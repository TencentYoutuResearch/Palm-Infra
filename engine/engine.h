#pragma once

#include "graph/graph.h"
#include "graph/execute.h"
#include "engine/backend.h"
#include "engine/accelerator_backend.h"
#include "engine/sampler.h"
#include "kernels/tensor.h"
#include "kernels/threading.h"
#include "kernels/moe_ssd.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// mollm — LLM inference engine
//
// Multi-graph: prefill graph (seq_len=N) + decode graph (seq_len=1).
// KV cache with embedded metadata header.
// Weights shared between graphs via path-dedup mapping.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CacheMetadata — embedded in KV cache buffer header (64 bytes)
//
// Layout of a cache buffer:
//   [0..63]       CacheMetadata
//   [64..]        key/value data (FP32)
// ---------------------------------------------------------------------------

struct CacheMetadata {
    uint64_t current_seq_len = 0;   // valid sequence length (past_len)
    uint64_t max_seq_len     = 0;   // buffer capacity (n_ctx)
    uint64_t num_kv_heads    = 0;
    uint64_t head_dim        = 0;
    uint64_t v_head_dim      = 0;
    uint64_t reserved[3]     = {0, 0, 0};

    static constexpr size_t SIZE = 64;
};

static_assert(sizeof(CacheMetadata) == CacheMetadata::SIZE, "CacheMetadata must be 64 bytes");

// Helper: get metadata pointer from cache tensor data
inline CacheMetadata* cache_meta(void* data) {
    return static_cast<CacheMetadata*>(data);
}

inline const CacheMetadata* cache_meta(const void* data) {
    return static_cast<const CacheMetadata*>(data);
}

// Helper: get key/value data pointer (after metadata header)
inline void* cache_data(void* data) {
    return static_cast<char*>(data) + CacheMetadata::SIZE;
}

inline const void* cache_data(const void* data) {
    return static_cast<const char*>(data) + CacheMetadata::SIZE;
}

// ---------------------------------------------------------------------------
// EngineConfig
// ---------------------------------------------------------------------------

enum class WeightLoadingMode {
    MMAP,
    RESIDENT,
};

enum class Device {
    CPU,
    METAL,
    CUDA,
};

// Controls what happens when the requested accelerator was not compiled in,
// cannot be initialized, or does not support the loaded model. Library users
// retain the historical CPU fallback by default; command-line tools require an
// explicitly selected accelerator so benchmarks cannot silently measure CPU.
enum class DeviceFallbackPolicy {
    ALLOW_CPU,
    REQUIRE_REQUESTED,
};

#if defined(__APPLE__)
inline constexpr bool kDefaultLockMoeSsdCache = true;
#else
inline constexpr bool kDefaultLockMoeSsdCache = false;
#endif

struct EngineConfig {
    static constexpr int kMinImageMaxPixels = 256 * 256;
    static constexpr int kDefaultImageMaxPixels = 512 * 512;
    static constexpr int kAbsoluteImageMaxPixels = 1024 * 1024;

    std::string package_path;         // .mollm single-file package (required)
    Device device = Device::CPU;      // METAL requires the MOLLM_METAL option
    DeviceFallbackPolicy device_fallback =
        DeviceFallbackPolicy::ALLOW_CPU;
    OperatorFallbackPolicy operator_fallback =
        OperatorFallbackPolicy::ALLOW_REFERENCE;
    int n_ctx = 4096;                 // max sequence length
    int rope_dim = 64;
    float rope_theta = 500000.f;
    int num_threads = default_worker_threads();
    WeightLoadingMode weight_loading = WeightLoadingMode::RESIDENT;
    // Host-side MoE expert cache. A non-zero value enables SSD offload for
    // packages carrying `moe_expert_storage` metadata.
    size_t moe_ssd_cache_bytes = 0;
    int moe_ssd_io_workers = 8;
    // Decode-only next-layer gate prefetch. Enabled with the shared SSD pool.
    bool moe_ssd_cross_layer_prefetch = true;
    // Number of early MoE layers to prioritize in the SSD expert cache.
    // Zero retains equal per-layer partitions.
    int moe_ssd_shallow_cache_layers = 0;
    // Shared capacity across all MoE layers. Disable to reproduce the legacy
    // equal-per-layer cache.
    bool moe_ssd_global_cache = true;
    // Wire dense mmap pages so a large expert cache cannot evict them.
    // Ignored unless SSD offload uses mmap weights.
    bool lock_dense_weights = true;
    // Expert buffers are anonymous memory. On macOS, wiring them prevents the
    // VM compressor from turning old cache hits into expensive decompressions.
    bool lock_moe_ssd_cache = kDefaultLockMoeSsdCache;
    // Experimental full-Metal decode for SSD-offloaded MoE packages. Expert
    // weights are loaded directly into Shared Metal buffers.
    bool metal_ssd_full = false;
    // Optional Chrome Trace / Perfetto JSON path. Empty disables tracing.
    std::string trace_path;
    // Experimental single-head MTP speculation. Zero keeps the ordinary
    // generation path even when the package contains an MTP graph.
    int mtp_draft_tokens = 0;

    SamplingParams sampling;

    // Output-only: set by load() from embedded package assets.
    // Callers use both paths so chat formatting follows the package's actual
    // template instead of guessing from its special-token vocabulary.
    std::string tokenizer_path;
    std::string chat_template_path;

    // When true, load the decode graph as the prefill graph too (so prefill()
    // runs the seq=1 graph). Used by test_e2e; enabled automatically for the
    // current RWKV v7 correctness-first path.
    bool use_decode_as_prefill = false;

    // When true, prefill pads short prompts to graph_seq_len (build-time
    // seq_len, typically 256) instead of running with the actual token count.
    // Stateful ops still receive n_real_tokens to skip padding positions.
    // For A/B benchmark comparison against DYNAMIC mode.
    bool static_padded = false;
    // CPU vision attention is quadratic in the number of image patches.
    // Resize ordinary high-resolution photos to a practical default budget.
    int image_max_pixels = kDefaultImageMaxPixels;
};

struct VisionEmbedding {
    std::vector<float> values; // token-major [tokens, hidden_size]
    int tokens = 0;
    int hidden_size = 0;
    int grid_t = 0;
    int grid_h = 0;
    int grid_w = 0;
};

// ---------------------------------------------------------------------------
// LLMEngine
// ---------------------------------------------------------------------------

class LLMEngine {
public:
    ~LLMEngine();

    /// Load prefill and decode graphs, initialise shared weights and KV caches.
    bool load(const EngineConfig& cfg);

    /// Process a full sequence of tokens (prefill).
    int prefill(const std::vector<int>& token_ids);

    /// Prefill one Qwen3.5 prompt containing exactly `vision.tokens`
    /// occurrences of image_token_id. The corresponding token embeddings are
    /// replaced by the vision encoder output and text RoPE uses Qwen's 3-axis
    /// multimodal positions.
    int prefill_with_image(const std::vector<int>& token_ids,
                           int image_token_id,
                           const VisionEmbedding& vision,
                           std::string* error = nullptr);

    /// Like prefill() but returns the raw hidden states instead of the sampled token.
    /// Used by tests for perplexity computation.
    Tensor prefill_hidden(const std::vector<int>& token_ids,
                          std::vector<float>* all_logits = nullptr,
                          std::vector<int>* all_top1 = nullptr);

    /// Like decode() but returns the raw hidden state instead of the sampled token.
    Tensor decode_hidden(int token_id);

    /// Process a single token (decode step).
    int decode(int token_id);

    /// Verify a short MTP draft while processing token_id. Every returned
    /// token is sampled by the target model; MTP only batches target work.
    bool speculative_decode(int token_id, std::vector<int>& output_tokens,
                            int max_output_tokens = 17,
                            int stop_token_id = -1);

    /// Run the packaged Qwen3.5 vision tower on processor-compatible flattened
    /// patches. `pixel_values` is token-major [grid_t*grid_h*grid_w, patch_dim].
    bool encode_vision_patches(const std::vector<float>& pixel_values,
                               int grid_t, int grid_h, int grid_w,
                               VisionEmbedding& output,
                               std::string* error = nullptr);

    /// Decode and preprocess one image using the package's Qwen3.5 processor
    /// metadata, then run the packaged vision tower.
    bool encode_image_file(const std::string& path, VisionEmbedding& output,
                           std::string* error = nullptr);

    /// Reset KV cache and past length.
    void reset();

    const EngineConfig& config() const { return cfg_; }
    const SamplingParams& sampling_params() const { return sampler_.params(); }
    bool set_sampling_params(const SamplingParams& params,
                             std::string* error = nullptr) {
        if (!sampler_.configure(params, error, true)) return false;
        cfg_.sampling = params;
        return true;
    }
    int past_len() const { return past_len_; }
    bool has_vision_encoder() const { return !graph_vision_.nodes.empty(); }
    bool has_mtp() const { return !graph_mtp_.nodes.empty(); }
    struct MtpStats {
        static constexpr size_t kMaxDraftDepth = 16;
        uint64_t steps = 0;
        uint64_t draft_calls = 0;
        uint64_t drafted = 0;
        uint64_t accepted = 0;
        uint64_t fallback_steps = 0;
        uint64_t verify_tokens = 0;
        uint64_t sync_tokens = 0;
        std::array<uint64_t, kMaxDraftDepth> attempts_by_depth{};
        std::array<uint64_t, kMaxDraftDepth> drafted_by_depth{};
        std::array<uint64_t, kMaxDraftDepth> accepted_by_depth{};
        double total_ms = 0.0;
        double draft_ms = 0.0;
        double draft_model_ms = 0.0;
        double verify_ms = 0.0;
        double sample_ms = 0.0;
        double sync_ms = 0.0;
    };
    const MtpStats& mtp_stats() const { return mtp_stats_; }
    // Package-level metadata fields (model_name, architecture, quantization,
    // num_layers, hidden_size, num_heads, n_ctx, vocab_size, prefill_seq_len).
    // Empty for packages without a metadata JSON section.
    const std::unordered_map<std::string, std::string>& package_metadata() const {
        return package_metadata_;
    }
    void set_profile_enabled(bool enabled);
    void reset_profiles();
    const ExecContext& prefill_exec_ctx() const { return exec_ctx_prefill_; }
    const ExecContext& decode_exec_ctx() const { return exec_ctx_decode_; }
    const ExecContext& mtp_exec_ctx() const { return exec_ctx_mtp_; }
    const ExecContext& mtp_verify_exec_ctx() const {
        return exec_ctx_mtp_verify_;
    }

    /// BufferPool memory stats (for leak detection in benchmarks).
    /// Returns {active_bytes, peak_bytes, acquire_count, release_count} from the prefill graph's pool.
    struct PoolStats { size_t active, peak, acquires, releases, freelist; };
    PoolStats prefill_pool_stats() const {
        return {graph_prefill_.runtime.pool.active_bytes(),
                graph_prefill_.runtime.pool.peak_bytes(),
                graph_prefill_.runtime.pool.acquire_count(),
                graph_prefill_.runtime.pool.release_count(),
                graph_prefill_.runtime.pool.pool_bytes()};
    }
    PoolStats decode_pool_stats() const {
        return {graph_decode_.runtime.pool.active_bytes(),
                graph_decode_.runtime.pool.peak_bytes(),
                graph_decode_.runtime.pool.acquire_count(),
                graph_decode_.runtime.pool.release_count(),
                graph_decode_.runtime.pool.pool_bytes()};
    }

    /// Release all non-INPUT/CONSTANT POOLED tensors from the prefill graph's
    /// pool. Called after prefill() completes to release the last chunk's
    /// intermediate buffers (which won't be reset by a subsequent
    /// execute_graph call since there's no next chunk).
    void release_prefill_buffers();

    /// Park worker threads (drop idle CPU). Auto-resumes on next prefill/decode.
    void park_workers() { thread_pool_.park(); }

    /// Touch mmap'd package weight pages so first-token latency does not pay
    /// lazy page-in cost. Returns the number of bytes covered.
    size_t warmup_package_weights();
    bool package_weights_mmap_backed() const {
        return package_weights_base_ != nullptr && !package_weights_resident_;
    }
    bool moe_ssd_offload_enabled() const { return moe_ssd_cache_ != nullptr; }
    MoeSsdCache::Stats moe_ssd_stats() const {
        return moe_ssd_cache_ ? moe_ssd_cache_->stats() : MoeSsdCache::Stats{};
    }
    void reset_moe_ssd_stats() {
        if (moe_ssd_cache_) moe_ssd_cache_->reset_stats();
    }
    /// Return raw logits. If all_positions=true, returns vocab_size*seq_len
    /// floats (seq_len blocks of vocab_size). Otherwise just the last position.
    std::vector<float> run_lmhead_raw(const Tensor& hidden, int n_tokens = 1,
                                       bool all_positions = false);

private:
    void prepare_accelerator_prefill_weights();
    bool decode_uses_metal_expert_cache() const;
    void release_vision_buffers();

    EngineConfig cfg_;
    Sampler sampler_;
    Graph graph_prefill_;
    Graph graph_decode_;
    Graph graph_vision_;
    Graph graph_mtp_;
    Graph graph_mtp_verify_;
    ExecContext exec_ctx_prefill_;
    ExecContext exec_ctx_decode_;
    ExecContext exec_ctx_vision_;
    ExecContext exec_ctx_mtp_;
    ExecContext exec_ctx_mtp_verify_;
    ThreadPool thread_pool_;
    CPUBackend cpu_backend_;     // owned by engine; assigned to ExecContexts
    // Active graph-resident accelerator. The engine is deliberately unaware
    // of backend-specific resource types.
    std::unique_ptr<AcceleratorBackend> accelerator_backend_;
    int past_len_ = 0;
    int mtp_past_len_ = 0;
    MtpStats mtp_stats_;

    // Shared mmap'd weight files (path → MappedFile)
    std::unordered_map<std::string, size_t> weight_map_;  // path → index into shared_weights_
    std::vector<MappedFile> shared_weights_;

    // .mollm package: raw mmap of the whole file
    void* package_mmap_ = nullptr;
    size_t package_mmap_size_ = 0;
    const uint8_t* package_weights_base_ = nullptr;
    size_t package_weights_size_ = 0;
    bool package_weights_resident_ = false;
    std::vector<uint8_t> package_weights_storage_;
    // weight filename → (offset, size) within weights region
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> package_weight_map_;
    // Aggregate expert ranges (relative to package_weights_base_) bypassed by
    // SSD offload. Dense-only warmup must leave these untouched.
    // Package-relative weight ranges intentionally excluded from dense mmap
    // warmup/mlock: paged experts and source tensors replaced by CPU
    // sidecars.
    std::vector<std::pair<uint64_t, uint64_t>>
        mmap_weight_exclusion_ranges_;
    std::vector<std::pair<void*, size_t>> locked_dense_ranges_;
    // Optional demand-paged storage for routed MoE experts.
    std::unique_ptr<MoeSsdCache> moe_ssd_cache_;
    // Full package-level metadata JSON fields (model_name, architecture,
    // quantization, num_layers, hidden_size, num_heads, n_ctx, vocab_size,
    // prefill_seq_len, ...). Populated in load_package(); exposed via
    // package_metadata() for CLI banner / display.
    std::unordered_map<std::string, std::string> package_metadata_;

    // Temp files extracted from the package (cleaned up in destructor).
    std::vector<std::string> temp_files_;

    // Load-time matmul layouts (path/layout suffix → packed buffer).
    std::unordered_map<std::string, std::vector<uint8_t>> packed_weights_;
    // Typed backend-specific layouts. Tensors borrow entries from this
    // engine-lifetime store; new backends should register layouts here rather
    // than add sidecar fields to Tensor.
    PreparedWeightMap prepared_weights_;

    // Engine-owned contiguous copy returned by prefill_hidden/decode_hidden.
    // Valid until the next hidden-output call on this engine.
    std::vector<uint8_t> hidden_output_copy_;
    std::vector<uint8_t> mtp_hidden_output_copy_;
    Tensor mtp_draft_hidden_device_;
    std::vector<float> mtp_pending_hidden_;
    // Qwen3.5 trains its MTP head on the backbone's pre-final-norm hidden
    // state. Keep a reusable host copy separate from the normalized graph
    // output consumed by lm_head.
    std::vector<uint8_t> mtp_target_hidden_copy_;

    // Engine-lifetime storage for KV cache and recurrent state. Graph pools
    // are reserved for per-execution temporaries.
    BufferPool persistent_pool_;

    // KV cache tensor pointers (per layer)
    struct CachePair {
        // Standard KV cache (full attention layers)
        Tensor* k = nullptr;
        Tensor* v = nullptr;
        int k_head_dim = 0;
        int k_num_heads = 0;
        int v_head_dim = 0;
        int v_num_heads = 0;

        // GDN recurrent state (linear attention layers)
        Tensor* gdn_state = nullptr;   // [v_dim, k_dim, num_heads] FP32
        Tensor* gdn_conv = nullptr;    // [groups, kernel-1] FP32
        Tensor* gdn_checkpoint = nullptr;
        Tensor* gdn_conv_checkpoint = nullptr;
        int gdn_v_dim = 0;
        int gdn_k_dim = 0;
        int gdn_num_heads = 0;
        int gdn_conv_groups = 0;
        int gdn_conv_kernel = 0;

        Tensor* rwkv_state = nullptr;
        Tensor* rwkv_att_shift = nullptr;
        Tensor* rwkv_ffn_shift = nullptr;

        bool is_linear_attn = false;
    };
    std::vector<CachePair> caches_;
    std::vector<CachePair> mtp_caches_;
    // Plain model-specific persistent tensors keyed by their serialized
    // `aux_state<N>` input id.
    std::unordered_map<int, Tensor*> auxiliary_states_;

    /// Embed tokens.
    Tensor embed(const std::vector<int>& token_ids, int pad_to = 0);

    Tensor build_causal_mask(int seq_len, int past_len,
                             bool initialize = true);
    void generate_rope_cache(int seq_len, int start_pos,
                             Tensor& cos, Tensor& sin);
    void generate_multimodal_rope_cache(int seq_len, int token_offset,
                                        Tensor& cos, Tensor& sin);
    bool prepare_multimodal_positions(const std::vector<int>& token_ids,
                                      int image_token_id,
                                      const VisionEmbedding& vision,
                                      std::string* error);

    /// Run lm_head on the last hidden state.
    int run_lmhead(const Tensor& hidden, int n_tokens = 1,
                   bool finish_accelerator_graph = false);

    /// Feed inputs, run graph, extract output.
    Tensor run_graph(Graph& graph, ExecContext& exec_ctx,
                     const Tensor& hidden, const Tensor& mask,
                     const Tensor& cos, const Tensor& sin,
                     const Tensor* token_ids = nullptr,
                     bool defer_accelerator_end = false,
                     const Tensor* target_hidden = nullptr,
                     int graph_position = -1,
                     int stop_after_node_index = -1);

    Tensor run_mtp_tokens(const std::vector<int>& token_ids,
                          const Tensor& target_hidden, int position,
                          int* draft_token = nullptr);
    bool update_mtp_cache(const std::vector<int>& token_ids,
                          const Tensor& target_hidden, int position);
    bool execute_mtp_tokens(const std::vector<int>& token_ids,
                            const Tensor& target_hidden, int position,
                            bool cache_only, Tensor* hidden_output,
                            int* draft_token);
    bool sync_mtp(const std::vector<int>& token_ids,
                  const Tensor& verified_hidden, int position,
                  int preserved_prefix = 0);
    bool restore_confirmed_target_state(int target_length);
    Tensor verify_target_tokens(const std::vector<int>& token_ids,
                                int position, std::vector<float>* logits,
                                std::vector<int>* top1);
    void set_cache_length(std::vector<CachePair>& caches, int length);
    Tensor capture_mtp_target_hidden(Graph& graph, const Tensor& fallback,
                                     Backend* backend);
    Tensor current_mtp_target_hidden(int tokens);

    /// Transactional public-load implementation and shared teardown path.
    bool load_impl(const EngineConfig& cfg);
    void clear_model_state();

    /// Load a single graph and set up its CONSTANT nodes from shared weights.
    bool load_graph(Graph& g, ExecContext& exec_ctx, const char* path);
    bool load_package(const std::string& path, std::string& pf_path,
                      std::string& dc_path, std::string& vi_path,
                      std::string& mtp_verify_path,
                      std::string& mtp_path,
                      std::string& tok_path, std::string& jinja_path);
    size_t lock_dense_package_weights();

    /// Allocate KV cache buffers with metadata header.
    bool allocate_caches(Graph& g, ExecContext& exec_ctx,
                         std::vector<CachePair>& caches, int n_ctx);
    Backend* persistent_backend() const;
    bool set_cache_lengths(std::vector<CachePair>& caches, uint64_t length);

    /// Process a single chunk of tokens (≤ graph_seq_len).
    /// Called by prefill() in a loop for chunked prefill.
    int prefill_chunk(const std::vector<int>& token_ids, int past);

    // weight tensors
    Tensor* embed_weight_ = nullptr;   // [vocab_size, hidden_dim], row-major FP16/FP32 for lookup
    Tensor* lm_head_weight_ = nullptr; // [vocab_size, hidden_dim], regular matmul weight
    Tensor* vision_pos_embed_ = nullptr; // [positions, vision_hidden]

    const VisionEmbedding* active_vision_ = nullptr;
    int active_image_token_id_ = -1;
    int active_vision_cursor_ = 0;
    int multimodal_base_past_ = 0;
    int rope_position_delta_ = 0;
    std::vector<int> multimodal_position_ids_; // axis-major [3, prompt_tokens]
};
