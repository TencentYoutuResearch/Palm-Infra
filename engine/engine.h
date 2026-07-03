#pragma once

#include "graph/graph.h"
#include "graph/execute.h"
#include "engine/backend.h"
#include "kernels/tensor.h"
#include "kernels/threading.h"

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

struct EngineConfig {
    std::string package_path;         // .mollm single-file package (required)
    int n_ctx = 4096;                 // max sequence length
    int rope_dim = 64;
    float rope_theta = 500000.f;
    int num_threads = 4;

    // Sampling params
    float temperature = 0.6f;         // 0 = greedy (argmax)
    int top_k = 50;                   // 0 = disabled
    float top_p = 0.9f;              // 0 = disabled
    unsigned int seed = 42;          // random seed for sampling

    // Output-only: set by load() when package contains a tokenizer.
    // Callers (e.g. CLI) read this to load the Tokenizer after engine.load().
    std::string tokenizer_path;

    // Test-only: when true, load the decode graph as the prefill graph too
    // (so prefill() runs the seq=1 decode graph). Used by test_e2e to verify
    // the decode path's PPL independently from the prefill path.
    bool use_decode_as_prefill = false;

    // When true, prefill pads short prompts to graph_seq_len (build-time
    // seq_len, typically 256) instead of running with the actual token count.
    // Stateful ops still receive n_real_tokens to skip padding positions.
    // For A/B benchmark comparison against DYNAMIC mode.
    bool static_padded = false;
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

    /// Like prefill() but returns the raw hidden states instead of the sampled token.
    /// Used by tests for perplexity computation.
    Tensor prefill_hidden(const std::vector<int>& token_ids);

    /// Like decode() but returns the raw hidden state instead of the sampled token.
    Tensor decode_hidden(int token_id);

    /// Process a single token (decode step).
    int decode(int token_id);

    /// Reset KV cache and past length.
    void reset();

    const EngineConfig& config() const { return cfg_; }
    int past_len() const { return past_len_; }
    Tensor* embed_weight() { return embed_weight_; }
    Tensor* lm_head_weight() { return lm_head_weight_; }
    void set_profile_enabled(bool enabled);
    void reset_profiles();
    const ExecContext& prefill_exec_ctx() const { return exec_ctx_prefill_; }
    const ExecContext& decode_exec_ctx() const { return exec_ctx_decode_; }

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

    // Dump ADD node outputs (last token) from prefill graph to dir.
    // Each layer has 2 ADD nodes: attention residual + MLP residual.
    void dump_prefill_add_outputs(const char* dir);

    /// Park worker threads (drop idle CPU). Auto-resumes on next prefill/decode.
    void park_workers() { thread_pool_.park(); }

    // exposed for testing
    Tensor build_causal_mask(int seq_len, int past_len);
    void generate_rope_cache(int seq_len, int start_pos,
                             Tensor& cos, Tensor& sin);
    /// Return raw logits. If all_positions=true, returns vocab_size*seq_len
    /// floats (seq_len blocks of vocab_size). Otherwise just the last position.
    std::vector<float> run_lmhead_raw(const Tensor& hidden, int n_tokens = 1,
                                       bool all_positions = false);

private:
    EngineConfig cfg_;
    Graph graph_prefill_;
    Graph graph_decode_;
    ExecContext exec_ctx_prefill_;
    ExecContext exec_ctx_decode_;
    ThreadPool thread_pool_;
    CPUBackend cpu_backend_;     // owned by engine; assigned to ExecContexts
    int past_len_ = 0;

    // Shared mmap'd weight files (path → MappedFile)
    std::unordered_map<std::string, size_t> weight_map_;  // path → index into shared_weights_
    std::vector<MappedFile> shared_weights_;

    // .mollm package: raw mmap of the whole file
    void* package_mmap_ = nullptr;
    size_t package_mmap_size_ = 0;
    const uint8_t* package_weights_base_ = nullptr;
    // weight filename → (offset, size) within weights region
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> package_weight_map_;
    // prefill_seq_len from package metadata
    int package_prefill_seq_len_ = 256;

    // Temp files extracted from the package (cleaned up in destructor).
    std::vector<std::string> temp_files_;

    // Load-time interleaved-packed FP16 weights (path → packed buffer)
    std::unordered_map<std::string, std::vector<uint8_t>> packed_weights_;

    // Engine-owned contiguous copy returned by prefill_hidden/decode_hidden.
    // Valid until the next hidden-output call on this engine.
    std::vector<uint8_t> hidden_output_copy_;

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
        Tensor* gdn_conv = nullptr;     // [groups, kernel-1] FP32
        int gdn_v_dim = 0;
        int gdn_k_dim = 0;
        int gdn_num_heads = 0;
        int gdn_conv_groups = 0;
        int gdn_conv_kernel = 0;

        bool is_linear_attn = false;
    };
    std::vector<CachePair> caches_;

    /// Embed tokens.
    Tensor embed(const std::vector<int>& token_ids, int pad_to = 0);

    /// Run lm_head on the last hidden state.
    int run_lmhead(const Tensor& hidden, int n_tokens = 1);

    /// Feed inputs, run graph, extract output.
    Tensor run_graph(Graph& graph, ExecContext& exec_ctx,
                     const Tensor& hidden, const Tensor& mask,
                     const Tensor& cos, const Tensor& sin);

    /// Load a single graph and set up its CONSTANT nodes from shared weights.
    bool load_graph(Graph& g, ExecContext& exec_ctx, const char* path);
    bool load_package(const std::string& path, std::string& pf_path,
                      std::string& dc_path, std::string& tok_path,
                      std::string& jin_path);

    /// Allocate KV cache buffers with metadata header.
    void allocate_caches(Graph& g, int n_ctx);

    /// Process a single chunk of tokens (≤ graph_seq_len).
    /// Called by prefill() in a loop for chunked prefill.
    int prefill_chunk(const std::vector<int>& token_ids, int past);

    // weight tensors
    Tensor* embed_weight_ = nullptr;   // [vocab_size, hidden_dim], row-major FP16/FP32 for lookup
    Tensor* lm_head_weight_ = nullptr; // [vocab_size, hidden_dim], regular matmul weight
};
