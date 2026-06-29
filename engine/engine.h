#pragma once

#include "graph/graph.h"
#include "graph/execute.h"
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
    std::string prefill_graph_path;   // .graph file for prefill
    std::string decode_graph_path;    // .graph file for decode
    std::string package_path;         // .mollm single-file package (alternative to above)
    std::string tokenizer_path;       // tokenizer.json path (may be set by package load)
    int n_ctx = 4096;                 // max sequence length
    int rope_dim = 64;
    float rope_theta = 500000.f;
    int num_threads = 4;

    // Sampling params
    float temperature = 0.6f;         // 0 = greedy (argmax)
    int top_k = 50;                   // 0 = disabled
    float top_p = 0.9f;              // 0 = disabled
    unsigned int seed = 42;          // random seed for sampling
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
    void set_profile_enabled(bool enabled);
    void reset_profiles();
    const ExecContext& prefill_exec_ctx() const { return exec_ctx_prefill_; }
    const ExecContext& decode_exec_ctx() const { return exec_ctx_decode_; }

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

    // Load-time interleaved-packed FP16 weights (path → packed buffer)
    std::unordered_map<std::string, std::vector<uint8_t>> packed_weights_;

    // Packed copy of embed_tokens for lm_head matmul (row-major original stays for embed lookup)
    std::vector<uint8_t> embed_packed_;

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
    Tensor embed(const std::vector<int>& token_ids);

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
    Tensor* embed_weight_ = nullptr;  // [vocab_size, hidden_dim]
};
