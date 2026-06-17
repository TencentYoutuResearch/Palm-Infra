#pragma once

#include "graph/graph.h"
#include "graph/execute.h"
#include "kernels/tensor.h"
#include "kernels/threading.h"

#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// mlllm — LLM inference engine
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
    int n_ctx = 4096;                 // max sequence length
    int rope_dim = 64;
    float rope_theta = 500000.f;
    int num_threads = 4;
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

    // exposed for testing
    Tensor build_causal_mask(int seq_len, int past_len);
    void generate_rope_cache(int seq_len, int start_pos,
                             Tensor& cos, Tensor& sin);

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

    // KV cache tensor pointers (per layer)
    struct CachePair {
        Tensor* k = nullptr;
        Tensor* v = nullptr;
        int k_head_dim = 0;      // head_dim for K cache (constant)
        int k_num_heads = 0;     // num_kv_heads for K cache (constant)
        int v_head_dim = 0;      // head_dim for V cache (constant)
        int v_num_heads = 0;     // num_kv_heads for V cache (constant)
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

    /// Allocate KV cache buffers with metadata header.
    void allocate_caches(Graph& g, int n_ctx);

    // weight tensors
    Tensor* embed_weight_ = nullptr;  // [vocab_size, hidden_dim]
};
