#pragma once

#include "graph/graph.h"
#include "graph/execute.h"
#include "kernels/tensor.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// PROJECT_NAME — LLM inference engine
//
// Three-phase: load → prefill(ids) → decode(id) × N
// KV cache is managed as persistent preallocated tensors with view height.
// ---------------------------------------------------------------------------

struct EngineConfig {
    std::string graph_path;      // .graph file
    int n_ctx = 4096;            // max sequence length
    int rope_dim = 64;
    float rope_theta = 500000.f;
    int num_threads = 4;
    std::string decoder_out_blob = "out";  // name of output node in graph
};

class LLMEngine {
public:
    ~LLMEngine();

    /// Load the graph from file and initialise runtime state.
    bool load(const EngineConfig& cfg);

    /// Process a full sequence of tokens (prefill).
    /// Returns the next token ID.
    int prefill(const std::vector<int>& token_ids);

    /// Process a single token (decode step).
    /// Returns the next token ID.
    int decode(int token_id);

    /// Reset KV cache and past length.
    void reset();

    const EngineConfig& config() const { return cfg_; }
    int past_len() const { return past_len_; }
    Tensor* embed_weight() { return embed_weight_; }
    const Graph& graph() const { return graph_; }

    // exposed for testing
    Tensor build_causal_mask(int seq_len, int past_len);
    void generate_rope_cache(int seq_len, int start_pos,
                             Tensor& cos, Tensor& sin);

private:
private:
    EngineConfig cfg_;
    Graph        graph_;
    ExecContext  exec_ctx_;
    int          past_len_ = 0;

    // KV cache tensor pointers (per layer)
    struct CachePair {
        Tensor* k = nullptr;
        Tensor* v = nullptr;
    };
    std::vector<CachePair> caches_;

    /// Embed tokens (placeholder — returns dummy data).
    Tensor embed(const std::vector<int>& token_ids);

    /// Run lm_head on the last hidden state (placeholder).
    int run_lmhead(const Tensor& hidden);

    /// Feed inputs, run graph, extract output.
    Tensor run_decoder(const Tensor& hidden, const Tensor& mask,
                       const Tensor& cos, const Tensor& sin);

    // weight tensors
    Tensor* embed_weight_ = nullptr;  // [vocab_size, hidden_dim]
};
