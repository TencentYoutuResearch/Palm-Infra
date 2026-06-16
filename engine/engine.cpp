#include "engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// LLMEngine
// ---------------------------------------------------------------------------

LLMEngine::~LLMEngine() {
    // Tensor data owned by BufferPool or EXTERNAL — no explicit free needed
}

bool LLMEngine::load(const EngineConfig& cfg) {
    cfg_ = cfg;

    if (!graph_load(graph_, cfg.graph_path.c_str())) {
        fprintf(stderr, "Engine: failed to load graph %s\n", cfg.graph_path.c_str());
        return false;
    }

    // ---- load weight files referenced by CONSTANT nodes ----
    std::string graph_dir = cfg.graph_path;
    size_t slash = graph_dir.find_last_of("/\\");
    if (slash != std::string::npos) graph_dir = graph_dir.substr(0, slash + 1);
    else graph_dir = "./";

    for (auto& node : graph_.nodes) {
        if (node.op_type == OpType::CONSTANT && !node.params.str.empty()) {
            std::string wpath = graph_dir + node.params.str[0];
            MappedFile mf;
            if (mf.open(wpath.c_str())) {
                // set tensor data from mmap'd weight
                auto& t = graph_.runtime.tensors[node.id];
                // Always trust the graph node's precision, not the file header
                t.prec = node.out_prec;
                t.shape[0] = node.out_shape[0];
                t.shape[1] = node.out_shape[1];
                t.shape[2] = node.out_shape[2];
                t.shape[3] = node.out_shape[3];
                t.compute_strides();
                t.data = const_cast<void*>(mf.data());
                t.mem_type = MemoryType::EXTERNAL;
                graph_.runtime.weights.push_back(std::move(mf));
            } else {
                fprintf(stderr, "Engine: failed to load weight %s\n", wpath.c_str());
            }
        }
    }

    // set up execution context
    exec_ctx_.graph = &graph_;
    exec_ctx_.pool  = &graph_.runtime.pool;
    prepare_execution(exec_ctx_);

    // find KV cache tensors (INPUT nodes with names like "cache_k0")
    int n_layers = 0;
    for (auto& node : graph_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty()) {
            const std::string& name = node.params.str[0];
            if (name.find("cache_k") == 0) {
                int layer_idx = std::stoi(name.substr(7));
                if (layer_idx >= (int)caches_.size()) {
                    caches_.resize(layer_idx + 1);
                }
                caches_[layer_idx].k = &graph_.runtime.tensors[node.id];
                if (layer_idx >= n_layers) n_layers = layer_idx + 1;
            } else if (name.find("cache_v") == 0) {
                int layer_idx = std::stoi(name.substr(7));
                if (layer_idx >= (int)caches_.size()) {
                    caches_.resize(layer_idx + 1);
                }
                caches_[layer_idx].v = &graph_.runtime.tensors[node.id];
                if (layer_idx >= n_layers) n_layers = layer_idx + 1;
            }
        }
    }

    printf("Engine: %d layers, %zu cache pairs, %zu nodes\n",
           n_layers, caches_.size(), graph_.nodes.size());

    reset();
    return true;
}

void LLMEngine::reset() {
    past_len_ = 0;

    // reinitialise KV cache tensors (preallocate to n_ctx)
    for (auto& cp : caches_) {
        if (cp.k) {
            // K: [head_dim, n_ctx, num_kv_heads]
            int hd = (int)cp.k->shape[0];
            int nkv = (int)cp.k->shape[2];
            size_t es = cp.k->element_size();
            void* buf = graph_.runtime.pool.acquire((size_t)hd * cfg_.n_ctx * nkv * es);
            *cp.k = Tensor::create(cp.k->prec, MemoryType::POOLED, hd, cfg_.n_ctx, nkv, 1, buf);
            cp.k->shape[1] = 0;  // valid length = 0
            cp.k->compute_strides();  // set capacity-based strides, then reset height
            cp.k->shape[1] = 0;
        }
        if (cp.v) {
            int vd = (int)cp.v->shape[0];
            int nkv = (int)cp.v->shape[2];
            size_t es = cp.v->element_size();
            void* buf = graph_.runtime.pool.acquire((size_t)vd * cfg_.n_ctx * nkv * es);
            *cp.v = Tensor::create(cp.v->prec, MemoryType::POOLED, vd, cfg_.n_ctx, nkv, 1, buf);
            cp.v->shape[1] = 0;
            cp.v->compute_strides();
            cp.v->shape[1] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// embed / lmhead (placeholder — real implementation depends on model format)
// ---------------------------------------------------------------------------

Tensor LLMEngine::embed(const std::vector<int>& /*token_ids*/) {
    // Placeholder: create a dummy hidden state.
    // Real implementation loads embed weights and does lookup.
    // For now, return a zero tensor of shape [hidden_dim, seq_len].
    int hidden_dim = 2048;
    int seq_len = (int)past_len_ + 1;  // will be set properly by caller
    void* buf = graph_.runtime.pool.acquire(hidden_dim * seq_len * sizeof(float));
    Tensor t = Tensor::create(Precision::FP32, MemoryType::POOLED, hidden_dim, seq_len, 1, 1, buf);
    std::memset(t.data, 0, t.nbytes());
    return t;
}

int LLMEngine::run_lmhead(const Tensor& /*hidden*/) {
    // Placeholder: return a dummy token.
    // Real implementation runs lm_head projection + argmax.
    return 0;
}

// ---------------------------------------------------------------------------
// rope / mask helpers
// ---------------------------------------------------------------------------

Tensor LLMEngine::build_causal_mask(int seq_len, int past_len) {
    int total = past_len + seq_len;
    void* buf = graph_.runtime.pool.acquire(total * seq_len * sizeof(float));
    Tensor mask = Tensor::create(Precision::FP32, MemoryType::POOLED, total, seq_len, 1, 1, buf);
    float* d = mask.ptr<float>();
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < total; j++) {
            d[i * total + j] = (j > past_len + i) ? -1e38f : 0.f;
        }
    }
    return mask;
}

void LLMEngine::generate_rope_cache(int seq_len, int start_pos,
                                    Tensor& cos, Tensor& sin) {
    int half = cfg_.rope_dim / 2;
    void* cb = graph_.runtime.pool.acquire(half * seq_len * sizeof(float));
    void* sb = graph_.runtime.pool.acquire(half * seq_len * sizeof(float));
    cos = Tensor::create(Precision::FP32, MemoryType::POOLED, half, seq_len, 1, 1, cb);
    sin = Tensor::create(Precision::FP32, MemoryType::POOLED, half, seq_len, 1, 1, sb);

    for (int n = 0; n < seq_len; n++) {
        int pos = start_pos + n;
        for (int i = 0; i < half; i++) {
            float theta = 1.0f / std::pow(cfg_.rope_theta, 2.0f * i / cfg_.rope_dim);
            float angle = pos * theta;
            cos.ptr<float>()[n * half + i] = std::cos(angle);
            sin.ptr<float>()[n * half + i] = std::sin(angle);
        }
    }
}

// ---------------------------------------------------------------------------
// decoder
// ---------------------------------------------------------------------------

Tensor LLMEngine::run_decoder(const Tensor& hidden, const Tensor& mask,
                               const Tensor& cos, const Tensor& sin) {
    auto& tensors = graph_.runtime.tensors;

    // Feed graph inputs
    for (auto& node : graph_.nodes) {
        if (node.op_type != OpType::INPUT) continue;
        if (node.params.str.empty()) continue;

        const std::string& name = node.params.str[0];
        Tensor* t = &tensors[node.id];

        if (name == "hidden") {
            *t = hidden;
        } else if (name == "mask") {
            *t = mask;
        } else if (name == "cos") {
            *t = cos;
        } else if (name == "sin") {
            *t = sin;
        }
        // cache_k/cache_v already set in reset() / updated by SDPA
    }

    execute_graph(exec_ctx_);

    // Extract output (last graph output node)
    if (!graph_.graph_outputs.empty()) {
        uint32_t out_id = graph_.graph_outputs.back();
        return tensors[out_id];
    }

    fprintf(stderr, "Engine: no graph output found\n");
    return Tensor();
}

// ---------------------------------------------------------------------------
// prefill / decode
// ---------------------------------------------------------------------------

int LLMEngine::prefill(const std::vector<int>& token_ids) {
    int n = (int)token_ids.size();
    if (n == 0) return -1;

    reset();

    Tensor h = embed(token_ids);
    // Override shape: embed should produce [hidden_dim, n]
    h.shape[1] = n;
    h.compute_strides();

    Tensor cos, sin;
    generate_rope_cache(n, 0, cos, sin);

    Tensor mask = build_causal_mask(n, 0);
    Tensor out = run_decoder(h, mask, cos, sin);

    past_len_ = n;
    return run_lmhead(out);
}

int LLMEngine::decode(int token_id) {
    Tensor h = embed({token_id});
    h.shape[1] = 1;
    h.compute_strides();

    Tensor cos, sin;
    generate_rope_cache(1, past_len_, cos, sin);

    Tensor mask = build_causal_mask(1, past_len_);
    Tensor out = run_decoder(h, mask, cos, sin);

    past_len_++;
    return run_lmhead(out);
}
