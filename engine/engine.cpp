#include "engine/engine.h"
#include "kernels/matmul.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// LLMEngine
// ---------------------------------------------------------------------------

LLMEngine::~LLMEngine() {
    // Tensor data owned by BufferPool or EXTERNAL — no explicit free needed
    // MappedFiles in shared_weights_ auto-close on destruction
}

// ---------------------------------------------------------------------------
// load_graph — load one graph and set up CONSTANT nodes from shared weights
// ---------------------------------------------------------------------------

bool LLMEngine::load_graph(Graph& g, ExecContext& exec_ctx, const char* path) {
    if (!graph_load(g, path)) {
        fprintf(stderr, "Engine: failed to load graph %s\n", path);
        return false;
    }

    // Resolve graph directory for relative weight paths
    std::string graph_dir = path;
    size_t slash = graph_dir.find_last_of("/\\");
    if (slash != std::string::npos) graph_dir = graph_dir.substr(0, slash + 1);
    else graph_dir = "./";

    for (auto& node : g.nodes) {
        if (node.op_type != OpType::CONSTANT || node.params.str.empty()) continue;

        std::string wpath = node.params.str[0];
        if (wpath[0] != '/' && (wpath.size() < 2 || wpath[1] != ':')) {
            wpath = graph_dir + wpath;
        }

        auto& t = g.runtime.tensors[node.id];

        // Helper: set up weight tensor.
        // Weight is stored as [N, K] row-major: W[n,k] = data[n*K + k].
        // The matmul uses B[n*K + k] directly (not B[n + k*stride]).
        auto setup_weight = [&](void* data) {
            t.prec = node.out_prec;
            int64_t dim0 = node.out_shape[0];  // N
            int64_t dim1 = node.out_shape[1];  // K
            t.shape[0] = dim0;
            t.shape[1] = dim1;
            t.shape[2] = node.out_shape[2];
            t.shape[3] = node.out_shape[3];
            // For embed_tokens, the stride should be hidden_dim, not vocab_size
            bool is_embed = (node.params.str[0].find("embed_tokens") != std::string::npos);
            if (is_embed) {
                size_t es = t.element_size();
                t.stride[0] = es;
                t.stride[1] = es * dim1;  // hidden_dim, not vocab_size!
                t.stride[2] = t.stride[1] * dim0;
                t.stride[3] = t.stride[2] * t.shape[2];
            } else {
                t.compute_strides();
            }
            t.data = data;
            t.mem_type = MemoryType::EXTERNAL;
        };

        // Check if this weight is already loaded
        auto it = weight_map_.find(wpath);
        if (it != weight_map_.end()) {
            // Reuse existing mmap
            setup_weight(const_cast<void*>(shared_weights_[it->second].data()));
            continue;
        }

        // Load new mmap
        MappedFile mf;
        if (!mf.open(wpath.c_str())) {
            fprintf(stderr, "Engine: failed to load weight %s\n", wpath.c_str());
            return false;
        }

        size_t idx = shared_weights_.size();
        weight_map_[wpath] = idx;
        shared_weights_.push_back(std::move(mf));

        setup_weight(const_cast<void*>(shared_weights_[idx].data()));
    }

    // Find embed_tokens weight
    for (auto& node : g.nodes) {
        if (node.op_type == OpType::CONSTANT && !node.params.str.empty()) {
            if (node.params.str[0].find("embed_tokens") != std::string::npos) {
                embed_weight_ = &g.runtime.tensors[node.id];
            }
        }
    }

    exec_ctx.graph = &g;
    exec_ctx.pool  = &g.runtime.pool;
    exec_ctx.thread_pool = &thread_pool_;
    prepare_execution(exec_ctx);

    return true;
}

// ---------------------------------------------------------------------------
// allocate_caches — allocate KV cache buffers with metadata header
// ---------------------------------------------------------------------------

void LLMEngine::allocate_caches(Graph& g, int n_ctx) {
    // Find cache INPUT nodes and initialise their tensor shapes
    for (auto& node : g.nodes) {
        if (node.op_type != OpType::INPUT || node.params.str.empty()) continue;
        const std::string& name = node.params.str[0];

        if (name.find("cache_k") == 0) {
            int layer_idx = std::stoi(name.substr(7));
            if (layer_idx >= (int)caches_.size()) caches_.resize(layer_idx + 1);
            Tensor& t = g.runtime.tensors[node.id];
            t.prec = node.out_prec;
            t.shape[0] = node.out_shape[0];
            t.shape[1] = node.out_shape[1];
            t.shape[2] = node.out_shape[2];
            t.shape[3] = node.out_shape[3];
            t.compute_strides();
            caches_[layer_idx].k = &t;
            caches_[layer_idx].k_head_dim = (int)node.out_shape[0];
            caches_[layer_idx].k_num_heads = (int)node.out_shape[2];
        } else if (name.find("cache_v") == 0) {
            int layer_idx = std::stoi(name.substr(7));
            if (layer_idx >= (int)caches_.size()) caches_.resize(layer_idx + 1);
            Tensor& t = g.runtime.tensors[node.id];
            t.prec = node.out_prec;
            t.shape[0] = node.out_shape[0];
            t.shape[1] = node.out_shape[1];
            t.shape[2] = node.out_shape[2];
            t.shape[3] = node.out_shape[3];
            t.compute_strides();
            caches_[layer_idx].v = &t;
            caches_[layer_idx].v_head_dim = (int)node.out_shape[0];
            caches_[layer_idx].v_num_heads = (int)node.out_shape[2];
        }
    }
}

// ---------------------------------------------------------------------------
// load — load both graphs, set up shared weights and caches
// ---------------------------------------------------------------------------

void LLMEngine::set_profile_enabled(bool enabled) {
    exec_ctx_prefill_.profile_enabled = enabled;
    exec_ctx_decode_.profile_enabled = enabled;
}

void LLMEngine::reset_profiles() {
    reset_profile_stats(exec_ctx_prefill_);
    reset_profile_stats(exec_ctx_decode_);
}

bool LLMEngine::load(const EngineConfig& cfg) {
    cfg_ = cfg;
    cfg_.num_threads = std::max(cfg_.num_threads, 1);
    thread_pool_.resize(cfg_.num_threads);
    exec_ctx_prefill_.thread_pool = &thread_pool_;
    exec_ctx_decode_.thread_pool = &thread_pool_;

    // Load prefill graph first (establishes shared weights)
    if (!load_graph(graph_prefill_, exec_ctx_prefill_, cfg.prefill_graph_path.c_str())) {
        return false;
    }
    allocate_caches(graph_prefill_, cfg.n_ctx);

    // Load decode graph (reuses shared weights via weight_map_)
    if (!load_graph(graph_decode_, exec_ctx_decode_, cfg.decode_graph_path.c_str())) {
        return false;
    }

    // After decode graph load, cache pointers still point to prefill graph.
    // We'll migrate them after prefill completes.

    reset();
    return true;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void LLMEngine::reset() {
    past_len_ = 0;

    for (auto& cp : caches_) {
        if (cp.k) {
            int hd = cp.k_head_dim;
            int nkv = cp.k_num_heads;
            size_t es = cp.k->element_size();

            // Total buffer size = metadata (64 bytes) + data
            size_t data_bytes = (size_t)hd * cfg_.n_ctx * nkv * es;
            size_t total = CacheMetadata::SIZE + data_bytes;

            void* buf = graph_prefill_.runtime.pool.acquire(total);
            std::memset(buf, 0, total);

            // Init metadata
            auto* meta = cache_meta(buf);
            meta->current_seq_len = 0;
            meta->max_seq_len     = cfg_.n_ctx;
            meta->num_kv_heads    = (uint64_t)nkv;
            meta->head_dim        = (uint64_t)hd;

            // The tensor shape is the full buffer size in dim0, cache metadata in header
            *cp.k = Tensor::create(cp.k->prec, MemoryType::POOLED,
                                   (int64_t)total / (int64_t)es, 1, 1, 1, buf);
        }
        if (cp.v) {
            int vd = cp.v_head_dim;
            int nkv = cp.v_num_heads;
            size_t es = cp.v->element_size();

            size_t data_bytes = (size_t)vd * cfg_.n_ctx * nkv * es;
            size_t total = CacheMetadata::SIZE + data_bytes;

            void* buf = graph_prefill_.runtime.pool.acquire(total);
            std::memset(buf, 0, total);

            auto* meta = cache_meta(buf);
            meta->current_seq_len = 0;
            meta->max_seq_len     = cfg_.n_ctx;
            meta->num_kv_heads    = (uint64_t)nkv;
            meta->v_head_dim      = (uint64_t)vd;

            *cp.v = Tensor::create(cp.v->prec, MemoryType::POOLED,
                                   (int64_t)total / (int64_t)es, 1, 1, 1, buf);
        }
    }
}

// ---------------------------------------------------------------------------
// embed
// ---------------------------------------------------------------------------

Tensor LLMEngine::embed(const std::vector<int>& token_ids) {
    int seq_len = (int)token_ids.size();

    if (embed_weight_ && embed_weight_->data) {
        // embed_weight_ shape: [vocab_size, hidden_dim] with compute_strides()
        // stride[0]=4, stride[1]=4*vocab_size
        // weight_stride = stride[1]/4 = vocab_size
        // Access: embed_data[d + tid * vocab_size] → row tid, col d ✓
        int vocab_size = (int)embed_weight_->shape[0];
        int hidden_dim = (int)embed_weight_->shape[1];
        int weight_stride = (int)(embed_weight_->stride[1] / sizeof(float));

        // Verify: check that vocab_size * hidden_dim fits in the mmap region
        size_t embed_nbytes = embed_weight_->nbytes();
        size_t expected = (size_t)vocab_size * hidden_dim * sizeof(float);
        if (embed_nbytes < expected) {
            fprintf(stderr, "  embed: weight buffer too small! nbytes=%zu expected=%zu\n", embed_nbytes, expected);
        }

        float* buf = new float[hidden_dim * seq_len];
        Tensor t = Tensor::create(Precision::FP32, MemoryType::OWNED, hidden_dim, seq_len, 1, 1, buf);

        const float* embed_data = embed_weight_->ptr<float>();
        for (int s = 0; s < seq_len; s++) {
            int tid = token_ids[s];
            if (tid < 0 || tid >= vocab_size) tid = 0;
            float* dst = t.ptr<float>() + s * hidden_dim;
            // embed_weight is [vocab_size, hidden_dim] row-major
            // W[tid, d] = data[tid * hidden_dim + d]
            // But our stride says stride[1] = vocab_size*4, so:
            //   w[d + tid * weight_stride] = data[tid * vocab_size + d]
            // That's WRONG! We need data[tid * hidden_dim + d]
            // weight_stride should be hidden_dim, not vocab_size!
            for (int d = 0; d < hidden_dim; d++) {
                dst[d] = embed_data[d + tid * weight_stride];
            }
        }
        return t;
    }

    int hidden_dim = 2048;
    void* buf = graph_prefill_.runtime.pool.acquire(hidden_dim * seq_len * sizeof(float));
    Tensor t = Tensor::create(Precision::FP32, MemoryType::POOLED, hidden_dim, seq_len, 1, 1, buf);
    std::memset(t.data, 0, t.nbytes());
    return t;
}

// ---------------------------------------------------------------------------
// run_lmhead
// ---------------------------------------------------------------------------

int LLMEngine::run_lmhead(const Tensor& hidden, int n_tokens) {
    if (!embed_weight_ || !embed_weight_->data) return 0;

    int vocab_size = (int)embed_weight_->shape[0];
    int hidden_dim = (int)embed_weight_->shape[1];
    int seq_len = (int)hidden.shape[1];

    // Read the last real token, not necessarily the last position
    // (which may be padded when graph seq_len > n_tokens).
    int last_pos = (n_tokens > 0 && n_tokens <= seq_len) ? n_tokens - 1 : seq_len - 1;

    // hidden is [hidden_dim, seq_len], we want [hidden_dim, 1] for matmul
    // embed_weight is [vocab_size, hidden_dim] — we use it as weight B
    // output will be [vocab_size, 1] — we take argmax

    // Create a view of the last hidden row as A: [hidden_dim, 1]
    Tensor A = hidden;
    A.shape[1] = 1;
    A.data = static_cast<char*>(hidden.data) + last_pos * hidden_dim * sizeof(float);
    A.compute_strides();

    // Output: [vocab_size, 1]
    void* c_buf = graph_prefill_.runtime.pool.acquire(vocab_size * sizeof(float));
    Tensor C = Tensor::create(Precision::FP32, MemoryType::POOLED, vocab_size, 1, 1, 1, c_buf);

    kernel_matmul_fp32(A, *embed_weight_, C, exec_ctx_decode_.thread_pool);

    const float* scores = C.ptr<float>();
    struct Candidate { int id; float score; };
    Candidate top5[5] = {{0,-1e38f},{0,-1e38f},{0,-1e38f},{0,-1e38f},{0,-1e38f}};

    for (int v = 0; v < vocab_size; v++) {
        float score = scores[v];
        for (int i = 0; i < 5; i++) {
            if (score > top5[i].score) {
                for (int j = 4; j > i; j--) top5[j] = top5[j-1];
                top5[i] = {v, score};
                break;
            }
        }
    }

    return top5[0].id;
}

// ---------------------------------------------------------------------------
// rope / mask helpers
// ---------------------------------------------------------------------------

Tensor LLMEngine::build_causal_mask(int seq_len, int past_len) {
    int total = past_len + seq_len;
    void* buf = graph_prefill_.runtime.pool.acquire(total * seq_len * sizeof(float));
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

    void* cb = graph_prefill_.runtime.pool.acquire(half * seq_len * sizeof(float));
    void* sb = graph_prefill_.runtime.pool.acquire(half * seq_len * sizeof(float));
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
// run_graph — feed inputs and execute a graph
// ---------------------------------------------------------------------------

Tensor LLMEngine::run_graph(Graph& graph, ExecContext& exec_ctx,
                             const Tensor& hidden, const Tensor& mask,
                             const Tensor& cos, const Tensor& sin) {
    auto& tensors = graph.runtime.tensors;

    // Feed graph inputs
    for (auto& node : graph.nodes) {
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
        // cache_k/cache_v are set in reset() and updated by SDPA
    }

    execute_graph(exec_ctx);

    if (!graph.graph_outputs.empty()) {
        uint32_t out_id = graph.graph_outputs.back();
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

    // Determine the graph's expected seq_len from its hidden input shape.
    // For prefill graph, this is typically 128; for decode graph, it's 1.
    // We need to generate rope cache and mask matching the graph's seq_len,
    // not the actual token count, because all intermediate tensors have
    // the graph's seq_len baked into their static shapes.
    int graph_seq_len = 1;
    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty()
            && node.params.str[0] == "hidden") {
            graph_seq_len = (int)node.out_shape[1];
            break;
        }
    }

    reset();

    Tensor h = embed(token_ids);
    // Pad hidden to the graph's seq_len if needed (graph may expect 128 tokens
    // but we only have n tokens).  The causal mask ensures padded positions
    // don't affect real tokens, but we must provide a valid buffer.
    if (n < graph_seq_len) {
        int hidden_dim = (int)h.shape[0];
        size_t padded_bytes = (size_t)hidden_dim * graph_seq_len * sizeof(float);
        void* padded_buf = graph_prefill_.runtime.pool.acquire(padded_bytes);
        std::memset(padded_buf, 0, padded_bytes);
        // Copy real embeddings to the start
        std::memcpy(padded_buf, h.data, (size_t)hidden_dim * n * sizeof(float));
        // Create new padded tensor
        h = Tensor::create(Precision::FP32, MemoryType::POOLED,
                           hidden_dim, graph_seq_len, 1, 1, padded_buf);
    }
    h.shape[1] = graph_seq_len;
    h.compute_strides();

    Tensor cos, sin;
    generate_rope_cache(graph_seq_len, 0, cos, sin);

    Tensor mask = build_causal_mask(graph_seq_len, 0);

    // Update cache metadata: set current_seq_len = 0 (fresh prefill)
    for (auto& cp : caches_) {
        if (cp.k) cache_meta(cp.k->data)->current_seq_len = 0;
        if (cp.v) cache_meta(cp.v->data)->current_seq_len = 0;
    }

    Tensor out = run_graph(graph_prefill_, exec_ctx_prefill_, h, mask, cos, sin);

    past_len_ = n;

    // Update cache metadata for decode
    for (auto& cp : caches_) {
        if (cp.k) cache_meta(cp.k->data)->current_seq_len = (uint64_t)n;
        if (cp.v) cache_meta(cp.v->data)->current_seq_len = (uint64_t)n;
    }

    // Migrate cache tensors from prefill graph to decode graph
    for (auto& node : graph_decode_.nodes) {
        if (node.op_type != OpType::INPUT || node.params.str.empty()) continue;
        const std::string& name = node.params.str[0];

        if (name.find("cache_k") == 0) {
            int layer_idx = std::stoi(name.substr(7));
            if (layer_idx < (int)caches_.size() && caches_[layer_idx].k) {
                graph_decode_.runtime.tensors[node.id] = *caches_[layer_idx].k;
            }
        } else if (name.find("cache_v") == 0) {
            int layer_idx = std::stoi(name.substr(7));
            if (layer_idx < (int)caches_.size() && caches_[layer_idx].v) {
                graph_decode_.runtime.tensors[node.id] = *caches_[layer_idx].v;
            }
        }
    }

    return run_lmhead(out, n);
}

int LLMEngine::decode(int token_id) {
    Tensor h = embed({token_id});
    h.shape[1] = 1;
    h.compute_strides();

    Tensor cos, sin;
    generate_rope_cache(1, past_len_, cos, sin);

    Tensor mask = build_causal_mask(1, past_len_);

    Tensor out = run_graph(graph_decode_, exec_ctx_decode_, h, mask, cos, sin);

    past_len_++;

    // Update cache metadata
    for (auto& cp : caches_) {
        if (cp.k) cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v) cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    return run_lmhead(out);
}
