#include "engine/engine.h"
#include "engine/input_prep.h"
#include "engine/sampler.h"
#include "kernels/matmul.h"
#include "kernels/trace.h"
#ifdef MOLLM_METAL
#include "engine/metal_backend.h"
// Downcast the owned Backend* to MetalBackend* (non-null iff Metal active).
static inline MetalBackend* as_metal(const std::unique_ptr<Backend>& b) {
    return static_cast<MetalBackend*>(b.get());
}
#endif

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int metal_ssd_prefill_min_tokens() {
    static int threshold = [] {
        // On M5 Pro the crossover is noisy at 32 tokens because each MoE
        // layer still introduces a GPU→CPU boundary; 64 is consistently on
        // the Metal-favorable side while protecting short interactive prompts.
        constexpr int kDefault = 64;
        const char* value = std::getenv("MOLLM_METAL_SSD_PREFILL_MIN_TOKENS");
        if (!value || !*value) return kDefault;
        char* end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        return end != value && *end == '\0' && parsed >= 1 &&
                       parsed <= INT32_MAX
                   ? static_cast<int>(parsed)
                   : kDefault;
    }();
    return threshold;
}

bool metal_ssd_reload_weights() {
    static bool enabled = [] {
        const char* value = std::getenv("MOLLM_METAL_SSD_RELOAD_WEIGHTS");
        return value && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

void release_pool_tensor(BufferPool& pool, Tensor& t) {
    if (t.data && t.mem_type == MemoryType::POOLED && t.nbytes() > 0) {
        if (t.owner_id != 0 && t.owner_id != pool.id()) {
            std::fprintf(stderr,
                         "release_pool_tensor: owner mismatch for %p (tensor "
                         "owner=%u, pool=%u)\n",
                         t.data, t.owner_id, pool.id());
            assert(false && "release_pool_tensor owner mismatch");
            return;
        }
        pool.release(t.data, t.nbytes());
    }
    t.data = nullptr;
    t.mem_type = MemoryType::NONE;
    t.owner_id = 0;
    t.storage_id = 0;
}

bool is_view_op(OpType op) {
    return op == OpType::PERMUTE || op == OpType::SLICE;
}

void clear_tensor_storage(Tensor& tensor) {
    tensor.data = nullptr;
    tensor.device_data = nullptr;
    tensor.device_offset = 0;
    tensor.mem_type = MemoryType::NONE;
    tensor.owner_id = 0;
    tensor.storage_id = 0;
}

std::vector<uint8_t> find_borrowed_views(const Graph& graph) {
    const auto& tensors = graph.runtime.tensors;
    std::vector<uint8_t> borrowed(graph.nodes.size(), 0);

    for (const auto& node : graph.nodes) {
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT)
            continue;

        bool is_borrowed = is_view_op(node.op_type);
        if (node.op_type == OpType::RESHAPE && !node.inputs.empty()) {
            is_borrowed =
                tensors[node.id].shares_storage_with(tensors[node.inputs[0]]);
        }
        borrowed[node.id] = is_borrowed ? 1 : 0;
    }
    return borrowed;
}

void release_graph_temporaries(Graph& graph, Backend* backend) {
    auto& pool = graph.runtime.pool;
    auto& tensors = graph.runtime.tensors;
    const auto borrowed_view = find_borrowed_views(graph);

    for (auto& node : graph.nodes) {
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT)
            continue;
        if (!borrowed_view[node.id])
            continue;
        clear_tensor_storage(tensors[node.id]);
    }

    for (auto& node : graph.nodes) {
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT)
            continue;
        if (borrowed_view[node.id])
            continue;
        Tensor& t = tensors[node.id];
        if (t.data && t.mem_type == MemoryType::POOLED && t.nbytes() > 0) {
            if (backend)
                backend->free_output(t, &pool);
            else
                release_pool_tensor(pool, t);
        }
        clear_tensor_storage(t);
    }
}

void clear_graph_borrowed_views(Graph& graph) {
    auto& tensors = graph.runtime.tensors;
    const auto borrowed_view = find_borrowed_views(graph);

    for (auto& node : graph.nodes) {
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT)
            continue;
        if (!borrowed_view[node.id])
            continue;
        clear_tensor_storage(tensors[node.id]);
    }
}

void invalidate_workspace_key(ExecContext& ctx) {
    ctx.workspace_shape_valid = false;
    ctx.workspace_runtime_seq_len = -1;
    ctx.workspace_runtime_batch = -1;
    ctx.workspace_static_padded = false;
    ctx.workspace_padded_seq_len = -1;
}

void finish_graph_temporaries(Graph& graph, ExecContext& ctx) {
    if (ctx.reuse_static_workspace || ctx.reuse_same_shape_workspace) {
        clear_graph_borrowed_views(graph);
    } else {
        release_graph_temporaries(graph, ctx.backend);
        invalidate_workspace_key(ctx);
    }
}

Tensor copy_tensor_contiguous(const Tensor& src,
                              std::vector<uint8_t>& storage) {
    size_t es = src.element_size();
    size_t bytes = (size_t)src.nelements() * es;
    storage.resize(bytes);

    Tensor dst = Tensor::create(src.prec, MemoryType::EXTERNAL, src.shape[0],
                                src.shape[1], src.shape[2], src.shape[3],
                                storage.data());

    if (!src.data || bytes == 0)
        return dst;

    if (src.is_contiguous()) {
        std::memcpy(dst.data, src.data, bytes);
        return dst;
    }

    char* dp = static_cast<char*>(dst.data);
    const char* sp_base = static_cast<const char*>(src.data);
    size_t flat = 0;
    for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
        for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
            for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                const char* sp = sp_base + i1 * src.stride[1] +
                                 i2 * src.stride[2] + i3 * src.stride[3];
                std::memcpy(dp + flat * es, sp, (size_t)src.shape[0] * es);
                flat += (size_t)src.shape[0];
            }
        }
    }
    return dst;
}

} // namespace

void LLMEngine::set_profile_enabled(bool enabled) {
    exec_ctx_prefill_.profile_enabled = enabled;
    exec_ctx_decode_.profile_enabled = enabled;
}

void LLMEngine::reset_profiles() {
    reset_profile_stats(exec_ctx_prefill_);
    reset_profile_stats(exec_ctx_decode_);
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void LLMEngine::reset() {
    release_graph_temporaries(graph_decode_, exec_ctx_decode_.backend);
    invalidate_workspace_key(exec_ctx_decode_);
    clear_graph_borrowed_views(graph_prefill_);
    past_len_ = 0;
    // KV cache: only clear metadata header (current_seq_len = 0).
    // GDN state: zero the entire recurrent state buffer (it's small, ~256KB
    // per layer, and GDN reads stale state without causal mask protection).
    // GDN conv state: also zero (short conv uses conv_state for continuity).
    for (auto& cp : caches_) {
        if (cp.k)
            cache_meta(cp.k->data)->current_seq_len = 0;
        if (cp.v)
            cache_meta(cp.v->data)->current_seq_len = 0;
        if (cp.gdn_state) {
            size_t sz = (size_t)cp.gdn_v_dim * cp.gdn_k_dim * cp.gdn_num_heads *
                        sizeof(float);
            std::memset(cp.gdn_state->data, 0, sz);
        }
        if (cp.gdn_conv) {
            size_t sz = (size_t)cp.gdn_conv_groups * (cp.gdn_conv_kernel - 1) *
                        sizeof(float);
            std::memset(cp.gdn_conv->data, 0, sz);
        }
        if (cp.rwkv_state)
            std::memset(cp.rwkv_state->data, 0, cp.rwkv_state->nbytes());
        if (cp.rwkv_att_shift)
            std::memset(cp.rwkv_att_shift->data, 0,
                        cp.rwkv_att_shift->nbytes());
        if (cp.rwkv_ffn_shift)
            std::memset(cp.rwkv_ffn_shift->data, 0,
                        cp.rwkv_ffn_shift->nbytes());
    }
}

// ---------------------------------------------------------------------------
// embed
// ---------------------------------------------------------------------------

Tensor LLMEngine::embed(const std::vector<int>& token_ids, int pad_to) {
    int n = (int)token_ids.size();
    int seq_len = (pad_to > n) ? pad_to : n;

    if (embed_weight_ && embed_weight_->data) {
        int vocab_size = (int)embed_weight_->shape[0];
        int hidden_dim = (int)embed_weight_->shape[1];

        // Allocate from the prefill graph's pool so the buffer is tracked and
        // reused. Previously this used `new float[]` with MemoryType::OWNED,
        // but Tensor has no destructor — the buffer leaked every call.
        size_t nbytes = (size_t)hidden_dim * seq_len * sizeof(float);
        void* buf = graph_prefill_.runtime.pool.acquire(nbytes);
        Tensor t = Tensor::create(Precision::FP32, MemoryType::POOLED,
                                  hidden_dim, seq_len, 1, 1, buf);
        t.owner_id = graph_prefill_.runtime.pool.id();
        t.storage_id = graph_prefill_.runtime.pool.storage_id(buf);
        // Zero-init so padding positions (if any) are well-defined zeros.
        std::memset(t.data, 0, nbytes);

        if (embed_weight_->prec == Precision::FP16) {
            // Row-major FP16 — simple direct access
            const __fp16* embed_data =
                reinterpret_cast<const __fp16*>(embed_weight_->data);
            for (int s = 0; s < n; s++) {
                int tid = token_ids[s];
                if (tid < 0 || tid >= vocab_size)
                    tid = 0;
                float* dst = t.ptr<float>() + s * hidden_dim;
                for (int d = 0; d < hidden_dim; d++) {
                    dst[d] = (float)embed_data[tid * hidden_dim + d];
                }
            }
        } else {
            // Row-major FP32
            const float* embed_data = embed_weight_->ptr<float>();
            for (int s = 0; s < n; s++) {
                int tid = token_ids[s];
                if (tid < 0 || tid >= vocab_size)
                    tid = 0;
                float* dst = t.ptr<float>() + s * hidden_dim;
                for (int d = 0; d < hidden_dim; d++) {
                    dst[d] = embed_data[tid * hidden_dim + d];
                }
            }
        }
        return t;
    }

    int hidden_dim = 2048;
    void* buf = graph_prefill_.runtime.pool.acquire(hidden_dim * seq_len *
                                                    sizeof(float));
    Tensor t = Tensor::create(Precision::FP32, MemoryType::POOLED, hidden_dim,
                              seq_len, 1, 1, buf);
    t.owner_id = graph_prefill_.runtime.pool.id();
    t.storage_id = graph_prefill_.runtime.pool.storage_id(buf);
    std::memset(t.data, 0, t.nbytes());
    return t;
}

// ---------------------------------------------------------------------------
// run_lmhead
// ---------------------------------------------------------------------------

int LLMEngine::run_lmhead(const Tensor& hidden, int n_tokens) {
    if (!lm_head_weight_ || !lm_head_weight_->data)
        return 0;

    int vocab_size = (int)lm_head_weight_->shape[0];
    int hidden_dim = (int)lm_head_weight_->shape[1];
    int seq_len = (int)hidden.shape[1];

    // Read the last real token, not necessarily the last position
    // (which may be padded when graph seq_len > n_tokens).
    int last_pos =
        (n_tokens > 0 && n_tokens <= seq_len) ? n_tokens - 1 : seq_len - 1;

    // hidden is [hidden_dim, seq_len], we want [hidden_dim, 1] for matmul
    // lm_head_weight is [vocab_size, hidden_dim] — we use it as weight B
    // output will be [vocab_size, 1] — we take argmax

    // Create a view of the last hidden row as A: [hidden_dim, 1]
    Tensor A = hidden;
    A.shape[1] = 1;
    A.data = static_cast<char*>(hidden.data) +
             last_pos * hidden_dim * sizeof(float);
    A.compute_strides();

    // Output: [vocab_size, 1]
    void* c_buf =
        graph_prefill_.runtime.pool.acquire(vocab_size * sizeof(float));
    Tensor C = Tensor::create(Precision::FP32, MemoryType::POOLED, vocab_size,
                              1, 1, 1, c_buf);
    C.owner_id = graph_prefill_.runtime.pool.id();
    C.storage_id = graph_prefill_.runtime.pool.storage_id(c_buf);

#ifdef MOLLM_METAL
    if (metal_backend_ && lm_head_weight_->prec == Precision::FP16 &&
        lm_head_weight_->device_data) {
        as_metal(metal_backend_)
            ->lm_head_gemv(A.ptr<float>(), *lm_head_weight_, C.ptr<float>(),
                           vocab_size, hidden_dim);
    } else
#endif
    {
        kernel_matmul_fp32(A, *lm_head_weight_, C,
                           exec_ctx_decode_.thread_pool);
    }

    float* scores = C.ptr<float>();
    int token = 0;
    {
        // Keep sampling separate from the enclosing decode span.  In a Chrome
        // trace this forms a clear token boundary: decode N -> sampler ->
        // decode N+1.
        mollm_trace::ScopedEvent trace_sampler("inference", "sampler", {},
                                               "rail_response");
        token = sample_token(scores, vocab_size, cfg_.temperature, cfg_.top_k,
                             cfg_.top_p, &cfg_.seed);
    }

    release_pool_tensor(graph_prefill_.runtime.pool, C);

    return token;
}

std::vector<float> LLMEngine::run_lmhead_raw(const Tensor& hidden, int n_tokens,
                                             bool all_positions) {
    if (!lm_head_weight_ || !lm_head_weight_->data)
        return {};

    int vocab_size = (int)lm_head_weight_->shape[0];
    int hidden_dim = (int)lm_head_weight_->shape[1];
    int seq_len = (int)hidden.shape[1];

    int n_pos = all_positions ? n_tokens : 1;
    std::vector<float> logits(n_pos * vocab_size);

    for (int p = 0; p < n_pos; p++) {
        int pos = all_positions
                      ? p
                      : ((n_tokens > 0 && n_tokens <= seq_len) ? n_tokens - 1
                                                               : seq_len - 1);

        Tensor A = hidden;
        A.shape[1] = 1;
        A.data =
            static_cast<char*>(hidden.data) + pos * hidden_dim * sizeof(float);
        A.compute_strides();

        Tensor C =
            Tensor::create(Precision::FP32, MemoryType::EXTERNAL, vocab_size, 1,
                           1, 1, logits.data() + p * vocab_size);

#ifdef MOLLM_METAL
        if (metal_backend_ && lm_head_weight_->prec == Precision::FP16 &&
            lm_head_weight_->device_data) {
            as_metal(metal_backend_)
                ->lm_head_gemv(A.ptr<float>(), *lm_head_weight_, C.ptr<float>(),
                               vocab_size, hidden_dim);
        } else
#endif
        {
            kernel_matmul_fp32(A, *lm_head_weight_, C,
                               exec_ctx_decode_.thread_pool);
        }
    }

    return logits;
}

// ---------------------------------------------------------------------------
// rope / mask helpers
// ---------------------------------------------------------------------------

Tensor LLMEngine::build_causal_mask(int seq_len, int past_len) {
    int total = past_len + seq_len;
    void* buf =
        graph_prefill_.runtime.pool.acquire(total * seq_len * sizeof(float));
    Tensor mask = Tensor::create(Precision::FP32, MemoryType::POOLED, total,
                                 seq_len, 1, 1, buf);
    mask.owner_id = graph_prefill_.runtime.pool.id();
    mask.storage_id = graph_prefill_.runtime.pool.storage_id(buf);
    mollm::detail::fill_causal_mask(mask.ptr<float>(), seq_len, past_len);
    return mask;
}

void LLMEngine::generate_rope_cache(int seq_len, int start_pos, Tensor& cos,
                                    Tensor& sin) {
    int half = cfg_.rope_dim / 2;

    void* cb =
        graph_prefill_.runtime.pool.acquire(half * seq_len * sizeof(float));
    void* sb =
        graph_prefill_.runtime.pool.acquire(half * seq_len * sizeof(float));
    cos = Tensor::create(Precision::FP32, MemoryType::POOLED, half, seq_len, 1,
                         1, cb);
    sin = Tensor::create(Precision::FP32, MemoryType::POOLED, half, seq_len, 1,
                         1, sb);
    cos.owner_id = graph_prefill_.runtime.pool.id();
    sin.owner_id = graph_prefill_.runtime.pool.id();
    cos.storage_id = graph_prefill_.runtime.pool.storage_id(cb);
    sin.storage_id = graph_prefill_.runtime.pool.storage_id(sb);

    mollm::detail::fill_rope_cache(cos.ptr<float>(), sin.ptr<float>(), seq_len,
                                   start_pos, cfg_.rope_dim, cfg_.rope_theta);
}

// ---------------------------------------------------------------------------
// run_graph — feed inputs and execute a graph
// ---------------------------------------------------------------------------

Tensor LLMEngine::run_graph(Graph& graph, ExecContext& exec_ctx,
                            const Tensor& hidden, const Tensor& mask,
                            const Tensor& cos, const Tensor& sin) {
    if (moe_ssd_cache_)
        moe_ssd_cache_->begin_forward_pass();
    auto& tensors = graph.runtime.tensors;

    // Feed graph inputs by borrowing the caller-owned/helper tensors directly.
    // hidden/mask/cos/sin lifetime is managed by the caller; cache/state INPUTs
    // point at engine persistent storage and are set up at load time.
    for (auto& node : graph.nodes) {
        if (node.op_type != OpType::INPUT)
            continue;
        if (node.params.str.empty())
            continue;

        const std::string& name = node.params.str[0];
        Tensor* t = &tensors[node.id];

        bool is_boundary = false;
        if (name == "hidden") {
            *t = hidden;
            is_boundary = true;
        } else if (name == "mask") {
            *t = mask;
            is_boundary = true;
        } else if (name == "cos") {
            *t = cos;
            is_boundary = true;
        } else if (name == "sin") {
            *t = sin;
            is_boundary = true;
        }
        // cache_k/cache_v/gdn state are persistent INPUT tensors.

#ifdef MOLLM_METAL
        // Boundary inputs are produced on the host (embed/rope/mask); upload
        // their bytes into a device buffer so GPU kernels can read them.
        // Cache/state INPUTs are already device-resident (allocate_caches).
        if (metal_backend_ && exec_ctx.backend == metal_backend_.get() &&
            is_boundary && t->data) {
            as_metal(metal_backend_)
                ->upload_input(*t, name, t->data, t->nbytes());
        }
#else
        (void)is_boundary;
#endif
    }

#ifdef MOLLM_METAL
    if (metal_backend_ && exec_ctx.backend == metal_backend_.get())
        metal_backend_->begin_graph();
#endif
    execute_graph(exec_ctx);
#ifdef MOLLM_METAL
    if (metal_backend_ && exec_ctx.backend == metal_backend_.get())
        metal_backend_->end_graph();
#endif

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

void LLMEngine::prepare_metal_prefill_weights() {
#ifdef MOLLM_METAL
    if (!moe_ssd_cache_ || !metal_backend_ ||
        exec_ctx_prefill_.backend != metal_backend_.get())
        return;
    auto* metal = as_metal(metal_backend_);
    if (metal->has_weight_copies()) return;

    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type != OpType::CONSTANT || node.params.str.empty())
            continue;
        Tensor& t = graph_prefill_.runtime.tensors[node.id];
        // SSD expert aggregates deliberately have no resident source pointer.
        if (!t.rowmajor_data) continue;
        // CPU fallback kernels may use a packed/interleaved t.data. Temporarily
        // expose the original mmap bytes only while recreating the Metal copy.
        void* cpu_data = t.data;
        t.data = const_cast<void*>(t.rowmajor_data);
        t.device_data = nullptr;
        t.device_offset = 0;
        metal->wrap_weight(t);
        t.data = cpu_data;
        const bool aggregate_expert =
            node.params.str[0].find("_experts_") != std::string::npos;
        metal->wrap_weight_int4_g128(t, aggregate_expert);
    }
#endif
}

void LLMEngine::release_metal_prefill_weights() {
#ifdef MOLLM_METAL
    if (!moe_ssd_cache_ || !metal_backend_ ||
        exec_ctx_prefill_.backend != metal_backend_.get())
        return;
    as_metal(metal_backend_)->release_weight_copies();
    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type != OpType::CONSTANT || node.params.str.empty())
            continue;
        Tensor& t = graph_prefill_.runtime.tensors[node.id];
        // KV caches and recurrent state are also CONSTANT nodes, but have no
        // mmap-backed row-major weight source and must stay device-resident.
        if (!t.rowmajor_data) continue;
        t.device_data = nullptr;
        t.device_offset = 0;
    }
#endif
}

void LLMEngine::release_prefill_buffers() {
    release_graph_temporaries(graph_prefill_, exec_ctx_prefill_.backend);
    invalidate_workspace_key(exec_ctx_prefill_);
}

int LLMEngine::prefill(const std::vector<int>& token_ids) {
    mollm_trace::ScopedEvent trace_prefill("inference", "prefill");
    int n = (int)token_ids.size();
    if (n == 0)
        return -1;

    Backend* saved_prefill_backend = exec_ctx_prefill_.backend;
    bool short_ssd_cpu_prefill = false;
#ifdef MOLLM_METAL
    const bool is_ssd_metal =
        moe_ssd_cache_ && metal_backend_ &&
        saved_prefill_backend == metal_backend_.get();
    const bool metal_weights_ready =
        is_ssd_metal && as_metal(metal_backend_)->has_weight_copies();
    short_ssd_cpu_prefill =
        is_ssd_metal &&
        (n < metal_ssd_prefill_min_tokens() ||
         (!metal_weights_ready && !metal_ssd_reload_weights()));
    if (short_ssd_cpu_prefill) {
        // Small-M GPU kernels plus one GPU→CPU synchronization per MoE layer
        // lose to the CPU path. Drop dense Metal copies before CPU work so UMA
        // pressure cannot slow the SSD expert kernels.
        release_metal_prefill_weights();
        release_graph_temporaries(graph_prefill_, saved_prefill_backend);
        invalidate_workspace_key(exec_ctx_prefill_);
        exec_ctx_prefill_.backend = &cpu_backend_;
    } else {
        prepare_metal_prefill_weights();
    }
#else
    prepare_metal_prefill_weights();
#endif
    auto finish_prefill_phase = [&] {
        // Hybrid decode is CPU-only, so no prefill workspace is useful after
        // the last chunk. Releasing it also makes backend switching explicit.
        if (moe_ssd_cache_) {
            release_graph_temporaries(graph_prefill_,
                                      exec_ctx_prefill_.backend);
            invalidate_workspace_key(exec_ctx_prefill_);
        }
        if (!short_ssd_cpu_prefill)
            release_metal_prefill_weights();
        exec_ctx_prefill_.backend = saved_prefill_backend;
    };

    // Determine the graph's expected seq_len from its hidden input shape.
    int graph_seq_len = 1;
    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty() &&
            node.params.str[0] == "hidden") {
            graph_seq_len = (int)node.out_shape[1];
            break;
        }
    }

    // Chunked prefill: split long prompts into graph_seq_len-sized chunks.
    // Each chunk is processed independently, appending to the KV cache.
    // past_len_ can exceed graph_seq_len (cache supports n_ctx); only each
    // chunk's token count must be <= graph_seq_len.
    int offset = 0;
    int last_token = -1;
    while (offset < n) {
        int remaining = n - offset;
        if (past_len_ >= cfg_.n_ctx) {
            fprintf(
                stderr,
                "prefill: context full (past=%d >= n_ctx=%d). Use /reset.\n",
                past_len_, cfg_.n_ctx);
            finish_prefill_phase();
            return -1;
        }
        int chunk_size = std::min(remaining, graph_seq_len);
        std::vector<int> chunk(token_ids.begin() + offset,
                               token_ids.begin() + offset + chunk_size);
        last_token = prefill_chunk(chunk, past_len_);
        if (last_token < 0) {
            finish_prefill_phase();
            return -1;
        }
        offset += chunk_size;
    }
    finish_prefill_phase();
    return last_token;
}

Tensor LLMEngine::prefill_hidden(const std::vector<int>& token_ids) {
    mollm_trace::ScopedEvent trace_prefill("inference", "prefill_hidden");
    int n = (int)token_ids.size();
    if (n == 0)
        return Tensor();

    Backend* saved_prefill_backend = exec_ctx_prefill_.backend;
    bool short_ssd_cpu_prefill = false;
#ifdef MOLLM_METAL
    const bool is_ssd_metal =
        moe_ssd_cache_ && metal_backend_ &&
        saved_prefill_backend == metal_backend_.get();
    const bool metal_weights_ready =
        is_ssd_metal && as_metal(metal_backend_)->has_weight_copies();
    short_ssd_cpu_prefill =
        is_ssd_metal &&
        (n < metal_ssd_prefill_min_tokens() ||
         (!metal_weights_ready && !metal_ssd_reload_weights()));
    if (short_ssd_cpu_prefill) {
        release_metal_prefill_weights();
        release_graph_temporaries(graph_prefill_, saved_prefill_backend);
        invalidate_workspace_key(exec_ctx_prefill_);
        exec_ctx_prefill_.backend = &cpu_backend_;
    } else {
        prepare_metal_prefill_weights();
    }
#else
    prepare_metal_prefill_weights();
#endif

    int graph_seq_len = 1;
    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty() &&
            node.params.str[0] == "hidden") {
            graph_seq_len = (int)node.out_shape[1];
            break;
        }
    }

    // Dynamic shape mode: chunked prefill is supported via repeated calls to
    // prefill_hidden (each chunk appends to KV cache). No padding — the
    // graph's SEQ dims are filled with the actual chunk size n at runtime.

    // DYNAMIC mode: no padding, runtime fills SEQ/MUL/ADD dims via DimExpr.
    // STATIC_PADDED mode: pad short prompts to graph_seq_len (A/B comparison).
    const bool use_padding = cfg_.static_padded && n < graph_seq_len;
    Tensor h;
    Tensor cos, sin;
    Tensor mask;

    if (use_padding) {
        exec_ctx_prefill_.runtime_seq_len = n;
        exec_ctx_prefill_.static_padded = true;
        exec_ctx_prefill_.padded_seq_len = graph_seq_len;
        inject_runtime_shapes(exec_ctx_prefill_);
        h = embed(token_ids, graph_seq_len); // zero-padded to graph_seq_len
        h.shape[1] = graph_seq_len;
        h.compute_strides();
        generate_rope_cache(graph_seq_len, past_len_, cos, sin);
        mask = build_causal_mask(graph_seq_len, past_len_);
    } else {
        exec_ctx_prefill_.runtime_seq_len = n;
        exec_ctx_prefill_.static_padded = false;
        exec_ctx_prefill_.padded_seq_len = -1;
        inject_runtime_shapes(exec_ctx_prefill_);
        h = embed(token_ids);
        h.shape[1] = n;
        h.compute_strides();
        generate_rope_cache(n, past_len_, cos, sin);
        mask = build_causal_mask(n, past_len_);
    }

    // n_real_tokens injection is done by inject_runtime_shapes() above.

    // Set cache metadata
    for (auto& cp : caches_) {
        if (cp.k)
            cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v)
            cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    mollm_set_matmul_profile_phase("prefill_graph");
    Tensor out =
        run_graph(graph_prefill_, exec_ctx_prefill_, h, mask, cos, sin);
    mollm_set_matmul_profile_phase("unscoped");
    Tensor copied = copy_tensor_contiguous(out, hidden_output_copy_);
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);

    past_len_ += n;

    for (auto& cp : caches_) {
        if (cp.k)
            cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v)
            cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    finish_graph_temporaries(graph_prefill_, exec_ctx_prefill_);
    if (moe_ssd_cache_) {
        release_graph_temporaries(graph_prefill_,
                                  exec_ctx_prefill_.backend);
        invalidate_workspace_key(exec_ctx_prefill_);
    }
    if (!short_ssd_cpu_prefill)
        release_metal_prefill_weights();
    exec_ctx_prefill_.backend = saved_prefill_backend;
    return copied;
}

int LLMEngine::prefill_chunk(const std::vector<int>& token_ids, int past) {
    int n = (int)token_ids.size();
    if (n == 0)
        return -1;

    // Determine the graph's expected seq_len from its hidden input shape.
    int graph_seq_len = 1;
    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty() &&
            node.params.str[0] == "hidden") {
            graph_seq_len = (int)node.out_shape[1];
            break;
        }
    }

    // Check that this chunk fits in the prefill graph's static seq_len.
    // `past` can be >= graph_seq_len (cache supports n_ctx=4096); only the
    // chunk size n must be <= graph_seq_len.
    if (n > graph_seq_len) {
        fprintf(stderr, "prefill_chunk: n=%d > graph_seq_len=%d\n", n,
                graph_seq_len);
        return -1;
    }

    // DYNAMIC mode: no padding. runtime fills SEQ/MUL/ADD dims via DimExpr
    // evaluation against runtime_seq_len. Symbolic reshape handles N*seq.
    //
    // STATIC_PADDED mode: pad short chunks (n < graph_seq_len) to
    // graph_seq_len. Stateful ops (GDN/SHORTCONV) receive n_real via params to
    // skip padding positions. Full chunks (n == graph_seq_len) skip padding
    // (identical work).
    const bool use_padding = cfg_.static_padded && n < graph_seq_len;
    Tensor h;
    Tensor cos, sin;
    Tensor mask;

    if (use_padding) {
        exec_ctx_prefill_.runtime_seq_len = n; // real token count
        exec_ctx_prefill_.static_padded = true;
        exec_ctx_prefill_.padded_seq_len = graph_seq_len;
        inject_runtime_shapes(exec_ctx_prefill_);
        h = embed(token_ids, graph_seq_len); // zero-padded to graph_seq_len
        h.shape[1] = graph_seq_len;
        h.compute_strides();
        generate_rope_cache(graph_seq_len, past, cos, sin);
        mask = build_causal_mask(graph_seq_len, past);
    } else {
        exec_ctx_prefill_.runtime_seq_len = n;
        exec_ctx_prefill_.static_padded = false;
        exec_ctx_prefill_.padded_seq_len = -1;
        inject_runtime_shapes(exec_ctx_prefill_);
        h = embed(token_ids);
        h.shape[1] = n;
        h.compute_strides();
        generate_rope_cache(n, past, cos, sin);
        mask = build_causal_mask(n, past);
    }

    // n_real_tokens injection is now done by inject_runtime_shapes() above.

    // Set cache metadata so SDPA knows the existing context length.
    for (auto& cp : caches_) {
        if (cp.k)
            cache_meta(cp.k->data)->current_seq_len = (uint64_t)past;
        if (cp.v)
            cache_meta(cp.v->data)->current_seq_len = (uint64_t)past;
    }

    mollm_set_matmul_profile_phase("prefill_graph");
    Tensor out =
        run_graph(graph_prefill_, exec_ctx_prefill_, h, mask, cos, sin);

    past_len_ = past + n;

    // Update cache metadata for decode (past + current prefill tokens)
    for (auto& cp : caches_) {
        if (cp.k)
            cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v)
            cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    // Cache migration (prefill→decode graph) is done once at load time.
    // Both graphs share the same physical cache buffers.

    mollm_set_matmul_profile_phase("prefill_lmhead");
    int token = run_lmhead(out, n);
    mollm_set_matmul_profile_phase("unscoped");
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);
    finish_graph_temporaries(graph_prefill_, exec_ctx_prefill_);
    return token;
}

int LLMEngine::decode(int token_id) {
    mollm_trace::ScopedEvent trace_decode("inference", "decode");
    Tensor h = embed({token_id});
    h.shape[1] = 1;
    h.compute_strides();

    Tensor cos, sin;
    generate_rope_cache(1, past_len_, cos, sin);

    Tensor mask = build_causal_mask(1, past_len_);

    mollm_set_matmul_profile_phase("decode_graph");
    Tensor out = run_graph(graph_decode_, exec_ctx_decode_, h, mask, cos, sin);

    past_len_++;

    // Update cache metadata
    for (auto& cp : caches_) {
        if (cp.k)
            cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v)
            cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    mollm_set_matmul_profile_phase("decode_lmhead");
    int token = run_lmhead(out);
    mollm_set_matmul_profile_phase("unscoped");
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);
    finish_graph_temporaries(graph_decode_, exec_ctx_decode_);
    return token;
}

Tensor LLMEngine::decode_hidden(int token_id) {
    mollm_trace::ScopedEvent trace_decode("inference", "decode_hidden");
    Tensor h = embed({token_id});
    h.shape[1] = 1;
    h.compute_strides();

    // Decode graph is all-STATIC (seq=1); no dynamic shape injection needed.
    exec_ctx_decode_.runtime_seq_len = -1;
    exec_ctx_decode_.static_padded = false;
    exec_ctx_decode_.padded_seq_len = -1;

    Tensor cos, sin;
    generate_rope_cache(1, past_len_, cos, sin);

    Tensor mask = build_causal_mask(1, past_len_);

    mollm_set_matmul_profile_phase("decode_graph");
    Tensor out = run_graph(graph_decode_, exec_ctx_decode_, h, mask, cos, sin);
    mollm_set_matmul_profile_phase("unscoped");
    Tensor copied = copy_tensor_contiguous(out, hidden_output_copy_);
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);

    past_len_++;

    for (auto& cp : caches_) {
        if (cp.k)
            cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v)
            cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    finish_graph_temporaries(graph_decode_, exec_ctx_decode_);
    return copied;
}
