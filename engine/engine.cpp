#include "engine/engine.h"
#include "engine/logical_prefill.h"
#include "runtime/cuda_host_memory_budget.h"
#include "engine/input_prep.h"
#include "engine/recurrent_checkpoint.h"
#include "engine/sampler.h"
#include "core/fp16.h"
#include "runtime/host_compute.h"
#include "runtime/trace.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <charconv>
#include <cstddef>
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
    tensor.device.buffer = nullptr;
    tensor.device.offset = 0;
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
        } else if (node.op_type == OpType::CONTIGUOUS &&
                   !node.inputs.empty()) {
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
                              std::vector<uint8_t>& storage,
                              Backend* backend) {
    size_t es = src.element_size();
    size_t bytes = (size_t)src.nelements() * es;
    storage.resize(bytes);

    Tensor dst = Tensor::create(src.prec, MemoryType::EXTERNAL, src.shape[0],
                                src.shape[1], src.shape[2], src.shape[3],
                                storage.data());

    if ((!src.data && !src.device.buffer) || bytes == 0)
        return dst;

    if (src.is_contiguous()) {
        if (backend) {
            if (!backend->copy_to_host(src, dst.data, bytes))
                return Tensor();
        } else {
            std::memcpy(dst.data, src.data, bytes);
        }
        return dst;
    }

    std::vector<uint8_t> source_storage(src.view_span_bytes());
    if (backend) {
        if (!backend->copy_to_host(
                src, source_storage.data(), source_storage.size()))
            return Tensor();
    } else {
        std::memcpy(source_storage.data(), src.data, source_storage.size());
    }
    char* dp = static_cast<char*>(dst.data);
    const char* sp_base =
        reinterpret_cast<const char*>(source_storage.data());
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

// A retained output may be wider than the ordinary lm_head hidden state.
bool copy_declared_mtp_state(Graph& graph, Backend* backend, int tokens,
                             std::vector<uint8_t>& storage, Tensor& result) {
    result = Tensor{};
    const auto found = graph.metadata.find("mtp_state_output_node");
    if (found == graph.metadata.end()) return true;
    uint32_t id = 0;
    const auto& value = found->second;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), id);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        id >= graph.runtime.tensors.size() ||
        std::find(graph.graph_outputs.begin(), graph.graph_outputs.end(), id) ==
            graph.graph_outputs.end()) {
        fprintf(stderr, "MTP: invalid retained-state output metadata\n");
        return false;
    }
    const Tensor& state = graph.runtime.tensors[id];
    if ((!state.data && !state.device.buffer) || state.prec != Precision::FP32 ||
        state.shape[0] <= 0 || state.shape[1] < tokens ||
        state.shape[2] != 1 || state.shape[3] != 1) {
        fprintf(stderr, "MTP: invalid retained-state tensor\n");
        return false;
    }
    Tensor real_state = state;
    real_state.shape[1] = tokens;
    result = copy_tensor_contiguous(real_state, storage, backend);
    return result.data != nullptr;
}

} // namespace

void LLMEngine::set_profile_enabled(bool enabled) {
    exec_ctx_prefill_.profile_enabled = enabled;
    exec_ctx_decode_.profile_enabled = enabled;
    exec_ctx_mtp_.profile_enabled = enabled;
}

void LLMEngine::reset_profiles() {
    reset_profile_stats(exec_ctx_prefill_);
    reset_profile_stats(exec_ctx_decode_);
    reset_profile_stats(exec_ctx_mtp_);
}

Backend* LLMEngine::persistent_backend() const {
    return accelerator_backend_
        ? static_cast<Backend*>(accelerator_backend_.get())
        : exec_ctx_prefill_.backend;
}

bool LLMEngine::set_cache_lengths(
    std::vector<CachePair>& caches, uint64_t length) {
    Backend* backend = persistent_backend();
    if (!backend)
        return false;
    for (auto& cache : caches) {
        for (Tensor* tensor : {cache.k, cache.v}) {
            if (tensor && !backend->copy_from_host(
                    &length, *tensor, sizeof(length),
                    offsetof(CacheMetadata, current_seq_len)))
                return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void LLMEngine::reset() {
    release_graph_temporaries(graph_decode_, exec_ctx_decode_.backend);
    invalidate_workspace_key(exec_ctx_decode_);
    release_graph_temporaries(graph_mtp_, exec_ctx_mtp_.backend);
    invalidate_workspace_key(exec_ctx_mtp_);
    clear_graph_borrowed_views(graph_prefill_);
    past_len_ = 0;
    mtp_past_len_ = 0;
    mtp_stats_ = MtpStats{};
    target_mtp_state_ = Tensor{};
    draft_mtp_state_ = Tensor{};
    std::fill(mtp_pending_hidden_.begin(), mtp_pending_hidden_.end(), 0.0f);
    rope_position_delta_ = 0;
    multimodal_position_ids_.clear();
    active_vision_ = nullptr;
    active_image_token_id_ = -1;
    active_vision_cursor_ = 0;
    sampler_.reset();
    // KV cache: only clear metadata header (current_seq_len = 0).
    // GDN state: zero the entire recurrent state buffer (it's small, ~256KB
    // per layer, and GDN reads stale state without causal mask protection).
    // GDN conv state: also zero (short conv uses conv_state for continuity).
    Backend* backend = persistent_backend();
    bool storage_reset = backend
        ? set_cache_lengths(caches_, 0) && set_cache_lengths(mtp_caches_, 0)
        : caches_.empty() && mtp_caches_.empty() && auxiliary_states_.empty();
    for (auto& cp : caches_) {
        if (cp.gdn_state) {
            size_t sz = (size_t)cp.gdn_v_dim * cp.gdn_k_dim * cp.gdn_num_heads *
                        sizeof(float);
            if (!backend || !backend->zero_tensor(*cp.gdn_state, sz))
                storage_reset = false;
        }
        if (cp.gdn_conv) {
            size_t sz = (size_t)cp.gdn_conv_groups * (cp.gdn_conv_kernel - 1) *
                        sizeof(float);
            if (!backend || !backend->zero_tensor(*cp.gdn_conv, sz))
                storage_reset = false;
        }
        for (Tensor* state : {
                 cp.rwkv_state, cp.rwkv_att_shift, cp.rwkv_ffn_shift}) {
            if (state &&
                (!backend || !backend->zero_tensor(*state, state->nbytes())))
                storage_reset = false;
        }
    }
    for (const auto& entry : auxiliary_states_) {
        Tensor* state = entry.second;
        if (state && state->data &&
            (!backend || !backend->zero_tensor(*state, state->nbytes())))
            storage_reset = false;
    }
    if (!storage_reset)
        std::fprintf(stderr, "Engine: failed to reset persistent state\n");
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
        if (active_vision_) {
            if (active_vision_->hidden_size != hidden_dim) {
                fprintf(stderr,
                        "embed: vision hidden size %d does not match text "
                        "hidden size %d\n",
                        active_vision_->hidden_size, hidden_dim);
            } else {
                for (int s = 0; s < n; ++s) {
                    if (token_ids[s] != active_image_token_id_) continue;
                    if (active_vision_cursor_ >= active_vision_->tokens) {
                        fprintf(stderr,
                                "embed: more image placeholder tokens than "
                                "vision embeddings\n");
                        break;
                    }
                    const float* src =
                        active_vision_->values.data() +
                        static_cast<size_t>(active_vision_cursor_) * hidden_dim;
                    std::memcpy(t.ptr<float>() +
                                    static_cast<size_t>(s) * hidden_dim,
                                src,
                                static_cast<size_t>(hidden_dim) *
                                    sizeof(float));
                    ++active_vision_cursor_;
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

int LLMEngine::run_lmhead(const Tensor& hidden, int n_tokens,
                          bool finish_accelerator_graph) {
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

    if (finish_accelerator_graph && sampler_.uses_plain_argmax() &&
        accelerator_backend_->supports_lm_head_argmax(*lm_head_weight_)) {
        const int token =
            accelerator_backend_->lm_head_argmax_device_and_end_graph(
                hidden, static_cast<size_t>(last_pos) * hidden_dim,
                *lm_head_weight_, vocab_size, hidden_dim);
        return accelerator_backend_->dispatch_failed() ? -1 : token;
    }

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

    if (finish_accelerator_graph) {
        assert(accelerator_backend_ && hidden.device.buffer &&
               lm_head_weight_->device.buffer);
        accelerator_backend_->lm_head_gemv_device_and_end_graph(
                hidden, (size_t)last_pos*(size_t)hidden_dim,
                *lm_head_weight_, C.ptr<float>(), vocab_size, hidden_dim);
    } else if (accelerator_backend_ &&
               accelerator_backend_->supports_lm_head(*lm_head_weight_)) {
        std::vector<float> host_activation;
        const float* activation = A.ptr<float>();
        if (hidden.device.buffer) {
            host_activation.resize(static_cast<size_t>(hidden_dim));
            if (!accelerator_backend_->copy_to_host(
                    hidden, host_activation.data(),
                    host_activation.size() * sizeof(float),
                    static_cast<size_t>(last_pos) * hidden_dim *
                        sizeof(float))) {
                release_pool_tensor(graph_prefill_.runtime.pool, C);
                return -1;
            }
            activation = host_activation.data();
        }
        accelerator_backend_->lm_head_gemv(
            activation, *lm_head_weight_, C.ptr<float>(), vocab_size,
            hidden_dim);
    } else {
        host_matmul_fp32(A, *lm_head_weight_, C,
                           exec_ctx_decode_.thread_pool);
    }

    if (accelerator_backend_ && accelerator_backend_->dispatch_failed()) {
        release_pool_tensor(graph_prefill_.runtime.pool, C);
        return -1;
    }

    float* scores = C.ptr<float>();
    int token = 0;
    {
        // Keep sampling separate from the enclosing decode span.  In a Chrome
        // trace this forms a clear token boundary: decode N -> sampler ->
        // decode N+1.
        mollm_trace::ScopedEvent trace_sampler("inference", "sampler", {},
                                               "rail_response");
        token = sampler_.sample(scores, vocab_size);
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

    // Speculative verification needs logits for every position in a tiny
    // incremental-prefill batch.  Submit one shared-weight small-M W4
    // projection instead of M standalone GEMVs (and M command-buffer waits).
    if (all_positions && n_pos >= 2 && n_pos <= 4 && hidden.is_contiguous() &&
        cfg_.device == Device::METAL && accelerator_backend_ &&
        accelerator_backend_->supports_lm_head_small_batch(
            *lm_head_weight_)) {
        if (accelerator_backend_->lm_head_small_batch(
                static_cast<const float*>(hidden.data), *lm_head_weight_,
                logits.data(), n_pos, vocab_size, hidden_dim))
            return logits;
    }

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

        if (accelerator_backend_ &&
            accelerator_backend_->supports_lm_head(*lm_head_weight_)) {
            std::vector<float> host_activation;
            const float* activation = A.ptr<float>();
            if (hidden.device.buffer) {
                host_activation.resize(static_cast<size_t>(hidden_dim));
                if (!accelerator_backend_->copy_to_host(
                        hidden, host_activation.data(),
                        host_activation.size() * sizeof(float),
                        static_cast<size_t>(pos) * hidden_dim *
                            sizeof(float)))
                    return {};
                activation = host_activation.data();
            }
            accelerator_backend_->lm_head_gemv(
                activation, *lm_head_weight_, C.ptr<float>(), vocab_size,
                hidden_dim);
        } else {
            host_matmul_fp32(A, *lm_head_weight_, C,
                               exec_ctx_decode_.thread_pool);
        }
    }

    return logits;
}

// ---------------------------------------------------------------------------
// rope / mask helpers
// ---------------------------------------------------------------------------

Tensor LLMEngine::build_causal_mask(int seq_len, int past_len,
                                    bool initialize) {
    int total = past_len + seq_len;
    void* buf =
        graph_prefill_.runtime.pool.acquire(total * seq_len * sizeof(float));
    Tensor mask = Tensor::create(Precision::FP32, MemoryType::POOLED, total,
                                 seq_len, 1, 1, buf);
    mask.owner_id = graph_prefill_.runtime.pool.id();
    mask.storage_id = graph_prefill_.runtime.pool.storage_id(buf);
    if (initialize)
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

    mollm::detail::fill_rope_cache(
        cos.ptr<float>(), sin.ptr<float>(), seq_len,
        start_pos + rope_position_delta_, cfg_.rope_dim, cfg_.rope_theta);
}

// ---------------------------------------------------------------------------
// run_graph — feed inputs and execute a graph
// ---------------------------------------------------------------------------

Tensor LLMEngine::run_graph(Graph& graph, ExecContext& exec_ctx,
                            const Tensor& hidden, const Tensor& mask,
                            const Tensor& cos, const Tensor& sin,
                            const Tensor* token_ids,
                            bool defer_accelerator_end,
                            const Tensor* target_hidden,
                            int position,
                            int stop_after_node_index) {
    if (moe_ssd_cache_)
        moe_ssd_cache_->begin_forward_pass();
    auto& tensors = graph.runtime.tensors;
    int32_t graph_position = static_cast<int32_t>(
        position >= 0 ? position : past_len_);
    Tensor position_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL,
        1, 1, 1, 1, &graph_position);
    int32_t graph_n_tokens = exec_ctx.runtime_seq_len > 0
        ? static_cast<int32_t>(exec_ctx.runtime_seq_len)
        : static_cast<int32_t>(hidden.shape[1]);
    Tensor n_tokens_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL,
        1, 1, 1, 1, &graph_n_tokens);

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
        } else if (name == "target_hidden" && target_hidden) {
            *t = *target_hidden;
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
        } else if (name == "token_ids" && token_ids) {
            *t = *token_ids;
            is_boundary = true;
        } else if (name == "position") {
            *t = position_tensor;
            is_boundary = true;
        } else if (name == "n_tokens") {
            *t = n_tokens_tensor;
            is_boundary = true;
        }
        // cache_k/cache_v/gdn state are persistent INPUT tensors.

        // Boundary inputs are produced on the host (embed/rope/mask); upload
        // their bytes into a device buffer so GPU kernels can read them.
        // Cache/state INPUTs are already device-resident (allocate_caches).
        if (accelerator_backend_ &&
            exec_ctx.backend == accelerator_backend_.get() &&
            is_boundary && t->data) {
            accelerator_backend_->upload_input(
                *t, name, t->data, t->nbytes());
        }
    }

    if (accelerator_backend_ &&
        exec_ctx.backend == accelerator_backend_.get())
        accelerator_backend_->begin_graph();
    execute_graph(exec_ctx, stop_after_node_index);
    if (accelerator_backend_ &&
        exec_ctx.backend == accelerator_backend_.get() &&
        !defer_accelerator_end)
        accelerator_backend_->end_graph();

    if (exec_ctx.backend->dispatch_failed())
        exec_ctx.execution_failed = true;

    if (exec_ctx.execution_failed)
        return Tensor();

    if (stop_after_node_index >= 0 &&
        stop_after_node_index < static_cast<int>(graph.nodes.size())) {
        return tensors[graph.nodes[stop_after_node_index].id];
    }
    if (!graph.graph_outputs.empty()) {
        uint32_t out_id = graph.graph_outputs.back();
        return tensors[out_id];
    }

    fprintf(stderr, "Engine: no graph output found\n");
    return Tensor();
}

void LLMEngine::set_cache_length(std::vector<CachePair>& caches, int length) {
    if (!set_cache_lengths(caches, static_cast<uint64_t>(length)))
        std::fprintf(stderr, "Engine: failed to update cache length\n");
}

Tensor LLMEngine::run_mtp_tokens(const std::vector<int>& token_ids,
                                  const Tensor& target_hidden,
                                  int position,
                                  int* draft_token) {
    Tensor output;
    if (!execute_mtp_tokens(token_ids, target_hidden, position, false,
                            &output, draft_token))
        return Tensor();
    return output;
}

bool LLMEngine::update_mtp_cache(const std::vector<int>& token_ids,
                                  const Tensor& target_hidden,
                                  int position) {
    return execute_mtp_tokens(token_ids, target_hidden, position, true,
                              nullptr, nullptr);
}

bool LLMEngine::execute_mtp_tokens(
    const std::vector<int>& token_ids, const Tensor& target_hidden,
    int position, bool cache_only, Tensor* hidden_output,
    int* draft_token) {
    if (!has_mtp() || token_ids.empty() || !target_hidden.data ||
        target_hidden.shape[1] != static_cast<int64_t>(token_ids.size())) {
        return false;
    }
    for (const auto& node : graph_mtp_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty() &&
            node.params.str[0] == "target_hidden" &&
            (target_hidden.prec != Precision::FP32 ||
             target_hidden.shape[0] != node.out_shape[0] ||
             target_hidden.shape[2] != 1 || target_hidden.shape[3] != 1)) {
            fprintf(stderr, "MTP: target state does not match draft input width\n");
            return false;
        }
    }
    if (draft_token)
        *draft_token = -1;

    int stop_after_node_index = -1;
    if (cache_only) {
        for (size_t i = 0; i < graph_mtp_.nodes.size(); ++i) {
            if (graph_mtp_.nodes[i].op_type == OpType::SDPA) {
                stop_after_node_index = static_cast<int>(i);
                break;
            }
        }
        if (stop_after_node_index < 0)
            return false;
    }

    const int n = static_cast<int>(token_ids.size());
    exec_ctx_mtp_.runtime_seq_len = n;
    exec_ctx_mtp_.static_padded = false;
    exec_ctx_mtp_.padded_seq_len = -1;
    inject_runtime_shapes(exec_ctx_mtp_);

    Tensor token_hidden = embed(token_ids);
    token_hidden.shape[1] = n;
    token_hidden.compute_strides();
    Tensor cos, sin;
    generate_rope_cache(n, position, cos, sin);
    Tensor mask = build_causal_mask(n, position);
    set_cache_length(mtp_caches_, position);

    static const bool cuda_mtp_device_argmax = [] {
        const char* value = std::getenv("MOLLM_CUDA_MTP_DEVICE_ARGMAX");
        return value && std::strcmp(value, "1") == 0;
    }();
    const bool fuse_accelerator_lm_head =
        !cache_only && draft_token &&
        (cfg_.device == Device::METAL ||
         (cfg_.device == Device::CUDA && cuda_mtp_device_argmax)) && accelerator_backend_ &&
        exec_ctx_mtp_.backend == accelerator_backend_.get() &&
        lm_head_weight_ &&
        accelerator_backend_->supports_lm_head_argmax(*lm_head_weight_);
    Tensor* device_hidden_copy = nullptr;
    if (fuse_accelerator_lm_head) {
        const int hidden = static_cast<int>(lm_head_weight_->shape[1]);
        if (!mtp_draft_hidden_device_.device.buffer) {
            mtp_draft_hidden_device_ = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL,
                hidden, 1, 1, 1, nullptr);
            accelerator_backend_->alloc_persistent(
                mtp_draft_hidden_device_,
                static_cast<size_t>(hidden) * sizeof(float));
        }
        device_hidden_copy = &mtp_draft_hidden_device_;
    }
    Tensor out = run_graph(
        graph_mtp_, exec_ctx_mtp_, token_hidden, mask, cos, sin,
        nullptr, fuse_accelerator_lm_head, &target_hidden, position,
        stop_after_node_index);

    if (fuse_accelerator_lm_head) {
        if (out.data && out.device.buffer) {
            const int vocab = static_cast<int>(lm_head_weight_->shape[0]);
            const int hidden = static_cast<int>(lm_head_weight_->shape[1]);
            // Keep projection and top-1 in the graph command stream.  MTP uses
            // greedy top-k=1 drafts, so copying the full vocabulary to CPU
            // would add a host transfer, allocation, and scan every depth.
            *draft_token =
                accelerator_backend_->lm_head_argmax_device_and_end_graph(
                out, static_cast<size_t>(n - 1) * hidden,
                *lm_head_weight_, vocab, hidden, 0, device_hidden_copy);
        } else {
            // run_graph() deliberately left the accelerator graph open for the
            // fused tail; close it even on a malformed/failed graph output.
            accelerator_backend_->end_graph();
        }
    }
    Tensor copied;
    bool retained_state_ok = true;
    if (!cache_only && out.data &&
        !exec_ctx_mtp_.backend->dispatch_failed()) {
        retained_state_ok = copy_declared_mtp_state(
                graph_mtp_, exec_ctx_mtp_.backend, n,
                draft_mtp_state_copy_, draft_mtp_state_);
        if (fuse_accelerator_lm_head && device_hidden_copy &&
            device_hidden_copy->device.buffer)
            copied = *device_hidden_copy;
        else
            copied = copy_tensor_contiguous(
                out, mtp_hidden_output_copy_, exec_ctx_mtp_.backend);
    }
    const bool succeeded = out.data && retained_state_ok &&
        !exec_ctx_mtp_.backend->dispatch_failed() &&
        (cache_only || copied.data);

    release_pool_tensor(graph_prefill_.runtime.pool, token_hidden);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);
    finish_graph_temporaries(graph_mtp_, exec_ctx_mtp_);
    if (!succeeded)
        return false;

    mtp_past_len_ = position + n;
    set_cache_length(mtp_caches_, mtp_past_len_);
    if (hidden_output)
        *hidden_output = copied;
    return true;
}

bool LLMEngine::sync_mtp(const std::vector<int>& token_ids,
                          const Tensor& verified_hidden, int position,
                          int preserved_prefix) {
    if (!has_mtp() || cfg_.mtp_draft_tokens <= 0)
        return true;
    mollm_trace::ScopedEvent trace_sync("inference", "mtp.sync");
    const int n = static_cast<int>(token_ids.size());
    if (n <= 0 || preserved_prefix < 0 || preserved_prefix > n ||
        !verified_hidden.data ||
        verified_hidden.prec != Precision::FP32 ||
        verified_hidden.shape[0] <= 0 || verified_hidden.shape[1] < n) {
        return false;
    }

    const int hidden = static_cast<int>(verified_hidden.shape[0]);
    if (mtp_pending_hidden_.empty())
        mtp_pending_hidden_.assign(static_cast<size_t>(hidden), 0.0f);
    if (static_cast<int>(mtp_pending_hidden_.size()) != hidden)
        return false;

    // The trained NextN stream is shifted left relative to the target:
    //
    //   MTP position t = (embedding of target token t + 1, target hidden t)
    //
    // There is therefore no MTP entry for the target's first token, and the
    // MTP cache is always one position shorter than the verified target
    // prefix. Do not insert a synthetic (token 0, zero hidden) KV entry: that
    // pair was not present during MTP training.
    int first_index = preserved_prefix;
    if (position + first_index == 0)
        ++first_index;
    const int remaining = n - first_index;
    const int resume_position = std::max(0, position + first_index - 1);

    // Drafting may have written a longer tail. The first drafted input uses
    // the exact pending target hidden state, so callers may preserve that KV
    // entry and overwrite only the recursively predicted suffix.
    if (preserved_prefix > 0 && mtp_past_len_ < resume_position)
        return false;
    mtp_past_len_ = resume_position;
    set_cache_length(mtp_caches_, resume_position);

    if (remaining > 0) {
        std::vector<int> suffix(
            token_ids.begin() + first_index, token_ids.end());
        std::vector<float> shifted(
            static_cast<size_t>(hidden) * remaining);
        for (int j = 0; j < remaining; ++j) {
            const int original_index = first_index + j;
            float* destination = shifted.data() +
                static_cast<size_t>(j) * hidden;
            if (original_index == 0) {
                std::copy(mtp_pending_hidden_.begin(),
                          mtp_pending_hidden_.end(), destination);
            } else {
                const float* source =
                    static_cast<const float*>(verified_hidden.data) +
                    static_cast<size_t>(original_index - 1) * hidden;
                std::copy(source, source + hidden, destination);
            }
        }
        Tensor shifted_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            hidden, remaining, 1, 1, shifted.data());
        if (!update_mtp_cache(suffix, shifted_tensor, resume_position))
            return false;
    }

    const float* last = static_cast<const float*>(verified_hidden.data) +
                        static_cast<size_t>(n - 1) * hidden;
    std::copy(last, last + hidden, mtp_pending_hidden_.begin());
    return true;
}

// ---------------------------------------------------------------------------
// prefill / decode
// ---------------------------------------------------------------------------

void LLMEngine::prepare_accelerator_prefill_weights() {
    if (!moe_ssd_cache_ || !accelerator_backend_ ||
        !accelerator_backend_->supports_moe_ssd_prefill_switching() ||
        exec_ctx_prefill_.backend != accelerator_backend_.get())
        return;
    if (accelerator_backend_->has_weight_copies()) return;

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
        t.device.buffer = nullptr;
        t.device.offset = 0;
        accelerator_backend_->wrap_weight(t);
        t.data = cpu_data;
        const bool aggregate_expert =
            mollm::detail::is_routed_expert_aggregate_ref(
                node.params.str[0]);
        accelerator_backend_->wrap_weight_int4(t, aggregate_expert);
    }
}

void LLMEngine::release_prefill_buffers() {
    release_graph_temporaries(graph_prefill_, exec_ctx_prefill_.backend);
    invalidate_workspace_key(exec_ctx_prefill_);
}

bool LLMEngine::decode_uses_metal_expert_cache() const {
    return cfg_.metal_ssd_full ||
           (cfg_.device == Device::METAL && accelerator_backend_ &&
            exec_ctx_decode_.moe_backend == accelerator_backend_.get());
}

void LLMEngine::release_vision_buffers() {
    release_graph_temporaries(graph_vision_, exec_ctx_vision_.backend);
    graph_vision_.runtime.pool.clear();
    invalidate_workspace_key(exec_ctx_vision_);
}

int LLMEngine::prefill(const std::vector<int>& token_ids) {
    mollm_trace::ScopedEvent trace_prefill("inference", "prefill");
    int n = (int)token_ids.size();
    if (n == 0)
        return -1;
    if(cfg_.cuda_prefill_logical_batch&&
        (!mollm_prefill::logical_length_supported(n,past_len_,cfg_.n_ctx)||active_vision_||
         exec_ctx_prefill_.profile_enabled||
         (std::getenv("MOLLM_TEST_CUDA_PREFILL_WINDOW_SCHEDULE")&&
          std::strcmp(std::getenv("MOLLM_TEST_CUDA_PREFILL_WINDOW_SCHEDULE"),"0"))||
         !mollm_memory::cuda_host_allocation_allowed(size_t(4)<<30,"logical prefill extra transient admission"))) {
        std::fprintf(stderr,"prefill: logical batch requires inputs fitting remaining context, no vision/profile/window scheduling, and4GiB transient headroom above the host guard\n");
        return -1;
    }
    mollm_prefill::LogicalBatchScope logical_scope(exec_ctx_prefill_.backend,cfg_.cuda_prefill_logical_batch);
    if(!logical_scope.entered){std::fprintf(stderr,"prefill: backend rejected logical batch admission\n");return -1;}
    if (past_len_ > cfg_.n_ctx || n > cfg_.n_ctx - past_len_) {
        fprintf(
            stderr,
            "prefill: %d tokens do not fit in remaining context "
            "(past=%d, n_ctx=%d). Use /reset or a shorter prompt.\n",
            n, past_len_, cfg_.n_ctx);
        return -1;
    }
    sampler_.accept(token_ids);

    Backend* saved_prefill_backend = exec_ctx_prefill_.backend;
    bool short_ssd_cpu_prefill = false;
    const bool switchable_ssd_accelerator =
        moe_ssd_cache_ && accelerator_backend_ &&
        accelerator_backend_->supports_moe_ssd_prefill_switching() &&
        saved_prefill_backend == accelerator_backend_.get();
    const bool metal_weights_ready =
        switchable_ssd_accelerator &&
        accelerator_backend_->has_weight_copies();
    short_ssd_cpu_prefill =
        switchable_ssd_accelerator &&
        (n < metal_ssd_prefill_min_tokens() ||
         (!metal_weights_ready && !metal_ssd_reload_weights()));
    if (short_ssd_cpu_prefill) {
        // Small-M GPU kernels plus one GPU→CPU synchronization per MoE layer
        // lose to the CPU path. Retain any existing dense Metal copies so a
        // later long prompt can reuse them without VM/allocation churn.
        release_graph_temporaries(graph_prefill_, saved_prefill_backend);
        invalidate_workspace_key(exec_ctx_prefill_);
        exec_ctx_prefill_.backend = cpu_backend_.get();
    } else {
        prepare_accelerator_prefill_weights();
    }
    auto finish_prefill_phase = [&] {
        // Hybrid decode is CPU-only, so no prefill workspace is useful after
        // the last chunk. Dense Metal weights remain cached for later prompts.
        if (moe_ssd_cache_) {
            release_graph_temporaries(graph_prefill_,
                                      exec_ctx_prefill_.backend);
            invalidate_workspace_key(exec_ctx_prefill_);
        }
        if (decode_uses_metal_expert_cache() && moe_ssd_cache_ &&
            !moe_ssd_cache_->clear_resident()) {
            fprintf(stderr,
                    "Engine: warning: CPU expert cache was still busy after "
                    "prefill\n");
        }
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

    int chunk_limit = cfg_.cuda_prefill_logical_batch?mollm_prefill::logical_chunk_limit(cfg_):cfg_.prefill_chunk_size > 0
        ? cfg_.prefill_chunk_size : graph_seq_len;
    if (chunk_limit > graph_seq_len) {
        bool dynamic_hidden = false;
        for (const auto& node : graph_prefill_.nodes) {
            if (node.op_type == OpType::INPUT && !node.params.str.empty() &&
                node.params.str[0] == "hidden") {
                dynamic_hidden = node.dim_expr[1].kind == DIM_SEQ;
                break;
            }
        }
        if (cfg_.static_padded || !dynamic_hidden) {
            fprintf(stderr, "prefill: larger chunk size requires dynamic hidden SEQ and no static padding\n");
            finish_prefill_phase();
            return -1;
        }
    }

    const char* window_selector = std::getenv("MOLLM_TEST_CUDA_PREFILL_WINDOW_SCHEDULE");
    if (window_selector && std::strcmp(window_selector, "0")) {
        if (std::strcmp(window_selector, "1") && std::strcmp(window_selector, "2")) {
            std::fprintf(stderr, "prefill: window schedule selector must be exact0/1/2\n");
            finish_prefill_phase(); return -1;
        }
        if (cfg_.device != Device::CUDA || !cfg_.offload_ple || !cfg_.cuda_prefill_reclaim ||
            // Resident CUDA uses mmap host weights and the same provider object
            // as SSD offload. The load-time readonly-last-use contract already
            // validates resident device ownership; neither is an SSD indicator.
            cfg_.moe_ssd_cache_bytes != 0 ||
            active_vision_ || cfg_.mtp_draft_tokens || !cfg_.mtp_companion_dir.empty() ||
            cfg_.static_padded || cfg_.use_decode_as_prefill ||
            cfg_.n_ctx > 4096 || chunk_limit != 1024 || n <= chunk_limit || n > 2048 ||
            !exec_ctx_prefill_.stateful_readonly_last_use || exec_ctx_prefill_.profile_enabled ||
            exec_ctx_prefill_.backend != accelerator_backend_.get()) {
            std::fprintf(stderr, "prefill: test-only window scheduling requires bounded CUDA resident+offload-PLE+reclaim+readonly-last-use, two windows (1025..2048 tokens), chunk1024/context<=4096, no MTP/vision/padding/SSD/profile\n");
            finish_prefill_phase(); return -1;
        }
        const int token = prefill_window_schedule(token_ids, past_len_, !std::strcmp(window_selector, "2"));
        finish_prefill_phase(); return token;
    }

    // Chunked prefill: default to the package length unless explicitly overridden.
    // Each chunk is processed independently, appending to the KV cache.
    // past_len_ can exceed graph_seq_len (cache supports n_ctx); only each
    // chunk's token count must be <= chunk_limit.
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
        int chunk_size =
            std::min({remaining, chunk_limit, cfg_.n_ctx - past_len_});
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

Tensor LLMEngine::prefill_hidden(const std::vector<int>& token_ids,
                                 std::vector<float>* all_logits,
                                 std::vector<int>* all_top1) {
    mollm_trace::ScopedEvent trace_prefill("inference", "prefill_hidden");
    int n = (int)token_ids.size();
    if (n == 0)
        return Tensor();
    if(cfg_.cuda_prefill_logical_batch&&
        (!mollm_prefill::logical_length_supported(n,past_len_,cfg_.n_ctx)||n>mollm_prefill::logical_chunk_limit(cfg_)||active_vision_||
         exec_ctx_prefill_.profile_enabled||
         !mollm_memory::cuda_host_allocation_allowed(size_t(4)<<30,"logical hidden prefill extra transient admission"))) {
        std::fprintf(stderr,"prefill_hidden: logical batch length/mode/headroom admission rejected\n");return Tensor();
    }
    mollm_prefill::LogicalBatchScope logical_scope(exec_ctx_prefill_.backend,cfg_.cuda_prefill_logical_batch);
    if(!logical_scope.entered){std::fprintf(stderr,"prefill_hidden: backend rejected logical batch admission\n");return Tensor();}
    if (past_len_ > cfg_.n_ctx || n > cfg_.n_ctx - past_len_) {
        fprintf(
            stderr,
            "prefill_hidden: %d tokens do not fit in remaining context "
            "(past=%d, n_ctx=%d).\n",
            n, past_len_, cfg_.n_ctx);
        return Tensor();
    }

    Backend* saved_prefill_backend = exec_ctx_prefill_.backend;
    bool short_ssd_cpu_prefill = false;
    const bool switchable_ssd_accelerator =
        moe_ssd_cache_ && accelerator_backend_ &&
        accelerator_backend_->supports_moe_ssd_prefill_switching() &&
        saved_prefill_backend == accelerator_backend_.get();
    const bool metal_weights_ready =
        switchable_ssd_accelerator &&
        accelerator_backend_->has_weight_copies();
    short_ssd_cpu_prefill =
        switchable_ssd_accelerator &&
        (n < metal_ssd_prefill_min_tokens() ||
         (!metal_weights_ready && !metal_ssd_reload_weights()));
    if (short_ssd_cpu_prefill) {
        release_graph_temporaries(graph_prefill_, saved_prefill_backend);
        invalidate_workspace_key(exec_ctx_prefill_);
        exec_ctx_prefill_.backend = cpu_backend_.get();
    } else {
        prepare_accelerator_prefill_weights();
    }

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

    // Set cache metadata through the persistent-storage backend so device-only
    // caches do not need a host-addressable payload.
    if (!set_cache_lengths(caches_, static_cast<uint64_t>(past_len_))) {
        release_pool_tensor(graph_prefill_.runtime.pool, h);
        release_pool_tensor(graph_prefill_.runtime.pool, mask);
        release_pool_tensor(graph_prefill_.runtime.pool, cos);
        release_pool_tensor(graph_prefill_.runtime.pool, sin);
        exec_ctx_prefill_.backend = saved_prefill_backend;
        return Tensor();
    }

    host_set_matmul_profile_phase("prefill_graph");
    std::vector<int32_t> graph_token_ids(
        static_cast<size_t>(h.shape[1]), 0);
    std::copy(token_ids.begin(), token_ids.end(), graph_token_ids.begin());
    Tensor token_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL,
        h.shape[1], 1, 1, 1, graph_token_ids.data());
    if (all_logits)
        all_logits->clear();
    if (all_top1)
        all_top1->clear();
    const bool fuse_accelerator_lm_head = (all_logits || all_top1) &&
        !use_padding && n >= 2 &&
        ((cfg_.device == Device::METAL && n <= 4) ||
         (cfg_.device == Device::CUDA && all_top1 && n <= 8)) && accelerator_backend_ &&
        exec_ctx_prefill_.backend == accelerator_backend_.get() &&
        lm_head_weight_ &&
        accelerator_backend_->supports_lm_head_small_batch(
            *lm_head_weight_);
    Tensor out = run_graph(
        graph_prefill_, exec_ctx_prefill_, h, mask, cos, sin,
        &token_tensor, fuse_accelerator_lm_head);
    if (fuse_accelerator_lm_head) {
        if (out.data && out.device.buffer) {
            const int vocab = static_cast<int>(lm_head_weight_->shape[0]);
            const int hidden = static_cast<int>(lm_head_weight_->shape[1]);
            if (all_top1) {
                all_top1->resize(static_cast<size_t>(n), -1);
                const bool top1_ok = accelerator_backend_
                    ->lm_head_small_batch_argmax_device_and_end_graph(
                        out, *lm_head_weight_, all_top1->data(),
                        n, vocab, hidden);
                if (!top1_ok ||
                    !std::all_of(
                        all_top1->begin(), all_top1->end(),
                        [vocab](int token) {
                            return token >= 0 && token < vocab;
                        })) {
                    all_top1->clear();
                }
            } else {
                all_logits->resize(static_cast<size_t>(n) * vocab);
                if (!accelerator_backend_
                         ->lm_head_small_batch_device_and_end_graph(
                             out, *lm_head_weight_, all_logits->data(),
                             n, vocab, hidden)) {
                    all_logits->clear();
                }
            }
        } else {
            accelerator_backend_->end_graph();
        }
    }
    host_set_matmul_profile_phase("unscoped");
    Tensor copied;
    if (out.data && !exec_ctx_prefill_.backend->dispatch_failed()) {
        copied = copy_tensor_contiguous(
            out, hidden_output_copy_, exec_ctx_prefill_.backend);
        if (has_mtp() && cfg_.mtp_draft_tokens > 0 &&
            !copy_declared_mtp_state(graph_prefill_, exec_ctx_prefill_.backend,
                                     n, target_mtp_state_copy_, target_mtp_state_))
            copied = Tensor{};
    }
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);

    if (copied.data) {
        const int next_past = past_len_ + n;
        if (set_cache_lengths(caches_, static_cast<uint64_t>(next_past)))
            past_len_ = next_past;
        else
            copied = Tensor();
    }

    finish_graph_temporaries(graph_prefill_, exec_ctx_prefill_);
    if (moe_ssd_cache_) {
        release_graph_temporaries(graph_prefill_,
                                  exec_ctx_prefill_.backend);
        invalidate_workspace_key(exec_ctx_prefill_);
    }
    if (decode_uses_metal_expert_cache() && moe_ssd_cache_ &&
        !moe_ssd_cache_->clear_resident()) {
        fprintf(stderr,
                "Engine: warning: CPU expert cache was still busy after "
                "prefill\n");
    }
    exec_ctx_prefill_.backend = saved_prefill_backend;
    return copied;
}

int LLMEngine::prefill_window_schedule(const std::vector<int>& token_ids, int past,
                                      bool chunk_major_reference) {
    const char* merge_selector = std::getenv("MOLLM_TEST_CUDA_MOE_WINDOW_MERGE");
    const bool merge_moe = merge_selector && !std::strcmp(merge_selector, "1");
    const char* fusion_selector = std::getenv("MOLLM_TEST_CUDA_WINDOW_SEGMENT_FUSIONS");
    const bool segment_fusions = fusion_selector && !std::strcmp(fusion_selector, "1");
    const char* inject_selector=std::getenv("MOLLM_TEST_CUDA_WINDOW_DEFER_MOE_INJECT");
    const bool defer_moe_inject=inject_selector&&!std::strcmp(inject_selector,"1");
    if(inject_selector&&std::strcmp(inject_selector,"0")&&!defer_moe_inject) {
        std::fprintf(stderr,"window schedule: deferred MoE inject selector must be exact0/1\n");return -1;
    }
    if(defer_moe_inject&&(!merge_moe||!segment_fusions||chunk_major_reference)) {
        std::fprintf(stderr,"window schedule: deferred MoE inject requires merged layer-major segment fusions\n");return -1;
    }
    const char* metadata_selector = std::getenv("MOLLM_TEST_CUDA_WINDOW_SELECTIVE_KV_METADATA");
    const bool selective_metadata = metadata_selector && !std::strcmp(metadata_selector, "1");
    const char* graph_selector = std::getenv("MOLLM_TEST_CUDA_WINDOW_SINGLE_GRAPH");
    const bool single_graph = graph_selector && !std::strcmp(graph_selector, "1");
    if (graph_selector && std::strcmp(graph_selector, "0") && !single_graph) {
        std::fprintf(stderr, "window schedule: single graph selector must be exact0/1\n"); return -1;
    }
    if (metadata_selector && std::strcmp(metadata_selector, "0") && !selective_metadata) {
        std::fprintf(stderr, "window schedule: selective metadata selector must be exact0/1\n"); return -1;
    }
    if (fusion_selector && std::strcmp(fusion_selector, "0") && !segment_fusions) {
        std::fprintf(stderr, "window schedule: segment fusion selector must be exact0/1\n"); return -1;
    }
    if (merge_selector && std::strcmp(merge_selector, "0") && !merge_moe) {
        std::fprintf(stderr, "window schedule: MoE merge selector must be exact0/1\n"); return -1;
    }
    if (merge_moe && chunk_major_reference) {
        std::fprintf(stderr, "window schedule: MoE merge requires layer-major mode1\n"); return -1;
    }
    std::vector<int> cuts;
    for (size_t i = 0; i < graph_prefill_.nodes.size(); ++i) {
        const auto& node = graph_prefill_.nodes[i];
        if (node.op_type != OpType::MOE) continue;
        if (i + 1 >= graph_prefill_.nodes.size() || node.id != i ||
            node.params.i32.size() < 4 || node.params.i32[0] != 2560 ||
            node.params.i32[1] != 512 || node.params.i32[2] != 10 ||
            graph_prefill_.nodes[i + 1].op_type != OpType::GR_INJECT ||
            graph_prefill_.nodes[i + 1].inputs.empty() ||
            graph_prefill_.nodes[i + 1].inputs[0] != node.id) {
            std::fprintf(stderr, "window schedule: unsupported MoE/carry graph boundary\n"); return -1;
        }
        cuts.push_back(int(i + 2));
    }
    if (cuts.size() != 48 || graph_prefill_.graph_outputs.empty()) {
        std::fprintf(stderr, "window schedule: requires48 Qwen3.8 MoE layers and an output\n"); return -1;
    }
    if (cuts.back() != int(graph_prefill_.nodes.size())) cuts.push_back(int(graph_prefill_.nodes.size()));
    std::vector<uint32_t> boundaries;
    for (const auto& node : graph_prefill_.nodes) {
        if (node.op_type != OpType::INPUT || node.params.str.empty()) continue;
        const auto& name = node.params.str[0];
        if (name == "hidden" || name == "mask" || name == "cos" || name == "sin" ||
            name == "token_ids" || name == "position" || name == "n_tokens") boundaries.push_back(node.id);
    }
    struct Window {
        Graph graph; ExecContext ctx;
        std::vector<uint32_t> private_ids;
        AcceleratorBackend* backend = nullptr;
        bool plans_prepared = false;
        int rows = 0, past = 0;
        ~Window() {
            if (!backend) return;
            if (plans_prepared) backend->release_graph_plan(graph);
            release_graph_temporaries(graph, backend);
            for (uint32_t id : private_ids) {
                auto& t = graph.runtime.tensors[id];
                if (t.device.buffer) backend->free_output(t, &graph.runtime.pool);
                clear_tensor_storage(t);
            }
        }
    } a, b;
    a.rows = 1024; b.rows = int(token_ids.size()) - a.rows;
    a.past = past; b.past = past + a.rows;
    auto initialize = [&](Window& w, int offset) {
        if (!initialize_execution_frame(graph_prefill_, w.graph, boundaries)) return false;
        w.backend = accelerator_backend_.get(); w.private_ids = boundaries;
        w.ctx.graph = &w.graph; w.ctx.pool = &w.graph.runtime.pool; w.ctx.backend = w.backend;
        w.ctx.thread_pool = &thread_pool_;
        w.ctx.conservative_stateful_liveness = true; w.ctx.stateful_readonly_last_use = true;
        w.ctx.conservative_liveness_seq_limit = 1024; w.ctx.runtime_seq_len = w.rows;
        w.ctx.segment_fusions_enabled = segment_fusions;
        w.plans_prepared = segment_fusions;
        // Pointer-bearing HC plans are explicitly dropped before frame storage.
        prepare_execution(w.ctx, segment_fusions);
        std::vector<int> ids(token_ids.begin() + offset, token_ids.begin() + offset + w.rows);
        Tensor hidden = embed(ids), cos, sin;
        generate_rope_cache(w.rows, w.past, cos, sin);
        Tensor mask = build_causal_mask(w.rows, w.past);
        int32_t position = w.past, count = w.rows;
        std::vector<int32_t> tokens(ids.begin(), ids.end());
        Tensor token = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, w.rows, 1, 1, 1, tokens.data());
        Tensor pos = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 1, 1, 1, 1, &position);
        Tensor n = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 1, 1, 1, 1, &count);
        bool ok = true;
        for (uint32_t id : boundaries) {
            const auto& name = w.graph.nodes[id].params.str[0];
            const Tensor& input = name == "hidden" ? hidden : name == "mask" ? mask :
                name == "cos" ? cos : name == "sin" ? sin : name == "token_ids" ? token :
                name == "position" ? pos : n;
            Tensor private_input = Tensor::create(input.prec, MemoryType::NONE,
                input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
            if (!input.data || !w.backend->alloc_output(private_input, input.nbytes(), &w.graph.runtime.pool) ||
                !w.backend->copy_from_host(input.data, private_input, input.nbytes())) ok = false;
            w.graph.runtime.tensors[id] = private_input;
            if (!ok) break;
        }
        for (Tensor* t : {&hidden, &cos, &sin, &mask}) release_pool_tensor(graph_prefill_.runtime.pool, *t);
        if (!ok || w.backend->dispatch_failed()) return false;
        inject_runtime_shapes(w.ctx); return true;
    };
    if (!initialize(a, 0) || !initialize(b, a.rows)) return -1;
    size_t metadata_writes = 0;
    size_t graph_ends = 0;
    struct GraphScope {
        AcceleratorBackend* backend;
        size_t& ends;
        GraphScope(AcceleratorBackend* b, size_t& e) : backend(b), ends(e) {
            if (backend) backend->begin_graph();
        }
        void finish() {
            if (backend) { auto* b = backend; backend = nullptr; b->end_graph(); ++ends; }
        }
        ~GraphScope() { finish(); }
    } outer(single_graph ? a.backend : nullptr, graph_ends);
    auto segment = [&](Window& w, int first, int end) {
        if (selective_metadata) {
            // Only SDPA reads a KV length in this Qwen graph. Restore the
            // active window's two headers immediately before its attention;
            // stateless/carry/GDN/PLE segments need no global header rewrite.
            const uint64_t length = uint64_t(w.past);
            for (int i = first; i < end; ++i) {
                const auto& node = w.graph.nodes[i];
                if (node.op_type == OpType::SDPA_MLA) return false;
                if (node.op_type != OpType::SDPA) continue;
                if (node.inputs.size() < 6) return false;
                for (int input : {4, 5}) {
                    const uint32_t id = node.inputs[input];
                    if (w.graph.nodes[id].op_type != OpType::INPUT ||
                        !w.backend->copy_from_host(&length, w.graph.runtime.tensors[id], sizeof(length),
                            offsetof(CacheMetadata, current_seq_len))) return false;
                    ++metadata_writes;
                }
            }
        } else {
            if (!set_cache_lengths(caches_, uint64_t(w.past))) return false;
            metadata_writes += caches_.size() * 2;
        }
        if (!single_graph) w.backend->begin_graph();
        execute_graph_segment(w.ctx, first, end);
        if (!single_graph) { w.backend->end_graph(); ++graph_ends; }
        return !w.ctx.execution_failed && !w.backend->dispatch_failed();
    };
    int merged_calls = 0;
    int deferred_inject_boundaries=0;
    if (chunk_major_reference) {
        for (Window* w : {&a, &b}) {
            int first = 0;
            for (int end : cuts) { if (!segment(*w, first, end)) return -1; first = end; }
        }
    } else {
        int first = 0;
        for (int end : cuts) {
            const int moe = end - 2;
            if (merge_moe && graph_prefill_.nodes[moe].op_type == OpType::MOE) {
                if (!segment(a, first, moe) || !segment(b, first, moe)) return -1;
                if (!single_graph) a.backend->begin_graph();
                const int merged = execute_graph_moe_pair(a.ctx, b.ctx, moe);
                if (!single_graph) { a.backend->end_graph(); ++graph_ends; }
                if (merged != 1 || a.backend->dispatch_failed()) {
                    std::fprintf(stderr, "window schedule: paired native MoE failed at%d\n", moe); return -1;
                }
                ++merged_calls;
                // Only move a stateless injection when the prepared fusion
                // plan identifies its later RMS consumer in BOTH windows.
                // Keep every stateful op and paired MoE in its original order.
                const int inject=moe+1;
                const bool defer=defer_moe_inject&&
                    size_t(inject)<a.ctx.gr_inject_rms_consumers.size()&&
                    size_t(inject)<b.ctx.gr_inject_rms_consumers.size()&&
                    a.ctx.gr_inject_rms_consumers[inject]>=end&&
                    a.ctx.gr_inject_rms_consumers[inject]==b.ctx.gr_inject_rms_consumers[inject];
                if(defer) {
                    ++deferred_inject_boundaries;
                    first=inject;
                    continue;
                }
                if (!segment(a, moe + 1, end) || !segment(b, moe + 1, end)) return -1;
            } else if (!segment(a, first, end) || !segment(b, first, end)) return -1;
            first = end;
        }
    }
    outer.finish();
    std::fprintf(stderr,"window schedule: deferred_moe_inject_boundaries=%d\n",deferred_inject_boundaries);
    if (a.backend->dispatch_failed()) return -1;
    const int next = past + int(token_ids.size());
    if (!set_cache_lengths(caches_, uint64_t(next))) return -1;
    past_len_ = next;
    if (const char* path = std::getenv("MOLLM_TEST_CUDA_WINDOW_DUMP")) {
        // Exclusive-create diagnostic evidence, never overwrite existing data.
        FILE* file = std::fopen(path, "wbx");
        if (!file) { std::perror("window schedule dump"); return -1; }
        std::vector<uint8_t> bytes(256 * 1024);
        auto record = [&](uint32_t id, const Tensor& t) {
            if (!t.data || t.nbytes() > size_t(64) * 1024 * 1024) return false;
            const uint64_t header[] = {id, uint64_t(t.prec), uint64_t(t.shape[0]), uint64_t(t.shape[1]),
                uint64_t(t.shape[2]), uint64_t(t.shape[3]), uint64_t(t.nbytes())};
            if (std::fwrite(header, 1, sizeof(header), file) != sizeof(header)) return false;
            for (size_t offset = 0; offset < t.nbytes();) {
                const size_t count = std::min(bytes.size(), t.nbytes() - offset);
                if (!accelerator_backend_->copy_to_host(t, bytes.data(), count, offset) ||
                    std::fwrite(bytes.data(), 1, count, file) != count) return false;
                offset += count;
            }
            return true;
        };
        bool ok = record(UINT32_MAX, a.graph.runtime.tensors[a.graph.graph_outputs.back()]) &&
            record(UINT32_MAX - 1, b.graph.runtime.tensors[b.graph.graph_outputs.back()]);
        std::vector<uint32_t> states;
        for (const auto& node : graph_prefill_.nodes) {
            const int index = node.op_type == OpType::SHORTCONV || node.op_type == OpType::PLE_DILATED_CONV ? 2 :
                node.op_type == OpType::GATED_DELTANET_PREFILL ? 7 : node.op_type == OpType::PLE_LOOKUP ? 1 : -1;
            if (index >= 0 && size_t(index) < node.inputs.size()) states.push_back(node.inputs[index]);
            if (node.op_type == OpType::SDPA && node.inputs.size() >= 6) {
                states.push_back(node.inputs[4]); states.push_back(node.inputs[5]);
            }
        }
        std::sort(states.begin(), states.end()); states.erase(std::unique(states.begin(), states.end()), states.end());
        for (uint32_t id : states) {
            if (!ok) break;
            ok = id < graph_prefill_.runtime.tensors.size() &&
                graph_prefill_.nodes[id].op_type == OpType::INPUT && record(id, graph_prefill_.runtime.tensors[id]);
        }
        const bool closed = std::fclose(file) == 0;
        if (!ok || !closed) { std::fprintf(stderr, "window schedule: incomplete diagnostic dump\n"); return -1; }
        std::fprintf(stderr, "window schedule: dumped both hidden windows and%zu declared state/KV inputs\n", states.size());
    }
    const Tensor& out = b.graph.runtime.tensors[b.graph.graph_outputs.back()];
    const int result = run_lmhead(out, b.rows);
    std::fprintf(stderr, "window schedule: mode=%s windows=%d+%d segments=%zu merged_moe=%d token=%d\n",
        chunk_major_reference ? "chunk-major-reference" : "layer-major", a.rows, b.rows, cuts.size(), merged_calls, result);
    std::fprintf(stderr, "window schedule: segment_fusions=%d selective_kv_metadata=%d segment_header_writes=%zu\n",
        segment_fusions, selective_metadata, metadata_writes);
    std::fprintf(stderr, "window schedule: single_device_graph=%d scheduler_graph_ends=%zu\n", single_graph, graph_ends);
    return result;
}

int LLMEngine::prefill_chunk(const std::vector<int>& token_ids, int past) {
    int n = (int)token_ids.size();
    if (n == 0)
        return -1;
    if(cfg_.cuda_prefill_logical_batch&&
       (!mollm_prefill::logical_length_supported(n,past,cfg_.n_ctx)||
        n>mollm_prefill::logical_chunk_limit(cfg_)||
        !mollm_memory::cuda_host_allocation_allowed(size_t(4)<<30,"logical prefill chunk transient admission"))) {
        std::fprintf(stderr,"prefill_chunk: logical batch length/headroom admission rejected\n");
        return -1;
    }

    // Determine the graph's expected seq_len from its hidden input shape.
    int graph_seq_len = 1;
    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty() &&
            node.params.str[0] == "hidden") {
            graph_seq_len = (int)node.out_shape[1];
            break;
        }
    }

    // prefill() validates larger overrides against the graph's dynamic SEQ.
    // Past length is independently bounded by the cache context capacity.
    const int chunk_limit = cfg_.cuda_prefill_logical_batch?mollm_prefill::logical_chunk_limit(cfg_):cfg_.prefill_chunk_size > 0
        ? cfg_.prefill_chunk_size : graph_seq_len;
    if (n > chunk_limit) {
        fprintf(stderr, "prefill_chunk: n=%d > chunk_limit=%d\n", n,
                chunk_limit);
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
        if (active_vision_)
            generate_multimodal_rope_cache(
                graph_seq_len, past - multimodal_base_past_, cos, sin);
        else
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
        if (active_vision_)
            generate_multimodal_rope_cache(
                n, past - multimodal_base_past_, cos, sin);
        else
            generate_rope_cache(n, past, cos, sin);
        mask = build_causal_mask(n, past);
    }

    // n_real_tokens injection is now done by inject_runtime_shapes() above.

    // Set cache metadata so SDPA knows the existing context length.
    if (!set_cache_lengths(caches_, static_cast<uint64_t>(past))) {
        release_pool_tensor(graph_prefill_.runtime.pool, h);
        release_pool_tensor(graph_prefill_.runtime.pool, mask);
        release_pool_tensor(graph_prefill_.runtime.pool, cos);
        release_pool_tensor(graph_prefill_.runtime.pool, sin);
        return -1;
    }

    host_set_matmul_profile_phase("prefill_graph");
    std::vector<int32_t> graph_token_ids(
        static_cast<size_t>(h.shape[1]), 0);
    std::copy(token_ids.begin(), token_ids.end(), graph_token_ids.begin());
    Tensor token_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL,
        h.shape[1], 1, 1, 1, graph_token_ids.data());
    Tensor out = run_graph(
        graph_prefill_, exec_ctx_prefill_, h, mask, cos, sin,
        &token_tensor);
    if (!out.data) {
        release_pool_tensor(graph_prefill_.runtime.pool, h);
        release_pool_tensor(graph_prefill_.runtime.pool, mask);
        release_pool_tensor(graph_prefill_.runtime.pool, cos);
        release_pool_tensor(graph_prefill_.runtime.pool, sin);
        finish_graph_temporaries(graph_prefill_, exec_ctx_prefill_);
        return -1;
    }

    if (has_mtp() && cfg_.mtp_draft_tokens > 0) {
        Tensor verified = copy_tensor_contiguous(
            out, hidden_output_copy_, exec_ctx_prefill_.backend);
        if (!copy_declared_mtp_state(graph_prefill_, exec_ctx_prefill_.backend,
                                    n, target_mtp_state_copy_, target_mtp_state_))
            verified = Tensor{};
        else if (target_mtp_state_.data)
            verified = target_mtp_state_;
        if (!verified.data || !sync_mtp(token_ids, verified, past)) {
            fprintf(stderr, "prefill_chunk: failed to synchronize MTP state\n");
            release_pool_tensor(graph_prefill_.runtime.pool, h);
            release_pool_tensor(graph_prefill_.runtime.pool, mask);
            release_pool_tensor(graph_prefill_.runtime.pool, cos);
            release_pool_tensor(graph_prefill_.runtime.pool, sin);
            finish_graph_temporaries(graph_prefill_, exec_ctx_prefill_);
            return -1;
        }
    }

    const int next_past = past + n;
    if (!set_cache_lengths(caches_, static_cast<uint64_t>(next_past))) {
        release_pool_tensor(graph_prefill_.runtime.pool, h);
        release_pool_tensor(graph_prefill_.runtime.pool, mask);
        release_pool_tensor(graph_prefill_.runtime.pool, cos);
        release_pool_tensor(graph_prefill_.runtime.pool, sin);
        finish_graph_temporaries(graph_prefill_, exec_ctx_prefill_);
        return -1;
    }
    past_len_ = next_past;

    // Cache migration (prefill→decode graph) is done once at load time.
    // Both graphs share the same physical cache buffers.

    host_set_matmul_profile_phase("prefill_lmhead");
    int token = run_lmhead(out, n);
    host_set_matmul_profile_phase("unscoped");
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);
    finish_graph_temporaries(graph_prefill_, exec_ctx_prefill_);
    return token;
}

int LLMEngine::decode(int token_id) {
    mollm_trace::ScopedEvent trace_decode("inference", "decode");
    if (past_len_ >= cfg_.n_ctx) {
        fprintf(stderr,
                "decode: context full (past=%d >= n_ctx=%d). Use /reset.\n",
                past_len_, cfg_.n_ctx);
        return -1;
    }
    Tensor h = embed({token_id});
    h.shape[1] = 1;
    h.compute_strides();

    Tensor cos, sin;
    generate_rope_cache(1, past_len_, cos, sin);

    Tensor mask = build_causal_mask(1, past_len_);

    const bool defer_accelerator_lmhead =
        accelerator_backend_ &&
        accelerator_backend_->is_device_resident() &&
        exec_ctx_decode_.backend == accelerator_backend_.get() &&
        lm_head_weight_ &&
        accelerator_backend_->supports_lm_head(*lm_head_weight_);

    host_set_matmul_profile_phase("decode_graph");
    int32_t graph_token_id = token_id;
    Tensor token_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL,
        1, 1, 1, 1, &graph_token_id);
    Tensor out = run_graph(
        graph_decode_, exec_ctx_decode_, h, mask, cos, sin,
        &token_tensor, defer_accelerator_lmhead);
    if (!out.data) {
        release_pool_tensor(graph_prefill_.runtime.pool, h);
        release_pool_tensor(graph_prefill_.runtime.pool, mask);
        release_pool_tensor(graph_prefill_.runtime.pool, cos);
        release_pool_tensor(graph_prefill_.runtime.pool, sin);
        finish_graph_temporaries(graph_decode_, exec_ctx_decode_);
        return -1;
    }

    if (has_mtp() && cfg_.mtp_draft_tokens > 0) {
        Tensor verified = copy_tensor_contiguous(
            out, hidden_output_copy_, exec_ctx_decode_.backend);
        if (!copy_declared_mtp_state(graph_decode_, exec_ctx_decode_.backend,
                                    1, target_mtp_state_copy_, target_mtp_state_))
            verified = Tensor{};
        else if (target_mtp_state_.data)
            verified = target_mtp_state_;
        if (!verified.data ||
            !sync_mtp(std::vector<int>{token_id}, verified, past_len_)) {
            fprintf(stderr, "decode: failed to synchronize MTP state\n");
            release_pool_tensor(graph_prefill_.runtime.pool, h);
            release_pool_tensor(graph_prefill_.runtime.pool, mask);
            release_pool_tensor(graph_prefill_.runtime.pool, cos);
            release_pool_tensor(graph_prefill_.runtime.pool, sin);
            finish_graph_temporaries(graph_decode_, exec_ctx_decode_);
            return -1;
        }
    }

    const int next_past = past_len_ + 1;
    if (!set_cache_lengths(caches_, static_cast<uint64_t>(next_past))) {
        release_pool_tensor(graph_prefill_.runtime.pool, h);
        release_pool_tensor(graph_prefill_.runtime.pool, mask);
        release_pool_tensor(graph_prefill_.runtime.pool, cos);
        release_pool_tensor(graph_prefill_.runtime.pool, sin);
        finish_graph_temporaries(graph_decode_, exec_ctx_decode_);
        return -1;
    }
    past_len_ = next_past;

    host_set_matmul_profile_phase("decode_lmhead");
    sampler_.accept(token_id);
    int token = run_lmhead(out, 1, defer_accelerator_lmhead);
    host_set_matmul_profile_phase("unscoped");
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);
    finish_graph_temporaries(graph_decode_, exec_ctx_decode_);
    return token;
}

bool LLMEngine::speculative_decode(int token_id,
                                   std::vector<int>& output_tokens,
                                   int max_output_tokens,
                                   int stop_token_id) {
    output_tokens.clear();
    if (max_output_tokens <= 0)
        return false;
    if (!has_mtp() || cfg_.mtp_draft_tokens <= 0) {
        const int token = decode(token_id);
        if (token < 0) return false;
        output_tokens.push_back(token);
        return true;
    }
    const int expected_mtp_len = std::max(0, past_len_ - 1);
    if (past_len_ >= cfg_.n_ctx || mtp_past_len_ != expected_mtp_len ||
        mtp_pending_hidden_.empty()) {
        fprintf(stderr,
                "speculative_decode: MTP state is not synchronized with target\n");
        return false;
    }

    const int max_drafts = std::min({
        cfg_.mtp_draft_tokens, cfg_.n_ctx - past_len_ - 1,
        std::max(0, max_output_tokens - 1)});
    mollm_trace::ScopedEvent trace_mtp("inference", "mtp_speculative_step");
    using MtpClock = std::chrono::steady_clock;
    const auto step_start = MtpClock::now();
    ++mtp_stats_.steps;
    const int target_start = past_len_;
    const int mtp_start = expected_mtp_len;
    const int hidden = static_cast<int>(mtp_pending_hidden_.size());
    const int vocab = static_cast<int>(lm_head_weight_->shape[0]);
    const bool plain_argmax = sampler_.uses_plain_argmax();
    Tensor state_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, hidden, 1, 1, 1,
        mtp_pending_hidden_.data());
    std::vector<float> cpu_state;
    std::vector<int> drafts;
    drafts.reserve(max_drafts);
    std::vector<std::vector<float>> draft_probabilities;
    if (!plain_argmax)
        draft_probabilities.reserve(max_drafts);
    std::vector<int> draft_history{token_id};
    std::vector<int> draft_input(1);
    int current = token_id;
    bool mtp_current_cached = false;

    const auto draft_start = MtpClock::now();
    for (int i = 0; i < max_drafts; ++i) {
        ++mtp_stats_.draft_calls;
        ++mtp_stats_.attempts_by_depth[static_cast<size_t>(i)];
        mollm_trace::ScopedEvent trace_draft("inference", "mtp.draft");
        int draft = -1;
        draft_input[0] = current;
        const auto draft_model_start = MtpClock::now();
        Tensor mtp_hidden = run_mtp_tokens(
            draft_input, state_tensor, mtp_start + i,
            plain_argmax ? &draft : nullptr);
        if (!mtp_hidden.data)
            return false;
        mtp_current_cached = true;
        if (plain_argmax && draft < 0) {
            std::vector<float> logits = run_lmhead_raw(mtp_hidden, 1, false);
            if (logits.empty())
                return false;
            draft = static_cast<int>(
                std::max_element(logits.begin(), logits.end()) -
                logits.begin());
        } else if (!plain_argmax) {
            std::vector<float> logits = run_lmhead_raw(mtp_hidden, 1, false);
            if (logits.size() != static_cast<size_t>(vocab))
                return false;
            draft_probabilities.emplace_back();
            sampler_.probabilities(
                logits.data(), vocab, draft_history,
                draft_probabilities.back());
            draft = sampler_.sample_probabilities(
                draft_probabilities.back());
        }
        mtp_stats_.draft_model_ms +=
            std::chrono::duration<double, std::milli>(
                MtpClock::now() - draft_model_start).count();

        drafts.push_back(draft);
        ++mtp_stats_.drafted_by_depth[static_cast<size_t>(i)];
        current = draft;
        draft_history.push_back(draft);
        const Tensor& next_state = draft_mtp_state_.data
            ? draft_mtp_state_ : mtp_hidden;
        if (next_state.shape[0] != hidden || next_state.shape[1] != 1)
            return false;
        if (next_state.device.buffer) {
            state_tensor = next_state;
        } else {
            const float* mtp_data =
                static_cast<const float*>(next_state.data);
            if (cpu_state.empty())
                cpu_state.resize(static_cast<size_t>(hidden));
            std::copy(mtp_data, mtp_data + hidden, cpu_state.begin());
            state_tensor = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL,
                hidden, 1, 1, 1, cpu_state.data());
        }
    }
    mtp_stats_.draft_ms +=
        std::chrono::duration<double, std::milli>(
            MtpClock::now() - draft_start).count();

    if (drafts.empty()) {
        ++mtp_stats_.fallback_steps;
    }

    std::vector<int> inputs;
    inputs.reserve(drafts.size() + 1);
    inputs.push_back(token_id);
    inputs.insert(inputs.end(), drafts.begin(), drafts.end());
    mtp_stats_.drafted += drafts.size();
    mtp_stats_.verify_tokens += inputs.size();

    // KV payload can be rewound by length, but GDN/conv/RWKV/plain auxiliary
    // buffers contain the state after the entire verification batch. Keep the
    // pre-verification state so rejected drafts cannot leak into generation.
    mollm::detail::RecurrentCheckpoint recurrent_checkpoint;
    Backend* state_backend = persistent_backend();
    struct PrefixStateScope {
        Backend* backend=nullptr;
        ~PrefixStateScope(){if(backend)backend->end_state_prefixes();}
    } prefix_scope;
    if (inputs.size() > 1) {
        const auto checkpoint_start = MtpClock::now();
        std::vector<Tensor*> states;
        for (const auto& cache : caches_) {
            for (Tensor* state : {cache.gdn_state, cache.gdn_conv,
                                 cache.rwkv_state, cache.rwkv_att_shift,
                                 cache.rwkv_ffn_shift})
                if (state) states.push_back(state);
        }
        for (const auto& state : auxiliary_states_)
            if (state.second) states.push_back(state.second);
        if (!states.empty() &&
            (!state_backend ||
             !recurrent_checkpoint.capture(*state_backend, states))) {
            fprintf(stderr, "speculative_decode: recurrent checkpoint failed\n");
            return false;
        }
        mtp_stats_.recurrent_checkpoint_bytes += recurrent_checkpoint.bytes();
        if (recurrent_checkpoint.device_resident())
            mtp_stats_.recurrent_device_checkpoint_bytes += recurrent_checkpoint.bytes();
        mtp_stats_.recurrent_checkpoint_ms +=
            std::chrono::duration<double, std::milli>(
                MtpClock::now() - checkpoint_start).count();
        if(state_backend && state_backend->begin_state_prefixes(states,static_cast<int>(inputs.size())))
            prefix_scope.backend=state_backend;
    }

    Tensor verified_hidden;
    std::vector<float> logits;
    std::vector<int> verified_top1;
    const auto verify_start = MtpClock::now();
    {
        mollm_trace::ScopedEvent trace_verify("inference", "mtp.verify");
        verified_hidden = prefill_hidden(
            inputs, plain_argmax ? nullptr : &logits,
            plain_argmax ? &verified_top1 : nullptr);
        if (!verified_hidden.data)
            return false;
        if (verified_top1.size() != inputs.size() && logits.empty())
            logits = run_lmhead_raw(
                verified_hidden, static_cast<int>(inputs.size()), true);
    }
    mtp_stats_.verify_ms +=
        std::chrono::duration<double, std::milli>(
            MtpClock::now() - verify_start).count();
    const bool gpu_top1 = verified_top1.size() == inputs.size();
    if (!gpu_top1 &&
        logits.size() != inputs.size() * static_cast<size_t>(vocab))
        return false;


    int kept = 0;
    std::vector<float> target_probabilities;
    const auto sample_start = MtpClock::now();
    for (size_t i = 0; i < inputs.size(); ++i) {
        sampler_.accept(inputs[i]);
        ++kept;
        int sampled = -1;
        bool accepted_draft = false;
        if (plain_argmax) {
            sampled = gpu_top1
                ? verified_top1[i]
                : sampler_.sample(
                      logits.data() + i * static_cast<size_t>(vocab), vocab);
            accepted_draft = i < drafts.size() && sampled == drafts[i];
        } else {
            sampler_.probabilities(
                logits.data() + i * static_cast<size_t>(vocab), vocab, {},
                target_probabilities);
            if (i < drafts.size()) {
                if (i >= draft_probabilities.size())
                    return false;
                const auto& proposal = draft_probabilities[i];
                const int proposed = drafts[i];
                sampled = sampler_.speculative_sample(
                    target_probabilities, proposal, proposed,
                    &accepted_draft);
            } else {
                sampled = sampler_.sample_probabilities(
                    target_probabilities);
            }
        }
        output_tokens.push_back(sampled);
        if (sampled == stop_token_id)
            break;
        if (i < drafts.size()) {
            if (!accepted_draft)
                break;
            ++mtp_stats_.accepted;
            ++mtp_stats_.accepted_by_depth[i];
        }
    }
    mtp_stats_.sample_ms +=
        std::chrono::duration<double, std::milli>(
            MtpClock::now() - sample_start).count();

    std::vector<int> verified_inputs(inputs.begin(), inputs.begin() + kept);
    if (kept < static_cast<int>(inputs.size()) &&
        !recurrent_checkpoint.empty()) {
        const auto rollback_start = MtpClock::now();
        const bool restored_prefix=prefix_scope.backend && state_backend->restore_state_prefix(kept);
        if (!restored_prefix && !recurrent_checkpoint.restore(*state_backend)) {
            fprintf(stderr, "speculative_decode: recurrent restore failed\n");
            return false;
        }
        past_len_ = restored_prefix ? target_start+kept : target_start;
        if (!set_cache_lengths(caches_, past_len_)) return false;
        // Replay only the accepted inputs. prefill_hidden does not mutate the
        // sampler or synchronize MTP. Its hidden copy replaces (and may
        // invalidate) the original verification output.
        if(restored_prefix) {
            verified_hidden.shape[1]=kept;verified_hidden.compute_strides();
            if(target_mtp_state_.data) {
                target_mtp_state_.shape[1]=kept;target_mtp_state_.compute_strides();
            }
        } else {
            // Prefix capture is no longer needed and must not intercept replay.
            if(prefix_scope.backend){prefix_scope.backend->end_state_prefixes();prefix_scope.backend=nullptr;}
            verified_hidden = prefill_hidden(verified_inputs);
            if (!verified_hidden.data) return false;
        }
        ++mtp_stats_.recurrent_rollbacks;
        if(!restored_prefix)mtp_stats_.recurrent_replay_tokens += kept;
        mtp_stats_.recurrent_rollback_ms +=
            std::chrono::duration<double, std::milli>(
                MtpClock::now() - rollback_start).count();
    } else {
        past_len_ = target_start + kept;
        if (!set_cache_lengths(caches_, past_len_)) return false;
        verified_hidden.shape[1] = kept;
        verified_hidden.compute_strides();
    }
    const int preserved_prefix = mtp_current_cached ? 1 : 0;
    mtp_stats_.sync_tokens += static_cast<uint64_t>(
        std::max(0, kept - preserved_prefix));
    const auto sync_start = MtpClock::now();
    Tensor synchronized_hidden = target_mtp_state_.data
        ? target_mtp_state_ : verified_hidden;
    synchronized_hidden.shape[1] = kept;
    synchronized_hidden.compute_strides();
    if (!sync_mtp(
            verified_inputs, synchronized_hidden, target_start,
            preserved_prefix))
        return false;
    mtp_stats_.sync_ms +=
        std::chrono::duration<double, std::milli>(
            MtpClock::now() - sync_start).count();
    mtp_stats_.total_ms +=
        std::chrono::duration<double, std::milli>(
            MtpClock::now() - step_start).count();
    return true;
}

Tensor LLMEngine::decode_hidden(int token_id) {
    mollm_trace::ScopedEvent trace_decode("inference", "decode_hidden");
    if (past_len_ >= cfg_.n_ctx) {
        fprintf(stderr,
                "decode_hidden: context full (past=%d >= n_ctx=%d). "
                "Use /reset.\n",
                past_len_, cfg_.n_ctx);
        return Tensor();
    }
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

    host_set_matmul_profile_phase("decode_graph");
    int32_t graph_token_id = token_id;
    Tensor token_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL,
        1, 1, 1, 1, &graph_token_id);
    Tensor out = run_graph(
        graph_decode_, exec_ctx_decode_, h, mask, cos, sin,
        &token_tensor);
    host_set_matmul_profile_phase("unscoped");
    Tensor copied;
    if (out.data)
        copied = copy_tensor_contiguous(
            out, hidden_output_copy_, exec_ctx_decode_.backend);
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);

    if (copied.data) {
        const int next_past = past_len_ + 1;
        if (set_cache_lengths(caches_, static_cast<uint64_t>(next_past)))
            past_len_ = next_past;
        else
            copied = Tensor();
    }

    finish_graph_temporaries(graph_decode_, exec_ctx_decode_);
    return copied;
}
