#include "graph/execute.h"
#include "engine/backend.h"
#include "kernels/matmul.h"
#include "kernels/norm.h"
#include "kernels/rope.h"
#include "kernels/attention.h"
#include "kernels/gdn.h"
#include "kernels/moe.h"
#include "kernels/rwkv.h"
#include "kernels/tensor.h"
#include "kernels/trace.h"
#include "kernels/activations.h"  // for Activation enum + sigmoid_f32_neon
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>

#if HAS_NEON
#include <arm_neon.h>

// sigmoid_f32_neon is now defined in kernels/activations.h (shared with
// matmul kernel for fused activation). Include that header above.
#endif

// ---------------------------------------------------------------------------
// DimExpr evaluation — substitute runtime seq_len into a symbolic dim expr.
// ---------------------------------------------------------------------------

static inline int64_t eval_dim(const DimExpr& e, int64_t out_shape_val,
                                int64_t seq_val, int64_t batch_val) {
    switch (e.kind) {
        case DIM_SEQ:   return seq_val;
        case DIM_MUL:   return (int64_t)e.coeff * seq_val;
        case DIM_ADD:   return (int64_t)e.coeff + seq_val;
        case DIM_BATCH: return batch_val;
        case DIM_CONST:
        default:        return out_shape_val;
    }
}

// ---------------------------------------------------------------------------
// materialize_strided
//
// Used by RESHAPE (non-contiguous input) and CONTIGUOUS ops. The generic
// 4-loop elementwise memcpy is slow when dim0 is contiguous (stride[0]==es):
// each iteration does a 4-byte memcpy with loop overhead. This helper detects
// the contiguous-inner-dim case and falls back to a single memcpy per (i1,i2,i3)
// row, cutting loop overhead by shape[0]×.
//
// For the fully contiguous case it collapses to a single memcpy.
static inline void materialize_strided(const Tensor& src, void* dst) {
    size_t es = src.element_size();
    char* dst_base = static_cast<char*>(dst);
    int64_t d0 = src.shape[0];
    int64_t row_bytes = d0 * es;

    // Fast path: dim0 contiguous → memcpy each row in one call.
    if (src.stride[0] == (size_t)es) {
        int64_t flat = 0;
        for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
            for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
                for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                    const char* src_ptr = static_cast<const char*>(src.data)
                        + i1 * src.stride[1]
                        + i2 * src.stride[2]
                        + i3 * src.stride[3];
                    std::memcpy(dst_base + flat * es, src_ptr, row_bytes);
                    flat += d0;
                }
            }
        }
        return;
    }

    // Slow path: element-wise copy (dim0 not contiguous).
    int64_t flat = 0;
    for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
        for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
            for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                for (int64_t i0 = 0; i0 < d0; i0++) {
                    const char* src_ptr = static_cast<const char*>(src.data)
                        + i0 * src.stride[0]
                        + i1 * src.stride[1]
                        + i2 * src.stride[2]
                        + i3 * src.stride[3];
                    std::memcpy(dst_base + flat * es, src_ptr, es);
                    flat++;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stride-aware elementwise iteration helper.
//
// Tensors are row-major 4-D [d0, d1, d2, d3] with stride[0] = es (innermost).
// "Contiguous" means all strides match the default row-major layout.
// When a tensor is a view (e.g. SLICE produces a non-contiguous dim-1 stride),
// we still want to read it efficiently.
//
// Layout invariant: stride[0] == es always (the dim-0 elements are always
// contiguous in memory). For non-contiguous tensors, stride[1] may be larger
// than shape[0]*es (e.g. SLICE dim=0 produces a view with stride[1] inherited
// from the parent's full N, not the sliced N/2).
//
// The iteration: outer loop over (d3, d2, d1), inner loop over d0 contiguous
// elements. Each inner run has `shape[0]` elements at stride es.
//
// `fn(float* dst_row, const float* src_row, int n)` is called per outer step
// with a contiguous run of n elements.
// ---------------------------------------------------------------------------
struct StrideIter {
    // Pre-compute outer-loop bounds and step sizes.
    int d1, d2, d3;             // outer-loop dims (excluding d0)
    size_t s1, s2, s3;          // byte strides for outer dims
    int n_inner;                // d0 (contiguous run length)
};

static inline StrideIter compute_stride_iter(const Tensor& t) {
    StrideIter it;
    it.n_inner = (int)t.shape[0];
    it.d1 = (int)t.shape[1];
    it.d2 = (int)t.shape[2];
    it.d3 = (int)t.shape[3];
    it.s1 = t.stride[1];
    it.s2 = t.stride[2];
    it.s3 = t.stride[3];
    return it;
}

// Check if a tensor has the default contiguous row-major layout.
static inline bool is_contiguous_layout(const Tensor& t) {
    if (t.stride[0] != t.element_size()) return false;
    size_t expected = t.element_size();
    if (t.stride[1] != expected * t.shape[0]) return false;
    if (t.stride[2] != t.stride[1] * t.shape[1]) return false;
    if (t.stride[3] != t.stride[2] * t.shape[2]) return false;
    return true;
}

static inline bool shares_storage(const Tensor& a, const Tensor& b) {
    if (a.storage_id != 0 && b.storage_id != 0) {
        return a.owner_id == b.owner_id && a.storage_id == b.storage_id;
    }
    return a.data == b.data;
}

// Apply a unary op (sigmoid/silu/etc) reading from `src` (may be strided view),
// writing to `dst` (assumed contiguous, freshly allocated).
// `kernel_4(dst, src)` processes 4 contiguous FP32 elements via NEON.
template <typename Kernel4>
static inline void unary_stride_aware(float* dst, const Tensor& src,
                                       Kernel4 kernel_4) {
    const char* src_base = static_cast<const char*>(src.data);
    int n_inner = (int)src.shape[0];

    if (is_contiguous_layout(src)) {
        // Fast path: linear memcpy-friendly layout.
        const float* sp = src.ptr<float>();
        int N = (int)src.nelements();
        int i = 0;
        for (; i + 3 < N; i += 4) {
            float32x4_t v = vld1q_f32(sp + i);
            vst1q_f32(dst + i, kernel_4(v));
        }
        for (; i < N; i++) {
            float32x4_t v = vld1q_f32(sp + i);
            float32x4_t r = kernel_4(v);
            dst[i] = vgetq_lane_f32(r, 0);
        }
        return;
    }

    // Strided view: iterate (d3, d2, d1) outer, d0 contiguous inner.
    StrideIter it = compute_stride_iter(src);
    float* dst_row = dst;
    for (int i3 = 0; i3 < it.d3; i3++) {
        const char* p3 = src_base + i3 * it.s3;
        for (int i2 = 0; i2 < it.d2; i2++) {
            const char* p2 = p3 + i2 * it.s2;
            for (int i1 = 0; i1 < it.d1; i1++) {
                const float* sp = reinterpret_cast<const float*>(p2 + i1 * it.s1);
                int i = 0;
                for (; i + 3 < n_inner; i += 4) {
                    float32x4_t v = vld1q_f32(sp + i);
                    vst1q_f32(dst_row + i, kernel_4(v));
                }
                for (; i < n_inner; i++) {
                    float32x4_t v = vld1q_f32(sp + i);
                    float32x4_t r = kernel_4(v);
                    dst_row[i] = vgetq_lane_f32(r, 0);
                }
                dst_row += n_inner;
            }
        }
    }
}

// Apply a binary op reading from `a` and `b` (either may be strided view),
// writing to `dst` (contiguous).
//
// `a` and `b` may have different strides — both are iterated by their own
// shape/stride. Their shapes (d0..d3) must match.
template <typename Kernel4>
static inline void binary_stride_aware(float* dst,
                                       const Tensor& a, const Tensor& b,
                                       Kernel4 kernel_4) {
    const char* a_base = static_cast<const char*>(a.data);
    const char* b_base = static_cast<const char*>(b.data);

    bool a_contig = is_contiguous_layout(a);
    bool b_contig = is_contiguous_layout(b);

    if (a_contig && b_contig) {
        // Fast path: both contiguous.
        const float* ap = a.ptr<float>();
        const float* bp = b.ptr<float>();
        int N = (int)a.nelements();
        int i = 0;
        for (; i + 7 < N; i += 8) {
            vst1q_f32(dst + i,     kernel_4(vld1q_f32(ap + i),     vld1q_f32(bp + i)));
            vst1q_f32(dst + i + 4, kernel_4(vld1q_f32(ap + i + 4), vld1q_f32(bp + i + 4)));
        }
        for (; i < N; i++) dst[i] = kernel_4(vdupq_n_f32(ap[i]), vdupq_n_f32(bp[i]))[0];
        return;
    }

    // At least one is strided. Iterate outer dims; inner dim-0 (shape[0]) is
    // always contiguous (stride[0] == es) for both inputs, but their outer
    // strides may differ.
    StrideIter ita = compute_stride_iter(a);
    StrideIter itb = compute_stride_iter(b);
    int n_inner = ita.n_inner;  // a.shape[0] == b.shape[0]
    float* dst_row = dst;
    for (int i3 = 0; i3 < ita.d3; i3++) {
        const char* pa3 = a_base + i3 * ita.s3;
        const char* pb3 = b_base + i3 * itb.s3;
        for (int i2 = 0; i2 < ita.d2; i2++) {
            const char* pa2 = pa3 + i2 * ita.s2;
            const char* pb2 = pb3 + i2 * itb.s2;
            for (int i1 = 0; i1 < ita.d1; i1++) {
                const float* ap = reinterpret_cast<const float*>(pa2 + i1 * ita.s1);
                const float* bp = reinterpret_cast<const float*>(pb2 + i1 * itb.s1);
                int i = 0;
                for (; i + 7 < n_inner; i += 8) {
                    vst1q_f32(dst_row + i,     kernel_4(vld1q_f32(ap + i),     vld1q_f32(bp + i)));
                    vst1q_f32(dst_row + i + 4, kernel_4(vld1q_f32(ap + i + 4), vld1q_f32(bp + i + 4)));
                }
                for (; i < n_inner; i++) dst_row[i] = kernel_4(vdupq_n_f32(ap[i]), vdupq_n_f32(bp[i]))[0];
                dst_row += n_inner;
            }
        }
    }
}

// Elementwise operations commonly apply a per-channel parameter [D, 1] to a
// sequence tensor [D, S].  The fast helper above deliberately assumes equal
// shapes; using it for this case walks past the one-row parameter buffer on
// token 1 and beyond.  Keep that fast path for equal tensors and use this
// stride-aware broadcast path only when a dimension in b is singleton.
template <typename ScalarOp>
static inline void binary_broadcast_from_b(float* dst, const Tensor& a,
                                           const Tensor& b, ScalarOp op) {
    const char* a_base = static_cast<const char*>(a.data);
    const char* b_base = static_cast<const char*>(b.data);
    size_t flat = 0;
    for (int64_t i3 = 0; i3 < a.shape[3]; ++i3) {
        const int64_t b3 = b.shape[3] == 1 ? 0 : i3;
        for (int64_t i2 = 0; i2 < a.shape[2]; ++i2) {
            const int64_t b2 = b.shape[2] == 1 ? 0 : i2;
            for (int64_t i1 = 0; i1 < a.shape[1]; ++i1) {
                const int64_t b1 = b.shape[1] == 1 ? 0 : i1;
                const float* ap = reinterpret_cast<const float*>(
                    a_base + i3 * a.stride[3] + i2 * a.stride[2] + i1 * a.stride[1]);
                const float* bp = reinterpret_cast<const float*>(
                    b_base + b3 * b.stride[3] + b2 * b.stride[2] + b1 * b.stride[1]);
                for (int64_t i0 = 0; i0 < a.shape[0]; ++i0) {
                    const int64_t b0 = b.shape[0] == 1 ? 0 : i0;
                    dst[flat + i0] = op(ap[i0], bp[b0]);
                }
                flat += a.shape[0];
            }
        }
    }
}

static inline bool broadcasts_to(const Tensor& b, const Tensor& a) {
    for (int d = 0; d < 4; ++d) {
        if (b.shape[d] != 1 && b.shape[d] != a.shape[d]) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// prepare_execution
// ---------------------------------------------------------------------------

void reset_profile_stats(ExecContext& ctx) {
    ctx.profile_stats.clear();
    if (!ctx.graph) return;

    const auto& nodes = ctx.graph->nodes;
    ctx.profile_stats.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        ctx.profile_stats[i].node_id = nodes[i].id;
        ctx.profile_stats[i].op_type = nodes[i].op_type;
        ctx.profile_stats[i].calls = 0;
        ctx.profile_stats[i].total_ns = 0;
    }
}

void prepare_execution(ExecContext& ctx) {
    auto& nodes = ctx.graph->nodes;
    const size_t N = nodes.size();
    ctx.release_queue.assign(N, {});
    reset_profile_stats(ctx);

    // Stateful/cache-mutating graphs need a stronger storage dependency model:
    // SDPA updates KV cache, GDN/SHORTCONV update recurrent state, and their
    // inputs may alias persistent engine storage. Keep them on end-of-call
    // cleanup until TensorStorage/storage-id tracking replaces pointer-based
    // borrowed-view inference.
    for (const auto& node : nodes) {
        if (node.op_type == OpType::SDPA || node.op_type == OpType::SDPA_MLA ||
            node.op_type == OpType::GATED_DELTANET_PREFILL ||
            node.op_type == OpType::GATED_DELTANET_DECODE ||
            node.op_type == OpType::SHORTCONV) {
            return;
        }
    }

    uint32_t max_id = 0;
    for (const auto& node : nodes) max_id = std::max(max_id, node.id);
    const size_t M = (size_t)max_id + 1;

    std::vector<uint32_t> owner_root(M, 0);
    std::vector<int> node_index(M, -1);
    std::vector<int> last_direct_use(M, -1);
    std::vector<int> last_storage_use(M, -1);
    std::vector<uint8_t> keep(M, 0);

    auto is_view_like = [](OpType op) {
        // RESHAPE may materialize at runtime for non-contiguous inputs, but it
        // may also borrow. Treating it as view-like for owner propagation keeps
        // producers alive for the borrowed case; runtime release skips it when
        // it actually borrowed and releases it when it materialized.
        return op == OpType::RESHAPE || op == OpType::PERMUTE || op == OpType::SLICE;
    };

    for (size_t i = 0; i < N; i++) {
        const auto& node = nodes[i];
        if (node.id >= M) continue;
        node_index[node.id] = (int)i;

        if (is_view_like(node.op_type) && !node.inputs.empty() && node.inputs[0] < M) {
            owner_root[node.id] = owner_root[node.inputs[0]];
        } else {
            owner_root[node.id] = node.id;
        }
        if (owner_root[node.id] >= M) owner_root[node.id] = node.id;

        for (uint32_t inp_id : node.inputs) {
            if (inp_id >= M) continue;
            last_direct_use[inp_id] = (int)i;
            uint32_t root = owner_root[inp_id];
            if (root < M) last_storage_use[root] = std::max(last_storage_use[root], (int)i);
        }
    }

    // Graph outputs are returned to the caller. If the output is a borrowed
    // view, its storage owner must also stay alive past execute_graph().
    for (uint32_t out_id : ctx.graph->graph_outputs) {
        if (out_id >= M) continue;
        keep[out_id] = 1;
        uint32_t root = owner_root[out_id];
        if (root < M) keep[root] = 1;
    }

    for (const auto& node : nodes) {
        if (node.id >= M) continue;
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        if (keep[node.id]) continue;

        int rel_at = std::max(last_direct_use[node.id], last_storage_use[node.id]);
        if (rel_at < 0) rel_at = node_index[node.id];  // dead materialized output
        if (rel_at >= 0 && (size_t)rel_at < ctx.release_queue.size()) {
            ctx.release_queue[rel_at].push_back(node.id);
        }
    }
}

// ---------------------------------------------------------------------------
// inject_runtime_shapes
//
// Fills INPUT tensors' actual shapes (substituting runtime_seq_len for any
// dim tagged DynamicKind::SEQ) and patches stateful op params (n_real_tokens
// for GATED_DELTANET_PREFILL.params.i32[6] and SHORTCONV.params.i32[1]).
//
// Called once before execute_graph() by the engine. After this call:
//   - INPUT tensors have their runtime shape filled in
//   - Downstream tensors are filled during execute_graph() dispatch
//   - stateful op params are set so kernels can skip padding positions
//
// In DYNAMIC mode (static_padded=false): runtime_seq_len is the actual token
// count and the graph executes with that seq_len — no padding.
// In STATIC_PADDED mode (static_padded=true): the graph executes with
// padded_seq_len, but stateful ops still need to know runtime_seq_len (the
// real token count) to skip padding positions.
// ---------------------------------------------------------------------------

void inject_runtime_shapes(ExecContext& ctx) {
    auto& nodes  = ctx.graph->nodes;
    auto& tensors = ctx.graph->runtime.tensors;
    const int seq_val = ctx.static_padded ? ctx.padded_seq_len : ctx.runtime_seq_len;
    const int n_real  = ctx.runtime_seq_len;  // always the real token count

    // Phase 1: INPUT tensors' actual shape.
    // Engine fills INPUT tensor data + shape before calling execute_graph;
    // for dynamic dims we override the shape[dim] with the evaluated DimExpr.
    for (auto& n : nodes) {
        if (n.op_type != OpType::INPUT) continue;
        auto& t = tensors[n.id];
        for (int d = 0; d < 4; d++) {
            if (n.dim_expr[d].is_dynamic()) {
                t.shape[d] = eval_dim(n.dim_expr[d], n.out_shape[d], seq_val, n_real);
            }
        }
        t.compute_strides();
    }

    // Phase 2: patch stateful op n_real_tokens + seq_len.
    // In STATIC_PADDED mode (current): graph runs at graph_seq_len (build-time
    // seq_len = 256). Stateful kernels read seq_len from params.i32[3] (GDN)
    // and need it to stay at build-time value so they iterate the full padded
    // range, but skip padding positions via n_real_tokens (params.i32[6]).
    // In DYNAMIC mode (future): runtime_seq_len replaces both seq_len and
    // n_real_tokens in params, kernels iterate only the real tokens.
    if (n_real <= 0) return;
    const bool patch_seq_len = !ctx.static_padded;  // DYNAMIC mode only
    for (auto& n : nodes) {
        if (n.op_type == OpType::RWKV7) {
            if(n.params.i32.size()<=3) n.params.i32.resize(4,0);
            n.params.i32[3]=n_real;
            if(patch_seq_len && n.params.i32.size()>2) n.params.i32[2]=n_real;
        } else if (n.op_type == OpType::RWKV_TOKEN_SHIFT) {
            if(n.params.i32.size()<=2) n.params.i32.resize(3,0);
            n.params.i32[2]=n_real;
            if(patch_seq_len && n.params.i32.size()>1) n.params.i32[1]=n_real;
        } else if (n.op_type == OpType::GATED_DELTANET_PREFILL) {
            if (n.params.i32.size() <= 6) n.params.i32.resize(7, 0);
            n.params.i32[6] = n_real;  // n_real_tokens (skip padding positions)
            if (patch_seq_len && n.params.i32.size() > 3) {
                n.params.i32[3] = n_real;  // seq_len = n_real in DYNAMIC mode
            }
        } else if (n.op_type == OpType::SHORTCONV) {
            if (n.params.i32.size() <= 1) n.params.i32.resize(2, 0);
            n.params.i32[1] = n_real;
            // SHORTCONV reads seq_len from input tensor shape (x.shape[1]),
            // which is set by the engine to graph_seq_len (padding mode) or
            // n_real (dynamic mode).
        }
    }
}

// ---------------------------------------------------------------------------
// CPUBackend::dispatch — kernel dispatcher for CPU (ARM NEON) backend.
//
// Routes OpType to the appropriate kernel. This is the only dispatch
// entry point for CPU; future NPU backend will have its own dispatch().
// ---------------------------------------------------------------------------

void CPUBackend::dispatch(const GraphNode& node,
                          const std::vector<const Tensor*>& inputs,
                          Tensor* output, ThreadPool* thread_pool) {
    const OpType op = node.op_type;
    const OpParams& params = node.params;
    switch (op) {
    case OpType::INPUT:
    case OpType::CONSTANT:
        // no-op — data is already in the tensor
        break;

    case OpType::RESHAPE:
        // Reshape preserves logical element order. For contiguous inputs this is
        // a zero-copy metadata change; for non-contiguous views we materialize a
        // contiguous output.
        //
        // Output shape priority for each dim d:
        //   - if node.dim_expr[d].is_dynamic(): use output->shape[d] (already
        //     filled by execute_graph main loop via eval_dim)
        //   - else: use params.i32[d] (build-time literal)
        if (!inputs.empty() && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            int64_t new_shape[4] = { output->shape[0], output->shape[1],
                                     output->shape[2], output->shape[3] };
            if (params.i32.size() >= 4) {
                for (int d = 0; d < 4; d++) {
                    if (!node.dim_expr[d].is_dynamic()) {
                        new_shape[d] = params.i32[d];
                    }
                    // dynamic dims: preserve output->shape[d] (runtime-filled)
                }
            }

            if (src.is_contiguous()) {
                *output = src;
                output->shape[0] = new_shape[0];
                output->shape[1] = new_shape[1];
                output->shape[2] = new_shape[2];
                output->shape[3] = new_shape[3];
                output->compute_strides();
            } else {
                output->shape[0] = new_shape[0];
                output->shape[1] = new_shape[1];
                output->shape[2] = new_shape[2];
                output->shape[3] = new_shape[3];
                output->compute_strides();
                materialize_strided(src, output->data);
            }
        }
        break;

    case OpType::MATMUL:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            // Fused activation: params.i32[0]=activation, [1]=act_n_begin, [2]=act_n_len.
            // Default: NONE, 0, -1 (whole output).
            Activation act = (Activation)graph_params::get_i32(params, 0, 0);
            int act_n_begin = graph_params::get_i32(params, 1, 0);
            int act_n_len = graph_params::get_i32(params, 2, -1);
            kernel_matmul_fp32(*inputs[0], *inputs[1], *output, thread_pool,
                                act, act_n_begin, act_n_len);
        }
        break;

    case OpType::GEMV_SPARSE_A:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            kernel_gemv_sparse_a(*inputs[0], *inputs[1], *output, thread_pool);
        }
        break;

    case OpType::SDPA:
    case OpType::SDPA_MLA: {
        std::vector<Tensor*> sdpa_outs = { output };
        kernel_sdpa(params, inputs, sdpa_outs, thread_pool);
        break;
    }
    case OpType::GATED_DELTANET_PREFILL: {
        std::vector<Tensor*> gdn_outs = { output };
        kernel_gdn_prefill(params, inputs, gdn_outs, thread_pool);
        break;
    }
    case OpType::GATED_DELTANET_DECODE: {
        std::vector<Tensor*> gdn_outs = { output };
        kernel_gdn_decode(params, inputs, gdn_outs, thread_pool);
        break;
    }
    case OpType::MOE: {
        int hidden_size = graph_params::get_i32(params, 0, output ? (int)output->shape[0] : 0);
        int num_experts = graph_params::get_i32(params, 1, 0);
        int top_k = graph_params::get_i32(params, 2, 0);
        int intermediate_size = graph_params::get_i32(params, 3, 0);
        int shared_intermediate_size = graph_params::get_i32(params, 4, intermediate_size);
        int router_score_func = graph_params::get_i32(params, 5, 0);
        bool norm_topk_prob = graph_params::get_i32(params, 6, 1) != 0;
        bool has_shared_expert = graph_params::get_i32(params, 7, 1) != 0;
        int n_group = graph_params::get_i32(params, 8, 1);
        int topk_group = graph_params::get_i32(params, 9, 1);
        float routed_scaling_factor = graph_params::get_f32(params, 0, 1.0f);
        if (output) {
            kernel_qwen3_moe(inputs, *output, thread_pool,
                             hidden_size, num_experts, top_k,
                             intermediate_size, shared_intermediate_size,
                             router_score_func, norm_topk_prob,
                             has_shared_expert, n_group, topk_group,
                             routed_scaling_factor);
        }
        break;
    }
    case OpType::RWKV_TOKEN_SHIFT:
        kernel_rwkv_token_shift(params, inputs, *output);
        break;
    case OpType::RWKV_MIX:
        kernel_rwkv_mix(params, inputs, *output);
        break;
    case OpType::RWKV_L2_NORM:
        kernel_rwkv_l2_norm(params, inputs, *output);
        break;
    case OpType::RWKV_GROUP_NORM:
        kernel_rwkv_group_norm(params, inputs, *output);
        break;
    case OpType::RWKV_BONUS:
        kernel_rwkv_bonus(params, inputs, *output);
        break;
    case OpType::RWKV_POST:
        kernel_rwkv_post(params, inputs, *output, thread_pool);
        break;
    case OpType::RWKV7:
        kernel_rwkv7(params, inputs, *output, thread_pool);
        break;
    case OpType::SHORTCONV: {
        // ShortConv: depth-wise causal conv1d + silu.
        // Input x comes from matmul: shape [groups, seq], data [seq, groups] row-major.
        //   x_data[s * groups + g] = value at (group g, position s).
        if (inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] && output) {
            const Tensor& x = *inputs[0];
            const Tensor& w = *inputs[1];
            Tensor& out = *output;

            int kernel_size = graph_params::get_i32(params, 0, 4);
            int groups = (int)x.shape[0];
            int seq_len = (int)x.shape[1];

            const float* x_data = x.ptr<float>();
            const float* w_data = w.ptr<float>();
            float* conv_state_data = reinterpret_cast<float*>(inputs[2]->data);
            float* out_data = out.ptr<float>();

#if HAS_NEON
            // Decode fast path: seq_len=1, single position per group.
            if (seq_len == 1 && kernel_size == 4) {
                int prefix_len = kernel_size - 1;  // 3

                for (int g = 0; g < groups; g++) {
                    float* cs = conv_state_data + g * prefix_len;
                    float xg = x_data[g];  // layout [seq, groups], seq=1

                    // Convolution: sum = cs0*w0 + cs1*w1 + cs2*w2 + xg*w3
                    const float* wp = w_data + g * kernel_size;
                    float sum = cs[0] * wp[0] + cs[1] * wp[1] + cs[2] * wp[2] + xg * wp[3];

                    // SiLU: sum * sigmoid(sum)
                    float sig = 1.f / (1.f + expf(-sum));
                    out_data[g * seq_len] = sum * sig;

                    // Update conv_state: shift left, append xg
                    cs[0] = cs[1];
                    cs[1] = cs[2];
                    cs[2] = xg;
                }
                break;
            }
#endif

            int n_real = graph_params::get_i32(params, 1, seq_len);
            int prefix_len = kernel_size - 1;
            int total_len = prefix_len + seq_len;
            int process_len = (n_real > 0 && n_real < seq_len) ? n_real : seq_len;

            std::vector<float> stated(groups * total_len);

            // Phase 1a: batch-copy x into stated with sequential memory access.
            // x data layout is [seq, groups] row-major: x_data[s*groups + g].
            // The per-group loop (s inner) has stride=groups*4 bytes → cache-thrash.
            // Flipping to seq-outer/groups-inner makes x_data access fully sequential.
            // stated[g*total_len + prefix_len + s] = x_data[s*groups + g]
            {
                float* st_base = stated.data();
                for (int s = 0; s < seq_len; s++) {
                    const float* x_row = x_data + s * groups;
                    float* st_row = st_base + prefix_len + s;
                    for (int g = 0; g < groups; g++)
                        st_row[g * total_len] = x_row[g];
                }
            }

            // Process groups in parallel. Each group's phases 1b/2/3 are
            // independent across groups — each touches only its own
            // stated[g*total_len .. (g+1)*total_len) and out_data[g*seq_len ..].
            auto process_group_range = [&](int /*tid*/, int g_start, int g_end) {
                for (int g = g_start; g < g_end; g++) {
                    float* dst = stated.data() + g * total_len;
                    float* cs = conv_state_data + g * prefix_len;

                    // Phase 1b: prepend conv_state (prefix_len elements, contiguous)
                    for (int p = 0; p < prefix_len; p++) dst[p] = cs[p];

                    // Phase 2: conv1d + silu
                    const float* w_ptr = w_data + g * kernel_size;
                    const float* st = stated.data() + g * total_len;
                    float* ot = out_data + g * seq_len;
                    for (int i = 0; i < seq_len; i++) ot[i] = 0.f;

#if HAS_NEON
                    float32x4_t w4 = vld1q_f32(w_ptr);
                    for (int i = 0; i < process_len; i++) {
                        float32x4_t st4 = vld1q_f32(st + i);
                        float sum = vaddvq_f32(vmulq_f32(st4, w4));
                        float sig = 1.f / (1.f + std::exp(-sum));
                        ot[i] = sum * sig;
                    }
#else
                    for (int i = 0; i < process_len; i++) {
                        float sum = 0.f;
                        for (int k = 0; k < kernel_size; k++)
                            sum += st[i + k] * w_ptr[k];
                        float sig = 1.f / (1.f + std::exp(-sum));
                        ot[i] = sum * sig;
                    }
#endif

                    // Phase 3: update conv_state (last prefix_len of stated)
                    if (process_len > 0) {
                        int last_real_pos = prefix_len + process_len - 1;
                        for (int p = 0; p < prefix_len; p++)
                            cs[p] = st[last_real_pos - prefix_len + 1 + p];
                    }
                }
            };

            // Parallelize over groups when worthwhile (same pattern as
            // gdn_prefill.cpp:100-103). groups is large (6144 for 0.8B,
            // 8192 for 4B), so 4 threads each get ~1500-2000 groups.
            if (thread_pool && groups >= 4) {
                int chunk = (groups + thread_pool->num_threads() - 1)
                            / thread_pool->num_threads();
                if (chunk < 1) chunk = 1;
                thread_pool->parallel_for(0, groups, chunk, process_group_range);
            } else {
                process_group_range(0, 0, groups);
            }
        }
        break;
    }
    case OpType::ROTARY_EMBED:
        if (inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] && output) {
            int rope_dim = graph_params::get_i32(params, 0, 64);
            bool interleave = graph_params::get_i32(params, 1, 1) != 0;
            kernel_rope(*inputs[0], *inputs[1], *inputs[2], rope_dim, interleave, *output);
        }
        break;

    case OpType::RMS_NORM:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            float eps = graph_params::get_f32(params, 0, 1e-6f);
            kernel_rms_norm(*inputs[0], *inputs[1], eps, *output);
        }
        break;

    case OpType::LAYER_NORM:
        if (inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] && output) {
            float eps = graph_params::get_f32(params, 0, 1e-5f);
            kernel_layer_norm(*inputs[0], *inputs[1], *inputs[2], eps, *output,
                              thread_pool);
        }
        break;

    case OpType::CONCAT:
        // Concatenate inputs along specified dimension. Materializes output and
        // respects input strides, so view inputs remain correct.
        //
        // Fast path for dim==0: for each (i1, i2, i3), the i0 elements are
        // contiguous in both src (stride[0]==es) and dst (stride[0]==es).
        // So we can memcpy the whole i0 block at once per (i1,i2,i3),
        // reducing memcpy calls from shape[0]*shape[1]*shape[2]*shape[3]
        // to shape[1]*shape[2]*shape[3].
        //
        // This is correct regardless of whether the outer dims are contiguous
        // — we still iterate i1/i2/i3 with their respective strides.
        if (!inputs.empty() && output) {
            int dim = graph_params::get_i32(params, 0, 0);
            size_t es = output->element_size();
            int64_t dim_offset = 0;
            char* dst_base = static_cast<char*>(output->data);

            if (dim == 0) {
                // Fast path: dim=0 concat with per-(i1,i2,i3) block copy.
                // i0 is contiguous in both src and dst (stride[0] == es).
                for (size_t i = 0; i < inputs.size(); i++) {
                    if (!inputs[i] || !inputs[i]->data) continue;
                    const Tensor& src = *inputs[i];
                    const char* src_base = static_cast<const char*>(src.data);
                    int64_t i0_bytes = (int64_t)src.shape[0] * es;
                    int64_t dst_o0_offset = dim_offset * output->stride[0];

                    for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
                        for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
                            for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                                const char* src_ptr = src_base
                                    + i1 * src.stride[1]
                                    + i2 * src.stride[2]
                                    + i3 * src.stride[3];
                                char* dst_ptr = dst_base + dst_o0_offset
                                    + i1 * output->stride[1]
                                    + i2 * output->stride[2]
                                    + i3 * output->stride[3];
                                std::memcpy(dst_ptr, src_ptr, i0_bytes);
                            }
                        }
                    }
                    dim_offset += src.shape[0];
                }
            } else {
                // General path: per-element copy for dim != 0
                for (size_t i = 0; i < inputs.size(); i++) {
                    if (!inputs[i] || !inputs[i]->data) continue;
                    const Tensor& src = *inputs[i];
                    const char* src_base = static_cast<const char*>(src.data);
                    for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
                        for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
                            for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                                for (int64_t i0 = 0; i0 < src.shape[0]; i0++) {
                                    int64_t o0 = i0, o1 = i1, o2 = i2, o3 = i3;
                                    if (dim == 1) o1 += dim_offset;
                                    else if (dim == 2) o2 += dim_offset;
                                    else if (dim == 3) o3 += dim_offset;

                                    const char* src_ptr = src_base
                                        + i0 * src.stride[0]
                                        + i1 * src.stride[1]
                                        + i2 * src.stride[2]
                                        + i3 * src.stride[3];
                                    char* dst_ptr = dst_base
                                        + o0 * output->stride[0]
                                        + o1 * output->stride[1]
                                        + o2 * output->stride[2]
                                        + o3 * output->stride[3];
                                    std::memcpy(dst_ptr, src_ptr, es);
                                }
                            }
                        }
                    }
                    dim_offset += src.shape[dim];
                }
            }
        }
        break;

    case OpType::ADD:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            const Tensor& a = *inputs[0];
            const Tensor& b = *inputs[1];
            int N = (int)a.nelements();
            float* o = output->ptr<float>();
            const float* ap = a.ptr<float>();
            const float* bp = b.ptr<float>();
            if (a.nelements() != b.nelements() && b.nelements() != 1 &&
                broadcasts_to(b, a)) {
                binary_broadcast_from_b(o, a, b,
                                        [](float av, float bv) { return av + bv; });
                break;
            }
            if(thread_pool&&thread_pool->num_threads()>1&&N>=32768&&
               a.is_contiguous()&&b.is_contiguous()&&output->is_contiguous()&&
               (a.nelements()==b.nelements()||b.nelements()==1)) {
                int chunk=(N+thread_pool->num_threads()-1)/thread_pool->num_threads();
                chunk=((chunk+7)/8)*8;
                thread_pool->parallel_for(0,N,chunk,[&](int,int begin,int end) {
                    int i=begin;
#if HAS_NEON
                    if(a.nelements()==b.nelements()) {
                        for(;i+7<end;i+=8) {
                            vst1q_f32(o+i,vaddq_f32(vld1q_f32(ap+i),vld1q_f32(bp+i)));
                            vst1q_f32(o+i+4,vaddq_f32(vld1q_f32(ap+i+4),vld1q_f32(bp+i+4)));
                        }
                    } else {
                        float32x4_t bv=vdupq_n_f32(bp[0]);
                        for(;i+7<end;i+=8) {
                            vst1q_f32(o+i,vaddq_f32(vld1q_f32(ap+i),bv));
                            vst1q_f32(o+i+4,vaddq_f32(vld1q_f32(ap+i+4),bv));
                        }
                    }
#endif
                    if(a.nelements()==b.nelements()) for(;i<end;++i)o[i]=ap[i]+bp[i];
                    else for(;i<end;++i)o[i]=ap[i]+bp[0];
                });
                break;
            }
#if HAS_NEON
            if (a.nelements() == b.nelements()) {
                int i = 0;
                for (; i + 7 < N; i += 8) {
                    vst1q_f32(o + i,     vaddq_f32(vld1q_f32(ap + i),     vld1q_f32(bp + i)));
                    vst1q_f32(o + i + 4, vaddq_f32(vld1q_f32(ap + i + 4), vld1q_f32(bp + i + 4)));
                }
                for (; i < N; i++) o[i] = ap[i] + bp[i];
            } else if (b.nelements() == 1) {
                float bv = bp[0];
                float32x4_t bv4 = vdupq_n_f32(bv);
                int i = 0;
                for (; i + 7 < N; i += 8) {
                    vst1q_f32(o + i,     vaddq_f32(vld1q_f32(ap + i),     bv4));
                    vst1q_f32(o + i + 4, vaddq_f32(vld1q_f32(ap + i + 4), bv4));
                }
                for (; i < N; i++) o[i] = ap[i] + bv;
            } else {
                for (int i = 0; i < N; i++) o[i] = ap[i] + bp[i];
            }
#else
            if (a.nelements() == b.nelements()) {
                for (int i = 0; i < N; i++) o[i] = ap[i] + bp[i];
            } else if (b.nelements() == 1) {
                float bv = bp[0];
                for (int i = 0; i < N; i++) o[i] = ap[i] + bv;
            } else {
                for (int i = 0; i < N; i++) o[i] = ap[i] + bp[i];
            }
#endif
        }
        break;

    case OpType::MUL:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            const Tensor& a = *inputs[0];
            const Tensor& b = *inputs[1];
            if (!a.data || !b.data) {
                fprintf(stderr, "MUL: null data! a.data=%p b.data=%p a_shape=[%lld,%lld,%lld,%lld] b_shape=[%lld,%lld,%lld,%lld] a_prec=%d b_prec=%d\n",
                        a.data, b.data,
                        a.shape[0], a.shape[1], a.shape[2], a.shape[3],
                        b.shape[0], b.shape[1], b.shape[2], b.shape[3],
                        (int)a.prec, (int)b.prec);
                break;
            }
            float* o = output->ptr<float>();
            if (a.nelements() != b.nelements() && broadcasts_to(b, a)) {
                binary_broadcast_from_b(o, a, b,
                                        [](float av, float bv) { return av * bv; });
                break;
            }
            int N=(int)a.nelements();
            if(thread_pool&&thread_pool->num_threads()>1&&N>=32768&&
               a.nelements()==b.nelements()&&a.is_contiguous()&&b.is_contiguous()&&
               output->is_contiguous()) {
                const float* ap=a.ptr<float>(); const float* bp=b.ptr<float>();
                int chunk=(N+thread_pool->num_threads()-1)/thread_pool->num_threads();
                chunk=((chunk+7)/8)*8;
                thread_pool->parallel_for(0,N,chunk,[&](int,int begin,int end) {
                    int i=begin;
#if HAS_NEON
                    for(;i+7<end;i+=8) {
                        vst1q_f32(o+i,vmulq_f32(vld1q_f32(ap+i),vld1q_f32(bp+i)));
                        vst1q_f32(o+i+4,vmulq_f32(vld1q_f32(ap+i+4),vld1q_f32(bp+i+4)));
                    }
#endif
                    for(;i<end;++i)o[i]=ap[i]*bp[i];
                });
                break;
            }
#if HAS_NEON
            binary_stride_aware(o, a, b,
                [](float32x4_t av, float32x4_t bv) { return vmulq_f32(av, bv); });
#else
            // Scalar fallback: iterate outer dims, inner contiguous run.
            const char* a_base = static_cast<const char*>(a.data);
            const char* b_base = static_cast<const char*>(b.data);
            StrideIter it = compute_stride_iter(a);
            float* dst_row = o;
            for (int i3 = 0; i3 < it.d3; i3++) {
                const char* pa3 = a_base + i3 * it.s3;
                const char* pb3 = b_base + i3 * it.s3;
                for (int i2 = 0; i2 < it.d2; i2++) {
                    const char* pa2 = pa3 + i2 * it.s2;
                    const char* pb2 = pb3 + i2 * it.s2;
                    for (int i1 = 0; i1 < it.d1; i1++) {
                        const float* ap = reinterpret_cast<const float*>(pa2 + i1 * it.s1);
                        const float* bp = reinterpret_cast<const float*>(pb2 + i1 * it.s1);
                        for (int i = 0; i < it.n_inner; i++) dst_row[i] = ap[i] * bp[i];
                        dst_row += it.n_inner;
                    }
                }
            }
#endif
        }
        break;

    case OpType::SILU:
        // SiLU: x * sigmoid(x). Stride-aware — handles SLICE views (gate/up
        // halves of merged gate_up matmul).
        if (inputs.size() >= 1 && inputs[0] && output) {
            float* o = output->ptr<float>();
#if HAS_NEON
            unary_stride_aware(o, *inputs[0],
                [](float32x4_t x) {
                    float32x4_t sv = sigmoid_f32_neon(x);
                    return vmulq_f32(x, sv);
                });
#else
            // Scalar fallback (stride-aware).
            const Tensor& src = *inputs[0];
            const char* src_base = static_cast<const char*>(src.data);
            StrideIter it = compute_stride_iter(src);
            float* dst_row = o;
            for (int i3 = 0; i3 < it.d3; i3++) {
                const char* p3 = src_base + i3 * it.s3;
                for (int i2 = 0; i2 < it.d2; i2++) {
                    const char* p2 = p3 + i2 * it.s2;
                    for (int i1 = 0; i1 < it.d1; i1++) {
                        const float* sp = reinterpret_cast<const float*>(p2 + i1 * it.s1);
                        for (int i = 0; i < it.n_inner; i++) {
                            float sx = 1.f / (1.f + std::exp(-sp[i]));
                            dst_row[i] = sp[i] * sx;
                        }
                        dst_row += it.n_inner;
                    }
                }
            }
#endif
        }
        break;

    case OpType::TANH:
        if (inputs.size() >= 1 && inputs[0] && output) {
            const Tensor& src=*inputs[0]; const char* base=(const char*)src.data;
            StrideIter it=compute_stride_iter(src); float* dst=output->ptr<float>();
            for(int i3=0;i3<it.d3;i3++) for(int i2=0;i2<it.d2;i2++) for(int i1=0;i1<it.d1;i1++) {
                const float* row=(const float*)(base+i3*it.s3+i2*it.s2+i1*it.s1);
                for(int i=0;i<it.n_inner;i++) dst[i]=std::tanh(row[i]);
                dst += it.n_inner;
            }
        }
        break;

    case OpType::SWIGLU:
        // Fused SwiGLU over a merged [2I, ...] tensor: out[i] = silu(gate[i]) * up[i],
        // gate = row[0..I), up = row[I..2I). Reads both halves from the single merged
        // row (stride-aware); output is dense [I, ...]. NOT a slice-view consumer.
        if (inputs.size() >= 1 && inputs[0] && output) {
            const Tensor& m = *inputs[0];
            const char* m_base = static_cast<const char*>(m.data);
            StrideIter it = compute_stride_iter(m);   // it.n_inner = 2I
            const int I = it.n_inner / 2;
            float* dst_row = output->ptr<float>();
            for (int i3 = 0; i3 < it.d3; i3++) {
                const char* p3 = m_base + i3 * it.s3;
                for (int i2 = 0; i2 < it.d2; i2++) {
                    const char* p2 = p3 + i2 * it.s2;
                    for (int i1 = 0; i1 < it.d1; i1++) {
                        const float* row = reinterpret_cast<const float*>(p2 + i1 * it.s1);
                        const float* gate = row;
                        const float* up   = row + I;
                        for (int i = 0; i < I; i++) {
                            float g = gate[i];
                            float sg = g / (1.f + std::exp(-g));  // silu(g)
                            dst_row[i] = sg * up[i];
                        }
                        dst_row += I;
                    }
                }
            }
        }
        break;

    case OpType::SIGMOID:
        if (inputs.size() >= 1 && inputs[0] && output) {
            float* o = output->ptr<float>();
#if HAS_NEON
            unary_stride_aware(o, *inputs[0],
                [](float32x4_t x) { return sigmoid_f32_neon(x); });
#else
            const Tensor& src = *inputs[0];
            const char* src_base = static_cast<const char*>(src.data);
            StrideIter it = compute_stride_iter(src);
            float* dst_row = o;
            for (int i3 = 0; i3 < it.d3; i3++) {
                const char* p3 = src_base + i3 * it.s3;
                for (int i2 = 0; i2 < it.d2; i2++) {
                    const char* p2 = p3 + i2 * it.s2;
                    for (int i1 = 0; i1 < it.d1; i1++) {
                        const float* sp = reinterpret_cast<const float*>(p2 + i1 * it.s1);
                        for (int i = 0; i < it.n_inner; i++)
                            dst_row[i] = 1.f / (1.f + std::exp(-sp[i]));
                        dst_row += it.n_inner;
                    }
                }
            }
#endif
        }
        break;

    case OpType::SIGMOID_EXACT:
        if (inputs.size() >= 1 && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            const int64_t total = src.nelements();
            if (thread_pool && thread_pool->num_threads() > 1 &&
                src.is_contiguous() && total >= 32768) {
                const float* input = src.ptr<float>();
                float* dst = output->ptr<float>();
                int chunk = (int)((total + thread_pool->num_threads() - 1) /
                                  thread_pool->num_threads());
                chunk = std::max(chunk, 4096);
                thread_pool->parallel_for(0, (int)total, chunk,
                    [&](int, int begin, int end) {
                        for (int i = begin; i < end; ++i)
                            dst[i] = 1.f / (1.f + std::exp(-input[i]));
                    });
                break;
            }
            const char* src_base = static_cast<const char*>(src.data);
            StrideIter it = compute_stride_iter(src);
            float* dst = output->ptr<float>();
            for (int i3 = 0; i3 < it.d3; i3++) {
                const char* p3 = src_base + i3 * it.s3;
                for (int i2 = 0; i2 < it.d2; i2++) {
                    const char* p2 = p3 + i2 * it.s2;
                    for (int i1 = 0; i1 < it.d1; i1++) {
                        const float* row = reinterpret_cast<const float*>(p2 + i1 * it.s1);
                        for (int i = 0; i < it.n_inner; i++)
                            dst[i] = 1.f / (1.f + std::exp(-row[i]));
                        dst += it.n_inner;
                    }
                }
            }
        }
        break;

    case OpType::EXP:
        if (inputs.size() >= 1 && inputs[0] && output) {
            float* o = output->ptr<float>();
#if HAS_NEON
            unary_stride_aware(o, *inputs[0],
                [](float32x4_t x) {
                    // Reuse fast_exp from sigmoid: exp(x) = sigmoid(x) / (1 - sigmoid(x))
                    // But simpler: use the polynomial exp approximation directly.
                    // Clamp to avoid overflow.
                    x = vmaxq_f32(x, vdupq_n_f32(-88.f));
                    x = vminq_f32(x, vdupq_n_f32(88.f));
                    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
                    float32x4_t t = vmulq_f32(x, log2e);
                    float32x4_t n = vrndmq_f32(t);
                    float32x4_t f = vsubq_f32(t, n);
                    int32x4_t ni = vcvtq_s32_f32(n);
                    float32x4_t pow2n = vreinterpretq_f32_s32(
                        vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));
                    const float32x4_t c0 = vdupq_n_f32(1.0f);
                    const float32x4_t c1 = vdupq_n_f32(0.6931472f);
                    const float32x4_t c2 = vdupq_n_f32(0.2402265f);
                    const float32x4_t c3 = vdupq_n_f32(0.0555049f);
                    const float32x4_t c4 = vdupq_n_f32(0.0096813f);
                    float32x4_t pow2f = vfmaq_f32(c3, c4, f);
                    pow2f = vfmaq_f32(c2, pow2f, f);
                    pow2f = vfmaq_f32(c1, pow2f, f);
                    pow2f = vfmaq_f32(c0, pow2f, f);
                    return vmulq_f32(pow2n, pow2f);
                });
#else
            const Tensor& src = *inputs[0];
            const char* src_base = static_cast<const char*>(src.data);
            StrideIter it = compute_stride_iter(src);
            float* dst_row = o;
            for (int i3 = 0; i3 < it.d3; i3++) {
                const char* p3 = src_base + i3 * it.s3;
                for (int i2 = 0; i2 < it.d2; i2++) {
                    const char* p2 = p3 + i2 * it.s2;
                    for (int i1 = 0; i1 < it.d1; i1++) {
                        const float* sp = reinterpret_cast<const float*>(p2 + i1 * it.s1);
                        for (int i = 0; i < it.n_inner; i++)
                            dst_row[i] = std::exp(sp[i]);
                        dst_row += it.n_inner;
                    }
                }
            }
#endif
        }
        break;

    case OpType::EXP_EXACT:
        if (inputs.size() >= 1 && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            const int64_t total = src.nelements();
            if (thread_pool && thread_pool->num_threads() > 1 &&
                src.is_contiguous() && total >= 32768) {
                const float* input = src.ptr<float>();
                float* dst = output->ptr<float>();
                int chunk = (int)((total + thread_pool->num_threads() - 1) /
                                  thread_pool->num_threads());
                chunk = std::max(chunk, 4096);
                thread_pool->parallel_for(0, (int)total, chunk,
                    [&](int, int begin, int end) {
                        for (int i=begin;i<end;++i) dst[i]=std::exp(input[i]);
                    });
                break;
            }
            const char* base=static_cast<const char*>(src.data);
            StrideIter it=compute_stride_iter(src);
            float* dst=output->ptr<float>();
            for(int i3=0;i3<it.d3;++i3) for(int i2=0;i2<it.d2;++i2)
                for(int i1=0;i1<it.d1;++i1) {
                    const float* row=reinterpret_cast<const float*>(
                        base+i3*it.s3+i2*it.s2+i1*it.s1);
                    for(int i=0;i<it.n_inner;++i) *dst++=std::exp(row[i]);
                }
        }
        break;

    case OpType::SOFTPLUS:
        if (inputs.size() >= 1 && inputs[0] && output) {
            float* o = output->ptr<float>();
#if HAS_NEON
            unary_stride_aware(o, *inputs[0],
                [](float32x4_t x) {
                    // softplus(x) = log(1 + exp(x))
                    // For large x, softplus ≈ x. For small x, ≈ exp(x).
                    x = vmaxq_f32(x, vdupq_n_f32(-20.f));
                    x = vminq_f32(x, vdupq_n_f32(20.f));
                    // exp(x) via the same polynomial as EXP
                    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
                    float32x4_t t = vmulq_f32(x, log2e);
                    float32x4_t n = vrndmq_f32(t);
                    float32x4_t f = vsubq_f32(t, n);
                    int32x4_t ni = vcvtq_s32_f32(n);
                    float32x4_t pow2n = vreinterpretq_f32_s32(
                        vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));
                    const float32x4_t c0 = vdupq_n_f32(1.0f);
                    const float32x4_t c1 = vdupq_n_f32(0.6931472f);
                    const float32x4_t c2 = vdupq_n_f32(0.2402265f);
                    const float32x4_t c3 = vdupq_n_f32(0.0555049f);
                    const float32x4_t c4 = vdupq_n_f32(0.0096813f);
                    float32x4_t pow2f = vfmaq_f32(c3, c4, f);
                    pow2f = vfmaq_f32(c2, pow2f, f);
                    pow2f = vfmaq_f32(c1, pow2f, f);
                    pow2f = vfmaq_f32(c0, pow2f, f);
                    float32x4_t ex = vmulq_f32(pow2n, pow2f);
                    // log(1 + ex) — approximate via log2:
                    // log2(1+ex) ≈ log2(max(ex, 1)) for ex >> 1
                    // But for correctness, use: vlogq_f32 if available, else scalar fallback.
                    // NEON doesn't have vlogq_f32, so we do scalar for the log part.
                    // Fast path: if x > 10, return x (softplus ≈ x).
                    uint32x4_t mask_large = vcgtq_f32(x, vdupq_n_f32(10.f));
                    float32x4_t result = x;  // fallback for large x
                    // For smaller values, compute log(1+exp) scalar.
                    float tmp[4];
                    vst1q_f32(tmp, ex);
                    for (int i = 0; i < 4; i++) tmp[i] = std::log1pf(tmp[i]);
                    float32x4_t log_result = vld1q_f32(tmp);
                    // Select: large x → x, small x → log(1+exp)
                    return vbslq_f32(mask_large, result, log_result);
                });
#else
            const Tensor& src = *inputs[0];
            const char* src_base = static_cast<const char*>(src.data);
            StrideIter it = compute_stride_iter(src);
            float* dst_row = o;
            for (int i3 = 0; i3 < it.d3; i3++) {
                const char* p3 = src_base + i3 * it.s3;
                for (int i2 = 0; i2 < it.d2; i2++) {
                    const char* p2 = p3 + i2 * it.s2;
                    for (int i1 = 0; i1 < it.d1; i1++) {
                        const float* sp = reinterpret_cast<const float*>(p2 + i1 * it.s1);
                        for (int i = 0; i < it.n_inner; i++) {
                            float v = sp[i];
                            if (v > 20.f) dst_row[i] = v;
                            else if (v < -20.f) dst_row[i] = std::exp(v);
                            else dst_row[i] = std::log1pf(std::exp(v));
                        }
                        dst_row += it.n_inner;
                    }
                }
            }
#endif
        }
        break;

    case OpType::SLICE:
        // zero-copy: slice produces a view of the parent and must preserve the
        // parent's stride layout.
        if (!inputs.empty() && inputs[0] && output) {
            int dim    = graph_params::get_i32(params, 0, 0);
            int offset = graph_params::get_i32(params, 1, 0);
            int size   = graph_params::get_i32(params, 2, (int)output->shape[dim]);
            *output = *inputs[0];
            if (offset >= 0) {
                size_t byte_off = (size_t)offset * inputs[0]->stride[dim];
                output->data = static_cast<char*>(inputs[0]->data) + byte_off;
            }
            output->shape[dim] = size;
        }
        break;

    case OpType::TILE:
        // Tile: replicate input along each dimension by the given multipliers.
        //
        // Fast path for dim=2-only tile (common MLA case:
        // k_rope [rope_dim, seq, 1] → [rope_dim, seq, num_heads]):
        // Each rep is a copy of src with the same (i0, i1) layout.
        // For each (i0, i1) pair, copy reps[2] times to dst.
        // This is correct regardless of stride layout.
        //
        // General path: per-element copy with div/mod.
        if (!inputs.empty() && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            size_t es = src.element_size();
            int reps[4] = {1, 1, 1, 1};
            for (int d = 0; d < 4 && d < (int)params.i32.size(); d++) {
                reps[d] = params.i32[d];
            }
            char* dst = static_cast<char*>(output->data);
            const char* s = static_cast<const char*>(src.data);

            bool only_dim2 = (reps[0] == 1 && reps[1] == 1 && reps[3] == 1 && reps[2] > 1);

            if (only_dim2) {
                // For each (i0, i1), copy src[i0][i1] to dst[i0][i1][r] for r=0..reps[2]-1.
                // src is [shape0, shape1, 1, 1], dst is [shape0, shape1, reps2, 1].
                // For fixed (i0, i1), src elements are contiguous (stride[0]==es
                // in inner dim). But i0 may not be contiguous across i1.
                // Safe approach: per (i0, i1) row, copy shape0 elements reps[2] times.
                int64_t i0_bytes = (int64_t)src.shape[0] * es;
                for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                    const char* src_row = s + i1 * src.stride[1];
                    // In dst, [i0=0..shape0-1, i1, r, 0] — for fixed i1, r,
                    // i0 is contiguous (stride[0]==es).
                    char* dst_row_base = dst + i1 * output->stride[1];
                    for (int r = 0; r < reps[2]; r++) {
                        char* dst_row = dst_row_base + r * output->stride[2];
                        std::memcpy(dst_row, src_row, i0_bytes);
                    }
                }
            } else {
                // General path
                int64_t total = output->nelements();
                for (int64_t o = 0; o < total; o++) {
                    int64_t idx = o;
                    int64_t src_off = 0;
                    for (int d = 0; d < 4; d++) {
                        int64_t dim_idx = idx % output->shape[d];
                        idx /= output->shape[d];
                        int64_t src_idx = dim_idx % src.shape[d];
                        src_off += src_idx * (int64_t)(src.stride[d] / es);
                    }
                    std::memcpy(dst + o * es, s + src_off * es, es);
                }
            }
        }
        break;

    case OpType::PERMUTE:
        if (!inputs.empty() && inputs[0] && output && params.i32.size() >= 4) {
            *output = inputs[0]->permute(params.i32[0], params.i32[1],
                                         params.i32[2], params.i32[3]);
        }
        break;

    case OpType::CONTIGUOUS:
        // Materialize to row-major contiguous buffer.
        if (!inputs.empty() && inputs[0] && inputs[0]->data && output && output->data) {
            materialize_strided(*inputs[0], output->data);
        }
        break;
        break;

    default:
        fprintf(stderr, "execute: unhandled op_type %u\n", (uint32_t)op);
        break;
    }
}

// ---------------------------------------------------------------------------
// execute_graph — pure executor, no shape inference
//
// With the multi-graph architecture, all shapes are static at graph-build
// time.  The prefill graph is built with seq_len=N, the decode graph with
// seq_len=1.  The only runtime-dynamic element is the KV cache, which uses
// embedded CacheMetadata for past_len tracking.
// ---------------------------------------------------------------------------

void execute_graph(ExecContext& ctx) {
    auto& nodes  = ctx.graph->nodes;
    auto& tensors = ctx.graph->runtime.tensors;
    auto* pool   = ctx.pool;
    // Device-resident backends (Metal) keep intermediates in device buffers,
    // so borrowed-view detection can't rely on host-pointer equality; classify
    // views by op type instead, and skip the host owner-id assertions.
    const bool device_resident = ctx.backend && ctx.backend->is_device_resident();

    // Compute seq_val once for dynamic shape injection.
    // In DYNAMIC mode this = runtime_seq_len; in STATIC_PADDED mode = padded_seq_len.
    const int seq_val = ctx.static_padded ? ctx.padded_seq_len : ctx.runtime_seq_len;
    const bool has_dynamic = (seq_val > 0);
    const bool same_shape_workspace =
        ctx.reuse_same_shape_workspace &&
        ctx.workspace_shape_valid &&
        ctx.workspace_runtime_seq_len == ctx.runtime_seq_len &&
        ctx.workspace_runtime_batch == ctx.runtime_batch &&
        ctx.workspace_static_padded == ctx.static_padded &&
        ctx.workspace_padded_seq_len == ctx.padded_seq_len;
    const bool reuse_workspace =
        (ctx.reuse_static_workspace && !has_dynamic) ||
        (has_dynamic && same_shape_workspace);

    // Reset non-INPUT/CONSTANT tensor state before execution. Borrowed views
    // must always be cleared because their source storage may be overwritten
    // or released between calls. Materialized tensors are released by default;
    // fully static graphs may opt into keeping them as reusable workspace.
    //
    // IMPORTANT: only release buffers owned by materialized ops. View ops
    // (zero-copy RESHAPE/PERMUTE/SLICE) borrow data from their inputs;
    // releasing their borrowed pointer would double-free the real owner.
    // INPUT/CONSTANT tensors keep their data (set by engine / load-time,
    // e.g. cache_k/cache_v which are mmap'd or load-time allocated).
    auto is_always_borrowed_view_op = [](OpType op) {
        return op == OpType::PERMUTE || op == OpType::SLICE;
    };
    // First classify borrowed views while all producer pointers are still
    // intact. View chains are common; mutating an earlier view before
    // classifying a later one would lose the pointer equality evidence.
    std::vector<uint8_t> borrowed_view(nodes.size(), 0);
    const std::vector<uint32_t> empty_release_batch;
    for (size_t i = 0; i < nodes.size(); i++) {
        auto& node = nodes[i];
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        auto& t = tensors[node.id];

        bool borrowed = is_always_borrowed_view_op(node.op_type);
        if (node.op_type == OpType::RESHAPE && !node.inputs.empty()) {
            const Tensor& src = tensors[node.inputs[0]];
            // On a device backend the output host pointer may be a device alias
            // (or null), so shares_storage() pointer equality is unreliable.
            // RESHAPE borrows iff its input is contiguous — known statically.
            borrowed = device_resident ? src.is_contiguous()
                                       : shares_storage(t, src);
        }
        borrowed_view[node.id] = borrowed ? 1 : 0;
    }

    // Clear borrowed views without releasing their borrowed pointer.
    for (size_t i = 0; i < nodes.size(); i++) {
        auto& node = nodes[i];
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        if (!borrowed_view[node.id]) continue;
        auto& t = tensors[node.id];
        t.data = nullptr;
        t.mem_type = MemoryType::NONE;
        t.owner_id = 0;
        t.storage_id = 0;
    }

    if (!reuse_workspace) {
        // Then release materialized outputs. Borrowed views were nulled above,
        // so RESHAPE only reaches this path when it actually materialized a
        // buffer.
        for (size_t i = 0; i < nodes.size(); i++) {
            auto& node = nodes[i];
            if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
            if (borrowed_view[node.id]) continue;
            auto& t = tensors[node.id];
            if (t.data && t.mem_type == MemoryType::POOLED && t.nbytes() > 0) {
                if (!device_resident && t.owner_id != 0 && t.owner_id != pool->id()) {
                    std::fprintf(stderr,
                                 "execute_graph: owner mismatch before release for node %u (%p owner=%u pool=%u)\n",
                                 node.id, t.data, t.owner_id, pool->id());
                    assert(false && "execute_graph owner mismatch");
                    return;
                }
                ctx.backend->free_output(t, pool);
            }
            t.data = nullptr;
            t.device_data = nullptr;
            t.device_offset = 0;
            t.mem_type = MemoryType::NONE;
            t.owner_id = 0;
            t.storage_id = 0;
        }
    }

    for (size_t i = 0; i < nodes.size(); i++) {
        auto& node = nodes[i];

        // skip inputs / constants — data is pre-set by the caller
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) {
            continue;
        }

        auto& out = tensors[node.id];
        auto start_time = std::chrono::steady_clock::time_point();
        if (ctx.profile_enabled && node.id < ctx.profile_stats.size()) {
            start_time = std::chrono::steady_clock::now();
        }

        // initialise output shape from node definition.
        // Dynamic dims evaluated against runtime seq_val via DimExpr.
        //
        // For zero-copy ops (RESHAPE/PERMUTE/SLICE) we must NOT skip this step
        // entirely in DYNAMIC mode: the dispatch reads output->shape[d] for the
        // dynamic dims, and skipping would leave stale values from the previous
        // execute_graph() call. Only skip when the graph is fully STATIC
        // (has_dynamic == false) — then shapes are constants and reuse is safe.
        //
        // When out.data is already set (zero-copy view inheriting from input),
        // we still must update shape[d] for dynamic dims, but we must NOT call
        // compute_strides() unconditionally — the dispatch path will handle
        // strides itself (see RESHAPE dispatch).
        bool skip_init = (node.op_type == OpType::RESHAPE ||
                          node.op_type == OpType::PERMUTE ||
                          node.op_type == OpType::SLICE)
                         && out.data != nullptr
                         && !has_dynamic;
        if (!skip_init) {
            // Always update dynamic dims via eval_dim, even for zero-copy views.
            // For static dims, fall back to build-time literal.
            bool any_dynamic = false;
            for (int d = 0; d < 4; d++) {
                if (has_dynamic && node.dim_expr[d].is_dynamic()) {
                    out.shape[d] = eval_dim(node.dim_expr[d], node.out_shape[d],
                                            seq_val, ctx.runtime_batch);
                    any_dynamic = true;
                } else {
                    out.shape[d] = node.out_shape[d];
                }
            }
            out.prec     = node.out_prec;
            // For zero-copy views with dynamic dims, defer stride computation
            // to dispatch (which knows whether to inherit from src or
            // materialise). Computing strides here with the new (correct)
            // shape would overwrite the inherited contiguous layout.
            if (!(any_dynamic && out.data != nullptr &&
                  (node.op_type == OpType::RESHAPE ||
                   node.op_type == OpType::PERMUTE ||
                   node.op_type == OpType::SLICE))) {
                out.compute_strides();
            }
        }

        // collect inputs before allocation so zero-copy views can avoid
        // acquiring a buffer they will immediately overwrite with borrowed data.
        std::vector<const Tensor*> inputs;
        inputs.reserve(node.inputs.size());
        for (uint32_t inp_id : node.inputs) {
            inputs.push_back(&tensors[inp_id]);
        }

        // allocate output if needed
        if (out.data == nullptr) {
            size_t nbytes = out.nbytes();
            bool needs_allocation = true;
            if (node.op_type == OpType::PERMUTE || node.op_type == OpType::SLICE) {
                needs_allocation = false;
            } else if (node.op_type == OpType::RESHAPE) {
                needs_allocation = !(inputs.size() >= 1 && inputs[0] && inputs[0]->is_contiguous());
            }
            if (nbytes > 0) {
                if (needs_allocation) {
                    // alloc_output() sets out.data/mem_type/owner_id/storage_id.
                    // Default (CPU) impl is the old host BufferPool path;
                    // a device backend allocates an MTLBuffer and records it in
                    // out.device_data.
                    void* buf = ctx.backend->alloc_output(out, nbytes, pool);
                    if (!buf) {
                        fprintf(stderr, "execute: pool acquire failed for node %u (%zu bytes)\n",
                                node.id, nbytes);
                        return;
                    }
                }
            }
        }

        // Fate-style cross-layer gate prediction. At the beginning of MoE
        // layer i, the gate input is already available. Feed its copied decode
        // vector to the real router from the next MoE layer on an idle SSD
        // worker, so its speculative reads can overlap this layer and the next
        // attention block. The next layer always recomputes its exact route.
        if (ctx.moe_cross_layer_prefetch && !device_resident &&
            node.op_type == OpType::MOE && inputs.size() >= 4 && inputs[0] &&
            inputs[0]->shape[1] == 1) {
            const auto next_it = std::find_if(nodes.begin() + static_cast<ptrdiff_t>(i + 1),
                                              nodes.end(),
                                              [](const auto& candidate) {
                                                  return candidate.op_type == OpType::MOE;
                                              });
            if (next_it != nodes.end() && next_it->inputs.size() >= 4) {
                const auto& next_inputs = next_it->inputs;
                const Tensor& next_router = tensors[next_inputs[1]];
                const Tensor& next_gate = tensors[next_inputs[2]];
                const Tensor& next_down = tensors[next_inputs[3]];
                const Tensor* next_bias = next_inputs.size() > 8
                    ? &tensors[next_inputs[8]] : nullptr;
                MoeGateConfig config;
                config.hidden_size = graph_params::get_i32(next_it->params, 0, 0);
                config.num_experts = graph_params::get_i32(next_it->params, 1, 0);
                config.top_k = graph_params::get_i32(next_it->params, 2, 0);
                config.router_score_func = graph_params::get_i32(next_it->params, 5, 0);
                config.n_group = graph_params::get_i32(next_it->params, 8, 1);
                config.topk_group = graph_params::get_i32(next_it->params, 9, 1);
                schedule_moe_cross_layer_prefetch(
                    *inputs[0], next_router, next_bias,
                    static_cast<const MoeSsdTensorSource*>(next_gate.moe_ssd_source),
                    static_cast<const MoeSsdTensorSource*>(next_down.moe_ssd_source), config);
            }
        }

        // Dispatch is the useful unit in a trace: it captures one graph op
        // (matmul, attention, MoE, norm, ...) without recording allocator and
        // liveness bookkeeping as fake compute work.
        const uint64_t trace_start = mollm_trace::now_ns();
        ctx.backend->dispatch(node, inputs, &out, ctx.thread_pool);
        if (trace_start != 0) {
            const std::string args =
                "{\"graph\":\"" + std::string(ctx.trace_label ? ctx.trace_label : "graph") +
                "\",\"node\":" + std::to_string(node.id) +
                ",\"shape\":[" + std::to_string(out.shape[0]) + "," +
                std::to_string(out.shape[1]) + "," + std::to_string(out.shape[2]) + "," +
                std::to_string(out.shape[3]) + "]}";
            mollm_trace::record_duration("graph", op_type_name(node.op_type), trace_start,
                                         mollm_trace::now_ns(), args);
        }

        // Node dumping is an opt-in diagnostic.  Do not probe the process
        // environment for every graph node in normal inference.
        static const bool dump_nodes_enabled = getenv("MOLLM_DUMP_NODES") != nullptr;
        if (dump_nodes_enabled && out.data && out.prec == Precision::FP32) {
            const float* d = (const float*)out.data;
            fprintf(stderr, "NODE %u op=%d shape=%lld,%lld,%lld,%lld  %.5f %.5f %.5f\n",
                    node.id, (int)node.op_type,
                    (long long)out.shape[0], (long long)out.shape[1],
                    (long long)out.shape[2], (long long)out.shape[3],
                    d[0], out.nelements()>1?d[1]:0, out.nelements()>2?d[2]:0);
        }
        // Release completed tensors. Classify borrowed views before mutating
        // any producer in this release batch; a producer and its view can have
        // the same last-use node.
        const auto& rels = reuse_workspace ? empty_release_batch : ctx.release_queue[i];
        std::vector<uint8_t> rel_borrowed(rels.size(), 0);
        for (size_t r = 0; r < rels.size(); r++) {
            uint32_t rel_id = rels[r];
            if (rel_id >= nodes.size()) continue;
            auto op = nodes[rel_id].op_type;
            if (op == OpType::SLICE || op == OpType::PERMUTE) {
                rel_borrowed[r] = 1;
            } else if (op == OpType::RESHAPE && !nodes[rel_id].inputs.empty()) {
                auto& t = tensors[rel_id];
                const Tensor& src = tensors[nodes[rel_id].inputs[0]];
                rel_borrowed[r] = (device_resident ? src.is_contiguous()
                                                   : shares_storage(t, src)) ? 1 : 0;
            }
        }

        for (size_t r = 0; r < rels.size(); r++) {
            uint32_t rel_id = rels[r];
            auto& t = tensors[rel_id];
            if (rel_id < nodes.size()) {
                auto op = nodes[rel_id].op_type;
                if (op == OpType::INPUT || op == OpType::CONSTANT)
                    continue;
                if (rel_borrowed[r]) {
                    t.data = nullptr;
                    t.device_data = nullptr;
                    t.device_offset = 0;
                    t.mem_type = MemoryType::NONE;
                    t.owner_id = 0;
                    t.storage_id = 0;
                    continue;
                }
            }
            if (t.data && t.mem_type == MemoryType::POOLED && t.nbytes() > 0) {
                if (!device_resident && t.owner_id != 0 && t.owner_id != pool->id()) {
                    std::fprintf(stderr,
                                 "execute_graph: owner mismatch in release_queue for node %u (%p owner=%u pool=%u)\n",
                                 rel_id, t.data, t.owner_id, pool->id());
                    assert(false && "execute_graph release_queue owner mismatch");
                    return;
                }
                ctx.backend->free_output(t, pool);
                t.data     = nullptr;
                t.device_data = nullptr;
                t.device_offset = 0;
                t.mem_type = MemoryType::NONE;
                t.owner_id = 0;
                t.storage_id = 0;
            }
        }

        if (ctx.profile_enabled && node.id < ctx.profile_stats.size()) {
            auto end_time = std::chrono::steady_clock::now();
            auto elapsed_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            auto& stat = ctx.profile_stats[node.id];
            stat.calls += 1;
            stat.total_ns += elapsed_ns;
        }
    }

    if (ctx.reuse_same_shape_workspace && has_dynamic) {
        ctx.workspace_shape_valid = true;
        ctx.workspace_runtime_seq_len = ctx.runtime_seq_len;
        ctx.workspace_runtime_batch = ctx.runtime_batch;
        ctx.workspace_static_padded = ctx.static_padded;
        ctx.workspace_padded_seq_len = ctx.padded_seq_len;
    }
}
