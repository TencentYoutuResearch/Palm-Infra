#include "graph/execute.h"
#include "kernels/matmul.h"
#include "kernels/norm.h"
#include "kernels/rope.h"
#include "kernels/attention.h"
#include "kernels/gdn.h"
#include "kernels/tensor.h"
#include "kernels/activations.h"  // for Activation enum + sigmoid_f32_neon
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#if HAS_NEON
#include <arm_neon.h>

// sigmoid_f32_neon is now defined in kernels/activations.h (shared with
// matmul kernel for fused activation). Include that header above.
#endif

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
    ctx.release_queue.resize(N);
    reset_profile_stats(ctx);
    // Phase 1: simple approach — don't free anything.
    // Memory is released when the BufferPool is destroyed.
    // Proper liveness analysis with view-awareness will be added later.
}

// ---------------------------------------------------------------------------
// kernel dispatcher
// ---------------------------------------------------------------------------

static void dispatch_kernel(OpType op, const OpParams& params,
                            const std::vector<const Tensor*>& inputs,
                            Tensor* output,
                            ThreadPool* thread_pool) {
    switch (op) {
    case OpType::INPUT:
    case OpType::CONSTANT:
        // no-op — data is already in the tensor
        break;

    case OpType::RESHAPE:
        // Reshape preserves logical element order. For contiguous inputs this is
        // a zero-copy metadata change; for non-contiguous views we materialize a
        // contiguous output.
        if (!inputs.empty() && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            int64_t new_shape[4] = { output->shape[0], output->shape[1], output->shape[2], output->shape[3] };
            if (params.i32.size() >= 4) {
                new_shape[0] = params.i32[0];
                new_shape[1] = params.i32[1];
                new_shape[2] = params.i32[2];
                new_shape[3] = params.i32[3];
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

                size_t es = src.element_size();
                char* dst = static_cast<char*>(output->data);
                int64_t flat = 0;
                for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
                    for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
                        for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                            for (int64_t i0 = 0; i0 < src.shape[0]; i0++) {
                                const char* src_ptr = static_cast<const char*>(src.data)
                                    + i0 * src.stride[0]
                                    + i1 * src.stride[1]
                                    + i2 * src.stride[2]
                                    + i3 * src.stride[3];
                                std::memcpy(dst + flat * es, src_ptr, es);
                                flat++;
                            }
                        }
                    }
                }
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
            int n_real = graph_params::get_i32(params, 1, seq_len); // 0 or missing → all positions

            const float* x_data = x.ptr<float>();
            const float* w_data = w.ptr<float>();
            float* conv_state_data = reinterpret_cast<float*>(inputs[2]->data);
            float* out_data = out.ptr<float>();

            int prefix_len = kernel_size - 1;
            int total_len = prefix_len + seq_len;

            std::vector<float> stated(groups * total_len);

            // Copy conv_state [groups, prefix_len] into stated
            for (int g = 0; g < groups; g++) {
                const float* cs = conv_state_data + g * prefix_len;
                float* dst = stated.data() + g * total_len;
                for (int p = 0; p < prefix_len; p++) dst[p] = cs[p];
            }
            // Copy x into stated. x data layout: [seq, groups] → x_data[s*groups + g].
            for (int s = 0; s < seq_len; s++) {
                for (int g = 0; g < groups; g++) {
                    stated[g * total_len + prefix_len + s] = x_data[s * groups + g];
                }
            }

            // Conv1d + silu for each group, each position.
            // Only process real tokens [0, n_real); padding positions remain zero.
            int process_len = (n_real > 0 && n_real < seq_len) ? n_real : seq_len;
            for (int g = 0; g < groups; g++) {
                const float* w_ptr = w_data + g * kernel_size;
                const float* st = stated.data() + g * total_len;
                float* ot = out_data + g * seq_len;

                // Zero out all positions first
                for (int i = 0; i < seq_len; i++) ot[i] = 0.f;

                for (int i = 0; i < process_len; i++) {
                    float sum = 0.f;
                    int base = prefix_len + i;
                    for (int k = 0; k < kernel_size; k++) {
                        int src_i = base - (kernel_size - 1) + k;
                        sum += st[src_i] * w_ptr[k];
                    }
                    float sig = 1.f / (1.f + std::exp(-sum));
                    ot[i] = sum * sig;
                }
            }

            // Update conv_state: save the last prefix_len positions of the
            // LAST REAL TOKEN's window (not the padded end).
            // For process_len tokens, the last relevant position in stated is
            // prefix_len + process_len - 1. We save the window ending there.
            if (process_len > 0) {
                int last_real_pos = prefix_len + process_len - 1;
                for (int g = 0; g < groups; g++) {
                    const float* st = stated.data() + g * total_len;
                    float* cs = conv_state_data + g * prefix_len;
                    for (int p = 0; p < prefix_len; p++) {
                        cs[p] = st[last_real_pos - prefix_len + 1 + p];
                    }
                }
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
                    float32x4_t mask_large = vcgtq_f32(x, vdupq_n_f32(10.f));
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
            const Tensor& src = *inputs[0];
            size_t es = src.element_size();
            char* dst = static_cast<char*>(output->data);
            int64_t flat = 0;
            for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
                for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
                    for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                        for (int64_t i0 = 0; i0 < src.shape[0]; i0++) {
                            const char* src_ptr = static_cast<const char*>(src.data)
                                + i0 * src.stride[0]
                                + i1 * src.stride[1]
                                + i2 * src.stride[2]
                                + i3 * src.stride[3];
                            std::memcpy(dst + flat * es, src_ptr, es);
                            flat++;
                        }
                    }
                }
            }
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

        // initialise output shape from node definition (static!)
        out.shape[0] = node.out_shape[0];
        out.shape[1] = node.out_shape[1];
        out.shape[2] = node.out_shape[2];
        out.shape[3] = node.out_shape[3];
        out.prec     = node.out_prec;
        out.compute_strides();

        // allocate output if needed
        if (out.data == nullptr) {
            size_t nbytes = out.nbytes();
            if (nbytes > 0) {
                void* buf = pool->acquire(nbytes);
                if (!buf) {
                    fprintf(stderr, "execute: pool acquire failed for node %u (%zu bytes)\n",
                            node.id, nbytes);
                    return;
                }
                out.data     = buf;
                out.mem_type = MemoryType::POOLED;
            }
        }

        // collect inputs
        std::vector<const Tensor*> inputs;
        inputs.reserve(node.inputs.size());
        for (uint32_t inp_id : node.inputs) {
            inputs.push_back(&tensors[inp_id]);
        }

        // dispatch
        dispatch_kernel(node.op_type, node.params, inputs, &out, ctx.thread_pool);

        // release completed tensors
        for (uint32_t rel_id : ctx.release_queue[i]) {
            auto& t = tensors[rel_id];
            if (rel_id < nodes.size()) {
                auto op = nodes[rel_id].op_type;
                if (op == OpType::INPUT || op == OpType::CONSTANT ||
                    op == OpType::SLICE || op == OpType::RESHAPE ||
                    op == OpType::PERMUTE || op == OpType::TILE ||
                    op == OpType::CONCAT)
                    continue;
            }
            if (t.data && t.mem_type == MemoryType::POOLED && t.nbytes() > 0) {
                pool->release(t.data, t.nbytes());
                t.data     = nullptr;
                t.mem_type = MemoryType::NONE;
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
}
