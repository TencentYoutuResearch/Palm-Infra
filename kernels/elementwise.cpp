#include "kernels/elementwise.h"
#include "kernels/activations.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#if HAS_NEON
#include <arm_neon.h>
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
    int d1, d2, d3;    // outer-loop dims (excluding d0)
    size_t s1, s2, s3; // byte strides for outer dims
    int n_inner;       // d0 (contiguous run length)
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
    if (t.stride[0] != t.element_size())
        return false;
    size_t expected = t.element_size();
    if (t.stride[1] != expected * t.shape[0])
        return false;
    if (t.stride[2] != t.stride[1] * t.shape[1])
        return false;
    if (t.stride[3] != t.stride[2] * t.shape[2])
        return false;
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
                const float* sp =
                    reinterpret_cast<const float*>(p2 + i1 * it.s1);
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
static inline void binary_stride_aware(float* dst, const Tensor& a,
                                       const Tensor& b, Kernel4 kernel_4) {
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
            vst1q_f32(dst + i, kernel_4(vld1q_f32(ap + i), vld1q_f32(bp + i)));
            vst1q_f32(dst + i + 4,
                      kernel_4(vld1q_f32(ap + i + 4), vld1q_f32(bp + i + 4)));
        }
        for (; i < N; i++)
            dst[i] = kernel_4(vdupq_n_f32(ap[i]), vdupq_n_f32(bp[i]))[0];
        return;
    }

    // At least one is strided. Iterate outer dims; inner dim-0 (shape[0]) is
    // always contiguous (stride[0] == es) for both inputs, but their outer
    // strides may differ.
    StrideIter ita = compute_stride_iter(a);
    StrideIter itb = compute_stride_iter(b);
    int n_inner = ita.n_inner; // a.shape[0] == b.shape[0]
    float* dst_row = dst;
    for (int i3 = 0; i3 < ita.d3; i3++) {
        const char* pa3 = a_base + i3 * ita.s3;
        const char* pb3 = b_base + i3 * itb.s3;
        for (int i2 = 0; i2 < ita.d2; i2++) {
            const char* pa2 = pa3 + i2 * ita.s2;
            const char* pb2 = pb3 + i2 * itb.s2;
            for (int i1 = 0; i1 < ita.d1; i1++) {
                const float* ap =
                    reinterpret_cast<const float*>(pa2 + i1 * ita.s1);
                const float* bp =
                    reinterpret_cast<const float*>(pb2 + i1 * itb.s1);
                int i = 0;
                for (; i + 7 < n_inner; i += 8) {
                    vst1q_f32(dst_row + i,
                              kernel_4(vld1q_f32(ap + i), vld1q_f32(bp + i)));
                    vst1q_f32(dst_row + i + 4, kernel_4(vld1q_f32(ap + i + 4),
                                                        vld1q_f32(bp + i + 4)));
                }
                for (; i < n_inner; i++)
                    dst_row[i] =
                        kernel_4(vdupq_n_f32(ap[i]), vdupq_n_f32(bp[i]))[0];
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
                    a_base + i3 * a.stride[3] + i2 * a.stride[2] +
                    i1 * a.stride[1]);
                const float* bp = reinterpret_cast<const float*>(
                    b_base + b3 * b.stride[3] + b2 * b.stride[2] +
                    b1 * b.stride[1]);
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
        if (b.shape[d] != 1 && b.shape[d] != a.shape[d])
            return false;
    }
    return true;
}

void kernel_elementwise(OpType op, const std::vector<const Tensor*>& inputs,
                        Tensor* output, ThreadPool* thread_pool) {
    switch (op) {
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
                binary_broadcast_from_b(
                    o, a, b, [](float av, float bv) { return av + bv; });
                break;
            }
            if (thread_pool && thread_pool->num_threads() > 1 && N >= 32768 &&
                a.is_contiguous() && b.is_contiguous() &&
                output->is_contiguous() &&
                (a.nelements() == b.nelements() || b.nelements() == 1)) {
                int chunk = (N + thread_pool->num_threads() - 1) /
                            thread_pool->num_threads();
                chunk = ((chunk + 7) / 8) * 8;
                thread_pool->parallel_for(
                    0, N, chunk, [&](int, int begin, int end) {
                        int i = begin;
#if HAS_NEON
                        if (a.nelements() == b.nelements()) {
                            for (; i + 7 < end; i += 8) {
                                vst1q_f32(o + i, vaddq_f32(vld1q_f32(ap + i),
                                                           vld1q_f32(bp + i)));
                                vst1q_f32(o + i + 4,
                                          vaddq_f32(vld1q_f32(ap + i + 4),
                                                    vld1q_f32(bp + i + 4)));
                            }
                        } else {
                            float32x4_t bv = vdupq_n_f32(bp[0]);
                            for (; i + 7 < end; i += 8) {
                                vst1q_f32(o + i,
                                          vaddq_f32(vld1q_f32(ap + i), bv));
                                vst1q_f32(o + i + 4,
                                          vaddq_f32(vld1q_f32(ap + i + 4), bv));
                            }
                        }
#endif
                        if (a.nelements() == b.nelements())
                            for (; i < end; ++i)
                                o[i] = ap[i] + bp[i];
                        else
                            for (; i < end; ++i)
                                o[i] = ap[i] + bp[0];
                    });
                break;
            }
#if HAS_NEON
            if (a.nelements() == b.nelements()) {
                int i = 0;
                for (; i + 7 < N; i += 8) {
                    vst1q_f32(o + i,
                              vaddq_f32(vld1q_f32(ap + i), vld1q_f32(bp + i)));
                    vst1q_f32(o + i + 4, vaddq_f32(vld1q_f32(ap + i + 4),
                                                   vld1q_f32(bp + i + 4)));
                }
                for (; i < N; i++)
                    o[i] = ap[i] + bp[i];
            } else if (b.nelements() == 1) {
                float bv = bp[0];
                float32x4_t bv4 = vdupq_n_f32(bv);
                int i = 0;
                for (; i + 7 < N; i += 8) {
                    vst1q_f32(o + i, vaddq_f32(vld1q_f32(ap + i), bv4));
                    vst1q_f32(o + i + 4, vaddq_f32(vld1q_f32(ap + i + 4), bv4));
                }
                for (; i < N; i++)
                    o[i] = ap[i] + bv;
            } else {
                for (int i = 0; i < N; i++)
                    o[i] = ap[i] + bp[i];
            }
#else
            if (a.nelements() == b.nelements()) {
                for (int i = 0; i < N; i++)
                    o[i] = ap[i] + bp[i];
            } else if (b.nelements() == 1) {
                float bv = bp[0];
                for (int i = 0; i < N; i++)
                    o[i] = ap[i] + bv;
            } else {
                for (int i = 0; i < N; i++)
                    o[i] = ap[i] + bp[i];
            }
#endif
        }
        break;

    case OpType::MUL:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            const Tensor& a = *inputs[0];
            const Tensor& b = *inputs[1];
            if (!a.data || !b.data) {
                fprintf(stderr,
                        "MUL: null data! a.data=%p b.data=%p "
                        "a_shape=[%lld,%lld,%lld,%lld] "
                        "b_shape=[%lld,%lld,%lld,%lld] a_prec=%d b_prec=%d\n",
                        a.data, b.data, a.shape[0], a.shape[1], a.shape[2],
                        a.shape[3], b.shape[0], b.shape[1], b.shape[2],
                        b.shape[3], (int)a.prec, (int)b.prec);
                break;
            }
            float* o = output->ptr<float>();
            if (a.nelements() != b.nelements() && broadcasts_to(b, a)) {
                binary_broadcast_from_b(
                    o, a, b, [](float av, float bv) { return av * bv; });
                break;
            }
            int N = (int)a.nelements();
            if (thread_pool && thread_pool->num_threads() > 1 && N >= 32768 &&
                a.nelements() == b.nelements() && a.is_contiguous() &&
                b.is_contiguous() && output->is_contiguous()) {
                const float* ap = a.ptr<float>();
                const float* bp = b.ptr<float>();
                int chunk = (N + thread_pool->num_threads() - 1) /
                            thread_pool->num_threads();
                chunk = ((chunk + 7) / 8) * 8;
                thread_pool->parallel_for(
                    0, N, chunk, [&](int, int begin, int end) {
                        int i = begin;
#if HAS_NEON
                        for (; i + 7 < end; i += 8) {
                            vst1q_f32(o + i, vmulq_f32(vld1q_f32(ap + i),
                                                       vld1q_f32(bp + i)));
                            vst1q_f32(o + i + 4,
                                      vmulq_f32(vld1q_f32(ap + i + 4),
                                                vld1q_f32(bp + i + 4)));
                        }
#endif
                        for (; i < end; ++i)
                            o[i] = ap[i] * bp[i];
                    });
                break;
            }
#if HAS_NEON
            binary_stride_aware(o, a, b, [](float32x4_t av, float32x4_t bv) {
                return vmulq_f32(av, bv);
            });
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
                        const float* ap =
                            reinterpret_cast<const float*>(pa2 + i1 * it.s1);
                        const float* bp =
                            reinterpret_cast<const float*>(pb2 + i1 * it.s1);
                        for (int i = 0; i < it.n_inner; i++)
                            dst_row[i] = ap[i] * bp[i];
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
            unary_stride_aware(o, *inputs[0], [](float32x4_t x) {
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
                        const float* sp =
                            reinterpret_cast<const float*>(p2 + i1 * it.s1);
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

    case OpType::GELU:
        // Match the fused MATMUL GELU convention used by both backends.
        if (inputs.size() >= 1 && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            const char* base = static_cast<const char*>(src.data);
            StrideIter it = compute_stride_iter(src);
            float* dst = output->ptr<float>();
            for (int i3 = 0; i3 < it.d3; ++i3)
                for (int i2 = 0; i2 < it.d2; ++i2)
                    for (int i1 = 0; i1 < it.d1; ++i1) {
                        const float* row = reinterpret_cast<const float*>(
                            base + i3 * it.s3 + i2 * it.s2 + i1 * it.s1);
                        for (int i = 0; i < it.n_inner; ++i) {
                            const float v = row[i];
                            const float inner =
                                0.7978845608f *
                                (v + 0.044715f * v * v * v);
                            *dst++ =
                                0.5f * v * (1.0f + std::tanh(inner));
                        }
                    }
        }
        break;

    case OpType::TANH:
        if (inputs.size() >= 1 && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            const char* base = (const char*)src.data;
            StrideIter it = compute_stride_iter(src);
            float* dst = output->ptr<float>();
            for (int i3 = 0; i3 < it.d3; i3++)
                for (int i2 = 0; i2 < it.d2; i2++)
                    for (int i1 = 0; i1 < it.d1; i1++) {
                        const float* row =
                            (const float*)(base + i3 * it.s3 + i2 * it.s2 +
                                           i1 * it.s1);
                        for (int i = 0; i < it.n_inner; i++)
                            dst[i] = std::tanh(row[i]);
                        dst += it.n_inner;
                    }
        }
        break;

    case OpType::SWIGLU:
        // Fused SwiGLU over a merged [2I, ...] tensor: out[i] = silu(gate[i]) *
        // up[i], gate = row[0..I), up = row[I..2I). Reads both halves from the
        // single merged row (stride-aware); output is dense [I, ...]. NOT a
        // slice-view consumer.
        if (inputs.size() >= 1 && inputs[0] && output) {
            const Tensor& m = *inputs[0];
            const char* m_base = static_cast<const char*>(m.data);
            StrideIter it = compute_stride_iter(m); // it.n_inner = 2I
            const int I = it.n_inner / 2;
            float* dst_row = output->ptr<float>();
            for (int i3 = 0; i3 < it.d3; i3++) {
                const char* p3 = m_base + i3 * it.s3;
                for (int i2 = 0; i2 < it.d2; i2++) {
                    const char* p2 = p3 + i2 * it.s2;
                    for (int i1 = 0; i1 < it.d1; i1++) {
                        const float* row =
                            reinterpret_cast<const float*>(p2 + i1 * it.s1);
                        const float* gate = row;
                        const float* up = row + I;
                        for (int i = 0; i < I; i++) {
                            float g = gate[i];
                            float sg = g / (1.f + std::exp(-g)); // silu(g)
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
            unary_stride_aware(o, *inputs[0], [](float32x4_t x) {
                return sigmoid_f32_neon(x);
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
                        const float* sp =
                            reinterpret_cast<const float*>(p2 + i1 * it.s1);
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
                thread_pool->parallel_for(
                    0, (int)total, chunk, [&](int, int begin, int end) {
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
                        const float* row =
                            reinterpret_cast<const float*>(p2 + i1 * it.s1);
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
            unary_stride_aware(o, *inputs[0], [](float32x4_t x) {
                // Reuse fast_exp from sigmoid: exp(x) = sigmoid(x) / (1 -
                // sigmoid(x)) But simpler: use the polynomial exp approximation
                // directly. Clamp to avoid overflow.
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
                        const float* sp =
                            reinterpret_cast<const float*>(p2 + i1 * it.s1);
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
                                              for (int i = begin; i < end; ++i)
                                                  dst[i] = std::exp(input[i]);
                                          });
                break;
            }
            const char* base = static_cast<const char*>(src.data);
            StrideIter it = compute_stride_iter(src);
            float* dst = output->ptr<float>();
            for (int i3 = 0; i3 < it.d3; ++i3)
                for (int i2 = 0; i2 < it.d2; ++i2)
                    for (int i1 = 0; i1 < it.d1; ++i1) {
                        const float* row = reinterpret_cast<const float*>(
                            base + i3 * it.s3 + i2 * it.s2 + i1 * it.s1);
                        for (int i = 0; i < it.n_inner; ++i)
                            *dst++ = std::exp(row[i]);
                    }
        }
        break;

    case OpType::SOFTPLUS:
        if (inputs.size() >= 1 && inputs[0] && output) {
            float* o = output->ptr<float>();
#if HAS_NEON
            unary_stride_aware(o, *inputs[0], [](float32x4_t x) {
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
                // But for correctness, use: vlogq_f32 if available, else scalar
                // fallback. NEON doesn't have vlogq_f32, so we do scalar for
                // the log part. Fast path: if x > 10, return x (softplus ≈ x).
                uint32x4_t mask_large = vcgtq_f32(x, vdupq_n_f32(10.f));
                float32x4_t result = x; // fallback for large x
                // For smaller values, compute log(1+exp) scalar.
                float tmp[4];
                vst1q_f32(tmp, ex);
                for (int i = 0; i < 4; i++)
                    tmp[i] = std::log1pf(tmp[i]);
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
                        const float* sp =
                            reinterpret_cast<const float*>(p2 + i1 * it.s1);
                        for (int i = 0; i < it.n_inner; i++) {
                            float v = sp[i];
                            if (v > 20.f)
                                dst_row[i] = v;
                            else if (v < -20.f)
                                dst_row[i] = std::exp(v);
                            else
                                dst_row[i] = std::log1pf(std::exp(v));
                        }
                        dst_row += it.n_inner;
                    }
                }
            }
#endif
        }
        break;

    default:
        std::fprintf(stderr, "elementwise: unsupported op_type %u\n",
                     static_cast<unsigned>(op));
        break;
    }
}
