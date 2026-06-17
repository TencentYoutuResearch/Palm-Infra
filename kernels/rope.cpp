#include "kernels/rope.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

// ---------------------------------------------------------------------------
// scalar RoPE
// ---------------------------------------------------------------------------

static void rope_scalar(const float* x, const float* cos, const float* sin,
                        float* out, int D, int N, int rope_dim, bool interleave,
                        int ldx, int ldo, int ldc, int lds) {
    int half = rope_dim / 2;

    for (int n = 0; n < N; n++) {
        const float* x_row  = x + n * ldx;
        float*       o_row  = out + n * ldo;
        const float* c_row  = cos + n * ldc;
        const float* s_row  = sin + n * lds;

        // copy non-rope part
        for (int d = rope_dim; d < D; d++) {
            o_row[d] = x_row[d];
        }

        // apply rotation
        if (interleave) {
            // pairs: (0,1), (2,3), ...
            for (int i = 0; i < half; i++) {
                float x0 = x_row[2 * i];
                float x1 = x_row[2 * i + 1];
                float c  = c_row[i];
                float s  = s_row[i];
                o_row[2 * i]     = x0 * c - x1 * s;
                o_row[2 * i + 1] = x0 * s + x1 * c;
            }
        } else {
            // pairs: (0, half), (1, half+1), ...
            for (int i = 0; i < half; i++) {
                float x0 = x_row[i];
                float x1 = x_row[i + half];
                float c  = c_row[i];
                float s  = s_row[i];
                o_row[i]          = x0 * c - x1 * s;
                o_row[i + half]   = x0 * s + x1 * c;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEON RoPE (interleave mode only for now — MLA uses this)
// ---------------------------------------------------------------------------
#if HAS_NEON

static void rope_neon_interleave(const float* x, const float* cos, const float* sin,
                                  float* out, int D, int N, int rope_dim,
                                  int ldx, int ldo, int ldc, int lds) {
    int half = rope_dim / 2;

    for (int n = 0; n < N; n++) {
        const float* x_row  = x + n * ldx;
        float*       o_row  = out + n * ldo;
        const float* c_row  = cos + n * ldc;
        const float* s_row  = sin + n * lds;

        // copy non-rope part
        for (int d = rope_dim; d < D; d++) {
            o_row[d] = x_row[d];
        }

        // apply rotation in blocks of 4 pairs (8 elements)
        int i = 0;
        for (; i + 3 < half; i += 4) {
            // load 4 cos/sin values
            float32x4_t c = vld1q_f32(c_row + i);
            float32x4_t s = vld1q_f32(s_row + i);

            // load 4 pairs (8 elements) — need to deinterleave
            // x: [x0,x1, x2,x3, x4,x5, x6,x7]  (8 elements, 4 pairs)
            float32x4x2_t xp = vld2q_f32(x_row + 2 * i);
            // xp.val[0] = [x0, x2, x4, x6]  (even)
            // xp.val[1] = [x1, x3, x5, x7]  (odd)

            // new_even = even * c - odd * s
            float32x4_t new_even = vsubq_f32(vmulq_f32(xp.val[0], c), vmulq_f32(xp.val[1], s));
            // new_odd  = even * s + odd * c
            float32x4_t new_odd  = vaddq_f32(vmulq_f32(xp.val[0], s), vmulq_f32(xp.val[1], c));

            // interleave back
            float32x4x2_t out_p;
            out_p.val[0] = new_even;
            out_p.val[1] = new_odd;
            vst2q_f32(o_row + 2 * i, out_p);
        }

        // scalar tail
        for (; i < half; i++) {
            float x0 = x_row[2 * i];
            float x1 = x_row[2 * i + 1];
            float c  = c_row[i];
            float s  = s_row[i];
            o_row[2 * i]     = x0 * c - x1 * s;
            o_row[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

static void rope_neon_halves(const float* x, const float* cos, const float* sin,
                              float* out, int D, int N, int rope_dim,
                              int ldx, int ldo, int ldc, int lds) {
    int half = rope_dim / 2;

    for (int n = 0; n < N; n++) {
        const float* x_row  = x + n * ldx;
        float*       o_row  = out + n * ldo;
        const float* c_row  = cos + n * ldc;
        const float* s_row  = sin + n * lds;

        // copy non-rope part
        for (int d = rope_dim; d < D; d++) {
            o_row[d] = x_row[d];
        }

        // apply rotation in blocks of 4
        int i = 0;
        for (; i + 3 < half; i += 4) {
            float32x4_t x0 = vld1q_f32(x_row + i);
            float32x4_t x1 = vld1q_f32(x_row + i + half);
            float32x4_t c  = vld1q_f32(c_row + i);
            float32x4_t s  = vld1q_f32(s_row + i);

            vst1q_f32(o_row + i,        vsubq_f32(vmulq_f32(x0, c), vmulq_f32(x1, s)));
            vst1q_f32(o_row + i + half, vaddq_f32(vmulq_f32(x0, s), vmulq_f32(x1, c)));
        }

        for (; i < half; i++) {
            float x0 = x_row[i];
            float x1 = x_row[i + half];
            float c  = c_row[i];
            float s  = s_row[i];
            o_row[i]        = x0 * c - x1 * s;
            o_row[i + half] = x0 * s + x1 * c;
        }
    }
}

#endif // HAS_NEON

// ---------------------------------------------------------------------------
// kernel_rope
// ---------------------------------------------------------------------------

void kernel_rope(const Tensor& x, const Tensor& cos, const Tensor& sin,
                 int rope_dim, bool interleave, Tensor& out) {
    int D = (int)x.shape[0];  // feature dimension
    int N = (int)x.shape[1];  // sequence dimension
    int C = (int)(x.shape[2] * x.shape[3]);  // channels/batch planes sharing cos/sin

    int ldx = (int)(x.stride[1] / sizeof(float));
    int ldo = (int)(out.stride[1] / sizeof(float));
    int ldc = (int)(cos.stride[1] / sizeof(float));
    int lds = (int)(sin.stride[1] / sizeof(float));

    const float* c_ptr = cos.ptr<float>();
    const float* s_ptr = sin.ptr<float>();

    for (int c = 0; c < C; c++) {
        int c2 = c % (int)x.shape[2];
        int c3 = c / (int)x.shape[2];
        const float* x_ptr = reinterpret_cast<const float*>(
            static_cast<const char*>(x.data) + c2 * x.stride[2] + c3 * x.stride[3]);
        float* o_ptr = reinterpret_cast<float*>(
            static_cast<char*>(out.data) + c2 * out.stride[2] + c3 * out.stride[3]);

        // copy x → out first (rope modifies in-place conceptually, but we
        // do separate output for safety)
        if (o_ptr != x_ptr) {
            for (int n = 0; n < N; n++) {
                std::memcpy(o_ptr + n * ldo, x_ptr + n * ldx, D * sizeof(float));
            }
        }

#if HAS_NEON
        if (interleave) {
            rope_neon_interleave(x_ptr, c_ptr, s_ptr, o_ptr, D, N, rope_dim, ldx, ldo, ldc, lds);
        } else {
            rope_neon_halves(x_ptr, c_ptr, s_ptr, o_ptr, D, N, rope_dim, ldx, ldo, ldc, lds);
        }
#else
        rope_scalar(x_ptr, c_ptr, s_ptr, o_ptr, D, N, rope_dim, interleave, ldx, ldo, ldc, lds);
#endif
    }
}
