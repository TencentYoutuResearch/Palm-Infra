#include "kernels/norm.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// scalar RMSNorm
// ---------------------------------------------------------------------------

static void rms_norm_scalar(const float* x, const float* weight,
                            float* out, int D, int N,
                            float eps, int ldx, int ldo) {
    for (int n = 0; n < N; n++) {
        const float* x_row = x + n * ldx;
        float*       o_row = out + n * ldo;

        float sum_sq = 0.f;
        for (int d = 0; d < D; d++) {
            float v = x_row[d];
            sum_sq += v * v;
        }
        float rms = 1.f / std::sqrt(sum_sq / D + eps);

        for (int d = 0; d < D; d++) {
            o_row[d] = x_row[d] * rms * weight[d];
        }
    }
}

// ---------------------------------------------------------------------------
// NEON RMSNorm
// ---------------------------------------------------------------------------
#if HAS_NEON

static void rms_norm_neon(const float* x, const float* weight,
                          float* out, int D, int N,
                          float eps, int ldx, int ldo) {
    for (int n = 0; n < N; n++) {
        const float* x_row = x + n * ldx;
        float*       o_row = out + n * ldo;

        // compute sum of squares
        float32x4_t sum_sq = vdupq_n_f32(0.f);
        int d = 0;
        for (; d + 7 < D; d += 8) {
            float32x4_t v0 = vld1q_f32(x_row + d);
            float32x4_t v1 = vld1q_f32(x_row + d + 4);
            sum_sq = vfmaq_f32(sum_sq, v0, v0);
            sum_sq = vfmaq_f32(sum_sq, v1, v1);
        }
        for (; d + 3 < D; d += 4) {
            float32x4_t v = vld1q_f32(x_row + d);
            sum_sq = vfmaq_f32(sum_sq, v, v);
        }
        float ss = vaddvq_f32(sum_sq);
        for (; d < D; d++) {
            float v = x_row[d];
            ss += v * v;
        }

        float rms = 1.f / std::sqrt(ss / D + eps);

        // apply rms * weight
        d = 0;
        for (; d + 7 < D; d += 8) {
            float32x4_t x0 = vld1q_f32(x_row + d);
            float32x4_t x1 = vld1q_f32(x_row + d + 4);
            float32x4_t w0 = vld1q_f32(weight + d);
            float32x4_t w1 = vld1q_f32(weight + d + 4);
            float32x4_t rms_vec = vdupq_n_f32(rms);
            vst1q_f32(o_row + d,     vmulq_f32(vmulq_f32(x0, rms_vec), w0));
            vst1q_f32(o_row + d + 4, vmulq_f32(vmulq_f32(x1, rms_vec), w1));
        }
        for (; d + 3 < D; d += 4) {
            float32x4_t xv = vld1q_f32(x_row + d);
            float32x4_t wv = vld1q_f32(weight + d);
            float32x4_t rv = vdupq_n_f32(rms);
            vst1q_f32(o_row + d, vmulq_f32(vmulq_f32(xv, rv), wv));
        }
        for (; d < D; d++) {
            o_row[d] = x_row[d] * rms * weight[d];
        }
    }
}

#endif // HAS_NEON

// ---------------------------------------------------------------------------
// kernel_rms_norm
// ---------------------------------------------------------------------------

void kernel_rms_norm(const Tensor& x, const Tensor& weight,
                     float eps, Tensor& out) {
    int D = (int)x.shape[0];  // feature dimension
    int N = (int)x.shape[1];  // number of rows

    // Safety check: weight is 1D vector, dim may be in shape[0] or shape[1]
    // after weight dim swap
    int64_t w_dim = weight.shape[0] > weight.shape[1] ? weight.shape[0] : weight.shape[1];
    if (w_dim < D) {
        fprintf(stderr, "RMSNorm: weight too small! weight=[%lld,%lld], x.dim0=%d\n",
                weight.shape[0], weight.shape[1], D);
        return;
    }

    int ldx = (int)(x.stride[1] / sizeof(float));
    int ldo = (int)(out.stride[1] / sizeof(float));

    const float* x_ptr = x.ptr<float>();
    const float* w_ptr = weight.ptr<float>();
    float*       o_ptr = out.ptr<float>();

#if HAS_NEON
    rms_norm_neon(x_ptr, w_ptr, o_ptr, D, N, eps, ldx, ldo);
#else
    rms_norm_scalar(x_ptr, w_ptr, o_ptr, D, N, eps, ldx, ldo);
#endif
}
