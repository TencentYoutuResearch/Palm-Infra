#include "kernels/attention.h"
#include "kernels/matmul.h"
#include "kernels/threading.h"
#include "engine/engine.h"  // for CacheMetadata, cache_meta, cache_data

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cfloat>

// ---------------------------------------------------------------------------
// FlashAttention-2 kernels for ARM NEON
// Ported from ncnn-upstream/src/layer/arm/sdpa_arm_flash.h
// 2D tiled (BrxBc) with online softmax - avoids O(MxN) attention matrix
//
// Two precision paths:
//   - FP32: Q/K/V/O all FP32, used as fallback
//   - FP16FML: Q/K/V converted to FP16 inside kernel, FP32 accumulate for
//     QK dot + softmax + PV, FP16FML (vfmlalq_lane_low/high_f16) for FMA.
//     Halves K/V memory bandwidth → faster on bandwidth-bound shapes.
// ---------------------------------------------------------------------------

#if HAS_NEON

// Fast vectorized exp approximation for NEON
// exp(x) = 2^(x * log2e) = 2^n * 2^f, polynomial approx for 2^f
static inline float32x4_t fast_exp_f32x4(float32x4_t x) {
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
}

static inline float fast_exp_f32(float x) {
    if (x < -88.f) return 0.f;
    if (x > 88.f) return INFINITY;
    float t = x * 1.4426950408889634f;
    float n = floorf(t);
    float f = t - n;
    union { float f; int32_t i; } pow2n;
    pow2n.i = ((int32_t)n + 127) << 23;
    float pow2f = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0555049f + f * 0.0096813f)));
    return pow2n.f * pow2f;
}

// Vectorized dot product for fp32, general d
static inline float dot_fp32_neon(const float* a, const float* b, int d) {
    float32x4_t s0 = vdupq_n_f32(0.f);
    float32x4_t s1 = vdupq_n_f32(0.f);
    int k = 0;
    for (; k + 7 < d; k += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + k), vld1q_f32(b + k));
        s1 = vfmaq_f32(s1, vld1q_f32(a + k + 4), vld1q_f32(b + k + 4));
    }
    for (; k + 3 < d; k += 4)
        s0 = vfmaq_f32(s0, vld1q_f32(a + k), vld1q_f32(b + k));
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; k < d; k++) s += a[k] * b[k];
    return s;
}

// Decode kernel: M=1, Bc=32, online softmax
static void flash_attn_fp32_decode(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d_k, int d_v, float scale, const float* mask)
{
    const int Bc = 32;
    float m = -FLT_MAX;
    float l = 0.f;
    memset(O, 0, d_v * sizeof(float));

    for (int jb = 0; jb < N; jb += Bc) {
        const int bc = (jb + Bc <= N) ? Bc : (N - jb);
        float scores[32];
        for (int j = 0; j < bc; j++) {
            scores[j] = dot_fp32_neon(Q, K + (jb + j) * d_k, d_k) * scale;
            if (mask) scores[j] += mask[jb + j];
        }
        float m_block = scores[0];
        for (int j = 1; j < bc; j++) m_block = fmaxf(m_block, scores[j]);
        float m_new = fmaxf(m, m_block);
        float alpha = (m_new == m) ? 1.f : fast_exp_f32(m - m_new);
        if (l != 0.f && alpha != 1.f) {
            float32x4_t valpha = vdupq_n_f32(alpha);
            int d = 0;
            for (; d + 7 < d_v; d += 8) {
                vst1q_f32(O + d, vmulq_f32(vld1q_f32(O + d), valpha));
                vst1q_f32(O + d + 4, vmulq_f32(vld1q_f32(O + d + 4), valpha));
            }
            for (; d + 3 < d_v; d += 4)
                vst1q_f32(O + d, vmulq_f32(vld1q_f32(O + d), valpha));
            for (; d < d_v; d++) O[d] *= alpha;
        }
        l *= alpha;
        for (int j = 0; j < bc; j++) {
            float p = fast_exp_f32(scores[j] - m_new);
            l += p;
            const float* vj = V + (jb + j) * d_v;
            float32x4_t vp = vdupq_n_f32(p);
            int d = 0;
            for (; d + 7 < d_v; d += 8) {
                vst1q_f32(O + d, vfmaq_f32(vld1q_f32(O + d), vp, vld1q_f32(vj + d)));
                vst1q_f32(O + d + 4, vfmaq_f32(vld1q_f32(O + d + 4), vp, vld1q_f32(vj + d + 4)));
            }
            for (; d + 3 < d_v; d += 4)
                vst1q_f32(O + d, vfmaq_f32(vld1q_f32(O + d), vp, vld1q_f32(vj + d)));
            for (; d < d_v; d++) O[d] += p * vj[d];
        }
        m = m_new;
    }
    if (l > 0.f) {
        float inv_l = 1.f / l;
        float32x4_t vinv = vdupq_n_f32(inv_l);
        int d = 0;
        for (; d + 3 < d_v; d += 4)
            vst1q_f32(O + d, vmulq_f32(vld1q_f32(O + d), vinv));
        for (; d < d_v; d++) O[d] *= inv_l;
    }
}

// Prefill kernel: Br=4, Bc=32, 2D tiled with online softmax
static void flash_attn_fp32_prefill(
    const float* Q, const float* K, const float* V, float* O,
    int M, int N, int d_k, int d_v, float scale, const float* mask)
{
    const int Br = 4;
    const int Bc = 32;
    float* row_m = (float*)malloc(M * sizeof(float));
    float* row_l = (float*)malloc(M * sizeof(float));
    memset(O, 0, M * d_v * sizeof(float));
    for (int i = 0; i < M; i++) { row_m[i] = -FLT_MAX; row_l[i] = 0.f; }

    for (int jb = 0; jb < N; jb += Bc) {
        const int bc = (jb + Bc <= N) ? Bc : (N - jb);
        for (int ib = 0; ib < M; ib += Br) {
            const int br = (ib + Br <= M) ? Br : (M - ib);
            float S[4][32];
            for (int i = 0; i < br; i++) {
                const float* qi = Q + (ib + i) * d_k;
                for (int j = 0; j < bc; j++) {
                    S[i][j] = dot_fp32_neon(qi, K + (jb + j) * d_k, d_k) * scale;
                    if (mask) S[i][j] += mask[(ib + i) * N + jb + j];
                }
            }
            for (int i = 0; i < br; i++) {
                const int row = ib + i;
                float m_old = row_m[row];
                float l_old = row_l[row];
                float* oi = O + row * d_v;
                float m_block = S[i][0];
                for (int j = 1; j < bc; j++) m_block = fmaxf(m_block, S[i][j]);
                float m_new = fmaxf(m_old, m_block);
                float alpha = fast_exp_f32(m_old - m_new);
                {
                    float32x4_t valpha = vdupq_n_f32(alpha);
                    int d = 0;
                    for (; d + 7 < d_v; d += 8) {
                        vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), valpha));
                        vst1q_f32(oi + d + 4, vmulq_f32(vld1q_f32(oi + d + 4), valpha));
                    }
                    for (; d + 3 < d_v; d += 4)
                        vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), valpha));
                    for (; d < d_v; d++) oi[d] *= alpha;
                }
                float l_block = 0.f;
                for (int j = 0; j < bc; j++) {
                    float p = fast_exp_f32(S[i][j] - m_new);
                    l_block += p;
                    const float* vj = V + (jb + j) * d_v;
                    float32x4_t vp = vdupq_n_f32(p);
                    int d = 0;
                    for (; d + 7 < d_v; d += 8) {
                        vst1q_f32(oi + d, vfmaq_f32(vld1q_f32(oi + d), vp, vld1q_f32(vj + d)));
                        vst1q_f32(oi + d + 4, vfmaq_f32(vld1q_f32(oi + d + 4), vp, vld1q_f32(vj + d + 4)));
                    }
                    for (; d + 3 < d_v; d += 4)
                        vst1q_f32(oi + d, vfmaq_f32(vld1q_f32(oi + d), vp, vld1q_f32(vj + d)));
                    for (; d < d_v; d++) oi[d] += p * vj[d];
                }
                row_l[row] = alpha * l_old + l_block;
                row_m[row] = m_new;
            }
        }
    }
    for (int i = 0; i < M; i++) {
        float* oi = O + i * d_v;
        float inv_l = (row_l[i] > 0.f) ? 1.f / row_l[i] : 0.f;
        float32x4_t vinv = vdupq_n_f32(inv_l);
        int d = 0;
        for (; d + 3 < d_v; d += 4)
            vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), vinv));
        for (; d < d_v; d++) oi[d] *= inv_l;
    }
    free(row_m);
    free(row_l);
}

// ===========================================================================
// FP16FML path: FP16 inputs (converted in-kernel from FP32), FP32 accumulate.
// Uses vfmlalq_low/high_f16 (FP16×FP16→FP32 widening FMA) and
// vfmlalq_lane_low/high_f16 (lane-broadcast variant) for PV accumulation.
// ===========================================================================

// General FP16 dot product, FP32 accumulate via FP16FML.
// Q, K are __fp16. Returns scalar FP32.
static inline float dot_fp16_neon(const __fp16* Q, const __fp16* K, int d) {
    float32x4_t s0 = vdupq_n_f32(0.f);
    float32x4_t s1 = vdupq_n_f32(0.f);
    int k = 0;
    for (; k + 15 < d; k += 16) {
        float16x8_t q0 = vld1q_f16(Q + k);
        float16x8_t q1 = vld1q_f16(Q + k + 8);
        float16x8_t k0 = vld1q_f16(K + k);
        float16x8_t k1 = vld1q_f16(K + k + 8);
        s0 = vfmlalq_low_f16(s0, q0, k0);
        s1 = vfmlalq_high_f16(s1, q0, k0);
        s0 = vfmlalq_low_f16(s0, q1, k1);
        s1 = vfmlalq_high_f16(s1, q1, k1);
    }
    for (; k + 7 < d; k += 8) {
        float16x8_t q0 = vld1q_f16(Q + k);
        float16x8_t k0 = vld1q_f16(K + k);
        s0 = vfmlalq_low_f16(s0, q0, k0);
        s1 = vfmlalq_high_f16(s1, q0, k0);
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; k < d; k++) s += (float)Q[k] * (float)K[k];
    return s;
}

// 4-way batched FP16 dot product: 4 K rows share one Q load.
// FP32 accumulate via FP16FML. Output 4 FP32 scores.
static inline void dot_fp16_4x(const __fp16* Q,
    const __fp16* K0, const __fp16* K1, const __fp16* K2, const __fp16* K3,
    int d_k, float* out)
{
    float32x4_t a0 = vdupq_n_f32(0.f);
    float32x4_t a1 = vdupq_n_f32(0.f);
    float32x4_t a2 = vdupq_n_f32(0.f);
    float32x4_t a3 = vdupq_n_f32(0.f);

    int k = 0;
    for (; k + 15 < d_k; k += 16) {
        float16x8_t qh0 = vld1q_f16(Q + k);
        float16x8_t qh1 = vld1q_f16(Q + k + 8);

        float16x8_t kh0 = vld1q_f16(K0 + k);
        float16x8_t kh1 = vld1q_f16(K0 + k + 8);
        a0 = vfmlalq_low_f16(a0, qh0, kh0);
        a0 = vfmlalq_high_f16(a0, qh0, kh0);
        a0 = vfmlalq_low_f16(a0, qh1, kh1);
        a0 = vfmlalq_high_f16(a0, qh1, kh1);

        kh0 = vld1q_f16(K1 + k);
        kh1 = vld1q_f16(K1 + k + 8);
        a1 = vfmlalq_low_f16(a1, qh0, kh0);
        a1 = vfmlalq_high_f16(a1, qh0, kh0);
        a1 = vfmlalq_low_f16(a1, qh1, kh1);
        a1 = vfmlalq_high_f16(a1, qh1, kh1);

        kh0 = vld1q_f16(K2 + k);
        kh1 = vld1q_f16(K2 + k + 8);
        a2 = vfmlalq_low_f16(a2, qh0, kh0);
        a2 = vfmlalq_high_f16(a2, qh0, kh0);
        a2 = vfmlalq_low_f16(a2, qh1, kh1);
        a2 = vfmlalq_high_f16(a2, qh1, kh1);

        kh0 = vld1q_f16(K3 + k);
        kh1 = vld1q_f16(K3 + k + 8);
        a3 = vfmlalq_low_f16(a3, qh0, kh0);
        a3 = vfmlalq_high_f16(a3, qh0, kh0);
        a3 = vfmlalq_low_f16(a3, qh1, kh1);
        a3 = vfmlalq_high_f16(a3, qh1, kh1);
    }
    for (; k + 7 < d_k; k += 8) {
        float16x8_t qh0 = vld1q_f16(Q + k);
        float16x8_t kh0 = vld1q_f16(K0 + k);
        a0 = vfmlalq_low_f16(a0, qh0, kh0);
        a0 = vfmlalq_high_f16(a0, qh0, kh0);
        kh0 = vld1q_f16(K1 + k);
        a1 = vfmlalq_low_f16(a1, qh0, kh0);
        a1 = vfmlalq_high_f16(a1, qh0, kh0);
        kh0 = vld1q_f16(K2 + k);
        a2 = vfmlalq_low_f16(a2, qh0, kh0);
        a2 = vfmlalq_high_f16(a2, qh0, kh0);
        kh0 = vld1q_f16(K3 + k);
        a3 = vfmlalq_low_f16(a3, qh0, kh0);
        a3 = vfmlalq_high_f16(a3, qh0, kh0);
    }

    out[0] = vaddvq_f32(a0);
    out[1] = vaddvq_f32(a1);
    out[2] = vaddvq_f32(a2);
    out[3] = vaddvq_f32(a3);

    // Scalar tail
    for (; k < d_k; k++) {
        float q = (float)Q[k];
        out[0] += q * (float)K0[k];
        out[1] += q * (float)K1[k];
        out[2] += q * (float)K2[k];
        out[3] += q * (float)K3[k];
    }
}

// FP16 decode kernel: M=1, Bc=64, online softmax, FP16FML PV accumulation.
// Q/K/V are FP16 (caller converts from FP32). O is FP32 output.
static void flash_attn_fp16_decode(
    const __fp16* Q, const __fp16* K, const __fp16* V, float* O,
    int N, int d_k, int d_v, float scale, const float* mask)
{
    const int Bc = 64;

    float m = -FLT_MAX;
    float l = 0.f;

    // O accumulator in FP32 (stays in registers for inner PV loop)
    float O_buf[256];
    float* O_fp32 = (d_v <= 256) ? O_buf : (float*)malloc(d_v * sizeof(float));
    memset(O_fp32, 0, d_v * sizeof(float));

    for (int jb = 0; jb < N; jb += Bc) {
        const int bc = (jb + Bc <= N) ? Bc : (N - jb);

        // 1. QK scores: 4-way batched FP16FML dot product
        float scores[64];
        {
            int j = 0;
            for (; j + 3 < bc; j += 4) {
                dot_fp16_4x(Q,
                            K + (jb + j) * d_k, K + (jb + j + 1) * d_k,
                            K + (jb + j + 2) * d_k, K + (jb + j + 3) * d_k,
                            d_k, scores + j);
                scores[j]   *= scale; scores[j+1] *= scale;
                scores[j+2] *= scale; scores[j+3] *= scale;
                if (mask) {
                    scores[j]   += mask[jb+j];
                    scores[j+1] += mask[jb+j+1];
                    scores[j+2] += mask[jb+j+2];
                    scores[j+3] += mask[jb+j+3];
                }
            }
            for (; j < bc; j++) {
                scores[j] = dot_fp16_neon(Q, K + (jb + j) * d_k, d_k) * scale;
                if (mask) scores[j] += mask[jb + j];
            }
        }

        // 2. Block max
        float m_block = scores[0];
        for (int j = 1; j < bc; j++)
            m_block = fmaxf(m_block, scores[j]);

        // 3. Online softmax rescale
        float m_new = fmaxf(m, m_block);
        float alpha = (m_new == m) ? 1.f : fast_exp_f32(m - m_new);
        if (l != 0.f && alpha != 1.f) {
            float32x4_t valpha = vdupq_n_f32(alpha);
            int d = 0;
            for (; d + 7 < d_v; d += 8) {
                vst1q_f32(O_fp32 + d, vmulq_f32(vld1q_f32(O_fp32 + d), valpha));
                vst1q_f32(O_fp32 + d + 4, vmulq_f32(vld1q_f32(O_fp32 + d + 4), valpha));
            }
            for (; d + 3 < d_v; d += 4)
                vst1q_f32(O_fp32 + d, vmulq_f32(vld1q_f32(O_fp32 + d), valpha));
            for (; d < d_v; d++) O_fp32[d] *= alpha;
        }
        l *= alpha;

        // 4. Convert scores → P = exp(s - m_new), accumulate l
        {
            float32x4_t vm_new = vdupq_n_f32(m_new);
            float32x4_t vl = vdupq_n_f32(0.f);
            int j = 0;
            for (; j + 3 < bc; j += 4) {
                float32x4_t s4 = vsubq_f32(vld1q_f32(scores + j), vm_new);
                float32x4_t p4 = fast_exp_f32x4(s4);
                vst1q_f32(scores + j, p4);
                vl = vaddq_f32(vl, p4);
            }
            l += vaddvq_f32(vl);
            for (; j < bc; j++) {
                scores[j] = fast_exp_f32(scores[j] - m_new);
                l += scores[j];
            }
        }

        // 5. PV accumulation with register-tiled FP16FML
        // Tile d_v into 64-wide blocks, keep O in 16 FP32 registers across all j
        {
            const int TILE_DV = 64;
            int d_start = 0;
            for (; d_start + TILE_DV <= d_v; d_start += TILE_DV) {
                float32x4_t o0  = vld1q_f32(O_fp32 + d_start + 0);
                float32x4_t o1  = vld1q_f32(O_fp32 + d_start + 4);
                float32x4_t o2  = vld1q_f32(O_fp32 + d_start + 8);
                float32x4_t o3  = vld1q_f32(O_fp32 + d_start + 12);
                float32x4_t o4  = vld1q_f32(O_fp32 + d_start + 16);
                float32x4_t o5  = vld1q_f32(O_fp32 + d_start + 20);
                float32x4_t o6  = vld1q_f32(O_fp32 + d_start + 24);
                float32x4_t o7  = vld1q_f32(O_fp32 + d_start + 28);
                float32x4_t o8  = vld1q_f32(O_fp32 + d_start + 32);
                float32x4_t o9  = vld1q_f32(O_fp32 + d_start + 36);
                float32x4_t o10 = vld1q_f32(O_fp32 + d_start + 40);
                float32x4_t o11 = vld1q_f32(O_fp32 + d_start + 44);
                float32x4_t o12 = vld1q_f32(O_fp32 + d_start + 48);
                float32x4_t o13 = vld1q_f32(O_fp32 + d_start + 52);
                float32x4_t o14 = vld1q_f32(O_fp32 + d_start + 56);
                float32x4_t o15 = vld1q_f32(O_fp32 + d_start + 60);

                for (int j = 0; j < bc; j++) {
                    float16x4_t vp = vdup_n_f16((__fp16)scores[j]);
                    const __fp16* vj = V + (jb + j) * d_v + d_start;

                    float16x8_t v01 = vld1q_f16(vj + 0);
                    float16x8_t v23 = vld1q_f16(vj + 8);
                    float16x8_t v45 = vld1q_f16(vj + 16);
                    float16x8_t v67 = vld1q_f16(vj + 24);
                    float16x8_t v89 = vld1q_f16(vj + 32);
                    float16x8_t vab = vld1q_f16(vj + 40);
                    float16x8_t vcd = vld1q_f16(vj + 48);
                    float16x8_t vef = vld1q_f16(vj + 56);

                    o0  = vfmlalq_lane_low_f16 (o0,  v01, vp, 0);
                    o1  = vfmlalq_lane_high_f16(o1,  v01, vp, 0);
                    o2  = vfmlalq_lane_low_f16 (o2,  v23, vp, 0);
                    o3  = vfmlalq_lane_high_f16(o3,  v23, vp, 0);
                    o4  = vfmlalq_lane_low_f16 (o4,  v45, vp, 0);
                    o5  = vfmlalq_lane_high_f16(o5,  v45, vp, 0);
                    o6  = vfmlalq_lane_low_f16 (o6,  v67, vp, 0);
                    o7  = vfmlalq_lane_high_f16(o7,  v67, vp, 0);
                    o8  = vfmlalq_lane_low_f16 (o8,  v89, vp, 0);
                    o9  = vfmlalq_lane_high_f16(o9,  v89, vp, 0);
                    o10 = vfmlalq_lane_low_f16 (o10, vab, vp, 0);
                    o11 = vfmlalq_lane_high_f16(o11, vab, vp, 0);
                    o12 = vfmlalq_lane_low_f16 (o12, vcd, vp, 0);
                    o13 = vfmlalq_lane_high_f16(o13, vcd, vp, 0);
                    o14 = vfmlalq_lane_low_f16 (o14, vef, vp, 0);
                    o15 = vfmlalq_lane_high_f16(o15, vef, vp, 0);
                }

                vst1q_f32(O_fp32 + d_start + 0,  o0);
                vst1q_f32(O_fp32 + d_start + 4,  o1);
                vst1q_f32(O_fp32 + d_start + 8,  o2);
                vst1q_f32(O_fp32 + d_start + 12, o3);
                vst1q_f32(O_fp32 + d_start + 16, o4);
                vst1q_f32(O_fp32 + d_start + 20, o5);
                vst1q_f32(O_fp32 + d_start + 24, o6);
                vst1q_f32(O_fp32 + d_start + 28, o7);
                vst1q_f32(O_fp32 + d_start + 32, o8);
                vst1q_f32(O_fp32 + d_start + 36, o9);
                vst1q_f32(O_fp32 + d_start + 40, o10);
                vst1q_f32(O_fp32 + d_start + 44, o11);
                vst1q_f32(O_fp32 + d_start + 48, o12);
                vst1q_f32(O_fp32 + d_start + 52, o13);
                vst1q_f32(O_fp32 + d_start + 56, o14);
                vst1q_f32(O_fp32 + d_start + 60, o15);
            }

            // 8-wide tail
            for (; d_start + 7 < d_v; d_start += 8) {
                float32x4_t o0 = vld1q_f32(O_fp32 + d_start);
                float32x4_t o1 = vld1q_f32(O_fp32 + d_start + 4);
                for (int j = 0; j < bc; j++) {
                    float16x4_t vp = vdup_n_f16((__fp16)scores[j]);
                    const __fp16* vj = V + (jb + j) * d_v + d_start;
                    float16x8_t v = vld1q_f16(vj);
                    o0 = vfmlalq_lane_low_f16(o0, v, vp, 0);
                    o1 = vfmlalq_lane_high_f16(o1, v, vp, 0);
                }
                vst1q_f32(O_fp32 + d_start, o0);
                vst1q_f32(O_fp32 + d_start + 4, o1);
            }

            // 4-wide tail
            for (; d_start + 3 < d_v; d_start += 4) {
                float32x4_t o0 = vld1q_f32(O_fp32 + d_start);
                for (int j = 0; j < bc; j++) {
                    float16x4_t vp = vdup_n_f16((__fp16)scores[j]);
                    const __fp16* vj = V + (jb + j) * d_v + d_start;
                    float16x4_t v = vld1_f16(vj);
                    o0 = vfmlalq_lane_low_f16(o0, vcombine_f16(v, v), vp, 0);
                }
                vst1q_f32(O_fp32 + d_start, o0);
            }

            // Scalar tail
            for (; d_start < d_v; d_start++) {
                float o0 = O_fp32[d_start];
                for (int j = 0; j < bc; j++)
                    o0 += scores[j] * (float)V[(jb + j) * d_v + d_start];
                O_fp32[d_start] = o0;
            }
        }

        m = m_new;
    }

    // Normalize
    if (l > 0.f) {
        float inv_l = 1.f / l;
        float32x4_t vinv = vdupq_n_f32(inv_l);
        int d = 0;
        for (; d + 7 < d_v; d += 8) {
            vst1q_f32(O + d, vmulq_f32(vld1q_f32(O_fp32 + d), vinv));
            vst1q_f32(O + d + 4, vmulq_f32(vld1q_f32(O_fp32 + d + 4), vinv));
        }
        for (; d + 3 < d_v; d += 4)
            vst1q_f32(O + d, vmulq_f32(vld1q_f32(O_fp32 + d), vinv));
        for (; d < d_v; d++) O[d] = O_fp32[d] * inv_l;
    } else {
        memcpy(O, O_fp32, d_v * sizeof(float));
    }

    if (d_v > 256) free(O_fp32);
}

// FP16 prefill kernel: Br=4, Bc=32, online softmax, FP16FML.
// Q/K/V are FP16 (caller converts from FP32). O is FP32 output.
// Simpler than ncnn's 8x12 micro-kernel — keeps the FP32 prefill structure
// but uses FP16FML for dot product and PV accumulation.
static void flash_attn_fp16_prefill(
    const __fp16* Q, const __fp16* K, const __fp16* V, float* O,
    int M, int N, int d_k, int d_v, float scale, const float* mask)
{
    const int Br = 4;
    const int Bc = 32;
    float* row_m = (float*)malloc(M * sizeof(float));
    float* row_l = (float*)malloc(M * sizeof(float));
    memset(O, 0, M * d_v * sizeof(float));
    for (int i = 0; i < M; i++) { row_m[i] = -FLT_MAX; row_l[i] = 0.f; }

    for (int jb = 0; jb < N; jb += Bc) {
        const int bc = (jb + Bc <= N) ? Bc : (N - jb);
        for (int ib = 0; ib < M; ib += Br) {
            const int br = (ib + Br <= M) ? Br : (M - ib);
            float S[4][32];
            for (int i = 0; i < br; i++) {
                const __fp16* qi = Q + (ib + i) * d_k;
                // 4-way batched dot for first 4 K rows, then scalar loop
                int j = 0;
                for (; j + 3 < bc; j += 4) {
                    dot_fp16_4x(qi,
                                K + (jb + j) * d_k, K + (jb + j + 1) * d_k,
                                K + (jb + j + 2) * d_k, K + (jb + j + 3) * d_k,
                                d_k, &S[i][j]);
                    S[i][j]   *= scale; S[i][j+1] *= scale;
                    S[i][j+2] *= scale; S[i][j+3] *= scale;
                    if (mask) {
                        S[i][j]   += mask[(ib + i) * N + jb + j];
                        S[i][j+1] += mask[(ib + i) * N + jb + j + 1];
                        S[i][j+2] += mask[(ib + i) * N + jb + j + 2];
                        S[i][j+3] += mask[(ib + i) * N + jb + j + 3];
                    }
                }
                for (; j < bc; j++) {
                    S[i][j] = dot_fp16_neon(qi, K + (jb + j) * d_k, d_k) * scale;
                    if (mask) S[i][j] += mask[(ib + i) * N + jb + j];
                }
            }
            for (int i = 0; i < br; i++) {
                const int row = ib + i;
                float m_old = row_m[row];
                float l_old = row_l[row];
                float* oi = O + row * d_v;
                float m_block = S[i][0];
                for (int j = 1; j < bc; j++) m_block = fmaxf(m_block, S[i][j]);
                float m_new = fmaxf(m_old, m_block);
                float alpha = fast_exp_f32(m_old - m_new);
                {
                    float32x4_t valpha = vdupq_n_f32(alpha);
                    int d = 0;
                    for (; d + 7 < d_v; d += 8) {
                        vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), valpha));
                        vst1q_f32(oi + d + 4, vmulq_f32(vld1q_f32(oi + d + 4), valpha));
                    }
                    for (; d + 3 < d_v; d += 4)
                        vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), valpha));
                    for (; d < d_v; d++) oi[d] *= alpha;
                }
                float l_block = 0.f;
                // PV accumulation: FP16FML
                for (int j = 0; j < bc; j++) {
                    float p = fast_exp_f32(S[i][j] - m_new);
                    l_block += p;
                    const __fp16* vj = V + (jb + j) * d_v;
                    float16x4_t vp = vdup_n_f16((__fp16)p);
                    int d = 0;
                    for (; d + 15 < d_v; d += 16) {
                        float16x8_t v0 = vld1q_f16(vj + d);
                        float16x8_t v1 = vld1q_f16(vj + d + 8);
                        vst1q_f32(oi + d,     vfmlalq_lane_low_f16 (
                                    vld1q_f32(oi + d),     v0, vp, 0));
                        vst1q_f32(oi + d + 4, vfmlalq_lane_high_f16(
                                    vld1q_f32(oi + d + 4), v0, vp, 0));
                        vst1q_f32(oi + d + 8, vfmlalq_lane_low_f16 (
                                    vld1q_f32(oi + d + 8), v1, vp, 0));
                        vst1q_f32(oi + d + 12, vfmlalq_lane_high_f16(
                                    vld1q_f32(oi + d + 12), v1, vp, 0));
                    }
                    for (; d + 7 < d_v; d += 8) {
                        float16x8_t v0 = vld1q_f16(vj + d);
                        vst1q_f32(oi + d,     vfmlalq_lane_low_f16 (
                                    vld1q_f32(oi + d),     v0, vp, 0));
                        vst1q_f32(oi + d + 4, vfmlalq_lane_high_f16(
                                    vld1q_f32(oi + d + 4), v0, vp, 0));
                    }
                    for (; d + 3 < d_v; d += 4) {
                        float16x8_t v0 = vld1q_f16(vj + d);
                        vst1q_f32(oi + d, vfmlalq_lane_low_f16(
                                    vld1q_f32(oi + d), v0, vp, 0));
                    }
                    for (; d < d_v; d++) oi[d] += p * (float)vj[d];
                }
                row_l[row] = alpha * l_old + l_block;
                row_m[row] = m_new;
            }
        }
    }
    for (int i = 0; i < M; i++) {
        float* oi = O + i * d_v;
        float inv_l = (row_l[i] > 0.f) ? 1.f / row_l[i] : 0.f;
        float32x4_t vinv = vdupq_n_f32(inv_l);
        int d = 0;
        for (; d + 3 < d_v; d += 4)
            vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), vinv));
        for (; d < d_v; d++) oi[d] *= inv_l;
    }
    free(row_m);
    free(row_l);
}

#endif // HAS_NEON

// ---------------------------------------------------------------------------
// Naive scalar fallback (for non-NEON platforms)
// ---------------------------------------------------------------------------

#if !HAS_NEON

static inline void softmax_row(float* row, int len) {
    float max_val = -INFINITY;
    for (int i = 0; i < len; i++) max_val = fmaxf(max_val, row[i]);
    float sum = 0.f;
    for (int i = 0; i < len; i++) { row[i] = expf(row[i] - max_val); sum += row[i]; }
    float inv_sum = 1.f / sum;
    for (int i = 0; i < len; i++) row[i] *= inv_sum;
}

static void naive_sdpa_head(
    const float* Q, const float* K, const float* V, float* O,
    int M, int N, int d_k, int d_v, float scale, const float* mask)
{
    float* qk_row = new float[N];
    for (int s = 0; s < M; s++) {
        const float* q = Q + s * d_k;
        for (int j = 0; j < N; j++) {
            float dot = 0.f;
            const float* k = K + j * d_k;
            for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
            qk_row[j] = dot * scale;
            if (mask) qk_row[j] += mask[s * N + j];
        }
        softmax_row(qk_row, N);
        float* o = O + s * d_v;
        memset(o, 0, d_v * sizeof(float));
        for (int j = 0; j < N; j++) {
            float a = qk_row[j];
            if (a == 0.f) continue;
            const float* v = V + j * d_v;
            for (int d = 0; d < d_v; d++) o[d] += a * v[d];
        }
    }
    delete[] qk_row;
}

#endif // !HAS_NEON

// ---------------------------------------------------------------------------
// kernel_sdpa
// ---------------------------------------------------------------------------

void kernel_sdpa(const OpParams& params,
                 const std::vector<const Tensor*>& inputs,
                 std::vector<Tensor*>& outputs,
                 ThreadPool* thread_pool) {
    int kv_cache    = graph_params::get_i32(params, 0, 2);
    int causal      = graph_params::get_i32(params, 1, 1);
    int num_heads   = graph_params::get_i32(params, 2, 16);
    int num_kv_heads= graph_params::get_i32(params, 3, 16);
    int head_dim    = graph_params::get_i32(params, 4, 192);
    int v_head_dim  = graph_params::get_i32(params, 5, 128);
    float scale     = graph_params::get_f32(params, 0, 0.f);
    if (scale == 0.f) scale = 1.f / std::sqrt((float)head_dim);

    int heads_per_group = num_heads / num_kv_heads;

    const Tensor& Q      = *inputs[0];
    const Tensor& K_cur  = *inputs[1];
    const Tensor& V_cur  = *inputs[2];
    const Tensor* mask   = (inputs.size() > 3 && inputs[3] && inputs[3]->data) ? inputs[3] : nullptr;
    const Tensor* K_cache= (inputs.size() > 4 && inputs[4] && inputs[4]->data) ? inputs[4] : nullptr;
    const Tensor* V_cache= (inputs.size() > 5 && inputs[5] && inputs[5]->data) ? inputs[5] : nullptr;

    Tensor& out       = *outputs[0];
    Tensor* K_cache_out = outputs.size() > 1 ? outputs[1] : nullptr;
    Tensor* V_cache_out = outputs.size() > 2 ? outputs[2] : nullptr;

    int src_seqlen = (int)Q.shape[1];
    int cur_seqlen = (int)K_cur.shape[1];
    int past_seqlen = 0;

    // ---- KV cache append ----
    // Cache may be FP16 (preferred for FP16FML SDPA) or FP32.
    // K_cur/V_cur are always FP32 (from matmul output). Convert on append
    // when cache is FP16.
    bool cache_is_fp16 = (kv_cache == 2 && K_cache && K_cache->data &&
                          K_cache->prec == Precision::FP16);
    if (kv_cache == 2 && K_cache && K_cache->data) {
        const CacheMetadata* meta = cache_meta(K_cache->data);
        past_seqlen = (int)meta->current_seq_len;
        size_t cache_es = K_cache->element_size();  // 2 for FP16, 4 for FP32
        size_t cur_es  = K_cur.element_size();      // always 4 (FP32)
        const void* k_cache_data = cache_data(K_cache->data);
        const void* v_cache_data = V_cache ? cache_data(V_cache->data) : nullptr;

        for (int g = 0; g < num_kv_heads; g++) {
            const unsigned char* ks = (const unsigned char*)K_cur.channel<unsigned char>(g);
            size_t k_cur_row_stride = K_cur.stride[1];
            unsigned char* kd = (unsigned char*)k_cache_data + g * (head_dim * meta->max_seq_len * cache_es);
            for (int s = 0; s < cur_seqlen; s++) {
                const float* src_k = (const float*)(ks + s * k_cur_row_stride);
                if (cache_is_fp16) {
                    __fp16* dst_k = (__fp16*)(kd + (past_seqlen + s) * head_dim * cache_es);
                    int d = 0;
                    for (; d + 7 < head_dim; d += 8) {
                        float32x4_t s0 = vld1q_f32(src_k + d);
                        float32x4_t s1 = vld1q_f32(src_k + d + 4);
                        vst1q_f16(dst_k + d, vcombine_f16(vcvt_f16_f32(s0),
                                                          vcvt_f16_f32(s1)));
                    }
                    for (; d + 3 < head_dim; d += 4) {
                        vst1_f16(dst_k + d, vcvt_f16_f32(vld1q_f32(src_k + d)));
                    }
                    for (; d < head_dim; d++) dst_k[d] = (__fp16)src_k[d];
                } else {
                    std::memcpy(kd + (past_seqlen + s) * head_dim * cache_es,
                                ks + s * k_cur_row_stride, head_dim * cache_es);
                }
            }
            const unsigned char* vs = (const unsigned char*)V_cur.channel<unsigned char>(g);
            size_t v_cur_row_stride = V_cur.stride[1];
            unsigned char* vd = (unsigned char*)v_cache_data + g * (v_head_dim * meta->max_seq_len * cache_es);
            for (int s = 0; s < cur_seqlen; s++) {
                const float* src_v = (const float*)(vs + s * v_cur_row_stride);
                if (cache_is_fp16) {
                    __fp16* dst_v = (__fp16*)(vd + (past_seqlen + s) * v_head_dim * cache_es);
                    int d = 0;
                    for (; d + 7 < v_head_dim; d += 8) {
                        float32x4_t s0 = vld1q_f32(src_v + d);
                        float32x4_t s1 = vld1q_f32(src_v + d + 4);
                        vst1q_f16(dst_v + d, vcombine_f16(vcvt_f16_f32(s0),
                                                          vcvt_f16_f32(s1)));
                    }
                    for (; d + 3 < v_head_dim; d += 4) {
                        vst1_f16(dst_v + d, vcvt_f16_f32(vld1q_f32(src_v + d)));
                    }
                    for (; d < v_head_dim; d++) dst_v[d] = (__fp16)src_v[d];
                } else {
                    std::memcpy(vd + (past_seqlen + s) * v_head_dim * cache_es,
                                vs + s * v_cur_row_stride, v_head_dim * cache_es);
                }
            }
        }
    } else if (kv_cache == 1 && K_cache && K_cache->data) {
        const CacheMetadata* meta = cache_meta(K_cache->data);
        past_seqlen = (int)meta->current_seq_len;
    }

    int dst_seqlen = past_seqlen + cur_seqlen;

    // ---- Get contiguous K/V pointers per head ----
    // After cache append, K/V are [max_seq_len, head_dim/v_head_dim] per head.
    // For kv_cache=0 (no cache), K/V come from K_cur/V_cur directly (FP32).
    // For kv_cache=2 with FP16 cache, returns __fp16*; otherwise float*.
    // Caller must check cache_is_fp16 to interpret the pointer type.
    auto get_k_ptr = [&](int kv_h) -> const void* {
        if (kv_cache == 2 && K_cache && K_cache->data) {
            return (const char*)cache_data(K_cache->data) +
                   kv_h * (head_dim * cache_meta(K_cache->data)->max_seq_len) * K_cache->element_size();
        }
        return (const void*)K_cur.channel<unsigned char>(kv_h);
    };
    auto get_v_ptr = [&](int kv_h) -> const void* {
        if (kv_cache == 2 && V_cache && V_cache->data) {
            return (const char*)cache_data(V_cache->data) +
                   kv_h * (v_head_dim * cache_meta(V_cache->data)->max_seq_len) * V_cache->element_size();
        }
        return (const void*)V_cur.channel<unsigned char>(kv_h);
    };

    const float* mask_ptr = mask ? (const float*)mask->channel<unsigned char>(0) : nullptr;

    // Build causal mask if needed (when no explicit mask but causal=1)
    float* causal_mask = nullptr;
    if (!mask_ptr && causal && src_seqlen > 1) {
        causal_mask = new float[src_seqlen * dst_seqlen];
        for (int i = 0; i < src_seqlen; i++) {
            for (int j = 0; j < dst_seqlen; j++) {
                causal_mask[i * dst_seqlen + j] =
                    (j <= past_seqlen + i) ? 0.f : -INFINITY;
            }
        }
        mask_ptr = causal_mask;
    }

#if HAS_NEON
    // ---- Flash attention path (parallel across heads) ----
    // FP16FML path:
    //   - K/V cache stored as FP16 (cache_is_fp16=true) → use directly, no
    //     per-call conversion. This is the key win: K/V are the big tensors
    //     (N × d_k / N × d_v), converting them per-call dominated the cost.
    //   - Q converted per-call (M × d_k, small).
    //   - If cache is FP32 (e.g. test cases), fall back to FP32 kernels.
    auto run_head = [&](int, int h_begin, int h_end) {
        // Per-shard FP16 Q buffer (small, M × d_k).
        __fp16* Q_h16 = nullptr;
        size_t q_size = (size_t)src_seqlen * head_dim;
        if (cache_is_fp16) {
            Q_h16 = new __fp16[q_size];
        }

        for (int h = h_begin; h < h_end; h++) {
            int kv_h = h / heads_per_group;
            float* O_head = (float*)out.channel<unsigned char>(h);

            if (cache_is_fp16) {
                // FP16FML path: K/V already FP16 in cache, only convert Q.
                const __fp16* K_head = (const __fp16*)get_k_ptr(kv_h);
                const __fp16* V_head = (const __fp16*)get_v_ptr(kv_h);
                const float* Q_head = (const float*)Q.channel<unsigned char>(h);

                // Convert Q [src_seqlen * d_k] to FP16
                {
                    size_t i = 0;
                    for (; i + 7 < q_size; i += 8) {
                        float32x4_t s0 = vld1q_f32(Q_head + i);
                        float32x4_t s1 = vld1q_f32(Q_head + i + 4);
                        vst1q_f16(Q_h16 + i, vcombine_f16(vcvt_f16_f32(s0),
                                                          vcvt_f16_f32(s1)));
                    }
                    for (; i + 3 < q_size; i += 4) {
                        vst1_f16(Q_h16 + i, vcvt_f16_f32(vld1q_f32(Q_head + i)));
                    }
                    for (; i < q_size; i++) Q_h16[i] = (__fp16)Q_head[i];
                }

                if (src_seqlen == 1) {
                    flash_attn_fp16_decode(Q_h16, K_head, V_head, O_head,
                                           dst_seqlen, head_dim, v_head_dim, scale, mask_ptr);
                } else {
                    flash_attn_fp16_prefill(Q_h16, K_head, V_head, O_head,
                                            src_seqlen, dst_seqlen, head_dim, v_head_dim,
                                            scale, mask_ptr);
                }
            } else {
                // FP32 cache fallback
                const float* Q_head = (const float*)Q.channel<unsigned char>(h);
                const float* K_head = (const float*)get_k_ptr(kv_h);
                const float* V_head = (const float*)get_v_ptr(kv_h);

                if (src_seqlen == 1) {
                    flash_attn_fp32_decode(Q_head, K_head, V_head, O_head,
                                           dst_seqlen, head_dim, v_head_dim, scale, mask_ptr);
                } else {
                    flash_attn_fp32_prefill(Q_head, K_head, V_head, O_head,
                                            src_seqlen, dst_seqlen, head_dim, v_head_dim,
                                            scale, mask_ptr);
                }
            }
        }

        delete[] Q_h16;
    };

    if (thread_pool && num_heads >= 2) {
        thread_pool->parallel_for(0, num_heads, 1, run_head);
    } else {
        run_head(0, 0, num_heads);
    }
#else
    // ---- Naive scalar fallback ----
    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_group;
        const float* Q_head = (const float*)Q.channel<unsigned char>(h);
        const float* K_head = get_k_ptr(kv_h);
        const float* V_head = get_v_ptr(kv_h);
        float* O_head = (float*)out.channel<unsigned char>(h);
        naive_sdpa_head(Q_head, K_head, V_head, O_head,
                        src_seqlen, dst_seqlen, head_dim, v_head_dim, scale, mask_ptr);
    }
#endif

    delete[] causal_mask;

    // ---- return cache views ----
    if (kv_cache == 2 && K_cache_out && K_cache) {
        *K_cache_out = *K_cache;
        K_cache_out->shape[1] = dst_seqlen;
        K_cache_out->stride[2] = K_cache_out->stride[1] * dst_seqlen;
        K_cache_out->stride[3] = K_cache_out->stride[2];
    }
    if (kv_cache == 2 && V_cache_out && V_cache) {
        *V_cache_out = *V_cache;
        V_cache_out->shape[1] = dst_seqlen;
        V_cache_out->stride[2] = V_cache_out->stride[1] * dst_seqlen;
        V_cache_out->stride[3] = V_cache_out->stride[2];
    }
}
