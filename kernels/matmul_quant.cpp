#include "kernels/matmul_internal.h"
#include "kernels/matmul_profile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#if HAS_NEON && defined(__aarch64__)
static inline void quantize_q8_block32_neon(const float* src, float& scale,
                                            int8x16_t& q_lo, int8x16_t& q_hi) {
    float32x4_t v0 = vld1q_f32(src + 0);
    float32x4_t v1 = vld1q_f32(src + 4);
    float32x4_t v2 = vld1q_f32(src + 8);
    float32x4_t v3 = vld1q_f32(src + 12);
    float32x4_t v4 = vld1q_f32(src + 16);
    float32x4_t v5 = vld1q_f32(src + 20);
    float32x4_t v6 = vld1q_f32(src + 24);
    float32x4_t v7 = vld1q_f32(src + 28);
    float32x4_t m0 = vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1));
    float32x4_t m1 = vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3));
    float32x4_t m2 = vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5));
    float32x4_t m3 = vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7));
    float amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));

    scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
    float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;

    int32x4_t q0 = vcvtnq_s32_f32(vmulq_n_f32(v0, inv_scale));
    int32x4_t q1 = vcvtnq_s32_f32(vmulq_n_f32(v1, inv_scale));
    int32x4_t q2 = vcvtnq_s32_f32(vmulq_n_f32(v2, inv_scale));
    int32x4_t q3 = vcvtnq_s32_f32(vmulq_n_f32(v3, inv_scale));
    int32x4_t q4 = vcvtnq_s32_f32(vmulq_n_f32(v4, inv_scale));
    int32x4_t q5 = vcvtnq_s32_f32(vmulq_n_f32(v5, inv_scale));
    int32x4_t q6 = vcvtnq_s32_f32(vmulq_n_f32(v6, inv_scale));
    int32x4_t q7 = vcvtnq_s32_f32(vmulq_n_f32(v7, inv_scale));
    int32x4_t qmin = vdupq_n_s32(-127);
    int32x4_t qmax = vdupq_n_s32(127);
    q0 = vmaxq_s32(qmin, vminq_s32(qmax, q0));
    q1 = vmaxq_s32(qmin, vminq_s32(qmax, q1));
    q2 = vmaxq_s32(qmin, vminq_s32(qmax, q2));
    q3 = vmaxq_s32(qmin, vminq_s32(qmax, q3));
    q4 = vmaxq_s32(qmin, vminq_s32(qmax, q4));
    q5 = vmaxq_s32(qmin, vminq_s32(qmax, q5));
    q6 = vmaxq_s32(qmin, vminq_s32(qmax, q6));
    q7 = vmaxq_s32(qmin, vminq_s32(qmax, q7));

    int16x8_t q01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
    int16x8_t q23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
    int16x8_t q45 = vcombine_s16(vqmovn_s32(q4), vqmovn_s32(q5));
    int16x8_t q67 = vcombine_s16(vqmovn_s32(q6), vqmovn_s32(q7));
    q_lo = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));
    q_hi = vcombine_s8(vqmovn_s16(q45), vqmovn_s16(q67));
}
#endif

void quantize_a_q8_blocks(const float* A, int M, int K, int lda, int K_storage,
                          std::vector<int8_t>& qA,
                          std::vector<float>& a_scales) {
    auto t0 = std::chrono::steady_clock::now();
    if (K_storage < K)
        K_storage = K;
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    if (K_storage == K) {
        qA.resize((size_t)M * K_storage);
    } else {
        qA.assign((size_t)M * K_storage, 0);
    }
    a_scales.resize((size_t)M * blocks_per_row);
    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * lda;
        int8_t* qa_row = qA.data() + (size_t)m * K_storage;
        float* s_row = a_scales.data() + (size_t)m * blocks_per_row;
        for (int qb = 0; qb < blocks_per_row; qb++) {
            int k_begin = qb * MATMUL_Q8_BLOCK;
            int k_end = std::min(k_begin + MATMUL_Q8_BLOCK, K);
            float amax = 0.f;
#if HAS_NEON && defined(__aarch64__)
            if (k_end - k_begin == MATMUL_Q8_BLOCK) {
                float scale = 1.f;
                int8x16_t q_lo;
                int8x16_t q_hi;
                quantize_q8_block32_neon(a_row + k_begin, scale, q_lo, q_hi);
                s_row[qb] = scale;
                vst1q_s8(qa_row + k_begin, q_lo);
                vst1q_s8(qa_row + k_begin + 16, q_hi);
                continue;
            }
#endif
            for (int k = k_begin; k < k_end; k++) {
                amax = std::max(amax, std::fabs(a_row[k]));
            }
            float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
            float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;
            s_row[qb] = scale;
            for (int k = k_begin; k < k_end; k++) {
                int q = (int)std::nearbyint(a_row[k] * inv_scale);
                q = std::max(-127, std::min(127, q));
                qa_row[k] = (int8_t)q;
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    matmul_record_q8_quant_a(ms);
}

#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
void quantize_a_q8_blocks_a4(const float* A, int M, int K, int lda,
                             std::vector<Q8A4Block>& qA4) {
    auto t0 = std::chrono::steady_clock::now();
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    int m_tiles = (M + 3) / 4;
    qA4.resize((size_t)m_tiles * blocks_per_row);
    for (int mt = 0; mt < m_tiles; mt++) {
        for (int qb = 0; qb < blocks_per_row; qb++) {
            int k_begin = qb * MATMUL_Q8_BLOCK;
            int k_end = std::min(k_begin + MATMUL_Q8_BLOCK, K);
            Q8A4Block& block = qA4[(size_t)mt * blocks_per_row + qb];
            for (int ar = 0; ar < 4; ar++) {
                int m = mt * 4 + ar;
                if (m >= M)
                    continue;

                const float* a_row = A + (size_t)m * lda;
#if HAS_NEON && defined(__aarch64__)
                if (k_end - k_begin == MATMUL_Q8_BLOCK) {
                    float scale = 1.f;
                    int8x16_t q_lo;
                    int8x16_t q_hi;
                    quantize_q8_block32_neon(a_row + k_begin, scale, q_lo,
                                             q_hi);
                    block.scales[ar] = scale;
                    vst1q_s8(block.even[ar], vuzp1q_s8(q_lo, q_hi));
                    vst1q_s8(block.odd[ar], vuzp2q_s8(q_lo, q_hi));
                    continue;
                }
#endif
                float amax = 0.f;
                for (int k = k_begin; k < k_end; k++) {
                    amax = std::max(amax, std::fabs(a_row[k]));
                }
                float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
                float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;
                block.scales[ar] = scale;
                for (int i = 0; i < 16; i++) {
                    int k0 = k_begin + i * 2;
                    int k1 = k0 + 1;
                    int q0 = (k0 < k_end)
                                 ? (int)std::nearbyint(a_row[k0] * inv_scale)
                                 : 0;
                    int q1 = (k1 < k_end)
                                 ? (int)std::nearbyint(a_row[k1] * inv_scale)
                                 : 0;
                    q0 = std::max(-127, std::min(127, q0));
                    q1 = std::max(-127, std::min(127, q1));
                    block.even[ar][i] = (int8_t)q0;
                    block.odd[ar][i] = (int8_t)q1;
                }
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    matmul_record_q8_quant_a(ms);
}
#endif

void quantize_a_q8_blocks_even_odd(const float* A, int K,
                                   std::vector<int8_t>& qA_even,
                                   std::vector<int8_t>& qA_odd,
                                   std::vector<float>& a_scales) {
    auto t0 = std::chrono::steady_clock::now();
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    qA_even.resize((size_t)blocks_per_row * 16);
    qA_odd.resize((size_t)blocks_per_row * 16);
    a_scales.resize((size_t)blocks_per_row);
    for (int qb = 0; qb < blocks_per_row; qb++) {
        int k_begin = qb * MATMUL_Q8_BLOCK;
        int k_end = std::min(k_begin + MATMUL_Q8_BLOCK, K);
        float amax = 0.f;
#if HAS_NEON && defined(__aarch64__)
        if (k_end - k_begin == MATMUL_Q8_BLOCK) {
            float scale = 1.f;
            int8x16_t q_lo;
            int8x16_t q_hi;
            quantize_q8_block32_neon(A + k_begin, scale, q_lo, q_hi);
            a_scales[qb] = scale;
            vst1q_s8(qA_even.data() + (size_t)qb * 16, vuzp1q_s8(q_lo, q_hi));
            vst1q_s8(qA_odd.data() + (size_t)qb * 16, vuzp2q_s8(q_lo, q_hi));
            continue;
        }
#endif
        for (int k = k_begin; k < k_end; k++) {
            amax = std::max(amax, std::fabs(A[k]));
        }
        float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
        float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;
        a_scales[qb] = scale;

        int8_t* even = qA_even.data() + (size_t)qb * 16;
        int8_t* odd = qA_odd.data() + (size_t)qb * 16;
        for (int i = 0; i < 16; i++) {
            int k0 = k_begin + i * 2;
            int k1 = k0 + 1;
            int q0 = (k0 < k_end) ? (int)std::nearbyint(A[k0] * inv_scale) : 0;
            int q1 = (k1 < k_end) ? (int)std::nearbyint(A[k1] * inv_scale) : 0;
            q0 = std::max(-127, std::min(127, q0));
            q1 = std::max(-127, std::min(127, q1));
            even[i] = (int8_t)q0;
            odd[i] = (int8_t)q1;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    matmul_record_q8_quant_a(ms);
}
