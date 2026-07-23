#include "kernels/matmul_internal.h"
#include "kernels/threading.h"

#include <algorithm>
#include <cstdint>
#include <vector>

static void matmul_int8_scalar_range(const float* A, const int8_t* B,
                                     const float* scales, int group_size,
                                     int groups_per_row, float* C, int M, int N,
                                     int K, int lda, int K_weight, int ldc,
                                     int m_begin, int m_end, int n_begin,
                                     int n_end, bool b_interleaved = false) {
    (void)M;
    (void)N;
    if (group_size <= 0)
        group_size = K;
    if (groups_per_row <= 0)
        groups_per_row = 1;

    for (int m = m_begin; m < m_end; m++) {
        float* c_row = C + m * ldc;
        for (int n = n_begin; n < n_end; n++) {
            const int8_t* b_row = B + n * K_weight;
            const float* s_row = scales + n * groups_per_row;
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                int g = k / group_size;
                int8_t q = b_interleaved
                               ? B[(n & ~7) * K_weight + k * 8 + (n & 7)]
                               : b_row[k];
                sum += A[k + m * lda] * ((float)q * s_row[g]);
            }
            c_row[n] = sum;
        }
    }
}

static inline int8_t load_b_int8_value(const int8_t* B, int n, int k,
                                       int K_weight, bool b_interleaved) {
    return b_interleaved ? B[(n & ~7) * K_weight + k * 8 + (n & 7)]
                         : B[n * K_weight + k];
}

static void matmul_int8_q8dot_scalar_range(
    const int8_t* qA, const float* a_scales, const int8_t* B,
    const float* scales, int group_size, int groups_per_row, float* C, int M,
    int N, int K, int K_weight, int ldc, int m_begin, int m_end, int n_begin,
    int n_end, bool b_interleaved) {
    (void)M;
    (void)N;
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    for (int m = m_begin; m < m_end; m++) {
        const int8_t* qa_row = qA + (size_t)m * K;
        const float* as_row = a_scales + (size_t)m * blocks_per_row;
        float* c_row = C + m * ldc;
        for (int n = n_begin; n < n_end; n++) {
            const float* bs_row = scales + n * groups_per_row;
            float sum = 0.f;
            int k0 = 0;
            while (k0 < K) {
                int q_block = k0 / MATMUL_Q8_BLOCK;
                int group = k0 / group_size;
                int k_end = std::min({K, (q_block + 1) * MATMUL_Q8_BLOCK,
                                      (group + 1) * group_size});
                int32_t dot = 0;
                for (int k = k0; k < k_end; k++) {
                    dot += (int32_t)qa_row[k] *
                           (int32_t)load_b_int8_value(B, n, k, K_weight,
                                                      b_interleaved);
                }
                sum += (float)dot * as_row[q_block] * bs_row[group];
                k0 = k_end;
            }
            c_row[n] = sum;
        }
    }
}

#if HAS_NEON
static void matmul_int8_neon_gemv_range(const float* A, const int8_t* B_packed,
                                        const float* scales, int group_size,
                                        int groups_per_row, float* C, int K,
                                        int n_begin, int n_end) {
    if (group_size <= 0)
        group_size = K;
    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        const int8_t* b_tile = &B_packed[(n & ~7) * K];

        float32x4_t acc0 = vdupq_n_f32(0.f);
        float32x4_t acc1 = vdupq_n_f32(0.f);

        for (int g = 0; g < groups_per_row; g++) {
            int k_begin = g * group_size;
            int k_end = std::min(k_begin + group_size, K);

            float scale_tmp[8];
            for (int j = 0; j < 8; j++) {
                scale_tmp[j] = (n + j < n_tile_end)
                                   ? scales[(n + j) * groups_per_row + g]
                                   : 0.f;
            }
            float32x4_t scale0 = vld1q_f32(scale_tmp);
            float32x4_t scale1 = vld1q_f32(scale_tmp + 4);

            for (int k = k_begin; k < k_end; k++) {
                int8x8_t b8 = vld1_s8(b_tile + k * 8);
                int16x8_t b16 = vmovl_s8(b8);
                float32x4_t b0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
                float32x4_t b1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));
                float a = A[k];
                acc0 = vfmaq_f32(acc0, b0, vmulq_n_f32(scale0, a));
                acc1 = vfmaq_f32(acc1, b1, vmulq_n_f32(scale1, a));
            }
        }

        float tmp[4];
        vst1q_f32(tmp, acc0);
        for (int j = 0; j < 4 && n + j < n_tile_end; j++)
            C[n + j] = tmp[j];
        vst1q_f32(tmp, acc1);
        for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++)
            C[n + 4 + j] = tmp[j];
    }
}

static void matmul_int8_neon_4x8_range(const float* A, const int8_t* B_packed,
                                       const float* scales, int group_size,
                                       int groups_per_row, float* C, int M,
                                       int N, int K, int lda, int ldc,
                                       int m_begin, int m_end) {
    (void)M;
    if (group_size <= 0)
        group_size = K;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);

        for (int n = 0; n < N; n += 8) {
            int n_tile_end = std::min(n + 8, N);
            const int8_t* b_tile = &B_packed[(n & ~7) * K];

            float32x4_t c0_lo = vdupq_n_f32(0.f);
            float32x4_t c0_hi = vdupq_n_f32(0.f);
            float32x4_t c1_lo = vdupq_n_f32(0.f);
            float32x4_t c1_hi = vdupq_n_f32(0.f);
            float32x4_t c2_lo = vdupq_n_f32(0.f);
            float32x4_t c2_hi = vdupq_n_f32(0.f);
            float32x4_t c3_lo = vdupq_n_f32(0.f);
            float32x4_t c3_hi = vdupq_n_f32(0.f);

            for (int g = 0; g < groups_per_row; g++) {
                int k_begin = g * group_size;
                int k_end = std::min(k_begin + group_size, K);

                float scale_tmp[8];
                for (int j = 0; j < 8; j++) {
                    scale_tmp[j] = (n + j < n_tile_end)
                                       ? scales[(n + j) * groups_per_row + g]
                                       : 0.f;
                }
                float32x4_t scale0 = vld1q_f32(scale_tmp);
                float32x4_t scale1 = vld1q_f32(scale_tmp + 4);

                for (int k = k_begin; k < k_end; k++) {
                    int8x8_t b8 = vld1_s8(b_tile + k * 8);
                    int16x8_t b16 = vmovl_s8(b8);
                    float32x4_t b0 = vmulq_f32(
                        vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16))), scale0);
                    float32x4_t b1 = vmulq_f32(
                        vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16))), scale1);

                    float a0 = A[k + (m + 0) * lda];
                    c0_lo = vfmaq_n_f32(c0_lo, b0, a0);
                    c0_hi = vfmaq_n_f32(c0_hi, b1, a0);
                    if (m + 1 < m_tile_end) {
                        float a1 = A[k + (m + 1) * lda];
                        c1_lo = vfmaq_n_f32(c1_lo, b0, a1);
                        c1_hi = vfmaq_n_f32(c1_hi, b1, a1);
                    }
                    if (m + 2 < m_tile_end) {
                        float a2 = A[k + (m + 2) * lda];
                        c2_lo = vfmaq_n_f32(c2_lo, b0, a2);
                        c2_hi = vfmaq_n_f32(c2_hi, b1, a2);
                    }
                    if (m + 3 < m_tile_end) {
                        float a3 = A[k + (m + 3) * lda];
                        c3_lo = vfmaq_n_f32(c3_lo, b0, a3);
                        c3_hi = vfmaq_n_f32(c3_hi, b1, a3);
                    }
                }
            }

            auto store_row = [&](int row, float32x4_t lo, float32x4_t hi) {
                if (row >= m_tile_end)
                    return;
                float* c_row = C + row * ldc;
                float tmp[4];
                vst1q_f32(tmp, lo);
                for (int j = 0; j < 4 && n + j < n_tile_end; j++)
                    c_row[n + j] = tmp[j];
                vst1q_f32(tmp, hi);
                for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++)
                    c_row[n + 4 + j] = tmp[j];
            };
            store_row(m + 0, c0_lo, c0_hi);
            store_row(m + 1, c1_lo, c1_hi);
            store_row(m + 2, c2_lo, c2_hi);
            store_row(m + 3, c3_lo, c3_hi);
        }
    }
}

static void
matmul_int8_q8dot_neon_gemv_range(const int8_t* qA, const float* a_scales,
                                  const int8_t* B_packed, const float* scales,
                                  int group_size, int groups_per_row, float* C,
                                  int K, int n_begin, int n_end) {
    if (group_size <= 0)
        group_size = K;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        const int8_t* b_tile = &B_packed[(n & ~7) * K];
        float32x4_t acc0 = vdupq_n_f32(0.f);
        float32x4_t acc1 = vdupq_n_f32(0.f);

        int k0 = 0;
        while (k0 < K) {
            int q_block = k0 / MATMUL_Q8_BLOCK;
            int group = k0 / group_size;
            int k_end = std::min(
                {K, (q_block + 1) * MATMUL_Q8_BLOCK, (group + 1) * group_size});
            int32x4_t dot0 = vdupq_n_s32(0);
            int32x4_t dot1 = vdupq_n_s32(0);

            for (int k = k0; k < k_end; k++) {
                int16x8_t b16 = vmovl_s8(vld1_s8(b_tile + k * 8));
                int16x8_t prod = vmulq_n_s16(b16, (int16_t)qA[k]);
                dot0 = vaddw_s16(dot0, vget_low_s16(prod));
                dot1 = vaddw_s16(dot1, vget_high_s16(prod));
            }

            float scale_tmp[8];
            float a_scale = a_scales[q_block];
            for (int j = 0; j < 8; j++) {
                scale_tmp[j] =
                    (n + j < n_tile_end)
                        ? a_scale * scales[(n + j) * groups_per_row + group]
                        : 0.f;
            }
            acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(dot0), vld1q_f32(scale_tmp));
            acc1 =
                vfmaq_f32(acc1, vcvtq_f32_s32(dot1), vld1q_f32(scale_tmp + 4));
            k0 = k_end;
        }

        float tmp[4];
        vst1q_f32(tmp, acc0);
        for (int j = 0; j < 4 && n + j < n_tile_end; j++)
            C[n + j] = tmp[j];
        vst1q_f32(tmp, acc1);
        for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++)
            C[n + 4 + j] = tmp[j];
    }
}

static void matmul_int8_q8dot_neon_4x8_range(
    const int8_t* qA, const float* a_scales, const int8_t* B_packed,
    const float* scales, int group_size, int groups_per_row, float* C, int M,
    int N, int K, int ldc, int m_begin, int m_end) {
    (void)M;
    if (group_size <= 0)
        group_size = K;
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);

        for (int n = 0; n < N; n += 8) {
            int n_tile_end = std::min(n + 8, N);
            const int8_t* b_tile = &B_packed[(n & ~7) * K];

            float32x4_t c0_lo = vdupq_n_f32(0.f);
            float32x4_t c0_hi = vdupq_n_f32(0.f);
            float32x4_t c1_lo = vdupq_n_f32(0.f);
            float32x4_t c1_hi = vdupq_n_f32(0.f);
            float32x4_t c2_lo = vdupq_n_f32(0.f);
            float32x4_t c2_hi = vdupq_n_f32(0.f);
            float32x4_t c3_lo = vdupq_n_f32(0.f);
            float32x4_t c3_hi = vdupq_n_f32(0.f);

            int k0 = 0;
            while (k0 < K) {
                int q_block = k0 / MATMUL_Q8_BLOCK;
                int group = k0 / group_size;
                int k_end = std::min({K, (q_block + 1) * MATMUL_Q8_BLOCK,
                                      (group + 1) * group_size});

                int32x4_t d0_lo = vdupq_n_s32(0);
                int32x4_t d0_hi = vdupq_n_s32(0);
                int32x4_t d1_lo = vdupq_n_s32(0);
                int32x4_t d1_hi = vdupq_n_s32(0);
                int32x4_t d2_lo = vdupq_n_s32(0);
                int32x4_t d2_hi = vdupq_n_s32(0);
                int32x4_t d3_lo = vdupq_n_s32(0);
                int32x4_t d3_hi = vdupq_n_s32(0);

                const int8_t* qa0 = qA + (size_t)(m + 0) * K;
                const int8_t* qa1 =
                    (m + 1 < m_tile_end) ? qA + (size_t)(m + 1) * K : qa0;
                const int8_t* qa2 =
                    (m + 2 < m_tile_end) ? qA + (size_t)(m + 2) * K : qa0;
                const int8_t* qa3 =
                    (m + 3 < m_tile_end) ? qA + (size_t)(m + 3) * K : qa0;

                for (int k = k0; k < k_end; k++) {
                    int16x8_t b16 = vmovl_s8(vld1_s8(b_tile + k * 8));
                    int16x4_t b_lo = vget_low_s16(b16);
                    int16x4_t b_hi = vget_high_s16(b16);

                    int16x8_t p0 = vmulq_n_s16(b16, (int16_t)qa0[k]);
                    d0_lo = vaddw_s16(d0_lo, vget_low_s16(p0));
                    d0_hi = vaddw_s16(d0_hi, vget_high_s16(p0));
                    if (m + 1 < m_tile_end) {
                        d1_lo =
                            vaddw_s16(d1_lo, vmul_n_s16(b_lo, (int16_t)qa1[k]));
                        d1_hi =
                            vaddw_s16(d1_hi, vmul_n_s16(b_hi, (int16_t)qa1[k]));
                    }
                    if (m + 2 < m_tile_end) {
                        d2_lo =
                            vaddw_s16(d2_lo, vmul_n_s16(b_lo, (int16_t)qa2[k]));
                        d2_hi =
                            vaddw_s16(d2_hi, vmul_n_s16(b_hi, (int16_t)qa2[k]));
                    }
                    if (m + 3 < m_tile_end) {
                        d3_lo =
                            vaddw_s16(d3_lo, vmul_n_s16(b_lo, (int16_t)qa3[k]));
                        d3_hi =
                            vaddw_s16(d3_hi, vmul_n_s16(b_hi, (int16_t)qa3[k]));
                    }
                }

                float b_scale_tmp[8];
                for (int j = 0; j < 8; j++) {
                    b_scale_tmp[j] =
                        (n + j < n_tile_end)
                            ? scales[(n + j) * groups_per_row + group]
                            : 0.f;
                }
                float32x4_t bs0 = vld1q_f32(b_scale_tmp);
                float32x4_t bs1 = vld1q_f32(b_scale_tmp + 4);

                auto add_row = [&](int row, int32x4_t lo, int32x4_t hi,
                                   float32x4_t& c_lo, float32x4_t& c_hi) {
                    float a_scale =
                        a_scales[(size_t)row * blocks_per_row + q_block];
                    c_lo = vfmaq_f32(c_lo, vcvtq_f32_s32(lo),
                                     vmulq_n_f32(bs0, a_scale));
                    c_hi = vfmaq_f32(c_hi, vcvtq_f32_s32(hi),
                                     vmulq_n_f32(bs1, a_scale));
                };
                add_row(m + 0, d0_lo, d0_hi, c0_lo, c0_hi);
                if (m + 1 < m_tile_end)
                    add_row(m + 1, d1_lo, d1_hi, c1_lo, c1_hi);
                if (m + 2 < m_tile_end)
                    add_row(m + 2, d2_lo, d2_hi, c2_lo, c2_hi);
                if (m + 3 < m_tile_end)
                    add_row(m + 3, d3_lo, d3_hi, c3_lo, c3_hi);

                k0 = k_end;
            }

            auto store_row = [&](int row, float32x4_t lo, float32x4_t hi) {
                if (row >= m_tile_end)
                    return;
                float* c_row = C + row * ldc;
                float tmp[4];
                vst1q_f32(tmp, lo);
                for (int j = 0; j < 4 && n + j < n_tile_end; j++)
                    c_row[n + j] = tmp[j];
                vst1q_f32(tmp, hi);
                for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++)
                    c_row[n + 4 + j] = tmp[j];
            };
            store_row(m + 0, c0_lo, c0_hi);
            store_row(m + 1, c1_lo, c1_hi);
            store_row(m + 2, c2_lo, c2_hi);
            store_row(m + 3, c3_lo, c3_hi);
        }
    }
}

#if defined(__ARM_FEATURE_DOTPROD)
static void matmul_int8_q8dot_neon_gemv_repacked_range(
    const int8_t* qA, const float* a_scales, const int8_t* B_repack,
    const float* scales, int group_size, int groups_per_row, float* C, int K,
    int K_padded, int n_begin, int n_end) {
    (void)K_padded;
    if (group_size <= 0)
        group_size = K;
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        int c_valid = n_tile_end - n;
        const int8_t* b_tile =
            B_repack + (size_t)(n / 8) * blocks_per_row * 8 * MATMUL_Q8_BLOCK;
        float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
        float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
        if (scale_per_channel) {
            load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                              bscale_lo_pc, bscale_hi_pc);
        }

        float32x4_t acc_lo = vdupq_n_f32(0.f);
        float32x4_t acc_hi = vdupq_n_f32(0.f);

        for (int qb = 0; qb < blocks_per_row; qb++) {
            int group = w8_scale_group(scale_mode, qb, group_size);
            const int8_t* b_block = b_tile + (size_t)qb * 8 * MATMUL_Q8_BLOCK;
            int32x4_t d0 = vdupq_n_s32(0);
            int32x4_t d1 = vdupq_n_s32(0);
            int32x4_t d2 = vdupq_n_s32(0);
            int32x4_t d3 = vdupq_n_s32(0);
            int32x4_t d4 = vdupq_n_s32(0);
            int32x4_t d5 = vdupq_n_s32(0);
            int32x4_t d6 = vdupq_n_s32(0);
            int32x4_t d7 = vdupq_n_s32(0);

            for (int half = 0; half < MATMUL_Q8_BLOCK; half += 16) {
                const int8_t* qa = qA + (size_t)qb * MATMUL_Q8_BLOCK + half;
                int8x16_t a_vec = vld1q_s8(qa);
                d0 = vdotq_s32(
                    d0, vld1q_s8(b_block + 0 * MATMUL_Q8_BLOCK + half), a_vec);
                d1 = vdotq_s32(
                    d1, vld1q_s8(b_block + 1 * MATMUL_Q8_BLOCK + half), a_vec);
                d2 = vdotq_s32(
                    d2, vld1q_s8(b_block + 2 * MATMUL_Q8_BLOCK + half), a_vec);
                d3 = vdotq_s32(
                    d3, vld1q_s8(b_block + 3 * MATMUL_Q8_BLOCK + half), a_vec);
                d4 = vdotq_s32(
                    d4, vld1q_s8(b_block + 4 * MATMUL_Q8_BLOCK + half), a_vec);
                d5 = vdotq_s32(
                    d5, vld1q_s8(b_block + 5 * MATMUL_Q8_BLOCK + half), a_vec);
                d6 = vdotq_s32(
                    d6, vld1q_s8(b_block + 6 * MATMUL_Q8_BLOCK + half), a_vec);
                d7 = vdotq_s32(
                    d7, vld1q_s8(b_block + 7 * MATMUL_Q8_BLOCK + half), a_vec);
            }

            int32x4_t p01 = vpaddq_s32(d0, d1);
            int32x4_t p23 = vpaddq_s32(d2, d3);
            int32x4_t p45 = vpaddq_s32(d4, d5);
            int32x4_t p67 = vpaddq_s32(d6, d7);
            int32x4_t dots_lo = vpaddq_s32(p01, p23);
            int32x4_t dots_hi = vpaddq_s32(p45, p67);

            float a_scale = a_scales[qb];
            float32x4_t bscale_lo = bscale_lo_pc;
            float32x4_t bscale_hi = bscale_hi_pc;
            if (!scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                  bscale_lo, bscale_hi);
            }
            float32x4_t scale_lo = vmulq_n_f32(bscale_lo, a_scale);
            float32x4_t scale_hi = vmulq_n_f32(bscale_hi, a_scale);

            acc_lo = vfmaq_f32(acc_lo, vcvtq_f32_s32(dots_lo), scale_lo);
            acc_hi = vfmaq_f32(acc_hi, vcvtq_f32_s32(dots_hi), scale_hi);
        }

        float tmp[4];
        vst1q_f32(tmp, acc_lo);
        for (int c = 0; c < 4 && c < c_valid; c++)
            C[n + c] = tmp[c];
        vst1q_f32(tmp, acc_hi);
        for (int c = 0; c < 4 && c + 4 < c_valid; c++)
            C[n + 4 + c] = tmp[c];
    }
}

static void matmul_int8_q8dot_neon_4x8_repacked_range(
    const int8_t* qA, const float* a_scales, const int8_t* B_repack,
    const float* scales, int group_size, int groups_per_row, float* C, int M,
    int N, int K, int K_padded, int ldc, int m_begin, int m_end, int n_begin,
    int n_end) {
    (void)M;
    (void)N;
    if (group_size <= 0)
        group_size = K;
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);
        int r_valid = m_tile_end - m;

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            const int8_t* b_tile = B_repack + (size_t)(n / 8) * blocks_per_row *
                                                  8 * MATMUL_Q8_BLOCK;
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float acc[4][8] = {};

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int group = w8_scale_group(scale_mode, qb, group_size);
                const int8_t* b_block =
                    b_tile + (size_t)qb * 8 * MATMUL_Q8_BLOCK;
                int32_t dots[4][8] = {};

                for (int half = 0; half < MATMUL_Q8_BLOCK; half += 16) {
                    int8x16_t b_vec[8];
                    for (int c = 0; c < 8; c++) {
                        b_vec[c] =
                            vld1q_s8(b_block + c * MATMUL_Q8_BLOCK + half);
                    }
                    for (int r = 0; r < r_valid; r++) {
                        const int8_t* qa = qA + (size_t)(m + r) * K_padded +
                                           qb * MATMUL_Q8_BLOCK + half;
                        int8x16_t a_vec = vld1q_s8(qa);
                        for (int c = 0; c < 8; c++) {
                            int32x4_t dot =
                                vdotq_s32(vdupq_n_s32(0), b_vec[c], a_vec);
                            dots[r][c] += vaddvq_s32(dot);
                        }
                    }
                }

                float32x4_t bscale_lo = bscale_lo_pc;
                float32x4_t bscale_hi = bscale_hi_pc;
                if (!scale_per_channel) {
                    load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                      bscale_lo, bscale_hi);
                }

                for (int r = 0; r < r_valid; r++) {
                    float a_scale =
                        a_scales[(size_t)(m + r) * blocks_per_row + qb];
                    float32x4_t acc_lo = vld1q_f32(acc[r]);
                    float32x4_t acc_hi = vld1q_f32(acc[r] + 4);
                    acc_lo =
                        vfmaq_f32(acc_lo, vcvtq_f32_s32(vld1q_s32(dots[r])),
                                  vmulq_n_f32(bscale_lo, a_scale));
                    acc_hi =
                        vfmaq_f32(acc_hi, vcvtq_f32_s32(vld1q_s32(dots[r] + 4)),
                                  vmulq_n_f32(bscale_hi, a_scale));
                    vst1q_f32(acc[r], acc_lo);
                    vst1q_f32(acc[r] + 4, acc_hi);
                }
            }

            for (int r = 0; r < r_valid; r++) {
                float* c_row = C + (m + r) * ldc;
                for (int c = 0; c < c_valid; c++) {
                    c_row[n + c] = acc[r][c];
                }
            }
        }
    }
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_int8_q8dot_neon_4x8_repacked_i8mm_range(
    const int8_t* qA, const float* a_scales, const int8_t* B_repack,
    const float* scales, int group_size, int groups_per_row, float* C, int M,
    int N, int K, int K_padded, int ldc, int m_begin, int m_end, int n_begin,
    int n_end) {
    (void)M;
    (void)N;
    if (group_size <= 0)
        group_size = K;
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);
        bool full_m_tile = (m + 4 <= m_end);

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);
            const int8_t* b_tile = B_repack + (size_t)(n / 8) * blocks_per_row *
                                                  8 * MATMUL_Q8_BLOCK;
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float32x4_t c0_lo = vdupq_n_f32(0.f);
            float32x4_t c0_hi = vdupq_n_f32(0.f);
            float32x4_t c1_lo = vdupq_n_f32(0.f);
            float32x4_t c1_hi = vdupq_n_f32(0.f);
            float32x4_t c2_lo = vdupq_n_f32(0.f);
            float32x4_t c2_hi = vdupq_n_f32(0.f);
            float32x4_t c3_lo = vdupq_n_f32(0.f);
            float32x4_t c3_hi = vdupq_n_f32(0.f);

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int group = w8_scale_group(scale_mode, qb, group_size);
                const int8_t* b_block =
                    b_tile + (size_t)qb * 8 * MATMUL_Q8_BLOCK;

                int32x4_t acc01_01 = vdupq_n_s32(0);
                int32x4_t acc01_23 = vdupq_n_s32(0);
                int32x4_t acc23_01 = vdupq_n_s32(0);
                int32x4_t acc23_23 = vdupq_n_s32(0);
                int32x4_t acc01_45 = vdupq_n_s32(0);
                int32x4_t acc01_67 = vdupq_n_s32(0);
                int32x4_t acc23_45 = vdupq_n_s32(0);
                int32x4_t acc23_67 = vdupq_n_s32(0);

                const int8_t* qa0 =
                    qA + (size_t)(m + 0) * K_padded + qb * MATMUL_Q8_BLOCK;
                const int8_t* qa1 =
                    (m + 1 < m_tile_end)
                        ? qA + (size_t)(m + 1) * K_padded + qb * MATMUL_Q8_BLOCK
                        : qa0;
                const int8_t* qa2 =
                    (m + 2 < m_tile_end)
                        ? qA + (size_t)(m + 2) * K_padded + qb * MATMUL_Q8_BLOCK
                        : qa0;
                const int8_t* qa3 =
                    (m + 3 < m_tile_end)
                        ? qA + (size_t)(m + 3) * K_padded + qb * MATMUL_Q8_BLOCK
                        : qa0;

                for (int off = 0; off < MATMUL_Q8_BLOCK; off += 8) {
                    int8x16_t a01 =
                        vcombine_s8(vld1_s8(qa0 + off), vld1_s8(qa1 + off));
                    int8x16_t a23 =
                        vcombine_s8(vld1_s8(qa2 + off), vld1_s8(qa3 + off));

                    int8x16_t b01 = vcombine_s8(
                        vld1_s8(b_block + 0 * MATMUL_Q8_BLOCK + off),
                        vld1_s8(b_block + 1 * MATMUL_Q8_BLOCK + off));
                    int8x16_t b23 = vcombine_s8(
                        vld1_s8(b_block + 2 * MATMUL_Q8_BLOCK + off),
                        vld1_s8(b_block + 3 * MATMUL_Q8_BLOCK + off));
                    int8x16_t b45 = vcombine_s8(
                        vld1_s8(b_block + 4 * MATMUL_Q8_BLOCK + off),
                        vld1_s8(b_block + 5 * MATMUL_Q8_BLOCK + off));
                    int8x16_t b67 = vcombine_s8(
                        vld1_s8(b_block + 6 * MATMUL_Q8_BLOCK + off),
                        vld1_s8(b_block + 7 * MATMUL_Q8_BLOCK + off));

                    acc01_01 = vmmlaq_s32(acc01_01, a01, b01);
                    acc01_23 = vmmlaq_s32(acc01_23, a01, b23);
                    acc23_01 = vmmlaq_s32(acc23_01, a23, b01);
                    acc23_23 = vmmlaq_s32(acc23_23, a23, b23);
                    acc01_45 = vmmlaq_s32(acc01_45, a01, b45);
                    acc01_67 = vmmlaq_s32(acc01_67, a01, b67);
                    acc23_45 = vmmlaq_s32(acc23_45, a23, b45);
                    acc23_67 = vmmlaq_s32(acc23_67, a23, b67);
                }

                int32x4_t row0_lo = vcombine_s32(vget_low_s32(acc01_01),
                                                 vget_low_s32(acc01_23));
                int32x4_t row0_hi = vcombine_s32(vget_low_s32(acc01_45),
                                                 vget_low_s32(acc01_67));
                int32x4_t row1_lo = vcombine_s32(vget_high_s32(acc01_01),
                                                 vget_high_s32(acc01_23));
                int32x4_t row1_hi = vcombine_s32(vget_high_s32(acc01_45),
                                                 vget_high_s32(acc01_67));
                int32x4_t row2_lo = vcombine_s32(vget_low_s32(acc23_01),
                                                 vget_low_s32(acc23_23));
                int32x4_t row2_hi = vcombine_s32(vget_low_s32(acc23_45),
                                                 vget_low_s32(acc23_67));
                int32x4_t row3_lo = vcombine_s32(vget_high_s32(acc23_01),
                                                 vget_high_s32(acc23_23));
                int32x4_t row3_hi = vcombine_s32(vget_high_s32(acc23_45),
                                                 vget_high_s32(acc23_67));

                float32x4_t bs0 = bscale_lo_pc;
                float32x4_t bs1 = bscale_hi_pc;
                if (!scale_per_channel) {
                    load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                      bs0, bs1);
                }

                if (full_m_tile) {
                    float a0 = a_scales[(size_t)(m + 0) * blocks_per_row + qb];
                    float a1 = a_scales[(size_t)(m + 1) * blocks_per_row + qb];
                    float a2 = a_scales[(size_t)(m + 2) * blocks_per_row + qb];
                    float a3 = a_scales[(size_t)(m + 3) * blocks_per_row + qb];
                    c0_lo = vfmaq_f32(c0_lo, vcvtq_f32_s32(row0_lo),
                                      vmulq_n_f32(bs0, a0));
                    c0_hi = vfmaq_f32(c0_hi, vcvtq_f32_s32(row0_hi),
                                      vmulq_n_f32(bs1, a0));
                    c1_lo = vfmaq_f32(c1_lo, vcvtq_f32_s32(row1_lo),
                                      vmulq_n_f32(bs0, a1));
                    c1_hi = vfmaq_f32(c1_hi, vcvtq_f32_s32(row1_hi),
                                      vmulq_n_f32(bs1, a1));
                    c2_lo = vfmaq_f32(c2_lo, vcvtq_f32_s32(row2_lo),
                                      vmulq_n_f32(bs0, a2));
                    c2_hi = vfmaq_f32(c2_hi, vcvtq_f32_s32(row2_hi),
                                      vmulq_n_f32(bs1, a2));
                    c3_lo = vfmaq_f32(c3_lo, vcvtq_f32_s32(row3_lo),
                                      vmulq_n_f32(bs0, a3));
                    c3_hi = vfmaq_f32(c3_hi, vcvtq_f32_s32(row3_hi),
                                      vmulq_n_f32(bs1, a3));
                } else {
                    auto add_row = [&](int row, int32x4_t lo, int32x4_t hi,
                                       float32x4_t& dst_lo,
                                       float32x4_t& dst_hi) {
                        if (row >= m_tile_end)
                            return;
                        float a_scale =
                            a_scales[(size_t)row * blocks_per_row + qb];
                        dst_lo = vfmaq_f32(dst_lo, vcvtq_f32_s32(lo),
                                           vmulq_n_f32(bs0, a_scale));
                        dst_hi = vfmaq_f32(dst_hi, vcvtq_f32_s32(hi),
                                           vmulq_n_f32(bs1, a_scale));
                    };
                    add_row(m + 0, row0_lo, row0_hi, c0_lo, c0_hi);
                    add_row(m + 1, row1_lo, row1_hi, c1_lo, c1_hi);
                    add_row(m + 2, row2_lo, row2_hi, c2_lo, c2_hi);
                    add_row(m + 3, row3_lo, row3_hi, c3_lo, c3_hi);
                }
            }

            if (full_m_tile && full_n_tile) {
                vst1q_f32(C + (m + 0) * ldc + n, c0_lo);
                vst1q_f32(C + (m + 0) * ldc + n + 4, c0_hi);
                vst1q_f32(C + (m + 1) * ldc + n, c1_lo);
                vst1q_f32(C + (m + 1) * ldc + n + 4, c1_hi);
                vst1q_f32(C + (m + 2) * ldc + n, c2_lo);
                vst1q_f32(C + (m + 2) * ldc + n + 4, c2_hi);
                vst1q_f32(C + (m + 3) * ldc + n, c3_lo);
                vst1q_f32(C + (m + 3) * ldc + n + 4, c3_hi);
            } else {
                auto store_row = [&](int row, float32x4_t lo, float32x4_t hi) {
                    if (row >= m_tile_end)
                        return;
                    float* c_row = C + row * ldc;
                    float tmp[4];
                    vst1q_f32(tmp, lo);
                    for (int c = 0; c < 4 && n + c < n_tile_end; c++)
                        c_row[n + c] = tmp[c];
                    vst1q_f32(tmp, hi);
                    for (int c = 0; c < 4 && n + 4 + c < n_tile_end; c++)
                        c_row[n + 4 + c] = tmp[c];
                };
                store_row(m + 0, c0_lo, c0_hi);
                store_row(m + 1, c1_lo, c1_hi);
                store_row(m + 2, c2_lo, c2_hi);
                store_row(m + 3, c3_lo, c3_hi);
            }
        }
    }
}

static inline void w8_i8mm_8x4_dot_rows(const int8_t* qa_block, int K_padded,
                                        const int8_t* b_block, int col_base,
                                        int32x4_t rows[8]) {
    const int8_t* qa0 = qa_block;
    const int8_t* qa1 = qa0 + K_padded;
    const int8_t* qa2 = qa1 + K_padded;
    const int8_t* qa3 = qa2 + K_padded;
    const int8_t* qa4 = qa3 + K_padded;
    const int8_t* qa5 = qa4 + K_padded;
    const int8_t* qa6 = qa5 + K_padded;
    const int8_t* qa7 = qa6 + K_padded;

    int32x4_t acc01_01 = vdupq_n_s32(0);
    int32x4_t acc01_23 = vdupq_n_s32(0);
    int32x4_t acc23_01 = vdupq_n_s32(0);
    int32x4_t acc23_23 = vdupq_n_s32(0);
    int32x4_t acc45_01 = vdupq_n_s32(0);
    int32x4_t acc45_23 = vdupq_n_s32(0);
    int32x4_t acc67_01 = vdupq_n_s32(0);
    int32x4_t acc67_23 = vdupq_n_s32(0);

    const int8_t* b0 = b_block + (col_base + 0) * MATMUL_Q8_BLOCK;
    const int8_t* b1 = b_block + (col_base + 1) * MATMUL_Q8_BLOCK;
    const int8_t* b2 = b_block + (col_base + 2) * MATMUL_Q8_BLOCK;
    const int8_t* b3 = b_block + (col_base + 3) * MATMUL_Q8_BLOCK;

    for (int off = 0; off < MATMUL_Q8_BLOCK; off += 8) {
        int8x16_t a01 = vcombine_s8(vld1_s8(qa0 + off), vld1_s8(qa1 + off));
        int8x16_t a23 = vcombine_s8(vld1_s8(qa2 + off), vld1_s8(qa3 + off));
        int8x16_t a45 = vcombine_s8(vld1_s8(qa4 + off), vld1_s8(qa5 + off));
        int8x16_t a67 = vcombine_s8(vld1_s8(qa6 + off), vld1_s8(qa7 + off));

        int8x16_t b01 = vcombine_s8(vld1_s8(b0 + off), vld1_s8(b1 + off));
        int8x16_t b23 = vcombine_s8(vld1_s8(b2 + off), vld1_s8(b3 + off));

        acc01_01 = vmmlaq_s32(acc01_01, a01, b01);
        acc01_23 = vmmlaq_s32(acc01_23, a01, b23);
        acc23_01 = vmmlaq_s32(acc23_01, a23, b01);
        acc23_23 = vmmlaq_s32(acc23_23, a23, b23);
        acc45_01 = vmmlaq_s32(acc45_01, a45, b01);
        acc45_23 = vmmlaq_s32(acc45_23, a45, b23);
        acc67_01 = vmmlaq_s32(acc67_01, a67, b01);
        acc67_23 = vmmlaq_s32(acc67_23, a67, b23);
    }

    rows[0] = vcombine_s32(vget_low_s32(acc01_01), vget_low_s32(acc01_23));
    rows[1] = vcombine_s32(vget_high_s32(acc01_01), vget_high_s32(acc01_23));
    rows[2] = vcombine_s32(vget_low_s32(acc23_01), vget_low_s32(acc23_23));
    rows[3] = vcombine_s32(vget_high_s32(acc23_01), vget_high_s32(acc23_23));
    rows[4] = vcombine_s32(vget_low_s32(acc45_01), vget_low_s32(acc45_23));
    rows[5] = vcombine_s32(vget_high_s32(acc45_01), vget_high_s32(acc45_23));
    rows[6] = vcombine_s32(vget_low_s32(acc67_01), vget_low_s32(acc67_23));
    rows[7] = vcombine_s32(vget_high_s32(acc67_01), vget_high_s32(acc67_23));
}

static void matmul_int8_q8dot_neon_8x8_repacked_i8mm_range(
    const int8_t* qA, const float* a_scales, const int8_t* B_repack,
    const float* scales, int group_size, int groups_per_row, float* C, int M,
    int N, int K, int K_padded, int ldc, int m_begin, int m_end, int n_begin,
    int n_end) {
    (void)M;
    (void)N;
    if (group_size <= 0)
        group_size = K;
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 8) {
        if (m + 8 > m_end) {
            matmul_int8_q8dot_neon_4x8_repacked_i8mm_range(
                qA, a_scales, B_repack, scales, group_size, groups_per_row, C,
                M, N, K, K_padded, ldc, m, m_end, n_begin, n_end);
            break;
        }

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);
            const int8_t* b_tile = B_repack + (size_t)(n / 8) * blocks_per_row *
                                                  8 * MATMUL_Q8_BLOCK;
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float32x4_t c_lo[8];
            float32x4_t c_hi[8];
            for (int r = 0; r < 8; r++) {
                c_lo[r] = vdupq_n_f32(0.f);
                c_hi[r] = vdupq_n_f32(0.f);
            }

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int group = w8_scale_group(scale_mode, qb, group_size);
                const int8_t* b_block =
                    b_tile + (size_t)qb * 8 * MATMUL_Q8_BLOCK;
                const int8_t* qa_block =
                    qA + (size_t)m * K_padded + qb * MATMUL_Q8_BLOCK;

                float32x4_t bs0 = bscale_lo_pc;
                float32x4_t bs1 = bscale_hi_pc;
                if (!scale_per_channel) {
                    load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                      bs0, bs1);
                }

                float a0 = a_scales[(size_t)(m + 0) * blocks_per_row + qb];
                float a1 = a_scales[(size_t)(m + 1) * blocks_per_row + qb];
                float a2 = a_scales[(size_t)(m + 2) * blocks_per_row + qb];
                float a3 = a_scales[(size_t)(m + 3) * blocks_per_row + qb];
                float a4 = a_scales[(size_t)(m + 4) * blocks_per_row + qb];
                float a5 = a_scales[(size_t)(m + 5) * blocks_per_row + qb];
                float a6 = a_scales[(size_t)(m + 6) * blocks_per_row + qb];
                float a7 = a_scales[(size_t)(m + 7) * blocks_per_row + qb];

                int32x4_t rows[8];
                w8_i8mm_8x4_dot_rows(qa_block, K_padded, b_block, 0, rows);
                c_lo[0] = vfmaq_f32(c_lo[0], vcvtq_f32_s32(rows[0]),
                                    vmulq_n_f32(bs0, a0));
                c_lo[1] = vfmaq_f32(c_lo[1], vcvtq_f32_s32(rows[1]),
                                    vmulq_n_f32(bs0, a1));
                c_lo[2] = vfmaq_f32(c_lo[2], vcvtq_f32_s32(rows[2]),
                                    vmulq_n_f32(bs0, a2));
                c_lo[3] = vfmaq_f32(c_lo[3], vcvtq_f32_s32(rows[3]),
                                    vmulq_n_f32(bs0, a3));
                c_lo[4] = vfmaq_f32(c_lo[4], vcvtq_f32_s32(rows[4]),
                                    vmulq_n_f32(bs0, a4));
                c_lo[5] = vfmaq_f32(c_lo[5], vcvtq_f32_s32(rows[5]),
                                    vmulq_n_f32(bs0, a5));
                c_lo[6] = vfmaq_f32(c_lo[6], vcvtq_f32_s32(rows[6]),
                                    vmulq_n_f32(bs0, a6));
                c_lo[7] = vfmaq_f32(c_lo[7], vcvtq_f32_s32(rows[7]),
                                    vmulq_n_f32(bs0, a7));

                w8_i8mm_8x4_dot_rows(qa_block, K_padded, b_block, 4, rows);
                c_hi[0] = vfmaq_f32(c_hi[0], vcvtq_f32_s32(rows[0]),
                                    vmulq_n_f32(bs1, a0));
                c_hi[1] = vfmaq_f32(c_hi[1], vcvtq_f32_s32(rows[1]),
                                    vmulq_n_f32(bs1, a1));
                c_hi[2] = vfmaq_f32(c_hi[2], vcvtq_f32_s32(rows[2]),
                                    vmulq_n_f32(bs1, a2));
                c_hi[3] = vfmaq_f32(c_hi[3], vcvtq_f32_s32(rows[3]),
                                    vmulq_n_f32(bs1, a3));
                c_hi[4] = vfmaq_f32(c_hi[4], vcvtq_f32_s32(rows[4]),
                                    vmulq_n_f32(bs1, a4));
                c_hi[5] = vfmaq_f32(c_hi[5], vcvtq_f32_s32(rows[5]),
                                    vmulq_n_f32(bs1, a5));
                c_hi[6] = vfmaq_f32(c_hi[6], vcvtq_f32_s32(rows[6]),
                                    vmulq_n_f32(bs1, a6));
                c_hi[7] = vfmaq_f32(c_hi[7], vcvtq_f32_s32(rows[7]),
                                    vmulq_n_f32(bs1, a7));
            }

            for (int r = 0; r < 8; r++) {
                float* c_row = C + (m + r) * ldc;
                if (full_n_tile) {
                    vst1q_f32(c_row + n, c_lo[r]);
                    vst1q_f32(c_row + n + 4, c_hi[r]);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, c_lo[r]);
                    for (int c = 0; c < 4 && c < c_valid; c++)
                        c_row[n + c] = tmp[c];
                    vst1q_f32(tmp, c_hi[r]);
                    for (int c = 0; c < 4 && c + 4 < c_valid; c++)
                        c_row[n + 4 + c] = tmp[c];
                }
            }
        }
    }
}
#endif
#endif
#endif

void matmul_dispatch_int8(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* thread_pool, Activation act,
                          int act_n_begin, int act_n_len, MatmulTimer& timer) {
    const int M = (int)A.shape[1];
    const int K = (int)A.shape[0];
    const int N = (int)B.shape[0];
    const int lda = (int)(A.stride[1] / sizeof(float));
    const int ldc = (int)(C.stride[1] / sizeof(float));
    const int K_weight = (int)B.shape[1];
    const float* a_ptr = A.ptr<float>();
    float* c_ptr = C.ptr<float>();
    const int8_t* b_int8 = reinterpret_cast<const int8_t*>(B.data);

    const float* scales = B.scales;
    int group_size = (int)B.group_size;
    int groups_per_row = (int)B.groups_per_row;
    bool b_interleaved = B.is_interleaved;
    const int8_t* b_q8_repack =
        reinterpret_cast<const int8_t*>(B.q8_repack_data);
    if (!scales || group_size <= 0 || groups_per_row <= 0) {
        timer.set_shape("int8_invalid_scales", M, N, K, group_size,
                        groups_per_row, b_q8_repack != nullptr, b_interleaved,
                        thread_pool ? thread_pool->num_threads() : 1);
        return;
    }

    constexpr int tile_m = HAS_NEON ? 8 : 1;
    int n_threads = thread_pool ? thread_pool->num_threads() : 1;
    bool shard_by_n = (N > M * 8 && M == 1);
    int chunk_size =
        (M == 1 || N == 1) ? g_matmul_config.gemv_chunk_size : tile_m;
    int total_dim = shard_by_n ? N : M;
    int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
    bool use_parallel = n_threads > 1 && n_chunks > 1;

#if HAS_NEON
#if defined(__ARM_FEATURE_DOTPROD)
    bool can_use_q8_repack = b_q8_repack && (group_size % MATMUL_Q8_BLOCK == 0);
#else
    bool can_use_q8_repack = false;
#endif
    bool use_q8_dot_gemv = can_use_q8_repack || b_interleaved;
    bool use_q8_dot_gemm = can_use_q8_repack;
    bool use_q8_dot_gemv_repack = use_q8_dot_gemv && can_use_q8_repack;
    bool use_q8_dot_gemm_repack = use_q8_dot_gemm && can_use_q8_repack;
    if (M == 1 && use_q8_dot_gemv) {
        const char* path =
            use_q8_dot_gemv_repack ? "q8dot_gemv_repack" : "q8dot_gemv";
        timer.set_shape(path, M, N, K, group_size, groups_per_row,
                        b_q8_repack != nullptr, b_interleaved, n_threads);
        std::vector<int8_t> qA;
        std::vector<float> a_scales;
        int K_padded =
            ((K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK) * MATMUL_Q8_BLOCK;
        quantize_a_q8_blocks(a_ptr, M, K, lda,
                             use_q8_dot_gemv_repack ? K_padded : K, qA,
                             a_scales);
        const int8_t* qA_data = qA.data();
        const float* a_scales_data = a_scales.data();
        if (!use_parallel) {
#if defined(__ARM_FEATURE_DOTPROD)
            if (use_q8_dot_gemv_repack) {
                matmul_int8_q8dot_neon_gemv_repacked_range(
                    qA_data, a_scales_data, b_q8_repack, scales, group_size,
                    groups_per_row, c_ptr, K, K_padded, 0, N);
            } else
#endif
            {
                matmul_int8_q8dot_neon_gemv_range(
                    qA_data, a_scales_data, b_int8, scales, group_size,
                    groups_per_row, c_ptr, K, 0, N);
            }
        } else {
            int n_chunk = std::max(N / (n_threads * 8), 64);
            n_chunk = ((n_chunk + 7) / 8) * 8;
            thread_pool->parallel_for(
                0, N, n_chunk, [&](int, int n_begin, int n_end) {
#if defined(__ARM_FEATURE_DOTPROD)
                    if (use_q8_dot_gemv_repack) {
                        matmul_int8_q8dot_neon_gemv_repacked_range(
                            qA_data, a_scales_data, b_q8_repack, scales,
                            group_size, groups_per_row, c_ptr, K, K_padded,
                            n_begin, n_end);
                    } else
#endif
                    {
                        matmul_int8_q8dot_neon_gemv_range(
                            qA_data, a_scales_data, b_int8, scales, group_size,
                            groups_per_row, c_ptr, K, n_begin, n_end);
                    }
                });
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation_gemv(c_ptr, N, act, act_n_begin, act_n_len);
        }
        return;
    }
    if (use_q8_dot_gemm) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
        bool use_q8_dot_gemm_i8mm = use_q8_dot_gemm_repack;
        bool use_q8_dot_gemm_i8mm_8x8 = use_q8_dot_gemm_i8mm;
#else
        bool use_q8_dot_gemm_i8mm = false;
        bool use_q8_dot_gemm_i8mm_8x8 = false;
#endif
        const char* path =
            use_q8_dot_gemm_i8mm_8x8
                ? "q8dot_gemm_repack_i8mm_8x8"
                : (use_q8_dot_gemm_i8mm
                       ? "q8dot_gemm_repack_i8mm"
                       : (use_q8_dot_gemm_repack ? "q8dot_gemm_repack"
                                                 : "q8dot_gemm"));
        timer.set_shape(path, M, N, K, group_size, groups_per_row,
                        b_q8_repack != nullptr, b_interleaved, n_threads);
        std::vector<int8_t> qA;
        std::vector<float> a_scales;
        int K_padded =
            ((K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK) * MATMUL_Q8_BLOCK;
        quantize_a_q8_blocks(a_ptr, M, K, lda,
                             use_q8_dot_gemm_repack ? K_padded : K, qA,
                             a_scales);
        const int8_t* qA_data = qA.data();
        const float* a_scales_data = a_scales.data();
        auto run_legacy_q8_gemm = [&](int m_begin, int m_end) {
            matmul_int8_q8dot_neon_4x8_range(
                qA_data, a_scales_data, b_int8, scales, group_size,
                groups_per_row, c_ptr, M, N, K, ldc, m_begin, m_end);
        };
#if defined(__ARM_FEATURE_DOTPROD)
        auto run_repacked_q8_gemm = [&](int m_begin, int m_end, int n_begin,
                                        int n_end) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
            if (use_q8_dot_gemm_i8mm_8x8) {
                matmul_int8_q8dot_neon_8x8_repacked_i8mm_range(
                    qA_data, a_scales_data, b_q8_repack, scales, group_size,
                    groups_per_row, c_ptr, M, N, K, K_padded, ldc, m_begin,
                    m_end, n_begin, n_end);
                return;
            }
            if (use_q8_dot_gemm_i8mm) {
                matmul_int8_q8dot_neon_4x8_repacked_i8mm_range(
                    qA_data, a_scales_data, b_q8_repack, scales, group_size,
                    groups_per_row, c_ptr, M, N, K, K_padded, ldc, m_begin,
                    m_end, n_begin, n_end);
                return;
            }
#endif
            matmul_int8_q8dot_neon_4x8_repacked_range(
                qA_data, a_scales_data, b_q8_repack, scales, group_size,
                groups_per_row, c_ptr, M, N, K, K_padded, ldc, m_begin, m_end,
                n_begin, n_end);
        };
#endif
        if (!use_parallel) {
#if defined(__ARM_FEATURE_DOTPROD)
            if (use_q8_dot_gemm_repack) {
                run_repacked_q8_gemm(0, M, 0, N);
            } else
#endif
            {
                run_legacy_q8_gemm(0, M);
            }
        } else {
#if defined(__ARM_FEATURE_DOTPROD)
            if (use_q8_dot_gemm_repack) {
                int n_block = 1024;
                n_block = ((n_block + 7) / 8) * 8;
                if (n_block > N)
                    n_block = ((N + 7) / 8) * 8;
                thread_pool->parallel_for_2d(
                    M, tile_m, N, n_block,
                    [&](int, int m_begin, int m_end, int n_begin, int n_end) {
                        run_repacked_q8_gemm(m_begin, m_end, n_begin, n_end);
                    });
            } else
#endif
            {
                thread_pool->parallel_for(
                    0, M, tile_m, [&](int, int m_begin, int m_end) {
#if defined(__ARM_FEATURE_DOTPROD)
                        if (use_q8_dot_gemm_repack) {
                            run_repacked_q8_gemm(m_begin, m_end, 0, N);
                        } else
#endif
                        {
                            run_legacy_q8_gemm(m_begin, m_end);
                        }
                    });
            }
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act, act_n_begin,
                                    act_n_len);
        }
        return;
    }
    if (M == 1 && b_interleaved) {
        timer.set_shape("native_w8_gemv", M, N, K, group_size, groups_per_row,
                        b_q8_repack != nullptr, b_interleaved, n_threads);
        if (!use_parallel) {
            matmul_int8_neon_gemv_range(a_ptr, b_int8, scales, group_size,
                                        groups_per_row, c_ptr, K, 0, N);
        } else {
            int n_chunk = std::max(N / (n_threads * 8), 64);
            n_chunk = ((n_chunk + 7) / 8) * 8;
            thread_pool->parallel_for(
                0, N, n_chunk, [&](int, int n_begin, int n_end) {
                    matmul_int8_neon_gemv_range(a_ptr, b_int8, scales,
                                                group_size, groups_per_row,
                                                c_ptr, K, n_begin, n_end);
                });
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation_gemv(c_ptr, N, act, act_n_begin, act_n_len);
        }
        return;
    }
    if (b_interleaved) {
        timer.set_shape("native_w8_gemm", M, N, K, group_size, groups_per_row,
                        b_q8_repack != nullptr, b_interleaved, n_threads);
        if (!use_parallel) {
            matmul_int8_neon_4x8_range(a_ptr, b_int8, scales, group_size,
                                       groups_per_row, c_ptr, M, N, K, lda, ldc,
                                       0, M);
        } else {
            thread_pool->parallel_for(
                0, M, tile_m, [&](int, int m_begin, int m_end) {
                    matmul_int8_neon_4x8_range(
                        a_ptr, b_int8, scales, group_size, groups_per_row,
                        c_ptr, M, N, K, lda, ldc, m_begin, m_end);
                });
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act, act_n_begin,
                                    act_n_len);
        }
        return;
    }
#endif

    if (b_interleaved && M == 1) {
        timer.set_shape("q8dot_scalar", M, N, K, group_size, groups_per_row,
                        b_q8_repack != nullptr, b_interleaved, n_threads);
        std::vector<int8_t> qA;
        std::vector<float> a_scales;
        quantize_a_q8_blocks(a_ptr, M, K, lda, K, qA, a_scales);
        const int8_t* qA_data = qA.data();
        const float* a_scales_data = a_scales.data();
        if (!use_parallel) {
            matmul_int8_q8dot_scalar_range(qA_data, a_scales_data, b_int8,
                                           scales, group_size, groups_per_row,
                                           c_ptr, M, N, K, K_weight, ldc, 0, M,
                                           0, N, b_interleaved);
        } else if (shard_by_n) {
            thread_pool->parallel_for(
                0, N, chunk_size, [&](int, int n_begin, int n_end) {
                    matmul_int8_q8dot_scalar_range(
                        qA_data, a_scales_data, b_int8, scales, group_size,
                        groups_per_row, c_ptr, M, N, K, K_weight, ldc, 0, M,
                        n_begin, n_end, b_interleaved);
                });
        } else {
            thread_pool->parallel_for(
                0, M, chunk_size, [&](int, int m_begin, int m_end) {
                    matmul_int8_q8dot_scalar_range(
                        qA_data, a_scales_data, b_int8, scales, group_size,
                        groups_per_row, c_ptr, M, N, K, K_weight, ldc, m_begin,
                        m_end, 0, N, b_interleaved);
                });
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act, act_n_begin,
                                    act_n_len);
        }
        return;
    }

    timer.set_shape("int8_scalar", M, N, K, group_size, groups_per_row,
                    b_q8_repack != nullptr, b_interleaved, n_threads);
    if (!use_parallel) {
        matmul_int8_scalar_range(a_ptr, b_int8, scales, group_size,
                                 groups_per_row, c_ptr, M, N, K, lda, K_weight,
                                 ldc, 0, M, 0, N, b_interleaved);
    } else if (shard_by_n) {
        thread_pool->parallel_for(
            0, N, chunk_size, [&](int, int n_begin, int n_end) {
                matmul_int8_scalar_range(a_ptr, b_int8, scales, group_size,
                                         groups_per_row, c_ptr, M, N, K, lda,
                                         K_weight, ldc, 0, M, n_begin, n_end,
                                         b_interleaved);
            });
    } else {
        thread_pool->parallel_for(
            0, M, chunk_size, [&](int, int m_begin, int m_end) {
                matmul_int8_scalar_range(a_ptr, b_int8, scales, group_size,
                                         groups_per_row, c_ptr, M, N, K, lda,
                                         K_weight, ldc, m_begin, m_end, 0, N,
                                         b_interleaved);
            });
    }
    if (act != Activation::NONE && act_n_len != 0) {
        matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act, act_n_begin,
                                act_n_len);
    }
    return;
}
