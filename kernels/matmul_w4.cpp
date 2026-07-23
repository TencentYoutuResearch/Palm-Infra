#include "kernels/matmul_internal.h"
#include "kernels/threading.h"

#include <algorithm>
#include <cstdint>
#include <vector>

static inline int8_t unpack_int4_signed(uint8_t byte, bool high_nibble) {
    int v = high_nibble ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    if (v >= 8)
        v -= 16;
    return (int8_t)v;
}

#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
static inline void load_int4x32_signed_scaled16(const uint8_t* src,
                                                int8x16_t& even_scaled,
                                                int8x16_t& odd_scaled) {
    uint8x16_t packed = vld1q_u8(src);
    even_scaled = vreinterpretq_s8_u8(vshlq_n_u8(packed, 4));
    odd_scaled = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0xF0)));
}

static inline int32x4_t q4_q8_dot32(int8x16_t q4_even, int8x16_t q4_odd,
                                    int8x16_t qa_even, int8x16_t qa_odd) {
    int32x4_t d = vdupq_n_s32(0);
    d = vdotq_s32(d, q4_even, qa_even);
    d = vdotq_s32(d, q4_odd, qa_odd);
    return d;
}

static inline float32x4_t q4_scaled16_dot_to_f32(int32x4_t dots) {
#if defined(__aarch64__)
    return vcvtq_n_f32_s32(dots, 4);
#else
    return vmulq_n_f32(vcvtq_f32_s32(dots), 1.0f / 16.0f);
#endif
}

static void matmul_int4_q8dot_neon_gemv_range(
    const int8_t* qA, const int8_t* qA_even_pre, const int8_t* qA_odd_pre,
    const float* a_scales, const uint8_t* B, const uint8_t* B_repack,
    const float* scales, int group_size, int groups_per_row, float* C, int K,
    int K_weight, int n_begin, int n_end) {
    if (group_size <= 0)
        group_size = K;
    int row_stride = (K_weight + 1) / 2;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    constexpr int bytes_per_block = MATMUL_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        int c_valid = n_tile_end - n;
        bool full_n_tile = (c_valid == 8);
        float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
        float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
        if (scale_per_channel) {
            load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                              bscale_lo_pc, bscale_hi_pc);
        }

        float32x4_t acc_lo = vdupq_n_f32(0.f);
        float32x4_t acc_hi = vdupq_n_f32(0.f);

        auto run_qblock = [&](int qb, float32x4_t bscale_lo,
                              float32x4_t bscale_hi) {
            int byte_off = qb * (MATMUL_Q8_BLOCK / 2);
            const uint8_t* b_repack_block =
                B_repack ? B_repack + ((size_t)(n / 8) * blocks_per_row + qb) *
                                          8 * bytes_per_block
                         : nullptr;

            int8x16_t qa_even;
            int8x16_t qa_odd;
            if (qA_even_pre && qA_odd_pre) {
                qa_even = vld1q_s8(qA_even_pre + (size_t)qb * 16);
                qa_odd = vld1q_s8(qA_odd_pre + (size_t)qb * 16);
            } else {
                int8x16_t qa0 = vld1q_s8(qA + (size_t)qb * MATMUL_Q8_BLOCK);
                int8x16_t qa1 =
                    vld1q_s8(qA + (size_t)qb * MATMUL_Q8_BLOCK + 16);
                qa_even = vuzp1q_s8(qa0, qa1);
                qa_odd = vuzp2q_s8(qa0, qa1);
            }

            int32x4_t d0 = vdupq_n_s32(0);
            int32x4_t d1 = vdupq_n_s32(0);
            int32x4_t d2 = vdupq_n_s32(0);
            int32x4_t d3 = vdupq_n_s32(0);
            int32x4_t d4 = vdupq_n_s32(0);
            int32x4_t d5 = vdupq_n_s32(0);
            int32x4_t d6 = vdupq_n_s32(0);
            int32x4_t d7 = vdupq_n_s32(0);

            auto dot_col = [&](int c, int32x4_t& d) {
                int8x16_t q4_even, q4_odd;
                const uint8_t* src =
                    b_repack_block
                        ? b_repack_block + (size_t)c * bytes_per_block
                        : B + (size_t)(n + c) * row_stride + byte_off;
                load_int4x32_signed_scaled16(src, q4_even, q4_odd);
                d = q4_q8_dot32(q4_even, q4_odd, qa_even, qa_odd);
            };
            if (full_n_tile) {
                dot_col(0, d0);
                dot_col(1, d1);
                dot_col(2, d2);
                dot_col(3, d3);
                dot_col(4, d4);
                dot_col(5, d5);
                dot_col(6, d6);
                dot_col(7, d7);
            } else {
                if (c_valid > 0)
                    dot_col(0, d0);
                if (c_valid > 1)
                    dot_col(1, d1);
                if (c_valid > 2)
                    dot_col(2, d2);
                if (c_valid > 3)
                    dot_col(3, d3);
                if (c_valid > 4)
                    dot_col(4, d4);
                if (c_valid > 5)
                    dot_col(5, d5);
                if (c_valid > 6)
                    dot_col(6, d6);
                if (c_valid > 7)
                    dot_col(7, d7);
            }

            int32x4_t p01 = vpaddq_s32(d0, d1);
            int32x4_t p23 = vpaddq_s32(d2, d3);
            int32x4_t p45 = vpaddq_s32(d4, d5);
            int32x4_t p67 = vpaddq_s32(d6, d7);
            int32x4_t dots_lo = vpaddq_s32(p01, p23);
            int32x4_t dots_hi = vpaddq_s32(p45, p67);

            float a_scale = a_scales[qb];
            acc_lo = vfmaq_f32(acc_lo, q4_scaled16_dot_to_f32(dots_lo),
                               vmulq_n_f32(bscale_lo, a_scale));
            acc_hi = vfmaq_f32(acc_hi, q4_scaled16_dot_to_f32(dots_hi),
                               vmulq_n_f32(bscale_hi, a_scale));
        };

        if (scale_per_channel) {
            for (int qb = 0; qb < blocks_per_row; qb++) {
                run_qblock(qb, bscale_lo_pc, bscale_hi_pc);
            }
        } else if (scale_mode == W8ScaleMode::PerGroup) {
            int qblocks_per_group = std::max(1, group_size / MATMUL_Q8_BLOCK);
            for (int group = 0; group < groups_per_row; group++) {
                float32x4_t bscale_lo;
                float32x4_t bscale_hi;
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                  bscale_lo, bscale_hi);
                int qb_begin = group * qblocks_per_group;
                int qb_end =
                    std::min(qb_begin + qblocks_per_group, blocks_per_row);
                for (int qb = qb_begin; qb < qb_end; qb++) {
                    run_qblock(qb, bscale_lo, bscale_hi);
                }
            }
        } else {
            for (int qb = 0; qb < blocks_per_row; qb++) {
                float32x4_t bscale_lo;
                float32x4_t bscale_hi;
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, qb,
                                  bscale_lo, bscale_hi);
                run_qblock(qb, bscale_lo, bscale_hi);
            }
        }

        if (full_n_tile) {
            vst1q_f32(C + n, acc_lo);
            vst1q_f32(C + n + 4, acc_hi);
        } else {
            float tmp[4];
            vst1q_f32(tmp, acc_lo);
            for (int c = 0; c < 4 && c < c_valid; c++)
                C[n + c] = tmp[c];
            vst1q_f32(tmp, acc_hi);
            for (int c = 0; c < 4 && c + 4 < c_valid; c++)
                C[n + 4 + c] = tmp[c];
        }
    }
}

static void matmul_int4_q8dot_neon_gemv_g128_range(
    const int8_t* qA_even_pre, const int8_t* qA_odd_pre, const float* a_scales,
    const Q4B8G128Block* B_g128, float* C, int K, int n_begin, int n_end) {
    int g128_per_row = K / 128;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        int c_valid = n_tile_end - n;

        float32x4_t acc_lo = vdupq_n_f32(0.f);
        float32x4_t acc_hi = vdupq_n_f32(0.f);
        const Q4B8G128Block* b_tile = B_g128 + (size_t)(n / 8) * g128_per_row;

        for (int g = 0; g < g128_per_row; g++) {
            const Q4B8G128Block& b_group = b_tile[g];
            float32x4_t bscale_lo = vld1q_f32(b_group.scales);
            float32x4_t bscale_hi = vld1q_f32(b_group.scales + 4);

            for (int qgi = 0; qgi < 4; qgi++) {
                int qb = g * 4 + qgi;
                int8x16_t qa_even = vld1q_s8(qA_even_pre + (size_t)qb * 16);
                int8x16_t qa_odd = vld1q_s8(qA_odd_pre + (size_t)qb * 16);

                int32x4_t d0 = vdupq_n_s32(0);
                int32x4_t d1 = vdupq_n_s32(0);
                int32x4_t d2 = vdupq_n_s32(0);
                int32x4_t d3 = vdupq_n_s32(0);
                int32x4_t d4 = vdupq_n_s32(0);
                int32x4_t d5 = vdupq_n_s32(0);
                int32x4_t d6 = vdupq_n_s32(0);
                int32x4_t d7 = vdupq_n_s32(0);

                auto dot_col = [&](int c, int32x4_t& d) {
                    int8x16_t q4_even;
                    int8x16_t q4_odd;
                    load_int4x32_signed_scaled16(b_group.q[qgi][c], q4_even,
                                                 q4_odd);
                    d = q4_q8_dot32(q4_even, q4_odd, qa_even, qa_odd);
                };

                if (c_valid > 0)
                    dot_col(0, d0);
                if (c_valid > 1)
                    dot_col(1, d1);
                if (c_valid > 2)
                    dot_col(2, d2);
                if (c_valid > 3)
                    dot_col(3, d3);
                if (c_valid > 4)
                    dot_col(4, d4);
                if (c_valid > 5)
                    dot_col(5, d5);
                if (c_valid > 6)
                    dot_col(6, d6);
                if (c_valid > 7)
                    dot_col(7, d7);

                int32x4_t p01 = vpaddq_s32(d0, d1);
                int32x4_t p23 = vpaddq_s32(d2, d3);
                int32x4_t p45 = vpaddq_s32(d4, d5);
                int32x4_t p67 = vpaddq_s32(d6, d7);
                int32x4_t dots_lo = vpaddq_s32(p01, p23);
                int32x4_t dots_hi = vpaddq_s32(p45, p67);

                float a_scale = a_scales[qb];
                acc_lo = vfmaq_f32(acc_lo, q4_scaled16_dot_to_f32(dots_lo),
                                   vmulq_n_f32(bscale_lo, a_scale));
                acc_hi = vfmaq_f32(acc_hi, q4_scaled16_dot_to_f32(dots_hi),
                                   vmulq_n_f32(bscale_hi, a_scale));
            }
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

static void matmul_int4_q8dot_neon_4x8_range(
    const int8_t* qA, const float* a_scales, const uint8_t* B,
    const uint8_t* B_repack, const float* scales, int group_size,
    int groups_per_row, float* C, int M, int N, int K, int K_padded,
    int K_weight, int ldc, int m_begin, int m_end) {
    (void)M;
    if (group_size <= 0)
        group_size = K;
    int row_stride = (K_weight + 1) / 2;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    constexpr int bytes_per_block = MATMUL_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);
        int r_valid = m_tile_end - m;

        for (int n = 0; n < N; n += 8) {
            int n_tile_end = std::min(n + 8, N);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float acc[4][8] = {};

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int group = w8_scale_group(scale_mode, qb, group_size);
                int byte_off = qb * (MATMUL_Q8_BLOCK / 2);
                const uint8_t* b_repack_block =
                    B_repack
                        ? B_repack + ((size_t)(n / 8) * blocks_per_row + qb) *
                                         8 * bytes_per_block
                        : nullptr;

                int8x16_t q4_even[8];
                int8x16_t q4_odd[8];
                int c_load_end = full_n_tile ? 8 : c_valid;
                for (int c = 0; c < c_load_end; c++) {
                    const uint8_t* src =
                        b_repack_block
                            ? b_repack_block + (size_t)c * bytes_per_block
                            : B + (size_t)(n + c) * row_stride + byte_off;
                    load_int4x32_signed_scaled16(src, q4_even[c], q4_odd[c]);
                }

                int32_t dots[4][8] = {};
                for (int r = 0; r < r_valid; r++) {
                    const int8_t* qa =
                        qA + (size_t)(m + r) * K_padded + qb * MATMUL_Q8_BLOCK;
                    int8x16_t qa0 = vld1q_s8(qa);
                    int8x16_t qa1 = vld1q_s8(qa + 16);
                    int8x16_t qa_even = vuzp1q_s8(qa0, qa1);
                    int8x16_t qa_odd = vuzp2q_s8(qa0, qa1);
                    for (int c = 0; c < c_valid; c++) {
                        int32x4_t d =
                            q4_q8_dot32(q4_even[c], q4_odd[c], qa_even, qa_odd);
                        dots[r][c] = vaddvq_s32(d);
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
                    acc_lo = vfmaq_f32(
                        acc_lo, q4_scaled16_dot_to_f32(vld1q_s32(dots[r])),
                        vmulq_n_f32(bscale_lo, a_scale));
                    acc_hi = vfmaq_f32(
                        acc_hi, q4_scaled16_dot_to_f32(vld1q_s32(dots[r] + 4)),
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

template <bool PackedA4>
static void matmul_int4_q8dot_neon_8x8_range(
    const int8_t* qA, const float* a_scales, const uint8_t* B,
    const uint8_t* B_repack, const float* scales, int group_size,
    int groups_per_row, float* C, int M, int N, int K, int K_padded,
    int K_weight, int ldc, int m_begin, int m_end, int n_begin, int n_end,
    const Q8A4Block* qA4) {
    (void)M;
    (void)N;
    if (group_size <= 0)
        group_size = K;
    int row_stride = (K_weight + 1) / 2;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    constexpr int bytes_per_block = MATMUL_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 8) {
        int m_tile_end = std::min(m + 8, m_end);
        int r_valid = m_tile_end - m;

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float32x4_t acc_lo[8];
            float32x4_t acc_hi[8];
            for (int r = 0; r < r_valid; r++) {
                acc_lo[r] = vdupq_n_f32(0.f);
                acc_hi[r] = vdupq_n_f32(0.f);
            }

            int qblocks_per_group =
                (scale_mode == W8ScaleMode::PerGroup)
                    ? std::max(1, group_size / MATMUL_Q8_BLOCK)
                    : 1;
            int cached_group = -1;
            float32x4_t cached_bscale_lo = vdupq_n_f32(0.f);
            float32x4_t cached_bscale_hi = vdupq_n_f32(0.f);

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int byte_off = qb * (MATMUL_Q8_BLOCK / 2);
                const uint8_t* b_repack_block =
                    B_repack
                        ? B_repack + ((size_t)(n / 8) * blocks_per_row + qb) *
                                         8 * bytes_per_block
                        : nullptr;

                int8x16_t q4_even[8];
                int8x16_t q4_odd[8];
                int c_load_end = full_n_tile ? 8 : c_valid;
                for (int c = 0; c < c_load_end; c++) {
                    const uint8_t* src =
                        b_repack_block
                            ? b_repack_block + (size_t)c * bytes_per_block
                            : B + (size_t)(n + c) * row_stride + byte_off;
                    load_int4x32_signed_scaled16(src, q4_even[c], q4_odd[c]);
                }

                float32x4_t bscale_lo = bscale_lo_pc;
                float32x4_t bscale_hi = bscale_hi_pc;
                if (!scale_per_channel) {
                    int group = (scale_mode == W8ScaleMode::PerGroup)
                                    ? qb / qblocks_per_group
                                    : qb;
                    if (group != cached_group) {
                        load_w8_b_scales8(scales, n, c_valid, groups_per_row,
                                          group, cached_bscale_lo,
                                          cached_bscale_hi);
                        cached_group = group;
                    }
                    bscale_lo = cached_bscale_lo;
                    bscale_hi = cached_bscale_hi;
                }

                for (int r = 0; r < r_valid; r++) {
                    int8x16_t qa_even;
                    int8x16_t qa_odd;
                    float a_scale;
                    if constexpr (PackedA4) {
                        int row = m + r;
                        const Q8A4Block& a_block =
                            qA4[(size_t)(row / 4) * blocks_per_row + qb];
                        int ar = row & 3;
                        qa_even = vld1q_s8(a_block.even[ar]);
                        qa_odd = vld1q_s8(a_block.odd[ar]);
                        a_scale = a_block.scales[ar];
                    } else {
                        const int8_t* qa = qA + (size_t)(m + r) * K_padded +
                                           qb * MATMUL_Q8_BLOCK;
                        int8x16_t qa0 = vld1q_s8(qa);
                        int8x16_t qa1 = vld1q_s8(qa + 16);
                        qa_even = vuzp1q_s8(qa0, qa1);
                        qa_odd = vuzp2q_s8(qa0, qa1);
                        a_scale =
                            a_scales[(size_t)(m + r) * blocks_per_row + qb];
                    }
                    int32x4_t d0 = vdupq_n_s32(0);
                    int32x4_t d1 = vdupq_n_s32(0);
                    int32x4_t d2 = vdupq_n_s32(0);
                    int32x4_t d3 = vdupq_n_s32(0);
                    int32x4_t d4 = vdupq_n_s32(0);
                    int32x4_t d5 = vdupq_n_s32(0);
                    int32x4_t d6 = vdupq_n_s32(0);
                    int32x4_t d7 = vdupq_n_s32(0);
                    if (full_n_tile) {
                        d0 =
                            q4_q8_dot32(q4_even[0], q4_odd[0], qa_even, qa_odd);
                        d1 =
                            q4_q8_dot32(q4_even[1], q4_odd[1], qa_even, qa_odd);
                        d2 =
                            q4_q8_dot32(q4_even[2], q4_odd[2], qa_even, qa_odd);
                        d3 =
                            q4_q8_dot32(q4_even[3], q4_odd[3], qa_even, qa_odd);
                        d4 =
                            q4_q8_dot32(q4_even[4], q4_odd[4], qa_even, qa_odd);
                        d5 =
                            q4_q8_dot32(q4_even[5], q4_odd[5], qa_even, qa_odd);
                        d6 =
                            q4_q8_dot32(q4_even[6], q4_odd[6], qa_even, qa_odd);
                        d7 =
                            q4_q8_dot32(q4_even[7], q4_odd[7], qa_even, qa_odd);
                    } else {
                        if (c_valid > 0)
                            d0 = q4_q8_dot32(q4_even[0], q4_odd[0], qa_even,
                                             qa_odd);
                        if (c_valid > 1)
                            d1 = q4_q8_dot32(q4_even[1], q4_odd[1], qa_even,
                                             qa_odd);
                        if (c_valid > 2)
                            d2 = q4_q8_dot32(q4_even[2], q4_odd[2], qa_even,
                                             qa_odd);
                        if (c_valid > 3)
                            d3 = q4_q8_dot32(q4_even[3], q4_odd[3], qa_even,
                                             qa_odd);
                        if (c_valid > 4)
                            d4 = q4_q8_dot32(q4_even[4], q4_odd[4], qa_even,
                                             qa_odd);
                        if (c_valid > 5)
                            d5 = q4_q8_dot32(q4_even[5], q4_odd[5], qa_even,
                                             qa_odd);
                        if (c_valid > 6)
                            d6 = q4_q8_dot32(q4_even[6], q4_odd[6], qa_even,
                                             qa_odd);
                        if (c_valid > 7)
                            d7 = q4_q8_dot32(q4_even[7], q4_odd[7], qa_even,
                                             qa_odd);
                    }

                    int32x4_t p01 = vpaddq_s32(d0, d1);
                    int32x4_t p23 = vpaddq_s32(d2, d3);
                    int32x4_t p45 = vpaddq_s32(d4, d5);
                    int32x4_t p67 = vpaddq_s32(d6, d7);
                    int32x4_t dots_lo = vpaddq_s32(p01, p23);
                    int32x4_t dots_hi = vpaddq_s32(p45, p67);

                    acc_lo[r] =
                        vfmaq_f32(acc_lo[r], q4_scaled16_dot_to_f32(dots_lo),
                                  vmulq_n_f32(bscale_lo, a_scale));
                    acc_hi[r] =
                        vfmaq_f32(acc_hi[r], q4_scaled16_dot_to_f32(dots_hi),
                                  vmulq_n_f32(bscale_hi, a_scale));
                }
            }

            for (int r = 0; r < r_valid; r++) {
                float* c_row = C + (m + r) * ldc;
                if (full_n_tile) {
                    vst1q_f32(c_row + n, acc_lo[r]);
                    vst1q_f32(c_row + n + 4, acc_hi[r]);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, acc_lo[r]);
                    for (int c = 0; c < 4 && c < c_valid; c++)
                        c_row[n + c] = tmp[c];
                    vst1q_f32(tmp, acc_hi[r]);
                    for (int c = 0; c < 4 && c + 4 < c_valid; c++)
                        c_row[n + 4 + c] = tmp[c];
                }
            }
        }
    }
}

template <bool PackedA4>
static void matmul_int4_q8dot_neon_8x8_g128packed_range(
    const int8_t* qA, const float* a_scales, const Q4B8G128Block* B_g128,
    float* C, int M, int N, int K, int K_padded, int ldc, int m_begin,
    int m_end, int n_begin, int n_end, const Q8A4Block* qA4) {
    (void)M;
    (void)N;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    int g128_per_row = K / 128;

    for (int m = m_begin; m < m_end; m += 8) {
        int m_tile_end = std::min(m + 8, m_end);
        int r_valid = m_tile_end - m;

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);

            float32x4_t acc_lo[8];
            float32x4_t acc_hi[8];
            for (int r = 0; r < r_valid; r++) {
                acc_lo[r] = vdupq_n_f32(0.f);
                acc_hi[r] = vdupq_n_f32(0.f);
            }

            const Q4B8G128Block* b_tile =
                B_g128 + (size_t)(n / 8) * g128_per_row;
            for (int g = 0; g < g128_per_row; g++) {
                const Q4B8G128Block& b_group = b_tile[g];
                float32x4_t bscale_lo = vld1q_f32(b_group.scales);
                float32x4_t bscale_hi = vld1q_f32(b_group.scales + 4);

                for (int qgi = 0; qgi < 4; qgi++) {
                    int qb = g * 4 + qgi;
                    int8x16_t q4_even[8];
                    int8x16_t q4_odd[8];
                    for (int c = 0; c < 8; c++) {
                        load_int4x32_signed_scaled16(b_group.q[qgi][c],
                                                     q4_even[c], q4_odd[c]);
                    }

                    for (int r = 0; r < r_valid; r++) {
                        int8x16_t qa_even;
                        int8x16_t qa_odd;
                        float a_scale;
                        if constexpr (PackedA4) {
                            int row = m + r;
                            const Q8A4Block& a_block =
                                qA4[(size_t)(row / 4) * blocks_per_row + qb];
                            int ar = row & 3;
                            qa_even = vld1q_s8(a_block.even[ar]);
                            qa_odd = vld1q_s8(a_block.odd[ar]);
                            a_scale = a_block.scales[ar];
                        } else {
                            const int8_t* qa = qA + (size_t)(m + r) * K_padded +
                                               qb * MATMUL_Q8_BLOCK;
                            int8x16_t qa0 = vld1q_s8(qa);
                            int8x16_t qa1 = vld1q_s8(qa + 16);
                            qa_even = vuzp1q_s8(qa0, qa1);
                            qa_odd = vuzp2q_s8(qa0, qa1);
                            a_scale =
                                a_scales[(size_t)(m + r) * blocks_per_row + qb];
                        }

                        int32x4_t d0 =
                            q4_q8_dot32(q4_even[0], q4_odd[0], qa_even, qa_odd);
                        int32x4_t d1 =
                            q4_q8_dot32(q4_even[1], q4_odd[1], qa_even, qa_odd);
                        int32x4_t d2 =
                            q4_q8_dot32(q4_even[2], q4_odd[2], qa_even, qa_odd);
                        int32x4_t d3 =
                            q4_q8_dot32(q4_even[3], q4_odd[3], qa_even, qa_odd);
                        int32x4_t d4 =
                            q4_q8_dot32(q4_even[4], q4_odd[4], qa_even, qa_odd);
                        int32x4_t d5 =
                            q4_q8_dot32(q4_even[5], q4_odd[5], qa_even, qa_odd);
                        int32x4_t d6 =
                            q4_q8_dot32(q4_even[6], q4_odd[6], qa_even, qa_odd);
                        int32x4_t d7 =
                            q4_q8_dot32(q4_even[7], q4_odd[7], qa_even, qa_odd);

                        int32x4_t p01 = vpaddq_s32(d0, d1);
                        int32x4_t p23 = vpaddq_s32(d2, d3);
                        int32x4_t p45 = vpaddq_s32(d4, d5);
                        int32x4_t p67 = vpaddq_s32(d6, d7);
                        int32x4_t dots_lo = vpaddq_s32(p01, p23);
                        int32x4_t dots_hi = vpaddq_s32(p45, p67);

                        acc_lo[r] = vfmaq_f32(acc_lo[r],
                                              q4_scaled16_dot_to_f32(dots_lo),
                                              vmulq_n_f32(bscale_lo, a_scale));
                        acc_hi[r] = vfmaq_f32(acc_hi[r],
                                              q4_scaled16_dot_to_f32(dots_hi),
                                              vmulq_n_f32(bscale_hi, a_scale));
                    }
                }
            }

            for (int r = 0; r < r_valid; r++) {
                float* c_row = C + (m + r) * ldc;
                if (full_n_tile) {
                    vst1q_f32(c_row + n, acc_lo[r]);
                    vst1q_f32(c_row + n + 4, acc_hi[r]);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, acc_lo[r]);
                    for (int c = 0; c < 4 && c < c_valid; c++)
                        c_row[n + c] = tmp[c];
                    vst1q_f32(tmp, acc_hi[r]);
                    for (int c = 0; c < 4 && c + 4 < c_valid; c++)
                        c_row[n + 4 + c] = tmp[c];
                }
            }
        }
    }
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
template <bool PackedA4>
static void matmul_int4_q8dot_neon_4x8_repacked_i8mm_range(
    const int8_t* qA, const float* a_scales, const uint8_t* B_repack,
    const float* scales, int group_size, int groups_per_row, float* C, int M,
    int N, int K, int K_padded, int ldc, int m_begin, int m_end,
    const Q8A4Block* qA4) {
    (void)M;
    if (group_size <= 0)
        group_size = K;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    constexpr int bytes_per_block = MATMUL_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);

        for (int n = 0; n < N; n += 8) {
            int n_tile_end = std::min(n + 8, N);
            int c_valid = n_tile_end - n;
            const uint8_t* b_tile = B_repack + (size_t)(n / 8) *
                                                   blocks_per_row * 8 *
                                                   bytes_per_block;

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
                const uint8_t* b_block =
                    b_tile + (size_t)qb * 8 * bytes_per_block;

                int8x16_t b_even[8];
                int8x16_t b_odd[8];
                for (int c = 0; c < 8; c++) {
                    if (c < c_valid) {
                        load_int4x32_signed_scaled16(
                            b_block + (size_t)c * bytes_per_block, b_even[c],
                            b_odd[c]);
                    } else {
                        b_even[c] = vdupq_n_s8(0);
                        b_odd[c] = vdupq_n_s8(0);
                    }
                }

                int32x4_t acc01_01 = vdupq_n_s32(0);
                int32x4_t acc01_23 = vdupq_n_s32(0);
                int32x4_t acc23_01 = vdupq_n_s32(0);
                int32x4_t acc23_23 = vdupq_n_s32(0);
                int32x4_t acc01_45 = vdupq_n_s32(0);
                int32x4_t acc01_67 = vdupq_n_s32(0);
                int32x4_t acc23_45 = vdupq_n_s32(0);
                int32x4_t acc23_67 = vdupq_n_s32(0);

                auto load_even_odd = [&](int row, int8x16_t& even,
                                         int8x16_t& odd, float& a_scale) {
                    int row_load = (row < m_tile_end) ? row : m;
                    if constexpr (PackedA4) {
                        const Q8A4Block& a_block =
                            qA4[(size_t)(row_load / 4) * blocks_per_row + qb];
                        int ar = row_load & 3;
                        even = vld1q_s8(a_block.even[ar]);
                        odd = vld1q_s8(a_block.odd[ar]);
                        a_scale = a_block.scales[ar];
                    } else {
                        const int8_t* qa = qA + (size_t)row_load * K_padded +
                                           qb * MATMUL_Q8_BLOCK;
                        int8x16_t qa0v = vld1q_s8(qa);
                        int8x16_t qa1v = vld1q_s8(qa + 16);
                        even = vuzp1q_s8(qa0v, qa1v);
                        odd = vuzp2q_s8(qa0v, qa1v);
                        a_scale =
                            a_scales[(size_t)row_load * blocks_per_row + qb];
                    }
                };

                int8x16_t a0_even, a0_odd, a1_even, a1_odd;
                int8x16_t a2_even, a2_odd, a3_even, a3_odd;
                float a_scale0, a_scale1, a_scale2, a_scale3;
                load_even_odd(m + 0, a0_even, a0_odd, a_scale0);
                load_even_odd(m + 1, a1_even, a1_odd, a_scale1);
                load_even_odd(m + 2, a2_even, a2_odd, a_scale2);
                load_even_odd(m + 3, a3_even, a3_odd, a_scale3);

                auto run_half = [&](bool high_half, bool odd_lane) {
                    auto half8 = [&](int8x16_t v) -> int8x8_t {
                        return high_half ? vget_high_s8(v) : vget_low_s8(v);
                    };
                    int8x16_t a01 =
                        vcombine_s8(half8(odd_lane ? a0_odd : a0_even),
                                    half8(odd_lane ? a1_odd : a1_even));
                    int8x16_t a23 =
                        vcombine_s8(half8(odd_lane ? a2_odd : a2_even),
                                    half8(odd_lane ? a3_odd : a3_even));

                    const int8x16_t* b = odd_lane ? b_odd : b_even;
                    int8x16_t b01 = vcombine_s8(half8(b[0]), half8(b[1]));
                    int8x16_t b23 = vcombine_s8(half8(b[2]), half8(b[3]));
                    int8x16_t b45 = vcombine_s8(half8(b[4]), half8(b[5]));
                    int8x16_t b67 = vcombine_s8(half8(b[6]), half8(b[7]));

                    acc01_01 = vmmlaq_s32(acc01_01, a01, b01);
                    acc01_23 = vmmlaq_s32(acc01_23, a01, b23);
                    acc23_01 = vmmlaq_s32(acc23_01, a23, b01);
                    acc23_23 = vmmlaq_s32(acc23_23, a23, b23);
                    acc01_45 = vmmlaq_s32(acc01_45, a01, b45);
                    acc01_67 = vmmlaq_s32(acc01_67, a01, b67);
                    acc23_45 = vmmlaq_s32(acc23_45, a23, b45);
                    acc23_67 = vmmlaq_s32(acc23_67, a23, b67);
                };

                run_half(false, false);
                run_half(true, false);
                run_half(false, true);
                run_half(true, true);

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

                auto add_row = [&](int row, int32x4_t lo, int32x4_t hi,
                                   float32x4_t& dst_lo, float32x4_t& dst_hi,
                                   float a_scale) {
                    if (row >= m_tile_end)
                        return;
                    dst_lo = vfmaq_f32(dst_lo, q4_scaled16_dot_to_f32(lo),
                                       vmulq_n_f32(bs0, a_scale));
                    dst_hi = vfmaq_f32(dst_hi, q4_scaled16_dot_to_f32(hi),
                                       vmulq_n_f32(bs1, a_scale));
                };
                add_row(m + 0, row0_lo, row0_hi, c0_lo, c0_hi, a_scale0);
                add_row(m + 1, row1_lo, row1_hi, c1_lo, c1_hi, a_scale1);
                add_row(m + 2, row2_lo, row2_hi, c2_lo, c2_hi, a_scale2);
                add_row(m + 3, row3_lo, row3_hi, c3_lo, c3_hi, a_scale3);
            }

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

#endif
#endif

static void matmul_int4_scalar_range(const float* A, const uint8_t* B,
                                     const float* scales, int group_size,
                                     int groups_per_row, float* C, int M, int N,
                                     int K, int lda, int K_weight, int ldc,
                                     int m_begin, int m_end, int n_begin,
                                     int n_end) {
    (void)M;
    (void)N;
    if (group_size <= 0)
        group_size = K;
    if (groups_per_row <= 0)
        groups_per_row = 1;
    int row_stride = (K_weight + 1) / 2;

    for (int m = m_begin; m < m_end; m++) {
        float* c_row = C + m * ldc;
        for (int n = n_begin; n < n_end; n++) {
            const uint8_t* b_row = B + (size_t)n * row_stride;
            const float* s_row = scales + n * groups_per_row;
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                uint8_t byte = b_row[k >> 1];
                int8_t q = unpack_int4_signed(byte, (k & 1) != 0);
                int g = k / group_size;
                sum += A[k + m * lda] * ((float)q * s_row[g]);
            }
            c_row[n] = sum;
        }
    }
}

void matmul_dispatch_int4(const Tensor& A, const Tensor& B, Tensor& C,
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
    const uint8_t* b_int4 = reinterpret_cast<const uint8_t*>(B.data);

    const float* scales = B.scales;
    int group_size = (int)B.group_size;
    int groups_per_row = (int)B.groups_per_row;
    const uint8_t* b_q4_repack =
        reinterpret_cast<const uint8_t*>(B.q4_repack_data);
    const auto* b_q4_g128 =
        reinterpret_cast<const Q4B8G128Block*>(B.q4_g128_data);
    int n_threads = thread_pool ? thread_pool->num_threads() : 1;
    if (!scales || group_size <= 0 || groups_per_row <= 0) {
        timer.set_shape("int4_invalid_scales", M, N, K, group_size,
                        groups_per_row, false, false, n_threads);
        return;
    }

    constexpr int tile_m = HAS_NEON ? 8 : 1;
    bool shard_by_n = (N > M * 8 && M == 1);
    int chunk_size =
        (M == 1 || N == 1) ? g_matmul_config.gemv_chunk_size : tile_m;
    int total_dim = shard_by_n ? N : M;
    int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
    bool use_parallel = n_threads > 1 && n_chunks > 1;

#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    bool has_direct_q4_g128 = B.is_q4_g128_packed && b_q4_g128;
    bool can_use_q4_dot =
        (B.is_q4_repacked || has_direct_q4_g128 || b_int4 != nullptr) &&
        (K % MATMUL_Q8_BLOCK == 0) && (group_size % MATMUL_Q8_BLOCK == 0);
    bool use_q4_repack = can_use_q4_dot && b_q4_repack;
    bool can_use_q4_bg128 =
        can_use_q4_dot && b_q4_g128 && group_size == 128 && (K % 128 == 0);
    if (M == 1 && can_use_q4_dot) {
        bool use_q4_gemv_bg128 = can_use_q4_bg128;
        const char* path =
            use_q4_gemv_bg128
                ? "q4dot_gemv_bg128"
                : (use_q4_repack ? "q4dot_gemv_repack" : "q4dot_gemv");
        timer.set_shape(path, M, N, K, group_size, groups_per_row,
                        use_q4_gemv_bg128 || use_q4_repack, false, n_threads);
        static thread_local Q4GemvScratch scratch;
        quantize_a_q8_blocks_even_odd(a_ptr, K, scratch.qA_even, scratch.qA_odd,
                                      scratch.a_scales);
        const int8_t* qA_even_data = scratch.qA_even.data();
        const int8_t* qA_odd_data = scratch.qA_odd.data();
        const float* a_scales_data = scratch.a_scales.data();
        if (!use_parallel) {
            if (use_q4_gemv_bg128) {
                matmul_int4_q8dot_neon_gemv_g128_range(
                    qA_even_data, qA_odd_data, a_scales_data, b_q4_g128, c_ptr,
                    K, 0, N);
            } else {
                matmul_int4_q8dot_neon_gemv_range(
                    nullptr, qA_even_data, qA_odd_data, a_scales_data, b_int4,
                    use_q4_repack ? b_q4_repack : nullptr, scales, group_size,
                    groups_per_row, c_ptr, K, K_weight, 0, N);
            }
        } else {
            int n_chunk = std::max(N / (n_threads * 8), 64);
            n_chunk = ((n_chunk + 7) / 8) * 8;
            thread_pool->parallel_for(
                0, N, n_chunk, [&](int, int n_begin, int n_end) {
                    if (use_q4_gemv_bg128) {
                        matmul_int4_q8dot_neon_gemv_g128_range(
                            qA_even_data, qA_odd_data, a_scales_data, b_q4_g128,
                            c_ptr, K, n_begin, n_end);
                    } else {
                        matmul_int4_q8dot_neon_gemv_range(
                            nullptr, qA_even_data, qA_odd_data, a_scales_data,
                            b_int4, use_q4_repack ? b_q4_repack : nullptr,
                            scales, group_size, groups_per_row, c_ptr, K,
                            K_weight, n_begin, n_end);
                    }
                });
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation_gemv(c_ptr, N, act, act_n_begin, act_n_len);
        }
        return;
    }
    if (can_use_q4_dot) {
        constexpr bool force_4x8 = false;
#if defined(__ARM_FEATURE_MATMUL_INT8)
        constexpr bool use_q4_dot_gemm_i8mm = false;
#else
        constexpr bool use_q4_dot_gemm_i8mm = false;
#endif
        constexpr bool use_q4_dot_gemm_a4 = true;
        bool use_q4_dot_gemm_bg128 =
            can_use_q4_bg128 && !use_q4_dot_gemm_i8mm && !force_4x8;
        const char* path =
            use_q4_dot_gemm_i8mm
                ? (use_q4_dot_gemm_a4 ? "q4dot_gemm_repack_i8mm_a4"
                                      : "q4dot_gemm_repack_i8mm")
                : (use_q4_dot_gemm_bg128
                       ? (use_q4_dot_gemm_a4 ? "q4dot_gemm_bg128_a4"
                                             : "q4dot_gemm_bg128")
                       : (use_q4_dot_gemm_a4
                              ? (use_q4_repack ? "q4dot_gemm_repack_a4"
                                               : "q4dot_gemm_a4")
                              : (use_q4_repack ? "q4dot_gemm_repack"
                                               : "q4dot_gemm")));
        timer.set_shape(path, M, N, K, group_size, groups_per_row,
                        use_q4_repack || use_q4_dot_gemm_bg128, false,
                        n_threads);
        std::vector<float> a_scales;
        std::vector<int8_t> qA;
        std::vector<Q8A4Block> qA4;
        if (use_q4_dot_gemm_a4) {
            quantize_a_q8_blocks_a4(a_ptr, M, K, lda, qA4);
        } else {
            quantize_a_q8_blocks(a_ptr, M, K, lda, K, qA, a_scales);
        }
        const int8_t* qA_data = qA.data();
        const float* a_scales_data = a_scales.data();
        const Q8A4Block* qA4_data = qA4.data();

        auto run_q4_gemm = [&](int m_begin, int m_end, int n_begin, int n_end) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
            if (use_q4_dot_gemm_i8mm) {
                if (use_q4_dot_gemm_a4) {
                    matmul_int4_q8dot_neon_4x8_repacked_i8mm_range<true>(
                        nullptr, nullptr, b_q4_repack, scales, group_size,
                        groups_per_row, c_ptr, M, N, K, K, ldc, m_begin, m_end,
                        qA4_data);
                } else {
                    matmul_int4_q8dot_neon_4x8_repacked_i8mm_range<false>(
                        qA_data, a_scales_data, b_q4_repack, scales, group_size,
                        groups_per_row, c_ptr, M, N, K, K, ldc, m_begin, m_end,
                        nullptr);
                }
                return;
            } else
#endif
                if (force_4x8) {
                matmul_int4_q8dot_neon_4x8_range(
                    qA_data, a_scales_data, b_int4,
                    use_q4_repack ? b_q4_repack : nullptr, scales, group_size,
                    groups_per_row, c_ptr, M, N, K, K, K_weight, ldc, m_begin,
                    m_end);
            } else if (use_q4_dot_gemm_bg128) {
                if (use_q4_dot_gemm_a4) {
                    matmul_int4_q8dot_neon_8x8_g128packed_range<true>(
                        nullptr, nullptr, b_q4_g128, c_ptr, M, N, K, K, ldc,
                        m_begin, m_end, n_begin, n_end, qA4_data);
                } else {
                    matmul_int4_q8dot_neon_8x8_g128packed_range<false>(
                        qA_data, a_scales_data, b_q4_g128, c_ptr, M, N, K, K,
                        ldc, m_begin, m_end, n_begin, n_end, nullptr);
                }
            } else if (use_q4_dot_gemm_a4) {
                matmul_int4_q8dot_neon_8x8_range<true>(
                    nullptr, nullptr, b_int4,
                    use_q4_repack ? b_q4_repack : nullptr, scales, group_size,
                    groups_per_row, c_ptr, M, N, K, K, K_weight, ldc, m_begin,
                    m_end, n_begin, n_end, qA4_data);
            } else {
                matmul_int4_q8dot_neon_8x8_range<false>(
                    qA_data, a_scales_data, b_int4,
                    use_q4_repack ? b_q4_repack : nullptr, scales, group_size,
                    groups_per_row, c_ptr, M, N, K, K, K_weight, ldc, m_begin,
                    m_end, n_begin, n_end, nullptr);
            }
        };

        if (!use_parallel) {
            run_q4_gemm(0, M, 0, N);
        } else {
            thread_pool->parallel_for(0, M, tile_m,
                                      [&](int, int m_begin, int m_end) {
                                          run_q4_gemm(m_begin, m_end, 0, N);
                                      });
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act, act_n_begin,
                                    act_n_len);
        }
        return;
    }
#endif

    timer.set_shape("int4_scalar", M, N, K, group_size, groups_per_row, false,
                    false, n_threads);
    if (!use_parallel) {
        matmul_int4_scalar_range(a_ptr, b_int4, scales, group_size,
                                 groups_per_row, c_ptr, M, N, K, lda, K_weight,
                                 ldc, 0, M, 0, N);
    } else if (shard_by_n) {
        thread_pool->parallel_for(
            0, N, chunk_size, [&](int, int n_begin, int n_end) {
                matmul_int4_scalar_range(a_ptr, b_int4, scales, group_size,
                                         groups_per_row, c_ptr, M, N, K, lda,
                                         K_weight, ldc, 0, M, n_begin, n_end);
            });
    } else {
        thread_pool->parallel_for(
            0, M, chunk_size, [&](int, int m_begin, int m_end) {
                matmul_int4_scalar_range(a_ptr, b_int4, scales, group_size,
                                         groups_per_row, c_ptr, M, N, K, lda,
                                         K_weight, ldc, m_begin, m_end, 0, N);
            });
    }
    if (act != Activation::NONE && act_n_len != 0) {
        matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act, act_n_begin,
                                act_n_len);
    }
    return;
}
