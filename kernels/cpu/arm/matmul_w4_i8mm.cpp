#include "kernels/cpu/matmul/matmul_internal.h"

#include <algorithm>
#include <cstdint>

#if HAS_NEON && defined(__ARM_FEATURE_MATMUL_INT8)
__attribute__((always_inline)) static inline void
i8mm_w4_8x4_accumulate_qblock(const int8_t* aq, const uint8_t* bq,
                              const float* a_scales, float32x4_t out[8]) {
    __asm__ __volatile__(
        "movi v16.4s, #0\n"
        "movi v17.4s, #0\n"
        "movi v18.4s, #0\n"
        "movi v19.4s, #0\n"
        "movi v20.4s, #0\n"
        "movi v21.4s, #0\n"
        "movi v22.4s, #0\n"
        "movi v23.4s, #0\n"

        // Load four existing BG128 columns, then ZIP their lower and upper
        // 8-byte halves into the column-pair vectors needed by SMMLA.
        "ldr q28, [%[bq], #0]\n"
        "ldr q30, [%[bq], #16]\n"
        "ldr q29, [%[bq], #32]\n"
        "ldr q31, [%[bq], #48]\n"
        "zip2 v24.2d, v28.2d, v30.2d\n"
        "zip1 v28.2d, v28.2d, v30.2d\n"
        "mov v30.16b, v24.16b\n"
        "zip2 v24.2d, v29.2d, v31.2d\n"
        "zip1 v29.2d, v29.2d, v31.2d\n"
        "mov v31.16b, v24.16b\n"

        // Even K[0:16] (low nibbles in the lower byte halves).
        "shl v24.16b, v28.16b, #4\n"
        "shl v25.16b, v29.16b, #4\n"
        "ldr q26, [%[aq], #0]\n"
        "smmla v16.4s, v26.16b, v24.16b\n"
        "smmla v17.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #16]\n"
        "smmla v18.4s, v26.16b, v24.16b\n"
        "smmla v19.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #32]\n"
        "smmla v20.4s, v26.16b, v24.16b\n"
        "smmla v21.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #48]\n"
        "smmla v22.4s, v26.16b, v24.16b\n"
        "smmla v23.4s, v26.16b, v25.16b\n"

        // Even K[16:32] (low nibbles in the upper byte halves).
        "shl v24.16b, v30.16b, #4\n"
        "shl v25.16b, v31.16b, #4\n"
        "ldr q26, [%[aq], #64]\n"
        "smmla v16.4s, v26.16b, v24.16b\n"
        "smmla v17.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #80]\n"
        "smmla v18.4s, v26.16b, v24.16b\n"
        "smmla v19.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #96]\n"
        "smmla v20.4s, v26.16b, v24.16b\n"
        "smmla v21.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #112]\n"
        "smmla v22.4s, v26.16b, v24.16b\n"
        "smmla v23.4s, v26.16b, v25.16b\n"

        // Odd K[0:16] and K[16:32] live in the corresponding high nibbles.
        "movi v27.16b, #0xf0\n"
        "and v24.16b, v28.16b, v27.16b\n"
        "and v25.16b, v29.16b, v27.16b\n"
        "ldr q26, [%[aq], #128]\n"
        "smmla v16.4s, v26.16b, v24.16b\n"
        "smmla v17.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #144]\n"
        "smmla v18.4s, v26.16b, v24.16b\n"
        "smmla v19.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #160]\n"
        "smmla v20.4s, v26.16b, v24.16b\n"
        "smmla v21.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #176]\n"
        "smmla v22.4s, v26.16b, v24.16b\n"
        "smmla v23.4s, v26.16b, v25.16b\n"

        "and v24.16b, v30.16b, v27.16b\n"
        "and v25.16b, v31.16b, v27.16b\n"
        "ldr q26, [%[aq], #192]\n"
        "smmla v16.4s, v26.16b, v24.16b\n"
        "smmla v17.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #208]\n"
        "smmla v18.4s, v26.16b, v24.16b\n"
        "smmla v19.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #224]\n"
        "smmla v20.4s, v26.16b, v24.16b\n"
        "smmla v21.4s, v26.16b, v25.16b\n"
        "ldr q26, [%[aq], #240]\n"
        "smmla v22.4s, v26.16b, v24.16b\n"
        "smmla v23.4s, v26.16b, v25.16b\n"

        // Keep the native 2-row x 2-column SMMLA matrix layout through all
        // qblocks. Pairwise scale vectors avoid transposing every qblock.
        "scvtf v16.4s, v16.4s, #4\n"
        "scvtf v17.4s, v17.4s, #4\n"
        "scvtf v18.4s, v18.4s, #4\n"
        "scvtf v19.4s, v19.4s, #4\n"
        "scvtf v20.4s, v20.4s, #4\n"
        "scvtf v21.4s, v21.4s, #4\n"
        "scvtf v22.4s, v22.4s, #4\n"
        "scvtf v23.4s, v23.4s, #4\n"

        "ldr q24, [%[as], #0]\n"
        "fmla %[o0].4s, v16.4s, v24.4s\n"
        "fmla %[o1].4s, v17.4s, v24.4s\n"
        "ldr q24, [%[as], #16]\n"
        "fmla %[o2].4s, v18.4s, v24.4s\n"
        "fmla %[o3].4s, v19.4s, v24.4s\n"
        "ldr q24, [%[as], #32]\n"
        "fmla %[o4].4s, v20.4s, v24.4s\n"
        "fmla %[o5].4s, v21.4s, v24.4s\n"
        "ldr q24, [%[as], #48]\n"
        "fmla %[o6].4s, v22.4s, v24.4s\n"
        "fmla %[o7].4s, v23.4s, v24.4s\n"
        : [o0] "+w"(out[0]), [o1] "+w"(out[1]), [o2] "+w"(out[2]),
          [o3] "+w"(out[3]), [o4] "+w"(out[4]), [o5] "+w"(out[5]),
          [o6] "+w"(out[6]), [o7] "+w"(out[7])
        : [aq] "r"(aq), [bq] "r"(bq), [as] "r"(a_scales)
        : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31");
}

void matmul_int4_i8mm_g128(
    const Q8A8I8MMBlock* qA8, const Q4B8G128Block* B_g128, float* C,
    int M, int N, int K, int ldc, int m_begin, int m_end, int n_begin,
    int n_end) {
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    int groups_per_row = K / 128;

    for (int m = m_begin; m < m_end; m += 8) {
        int r_valid = std::min(8, M - m);
        const Q8A8I8MMBlock* a_tile =
            qA8 + (size_t)(m / 8) * blocks_per_row;
        for (int n = n_begin; n < n_end; n += 8) {
            int c_valid = std::min({8, N - n, n_end - n});
            const Q4B8G128Block* b_tile =
                B_g128 + (size_t)(n / 8) * groups_per_row;
            for (int half = 0; half * 4 < c_valid; half++) {
                float32x4_t out[8];
                for (int r = 0; r < 8; r++) {
                    out[r] = vdupq_n_f32(0.f);
                }

                for (int g = 0; g < groups_per_row; g++) {
                    const Q4B8G128Block& bg = b_tile[g];
                    float32x4_t group[8];
                    for (int r = 0; r < 8; r++) {
                        group[r] = vdupq_n_f32(0.f);
                    }
                    for (int qgi = 0; qgi < 4; qgi++) {
                        const Q8A8I8MMBlock& ab = a_tile[g * 4 + qgi];
                        i8mm_w4_8x4_accumulate_qblock(
                            &ab.q[0][0][0], &bg.q[qgi][half * 4][0],
                            &ab.scale_pairs[0][0], group);
                    }
                    float32x2_t bscale01 =
                        vld1_f32(bg.scales + half * 4);
                    float32x2_t bscale23 =
                        vld1_f32(bg.scales + half * 4 + 2);
                    float32x4_t bscale_pair01 =
                        vcombine_f32(bscale01, bscale01);
                    float32x4_t bscale_pair23 =
                        vcombine_f32(bscale23, bscale23);
                    for (int rp = 0; rp < 4; rp++) {
                        out[rp * 2] = vfmaq_f32(
                            out[rp * 2], group[rp * 2], bscale_pair01);
                        out[rp * 2 + 1] = vfmaq_f32(
                            out[rp * 2 + 1], group[rp * 2 + 1],
                            bscale_pair23);
                    }
                }

                int half_cols = std::min(4, c_valid - half * 4);
                for (int rp = 0; rp < 4; rp++) {
                    float32x4_t row0 =
                        vcombine_f32(vget_low_f32(out[rp * 2]),
                                     vget_low_f32(out[rp * 2 + 1]));
                    float32x4_t row1 =
                        vcombine_f32(vget_high_f32(out[rp * 2]),
                                     vget_high_f32(out[rp * 2 + 1]));
                    for (int rr = 0; rr < 2; rr++) {
                        int r = rp * 2 + rr;
                        if (r >= r_valid)
                            continue;
                        float* dst =
                            C + (size_t)(m + r) * ldc + n + half * 4;
                        float32x4_t row = rr == 0 ? row0 : row1;
                        if (half_cols == 4) {
                            vst1q_f32(dst, row);
                            continue;
                        }
                        float tmp[4];
                        vst1q_f32(tmp, row);
                        for (int c = 0; c < half_cols; c++)
                            dst[c] = tmp[c];
                    }
                }
            }
        }
    }
}

void matmul_int4_i8mm_g32(
    const Q8A8I8MMBlock* qA8, const Q4B8G32Block* B_g32, float* C,
    int M, int N, int K, int ldc, int m_begin, int m_end, int n_begin,
    int n_end) {
    int blocks_per_row = K / MATMUL_Q8_BLOCK;

    for (int m = m_begin; m < m_end; m += 8) {
        int r_valid = std::min(8, M - m);
        const Q8A8I8MMBlock* a_tile =
            qA8 + (size_t)(m / 8) * blocks_per_row;
        for (int n = n_begin; n < n_end; n += 8) {
            int c_valid = std::min({8, N - n, n_end - n});
            const Q4B8G32Block* b_tile =
                B_g32 + (size_t)(n / 8) * blocks_per_row;
            for (int half = 0; half * 4 < c_valid; half++) {
                float32x4_t out[8];
                for (int r = 0; r < 8; r++)
                    out[r] = vdupq_n_f32(0.f);

                for (int qb = 0; qb < blocks_per_row; qb++) {
                    const Q8A8I8MMBlock& ab = a_tile[qb];
                    const Q4B8G32Block& bb = b_tile[qb];
                    float32x4_t block[8];
                    for (int r = 0; r < 8; r++)
                        block[r] = vdupq_n_f32(0.f);
                    i8mm_w4_8x4_accumulate_qblock(
                        &ab.q[0][0][0], &bb.q[half * 4][0],
                        &ab.scale_pairs[0][0], block);

                    float32x2_t bscale01 =
                        vld1_f32(bb.scales + half * 4);
                    float32x2_t bscale23 =
                        vld1_f32(bb.scales + half * 4 + 2);
                    float32x4_t bscale_pair01 =
                        vcombine_f32(bscale01, bscale01);
                    float32x4_t bscale_pair23 =
                        vcombine_f32(bscale23, bscale23);
                    for (int rp = 0; rp < 4; rp++) {
                        out[rp * 2] = vfmaq_f32(
                            out[rp * 2], block[rp * 2], bscale_pair01);
                        out[rp * 2 + 1] = vfmaq_f32(
                            out[rp * 2 + 1], block[rp * 2 + 1],
                            bscale_pair23);
                    }
                }

                int half_cols = std::min(4, c_valid - half * 4);
                for (int rp = 0; rp < 4; rp++) {
                    float32x4_t row0 =
                        vcombine_f32(vget_low_f32(out[rp * 2]),
                                     vget_low_f32(out[rp * 2 + 1]));
                    float32x4_t row1 =
                        vcombine_f32(vget_high_f32(out[rp * 2]),
                                     vget_high_f32(out[rp * 2 + 1]));
                    for (int rr = 0; rr < 2; rr++) {
                        int r = rp * 2 + rr;
                        if (r >= r_valid)
                            continue;
                        float* dst =
                            C + (size_t)(m + r) * ldc + n + half * 4;
                        float32x4_t row = rr == 0 ? row0 : row1;
                        if (half_cols == 4) {
                            vst1q_f32(dst, row);
                            continue;
                        }
                        float tmp[4];
                        vst1q_f32(tmp, row);
                        for (int c = 0; c < half_cols; c++)
                            dst[c] = tmp[c];
                    }
                }
            }
        }
    }
}

#endif
