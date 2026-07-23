#include "kernels/matmul.h"
#include "kernels/matmul_internal.h"
#include "kernels/matmul_profile.h"
#include "kernels/threading.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace {

struct LocalActivationRange {
    int begin;
    int length;
};

LocalActivationRange local_activation_range(int global_begin,
                                            int global_length,
                                            int shard_begin, int shard_end) {
    const int begin = std::max(0, global_begin - shard_begin);
    if (global_length < 0)
        return {begin, -1};
    const int end = std::min(shard_end - shard_begin,
                             global_begin + global_length - shard_begin);
    return {begin, std::max(0, end - begin)};
}

}  // namespace

// ---------------------------------------------------------------------------
// scalar matmul (fallback)
// ---------------------------------------------------------------------------

// After weight dim swap: B has shape [N, K] (N=output, K=input).
// Weight file stores row-major [N, K]: data[n*K + k] = W[n, k].
// Tensor access: B.at<float>(n, k) = data[n*K + k].
// C[m,n] = sum_k A[k + m*lda] * B[n*K_weight + k] = A[m,:] @ W[n,:].
[[maybe_unused]] static void matmul_fp32_scalar_range(
    const float* A, const float* B, float* C, int M, int N, int K, int lda,
    int K_weight, int ldc, int m_begin, int m_end) {
    (void)M;
    for (int m = m_begin; m < m_end; m++) {
        float* c_row = C + m * ldc;
        for (int n = 0; n < N; n++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += A[k + m * lda] * B[n * K_weight + k];
            }
            c_row[n] = sum;
        }
    }
}

#if HAS_NEON
static std::unique_ptr<__fp16[]>
pack_a_interleaved_full(const float* A_original, int M, int K, int lda) {
    auto t0 = std::chrono::steady_clock::now();
    int M_padded = ((M + 7) / 8) * 8; // round up to multiple of 8
    auto dst = std::make_unique<__fp16[]>((size_t)M_padded * K);
    for (int m_tile = 0; m_tile < M_padded; m_tile += 8) {
        int tile_valid = std::min(8, M - m_tile);
        if (tile_valid < 0)
            tile_valid = 0;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < tile_valid; j++) {
                dst[m_tile * K + k * 8 + j] =
                    (__fp16)A_original[k + (m_tile + j) * lda];
            }
            for (int j = tile_valid; j < 8; j++) {
                dst[m_tile * K + k * 8 + j] = (__fp16)0.f;
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    matmul_record_pack_a(ms);
    return dst;
}

static void matmul_fp16_neon_8x8_range(const float* A, const __fp16* B,
                                       float* C, int M, int N, int K, int lda,
                                       int K_weight, int ldc, int m_begin,
                                       int m_end) {
    const int K_BLOCK =
        g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    // FP16: twice as many elements fit in cache, so double K_BLOCK.
    const int K_BLOCK_FP16 = K_BLOCK * 2;

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK_FP16) {
        int k_end = std::min(k_outer + K_BLOCK_FP16, K);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                float32x4_t c[8][2];
                bool first_block = (k_outer == 0);
                if (first_block) {
                    for (int r = 0; r < 8; r++) {
                        c[r][0] = vdupq_n_f32(0.f);
                        c[r][1] = vdupq_n_f32(0.f);
                    }
                } else {
                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            c[r][0] = vld1q_f32(&C[row * ldc + n]);
                            if (n + 4 < n_end) {
                                c[r][1] = vld1q_f32(&C[row * ldc + n + 4]);
                            } else {
                                c[r][1] = vdupq_n_f32(0.f);
                            }
                        } else {
                            c[r][0] = vdupq_n_f32(0.f);
                            c[r][1] = vdupq_n_f32(0.f);
                        }
                    }
                }

                for (int k = k_outer; k < k_end; k++) {
                    // Load 8 FP16 B values, convert to FP32.
                    float32x4_t b0, b1;
                    {
                        __fp16 tmp[4] = {(__fp16)0.f, (__fp16)0.f, (__fp16)0.f,
                                         (__fp16)0.f};
                        for (int j = 0; j < 4 && n + j < n_end; j++) {
                            tmp[j] = B[(n + j) * K_weight + k];
                        }
                        b0 = vcvt_f32_f16(vld1_f16(tmp));
                    }
                    {
                        __fp16 tmp[4] = {(__fp16)0.f, (__fp16)0.f, (__fp16)0.f,
                                         (__fp16)0.f};
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++) {
                            tmp[j] = B[(n + 4 + j) * K_weight + k];
                        }
                        b1 = vcvt_f32_f16(vld1_f16(tmp));
                    }

                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            float a_val = A[k + row * lda];
                            c[r][0] = vfmaq_n_f32(c[r][0], b0, a_val);
                            c[r][1] = vfmaq_n_f32(c[r][1], b1, a_val);
                        }
                    }
                }

                // Write back
                for (int r = 0; r < 8; r++) {
                    int row = m + r;
                    if (row < m_tile_end && row < m_global_end) {
                        float tmp[4];
                        vst1q_f32(tmp, c[r][0]);
                        for (int j = 0; j < 4 && n + j < n_end; j++)
                            C[row * ldc + n + j] = tmp[j];
                        vst1q_f32(tmp, c[r][1]);
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++)
                            C[row * ldc + n + 4 + j] = tmp[j];
                    }
                }
            }
        }
    }
}

// ---- FP16 packed kernel (B pre-packed interleaved, contiguous load) ----
//
// B_packed: load-time interleaved layout [N/8, K, 8].
// For fixed k, B_packed[(n & ~7) * K + k * 8 + 0..7] are 8 consecutive FP16.
// K-blocking loop is internal (cache blocking only, no packing).
static void matmul_fp16_neon_8x8_range_packed(const float* A,
                                              const __fp16* B_packed, float* C,
                                              int M, int N, int K, int lda,
                                              int ldc, int m_begin, int m_end) {
    const int K_BLOCK =
        g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2; // FP16: 2x cache density

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK_FP16) {
        int k_end = std::min(k_outer + K_BLOCK_FP16, K);
        bool first_block = (k_outer == 0);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                float32x4_t c[8][2];
                if (first_block) {
                    for (int r = 0; r < 8; r++) {
                        c[r][0] = vdupq_n_f32(0.f);
                        c[r][1] = vdupq_n_f32(0.f);
                    }
                } else {
                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            c[r][0] = vld1q_f32(&C[row * ldc + n]);
                            if (n + 4 < n_end) {
                                c[r][1] = vld1q_f32(&C[row * ldc + n + 4]);
                            } else {
                                c[r][1] = vdupq_n_f32(0.f);
                            }
                        } else {
                            c[r][0] = vdupq_n_f32(0.f);
                            c[r][1] = vdupq_n_f32(0.f);
                        }
                    }
                }

                for (int k = k_outer; k < k_end; k++) {
                    // Load 8 contiguous FP16 values from pre-packed B
                    float16x8_t b_vec =
                        vld1q_f16(&B_packed[(n & ~7) * K + k * 8]);
                    float32x4_t b0 = vcvt_f32_f16(vget_low_f16(b_vec));
                    float32x4_t b1 = vcvt_f32_f16(vget_high_f16(b_vec));

                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            float a_val = A[k + row * lda];
                            c[r][0] = vfmaq_n_f32(c[r][0], b0, a_val);
                            c[r][1] = vfmaq_n_f32(c[r][1], b1, a_val);
                        }
                    }
                }

                // Write back
                for (int r = 0; r < 8; r++) {
                    int row = m + r;
                    if (row < m_tile_end && row < m_global_end) {
                        float tmp[4];
                        vst1q_f32(tmp, c[r][0]);
                        for (int j = 0; j < 4 && n + j < n_end; j++)
                            C[row * ldc + n + j] = tmp[j];
                        vst1q_f32(tmp, c[r][1]);
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++)
                            C[row * ldc + n + 4 + j] = tmp[j];
                    }
                }
            }
        }
    }
}

// ---- FP16 GEMV kernel (M=1, B pre-packed interleaved) ----
//
// C[n] = sum_k A[k] * B[n, k]
//
// A:        FP32 [K]              (single row, M=1)
// B_packed: interleaved FP16 [N/8, K, 8]
//           For fixed k, B_packed[(n & ~7) * K + k * 8 + 0..7] = 8 N values
// C:        FP32 [N] output
//
// n_begin must be 8-aligned (parallel_for grain_size is aligned). n_end may
// be any value; the trailing partial tile is masked on store.
//
// Two accumulate modes:
//   FP32 acc (default fallback): 1 vld + 2 vcvt + 2 vfmaq_n_f32 = 5
//   instr/K-step
//     FMA-bound at ~48 GF/s (4 threads). Higher precision.
//   FP16 acc (use_fp16_accumulate=true): 1 vld + 1 vfmaq_n_f16 = 2 instr/K-step
//     Bandwidth-bound at ~60 GF/s (4 threads). K_BLOCK store/reload for
//     precision.

static void matmul_fp16_neon_gemv_range(const float* A, const __fp16* B_packed,
                                        float* C, int K, int n_begin,
                                        int n_end) {
    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);

        float32x4_t acc0 = vdupq_n_f32(0.f); // n+0..n+3
        float32x4_t acc1 = vdupq_n_f32(0.f); // n+4..n+7

        const __fp16* b_tile = &B_packed[(n & ~7) * K];
        for (int k = 0; k < K; k++) {
            float a_val = A[k];
            float16x8_t b_vec = vld1q_f16(b_tile + k * 8);
            acc0 = vfmaq_n_f32(acc0, vcvt_f32_f16(vget_low_f16(b_vec)), a_val);
            acc1 = vfmaq_n_f32(acc1, vcvt_f32_f16(vget_high_f16(b_vec)), a_val);
        }

        // Store partial tile (handles n_end not 8-aligned).
        float tmp[4];
        vst1q_f32(tmp, acc0);
        for (int j = 0; j < 4 && n + j < n_tile_end; j++)
            C[n + j] = tmp[j];
        vst1q_f32(tmp, acc1);
        for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++)
            C[n + 4 + j] = tmp[j];
    }
}

// FP16 accumulate variant: vfmaq_n_f16 does 8-lane FP16×FP16→FP16 FMA with
// scalar broadcast. K_BLOCK store/reload (FP16→FP32→FP16) between blocks
// limits within-block FP16 accumulation range for precision.
//
// 8-way K-unroll with 8 independent FP16 accumulators to fully hide FMA
// latency. Apple M5 Pro FP16 FMA latency = 2 cycles, throughput = 2/cycle.
// With N independent acc chains, CPI = max(1, 2/N) cycles/K-step.
//   N=1 (single-acc):  CPI=2  → latency-bound
//   N=2 (2-way unroll): CPI=1  → at FMA throughput ceiling
//   N=8 (8-way unroll): CPI=1  → same throughput, but lower loop overhead
//                                 per FMA, better ILP for issue stage.
// Inspired by ggml's SVE GEMV which uses 4-acc × 8-way K-unroll.
static void
matmul_fp16_neon_gemv_range_fp16acc(const float* A, const __fp16* B_packed,
                                    float* C, int K, int n_begin, int n_end,
                                    Activation act = Activation::NONE,
                                    int act_n_begin = 0, int act_n_len = -1) {
    const int K_BLOCK =
        g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2; // FP16: 2x cache density

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        const __fp16* b_tile = &B_packed[(n & ~7) * K];

        // 8 independent FP16 acc chains to fully hide FMA latency.
        // Each K step writes to one acc, rotating through acc[0..7].
        // After 8 K steps, every acc has exactly 1 FMA → fully independent.
        float16x8_t acc[8];

        int k_outer = 0;
        for (; k_outer < K; k_outer += K_BLOCK_FP16) {
            int k_end = std::min(k_outer + K_BLOCK_FP16, K);

            // Initialize acc for this K-block.
            if (k_outer == 0) {
                for (int i = 0; i < 8; i++)
                    acc[i] = vdupq_n_f16((__fp16)0.f);
            } else {
                // Reload partial sums from C (FP32 → FP16) — split into 8
                // chains. acc[0] carries the merged partial from prior block.
                float tmp[8];
                for (int j = 0; j < 8; j++) {
                    tmp[j] = (n + j < n_tile_end) ? C[n + j] : 0.f;
                }
                acc[0] = vcombine_f16(vcvt_f16_f32(vld1q_f32(tmp)),
                                      vcvt_f16_f32(vld1q_f32(tmp + 4)));
                for (int i = 1; i < 8; i++)
                    acc[i] = vdupq_n_f16((__fp16)0.f);
            }

            // 8-way K-unroll: 8 independent FMA chains.
            // Each K step: load A[k], load B[k*8], FMA into acc[k%8].
            // 8 acc chains → FMA latency (2 cycles) fully hidden.
            int k = k_outer;
            int k_unrolled_end = k_outer + ((k_end - k_outer) & ~7);
            for (; k < k_unrolled_end; k += 8) {
                __fp16 a0 = (__fp16)A[k + 0];
                __fp16 a1 = (__fp16)A[k + 1];
                __fp16 a2 = (__fp16)A[k + 2];
                __fp16 a3 = (__fp16)A[k + 3];
                __fp16 a4 = (__fp16)A[k + 4];
                __fp16 a5 = (__fp16)A[k + 5];
                __fp16 a6 = (__fp16)A[k + 6];
                __fp16 a7 = (__fp16)A[k + 7];
                float16x8_t b0 = vld1q_f16(b_tile + (k + 0) * 8);
                float16x8_t b1 = vld1q_f16(b_tile + (k + 1) * 8);
                float16x8_t b2 = vld1q_f16(b_tile + (k + 2) * 8);
                float16x8_t b3 = vld1q_f16(b_tile + (k + 3) * 8);
                float16x8_t b4 = vld1q_f16(b_tile + (k + 4) * 8);
                float16x8_t b5 = vld1q_f16(b_tile + (k + 5) * 8);
                float16x8_t b6 = vld1q_f16(b_tile + (k + 6) * 8);
                float16x8_t b7 = vld1q_f16(b_tile + (k + 7) * 8);
                acc[0] = vfmaq_n_f16(acc[0], b0, a0);
                acc[1] = vfmaq_n_f16(acc[1], b1, a1);
                acc[2] = vfmaq_n_f16(acc[2], b2, a2);
                acc[3] = vfmaq_n_f16(acc[3], b3, a3);
                acc[4] = vfmaq_n_f16(acc[4], b4, a4);
                acc[5] = vfmaq_n_f16(acc[5], b5, a5);
                acc[6] = vfmaq_n_f16(acc[6], b6, a6);
                acc[7] = vfmaq_n_f16(acc[7], b7, a7);
            }
            // Tail (K % 8 != 0 within block): fall back to rotating acc[0..7].
            for (; k < k_end; k++) {
                __fp16 a0 = (__fp16)A[k];
                float16x8_t b0 = vld1q_f16(b_tile + k * 8);
                int idx = (k - k_outer) & 7;
                acc[idx] = vfmaq_n_f16(acc[idx], b0, a0);
            }

            // Merge 8 acc chains into acc[0] (pairwise reduction).
            acc[0] = vaddq_f16(acc[0], acc[1]);
            acc[2] = vaddq_f16(acc[2], acc[3]);
            acc[4] = vaddq_f16(acc[4], acc[5]);
            acc[6] = vaddq_f16(acc[6], acc[7]);
            acc[0] = vaddq_f16(acc[0], acc[2]);
            acc[4] = vaddq_f16(acc[4], acc[6]);
            acc[0] = vaddq_f16(acc[0], acc[4]);

            // Store partial sums to C as FP32 (reloaded if more blocks remain).
            // On the last K-block, also apply fused activation in-place.
            // Single-store path: convert FP16 acc → FP32, apply act if needed,
            // store. Avoids a separate activation pass after the K-block loop.
            bool is_last_block = (k_outer + K_BLOCK_FP16 >= K);
            bool apply_act = (act != Activation::NONE) && is_last_block;

            float32x4_t lo = vcvt_f32_f16(vget_low_f16(acc[0]));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(acc[0]));
            if (apply_act) {
                // Apply activation per-column.
                // For typical 8-wide N-tile, columns are either all-active or
                // all-inactive (since act_n_len is multiple of 8 for SwiGLU
                // gate). Fast path: if all 4 in [act_n_begin,
                // act_n_begin+act_n_len), vectorize; else scalar per-column.
                bool lo_active =
                    (act_n_len < 0) ||
                    (activation_applies_at(n + 0, act_n_begin, act_n_len) &&
                     activation_applies_at(n + 1, act_n_begin, act_n_len) &&
                     activation_applies_at(n + 2, act_n_begin, act_n_len) &&
                     activation_applies_at(n + 3, act_n_begin, act_n_len));
                if (lo_active) {
                    lo = apply_activation_f32_neon(lo, act);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, lo);
                    for (int jj = 0; jj < 4; jj++) {
                        if (activation_applies_at(n + jj, act_n_begin,
                                                  act_n_len)) {
                            tmp[jj] = apply_activation_scalar(tmp[jj], act);
                        }
                    }
                    lo = vld1q_f32(tmp);
                }
                bool hi_active =
                    (act_n_len < 0) ||
                    (activation_applies_at(n + 4, act_n_begin, act_n_len) &&
                     activation_applies_at(n + 5, act_n_begin, act_n_len) &&
                     activation_applies_at(n + 6, act_n_begin, act_n_len) &&
                     activation_applies_at(n + 7, act_n_begin, act_n_len));
                if (hi_active) {
                    hi = apply_activation_f32_neon(hi, act);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, hi);
                    for (int jj = 0; jj < 4; jj++) {
                        if (activation_applies_at(n + 4 + jj, act_n_begin,
                                                  act_n_len)) {
                            tmp[jj] = apply_activation_scalar(tmp[jj], act);
                        }
                    }
                    hi = vld1q_f32(tmp);
                }
            }

            float tmp[4];
            vst1q_f32(tmp, lo);
            for (int j = 0; j < 4 && n + j < n_tile_end; j++)
                C[n + j] = tmp[j];
            vst1q_f32(tmp, hi);
            for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++)
                C[n + 4 + j] = tmp[j];
        }

        // Note: fused activation is applied inline above on the last K-block
        // store, so no separate post-loop pass is needed here.
    }
}

static void matmul_fp16_neon_8x8_range_packed_a(const __fp16* A_packed,
                                                const __fp16* B_packed,
                                                float* C, int M, int N, int K,
                                                int ldc, int m_begin,
                                                int m_end) {
    const int K_BLOCK =
        g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2;

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK_FP16) {
        int k_end = std::min(k_outer + K_BLOCK_FP16, K);
        bool first_block = (k_outer == 0);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                // Column-major accumulators: c[j][half]
                // c[j][0] = [C[m+0][n+j], C[m+1][n+j], C[m+2][n+j],
                // C[m+3][n+j]] c[j][1] = [C[m+4][n+j], C[m+5][n+j],
                // C[m+6][n+j], C[m+7][n+j]]
                float32x4_t c[8][2];
                if (first_block) {
                    for (int j = 0; j < 8; j++) {
                        c[j][0] = vdupq_n_f32(0.f);
                        c[j][1] = vdupq_n_f32(0.f);
                    }
                } else {
                    // Gather C columns into column-major accumulators.
                    // C is row-major: C[(m+r) * ldc + (n+j)].
                    // Build c[j][0] = [C[m+0][n+j], C[m+1][n+j], C[m+2][n+j],
                    // C[m+3][n+j]] Use vld4q_f32 to de-interleave 4 rows × 4
                    // cols block. For each 4x4 block, load 4 rows and
                    // transpose.
                    auto load_col4 = [&](int col,
                                         int row_start) -> float32x4_t {
                        // Gather C[row_start..row_start+3][col] into a vector
                        float tmp[4];
                        for (int r = 0; r < 4; r++) {
                            int row = row_start + r;
                            if (row < m_tile_end && row < m_global_end &&
                                col < n_end) {
                                tmp[r] = C[row * ldc + col];
                            } else {
                                tmp[r] = 0.f;
                            }
                        }
                        return vld1q_f32(tmp);
                    };
                    for (int j = 0; j < 8; j++) {
                        c[j][0] = load_col4(n + j, m + 0);
                        c[j][1] = load_col4(n + j, m + 4);
                    }
                }

                // K loop: lane-FMA
                // A_packed: A_packed[(m & ~7) * K + k * 8 + 0..7] = 8 M values
                // at K=k B_packed: B_packed[(n & ~7) * K + k * 8 + 0..7] = 8 N
                // values at K=k
                for (int k = k_outer; k < k_end; k++) {
                    float16x8_t a_vec =
                        vld1q_f16(&A_packed[(m & ~7) * K + k * 8]);
                    float16x8_t b_vec =
                        vld1q_f16(&B_packed[(n & ~7) * K + k * 8]);

                    // For each N column j, broadcast b_vec[j] and FMA with
                    // a_vec low half (rows 0..3) and high half (rows 4..7)
                    c[0][0] = vfmlalq_laneq_low_f16(c[0][0], a_vec, b_vec, 0);
                    c[0][1] = vfmlalq_laneq_high_f16(c[0][1], a_vec, b_vec, 0);
                    c[1][0] = vfmlalq_laneq_low_f16(c[1][0], a_vec, b_vec, 1);
                    c[1][1] = vfmlalq_laneq_high_f16(c[1][1], a_vec, b_vec, 1);
                    c[2][0] = vfmlalq_laneq_low_f16(c[2][0], a_vec, b_vec, 2);
                    c[2][1] = vfmlalq_laneq_high_f16(c[2][1], a_vec, b_vec, 2);
                    c[3][0] = vfmlalq_laneq_low_f16(c[3][0], a_vec, b_vec, 3);
                    c[3][1] = vfmlalq_laneq_high_f16(c[3][1], a_vec, b_vec, 3);
                    c[4][0] = vfmlalq_laneq_low_f16(c[4][0], a_vec, b_vec, 4);
                    c[4][1] = vfmlalq_laneq_high_f16(c[4][1], a_vec, b_vec, 4);
                    c[5][0] = vfmlalq_laneq_low_f16(c[5][0], a_vec, b_vec, 5);
                    c[5][1] = vfmlalq_laneq_high_f16(c[5][1], a_vec, b_vec, 5);
                    c[6][0] = vfmlalq_laneq_low_f16(c[6][0], a_vec, b_vec, 6);
                    c[6][1] = vfmlalq_laneq_high_f16(c[6][1], a_vec, b_vec, 6);
                    c[7][0] = vfmlalq_laneq_low_f16(c[7][0], a_vec, b_vec, 7);
                    c[7][1] = vfmlalq_laneq_high_f16(c[7][1], a_vec, b_vec, 7);
                }

                // Write back: column-major → row-major C
                // c[j][0] = [C[m+0][n+j], C[m+1][n+j], C[m+2][n+j],
                // C[m+3][n+j]] Need: C[(m+r) * ldc + (n+j)] = c[j][half][r]
                auto store_col4 = [&](int col, int row_start, float32x4_t val) {
                    float tmp[4];
                    vst1q_f32(tmp, val);
                    for (int r = 0; r < 4; r++) {
                        int row = row_start + r;
                        if (row < m_tile_end && row < m_global_end &&
                            col < n_end) {
                            C[row * ldc + col] = tmp[r];
                        }
                    }
                };
                for (int j = 0; j < 8; j++) {
                    store_col4(n + j, m + 0, c[j][0]);
                    store_col4(n + j, m + 4, c[j][1]);
                }
            }
        }
    }
}

// ---- FP16 lane-FMA kernel with FP16 accumulation ----
//
// Same interface as above but accumulates in float16x8_t using vfmaq_lane_f16.
// 2x FMA throughput vs FP32 widening path. Accumulator is row-major
// (c[j] = 8 M rows for N column j), writeback is direct store.
//
// Precision: FP16 accumulation over K_BLOCK (1024) terms. Between K-blocks,
// partial results are stored to C (FP32) and reloaded (FP16), providing
// FP32 intermediate precision across blocks.
static void matmul_fp16_neon_8x8_range_packed_a_fp16acc(
    const __fp16* A_packed, const __fp16* B_packed, float* C, int M, int N,
    int K, int ldc, int m_begin, int m_end, Activation act = Activation::NONE,
    int act_n_begin = 0, int act_n_len = -1) {
    const int K_BLOCK =
        g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2;

    // Fused activation fast path: when act == NONE or act_n_len covers whole N,
    // we can skip per-column check. We still need to apply after all K-blocks
    // (activation must not be applied to intermediate partial sums).
    bool act_enabled = (act != Activation::NONE) && (act_n_len != 0);
    bool act_full_N =
        act_enabled && (act_n_len < 0 || (act_n_begin == 0 && act_n_len >= N));

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK_FP16) {
        int k_end = std::min(k_outer + K_BLOCK_FP16, K);
        bool first_block = (k_outer == 0);
        bool last_block = (k_outer + K_BLOCK_FP16 >= K);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                // Row-major accumulators: c[j] = 8 M values for N column n+j
                // Layout matches C: c[j][r] = C[(m+r)][n+j]
                float16x8_t c[8];
                if (first_block) {
                    for (int j = 0; j < 8; j++)
                        c[j] = vdupq_n_f16((__fp16)0.f);
                } else {
                    // Reload from C (FP32 → FP16)
                    for (int j = 0; j < 8; j++) {
                        if (n + j < n_end) {
                            float tmp[8];
                            for (int r = 0; r < 8; r++) {
                                int row = m + r;
                                tmp[r] =
                                    (row < m_tile_end && row < m_global_end)
                                        ? C[row * ldc + n + j]
                                        : 0.f;
                            }
                            c[j] =
                                vcombine_f16(vcvt_f16_f32(vld1q_f32(tmp)),
                                             vcvt_f16_f32(vld1q_f32(tmp + 4)));
                        } else {
                            c[j] = vdupq_n_f16((__fp16)0.f);
                        }
                    }
                }

                // K loop: FP16 lane-FMA, 2-way unroll.
                // 8 N-columns already provide 8 independent acc chains (>>2),
                // enough to hide FMA latency (lat=2, throughput=2/cycle).
                // 2-way unroll mainly amortizes loop branch overhead.
                // Note: 8-way unroll (64 acc) was tried and reverted — it
                // caused register spill on Apple Silicon (32 NEON Q regs),
                // hurting prefill.
                float16x8_t c1[8];
                for (int j = 0; j < 8; j++)
                    c1[j] = vdupq_n_f16((__fp16)0.f);

                int k = k_outer;
                for (; k + 1 < k_end; k += 2) {
                    float16x8_t a0 = vld1q_f16(&A_packed[(m & ~7) * K + k * 8]);
                    float16x8_t a1 =
                        vld1q_f16(&A_packed[(m & ~7) * K + (k + 1) * 8]);
                    float16x8_t b0 = vld1q_f16(&B_packed[(n & ~7) * K + k * 8]);
                    float16x8_t b1 =
                        vld1q_f16(&B_packed[(n & ~7) * K + (k + 1) * 8]);
                    float16x4_t b0_low = vget_low_f16(b0),
                                b0_high = vget_high_f16(b0);
                    float16x4_t b1_low = vget_low_f16(b1),
                                b1_high = vget_high_f16(b1);

                    c[0] = vfmaq_lane_f16(c[0], a0, b0_low, 0);
                    c[1] = vfmaq_lane_f16(c[1], a0, b0_low, 1);
                    c[2] = vfmaq_lane_f16(c[2], a0, b0_low, 2);
                    c[3] = vfmaq_lane_f16(c[3], a0, b0_low, 3);
                    c[4] = vfmaq_lane_f16(c[4], a0, b0_high, 0);
                    c[5] = vfmaq_lane_f16(c[5], a0, b0_high, 1);
                    c[6] = vfmaq_lane_f16(c[6], a0, b0_high, 2);
                    c[7] = vfmaq_lane_f16(c[7], a0, b0_high, 3);

                    c1[0] = vfmaq_lane_f16(c1[0], a1, b1_low, 0);
                    c1[1] = vfmaq_lane_f16(c1[1], a1, b1_low, 1);
                    c1[2] = vfmaq_lane_f16(c1[2], a1, b1_low, 2);
                    c1[3] = vfmaq_lane_f16(c1[3], a1, b1_low, 3);
                    c1[4] = vfmaq_lane_f16(c1[4], a1, b1_high, 0);
                    c1[5] = vfmaq_lane_f16(c1[5], a1, b1_high, 1);
                    c1[6] = vfmaq_lane_f16(c1[6], a1, b1_high, 2);
                    c1[7] = vfmaq_lane_f16(c1[7], a1, b1_high, 3);
                }
                // Tail (odd K within block)
                if (k < k_end) {
                    float16x8_t a0 = vld1q_f16(&A_packed[(m & ~7) * K + k * 8]);
                    float16x8_t b0 = vld1q_f16(&B_packed[(n & ~7) * K + k * 8]);
                    float16x4_t b0_low = vget_low_f16(b0),
                                b0_high = vget_high_f16(b0);
                    c[0] = vfmaq_lane_f16(c[0], a0, b0_low, 0);
                    c[1] = vfmaq_lane_f16(c[1], a0, b0_low, 1);
                    c[2] = vfmaq_lane_f16(c[2], a0, b0_low, 2);
                    c[3] = vfmaq_lane_f16(c[3], a0, b0_low, 3);
                    c[4] = vfmaq_lane_f16(c[4], a0, b0_high, 0);
                    c[5] = vfmaq_lane_f16(c[5], a0, b0_high, 1);
                    c[6] = vfmaq_lane_f16(c[6], a0, b0_high, 2);
                    c[7] = vfmaq_lane_f16(c[7], a0, b0_high, 3);
                }
                // Merge c += c1
                for (int j = 0; j < 8; j++)
                    c[j] = vaddq_f16(c[j], c1[j]);

                // Write back: FP16 → FP32, store to C
                // On the last K-block, also apply fused activation if enabled.
                // Batch-check activation status per N-tile (8 columns): SwiGLU
                // gate (act_n_len multiple of 8) → all 8 columns same status,
                // so we can skip per-column check entirely.
                bool tile_active =
                    act_enabled && last_block &&
                    (act_full_N ||
                     (activation_applies_at(n + 0, act_n_begin, act_n_len) &&
                      activation_applies_at(n + 7, act_n_begin, act_n_len)));
                for (int j = 0; j < 8 && n + j < n_end; j++) {
                    float32x4_t lo = vcvt_f32_f16(vget_low_f16(c[j]));
                    float32x4_t hi = vcvt_f32_f16(vget_high_f16(c[j]));
                    if (tile_active) {
                        lo = apply_activation_f32_neon(lo, act);
                        hi = apply_activation_f32_neon(hi, act);
                    } else if (act_enabled && last_block) {
                        // Mixed tile (rare — only at gate/up boundary).
                        // Per-column scalar apply via lane extract/insert.
                        if (activation_applies_at(n + j, act_n_begin,
                                                  act_n_len)) {
                            // Apply to all 8 M-rows for this N column.
                            // lo = [m0,m1,m2,m3], hi = [m4,m5,m6,m7]
                            float32x4_t a_lo =
                                apply_activation_f32_neon(lo, act);
                            float32x4_t a_hi =
                                apply_activation_f32_neon(hi, act);
                            lo = a_lo;
                            hi = a_hi;
                        }
                    }
                    float tmp[8];
                    vst1q_f32(tmp, lo);
                    vst1q_f32(tmp + 4, hi);
                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            C[row * ldc + n + j] = tmp[r];
                        }
                    }
                }
            }
        }
    }
}

// ---- FP32 kernel (kept for backward compat) ----
static void matmul_fp32_neon_8x8_range(const float* A, const float* B, float* C,
                                       int M, int N, int K, int lda,
                                       int K_weight, int ldc, int m_begin,
                                       int m_end) {
    const int K_BLOCK =
        g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK) {
        int k_end = std::min(k_outer + K_BLOCK, K);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                float32x4_t c[8]
                             [2]; // c[row][col_hi/lo]: hi=cols 0-3, lo=cols 4-7
                bool first_block = (k_outer == 0);
                if (first_block) {
                    for (int r = 0; r < 8; r++) {
                        c[r][0] = vdupq_n_f32(0.f);
                        c[r][1] = vdupq_n_f32(0.f);
                    }
                } else {
                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            c[r][0] = vld1q_f32(&C[row * ldc + n]);
                            if (n + 4 < n_end) {
                                c[r][1] = vld1q_f32(&C[row * ldc + n + 4]);
                            } else {
                                c[r][1] = vdupq_n_f32(0.f);
                            }
                        } else {
                            c[r][0] = vdupq_n_f32(0.f);
                            c[r][1] = vdupq_n_f32(0.f);
                        }
                    }
                }

                for (int k = k_outer; k < k_end; k++) {
                    // Load B columns n..n+3
                    float tmp_b0[4] = {0.f, 0.f, 0.f, 0.f};
                    float tmp_b1[4] = {0.f, 0.f, 0.f, 0.f};
                    for (int j = 0; j < 4 && n + j < n_end; j++) {
                        tmp_b0[j] = B[(n + j) * K_weight + k];
                    }
                    for (int j = 0; j < 4 && n + 4 + j < n_end; j++) {
                        tmp_b1[j] = B[(n + 4 + j) * K_weight + k];
                    }
                    float32x4_t b0 = vld1q_f32(tmp_b0);
                    float32x4_t b1 = vld1q_f32(tmp_b1);

                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            float a_val = A[k + row * lda];
                            c[r][0] = vfmaq_n_f32(c[r][0], b0, a_val);
                            c[r][1] = vfmaq_n_f32(c[r][1], b1, a_val);
                        }
                    }
                }

                // Write back
                for (int r = 0; r < 8; r++) {
                    int row = m + r;
                    if (row < m_tile_end && row < m_global_end) {
                        float tmp[4];
                        vst1q_f32(tmp, c[r][0]);
                        for (int j = 0; j < 4 && n + j < n_end; j++)
                            C[row * ldc + n + j] = tmp[j];
                        vst1q_f32(tmp, c[r][1]);
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++)
                            C[row * ldc + n + 4 + j] = tmp[j];
                    }
                }
            }
        }
    }
}

#endif // HAS_NEON

static void matmul_fp32_range(const float* A, const float* B, float* C, int M,
                              int N, int K, int lda, int K_weight, int ldc,
                              int m_begin, int m_end) {
#if HAS_NEON
    matmul_fp32_neon_8x8_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin,
                               m_end);
#else
    matmul_fp32_scalar_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin,
                             m_end);
#endif
}

// FP16 variant: B is __fp16*, A and C are float*.
static void matmul_fp16_range(const float* A, const __fp16* B, float* C, int M,
                              int N, int K, int lda, int K_weight, int ldc,
                              int m_begin, int m_end) {
#if HAS_NEON
    matmul_fp16_neon_8x8_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin,
                               m_end);
#else
    // Scalar fallback: convert each FP16 to FP32 on the fly.
    for (int m = m_begin; m < m_end; m++) {
        float* c_row = C + m * ldc;
        for (int n = 0; n < N; n++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += A[k + m * lda] * (float)B[n * K_weight + k];
            }
            c_row[n] = sum;
        }
    }
#endif
}

// Like matmul_fp32_range but shards by N (output dimension) instead of M.
// Used when N >> M (e.g. lm_head where M=1, N=vocab_size).
static void matmul_fp32_range_n(const float* A, const float* B, float* C, int M,
                                int N, int K, int lda, int K_weight, int ldc,
                                int n_begin, int n_end) {
    (void)N;
#if HAS_NEON
    // Decompose into the existing 8x8 NEON kernel by limiting N range.
    // We pass the full M range but only the [n_begin, n_end) columns of C.
    // The NEON kernel writes to C[row * ldc + col], so we offset C by n_begin.
    matmul_fp32_neon_8x8_range(A, B + n_begin * K_weight, C + n_begin, M,
                               n_end - n_begin, K, lda, K_weight, ldc, 0, M);
#else
    matmul_fp32_scalar_range(A, B + n_begin * K_weight, C + n_begin, M,
                             n_end - n_begin, K, lda, K_weight, ldc, 0, M);
#endif
}

// ---------------------------------------------------------------------------
// Dense FP32/FP16 dispatch
// ---------------------------------------------------------------------------

void matmul_dispatch_dense(const Tensor& A, const Tensor& B, Tensor& C,
                           ThreadPool* thread_pool, Activation act,
                           int act_n_begin, int act_n_len,
                           MatmulTimer& _timer) {
    int M = (int)A.shape[1];
    int K = (int)A.shape[0];
    int N = (int)B.shape[0];

    int lda = (int)(A.stride[1] / sizeof(float));
    int ldc = (int)(C.stride[1] / sizeof(float));
    int K_weight = (int)B.shape[1];
    const float* a_ptr = A.ptr<float>();
    float* c_ptr = C.ptr<float>();

    // Quantized weights are dispatched by kernel_matmul_fp32 before reaching
    // this dense implementation.
    bool is_fp16 = (B.prec == Precision::FP16);
    const __fp16* b_fp16 =
        is_fp16 ? reinterpret_cast<const __fp16*>(B.data) : nullptr;
    const float* b_fp32 = is_fp16 ? nullptr : B.ptr<float>();

    // K_weight is the stride between consecutive k rows in the repacked layout.
    // For repacked [K, N]: K_weight = original N.
    // For non-repacked [N, K]: K_weight = original K (the inner dim of B).
    // We determine this by comparing K_weight with K: if they differ, it's
    // repacked.
    bool is_repacked = (K_weight != K);

    // ---- Interleaved packing path (FP16 + NEON) ----
    // B is pre-packed at load time (engine) or by the caller (bench/test).
    // A is packed per-call (small overhead, M×K << N×K).
    // - M >= 8: lane-FMA kernel (A+B both packed)
    //   - FP16 accumulate: vfmaq_lane_f16 (2x throughput, default)
    //   - FP32 widen accumulate: vfmlalq_laneq_f16 (precision fallback)
    // - M <  8: scalar-A kernel (vfmaq_n_f32, only B packed) — GEMV path
    // Accelerate's SGEMM reaches the Apple AMX path for large prefill
    // matrices. Like llama.cpp's BLAS backend, convert the immutable FP16
    // weight to a temporary FP32 plane per call. mollm's NEON kernel wins for
    // short prompts; model-level sweeps put the crossover at roughly M=96.
#if defined(__APPLE__)
    constexpr int accelerate_min_m = 96;
    if (is_fp16 && B.rowmajor_data && M >= accelerate_min_m && N >= 32 &&
        K >= 32) {
        int n_threads = thread_pool ? thread_pool->num_threads() : 1;
        _timer.set_shape("accelerate_sgemm", M, N, K, 0, 0, false, false,
                         n_threads);
        const __fp16* src = reinterpret_cast<const __fp16*>(B.rowmajor_data);
        std::vector<float> bf((size_t)N * K);
        auto convert_rows = [&](int, int begin, int end) {
            for (int n = begin; n < end; ++n) {
                const __fp16* in = src + (size_t)n * K;
                float* out = bf.data() + (size_t)n * K;
                int k = 0;
#if HAS_NEON
                for (; k + 7 < K; k += 8) {
                    float16x8_t h = vld1q_f16(in + k);
                    vst1q_f32(out + k, vcvt_f32_f16(vget_low_f16(h)));
                    vst1q_f32(out + k + 4, vcvt_f32_f16(vget_high_f16(h)));
                }
#endif
                for (; k < K; ++k)
                    out[k] = (float)in[k];
            }
        };
        if (thread_pool && n_threads > 1) {
            int chunk = std::max(1, (N + n_threads - 1) / n_threads);
            thread_pool->parallel_for(0, N, chunk, convert_rows);
        } else
            convert_rows(0, 0, N);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.f,
                    a_ptr, lda, bf.data(), K, 0.f, c_ptr, ldc);
        if (act != Activation::NONE && act_n_len != 0) {
            if (thread_pool && n_threads > 1) {
                int chunk = std::max(1, (M + n_threads - 1) / n_threads);
                thread_pool->parallel_for(
                    0, M, chunk, [&](int, int begin, int end) {
                        matmul_apply_activation(c_ptr, M, N, ldc, begin, end,
                                                act, act_n_begin, act_n_len);
                    });
            } else
                matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act,
                                        act_n_begin, act_n_len);
        }
        return;
    }
#endif
    bool use_interleave = is_fp16 && HAS_NEON && !is_repacked &&
                          g_matmul_config.use_interleave_pack;
    bool use_lane_fma = use_interleave && (M >= 8);
    bool use_fp16_acc =
        g_matmul_config.use_fp16_accumulate && !g_mollm_force_fp32_acc;

    // Select lane-FMA kernel based on accumulation mode.
    // `act`/`act_n_begin`/`act_n_len` are in local shard coords (caller already
    // translated global → local).
    auto dispatch_lane_fma = [&](const __fp16* a_packed, const __fp16* b_packed,
                                 float* c, int m_len, int n_len, int ldc_,
                                 int m_begin_, int m_end_, Activation act_,
                                 int act_n_begin_, int act_n_len_) {
        if (use_fp16_acc) {
            matmul_fp16_neon_8x8_range_packed_a_fp16acc(
                a_packed, b_packed, c, m_len, n_len, K, ldc_, m_begin_, m_end_,
                act_, act_n_begin_, act_n_len_);
        } else {
            // FP32 acc lane-FMA kernel — activation not yet supported here.
            // Fall back: run kernel without act, then apply act in a separate
            // pass.
            matmul_fp16_neon_8x8_range_packed_a(
                a_packed, b_packed, c, m_len, n_len, K, ldc_, m_begin_, m_end_);
            if (act_ != Activation::NONE && act_n_len_ != 0) {
                matmul_apply_activation(c, m_len, n_len, ldc_, m_begin_, m_end_,
                                        act_, act_n_begin_, act_n_len_);
            }
        }
    };

    // Helper: apply activation post-hoc to a [M, N] C row-major block,
    // for columns in [act_n_begin, act_n_begin + act_n_len).
    // Used by FP32 acc fallback kernels that don't fuse activation in
    // writeback. (Defined below as a lambda for closure over nothing — pure
    // function.)

    if (use_interleave) {
        constexpr int tile_m = HAS_NEON ? 8 : 1;
        int n_threads = thread_pool ? thread_pool->num_threads() : 1;
        bool shard_by_n = (N > M * 8 && M == 1);
        int chunk_size = tile_m;
        if (M == 1 || N == 1)
            chunk_size = g_matmul_config.gemv_chunk_size;

        int total_dim = shard_by_n ? N : M;
        int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
        bool use_parallel = n_threads > 1 && n_chunks > 1;

        const __fp16* b_packed = b_fp16;

        // ---- GEMV path: M == 1, dedicated kernel ----
        if (M == 1) {
            _timer.set_shape(use_fp16_acc ? "fp16_gemv_interleaved_fp16acc"
                                          : "fp16_gemv_interleaved_fp32acc",
                             M, N, K, 0, 0, false, false, n_threads);
            if (use_fp16_acc) {
                if (!use_parallel) {
                    matmul_fp16_neon_gemv_range_fp16acc(a_ptr, b_packed, c_ptr,
                                                        K, 0, N, act,
                                                        act_n_begin, act_n_len);
                } else {
                    int n_chunk = std::max(N / (n_threads * 8), 64);
                    n_chunk = ((n_chunk + 7) / 8) * 8;
                    thread_pool->parallel_for(
                        0, N, n_chunk, [&](int, int n_begin, int n_end) {
                            // Translate global act range to local shard coords.
                            // GEMV kernel uses absolute n indices (C + n_begin
                            // offset is added by kernel itself), so pass global
                            // range intersected with shard.
                            const auto local = local_activation_range(
                                act_n_begin, act_n_len, n_begin, n_end);
                            matmul_fp16_neon_gemv_range_fp16acc(
                                a_ptr, b_packed, c_ptr, K, n_begin, n_end, act,
                                n_begin + local.begin, local.length);
                        });
                }
            } else {
                if (!use_parallel) {
                    matmul_fp16_neon_gemv_range(a_ptr, b_packed, c_ptr, K, 0,
                                                N);
                    if (act != Activation::NONE && act_n_len != 0) {
                        matmul_apply_activation_gemv(c_ptr, N, act, act_n_begin,
                                                     act_n_len);
                    }
                } else {
                    int n_chunk = std::max(N / (n_threads * 8), 64);
                    n_chunk = ((n_chunk + 7) / 8) * 8;
                    thread_pool->parallel_for(
                        0, N, n_chunk, [&](int, int n_begin, int n_end) {
                            matmul_fp16_neon_gemv_range(a_ptr, b_packed, c_ptr,
                                                        K, n_begin, n_end);
                            const auto local = local_activation_range(
                                act_n_begin, act_n_len, n_begin, n_end);
                            matmul_apply_activation_gemv(
                                c_ptr + n_begin, n_end - n_begin, act,
                                local.begin, local.length);
                        });
                }
            }
            return;
        }

        _timer.set_shape(use_lane_fma
                             ? (use_fp16_acc ? "fp16_gemm_interleaved_fp16acc"
                                             : "fp16_gemm_interleaved_fp32acc")
                             : "fp16_gemm_interleaved_scalar_a",
                         M, N, K, 0, 0, false, false, n_threads);

        std::unique_ptr<__fp16[]> a_packed;
        if (use_lane_fma && !use_parallel) {
            a_packed = pack_a_interleaved_full(a_ptr, M, K, lda);
        }

        if (!use_parallel) {
            if (use_lane_fma) {
                dispatch_lane_fma(a_packed.get(), b_packed, c_ptr, M, N, ldc,
                                  0, M, act, act_n_begin, act_n_len);
            } else {
                matmul_fp16_neon_8x8_range_packed(a_ptr, b_packed, c_ptr, M, N,
                                                  K, lda, ldc, 0, M);
                if (act != Activation::NONE && act_n_len != 0) {
                    matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act,
                                            act_n_begin, act_n_len);
                }
            }
        } else if (shard_by_n) {
            // GEMV-ish: shard_by_n with M<8 (use the scalar-A kernel).
            int n_chunk = std::max(chunk_size, 8);
            n_chunk = ((n_chunk + 7) / 8) * 8;
            thread_pool->parallel_for(
                0, N, n_chunk, [&](int, int n_begin, int n_end) {
                    int n_begin_aligned = n_begin & ~7;
                    matmul_fp16_neon_8x8_range_packed(
                        a_ptr, b_packed + n_begin_aligned * K, c_ptr + n_begin,
                        M, n_end - n_begin, K, lda, ldc, 0, M);
                    const auto local = local_activation_range(
                        act_n_begin, act_n_len, n_begin, n_end);
                    if (act != Activation::NONE && local.length != 0) {
                        matmul_apply_activation(c_ptr + n_begin, M,
                                                n_end - n_begin, ldc, 0, M, act,
                                                local.begin, local.length);
                    }
                });
        } else {
            // GEMM (M>=8): 2D atomic-steal scheduling.
            // Jobs = (M/8) × (N/N_BLOCK), each job = 8 rows × N_BLOCK cols.
            // N_BLOCK=256 reduces job count + atomic contention.
            constexpr int N_BLOCK = 256;
            int n_block = ((N_BLOCK + 7) / 8) * 8; // align to 8
            if (n_block > N)
                n_block = ((N + 7) / 8) * 8;

            // Pre-pack A per M-tile ONCE (not per job).
            int m_tiles = (M + tile_m - 1) / tile_m;
            std::vector<std::unique_ptr<__fp16[]>> a_packed_tiles(m_tiles);
            for (int i = 0; i < m_tiles; i++) {
                int m_begin = i * tile_m;
                int m_len = std::min(tile_m, M - m_begin);
                a_packed_tiles[i] = pack_a_interleaved_full(
                    a_ptr + m_begin * lda, m_len, K, lda);
            }

            thread_pool->parallel_for_2d(
                M, tile_m, N, n_block,
                [&](int, int m_begin, int m_end, int n_begin, int n_end) {
                    int m_len = m_end - m_begin;
                    int n_len = n_end - n_begin;
                    int tile_idx = m_begin / tile_m;
                    const __fp16* a_slice_packed =
                        a_packed_tiles[tile_idx].get();
                    int n_begin_aligned = n_begin & ~7;

                    // Translate global act range to local shard coords.
                    const auto local = local_activation_range(
                        act_n_begin, act_n_len, n_begin, n_end);

                    dispatch_lane_fma(
                        a_slice_packed, b_packed + n_begin_aligned * K,
                        c_ptr + m_begin * ldc + n_begin, m_len, n_len, ldc, 0,
                        m_len, act, local.begin, local.length);
                });
        }
        return;
    }

    // ---- Standard path (FP32 or non-packed FP16) ----

    constexpr int tile_m = HAS_NEON ? 8 : 1;
    int n_threads = thread_pool ? thread_pool->num_threads() : 1;

    // Decide sharding dimension adaptively, similar to ggml:
    //   If N >> M, shard by N (e.g. lm_head: M=1, N=vocab_size).
    //   Otherwise shard by M (the common case).
    bool shard_by_n = (N > M * 8 && M == 1);

    // Decide chunk size adaptively.
    // For GEMV-like shapes, use a larger chunk to reduce per-chunk overhead.
    int chunk_size = tile_m;
    if (M == 1 || N == 1) {
        chunk_size = g_matmul_config.gemv_chunk_size;
    }

    int total_dim = shard_by_n ? N : M;
    int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
    bool use_parallel = n_threads > 1 && n_chunks > 1;

    _timer.set_shape(
        is_fp16 ? (is_repacked ? "fp16_standard_repacked" : "fp16_standard")
                : "fp32_standard",
        M, N, K, 0, 0, false, false, n_threads);

    if (!use_parallel) {
        if (is_fp16) {
            matmul_fp16_range(a_ptr, b_fp16, c_ptr, M, N, K, lda, K_weight, ldc,
                              0, M);
        } else {
            matmul_fp32_range(a_ptr, b_fp32, c_ptr, M, N, K, lda, K_weight, ldc,
                              0, M);
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act, act_n_begin,
                                    act_n_len);
        }
        return;
    }

    if (shard_by_n) {
        if (is_fp16) {
            thread_pool->parallel_for(
                0, N, chunk_size, [&](int, int n_begin, int n_end) {
                    matmul_fp16_range(a_ptr, b_fp16 + n_begin * K_weight,
                                      c_ptr + n_begin, M, n_end - n_begin, K,
                                      lda, K_weight, ldc, 0, M);
                    const auto local = local_activation_range(
                        act_n_begin, act_n_len, n_begin, n_end);
                    if (act != Activation::NONE && local.length != 0) {
                        matmul_apply_activation(c_ptr + n_begin, M,
                                                n_end - n_begin, ldc, 0, M, act,
                                                local.begin, local.length);
                    }
                });
        } else {
            thread_pool->parallel_for(
                0, N, chunk_size, [&](int, int n_begin, int n_end) {
                    matmul_fp32_range_n(a_ptr, b_fp32, c_ptr, M, N, K, lda,
                                        K_weight, ldc, n_begin, n_end);
                    const auto local = local_activation_range(
                        act_n_begin, act_n_len, n_begin, n_end);
                    if (act != Activation::NONE && local.length != 0) {
                        matmul_apply_activation(c_ptr + n_begin, M,
                                                n_end - n_begin, ldc, 0, M, act,
                                                local.begin, local.length);
                    }
                });
        }
    } else {
        if (is_fp16) {
            thread_pool->parallel_for(
                0, M, chunk_size, [&](int, int m_begin, int m_end) {
                    matmul_fp16_range(a_ptr, b_fp16, c_ptr, M, N, K, lda,
                                      K_weight, ldc, m_begin, m_end);
                });
        } else {
            thread_pool->parallel_for(
                0, M, chunk_size, [&](int, int m_begin, int m_end) {
                    matmul_fp32_range(a_ptr, b_fp32, c_ptr, M, N, K, lda,
                                      K_weight, ldc, m_begin, m_end);
                });
        }
    }

    // M-sharded path: activation can't be applied per-shard (shards split M,
    // not N). Apply once after all shards complete.
    // (shard_by_n branches above already apply per-shard.)
    if (!shard_by_n && act != Activation::NONE && act_n_len != 0) {
        matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act, act_n_begin,
                                act_n_len);
    }
}
