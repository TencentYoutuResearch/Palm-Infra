#include "kernels/matmul.h"
#include "kernels/threading.h"

#include <algorithm>

MatmulConfig g_matmul_config;

// ---------------------------------------------------------------------------
// scalar matmul (fallback)
// ---------------------------------------------------------------------------

// After weight dim swap: B has shape [N, K] (N=output, K=input).
// Weight file stores row-major [N, K]: data[n*K + k] = W[n, k].
// Tensor access: B.at<float>(n, k) = data[n*K + k].
// C[m,n] = sum_k A[k + m*lda] * B[n*K_weight + k] = A[m,:] @ W[n,:].
static void matmul_fp32_scalar_range(const float* A, const float* B, float* C,
                                     int M, int N, int K,
                                     int lda, int K_weight, int ldc,
                                     int m_begin, int m_end) {
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

// ---------------------------------------------------------------------------
// NEON matmul — TILE_M=8, TILE_N=8, FP16 storage + FP32 accumulate
//
// B is stored as float16 ([N, K] or [K, N] repacked).
// Each inner iteration loads 8 float16 values (one vld1q_f16 or gather),
// converts to float32 via vcvt_f32_f16, then FMA into float32 accumulators.
// This halves the memory bandwidth for B compared to FP32 storage.
//
// For gather (row-major) layout, we load via uint16x4_t + vget_lane + vcvt
// to avoid stack temporaries.
// ---------------------------------------------------------------------------
#if HAS_NEON

// ---------------------------------------------------------------------------
// B interleaved packing — transform B[N,K] to tile-of-8 transposed layout.
//
// For each N-tile of 8 rows, transpose so that for fixed k,
// B_packed[tile_base + k*8 + 0..7] are 8 consecutive FP16 values.
// This enables vld1q_f16 contiguous load instead of strided gather.
//
// Full-matrix version: pack entire B [N, K] → interleaved [N/8, K, 8].
// Caller owns the returned buffer (must delete[]).
// ---------------------------------------------------------------------------
__fp16* pack_b_interleaved_full(const __fp16* B_original, int N, int K, int K_weight) {
    __fp16* dst = new __fp16[(size_t)N * K];
    for (int n_tile = 0; n_tile < N; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < tile_valid; j++) {
                dst[n_tile * K + k * 8 + j] = B_original[(n_tile + j) * K_weight + k];
            }
            for (int j = tile_valid; j < 8; j++) {
                dst[n_tile * K + k * 8 + j] = (__fp16)0.f;
            }
        }
    }
    return dst;
}

static void matmul_fp16_neon_8x8_range(const float* A, const __fp16* B, float* C,
                                       int M, int N, int K,
                                       int lda, int K_weight, int ldc,
                                       int m_begin, int m_end) {
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
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
                        __fp16 tmp[4] = {(__fp16)0.f, (__fp16)0.f, (__fp16)0.f, (__fp16)0.f};
                        for (int j = 0; j < 4 && n + j < n_end; j++) {
                            tmp[j] = B[(n + j) * K_weight + k];
                        }
                        b0 = vcvt_f32_f16(vld1_f16(tmp));
                    }
                    {
                        __fp16 tmp[4] = {(__fp16)0.f, (__fp16)0.f, (__fp16)0.f, (__fp16)0.f};
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
                        for (int j = 0; j < 4 && n + j < n_end; j++) C[row * ldc + n + j] = tmp[j];
                        vst1q_f32(tmp, c[r][1]);
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++) C[row * ldc + n + 4 + j] = tmp[j];
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
static void matmul_fp16_neon_8x8_range_packed(
    const float* A, const __fp16* B_packed, float* C,
    int M, int N, int K,
    int lda, int ldc,
    int m_begin, int m_end)
{
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2;  // FP16: 2x cache density

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
                    float16x8_t b_vec = vld1q_f16(&B_packed[(n & ~7) * K + k * 8]);
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
                        for (int j = 0; j < 4 && n + j < n_end; j++) C[row * ldc + n + j] = tmp[j];
                        vst1q_f32(tmp, c[r][1]);
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++) C[row * ldc + n + 4 + j] = tmp[j];
                    }
                }
            }
        }
    }
}

// ---- FP32 kernel (kept for backward compat) ----
static void matmul_fp32_neon_8x8_range(const float* A, const float* B, float* C,
                                       int M, int N, int K,
                                       int lda, int K_weight, int ldc,
                                       int m_begin, int m_end) {
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK) {
        int k_end = std::min(k_outer + K_BLOCK, K);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                float32x4_t c[8][2]; // c[row][col_hi/lo]: hi=cols 0-3, lo=cols 4-7
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
                        for (int j = 0; j < 4 && n + j < n_end; j++) C[row * ldc + n + j] = tmp[j];
                        vst1q_f32(tmp, c[r][1]);
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++) C[row * ldc + n + 4 + j] = tmp[j];
                    }
                }
            }
        }
    }
}

#endif // HAS_NEON

static void matmul_fp32_range(const float* A, const float* B, float* C,
                              int M, int N, int K,
                              int lda, int K_weight, int ldc,
                              int m_begin, int m_end) {
#if HAS_NEON
    matmul_fp32_neon_8x8_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin, m_end);
#else
    matmul_fp32_scalar_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin, m_end);
#endif
}

// FP16 variant: B is __fp16*, A and C are float*.
static void matmul_fp16_range(const float* A, const __fp16* B, float* C,
                              int M, int N, int K,
                              int lda, int K_weight, int ldc,
                              int m_begin, int m_end) {
#if HAS_NEON
    matmul_fp16_neon_8x8_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin, m_end);
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
static void matmul_fp32_range_n(const float* A, const float* B, float* C,
                                int M, int N, int K,
                                int lda, int K_weight, int ldc,
                                int n_begin, int n_end) {
#if HAS_NEON
    // Decompose into the existing 8x8 NEON kernel by limiting N range.
    // We pass the full M range but only the [n_begin, n_end) columns of C.
    // The NEON kernel writes to C[row * ldc + col], so we offset C by n_begin.
    matmul_fp32_neon_8x8_range(A, B + n_begin * K_weight, C + n_begin,
                               M, n_end - n_begin, K,
                               lda, K_weight, ldc,
                               0, M);
#else
    matmul_fp32_scalar_range(A, B + n_begin * K_weight, C + n_begin,
                             M, n_end - n_begin, K,
                             lda, K_weight, ldc,
                             0, M);
#endif
}

// ---------------------------------------------------------------------------
// kernel_matmul_fp32
// ---------------------------------------------------------------------------

void kernel_matmul_fp32(const Tensor& A, const Tensor& B, Tensor& C,
                        ThreadPool* thread_pool) {
    int M = (int)A.shape[1];
    int K = (int)A.shape[0];
    int N = (int)B.shape[0];

    int lda = (int)(A.stride[1] / sizeof(float));
    int ldc = (int)(C.stride[1] / sizeof(float));
    int K_weight = (int)B.shape[1];

    const float* a_ptr = A.ptr<float>();
    float* c_ptr = C.ptr<float>();

    // Detect FP16 weight: B.prec == FP16.
    bool is_fp16 = (B.prec == Precision::FP16);
    const __fp16* b_fp16 = is_fp16 ? reinterpret_cast<const __fp16*>(B.data) : nullptr;
    const float* b_fp32 = is_fp16 ? nullptr : B.ptr<float>();

    // K_weight is the stride between consecutive k rows in the repacked layout.
    // For repacked [K, N]: K_weight = original N.
    // For non-repacked [N, K]: K_weight = original K (the inner dim of B).
    // We determine this by comparing K_weight with K: if they differ, it's repacked.
    bool is_repacked = (K_weight != K);

    // ---- Interleaved packing path (FP16 + NEON) ----
    // B is pre-packed at load time (engine) or by the caller (bench/test).
    // No per-call packing overhead — works for both GEMM and GEMV.
    bool use_interleave = is_fp16 && HAS_NEON && !is_repacked
                       && g_matmul_config.use_interleave_pack;

    if (use_interleave) {
        constexpr int tile_m = HAS_NEON ? 8 : 1;
        int n_threads = thread_pool ? thread_pool->num_threads() : 1;
        bool shard_by_n = (N > M * 8 && M == 1);
        int chunk_size = tile_m;
        if (M == 1 || N == 1) chunk_size = g_matmul_config.gemv_chunk_size;

        int total_dim = shard_by_n ? N : M;
        int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
        bool use_parallel = n_threads > 1 && n_chunks > 1;

        // B is already packed: B_packed layout is [N/8, K, 8]
        const __fp16* b_packed = b_fp16;

        if (!use_parallel) {
            matmul_fp16_neon_8x8_range_packed(
                a_ptr, b_packed, c_ptr,
                M, N, K, lda, ldc, 0, M);
        } else if (shard_by_n) {
            int n_chunk = std::max(chunk_size, 8);
            n_chunk = ((n_chunk + 7) / 8) * 8;  // round up to multiple of 8
            thread_pool->parallel_for(0, N, n_chunk,
                [&](int, int n_begin, int n_end) {
                    int n_begin_aligned = n_begin & ~7;
                    matmul_fp16_neon_8x8_range_packed(
                        a_ptr,
                        b_packed + n_begin_aligned * K,  // offset to tile start
                        c_ptr + n_begin,
                        M, n_end - n_begin,
                        K, lda, ldc, 0, M);
                });
        } else {
            thread_pool->parallel_for(0, M, chunk_size,
                [&](int, int m_begin, int m_end) {
                    matmul_fp16_neon_8x8_range_packed(
                        a_ptr, b_packed, c_ptr,
                        M, N, K, lda, ldc, m_begin, m_end);
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

    if (!use_parallel) {
        if (is_fp16) {
            matmul_fp16_range(a_ptr, b_fp16, c_ptr, M, N, K, lda, K_weight, ldc, 0, M);
        } else {
            matmul_fp32_range(a_ptr, b_fp32, c_ptr, M, N, K, lda, K_weight, ldc, 0, M);
        }
        return;
    }

    if (shard_by_n) {
        if (is_fp16) {
            thread_pool->parallel_for(0, N, chunk_size,
                                      [&](int, int n_begin, int n_end) {
                                          matmul_fp16_range(a_ptr, b_fp16 + n_begin * K_weight, c_ptr + n_begin,
                                                            M, n_end - n_begin, K, lda, K_weight, ldc, 0, M);
                                      });
        } else {
            thread_pool->parallel_for(0, N, chunk_size,
                                      [&](int, int n_begin, int n_end) {
                                          matmul_fp32_range_n(a_ptr, b_fp32, c_ptr,
                                                              M, N, K, lda, K_weight, ldc,
                                                              n_begin, n_end);
                                      });
        }
    } else {
        if (is_fp16) {
            thread_pool->parallel_for(0, M, chunk_size,
                                      [&](int, int m_begin, int m_end) {
                                          matmul_fp16_range(a_ptr, b_fp16, c_ptr,
                                                            M, N, K, lda, K_weight, ldc,
                                                            m_begin, m_end);
                                      });
        } else {
            thread_pool->parallel_for(0, M, chunk_size,
                                      [&](int, int m_begin, int m_end) {
                                          matmul_fp32_range(a_ptr, b_fp32, c_ptr,
                                                            M, N, K, lda, K_weight, ldc,
                                                            m_begin, m_end);
                                      });
        }
    }
}
