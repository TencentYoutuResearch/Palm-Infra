#include "kernels/matmul.h"
#include "kernels/threading.h"

#include <algorithm>

// ---------------------------------------------------------------------------
// NEON intrinsics (ARM only)
// ---------------------------------------------------------------------------
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

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
// NEON matmul — TILE_M=4, TILE_N=4
// ---------------------------------------------------------------------------
#if HAS_NEON

static void matmul_fp32_neon_4x4_range(const float* A, const float* B, float* C,
                                       int M, int N, int K,
                                       int lda, int K_weight, int ldc,
                                       int m_begin, int m_end) {
    // K-blocking: split K into blocks that fit in L1 cache.
    // A typical L1 data cache is 32-64 KB.  Each K-step touches 4 rows of A
    // (one scalar per row) and up to 4 columns of B (4 floats).  Keeping
    // the working set around 16 KB is safe.  With float32 elements:
    //   16 KB / 4 bytes = 4096 elements
    //   per K-step: B loads 4 floats  → 16 bytes
    //               A loads 1 scalar  → 4 bytes
    //   4096 / 16 ≈ 256, so K_BLOCK = 256 is a reasonable default.
    constexpr int K_BLOCK = 256;

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK) {
        int k_end = std::min(k_outer + K_BLOCK, K);

        for (int m = m_begin; m < m_end; m += 4) {
            int m_tile_end = std::min(m + 4, m_end);
            int m_global_end = std::min(m + 4, M);

            for (int n = 0; n < N; n += 4) {
                int n_end = std::min(n + 4, N);

                // For the first K block, zero the accumulators.
                // For subsequent blocks, reload the partial sums so we can
                // continue accumulating.
                float32x4_t c0, c1, c2, c3;
                bool first_block = (k_outer == 0);
                if (first_block) {
                    c0 = vdupq_n_f32(0.f);
                    c1 = vdupq_n_f32(0.f);
                    c2 = vdupq_n_f32(0.f);
                    c3 = vdupq_n_f32(0.f);
                } else {
                    float tmp[4];
                    int row0 = m + 0, row1 = m + 1, row2 = m + 2, row3 = m + 3;
                    if (row0 < m_tile_end && row0 < m_global_end) {
                        for (int j = 0; j < 4 && n + j < n_end; j++) tmp[j] = C[row0 * ldc + n + j];
                        c0 = vld1q_f32(tmp);
                    } else { c0 = vdupq_n_f32(0.f); }
                    if (row1 < m_tile_end && row1 < m_global_end) {
                        for (int j = 0; j < 4 && n + j < n_end; j++) tmp[j] = C[row1 * ldc + n + j];
                        c1 = vld1q_f32(tmp);
                    } else { c1 = vdupq_n_f32(0.f); }
                    if (row2 < m_tile_end && row2 < m_global_end) {
                        for (int j = 0; j < 4 && n + j < n_end; j++) tmp[j] = C[row2 * ldc + n + j];
                        c2 = vld1q_f32(tmp);
                    } else { c2 = vdupq_n_f32(0.f); }
                    if (row3 < m_tile_end && row3 < m_global_end) {
                        for (int j = 0; j < 4 && n + j < n_end; j++) tmp[j] = C[row3 * ldc + n + j];
                        c3 = vld1q_f32(tmp);
                    } else { c3 = vdupq_n_f32(0.f); }
                }

                for (int k = k_outer; k < k_end; k++) {
                    float tmp_b[4] = {0.f, 0.f, 0.f, 0.f};
                    for (int j = 0; j < 4 && n + j < n_end; j++) {
                        tmp_b[j] = B[(n + j) * K_weight + k];
                    }
                    float32x4_t b_vec = vld1q_f32(tmp_b);

                    if (m + 0 < m_tile_end && m + 0 < m_global_end) {
                        c0 = vfmaq_n_f32(c0, b_vec, A[k + (m + 0) * lda]);
                    }
                    if (m + 1 < m_tile_end && m + 1 < m_global_end) {
                        c1 = vfmaq_n_f32(c1, b_vec, A[k + (m + 1) * lda]);
                    }
                    if (m + 2 < m_tile_end && m + 2 < m_global_end) {
                        c2 = vfmaq_n_f32(c2, b_vec, A[k + (m + 2) * lda]);
                    }
                    if (m + 3 < m_tile_end && m + 3 < m_global_end) {
                        c3 = vfmaq_n_f32(c3, b_vec, A[k + (m + 3) * lda]);
                    }
                }

                // Write back after every K block so the output is always
                // up-to-date (needed because the next block will reload).
                float tmp_out[4];
                if (m + 0 < m_tile_end && m + 0 < m_global_end) {
                    vst1q_f32(tmp_out, c0);
                    for (int j = 0; j < n_end - n; j++) C[(m + 0) * ldc + n + j] = tmp_out[j];
                }
                if (m + 1 < m_tile_end && m + 1 < m_global_end) {
                    vst1q_f32(tmp_out, c1);
                    for (int j = 0; j < n_end - n; j++) C[(m + 1) * ldc + n + j] = tmp_out[j];
                }
                if (m + 2 < m_tile_end && m + 2 < m_global_end) {
                    vst1q_f32(tmp_out, c2);
                    for (int j = 0; j < n_end - n; j++) C[(m + 2) * ldc + n + j] = tmp_out[j];
                }
                if (m + 3 < m_tile_end && m + 3 < m_global_end) {
                    vst1q_f32(tmp_out, c3);
                    for (int j = 0; j < n_end - n; j++) C[(m + 3) * ldc + n + j] = tmp_out[j];
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
    matmul_fp32_neon_4x4_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin, m_end);
#else
    matmul_fp32_scalar_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin, m_end);
#endif
}

// Like matmul_fp32_range but shards by N (output dimension) instead of M.
// Used when N >> M (e.g. lm_head where M=1, N=vocab_size).
static void matmul_fp32_range_n(const float* A, const float* B, float* C,
                                int M, int N, int K,
                                int lda, int K_weight, int ldc,
                                int n_begin, int n_end) {
#if HAS_NEON
    // Decompose into the existing 4x4 NEON kernel by limiting N range.
    // We pass the full M range but only the [n_begin, n_end) columns of C.
    // The NEON kernel writes to C[row * ldc + col], so we offset C by n_begin.
    matmul_fp32_neon_4x4_range(A, B + n_begin * K_weight, C + n_begin,
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
    const float* b_ptr = B.ptr<float>();
    float* c_ptr = C.ptr<float>();

    constexpr int tile_m = HAS_NEON ? 4 : 1;
    int n_threads = thread_pool ? thread_pool->num_threads() : 1;

    // Decide sharding dimension adaptively, similar to ggml:
    //   If N >> M, shard by N (e.g. lm_head: M=1, N=vocab_size).
    //   Otherwise shard by M (the common case).
    bool shard_by_n = (N > M * 8 && M == 1);

    // Decide chunk size adaptively.
    // For GEMV-like shapes, use a larger chunk to reduce per-chunk overhead.
    int chunk_size = tile_m;
    if (M == 1 || N == 1) {
        chunk_size = 64;
    }

    int total_dim = shard_by_n ? N : M;
    int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
    bool use_parallel = n_threads > 1 && n_chunks > 1;

    if (!use_parallel) {
        matmul_fp32_range(a_ptr, b_ptr, c_ptr, M, N, K, lda, K_weight, ldc, 0, M);
        return;
    }

    if (shard_by_n) {
        thread_pool->parallel_for(0, N, chunk_size,
                                  [&](int, int n_begin, int n_end) {
                                      matmul_fp32_range_n(a_ptr, b_ptr, c_ptr,
                                                          M, N, K, lda, K_weight, ldc,
                                                          n_begin, n_end);
                                  });
    } else {
        thread_pool->parallel_for(0, M, chunk_size,
                                  [&](int, int m_begin, int m_end) {
                                      matmul_fp32_range(a_ptr, b_ptr, c_ptr,
                                                        M, N, K, lda, K_weight, ldc,
                                                        m_begin, m_end);
                                  });
    }
}
