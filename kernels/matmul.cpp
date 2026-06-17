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
    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);
        int m_global_end = std::min(m + 4, M);

        for (int n = 0; n < N; n += 4) {
            int n_end = std::min(n + 4, N);

            float32x4_t c0 = vdupq_n_f32(0.f);
            float32x4_t c1 = vdupq_n_f32(0.f);
            float32x4_t c2 = vdupq_n_f32(0.f);
            float32x4_t c3 = vdupq_n_f32(0.f);

            for (int k = 0; k < K; k++) {
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

    // Decide chunk size adaptively, similar to ggml's approach.
    // For GEMV-like shapes (M==1 or N==1), use a larger chunk to reduce
    // per-chunk overhead.  Otherwise keep chunks small so work is well
    // distributed.
    int chunk_size = tile_m;
    if (M == 1 || N == 1) {
        chunk_size = 64;
    }

    int n_chunks = (M + chunk_size - 1) / chunk_size;
    bool use_parallel = n_threads > 1 && n_chunks > 1;

    if (!use_parallel) {
        matmul_fp32_range(a_ptr, b_ptr, c_ptr, M, N, K, lda, K_weight, ldc, 0, M);
        return;
    }

    thread_pool->parallel_for(0, M, chunk_size,
                              [&](int, int m_begin, int m_end) {
                                  matmul_fp32_range(a_ptr, b_ptr, c_ptr,
                                                    M, N, K, lda, K_weight, ldc,
                                                    m_begin, m_end);
                              });
}
