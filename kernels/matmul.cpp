#include "kernels/matmul.h"

#include <algorithm>
#include <cstring>

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
// Weight file stores row-major [K, N]: data[k*N + n] = weight[n,k].
// Tensor access: B.at<float>(n, k) = data[n + k*N] = data[k*N + n].
// C[m,n] = sum_k A[k + m*lda] * B[n + k*ldb] = sum_k A[k,m] * W[n,k] = (A @ W^T)[m,n]
static void matmul_fp32_scalar(const float* A, const float* B, float* C,
                               int M, int N, int K,
                               int lda, int K_weight, int ldc) {
    for (int m = 0; m < M; m++) {
        float* c_row = C + m * ldc;
        for (int n = 0; n < N; n++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                // A[k,m] * W[n,k] = A[k + m*lda] * B[n*K_weight + k]
                sum += A[k + m * lda] * B[n * K_weight + k];
            }
            c_row[n] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// NEON matmul — TILE_M=4, TILE_N=4, three-phase K loop
//
// The 4x4 tile is accumulated into 4 float32x4_t registers:
//   c0[n] = A_row0 · B_col_n     (for n=0..3)
//   c1[n] = A_row1 · B_col_n
//   c2[n] = A_row2 · B_col_n
//   c3[n] = A_row3 · B_col_n
//
// Each register's lane j holds the dot product for column (n+j).
// ---------------------------------------------------------------------------
#if HAS_NEON

static void matmul_fp32_neon_4x4(const float* A, const float* B, float* C,
                                 int M, int N, int K,
                                 int lda, int K_weight, int ldc) {
    for (int m = 0; m < M; m += 4) {
        int m_end = std::min(m + 4, M);

        for (int n = 0; n < N; n += 4) {
            int n_end = std::min(n + 4, N);

            // accumulators for 4 rows, each holding up to 4 columns
            float32x4_t c0 = vdupq_n_f32(0.f);
            float32x4_t c1 = vdupq_n_f32(0.f);
            float32x4_t c2 = vdupq_n_f32(0.f);
            float32x4_t c3 = vdupq_n_f32(0.f);

            // ---- K loop ----
            for (int k = 0; k < K; k++) {
                // Weight is [N, K_weight] row-major: W[n,k] = data[n*K_weight + k]
                // Load 4 elements for columns n..n+3, row k
                float32x4_t b_vec;
                {
                    float tmp[4];
                    for (int j = 0; j < 4 && n + j < n_end; j++) {
                        tmp[j] = B[(n + j) * K_weight + k];
                    }
                    b_vec = vld1q_f32(tmp);
                }

                // Row 0
                if (m + 0 < m_end) {
                    float a0 = A[k + (m + 0) * lda];
                    c0 = vfmaq_n_f32(c0, b_vec, a0);
                }
                // Row 1
                if (m + 1 < m_end) {
                    float a1 = A[k + (m + 1) * lda];
                    c1 = vfmaq_n_f32(c1, b_vec, a1);
                }
                // Row 2
                if (m + 2 < m_end) {
                    float a2 = A[k + (m + 2) * lda];
                    c2 = vfmaq_n_f32(c2, b_vec, a2);
                }
                // Row 3
                if (m + 3 < m_end) {
                    float a3 = A[k + (m + 3) * lda];
                    c3 = vfmaq_n_f32(c3, b_vec, a3);
                }
            }

            // write back
            float tmp[4];
            if (m + 0 < m_end) { vst1q_f32(tmp, c0); for (int j = 0; j < n_end - n; j++) C[(m + 0) * ldc + n + j] = tmp[j]; }
            if (m + 1 < m_end) { vst1q_f32(tmp, c1); for (int j = 0; j < n_end - n; j++) C[(m + 1) * ldc + n + j] = tmp[j]; }
            if (m + 2 < m_end) { vst1q_f32(tmp, c2); for (int j = 0; j < n_end - n; j++) C[(m + 2) * ldc + n + j] = tmp[j]; }
            if (m + 3 < m_end) { vst1q_f32(tmp, c3); for (int j = 0; j < n_end - n; j++) C[(m + 3) * ldc + n + j] = tmp[j]; }
        }
    }
}

#endif // HAS_NEON

// ---------------------------------------------------------------------------
// kernel_matmul_fp32
// ---------------------------------------------------------------------------

void kernel_matmul_fp32(const Tensor& A, const Tensor& B, Tensor& C) {
    // A: [K, M] — input activations (K=features, M=seq_len)
    // B: [N, K] — weight matrix (N=output, K=input), stored row-major
    // C: [N, M] — output
    // Compute: C[m,n] = sum_k A[k,m] * W[n,k]
    int M = (int)A.shape[1];  // seq_len
    int K = (int)A.shape[0];  // features
    // N = B.shape[0] (weight is [N, K])
    int N = (int)B.shape[0];
    
    int lda = (int)(A.stride[1] / sizeof(float));
    int ldc = (int)(C.stride[1] / sizeof(float));
    
    // Weight stride: B is [N, K] row-major, so W[n,k] = data[n*K + k]
    int K_weight = (int)B.shape[1];

    const float* a_ptr = A.ptr<float>();
    const float* b_ptr = B.ptr<float>();
    float*       c_ptr = C.ptr<float>();

    // zero output
    for (int i = 0; i < M; i++) {
        std::memset(c_ptr + i * ldc, 0, N * sizeof(float));
    }

#if HAS_NEON
    matmul_fp32_neon_4x4(a_ptr, b_ptr, c_ptr, M, N, K, lda, K_weight, ldc);
#else
    matmul_fp32_scalar(a_ptr, b_ptr, c_ptr, M, N, K, lda, K_weight, ldc);
#endif

#if HAS_NEON
    matmul_fp32_neon_4x4(a_ptr, b_ptr, c_ptr, M, N, K, lda, K_weight, ldc);
#else
    matmul_fp32_scalar(a_ptr, b_ptr, c_ptr, M, N, K, lda, K_weight, ldc);
#endif
}
