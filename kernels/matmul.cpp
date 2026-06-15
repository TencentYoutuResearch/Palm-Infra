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

static void matmul_fp32_scalar(const float* A, const float* B, float* C,
                               int M, int N, int K,
                               int lda, int ldb, int ldc) {
    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * lda;
        float*       c_row = C + m * ldc;
        for (int n = 0; n < N; n++) {
            float sum = 0.f;
            const float* b_col = B + n;
            for (int k = 0; k < K; k++) {
                sum += a_row[k] * b_col[k * ldb];
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
                                 int lda, int ldb, int ldc) {
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
                // Load 4 elements from B: B[k, n], B[k, n+1], B[k, n+2], B[k, n+3]
                // B is K×N row-major, so row k starts at B + k*ldb
                float32x4_t b_vec;
                {
                    const float* b_row = B + k * ldb + n;
                    if (n + 4 <= N) {
                        b_vec = vld1q_f32(b_row);
                    } else {
                        // partial tile — pad with zeros
                        float tmp[4] = {0, 0, 0, 0};
                        for (int j = 0; j < n_end - n; j++) tmp[j] = b_row[j];
                        b_vec = vld1q_f32(tmp);
                    }
                }

                // Row 0
                if (m + 0 < m_end) {
                    float a0 = A[(m + 0) * lda + k];
                    c0 = vfmaq_n_f32(c0, b_vec, a0);
                }
                // Row 1
                if (m + 1 < m_end) {
                    float a1 = A[(m + 1) * lda + k];
                    c1 = vfmaq_n_f32(c1, b_vec, a1);
                }
                // Row 2
                if (m + 2 < m_end) {
                    float a2 = A[(m + 2) * lda + k];
                    c2 = vfmaq_n_f32(c2, b_vec, a2);
                }
                // Row 3
                if (m + 3 < m_end) {
                    float a3 = A[(m + 3) * lda + k];
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
    // shapes: A[M,K] * B[K,N] → C[M,N]
    int M = (int)A.shape[1];  // rows of A
    int K = (int)A.shape[0];  // inner dim
    int N = (int)B.shape[0];  // cols of B

    int lda = (int)(A.stride[1] / sizeof(float));
    int ldb = (int)(B.stride[1] / sizeof(float));
    int ldc = (int)(C.stride[1] / sizeof(float));

    const float* a_ptr = A.ptr<float>();
    const float* b_ptr = B.ptr<float>();
    float*       c_ptr = C.ptr<float>();

    // zero output
    for (int i = 0; i < M; i++) {
        std::memset(c_ptr + i * ldc, 0, N * sizeof(float));
    }

#if HAS_NEON
    matmul_fp32_neon_4x4(a_ptr, b_ptr, c_ptr, M, N, K, lda, ldb, ldc);
#else
    matmul_fp32_scalar(a_ptr, b_ptr, c_ptr, M, N, K, lda, ldb, ldc);
#endif
}
