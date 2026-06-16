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
                // Load 4 elements from B: B[n, k], B[n+1, k], B[n+2, k], B[n+3, k]
                // B is N×K row-major: row n starts at B + n*ldb
                float32x4_t b_vec;
                {
                    // Gather: B[n][k], B[n+1][k], B[n+2][k], B[n+3][k]
                    if (n + 4 <= N) {
                        float b_tmp[4];
                        b_tmp[0] = B[(n+0) * ldb + k];
                        b_tmp[1] = B[(n+1) * ldb + k];
                        b_tmp[2] = B[(n+2) * ldb + k];
                        b_tmp[3] = B[(n+3) * ldb + k];
                        b_vec = vld1q_f32(b_tmp);
                    } else {
                        float tmp[4] = {0, 0, 0, 0};
                        for (int j = 0; j < n_end - n; j++) tmp[j] = B[(n+j) * ldb + k];
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
    // Detect N from B: the dimension that is NOT K
    int N;
    if ((int)B.shape[0] == K) {
        N = (int)B.shape[1];  // B.shape[0] is K, B.shape[1] is N
    } else if ((int)B.shape[1] == K) {
        N = (int)B.shape[0];  // B.shape[1] is K, B.shape[0] is N
    } else {
        N = (int)B.shape[0];  // fallback
    }

    static int call_count = 0;
    if (call_count < 3) {
        // Compute dot product manually for first 3 output elements to verify
        float manual[3] = {0,0,0};
        const float* a = A.ptr<float>();
        const float* b = B.ptr<float>();
        int ldb_elems = (int)B.stride[1] / (int)sizeof(float);
        fprintf(stderr, "  C++ MATMUL[%d] ldb_elems=%d (stride[1]=%zu)\n",
                call_count, ldb_elems, B.stride[1]);
        for (int k = 0; k < K; k++) {
            manual[0] += a[k] * b[0 * ldb_elems + k];
            manual[1] += a[k] * b[1 * ldb_elems + k];
            manual[2] += a[k] * b[2 * ldb_elems + k];
        }
        fprintf(stderr, "  C++ MATMUL[%d] manual[0..2]: %.4f %.4f %.4f\n",
                call_count, manual[0], manual[1], manual[2]);
        fprintf(stderr, "  C++ MATMUL[%d]: M=%d K=%d N=%d A[0..2]=%.4f %.4f %.4f B[0..2]=%.4f %.4f %.4f\n",
                call_count, M, K, N,
                A.ptr<float>()[0], A.ptr<float>()[1], A.ptr<float>()[2],
                B.ptr<float>()[0], B.ptr<float>()[1], B.ptr<float>()[2]);
        call_count++;
    }

    int lda = (int)(A.stride[1] / sizeof(float));
    int ldb = (int)(B.stride[1] / sizeof(float));
    int ldc = (int)(C.stride[1] / sizeof(float));

    static int dbg = 0;
    if (dbg < 8) {
        fprintf(stderr, "  matmul_fp32[%d]: M=%d N=%d K=%d lda=%d ldb=%d ldc=%d C.shape=[%lld,%lld] nbytes=%zu\n",
                dbg, M, N, K, lda, ldb, ldc, C.shape[0], C.shape[1], C.nbytes());
        dbg++;
    }

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

    static int out_count = 0;
    if (out_count < 3) {
        fprintf(stderr, "  C++ MATMUL[%d] out[0..2]: %.4f %.4f %.4f\n",
                out_count, c_ptr[0], c_ptr[1], c_ptr[2]);
        out_count++;
    }
}
