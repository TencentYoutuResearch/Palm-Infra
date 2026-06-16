#include "kernels/tensor.h"
#include "kernels/matmul.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

static void fill_rand(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

// reference matmul for N×K layout: B[n,k] at B[n*K+k]
static void ref_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

static bool check_approx(const float* got, const float* ref, int n, float tol = 1e-4f) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(got[i] - ref[i]) > tol) {
            fprintf(stderr, "  mismatch at %d: got %f, expected %f\n", i, got[i], ref[i]);
            return false;
        }
    }
    return true;
}

int main() {
    srand(42);

    // ---- small matmul: 4x4 * 4x4 = 4x4 ----
    {
        int M = 4, K = 4, N = 4;
        float* a_data = new float[M * K];
        float* b_data = new float[K * N];
        fill_rand(a_data, M * K);
        fill_rand(b_data, K * N);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, N, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data, b_data, ref_c, M, N, K);

        CHECK(check_approx(c_data, ref_c, M * N), "4x4 * 4x4");

        delete[] a_data; delete[] b_data; delete[] c_data; delete[] ref_c;
    }

    // ---- rectangular: 8x16 * 16x4 = 8x4 ----
    {
        int M = 8, K = 16, N = 4;
        float* a_data = new float[M * K];
        float* b_data = new float[K * N];
        fill_rand(a_data, M * K);
        fill_rand(b_data, K * N);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, N, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data, b_data, ref_c, M, N, K);

        // Debug specific element
        int idx = 1; // second element
        if (M == 8 && N == 4 && K == 16) {
            fprintf(stderr, "  DEBUG 8x16*16x4: c_data[%d]=%.6f ref_c[%d]=%.6f\n", idx, c_data[idx], idx, ref_c[idx]);
            // Manual compute C[0,1]
            float manual_c01 = 0;
            for (int k = 0; k < K; k++) manual_c01 += a_data[0*K+k] * b_data[1*K+k];
            fprintf(stderr, "  manual C[0,1]=%.6f (a_data[0], b_data[1])\n", manual_c01);
        }

        CHECK(check_approx(c_data, ref_c, M * N), "8x16 * 16x4");

        delete[] a_data; delete[] b_data; delete[] c_data; delete[] ref_c;
    }

    // ---- GEMV (M=1): 1x256 * 256x64 = 1x64 ----
    {
        int M = 1, K = 256, N = 64;
        float* a_data = new float[M * K];
        float* b_data = new float[K * N];
        fill_rand(a_data, M * K);
        fill_rand(b_data, K * N);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, N, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data, b_data, ref_c, M, N, K);

        CHECK(check_approx(c_data, ref_c, M * N), "1x256 * 256x64 (GEMV)");

        delete[] a_data; delete[] b_data; delete[] c_data; delete[] ref_c;
    }

    // ---- odd K: 3x17 * 17x5 = 3x5 ----
    {
        int M = 3, K = 17, N = 5;
        float* a_data = new float[M * K];
        float* b_data = new float[K * N];
        fill_rand(a_data, M * K);
        fill_rand(b_data, K * N);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, N, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data, b_data, ref_c, M, N, K);

        CHECK(check_approx(c_data, ref_c, M * N), "3x17 * 17x5 (odd K)");

        delete[] a_data; delete[] b_data; delete[] c_data; delete[] ref_c;
    }

    // ---- large: 32x256 * 256x128 = 32x128 ----
    {
        int M = 32, K = 256, N = 128;
        float* a_data = new float[M * K];
        float* b_data = new float[K * N];
        fill_rand(a_data, M * K);
        fill_rand(b_data, K * N);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, N, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data, b_data, ref_c, M, N, K);

        CHECK(check_approx(c_data, ref_c, M * N), "32x256 * 256x128");

        delete[] a_data; delete[] b_data; delete[] c_data; delete[] ref_c;
    }

    // ---- odd N: 4x8 * 8x3 = 4x3 ----
    {
        int M = 4, K = 8, N = 3;
        float* a_data = new float[M * K];
        float* b_data = new float[K * N];
        fill_rand(a_data, M * K);
        fill_rand(b_data, K * N);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, N, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data, b_data, ref_c, M, N, K);

        CHECK(check_approx(c_data, ref_c, M * N), "4x8 * 8x3 (odd N)");

        delete[] a_data; delete[] b_data; delete[] c_data; delete[] ref_c;
    }

    if (failures == 0) {
        printf("\nAll matmul tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
