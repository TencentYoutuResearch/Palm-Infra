#include "kernels/tensor.h"
#include "kernels/matmul.h"
#include "kernels/threading.h"
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

// Reference matmul: C = A @ W^T
// A: [K, M] — activations (K features, M seq_len)
// W: [N, K] — weight (N output, K input), row-major: W[n,k] = data[n*K + k]
// C: [N, M] — output
static void ref_matmul(const float* A, const float* W, float* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[k + m * K] * W[n * K + k];
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
        float* b_data = new float[N * K];  // [N, K] layout
        fill_rand(a_data, M * K);
        fill_rand(b_data, N * K);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1, b_data);
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
        float* b_data = new float[N * K];
        fill_rand(a_data, M * K);
        fill_rand(b_data, N * K);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data, b_data, ref_c, M, N, K);

        CHECK(check_approx(c_data, ref_c, M * N), "8x16 * 16x4");

        delete[] a_data; delete[] b_data; delete[] c_data; delete[] ref_c;
    }

    // ---- GEMV (M=1): 1x256 * 256x64 = 1x64 ----
    {
        int M = 1, K = 256, N = 64;
        float* a_data = new float[M * K];
        float* b_data = new float[N * K];
        fill_rand(a_data, M * K);
        fill_rand(b_data, N * K);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1, b_data);
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
        float* b_data = new float[N * K];
        fill_rand(a_data, M * K);
        fill_rand(b_data, N * K);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1, b_data);
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
        float* b_data = new float[N * K];
        fill_rand(a_data, M * K);
        fill_rand(b_data, N * K);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1, b_data);
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
        float* b_data = new float[N * K];
        fill_rand(a_data, M * K);
        fill_rand(b_data, N * K);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data, b_data, ref_c, M, N, K);

        CHECK(check_approx(c_data, ref_c, M * N), "4x8 * 8x3 (odd N)");

        delete[] a_data; delete[] b_data; delete[] c_data; delete[] ref_c;
    }

    // ---- multithreaded large case ----
    {
        int M = 32, K = 256, N = 128;
        float* a_data = new float[M * K];
        float* b_data = new float[N * K];
        fill_rand(a_data, M * K);
        fill_rand(b_data, N * K);

        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

        ref_matmul(a_data, b_data, ref_c, M, N, K);

        for (int num_threads : {1, 2, 4}) {
            ThreadPool pool(num_threads);
            kernel_matmul_fp32(A, B, C, &pool);

            char msg[64];
            std::snprintf(msg, sizeof(msg), "32x256 * 256x128 threads=%d", num_threads);
            CHECK(check_approx(c_data, ref_c, M * N), msg);
        }

        delete[] a_data; delete[] b_data; delete[] c_data; delete[] ref_c;
    }

    // ---- FP16 tests with interleaved packing (pre-packed B) ----
    {
        // Helper: pack B then run matmul
        auto test_fp16 = [&](int M, int K, int N, const char* label) {
            float* a_data = new float[M * K];
            __fp16* b_data = new __fp16[N * K];
            float* c_data = new float[M * N];
            float* ref_c  = new float[M * N];
            float* tmp_b  = new float[N * K];

            fill_rand(a_data, M * K);
            fill_rand(tmp_b, N * K);
            for (int i = 0; i < N * K; i++) b_data[i] = (__fp16)tmp_b[i];

            // Pre-pack B (mirrors engine load-time packing)
            __fp16* b_packed = pack_b_interleaved_full(b_data, N, K, K);

            Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
            Tensor B = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, N, K, 1, 1, b_packed);
            Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

            // Enable interleaved packing
            g_matmul_config.use_interleave_pack = true;
            kernel_matmul_fp32(A, B, C);

            ref_matmul(a_data, tmp_b, ref_c, M, N, K);

            // Use larger tolerance for FP16 due to truncation error accumulation
            float tol = 1e-2f;
            CHECK(check_approx(c_data, ref_c, M * N, tol), label);

            delete[] a_data; delete[] b_data; delete[] b_packed;
            delete[] c_data; delete[] ref_c; delete[] tmp_b;
        };

        test_fp16(32, 256, 128, "FP16 32x256 * 256x128 (interleave)");
        test_fp16(1, 256, 64, "FP16 1x256 * 256x64 GEMV (interleave)");
        test_fp16(4, 8, 3, "FP16 4x8 * 8x3 odd N (interleave)");
        test_fp16(4, 8, 5, "FP16 4x8 * 8x5 odd N (interleave)");
        test_fp16(4, 8, 7, "FP16 4x8 * 8x7 odd N (interleave)");
        test_fp16(4, 8, 9, "FP16 4x8 * 8x9 N=9 (interleave)");
        test_fp16(4, 8, 16, "FP16 4x8 * 8x16 N=16 (interleave)");
        test_fp16(1, 8, 3, "FP16 1x8 * 8x3 GEMV odd N (interleave)");
        // Lane-FMA path (M >= 8)
        test_fp16(8, 64, 8, "FP16 8x64 * 64x8 M=8 (lane-fma)");
        test_fp16(16, 128, 32, "FP16 16x128 * 128x32 (lane-fma)");
        test_fp16(64, 256, 64, "FP16 64x256 * 256x64 (lane-fma)");
        test_fp16(8, 17, 8, "FP16 8x17 * 17x8 odd K (lane-fma)");
        test_fp16(8, 64, 3, "FP16 8x64 * 64x3 odd N (lane-fma)");
    }

    // ---- FP16 with interleave disabled (fallback path) ----
    {
        int M = 32, K = 256, N = 128;
        float* a_data = new float[M * K];
        __fp16* b_data = new __fp16[N * K];
        float* c_data = new float[M * N];
        float* ref_c  = new float[M * N];
        float* tmp_b  = new float[N * K];

        fill_rand(a_data, M * K);
        fill_rand(tmp_b, N * K);
        for (int i = 0; i < N * K; i++) b_data[i] = (__fp16)tmp_b[i];

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, N, K, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

        // Disable interleaved packing
        g_matmul_config.use_interleave_pack = false;
        kernel_matmul_fp32(A, B, C);

        ref_matmul(a_data, tmp_b, ref_c, M, N, K);
        CHECK(check_approx(c_data, ref_c, M * N, 1e-2f), "FP16 32x256 * 256x128 (no interleave)");

        delete[] a_data; delete[] b_data; delete[] c_data; delete[] ref_c; delete[] tmp_b;

        // Restore default
        g_matmul_config.use_interleave_pack = true;
    }

    if (failures == 0) {
        printf("\nAll matmul tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
