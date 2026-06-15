#include "kernels/tensor.h"
#include "kernels/norm.h"
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
        data[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }
}

// reference RMSNorm
static void ref_rms_norm(const float* x, const float* w, float* out,
                         int D, int N, float eps) {
    for (int n = 0; n < N; n++) {
        float sum_sq = 0;
        for (int d = 0; d < D; d++) sum_sq += x[n * D + d] * x[n * D + d];
        float rms = 1.f / std::sqrt(sum_sq / D + eps);
        for (int d = 0; d < D; d++) out[n * D + d] = x[n * D + d] * rms * w[d];
    }
}

static bool check_approx(const float* got, const float* ref, int n, float tol = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(got[i] - ref[i]) > tol) {
            fprintf(stderr, "  mismatch at %d: got %f, expected %f (diff %e)\n",
                    i, got[i], ref[i], std::fabs(got[i] - ref[i]));
            return false;
        }
    }
    return true;
}

int main() {
    srand(42);

    // ---- small: D=8, N=1 ----
    {
        int D = 8, N = 1;
        float* x_data = new float[D * N];
        float* w_data = new float[D];
        fill_rand(x_data, D * N);
        fill_rand(w_data, D);

        float* o_data = new float[D * N];
        float* ref    = new float[D * N];

        Tensor x = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, x_data);
        Tensor w = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, 1, 1, 1, w_data);
        Tensor o = Tensor::create(Precision::FP32, MemoryType::OWNED, D, N, 1, 1, o_data);

        kernel_rms_norm(x, w, 1e-6f, o);
        ref_rms_norm(x_data, w_data, ref, D, N, 1e-6f);

        CHECK(check_approx(o_data, ref, D * N), "RMSNorm D=8 N=1");

        delete[] x_data; delete[] w_data; delete[] o_data; delete[] ref;
    }

    // ---- medium: D=64, N=4 ----
    {
        int D = 64, N = 4;
        float* x_data = new float[D * N];
        float* w_data = new float[D];
        fill_rand(x_data, D * N);
        fill_rand(w_data, D);

        float* o_data = new float[D * N];
        float* ref    = new float[D * N];

        Tensor x = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, x_data);
        Tensor w = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, 1, 1, 1, w_data);
        Tensor o = Tensor::create(Precision::FP32, MemoryType::OWNED, D, N, 1, 1, o_data);

        kernel_rms_norm(x, w, 1e-6f, o);
        ref_rms_norm(x_data, w_data, ref, D, N, 1e-6f);

        CHECK(check_approx(o_data, ref, D * N), "RMSNorm D=64 N=4");

        delete[] x_data; delete[] w_data; delete[] o_data; delete[] ref;
    }

    // ---- MLA typical: D=2048, N=1 (single row decode) ----
    {
        int D = 2048, N = 1;
        float* x_data = new float[D * N];
        float* w_data = new float[D];
        fill_rand(x_data, D * N);
        fill_rand(w_data, D);

        float* o_data = new float[D * N];
        float* ref    = new float[D * N];

        Tensor x = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, x_data);
        Tensor w = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, 1, 1, 1, w_data);
        Tensor o = Tensor::create(Precision::FP32, MemoryType::OWNED, D, N, 1, 1, o_data);

        kernel_rms_norm(x, w, 1e-6f, o);
        ref_rms_norm(x_data, w_data, ref, D, N, 1e-6f);

        CHECK(check_approx(o_data, ref, D * N, 1e-4f), "RMSNorm D=2048 N=1 (decode)");

        delete[] x_data; delete[] w_data; delete[] o_data; delete[] ref;
    }

    // ---- odd D: D=17, N=3 ----
    {
        int D = 17, N = 3;
        float* x_data = new float[D * N];
        float* w_data = new float[D];
        fill_rand(x_data, D * N);
        fill_rand(w_data, D);

        float* o_data = new float[D * N];
        float* ref    = new float[D * N];

        Tensor x = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, x_data);
        Tensor w = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, 1, 1, 1, w_data);
        Tensor o = Tensor::create(Precision::FP32, MemoryType::OWNED, D, N, 1, 1, o_data);

        kernel_rms_norm(x, w, 1e-5f, o);
        ref_rms_norm(x_data, w_data, ref, D, N, 1e-5f);

        CHECK(check_approx(o_data, ref, D * N), "RMSNorm D=17 N=3 (odd D)");

        delete[] x_data; delete[] w_data; delete[] o_data; delete[] ref;
    }

    // ---- zero input: all zeros ----
    {
        int D = 32, N = 2;
        float* x_data = new float[D * N]();  // zero-initialized
        float* w_data = new float[D];
        fill_rand(w_data, D);

        float* o_data = new float[D * N];
        float* ref    = new float[D * N];

        Tensor x = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, x_data);
        Tensor w = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, 1, 1, 1, w_data);
        Tensor o = Tensor::create(Precision::FP32, MemoryType::OWNED, D, N, 1, 1, o_data);

        kernel_rms_norm(x, w, 1e-6f, o);
        ref_rms_norm(x_data, w_data, ref, D, N, 1e-6f);

        // zero input → output should be zero (rms = 1/sqrt(eps), but x=0 → out=0)
        for (int i = 0; i < D * N; i++) {
            if (std::fabs(o_data[i]) > 1e-6f) {
                printf("  FAIL: zero input produced non-zero output\n");
                failures++;
                break;
            }
        }
        printf("  PASS: RMSNorm zero input\n");

        delete[] x_data; delete[] w_data; delete[] o_data; delete[] ref;
    }

    if (failures == 0) {
        printf("\nAll norm tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
