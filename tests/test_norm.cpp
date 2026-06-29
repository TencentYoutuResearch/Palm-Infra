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

    // ---- strided input: ldx > D (simulates permuted/reshaped view) ----
    // Validates that kernel_rms_norm correctly uses stride[1] instead of
    // assuming row-major contiguous layout. This is required for the
    // qwen35.py full_attn optimization where rms_norm reads a permuted
    // (strided) tensor without an explicit contiguous() copy.
    {
        int D = 256, N = 4;
        // Physical buffer with extra columns so ldx > D.
        int ldx = D + 64;  // strided: 64 floats of padding between rows
        float* x_buf = new float[N * ldx];
        float* w_data = new float[D];
        fill_rand(x_buf, N * ldx);
        fill_rand(w_data, D);

        // Build strided tensor: shape=[D, N], stride set so row n starts at
        // x_buf + n*ldx (in elements). stride[1] = ldx * sizeof(float).
        Tensor x = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, x_buf);
        x.stride[1] = (size_t)ldx * sizeof(float);  // override default contiguous stride

        Tensor w = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, 1, 1, 1, w_data);

        // Output also strided with same ldx (kernel writes via stride[1]).
        float* o_buf = new float[N * ldx]();
        Tensor o = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, o_buf);
        o.stride[1] = (size_t)ldx * sizeof(float);

        kernel_rms_norm(x, w, 1e-6f, o);

        // Reference computed with the same strided layout.
        bool ok = true;
        for (int n = 0; n < N; n++) {
            float sum_sq = 0;
            for (int d = 0; d < D; d++) sum_sq += x_buf[n * ldx + d] * x_buf[n * ldx + d];
            float rms = 1.f / std::sqrt(sum_sq / D + 1e-6f);
            for (int d = 0; d < D; d++) {
                float expected = x_buf[n * ldx + d] * rms * w_data[d];
                if (std::fabs(o_buf[n * ldx + d] - expected) > 1e-4f) {
                    fprintf(stderr, "  strided mismatch n=%d d=%d: got %f expected %f\n",
                            n, d, o_buf[n * ldx + d], expected);
                    ok = false;
                }
            }
        }
        CHECK(ok, "RMSNorm strided D=256 N=4 ldx=320");

        delete[] x_buf; delete[] w_data; delete[] o_buf;
    }

    if (failures == 0) {
        printf("\nAll norm tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
