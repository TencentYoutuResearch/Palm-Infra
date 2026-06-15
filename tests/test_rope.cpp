#include "kernels/tensor.h"
#include "kernels/rope.h"
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

// reference RoPE (interleave)
static void ref_rope_interleave(const float* x, const float* cos, const float* sin,
                                float* out, int D, int N, int rope_dim) {
    int half = rope_dim / 2;
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) out[n * D + d] = x[n * D + d];
        for (int i = 0; i < half; i++) {
            float x0 = x[n * D + 2 * i];
            float x1 = x[n * D + 2 * i + 1];
            float c  = cos[n * half + i];
            float s  = sin[n * half + i];
            out[n * D + 2 * i]     = x0 * c - x1 * s;
            out[n * D + 2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

// reference RoPE (halves)
static void ref_rope_halves(const float* x, const float* cos, const float* sin,
                            float* out, int D, int N, int rope_dim) {
    int half = rope_dim / 2;
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) out[n * D + d] = x[n * D + d];
        for (int i = 0; i < half; i++) {
            float x0 = x[n * D + i];
            float x1 = x[n * D + i + half];
            float c  = cos[n * half + i];
            float s  = sin[n * half + i];
            out[n * D + i]        = x0 * c - x1 * s;
            out[n * D + i + half] = x0 * s + x1 * c;
        }
    }
}

static bool check_approx(const float* got, const float* ref, int n, float tol = 1e-5f) {
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

    // ---- interleave: D=8, N=2, rope_dim=8 ----
    {
        int D = 8, N = 2, rope_dim = 8, half = rope_dim / 2;
        float* x_data   = new float[D * N];
        float* cos_data = new float[half * N];
        float* sin_data = new float[half * N];
        fill_rand(x_data, D * N);
        fill_rand(cos_data, half * N);
        fill_rand(sin_data, half * N);

        float* o_data = new float[D * N];
        float* ref    = new float[D * N];

        Tensor x   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, x_data);
        Tensor c   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, half, N, 1, 1, cos_data);
        Tensor s   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, half, N, 1, 1, sin_data);
        Tensor o   = Tensor::create(Precision::FP32, MemoryType::OWNED, D, N, 1, 1, o_data);

        kernel_rope(x, c, s, rope_dim, true, o);
        ref_rope_interleave(x_data, cos_data, sin_data, ref, D, N, rope_dim);

        CHECK(check_approx(o_data, ref, D * N), "RoPE interleave D=8 N=2");

        delete[] x_data; delete[] cos_data; delete[] sin_data;
        delete[] o_data; delete[] ref;
    }

    // ---- halves: D=8, N=2, rope_dim=8 ----
    {
        int D = 8, N = 2, rope_dim = 8, half = rope_dim / 2;
        float* x_data   = new float[D * N];
        float* cos_data = new float[half * N];
        float* sin_data = new float[half * N];
        fill_rand(x_data, D * N);
        fill_rand(cos_data, half * N);
        fill_rand(sin_data, half * N);

        float* o_data = new float[D * N];
        float* ref    = new float[D * N];

        Tensor x   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, x_data);
        Tensor c   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, half, N, 1, 1, cos_data);
        Tensor s   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, half, N, 1, 1, sin_data);
        Tensor o   = Tensor::create(Precision::FP32, MemoryType::OWNED, D, N, 1, 1, o_data);

        kernel_rope(x, c, s, rope_dim, false, o);
        ref_rope_halves(x_data, cos_data, sin_data, ref, D, N, rope_dim);

        CHECK(check_approx(o_data, ref, D * N), "RoPE halves D=8 N=2");

        delete[] x_data; delete[] cos_data; delete[] sin_data;
        delete[] o_data; delete[] ref;
    }

    // ---- MLA typical: D=192, rope_dim=64, N=1 ----
    {
        int D = 192, N = 1, rope_dim = 64, half = rope_dim / 2;
        float* x_data   = new float[D * N];
        float* cos_data = new float[half * N];
        float* sin_data = new float[half * N];
        fill_rand(x_data, D * N);
        fill_rand(cos_data, half * N);
        fill_rand(sin_data, half * N);

        float* o_data = new float[D * N];
        float* ref    = new float[D * N];

        Tensor x   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, x_data);
        Tensor c   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, half, N, 1, 1, cos_data);
        Tensor s   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, half, N, 1, 1, sin_data);
        Tensor o   = Tensor::create(Precision::FP32, MemoryType::OWNED, D, N, 1, 1, o_data);

        kernel_rope(x, c, s, rope_dim, true, o);
        ref_rope_interleave(x_data, cos_data, sin_data, ref, D, N, rope_dim);

        CHECK(check_approx(o_data, ref, D * N), "RoPE MLA D=192 rope=64 N=1");

        delete[] x_data; delete[] cos_data; delete[] sin_data;
        delete[] o_data; delete[] ref;
    }

    // ---- non-rope part preserved: D=128, rope_dim=64 ----
    {
        int D = 128, N = 1, rope_dim = 64, half = rope_dim / 2;
        float* x_data   = new float[D * N];
        float* cos_data = new float[half * N];
        float* sin_data = new float[half * N];
        fill_rand(x_data, D * N);
        fill_rand(cos_data, half * N);
        fill_rand(sin_data, half * N);

        float* o_data = new float[D * N];
        float* ref    = new float[D * N];

        Tensor x   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, D, N, 1, 1, x_data);
        Tensor c   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, half, N, 1, 1, cos_data);
        Tensor s   = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, half, N, 1, 1, sin_data);
        Tensor o   = Tensor::create(Precision::FP32, MemoryType::OWNED, D, N, 1, 1, o_data);

        kernel_rope(x, c, s, rope_dim, true, o);
        ref_rope_interleave(x_data, cos_data, sin_data, ref, D, N, rope_dim);

        CHECK(check_approx(o_data, ref, D * N), "RoPE non-rope preserved D=128 rope=64");

        delete[] x_data; delete[] cos_data; delete[] sin_data;
        delete[] o_data; delete[] ref;
    }

    if (failures == 0) {
        printf("\nAll rope tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
