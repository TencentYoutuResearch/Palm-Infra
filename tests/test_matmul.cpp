#include "kernels/tensor.h"
#include "kernels/matmul.h"
#include "kernels/threading.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    setenv("MOLLM_W4_PACKED_BG128", "1", 1);

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

            // FP16 accumulation has ~0.5% relative error; use loose tolerance
            float tol = 0.5f;
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

    // ---- INT8 weight path: per-channel and per-group scales ----
    {
        int M = 2, K = 4, N = 3;
        float a_data[8] = {
            1.0f, -2.0f, 0.5f, 3.0f,
            -1.0f, 4.0f, 2.0f, -0.5f,
        };
        int8_t q_data[12] = {
            1, 2, -3, 4,
            -2, 1, 5, -1,
            3, -4, 2, 1,
        };
        float c_data[6];
        float ref_c[6];

        auto run_int8_case = [&](const float* scales, uint32_t group_size,
                                 uint32_t groups_per_row, const char* label,
                                 bool interleaved = false) {
            float deq[12];
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    deq[n * K + k] = (float)q_data[n * K + k]
                        * scales[n * groups_per_row + k / (int)group_size];
                }
            }

            int8_t* packed = interleaved ? pack_b_interleaved_int8_full(q_data, N, K, K) : nullptr;
            Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
            Tensor B = Tensor::create(Precision::INT8, MemoryType::EXTERNAL, N, K, 1, 1,
                                      interleaved ? (void*)packed : (void*)q_data);
            Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, c_data);
            B.scales = scales;
            B.group_size = group_size;
            B.groups_per_row = groups_per_row;
            B.num_groups = (uint32_t)(N * groups_per_row);
            B.is_interleaved = interleaved;

            kernel_matmul_fp32(A, B, C);
            ref_matmul(a_data, deq, ref_c, M, N, K);
            CHECK(check_approx(c_data, ref_c, M * N, interleaved ? 8e-2f : 1e-5f), label);
            if (packed) delete[] packed;
        };

        float pc_scales[3] = {0.5f, 0.25f, 0.125f};
        run_int8_case(pc_scales, 4, 1, "INT8 per-channel matmul");
        run_int8_case(pc_scales, 4, 1, "INT8 per-channel matmul interleaved", true);

        float pg_scales[6] = {0.5f, 0.25f, 0.125f, 0.75f, 1.0f, 0.0625f};
        run_int8_case(pg_scales, 2, 2, "INT8 per-group matmul");
        run_int8_case(pg_scales, 2, 2, "INT8 per-group matmul interleaved", true);
    }

    // ---- INT4 packed weight path: per-group scales, odd K tail ----
    {
        int M = 2, K = 5, N = 3;
        uint32_t group_size = 2;
        uint32_t groups_per_row = 3;
        float a_data[10] = {
            1.0f, -2.0f, 0.5f, 3.0f, -1.5f,
            -1.0f, 4.0f, 2.0f, -0.5f, 0.25f,
        };
        int8_t q_data[15] = {
            1, 2, -3, 4, -7,
            -2, 1, 5, -1, 6,
            3, -4, 2, 1, -5,
        };
        uint8_t packed[9] = {};
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                uint8_t nibble = (uint8_t)q_data[n * K + k] & 0x0F;
                int byte_idx = n * ((K + 1) / 2) + (k >> 1);
                if (k & 1) packed[byte_idx] |= (uint8_t)(nibble << 4);
                else packed[byte_idx] |= nibble;
            }
        }
        float scales[9] = {
            0.5f, 0.25f, 0.125f,
            0.75f, 1.0f, 0.0625f,
            0.2f, 0.4f, 0.8f,
        };
        float deq[15];
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                deq[n * K + k] = (float)q_data[n * K + k]
                    * scales[n * groups_per_row + k / (int)group_size];
            }
        }

        float c_data[6];
        float ref_c[6];
        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, N, K, 1, 1, packed);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, c_data);
        B.scales = scales;
        B.group_size = group_size;
        B.groups_per_row = groups_per_row;
        B.num_groups = (uint32_t)(N * groups_per_row);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data, deq, ref_c, M, N, K);
        CHECK(check_approx(c_data, ref_c, M * N, 1e-5f), "INT4 per-group packed matmul");
    }

    {
        int M = 1, K = 32, N = 9;
        uint32_t group_size = 32;
        uint32_t groups_per_row = 1;
        std::vector<float> a_data(M * K);
        std::vector<int8_t> q_data(N * K);
        std::vector<uint8_t> packed((size_t)N * ((K + 1) / 2), 0);
        std::vector<float> scales(N * groups_per_row);
        std::vector<float> deq(N * K);
        std::vector<float> c_data(M * N);
        std::vector<float> ref_c(M * N);

        for (int k = 0; k < K; k++) {
            a_data[k] = ((k % 17) - 8) * 0.03125f;
        }
        int row_stride = (K + 1) / 2;
        for (int n = 0; n < N; n++) {
            scales[n] = 0.01f + 0.001f * (float)(n % 5);
            for (int k = 0; k < K; k++) {
                int8_t q = (int8_t)(((n * 7 + k * 3) % 15) - 7);
                q_data[n * K + k] = q;
                deq[n * K + k] = (float)q * scales[n];
                uint8_t nibble = (uint8_t)q & 0x0F;
                uint8_t& byte = packed[(size_t)n * row_stride + (k >> 1)];
                if (k & 1) byte |= (uint8_t)(nibble << 4);
                else byte |= nibble;
            }
        }

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data.data());
        Tensor B = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, N, K, 1, 1, packed.data());
        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, c_data.data());
        B.scales = scales.data();
        B.group_size = group_size;
        B.groups_per_row = groups_per_row;
        B.num_groups = (uint32_t)(N * groups_per_row);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data.data(), deq.data(), ref_c.data(), M, N, K);
        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 2e-2f),
              "INT4 Q8-dot GEMV packed matmul");

        uint8_t* q4_repack = pack_b_q4dot_int4_full(packed.data(), N, K, K);
        B.q4_repack_data = q4_repack;
        kernel_matmul_fp32(A, B, C);
        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 2e-2f),
              "INT4 Q8-dot GEMV repacked matmul");

        Tensor B_direct = Tensor::create(Precision::INT4, MemoryType::EXTERNAL,
                                         N, K, 1, 1, q4_repack);
        B_direct.scales = scales.data();
        B_direct.group_size = group_size;
        B_direct.groups_per_row = groups_per_row;
        B_direct.num_groups = (uint32_t)(N * groups_per_row);
        B_direct.is_q4_repacked = true;
        B_direct.q4_repack_data = q4_repack;
        kernel_matmul_fp32(A, B_direct, C);
        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 2e-2f),
              "INT4 Q8-dot GEMV direct q4 layout matmul");
        delete[] q4_repack;
    }

    {
        int M = 1, K = 128, N = 13;
        uint32_t group_size = 128;
        uint32_t groups_per_row = 1;
        std::vector<float> a_data(M * K);
        std::vector<int8_t> q_data(N * K);
        std::vector<uint8_t> packed((size_t)N * ((K + 1) / 2), 0);
        std::vector<float> scales(N * groups_per_row);
        std::vector<float> deq(N * K);
        std::vector<float> c_repack(M * N);
        std::vector<float> c_bg128(M * N);
        std::vector<float> ref_c(M * N);

        for (int k = 0; k < K; k++) {
            a_data[k] = ((k % 23) - 11) * 0.015625f;
        }
        int row_stride = (K + 1) / 2;
        for (int n = 0; n < N; n++) {
            scales[n] = 0.005f + 0.0005f * (float)(n % 7);
            for (int k = 0; k < K; k++) {
                int8_t q = (int8_t)(((n * 17 + k * 9) % 15) - 7);
                q_data[n * K + k] = q;
                deq[n * K + k] = (float)q * scales[n];
                uint8_t nibble = (uint8_t)q & 0x0F;
                uint8_t& byte = packed[(size_t)n * row_stride + (k >> 1)];
                if (k & 1) byte |= (uint8_t)(nibble << 4);
                else byte |= nibble;
            }
        }

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data.data());
        Tensor B = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, N, K, 1, 1, packed.data());
        Tensor C_repack = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                         N, M, 1, 1, c_repack.data());
        Tensor C_bg128 = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                        N, M, 1, 1, c_bg128.data());
        B.scales = scales.data();
        B.group_size = group_size;
        B.groups_per_row = groups_per_row;
        B.num_groups = (uint32_t)(N * groups_per_row);

        uint8_t* q4_repack = pack_b_q4dot_int4_full(packed.data(), N, K, K);
        uint8_t* q4_g128 = pack_b_q4dot_g128_full(
            q4_repack, scales.data(), N, K, groups_per_row);
        B.q4_repack_data = q4_repack;

        kernel_matmul_fp32(A, B, C_repack);
        B.q4_g128_data = q4_g128;
        kernel_matmul_fp32(A, B, C_bg128);
        ref_matmul(a_data.data(), deq.data(), ref_c.data(), M, N, K);

        CHECK(check_approx(c_bg128.data(), c_repack.data(), M * N, 1e-5f),
              "INT4 Q8-dot GEMV BG128 matches q4dot repack");
        CHECK(check_approx(c_bg128.data(), ref_c.data(), M * N, 5e-2f),
              "INT4 Q8-dot GEMV BG128 reference");
        delete[] q4_repack;
        delete[] q4_g128;
    }

    {
        int M = 8, K = 128, N = 13;
        uint32_t group_size = 128;
        uint32_t groups_per_row = 1;
        std::vector<float> a_data(M * K);
        std::vector<int8_t> q_data(N * K);
        std::vector<uint8_t> packed((size_t)N * ((K + 1) / 2), 0);
        std::vector<float> scales(N * groups_per_row);
        std::vector<float> deq(N * K);
        std::vector<float> c_repack(M * N);
        std::vector<float> c_bg128(M * N);
        std::vector<float> ref_c(M * N);

        for (int i = 0; i < M * K; i++) {
            a_data[i] = ((i % 23) - 11) * 0.015625f;
        }
        int row_stride = (K + 1) / 2;
        for (int n = 0; n < N; n++) {
            scales[n] = 0.005f + 0.0005f * (float)(n % 7);
            for (int k = 0; k < K; k++) {
                int8_t q = (int8_t)(((n * 17 + k * 9) % 15) - 7);
                q_data[n * K + k] = q;
                deq[n * K + k] = (float)q * scales[n];
                uint8_t nibble = (uint8_t)q & 0x0F;
                uint8_t& byte = packed[(size_t)n * row_stride + (k >> 1)];
                if (k & 1) byte |= (uint8_t)(nibble << 4);
                else byte |= nibble;
            }
        }

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data.data());
        Tensor B = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, N, K, 1, 1, packed.data());
        Tensor C_repack = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                         N, M, 1, 1, c_repack.data());
        Tensor C_bg128 = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                        N, M, 1, 1, c_bg128.data());
        B.scales = scales.data();
        B.group_size = group_size;
        B.groups_per_row = groups_per_row;
        B.num_groups = (uint32_t)(N * groups_per_row);

        uint8_t* q4_repack = pack_b_q4dot_int4_full(packed.data(), N, K, K);
        uint8_t* q4_g128 = pack_b_q4dot_g128_full(
            q4_repack, scales.data(), N, K, groups_per_row);
        B.q4_repack_data = q4_repack;

        kernel_matmul_fp32(A, B, C_repack);
        B.q4_g128_data = q4_g128;
        kernel_matmul_fp32(A, B, C_bg128);
        ref_matmul(a_data.data(), deq.data(), ref_c.data(), M, N, K);

        CHECK(check_approx(c_bg128.data(), c_repack.data(), M * N, 1e-5f),
              "INT4 Q8-dot GEMM BG128 matches q4dot repack");
        CHECK(check_approx(c_bg128.data(), ref_c.data(), M * N, 5e-2f),
              "INT4 Q8-dot GEMM BG128 reference");
        delete[] q4_repack;
        delete[] q4_g128;
    }

    {
        int M = 4, K = 64, N = 9;
        uint32_t group_size = 32;
        uint32_t groups_per_row = 2;
        std::vector<float> a_data(M * K);
        std::vector<int8_t> q_data(N * K);
        std::vector<uint8_t> packed((size_t)N * ((K + 1) / 2), 0);
        std::vector<float> scales(N * groups_per_row);
        std::vector<float> deq(N * K);
        std::vector<float> c_data(M * N);
        std::vector<float> ref_c(M * N);

        for (int i = 0; i < M * K; i++) {
            a_data[i] = ((i % 19) - 9) * 0.03125f;
        }
        int row_stride = (K + 1) / 2;
        for (int n = 0; n < N; n++) {
            for (uint32_t g = 0; g < groups_per_row; g++) {
                scales[n * groups_per_row + g] = 0.01f + 0.001f * (float)((n + g) % 5);
            }
            for (int k = 0; k < K; k++) {
                int8_t q = (int8_t)(((n * 11 + k * 5) % 15) - 7);
                q_data[n * K + k] = q;
                deq[n * K + k] = (float)q * scales[n * groups_per_row + k / (int)group_size];
                uint8_t nibble = (uint8_t)q & 0x0F;
                uint8_t& byte = packed[(size_t)n * row_stride + (k >> 1)];
                if (k & 1) byte |= (uint8_t)(nibble << 4);
                else byte |= nibble;
            }
        }

        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data.data());
        Tensor B = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, N, K, 1, 1, packed.data());
        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, c_data.data());
        B.scales = scales.data();
        B.group_size = group_size;
        B.groups_per_row = groups_per_row;
        B.num_groups = (uint32_t)(N * groups_per_row);

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data.data(), deq.data(), ref_c.data(), M, N, K);
        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 2e-2f),
              "INT4 Q8-dot GEMM packed matmul");

        uint8_t* q4_repack = pack_b_q4dot_int4_full(packed.data(), N, K, K);
        B.q4_repack_data = q4_repack;
        kernel_matmul_fp32(A, B, C);
        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 2e-2f),
              "INT4 Q8-dot GEMM repacked matmul");

        Tensor B_direct = Tensor::create(Precision::INT4, MemoryType::EXTERNAL,
                                         N, K, 1, 1, q4_repack);
        B_direct.scales = scales.data();
        B_direct.group_size = group_size;
        B_direct.groups_per_row = groups_per_row;
        B_direct.num_groups = (uint32_t)(N * groups_per_row);
        B_direct.is_q4_repacked = true;
        B_direct.q4_repack_data = q4_repack;
        kernel_matmul_fp32(A, B_direct, C);
        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 2e-2f),
              "INT4 Q8-dot GEMM direct q4 layout matmul");
        delete[] q4_repack;
    }

    {
        int M = 1, K = 4, N = 3;
        float a_data[4] = {1.0f, -2.0f, 0.5f, 3.0f};
        int8_t q_data[12] = {
            1, 2, -3, 4,
            -2, 1, 5, -1,
            3, -4, 2, 1,
        };
        float scales[3] = {0.5f, 0.25f, 0.125f};
        float deq[12];
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                deq[n * K + k] = (float)q_data[n * K + k] * scales[n];
            }
        }

        int8_t* packed = pack_b_interleaved_int8_full(q_data, N, K, K);
        float c_data[3];
        float ref_c[3];
        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::INT8, MemoryType::EXTERNAL, N, K, 1, 1, packed);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, c_data);
        B.scales = scales;
        B.group_size = K;
        B.groups_per_row = 1;
        B.num_groups = N;
        B.is_interleaved = true;

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data, deq, ref_c, M, N, K);
        CHECK(check_approx(c_data, ref_c, M * N, 8e-2f), "INT8 Q8-dot GEMV interleaved");
        delete[] packed;
    }

    {
        int M = 1, K = 32, N = 8;
        std::vector<float> a_data(M * K);
        std::vector<int8_t> q_data(N * K);
        std::vector<float> scales(N);
        std::vector<float> deq(N * K);
        std::vector<float> c_data(M * N);
        std::vector<float> ref_c(M * N);

        for (int k = 0; k < K; k++) {
            a_data[k] = ((k % 13) - 6) * 0.03125f;
        }
        for (int n = 0; n < N; n++) {
            scales[n] = 0.01f + 0.001f * (float)(n % 3);
            for (int k = 0; k < K; k++) {
                q_data[n * K + k] = (int8_t)(((n * 11 + k * 5) % 63) - 31);
                deq[n * K + k] = (float)q_data[n * K + k] * scales[n];
            }
        }

        int8_t* packed = pack_b_interleaved_int8_full(q_data.data(), N, K, K);
        int8_t* q8_repack = pack_b_q8dot_int8_full(q_data.data(), N, K, K);
        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data.data());
        Tensor B = Tensor::create(Precision::INT8, MemoryType::EXTERNAL, N, K, 1, 1, packed);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, c_data.data());
        B.scales = scales.data();
        B.group_size = K;
        B.groups_per_row = 1;
        B.num_groups = N;
        B.is_interleaved = true;
        B.q8_repack_data = q8_repack;

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data.data(), deq.data(), ref_c.data(), M, N, K);
        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 8e-2f),
              "INT8 Q8-dot GEMV repackable interleaved");
        delete[] packed;
        delete[] q8_repack;
    }

    {
        int M = 4, K = 32, N = 8;
        std::vector<float> a_data(M * K);
        std::vector<int8_t> q_data(N * K);
        std::vector<float> scales(N);
        std::vector<float> deq(N * K);
        std::vector<float> c_data(M * N);
        std::vector<float> ref_c(M * N);

        for (int i = 0; i < M * K; i++) {
            a_data[i] = ((i % 17) - 8) * 0.03125f;
        }
        for (int n = 0; n < N; n++) {
            scales[n] = 0.01f + 0.001f * (float)(n % 3);
            for (int k = 0; k < K; k++) {
                q_data[n * K + k] = (int8_t)(((n * 13 + k * 7) % 63) - 31);
                deq[n * K + k] = (float)q_data[n * K + k] * scales[n];
            }
        }

        int8_t* packed = pack_b_interleaved_int8_full(q_data.data(), N, K, K);
        int8_t* q8_repack = pack_b_q8dot_int8_full(q_data.data(), N, K, K);
        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data.data());
        Tensor B = Tensor::create(Precision::INT8, MemoryType::EXTERNAL, N, K, 1, 1, packed);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, c_data.data());
        B.scales = scales.data();
        B.group_size = K;
        B.groups_per_row = 1;
        B.num_groups = N;
        B.is_interleaved = true;
        B.q8_repack_data = q8_repack;

        kernel_matmul_fp32(A, B, C);
        ref_matmul(a_data.data(), deq.data(), ref_c.data(), M, N, K);
        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 8e-2f),
              "INT8 Q8-dot GEMM repackable interleaved");
        delete[] packed;
        delete[] q8_repack;
    }

    if (failures == 0) {
        printf("\nAll matmul tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
