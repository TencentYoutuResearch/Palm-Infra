#include "kernels/tensor.h"
#include "kernels/matmul.h"
#include "kernels/threading.h"
#include <algorithm>
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

static float relative_l2_error(const float* got, const float* ref, int n) {
    double error = 0.0;
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        const double delta =
            static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        error += delta * delta;
        norm += static_cast<double>(ref[i]) * ref[i];
    }
    return norm > 0.0
        ? static_cast<float>(std::sqrt(error / norm))
        : 0.0f;
}

static uint8_t reference_encode_fp8(float value) {
    if (std::isnan(value)) return 0x7f;
    const uint8_t sign = std::signbit(value) ? 0x80 : 0;
    const float magnitude = std::min(std::fabs(value), 448.0f);
    int best = 0;
    float best_distance = magnitude;
    for (int code = 1; code <= 126; ++code) {
        const float distance =
            std::fabs(magnitude -
                      decode_fp8_e4m3fn(static_cast<uint8_t>(code)));
        if (distance < best_distance ||
            (distance == best_distance && (code & 1) == 0)) {
            best = code;
            best_distance = distance;
        }
    }
    return static_cast<uint8_t>(sign | best);
}

static void reference_fp8_ue8m0_activation(const float* input, float* output,
                                           int M, int K) {
    constexpr int block_size = 128;
    for (int m = 0; m < M; ++m) {
        for (int begin = 0; begin < K; begin += block_size) {
            const int end = std::min(begin + block_size, K);
            float maximum = 1.0e-4f;
            for (int k = begin; k < end; ++k) {
                maximum = std::max(
                    maximum, std::fabs(input[m * K + k]));
            }
            const int exponent = static_cast<int>(
                std::ceil(std::log2(maximum / 448.0f)));
            const float scale = std::ldexp(1.0f, exponent);
            for (int k = begin; k < end; ++k) {
                output[m * K + k] =
                    decode_fp8_e4m3fn(encode_fp8_e4m3fn(
                        std::clamp(
                            input[m * K + k] / scale,
                            -448.0f, 448.0f))) *
                    scale;
            }
        }
    }
}

int main() {
    srand(42);

    // ---- FP8/MXFP4 scalar format decoding ----
    {
        CHECK(decode_e8m0(127) == 1.0f &&
                  decode_e8m0(128) == 2.0f,
              "E8M0 power-of-two scales");
        bool all_e8m0_values_match = true;
        for (int value = 0; value < 255; ++value) {
            if (decode_e8m0(static_cast<uint8_t>(value)) !=
                std::ldexp(1.0f, value - 127)) {
                all_e8m0_values_match = false;
                break;
            }
        }
        CHECK(all_e8m0_values_match,
              "E8M0 bit decoding matches every finite power");
        CHECK(decode_fp8_e4m3fn(0x38) == 1.0f &&
                  decode_fp8_e4m3fn(0x7e) == 448.0f &&
                  decode_fp8_e4m3fn(0xb8) == -1.0f,
              "FP8 E4M3FN decoding");
        CHECK(encode_fp8_e4m3fn(1.0f) == 0x38 &&
                  encode_fp8_e4m3fn(-1.0f) == 0xb8 &&
                  encode_fp8_e4m3fn(500.0f) == 0x7e,
              "FP8 E4M3FN encoding");
        bool fp8_rounding_ok = true;
        for (int i = 0; i <= 200000; ++i) {
            const float value =
                -500.0f + 1000.0f * static_cast<float>(i) / 200000.0f;
            if (encode_fp8_e4m3fn(value) !=
                reference_encode_fp8(value)) {
                fp8_rounding_ok = false;
                break;
            }
        }
        for (int code = 0; code < 126 && fp8_rounding_ok; ++code) {
            const float lower =
                decode_fp8_e4m3fn(static_cast<uint8_t>(code));
            const float upper =
                decode_fp8_e4m3fn(static_cast<uint8_t>(code + 1));
            const float midpoint = (lower + upper) * 0.5f;
            const float probes[] = {
                std::nextafter(midpoint, lower), midpoint,
                std::nextafter(midpoint, upper)};
            for (float probe : probes) {
                if (encode_fp8_e4m3fn(probe) !=
                    reference_encode_fp8(probe)) {
                    fp8_rounding_ok = false;
                    break;
                }
            }
        }
        CHECK(fp8_rounding_ok,
              "FP8 E4M3FN fast encoder matches nearest-even reference");
        CHECK(decode_mxfp4_e2m1(0x1) == 0.5f &&
                  decode_mxfp4_e2m1(0x7) == 6.0f &&
                  decode_mxfp4_e2m1(0xf) == -6.0f,
              "MXFP4 E2M1 decoding");
    }

    // ---- exact F32 x MXFP4 and fast Q8 x MXFP4 GEMV ----
    {
        constexpr int M = 2;
        constexpr int N = 3;
        constexpr int K = 32;
        float a[M * K];
        for (int i = 0; i < M * K; ++i)
            a[i] = static_cast<float>((i % 13) - 6) / 6.0f;
        uint8_t packed[N * K / 2];
        for (int i = 0; i < N * K / 2; ++i) {
            const uint8_t low = static_cast<uint8_t>(i % 16);
            const uint8_t high = static_cast<uint8_t>((15 - i) % 16);
            packed[i] = low | static_cast<uint8_t>(high << 4);
        }
        uint8_t scales[N] = {127, 128, 126}; // 1, 2, .5
        float dequantized[N * K];
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                const uint8_t byte = packed[n * (K / 2) + k / 2];
                const uint8_t nibble =
                    (k & 1) ? byte >> 4 : byte & 0x0f;
                dequantized[n * K + k] =
                    decode_mxfp4_e2m1(nibble) * decode_e8m0(scales[n]);
            }
        }
        float exact[M * N] = {};
        float reference[M * N] = {};
        Tensor A = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a);
        Tensor B = Tensor::create(
            Precision::MXFP4, MemoryType::EXTERNAL, N, K, 1, 1, packed);
        Tensor C = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, exact);
        B.e8m0_scales = scales;
        B.group_size = 32;
        B.groups_per_row = 1;
        B.num_groups = N;
        kernel_matmul_mxfp4_reference(A, B, C);
        ref_matmul(a, dequantized, reference, M, N, K);
        CHECK(check_approx(exact, reference, M * N, 1e-5f),
              "F32 x MXFP4 exact reference");

        float fast[N] = {};
        Tensor A1 = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1, a);
        Tensor C1 = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1, fast);
        kernel_matmul_fp32(A1, B, C1);
        float fp8_a[K];
        float fp8_reference[N] = {};
        reference_fp8_ue8m0_activation(a, fp8_a, 1, K);
        ref_matmul(fp8_a, dequantized, fp8_reference, 1, N, K);
        CHECK(relative_l2_error(fast, fp8_reference, N) < 0.015f,
              "FP8 x MXFP4 decode GEMV");

        float small_batch[M * N] = {};
        float rowwise[M * N] = {};
        Tensor small_batch_output = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            N, M, 1, 1, small_batch);
        kernel_matmul_fp32(A, B, small_batch_output);
        for (int m = 0; m < M; ++m) {
            Tensor row_input = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL,
                K, 1, 1, 1, a + m * K);
            Tensor row_output = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL,
                N, 1, 1, 1, rowwise + m * N);
            kernel_matmul_fp32(row_input, B, row_output);
        }
        CHECK(check_approx(
                  small_batch, rowwise, M * N, 1e-5f),
              "MXFP4 small GEMM matches individual GEMVs");
    }

    // ---- F32 x block-scaled FP8 E4M3 ----
    {
        constexpr int M = 2;
        constexpr int N = 3;
        constexpr int K = 4;
        float a[M * K] = {1, 2, -1, .5f, -2, 1, .25f, 3};
        uint8_t weights[N * K] = {
            0x38, 0x40, 0xb8, 0x30,
            0x40, 0x38, 0x00, 0xc0,
            0x30, 0xb0, 0x38, 0x40,
        };
        uint8_t scales[1] = {128}; // one 128x128 tile, scale=2
        float dequantized[N * K];
        for (int i = 0; i < N * K; ++i)
            dequantized[i] =
                decode_fp8_e4m3fn(weights[i]) * 2.0f;
        float output[M * N] = {};
        float reference[M * N] = {};
        Tensor A = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a);
        Tensor B = Tensor::create(
            Precision::FP8_E4M3, MemoryType::EXTERNAL,
            N, K, 1, 1, weights);
        Tensor C = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, output);
        B.e8m0_scales = scales;
        B.group_size = 128;
        B.groups_per_row = 1;
        B.num_groups = 1;
        B.is_fp8_block128 = true;
        CHECK(kernel_matmul_fp8_weight_f32_activation(A, B, C),
              "direct F32 x FP8 E4M3 dispatch");
        ref_matmul(a, dequantized, reference, M, N, K);
        CHECK(check_approx(output, reference, M * N, 1e-5f),
              "F32 x FP8 E4M3 block-128 reference");
    }

    // Exercise multiple FP8 scale tiles in both dimensions.  The grouped
    // DeepSeek-V4 output projection uses this direct (unquantized activation)
    // path with large matrices.
    {
        constexpr int M = 3;
        constexpr int N = 256;
        constexpr int K = 256;
        float a[M * K];
        uint8_t weights[N * K];
        uint8_t scales[(N / 128) * (K / 128)] = {
            126, 127, 128, 129,
        };
        float output[M * N] = {};
        float reference[M * N] = {};
        for (int i = 0; i < M * K; ++i)
            a[i] = static_cast<float>((i * 17) % 61 - 30) / 31.0f;
        for (int i = 0; i < N * K; ++i) {
            const int magnitude = 1 + (i * 13) % 96;
            weights[i] = static_cast<uint8_t>(
                magnitude | ((i % 7 == 0) ? 0x80 : 0));
        }
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const float scale = decode_e8m0(
                        scales[(n / 128) * (K / 128) + k / 128]);
                    sum += a[m * K + k] *
                           decode_fp8_e4m3fn(weights[n * K + k]) *
                           scale;
                }
                reference[m * N + n] = sum;
            }
        }
        Tensor A = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a);
        Tensor B = Tensor::create(
            Precision::FP8_E4M3, MemoryType::EXTERNAL,
            N, K, 1, 1, weights);
        Tensor C = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, output);
        B.e8m0_scales = scales;
        B.group_size = 128;
        B.groups_per_row = K / 128;
        B.num_groups = sizeof(scales);
        B.is_fp8_block128 = true;
        CHECK(kernel_matmul_fp8_weight_f32_activation(A, B, C),
              "direct F32 x FP8 E4M3 multi-tile dispatch");
        CHECK(relative_l2_error(output, reference, M * N) < 2e-5f,
              "direct F32 x FP8 E4M3 multi-tile reference");
    }

    // ---- fast Q8-dot F32 x FP8 E4M3 GEMV ----
    {
        constexpr int N = 5;
        constexpr int K = 128;
        float a[K];
        uint8_t weights[N * K];
        for (int k = 0; k < K; ++k)
            a[k] = static_cast<float>((k % 17) - 8) / 8.0f;
        for (int i = 0; i < N * K; ++i) {
            // Cover signs, subnormals and all finite normal exponents while
            // avoiding the two E4M3FN NaN encodings.
            uint8_t value = static_cast<uint8_t>((i * 29) & 0xfe);
            weights[i] = value;
        }
        uint8_t scales[1] = {125}; // 0.25
        float dequantized[N * K];
        for (int i = 0; i < N * K; ++i)
            dequantized[i] =
                decode_fp8_e4m3fn(weights[i]) * 0.25f;
        float output[N] = {};
        float reference[N] = {};
        Tensor A = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1, a);
        Tensor B = Tensor::create(
            Precision::FP8_E4M3, MemoryType::EXTERNAL,
            N, K, 1, 1, weights);
        Tensor C = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1, output);
        B.e8m0_scales = scales;
        B.group_size = 128;
        B.groups_per_row = 1;
        B.num_groups = 1;
        B.is_fp8_block128 = true;
        kernel_matmul_fp32(A, B, C);
        float fallback_output[N];
        std::copy(output, output + N, fallback_output);
        float fp8_a[K];
        reference_fp8_ue8m0_activation(a, fp8_a, 1, K);
        ref_matmul(fp8_a, dequantized, reference, 1, N, K);
        CHECK(relative_l2_error(output, reference, N) < 0.025f,
              "Q8-dot FP8 x FP8 E4M3 GEMV");

        const size_t packed_bytes = pack_fp8_e4m3_q8dot_bytes(N, K);
        std::vector<int8_t> packed(packed_bytes);
        std::vector<float> packed_scales(
            static_cast<size_t>(N) * K / 32);
        CHECK(pack_fp8_e4m3_q8dot(
                  weights, scales, N, K, packed.data(),
                  packed_scales.data()),
              "pack native FP8 to Q8-dot sidecar");
        if (mollm::cpu::capabilities().fp16_interleaved_weights) {
            std::fill(output, output + N, 0.0f);
            B.q8_repack_data = packed.data();
            B.fp8_q8_scales = packed_scales.data();
            kernel_matmul_fp32(A, B, C);
            CHECK(relative_l2_error(output, reference, N) < 0.025f,
                  "packed Q8-dot FP8 x FP8 E4M3 GEMV");
            CHECK(check_approx(output, fallback_output, N, 1e-5f),
                  "packed and portable FP8 Q8-dot paths agree");
        } else {
            CHECK(true, "scalar provider keeps FP8 weights on reference path");
        }
    }

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

    // ---- fused ReLU-squared writeback ----
    {
        float a_data[4] = {-2.f, 1.f, 3.f, -4.f};
        float b_data[4] = {1.f, 0.f, 0.f, 1.f};
        float c_data[4] = {};
        const float expected[4] = {0.f, 1.f, 9.f, 0.f};
        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  2, 2, 1, 1, a_data);
        Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  2, 2, 1, 1, b_data);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  2, 2, 1, 1, c_data);
        kernel_matmul_fp32(A, B, C, nullptr, Activation::RELU_SQUARED);
        CHECK(check_approx(c_data, expected, 4),
              "matmul fused ReLU-squared activation");
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

    // ---- row-major FP16 M=2 matches two independent GEMVs bit-for-bit ----
    {
        auto test_fp16_m2 = [&](int K, int N, int num_threads,
                                const char* shape_label) {
            constexpr int M = 2;
            std::vector<float> a(M * K);
            std::vector<__fp16> weights(N * K);
            std::vector<float> batched(M * N, -1.0f);
            std::vector<float> rowwise(M * N, -2.0f);

            for (int i = 0; i < M * K; ++i) {
                a[i] = static_cast<float>((i * 37 + 11) % 257 - 128) /
                       128.0f;
            }
            for (int i = 0; i < N * K; ++i) {
                weights[i] = static_cast<__fp16>(
                    static_cast<float>((i * 29 + 7) % 193 - 96) / 256.0f);
            }

            Tensor A = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1,
                a.data());
            Tensor B = Tensor::create(
                Precision::FP16, MemoryType::EXTERNAL, N, K, 1, 1,
                weights.data());
            Tensor C = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1,
                batched.data());
            ThreadPool pool(num_threads);
            kernel_matmul_fp32(A, B, C, &pool);

            for (int row = 0; row < M; ++row) {
                Tensor row_a = Tensor::create(
                    Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1,
                    a.data() + row * K);
                Tensor row_c = Tensor::create(
                    Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                    rowwise.data() + row * N);
                kernel_matmul_fp32(row_a, B, row_c, &pool);
            }

            for (int row = 0; row < M; ++row) {
                char message[160];
                std::snprintf(
                    message, sizeof(message),
                    "row-major FP16 M=2 row=%d bitwise parity %s threads=%d",
                    row, shape_label, num_threads);
                CHECK(std::memcmp(batched.data() + row * N,
                                  rowwise.data() + row * N,
                                  N * sizeof(float)) == 0,
                      message);
            }
        };

        for (int num_threads : {1, 4}) {
            test_fp16_m2(37, 131, num_threads, "odd K/N tails");
            // One eighth of the official 4B fused GDN input projection.
            test_fp16_m2(320, 1544, num_threads,
                         "scaled 12352x2560 projection");
        }
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
            B.is_interleaved = true;

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
        B.is_q4_g128_packed = true;
        B.scales = nullptr;
        kernel_matmul_fp32(A, B, C_bg128);
        ref_matmul(a_data.data(), deq.data(), ref_c.data(), M, N, K);

        CHECK(check_approx(c_bg128.data(), c_repack.data(), M * N, 1e-5f),
              "INT4 Q8-dot GEMV BG128 matches q4dot repack");
        CHECK(check_approx(c_bg128.data(), ref_c.data(), M * N, 5e-2f),
              "INT4 Q8-dot GEMV BG128 reference");
        delete[] q4_repack;
        delete[] q4_g128;
    }

    // Several independent expert GEMVs can share one thread-pool dispatch.
    // Compare the batched path against the regular production kernel, including
    // the common MoE gate/up case where every expert sees the same activation.
    if (matmul_int4_q4dot_kernel_available()) {
        constexpr int batch = 3;
        constexpr int M = 1;
        constexpr int K = 128;
        constexpr int N = 64;
        std::vector<float> input0(K), input1(K);
        for (int k = 0; k < K; ++k) {
            input0[k] = ((k * 7) % 29 - 14) * 0.015625f;
            input1[k] = ((k * 11) % 31 - 15) * 0.0125f;
        }

        std::vector<std::vector<uint8_t>> packed(
            batch, std::vector<uint8_t>((size_t)N * K / 2, 0));
        std::vector<std::vector<float>> scales(
            batch, std::vector<float>(N));
        std::vector<uint8_t*> q4dot(batch, nullptr);
        std::vector<uint8_t*> bg128(batch, nullptr);
        std::vector<std::vector<float>> expected(
            batch, std::vector<float>(N));
        std::vector<std::vector<float>> actual(
            batch, std::vector<float>(N));
        std::vector<Tensor> inputs;
        std::vector<Tensor> weights;
        std::vector<Tensor> outputs;
        inputs.reserve(batch);
        weights.reserve(batch);
        outputs.reserve(batch);
        for (int i = 0; i < batch; ++i) {
            for (int n = 0; n < N; ++n) {
                scales[i][n] = 0.006f + 0.00025f * ((n + i) % 9);
                for (int k = 0; k < K; ++k) {
                    int q = ((n * 13 + k * 5 + i * 3) % 15) - 7;
                    size_t index = (size_t)n * (K / 2) + k / 2;
                    if (k & 1)
                        packed[i][index] |= (uint8_t)((q & 15) << 4);
                    else
                        packed[i][index] |= (uint8_t)(q & 15);
                }
            }
            q4dot[i] =
                pack_b_q4dot_int4_full(packed[i].data(), N, K, K);
            bg128[i] = pack_b_q4dot_g128_full(
                q4dot[i], scales[i].data(), N, K, 1);

            float* input = i < 2 ? input0.data() : input1.data();
            inputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, input));
            weights.push_back(Tensor::create(
                Precision::INT4, MemoryType::EXTERNAL, N, K, 1, 1,
                bg128[i]));
            // BG128 blocks embed their scales; SSD-loaded experts intentionally
            // omit the redundant scale sidecar.
            weights.back().scales = nullptr;
            weights.back().group_size = 128;
            weights.back().groups_per_row = 1;
            weights.back().is_q4_g128_packed = true;
            weights.back().q4_g128_data = bg128[i];
            outputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1,
                actual[i].data()));

            Tensor reference = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1,
                expected[i].data());
            kernel_matmul_fp32(inputs.back(), weights.back(), reference);
        }

        ThreadPool pool(4);
        CHECK(kernel_matmul_int4_gemv_batch(
                  inputs, weights, outputs, &pool),
              "INT4 BG128 batched GEMV is supported");
        for (int i = 0; i < batch; ++i) {
            CHECK(check_approx(
                      actual[i].data(), expected[i].data(), N, 1e-5f),
                  "INT4 BG128 batched GEMV matches individual GEMV");
            delete[] q4dot[i];
            delete[] bg128[i];
        }
    }

    // DeepSeek-V4's compressor has two FP32 projections with the same input.
    // The batched path must only coalesce worker dispatch, not arithmetic.
    {
        constexpr int batch = 2;
        constexpr int K = 257;
        constexpr int N = 70;
        std::vector<float> input(K);
        for (int k = 0; k < K; ++k)
            input[k] = ((k * 17) % 43 - 21) * 0.0078125f;
        std::vector<std::vector<float>> weight_data(
            batch, std::vector<float>((size_t)N * K));
        std::vector<std::vector<float>> expected(
            batch, std::vector<float>(N));
        std::vector<std::vector<float>> actual(
            batch, std::vector<float>(N));
        std::vector<Tensor> inputs;
        std::vector<Tensor> weights;
        std::vector<Tensor> outputs;
        for (int i = 0; i < batch; ++i) {
            for (int n = 0; n < N; ++n)
                for (int k = 0; k < K; ++k)
                    weight_data[i][(size_t)n * K + k] =
                        ((n * 11 + k * 5 + i * 7) % 37 - 18) * 0.00390625f;
            inputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1,
                input.data()));
            weights.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1,
                weight_data[i].data()));
            outputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                actual[i].data()));
            Tensor reference = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                expected[i].data());
            ThreadPool reference_pool(4);
            kernel_matmul_fp32(
                inputs.back(), weights.back(), reference, &reference_pool);
        }

        ThreadPool pool(4);
        CHECK(kernel_matmul_fp32_gemv_batch(
                  inputs, weights, outputs, &pool),
              "FP32 batched GEMV is supported");
        for (int i = 0; i < batch; ++i) {
            CHECK(std::memcmp(
                      actual[i].data(), expected[i].data(),
                      N * sizeof(float)) == 0,
                  "FP32 batched GEMV exactly matches individual GEMV");
        }
    }

    // FP32 graph weights converted from a BF16 checkpoint can use an exact
    // two-byte sidecar. Include values outside the exactly-representable FP16
    // subset and require bit-identical FP32 accumulation results.
    {
        constexpr int batch = 2;
        constexpr int K = 257;
        constexpr int N = 70;
        std::vector<float> input(K);
        for (int k = 0; k < K; ++k)
            input[k] = ((k * 7) % 31 - 15) * 0.009765625f;
        std::vector<std::vector<float>> weight_data(
            batch, std::vector<float>((size_t)N * K));
        std::vector<std::vector<float>> expected(
            batch, std::vector<float>(N));
        std::vector<std::vector<float>> actual(
            batch, std::vector<float>(N));
        std::vector<Tensor> inputs;
        std::vector<Tensor> weights;
        std::vector<Tensor> outputs;
        PackedWeightMap packed;
        PreparedWeightMap prepared;
        for (int i = 0; i < batch; ++i) {
            for (size_t j = 0; j < weight_data[i].size(); ++j) {
                const uint32_t bits =
                    ((j + i) & 1 ? 0x80000000u : 0u) |
                    (static_cast<uint32_t>(110 + (j % 28)) << 23) |
                    (static_cast<uint32_t>((j * 13 + i * 5) % 128) << 16);
                std::memcpy(&weight_data[i][j], &bits, sizeof(bits));
            }
            inputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1,
                input.data()));
            weights.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1,
                weight_data[i].data()));
            Tensor reference_weight = weights.back();
            Tensor reference = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                expected[i].data());
            kernel_matmul_fp32(
                inputs.back(), reference_weight, reference);
            prepare_matmul_weight(
                weights.back(), "exact_bf16_" + std::to_string(i),
                weight_data[i].data(), packed, prepared);
            const bool expects_bf16_sidecar =
                mollm::cpu::capabilities().fp16_interleaved_weights;
            CHECK((weights.back().fp32_bf16_data != nullptr) ==
                      expects_bf16_sidecar,
                  "exact BF16 sidecar follows provider capability");
            outputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                actual[i].data()));
        }
        ThreadPool pool(4);
        CHECK(kernel_matmul_fp32_gemv_batch(
                  inputs, weights, outputs, &pool),
              "exact BF16 sidecars support batched GEMV");
        for (int i = 0; i < batch; ++i) {
            CHECK(std::memcmp(
                      actual[i].data(), expected[i].data(),
                      N * sizeof(float)) == 0,
                  "exact BF16 sidecar preserves FP32 GEMV bits");
        }
    }

    {
        constexpr int M = 5;
        constexpr int K = 129;
        constexpr int N = 70;
        std::vector<float> input((size_t)M * K);
        std::vector<float> weight_data((size_t)N * K);
        std::vector<float> expected((size_t)M * N);
        std::vector<float> actual((size_t)M * N);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = ((i * 11) % 41 - 20) * 0.0068359375f;
        for (size_t i = 0; i < weight_data.size(); ++i) {
            const uint32_t bits =
                (i & 1 ? 0x80000000u : 0u) |
                (static_cast<uint32_t>(112 + (i % 24)) << 23) |
                (static_cast<uint32_t>((i * 9) % 128) << 16);
            std::memcpy(&weight_data[i], &bits, sizeof(bits));
        }
        Tensor a = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1,
            input.data());
        Tensor weight = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1,
            weight_data.data());
        Tensor reference = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1,
            expected.data());
        Tensor output = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1,
            actual.data());
        kernel_matmul_fp32(a, weight, reference);
        PackedWeightMap packed;
        PreparedWeightMap prepared;
        prepare_matmul_weight(
            weight, "exact_bf16_gemm", weight_data.data(), packed, prepared);
        ThreadPool pool(4);
        kernel_matmul_fp32(a, weight, output, &pool);
        CHECK(std::memcmp(
                  actual.data(), expected.data(),
                  actual.size() * sizeof(float)) == 0,
              "exact BF16 sidecar preserves FP32 GEMM bits");
    }

    // BG32 uses the same batched-GEMV API with one embedded scale per
    // 32-value dot block.
    if (matmul_int4_q4dot_kernel_available()) {
        constexpr int batch = 2;
        constexpr int K = 64;
        constexpr int N = 32;
        constexpr int groups_per_row = K / 32;
        std::vector<std::vector<float>> input(
            batch, std::vector<float>(K));
        for (int i = 0; i < batch; ++i)
            for (int k = 0; k < K; ++k)
                input[i][k] =
                    ((k * (7 + i * 4)) % (29 + i * 2) - 14) * 0.015625f;

        std::vector<std::vector<uint8_t>> packed(
            batch, std::vector<uint8_t>((size_t)N * K / 2, 0));
        std::vector<std::vector<float>> scales(
            batch, std::vector<float>((size_t)N * groups_per_row));
        std::vector<uint8_t*> q4dot(batch, nullptr);
        std::vector<uint8_t*> bg32(batch, nullptr);
        std::vector<std::vector<float>> expected(
            batch, std::vector<float>(N));
        std::vector<std::vector<float>> actual(
            batch, std::vector<float>(N));
        std::vector<Tensor> inputs;
        std::vector<Tensor> weights;
        std::vector<Tensor> outputs;
        for (int i = 0; i < batch; ++i) {
            for (int n = 0; n < N; ++n) {
                for (int g = 0; g < groups_per_row; ++g)
                    scales[i][n * groups_per_row + g] =
                        0.006f + 0.00025f * ((n + g + i) % 9);
                for (int k = 0; k < K; ++k) {
                    int q = ((n * 13 + k * 5 + i * 3) % 15) - 7;
                    size_t index = (size_t)n * (K / 2) + k / 2;
                    if (k & 1)
                        packed[i][index] |= (uint8_t)((q & 15) << 4);
                    else
                        packed[i][index] |= (uint8_t)(q & 15);
                }
            }
            q4dot[i] =
                pack_b_q4dot_int4_full(packed[i].data(), N, K, K);
            bg32[i] = pack_b_q4dot_g32_full(
                q4dot[i], scales[i].data(), N, K, groups_per_row);
            inputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1,
                input[i].data()));
            weights.push_back(Tensor::create(
                Precision::INT4, MemoryType::EXTERNAL, N, K, 1, 1,
                bg32[i]));
            weights.back().group_size = 32;
            weights.back().groups_per_row = groups_per_row;
            weights.back().is_q4_g32_packed = true;
            weights.back().q4_g32_data = bg32[i];
            outputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                actual[i].data()));

            Tensor reference = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                expected[i].data());
            kernel_matmul_fp32(inputs.back(), weights.back(), reference);
        }

        ThreadPool pool(4);
        CHECK(kernel_matmul_int4_gemv_batch(
                  inputs, weights, outputs, &pool),
              "INT4 BG32 batched GEMV is supported");
        for (int i = 0; i < batch; ++i) {
            CHECK(check_approx(
                      actual[i].data(), expected[i].data(), N, 1e-5f),
                  "INT4 BG32 batched GEMV matches individual GEMV");
            delete[] q4dot[i];
            delete[] bg32[i];
        }
    }

    // Small RWKV LoRA projections use W8. Verify that independent inputs can
    // share one worker dispatch without changing quantization or output.
    if (matmul_int4_q4dot_kernel_available()) {
        constexpr int batch = 2;
        constexpr int K = 128;
        constexpr int N = 64;
        std::vector<std::vector<float>> input(
            batch, std::vector<float>(K));
        std::vector<std::vector<int8_t>> weight_data(
            batch, std::vector<int8_t>((size_t)N * K));
        std::vector<std::vector<float>> scales(
            batch, std::vector<float>(N));
        std::vector<int8_t*> packed(batch, nullptr);
        std::vector<std::vector<float>> expected(
            batch, std::vector<float>(N));
        std::vector<std::vector<float>> actual(
            batch, std::vector<float>(N));
        std::vector<Tensor> inputs, weights, outputs;
        for (int i = 0; i < batch; ++i) {
            for (int k = 0; k < K; ++k)
                input[i][k] =
                    ((k * (5 + i * 2)) % 31 - 15) * 0.0125f;
            for (int n = 0; n < N; ++n) {
                scales[i][n] = 0.003f + 0.0001f * ((n + i) % 7);
                for (int k = 0; k < K; ++k)
                    weight_data[i][(size_t)n * K + k] =
                        (int8_t)(((n * 13 + k * 7 + i * 5) % 255) - 127);
            }
            packed[i] = pack_b_q8dot_int8_full(
                weight_data[i].data(), N, K, K);
            inputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1,
                input[i].data()));
            weights.push_back(Tensor::create(
                Precision::INT8, MemoryType::EXTERNAL, N, K, 1, 1,
                weight_data[i].data()));
            weights.back().scales = scales[i].data();
            weights.back().group_size = 128;
            weights.back().groups_per_row = 1;
            weights.back().q8_repack_data = packed[i];
            outputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                actual[i].data()));
            Tensor reference = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                expected[i].data());
            kernel_matmul_fp32(inputs.back(), weights.back(), reference);
        }

        ThreadPool pool(4);
        CHECK(kernel_matmul_int8_gemv_batch(
                  inputs, weights, outputs, &pool),
              "INT8 batched GEMV is supported");
        for (int i = 0; i < batch; ++i) {
            CHECK(check_approx(
                      actual[i].data(), expected[i].data(), N, 1e-5f),
                  "INT8 batched GEMV matches individual GEMV");
            delete[] packed[i];
        }
    }

    // DeepSeek-V4's shared gate/up projections use the same activation and
    // native FP8 weights. The batched path must preserve the individual FP8
    // activation quantization while reusing it across both GEMVs.
    if (matmul_int4_q4dot_kernel_available()) {
        constexpr int batch = 2;
        constexpr int K = 128;
        constexpr int N = 64;
        constexpr int q8_groups = K / 32;
        std::vector<float> input(K);
        for (int k = 0; k < K; ++k)
            input[k] = ((k * 11) % 41 - 20) * 0.03125f;

        std::vector<std::vector<uint8_t>> weight_data(
            batch, std::vector<uint8_t>((size_t)N * K));
        std::vector<std::vector<uint8_t>> e8m0_scales(
            batch, std::vector<uint8_t>(1));
        std::vector<std::vector<int8_t>> packed(
            batch,
            std::vector<int8_t>(pack_fp8_e4m3_q8dot_bytes(N, K)));
        std::vector<std::vector<float>> packed_scales(
            batch, std::vector<float>((size_t)N * q8_groups));
        std::vector<std::vector<float>> expected(
            batch, std::vector<float>(N));
        std::vector<std::vector<float>> actual(
            batch, std::vector<float>(N));
        std::vector<Tensor> inputs, weights, outputs;
        for (int i = 0; i < batch; ++i) {
            e8m0_scales[i][0] = static_cast<uint8_t>(127 - i);
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; ++k) {
                    const float value =
                        ((n * 7 + k * (3 + i * 2)) % 31 - 15) * 0.125f;
                    weight_data[i][(size_t)n * K + k] =
                        encode_fp8_e4m3fn(value);
                }
            }
            CHECK(pack_fp8_e4m3_q8dot(
                      weight_data[i].data(), e8m0_scales[i].data(),
                      N, K, packed[i].data(), packed_scales[i].data()),
                  "pack FP8 sidecar for batched GEMV");
            // Deliberately reuse the exact Tensor/data pointer. This is the
            // shared-expert gate/up case and should quantize the activation
            // only once.
            inputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1,
                input.data()));
            weights.push_back(Tensor::create(
                Precision::FP8_E4M3, MemoryType::EXTERNAL, N, K, 1, 1,
                weight_data[i].data()));
            weights.back().e8m0_scales = e8m0_scales[i].data();
            weights.back().group_size = 128;
            weights.back().groups_per_row = 1;
            weights.back().num_groups = 1;
            weights.back().is_fp8_block128 = true;
            weights.back().q8_repack_data = packed[i].data();
            weights.back().fp8_q8_scales = packed_scales[i].data();
            outputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                actual[i].data()));
            Tensor reference = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                expected[i].data());
            kernel_matmul_fp32(inputs.back(), weights.back(), reference);
        }

        ThreadPool pool(4);
        CHECK(kernel_matmul_int8_gemv_batch(
                  inputs, weights, outputs, &pool),
              "FP8 sidecar batched GEMV is supported");
        for (int i = 0; i < batch; ++i) {
            CHECK(check_approx(
                      actual[i].data(), expected[i].data(), N, 1e-5f),
                  "FP8 batched GEMV matches individual GEMV");
        }
    }

    // Ready DeepSeek experts can be executed in one worker-pool dispatch.
    // Gate/up routes share the hidden input; down routes keep independent
    // intermediate inputs. Cover both forms in the same batch.
    {
        constexpr int batch = 3;
        constexpr int K = 128;
        constexpr int N = 64;
        constexpr int groups = K / 32;
        std::vector<float> shared_input(K);
        std::vector<float> distinct_input(K);
        for (int k = 0; k < K; ++k) {
            shared_input[k] = ((k * 13) % 47 - 23) * 0.01953125f;
            distinct_input[k] = ((k * 17) % 53 - 26) * 0.015625f;
        }
        std::vector<std::vector<uint8_t>> packed(
            batch, std::vector<uint8_t>((size_t)N * K / 2));
        std::vector<std::vector<uint8_t>> scales(
            batch, std::vector<uint8_t>((size_t)N * groups));
        std::vector<std::vector<float>> expected(
            batch, std::vector<float>(N));
        std::vector<std::vector<float>> actual(
            batch, std::vector<float>(N));
        std::vector<Tensor> inputs, weights, outputs;
        for (int i = 0; i < batch; ++i) {
            for (int n = 0; n < N; ++n) {
                for (int group = 0; group < groups; ++group)
                    scales[i][(size_t)n * groups + group] =
                        static_cast<uint8_t>(126 + ((n + group + i) % 3));
                for (int k = 0; k < K; ++k) {
                    const uint8_t nibble =
                        static_cast<uint8_t>(
                            (n * 5 + k * (3 + i * 2)) & 15);
                    uint8_t& byte =
                        packed[i][(size_t)n * (K / 2) + k / 2];
                    if (k & 1)
                        byte |= static_cast<uint8_t>(nibble << 4);
                    else
                        byte = nibble;
                }
            }
            float* input_data =
                i < 2 ? shared_input.data() : distinct_input.data();
            inputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1,
                input_data));
            weights.push_back(Tensor::create(
                Precision::MXFP4, MemoryType::EXTERNAL, N, K, 1, 1,
                packed[i].data()));
            weights.back().e8m0_scales = scales[i].data();
            weights.back().group_size = 32;
            weights.back().groups_per_row = groups;
            weights.back().num_groups = N * groups;
            outputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                actual[i].data()));
            Tensor reference = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
                expected[i].data());
            kernel_matmul_fp32(inputs.back(), weights.back(), reference);
        }
        ThreadPool pool(4);
        CHECK(kernel_matmul_mxfp4_gemv_batch(
                  inputs, weights, outputs, &pool),
              "MXFP4 batched GEMV is supported");
        for (int i = 0; i < batch; ++i) {
            CHECK(check_approx(
                      actual[i].data(), expected[i].data(), N, 1e-5f),
                  "MXFP4 batched GEMV matches individual GEMV");
        }
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
        B.is_q4_g128_packed = true;
        B.scales = nullptr;
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
        int M = 16, K = 128, N = 19;
        uint32_t group_size = 128;
        uint32_t groups_per_row = 1;
        std::vector<float> a_data(M * K);
        std::vector<int8_t> q_data(N * K);
        std::vector<uint8_t> packed((size_t)N * ((K + 1) / 2), 0);
        std::vector<float> scales(N * groups_per_row);
        std::vector<float> deq(N * K);
        std::vector<float> c_data(M * N);
        std::vector<float> ref_c(M * N);

        for (int i = 0; i < M * K; i++) {
            a_data[i] = ((i % 29) - 14) * 0.013671875f;
        }
        int row_stride = (K + 1) / 2;
        for (int n = 0; n < N; n++) {
            scales[n] = 0.0045f + 0.0004f * (float)(n % 9);
            for (int k = 0; k < K; k++) {
                int8_t q = (int8_t)(((n * 19 + k * 7) % 15) - 7);
                q_data[n * K + k] = q;
                deq[n * K + k] = (float)q * scales[n];
                uint8_t nibble = (uint8_t)q & 0x0F;
                uint8_t& byte = packed[(size_t)n * row_stride + (k >> 1)];
                if (k & 1) byte |= (uint8_t)(nibble << 4);
                else byte |= nibble;
            }
        }

        uint8_t* q4_repack = pack_b_q4dot_int4_full(packed.data(), N, K, K);
        uint8_t* q4_g128 = pack_b_q4dot_g128_full(
            q4_repack, scales.data(), N, K, groups_per_row);
        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data.data());
        Tensor B = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, N, K, 1, 1, packed.data());
        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1, c_data.data());
        B.scales = scales.data();
        B.group_size = group_size;
        B.groups_per_row = groups_per_row;
        B.num_groups = (uint32_t)(N * groups_per_row);
        B.q4_repack_data = q4_repack;
        B.q4_g128_data = q4_g128;

        ThreadPool pool(4);
        kernel_matmul_fp32(A, B, C, &pool);
        ref_matmul(a_data.data(), deq.data(), ref_c.data(), M, N, K);

        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 5e-2f),
              "INT4 Q8-dot GEMM BG128 2D odd N threads=4");
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
        std::vector<float> c_repack = c_data;

        uint8_t* q4_g32 = pack_b_q4dot_g32_full(
            q4_repack, scales.data(), N, K, groups_per_row);
        B.q4_g32_data = q4_g32;
        kernel_matmul_fp32(A, B, C);
        CHECK(check_approx(c_data.data(), c_repack.data(), M * N, 1e-5f),
              "INT4 Q8-dot GEMM BG32 matches q4dot repack");
        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 2e-2f),
              "INT4 Q8-dot GEMM BG32 reference");

        uint8_t* q4_vnni = nullptr;
        PreparedWeight prepared;
        if (mollm::cpu::capabilities().x86_avx512_vnni) {
            q4_vnni = pack_b_q4_vnni_full(q4_g32, N, K);
            const size_t vnni_bytes = pack_b_q4_vnni_bytes(N, K);
            prepared.layout(WeightLayout::X86_VNNI_Q4_G32)
                .assign(q4_vnni, q4_vnni + vnni_bytes);
            B.prepared_weight = &prepared;
            kernel_matmul_fp32(A, B, C);
            CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 2e-2f),
                  "INT4 Q8-dot GEMM BG32 x86 VNNI reference");
        }

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
        delete[] q4_g32;
        delete[] q4_vnni;
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

    {
        int M = 16, K = 64, N = 19;
        std::vector<float> a_data(M * K);
        std::vector<int8_t> q_data(N * K);
        std::vector<float> scales(N);
        std::vector<float> deq(N * K);
        std::vector<float> c_data(M * N);
        std::vector<float> ref_c(M * N);

        for (int i = 0; i < M * K; i++) {
            a_data[i] = ((i % 23) - 11) * 0.015625f;
        }
        for (int n = 0; n < N; n++) {
            scales[n] = 0.0075f + 0.0007f * (float)(n % 5);
            for (int k = 0; k < K; k++) {
                q_data[n * K + k] = (int8_t)(((n * 17 + k * 9) % 95) - 47);
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

        ThreadPool pool(4);
        kernel_matmul_fp32(A, B, C, &pool);
        ref_matmul(a_data.data(), deq.data(), ref_c.data(), M, N, K);
        CHECK(check_approx(c_data.data(), ref_c.data(), M * N, 8e-2f),
              "INT8 Q8-dot GEMM repackable interleaved 2D odd N");
        delete[] packed;
        delete[] q8_repack;
    }

    // Large FP16 matmul with the original row-major weight retained beside
    // the interleaved CPU pack.  This is an exactness test for the sidecar
    // path, so do not exercise the separately-tested FP16 accumulation mode.
    // On Apple it exercises Accelerate SGEMM; on ARM it exercises the FP32
    // accumulation NEON path.
    {
        int M=96,K=64,N=64;
        std::vector<float> a(M*K),w32(N*K),out(M*N),ref(M*N);
        std::vector<__fp16> w16(N*K);
        fill_rand(a.data(),M*K); fill_rand(w32.data(),N*K);
        for(int i=0;i<N*K;++i) w16[i]=(__fp16)w32[i];
        for(int i=0;i<N*K;++i) w32[i]=(float)w16[i];
        __fp16* packed=pack_b_interleaved_full(w16.data(),N,K,K);
        Tensor A=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,K,M,1,1,a.data());
        Tensor B=Tensor::create(Precision::FP16,MemoryType::EXTERNAL,N,K,1,1,packed);
        Tensor C=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,N,M,1,1,out.data());
        B.is_interleaved=true; B.rowmajor_data=w16.data();
        ThreadPool pool(4);
        const bool previous_force_fp32_acc = g_mollm_force_fp32_acc;
        g_mollm_force_fp32_acc = true;
        kernel_matmul_fp32(A,B,C,&pool);
        g_mollm_force_fp32_acc = previous_force_fp32_acc;
        ref_matmul(a.data(),w32.data(),ref.data(),M,N,K);
        // NEON uses fused FP32 FMA while the scalar reference accumulates
        // separately; the resulting cross-platform rounding delta is below
        // 2e-3 for this K=64 exact-value sidecar check.
        CHECK(check_approx(out.data(),ref.data(),M*N,2e-3f),
              "FP16 large GEMM row-major sidecar");
        delete[] packed;
    }

    // Sparse-A decode GEMV: exact zeros from ReLU-squared are omitted while
    // FP16/W8/W4 results remain consistent with their dequantized references.
    {
        constexpr int K = 128, N = 16;
        std::vector<float> a(K, 0.f), w(N*K), ref(N), out(N);
        a[3] = 0.5f; a[37] = 1.25f; a[68] = 0.75f; a[111] = 2.f;
        for (int i = 0; i < N*K; ++i) w[i] = ((i * 13) % 31 - 15) * 0.01f;
        ref_matmul(a.data(), w.data(), ref.data(), 1, N, K);
        std::vector<__fp16> h(N*K);
        for (int i = 0; i < N*K; ++i) h[i] = (__fp16)w[i];
        __fp16* hp = pack_b_interleaved_full(h.data(), N, K, K);
        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1, a.data());
        Tensor B = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, N, K, 1, 1, hp);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1, out.data());
        if (mollm::cpu::capabilities().arm_neon) {
            kernel_gemv_sparse_a(A, B, C);
            CHECK(check_approx(out.data(), ref.data(), N, 3e-3f),
                  "sparse-A FP16 GEMV");
        } else {
            CHECK(true, "sparse-A FP16 is an ARM provider optimization");
        }
        delete[] hp;

        std::vector<int8_t> q8(N*K);
        std::vector<float> s8(N), deq8(N*K), ref8(N);
        for (int n = 0; n < N; ++n) {
            s8[n] = 0.01f;
            for (int k = 0; k < K; ++k) {
                q8[n*K+k] = (int8_t)(((n*7+k*5)%31)-15);
                deq8[n*K+k] = q8[n*K+k] * s8[n];
            }
        }
        int8_t* q8p = pack_b_interleaved_int8_full(q8.data(), N, K, K);
        B = Tensor::create(Precision::INT8, MemoryType::EXTERNAL, N, K, 1, 1, q8.data());
        B.sparse_data=q8p; B.scales=s8.data(); B.group_size=K; B.groups_per_row=1;
        ref_matmul(a.data(), deq8.data(), ref8.data(), 1, N, K);
        kernel_gemv_sparse_a(A, B, C);
        CHECK(check_approx(out.data(), ref8.data(), N, 4e-3f), "sparse-A W8 GEMV");
        delete[] q8p;

        std::vector<uint8_t> q4rows((size_t)N*K/2, 0);
        std::vector<float> s4(N, 0.02f), deq4(N*K), ref4(N);
        for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) {
            int q = ((n*3+k*5)%15)-7;
            size_t idx=(size_t)n*(K/2)+k/2;
            if(k&1) q4rows[idx]|=(uint8_t)((q&15)<<4); else q4rows[idx]|=(uint8_t)(q&15);
            deq4[n*K+k]=q*s4[n];
        }
        uint8_t* q4dot=pack_b_q4dot_int4_full(q4rows.data(),N,K,K);
        uint8_t* bg=pack_b_q4dot_g128_full(q4dot,s4.data(),N,K,1);
        int8_t* q4s=pack_b_sparse_int4_g128_full(bg,N,K);
        B=Tensor::create(Precision::INT4,MemoryType::EXTERNAL,N,K,1,1,bg);
        B.sparse_data=q4s; B.q4_g128_data=bg; B.scales=s4.data();
        B.group_size=128; B.groups_per_row=1;
        ref_matmul(a.data(),deq4.data(),ref4.data(),1,N,K);
        if (mollm::cpu::capabilities().arm_neon) {
            kernel_gemv_sparse_a(A, B, C);
            CHECK(check_approx(out.data(), ref4.data(), N, 4e-3f),
                  "sparse-A W4 GEMV");
        } else {
            CHECK(true, "sparse-A W4 is an ARM provider optimization");
        }
        delete[] q4dot; delete[] bg; delete[] q4s;
    }

    // Production-scale FP16 sparse accumulation: RWKV FFN value projections
    // have thousands of inputs and roughly half survive ReLU-squared.
    {
        constexpr int K = 8192, N = 256;
        std::vector<float> a(K, 0.f), w(N*K), ref(N), out(N);
        for (int k = 0; k < K; k += 2)
            a[k] = 0.001f * float((k * 17) % 101);
        std::vector<__fp16> h(N*K);
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                h[n*K+k] = (__fp16)(0.0002f * float((n*13+k*7)%63-31));
                w[n*K+k] = (float)h[n*K+k];
            }
        }
        ref_matmul(a.data(), w.data(), ref.data(), 1, N, K);
        __fp16* hp = pack_b_interleaved_full(h.data(), N, K, K);
        Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  K, 1, 1, 1, a.data());
        Tensor B = Tensor::create(Precision::FP16, MemoryType::EXTERNAL,
                                  N, K, 1, 1, hp);
        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  N, 1, 1, 1, out.data());
        ThreadPool pool(4);
        if (mollm::cpu::capabilities().arm_neon) {
            kernel_gemv_sparse_a(A, B, C, &pool);
            CHECK(check_approx(out.data(), ref.data(), N, 1e-4f),
                  "sparse-A FP16 GEMV production K");
        } else {
            CHECK(true, "sparse-A production path is an ARM provider optimization");
        }
        delete[] hp;
    }

    if (failures == 0) {
        printf("\nAll matmul tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
