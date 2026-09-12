#include "kernels/cpu/matmul/matmul.h"
#include "runtime/threading.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool close(float lhs, float rhs, float relative_tolerance = 1e-5f) {
    return std::fabs(lhs - rhs) <=
        relative_tolerance * std::max(1.0f, std::fabs(rhs));
}

}  // namespace

int main() {
    constexpr int M = 2;
    constexpr int N = 11;
    constexpr int K = 16;
    const float e2m1[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    std::vector<float> a(M * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            a[m * K + k] = (m + 1) * (k - 7.0f) / 8.0f;

    std::vector<uint8_t> packed(N * K / 2);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            const uint8_t low = static_cast<uint8_t>((k + n) & 15);
            const uint8_t high = static_cast<uint8_t>((k + n + 1) & 15);
            packed[n * (K / 2) + k / 2] =
                static_cast<uint8_t>(low | (high << 4));
        }
    }
    // E4M3FN encodings of 1.0, 2.0, 0.5, repeated to cover two four-row
    // vector tiles and the scalar tail.
    std::vector<uint8_t> block_scales(N);
    std::vector<float> row_scales(N);
    for (int n = 0; n < N; ++n) {
        constexpr uint8_t codes[3] = {0x38, 0x40, 0x30};
        block_scales[n] = codes[n % 3];
        row_scales[n] = 0.25f * static_cast<float>(1 + ((n * 3) & 7));
    }
    std::vector<float> output(M * N, 0.0f);

    Tensor input = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a.data());
    Tensor weight = Tensor::create(
        Precision::NVFP4, MemoryType::EXTERNAL, N, K, 1, 1,
        packed.data());
    weight.group_size = 16;
    weight.groups_per_row = 1;
    weight.num_groups = N;
    weight.nvfp4_scales = block_scales.data();
    weight.nvfp4_row_scales = row_scales.data();
    Tensor result = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, N, M, 1, 1,
        output.data());
    kernel_matmul_fp32(input, weight, result, nullptr);

    const float decoded_block_scale_values[3] = {1.0f, 2.0f, 0.5f};
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k)
                expected += a[m * K + k] * e2m1[(k + n) & 15];
            expected *= decoded_block_scale_values[n % 3] * row_scales[n];
            if (!close(output[m * N + n], expected)) {
                std::fprintf(stderr,
                             "NVFP4 mismatch m=%d n=%d: %.8f != %.8f\n",
                             m, n, output[m * N + n], expected);
                return 1;
            }
        }
    }

    std::vector<float> batch_output(M * N, 0.0f);
    std::vector<Tensor> batch_inputs;
    std::vector<Tensor> batch_weights;
    std::vector<Tensor> batch_outputs;
    for (int m = 0; m < M; ++m) {
        batch_inputs.push_back(Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1,
            a.data() + m * K));
        batch_weights.push_back(weight);
        batch_outputs.push_back(Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, N, 1, 1, 1,
            batch_output.data() + m * N));
    }
    ThreadPool thread_pool(4);
    std::vector<const Tensor*> pairs;
    for (int m = 0; m < M; ++m) {
        pairs.push_back(&batch_inputs[m]);
        pairs.push_back(&batch_weights[m]);
    }
    Tensor batch_result = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, N, 1, M, 1,
        batch_output.data());
    // Exercise the public dispatcher: AArch64 takes the fused batch kernel,
    // while other targets must retain the portable per-matrix fallback.
    kernel_matmul_batch(pairs, batch_result, &thread_pool);
    const float batch_tolerance = 5e-3f;
    for (int i = 0; i < M * N; ++i) {
        if (!close(batch_output[i], output[i], batch_tolerance)) {
            std::fprintf(stderr,
                         "NVFP4 batch mismatch i=%d: %.8f != %.8f\n",
                         i, batch_output[i], output[i]);
            return 1;
        }
    }
    {
        constexpr int pair_n = 8;
        std::vector<uint8_t> pair_packed(pair_n * K / 2);
        if (!pack_nvfp4_q8_pairs(
                packed.data(), pair_packed.data(), pair_n, K)) {
            std::fprintf(stderr, "NVFP4 pair pack failed\n");
            return 1;
        }
        Tensor pair_weight = weight;
        pair_weight.shape[0] = pair_n;
        pair_weight.num_groups = pair_n;
        pair_weight.nvfp4_q8_pair_data = pair_packed.data();
        std::vector<Tensor> pair_weights(M, pair_weight);
        std::vector<float> pair_output(M * pair_n, 0.0f);
        std::vector<Tensor> pair_outputs;
        for (int m = 0; m < M; ++m) {
            pair_outputs.push_back(Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL, pair_n, 1, 1, 1,
                pair_output.data() + m * pair_n));
        }
        if (kernel_matmul_nvfp4_gemv_batch(
                batch_inputs, pair_weights, pair_outputs, &thread_pool)) {
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < pair_n; ++n) {
                    if (!close(
                            pair_output[m * pair_n + n],
                            batch_output[m * N + n], batch_tolerance)) {
                        std::fprintf(
                            stderr,
                            "NVFP4 pair-packed mismatch m=%d n=%d: "
                            "%.8f != %.8f\n",
                            m, n, pair_output[m * pair_n + n],
                            batch_output[m * N + n]);
                        return 1;
                    }
                }
            }
        }

        // The non-batched decode dispatcher is used when only one routed
        // expert is ready. It must select the pair kernel as well.
        std::vector<float> single_output(pair_n, 0.0f);
        Tensor single_result = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, pair_n, 1, 1, 1,
            single_output.data());
        kernel_matmul_fp32(
            batch_inputs[0], pair_weight, single_result, &thread_pool);
        for (int n = 0; n < pair_n; ++n) {
            if (!close(single_output[n], batch_output[n], batch_tolerance)) {
                std::fprintf(
                    stderr,
                    "NVFP4 pair-packed single mismatch n=%d: %.8f != %.8f\n",
                    n, single_output[n], batch_output[n]);
                return 1;
            }
        }

        // Prefill and speculative multi-token calls consume the same package
        // layout through the floating-point kernels.
        std::vector<float> pair_prefill_output(M * pair_n, 0.0f);
        Tensor pair_prefill_result = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, pair_n, M, 1, 1,
            pair_prefill_output.data());
        kernel_matmul_fp32(
            input, pair_weight, pair_prefill_result, &thread_pool);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < pair_n; ++n) {
                if (!close(pair_prefill_output[m * pair_n + n],
                           output[m * N + n])) {
                    std::fprintf(
                        stderr,
                        "NVFP4 pair-packed prefill mismatch m=%d n=%d: "
                        "%.8f != %.8f\n",
                        m, n, pair_prefill_output[m * pair_n + n],
                        output[m * N + n]);
                    return 1;
                }
            }
        }
    }
    std::puts("NVFP4 tests passed");
    return 0;
}
