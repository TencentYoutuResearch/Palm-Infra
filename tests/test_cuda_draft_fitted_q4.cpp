#include "backends/cuda/backend.h"
#include "kernels/cuda/draft_compact_head.h"

#include "cuda_test_support.h"

#include <algorithm>
#include <cfenv>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using mollm_test::check;
using mollm_test::GuardedBuffer;
using mollm_test::require;

// Scalar CPU reference: integer dot products, scalar division and explicit
// BF16 rounding. It does not call the GPU quantizer or its reduction helpers.
uint16_t cpu_group(const int8_t* weights, float scale, uint8_t* codes, bool fitted = true) {
    int maximum = 0, quantized[32] = {};
    for (int i = 0; i < 32; ++i) maximum = std::max(maximum, std::abs(int(weights[i])));
    float stored_maximum = float(maximum);
    if (fitted) {
        float step = float(maximum) / 7.f;
        for (int iteration = 0; iteration < 5; ++iteration) {
            int numerator = 0, denominator = 0;
            for (int i = 0; i < 32; ++i) {
                quantized[i] = step > 0.f
                    ? std::clamp(int(std::nearbyint(float(weights[i]) / step)), -7, 7) : 0;
                numerator += int(weights[i]) * quantized[i];
                denominator += quantized[i] * quantized[i];
            }
            step = denominator ? float(numerator) / float(denominator) : 0.f;
        }
        stored_maximum = 7.f * step;
    } else {
        for (int i = 0; i < 32; ++i)
            quantized[i] = maximum ? std::clamp(
                int(std::nearbyint(float(weights[i]) * (7.f / float(maximum)))), -7, 7) : 0;
    }
    for (int i = 0; i < 16; ++i)
        codes[i] = uint8_t((quantized[2 * i] + 8) | ((quantized[2 * i + 1] + 8) << 4));
    const uint32_t bits = mollm_test::to_bits(stored_maximum * scale);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

bool nan_bf16(uint16_t value) {
    return (value & 0x7f80u) == 0x7f80u && (value & 0x7fu);
}

void compare(const std::vector<mollm_cuda::Q4CompactG32>& actual,
             const std::vector<mollm_cuda::Q4CompactG32>& expected) {
    require(actual.size() == expected.size(), "packed count");
    for (size_t i = 0; i < actual.size(); ++i) {
        require(!std::memcmp(actual[i].q, expected[i].q, sizeof(actual[i].q)), "CPU code bytes");
        for (int row = 0; row < 8; ++row)
            require(actual[i].scales[row] == expected[i].scales[row] ||
                        (nan_bf16(actual[i].scales[row]) && nan_bf16(expected[i].scales[row])),
                    "CPU BF16 scale bits / NaN semantics");
    }
}

void test_packing() {
    using Block = mollm_cuda::Q4CompactG32;
    const auto pack = mollm_cuda::pack_q8_draft_head_compact_fitted;
    require(pack(nullptr, nullptr, nullptr, 1, 32, cudaStreamPerThread) == cudaErrorInvalidValue,
            "null inputs rejected");
    for (int inner : {32, 64, 2560}) {
        for (int rows : {1, 17, 257}) {
            std::vector<int8_t> weights(size_t(rows) * inner);
            std::vector<float> scales(rows);
            for (int row = 0; row < rows; ++row) {
                scales[row] = (row % 3 == 0 ? -1.f : 1.f) * (.001f * (1 + row % 7));
                for (int j = 0; j < inner; ++j)
                    weights[size_t(row) * inner + j] = row == 0 ? 0 : int8_t((row * 13 + j * 29) % 256 - 128);
                if (row % 13 == 8) scales[row] = std::numeric_limits<float>::denorm_min();
                if (row % 13 == 9) scales[row] = INFINITY;
                if (row % 13 == 10) scales[row] = -INFINITY;
                if (row % 13 == 11) scales[row] = NAN;
                if (row % 13 == 12) scales[row] = -0.f;
            }
            std::vector<Block> expected(size_t((rows + 7) / 8) * (inner / 32));
            const auto reference = [&] {
                std::fill(expected.begin(), expected.end(), Block{});
                for (int row = 0; row < rows; ++row) {
                    for (int group = 0; group < inner / 32; ++group) {
                        auto& block = expected[size_t(row / 8) * (inner / 32) + group];
                        block.scales[row % 8] = cpu_group(
                            weights.data() + size_t(row) * inner + group * 32, scales[row],
                            block.q[row % 8]);
                    }
                }
            };
            reference();
            GuardedBuffer source_weights(weights.size());
            GuardedBuffer source_scales(scales.size() * sizeof(float));
            GuardedBuffer packed(expected.size() * sizeof(Block));
            source_weights.upload(weights.data());
            source_scales.upload(scales.data());
            const auto* device_weights = source_weights.as<int8_t>();
            const auto* device_scales = source_scales.as<float>();
            auto* output = packed.as<Block>();
            const auto read_and_compare = [&] {
                compare(packed.download<Block>(), expected);
            };
            check(pack(device_weights, device_scales, output, rows, inner, cudaStreamPerThread));
            read_and_compare();
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            const auto stream = cudaStreamPerThread;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            check(pack(device_weights, device_scales, output, rows, inner, stream));
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 2; ++replay) {
                weights[0] = replay ? -128 : 127;
                source_weights.upload(weights.data());
                reference();
                check(cudaGraphLaunch(executable, stream));
                check(cudaStreamSynchronize(stream));
                read_and_compare();
            }
            check(cudaGraphExecDestroy(executable));
            check(cudaGraphDestroy(graph));
            const auto* unaligned_scales =
                reinterpret_cast<const float*>(reinterpret_cast<const char*>(device_scales) + 1);
            auto* unaligned_output = reinterpret_cast<Block*>(reinterpret_cast<char*>(output) + 1);
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            require(pack(device_weights, device_scales, output, INT_MAX, inner, stream) ==
                        cudaErrorInvalidValue,
                    "row overflow rejected");
            require(pack(device_weights, device_scales, output, rows, inner - 1, stream) ==
                        cudaErrorInvalidValue,
                    "inner alignment rejected");
            require(pack(device_weights, unaligned_scales, output, rows, inner, stream) ==
                        cudaErrorInvalidValue,
                    "scale alignment rejected");
            require(pack(device_weights, device_scales, unaligned_output, rows, inner, stream) ==
                        cudaErrorInvalidValue,
                    "output alignment rejected");
            check(cudaStreamEndCapture(stream, &graph));
            size_t nodes = 1;
            check(cudaGraphGetNodes(graph, nullptr, &nodes));
            require(nodes == 0, "invalid calls do not enqueue work");
            check(cudaGraphDestroy(graph));
            packed.check_guards();
            require(source_weights.download<int8_t>() == weights &&
                        mollm_test::same_bits(source_scales.download<float>(), scales),
                    "source weights and scales immutable");
        }
    }
    std::puts("fitted Q4 primitive: CPU codes/scales, padded rows, graphs, guards, immutable sources PASS");
}

void test_backend() {
    constexpr int rows = 248320, inner = 2560;
    std::vector<int8_t> weights(size_t(rows) * inner, 0);
    std::vector<float> scales(rows, 1.f);
    weights[0] = weights[inner] = 100;
    weights[1] = 12;
    weights[inner + 1] = 13;
    CudaBackend backend;
    Tensor weight = Tensor::create(Precision::INT8, MemoryType::EXTERNAL, rows, inner, 1, 1, weights.data());
    weight.rowmajor_data = weights.data();
    weight.scales = scales.data();
    weight.group_size = inner;
    weight.groups_per_row = 1;
    backend.wrap_weight(weight);
    Tensor input = Tensor::create(Precision::FP32, MemoryType::NONE, inner, 1);
    require(backend.alloc_output(input, input.nbytes(), nullptr), "input allocation");
    std::vector<float> values(inner, 0.f);
    values[1] = 1.f;
    require(backend.copy_from_host(values.data(), input, inner * sizeof(float)), "input upload");
    int device, major, minor;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    const bool fitted = major == 12 && minor == 1;
    const bool screened = fitted;
    float logits[2];
    for (int row = 0; row < 2; ++row) {
        uint8_t codes[16];
        const auto code = cpu_group(weights.data() + size_t(row) * inner, 1.f, codes, fitted);
        const float maximum = mollm_test::from_bits(uint32_t(code) << 16);
        logits[row] = float(int(codes[0] >> 4) - 8) * (maximum / 7.f);
    }
    const int expected = logits[1] > logits[0] ? 1 : 0;
    require(expected == (fitted ? 1 : 0), "fixture distinguishes fitted and legacy Q4");
    require(backend.lm_head_argmax_device_and_end_graph(input, 0, weight, rows, inner) == 1,
            "original target starts exact");
    const int active_rows = screened ? 1024 : rows;
    for (int repeat = 0; repeat < 2; ++repeat) {
        // Without a probability request every device drafts with legacy Q4,
        // which ties the fixture rows and keeps the first.
        require(backend.lm_head_draft_argmax_device_and_end_graph(input, 0, weight, rows, inner) == 0,
                "no-probability legacy Q4 cache restored");
        float probability = -99.f;
        require(backend.lm_head_draft_argmax_device_and_end_graph(
                    input, 0, weight, rows, inner, 0, nullptr, &probability) == expected,
                "probability request selects fitted / legacy Q4");
        const double maximum = std::max(logits[0], logits[1]);
        const double denominator = std::exp(double(logits[0]) - maximum) +
                                   std::exp(double(logits[1]) - maximum) +
                                   (active_rows - 2.) * std::exp(-maximum);
        const double reference = std::exp(double(logits[expected]) - maximum) / denominator;
        require(std::abs(double(probability) - reference) < 3e-6 * reference,
                "normalized draft probability versus CPU double softmax");
        require(backend.lm_head_argmax_device_and_end_graph(input, 0, weight, rows, inner) == 1,
                "target remains exact after draft scope");
    }
    std::fill(values.begin(), values.end(), 0.f);
    require(backend.copy_from_host(values.data(), input, inner * sizeof(float)), "changed input upload");
    float probability = -99.f;
    require(backend.lm_head_draft_argmax_device_and_end_graph(
                input, 0, weight, rows, inner, 0, nullptr, &probability) == 0,
            "zero-input stable tie");
    require(std::abs(double(probability) - 1. / active_rows) < 3e-6 / active_rows,
            "uniform probability over full or screened vocabulary");
    require(backend.lm_head_argmax_device_and_end_graph(input, 0, weight, rows, inner) == 0 &&
                !backend.dispatch_failed(),
            "changed target exact, no dispatch failure");
    backend.free_output(input, nullptr);
    std::puts("fitted Q4 full vocabulary: probability, cache transitions, target isolation PASS");
}

}  // namespace

int main() {
    std::fesetround(FE_TONEAREST);
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    test_packing();
    test_backend();
    std::puts("fitted Q4 packing, normalized probability, cache transitions and target isolation PASS");
}
