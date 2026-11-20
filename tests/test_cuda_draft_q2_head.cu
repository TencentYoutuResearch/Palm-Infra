#include "backends/cuda/backend.h"
#include "kernels/cuda/draft_compact_head.h"
#include "kernels/cuda/matmul.h"

#include "cuda_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Block = mollm_cuda::Q2DraftG32;
using mollm_test::check;
using mollm_test::GuardedBuffer;
using mollm_test::require;

// Typed device array with stream-ordered transfers (no guard bands).
template<class T> struct Device {
    T* data = nullptr;
    size_t count;
    explicit Device(size_t n) : count(n) { check(cudaMalloc(&data, n * sizeof(T))); }
    ~Device() { cudaFree(data); }
    void put(const std::vector<T>& values, cudaStream_t stream) {
        require(values.size() == count, "upload extent");
        check(cudaMemcpyAsync(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    std::vector<T> get(cudaStream_t stream) const {
        std::vector<T> values(count);
        check(cudaMemcpyAsync(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return values;
    }
};

// Scalar CPU fit, independent of the warp shuffle/packing implementation.
static std::vector<Block> reference_pack(
    const std::vector<int8_t>& weights, const std::vector<float>& scales, int n, int k) {
    std::vector<Block> result(size_t(n / 8) * (k / 32));
    for (int row = 0; row < n; ++row) {
        for (int group = 0; group < k / 32; ++group) {
            const int8_t* values = weights.data() + size_t(row) * k + group * 32;
            int maximum = 0;
            for (int j = 0; j < 32; ++j) maximum = std::max(maximum, std::abs(int(values[j])));
            float step = float(maximum) / 3.f;
            for (int iteration = 0; iteration < 4; ++iteration) {
                int numerator = 0, denominator = 0;
                for (int j = 0; j < 32; ++j) {
                    const int magnitude = std::abs(int(values[j])) > 2.f * step ? 3 : 1;
                    const int q = values[j] < 0 ? -magnitude : magnitude;
                    numerator += int(values[j]) * q;
                    denominator += q * q;
                }
                step = float(numerator) / float(denominator);
            }
            int codes[32], numerator = 0, denominator = 0;
            for (int j = 0; j < 32; ++j) {
                const int magnitude = std::abs(int(values[j])) > 2.f * step ? 3 : 1;
                const int q = values[j] < 0 ? -magnitude : magnitude;
                codes[j] = (q + 3) / 2;
                numerator += int(values[j]) * q;
                denominator += q * q;
            }
            auto& block = result[size_t(row / 8) * (k / 32) + group];
            const float fitted = (3.f * (float(numerator) / float(denominator))) * scales[row];
            const uint32_t bits = mollm_test::to_bits(fitted);
            block.scales[row % 8] = uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
            for (int packet = 0; packet < 4; ++packet) {
                unsigned word = 0;
                for (int j = 0; j < 8; ++j)
                    word |= unsigned(codes[packet * 4 + (j < 4 ? j : j + 12)]) << (j * 2);
                block.codes[row % 8][packet] = uint16_t(word);
            }
        }
    }
    return result;
}

// Materialize coefficients as ordinary Q4 with FP32 scales. The established
// native wide GEMV supplies an independent oracle for all output bits.
__global__ void expand_reference(const Block* input, Q4B8G32Block* output, int n, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int group = 0; group < k / 32; ++group) {
        const auto& source = input[size_t(row / 8) * (k / 32) + group];
        auto& destination = output[size_t(row / 8) * (k / 32) + group];
        destination.scales[row % 8] = __fdiv_rn(
            __uint_as_float(unsigned(source.scales[row % 8]) << 16), 3.f);
        for (int packet = 0; packet < 4; ++packet) {
            for (int j = 0; j < 8; ++j) {
                const int column = packet * 4 + (j < 4 ? j : j + 12);
                const int value = int((source.codes[row % 8][packet] >> (j * 2)) & 3) * 2 - 3;
                auto& byte = destination.q[row % 8][column / 2];
                if (column % 2)
                    byte |= (value + 8) << 4;
                else
                    byte = value + 8;
            }
        }
    }
}

static void test_operator(int n, int k, bool cpu_oracle) {
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const size_t blocks = size_t(n / 8) * (k / 32), bytes = blocks * sizeof(Block);
    std::vector<int8_t> weights(size_t(n) * k);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = int((i * 1664525u + 1013904223u) >> 16) % 256 - 128;
    for (int row = 0; row < n; row += 11)
        std::fill_n(weights.data() + size_t(row) * k, 32, int8_t(0));
    std::vector<float> scales(n);
    for (int row = 0; row < n; ++row)
        scales[row] = std::ldexp(float(row % 11 + 1), -12) * (row % 3 ? 1.f : -1.f);

    Device<int8_t> dw(weights.size());
    Device<float> ds(n);
    Device<float> input(k);
    Device<Q4B8G32Block> expanded(blocks);
    GuardedBuffer packed(bytes);
    GuardedBuffer output(n * sizeof(float));
    GuardedBuffer reference(n * sizeof(float));
    auto* q2 = packed.as<Block>();
    dw.put(weights, stream);
    ds.put(scales, stream);
    for (int bad_rows : {0, 17, INT32_MAX})
        require(mollm_cuda::pack_q8_draft_head_q2(dw.data, ds.data, q2, bad_rows, k, stream)
                    == cudaErrorInvalidValue, "reject invalid rows");
    for (int bad_inner : {0, 33, 2592})
        require(mollm_cuda::launch_q2_draft_head_gemv(input.data, q2, output.as<float>(),
                    n, bad_inner, stream) == cudaErrorInvalidValue, "reject invalid K");
    require(mollm_cuda::launch_q2_draft_head_gemv(input.data + 1, q2, output.as<float>(),
                n, k, stream) == cudaErrorInvalidValue, "reject unaligned input");
    require(mollm_cuda::pack_q8_draft_head_q2(dw.data, ds.data,
                reinterpret_cast<Block*>(packed.as<unsigned char>() + 1), n, k, stream)
                == cudaErrorInvalidValue, "reject unaligned weights");
    require(mollm_cuda::pack_q8_draft_head_q2(nullptr, ds.data, q2, n, k, stream)
                == cudaErrorInvalidValue, "reject null source");
    require(mollm_cuda::launch_q2_draft_head_gemv(input.data, q2, nullptr, n, k, stream)
                == cudaErrorInvalidValue, "reject null output");
    check(mollm_cuda::pack_q8_draft_head_q2(dw.data, ds.data, q2, n, k, stream));
    check(cudaStreamSynchronize(stream));
    const auto original_packed = packed.download<unsigned char>();
    if (cpu_oracle) {
        const auto expected = reference_pack(weights, scales, n, k);
        require(!std::memcmp(expected.data(), original_packed.data(), bytes),
                "packed codes/scales versus independent CPU quantizer");
    }
    expand_reference<<<(n + 127) / 128, 128, 0, stream>>>(q2, expanded.data, n, k);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    check(mollm_cuda::launch_q2_draft_head_gemv(input.data, q2, output.as<float>(), n, k, stream));
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int fixture = 0; fixture < 5; ++fixture) {
        std::vector<float> values(k);
        for (int j = 0; j < k; ++j)
            values[j] = float(int((unsigned(j + fixture * k) * 1103515245u) >> 16) % 2001 - 1000) * .001f;
        input.put(values, stream);
        check(cudaMemsetAsync(reference.data(), 0x5a, reference.bytes(), stream));
        // The reference launcher uses the per-thread default stream.
        check(cudaStreamSynchronize(stream));
        mollm_cuda::launch_q4_g32_uncached_wide_gemv(input.data, expanded.data, reference.as<float>(), n, k);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        const auto expected = reference.download<float>();
        reference.check_guards();
        for (bool replay : {false, true}) {
            check(cudaMemsetAsync(output.data(), 0x5a, output.bytes(), stream));
            if (replay)
                check(cudaGraphLaunch(executable, stream));
            else
                check(mollm_cuda::launch_q2_draft_head_gemv(input.data, q2, output.as<float>(), n, k,
                                                            stream));
            check(cudaStreamSynchronize(stream));
            require(mollm_test::same_bits(output.download<float>(), expected),
                    "native Q4 reference, eager/mutable graph");
            output.check_guards();
        }
        require(input.get(stream) == values, "input remains unchanged");
    }
    require(packed.download<unsigned char>() == original_packed, "packed weights remain unchanged");
    packed.check_guards();
    require(dw.get(stream) == weights && ds.get(stream) == scales, "source weights/scales remain unchanged");
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));
    std::printf("draft Q2 operator passed n=%d k=%d\n", n, k);
}

static void test_backend_cache() {
    constexpr int n = 248320, k = 2560;
    std::vector<int8_t> weights(size_t(n) * k, 0);
    std::vector<float> scales(n, 1.f), values(k, 0.f);
    weights[0] = weights[k] = 100;
    weights[1] = 14;
    weights[k + 1] = 15;
    values[1] = 1.f;
    CudaBackend backend;
    require(backend.available(), "backend available");
    Tensor weight = Tensor::create(Precision::INT8, MemoryType::EXTERNAL, n, k, 1, 1, weights.data());
    weight.rowmajor_data = weights.data();
    weight.scales = scales.data();
    weight.group_size = k;
    weight.groups_per_row = 1;
    backend.wrap_weight(weight);
    Tensor input = Tensor::create(Precision::FP32, MemoryType::NONE, k, 1);
    require(backend.alloc_output(input, input.nbytes(), nullptr), "backend input allocation");
    require(backend.copy_from_host(values.data(), input, k * sizeof(float)), "backend input upload");
    const int draft_id = 1;
    // These weights distinguish Q4 (tie ->0), fitted Q2 (->1), and exact Q8 (->1).
    for (int repeat = 0; repeat < 2; ++repeat) {
        require(backend.lm_head_argmax_device_and_end_graph(input, 0, weight, n, k) == 1,
                "target remains exact before draft");
        require(backend.lm_head_draft_argmax_device_and_end_graph(input, 0, weight, n, k) == draft_id,
                "no-probability draft chooses configured representation");
        float probability = 99.f;
        const int id = backend.lm_head_draft_argmax_device_and_end_graph(
            input, 0, weight, n, k, 0, nullptr, &probability);
        require(id == 0, "probability request retains previous Q4/Q8 choice");
        const double expected = 1. / (2. + (n - 2.) * std::exp(-double(100.f / 7.f)));
        require(std::abs(double(probability) - expected) < 3e-6 * expected,
                "probability uses complete Q4 logits");
        require(backend.lm_head_draft_argmax_device_and_end_graph(input, 0, weight, n, k) == draft_id,
                "cache layout/probability scope restored for next draft");
        require(backend.lm_head_argmax_device_and_end_graph(input, 0, weight, n, k) == 1,
                "draft/cache switches do not modify target head");
    }
    std::fill(values.begin(), values.end(), 0.f);
    require(backend.copy_from_host(values.data(), input, k * sizeof(float)), "changed backend input");
    require(backend.lm_head_draft_argmax_device_and_end_graph(input, 0, weight, n, k) == 0,
            "zero-input draft tie");
    float probability = 99.f;
    require(backend.lm_head_draft_argmax_device_and_end_graph(
                input, 0, weight, n, k, 0, nullptr, &probability) == 0, "zero-input Q4/Q8 tie");
    require(std::abs(double(probability) - 1. / n) < 3e-6 / n, "uniform Q4 probability");
    require(backend.lm_head_argmax_device_and_end_graph(input, 0, weight, n, k) == 0,
            "target sees changed input");
    require(!backend.dispatch_failed(), "backend remains healthy");
    backend.free_output(input, nullptr);
    std::puts("draft Q2 backend role, representation switching, probability and target isolation passed");
}

int main(int argc, char** argv) {
    const char* mode = argc == 2 ? argv[1] : "";
    const bool backend = std::strcmp(mode, "--backend") == 0;
    const bool large = std::strcmp(mode, "--large") == 0;
    if (argc > 2 || (argc == 2 && !backend && !large)) {
        std::fprintf(stderr, "usage: %s [--large | --backend]\n", argv[0]);
        return 2;
    }
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    if (backend) {
        test_backend_cache();
    } else if (large) {
        test_operator(248320, 2560, false);
    } else {
        for (int k : {32, 64, 2560}) test_operator(40, k, true);
    }
    std::puts("draft Q2 PASS");
}
