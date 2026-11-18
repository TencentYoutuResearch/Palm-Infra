#include "kernels/cuda/matmul.h"
#include "backends/cuda/backend.h"
#include "graph/graph.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class T> struct Buffer {
    T* data = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) { check(cudaMalloc(&data, n * sizeof(T))); }
    ~Buffer() { if (data) cudaFree(data); }
    std::vector<T> read(cudaStream_t stream) const {
        std::vector<T> result(count);
        check(cudaMemcpyAsync(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return result;
    }
};
float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
void run(int columns, int inner, bool backend_check, bool fixed) {
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<Q4B8G32Block> weights(size_t(columns / 8) * (inner / 32));
    uint32_t state = 813171;
    auto random = [&] { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (auto& block : weights) for (int row = 0; row < 8; ++row) {
        block.scales[row] = float((random() % 1024) + 1) * 0.000013f;
        for (auto& code : block.q[row]) code = static_cast<uint8_t>(random());
    }
    Buffer<Q4B8G32Block> w(weights.size());
    Buffer<float> x(3 * inner + 4), expected(3 * columns + 32), actual(3 * columns + 32);
    check(cudaMemcpyAsync(w.data, weights.data(), weights.size() * sizeof(weights[0]), cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
    auto launch = [&](const float* input, const Q4B8G32Block* weight, float* output, int n, int k) {
        auto kernel = fixed ? mollm_cuda::launch_q4_g32_m3_fixed : mollm_cuda::launch_q4_g32_m3_adjacent;
        return kernel(input, weight, output, n, k, stream);
    };
    if (!backend_check) {
        cudaGraph_t empty;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        require(launch(nullptr, w.data, actual.data, columns, inner) == cudaErrorInvalidValue, "null input accepted");
        require(launch(x.data, nullptr, actual.data, columns, inner) == cudaErrorInvalidValue, "null weight accepted");
        require(launch(x.data, w.data, nullptr, columns, inner) == cudaErrorInvalidValue, "null output accepted");
        require(launch(x.data + 2, w.data, actual.data, columns, inner) == cudaErrorInvalidValue, "8-byte input accepted");
        require(launch(x.data, reinterpret_cast<const Q4B8G32Block*>(reinterpret_cast<const char*>(w.data) + 1), actual.data, columns, inner) == cudaErrorInvalidValue, "unaligned weight accepted");
        require(launch(x.data, w.data, reinterpret_cast<float*>(reinterpret_cast<char*>(actual.data) + 1), columns, inner) == cudaErrorInvalidValue, "unaligned output accepted");
        const std::vector<std::pair<int, int>> invalid = fixed ?
            std::vector<std::pair<int, int>>{{320, 10240}, {640, 2560}, {320, 2560}, {0, 10240}} :
            std::vector<std::pair<int, int>>{{2560, 6144}, {512, 2560}, {320, 2560}, {0, 10240}};
        for (const auto shape : invalid)
            require(launch(x.data, w.data, actual.data, shape.first, shape.second) == cudaErrorInvalidValue, "unqualified shape accepted");
        check(cudaStreamEndCapture(stream, &empty));
        size_t count = 0;
        check(cudaGraphGetNodes(empty, nullptr, &count));
        require(count == 0, "invalid calls enqueued work");
        check(cudaGraphDestroy(empty));
    }
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    check(launch(x.data, w.data, actual.data, columns, inner));
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    std::vector<float> original(3 * inner);
    for (float& value : original) value = float(int(random() % 2001) - 1000) * 0.00073f;
    for (int change = 0; change < 9; ++change) {
        auto values = original;
        for (size_t i = 0; i < values.size(); ++i) {
            if (change == 1) values[i] = -values[i];
            else if (change == 2) values[i] = values[i] * 0.5f + float(int(i % 13) - 6) * 0.00017f;
            else if (change == 3) values[i] = 0.f;
            else if (change == 4) values[i] = from_bits(0x80000000u);
            else if (change == 5) values[i] = i % 4 ? 0.f : from_bits(2);
        }
        if (change == 6) values[13] = from_bits(0x7fc12345u);
        if (change == 7) values[17] = from_bits(0x7f800000u);
        if (change == 8) { values[17] = from_bits(0x7f800000u); values[23] = from_bits(0xff800000u); }
        check(cudaMemcpyAsync(x.data, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        check(cudaMemsetAsync(expected.data, 0x5a, expected.count * sizeof(float), stream));
        check(cudaMemsetAsync(actual.data, 0x5a, actual.count * sizeof(float), stream));
        check(mollm_cuda::launch_q4_g32_small_batch_reduced(x.data, w.data, expected.data, 3, columns, inner, inner, columns, stream));
        check(launch(x.data, w.data, actual.data, columns, inner));
        const auto reference = expected.read(stream);
        auto result = actual.read(stream);
        require(!std::memcmp(reference.data(), result.data(), result.size() * sizeof(float)), "eager native FP32/guard mismatch");
        check(cudaMemsetAsync(actual.data, 0x5a, actual.count * sizeof(float), stream));
        check(cudaGraphLaunch(exec, stream));
        result = actual.read(stream);
        require(!std::memcmp(reference.data(), result.data(), result.size() * sizeof(float)), "changed graph/guard mismatch");
        if (backend_check && change < 3) {
            auto raw = weights;
            for (auto& block : raw) for (auto& row : block.q) for (auto& code : row) code ^= 0x88;
            CudaBackend backend;
            backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
            Tensor weight = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, columns, inner, 1, 1, raw.data());
            weight.rowmajor_data = raw.data(); weight.q4_g32_data = raw.data();
            weight.is_q4_g32_packed = true; weight.group_size = 32; weight.groups_per_row = inner / 32;
            backend.wrap_weight_int4(weight);
            Tensor input = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, inner, 3);
            Tensor output = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, columns, 3);
            backend.alloc_persistent(input, input.nbytes(), PersistentHostAccess::NONE);
            backend.alloc_persistent(output, output.nbytes(), PersistentHostAccess::NONE);
            require(backend.copy_from_host(values.data(), input, input.nbytes()), "backend input copy failed");
            GraphNode node; node.op_type = OpType::MATMUL;
            backend.dispatch(node, {&input, &weight}, &output, nullptr);
            std::vector<float> got(3 * columns);
            require(!backend.dispatch_failed() && backend.copy_to_host(output, got.data(), output.nbytes()), "backend dispatch/copy failed");
            require(!std::memcmp(reference.data(), got.data(), got.size() * sizeof(float)), "backend native FP32 mismatch");
        }
    }
    const auto final_weights = w.read(stream);
    require(!std::memcmp(final_weights.data(), weights.data(), weights.size() * sizeof(weights[0])), "weights changed");
    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
}
}
int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    bool backend_check = false, fixed = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0) backend_check = true;
        else if (std::strcmp(argv[i], "--fixed") == 0) fixed = true;
        else { std::fprintf(stderr, "Usage: %s [--fixed] [--backend]\n", argv[0]); return 2; }
    }
    if (backend_check) {
        setenv("MOLLM_CUDA_Q4_SMALL_BATCH_EXACT", "1", 1);
        setenv("MOLLM_CUDA_Q4_SMALL_BATCH_EXACT_REUSE", "1", 1);
    }
    try {
        if (fixed) { run(2560, 6144, backend_check, true); run(512, 2560, backend_check, true); }
        else { run(320, 10240, backend_check, false); run(640, 2560, backend_check, false); }
        std::puts("M3 projections: native FP32/NaN/Inf/signed-zero/subnormal, mutable graph, guards, immutable weights and invalid capture PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "M3 projections: %s\n", error.what()); return 1;
    }
}
