#include "kernels/cuda/matmul.h"
#include "backends/cuda/backend.h"
#include "graph/graph.h"

#include "cuda_test_support.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

using mollm_test::check;
using mollm_test::from_bits;
using mollm_test::GuardedBuffer;
using mollm_test::require;
using mollm_test::same_bits;

constexpr int tokens = 3;
// Extra output elements after the rows; they must keep the 0x5a fill.
constexpr int tail = 32;
constexpr int input_changes = 9;

template<class T> T* misaligned(T* pointer) {
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(pointer) + 1);
}

// Change 0 is the original input; 1..2 perturb it; 3..8 cover zeros, signed
// zeros, subnormals, a NaN payload and infinities.
std::vector<float> changed_input(const std::vector<float>& original, int change) {
    std::vector<float> values = original;
    for (size_t i = 0; i < values.size(); ++i) {
        if (change == 1)
            values[i] = -values[i];
        else if (change == 2)
            values[i] = values[i] * 0.5f + float(int(i % 13) - 6) * 0.00017f;
        else if (change == 3)
            values[i] = 0.f;
        else if (change == 4)
            values[i] = from_bits(0x80000000u);
        else if (change == 5)
            values[i] = i % 4 ? 0.f : from_bits(2);
    }
    if (change == 6) values[13] = from_bits(0x7fc12345u);
    if (change == 7) values[17] = from_bits(0x7f800000u);
    if (change == 8) {
        values[17] = from_bits(0x7f800000u);
        values[23] = from_bits(0xff800000u);
    }
    return values;
}

// Runs the same projection through CudaBackend dispatch and returns its output.
std::vector<float> backend_matmul(const std::vector<Q4B8G32Block>& weights, const std::vector<float>& values,
                                  int columns, int inner) {
    auto raw = weights;
    for (auto& block : raw) {
        for (auto& row : block.q) {
            for (auto& code : row)
                code ^= 0x88;
        }
    }
    CudaBackend backend;
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    Tensor weight = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, columns, inner, 1, 1, raw.data());
    weight.rowmajor_data = raw.data();
    weight.q4_g32_data = raw.data();
    weight.is_q4_g32_packed = true;
    weight.group_size = 32;
    weight.groups_per_row = inner / 32;
    backend.wrap_weight_int4(weight);
    Tensor input = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, inner, tokens);
    Tensor output = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, columns, tokens);
    backend.alloc_persistent(input, input.nbytes(), PersistentHostAccess::NONE);
    backend.alloc_persistent(output, output.nbytes(), PersistentHostAccess::NONE);
    require(backend.copy_from_host(values.data(), input, input.nbytes()), "backend input copy failed");
    GraphNode node;
    node.op_type = OpType::MATMUL;
    backend.dispatch(node, {&input, &weight}, &output, nullptr);
    std::vector<float> got(size_t(tokens) * columns);
    require(!backend.dispatch_failed() && backend.copy_to_host(output, got.data(), output.nbytes()),
            "backend dispatch/copy failed");
    return got;
}

void run(int columns, int inner, bool backend_check, bool fixed) {
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    std::vector<Q4B8G32Block> weights(size_t(columns / 8) * (inner / 32));
    uint32_t state = 813171;
    const auto random = [&state] {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    for (auto& block : weights) {
        for (int row = 0; row < 8; ++row) {
            block.scales[row] = float((random() % 1024) + 1) * 0.000013f;
            for (auto& code : block.q[row])
                code = static_cast<uint8_t>(random());
        }
    }
    GuardedBuffer weight_buffer(weights.size() * sizeof(Q4B8G32Block));
    GuardedBuffer input_buffer((size_t(tokens) * inner + 4) * sizeof(float));
    GuardedBuffer expected_buffer((size_t(tokens) * columns + tail) * sizeof(float));
    GuardedBuffer actual_buffer((size_t(tokens) * columns + tail) * sizeof(float));
    auto* w = weight_buffer.as<Q4B8G32Block>();
    float* x = input_buffer.as<float>();
    float* expected = expected_buffer.as<float>();
    float* actual = actual_buffer.as<float>();
    weight_buffer.upload(weights.data());
    // The work below runs on a non-blocking stream; finish the upload first.
    check(cudaDeviceSynchronize(), "weight upload");

    const auto launch = [&](const float* input, const Q4B8G32Block* weight, float* output, int n, int k) {
        auto kernel = fixed ? mollm_cuda::launch_q4_g32_m3_fixed : mollm_cuda::launch_q4_g32_m3_adjacent;
        return kernel(input, weight, output, n, k, stream);
    };
    const auto rejected = [&](const float* input, const Q4B8G32Block* weight, float* output, int n, int k) {
        return launch(input, weight, output, n, k) == cudaErrorInvalidValue;
    };
    if (!backend_check) {
        // Rejected calls must not enqueue work into a capturing stream.
        cudaGraph_t empty;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        require(rejected(nullptr, w, actual, columns, inner), "null input accepted");
        require(rejected(x, nullptr, actual, columns, inner), "null weight accepted");
        require(rejected(x, w, nullptr, columns, inner), "null output accepted");
        require(rejected(x + 2, w, actual, columns, inner), "8-byte input accepted");
        require(rejected(x, misaligned(w), actual, columns, inner), "unaligned weight accepted");
        require(rejected(x, w, misaligned(actual), columns, inner), "unaligned output accepted");
        const std::vector<std::pair<int, int>> invalid = fixed
            ? std::vector<std::pair<int, int>>{{320, 10240}, {640, 2560}, {320, 2560}, {0, 10240}}
            : std::vector<std::pair<int, int>>{{2560, 6144}, {512, 2560}, {320, 2560}, {0, 10240}};
        for (const auto& shape : invalid)
            require(rejected(x, w, actual, shape.first, shape.second), "unqualified shape accepted");
        check(cudaStreamEndCapture(stream, &empty), "end capture");
        size_t nodes = 0;
        check(cudaGraphGetNodes(empty, nullptr, &nodes), "graph nodes");
        require(nodes == 0, "invalid calls enqueued work");
        check(cudaGraphDestroy(empty), "graph destroy");
    }

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    check(launch(x, w, actual, columns, inner), "M3 launch");
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "graph instantiate");
    std::vector<float> original(size_t(tokens) * inner);
    for (float& value : original)
        value = float(int(random() % 2001) - 1000) * 0.00073f;
    for (int change = 0; change < input_changes; ++change) {
        const std::vector<float> values = changed_input(original, change);
        check(cudaMemcpyAsync(x, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice,
                              stream),
              "input upload");
        check(cudaMemsetAsync(expected, 0x5a, expected_buffer.bytes(), stream), "expected fill");
        check(cudaMemsetAsync(actual, 0x5a, actual_buffer.bytes(), stream), "actual fill");
        check(mollm_cuda::launch_q4_g32_small_batch_reduced(x, w, expected, tokens, columns, inner, inner,
                                                            columns, stream),
              "reference launch");
        check(launch(x, w, actual, columns, inner), "M3 launch");
        check(cudaStreamSynchronize(stream), "eager");
        const auto reference = expected_buffer.download<float>();
        require(same_bits(reference, actual_buffer.download<float>()), "eager native FP32/guard mismatch");

        check(cudaMemsetAsync(actual, 0x5a, actual_buffer.bytes(), stream), "actual fill");
        check(cudaGraphLaunch(exec, stream), "graph launch");
        check(cudaStreamSynchronize(stream), "graph");
        require(same_bits(reference, actual_buffer.download<float>()), "changed graph/guard mismatch");

        if (backend_check && change < 3) {
            const std::vector<float> got = backend_matmul(weights, values, columns, inner);
            require(!std::memcmp(reference.data(), got.data(), got.size() * sizeof(float)),
                    "backend native FP32 mismatch");
        }
    }
    check(cudaStreamSynchronize(stream), "final");
    require(same_bits(weight_buffer.download<Q4B8G32Block>(), weights), "weights changed");
    for (const GuardedBuffer* buffer : {&weight_buffer, &input_buffer, &expected_buffer, &actual_buffer})
        buffer->check_guards();
    check(cudaGraphExecDestroy(exec), "graph exec destroy");
    check(cudaGraphDestroy(graph), "graph destroy");
    check(cudaStreamDestroy(stream), "stream destroy");
}

}  // namespace

int main(int argc, char** argv) {
    bool backend_check = false;
    bool fixed = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0) {
            backend_check = true;
        } else if (std::strcmp(argv[i], "--fixed") == 0) {
            fixed = true;
        } else {
            std::fprintf(stderr, "usage: %s [--fixed] [--backend]\n", argv[0]);
            return 2;
        }
    }
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    if (fixed) {
        run(2560, 6144, backend_check, true);
        run(512, 2560, backend_check, true);
    } else {
        run(320, 10240, backend_check, false);
        run(640, 2560, backend_check, false);
    }
    std::puts("M3 projections: native FP32/NaN/Inf/signed-zero/subnormal, mutable graph, guards, "
              "immutable weights and invalid capture PASS");
    return 0;
}
