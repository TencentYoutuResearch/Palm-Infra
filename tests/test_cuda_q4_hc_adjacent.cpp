#include "kernels/cuda/matmul.h"

#include "cuda_test_support.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using mollm_test::check;
using mollm_test::from_bits;
using mollm_test::GuardedBuffer;
using mollm_test::require;
using mollm_test::same_bits;
using mollm_test::to_bits;

constexpr int inner = 10240;
constexpr int down_rows = 320;
constexpr int inject_rows = 4;
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
            values[i] = (i % 4 == 0) ? from_bits(2) : 0.f;
    }
    if (change == 6) values[13] = from_bits(0x7fc12345u);
    if (change == 7) values[17] = from_bits(0x7f800000u);
    if (change == 8) {
        values[17] = from_bits(0x7f800000u);
        values[23] = from_bits(0xff800000u);
    }
    return values;
}

void require_tail(const std::vector<float>& output, size_t rows, const char* message) {
    for (size_t i = rows; i < output.size(); ++i)
        require(to_bits(output[i]) == 0x5a5a5a5au, message);
}

void run() {
    check(mollm_cuda::configure_q4_g32_hc_adjacent_cache(), "configure HC adjacent cache");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");

    std::vector<Q4B8G32Block> down(size_t(down_rows / 8) * (inner / 32));
    std::vector<Q4B8G32Block> inject(inner / 32);
    uint32_t state = 813171;
    const auto random = [&state] {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    // Every nibble pattern is valid for the biased CUDA BG32 layout.
    for (auto* weights : {&down, &inject}) {
        for (auto& block : *weights) {
            for (int row = 0; row < 8; ++row) {
                block.scales[row] = float((random() % 1024) + 1) * 0.000013f;
                for (auto& code : block.q[row])
                    code = static_cast<uint8_t>(random());
            }
        }
    }

    GuardedBuffer down_weight_buffer(down.size() * sizeof(Q4B8G32Block));
    GuardedBuffer inject_weight_buffer(inject.size() * sizeof(Q4B8G32Block));
    GuardedBuffer input_buffer((inner + 4) * sizeof(float));
    GuardedBuffer expected_down_buffer((down_rows + tail) * sizeof(float));
    GuardedBuffer expected_inject_buffer((inject_rows + tail) * sizeof(float));
    GuardedBuffer actual_down_buffer((down_rows + tail) * sizeof(float));
    GuardedBuffer actual_inject_buffer((inject_rows + tail) * sizeof(float));
    auto* down_weight = down_weight_buffer.as<Q4B8G32Block>();
    auto* inject_weight = inject_weight_buffer.as<Q4B8G32Block>();
    float* x = input_buffer.as<float>();
    float* expected_down = expected_down_buffer.as<float>();
    float* expected_inject = expected_inject_buffer.as<float>();
    float* actual_down = actual_down_buffer.as<float>();
    float* actual_inject = actual_inject_buffer.as<float>();
    down_weight_buffer.upload(down.data());
    inject_weight_buffer.upload(inject.data());
    // The work below runs on a non-blocking stream; finish the uploads first.
    check(cudaDeviceSynchronize(), "weight upload");

    const auto launch = [&](const float* input, const Q4B8G32Block* a, const Q4B8G32Block* b, float* y,
                            float* z) {
        return mollm_cuda::launch_q4_g32_hc_adjacent_projections(input, a, b, y, z, stream);
    };
    const auto rejected = [&](const float* input, const Q4B8G32Block* a, const Q4B8G32Block* b, float* y,
                              float* z) {
        return launch(input, a, b, y, z) == cudaErrorInvalidValue;
    };

    // Rejected calls must not enqueue work into a capturing stream.
    cudaGraph_t empty;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    require(rejected(nullptr, down_weight, inject_weight, actual_down, actual_inject), "null input accepted");
    require(rejected(x, nullptr, inject_weight, actual_down, actual_inject), "null down weight accepted");
    require(rejected(x, down_weight, nullptr, actual_down, actual_inject), "null inject weight accepted");
    require(rejected(x, down_weight, inject_weight, nullptr, actual_inject), "null down output accepted");
    require(rejected(x, down_weight, inject_weight, actual_down, nullptr), "null inject output accepted");
    require(rejected(x + 2, down_weight, inject_weight, actual_down, actual_inject), "8-byte input accepted");
    require(rejected(x, misaligned(down_weight), inject_weight, actual_down, actual_inject),
            "unaligned weight accepted");
    require(rejected(x, down_weight, inject_weight, misaligned(actual_down), actual_inject),
            "unaligned output accepted");
    check(cudaStreamEndCapture(stream, &empty), "end capture");
    size_t nodes = 0;
    check(cudaGraphGetNodes(empty, nullptr, &nodes), "graph nodes");
    require(nodes == 0, "invalid calls enqueued work");
    check(cudaGraphDestroy(empty), "graph destroy");

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    check(launch(x, down_weight, inject_weight, actual_down, actual_inject), "HC adjacent launch");
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "graph instantiate");

    std::vector<float> original(inner);
    for (float& value : original)
        value = float(int(random() % 2001) - 1000) * 0.00073f;
    for (int change = 0; change < input_changes; ++change) {
        const std::vector<float> values = changed_input(original, change);
        check(cudaMemcpyAsync(x, values.data(), inner * sizeof(float), cudaMemcpyHostToDevice, stream),
              "input upload");
        for (GuardedBuffer* buffer :
             {&expected_down_buffer, &expected_inject_buffer, &actual_down_buffer, &actual_inject_buffer})
            check(cudaMemsetAsync(buffer->data(), 0x5a, buffer->bytes(), stream), "output fill");
        // Independent existing 256-thread kernels supply the exact native reference.
        check(mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(x, down_weight, expected_down, down_rows, inner,
                                                              1, true, stream),
              "down reference launch");
        check(mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(x, inject_weight, expected_inject, inject_rows,
                                                              inner, 1, true, stream),
              "inject reference launch");
        check(launch(x, down_weight, inject_weight, actual_down, actual_inject), "HC adjacent launch");
        check(cudaStreamSynchronize(stream), "eager");
        const auto reference_down = expected_down_buffer.download<float>();
        const auto reference_inject = expected_inject_buffer.download<float>();
        require(same_bits(reference_down, actual_down_buffer.download<float>()) &&
                    same_bits(reference_inject, actual_inject_buffer.download<float>()),
                "eager FP32/tree mismatch");

        check(cudaMemsetAsync(actual_down, 0x5a, actual_down_buffer.bytes(), stream), "output fill");
        check(cudaMemsetAsync(actual_inject, 0x5a, actual_inject_buffer.bytes(), stream), "output fill");
        check(cudaGraphLaunch(exec, stream), "graph launch");
        check(cudaStreamSynchronize(stream), "graph");
        const auto result_down = actual_down_buffer.download<float>();
        const auto result_inject = actual_inject_buffer.download<float>();
        require(same_bits(reference_down, result_down) && same_bits(reference_inject, result_inject),
                "changed-input graph/guard mismatch");
        require_tail(result_down, down_rows, "down guard changed");
        require_tail(result_inject, inject_rows, "inject guard changed");
    }
    require(same_bits(down_weight_buffer.download<Q4B8G32Block>(), down) &&
                same_bits(inject_weight_buffer.download<Q4B8G32Block>(), inject),
            "weights changed");
    for (const GuardedBuffer* buffer :
         {&down_weight_buffer, &inject_weight_buffer, &input_buffer, &expected_down_buffer,
          &expected_inject_buffer, &actual_down_buffer, &actual_inject_buffer})
        buffer->check_guards();
    check(cudaGraphExecDestroy(exec), "graph exec destroy");
    check(cudaGraphDestroy(graph), "graph destroy");
    check(cudaStreamDestroy(stream), "stream destroy");
}

}  // namespace

int main() {
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    run();
    std::puts("HC adjacent: native FP32/NaN/Inf/signed-zero/subnormal, changed graph, guards, "
              "immutable weights and empty invalid capture PASS");
    return 0;
}
