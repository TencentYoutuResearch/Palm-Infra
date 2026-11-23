// Latency of the scalar Q8 greedy head screen against the full Q8 GEMV at the
// production head shape (248320 x 2560). Correctness is covered by
// test_cuda_q8_head_screen; this only confirms both paths agree on the argmax
// before timing them.
#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/q8_head_screen.h"

#include "tests/cuda_test_support.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using mollm_cuda::ArgMaxPair;
using mollm_test::GuardedBuffer;
using mollm_test::check;
using mollm_test::require;

constexpr int rows = 248320;
constexpr int inner = 2560;
constexpr int warmup_iterations = 3;
constexpr int timed_iterations = 20;

// Same deterministic weights as test_cuda_q8_head_screen; rows 0 and 1 tie.
int8_t weight_code(size_t index) {
    uint32_t x = uint32_t(index);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return int8_t(x & 255);
}

std::vector<int8_t> make_weights() {
    std::vector<int8_t> weights(size_t(rows) * inner);
    for (int row = 0; row < rows; ++row) {
        const size_t source = size_t(row == 1 ? 0 : row) * inner;
        for (int column = 0; column < inner; ++column)
            weights[size_t(row) * inner + column] = weight_code(source + column);
    }
    return weights;
}

std::vector<float> make_scales() {
    std::vector<float> scales(rows);
    for (int row = 0; row < rows; ++row)
        scales[row] = row < 2 ? .001f : (row % 13 == 0 ? -.001f : .001f);
    return scales;
}

// Input 0 is all zeros, input 1 is correlated with row 0, input 2 is a sine wave.
std::vector<float> make_input(int fixture, const std::vector<int8_t>& weights) {
    std::vector<float> input(inner, 0.f);
    for (int i = 0; i < inner; ++i) {
        if (fixture == 1)
            input[i] = float(weights[i]) / 128.f;
        else if (fixture == 2)
            input[i] = std::sin(float(i) * .31f);
    }
    return input;
}

}  // namespace

int main() {
    if (!mollm_test::cuda_device_available())
        return mollm_test::skip_status;
    const std::vector<int8_t> weights = make_weights();
    const std::vector<float> scales = make_scales();

    GuardedBuffer w(weights.size());
    GuardedBuffer packed(size_t(rows) * inner / 4 * 3);
    GuardedBuffer sf(size_t(rows) * sizeof(float));
    GuardedBuffer x(inner * sizeof(float));
    GuardedBuffer approximate(size_t(rows) * sizeof(float));
    GuardedBuffer output(size_t(rows) * sizeof(float));
    GuardedBuffer reference(size_t(rows) * sizeof(float));
    GuardedBuffer norm(sizeof(double));
    GuardedBuffer lower(257 * sizeof(float));
    GuardedBuffer count(sizeof(unsigned));
    GuardedBuffer ids(size_t(rows) * sizeof(int));
    GuardedBuffer argmax_partial(256 * sizeof(ArgMaxPair));
    GuardedBuffer argmax_result(sizeof(ArgMaxPair));
    w.upload(weights.data());
    sf.upload(scales.data());
    check(cudaDeviceSynchronize(), "upload");

    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    check(mollm_cuda::pack_q8_head_screen(w.as<int8_t>(), packed.as<uint8_t>(), rows, stream), "pack");
    check(cudaStreamSynchronize(stream), "pack");

    const mollm_cuda::Q8HeadScreenScratch scratch{approximate.as<float>(), lower.as<float>(),
                                                  norm.as<double>(), count.as<unsigned>(),
                                                  ids.as<int>()};
    auto launch = [&](bool screen) {
        if (screen) {
            check(mollm_cuda::launch_q8_head_screen(x.as<float>(), w.as<int8_t>(), packed.as<uint8_t>(),
                                                    sf.as<float>(), output.as<float>(), rows, scratch,
                                                    stream),
                  "Q8 head screen");
        } else {
            check(mollm_cuda::launch_q8_tuned_gemv(x.as<float>(), w.as<int8_t>(), sf.as<float>(),
                                                   reference.as<float>(), rows, inner, 8, false, stream),
                  "Q8 GEMV");
        }
    };
    auto argmax = [&](const float* values) {
        mollm_cuda::launch_argmax(values, rows, argmax_partial.as<ArgMaxPair>(), 256,
                                  argmax_result.as<ArgMaxPair>());
        check(cudaGetLastError(), "argmax");
        return argmax_result.download<ArgMaxPair>()[0].index;
    };

    cudaEvent_t begin;
    cudaEvent_t end;
    check(cudaEventCreate(&begin), "event");
    check(cudaEventCreate(&end), "event");
    for (int fixture = 0; fixture < 3; ++fixture) {
        const std::vector<float> input = make_input(fixture, weights);
        check(cudaMemcpyAsync(x.data(), input.data(), x.bytes(), cudaMemcpyHostToDevice, stream), "input");
        launch(false);
        launch(true);
        check(cudaStreamSynchronize(stream), "sanity run");
        require(argmax(output.as<float>()) == argmax(reference.as<float>()),
                "screened argmax differs from full Q8");
        // Interleave the two paths (GEMV, screen, screen, GEMV) to expose drift.
        for (bool screen : {false, true, true, false}) {
            for (int i = 0; i < warmup_iterations; ++i)
                launch(screen);
            check(cudaEventRecord(begin, stream), "record");
            for (int i = 0; i < timed_iterations; ++i)
                launch(screen);
            check(cudaEventRecord(end, stream), "record");
            check(cudaEventSynchronize(end), "synchronize");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, begin, end), "elapsed");
            std::printf("fixture=%d path=%s ms=%.6f\n", fixture, screen ? "screen" : "q8_gemv",
                        ms / timed_iterations);
        }
    }
    check(cudaEventDestroy(begin), "event");
    check(cudaEventDestroy(end), "event");
    check(cudaStreamDestroy(stream), "stream");
    return 0;
}
