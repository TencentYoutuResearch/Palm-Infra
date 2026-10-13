#include "kernels/cuda/elementwise.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(cudaError_t error) {
    if (error != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); std::exit(1); }
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    // The reference launcher must remain the original kernel in this process.
    setenv("MOLLM_CUDA_GROUP_RMS_CACHE", "0", 1);
    constexpr int width = 2560;
    for (int groups : {1, 4, 7}) for (int rows : {1, 3}) {
        size_t count = size_t(width) * groups * rows;
        std::vector<float> input(count), weight(width * groups), baseline(count + 8), actual(count + 8);
        float *x, *w, *reference, *output;
        check(cudaMalloc(&x, count * 4)); check(cudaMalloc(&w, weight.size() * 4));
        check(cudaMalloc(&reference, (count + 8) * 4)); check(cudaMalloc(&output, (count + 8) * 4));
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaGraph_t graph; cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        if (!mollm_cuda::launch_group_rms_norm_cached(x, w, output, width, groups, rows, 1e-6f, stream)) return 1;
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int iteration = 0; iteration < 4; ++iteration) {
            for (size_t i = 0; i < count; ++i)
                input[i] = iteration == 0 ? 0.f : std::sin(float(i) * .173f + iteration) * (1 + i % 17);
            for (size_t i = 0; i < weight.size(); ++i) weight[i] = std::cos(float(i) * .19f + iteration);
            check(cudaMemcpy(x, input.data(), count * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(w, weight.data(), weight.size() * 4, cudaMemcpyHostToDevice));
            check(cudaMemset(reference, 0x7f, (count + 8) * 4));
            check(cudaMemset(output, 0x7f, (count + 8) * 4));
            mollm_cuda::launch_group_rms_norm(x, w, reference, width, groups, rows, 1e-6f);
            check(cudaDeviceSynchronize());
            if (iteration % 2) check(cudaGraphLaunch(executable, stream));
            else if (!mollm_cuda::launch_group_rms_norm_cached(x, w, output, width, groups, rows, 1e-6f, stream)) return 1;
            check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(baseline.data(), reference, (count + 8) * 4, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(actual.data(), output, (count + 8) * 4, cudaMemcpyDeviceToHost));
            if (std::memcmp(baseline.data(), actual.data(), (count + 8) * 4)) return 2;
            for (int group = 0; group < groups * rows; ++group) {
                double sum = 0;
                for (int d = 0; d < width; ++d) { double value = input[size_t(group) * width + d]; sum += value * value; }
                double inverse = 1 / std::sqrt(sum / width + 1e-6);
                for (int d = 0; d < width; ++d) {
                    size_t index = size_t(group) * width + d;
                    double expected = input[index] * inverse * weight[(group % groups) * width + d];
                    if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected) > 2e-6 * (1 + std::abs(expected))) return 3;
                }
            }
        }
        if (mollm_cuda::launch_group_rms_norm_cached(x, w, output, 2559, groups, rows, 1e-6f) ||
            mollm_cuda::launch_group_rms_norm_cached(x, w, output, width, 0, rows, 1e-6f) ||
            mollm_cuda::launch_group_rms_norm_cached(nullptr, w, output, width, groups, rows, 1e-6f)) return 4;
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
        check(cudaFree(x)); check(cudaFree(w)); check(cudaFree(reference)); check(cudaFree(output));
    }
    std::puts("group RMS cached: eager/changed-input graph, guards, bitwise reference and CPU double PASS");
}
