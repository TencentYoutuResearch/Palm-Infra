#include "kernels/cuda/reduction.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <type_traits>

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

// Diagnostic only: preserve the virtual 256-thread accumulation and reduction
// tree, while assigning multiple virtual threads to each physical thread.
template<int Threads, bool Cache = false>
__global__ void norm(const float* x, const float* w, float* y,
                     int width, int groups) {
    constexpr int Slots = 256 / Threads;
    float sums[Slots] = {};
    float cached[10];
    const size_t base = size_t(blockIdx.x) * width;
    const size_t wb = size_t(blockIdx.x % groups) * width;
    if constexpr (Cache) {
        static_assert(Threads == 256);
#pragma unroll
        for (int step = 0; step < 10; ++step) {
            cached[step] = x[base + threadIdx.x + step * 256];
            sums[0] += cached[step] * cached[step];
        }
    } else {
        for (int start = threadIdx.x; start < width; start += 256) {
#pragma unroll
            for (int slot = 0; slot < Slots; ++slot) {
                const int dim = start + slot * Threads;
                if (dim < width) {
                    const float value = x[base + dim];
                    sums[slot] += value * value;
                }
            }
        }
    }
#pragma unroll
    for (int stride = Slots / 2; stride > 0; stride >>= 1) {
#pragma unroll
        for (int slot = 0; slot < stride; ++slot)
            sums[slot] += sums[slot + stride];
    }
    float total;
    if constexpr (Threads == 32) {
        total = mollm_cuda::detail::warp_reduce_sum(sums[0]);
        total = __shfl_sync(0xffffffffu, total, 0);
    } else {
        __shared__ float reduction[Threads];
        total = mollm_cuda::detail::block_reduce_sum<Threads>(sums[0], reduction);
    }
    const float inverse = rsqrtf(total / width + 1e-6f);
    if constexpr (Cache) {
#pragma unroll
        for (int step = 0; step < 10; ++step) {
            const int dim = threadIdx.x + step * 256;
            y[base + dim] = cached[step] * inverse * w[wb + dim];
        }
    } else for (int dim = threadIdx.x; dim < width; dim += Threads)
        y[base + dim] = x[base + dim] * inverse * w[wb + dim];
}

template<int Threads, bool Cache = false>
static float run(const float* x, const float* w, float* y, int width,
                 int groups, int rows, cudaStream_t stream) {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 1024; ++i)
        norm<Threads, Cache><<<rows * groups, Threads, 0, stream>>>(x, w, y, width, groups);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 2; ++i) check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    check(cudaEventRecord(begin, stream));
    for (int i = 0; i < 4; ++i) check(cudaGraphLaunch(executable, stream));
    check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
    float ms;
    check(cudaEventElapsedTime(&ms, begin, end));
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    return ms * 1000 / 4096;
}

int main() {
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int width : {1, 31, 257, 640, 2560, 4097}) {
        for (int rows : {1, 3}) {
            constexpr int groups = 4;
            const size_t count = size_t(width) * groups * rows;
            std::vector<float> x(count), w(width * groups), reference(count + 8), actual(count + 8);
            for (size_t i = 0; i < count; ++i) x[i] = std::sin(float(i) * .731f) * (1 + i % 13);
            for (size_t i = 0; i < w.size(); ++i) w[i] = .1f + std::cos(float(i) * .19f);
            float *dx, *dw, *dy;
            check(cudaMalloc(&dx, count * 4)); check(cudaMalloc(&dw, w.size() * 4));
            check(cudaMalloc(&dy, (count + 8) * 4));
            check(cudaMemcpy(dx, x.data(), count * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
            for (int order = 0; order < 3; ++order) {
                check(cudaMemsetAsync(dy, 0x7f, (count + 8) * 4, stream));
                const float baseline = run<256>(dx, dw, dy, width, groups, rows, stream);
                check(cudaMemcpy(reference.data(), dy, (count + 8) * 4, cudaMemcpyDeviceToHost));
                auto candidate = [&](auto tag) {
                    constexpr int threads = decltype(tag)::value;
                    check(cudaMemsetAsync(dy, 0x7f, (count + 8) * 4, stream));
                    float us = run<threads>(dx, dw, dy, width, groups, rows, stream);
                    check(cudaMemcpy(actual.data(), dy, (count + 8) * 4, cudaMemcpyDeviceToHost));
                    if (std::memcmp(reference.data(), actual.data(), (count + 8) * 4)) {
                        std::fprintf(stderr, "mismatch width=%d rows=%d threads=%d\n", width, rows, threads);
                        std::exit(2);
                    }
                    std::printf("width=%d rows=%d order=%d threads=%d baseline_us=%.4f candidate_us=%.4f exact=1\n",
                                width, rows, order, threads, baseline, us);
                };
                if (order % 2) { candidate(std::integral_constant<int, 128>{}); candidate(std::integral_constant<int, 64>{}); candidate(std::integral_constant<int, 32>{}); }
                else { candidate(std::integral_constant<int, 32>{}); candidate(std::integral_constant<int, 64>{}); candidate(std::integral_constant<int, 128>{}); }
                if (width == 2560) {
                    const float us = run<256, true>(dx, dw, dy, width, groups, rows, stream);
                    check(cudaMemcpy(actual.data(), dy, (count + 8) * 4, cudaMemcpyDeviceToHost));
                    if (std::memcmp(reference.data(), actual.data(), (count + 8) * 4)) std::exit(3);
                    std::printf("width=%d rows=%d order=%d cache=1 baseline_us=%.4f candidate_us=%.4f exact=1\n", width, rows, order, baseline, us);
                }
            }
            check(cudaFree(dx)); check(cudaFree(dw)); check(cudaFree(dy));
        }
    }
    check(cudaStreamDestroy(stream));
}
