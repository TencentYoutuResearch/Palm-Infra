#include "kernels/cuda/elementwise.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); std::exit(1);
    }
}
template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(&data, n * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
};
static bool run(int hidden, int streams, int rows, bool benchmark) {
    const size_t count = size_t(hidden) * streams * rows, outcount = size_t(hidden) * rows;
    Buffer<float> x(count), logits(count), gates(count), baseline(outcount + 8), fused(outcount + 8);
    std::vector<float> host_x(count), host_logits(count), expected(outcount + 8), actual(outcount + 8);
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t exec;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    auto candidate = [&](cudaStream_t target_stream) {
        mollm_cuda::launch_gr_reduce_sigmoid(x.data, logits.data, fused.data,
            hidden, streams, rows, target_stream);
    };
    candidate(stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    auto reference = [&] {
        mollm_cuda::launch_unary(logits.data, gates.data, count, 3);
        mollm_cuda::launch_gr_reduce(x.data, gates.data, baseline.data, hidden, streams, rows);
        check(cudaGetLastError());
    };
    for (int replay = 0; replay < 4; ++replay) {
        for (size_t i = 0; i < count; ++i) {
            host_x[i] = std::sin(float(i % 10007) * .17f + replay) * (1 + i % 17);
            host_logits[i] = std::cos(float(i % 10009) * .19f + replay) * (replay ? 12.f : 0.f);
            if (i % 101 == 0) host_logits[i] = INFINITY;
            if (i % 103 == 0) host_logits[i] = -INFINITY;
            if (i % 107 == 0) host_logits[i] = -0.f;
            if (i % 109 == 0) host_logits[i] = 80.f;
            if (i % 113 == 0) host_logits[i] = -80.f;
        }
        check(cudaMemcpy(x.data, host_x.data(), count * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(logits.data, host_logits.data(), count * 4, cudaMemcpyHostToDevice));
        check(cudaMemset(baseline.data, 0x7f, (outcount + 8) * 4));
        check(cudaMemset(fused.data, 0x7f, (outcount + 8) * 4));
        reference(); check(cudaDeviceSynchronize());
        check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(expected.data(), baseline.data, (outcount + 8) * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(actual.data(), fused.data, (outcount + 8) * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(expected.data(), actual.data(), (outcount + 8) * 4)) {
            std::fprintf(stderr, "sigmoid/GR mismatch hidden=%d streams=%d rows=%d replay=%d\n", hidden, streams, rows, replay);
            return false;
        }
        for (size_t i = outcount; i < outcount + 8; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &actual[i], sizeof(bits));
            if (bits != 0x7f7f7f7fu) return false;
        }
    }
    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
    std::printf("bit_exact=true hidden=%d streams=%d rows=%d changed_input_graph=true\n", hidden, streams, rows);
    if (benchmark) {
        cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        for (int iteration = 0; iteration < 7; ++iteration)
            for (int order = 0; order < 2; ++order) {
                const bool use_fused = (iteration + order) % 2;
                auto launch = [&] {
                    if (use_fused) candidate(cudaStreamPerThread);
                    else reference();
                };
                launch(); check(cudaDeviceSynchronize()); check(cudaEventRecord(start));
                for (int repeat = 0; repeat < 32; ++repeat) launch();
                check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
                float ms; check(cudaEventElapsedTime(&ms, start, end));
                std::printf("timing iteration=%d fused=%d us=%.3f\n", iteration, use_fused, ms * 1000 / 32);
            }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    }
    return true;
}
int main(int argc, char** argv) {
    const bool timing = argc == 1;
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--check-only"))) return 64;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    for (int hidden : {1, 19, 2560})
      for (int streams : {1, 4, 7})
        for (int rows : {1, 3, 857})
            if (!run(hidden, streams, rows, timing && hidden == 2560 && streams == 4 && rows == 857)) return 1;
    return 0;
}
