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
static bool tiled_reduce = false;
static bool tiled_four = false;
static bool vector_reduce = false;
static bool run(int hidden, int streams, int rows, bool benchmark) {
    const size_t count = size_t(hidden) * streams * rows, outcount = size_t(hidden) * rows;
    Buffer<float> x(count), logits(count), gates(count), baseline(outcount + 8), fused(outcount + 8);
    std::vector<float> host_x(count), host_logits(count), expected(outcount + 8), actual(outcount + 8);
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t exec;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    auto candidate = [&](cudaStream_t target_stream) {
        if (vector_reduce && hidden % 4 == 0 && streams == 4) {
            if (!mollm_cuda::launch_gr_reduce_sigmoid_vector4(x.data,logits.data,fused.data,
                hidden,streams,rows,target_stream)) std::exit(2);
        } else if (tiled_reduce) {
            if (!mollm_cuda::launch_gr_reduce_sigmoid_tiled(x.data, logits.data, fused.data,
                hidden, streams, rows, target_stream, tiled_four)) std::exit(2);
        } else mollm_cuda::launch_gr_reduce_sigmoid(x.data, logits.data, fused.data,
            hidden, streams, rows, target_stream);
    };
    candidate(stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    auto reference = [&] {
        if (tiled_reduce || vector_reduce) {
            mollm_cuda::launch_gr_reduce_sigmoid(x.data, logits.data, baseline.data, hidden, streams, rows);
            check(cudaGetLastError());
            return;
        }
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
static bool run_inject(int hidden, int streams, int rows, bool timing) {
    const size_t branch_count = size_t(hidden) * rows;
    const size_t count = branch_count * streams, gate_count = size_t(rows) * streams;
    Buffer<float> branch(branch_count), residual(count), gates(gate_count), baseline(count + 8), tiled(count + 8);
    std::vector<float> b(branch_count), r(count), g(gate_count), expected(count + 8), actual(count + 8);
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t exec;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    if (!mollm_cuda::launch_gr_inject_tiled(branch.data, residual.data, gates.data, tiled.data,
            hidden, streams, rows, stream)) return false;
    check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 4; ++replay) {
        for (size_t i = 0; i < b.size(); ++i) b[i] = std::sin(float(i % 10007) * .17f + replay);
        for (size_t i = 0; i < r.size(); ++i) r[i] = std::cos(float(i % 10009) * .19f + replay);
        for (size_t i = 0; i < g.size(); ++i) g[i] = (int((i + replay) % 11) - 5) / 3.f;
        check(cudaMemcpy(branch.data, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(residual.data, r.data(), r.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(gates.data, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemset(baseline.data, 0x7f, (count + 8) * 4));
        check(cudaMemset(tiled.data, 0x7f, (count + 8) * 4));
        mollm_cuda::launch_gr_inject(branch.data, residual.data, gates.data, baseline.data, hidden, streams, rows);
        check(cudaDeviceSynchronize());
        check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(expected.data(), baseline.data, (count + 8) * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(actual.data(), tiled.data, (count + 8) * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(expected.data(), actual.data(), (count + 8) * 4)) {
            std::fprintf(stderr, "GR inject mismatch hidden=%d streams=%d rows=%d replay=%d\n", hidden, streams, rows, replay);
            return false;
        }
    }
    if (mollm_cuda::launch_gr_inject_tiled(nullptr, residual.data, gates.data, tiled.data, hidden, streams, rows) ||
        mollm_cuda::launch_gr_inject_tiled(branch.data, residual.data, gates.data, tiled.data, 0, streams, rows) ||
        mollm_cuda::launch_gr_inject_tiled(branch.data, residual.data, gates.data, tiled.data, hidden, 65536, rows) ||
        mollm_cuda::launch_gr_inject_tiled(branch.data, residual.data, gates.data, tiled.data, hidden, streams, 65536) ||
        mollm_cuda::launch_gr_inject_tiled(branch.data, residual.data, gates.data, tiled.data, 2147483647, 65535, 65535)) return false;
    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
    std::printf("inject_bit_exact=true hidden=%d streams=%d rows=%d changed_input_graph=true\n", hidden, streams, rows);
    if (timing) {
        cudaGraphExec_t timed[2];
        for (int version = 0; version < 2; ++version) {
            check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            for (int repeat = 0; repeat < 32; ++repeat) {
                if (version) {
                    if (!mollm_cuda::launch_gr_inject_tiled(branch.data, residual.data, gates.data, tiled.data,
                            hidden, streams, rows)) std::exit(2);
                } else mollm_cuda::launch_gr_inject(branch.data, residual.data, gates.data, baseline.data, hidden, streams, rows);
            }
            cudaGraph_t timed_graph;
            check(cudaStreamEndCapture(cudaStreamPerThread, &timed_graph));
            check(cudaGraphInstantiate(&timed[version], timed_graph, nullptr, nullptr, 0));
            check(cudaGraphDestroy(timed_graph));
        }
        cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        for (int iteration = 0; iteration < 7; ++iteration)
            for (int order = 0; order < 2; ++order) {
                const bool use_tiled = (iteration + order) % 2;
                check(cudaGraphLaunch(timed[use_tiled], cudaStreamPerThread));
                check(cudaDeviceSynchronize()); check(cudaEventRecord(start));
                check(cudaGraphLaunch(timed[use_tiled], cudaStreamPerThread));
                check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
                float ms; check(cudaEventElapsedTime(&ms, start, end));
                std::printf("inject_graph_timing rows=%d iteration=%d tiled=%d us=%.3f\n", rows, iteration, use_tiled, ms * 1000 / 32);
            }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
        check(cudaGraphExecDestroy(timed[0])); check(cudaGraphExecDestroy(timed[1]));
    }
    return true;
}

int main(int argc, char** argv) {
    vector_reduce = argc == 2 && !std::strcmp(argv[1], "--reduce-vector4");
    tiled_four = argc == 2 && !std::strcmp(argv[1], "--reduce-tiled-four");
    tiled_reduce = tiled_four || (argc == 2 && !std::strcmp(argv[1], "--reduce-tiled"));
    const bool inject = argc == 2 && (!std::strcmp(argv[1], "--inject") || !std::strcmp(argv[1], "--inject-check"));
    const bool timing = argc == 1 || tiled_reduce || vector_reduce || (argc == 2 && !std::strcmp(argv[1], "--inject"));
    if (argc > 2 || (argc == 2 && !inject && !tiled_reduce && !vector_reduce && std::strcmp(argv[1], "--check-only"))) return 64;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    if (tiled_reduce || vector_reduce) {
        Buffer<float> stub(16);
        auto* p = stub.data;
        if (mollm_cuda::launch_gr_reduce_sigmoid_tiled(nullptr,p,p,4,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_tiled(p,nullptr,p,4,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_tiled(p,p,nullptr,4,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_tiled(p,p,p,0,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_tiled(p,p,p,4,0,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_tiled(p,p,p,4,4,0) ||
            mollm_cuda::launch_gr_reduce_sigmoid_tiled(p,p,p,4,4,65536)) return 1;
        if (mollm_cuda::launch_gr_reduce_sigmoid_vector4(nullptr,p,p,4,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_vector4(p,nullptr,p,4,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_vector4(p,p,nullptr,4,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_vector4(p,p,p,0,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_vector4(p,p,p,3,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_vector4(p,p,p,4,3,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_vector4(p,p,p,4,4,0) ||
            mollm_cuda::launch_gr_reduce_sigmoid_vector4(p,p,p,4,4,65536) ||
            mollm_cuda::launch_gr_reduce_sigmoid_vector4(p+1,p,p,4,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_vector4(p,p+1,p,4,4,1) ||
            mollm_cuda::launch_gr_reduce_sigmoid_vector4(p,p,p+1,4,4,1)) return 1;
        check(cudaGetLastError());
        std::puts("invalid_geometry_cases=18 rejected=true");
    }
    for (int hidden : vector_reduce ? std::vector<int>{4,28,260,2560} : std::vector<int>{1,19,2560})
      for (int streams : {1, 4, 7})
        for (int rows : {1, 3, 857})
            if (inject) {
                if (!run_inject(hidden, streams, rows, timing && hidden == 2560 && streams == 4 && (rows == 1 || rows == 857))) return 1;
            } else if (!run(hidden, streams, rows, timing && hidden == 2560 && streams == 4 && rows == 857)) return 1;
    return 0;
}
