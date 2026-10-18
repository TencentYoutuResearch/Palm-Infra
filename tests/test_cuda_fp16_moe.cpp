#include "backends/cuda/backend.h"
#include "backends/cpu/backend.h"
#include "graph/graph.h"
#include "core/fp16.h"
#include "kernels/cuda/moe.h"
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

static Tensor allocate(CudaBackend& backend, int n, int rows = 1) {
    Tensor t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, n, rows);
    backend.alloc_persistent(t, t.nbytes(), PersistentHostAccess::NONE);
    return t;
}

static bool projections(int n, int k) {
    CudaBackend backend;
    std::vector<mollm::fp16_t> weights(2 * n * k);
    std::vector<float> inputs(2 * k), expected(2 * (n + 8), -123), actual = expected;
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = (static_cast<int>(i % 17) - 8) / 32.0f;
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = (static_cast<int>(i % 13) - 6) / 16.0f;
    for (int job = 0; job < 2; ++job)
        for (int row = 0; row < n; ++row) {
            double sum = 0;
            for (int col = 0; col < k; ++col)
                sum += static_cast<float>(weights[(job * n + row) * k + col]) * inputs[job * k + col];
            expected[job * (n + 8) + row] = static_cast<float>(sum);
        }
    Tensor weight = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, 2 * n, k, 1, 1, weights.data());
    backend.wrap_weight(weight);
    Tensor x = allocate(backend, 2 * k), y = allocate(backend, actual.size());
    if (!backend.copy_from_host(inputs.data(), x, x.nbytes()) ||
        !backend.copy_from_host(actual.data(), y, y.nbytes())) return false;
    std::array<mollm_cuda::Nvfp4Projection, 2> jobs{};
    for (int j = 0; j < 2; ++j) {
        jobs[j].weight = static_cast<const uint8_t*>(weight.device.buffer) + static_cast<size_t>(j) * n * k * 2;
        jobs[j].input = static_cast<const float*>(x.device.buffer) + j * k;
        jobs[j].output = static_cast<float*>(y.device.buffer) + j * (n + 8);
    }
    Tensor descriptors = allocate(backend, sizeof(jobs) / sizeof(float));
    if (!backend.copy_from_host(jobs.data(), descriptors, sizeof(jobs))) return false;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    bool ok = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess;
    if (ok) ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    if (ok) ok = mollm_cuda::launch_fp16_expert_projections(
        static_cast<const mollm_cuda::Nvfp4Projection*>(descriptors.device.buffer), 2, n, k, stream) == cudaSuccess;
    if (ok) ok = cudaStreamEndCapture(stream, &graph) == cudaSuccess;
    if (ok) ok = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) == cudaSuccess;
    for (int replay = 0; replay < 2 && ok; ++replay) {
        ok = cudaGraphLaunch(executable, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess;
        if (ok) ok = backend.copy_to_host(y, actual.data(), y.nbytes());
        if (ok) for (size_t i = 0; i < actual.size(); ++i)
            if (actual[i] != expected[i]) { ok = false; break; }
    }
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    if (stream) cudaStreamDestroy(stream);
    if (!ok) std::fprintf(stderr, "FP16 projection failed N=%d K=%d\n", n, k);
    return ok;
}

static bool moe(int rows, int score, bool normalize) {
    CudaBackend backend;
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    CPUBackend cpu;
    constexpr int h = 32, e = 7, top = 3, inter = 16, shared = 16;
    std::array<std::vector<mollm::fp16_t>, 7> storage;
    std::array<Tensor, 8> host, device;
    std::vector<float> x(rows * h), expected(rows * h), actual(rows * h);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>(i % 11) - 5) / 16.0f;
    host[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, h, rows, 1, 1, x.data());
    device[0] = allocate(backend, h, rows);
    if (!backend.copy_from_host(x.data(), device[0], device[0].nbytes())) return false;
    const int ns[] = {e, e * 2 * inter, e * h, shared, shared, h, 1};
    const int ks[] = {h, h, inter, h, h, shared, h};
    for (int w = 0; w < 7; ++w) {
        storage[w].resize(ns[w] * ks[w]);
        for (size_t i = 0; i < storage[w].size(); ++i)
            storage[w][i] = (static_cast<int>((i * 7 + w * 3) % 19) - 9) / 64.0f;
        host[w + 1] = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, ns[w], ks[w], 1, 1, storage[w].data());
        device[w + 1] = host[w + 1];
        backend.wrap_weight(device[w + 1]);
    }
    Tensor cy = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, h, rows, 1, 1, expected.data());
    Tensor dy = allocate(backend, h, rows);
    GraphNode node;
    node.op_type = OpType::MOE;
    node.params.i32 = {h, e, top, inter, shared, score, normalize ? 1 : 0, 1, 1, 1, 1, -1, -1, -1};
    node.params.f32 = {0.75f, 0};
    std::vector<const Tensor*> hp, dp;
    for (int i = 0; i < 8; ++i) { hp.push_back(&host[i]); dp.push_back(&device[i]); }
    cpu.dispatch(node, hp, &cy, nullptr);
    for (int repeat = 0; repeat < 2; ++repeat) {
        backend.dispatch(node, dp, &dy, nullptr);
        if (cpu.dispatch_failed() || backend.dispatch_failed() ||
            !backend.copy_to_host(dy, actual.data(), dy.nbytes())) return false;
        for (size_t i = 0; i < actual.size(); ++i)
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 2e-4f) {
                std::fprintf(stderr, "FP16 MoE mismatch row_count=%d score=%d i=%zu actual=%g expected=%g\n",
                    rows, score, i, actual[i], expected[i]);
                return false;
            }
    }
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
        return false;
    backend.dispatch(node, dp, &dy, nullptr);
    if (cudaStreamEndCapture(cudaStreamPerThread, &graph) != cudaSuccess ||
        backend.dispatch_failed() || !graph) return false;
    bool ok = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) == cudaSuccess;
    for (int replay = 0; replay < 2 && ok; ++replay) {
        for (float& value : x) value = -value;
        cpu.dispatch(node, hp, &cy, nullptr);
        ok = backend.copy_from_host(x.data(), device[0], device[0].nbytes()) &&
             cudaGraphLaunch(executable, cudaStreamPerThread) == cudaSuccess &&
             cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess &&
             backend.copy_to_host(dy, actual.data(), dy.nbytes());
        if (ok) for (size_t i = 0; i < actual.size(); ++i)
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 2e-4f) {
                std::fprintf(stderr, "FP16 MoE graph mismatch rows=%d score=%d index=%zu\n", rows, score, i);
                ok = false; break;
            }
    }
    if (executable) cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    if (!ok) return false;
#endif
    return true;
}

int main() {
    { CudaBackend probe; if (!probe.available()) return 77; }
    if (mollm_cuda::launch_fp16_expert_projections(nullptr, 1, 1, 1) != cudaErrorInvalidValue) return 1;
    for (int n : {1, 7, 9, 31, 33})
        for (int k : {1, 7, 31, 32, 33, 640, 2560})
            if (!projections(n, k)) return 1;
    for (int rows : {1, 3, 8})
        for (int score : {0, 1, 2})
            for (bool normalize : {false, true})
                if (!moe(rows, score, normalize)) return 1;
    std::printf("FP16 CUDA experts: 35 projection/graph/guard cases and 18 native MoE CPU comparisons passed\n");
    return 0;
}
