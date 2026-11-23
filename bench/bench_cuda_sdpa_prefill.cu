#include "kernels/cuda/attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool paired_mode = false;
static bool paired_qk = false;
static bool paired_reload = false;

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}
template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(&data, count * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
};

static bool run(int queries, int keys, int dim, bool masked, bool causal,
                bool benchmark = false, bool interleaved = false) {
    const int heads = benchmark ? 24 : 4, kv_heads = 2;
    const int capacity = keys + 3;
    const size_t qs = interleaved ? dim * heads + 2 : dim + 2;
    const size_t qh = interleaved ? dim : qs * queries + 4;
    const size_t os = interleaved ? dim * heads + 3 : dim + 3;
    const size_t oh = interleaved ? dim : os * queries + 5;
    const size_t qsize = std::max(qh * heads, qh * (heads - 1) + qs * queries + 4);
    const size_t outsize = std::max(oh * heads, oh * (heads - 1) + os * queries + 5);
    const size_t kvsize = size_t(capacity) * dim * kv_heads;
    std::vector<float> q(qsize), mask(size_t(queries) * (keys + 2));
    std::vector<__half> k(kvsize), v(kvsize);
    for (size_t i = 0; i < qsize; ++i) q[i] = (int(i % 31) - 15) / 17.f;
    for (size_t i = 0; i < kvsize; ++i) {
        k[i] = __float2half_rn((int(i % 29) - 14) / 19.f);
        v[i] = __float2half_rn((int(i % 37) - 18) / 23.f);
    }
    for (int row = 0; row < queries; ++row)
        for (int col = 0; col < keys; ++col)
            mask[size_t(row) * (keys + 2) + col] =
                col > keys - queries + row ? -INFINITY : (col % 7 - 3) / 10.f;
    Buffer<float> dq(qsize), dm(mask.size()), scores(size_t(heads) * queries * keys);
    Buffer<float> output(outsize);
    Buffer<__half> dk(kvsize), dv(kvsize);
    check(cudaMemcpy(dq.data, q.data(), qsize * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dm.data, mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dk.data, k.data(), kvsize * sizeof(__half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dv.data, v.data(), kvsize * sizeof(__half), cudaMemcpyHostToDevice));
    auto launch = [&](bool skip_causal_qk) {
        if (!mollm_cuda::try_launch_sdpa_prefill(dq.data, dk.data, dv.data,
                scores.data, output.data, masked ? dm.data : nullptr,
                heads, kv_heads, queries, keys, keys - queries, dim, dim,
                capacity, true, true, causal, 1.f / std::sqrt(float(dim)),
                1, qs, qh, 1, keys + 2, 1, os, oh, paired_mode ? false : skip_causal_qk,
                paired_mode && skip_causal_qk, paired_qk && skip_causal_qk,
                paired_reload && skip_causal_qk)) std::exit(2);
        check(cudaGetLastError());
    };
    std::vector<float> reference(outsize), actual(outsize);
    check(cudaMemset(output.data, 0x3f, outsize * sizeof(float)));
    launch(false);
    check(cudaMemcpy(reference.data(), output.data, outsize * sizeof(float), cudaMemcpyDeviceToHost));
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    // The launcher uses the default stream; capture that stream explicitly.
    check(cudaStreamBeginCapture(nullptr, cudaStreamCaptureModeGlobal));
    launch(true);
    cudaGraph_t graph;
    check(cudaStreamEndCapture(nullptr, &graph));
    cudaGraphExec_t exec;
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; ++replay) {
        if (paired_mode) {
            for (size_t i = 0; i < qsize; ++i) q[i] = (int((i + replay * 7) % 31) - 15) / 17.f;
            for (size_t i = 0; i < kvsize; ++i) {
                k[i] = __float2half_rn((int((i + replay * 3) % 29) - 14) / 19.f);
                v[i] = __float2half_rn((int((i + replay * 5) % 37) - 18) / 23.f);
            }
            for (int row = 0; row < queries; ++row) for (int col = 0; col < keys; ++col)
                mask[size_t(row) * (keys + 2) + col] = col > keys - queries + row
                    ? (replay == 1 ? -1e38f : -INFINITY) : (col % 7 - 3) / 10.f;
            check(cudaMemcpy(dq.data,q.data(),qsize*sizeof(float),cudaMemcpyHostToDevice));
            check(cudaMemcpy(dk.data,k.data(),kvsize*sizeof(__half),cudaMemcpyHostToDevice));
            check(cudaMemcpy(dv.data,v.data(),kvsize*sizeof(__half),cudaMemcpyHostToDevice));
            check(cudaMemcpy(dm.data,mask.data(),mask.size()*sizeof(float),cudaMemcpyHostToDevice));
            check(cudaMemset(output.data,0x3f,outsize*sizeof(float))); launch(false);
            check(cudaMemcpy(reference.data(),output.data,outsize*sizeof(float),cudaMemcpyDeviceToHost));
        }
        check(cudaMemset(output.data, 0x3f, outsize * sizeof(float)));
        check(cudaGraphLaunch(exec, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(), output.data, outsize * sizeof(float), cudaMemcpyDeviceToHost));
        const bool matches = !std::memcmp(reference.data(),actual.data(),outsize*sizeof(float));
        if (!matches) {
            std::fprintf(stderr, "not bit-exact q=%d k=%d dim=%d mask=%d causal=%d\n",
                         queries, keys, dim, masked, causal);
            return false;
        }
    }
    check(cudaGraphExecDestroy(exec));
    check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));
    std::printf("check_pass=true paired=%d interleaved=%d q=%d k=%d dim=%d mask=%d causal=%d\n",
                paired_mode, interleaved, queries, keys, dim, masked, causal);
    if (benchmark) {
        cudaEvent_t start, end;
        check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        for (int iteration = 0; iteration < 7; ++iteration)
            for (int order = 0; order < 2; ++order) {
                const bool skip_causal_qk = (iteration + order) % 2;
                launch(skip_causal_qk); check(cudaDeviceSynchronize());
                check(cudaEventRecord(start));
                for (int call = 0; call < 12; ++call) launch(skip_causal_qk);
                check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
                float ms; check(cudaEventElapsedTime(&ms, start, end));
                std::printf("timing paired=%d interleaved=%d mask=%d iteration=%d skip_causal_qk=%d us=%.3f\n", paired_mode, interleaved, masked, iteration, skip_causal_qk, ms * 1000 / 12);
            }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    }
    return true;
}
int main(int argc, char** argv) {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    if (argc == 2 && (!std::strcmp(argv[1],"--paired") || !std::strcmp(argv[1],"--paired-qk") || !std::strcmp(argv[1],"--paired-qk-reload"))) {
        paired_mode = true; paired_qk = std::strcmp(argv[1],"--paired") != 0;
        paired_reload = !std::strcmp(argv[1],"--paired-qk-reload");
    }
    else if (argc != 1) return 2;
    if (paired_mode) {
        // Every case must reject before launching: these buffers deliberately
        // cannot hold even the nominal valid shape.
        Buffer<float> stub(8);
        auto rejected = [&](int h, int kh, int q, int k, int past, int capacity,
                            bool pair = true, bool qk = false, bool reload = false) {
            return !mollm_cuda::try_launch_sdpa_prefill(stub.data,stub.data,stub.data,
                stub.data,stub.data,nullptr,h,kh,q,k,past,256,256,capacity,
                true,true,true,1.f,1,256,768,1,3,1,256,768,false,pair,qk,reload);
        };
        if (!rejected(0,2,3,3,0,3) || !rejected(65536,2,3,3,0,3) ||
            !rejected(4,0,3,3,0,3) || !rejected(3,2,3,3,0,3) ||
            !rejected(4,2,INT32_MAX,3,0,3) || !rejected(4,2,3,0,0,3) ||
            !rejected(4,2,3,INT32_MAX,0,INT32_MAX) || !rejected(4,2,3,3,0,2) ||
            !rejected(4,2,3,3,-1,3) || !rejected(4,2,3,3,INT32_MAX,3) ||
            !rejected(4,2,3,3,0,3,false,true,false) ||
            !rejected(4,2,3,3,0,3,true,false,true)) return 2;
        check(cudaGetLastError());
        std::puts("paired_invalid_arguments_rejected=12");
    }
    for (int dim : {128, 256})
        for (int keys : {3, 7, 129, 857, 1024, 1025})
            for (bool mask : {false, true})
                for (bool causal : {false, true})
                    if (!run(std::min(keys, 17), keys, dim, mask, causal)) return 1;
    bool ok = run(857, 857, 256, false, true, true);
    if (paired_mode) {
        ok &= run(857,857,256,true,true,true);
        for (int dim : {128,256}) for (int keys : {129,857,1025}) for (bool mask : {false,true})
            ok &= run(17,keys,dim,mask,true,false,true);
        ok &= run(857,857,256,true,true,true,true);
    }
    return ok ? 0 : 1;
}
