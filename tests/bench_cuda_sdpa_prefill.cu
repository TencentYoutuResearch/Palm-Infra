#include "kernels/cuda/attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool gemm_mode = false;
static bool batched_gemm = false;
static bool ordered_pv = false;
static bool check_only = false;
static bool focused_sanitize = false;
static bool paired_mode = false;
static bool paired_qk = false;
static bool paired_reload = false;
static cublasHandle_t gemm_handle = nullptr;

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
    std::vector<uint8_t> logical_output(outsize,0);
    for (int h = 0; h < heads; ++h) for (int q = 0; q < queries; ++q)
        std::fill(logical_output.begin()+h*oh+q*os,logical_output.begin()+h*oh+q*os+dim,1);
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
    Buffer<float> key32(gemm_mode ? size_t(kv_heads) * keys * dim : 1);
    Buffer<float> value32(gemm_mode ? size_t(kv_heads) * keys * dim : 1);
    Buffer<float*> pointers(batched_gemm ? size_t(heads) * 5 : 1);
    check(cudaMemcpy(dq.data, q.data(), qsize * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dm.data, mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dk.data, k.data(), kvsize * sizeof(__half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dv.data, v.data(), kvsize * sizeof(__half), cudaMemcpyHostToDevice));
    auto launch = [&](bool skip_causal_qk) {
        if (gemm_mode && skip_causal_qk) {
            mollm_cuda::SdpaGemmPrefillParams p{heads,kv_heads,queries,keys,keys-queries,dim,capacity,
                causal,1.f/std::sqrt(float(dim)),qs,qh,1,size_t(keys+2),os,oh};
            auto status = mollm_cuda::launch_sdpa_prefill_fp32_gemm(gemm_handle,dq.data,dk.data,dv.data,
                scores.data,output.data,masked?dm.data:nullptr,key32.data,value32.data,p,cudaStreamPerThread,
                batched_gemm ? pointers.data : nullptr,ordered_pv);
            if (status != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr,"cuBLAS status=%d\n",int(status)); std::exit(2); }
            return;
        }
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
    if (gemm_mode) { launch(true); check(cudaDeviceSynchronize()); }
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
        if (gemm_mode || paired_mode) {
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
        bool matches = true;
        if (gemm_mode) {
            float max_error = 0.f;
            for (size_t i = 0; i < outsize; ++i) {
                const bool logical = logical_output[i] != 0;
                if (!logical) matches &= !std::memcmp(&reference[i],&actual[i],sizeof(float));
                else {
                    const float error = std::fabs(reference[i] - actual[i]);
                    max_error = std::max(max_error,error);
                    matches &= std::isfinite(actual[i]) && error <= 2e-5f * (1.f + std::fabs(reference[i]));
                }
            }
            std::printf("gemm_error q=%d k=%d dim=%d mask=%d causal=%d replay=%d max_abs=%g\n",queries,keys,dim,masked,causal,replay,max_error);
        } else matches = !std::memcmp(reference.data(),actual.data(),outsize*sizeof(float));
        if (!matches) {
            std::fprintf(stderr, "not bit-exact q=%d k=%d dim=%d mask=%d causal=%d\n",
                         queries, keys, dim, masked, causal);
            return false;
        }
    }
    check(cudaGraphExecDestroy(exec));
    check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));
    std::printf("check_pass=true gemm=%d batched=%d paired=%d interleaved=%d q=%d k=%d dim=%d mask=%d causal=%d\n",
                gemm_mode, batched_gemm, paired_mode, interleaved, queries, keys, dim, masked, causal);
    if (benchmark&&!check_only) {
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
                std::printf("timing gemm=%d batched=%d paired=%d interleaved=%d mask=%d iteration=%d skip_causal_qk=%d us=%.3f\n", gemm_mode, batched_gemm, paired_mode, interleaved, masked, iteration, skip_causal_qk, ms * 1000 / 12);
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
    else if (argc == 2 && (!std::strcmp(argv[1],"--gemm") || !std::strcmp(argv[1],"--gemm-batched") || !std::strcmp(argv[1],"--gemm-ordered-pv") || !std::strcmp(argv[1],"--gemm-ordered-pv-check") || !std::strcmp(argv[1],"--gemm-ordered-pv-sanitize"))) {
        gemm_mode = true;
        focused_sanitize=!std::strcmp(argv[1],"--gemm-ordered-pv-sanitize");
        check_only=focused_sanitize||!std::strcmp(argv[1],"--gemm-ordered-pv-check");
        ordered_pv=check_only||!std::strcmp(argv[1],"--gemm-ordered-pv");
        batched_gemm = ordered_pv||!std::strcmp(argv[1],"--gemm-batched");
        if(ordered_pv)std::puts("diagnostic_stage=GEMM-QK-softmax-with-ordered-FP32-PV production-selector=none");
        if (cublasCreate(&gemm_handle) != CUBLAS_STATUS_SUCCESS ||
            cublasSetStream(gemm_handle,cudaStreamPerThread) != CUBLAS_STATUS_SUCCESS) return 2;
        Buffer<float> stub(8);
        const mollm_cuda::SdpaGemmPrefillParams valid{4,2,3,3,0,128,3,true,1.f,128,384,1,3,128,384};
        auto rejected = [&](const mollm_cuda::SdpaGemmPrefillParams& p, cudaStream_t stream = cudaStreamPerThread) {
            return mollm_cuda::launch_sdpa_prefill_fp32_gemm(gemm_handle,stub.data,stub.data,stub.data,
                stub.data,stub.data,stub.data,stub.data,stub.data,p,stream) == CUBLAS_STATUS_INVALID_VALUE;
        };
        auto bad = valid; bad.heads = 0; if (!rejected(bad)) return 2;
        bad = valid; bad.kv_heads = 0; if (!rejected(bad)) return 2;
        bad = valid; bad.heads = 3; if (!rejected(bad)) return 2;
        bad = valid; bad.queries = 1; if (!rejected(bad)) return 2;
        bad = valid; bad.capacity = 2; if (!rejected(bad)) return 2;
        bad = valid; bad.past = 1; if (!rejected(bad)) return 2;
        bad = valid; bad.dimension = 64; if (!rejected(bad)) return 2;
        bad = valid; bad.query_position_stride = 127; if (!rejected(bad)) return 2;
        bad = valid; bad.output_position_stride = 127; if (!rejected(bad)) return 2;
        bad = valid; bad.query_head_stride = SIZE_MAX; if (!rejected(bad)) return 2;
        bad = valid; bad.mask_column_stride = SIZE_MAX; if (!rejected(bad)) return 2;
        bad = valid; bad.keys = bad.capacity = INT32_MAX; if (!rejected(bad)) return 2;
        if (!rejected(valid,cudaStreamLegacy)) return 2;
        if (cublasSetPointerMode(gemm_handle,CUBLAS_POINTER_MODE_DEVICE) != CUBLAS_STATUS_SUCCESS ||
            !rejected(valid) || cublasSetPointerMode(gemm_handle,CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS) return 2;
        std::puts("gemm_invalid_arguments_rejected=14");
    } else if (argc != 1) return 2;
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
    if(focused_sanitize) {
        std::puts("sanitizer_scope=finite_changed_inputs3_direct_capture_padding_guards_D128_D256_padded_interleaved_Q3K17_Q17K16384_no_timing_not_full_suite");
        bool ok=true;
        for(int dim:{128,256})for(bool interleaved:{false,true})
            ok &= run(3,17,dim,true,true,false,interleaved);
        for(bool interleaved:{false,true})ok &= run(17,16384,256,true,true,false,interleaved);
        cublasDestroy(gemm_handle);
        return ok?0:1;
    }
    for (int dim : {128, 256})
        for (int keys : {3, 7, 129, 857, 1024, 1025})
            for (bool mask : {false, true})
                for (bool causal : {false, true})
                    if (!run(std::min(keys, 17), keys, dim, mask, causal)) return 1;
    bool ok = run(857, 857, 256, false, true, true);
    if (gemm_mode || paired_mode) {
        ok &= run(857,857,256,true,true,true);
        if (batched_gemm || paired_mode) {
            for (int dim : {128,256}) for (int keys : {129,857,1025}) for (bool mask : {false,true})
                ok &= run(17,keys,dim,mask,true,false,true);
            ok &= run(857,857,256,true,true,true,true);
        }
        if(ordered_pv)for(bool interleaved:{false,true})
            ok &= run(17,16384,256,true,true,false,interleaved);
        if (gemm_mode) cublasDestroy(gemm_handle);
    }
    return ok ? 0 : 1;
}
