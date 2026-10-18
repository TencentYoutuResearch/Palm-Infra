#include "kernels/cuda/moe.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1);
    }
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
};
template<class T> void copy(T* ptr, const std::vector<T>& data) {
    check(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template<class T> std::vector<T> read(const T* ptr, size_t count) {
    std::vector<T> result(count);
    check(cudaMemcpy(result.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost)); return result;
}
void equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size()*sizeof(float))) {
        std::fprintf(stderr, "decode byte mismatch\n"); std::exit(2);
    }
}
}

int main(int argc, char** argv) {
    const bool timing = argc == 2 && !std::strcmp(argv[1], "--timing");
    if (argc > 1 && !timing) return 64;
    const bool scale_view=std::getenv("MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW")&&
        !std::strcmp(std::getenv("MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW"),"1");
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop,0));
    if (prop.major != 12 || prop.minor > 1) return 77;
    constexpr int rows = 1280, inner = 2560, routes = 10, sets = 2;
    constexpr size_t codes = size_t(rows)*inner/2, scales = size_t(rows)*inner/16;
    constexpr size_t stride = codes + scales + rows*sizeof(float);
    constexpr size_t outputs = size_t(routes)*rows;
    Buffer<uint8_t> weights(sets*routes*stride), view(sets*routes*codes+16);
    Buffer<uint8_t> scale_major(sets*routes*scales+16);
    Buffer<float> input(size_t(sets)*routes*inner), baseline(outputs+8), candidate(outputs+8);
    Buffer<float> partial_base(outputs*8), partial_code(outputs*8);
    Buffer<uint8_t> prepared(routes*2*inner/2), prepared_scales(routes*2*inner/16);
    Buffer<mollm_cuda::Nvfp4Projection> base_jobs(sets*routes), code_jobs(sets*routes);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto launch = [&](bool code, int mode, int set) {
        if (mode >= 2) {
            const int parts = mode == 3 ? 2 : mode == 4 ? 8 : 1;
            check(mollm_cuda::launch_nvfp4_mma_projections(
                (code?code_jobs.p:base_jobs.p)+set*routes,routes,rows,inner,640,stream,
                code?partial_code.p:partial_base.p,parts,true,
                mode==5?prepared.p:nullptr,mode==5?prepared_scales.p:nullptr,code||scale_view));
        } else if (code||scale_view) check(mollm_cuda::launch_nvfp4_gate_code_projections(
            (code?code_jobs.p:base_jobs.p)+set*routes, routes, rows, inner, mode==1, stream));
        else if (mode==1) check(mollm_cuda::launch_nvfp4_pipelined_projections(
            base_jobs.p+set*routes, routes, rows, inner, true, stream));
        else {
            mollm_cuda::launch_nvfp4_projections(base_jobs.p+set*routes, routes, rows, inner, stream);
            check(cudaGetLastError());
        }
    };
    check(cudaMemset(candidate.p, 0x5a, (outputs+8)*sizeof(float)));
    check(cudaDeviceSynchronize());
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    int rejects = 0;
    for (int id = 0; id < 7; ++id) {
        const auto* jobs = code_jobs.p;
        int n = routes, r = rows, k = inner;
        switch (id) {
            case 0: jobs = nullptr; break;
            case 1: n = 0; break;
            case 2: n = -1; break;
            case 3: n = 65536; break;
            case 4: r = 640; break;
            case 5: k = 640; break;
            case 6: jobs = reinterpret_cast<const mollm_cuda::Nvfp4Projection*>(
                reinterpret_cast<const uint8_t*>(jobs)+1); break;
        }
        if (mollm_cuda::launch_nvfp4_gate_code_projections(jobs,n,r,k,true,stream)
            != cudaErrorInvalidValue) return 3;
        ++rejects;
    }
    for (int id = 0; id < 4; ++id) {
        const auto* jobs = code_jobs.p;
        int r = rows, k = inner;
        bool canonical = true;
        switch (id) {
            case 0: jobs = nullptr; break;
            case 1: r = 1024; break;
            case 2: k = 1280; break;
            case 3: canonical = false; break;
        }
        if (mollm_cuda::launch_nvfp4_mma_projections(jobs,routes,r,k,640,stream,
            nullptr,1,canonical,nullptr,nullptr,true) != cudaErrorInvalidValue) return 3;
        ++rejects;
    }
    cudaGraph_t empty;
    check(cudaStreamEndCapture(stream, &empty));
    size_t nodes = 0;
    check(cudaGraphGetNodes(empty, nullptr, &nodes));
    if (nodes) return 4;
    check(cudaGraphDestroy(empty));
    const auto sentinel = read(reinterpret_cast<const uint8_t*>(candidate.p), (outputs+8)*sizeof(float));
    for (uint8_t b : sentinel) if (b != 0x5a) return 5;
    int changed = 0;
    for (bool pair : {false, true}) {
        std::vector<uint8_t> payload(sets*routes*stride);
        for (int expert = 0; expert < sets*routes; ++expert) {
            auto* p = payload.data()+size_t(expert)*stride;
            for (int row = 0; row < rows; ++row) {
                const float global = (1+(row+expert)%7)/64.f;
                std::memcpy(p+codes+scales+row*sizeof(float), &global, sizeof(float));
                for (int group = 0; group < inner/16; ++group) {
                    const uint8_t sf[] = {0,1,7,8,48,56,64,126};
                    p[codes+size_t(row)*(inner/16)+group] = sf[(row+group+expert)%8];
                    for (int byte = 0; byte < 8; ++byte) {
                        const size_t offset = pair ? size_t(row/4)*(inner/16)*32+group*32+(row%4)*8+byte
                            : size_t(row)*(inner/2)+group*8+byte;
                        p[offset] = (row*29+group*17+byte*13+expert*37)&255;
                    }
                }
            }
        }
        copy(weights.p, payload);
        check(cudaMemset(view.p, 0x5a, sets*routes*codes+16));
        for (int expert = 0; expert < sets*routes; ++expert) {
            mollm_cuda::Nvfp4Projection source{};
            source.weight = weights.p+size_t(expert)*stride; source.pair_layout = pair;
            check(mollm_cuda::launch_repack_nvfp4_gate_codes(source, view.p+size_t(expert)*codes,
                rows,inner,stream));
            if(scale_view) {
                source.block_scales=weights.p+size_t(expert)*stride+codes;
                check(mollm_cuda::launch_repack_nvfp4_gate_scales(source,
                    scale_major.p+size_t(expert)*scales,rows,inner,stream));
            }
        }
        check(cudaStreamSynchronize(stream));
        const auto original_view = read(view.p, sets*routes*codes+16);
        for (int mode = 0; mode < 6; ++mode) {
            std::vector<mollm_cuda::Nvfp4Projection> a(sets*routes), b(sets*routes);
            const auto update = [&](int variant) {
                std::vector<float> values(size_t(sets)*routes*inner);
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = (int((i*17+variant*31)%257)-128)/128.f;
                copy(input.p, values);
                for (int set = 0; set < sets; ++set) for (int route = 0; route < routes; ++route) {
                    const int expert = set*routes+(route*3+variant)%routes;
                    const auto* ptr = weights.p+size_t(expert)*stride;
                    a[set*routes+route] = {ptr,ptr+codes,reinterpret_cast<const float*>(ptr+codes+scales),
                        input.p+size_t(set*routes+route)*inner,baseline.p+size_t(route)*rows,pair};
                    a[set*routes+route].input_global = (1+route%3)/16.f;
                    a[set*routes+route].input_global_second = (1+route%5)/16.f;
                    a[set*routes+route].output_scale = (1+route%7)/8.f;
                    // Isolate the scale-layout change: both arms use the
                    // already-qualified K-major code payload in this mode.
                    if(scale_view)a[set*routes+route].weight=view.p+size_t(expert)*codes;
                    b[set*routes+route] = a[set*routes+route];
                    b[set*routes+route].weight = view.p+size_t(expert)*codes;
                    if(scale_view) {
                        b[set*routes+route].block_scales=scale_major.p+size_t(expert)*scales;
                        b[set*routes+route].gate_scales_kmajor=true;
                    }
                    b[set*routes+route].output = candidate.p+size_t(route)*rows;
                }
                copy(base_jobs.p,a); copy(code_jobs.p,b);
            };
            update(0);
            cudaGraph_t graphs[sets]; cudaGraphExec_t execs[sets];
            for (int set = 0; set < sets; ++set) {
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                launch(false,mode,set); launch(true,mode,set);
                check(cudaStreamEndCapture(stream,&graphs[set]));
                check(cudaGraphInstantiate(&execs[set],graphs[set],nullptr,nullptr,0));
            }
            for (int variant = 0; variant < 8; ++variant) {
                update(variant);
                for (int set = 0; set < sets; ++set) {
                    check(cudaMemsetAsync(baseline.p,0x5a,(outputs+8)*sizeof(float),stream));
                    check(cudaMemsetAsync(candidate.p,0x5a,(outputs+8)*sizeof(float),stream));
                    check(cudaGraphLaunch(execs[set],stream)); check(cudaStreamSynchronize(stream));
                    const auto ref = read(baseline.p,outputs+8), result = read(candidate.p,outputs+8);
                    equal(ref,result);
                    const auto* tail = reinterpret_cast<const uint8_t*>(result.data()+outputs);
                    for (int i = 0; i < 8*sizeof(float); ++i) if (tail[i] != 0x5a) return 6;
                    ++changed;
                }
            }
            for (int set = 0; set < sets; ++set) {
                check(cudaGraphExecDestroy(execs[set])); check(cudaGraphDestroy(graphs[set]));
            }
            if (timing && mode==1) {
                cudaGraph_t graph[2]; cudaGraphExec_t exec[2];
                for (int code = 0; code < 2; ++code) {
                    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                    for (int repeat = 0; repeat < 16; ++repeat)
                        for (int set = 0; set < sets; ++set) launch(code,mode,set);
                    check(cudaStreamEndCapture(stream,&graph[code]));
                    check(cudaGraphInstantiate(&exec[code],graph[code],nullptr,nullptr,0));
                }
                cudaEvent_t start,end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
                for (int repeat = 0; repeat < 4; ++repeat)
                    for (int code = 0; code < 2; ++code) check(cudaGraphLaunch(exec[code],stream));
                check(cudaStreamSynchronize(stream));
                for (int repeat = 0; repeat < 7; ++repeat) {
                    float us[2];
                    for (int slot = 0; slot < 2; ++slot) {
                        const int code = (repeat+slot)&1;
                        check(cudaEventRecord(start,stream));
                        for (int batch = 0; batch < 16; ++batch) check(cudaGraphLaunch(exec[code],stream));
                        check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
                        float ms; check(cudaEventElapsedTime(&ms,start,end)); us[code]=ms*1000/(16*32);
                    }
                    std::printf("gate_code_decode_timing pair=%d repeat=%d baseline_us=%.6f kmajor_us=%.6f scale_view=%d scope=ten_routes_two_rotating_synthetic_sets_gate_only_not_model\n",
                        pair,repeat,us[0],us[1],scale_view);
                }
                check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
                for (int code = 0; code < 2; ++code) {
                    check(cudaGraphExecDestroy(exec[code])); check(cudaGraphDestroy(graph[code]));
                }
            }
        }
        if (read(view.p,original_view.size()) != original_view || read(weights.p,payload.size()) != payload) return 7;
    }
    check(cudaStreamDestroy(stream));
    std::printf("gate_code_decode rejected=%d changed_replays=%d rows=1280 inner=2560 routes=10 sets=2 layouts=ROW,PAIR simt_pipeline=0,1 mma_parts=1,2,8 mma_prequantized=1 split_row=640 byte_exact=1 weights_immutable=1 guards=1\n",rejects,changed);
}
