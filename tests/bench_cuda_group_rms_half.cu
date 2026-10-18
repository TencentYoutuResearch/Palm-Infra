#include "kernels/cuda/elementwise.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <algorithm>

// Diagnostic only. The baseline calls the actual production RMS kernel.
// Candidate calls the production shadow entry and keeps the reduction order.
static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}
__global__ void convert(const float* input, __half* output, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2half(input[i]);
}
static bool cached_input = false;
static bool inject_rms = false;
static float *inject_branch = nullptr, *inject_gates = nullptr, *inject_carry = nullptr;
static cudaGraphExec_t capture(bool fusion, float* x, float* w, float* y,
                              __half* h, int width, int groups, int rows, int calls) {
    const size_t count = size_t(width) * groups * rows;
    check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < calls; ++i) {
        if (inject_rms) {
            if (fusion) {
                if (!mollm_cuda::launch_gr_inject_group_rms_half(inject_branch,x,
                        inject_gates,w,inject_carry,y,reinterpret_cast<uint16_t*>(h),
                        width,groups,rows,1e-6f,cudaStreamPerThread)) std::exit(4);
            } else {
                mollm_cuda::launch_gr_inject(inject_branch,x,inject_gates,inject_carry,width,groups,rows);
                if (!mollm_cuda::launch_group_rms_norm_half(inject_carry,w,y,
                        reinterpret_cast<uint16_t*>(h),width,groups,rows,1e-6f)) std::exit(4);
            }
            continue;
        }
        if (fusion || cached_input) {
            if (!mollm_cuda::launch_group_rms_norm_half(x,w,y,
                    reinterpret_cast<uint16_t*>(h),width,groups,rows,1e-6f,
                    cached_input && fusion && width == 2560)) std::exit(4);
        }
        else {
            mollm_cuda::launch_group_rms_norm(x, w, y, width, groups, rows, 1e-6f);
            convert<<<(count + 255) / 256, 256>>>(y, h, count);
        }
    }
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    check(cudaGraphDestroy(graph));
    return executable;
}
static float timed(cudaGraphExec_t graph, int calls) {
    for (int i = 0; i < 2; ++i) check(cudaGraphLaunch(graph, cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    check(cudaEventRecord(begin, cudaStreamPerThread));
    for (int i = 0; i < 3; ++i) check(cudaGraphLaunch(graph, cudaStreamPerThread));
    check(cudaEventRecord(end, cudaStreamPerThread)); check(cudaEventSynchronize(end));
    float ms;
    check(cudaEventElapsedTime(&ms, begin, end));
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
    return ms * 1000 / (3 * calls);
}
int main(int argc,char** argv) {
    cached_input = argc == 2 && !std::strcmp(argv[1],"--cached-input");
    inject_rms = argc == 2 && !std::strcmp(argv[1],"--inject-rms");
    if (argc > 2 || (argc == 2 && !cached_input && !inject_rms)) return 64;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    // Baseline must not select its different single-row cached specialization.
    setenv("MOLLM_CUDA_GROUP_RMS_CACHE", "0", 1);
    for (int width : inject_rms ? std::vector<int>{2560} : std::vector<int>{1,31,257,640,2560,4097})
    for (int rows : inject_rms ? std::vector<int>{1,3,17,256,345,512,719,857,1024} : std::vector<int>{1,3,17,256,857}) {
        constexpr int groups = 4, guard = 8;
        const size_t count = size_t(width) * groups * rows;
        std::vector<float> x(count), w(width * groups), ref(count + guard), actual(count + guard);
        std::vector<unsigned short> href(count + guard), hactual(count + guard);
        std::vector<float> branch(inject_rms ? count/groups : 0), gates(inject_rms ? rows*groups : 0);
        std::vector<float> carry_ref(inject_rms ? count+guard : 0), carry_actual(carry_ref.size());
        for (size_t i = 0; i < w.size(); ++i) w[i] = .1f + std::cos(float(i) * .19f);
        float *dx, *dw, *dy;
        __half* dh;
        check(cudaMalloc(&dx, count * 4)); check(cudaMalloc(&dw, w.size() * 4));
        check(cudaMalloc(&dy, (count + guard) * 4)); check(cudaMalloc(&dh, (count + guard) * 2));
        if (inject_rms) {
            check(cudaMalloc(&inject_branch, branch.size()*4));
            check(cudaMalloc(&inject_gates, gates.size()*4));
            check(cudaMalloc(&inject_carry, (count+guard)*4));
            if (rows==1) {
                int rejected=0;
                auto reject=[&](const float* b,const float* x,const float* g,const float* w,
                                float* c,float* y,uint16_t* h,int width,int groups,int rows) {
                    if (mollm_cuda::launch_gr_inject_group_rms_half(b,x,g,w,c,y,h,
                            width,groups,rows,1e-6f,cudaStreamPerThread)) std::exit(5);
                    ++rejected;
                };
                auto h=reinterpret_cast<uint16_t*>(dh);
                reject(nullptr,dx,inject_gates,dw,inject_carry,dy,h,2560,4,1);
                reject(inject_branch,nullptr,inject_gates,dw,inject_carry,dy,h,2560,4,1);
                reject(inject_branch,dx,nullptr,dw,inject_carry,dy,h,2560,4,1);
                reject(inject_branch,dx,inject_gates,nullptr,inject_carry,dy,h,2560,4,1);
                reject(inject_branch,dx,inject_gates,dw,nullptr,dy,h,2560,4,1);
                reject(inject_branch,dx,inject_gates,dw,inject_carry,nullptr,h,2560,4,1);
                reject(inject_branch,dx,inject_gates,dw,inject_carry,dy,nullptr,2560,4,1);
                for(int width:{0,640,-1})reject(inject_branch,dx,inject_gates,dw,inject_carry,dy,h,width,4,1);
                for(int groups:{0,1,-1})reject(inject_branch,dx,inject_gates,dw,inject_carry,dy,h,2560,groups,1);
                for(int rows:{0,-1,4097})reject(inject_branch,dx,inject_gates,dw,inject_carry,dy,h,2560,4,rows);
                reject(inject_branch,dx,inject_gates,dw,dx,dy,h,2560,4,1);
                reject(inject_branch,dx,inject_gates,dw,inject_carry,dx,h,2560,4,1);
                reject(inject_branch,dx,inject_gates,dw,inject_carry,dy,reinterpret_cast<uint16_t*>(dx),2560,4,1);
                reject(inject_branch,dx,inject_gates,dw,inject_carry,inject_carry,h,2560,4,1);
                reject(inject_branch,dx,inject_gates,dw,inject_carry,inject_carry+1,h,2560,4,1);
                reject(inject_branch,dx,inject_gates,dw,dx+1,dy,h,2560,4,1);
                reject(inject_branch,dx,inject_gates,dw,inject_carry,dy,reinterpret_cast<uint16_t*>(dy+1),2560,4,1);
                reject(inject_branch,dx,inject_gates,dw,reinterpret_cast<float*>(UINTPTR_MAX-4),dy,h,2560,4,1);
                check(cudaGetLastError());
                std::printf("inject_rms_invalid_arguments=%d rejected=1\n",rejected);
            }
        }
        if (width==1 && rows==1) {
            auto reject=[&](const float* input,const float* weight,float* output,
                            uint16_t* shadow,int width,int groups,int rows) {
                if (mollm_cuda::launch_group_rms_norm_half(input,weight,output,shadow,
                        width,groups,rows,1e-6f)) std::exit(5);
            };
            auto half=reinterpret_cast<uint16_t*>(dh);
            reject(nullptr,dw,dy,half,1,4,1);
            reject(dx,nullptr,dy,half,1,4,1);
            reject(dx,dw,nullptr,half,1,4,1);
            reject(dx,dw,dy,nullptr,1,4,1);
            reject(dx,dw,dy,half,0,4,1);
            reject(dx,dw,dy,half,1,0,1);
            reject(dx,dw,dy,half,1,4,0);
            reject(dx,dw,dy,half,1,4,2147483647);
            if (mollm_cuda::launch_group_rms_norm_half(dx,dw,dy,half,1,4,1,1e-6f,true)) return 5;
            check(cudaGetLastError());
            std::puts("invalid_arguments=8 rejected=1");
        }
        check(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        cudaGraphExec_t graphs[2] = {
            capture(false, dx, dw, dy, dh, width, groups, rows, 1),
            capture(true, dx, dw, dy, dh, width, groups, rows, 1)};
        const int variants = inject_rms ? 8 : 4;
        for (int variant = 0; variant < variants; ++variant) {
            for (size_t i = 0; i < w.size(); ++i)
                w[i] = .1f + std::cos(float(i) * .19f + variant);
            check(cudaMemcpy(dw,w.data(),w.size() * 4,cudaMemcpyHostToDevice));
            for (size_t i = 0; i < count; ++i)
                x[i] = variant == 3 ? 0.f : std::sin(float(i) * .731f + variant) * (1 + i % 13);
            if (inject_rms) {
                for (size_t i=0;i<branch.size();++i) branch[i]=std::cos(float(i)*.137f+variant)*3.f;
                for (size_t i=0;i<gates.size();++i) gates[i]=std::sin(float(i)*.91f+variant);
                if (variant==3 || variant==4 || variant==5) {
                    std::fill(x.begin(),x.end(),variant==4 ? -0.f : variant==5 ? std::numeric_limits<float>::denorm_min() : 0.f);
                    std::fill(branch.begin(),branch.end(),variant==4 ? -0.f : 0.f);
                    std::fill(gates.begin(),gates.end(),1.f);
                }
                if (variant==6) { x[0]=std::numeric_limits<float>::infinity(); branch[1]=-std::numeric_limits<float>::infinity(); }
                if (variant==5) x[0]=std::numeric_limits<float>::min()/128.f;
                if (variant==7) { x[0]=std::numeric_limits<float>::quiet_NaN(); gates[0]=std::numeric_limits<float>::quiet_NaN(); }
                check(cudaMemcpy(inject_branch,branch.data(),branch.size()*4,cudaMemcpyHostToDevice));
                check(cudaMemcpy(inject_gates,gates.data(),gates.size()*4,cudaMemcpyHostToDevice));
            }
            check(cudaMemcpy(dx, x.data(), count * 4, cudaMemcpyHostToDevice));
            for (int mode = 0; mode < 2; ++mode) {
                check(cudaMemset(dy, 0x7f, (count + guard) * 4));
                check(cudaMemset(dh, 0x5a, (count + guard) * 2));
                if (inject_rms) check(cudaMemset(inject_carry,0x7f,(count+guard)*4));
                check(cudaGraphLaunch(graphs[mode], cudaStreamPerThread));
                check(cudaStreamSynchronize(cudaStreamPerThread));
                check(cudaMemcpy(mode ? actual.data() : ref.data(), dy, (count + guard) * 4, cudaMemcpyDeviceToHost));
                check(cudaMemcpy(mode ? hactual.data() : href.data(), dh, (count + guard) * 2, cudaMemcpyDeviceToHost));
                if (inject_rms) check(cudaMemcpy(mode ? carry_actual.data() : carry_ref.data(),inject_carry,(count+guard)*4,cudaMemcpyDeviceToHost));
            }
            if (std::memcmp(ref.data(), actual.data(), (count + guard) * 4) ||
                std::memcmp(href.data(), hactual.data(), (count + guard) * 2) ||
                (inject_rms && std::memcmp(carry_ref.data(),carry_actual.data(),(count+guard)*4))) {
                std::fprintf(stderr, "mismatch width=%d rows=%d variant=%d\n", width, rows, variant);
                return 2;
            }
            // Verify guards independently, not only that both paths agree.
            if(inject_rms && variant==4) {
                uint32_t bits;std::memcpy(&bits,&carry_actual[0],4);
                if(bits!=0x80000000u)return 3;
            }
            if(inject_rms && variant==5 && (carry_actual[0]==0.f || actual[0]==0.f))return 3;
            for (size_t i = count; i < count + guard; ++i) {
                unsigned bits;
                std::memcpy(&bits, &actual[i], sizeof(bits));
                if (bits != 0x7f7f7f7f || hactual[i] != 0x5a5a) return 3;
                if (inject_rms) { std::memcpy(&bits,&carry_actual[i],4); if(bits!=0x7f7f7f7f)return 3; }
            }
        }
        std::printf("width=%d rows=%d variants=%d fp32_exact=1 fp16_exact=1 guards=1 inject_rms=%d\n", width, rows,variants,int(inject_rms));
        for (auto graph : graphs) check(cudaGraphExecDestroy(graph));
        if (width == 2560 && (rows == 256 || rows == 857 || (inject_rms && (rows==512 || rows==719 || rows==1024))) &&
            !std::getenv("MOLLM_GROUP_RMS_HALF_CHECK_ONLY")) {
            // Time a varied-input case already checked above, not the zero fixture.
            for (size_t i = 0; i < count; ++i)
                x[i] = std::sin(float(i) * .731f + 2) * (1 + i % 13);
            for (size_t i = 0; i < w.size(); ++i)
                w[i] = .1f + std::cos(float(i) * .19f + 2);
            check(cudaMemcpy(dx,x.data(),count * 4,cudaMemcpyHostToDevice));
            check(cudaMemcpy(dw,w.data(),w.size() * 4,cudaMemcpyHostToDevice));
            if (inject_rms) {
                for (size_t i=0;i<branch.size();++i)branch[i]=std::cos(float(i)*.137f+2)*3.f;
                for (size_t i=0;i<gates.size();++i)gates[i]=std::sin(float(i)*.91f+2);
                check(cudaMemcpy(inject_branch,branch.data(),branch.size()*4,cudaMemcpyHostToDevice));
                check(cudaMemcpy(inject_gates,gates.data(),gates.size()*4,cudaMemcpyHostToDevice));
            }
            constexpr int calls = 64;
            graphs[0] = capture(false, dx, dw, dy, dh, width, groups, rows, calls);
            graphs[1] = capture(true, dx, dw, dy, dh, width, groups, rows, calls);
            for (int order = 0; order < 7; ++order) {
                float us[2];
                for (int step = 0; step < 2; ++step) {
                    const int mode = step ^ (order & 1);
                    us[mode] = timed(graphs[mode], calls);
                }
                std::printf("width=%d rows=%d order=%d baseline_us=%.4f fused_us=%.4f\n",
                            width, rows, order, us[0], us[1]);
            }
            for (auto graph : graphs) check(cudaGraphExecDestroy(graph));
        }
        check(cudaFree(dx)); check(cudaFree(dw)); check(cudaFree(dy)); check(cudaFree(dh));
        if (inject_rms) {
            check(cudaFree(inject_branch));check(cudaFree(inject_gates));check(cudaFree(inject_carry));
            inject_branch=inject_gates=inject_carry=nullptr;
        }
    }
}
