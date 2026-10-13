// Bounded instruction diagnostic, NOT a GEMM/model benchmark. No weight loads,
// scaling/dequantization, grouping, staging or routing cost is represented.
// Build: nvcc -O2 --generate-code=arch=compute_121a,code=sm_121a -I. tools/probe_cuda_tensor_formats.cu -o build_cuda/probe_cuda_tensor_formats
#include "runtime/cuda_host_memory_budget.h"
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM120_ALL) && !defined(__CUDA_ARCH_FEAT_SM121_ALL)
#error "Tensor format probe requires sm_120a or sm_121a; refuse unsupported image at build time"
#endif

static void check(cudaError_t result) {
    if(result!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(result));std::exit(1); }
}
constexpr int blocks=384, threads=128, steps=1024;

template<bool FP4,int Chains>
__global__ void tensor_probe(float* output) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    float a[Chains][4]={};
    const unsigned one=FP4?0x22222222u:0x3f803f80u;
    const unsigned scale=0x38383838u;
#pragma unroll 1
    for(int k=0;k<steps;++k) {
#pragma unroll
        for(int c=0;c<Chains;++c) {
            if constexpr(FP4) asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3}, %5,{0,0}, %5,{0,0};"
                : "+f"(a[c][0]),"+f"(a[c][1]),"+f"(a[c][2]),"+f"(a[c][3]) : "r"(one),"r"(scale));
            else {
#pragma unroll
                for(int q=0;q<4;++q) asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};"
                    : "+f"(a[c][0]),"+f"(a[c][1]),"+f"(a[c][2]),"+f"(a[c][3]) : "r"(one));
            }
        }
    }
#pragma unroll
    for(int c=0;c<Chains;++c)for(int i=0;i<4;++i)
        output[((blockIdx.x*blockDim.x+threadIdx.x)*Chains+c)*4+i]=a[c][i];
#endif
}

template<bool FP4,int Chains>
static float measure(float* output) {
    tensor_probe<FP4,Chains><<<blocks,threads>>>(output);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    std::vector<float> actual(blocks*threads*Chains*4);
    check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(float v:actual)if(v!=float(steps*64)) {std::fprintf(stderr,"result mismatch %g\n",v);std::exit(1);}
    cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    for(int i=0;i<16;++i)tensor_probe<FP4,Chains><<<blocks,threads>>>(output);
    check(cudaGetLastError());check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));
    float ms;check(cudaEventElapsedTime(&ms,start,stop));
    check(cudaEventDestroy(start));check(cudaEventDestroy(stop));return ms*1000/16;
}

template<int Chains> static void compare(float* output) {
    for(int repeat=0;repeat<7;++repeat)for(int order=0;order<2;++order) {
        const bool fp4=((repeat+order)%2)!=0;
        float us=fp4?measure<true,Chains>(output):measure<false,Chains>(output);
        std::printf("tensor_instruction_probe chains=%d fp4=%d repeat=%d us=%.3f exact_ones=true\n",Chains,fp4,repeat,us);
        std::fflush(stdout);
    }
}

int main() {
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=12 || prop.minor>1)return 77;
    constexpr size_t bytes=size_t(blocks)*threads*4*4*sizeof(float);
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);
    if(!mollm_memory::cuda_host_allocation_allowed(bytes,"bounded tensor instruction probe"))return 2;
    float* output=nullptr;check(cudaMalloc(&output,bytes));
    compare<1>(output);compare<4>(output);check(cudaFree(output));
}
