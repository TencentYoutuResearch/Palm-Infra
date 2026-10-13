// Fixed-size numerical diagnostic, NOT a GEMM or throughput benchmark.
// nvcc -O2 --generate-code=arch=compute_121a,code=sm_121a -I. tools/probe_cuda_scaled_tensor_formats.cu -o build_cuda/probe_cuda_scaled_tensor_formats
#include "runtime/cuda_host_memory_budget.h"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM121_ALL)
#error "Scaled tensor probe requires an sm_121a image; reject unsupported image before launch"
#endif

constexpr int scales=127; // Positive finite E4M3 encodings0..126, excluding NaN127.
constexpr int cases=16*16*scales*scales;
struct Result { float fp4,bf16; };

static void check(cudaError_t status) {
    if(status!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(1);}
}

__device__ float fp4_value(unsigned code) {
    const float values[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    return code&8 ? -values[code&7] : values[code&7];
}

__global__ void scaled_probe(Result* results,unsigned* errors) {
#if defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const int index=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(index>=cases)return; // Whole-warp uniform branch.
    const int lane=threadIdx.x%32;
    const unsigned sa=index%scales,sb=(index/scales)%scales;
    const unsigned ca=(index/(scales*scales))%16,cb=(index/(scales*scales*16))%16;
    const float as=__half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(sa,__NV_E4M3)));
    const float bs=__half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3)));
    const float av=fp4_value(ca)*as,bv=fp4_value(cb)*bs;
    const __nv_bfloat16 ah=__float2bfloat16(av),bh=__float2bfloat16(bv);
    const unsigned aa=__bfloat16_as_ushort(ah)*0x10001u,bb=__bfloat16_as_ushort(bh)*0x10001u;
    const unsigned ac=ca*0x11111111u,bc=cb*0x11111111u;
    const unsigned ap=sa*0x1010101u,bp=sb*0x1010101u;
    float x[4]={},y[4]={};
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3}, %6,{0,0}, %7,{0,0};"
        : "+f"(x[0]),"+f"(x[1]),"+f"(x[2]),"+f"(x[3]) : "r"(ac),"r"(bc),"r"(ap),"r"(bp));
#pragma unroll
    for(int k=0;k<4;++k) asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
        : "+f"(y[0]),"+f"(y[1]),"+f"(y[2]),"+f"(y[3]) : "r"(aa),"r"(bb));
    bool bad=__bfloat162float(ah)!=av || __bfloat162float(bh)!=bv;
    const float expected=(av*bv)*64.f;
#pragma unroll
    for(int i=0;i<4;++i)bad=bad || x[i]!=expected || y[i]!=expected;
    if(bad)atomicAdd(errors,1u);
    if(lane==0)results[index]={x[0],y[0]};
#endif
}

int main() {
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=12 || prop.minor!=1)return 77;
    constexpr size_t bytes=size_t(cases)*sizeof(Result);
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);
    if(!mollm_memory::cuda_host_allocation_allowed(bytes+sizeof(unsigned),"scaled tensor diagnostic"))return 2;
    Result* output=nullptr;unsigned* errors=nullptr;
    check(cudaMalloc(&output,bytes));check(cudaMalloc(&errors,sizeof(unsigned)));
    check(cudaMemset(errors,0,sizeof(unsigned)));
    constexpr int warps_per_block=4;
    scaled_probe<<<(cases+warps_per_block-1)/warps_per_block,warps_per_block*32>>>(output,errors);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    unsigned bad=0;check(cudaMemcpy(&bad,errors,sizeof(unsigned),cudaMemcpyDeviceToHost));
    std::vector<Result> values(cases);check(cudaMemcpy(values.data(),output,bytes,cudaMemcpyDeviceToHost));
    const float codes[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    float scale_values[scales];
    for(int code=0;code<scales;++code) {
        const int exponent=code>>3,mantissa=code&7;
        scale_values[code]=exponent ? std::ldexp(1.f+mantissa/8.f,exponent-7) : std::ldexp(float(mantissa),-9);
    }
    for(int i=0;i<cases;++i) {
        const int sa=i%scales,sb=(i/scales)%scales;
        const int ca=(i/(scales*scales))%16,cb=(i/(scales*scales*16))%16;
        const float expected=((codes[ca]*scale_values[sa])*(codes[cb]*scale_values[sb]))*64.f;
        if(values[i].fp4!=expected || values[i].bf16!=expected) {
            std::fprintf(stderr,"scaled probe mismatch case=%d codes=%d/%d scales=%d/%d expected=%g fp4=%g bf16=%g\n",i,ca,cb,sa,sb,expected,values[i].fp4,values[i].bf16);
            bad=1;break;
        }
    }
    check(cudaFree(output));check(cudaFree(errors));
    std::printf("scaled_tensor_probe cases=%d output_bytes=%zu mismatching_lanes=%u scope=uniform_tiles_not_general_gemm_or_model_quality\n",cases,bytes,bad);
    return bad ? 1 : 0;
}
