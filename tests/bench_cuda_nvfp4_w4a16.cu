// Independent W4A16 diagnostic. No backend integration or model default change.
// Weights remain packed; decoded BF16 operands exist only in registers.
#include "kernels/cuda/moe.h"
#include "runtime/cuda_host_memory_budget.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void ck(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);} }
template<class T>T* alloc(size_t n){T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
float f4(unsigned c){const float a[8]={0,.5f,1,1.5f,2,3,4,6};return c&8?-a[c&7]:a[c&7];}
float f8(unsigned c){int e=c>>3,m=c&7;return e?std::ldexp(1.f+m/8.f,e-7):std::ldexp(float(m),-9);}
__host__ __device__ size_t offset(int r,int k,int inner,bool pair){
    return pair?(size_t(r/4)*(inner/16)+k/16)*32+(r%4)*8+(k%16)/2:size_t(r)*(inner/2)+k/2;
}
__device__ uint16_t decode_bits(unsigned c){
    unsigned m=c&7;
    unsigned bits=m==0?0:(m==1?0x3f00:((126+(m>>1))<<7)+((m&1)<<6));
    return bits|((c&8)<<12);
}
__device__ float decode(unsigned c){
    __nv_bfloat16_raw raw;raw.x=decode_bits(c);
    return __bfloat162float(__nv_bfloat16(raw));
}
__device__ uint32_t raw_code(uint8_t c){return uint32_t(decode_bits(c&15))|(uint32_t(decode_bits(c>>4))<<16);}
__device__ uint32_t lookup_code(uint8_t c){
    // Two register byte permutations: independent low/high BF16 byte tables.
    unsigned selector=(c&7)|((c&0x70)<<4);
    unsigned low=__byte_perm(0xc0800000u,0xc0804000u,selector)&0x00ff00ffu;
    unsigned high=__byte_perm(0x3f3f3f00u,0x40404040u,selector)&0x00ff00ffu;
    return low|(high<<8)|((unsigned(c)&8)<<12)|((unsigned(c)&0x80)<<24);
}
__global__ void check_codes(uint32_t* out){
    unsigned c=threadIdx.x;out[c]=lookup_code(c);out[c+256]=raw_code(c);
}
__device__ uint32_t scaled_code(uint8_t c,float s){
    __nv_bfloat162_raw b=__floats2bfloat162_rn(decode(c&15)*s,decode(c>>4)*s);
    return uint32_t(b.x)|(uint32_t(b.y)<<16);
}
__device__ uint32_t operand(const mollm_cuda::Nvfp4Projection& j,int r,int k,int inner){
    uint8_t c=j.weight[offset(r,k,inner,j.pair_layout)];
    float s=__half2float(__nv_cvt_fp8_to_halfraw(j.block_scales[size_t(r)*(inner/16)+k/16],__NV_E4M3));
    return scaled_code(c,s);
}
__device__ uint32_t activation(const __nv_bfloat16* x,int col,int k,int inner,int size){
    if(col>=size)return 0;
    const __nv_bfloat16_raw a=x[size_t(col)*inner+k],b=x[size_t(col)*inner+k+1];
    return uint32_t(a.x)|(uint32_t(b.x)<<16);
}
template<int Tile>struct Stage {uint32_t codes;float s0,s1;uint32_t b[Tile/8][2];};
template<int Tile>__device__ Stage<Tile> load_stage(const mollm_cuda::Nvfp4Projection& j,
    const __nv_bfloat16* input,int r,int c,int k,int inner,int size,int lane){
    Stage<Tile> v;
    v.codes=reinterpret_cast<const uint32_t*>(j.weight)[(size_t(r/16)*(inner/16)+k/16)*32+lane];
    v.s0=__half2float(__nv_cvt_fp8_to_halfraw(j.block_scales[size_t(r)*(inner/16)+k/16],__NV_E4M3));
    v.s1=__half2float(__nv_cvt_fp8_to_halfraw(j.block_scales[size_t(r+8)*(inner/16)+k/16],__NV_E4M3));
    #pragma unroll
    for(int b=0;b<Tile/8;++b){v.b[b][0]=activation(input,c+b*8+lane/4,k+2*(lane%4),inner,size);
        v.b[b][1]=activation(input,c+b*8+lane/4,k+8+2*(lane%4),inner,size);}
    return v;
}
template<bool Fragment,int Tile=8,bool ScaleAfter=false,bool Pipeline=false>
__global__ void project(const mollm_cuda::Nvfp4Projection* jobs,
    const mollm_cuda::Nvfp4MmaBatch* groups,const __nv_bfloat16* x,int rows,int inner){
    auto g=groups[blockIdx.y];auto j=jobs[g.first];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,q=lane/4,t=lane%4;
    int r=blockIdx.x*64+warp*16+q,c=blockIdx.z*Tile;
    if(r>=rows)return; // All test shapes have complete M16 warp tiles.
    const auto* input=x+size_t(g.first)*inner;
    float d[Tile/8][4]={};
    Stage<Tile> current;
    if constexpr(Pipeline)current=load_stage<Tile>(j,input,r,c,0,inner,g.size,lane);
    for(int k=0;k<inner;k+=16){
        Stage<Tile> next;
        if constexpr(Pipeline){if(k+16<inner)next=load_stage<Tile>(j,input,r,c,k+16,inner,g.size,lane);}
        uint32_t a0,a1,a2,a3;float s0=1,s1=1;
        if constexpr(Fragment){
            uint32_t codes;
            if constexpr(Pipeline){codes=current.codes;s0=current.s0;s1=current.s1;}
            else{codes=reinterpret_cast<const uint32_t*>(j.weight)[(size_t(r/16)*(inner/16)+k/16)*32+lane];
                s0=__half2float(__nv_cvt_fp8_to_halfraw(j.block_scales[size_t(r)*(inner/16)+k/16],__NV_E4M3));
                s1=__half2float(__nv_cvt_fp8_to_halfraw(j.block_scales[size_t(r+8)*(inner/16)+k/16],__NV_E4M3));}
            if constexpr(ScaleAfter){a0=raw_code(codes);a1=raw_code(codes>>8);a2=raw_code(codes>>16);a3=raw_code(codes>>24);}
            else{a0=scaled_code(codes,s0);a1=scaled_code(codes>>8,s1);
                a2=scaled_code(codes>>16,s0);a3=scaled_code(codes>>24,s1);}
        }else{
            a0=operand(j,r,k+2*t,inner);a1=operand(j,r+8,k+2*t,inner);
            a2=operand(j,r,k+8+2*t,inner);a3=operand(j,r+8,k+8+2*t,inner);
        }
        #pragma unroll
        for(int b=0;b<Tile/8;++b){
        uint32_t b0,b1;
        if constexpr(Pipeline){b0=current.b[b][0];b1=current.b[b][1];}
        else{b0=activation(input,c+b*8+q,k+2*t,inner,g.size);
            b1=activation(input,c+b*8+q,k+8+2*t,inner,g.size);}
        float p0=ScaleAfter?0:d[b][0],p1=ScaleAfter?0:d[b][1];
        float p2=ScaleAfter?0:d[b][2],p3=ScaleAfter?0:d[b][3];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(p0),"+f"(p1),"+f"(p2),"+f"(p3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        if constexpr(ScaleAfter){d[b][0]=fmaf(p0,s0,d[b][0]);d[b][1]=fmaf(p1,s0,d[b][1]);
            d[b][2]=fmaf(p2,s1,d[b][2]);d[b][3]=fmaf(p3,s1,d[b][3]);}
        else{d[b][0]=p0;d[b][1]=p1;d[b][2]=p2;d[b][3]=p3;}
        }
        if constexpr(Pipeline){if(k+16<inner)current=next;}
    }
    for(int b=0;b<Tile/8;++b)for(int v=0;v<4;++v){int col=c+b*8+2*t+(v&1),row=r+(v>=2?8:0);
        if(col<g.size){auto out=jobs[g.first+col];out.output[row]=(d[b][v]*j.row_scales[row])*out.output_scale;}
    }
}
struct Stage32 {Stage<16> first;uint32_t codes;float s0,s1;};
template<int InlineScales=0>__device__ Stage32 load_stage32(const mollm_cuda::Nvfp4Projection& j,
    const __nv_bfloat16* input,int r,int c,int k,int inner,int size,int lane){
    Stage32 v;
    if constexpr(InlineScales){
        const uint8_t *p0,*p1,*sf0,*sf1;
        if constexpr(InlineScales==1){
            size_t tile=size_t(r/16)*(inner/16)+k/16;p0=j.weight+tile*144;
            p1=p0+(inner/16)*144;sf0=p0+128;sf1=p1+128;
        }else{
            // M32/K32 tile: four M16/K16 code fragments then their scale tuples.
            // 576-byte stride preserves 64-byte alignment, unlike 144-byte slices.
            const uint8_t* p=j.weight+(size_t(r/32)*(inner/32)+k/32)*576;
            int sub=(k/16)&1;p0=p+sub*128;p1=p+(sub+2)*128;
            sf0=p+512+sub*16;sf1=p+512+(sub+2)*16;
        }
        v.first.codes=reinterpret_cast<const uint32_t*>(p0)[lane];
        v.first.s0=__half2float(__nv_cvt_fp8_to_halfraw(sf0[lane/4],__NV_E4M3));
        v.first.s1=__half2float(__nv_cvt_fp8_to_halfraw(sf0[8+lane/4],__NV_E4M3));
        v.codes=reinterpret_cast<const uint32_t*>(p1)[lane];
        v.s0=__half2float(__nv_cvt_fp8_to_halfraw(sf1[lane/4],__NV_E4M3));
        v.s1=__half2float(__nv_cvt_fp8_to_halfraw(sf1[8+lane/4],__NV_E4M3));
        #pragma unroll
        for(int b=0;b<2;++b){v.first.b[b][0]=activation(input,c+b*8+lane/4,k+2*(lane%4),inner,size);
            v.first.b[b][1]=activation(input,c+b*8+lane/4,k+8+2*(lane%4),inner,size);}
    }else{
        v.first=load_stage<16>(j,input,r,c,k,inner,size,lane);r+=16;
        v.codes=reinterpret_cast<const uint32_t*>(j.weight)[(size_t(r/16)*(inner/16)+k/16)*32+lane];
        v.s0=__half2float(__nv_cvt_fp8_to_halfraw(j.block_scales[size_t(r)*(inner/16)+k/16],__NV_E4M3));
        v.s1=__half2float(__nv_cvt_fp8_to_halfraw(j.block_scales[size_t(r+8)*(inner/16)+k/16],__NV_E4M3));
    }
    return v;
}
template<bool Pipeline,int InlineScales=0,bool Lookup=false,bool ExpertMajor=false,bool Compact=false,bool Bf16Output=false>
__global__ void project_m32(const mollm_cuda::Nvfp4Projection* jobs,
    const mollm_cuda::Nvfp4MmaBatch* groups,const __nv_bfloat16* x,int rows,int inner,const int* group_count=nullptr){
    if constexpr(Compact){if(blockIdx.y>=*group_count)return;}
    auto g=groups[ExpertMajor?blockIdx.z:blockIdx.y];auto j=jobs[g.first];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,q=lane/4,t=lane%4;
    int r=blockIdx.x*128+warp*32+q,c=Compact?0:(ExpertMajor?blockIdx.y:blockIdx.z)*16;
    if(r>=rows)return; // Complete M32 tiles; small fixture has two inactive warps.
    const auto* input=x+size_t(g.first)*inner;
    float d[2][2][4]={};Stage32 current;
    if constexpr(Pipeline)current=load_stage32<InlineScales>(j,input,r,c,0,inner,g.size,lane);
    for(int k=0;k<inner;k+=16){
        Stage32 next;
        if constexpr(Pipeline){if(k+16<inner)next=load_stage32<InlineScales>(j,input,r,c,k+16,inner,g.size,lane);}
        else current=load_stage32<InlineScales>(j,input,r,c,k,inner,g.size,lane);
        #pragma unroll
        for(int m=0;m<2;++m){
            uint32_t codes=m?current.codes:current.first.codes;
            float s0=m?current.s0:current.first.s0,s1=m?current.s1:current.first.s1;
            uint32_t a0,a1,a2,a3;
            if constexpr(Lookup){a0=lookup_code(codes);a1=lookup_code(codes>>8);a2=lookup_code(codes>>16);a3=lookup_code(codes>>24);}
            else{a0=raw_code(codes);a1=raw_code(codes>>8);a2=raw_code(codes>>16);a3=raw_code(codes>>24);}
            #pragma unroll
            for(int b=0;b<2;++b){
                float p0=0,p1=0,p2=0,p3=0;
                uint32_t b0=current.first.b[b][0],b1=current.first.b[b][1];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(p0),"+f"(p1),"+f"(p2),"+f"(p3)
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
                d[m][b][0]=fmaf(p0,s0,d[m][b][0]);d[m][b][1]=fmaf(p1,s0,d[m][b][1]);
                d[m][b][2]=fmaf(p2,s1,d[m][b][2]);d[m][b][3]=fmaf(p3,s1,d[m][b][3]);
            }
        }
        if constexpr(Pipeline){if(k+16<inner)current=next;}
    }
    #pragma unroll
    for(int m=0;m<2;++m)for(int b=0;b<2;++b)for(int v=0;v<4;++v){
        int col=c+b*8+2*t+(v&1),row=r+m*16+(v>=2?8:0);
        if(col<g.size){auto out=jobs[g.first+col];float value=(d[m][b][v]*j.row_scales[row])*out.output_scale;
            if constexpr(Bf16Output)reinterpret_cast<__nv_bfloat16*>(out.output)[row]=__float2bfloat16_rn(value);
            else out.output[row]=value;}
    }
}
template<class F>float timing(F launch){
    cudaGraph_t g;cudaGraphExec_t ex;cudaEvent_t a,b;
    ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));launch();
    ck(cudaStreamEndCapture(cudaStreamPerThread,&g));ck(cudaGraphInstantiateWithFlags(&ex,g,0));
    ck(cudaGraphLaunch(ex,cudaStreamPerThread));ck(cudaStreamSynchronize(cudaStreamPerThread));
    ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a,cudaStreamPerThread));
    for(int i=0;i<32;++i)ck(cudaGraphLaunch(ex,cudaStreamPerThread));
    ck(cudaEventRecord(b,cudaStreamPerThread));ck(cudaEventSynchronize(b));float ms;
    ck(cudaEventElapsedTime(&ms,a,b));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
    ck(cudaGraphExecDestroy(ex));ck(cudaGraphDestroy(g));return ms*1000/32;
}
}
#ifndef MOLLM_W4A16_KERNELS_ONLY
int main(int argc,char** argv){
    bool small=argc==2&&!std::strcmp(argv[1],"--checks-only");
    bool gate=argc==2&&!std::strcmp(argv[1],"--gate");
    if(argc>1&&!small&&!gate){std::fprintf(stderr,"usage: %s [--checks-only | --gate]\n",argv[0]);return 2;}
    cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));if(prop.major!=12||prop.minor!=1)return 77;
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);
    setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","4",1);
    int rows=small?64:(gate?1280:2560),inner=small?64:(gate?2560:640),experts=small?3:32,maxn=64;
    std::printf("diagnostic projection rows=%d inner=%d experts=%d; native comparator=N8 FP32, candidates preplanned; NOT production fused gate/up or BF16 down\n",rows,inner,experts);
    size_t codes=size_t(rows)*inner/2,sfs=size_t(rows)*inner/16,stride=codes+sfs+rows*sizeof(float);
    int capacity=experts*maxn;
    size_t budget=4*experts*stride+size_t(capacity)*(inner*7+rows*4+4*sizeof(mollm_cuda::Nvfp4Projection)+sizeof(mollm_cuda::Nvfp4MmaBatch))+4096;
    if(!mollm_memory::cuda_host_allocation_allowed(budget,"bounded register W4A16 diagnostic"))return 2;
    auto* code_check=alloc<uint32_t>(512);std::vector<uint32_t> decoded(512);
    check_codes<<<1,256,0,cudaStreamPerThread>>>(code_check);ck(cudaGetLastError());
    ck(cudaStreamSynchronize(cudaStreamPerThread));ck(cudaMemcpy(decoded.data(),code_check,2048,cudaMemcpyDeviceToHost));
    for(unsigned c=0;c<256;++c){
        __nv_bfloat16_raw lo=__float2bfloat16(f4(c&15)),hi=__float2bfloat16(f4(c>>4));
        uint32_t expected=uint32_t(lo.x)|(uint32_t(hi.x)<<16);
        if(decoded[c]!=expected||decoded[c+256]!=expected){std::fprintf(stderr,"exhaustive code check failed %u\n",c);return 1;}
    }
    ck(cudaFree(code_check));std::puts("PASS all256 packed E2M1 bytes: lookup/arithmetic/independent BF16 bits, signed zeros included");
    auto* w=alloc<uint8_t>(experts*stride);auto* x=alloc<float>(size_t(capacity)*inner);
    auto* fw=alloc<uint8_t>(experts*stride);auto* fjobs=alloc<mollm_cuda::Nvfp4Projection>(capacity);
    auto* iw=alloc<uint8_t>(2*experts*stride);auto* ijobs=alloc<mollm_cuda::Nvfp4Projection>(2*capacity);
    auto* bw=iw+experts*stride;auto* bjobs=ijobs+capacity;
    auto* bx=alloc<__nv_bfloat16>(size_t(capacity)*inner);auto* y=alloc<float>(size_t(capacity)*rows+8);
    auto* jobs=alloc<mollm_cuda::Nvfp4Projection>(capacity);auto* groups=alloc<mollm_cuda::Nvfp4MmaBatch>(experts);
    auto* plan=alloc<mollm_cuda::Nvfp4MmaBatch>(capacity);
    auto* packed=alloc<uint8_t>(size_t(capacity)*inner/2);auto* scales=alloc<uint8_t>(size_t(capacity)*inner/16);
    std::vector<uint8_t> payload(experts*stride);
    size_t checks=0;double worst=0;
    for(bool pair:{false,true}){
        for(int e=0;e<experts;++e){auto* p=payload.data()+e*stride;
            auto* global=reinterpret_cast<float*>(p+codes+sfs);
            for(int r=0;r<rows;++r){global[r]=(1+(r+e)%7)/64.f;
                for(int k=0;k<inner;k+=16){const uint8_t sf[8]={0,1,7,8,48,56,64,126};
                    p[codes+size_t(r)*(inner/16)+k/16]=sf[(r+k/16+e)%8];
                    for(int b=0;b<8;++b)p[offset(r,k+2*b,inner,pair)]=(r*29+k*17+b*13+e*7)&255;
                }
            }
        }
        ck(cudaMemcpy(w,payload.data(),payload.size(),cudaMemcpyHostToDevice));
        auto fragment_payload=payload;
        for(int e=0;e<experts;++e)for(int m=0;m<rows;m+=16)for(int k=0;k<inner;k+=16)for(int lane=0;lane<32;++lane){
            int r=m+lane/4,t=lane%4;
            size_t dst=e*stride+((size_t(m/16)*(inner/16)+k/16)*32+lane)*4;
            const auto* src=payload.data()+e*stride;
            fragment_payload[dst]=src[offset(r,k+2*t,inner,pair)];
            fragment_payload[dst+1]=src[offset(r+8,k+2*t,inner,pair)];
            fragment_payload[dst+2]=src[offset(r,k+8+2*t,inner,pair)];
            fragment_payload[dst+3]=src[offset(r+8,k+8+2*t,inner,pair)];
        }
        ck(cudaMemcpy(fw,fragment_payload.data(),fragment_payload.size(),cudaMemcpyHostToDevice));
        auto inline_payload=payload;
        for(int e=0;e<experts;++e)for(int m=0;m<rows;m+=16)for(int k=0;k<inner;k+=16){
            size_t tile=size_t(m/16)*(inner/16)+k/16,dst=e*stride+tile*144;
            std::memcpy(inline_payload.data()+dst,fragment_payload.data()+e*stride+tile*128,128);
            for(int r=0;r<16;++r)inline_payload[dst+128+r]=payload[e*stride+codes+size_t(m+r)*(inner/16)+k/16];
        }
        ck(cudaMemcpy(iw,inline_payload.data(),inline_payload.size(),cudaMemcpyHostToDevice));
        auto blocked_payload=payload;
        for(int e=0;e<experts;++e)for(int m=0;m<rows;m+=32)for(int k=0;k<inner;k+=32){
            size_t tile=size_t(m/32)*(inner/32)+k/32,dst=e*stride+tile*576;
            for(int mh=0;mh<2;++mh)for(int kh=0;kh<2;++kh){int sub=mh*2+kh;
                size_t src=e*stride+(size_t(m/16+mh)*(inner/16)+k/16+kh)*128;
                std::memcpy(blocked_payload.data()+dst+sub*128,fragment_payload.data()+src,128);
                for(int r=0;r<16;++r)blocked_payload[dst+512+sub*16+r]=payload[e*stride+codes+size_t(m+mh*16+r)*(inner/16)+k/16+kh];
            }
        }
        ck(cudaMemcpy(bw,blocked_payload.data(),blocked_payload.size(),cudaMemcpyHostToDevice));
        for(int n:{1,7,8,17,32,64}){
            int count=experts*n;std::vector<mollm_cuda::Nvfp4Projection> host(count);
            std::vector<mollm_cuda::Nvfp4MmaBatch> hg(experts);
            for(int e=0;e<experts;++e){hg[e]={e*n,n};for(int j=0;j<n;++j){int i=e*n+j;auto* p=w+e*stride;
                host[i]={p,p+codes,reinterpret_cast<float*>(p+codes+sfs),x+size_t(i)*inner,y+size_t(i)*rows,pair,false,1,1,(j+1)/float(n)};
            }}
            ck(cudaMemcpy(jobs,host.data(),count*sizeof(host[0]),cudaMemcpyHostToDevice));
            auto fragment_host=host;
            for(int i=0;i<count;++i){auto* p=fw+(i/n)*stride;fragment_host[i].weight=p;fragment_host[i].block_scales=p+codes;fragment_host[i].row_scales=reinterpret_cast<float*>(p+codes+sfs);}
            ck(cudaMemcpy(fjobs,fragment_host.data(),count*sizeof(host[0]),cudaMemcpyHostToDevice));
            auto inline_host=host;
            for(int i=0;i<count;++i){auto* p=iw+(i/n)*stride;inline_host[i].weight=p;inline_host[i].block_scales=nullptr;inline_host[i].row_scales=reinterpret_cast<float*>(p+codes+sfs);}
            ck(cudaMemcpy(ijobs,inline_host.data(),count*sizeof(host[0]),cudaMemcpyHostToDevice));
            for(int i=0;i<count;++i){auto* p=bw+(i/n)*stride;inline_host[i].weight=p;inline_host[i].row_scales=reinterpret_cast<float*>(p+codes+sfs);}
            ck(cudaMemcpy(bjobs,inline_host.data(),count*sizeof(host[0]),cudaMemcpyHostToDevice));
            ck(cudaMemcpy(groups,hg.data(),experts*sizeof(hg[0]),cudaMemcpyHostToDevice));
            std::vector<float> exact_separate[3],exact_blocked[3];
            for(int mode:{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}){
            auto candidate=[&](){
                dim3 grid((rows+63)/64,experts,(n+7)/8);
                if(mode==15){grid=dim3((rows+127)/128,(n+15)/16,experts);project_m32<false,2,true,true><<<grid,128,0,cudaStreamPerThread>>>(bjobs,groups,bx,rows,inner);}
                else if(mode==14){grid=dim3((rows+127)/128,(n+15)/16,experts);project_m32<false,0,true,true><<<grid,128,0,cudaStreamPerThread>>>(fjobs,groups,bx,rows,inner);}
                else if(mode==13){grid.x=(rows+127)/128;grid.z=(n+15)/16;project_m32<false,2,true><<<grid,128,0,cudaStreamPerThread>>>(bjobs,groups,bx,rows,inner);}
                else if(mode==12){grid.x=(rows+127)/128;grid.z=(n+15)/16;project_m32<false,0,true><<<grid,128,0,cudaStreamPerThread>>>(fjobs,groups,bx,rows,inner);}
                else if(mode==11){grid.x=(rows+127)/128;grid.z=(n+15)/16;project_m32<true,2><<<grid,128,0,cudaStreamPerThread>>>(bjobs,groups,bx,rows,inner);}
                else if(mode==10){grid.x=(rows+127)/128;grid.z=(n+15)/16;project_m32<false,2><<<grid,128,0,cudaStreamPerThread>>>(bjobs,groups,bx,rows,inner);}
                else if(mode==9){grid.x=(rows+127)/128;grid.z=(n+15)/16;project_m32<true,true><<<grid,128,0,cudaStreamPerThread>>>(ijobs,groups,bx,rows,inner);}
                else if(mode==8){grid.x=(rows+127)/128;grid.z=(n+15)/16;project_m32<false,true><<<grid,128,0,cudaStreamPerThread>>>(ijobs,groups,bx,rows,inner);}
                else if(mode==7){grid.x=(rows+127)/128;grid.z=(n+15)/16;project_m32<false><<<grid,128,0,cudaStreamPerThread>>>(fjobs,groups,bx,rows,inner);}
                else if(mode==6){grid.x=(rows+127)/128;grid.z=(n+15)/16;project_m32<true><<<grid,128,0,cudaStreamPerThread>>>(fjobs,groups,bx,rows,inner);}
                else if(mode==5){grid.z=(n+15)/16;project<true,16,false,true><<<grid,128,0,cudaStreamPerThread>>>(fjobs,groups,bx,rows,inner);}
                else if(mode==4){grid.z=(n+15)/16;project<true,16,true,true><<<grid,128,0,cudaStreamPerThread>>>(fjobs,groups,bx,rows,inner);}
                else if(mode==3){grid.z=(n+15)/16;project<true,16,true><<<grid,128,0,cudaStreamPerThread>>>(fjobs,groups,bx,rows,inner);}
                else if(mode==2){grid.z=(n+15)/16;project<true,16><<<grid,128,0,cudaStreamPerThread>>>(fjobs,groups,bx,rows,inner);}
                else if(mode==1)project<true><<<grid,128,0,cudaStreamPerThread>>>(fjobs,groups,bx,rows,inner);
                else project<false><<<grid,128,0,cudaStreamPerThread>>>(jobs,groups,bx,rows,inner);
                ck(cudaGetLastError());};
            auto native=[&](){ck(mollm_cuda::launch_nvfp4_mma_grouped(jobs,count,plan,rows,inner,0,cudaStreamPerThread,packed,scales,1,true));};
            std::vector<float> hx(size_t(count)*inner),hy(size_t(count)*rows+8);
            std::vector<__nv_bfloat16> hb(hx.size());
            cudaGraph_t check_graph;cudaGraphExec_t check_exec;
            ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));candidate();
            ck(cudaStreamEndCapture(cudaStreamPerThread,&check_graph));
            ck(cudaGraphInstantiateWithFlags(&check_exec,check_graph,0));
            for(int variant=0;variant<3;++variant){
                for(size_t i=0;i<hx.size();++i){hb[i]=__float2bfloat16(variant==2?0.f:(int((i*13+variant*11)%101)-50)/37.f);hx[i]=__bfloat162float(hb[i]);}
                ck(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
                ck(cudaMemcpy(bx,hb.data(),hb.size()*2,cudaMemcpyHostToDevice));
                ck(cudaMemset(y,0xff,hy.size()*4));ck(cudaMemset(y+size_t(count)*rows,0x5a,32));
                ck(cudaGraphLaunch(check_exec,cudaStreamPerThread));
                ck(cudaStreamSynchronize(cudaStreamPerThread));ck(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));
                if(mode==7)exact_separate[variant].assign(hy.begin(),hy.begin()+size_t(count)*rows);
                if(mode==10)exact_blocked[variant].assign(hy.begin(),hy.begin()+size_t(count)*rows);
                if(mode>=12){const auto& reference=(mode==12||mode==14)?exact_separate[variant]:exact_blocked[variant];
                    if(reference.size()!=size_t(count)*rows||std::memcmp(reference.data(),hy.data(),reference.size()*4)){
                        std::fprintf(stderr,"lookup full-output byte comparison failed mode%d variant%d\n",mode,variant);return 1;}
                }
                for(size_t i=0;i<size_t(count)*rows;++i)if(!std::isfinite(hy[i])){std::fprintf(stderr,"unwritten/nonfinite %zu\n",i);return 1;}
                for(int b=0;b<32;++b)if(reinterpret_cast<uint8_t*>(hy.data()+size_t(count)*rows)[b]!=0x5a)return 1;
                size_t samples=small?size_t(count)*rows:256;
                for(size_t s=0;s<samples;++s){size_t idx=small?s:(s*7919+variant*97)%(size_t(count)*rows);
                    int i=idx/rows,r=idx%rows,e=i/n;auto* p=payload.data()+e*stride;double sum=0,absum=0;
                    for(int k=0;k<inner;++k){uint8_t code=p[offset(r,k,inner,pair)];
                        float decoded=f4(k&1?code>>4:code&15)*f8(p[codes+size_t(r)*(inner/16)+k/16]);
                        double term=double(__bfloat162float(__float2bfloat16(decoded)))*hx[size_t(i)*inner+k];sum+=term;absum+=std::abs(term);
                    }
                    float global;std::memcpy(&global,p+codes+sfs+r*4,4);double factor=double(global)*host[i].output_scale;
                    double error=std::abs(hy[idx]-sum*factor),bound=std::max(1e-7,absum*factor*3e-4);
                    worst=std::max(worst,error/bound);++checks;
                    if(error>bound){std::fprintf(stderr,"oracle failed pair%d n%d index%zu got%g ref%g bound%g\n",pair,n,idx,hy[idx],sum*factor,bound);return 1;}
                }
            }
            ck(cudaGraphExecDestroy(check_exec));ck(cudaGraphDestroy(check_graph));
            if(!small){
                for(size_t i=0;i<hx.size();++i){hb[i]=__float2bfloat16((int((i*13)%101)-50)/37.f);hx[i]=__bfloat162float(hb[i]);}
                ck(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
                ck(cudaMemcpy(bx,hb.data(),hb.size()*2,cudaMemcpyHostToDevice));
                // Native includes device grouping/FP4 quantization, candidate uses supplied
                // groups, fixture-prepacked weights (modes 1..15), and BF16 inputs.
                // Both publish FP32, unlike the retained backend's BF16 down output.
                // This is an optimistic bound, NOT a fair backend win.
                for(int rep=0;rep<3;++rep){float a,b;if(rep&1){b=timing(candidate);a=timing(native);}else{a=timing(native);b=timing(candidate);}
                    std::printf("pair=%d mode=%d routes/expert=%d native_A4_plan_quant_us=%.3f W4A16_preplanned_us=%.3f ratio=%.4f\n",pair,mode,n,a,b,b/a);
                }
            }
            }
        }
    }
    std::printf("PASS independent BF16-weight/BF16-input double oracle checks=%zu worst_bound_fraction=%.6g; full finite/tail guards; no model-quality claim\n",checks,worst);
    std::puts("PASS lookup projections byte-identical to respective arithmetic W4A16 modes for all three captured input variants");
    for(void* p:{(void*)w,(void*)fw,(void*)fjobs,(void*)iw,(void*)ijobs,(void*)x,(void*)bx,(void*)y,(void*)jobs,(void*)groups,(void*)plan,(void*)packed,(void*)scales})ck(cudaFree(p));
    return 0;
}
#endif
