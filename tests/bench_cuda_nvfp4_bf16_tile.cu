// Standalone canonical FP4 -> transient BF16 tile diagnostic. No backend selector.
#include "kernels/cuda/moe.h"
#include "runtime/cuda_host_memory_budget.h"
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int rows=1280,inner=2560,split=640,experts=32;
void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
template<class T>T* allocate(size_t n){T* p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));return p;}
__device__ uint32_t scaled_pair(uint8_t code,uint8_t scale) {
    const float2 v=__half22float2(__nv_cvt_fp4x2_to_halfraw2(code,__NV_E2M1));
    const float s=__half2float(__nv_cvt_fp8_to_halfraw(scale,__NV_E4M3));
    const __nv_bfloat162_raw b=__floats2bfloat162_rn(v.x*s,v.y*s);
    return uint32_t(b.x)|(uint32_t(b.y)<<16);
}
template<int Stage>
__global__ void bf16_tile(const mollm_cuda::Nvfp4Projection* jobs,const mollm_cuda::Nvfp4MmaBatch* groups,
    const uint8_t* packed,const uint8_t* scales,int count) {
    const auto group=groups[blockIdx.y];
    const auto job=jobs[group.first];
    const int base_row=blockIdx.x*128,role=base_row>=split;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int mbase=(warp/2)*32,nbase=(warp%2)*32,quad=lane/4,pair=lane%4;
    __shared__ uint32_t a[128][Stage/2],b[64][Stage/2];
    float accum[2][4][4]={};
    for(int k=0;k<inner;k+=Stage) {
        for(int i=threadIdx.x;i<128*(Stage/2);i+=blockDim.x) {
            const int local=i/(Stage/2),p=i%(Stage/2),row=base_row+local;
            const size_t offset=job.pair_layout ? (size_t(row/4)*(inner/16)+k/16+p/8)*32+(row%4)*8+p%8
                : size_t(row)*(inner/2)+k/2+p;
            a[local][p]=scaled_pair(job.weight[offset],job.block_scales[size_t(row)*(inner/16)+k/16+p/8]);
        }
        for(int i=threadIdx.x;i<64*(Stage/2);i+=blockDim.x) {
            const int column=i/(Stage/2),p=i%(Stage/2),index=group.first+column+role*count;
            b[column][p]=column<group.size ? scaled_pair(packed[size_t(index)*(inner/2)+k/2+p],
                scales[size_t(index)*(inner/16)+k/16+p/8]) : 0;
        }
        __syncthreads();
#pragma unroll
        for(int step=0;step<Stage/16;++step) {
#pragma unroll
        for(int m=0;m<2;++m) {
            const int r=mbase+m*16+quad;
            const int p=step*8+pair;
            const uint32_t a0=a[r][p],a1=a[r+8][p],a2=a[r][p+4],a3=a[r+8][p+4];
#pragma unroll
            for(int n=0;n<4;++n) {
                const int c=nbase+n*8+quad;
                const uint32_t b0=b[c][p],b1=b[c][p+4];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(accum[m][n][0]),"+f"(accum[m][n][1]),"+f"(accum[m][n][2]),"+f"(accum[m][n][3])
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        }
        __syncthreads();
    }
#pragma unroll
    for(int m=0;m<2;++m)for(int n=0;n<4;++n)for(int v=0;v<4;++v) {
        const int column=nbase+n*8+pair*2+(v&1),row=base_row+mbase+m*16+quad+(v>=2?8:0);
        if(column<group.size) {
            const auto out=jobs[group.first+column];
            const float global=role?out.input_global_second:out.input_global;
            out.output[row]=((accum[m][n][v]*global)*job.row_scales[row])*out.output_scale;
        }
    }
}
float fp4(unsigned code){const float v[8]={0,.5,1,1.5,2,3,4,6};return code&8?-v[code&7]:v[code&7];}
float fp8(unsigned code){const int e=code>>3,m=code&7;return e?std::ldexp(1.f+m/8.f,e-7):std::ldexp(float(m),-9);}
}

int main(int argc,char** argv) {
    const bool checks_only=argc==2&&std::strcmp(argv[1],"--checks-only")==0;
    if(argc>1&&!checks_only){std::fprintf(stderr,"usage: %s [--checks-only]\n",argv[0]);return 2;}
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));if(prop.major!=12||prop.minor!=1)return 77;
    setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","4",1);
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);
    constexpr size_t code_bytes=size_t(rows)*inner/2,scale_bytes=size_t(rows)*inner/16;
    constexpr size_t stride=code_bytes+scale_bytes+rows*sizeof(float),max_count=experts*64;
    constexpr size_t allocation_budget=experts*stride+max_count*(inner+rows)*sizeof(float)+
        max_count*inner+max_count*inner/8+max_count*sizeof(mollm_cuda::Nvfp4Projection)+
        (max_count+experts)*sizeof(mollm_cuda::Nvfp4MmaBatch)+32;
    if(!mollm_memory::cuda_host_allocation_allowed(allocation_budget,"bounded BF16 tile diagnostic"))return 2;
    auto* weight=allocate<uint8_t>(experts*stride);auto* input=allocate<float>(max_count*inner);
    auto* output=allocate<float>(max_count*rows+8);auto* jobs=allocate<mollm_cuda::Nvfp4Projection>(max_count);
    auto* plan=allocate<mollm_cuda::Nvfp4MmaBatch>(max_count);auto* groups=allocate<mollm_cuda::Nvfp4MmaBatch>(experts);
    auto* packed=allocate<uint8_t>(max_count*inner);auto* scales=allocate<uint8_t>(max_count*inner/8);
    std::vector<uint8_t> payload(experts*stride);
    int checks=0,replays=0;size_t strict_failures=0,cpu_strict_failures=0,ref_cpu_strict_failures=0;
    double worst=0,cpu_worst=0,ref_cpu_worst=0;
    for(bool pair_layout:{false,true}) {
        for(int e=0;e<experts;++e) {
            auto* p=payload.data()+e*stride;auto* global=reinterpret_cast<float*>(p+code_bytes+scale_bytes);
            for(int r=0;r<rows;++r) {
                global[r]=(1+(r+e)%7)/64.f;
                for(int g=0;g<inner/16;++g) {
                    const uint8_t sc[]={0,1,8,24,48,56,64,126};p[code_bytes+size_t(r)*160+g]=sc[(r+g+e)%8];
                    for(int b=0;b<8;++b) {
                        const size_t i=pair_layout?(size_t(r/4)*160+g)*32+(r%4)*8+b:size_t(r)*1280+g*8+b;
                        p[i]=(r*29+g*17+b*13+e*7)&255;
                    }
                }
            }
        }
        check(cudaMemcpy(weight,payload.data(),payload.size(),cudaMemcpyHostToDevice));
        for(int n:{1,7,8,16,17,31,32,33,63,64}) {
            const int count=n*experts;
            std::vector<mollm_cuda::Nvfp4Projection> descriptors(count);
            std::vector<mollm_cuda::Nvfp4MmaBatch> blocks(experts);
            for(int e=0;e<experts;++e) {
                blocks[e]={e*n,n};const auto* p=weight+e*stride;
                for(int j=0;j<n;++j) {const int i=e*n+j;
                    descriptors[i]={p,p+code_bytes,reinterpret_cast<const float*>(p+code_bytes+scale_bytes),
                        input+size_t(i)*inner,output+size_t(i)*rows,pair_layout,false,false,.75f+e*.03f,1.25f+e*.03f,(j+1)/float(n)};
                }
            }
            check(cudaMemcpy(jobs,descriptors.data(),descriptors.size()*sizeof(descriptors[0]),cudaMemcpyHostToDevice));
            check(cudaMemcpy(groups,blocks.data(),blocks.size()*sizeof(blocks[0]),cudaMemcpyHostToDevice));
            auto original=[&](){check(mollm_cuda::launch_nvfp4_mma_grouped(jobs,count,plan,rows,inner,split,cudaStreamPerThread,
                packed,scales,1,true,false,nullptr,0,32,false,false,true));};
            auto candidate=[&](){check(mollm_cuda::launch_nvfp4_quantize_probe(jobs,count,inner,2,packed,scales,nullptr,1,0,cudaStreamPerThread));
                bf16_tile<16><<<dim3(rows/128,experts),256,0,cudaStreamPerThread>>>(jobs,groups,packed,scales,count);check(cudaGetLastError());};
            std::vector<float> x(size_t(count)*inner),ref(size_t(count)*rows+8),actual(ref.size());
            std::vector<uint8_t> qp(size_t(count)*inner),qs(size_t(count)*inner/8);
            cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
            check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));candidate();
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&executable,graph,0));
            for(int variant=0;variant<3;++variant) {
                for(size_t i=0;i<x.size();++i)x[i]=variant==2?0.f:(int((i*13+variant*11)%101)-50)/37.f;
                check(cudaMemcpy(input,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
                check(cudaMemset(output,0xff,ref.size()*sizeof(float)));check(cudaMemset(output+size_t(count)*rows,0x5a,32));
                original();check(cudaStreamSynchronize(cudaStreamPerThread));
                check(cudaMemcpy(ref.data(),output,ref.size()*sizeof(float),cudaMemcpyDeviceToHost));
                check(cudaMemset(output,0xff,ref.size()*sizeof(float)));check(cudaMemset(output+size_t(count)*rows,0x5a,32));
                check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
                check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
                const auto* ref_guard=reinterpret_cast<const uint8_t*>(ref.data()+size_t(count)*rows);
                const auto* actual_guard=reinterpret_cast<const uint8_t*>(actual.data()+size_t(count)*rows);
                for(int i=0;i<32;++i)if(ref_guard[i]!=0x5a||actual_guard[i]!=0x5a)return 1;
                for(size_t i=0;i<size_t(count)*rows;++i) {
                    const double error=std::abs(double(actual[i])-ref[i])/(1+std::abs(double(ref[i])));worst=std::max(worst,error);
                    if(!std::isfinite(actual[i])||!std::isfinite(ref[i])) {
                        std::fprintf(stderr,"tile mismatch pair=%d n=%d variant=%d index=%zu ref=%g got=%g normalized_error=%g\n",pair_layout,n,variant,i,ref[i],actual[i],error);return 1;
                    }
                    if(error>1e-4)++strict_failures;
                }
                check(cudaMemcpy(qp.data(),packed,qp.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(qs.data(),scales,qs.size(),cudaMemcpyDeviceToHost));
                for(int e:{0,experts-1})for(int j:{0,n/2,n-1})for(int r:{0,7,16,31,64,639,640,1279}) {
                    const int index=e*n+j,source=index+(r>=split?count:0);double sum=0,sum_abs=0;
                    for(int k=0;k<inner;++k) {
                        const int g=k/16,b=(k%16)/2;
                        const size_t w=pair_layout?(size_t(r/4)*160+g)*32+(r%4)*8+b:size_t(r)*1280+g*8+b;
                        const uint8_t wc=payload[e*stride+w],bc=qp[size_t(source)*1280+k/2];
                        const float av=fp4((wc>>((k&1)*4))&15)*fp8(payload[e*stride+code_bytes+size_t(r)*160+g]);
                        const float bv=fp4((bc>>((k&1)*4))&15)*fp8(qs[size_t(source)*160+g]);
                        sum+=double(av)*bv;sum_abs+=std::abs(double(av)*bv);
                    }
                    const auto& d=descriptors[index];const float rg=(1+(r+e)%7)/64.f;
                    const float expected=((float(sum)*(r>=split?d.input_global_second:d.input_global))*rg)*d.output_scale;
                    const double error=std::abs(double(actual[size_t(index)*rows+r])-expected)/(1+std::abs(double(expected)));
                    cpu_worst=std::max(cpu_worst,error);if(error>1e-4)++cpu_strict_failures;
                    const double ref_error=std::abs(double(ref[size_t(index)*rows+r])-expected)/(1+std::abs(double(expected)));
                    ref_cpu_worst=std::max(ref_cpu_worst,ref_error);if(ref_error>1e-4)++ref_cpu_strict_failures;
                    // A diagnostic forward-error bound, not model-quality acceptance.
                    const double post=std::abs(double(r>=split?d.input_global_second:d.input_global)*rg*d.output_scale);
                    const double bound=8*0x1p-23*inner*sum_abs*post+1e-6;
                    if(std::abs(double(actual[size_t(index)*rows+r])-expected)>bound||
                       std::abs(double(ref[size_t(index)*rows+r])-expected)>bound) {
                        std::fprintf(stderr,"CPU forward-error bound exceeded error=%g bound=%g\n",error,bound);return 1;
                    }
                }
                ++checks;++replays;
            }
            if(!checks_only&&(n==33||n==64)) {
                for(size_t i=0;i<x.size();++i)x[i]=(int(i*13%101)-50)/37.f;
                check(cudaMemcpy(input,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
                auto measure=[&](bool bf16){cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a,cudaStreamPerThread));
                    for(int i=0;i<8;++i){if(bf16)candidate();else original();}check(cudaEventRecord(b,cudaStreamPerThread));check(cudaEventSynchronize(b));
                    float ms;check(cudaEventElapsedTime(&ms,a,b));check(cudaEventDestroy(a));check(cudaEventDestroy(b));return ms*1000/8;};
                original();candidate();check(cudaStreamSynchronize(cudaStreamPerThread));
                for(int repeat=0;repeat<7;++repeat) {float a,b;if(repeat&1){b=measure(true);a=measure(false);}else{a=measure(false);b=measure(true);}
                    std::printf("bf16_tile_probe pair=%d routes_per_expert=%d experts=%d repeat=%d fp4_async_us=%.3f bf16_tile_us=%.3f stage=16 scope=synthetic_group_queue_including_quantization_candidate_excludes_planning\n",pair_layout,n,experts,repeat,a,b);std::fflush(stdout);}
            }
            check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
        }
    }
    for(void* p:{static_cast<void*>(weight),static_cast<void*>(input),static_cast<void*>(output),static_cast<void*>(jobs),static_cast<void*>(plan),static_cast<void*>(groups),static_cast<void*>(packed),static_cast<void*>(scales)})check(cudaFree(p));
    std::printf("bf16_tile_checks=%d changed_input_captured_replays=%d worst_vs_fp4=%g worst_vs_cpu_double=%g ref_worst_vs_cpu_double=%g strict_failures=%zu cpu_strict_failures=%zu ref_cpu_strict_failures=%zu no_model_selector=1\n",checks,replays,worst,cpu_worst,ref_cpu_worst,strict_failures,cpu_strict_failures,ref_cpu_strict_failures);
    return strict_failures||cpu_strict_failures?1:0;
}
