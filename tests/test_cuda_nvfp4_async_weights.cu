#include "kernels/cuda/moe.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void check(cudaError_t status) {
    if(status!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(1);}
}
template<class T> T* allocate(size_t n) {
    T* p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));return p;
}
struct Args {
    const mollm_cuda::Nvfp4Projection* jobs;
    mollm_cuda::Nvfp4MmaBatch* batches;
    uint8_t* packed;uint8_t* scales;
    int count,rows=1280,inner=2560,split=640,minimum=1,width=32,hint=0;
    bool canonical=true,specialized=false,prepared=false;
    mollm_cuda::Nvfp4MmaBatch* compact=nullptr;
    bool code_kmajor=false,bf16=false,early=false,parallel=false,vector=false,singles=false;
};
cudaError_t launch(const Args& a,bool async) {
    return mollm_cuda::launch_nvfp4_mma_grouped(a.jobs,a.count,a.batches,a.rows,a.inner,
        a.split,cudaStreamPerThread,a.packed,a.scales,a.minimum,a.canonical,a.singles,
        a.compact,a.hint,a.width,a.specialized,a.prepared,async,a.bf16,a.early,a.parallel,a.vector,a.code_kmajor);
}
}

int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=12 || prop.minor>1)return 77;
    setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","4",1);
    const char* code_option=std::getenv("MOLLM_TEST_NVFP4_GROUP_CODE_KMAJOR");
    const bool code_kmajor=code_option&&!std::strcmp(code_option,"1");
    const char* scale_option=std::getenv("MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW");
    const bool scale_view=scale_option&&!std::strcmp(scale_option,"1");
    if(scale_view&&!code_kmajor)return 64;
    constexpr int rows=1280,inner=2560,sets=2;
    const int runs[]={1,7,8,16,17,32,33};
    constexpr size_t codes=size_t(rows)*inner/2,scales=size_t(rows)*inner/16;
    constexpr size_t stride=codes+scales+rows*sizeof(float);
    int count=0;for(int n:runs)count+=n;
    std::vector<uint8_t> payload(stride*7*sets);
    auto* weights=allocate<uint8_t>(payload.size());
    auto* scale_major=allocate<uint8_t>(sets*7*scales);
    auto* input=allocate<float>(size_t(count)*inner);
    auto* output=allocate<float>(size_t(count)*rows+8);
    auto* jobs=allocate<mollm_cuda::Nvfp4Projection>(sets*count);
    auto* code_jobs=allocate<mollm_cuda::Nvfp4Projection>(code_kmajor?sets*count:1);
    auto* code_view=allocate<uint8_t>(code_kmajor?sets*7*codes+16:16);
    auto* plan=allocate<mollm_cuda::Nvfp4MmaBatch>(count+1);
    auto* compact=allocate<mollm_cuda::Nvfp4MmaBatch>(count+2);
    auto* packed=allocate<uint8_t>(size_t(count)*inner+8);
    auto* quant_scales=allocate<uint8_t>(size_t(count)*inner/8+8);
    Args args{jobs,plan,packed,quant_scales,count};
    args.code_kmajor=code_kmajor;
    check(cudaMemset(plan,0x5a,(count+1)*sizeof(*plan)));
    check(cudaMemset(output,0x5a,(size_t(count)*rows+8)*sizeof(float)));
    int rejects=0;
    cudaGraph_t reject_graph=nullptr;
    if(code_kmajor)check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    for(int id=0;id<20;++id) {
        Args bad=args;
        switch(id) {
            case 0:bad.jobs=nullptr;break;
            case 1:bad.batches=nullptr;break;
            case 2:bad.count=0;break;
            case 3:bad.rows=640;break;
            case 4:bad.inner=640;break;
            case 5:bad.width=8;break;
            case 6:bad.width=16;break;
            case 7:bad.canonical=false;break;
            case 8:bad.packed=nullptr;break;
            case 9:bad.scales=nullptr;break;
            case 10:bad.compact=plan;break;
            case 11:bad.specialized=true;break;
            case 12:bad.minimum=0;break;
            case 13:bad.minimum=9;break;
            case 14:bad.split=1;break;
            case 15:bad.hint=1;break;
            case 16:bad.count=65536;break;
            case 17:bad.prepared=true;break;
            case 18:setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","2",1);break;
            case 19:setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","8",1);break;
        }
        if(launch(bad,true)!=cudaErrorInvalidValue)return 1;
        setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","4",1);++rejects;
    }
    if(code_kmajor) {
        check(cudaStreamEndCapture(cudaStreamPerThread,&reject_graph));
        size_t nodes=0;check(cudaGraphGetNodes(reject_graph,nullptr,&nodes));
        if(nodes)return 1;check(cudaGraphDestroy(reject_graph));
    }
    check(cudaDeviceSynchronize());
    std::vector<uint8_t> sentinel((count+1)*sizeof(*plan));
    check(cudaMemcpy(sentinel.data(),plan,sentinel.size(),cudaMemcpyDeviceToHost));
    if(!std::all_of(sentinel.begin(),sentinel.end(),[](uint8_t x){return x==0x5a;}))return 1;
    sentinel.resize((size_t(count)*rows+8)*sizeof(float));
    check(cudaMemcpy(sentinel.data(),output,sentinel.size(),cudaMemcpyDeviceToHost));
    if(!std::all_of(sentinel.begin(),sentinel.end(),[](uint8_t x){return x==0x5a;}))return 1;
    std::vector<float> values(size_t(count)*inner),reference(size_t(count)*rows+8),actual(reference.size());
    std::vector<mollm_cuda::Nvfp4Projection> descriptors(sets*count);
    int replays=0;
    for(bool pair:{false,true}) {
        for(int expert=0;expert<7*sets;++expert) {
            auto* base=payload.data()+expert*stride;
            auto* global=reinterpret_cast<float*>(base+codes+scales);
            for(int row=0;row<rows;++row) {
                global[row]=(1+(row+expert)%7)/64.f;
                for(int group=0;group<inner/16;++group) {
                    const uint8_t scale_codes[]={0,1,7,8,48,56,64,126};
                    base[codes+size_t(row)*160+group]=scale_codes[(row+group+expert)%8];
                    for(int byte=0;byte<8;++byte) {
                        const size_t offset=pair?size_t(row/4)*5120+(row%4)*8+group*32+byte:
                            size_t(row)*1280+group*8+byte;
                        base[offset]=(row*29+group*17+byte*13+expert*7)&255;
                    }
                }
            }
        }
        check(cudaMemcpy(weights,payload.data(),payload.size(),cudaMemcpyHostToDevice));
        std::vector<uint8_t> original_view;
        if(code_kmajor) {
            check(cudaMemset(code_view,0x5a,sets*7*codes+16));
            for(int expert=0;expert<sets*7;++expert) {
                mollm_cuda::Nvfp4Projection source{};
                source.weight=weights+expert*stride;source.pair_layout=pair;
                check(mollm_cuda::launch_repack_nvfp4_gate_codes(source,code_view+expert*codes,rows,inner,cudaStreamPerThread));
                if(scale_view) {
                    source.block_scales=weights+expert*stride+codes;
                    check(mollm_cuda::launch_repack_nvfp4_gate_scales(source,
                        scale_major+expert*scales,rows,inner,cudaStreamPerThread));
                }
            }
            check(cudaStreamSynchronize(cudaStreamPerThread));
            original_view.resize(sets*7*codes+16);
            check(cudaMemcpy(original_view.data(),code_view,original_view.size(),cudaMemcpyDeviceToHost));
        }
        for(int set=0;set<sets;++set) {
            int j=0;
            for(int expert=0;expert<7;++expert)for(int token=0;token<runs[expert];++token,++j) {
                const auto* base=weights+(set*7+expert)*stride;
                descriptors[set*count+j]={base,base+codes,reinterpret_cast<const float*>(base+codes+scales),
                    input+size_t(j)*inner,output+size_t(j)*rows,pair,false,false,.75f+expert*.03f,
                    1.25f+expert*.03f,(j+1)/float(count)};
            }
        }
        check(cudaMemcpy(jobs,descriptors.data(),descriptors.size()*sizeof(descriptors[0]),cudaMemcpyHostToDevice));
        if(code_kmajor) {
            auto views=descriptors;
            for(auto& descriptor:views) {
                const size_t expert=(descriptor.weight-weights)/stride;
                descriptor.weight=code_view+expert*codes;
            }
            check(cudaMemcpy(code_jobs,views.data(),views.size()*sizeof(views[0]),cudaMemcpyHostToDevice));
        }
        for(int consumer=0;consumer<(code_kmajor?17:1);++consumer)
        for(int set=0;set<sets;++set)for(int split:{0,640})for(int minimum=1;minimum<=8;++minimum) {
            const bool async=consumer<=3;
            if(consumer>=1&&consumer<=3&&(split!=640||minimum!=1))continue;
            args=Args{(code_kmajor?code_jobs:jobs)+set*count,plan,packed,quant_scales,count};
            args.code_kmajor=code_kmajor;args.split=split;args.minimum=minimum;
            if(consumer>=1&&consumer<=3)args.bf16=true;
            if(consumer==2)args.early=true;
            if(consumer==3){args.early=true;args.vector=true;args.parallel=true;}
            if(consumer==5||consumer==11||consumer==15)args.width=16;
            if(consumer==6||consumer==7||consumer==8||consumer==9||consumer==10||consumer==13||consumer==16)args.width=8;
            if(consumer==7||consumer==9){args.packed=nullptr;args.scales=nullptr;}
            if(consumer>=8&&consumer<=12){args.compact=compact;args.hint=1;}
            if(consumer==10)args.specialized=true;
            if(consumer==13)args.singles=true;
            if(consumer>=14)args.parallel=true;
            if(code_kmajor) {
                auto output_jobs=descriptors;
                for(size_t index=0;index<output_jobs.size();++index)
                    output_jobs[index].output=reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(output)+
                        (index%count)*rows*(args.bf16?sizeof(uint16_t):sizeof(float)));
                check(cudaMemcpy(jobs,output_jobs.data(),output_jobs.size()*sizeof(output_jobs[0]),cudaMemcpyHostToDevice));
                for(auto& descriptor:output_jobs) {
                    const size_t expert=(descriptor.weight-weights)/stride;
                    descriptor.weight=code_view+expert*codes;
                    if(scale_view) {
                        descriptor.block_scales=scale_major+expert*scales;
                        descriptor.gate_scales_kmajor=true;
                    }
                }
                check(cudaMemcpy(code_jobs,output_jobs.data(),output_jobs.size()*sizeof(output_jobs[0]),cudaMemcpyHostToDevice));
            }
            auto reference_args=args;reference_args.jobs=jobs+set*count;reference_args.code_kmajor=false;
            cudaGraph_t graph=nullptr;cudaGraphExec_t replay=nullptr;
            check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            check(launch(args,async));
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            check(cudaGraphInstantiateWithFlags(&replay,graph,0));
            for(int variant=0;variant<3;++variant) {
                for(size_t i=0;i<values.size();++i)values[i]=variant==2?0.f:
                    (int((i*13+variant*11+set)%101)-50)/37.f;
                if(variant==1)for(size_t i=0;i<values.size();i+=257)values[i]=-0.f;
                check(cudaMemcpy(input,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice));
                check(cudaMemset(output,0x5a,actual.size()*sizeof(float)));
                check(cudaMemset(packed,0x5a,size_t(count)*inner+8));
                check(cudaMemset(quant_scales,0x5a,size_t(count)*inner/8+8));
                check(launch(reference_args,code_kmajor?async:false));
                check(cudaMemcpy(reference.data(),output,reference.size()*sizeof(float),cudaMemcpyDeviceToHost));
                std::vector<uint8_t> reference_packed(size_t(count)*inner+8),reference_scales(size_t(count)*inner/8+8);
                if(code_kmajor) {
                    check(cudaMemcpy(reference_packed.data(),packed,reference_packed.size(),cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(reference_scales.data(),quant_scales,reference_scales.size(),cudaMemcpyDeviceToHost));
                }
                check(cudaMemset(output,0x5a,actual.size()*sizeof(float)));
                check(cudaMemset(packed,0x5a,size_t(count)*inner+8));
                check(cudaMemset(quant_scales,0x5a,size_t(count)*inner/8+8));
                check(cudaGraphLaunch(replay,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
                check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
                const auto* guard=reinterpret_cast<const uint8_t*>(actual.data()+size_t(count)*rows);
                if(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float)) ||
                    !std::all_of(guard,guard+32,[](uint8_t x){return x==0x5a;})) {
                    std::fprintf(stderr,"async mismatch pair=%d set=%d split=%d min=%d variant=%d consumer=%d\n",pair,set,split,minimum,variant,consumer);return 1;
                }
                if(code_kmajor) {
                    std::vector<uint8_t> actual_packed(reference_packed.size()),actual_scales(reference_scales.size());
                    check(cudaMemcpy(actual_packed.data(),packed,actual_packed.size(),cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(actual_scales.data(),quant_scales,actual_scales.size(),cudaMemcpyDeviceToHost));
                    if(actual_packed!=reference_packed||actual_scales!=reference_scales)return 2;
                    const size_t written=size_t(count)*rows*(args.bf16?sizeof(uint16_t):sizeof(float));
                    const auto* bytes=reinterpret_cast<const uint8_t*>(actual.data());
                    if(!std::all_of(bytes+written,bytes+actual.size()*sizeof(float),[](uint8_t x){return x==0x5a;})) {
                        std::fprintf(stderr,"output guard mismatch pair=%d consumer=%d split=%d min=%d\n",pair,consumer,split,minimum);return 3;
                    }
                }
                ++replays;
            }
            check(cudaGraphExecDestroy(replay));check(cudaGraphDestroy(graph));
        }
        if(code_kmajor) {
            std::vector<uint8_t> after(original_view.size()),original_weights(payload.size());
            check(cudaMemcpy(after.data(),code_view,after.size(),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(original_weights.data(),weights,original_weights.size(),cudaMemcpyDeviceToHost));
            if(after!=original_view||original_weights!=payload)return 4;
        }
    }
    for(void* p:{static_cast<void*>(weights),static_cast<void*>(input),static_cast<void*>(output),
        static_cast<void*>(jobs),static_cast<void*>(code_jobs),static_cast<void*>(code_view),static_cast<void*>(scale_major),static_cast<void*>(compact),static_cast<void*>(plan),static_cast<void*>(packed),static_cast<void*>(quant_scales)})check(cudaFree(p));
    std::printf("async_weights guarded_full_bytes_replays=%d rejected=%d code_kmajor=%d consumers=%d rejection_storage_untouched=true\n",replays,rejects,code_kmajor,code_kmajor?17:1);
}
