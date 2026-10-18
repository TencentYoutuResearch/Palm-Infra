// Bounded diagnostic: BF16 physical down outputs, unchanged rounding boundary.
#include "kernels/cuda/moe.h"
#include "runtime/cuda_host_memory_budget.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static void check(cudaError_t s){if(s!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(s));std::exit(1);}}
template<class T>struct Buffer{T* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){cudaFree(p);}};
template<class T>static void copy(T* p,const std::vector<T>& v){check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>static std::vector<T> read(T* p,size_t n){std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
static uint16_t code(float x){uint32_t b;std::memcpy(&b,&x,4);if((b&0x7f800000u)!=0x7f800000u)b+=0x7fffu+((b>>16)&1u);return b>>16;}
static float from_bits(uint32_t b){float x;std::memcpy(&x,&b,4);return x;}
template<class T>static void guard(const std::vector<T>& v,size_t n){const auto* p=reinterpret_cast<const uint8_t*>(v.data()+n);for(size_t i=0;i<8*sizeof(T);++i)if(p[i]!=0x5a)std::exit(2);}
static double time_graph(cudaGraphExec_t graph){
    check(cudaGraphLaunch(graph,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
    cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a,cudaStreamPerThread));
    for(int i=0;i<32;++i)check(cudaGraphLaunch(graph,cudaStreamPerThread));
    check(cudaEventRecord(b,cudaStreamPerThread));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));check(cudaEventDestroy(a));check(cudaEventDestroy(b));return ms*1000/32;
}
static void combine_edges(){
    constexpr int rows=2,h=2560,k=10;constexpr size_t n=rows*h,r=n*k;
    Buffer<float> routed(r),shared(n),gate(rows),a(n+8),b(n+8);Buffer<uint16_t> packed(r);
    const uint32_t bits[]={0,0x80000000,1,0x80000001,0x3f808000,0x3f818000,0xbf808000,0xbf818000,
        0x007fffff,0x7f800000,0xff800000,0x7fc01234,0x7f800001,0xffffffff,0x3f800000};
    std::vector<float> x(r),s(n),g{80.f,-80.f};std::vector<uint16_t> z(r);
    for(size_t i=0;i<r;++i){x[i]=from_bits(bits[i%15]);z[i]=code(x[i]);}
    for(size_t i=0;i<n;++i)s[i]=from_bits(bits[(i+7)%15]);
    copy(routed.p,x);copy(packed.p,z);copy(shared.p,s);copy(gate.p,g);
    check(cudaMemset(a.p,0x5a,(n+8)*4));check(cudaMemset(b.p,0x5a,(n+8)*4));
    mollm_cuda::launch_moe_combine(routed.p,shared.p,gate.p,a.p,rows,h,k,cudaStreamPerThread);check(cudaGetLastError());
    check(mollm_cuda::launch_moe_combine_bf16(packed.p,shared.p,gate.p,b.p,rows,h,k,cudaStreamPerThread));
    const auto av=read(a.p,n+8),bv=read(b.p,n+8);guard(av,n);guard(bv,n);
    if(std::memcmp(av.data(),bv.data(),av.size()*4))std::exit(3);
    int rejected=0;
    auto reject=[&](cudaError_t e){if(e!=cudaErrorInvalidValue)std::exit(4);++rejected;};
    reject(mollm_cuda::launch_moe_combine_bf16(nullptr,shared.p,gate.p,b.p,rows,h,k));
    reject(mollm_cuda::launch_moe_combine_bf16(packed.p,nullptr,gate.p,b.p,rows,h,k));
    reject(mollm_cuda::launch_moe_combine_bf16(packed.p,shared.p,nullptr,b.p,rows,h,k));
    reject(mollm_cuda::launch_moe_combine_bf16(packed.p,shared.p,gate.p,nullptr,rows,h,k));
    for(int bad:{0,-1,4097})reject(mollm_cuda::launch_moe_combine_bf16(packed.p,shared.p,gate.p,b.p,bad,h,k));
    reject(mollm_cuda::launch_moe_combine_bf16(packed.p,shared.p,gate.p,b.p,rows,640,k));
    for(int bad:{0,-1,33})reject(mollm_cuda::launch_moe_combine_bf16(packed.p,shared.p,gate.p,b.p,rows,h,bad));
    reject(mollm_cuda::launch_moe_combine_bf16(packed.p,shared.p,gate.p,shared.p,rows,h,k));
    reject(mollm_cuda::launch_moe_combine_bf16(reinterpret_cast<uint16_t*>(UINTPTR_MAX-1),shared.p,gate.p,b.p,rows,h,k));
    const auto after=read(b.p,n+8);
    if(std::memcmp(after.data(),bv.data(),bv.size()*4))std::exit(4);
    std::printf("combine_edges full_bytes=true signed_zero_subnormal_ties_inf_nan=true rejected=%d\n",rejected);
}
static void route_pointers(){
    constexpr int rows=17,experts=512,top=10,h=2560,inner=640,count=rows*top;
    Buffer<float> logits(rows*experts),input(rows*h),gu(count*inner*2),act(count*inner),down(count*h),weights(count);
    Buffer<mollm_cuda::Nvfp4Projection> tables(2*experts),gate(count),plain(count),packed(count);
    std::vector<float> scores(rows*experts);
    for(size_t i=0;i<scores.size();++i)scores[i]=(int(i*73%997)-498)/97.f;
    copy(logits.p,scores);
    std::vector<mollm_cuda::Nvfp4Projection> t(2*experts);
    for(int i=0;i<2*experts;++i)t[i]={reinterpret_cast<const uint8_t*>(uintptr_t(4096+i*256)),
        reinterpret_cast<const uint8_t*>(uintptr_t(8192+i*256)),nullptr,nullptr,nullptr,true,false,false,.75f,1.f,1.f};
    copy(tables.p,t);int checked=0;
    setenv("MOLLM_CUDA_REGISTER_ROUTES","1",1);
    for(int selector:{-1,0,1,2})for(int score:{0,1,2})for(bool down_bf16:{false,true})for(bool gate_bf16:{false,true}){
        mollm_cuda::launch_moe_resident_routes(logits.p,rows,experts,top,score,true,1.f,tables.p,
            input.p,gu.p,act.p,down.p,h,inner,gate.p,plain.p,weights.p,cudaStreamPerThread,selector,false);
        const auto g=read(gate.p,count),p=read(plain.p,count);
        const auto w=read(weights.p,count);
        mollm_cuda::launch_moe_resident_routes(logits.p,rows,experts,top,score,true,1.f,tables.p,
            input.p,gu.p,act.p,down.p,h,inner,gate.p,packed.p,weights.p,cudaStreamPerThread,selector,down_bf16,gate_bf16);
        const auto g2=read(gate.p,count),p2=read(packed.p,count);
        const auto w2=read(weights.p,count);
        if(std::memcmp(w.data(),w2.data(),count*4))std::exit(7);
        for(int j=0;j<count;++j){auto expected=p[j];
            if(expected.output!=down.p+size_t(j)*h)std::exit(7);
            if(down_bf16)expected.output=reinterpret_cast<float*>(reinterpret_cast<uint16_t*>(down.p)+size_t(j)*h);
            auto expected_gate=g[j];
            if(expected_gate.output!=gu.p+size_t(j)*inner*2)std::exit(7);
            if(gate_bf16)expected_gate.output=reinterpret_cast<float*>(reinterpret_cast<uint16_t*>(gu.p)+size_t(j)*inner*2);
            // Compare fields, not unspecified structure padding bytes.
            auto equal=[](const auto& a,const auto& b){return a.weight==b.weight&&a.block_scales==b.block_scales&&
                a.row_scales==b.row_scales&&a.input==b.input&&a.output==b.output&&a.pair_layout==b.pair_layout&&
                a.mma_layout==b.mma_layout&&a.input_global==b.input_global&&a.input_global_second==b.input_global_second&&a.output_scale==b.output_scale;};
            if(!equal(expected,p2[j])||!equal(expected_gate,g2[j]))std::exit(7);
        }
        ++checked;
    }
    std::printf("route_bf16_pointer_cases=%d only_output_stride_changed=true\n",checked);
}
static void run(const std::vector<int>& runs,bool pair,bool timing){
    constexpr int h=2560,inner=640,sets=2,top=10;
    const int experts=runs.size(),count=std::accumulate(runs.begin(),runs.end(),0),rows=count/top;
    const size_t values=size_t(count)*h,out=size_t(rows)*h;
    constexpr size_t wbytes=size_t(h)*inner/2,sbytes=size_t(h)*inner/16,stride=wbytes+sbytes+h*4;
    Buffer<uint8_t> weights(size_t(sets)*experts*stride),packed(size_t(count)*inner/2),scales(size_t(count)*inner/16);
    Buffer<float> input(size_t(count)*inner),fp(values+8),shared(out),gate(rows),a(out+8),b(out+8);
    Buffer<uint16_t> bf(values+8);Buffer<mollm_cuda::Nvfp4Projection> jobs(size_t(sets)*count*2);
    Buffer<mollm_cuda::Nvfp4MmaBatch> plan(count+1);
    std::vector<uint8_t> payload(size_t(sets)*experts*stride);
    for(int expert=0;expert<sets*experts;++expert){auto* p=payload.data()+size_t(expert)*stride;
        for(int row=0;row<h;++row){const float global=(1+(row+expert)%7)/64.f;std::memcpy(p+wbytes+sbytes+row*4,&global,4);
            for(int group=0;group<40;++group){const uint8_t codes[]={0,1,7,8,48,56,64,126};p[wbytes+size_t(row)*40+group]=codes[(row+group+expert)%8];
                for(int byte=0;byte<8;++byte){const size_t offset=pair?size_t(row/4)*1280+(row%4)*8+group*32+byte:size_t(row)*320+group*8+byte;p[offset]=(row*29+group*17+byte*13+expert*7)&255;}
            }
        }
    }
    copy(weights.p,payload);payload.clear();payload.shrink_to_fit();
    std::vector<mollm_cuda::Nvfp4Projection> descriptors(size_t(sets)*count*2);
    int j=0;for(int expert=0;expert<experts;++expert)for(int token=0;token<runs[expert];++token,++j){
        const size_t route=count==120?(size_t(j)*7)%count:size_t(j); // Explicit nontrivial output permutation in the small fixture.
        for(int set=0;set<sets;++set){const auto* p=weights.p+size_t(set*experts+expert)*stride;
            auto d=mollm_cuda::Nvfp4Projection{p,p+wbytes,reinterpret_cast<const float*>(p+wbytes+sbytes),input.p+size_t(j)*inner,fp.p+route*h,pair,false,false,.75f+expert*.001f,1.f,(j+1)/float(count)};
            descriptors[set*count+j]=d;d.output=reinterpret_cast<float*>(bf.p+route*h);descriptors[(sets+set)*count+j]=d;
        }
    }
    copy(jobs.p,descriptors);
    auto producer=[&](bool compressed,bool prepared,int set,cudaStream_t stream){return mollm_cuda::launch_nvfp4_mma_grouped(jobs.p+(set+(compressed?sets:0))*count,count,plan.p,h,inner,0,stream,packed.p,scales.p,1,true,false,nullptr,0,8,false,prepared,false,compressed);};
    check(cudaMemset(fp.p,0x5a,(values+8)*4));check(cudaMemset(bf.p,0x5a,(values+8)*2));check(cudaMemset(plan.p,0x5a,(count+1)*sizeof(*plan.p)));
    int rejects=0;for(int id=0;id<10;++id){
        const int nr=id==0?1280:h,nk=id==1?2560:inner,split=id==2?64:0,min=id==3?2:1,width=id==4?16:8;
        const auto status=mollm_cuda::launch_nvfp4_mma_grouped(jobs.p+sets*count,count,plan.p,nr,nk,split,cudaStreamPerThread,packed.p,scales.p,min,id!=5,id==6,id==7?plan.p:nullptr,0,width,id==8,false,id==9,true);
        if(status!=cudaErrorInvalidValue)std::exit(4);++rejects;
    }
    const auto sentinel=read(bf.p,values+8);for(auto v:sentinel)if(v!=0x5a5a)std::exit(4);
    const auto plan_sentinel=read(reinterpret_cast<uint8_t*>(plan.p),(count+1)*sizeof(*plan.p));for(auto v:plan_sentinel)if(v!=0x5a)std::exit(4);
    auto chain=[&](bool compressed,bool prepared,int set){check(producer(compressed,prepared,set,cudaStreamPerThread));
        if(compressed)check(mollm_cuda::launch_moe_combine_bf16(bf.p,shared.p,gate.p,b.p,rows,h,top,cudaStreamPerThread));
        else{mollm_cuda::launch_moe_combine(fp.p,shared.p,gate.p,a.p,rows,h,top,cudaStreamPerThread);check(cudaGetLastError());}
    };
    int checked=0;
    for(int set=0;set<sets;++set)for(bool prepared:{false,true}){
        cudaGraph_t graph;cudaGraphExec_t replay;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));chain(true,prepared,set);
        check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&replay,graph,0));check(cudaGraphDestroy(graph));
        for(int variant=0;variant<4;++variant){
            std::vector<float> x(size_t(count)*inner),s(out),g(rows);
            for(size_t i=0;i<x.size();++i)x[i]=variant==2?0.f:variant==3?-0.f:(int((i*13+variant*11+set)%101)-50)/37.f;
            for(size_t i=0;i<s.size();++i)s[i]=(int((i+variant)%31)-15)/7.f;
            for(int i=0;i<rows;++i)g[i]=(i%9-4)*4.f;
            copy(input.p,x);copy(shared.p,s);copy(gate.p,g);
            check(cudaMemset(fp.p,0x5a,(values+8)*4));check(cudaMemset(bf.p,0x5a,(values+8)*2));check(cudaMemset(a.p,0x5a,(out+8)*4));check(cudaMemset(b.p,0x5a,(out+8)*4));
            chain(false,false,set);check(cudaGraphLaunch(replay,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            const auto f=read(fp.p,values+8);const auto q=read(bf.p,values+8);guard(f,values);guard(q,values);
            for(size_t i=0;i<values;++i)if(code(f[i])!=q[i]){std::fprintf(stderr,"down BF16 code mismatch pair=%d prepared=%d set=%d variant=%d index=%zu\n",pair,prepared,set,variant,i);std::exit(5);}
            const auto av=read(a.p,out+8),bv=read(b.p,out+8);guard(av,out);guard(bv,out);
            if(std::memcmp(av.data(),bv.data(),av.size()*4))std::exit(6);++checked;
        }
        check(cudaGraphExecDestroy(replay));
    }
    std::printf("down_bf16 pair=%d jobs=%d captured_replays=%d rejection_count=%d rounded_down_codes_and_full_combine_bytes=true guards=true\n",pair,count,checked,rejects);std::fflush(stdout);
    if(timing){ // Realistic down consumes prequantized inputs, like fused activation preparation.
        std::vector<float> x(size_t(count)*inner);for(size_t i=0;i<x.size();++i)x[i]=(int(i%101)-50)/37.f;copy(input.p,x);
        cudaGraphExec_t graphs[2];for(int mode=0;mode<2;++mode){cudaGraph_t graph;
            check(producer(false,false,0,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));for(int set=0;set<sets;++set)chain(mode,true,set);
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&graphs[mode],graph,0));check(cudaGraphDestroy(graph));
        }
        for(int repeat=0;repeat<7;++repeat){double us[2];for(int order=0;order<2;++order){const int mode=order^(repeat&1);us[mode]=time_graph(graphs[mode])/sets;}
            std::printf("down_bf16_timing pair=%d jobs=%d repeat=%d fp32_us=%.3f bf16_us=%.3f scope=prepared_down_plus_combine_synthetic_not_model\n",pair,count,repeat,us[0],us[1]);std::fflush(stdout);
        }
        for(auto graph:graphs)check(cudaGraphExecDestroy(graph));
    }
}
int main(int argc,char** argv){
    const bool only=argc==2&&!std::strcmp(argv[1],"--check-only");std::vector<int> runs{1,7,8,16,17,32,39};
    if(argc>1&&!only){
        if(argc!=5||std::strcmp(argv[1],"--rows")||std::strcmp(argv[3],"--counts"))return 64;
        if(std::strcmp(argv[2],"1024")&&std::strcmp(argv[2],"719")&&std::strcmp(argv[2],"1743"))return 64;
        runs.clear();const char* p=argv[4];while(*p){const char* begin=p;while(*p>='0'&&*p<='9')++p;if(p==begin||p-begin>5||(*p&&*p!=','))return 64;
            const long n=std::strtol(begin,nullptr,10);if(n<0||n>65535)return 64;runs.push_back(n);if(!*p)break;++p;if(!*p)return 64;}
        if(runs.size()!=512||std::accumulate(runs.begin(),runs.end(),0)!=std::atoi(argv[2])*10)return 64;
    }
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));if(prop.major!=12||prop.minor>1)return 77;
    setenv("MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS","4",1);setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);
    if(!mollm_memory::cuda_host_allocation_allowed(size_t(3)<<30,"bounded BF16 down output diagnostic"))return 1;
    combine_edges();route_pointers();for(bool pair:{true,false})run(runs,pair,!only);
}
