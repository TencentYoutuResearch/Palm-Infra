// Same-process frozen-route window comparison, not model scheduling or TPS.
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
static void require(bool ok){if(!ok)std::exit(2);}
template<class T>struct Buffer{T* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){cudaFree(p);}};
template<class T>static void copy(T* p,const std::vector<T>& v){check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>static std::vector<T> read(T* p,size_t n){std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T>static void guard(const std::vector<T>& v,size_t n){const auto* p=reinterpret_cast<const uint8_t*>(v.data()+n);for(size_t i=0;i<8*sizeof(T);++i)require(p[i]==0x5a);}
static double time_graph(cudaGraphExec_t graph){
    check(cudaGraphLaunch(graph,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
    cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a,cudaStreamPerThread));
    for(int i=0;i<32;++i)check(cudaGraphLaunch(graph,cudaStreamPerThread));
    check(cudaEventRecord(b,cudaStreamPerThread));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));
    check(cudaEventDestroy(a));check(cudaEventDestroy(b));return ms*1000/32;
}
static bool parse(const char* p,std::vector<int>& v){
    v.clear();while(*p){const char* start=p;while(*p>='0'&&*p<='9')++p;
        if(start==p||p-start>5||(*p&&*p!=','))return false;
        const long n=std::strtol(start,nullptr,10);if(n>65535)return false;v.push_back(n);
        if(!*p)break;++p;if(!*p)return false;
    }return !v.empty();
}
static bool same_route(const mollm_cuda::Nvfp4Projection& a,const mollm_cuda::Nvfp4Projection& b){
    return a.weight==b.weight&&a.block_scales==b.block_scales&&a.row_scales==b.row_scales&&
        a.input==b.input&&a.output==b.output&&a.pair_layout==b.pair_layout&&a.mma_layout==b.mma_layout&&
        a.input_global==b.input_global&&a.input_global_second==b.input_global_second&&a.output_scale==b.output_scale;
}
static void run(const std::vector<int>& ca,const std::vector<int>& cb,bool pair,bool timing){
    constexpr int h=1280,k=2560,inner=640,sets=2;
    const int na=std::accumulate(ca.begin(),ca.end(),0),nb=std::accumulate(cb.begin(),cb.end(),0);
    const int count=na+nb,experts=ca.size(),rows=na/10+nb/10;
    constexpr size_t wb=size_t(h)*k/2,sb=size_t(h)*k/16,stride=wb+sb+h*4;
    const size_t av=size_t(count)*inner,pv=size_t(count)*inner/2,sv=size_t(count)*inner/16;
    Buffer<uint8_t> weights(size_t(sets)*experts*stride),packed(size_t(count)*k+8),scales(size_t(count)*k/8+8);
    Buffer<float> input(size_t(rows)*k),act0(av+8),act1(av+8);
    Buffer<uint16_t> unused(size_t(count)*h+8);
    Buffer<uint8_t> pack0(pv+8),pack1(pv+8),scale0(sv+8),scale1(sv+8);
    Buffer<mollm_cuda::Nvfp4Projection> gate0(size_t(sets)*count),gate1(size_t(sets)*count),down0(count),down1(count);
    Buffer<mollm_cuda::Nvfp4MmaBatch> plan(count+1);
    std::vector<uint8_t> payload(size_t(sets)*experts*stride);
    for(int e=0;e<sets*experts;++e){auto* p=payload.data()+size_t(e)*stride;
        for(int row=0;row<h;++row){const float global=(1+(row+e)%7)/64.f;std::memcpy(p+wb+sb+row*4,&global,4);
            for(int group=0;group<160;++group){const uint8_t q[]={0,1,7,8,48,56,64,126};p[wb+size_t(row)*160+group]=q[(row+group+e)%8];
                for(int b=0;b<8;++b){const size_t offset=pair?size_t(row/4)*5120+(row%4)*8+group*32+b:size_t(row)*1280+group*8+b;
                    p[offset]=(row*29+group*17+b*13+e*7)&255;}
            }
        }
    }
    copy(weights.p,payload);payload.clear();payload.shrink_to_fit();
    std::vector<int> permutation;permutation.reserve(count);int ia=0,ib=0;
    std::vector<mollm_cuda::Nvfp4Projection> g0(size_t(sets)*count),g1(g0.size()),d0(count),d1(count);
    auto route=[&](int e,int r,int set){
        const auto* p=weights.p+size_t(set*experts+e)*stride;
        const int token=r<na?r%(na/10):na/10+(r-na)%(nb/10);
        return mollm_cuda::Nvfp4Projection{p,p+wb,reinterpret_cast<const float*>(p+wb+sb),input.p+size_t(token)*k,
            reinterpret_cast<float*>(unused.p+size_t(r)*h),pair,false,false,.75f+e*.001f,1.25f+e*.001f,(r%41-20)/31.f};
    };
    for(int e=0;e<experts;++e){
        for(int t=0;t<ca[e];++t)permutation.push_back(ia+t);
        for(int t=0;t<cb[e];++t)permutation.push_back(na+ib+t);
        for(int set=0;set<sets;++set){
            for(int t=0;t<ca[e];++t)g0[size_t(set)*count+ia+t]=route(e,ia+t,set);
            for(int t=0;t<cb[e];++t)g0[size_t(set)*count+na+ib+t]=route(e,na+ib+t,set);
        }
        ia+=ca[e];ib+=cb[e];
    }
    require(ia==na&&ib==nb&&permutation.size()==size_t(count));
    std::vector<int> seen(count);
    for(int i=0;i<count;++i){const int r=permutation[i];require(r>=0&&r<count&&!seen[r]++);
        for(int set=0;set<sets;++set){g1[size_t(set)*count+i]=g0[size_t(set)*count+r];require(same_route(g1[size_t(set)*count+i],g0[size_t(set)*count+r]));}
        d0[r].input=act0.p+size_t(r)*inner;
        d0[r].input_global=.9f+float((r*7)%19)*.003f;
        d1[i]=d0[r];d1[i].input=act1.p+size_t(r)*inner;
    }
    copy(gate0.p,g0);copy(gate1.p,g1);copy(down0.p,d0);copy(down1.p,d1);
    auto launch=[&](int mode,int set,bool publish){
        auto* a=mode?act1.p:act0.p;auto* p=mode?pack1.p:pack0.p;auto* s=mode?scale1.p:scale0.p;
        const auto* g=(mode?gate1.p:gate0.p)+size_t(set)*count;const auto* d=mode?down1.p:down0.p;
        auto call=[&](int first,int n){check(mollm_cuda::launch_nvfp4_mma_activation_diagnostic(g+first,d+first,n,plan.p,
            packed.p,scales.p,a+size_t(first)*inner,p+size_t(first)*inner/2,s+size_t(first)*inner/16,publish,cudaStreamPerThread));};
        if(mode)call(0,count);else{call(0,na);call(na,nb);}
    };
    auto reset=[&](){for(auto p:{act0.p,act1.p})check(cudaMemset(p,0x5a,(av+8)*4));
        for(auto p:{pack0.p,pack1.p})check(cudaMemset(p,0x5a,pv+8));for(auto p:{scale0.p,scale1.p})check(cudaMemset(p,0x5a,sv+8));
        check(cudaMemset(unused.p,0x5a,(size_t(count)*h+8)*2));check(cudaMemset(plan.p,0x5a,(count+1)*sizeof(*plan.p)));
        check(cudaMemset(packed.p,0x5a,size_t(count)*k+8));check(cudaMemset(scales.p,0x5a,size_t(count)*k/8+8));};
    auto compare=[&](bool publish){
        const auto a=read(act0.p,av+8),b=read(act1.p,av+8);guard(a,av);guard(b,av);require(!std::memcmp(a.data(),b.data(),(av+8)*4));
        if(!publish)for(auto x:read(reinterpret_cast<uint8_t*>(act0.p),(av+8)*4))require(x==0x5a);
        const auto p=read(pack0.p,pv+8),q=read(pack1.p,pv+8),s=read(scale0.p,sv+8),t=read(scale1.p,sv+8);
        guard(p,pv);guard(q,pv);guard(s,sv);guard(t,sv);
        for(int i=0;i<count;++i){const int r=permutation[i];require(!std::memcmp(p.data()+size_t(r)*320,q.data()+size_t(i)*320,320));
            require(!std::memcmp(s.data()+size_t(r)*40,t.data()+size_t(i)*40,40));}
        for(auto x:read(unused.p,size_t(count)*h+8))require(x==0x5a5a);
        guard(read(packed.p,size_t(count)*k+8),size_t(count)*k);guard(read(scales.p,size_t(count)*k/8+8),size_t(count)*k/8);
        const auto tail=read(reinterpret_cast<uint8_t*>(plan.p+count),sizeof(*plan.p));for(auto x:tail)require(x==0x5a);
    };
    int checks=0;
    for(bool publish:{false,true})for(int set=0;set<sets;++set){
        cudaGraphExec_t graphs[2];for(int mode=0;mode<2;++mode){cudaGraph_t graph;
            check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));launch(mode,set,publish);
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&graphs[mode],graph,0));check(cudaGraphDestroy(graph));}
        for(int variant=0;variant<4;++variant){std::vector<float> x(size_t(rows)*k);
            for(size_t i=0;i<x.size();++i)x[i]=variant==2?0.f:variant==3?-0.f:(int((i*13+variant*11+set)%101)-50)/37.f;
            copy(input.p,x);reset();for(auto graph:graphs)check(cudaGraphLaunch(graph,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));compare(publish);++checks;
        }
        for(auto graph:graphs)check(cudaGraphExecDestroy(graph));
    }
    std::printf("route_windows pair=%d jobs=%d na=%d nb=%d route_identity_permutation=true captured_pairs=%d bytes_equal=true guards=true\n",pair,count,na,nb,checks);std::fflush(stdout);
    if(timing){std::vector<float> x(size_t(rows)*k);for(size_t i=0;i<x.size();++i)x[i]=(int(i%101)-50)/37.f;copy(input.p,x);
        cudaGraphExec_t graphs[2];for(int mode=0;mode<2;++mode){cudaGraph_t graph;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            for(int set=0;set<sets;++set)launch(mode,set,false);check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            check(cudaGraphInstantiateWithFlags(&graphs[mode],graph,0));check(cudaGraphDestroy(graph));}
        for(int repeat=0;repeat<7;++repeat){double us[2];for(int order=0;order<2;++order){const int mode=order^(repeat&1);us[mode]=time_graph(graphs[mode])/sets;}
            std::printf("route_windows_timing pair=%d jobs=%d repeat=%d split_us=%.3f merged_us=%.3f scope=same_process_same_route_inputs_weights_packed_gate_chain_not_model\n",pair,count,repeat,us[0],us[1]);std::fflush(stdout);}
        reset();for(auto graph:graphs)check(cudaGraphLaunch(graph,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));compare(false);
        for(auto graph:graphs)check(cudaGraphExecDestroy(graph));
    }
}
int main(int argc,char** argv){
    bool only=argc==2&&!std::strcmp(argv[1],"--check-only");
    std::vector<int> a{1,7,8,16,17,32,39},b{2,8,9,15,16,17,13};
    if(argc>1&&!only){if(argc!=5||std::strcmp(argv[1],"--counts-a")||std::strcmp(argv[3],"--counts-b")||
        !parse(argv[2],a)||!parse(argv[4],b)||a.size()!=512||b.size()!=512)return 64;}
    const int na=std::accumulate(a.begin(),a.end(),0),nb=std::accumulate(b.begin(),b.end(),0);
    if(a.size()!=b.size()||na<=0||nb<=0||na%10||nb%10||na+nb>40960)return 64;
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));if(prop.major!=12||prop.minor>1)return 77;
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);
    if(!mollm_memory::cuda_host_allocation_allowed(size_t(5)<<30,"bounded route-window diagnostic"))return 1;
    for(bool pair:{true,false})run(a,b,pair,!only);
}
