// Independent sparse metadata planner comparison; no model/expert payload.
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
static bool equal(const mollm_cuda::Nvfp4Projection& a,const mollm_cuda::Nvfp4Projection& b){return a.weight==b.weight&&a.block_scales==b.block_scales&&a.row_scales==b.row_scales&&a.pair_layout==b.pair_layout&&a.mma_layout==b.mma_layout;}
static double time_graph(cudaGraphExec_t graph){
    check(cudaGraphLaunch(graph,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));cudaEvent_t a,b;
    check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a,cudaStreamPerThread));
    for(int i=0;i<64;++i)check(cudaGraphLaunch(graph,cudaStreamPerThread));
    check(cudaEventRecord(b,cudaStreamPerThread));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));
    check(cudaEventDestroy(a));check(cudaEventDestroy(b));return ms*1000/64;
}
static void run(const std::vector<int>& counts,bool timing){
    const int count=std::accumulate(counts.begin(),counts.end(),0);
    Buffer<mollm_cuda::Nvfp4Projection> jobs(count);
    Buffer<mollm_cuda::Nvfp4MmaBatch> scalar(count+8),parallel(count+8);
    std::vector<mollm_cuda::Nvfp4Projection> original(count);int index=0;
    for(size_t expert=0;expert<counts.size();++expert)for(int j=0;j<counts[expert];++j,++index){
        original[index].weight=reinterpret_cast<const uint8_t*>(0x1000+expert*256);
        original[index].block_scales=reinterpret_cast<const uint8_t*>(0x100000+expert*256);
        original[index].row_scales=reinterpret_cast<const float*>(0x200000+expert*256);
        original[index].pair_layout=expert%2;original[index].mma_layout=expert%3==0;
    }
    int checks=0,rejected=0;
    check(cudaMemset(parallel.p,0x5a,size_t(count+8)*sizeof(*parallel.p)));
    for(int variant=0;variant<6;++variant){const auto status=mollm_cuda::launch_nvfp4_batch_plan_diagnostic(
        variant==0?nullptr:jobs.p,variant==1?0:variant==2?65536:count,
        variant==3?nullptr:parallel.p,variant==4?7:variant==5?64:32,true,cudaStreamPerThread);
        if(status!=cudaErrorInvalidValue)std::exit(4);++rejected;}
    for(int variant=0;variant<2;++variant){const auto status=mollm_cuda::launch_nvfp4_mma_grouped(jobs.p,count,parallel.p,
        2560,640,0,cudaStreamPerThread,nullptr,nullptr,1,true,false,variant==0?parallel.p:nullptr,0,8,variant==1,false,false,false,false,true);
        if(status!=cudaErrorInvalidValue)std::exit(4);++rejected;}
    std::vector<unsigned char> untouched(size_t(count+8)*sizeof(*parallel.p));
    check(cudaMemcpy(untouched.data(),parallel.p,untouched.size(),cudaMemcpyDeviceToHost));for(auto x:untouched)if(x!=0x5a)std::exit(4);
    for(int width:{8,16,32}){
        cudaGraphExec_t graphs[2];for(int mode=0;mode<2;++mode){cudaGraph_t graph;
            check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            check(mollm_cuda::launch_nvfp4_batch_plan_diagnostic(jobs.p,count,mode?parallel.p:scalar.p,width,mode,cudaStreamPerThread));
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&graphs[mode],graph,0));check(cudaGraphDestroy(graph));}
        for(int replay=0;replay<4;++replay){auto descriptors=original;
            if(replay==1)std::reverse(descriptors.begin(),descriptors.end());
            if(replay==2)for(int i=0;i<count;++i){if(i%7==0)descriptors[i].block_scales=reinterpret_cast<const uint8_t*>(0x400000);
                if(i%11==0)descriptors[i].row_scales=reinterpret_cast<const float*>(0x500000);if(i%13==0)descriptors[i].mma_layout=!descriptors[i].mma_layout;}
            if(replay==3)for(int i=0;i<count;++i){descriptors[i]=original[0];descriptors[i].pair_layout=i%3==0;}
            check(cudaMemcpy(jobs.p,descriptors.data(),size_t(count)*sizeof(*jobs.p),cudaMemcpyHostToDevice));
            std::vector<mollm_cuda::Nvfp4MmaBatch> expected(count+8);std::memset(expected.data(),0x5a,expected.size()*sizeof(expected[0]));
            for(int begin=0;begin<count;){int end=begin+1;while(end<count&&equal(descriptors[begin],descriptors[end]))++end;
                for(int first=begin;first<end;first+=width){const int size=std::min(width,end-first);for(int i=first;i<first+size;++i)expected[i]={i==first?first:-1,size};}begin=end;}
            for(int mode=0;mode<2;++mode){auto* output=mode?parallel.p:scalar.p;check(cudaMemset(output,0x5a,size_t(count+8)*sizeof(*output)));
                check(cudaGraphLaunch(graphs[mode],cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));std::vector<mollm_cuda::Nvfp4MmaBatch> actual(count+8);
                check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(actual[0]),cudaMemcpyDeviceToHost));
                if(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(actual[0]))){std::fprintf(stderr,"planner mismatch count=%d width=%d replay=%d parallel=%d\n",count,width,replay,mode);std::exit(5);}++checks;}
        }
        check(cudaMemcpy(jobs.p,original.data(),size_t(count)*sizeof(*jobs.p),cudaMemcpyHostToDevice));
        if(timing)for(int repeat=0;repeat<7;++repeat){double us[2];for(int order=0;order<2;++order){const int mode=order^(repeat&1);us[mode]=time_graph(graphs[mode]);}
            std::printf("batch_plan_timing count=%d width=%d repeat=%d scalar_us=%.3f parallel_us=%.3f scope=memset_plus_metadata_plan_not_model\n",count,width,repeat,us[0],us[1]);std::fflush(stdout);}
        for(auto graph:graphs)check(cudaGraphExecDestroy(graph));
    }
    std::printf("batch_plan_check count=%d captured_checks=%d rejects=%d cpu_expected_full_bytes_guards=true\n",count,checks,rejected);std::fflush(stdout);
}
int main(int argc,char** argv){
    const bool small=argc==2&&!std::strcmp(argv[1],"--small-check-only");
    const bool only=small||(argc==2&&!std::strcmp(argv[1],"--check-only"));std::vector<int> counts{1,7,8,16,17,32,33};
    if(argc>1&&!only){if(argc!=5||std::strcmp(argv[1],"--rows")||std::strcmp(argv[3],"--counts")||
        (std::strcmp(argv[2],"1024")&&std::strcmp(argv[2],"719")))return 64;
        counts.clear();const char* p=argv[4];while(*p){const char* begin=p;while(*p>='0'&&*p<='9')++p;
            if(p==begin||p-begin>5||(*p&&*p!=','))return 64;long n=std::strtol(begin,nullptr,10);if(n<0||n>65535)return 64;counts.push_back(n);if(!*p)break;++p;if(!*p)return 64;}
        if(counts.size()!=512||std::accumulate(counts.begin(),counts.end(),0)!=std::atoi(argv[2])*10)return 64;}
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);if(!mollm_memory::cuda_host_allocation_allowed(size_t(1)<<30,"bounded planner diagnostic"))return 1;
    if(only){for(int count:{1,31,32,33,127,128,129,1024,65535})if(!small||count!=65535)run({count},false);run(counts,false);}
    else run(counts,true);
}
