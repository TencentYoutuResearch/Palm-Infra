// Synthetic compound-operator diagnostic; no model/backend selector.
#include "kernels/cuda/elementwise.h"
#include "runtime/cuda_host_memory_budget.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static void check(cudaError_t status){
    if(status!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(1);}
}
template<class T>struct Buffer {
    T* data=nullptr;
    explicit Buffer(size_t size){check(cudaMalloc(&data,size*sizeof(T)));}
    ~Buffer(){cudaFree(data);}
};
template<class T>static std::vector<T> read(T* pointer,size_t size){
    std::vector<T> result(size);check(cudaMemcpy(result.data(),pointer,size*sizeof(T),cudaMemcpyDeviceToHost));return result;
}
template<class T>static void copy(T* pointer,const std::vector<T>& source){
    check(cudaMemcpy(pointer,source.data(),source.size()*sizeof(T),cudaMemcpyHostToDevice));
}
template<class T>static void compare(const std::vector<T>& a,const std::vector<T>& b,const char* label,int rows,int variant){
    if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))){
        std::fprintf(stderr,"deferred RMS mismatch %s rows=%d variant=%d\n",label,rows,variant);std::exit(2);
    }
    const auto* guard=reinterpret_cast<const unsigned char*>(b.data()+b.size()-8);
    for(size_t i=0;i<8*sizeof(T);++i)if(guard[i]!=0x5a)std::exit(3);
}
static float time_graph(cudaGraphExec_t graph){
    check(cudaGraphLaunch(graph,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
    cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
    check(cudaEventRecord(start,cudaStreamPerThread));
    for(int i=0;i<32;++i)check(cudaGraphLaunch(graph,cudaStreamPerThread));
    check(cudaEventRecord(end,cudaStreamPerThread));check(cudaEventSynchronize(end));
    float ms;check(cudaEventElapsedTime(&ms,start,end));check(cudaEventDestroy(start));check(cudaEventDestroy(end));return ms*1000/32;
}
int main(int argc,char** argv){
    const bool checks_only=argc==2&&!std::strcmp(argv[1],"--check-only");
    if(argc>1&&!checks_only)return 64;
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);
    if(!mollm_memory::cuda_host_allocation_allowed(512*1024*1024,"bounded deferred RMS diagnostic"))return 4;
    constexpr int width=2560,streams=4,max_rows=1024,guard=8;
    constexpr size_t capacity=size_t(width)*streams*max_rows;
    Buffer<float> residual(capacity),branch(capacity/streams),gates(max_rows*streams),weight(width*streams),logits(capacity);
    Buffer<float> carry(capacity+guard),normalized(capacity+guard),inverses(max_rows*streams+guard),output(capacity/streams+guard);
    Buffer<uint16_t> half(capacity+guard);
    int checks=0,rejections=0;
    auto reject=[&](bool accepted){if(accepted)std::exit(5);++rejections;};
    reject(mollm_cuda::launch_gr_inject_rms_deferred(nullptr,residual.data,gates.data,weight.data,carry.data,half.data,inverses.data,1,1e-6f));
    reject(mollm_cuda::launch_gr_inject_rms_deferred(branch.data,residual.data,gates.data,weight.data,residual.data,half.data,inverses.data,1,1e-6f));
    reject(mollm_cuda::launch_gr_inject_rms_deferred(branch.data,residual.data,gates.data,weight.data,carry.data,reinterpret_cast<uint16_t*>(carry.data+1),inverses.data,1,1e-6f));
    reject(mollm_cuda::launch_gr_inject_rms_deferred(branch.data,residual.data,gates.data,weight.data,carry.data,half.data,reinterpret_cast<float*>(UINTPTR_MAX-4),1,1e-6f));
    reject(mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(carry.data,logits.data,weight.data,inverses.data,carry.data,1));
    reject(mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(carry.data,logits.data,weight.data,inverses.data,output.data,0));
    reject(mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(carry.data,logits.data,weight.data,inverses.data,output.data,4097));
    check(cudaGetLastError());
    for(int rows:{1,3,17,65,256,512,719,1024})for(float epsilon:{1e-6f,1e-3f}){
        const size_t count=size_t(rows)*width*streams,out_count=count/streams;
        auto launch=[&](bool deferred){
            if(deferred){
                if(!mollm_cuda::launch_gr_inject_rms_deferred(branch.data,residual.data,gates.data,weight.data,carry.data,half.data,inverses.data,rows,epsilon,cudaStreamPerThread)||
                   !mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(carry.data,logits.data,weight.data,inverses.data,output.data,rows,cudaStreamPerThread))std::exit(6);
            }else{
                if(!mollm_cuda::launch_gr_inject_group_rms_half(branch.data,residual.data,gates.data,weight.data,carry.data,normalized.data,half.data,width,streams,rows,epsilon,cudaStreamPerThread))std::exit(6);
                mollm_cuda::launch_gr_reduce_sigmoid(normalized.data,logits.data,output.data,width,streams,rows,cudaStreamPerThread);check(cudaGetLastError());
            }
        };
        cudaGraphExec_t graphs[3];
        for(int mode=0;mode<2;++mode){
            cudaGraph_t graph;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));launch(mode);
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&graphs[mode],graph,0));check(cudaGraphDestroy(graph));
        }
        {
            cudaGraph_t graph;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            if(!mollm_cuda::launch_gr_rms_deferred_materialize(carry.data,weight.data,inverses.data,normalized.data,rows,cudaStreamPerThread))return 6;
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&graphs[2],graph,0));check(cudaGraphDestroy(graph));
        }
        auto inputs=[&](int variant){
            std::vector<float> x(count),b(out_count),g(rows*streams),w(width*streams),l(count);
            for(size_t i=0;i<count;++i){x[i]=std::sin(float(i%8191)*.137f+variant)*3;l[i]=std::cos(float(i%8191)*.173f+variant)*12;}
            for(size_t i=0;i<out_count;++i)b[i]=std::cos(float(i%8191)*.11f+variant)*2;
            for(size_t i=0;i<g.size();++i)g[i]=std::sin(float(i)*.71f+variant);
            for(size_t i=0;i<w.size();++i)w[i]=std::cos(float(i)*.19f+variant);
            if(variant==2||variant==3||variant==4){
                std::fill(x.begin(),x.end(),variant==3?-0.f:0.f);std::fill(b.begin(),b.end(),variant==3?-0.f:0.f);std::fill(g.begin(),g.end(),1.f);
                if(variant==4)x[0]=std::numeric_limits<float>::min()/128.f;
            }
            if(variant==5){x[0]=INFINITY;b[1]=-INFINITY;}
            if(variant==6)x[0]=std::numeric_limits<float>::quiet_NaN();
            if(variant==7){l[0]=INFINITY;l[1]=-INFINITY;l[2]=-0.f;l[3]=80.f;l[4]=-80.f;}
            copy(residual.data,x);copy(branch.data,b);copy(gates.data,g);copy(weight.data,w);copy(logits.data,l);
        };
        for(int variant=0;variant<8;++variant){
            inputs(variant);
            std::vector<float> reference_carry,reference_out,reference_normalized;std::vector<uint16_t> reference_half;
            for(int mode=0;mode<2;++mode){
                check(cudaMemset(carry.data,0x5a,(count+guard)*4));check(cudaMemset(half.data,0x5a,(count+guard)*2));
                check(cudaMemset(output.data,0x5a,(out_count+guard)*4));check(cudaMemset(inverses.data,0x5a,(rows*streams+guard)*4));
                check(cudaMemset(normalized.data,0x5a,(count+guard)*4));
                check(cudaGraphLaunch(graphs[mode],cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
                auto c=read(carry.data,count+guard);auto h=read(half.data,count+guard);auto o=read(output.data,out_count+guard);
                if(mode==0){reference_carry=std::move(c);reference_half=std::move(h);reference_out=std::move(o);reference_normalized=read(normalized.data,count+guard);}
                else{
                    compare(reference_carry,c,"carry",rows,variant);compare(reference_half,h,"half",rows,variant);compare(reference_out,o,"reduced",rows,variant);
                    const auto inv=read(inverses.data,size_t(rows)*streams+guard);
                    const auto* inv_guard=reinterpret_cast<const unsigned char*>(inv.data()+rows*streams);
                    for(int i=0;i<guard*4;++i)if(inv_guard[i]!=0x5a)return 3;
                    const auto norm=read(normalized.data,count+guard);
                    const auto* bytes=reinterpret_cast<const unsigned char*>(norm.data());
                    for(size_t i=0;i<norm.size()*4;++i)if(bytes[i]!=0x5a)return 3;
                    check(cudaGraphLaunch(graphs[2],cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
                    const auto materialized=read(normalized.data,count+guard);
                    compare(reference_normalized,materialized,"materialized",rows,variant);
                    if(variant==3){uint32_t bits;std::memcpy(&bits,&c[0],4);if(bits!=0x80000000u)return 3;}
                    if(variant==4&&(c[0]==0.f||materialized[0]==0.f))return 3;
                    if(variant!=5&&variant!=6){
                        for(int row:{0,rows-1})for(int s=0;s<streams;++s){
                            const size_t base=(size_t(row)*streams+s)*width;double sum=0;
                            for(int d=0;d<width;++d)sum+=double(c[base+d])*c[base+d];
                            const double expected=1/std::sqrt(sum/width+epsilon);
                            if(!std::isfinite(inv[row*streams+s])||std::abs(inv[row*streams+s]-expected)>2e-6*expected)return 7;
                        }
                    }
                    ++checks;
                }
            }
        }
        std::printf("deferred_rms rows=%d epsilon=%g replays=8 exact_carry_half_reduced_materialized=1 guards=1 normalized_unwritten=1\n",rows,epsilon);
        if(!checks_only&&epsilon==1e-6f&&rows>=256){
            inputs(1);
            for(int pair=0;pair<7;++pair){float us[2];for(int step=0;step<2;++step){const int mode=step^(pair&1);us[mode]=time_graph(graphs[mode]);}
                std::printf("deferred_rms_timing rows=%d pair=%d baseline_us=%.3f deferred_us=%.3f scope=synthetic_producer_consumer_excludes_dense_projections\n",rows,pair,us[0],us[1]);std::fflush(stdout);
            }
            // Refresh both representations for the same input. Consumer-only
            // timings exclude producer writes and intervening HC projections.
            launch(false);launch(true);check(cudaStreamSynchronize(cudaStreamPerThread));
            cudaGraphExec_t consumers[2];
            for(int mode=0;mode<2;++mode){
                cudaGraph_t graph;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
                if(mode){
                    if(!mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(carry.data,logits.data,weight.data,inverses.data,output.data,rows,cudaStreamPerThread))return 6;
                }else{
                    mollm_cuda::launch_gr_reduce_sigmoid(normalized.data,logits.data,output.data,width,streams,rows,cudaStreamPerThread);check(cudaGetLastError());
                }
                check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&consumers[mode],graph,0));check(cudaGraphDestroy(graph));
            }
            std::vector<float> consumer_reference;
            for(int mode=0;mode<2;++mode){
                check(cudaMemset(output.data,0x5a,(out_count+guard)*4));
                check(cudaGraphLaunch(consumers[mode],cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
                auto actual=read(output.data,out_count+guard);
                if(mode)compare(consumer_reference,actual,"consumer_only",rows,1);
                else consumer_reference=std::move(actual);
            }
            for(int pair=0;pair<7;++pair){float us[2];for(int step=0;step<2;++step){const int mode=step^(pair&1);us[mode]=time_graph(consumers[mode]);}
                std::printf("deferred_rms_consumer rows=%d pair=%d materialized_us=%.3f deferred_us=%.3f scope=consumer_excludes_producer_and_dense_projections\n",rows,pair,us[0],us[1]);std::fflush(stdout);
            }
            for(auto graph:consumers)check(cudaGraphExecDestroy(graph));
        }
        for(auto graph:graphs)check(cudaGraphExecDestroy(graph));
    }
    std::printf("deferred_rms_checks=%d rejected_arguments=%d no_backend_selector=1\n",checks,rejections);
}
