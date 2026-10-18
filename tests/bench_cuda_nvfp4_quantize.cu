#include "kernels/cuda/moe.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void check(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);} }
template<class T> struct Buffer {
    T* data=nullptr;
    explicit Buffer(size_t n){check(cudaMalloc(&data,n*sizeof(T)));}
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){cudaFree(data);}
};
static bool run(int count,int inner,int variants,int minimum,bool filtered,bool timing,bool shared_inputs=false) {
    const int source_rows=shared_inputs?count/10:count;
    const size_t input_count=size_t(source_rows)*inner, packed_count=size_t(count)*inner*variants/2,
        scale_count=size_t(count)*inner*variants/16;
    Buffer<float> input(input_count);
    Buffer<mollm_cuda::Nvfp4Projection> jobs(count);
    Buffer<mollm_cuda::Nvfp4MmaBatch> batches(count);
    Buffer<uint8_t> packed[5]={Buffer<uint8_t>{packed_count+8},Buffer<uint8_t>{packed_count+8},Buffer<uint8_t>{packed_count+8},Buffer<uint8_t>{packed_count+8},Buffer<uint8_t>{packed_count+8}};
    Buffer<uint8_t> scales[5]={Buffer<uint8_t>{scale_count+8},Buffer<uint8_t>{scale_count+8},Buffer<uint8_t>{scale_count+8},Buffer<uint8_t>{scale_count+8},Buffer<uint8_t>{scale_count+8}};
    std::vector<float> values(input_count);
    std::vector<mollm_cuda::Nvfp4Projection> descriptors(count);
    std::vector<mollm_cuda::Nvfp4MmaBatch> plan(count);
    std::vector<uint8_t> a(std::max(packed_count,scale_count)+8),b(a.size());
    const auto launch=[&](int mode){check(mollm_cuda::launch_nvfp4_quantize_probe(jobs.data,count,inner,variants,
        packed[mode].data,scales[mode].data,filtered?batches.data:nullptr,minimum,mode,cudaStreamPerThread));};
    cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));launch(0);launch(1);launch(2);launch(3);launch(4);
    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    constexpr int replays=10;
    for(int replay=0;replay<replays;++replay) {
        for(size_t i=0;i<input_count;++i) values[i]=float(int((i*7+replay)%97)-48)/29.f;
        if(replay==2) std::fill(values.begin(),values.end(),0.f);
        if(replay==3) for(size_t i=0;i<input_count;++i) values[i]=(i&1)?-0.f:0.f;
        if(replay==4){values[0]=NAN;values[1]=INFINITY;values[2]=-INFINITY;values[3]=1e-38f;}
        if(replay==7) std::fill(values.begin(),values.end(),NAN);
        if(replay==8) std::fill(values.begin(),values.end(),INFINITY);
        if(replay==9) for(size_t i=0;i<input_count;++i) values[i]=(i%32<16)?1e-38f:-1e-38f;
        for(int i=0;i<count;++i) {
            descriptors[i]={};descriptors[i].input=input.data+size_t((count-1-i+replay)%source_rows)*inner;
            descriptors[i].input_global=.0013892765f*(1.f+(i%7)*.125f);
            descriptors[i].input_global_second=.0031854538f*(1.f+(i%5)*.25f);
            if(replay==5 && i%3==0) descriptors[i].input_global=0.f;
            if(replay==6) {
                const float exceptional[]={-0.f,-.0013892765f,NAN,INFINITY,-INFINITY};
                descriptors[i].input_global=exceptional[i%5];
                descriptors[i].input_global_second=exceptional[(i+1)%5];
            }
            plan[i]={i,(i+replay)%9};
        }
        check(cudaMemcpy(input.data,values.data(),input_count*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(jobs.data,descriptors.data(),count*sizeof(descriptors[0]),cudaMemcpyHostToDevice));
        check(cudaMemcpy(batches.data,plan.data(),count*sizeof(plan[0]),cudaMemcpyHostToDevice));
        for(int mode=0;mode<5;++mode){check(cudaMemset(packed[mode].data,0xa5,packed_count+8));check(cudaMemset(scales[mode].data,0xa5,scale_count+8));}
        check(cudaGraphLaunch(executable,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
        for(int field=0;field<2;++field) for(int mode=1;mode<5;++mode) {
            const size_t bytes=(field?scale_count:packed_count)+8;
            check(cudaMemcpy(a.data(),field?scales[0].data:packed[0].data,bytes,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(b.data(),field?scales[mode].data:packed[mode].data,bytes,cudaMemcpyDeviceToHost));
            if(std::memcmp(a.data(),b.data(),bytes)){std::fprintf(stderr,"mismatch count=%d K=%d variants=%d min=%d filtered=%d replay=%d field=%d\n",count,inner,variants,minimum,filtered,replay,field);return false;}
            for(size_t i=bytes-8;i<bytes;++i) if(a[i]!=0xa5 || b[i]!=0xa5)return false;
            if(filtered) for(int job=0;job<count*variants;++job) if(plan[job%count].size<minimum) {
                const size_t stride=inner/(field?16:2);
                for(size_t i=size_t(job)*stride;i<size_t(job+1)*stride;++i) if(a[i]!=0xa5 || b[i]!=0xa5)return false;
            }
        }
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
    std::printf("count=%d K=%d variants=%d min=%d filtered=%d shared=%d captures=%d modes=5 exact=1 guards=1\n",count,inner,variants,minimum,filtered,shared_inputs,replays);
    if(timing && !std::getenv("MOLLM_NVFP4_QUANTIZE_CHECK_ONLY")) {
        // Restore finite varied inputs, calibrated globals and active batches.
        for(size_t i=0;i<input_count;++i)values[i]=float(int(i%97)-48)/29.f;
        for(int i=0;i<count;++i){descriptors[i].input_global=.0013892765f;plan[i].size=8;}
        check(cudaMemcpy(input.data,values.data(),input_count*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(jobs.data,descriptors.data(),count*sizeof(descriptors[0]),cudaMemcpyHostToDevice));
        check(cudaMemcpy(batches.data,plan.data(),count*sizeof(plan[0]),cudaMemcpyHostToDevice));
        cudaGraphExec_t graphs[5];constexpr int calls=16;
        for(int mode=0;mode<5;++mode){check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            for(int i=0;i<calls;++i)launch(mode);check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            check(cudaGraphInstantiate(&graphs[mode],graph,nullptr,nullptr,0));check(cudaGraphDestroy(graph));}
        for(int order=0;order<7;++order)for(int step=0;step<5;++step){const int mode=(order&1)?4-step:step;
            for(int i=0;i<2;++i)check(cudaGraphLaunch(graphs[mode],cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));check(cudaEventRecord(start,cudaStreamPerThread));
            for(int i=0;i<3;++i)check(cudaGraphLaunch(graphs[mode],cudaStreamPerThread));check(cudaEventRecord(end,cudaStreamPerThread));check(cudaEventSynchronize(end));
            float ms;check(cudaEventElapsedTime(&ms,start,end));std::printf("timing count=%d K=%d variants=%d shared=%d order=%d mode=%d us=%.3f\n",count,inner,variants,shared_inputs,order,mode,ms*1000/(3*calls));
            check(cudaEventDestroy(start));check(cudaEventDestroy(end));}
        for(auto executable:graphs)check(cudaGraphExecDestroy(executable));
    }
    return true;
}
static bool run_cached_max(int count, int inner, int variants, bool filtered, bool shared, bool timing,
    int filter_minimum = 4) {
    const int rows = shared ? (count + 9) / 10 : count, minimum = filtered ? filter_minimum : 1;
    const size_t inputs = size_t(rows) * inner, bytes = size_t(count) * variants * inner / 2;
    const size_t scale_bytes = size_t(count) * variants * inner / 16, groups = size_t(rows) * inner / 16;
    Buffer<float> input(inputs), external(inner), maxima(groups + 2);
    Buffer<mollm_cuda::Nvfp4Projection> jobs(count);
    Buffer<mollm_cuda::Nvfp4MmaBatch> batches(count);
    Buffer<uint8_t> packed[2] = {Buffer<uint8_t>{bytes+8}, Buffer<uint8_t>{bytes+8}};
    Buffer<uint8_t> scales[2] = {Buffer<uint8_t>{scale_bytes+8}, Buffer<uint8_t>{scale_bytes+8}};
    std::vector<float> values(inputs);
    std::vector<mollm_cuda::Nvfp4Projection> descriptors(count);
    std::vector<mollm_cuda::Nvfp4MmaBatch> plan(count);
    const auto launch = [&](int mode) {
        if (!mode) check(mollm_cuda::launch_nvfp4_quantize_probe(jobs.data,count,inner,variants,
            packed[0].data,scales[0].data,filtered?batches.data:nullptr,minimum,0,cudaStreamPerThread));
        else check(mollm_cuda::launch_nvfp4_quantize_cached_max_probe(jobs.data,count,inner,variants,
            packed[1].data,scales[1].data,filtered?batches.data:nullptr,minimum,input.data,rows,
            maxima.data,cudaStreamPerThread));
    };
    cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    launch(0); launch(1);
    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
    check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for (int replay = 0; replay < 10; ++replay) {
        for (size_t i=0;i<inputs;++i) values[i]=float(int((i*7+replay)%97)-48)/29.f;
        if (replay==2) std::fill(values.begin(),values.end(),0.f);
        if (replay==3) for(size_t i=0;i<inputs;++i) values[i]=(i&1)?-0.f:0.f;
        if (replay==4) {values[0]=NAN;if(inputs>1)values[1]=INFINITY;if(inputs>2)values[2]=-INFINITY;}
        if (replay==7) std::fill(values.begin(),values.end(),NAN);
        if (replay==8) std::fill(values.begin(),values.end(),INFINITY);
        if (replay==9) for(size_t i=0;i<inputs;++i)values[i]=(i&1)?-1e-38f:1e-38f;
        for (int i=0;i<count;++i) {
            descriptors[i]={}; descriptors[i].input=input.data+size_t((i+replay)%rows)*inner;
            descriptors[i].input_global=.0013892765f*(1.f+(i%7)*.125f);
            descriptors[i].input_global_second=.0031854538f*(1.f+(i%5)*.25f);
            if (replay==5 && i%3==0) descriptors[i].input_global=0.f;
            if (replay==6) {
                const float exceptional[]={-0.f,-.0013892765f,NAN,INFINITY,-INFINITY};
                descriptors[i].input_global=exceptional[i%5];
                descriptors[i].input_global_second=exceptional[(i+1)%5];
            }
            plan[i]={i,(i+replay)%9};
        }
        // Exercise unrecognized and shifted (but still valid) row pointers.
        if (replay==1) descriptors[0].input=external.data;
        if (replay==4 && rows>1) descriptors[0].input=input.data+1;
        check(cudaMemcpy(input.data,values.data(),inputs*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemcpy(external.data,values.data(),inner*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemcpy(jobs.data,descriptors.data(),count*sizeof(descriptors[0]),cudaMemcpyHostToDevice));
        check(cudaMemcpy(batches.data,plan.data(),count*sizeof(plan[0]),cudaMemcpyHostToDevice));
        for(int mode=0;mode<2;++mode) {
            check(cudaMemset(packed[mode].data,0xa5,bytes+8));
            check(cudaMemset(scales[mode].data,0xa5,scale_bytes+8));
        }
        check(cudaMemset(maxima.data,0xa5,(groups+2)*sizeof(float)));
        check(cudaGraphLaunch(executable,cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        for(int field=0;field<2;++field) {
            const size_t size=(field?scale_bytes:bytes)+8;
            std::vector<uint8_t> a(size),b(size);
            check(cudaMemcpy(a.data(),field?scales[0].data:packed[0].data,size,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(b.data(),field?scales[1].data:packed[1].data,size,cudaMemcpyDeviceToHost));
            if(a!=b) {std::fprintf(stderr,"cached-max mismatch count=%d K=%d replay=%d field=%d\n",count,inner,replay,field);return false;}
            for(size_t i=size-8;i<size;++i)if(a[i]!=0xa5)return false;
            if(filtered)for(int job=0;job<count*variants;++job)if(plan[job%count].size<minimum) {
                const size_t stride=inner/(field?16:2);
                for(size_t i=size_t(job)*stride;i<size_t(job+1)*stride;++i)if(a[i]!=0xa5)return false;
            }
        }
        uint8_t guard[8]; check(cudaMemcpy(guard,maxima.data+groups,8,cudaMemcpyDeviceToHost));
        for(uint8_t byte:guard)if(byte!=0xa5)return false;
    }
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    std::printf("cached-max count=%d K=%d variants=%d shared=%d filtered=%d minimum=%d captures=10 packed_scales_exact=1 guards=1 pointer_fallbacks=1\n",
        count,inner,variants,shared,filtered,minimum);
    if (timing && !std::getenv("MOLLM_NVFP4_QUANTIZE_CHECK_ONLY")) {
        for(size_t i=0;i<inputs;++i)values[i]=float(int(i%97)-48)/29.f;
        for(int i=0;i<count;++i) {
            descriptors[i].input=input.data+size_t(i%rows)*inner;
            descriptors[i].input_global=.0013892765f; descriptors[i].input_global_second=.0031854538f;
            plan[i].size=8;
        }
        check(cudaMemcpy(input.data,values.data(),inputs*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemcpy(jobs.data,descriptors.data(),count*sizeof(descriptors[0]),cudaMemcpyHostToDevice));
        check(cudaMemcpy(batches.data,plan.data(),count*sizeof(plan[0]),cudaMemcpyHostToDevice));
        cudaGraphExec_t graphs[2]; constexpr int calls=16;
        for(int mode=0;mode<2;++mode) {
            check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            for(int i=0;i<calls;++i)launch(mode);
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            check(cudaGraphInstantiate(&graphs[mode],graph,nullptr,nullptr,0));check(cudaGraphDestroy(graph));
        }
        for(int order=0;order<7;++order)for(int step=0;step<2;++step) {
            const int mode=(order&1)?1-step:step;
            for(int i=0;i<2;++i)check(cudaGraphLaunch(graphs[mode],cudaStreamPerThread));
            check(cudaStreamSynchronize(cudaStreamPerThread));
            cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
            check(cudaEventRecord(start,cudaStreamPerThread));
            for(int i=0;i<3;++i)check(cudaGraphLaunch(graphs[mode],cudaStreamPerThread));
            check(cudaEventRecord(end,cudaStreamPerThread));check(cudaEventSynchronize(end));
            float ms;check(cudaEventElapsedTime(&ms,start,end));
            std::printf("cached-max-timing count=%d K=%d shared=%d order=%d mode=%d us=%.3f includes_stats=1\n",count,inner,shared,order,mode,ms*1000/(3*calls));
            check(cudaEventDestroy(start));check(cudaEventDestroy(end));
        }
        for(auto g:graphs)check(cudaGraphExecDestroy(g));
    }
    return true;
}

static bool vector_alignment_checks(){
    constexpr int inner=64,bytes=64,scale_bytes=8;
    Buffer<float> input(inner+8);Buffer<mollm_cuda::Nvfp4Projection> jobs(1);
    Buffer<uint8_t> packed[2]={Buffer<uint8_t>{bytes+32},Buffer<uint8_t>{bytes+32}};
    Buffer<uint8_t> scales[2]={Buffer<uint8_t>{scale_bytes+32},Buffer<uint8_t>{scale_bytes+32}};
    for(int in_shift=0;in_shift<4;++in_shift)for(int out_shift=0;out_shift<4;++out_shift){
        mollm_cuda::Nvfp4Projection d{};d.input=input.data+in_shift;
        d.input_global=.0013892765f;d.input_global_second=.0031854538f;
        check(cudaMemcpy(jobs.data,&d,sizeof(d),cudaMemcpyHostToDevice));
        cudaGraph_t graph;cudaGraphExec_t replay;
        check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        for(int mode=0;mode<2;++mode)check(mollm_cuda::launch_nvfp4_quantize_probe(
            jobs.data,1,inner,2,packed[mode].data+out_shift,scales[mode].data+1,
            nullptr,1,mode?4:0,cudaStreamPerThread));
        check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        check(cudaGraphInstantiateWithFlags(&replay,graph,0));check(cudaGraphDestroy(graph));
        for(int changed=0;changed<3;++changed){
            std::vector<float> values(inner+8);for(size_t i=0;i<values.size();++i)
                values[i]=changed==1?NAN:changed==2?((i&1)?-0.f:0.f):float(int(i%17)-8)/3.f;
            check(cudaMemcpy(input.data,values.data(),values.size()*4,cudaMemcpyHostToDevice));
            for(int mode=0;mode<2;++mode){check(cudaMemset(packed[mode].data,0xa5,bytes+32));check(cudaMemset(scales[mode].data,0xa5,scale_bytes+32));}
            check(cudaGraphLaunch(replay,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            for(int field=0;field<2;++field){const size_t n=(field?scale_bytes:bytes)+32,shift=field?1:out_shift,used=field?scale_bytes:bytes;
                std::vector<uint8_t> a(n),b(n);
                check(cudaMemcpy(a.data(),field?scales[0].data:packed[0].data,n,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(b.data(),field?scales[1].data:packed[1].data,n,cudaMemcpyDeviceToHost));
                if(a!=b)return false;
                for(size_t i=0;i<n;++i)if((i<shift||i>=shift+used)&&b[i]!=0xa5)return false;
            }
            std::vector<float> after(values.size());check(cudaMemcpy(after.data(),input.data,after.size()*4,cudaMemcpyDeviceToHost));
            if(std::memcmp(values.data(),after.data(),after.size()*4))return false;
        }
        check(cudaGraphExecDestroy(replay));
    }
    std::printf("vector_alignment captured_checks=48 packed_scales_full_bytes_guards_input_unchanged=true\n");return true;
}

int main(){int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    if (std::getenv("MOLLM_NVFP4_SHARED_MAX_PROBE")) {
        Buffer<mollm_cuda::Nvfp4Projection> jobs(1);
        Buffer<float> input(16), maxima(1);
        Buffer<uint8_t> packed(8),scales(1);
        for(int fault=0;fault<19;++fault) {
            const auto status=mollm_cuda::launch_nvfp4_quantize_cached_max_probe(
                fault==0?nullptr:jobs.data,
                fault==5?0:fault==6?-1:fault==7?65536:1,
                fault==8?0:fault==9?-16:fault==10?17:fault==11?2576:16,
                fault==12?0:fault==13?3:1,
                fault==1?nullptr:packed.data,fault==2?nullptr:scales.data,nullptr,
                fault==14?0:fault==15?9:1,fault==3?nullptr:input.data,
                fault==16?0:fault==17?-1:fault==18?65536:1,
                fault==4?nullptr:maxima.data,cudaStreamPerThread);
            if(status!=cudaErrorInvalidValue) {std::fprintf(stderr,"cached-max invalid argument accepted fault=%d\n",fault);return 1;}
        }
        std::printf("cached-max invalid_arguments_rejected=19\n");
        for(int count:{1,3,30})for(int inner:{16,64,640,2560})for(int variants:{1,2})
            for(bool filtered:{false,true})if(!run_cached_max(count,inner,variants,filtered,true,false))return 1;
        if(!run_cached_max(30,64,2,true,true,false,8) ||
            !run_cached_max(65535,16,2,true,true,false,8))return 1;
        for(bool shared:{false,true})for(int count:{8570,17430})
            if(!run_cached_max(count,2560,2,false,shared,true))return 1;
        return 0;
    }
    {
        Buffer<mollm_cuda::Nvfp4Projection> jobs(1);
        Buffer<uint8_t> packed(16),scales(16);
        int rejected=0;
        for(int mode:{0,1,2,3,4}) for(int fault=0;fault<16;++fault) {
            const auto result=mollm_cuda::launch_nvfp4_quantize_probe(
                fault==0?nullptr:jobs.data,
                fault==3?0:fault==4?-1:fault==5?65536:1,
                fault==6?0:fault==7?-16:fault==8?17:fault==9?2576:16,
                fault==10?0:fault==11?3:1,
                fault==1?nullptr:packed.data,fault==2?nullptr:scales.data,nullptr,
                fault==12?0:fault==13?9:1,
                fault==14?-1:fault==15?5:mode,cudaStreamPerThread);
            if(result!=cudaErrorInvalidValue){std::fprintf(stderr,"invalid argument accepted mode=%d fault=%d\n",mode,fault);return 1;}
            ++rejected;
        }
        std::printf("invalid arguments rejected=%d\n",rejected);
    }
    if(!vector_alignment_checks())return 1;
    for(int count:{1,3,17})for(int inner:{16,64,640,2560})for(int variants:{1,2})
        for(int minimum:{1,4,8})for(bool filtered:{false,true})if(!run(count,inner,variants,minimum,filtered,false))return 1;
    if(!run(65535,16,2,8,true,false))return 1;
    if(!run(30,64,2,4,true,false,true))return 1;
    for(bool shared:{false,true})for(int count:{7190,8570,10240,17430})if(!run(count,2560,2,1,true,true,shared))return 1;
    return 0;
}
