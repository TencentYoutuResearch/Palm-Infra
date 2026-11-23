#include "kernels/cuda/moe.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t result) {
    if (result!=cudaSuccess) {
        std::fprintf(stderr,"%s\n",cudaGetErrorString(result));std::exit(1);
    }
}
template<class T> static T* allocate(size_t count) {
    T* p=nullptr;check(cudaMalloc(&p,count*sizeof(T)));return p;
}
static cudaGraphExec_t capture(const mollm_cuda::Nvfp4Projection* jobs,int count,
    int inner,const float* gate,float* activated,uint8_t* packed,uint8_t* scales,
    int fusion_mode,int calls,cudaStream_t stream) {
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for (int i=0;i<calls;++i) check(mollm_cuda::launch_nvfp4_activation_prepare(
        jobs,count,inner,gate,activated,packed,scales,fusion_mode,stream));
    cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamEndCapture(stream,&graph));
    check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    check(cudaGraphDestroy(graph));return executable;
}
static float time(cudaGraphExec_t graph,int calls,cudaStream_t stream) {
    for (int i=0;i<2;++i) check(cudaGraphLaunch(graph,stream));
    check(cudaStreamSynchronize(stream));
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    check(cudaEventRecord(begin,stream));
    for (int i=0;i<3;++i) check(cudaGraphLaunch(graph,stream));
    check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
    float ms;check(cudaEventElapsedTime(&ms,begin,end));
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));return ms*1000/(3*calls);
}
int main() {
    int devices=0;if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0));
    if (properties.major<12) return 77;
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for (int inner : {16,64,640}) for (int count : {1,17,103,2560,8570,65535}) {
        if (count==65535 && inner!=16) continue;
        constexpr size_t guard=8;
        const size_t elements=size_t(count)*inner,bytes=elements/2,groups=elements/16;
        std::vector<float> input(2*elements),reference(elements+guard),actual(elements+guard);
        std::vector<uint8_t> pref(bytes+guard),pactual(bytes+guard),sref(groups+guard),sactual(groups+guard);
        std::vector<mollm_cuda::Nvfp4Projection> jobs(count);
        auto* dx=allocate<float>(input.size());auto* da=allocate<float>(elements+guard);
        auto* dp=allocate<uint8_t>(bytes+guard);auto* ds=allocate<uint8_t>(groups+guard);
        auto* dj=allocate<mollm_cuda::Nvfp4Projection>(count);
        if (inner==16 && count==1) {
            auto reject=[&](const mollm_cuda::Nvfp4Projection* jobs,int count,int width,
                            const float* gate,float* activated,uint8_t* packed,uint8_t* scales) {
                for (int mode : {0,1,2,3})
                    if (mollm_cuda::launch_nvfp4_activation_prepare(jobs,count,width,gate,
                        activated,packed,scales,mode,stream)!=cudaErrorInvalidValue) std::exit(4);
            };
            reject(nullptr,1,16,dx,da,dp,ds);reject(dj,1,16,nullptr,da,dp,ds);
            reject(dj,1,16,dx,nullptr,dp,ds);reject(dj,1,16,dx,da,nullptr,ds);
            reject(dj,1,16,dx,da,dp,nullptr);reject(dj,-1,16,dx,da,dp,ds);
            reject(dj,65536,16,dx,da,dp,ds);reject(dj,1,0,dx,da,dp,ds);
            reject(dj,1,15,dx,da,dp,ds);reject(dj,1,2147483632,dx,da,dp,ds);
            reject(dj,65535,1048576,dx,da,dp,ds);
            for (int mode : {0,1,2,3}) check(mollm_cuda::launch_nvfp4_activation_prepare(
                nullptr,0,16,nullptr,nullptr,nullptr,nullptr,mode,stream));
            for (int mode : {-1,4})
                if (mollm_cuda::launch_nvfp4_activation_prepare(dj,1,16,dx,da,dp,ds,mode,stream)!=cudaErrorInvalidValue) return 4;
            check(cudaGetLastError());std::puts("invalid_shapes=11 modes=4 invalid_modes=2 rejected=1 empty_ok=1");
        }
        cudaGraphExec_t graphs[4];
        for (int mode=0;mode<4;++mode) graphs[mode]=capture(dj,count,inner,dx,da,dp,ds,mode,1,stream);
        for (int variant=0;variant<3;++variant) {
            for (size_t i=0;i<input.size();++i)
                input[i]=variant==0?0.f:std::sin(float(i)*.19f+variant)*(1+i%17);
            for (int i=0;i<count;++i) {
                const int route=variant==0?i:((count-1)*size_t(i)+variant)%count;
                jobs[i]={};jobs[i].input=da+size_t(route)*inner;
                jobs[i].input_global=.125f*(1+(route+variant)%19);
            }
            check(cudaMemcpyAsync(dx,input.data(),input.size()*4,cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(dj,jobs.data(),jobs.size()*sizeof(jobs[0]),cudaMemcpyHostToDevice,stream));
            for (int mode=0;mode<4;++mode) {
                check(cudaMemsetAsync(da,0x7f,(elements+guard)*4,stream));
                check(cudaMemsetAsync(dp,0x5a,bytes+guard,stream));
                check(cudaMemsetAsync(ds,0x6b,groups+guard,stream));
                check(cudaGraphLaunch(graphs[mode],stream));check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(mode?actual.data():reference.data(),da,(elements+guard)*4,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(mode?pactual.data():pref.data(),dp,bytes+guard,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(mode?sactual.data():sref.data(),ds,groups+guard,cudaMemcpyDeviceToHost));
                if (mode && ((mode!=3 && std::memcmp(reference.data(),actual.data(),(elements+guard)*4)) ||
                pref!=pactual || sref!=sactual)) {
                    std::fprintf(stderr,"mismatch inner=%d count=%d variant=%d mode=%d\n",inner,count,variant,mode);return 2;
                }
                if (mode==3) for (const float value : actual) {
                    unsigned bits;std::memcpy(&bits,&value,4);
                    if (bits!=0x7f7f7f7f) return 3;
                }
                if (mode) for (size_t i=0;i<guard;++i) {
                    unsigned bits;std::memcpy(&bits,&actual[elements+i],4);
                    if (bits!=0x7f7f7f7f || pactual[bytes+i]!=0x5a || sactual[groups+i]!=0x6b) return 3;
                }
            }
        }
        std::printf("inner=%d routes=%d variants=3 activation_exact=1 packed_exact=1 scales_exact=1 packed_only_no_writes=1 guards=1\n",inner,count);
        for (auto graph : graphs) check(cudaGraphExecDestroy(graph));
        if (inner==640 && count>=2560 && !std::getenv("MOLLM_NVFP4_ACTIVATION_CHECK_ONLY")) {
            constexpr int calls=64;
            for (int mode=0;mode<4;++mode) graphs[mode]=capture(dj,count,inner,dx,da,dp,ds,mode,calls,stream);
            for (int order=0;order<7;++order) {
                float us[4];for (int step=0;step<4;++step) {
                    const int mode=(step+order)%4;us[mode]=time(graphs[mode],calls,stream);
                }
                std::printf("inner=%d routes=%d order=%d baseline_us=%.4f scalar_us=%.4f subgroup_us=%.4f packed_only_us=%.4f\n",inner,count,order,us[0],us[1],us[2],us[3]);
            }
            for (auto graph : graphs) check(cudaGraphExecDestroy(graph));
        }
        check(cudaFree(dx));check(cudaFree(da));check(cudaFree(dp));check(cudaFree(ds));check(cudaFree(dj));
    }
    check(cudaStreamDestroy(stream));
}
