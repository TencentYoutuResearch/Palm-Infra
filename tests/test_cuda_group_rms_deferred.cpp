#include "kernels/cuda/elementwise.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t e) {
    if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}
}
int main() {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    for(int rows:{17,719,1024,1025,1743,2048}) {
        const size_t count=size_t(rows)*10240;
        std::vector<float> input(count),weight(10240),reference(count+8),actual(count+8),inv(rows*4+8);
        std::vector<uint16_t> half_ref(count+8),half_actual(count+8);
        float *x,*w,*r,*out,*inverses;uint16_t *hr,*ha;
        check(cudaMalloc(&x,count*4));check(cudaMalloc(&w,10240*4));
        check(cudaMalloc(&r,(count+8)*4));check(cudaMalloc(&out,(count+8)*4));
        check(cudaMalloc(&inverses,inv.size()*4));
        check(cudaMalloc(&hr,(count+8)*2));check(cudaMalloc(&ha,(count+8)*2));
        for(int pass=0;pass<4;++pass) {
            for(size_t i=0;i<count;++i)input[i]=pass==0?0.f:std::sin(float(i%1009)*.017f+pass)*(1+i%17);
            for(size_t i=0;i<10240;++i)weight[i]=std::cos(float(i)*.013f+pass);
            check(cudaMemcpy(x,input.data(),count*4,cudaMemcpyHostToDevice));
            check(cudaMemcpy(w,weight.data(),10240*4,cudaMemcpyHostToDevice));
            check(cudaMemset(r,0x53,(count+8)*4));check(cudaMemset(out,0x53,(count+8)*4));
            check(cudaMemset(hr,0x5d,(count+8)*2));check(cudaMemset(ha,0x5d,(count+8)*2));
            check(cudaMemset(inverses,0x53,inv.size()*4));
            const float epsilon=pass==3?1e-5f:1e-6f;
            if(!mollm_cuda::launch_group_rms_norm_half(x,w,r,hr,2560,4,rows,epsilon)||
               !mollm_cuda::launch_group_rms_norm_deferred(x,w,ha,inverses,rows,epsilon,cudaStreamPerThread))return 1;
            check(cudaDeviceSynchronize());
            check(cudaMemcpy(half_ref.data(),hr,(count+8)*2,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(half_actual.data(),ha,(count+8)*2,cudaMemcpyDeviceToHost));
            if(std::memcmp(half_ref.data(),half_actual.data(),(count+8)*2))return 2;
            check(cudaMemcpy(actual.data(),out,(count+8)*4,cudaMemcpyDeviceToHost));
            for(float value:actual){uint32_t bits;std::memcpy(&bits,&value,4);if(bits!=0x53535353u)return 3;}
            if(!mollm_cuda::launch_gr_rms_deferred_materialize(x,w,inverses,out,rows,cudaStreamPerThread))return 4;
            check(cudaDeviceSynchronize());
            check(cudaMemcpy(reference.data(),r,(count+8)*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(actual.data(),out,(count+8)*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(inv.data(),inverses,inv.size()*4,cudaMemcpyDeviceToHost));
            if(std::memcmp(reference.data(),actual.data(),(count+8)*4))return 5;
            for(size_t i=0;i<count;++i)if(!std::isfinite(actual[i]))return 6;
            for(size_t i=0;i<size_t(rows)*4;++i)if(!std::isfinite(inv[i])||inv[i]<=0)return 7;
            for(size_t i=size_t(rows)*4;i<inv.size();++i){uint32_t bits;std::memcpy(&bits,&inv[i],4);if(bits!=0x53535353u)return 8;}
        }
        if(mollm_cuda::launch_group_rms_norm_deferred(x,w,ha,inverses,15,1e-6f)||
           mollm_cuda::launch_group_rms_norm_deferred(x,w,ha,inverses,2049,1e-6f)||
           mollm_cuda::launch_group_rms_norm_deferred(nullptr,w,ha,inverses,rows,1e-6f)||
           mollm_cuda::launch_group_rms_norm_deferred(x,w,reinterpret_cast<uint16_t*>(x),inverses,rows,1e-6f)||
           mollm_cuda::launch_group_rms_norm_deferred(x,w,ha,w,rows,1e-6f)||
           mollm_cuda::launch_group_rms_norm_deferred(x,w,ha,inverses,rows,0.f)||
           mollm_cuda::launch_group_rms_norm_deferred(x,w,ha,inverses,rows,NAN))return 9;
        check(cudaFree(x));check(cudaFree(w));check(cudaFree(r));check(cudaFree(out));
        check(cudaFree(inverses));check(cudaFree(hr));check(cudaFree(ha));
    }
    std::puts("standalone deferred RMS:12 changed-input cases; exact FP16 shadow/FP32 reconstruction, finite inverses, canaries/alias/shape guards PASS");
    return 0;
}
