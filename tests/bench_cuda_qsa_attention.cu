#include "kernels/cuda/qsa.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void check(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);} }
struct Buffer { void* data=nullptr; explicit Buffer(size_t n){check(cudaMalloc(&data,n));} ~Buffer(){cudaFree(data);} };
void bench(int rows,int keys) {
    mollm_cuda::QsaSelectedAttentionParams p;
    p.heads=24;p.kv_heads=2;p.queries=rows;p.keys=keys;p.past=keys-rows;
    p.key_dim=p.value_dim=256;p.slots=2051;p.scale=1.f/16;
    p.query_position_stride=p.output_position_stride=256;
    p.query_head_stride=p.output_head_stride=size_t(rows)*256;
    p.key_position_stride=p.value_position_stride=256;
    p.key_head_stride=p.value_head_stride=size_t(keys)*256;
    p.index_position_stride=p.slots;
    std::vector<float> query(size_t(rows)*24*256);
    for(size_t i=0;i<query.size();++i)query[i]=float(int(i%71)-35)/100.f;
    std::vector<__half> cache(size_t(keys)*2*256);
    for(size_t i=0;i<cache.size();++i)cache[i]=__float2half(float(int(i%61)-30)/100.f);
    std::vector<int32_t> ids(size_t(rows)*p.slots,-1);
    for(int r=0;r<rows;++r)for(int s=0;s<2048;++s)
        ids[size_t(r)*p.slots+s]=int(int64_t(s)*(p.past+r+1)/2048);
    Buffer dq(query.size()*4),dk(cache.size()*2),dv(cache.size()*2),di(ids.size()*4),out(query.size()*4);
    Buffer scores(size_t(rows)*p.heads*p.slots*4);
    check(cudaMemcpy(dq.data,query.data(),query.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dk.data,cache.data(),cache.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dv.data,cache.data(),cache.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(di.data,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
    auto launch=[&] {check(mollm_cuda::launch_qsa_selected_attention(static_cast<float*>(dq.data),dk.data,dv.data,true,
        static_cast<int32_t*>(di.data),static_cast<float*>(out.data),p));};
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    for(int mode:{0,1,2,2,1,0}) {
        p.reuse_gqa=mode==1;p.partitions=mode==2?16:1;
        p.selected_scores=static_cast<float*>(scores.data);
        for(int i=0;i<5;++i)launch();
        check(cudaDeviceSynchronize());
        const int repeats=rows>1?10:100;
        check(cudaEventRecord(begin));for(int i=0;i<repeats;++i)launch();check(cudaEventRecord(end));
        check(cudaEventSynchronize(end));float elapsed=0;check(cudaEventElapsedTime(&elapsed,begin,end));
        std::printf("qsa_attention queries=%d keys=%d mode=%d ms=%.6f\n",rows,keys,mode,elapsed/repeats);
    }
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
}
}
int main(){int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    bench(1,8192);bench(1,32768);bench(32,8192);bench(128,8192);
    bench(512,8192);bench(2048,8192);bench(2048,32768);}
