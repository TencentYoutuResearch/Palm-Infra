// Private structural probe. No production selector or weight-format change.
#include "kernels/cuda/moe.h"
#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/reduction.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <array>
#include <limits>
#include <type_traits>

static void ck(cudaError_t e) {if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
template<class T> struct Buffer {T* p;size_t n;explicit Buffer(size_t size):n(size){ck(cudaMalloc(&p,(n+8)*sizeof(T)));}~Buffer(){cudaFree(p);}};

template<class T> static void same(Buffer<T>& a,Buffer<T>& b){
    std::vector<T>x(a.n+8),y(b.n+8);ck(cudaMemcpy(x.data(),a.p,x.size()*sizeof(T),cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(y.data(),b.p,y.size()*sizeof(T),cudaMemcpyDeviceToHost));
    if(std::memcmp(x.data(),y.data(),x.size()*sizeof(T))){std::fprintf(stderr,"byte mismatch\n");std::exit(2);}
    const auto* tail=reinterpret_cast<const unsigned char*>(y.data()+b.n);
    for(size_t i=0;i<8*sizeof(T);++i)if(tail[i]!=0x5a)std::exit(3);
}
template<class T> static std::vector<T> fill(Buffer<T>& b,int variant,float magnitude){
    std::vector<T> v(b.n+8);for(size_t i=0;i<b.n;++i){
        const float x=variant==2?0.f:variant==3?(i&1?-0.f:0.f):std::sin(float((i*17+variant*13)%7919)*.037f)*magnitude;
        if constexpr(sizeof(T)==2){float y=x;unsigned u;std::memcpy(&u,&y,4);u+=0x7fff+((u>>16)&1);v[i]=T(u>>16);}
        else v[i]=x;
    }
    std::memset(v.data()+b.n,0x5a,8*sizeof(T));ck(cudaMemcpy(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return v;
}
template<class T>static void unchanged(Buffer<T>& b,const std::vector<T>& original){std::vector<T>v(b.n+8);
    ck(cudaMemcpy(v.data(),b.p,v.size()*sizeof(T),cudaMemcpyDeviceToHost));
    if(std::memcmp(v.data(),original.data(),v.size()*sizeof(T))){std::fprintf(stderr,"source modified\n");std::exit(6);}}
static void run(int rows,bool check_only){
    const char* option=std::getenv("MOLLM_TEST_MOE_HC_PUBLISH_BRANCH");
    const bool publish_branch=!(option&&!std::strcmp(option,"0"));
    const size_t n=size_t(rows)*2560,w=n*4;
    // The production compact API conservatively reserves FP32 owner extents.
    Buffer<uint16_t> routed(n*10),residual(w*2),carry_a(w*2),carry_b(w*2),half_a(w),half_b(w);
    Buffer<float> shared(n),scale(rows),gates(rows*4),gamma(10240),branch(n),branch_b(n),inv_a(rows*4),inv_b(rows*4);
    if(rows==16){
        const std::array<const void*,6> in={routed.p,shared.p,scale.p,residual.p,gates.p,gamma.p};
        const std::array<void*,4> out={branch_b.p,carry_b.p,half_b.p,inv_b.p};
        int rejects=0;
        auto reject=[&](std::array<const void*,6> a,std::array<void*,4> b,int r=16,float eps=1e-6f){
            if(mollm_cuda::launch_moe_combine_gr_inject_rms_deferred(static_cast<const uint16_t*>(a[0]),
                static_cast<const float*>(a[1]),static_cast<const float*>(a[2]),static_cast<const float*>(a[3]),
                static_cast<const float*>(a[4]),static_cast<const float*>(a[5]),static_cast<float*>(b[0]),
                static_cast<float*>(b[1]),static_cast<uint16_t*>(b[2]),static_cast<float*>(b[3]),r,eps,nullptr,publish_branch))std::exit(7);
            ++rejects;};
        ck(cudaMemset(branch_b.p,0x5a,(n+8)*4));ck(cudaMemset(carry_b.p,0x5a,(carry_b.n+8)*2));
        ck(cudaMemset(half_b.p,0x5a,(w+8)*2));ck(cudaMemset(inv_b.p,0x5a,(inv_b.n+8)*4));
        cudaGraph_t invalid;ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        for(int i=0;i<6;++i){auto a=in;a[i]=nullptr;reject(a,out);a=in;
            a[i]=static_cast<const char*>(a[i])+1;reject(a,out);a=in;
            a[i]=reinterpret_cast<const void*>(UINTPTR_MAX-3);reject(a,out);}
        for(int i=0;i<4;++i){auto b=out;b[i]=nullptr;reject(in,b);b=out;
            b[i]=static_cast<char*>(b[i])+1;reject(in,b);b=out;
            b[i]=reinterpret_cast<void*>(UINTPTR_MAX-3);reject(in,b);
            for(int j=0;j<6;++j){b=out;b[i]=const_cast<void*>(in[j]);reject(in,b);}
            for(int j=0;j<i;++j){b=out;b[i]=out[j];reject(in,b);}}
        for(int r:{-1,0,15,2049})reject(in,out,r);
        for(float eps:{0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})reject(in,out,16,eps);
        ck(cudaStreamEndCapture(cudaStreamPerThread,&invalid));size_t nodes=0;ck(cudaGraphGetNodes(invalid,nullptr,&nodes));
        if(nodes)std::exit(8);ck(cudaGraphDestroy(invalid));
        auto sentinel=[](auto& b){using T=typename std::remove_pointer<decltype(b.p)>::type;
            std::vector<T> v(b.n+8);ck(cudaMemcpy(v.data(),b.p,v.size()*sizeof(T),cudaMemcpyDeviceToHost));
            const auto* bytes=reinterpret_cast<const unsigned char*>(v.data());
            for(size_t i=0;i<v.size()*sizeof(T);++i)if(bytes[i]!=0x5a)std::exit(9);};
        sentinel(branch_b);sentinel(carry_b);sentinel(half_b);sentinel(inv_b);
        std::printf("invalid_calls=%d empty capture and output sentinels preserved\n",rejects);
    }
    auto launch=[&](bool candidate){if(candidate){
            if(!mollm_cuda::launch_moe_combine_gr_inject_rms_deferred(routed.p,shared.p,scale.p,
                reinterpret_cast<const float*>(residual.p),gates.p,gamma.p,branch_b.p,
                reinterpret_cast<float*>(carry_b.p),half_b.p,inv_b.p,rows,1e-6f,nullptr,publish_branch))std::exit(5);
        }
        else {ck(mollm_cuda::launch_moe_combine_bf16(routed.p,shared.p,scale.p,branch.p,rows,2560,10));
            if(!mollm_cuda::launch_gr_inject_rms_deferred(branch.p,reinterpret_cast<const float*>(residual.p),gates.p,gamma.p,
                reinterpret_cast<float*>(carry_a.p),half_a.p,inv_a.p,rows,1e-6f,nullptr,true,true))std::exit(4);}
        ck(cudaGetLastError());};
    cudaGraph_t graphs[2];cudaGraphExec_t execs[2];
    for(int c=0;c<2;++c){ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));launch(c);
        ck(cudaStreamEndCapture(cudaStreamPerThread,&graphs[c]));ck(cudaGraphInstantiate(&execs[c],graphs[c],nullptr,nullptr,0));}
    for(int variant=0;variant<4;++variant){auto r=fill(routed,variant,.125f),res=fill(residual,variant,1.f);
        auto sh=fill(shared,variant,.25f),sc=fill(scale,variant,3.f),g=fill(gates,variant,.5f),gm=fill(gamma,variant,1.f);
        ck(cudaMemset(carry_a.p,0x5a,(carry_a.n+8)*2));ck(cudaMemset(carry_b.p,0x5a,(carry_b.n+8)*2));
        ck(cudaMemset(half_a.p,0x5a,(w+8)*2));ck(cudaMemset(half_b.p,0x5a,(w+8)*2));
        ck(cudaMemset(inv_a.p,0x5a,(inv_a.n+8)*4));ck(cudaMemset(inv_b.p,0x5a,(inv_b.n+8)*4));
        ck(cudaMemset(branch.p,0x5a,(n+8)*4));ck(cudaMemset(branch_b.p,0x5a,(n+8)*4));
        for(int repeat=0;repeat<2;++repeat){for(int c=0;c<2;++c)ck(cudaGraphLaunch(execs[c],cudaStreamPerThread));
            ck(cudaDeviceSynchronize());same(carry_a,carry_b);same(half_a,half_b);same(inv_a,inv_b);
            if(publish_branch)same(branch,branch_b);
            else {std::vector<uint32_t> bits(n+8);ck(cudaMemcpy(bits.data(),branch_b.p,bits.size()*4,cudaMemcpyDeviceToHost));
                for(auto v:bits)if(v!=0x5a5a5a5a)std::exit(10);}}
        unchanged(routed,r);unchanged(residual,res);unchanged(shared,sh);unchanged(scale,sc);unchanged(gates,g);unchanged(gamma,gm);
    }
    std::printf("rows=%d publish_branch=%d eight changed captured comparisons byte-exact\n",rows,publish_branch);
    if(!check_only){fill(routed,0,.125f);fill(residual,0,1.f);fill(shared,0,.25f);
        fill(scale,0,3.f);fill(gates,0,.5f);fill(gamma,0,1.f);
        cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
        for(int c=0;c<2;++c)for(int i=0;i<8;++i)ck(cudaGraphLaunch(execs[c],cudaStreamPerThread));
        ck(cudaDeviceSynchronize());
        for(int pair=0;pair<7;++pair){float us[2];for(int j=0;j<2;++j){const int c=j^(pair&1);
            ck(cudaEventRecord(start));for(int i=0;i<32;++i)ck(cudaGraphLaunch(execs[c],cudaStreamPerThread));ck(cudaEventRecord(stop));ck(cudaEventSynchronize(stop));
            float ms;ck(cudaEventElapsedTime(&ms,start,stop));us[c]=ms*1000/32;}
            std::printf("pair=%d baseline_us=%.6f fused_us=%.6f\n",pair,us[0],us[1]);}
        ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));}
    for(int c=0;c<2;++c){ck(cudaGraphExecDestroy(execs[c]));ck(cudaGraphDestroy(graphs[c]));}
}
int main(int argc,char**argv){int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    const bool check_only=argc==2&&!std::strcmp(argv[1],"--check-only");
    run(16,true);
    if(argc==2&&!std::strcmp(argv[1],"--boundaries")){for(int rows:{32,512,1024,1025,1743,2048})run(rows,true);}
    else if(!check_only)run(1743,false);return 0;}
