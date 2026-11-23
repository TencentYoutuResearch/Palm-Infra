// Private lossless-carry diagnostic. No backend, graph, or weight-format change.
#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/reduction.cuh"
#include "runtime/cuda_host_memory_budget.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void ck(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
template<class T>struct Buffer{T* p=nullptr;explicit Buffer(size_t n){ck(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){cudaFree(p);}};
template<class T>static void upload(T* p,const std::vector<T>& x){ck(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>static std::vector<T> read(T* p,size_t n){std::vector<T> x(n);ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
template<class T>static void same(T* a,T* b,size_t n,const char* what){auto x=read(a,n+8),y=read(b,n+8);
    if(std::memcmp(x.data(),y.data(),x.size()*sizeof(T))){std::fprintf(stderr,"compact carry mismatch: %s\n",what);std::exit(2);}
    const auto* tail=reinterpret_cast<const unsigned char*>(y.data()+n);
    for(size_t i=0;i<8*sizeof(T);++i)if(tail[i]!=0x5a)std::exit(3);}
__device__ static float rounded(float x){uint32_t u=__float_as_uint(x);
    if((u&0x7f800000u)!=0x7f800000u)u+=0x7fffu+((u>>16)&1u);
    return __uint_as_float(u&0xffff0000u);}
template<bool CompactResidual>__global__ static void producer(const float* branch,const void* residual,
    const float* gates,const float* gamma,uint16_t* carry,__half* half,float* inverses,float epsilon){
    const size_t base=size_t(blockIdx.x)*2560,row=blockIdx.x/4;
    const int gb=(blockIdx.x%4)*2560;const float gate=gates[blockIdx.x];
    float values[10],sum=0.f;
#pragma unroll
    for(int step=0;step<10;++step){int dim=threadIdx.x+step*256;
        float r;if constexpr(CompactResidual)r=__uint_as_float(uint32_t(static_cast<const uint16_t*>(residual)[base+dim])<<16);
        else r=static_cast<const float*>(residual)[base+dim];
        const float v=rounded(r+gate*branch[row*2560+dim]);values[step]=v;
        carry[base+dim]=uint16_t(__float_as_uint(v)>>16);sum+=v*v;}
    __shared__ float reduction[256];
    const float total=mollm_cuda::detail::block_reduce_sum<256>(sum,reduction);
    const float inverse=rsqrtf(total/2560+epsilon);
    if(threadIdx.x==0)inverses[blockIdx.x]=inverse;
#pragma unroll
    for(int step=0;step<10;++step){int dim=threadIdx.x+step*256;
        half[base+dim]=__float2half(values[step]*inverse*gamma[gb+dim]);}
}
__global__ static void consumer(const uint16_t* carry,const float* logits,const float* gamma,
    const float* inv,float* output,size_t n){
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(index>=n)return;
    const size_t row=index/2560,dim=index%2560;float sum=0.f;
#pragma unroll
    for(int s=0;s<4;++s){const size_t source=(row*4+s)*2560+dim;
        const float value=__uint_as_float(uint32_t(carry[source])<<16);
        const float norm=(value*inv[row*4+s])*gamma[s*2560+dim];
        const float gate=1.f/(1.f+expf(-logits[source]));sum+=gate*norm;}
    output[index]=rounded(sum*.25f);
}
template<int InputPitch=320,int Columns=16>
__global__ void compact_hc_up_reduce_cuda(const __half* input,const __half* weight,
    const uint16_t* carry,const float* gamma,const float* inv,float* output,int rows) {
    using namespace nvcuda;
    constexpr int hidden=2560,k=320;
    const int first_token=blockIdx.y*16,first_dim=blockIdx.x*Columns;
    const int s=threadIdx.x/32;
    __shared__ __align__(32) __half x[16*InputPitch];
    __shared__ __align__(32) float logits[4][16*Columns];
    for(int i=threadIdx.x;i<16*k;i+=128){const int token=first_token+i/k;
        if constexpr(InputPitch==320)
            x[i]=token<rows?input[size_t(token)*k+i%k]:__float2half(0.f);
        else x[(i/k)*InputPitch+i%k]=token<rows?input[size_t(token)*k+i%k]:__float2half(0.f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,float> c;
    wmma::fragment<wmma::accumulator,16,16,16,float> c_second;
    wmma::fill_fragment(c,0.f);
    if constexpr(Columns==32)wmma::fill_fragment(c_second,0.f);
    for(int j=0;j<k;j+=16){
        wmma::load_matrix_sync(a,x+j,InputPitch);
        wmma::load_matrix_sync(b,weight+size_t(s*hidden+first_dim)*k+j,k);
        wmma::mma_sync(c,a,b,c);
        if constexpr(Columns==32){
            wmma::load_matrix_sync(b,weight+size_t(s*hidden+first_dim+16)*k+j,k);
            wmma::mma_sync(c_second,a,b,c_second);
        }
    }
    wmma::store_matrix_sync(logits[s],c,Columns,wmma::mem_row_major);
    if constexpr(Columns==32)wmma::store_matrix_sync(logits[s]+16,c_second,Columns,wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<16*Columns;i+=128){
        const int row=first_token+i/Columns,dim=first_dim+i%Columns;
        if(row>=rows)continue;
        float sum=0.f;
#pragma unroll
        for(int stream=0;stream<4;++stream){
            const size_t source=(size_t(row)*4+stream)*hidden+dim;
            const float recovered=__uint_as_float(uint32_t(carry[source])<<16);
            const float normalized=(recovered*inv[row*4+stream])*gamma[stream*hidden+dim];
            const float gate=1.f/(1.f+expf(-logits[stream][i]));
            sum+=gate*normalized;
        }
        output[size_t(row)*hidden+dim]=rounded(sum*.25f);
    }
}

__global__ static void materialize(const uint16_t* input,float* output,size_t n){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)output[i]=__uint_as_float(uint32_t(input[i])<<16);}
static float time_graph(cudaGraphExec_t g){cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    ck(cudaGraphLaunch(g,cudaStreamPerThread));ck(cudaStreamSynchronize(cudaStreamPerThread));
    ck(cudaEventRecord(a,cudaStreamPerThread));for(int i=0;i<32;++i)ck(cudaGraphLaunch(g,cudaStreamPerThread));
    ck(cudaEventRecord(b,cudaStreamPerThread));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));
    ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));return ms*1000/32;}
int main(int argc,char** argv){
    const bool library=argc>1&&!std::strcmp(argv[argc-1],"--library");
    const int mode_argc=argc-int(library);
    const bool hc_full=mode_argc==2&&!std::strcmp(argv[1],"--hc-full");
    const bool hc_up=hc_full||(mode_argc==2&&(!std::strcmp(argv[1],"--hc-up")||!std::strcmp(argv[1],"--hc-up-checks-only")));
    const bool small=mode_argc==2&&(!std::strcmp(argv[1],"--checks-only")||!std::strcmp(argv[1],"--hc-up-checks-only"));
    if(mode_argc>1&&!small&&!hc_up)return 64;
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB","102400",1);
    if(!mollm_memory::cuda_host_allocation_allowed(2ull*1024*1024*1024,"compact carry bounded diagnostic"))return 4;
    {
        constexpr size_t codes=65536;
        Buffer<uint16_t> raw(codes+8);Buffer<float> restored(codes+8);
        std::vector<uint16_t> bits(codes+8,0x5a5a);std::vector<uint32_t> expected(codes+8,0x5a5a5a5a);
        for(size_t i=0;i<codes;++i){bits[i]=uint16_t(i);expected[i]=uint32_t(i)<<16;}
        upload(raw.p,bits);ck(cudaMemset(restored.p,0x5a,(codes+8)*4));
        materialize<<<codes/256,256,0,cudaStreamPerThread>>>(raw.p,restored.p,codes);ck(cudaGetLastError());ck(cudaStreamSynchronize(cudaStreamPerThread));
        auto actual=read(restored.p,codes+8);auto source=read(raw.p,codes+8);
        if(std::memcmp(actual.data(),expected.data(),expected.size()*4)||std::memcmp(source.data(),bits.data(),bits.size()*2))return 6;
        std::printf("compact-carry exhaustive_restore=65536 including signed/subnormal/Inf/NaN payload bits and guards\n");
    }
    constexpr size_t max_rows=mollm::hc::prefill_max_rows,capacity=max_rows*10240,guard=8;
    Buffer<float> residual(capacity),branch(capacity/4),gates(max_rows*4),gamma(10240),logits(capacity);
    Buffer<float> old0(capacity+guard),old1(capacity+guard),decoded0(capacity+guard),decoded1(capacity+guard);
    // Library flags retain FULL logical FP32 capacity and alias spans.
    Buffer<uint16_t> new0(2*capacity+guard),new1(2*capacity+guard),h0(capacity+guard),h1(capacity+guard),nh0(capacity+guard),nh1(capacity+guard);
    Buffer<float> i0(max_rows*4+guard),i1(max_rows*4+guard),ni0(max_rows*4+guard),ni1(max_rows*4+guard),y(capacity/4+guard),ny(capacity/4+guard);
    Buffer<uint16_t> low(max_rows*320),up_weight(10240*320);
    Buffer<uint16_t> down_weight(hc_full?10240*320:1),full_low0(hc_full?max_rows*320+8:1),full_low1(hc_full?max_rows*320+8:1);
    Buffer<float> down0(hc_full?max_rows*320+8:1),down1(hc_full?max_rows*320+8:1);
    cublasHandle_t handle=nullptr;
    if(hc_full){if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS||cublasSetStream(handle,cudaStreamPerThread)!=CUBLAS_STATUS_SUCCESS||cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH)!=CUBLAS_STATUS_SUCCESS)return 7;
        std::vector<uint16_t> w(10240*320);for(size_t j=0;j<w.size();++j){const __half h=__float2half(std::sin(float(j%7907)*.041f)*.003f);std::memcpy(&w[j],&h,2);}upload(down_weight.p,w);}
    if(hc_up){std::vector<uint16_t> w(10240*320);
        for(size_t j=0;j<w.size();++j){const __half h=__float2half(std::sin(float(j%7919)*.037f)*(.003f+(j%13)*.003f));std::memcpy(&w[j],&h,2);}upload(up_weight.p,w);}
    int cases=0;
    for(int rows:{1,3,17,65,256,719,1024,1025,1743,2048}){if((small&&rows>17)||((hc_up||library)&&rows<16))continue;
        const size_t n=size_t(rows)*10240,out=n/4,inv=size_t(rows)*4;
        const size_t low_count=size_t(rows)*320;
        if(library){
            cudaGraph_t invalid;ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            auto* packed0=reinterpret_cast<float*>(new0.p);auto* packed1=reinterpret_cast<float*>(new1.p);
            const auto rejected=[](bool accepted){if(accepted)std::exit(8);};
            rejected(mollm_cuda::launch_gr_inject_rms_deferred(branch.p,residual.p,gates.p,gamma.p,packed0,nh0.p,ni0.p,15,1e-6f,cudaStreamPerThread,true));
            rejected(mollm_cuda::launch_gr_inject_rms_deferred(branch.p,residual.p,gates.p,gamma.p,packed0,nh0.p,ni0.p,2049,1e-6f,cudaStreamPerThread,true));
            rejected(mollm_cuda::launch_gr_inject_rms_deferred(branch.p,packed0,gates.p,gamma.p,packed1,nh0.p,ni0.p,rows,1e-6f,cudaStreamPerThread,false,true));
            for(float eps:{0.f,NAN})rejected(mollm_cuda::launch_gr_inject_rms_deferred(branch.p,residual.p,gates.p,gamma.p,packed0,nh0.p,ni0.p,rows,eps,cudaStreamPerThread,true));
            rejected(mollm_cuda::launch_gr_inject_rms_deferred(branch.p,packed0,gates.p,gamma.p,packed0+n/2,nh0.p,ni0.p,rows,1e-6f,cudaStreamPerThread,true,true));
            rejected(mollm_cuda::launch_gr_inject_rms_deferred(branch.p,residual.p,gates.p,gamma.p,reinterpret_cast<float*>(new0.p+1),nh0.p,ni0.p,rows,1e-6f,cudaStreamPerThread,true));
            rejected(mollm_cuda::launch_gr_inject_rms_deferred(branch.p,residual.p,gates.p,gamma.p,packed0,new0.p,ni0.p,rows,1e-6f,cudaStreamPerThread,true));
            rejected(mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(packed0,logits.p,gamma.p,ni0.p,packed0+n/2,rows,cudaStreamPerThread,true));
            rejected(mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(packed0,logits.p,gamma.p,ni0.p,ny.p,15,cudaStreamPerThread,true));
            rejected(mollm_cuda::launch_gr_rms_deferred_materialize(packed0,gamma.p,ni0.p,packed0+n/2,rows,cudaStreamPerThread,true));
            rejected(mollm_cuda::launch_gr_rms_deferred_materialize(packed0,gamma.p,ni0.p,decoded1.p,2049,cudaStreamPerThread,true));
            rejected(mollm_cuda::launch_hc_up_reduce_fused(low.p,up_weight.p,packed0,gamma.p,ni0.p,packed0+n/2,rows,cudaStreamPerThread,336,32,true));
            rejected(mollm_cuda::launch_hc_up_reduce_fused(low.p,up_weight.p,packed0,gamma.p,ni0.p,ny.p,rows,cudaStreamPerThread,321,32,true));
            ck(cudaStreamEndCapture(cudaStreamPerThread,&invalid));size_t nodes=0;ck(cudaGraphGetNodes(invalid,nullptr,&nodes));if(nodes)return 8;ck(cudaGraphDestroy(invalid));
        }
        auto down=[&](int mode){const float alpha=1.f,beta=0.f;float* f=mode?down1.p:down0.p;uint16_t* a=mode?full_low1.p:full_low0.p;
            if(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,320,rows,10240,&alpha,down_weight.p,CUDA_R_16F,10240,
                mode?nh0.p:h0.p,CUDA_R_16F,10240,&beta,f,CUDA_R_32F,320,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS)std::exit(7);
            mollm_cuda::launch_fp32_to_fp16(f,reinterpret_cast<__half*>(a),low_count);};
        if(hc_full){ck(cudaMemset(h0.p,0,n*2));ck(cudaMemset(nh0.p,0,n*2));down(0);down(1);ck(cudaStreamSynchronize(cudaStreamPerThread));}
        for(float epsilon:{1e-6f,1e-3f}){
            cudaGraphExec_t graphs[2];
            for(int mode=0;mode<2;++mode){cudaGraph_t g;ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
                if(!mode){
                    if(!mollm_cuda::launch_gr_inject_rms_deferred(branch.p,residual.p,gates.p,gamma.p,old0.p,h0.p,i0.p,rows,epsilon,cudaStreamPerThread))return 5;
                    if(hc_full)down(0);
                    if(hc_up){if(!mollm_cuda::launch_hc_up_reduce_fused(hc_full?full_low0.p:low.p,up_weight.p,old0.p,gamma.p,i0.p,y.p,rows,cudaStreamPerThread,336,32))return 5;}
                    else if(!mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(old0.p,logits.p,gamma.p,i0.p,y.p,rows,cudaStreamPerThread))return 5;
                    if(!mollm_cuda::launch_gr_inject_rms_deferred(y.p,old0.p,gates.p,gamma.p,old1.p,h1.p,i1.p,rows,epsilon,cudaStreamPerThread))return 5;
                }else{
                    if(library){if(!mollm_cuda::launch_gr_inject_rms_deferred(branch.p,residual.p,gates.p,gamma.p,reinterpret_cast<float*>(new0.p),nh0.p,ni0.p,rows,epsilon,cudaStreamPerThread,true,false))return 5;}
                    else producer<false><<<rows*4,256,0,cudaStreamPerThread>>>(branch.p,residual.p,gates.p,gamma.p,new0.p,reinterpret_cast<__half*>(nh0.p),ni0.p,epsilon);
                    if(hc_full)down(1);
                    if(library){
                        if(hc_up){if(!mollm_cuda::launch_hc_up_reduce_fused(hc_full?full_low1.p:low.p,up_weight.p,reinterpret_cast<const float*>(new0.p),gamma.p,ni0.p,ny.p,rows,cudaStreamPerThread,336,32,true))return 5;}
                        else if(!mollm_cuda::launch_gr_reduce_sigmoid_deferred_rms(reinterpret_cast<const float*>(new0.p),logits.p,gamma.p,ni0.p,ny.p,rows,cudaStreamPerThread,true))return 5;
                        if(!mollm_cuda::launch_gr_inject_rms_deferred(ny.p,reinterpret_cast<const float*>(new0.p),gates.p,gamma.p,reinterpret_cast<float*>(new1.p),nh1.p,ni1.p,rows,epsilon,cudaStreamPerThread,true,true))return 5;
                    }else{
                        if(hc_up)compact_hc_up_reduce_cuda<336,32><<<dim3(80,(rows+15)/16),128,0,cudaStreamPerThread>>>(reinterpret_cast<const __half*>(hc_full?full_low1.p:low.p),reinterpret_cast<const __half*>(up_weight.p),new0.p,gamma.p,ni0.p,ny.p,rows);
                        else consumer<<<(out+255)/256,256,0,cudaStreamPerThread>>>(new0.p,logits.p,gamma.p,ni0.p,ny.p,out);
                        producer<true><<<rows*4,256,0,cudaStreamPerThread>>>(ny.p,new0.p,gates.p,gamma.p,new1.p,reinterpret_cast<__half*>(nh1.p),ni1.p,epsilon);
                    }
                }
                ck(cudaGetLastError());ck(cudaStreamEndCapture(cudaStreamPerThread,&g));ck(cudaGraphInstantiateWithFlags(&graphs[mode],g,0));ck(cudaGraphDestroy(g));}
            for(int variant=0;variant<6;++variant){
                if(hc_up){std::vector<uint16_t> a(size_t(rows)*320);
                    for(size_t j=0;j<a.size();++j){const __half h=__float2half(std::cos(float(j%8191)*.071f+variant));std::memcpy(&a[j],&h,2);}upload(low.p,a);}
                std::vector<float> r(n),b(out),gt(inv),w(10240),l(n);
                for(size_t j=0;j<n;++j){r[j]=std::sin(float(j%8191)*.137f+variant)*3;l[j]=std::cos(float(j%8191)*.173f+variant)*12;}
                for(size_t j=0;j<out;++j)b[j]=std::cos(float(j%8191)*.11f+variant)*2;
                for(size_t j=0;j<inv;++j)gt[j]=std::sin(float(j%73)*.17f+variant);
                for(size_t j=0;j<w.size();++j)w[j]=.5f+std::sin(float(j%1023)*.031f+variant)*.3f;
                if(variant>=3){const uint32_t codes[]={0,0x80000000u,0x00010000u,0x80010000u,0x7f800000u,0xff800000u,0x7fc10000u,0xffc20000u};
                    for(int j=0;j<8;++j)std::memcpy(&r[(j*2560)%n],&codes[j],4);
                    std::fill(b.begin(),b.end(),0.f);std::fill(gt.begin(),gt.end(),0.f);}
                upload(residual.p,r);upload(branch.p,b);upload(gates.p,gt);upload(gamma.p,w);upload(logits.p,l);
                for(float* p:{old0.p,old1.p,decoded0.p,decoded1.p})ck(cudaMemset(p,0x5a,(n+guard)*4));
                for(uint16_t* p:{new0.p,new1.p,h0.p,h1.p,nh0.p,nh1.p})ck(cudaMemset(p,0x5a,(n+guard)*2));
                for(float* p:{i0.p,i1.p,ni0.p,ni1.p})ck(cudaMemset(p,0x5a,(inv+guard)*4));
                for(float* p:{y.p,ny.p})ck(cudaMemset(p,0x5a,(out+guard)*4));
                if(hc_full){for(float* p:{down0.p,down1.p})ck(cudaMemset(p,0x5a,(low_count+guard)*4));for(uint16_t* p:{full_low0.p,full_low1.p})ck(cudaMemset(p,0x5a,(low_count+guard)*2));}
                for(auto g:graphs)ck(cudaGraphLaunch(g,cudaStreamPerThread));
                materialize<<<(n+255)/256,256,0,cudaStreamPerThread>>>(new0.p,decoded0.p,n);
                materialize<<<(n+255)/256,256,0,cudaStreamPerThread>>>(new1.p,decoded1.p,n);ck(cudaGetLastError());ck(cudaStreamSynchronize(cudaStreamPerThread));
                same(old0.p,decoded0.p,n,"first carry");same(old1.p,decoded1.p,n,"next carry");
                if(hc_full){same(down0.p,down1.p,low_count,"HC down FP32");same(full_low0.p,full_low1.p,low_count,"HC down FP16 shadow");}
                same(h0.p,nh0.p,n,"first half");same(h1.p,nh1.p,n,"next half");same(i0.p,ni0.p,inv,"first inverse");same(i1.p,ni1.p,inv,"next inverse");same(y.p,ny.p,out,"reduced output");
                if(library){
                    if(!mollm_cuda::launch_gr_rms_deferred_materialize(old0.p,gamma.p,i0.p,decoded0.p,rows,cudaStreamPerThread)||
                       !mollm_cuda::launch_gr_rms_deferred_materialize(reinterpret_cast<const float*>(new0.p),gamma.p,ni0.p,decoded1.p,rows,cudaStreamPerThread,true))return 5;
                    ck(cudaStreamSynchronize(cudaStreamPerThread));same(decoded0.p,decoded1.p,n,"normalized materialization");
                    if(hc_up)for(int pitch:{320,328,336})for(int columns:{16,32}){
                        if(!mollm_cuda::launch_hc_up_reduce_fused(hc_full?full_low0.p:low.p,up_weight.p,old0.p,gamma.p,i0.p,y.p,rows,cudaStreamPerThread,pitch,columns)||
                           !mollm_cuda::launch_hc_up_reduce_fused(hc_full?full_low1.p:low.p,up_weight.p,reinterpret_cast<const float*>(new0.p),gamma.p,ni0.p,ny.p,rows,cudaStreamPerThread,pitch,columns,true))return 5;
                        ck(cudaStreamSynchronize(cudaStreamPerThread));same(y.p,ny.p,out,"HC up geometry");
                    }
                }
                for(auto* p:{new0.p,new1.p}){auto tail=read(p+n,guard);for(auto v:tail)if(v!=0x5a5a)std::exit(3);}
                ++cases;
            }
            // Last finite inputs for timing; do not benchmark exceptional-only tensors.
            if(!small){std::vector<float> r(n),b(out),gt(inv),w(10240),l(n);
                if(hc_up){std::vector<uint16_t> a(size_t(rows)*320);
                    for(size_t j=0;j<a.size();++j){const __half h=__float2half(std::cos(float(j%8191)*.071f));std::memcpy(&a[j],&h,2);}upload(low.p,a);}
                for(size_t j=0;j<n;++j){r[j]=std::sin(float(j%8191)*.137f)*3;l[j]=std::cos(float(j%8191)*.173f)*12;}
                for(size_t j=0;j<out;++j)b[j]=std::cos(float(j%8191)*.11f)*2;
                for(size_t j=0;j<inv;++j)gt[j]=std::sin(float(j%73)*.17f);
                for(size_t j=0;j<w.size();++j)w[j]=.5f+std::sin(float(j%1023)*.031f)*.3f;
                upload(residual.p,r);upload(branch.p,b);upload(gates.p,gt);upload(gamma.p,w);upload(logits.p,l);
                for(auto g:graphs)ck(cudaGraphLaunch(g,cudaStreamPerThread));
                materialize<<<(n+255)/256,256,0,cudaStreamPerThread>>>(new0.p,decoded0.p,n);
                materialize<<<(n+255)/256,256,0,cudaStreamPerThread>>>(new1.p,decoded1.p,n);ck(cudaGetLastError());ck(cudaStreamSynchronize(cudaStreamPerThread));
                same(old0.p,decoded0.p,n,"timed first carry");same(old1.p,decoded1.p,n,"timed next carry");
                if(hc_full){same(down0.p,down1.p,low_count,"timed HC down FP32");same(full_low0.p,full_low1.p,low_count,"timed HC down FP16 shadow");}
                same(h0.p,nh0.p,n,"timed first half");same(h1.p,nh1.p,n,"timed next half");
                same(i0.p,ni0.p,inv,"timed first inverse");same(i1.p,ni1.p,inv,"timed next inverse");same(y.p,ny.p,out,"timed reduced output");++cases;
                for(int round=0;round<7;++round){float a,c;if(round%2){c=time_graph(graphs[1]);a=time_graph(graphs[0]);}else{a=time_graph(graphs[0]);c=time_graph(graphs[1]);}
                    std::printf("compact-carry rows=%d epsilon=%g round=%d float_us=%.3f compact_us=%.3f hc_up=%d hc_full=%d\n",rows,epsilon,round,a,c,int(hc_up),int(hc_full));}}
            for(auto g:graphs)ck(cudaGraphExecDestroy(g));
        }
    }
    if(handle&&cublasDestroy(handle)!=CUBLAS_STATUS_SUCCESS)return 7;
    std::printf("compact-carry checks=%d hc_up=%d hc_full=%d library=%d full byte comparisons including guards and exceptional values; no model integration\n",cases,int(hc_up),int(hc_full),int(library));
}
