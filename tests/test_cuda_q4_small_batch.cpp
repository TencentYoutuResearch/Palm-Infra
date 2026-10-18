#include "kernels/cuda/matmul.h"
#include "backends/cuda/backend.h"
#include "graph/graph.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void check(cudaError_t s) {
    if(s != cudaSuccess) {std::fprintf(stderr,"%s\n",cudaGetErrorString(s));std::exit(1);}
}
template<class T> static T* allocate(size_t n) {T* p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class F> static double measure(F f) {
    for(int i=0;i<16;++i)f();
    cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a));
    for(int i=0;i<100;++i)f();
    check(cudaEventRecord(b));check(cudaEventSynchronize(b));float ms;
    check(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms*10;
}
static bool run(int m,int n,int k,bool timing=false,bool wide=false,bool reduced=false,bool exact_reuse=false,
                int stride_padding=4,int alignment_offset=0) {
    const int lda=k+stride_padding,ldc=n+9;
    std::vector<Q4B8G32Block> weights(static_cast<size_t>((n+7)/8)*(k/32));
    for(size_t b=0;b<weights.size();++b)for(int r=0;r<8;++r) {
        weights[b].scales[r]=0.01317f*(1+(b+r)%7);
        for(int p=0;p<16;++p)weights[b].q[r][p]=((b*7+r*3+p)%16)|(((b*3+r+p*5)%16)<<4);
    }
    std::vector<float> x(m*lda,999),reference(m*ldc,-123),actual=reference;
    for(int t=0;t<m;++t)for(int c=0;c<k;++c)x[t*lda+c]=((c*13+t*17)%101-50)/137.f;
    auto*w=allocate<Q4B8G32Block>(weights.size());auto*a_storage=allocate<float>(x.size()+alignment_offset);
    auto*a=a_storage+alignment_offset;auto*y=allocate<float>(actual.size());
    check(cudaMemcpy(w,weights.data(),weights.size()*sizeof(weights[0]),cudaMemcpyHostToDevice));
    check(cudaMemcpy(a,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(y,reference.data(),reference.size()*4,cudaMemcpyHostToDevice));
    auto serial=[&](){for(int t=0;t<m;++t)mollm_cuda::launch_q4_g32_dense_gemv(a+t*lda,w,y+t*ldc,n,k);};
    auto batch=[&](cudaStream_t stream=nullptr){
        auto launch=exact_reuse?mollm_cuda::launch_q4_g32_small_batch_exact:reduced?mollm_cuda::launch_q4_g32_small_batch_reduced:
            wide?mollm_cuda::launch_q4_g32_small_batch_wide:mollm_cuda::launch_q4_g32_small_batch;
        check(launch(a,w,y,m,n,k,lda,ldc,stream));
    };
    auto baseline=[&](){if(reduced)check(mollm_cuda::launch_q4_g32_small_batch(a,w,y,m,n,k,lda,ldc));else serial();};
    baseline();check(cudaMemcpy(reference.data(),y,reference.size()*4,cudaMemcpyDeviceToHost));
    cudaGraph_t graph;cudaGraphExec_t executable;cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    batch(stream);
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int replay=0;replay<2;++replay) {
        if(replay) {
            for(float& value:x)value=-value;
            check(cudaMemcpy(a,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
            baseline();check(cudaMemcpy(reference.data(),y,reference.size()*sizeof(float),cudaMemcpyDeviceToHost));
        }
        std::fill(actual.begin(),actual.end(),-123.f);
        // Nonblocking graph streams do not wait for the default stream's
        // pageable H2D reset. Queue the sentinel on the graph stream itself.
        check(cudaMemcpyAsync(y,actual.data(),actual.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(),y,actual.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<actual.size();++i) {
            const float tol=i%ldc>=static_cast<size_t>(n)?0:2e-5f*(1+std::fabs(reference[i]));
            if(!std::isfinite(actual[i])||std::fabs(actual[i]-reference[i])>tol||
                (!wide&&n<=4096&&k>=2048&&actual[i]!=reference[i])||
                (wide&&n>=9216&&k>=2048&&actual[i]!=reference[i])) {
                std::fprintf(stderr,"mismatch M=%d N=%d K=%d exact=%d padding=%d offset=%d replay=%d i=%zu %g != %g\n",m,n,k,exact_reuse,stride_padding,alignment_offset,replay,i,actual[i],reference[i]);return false;
            }
            if((reduced||exact_reuse)&&actual[i]!=reference[i])return false;
        }
    }
    if(timing) {
        auto*d=allocate<__half>(static_cast<size_t>(n)*k);auto*h=allocate<__half>(static_cast<size_t>(m)*k);
        auto*c=allocate<float>(static_cast<size_t>(m)*n);auto*ax=allocate<float>(static_cast<size_t>(m)*k);
        std::vector<float> host(static_cast<size_t>(m)*k);
        for(int t=0;t<m;++t)std::copy_n(x.data()+t*lda,k,host.data()+t*k);
        check(cudaMemcpy(ax,host.data(),host.size()*4,cudaMemcpyHostToDevice));
        if(k>=2048 && static_cast<int64_t>(n)*k>=4*1024*1024) {
            CudaBackend backend;
            backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
            auto raw=weights;
            for(auto& block:raw)for(auto& row:block.q)for(auto& byte:row)byte^=0x88;
            Tensor weight=Tensor::create(Precision::INT4,MemoryType::EXTERNAL,n,k,1,1,raw.data());
            weight.rowmajor_data=raw.data();weight.q4_g32_data=raw.data();
            weight.is_q4_g32_packed=true;weight.group_size=32;weight.groups_per_row=k/32;
            backend.wrap_weight_int4(weight);
            Tensor input=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,k,m);
            Tensor output=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,n,m);
            backend.alloc_persistent(input,input.nbytes(),PersistentHostAccess::NONE);
            backend.alloc_persistent(output,output.nbytes(),PersistentHostAccess::NONE);
            if(!backend.copy_from_host(host.data(),input,input.nbytes()))return false;
            GraphNode node;node.op_type=OpType::MATMUL;
            backend.dispatch(node,{&input,&weight},&output,nullptr);
            std::vector<float> got(static_cast<size_t>(m)*n);
            if(backend.dispatch_failed()||!backend.copy_to_host(output,got.data(),output.nbytes()))return false;
            for(int t=0;t<m;++t)for(int r=0;r<n;++r)
                if(std::fabs(got[t*n+r]-reference[t*ldc+r])>2e-5f*(1+std::fabs(reference[t*ldc+r])))return false;
        }
        cublasHandle_t handle;if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)return false;
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
        if(cublasSetStream(handle,cudaStreamPerThread)!=CUBLAS_STATUS_SUCCESS)return false;
#endif
        auto gemm=[&](){
            mollm_cuda::launch_prepare_q4_g32_dense_gemm(w,reinterpret_cast<__half2*>(d),ax,h,static_cast<size_t>(m)*k,n,k/32);
            if(!mollm_cuda::run_dense_matmul(handle,d,CUDA_R_16F,h,CUDA_R_16F,c,m,n,k,k))std::exit(1);
        };
        auto old_batch=[&](){check(mollm_cuda::launch_q4_g32_small_batch(a,w,y,m,n,k,lda,ldc));};
        std::printf("wide=%d reduced=%d exact_reuse=%d M=%d N=%d K=%d serial_us=%.2f baseline_batch_us=%.2f batch_us=%.2f expand_gemm_us=%.2f\n",
            wide,reduced,exact_reuse,m,n,k,measure(serial),measure(old_batch),measure(batch),measure(gemm));
        cublasDestroy(handle);cudaFree(d);cudaFree(h);cudaFree(c);cudaFree(ax);
    }
    cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);cudaStreamDestroy(stream);cudaFree(w);cudaFree(a_storage);cudaFree(y);return true;
}
static bool compact_exact(int m, int n, int padding, bool integer_scale, int cache_hint = 0) {
    constexpr int k = 2560;
    const int lda = k + padding, ldc = n + 9;
    std::vector<mollm_cuda::Q4CompactG32> weights(static_cast<size_t>((n+7)/8)*(k/32));
    for (size_t b=0;b<weights.size();++b) for (int r=0;r<8;++r) {
        // Include zero, subnormal maxima, and both normal decoder branches.
        const uint16_t codes[] = {0,1,0x0080,0x0180,0x3f80,0x3fe0,0x4001,0x417f};
        weights[b].scales[r] = codes[(b+r)%8];
        for(int p=0;p<16;++p) weights[b].q[r][p] = ((b+r+p)%16)|(((b*3+r+p*5)%16)<<4);
    }
    std::vector<float> x(m*lda,999.f), reference(m*ldc,-123.f), actual=reference;
    for(int t=0;t<m;++t)for(int c=0;c<k;++c)x[t*lda+c]=((c*13+t*17)%101-50)/137.f;
    auto*w=allocate<mollm_cuda::Q4CompactG32>(weights.size());
    auto*a=allocate<float>(x.size());auto*y=allocate<float>(actual.size());
    check(cudaMemcpy(w,weights.data(),weights.size()*sizeof(weights[0]),cudaMemcpyHostToDevice));
    check(cudaMemcpy(a,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
    if (cache_hint) {
        check(cudaMemcpy(y,actual.data(),actual.size()*sizeof(float),cudaMemcpyHostToDevice));
        if (mollm_cuda::launch_q4_compact_cache_probe(a+1,w,y,n,k,integer_scale,cache_hint)!=cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_compact_cache_probe(a,w,y,4096,k,integer_scale,cache_hint)!=cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_compact_cache_probe(a,w,y,n,4097,integer_scale,cache_hint)!=cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_compact_cache_probe(a,w,y,n,k,integer_scale,10)!=cudaErrorInvalidValue) {
            std::fprintf(stderr,"compact cache probe invalid argument accepted N=%d integer=%d hint=%d\n",n,integer_scale,cache_hint);
            return false;
        }
        check(cudaMemcpy(actual.data(),y,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
        if (std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(float))) return false;
    }
    auto serial=[&](){for(int t=0;t<m;++t) {
        if(integer_scale)check(mollm_cuda::launch_q4_compact_integer_gemv(a+t*lda,w,y+t*ldc,n,k));
        else mollm_cuda::launch_q4_g32_compact_gemv(a+t*lda,w,y+t*ldc,n,k);
    }};
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    if (cache_hint)
        check(mollm_cuda::launch_q4_compact_cache_probe(a,w,y,n,k,integer_scale,cache_hint,stream));
    else check(mollm_cuda::launch_q4_compact_small_batch_exact(a,w,y,m,n,k,lda,ldc,integer_scale,stream));
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int replay=0;replay<2;++replay) {
        if(replay)for(float& value:x)value=-value;
        check(cudaMemcpy(a,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
        std::fill(reference.begin(),reference.end(),-123.f);
        check(cudaMemcpy(y,reference.data(),reference.size()*sizeof(float),cudaMemcpyHostToDevice));
        serial();check(cudaMemcpy(reference.data(),y,reference.size()*sizeof(float),cudaMemcpyDeviceToHost));
        std::fill(actual.begin(),actual.end(),-123.f);
        check(cudaMemcpyAsync(y,actual.data(),actual.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(),y,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<actual.size();++i)if(!std::isfinite(actual[i])||
            std::memcmp(&actual[i],&reference[i],sizeof(float))) {
            std::fprintf(stderr,"compact mismatch M=%d N=%d padding=%d integer=%d replay=%d i=%zu %g != %g\n",
                m,n,padding,integer_scale,replay,i,actual[i],reference[i]);return false;
        }
    }
    cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);cudaStreamDestroy(stream);
    cudaFree(w);cudaFree(a);cudaFree(y);return true;
}
int main() {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    if(mollm_cuda::launch_q4_compact_small_batch_exact(nullptr,nullptr,nullptr,4,8,32,32,8,false)!=cudaErrorInvalidValue)return 1;
    for(int m:{2,3,4,5,7,8})for(int n:{13312,16480})for(int padding:{0,3})
        if(!compact_exact(m,n,padding,n==13312))return 1;
    for(int n:{13312,16480})for(int padding:{0,3})for(int hint:{1,2})
        if(!compact_exact(1,n,padding,n==13312,hint))return 1;
    if (mollm_cuda::launch_q4_compact_cache_probe(nullptr,nullptr,nullptr,16480,2560,false,2)!=cudaErrorInvalidValue) return 1;
    setenv("MOLLM_CUDA_Q4_SMALL_BATCH","1",1);
    if(mollm_cuda::launch_q4_g32_small_batch(nullptr,nullptr,nullptr,4,8,32,32,8)!=cudaErrorInvalidValue)return 1;
    if(mollm_cuda::launch_q4_g32_small_batch_exact(nullptr,nullptr,nullptr,4,8,32,32,8)!=cudaErrorInvalidValue)return 1;
    auto* probe=allocate<float>(128);
    auto* weight=reinterpret_cast<const Q4B8G32Block*>(probe);
    if(mollm_cuda::launch_q4_g32_small_batch_wide(probe,weight,probe,2,1,32,33,1)!=cudaErrorInvalidValue ||
       mollm_cuda::launch_q4_g32_small_batch_wide(probe+1,weight,probe,2,1,32,32,1)!=cudaErrorInvalidValue ||
       mollm_cuda::launch_q4_g32_small_batch_wide(probe,weight,probe,2,INT32_MAX,32,32,INT32_MAX)!=cudaErrorInvalidValue)return 1;
    cudaFree(probe);
    for(int m:{2,3,4,5,7,8})for(int n:{7,320,2560})for(int k:{32,2560})if(!run(m,n,k))return 1;
    for(int m:{4,8})if(!run(m,320,10240,true)||!run(m,10240,2560,true)||!run(m,2560,6144,true)||!run(m,10240,320,true))return 1;
    for(int m:{2,3,4,5,7,8})for(int n:{7,320,8199,10240})
        if(!run(m,n,2560,false,true))return 1;
    for(int m:{4,8})if(!run(m,10240,2560,true,true)||!run(m,20608,2560,true,true))return 1;
    for(int m:{2,3,4,5,7,8})for(int n:{7,320,2560,10240})
        if(!run(m,n,2560,false,false,true))return 1;
    for(int m:{4,8})if(!run(m,10240,2560,true,false,true)||!run(m,20608,2560,true,false,true))return 1;
    for(int m:{2,3,4,5,7,8})for(int n:{7,320,8199,10240})
        if(!run(m,n,2560,false,false,false,true))return 1;
    for(int m:{4,8})if(!run(m,10240,2560,true,false,false,true)||!run(m,2560,6144,true,false,false,true))return 1;
    for(int m:{2,3,8})for(int n:{320,10240})
        if(!run(m,n,2560,false,false,false,true,3)||
           !run(m,n,2560,false,false,false,true,4,1))return 1;
    if(!run(4,320,32,false,false,false,true)||!run(4,10240,320,false,false,false,true))return 1;
    for(int m:{5,7})
        if(!run(m,2560,6144,false,false,false,true)||!run(m,320,10240,false,false,false,true))return 1;
    for(int n:{7,320,8199,10240})if(!run(6,n,2560,false,false,false,true))return 1;
    if(!run(6,2560,6144,false,false,false,true)||!run(6,320,10240,false,false,false,true))return 1;
    std::puts("Q4 small-batch numeric/guard/graph cases passed (including 24 compact fixtures)");return 0;
}
