#include "kernels/cuda/matmul.h"
#include "kernels/cuda/q8_head_screen.h"
#include "kernels/cuda/elementwise.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CK(call) do { auto status=(call); if(status!=cudaSuccess) { \
    std::fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(status)); std::exit(2); } } while(0)
#define NEED(value) do { if(!(value)) { \
    std::fprintf(stderr,"batch screen assertion at line %d\n",__LINE__); std::exit(3); } } while(0)

struct Buffer {
    unsigned char *base, *data;
    size_t bytes;
    std::vector<unsigned char> original;
    explicit Buffer(size_t n):bytes(n) {
        CK(cudaMalloc(&base,n+512)); data=base+256;
        CK(cudaMemset(base,0x5a,n+512));
        CK(cudaStreamSynchronize(cudaStreamPerThread));
    }
    ~Buffer() { cudaFree(base); }
    template<class T> T* as() { return reinterpret_cast<T*>(data); }
    void upload(const void* source) {
        CK(cudaMemcpy(data,source,bytes,cudaMemcpyHostToDevice));
        CK(cudaStreamSynchronize(cudaStreamPerThread));
        auto* p=static_cast<const unsigned char*>(source);
        original.assign(p,p+bytes);
    }
    void guards(bool immutable=false) {
        std::vector<unsigned char> host(bytes+512);
        CK(cudaMemcpy(host.data(),base,host.size(),cudaMemcpyDeviceToHost));
        for(int i=0;i<256;++i) NEED(host[i]==0x5a && host[bytes+256+i]==0x5a);
        if(immutable) NEED(original.size()==bytes && !std::memcmp(host.data()+256,original.data(),bytes));
    }
    template<class T> std::vector<T> read() {
        NEED(bytes%sizeof(T)==0); std::vector<T> host(bytes/sizeof(T));
        CK(cudaMemcpy(host.data(),data,bytes,cudaMemcpyDeviceToHost)); return host;
    }
};

static uint32_t mix32(uint32_t v) {
    v^=v>>16; v*=0x7feb352du; v^=v>>15; v*=0x846ca68bu; return v^(v>>16);
}
static bool same(float a,float b) {
    return (std::isnan(a)&&std::isnan(b)) || !std::memcmp(&a,&b,sizeof(float));
}

int main(int argc,char** argv) {
    int m=argc==2?std::atoi(argv[1]):3;
    NEED(argc<=2 && (m>=3 && m<=8));
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    constexpr int k=2560,n=263; // Both row-grid tails and distinct queries.
    std::vector<unsigned char> weights(size_t(n)*k);
    std::vector<float> scales(n);
    for(size_t i=0;i<weights.size();++i) {
        size_t j=i/k==1?i-k:i; weights[i]=mix32(uint32_t(j))&255;
    }
    for(int i=0;i<n;++i) scales[i]=i%13==0?-.001f:.001f;
    scales[0]=scales[1]=.001f; // Equal winning rows for the correlation fixture.

    Buffer w(weights.size()),packed(size_t(n)*1920),sf(n*4),x(m*k*4);
    Buffer reference(m*n*4),output(m*n*4),scalar(m*n*4),approx(m*n*4);
    Buffer lower(m*257*4),norm(m*8),count(m*4),indices(m*n*4);
    Buffer scalar_approx(m*n*4),scalar_lower(m*257*4),scalar_norm(m*8);
    Buffer scalar_count(m*4),scalar_indices(m*n*4);
    Buffer partial(256*sizeof(mollm_cuda::ArgMaxPair)),result(sizeof(mollm_cuda::ArgMaxPair));
    w.upload(weights.data()); sf.upload(scales.data());
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CK(mollm_cuda::pack_q8_head_screen(w.as<int8_t>(),packed.data,n,stream));
    CK(cudaStreamSynchronize(stream)); packed.original=packed.read<unsigned char>();
    mollm_cuda::Q8HeadScreenBatch3Scratch scratch{
        approx.as<float>(),lower.as<float>(),norm.as<double>(),count.as<unsigned>(),indices.as<int>()};
    auto screen=[&](const float* source,float* target,int rows,
                    mollm_cuda::Q8HeadScreenBatch3Scratch work) {
        if(m==3)return mollm_cuda::launch_q8_head_screen_batch3(source,w.as<int8_t>(),
            packed.data,sf.as<float>(),target,rows,work,stream);
        mollm_cuda::Q8HeadScreenBatchScratch wide{work.approximate,work.lower,work.norm,
            work.count,work.indices,size_t(m)*n,size_t(m)*257,size_t(m),size_t(m),size_t(m)*n};
        return mollm_cuda::launch_q8_head_screen_batch(source,w.as<int8_t>(),packed.data,
            sf.as<float>(),target,m,rows,wide,stream);
    };
    auto launch=[&](int mode) {
        if(mode==0) {
            CK(mollm_cuda::launch_q8_small_batch_exact(x.as<float>(),w.as<int8_t>(),sf.as<float>(),
                reference.as<float>(),m,n,k,k,n,stream));
        } else if(mode==1) {
            CK(screen(x.as<float>(),output.as<float>(),n,scratch));
        } else {
            for(int t=0;t<m;++t) {
                CK(mollm_cuda::launch_q8_head_screen(x.as<float>()+t*k,w.as<int8_t>(),packed.data,
                    sf.as<float>(),scalar.as<float>()+t*n,n,
                    {scalar_approx.as<float>()+t*n,scalar_lower.as<float>()+t*257,
                     scalar_norm.as<double>()+t,scalar_count.as<unsigned>()+t,
                     scalar_indices.as<int>()+t*n},stream));
            }
        }
    };
    NEED(mollm_cuda::launch_q8_head_screen_batch3(x.as<float>(),w.as<int8_t>(),packed.data,
        sf.as<float>(),output.as<float>(),0,scratch,stream)==cudaErrorInvalidValue);
    NEED(mollm_cuda::launch_q8_head_screen_batch3(x.as<float>()+1,w.as<int8_t>(),packed.data,
        sf.as<float>(),output.as<float>(),n,scratch,stream)==cudaErrorInvalidValue);
    NEED(mollm_cuda::launch_q8_head_screen_batch3(x.as<float>(),w.as<int8_t>(),packed.data+1,
        sf.as<float>(),output.as<float>(),n,scratch,stream)==cudaErrorInvalidValue);
    auto incomplete=scratch; incomplete.lower=nullptr;
    NEED(mollm_cuda::launch_q8_head_screen_batch3(x.as<float>(),w.as<int8_t>(),packed.data,
        sf.as<float>(),output.as<float>(),n,incomplete,stream)==cudaErrorInvalidValue);
    if(m!=3) {
        mollm_cuda::Q8HeadScreenBatchScratch wide{scratch.approximate,scratch.lower,scratch.norm,
            scratch.count,scratch.indices,size_t(m)*n,size_t(m)*257,size_t(m),size_t(m),size_t(m)*n};
        auto invalid=[&](int queries,int rows,mollm_cuda::Q8HeadScreenBatchScratch work){
            NEED(mollm_cuda::launch_q8_head_screen_batch(x.as<float>(),w.as<int8_t>(),packed.data,
                sf.as<float>(),output.as<float>(),queries,rows,work,stream)==cudaErrorInvalidValue);
        };
        for(int width:{0,1,2,3,9,16})invalid(width,n,wide);
        invalid(m,0,wide);invalid(m,INT32_MAX,wide);
        for(int field=0;field<5;++field){auto bad=wide;
            switch(field){case 0:--bad.approximate_capacity;break;case 1:--bad.lower_capacity;break;
                case 2:--bad.norm_capacity;break;case 3:--bad.count_capacity;break;case 4:--bad.indices_capacity;break;}
            invalid(m,n,bad);
        }
        auto bad=wide;bad.norm=reinterpret_cast<double*>(reinterpret_cast<char*>(wide.norm)+4);invalid(m,n,bad);
        bad=wide;bad.lower=nullptr;invalid(m,n,bad);
        NEED(mollm_cuda::launch_q8_head_screen_batch(x.as<float>()+1,w.as<int8_t>(),packed.data,
            sf.as<float>(),output.as<float>(),m,n,wide,stream)==cudaErrorInvalidValue);
        NEED(mollm_cuda::launch_q8_head_screen_batch(x.as<float>(),w.as<int8_t>(),packed.data+1,
            sf.as<float>(),output.as<float>(),m,n,wide,stream)==cudaErrorInvalidValue);
    }
    auto argmax=[&](float* values) {
        mollm_cuda::launch_argmax(values,n,partial.as<mollm_cuda::ArgMaxPair>(),256,
            result.as<mollm_cuda::ArgMaxPair>());
        CK(cudaGetLastError()); mollm_cuda::ArgMaxPair top;
        CK(cudaMemcpy(&top,result.data,sizeof(top),cudaMemcpyDeviceToHost));
        CK(cudaStreamSynchronize(cudaStreamPerThread)); return top.index;
    };
    cudaGraph_t graphs[3]; cudaGraphExec_t executable[3];
    for(int mode=0;mode<3;++mode) {
        CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); launch(mode);
        CK(cudaStreamEndCapture(stream,&graphs[mode]));
        CK(cudaGraphInstantiate(&executable[mode],graphs[mode],nullptr,nullptr,0));
    }
    for(int fixture=0;fixture<20;++fixture) {
        std::vector<float> input(m*k),current_scales=scales;
        for(int t=0;t<m;++t) for(int j=0;j<k;++j) input[t*k+j]=std::sin(j*.31f+t*.07f);
        switch(fixture) {
            case 0:std::fill(input.begin(),input.end(),0.f);break;
            case 1:for(int t=0;t<m;++t) for(int j=0;j<k;++j)
                input[t*k+j]=float(int8_t(weights[j]))/128.f;break;
            case 3:for(auto& v:input)v*=1e16f;break;
            case 4:for(auto& v:input)v*=1e20f;break;
            case 5:for(auto& v:input)v*=1e-40f;break;
            case 6:std::fill(current_scales.begin(),current_scales.end(),1e-40f);break;
            case 7:std::fill(current_scales.begin(),current_scales.end(),1e10f);break;
            case 8:std::fill(current_scales.begin(),current_scales.end(),1e11f);break;
            case 9:input[17]=std::numeric_limits<float>::quiet_NaN();break;
            case 10:input[k+17]=INFINITY;break;
            case 11:current_scales[0]=INFINITY;current_scales[1]=NAN;break;
            case 12:std::fill(input.begin(),input.end(),0.f);
                std::fill(current_scales.begin(),current_scales.end(),std::numeric_limits<float>::max());break;
            case 13:std::fill(current_scales.begin(),current_scales.end(),NAN);break;
            case 15:for(auto& v:input)v*=1e-40f;std::fill(current_scales.begin(),current_scales.end(),1e10f);break;
            case 16:for(auto& v:input)v*=1e-30f;std::fill(current_scales.begin(),current_scales.end(),1e-10f);break;
            case 17:for(auto& v:input)v*=1e18f;std::fill(current_scales.begin(),current_scales.end(),1e-40f);break;
            case 18:case 19:for(size_t j=0;j<input.size();++j)input[j]=(j%2?-1.f:1.f)*(1e20f/2560)*(fixture==18?.999f:1.001f);std::fill(current_scales.begin(),current_scales.end(),-1e-20f);break;
            case 14:std::fill(current_scales.begin(),current_scales.end(),-0.f);break;
        }
        x.upload(input.data()); sf.upload(current_scales.data());
        for(int phase=0;phase<2;++phase) {
            for(int mode=0;mode<3;++mode) {
                if(phase) CK(cudaGraphLaunch(executable[mode],stream)); else launch(mode);
            }
            CK(cudaStreamSynchronize(stream));
            auto ref=reference.read<float>(),got=output.read<float>(),old=scalar.read<float>();
            auto ap=approx.read<float>(),old_ap=scalar_approx.read<float>();
            auto retained=count.read<unsigned>(),old_count=scalar_count.read<unsigned>();
            auto selected=indices.read<int>(); NEED(retained==old_count);
            for(int t=0;t<m;++t) {
                NEED(retained[t]>0 && retained[t]<=n); std::vector<bool> seen(n);
                for(unsigned j=0;j<retained[t];++j) {
                    int row=selected[t*n+j]; NEED(row>=0 && row<n && !seen[row]); seen[row]=true;
                    NEED(same(ref[t*n+row],got[t*n+row]));
                }
                for(int row=0;row<n;++row) {
                    NEED(same(ap[t*n+row],old_ap[t*n+row]));
                    NEED(same(old[t*n+row],got[t*n+row]));
                }
                NEED(argmax(reference.as<float>()+t*n)==argmax(output.as<float>()+t*n));
                if(fixture==0 || fixture==1) NEED(argmax(output.as<float>()+t*n)==0);
            }
            for(Buffer* p:{&output,&reference,&scalar,&approx,&lower,&norm,&count,&indices,
                    &scalar_approx,&scalar_lower,&scalar_norm,&scalar_count,&scalar_indices}) p->guards();
            w.guards(true); packed.guards(true); x.guards(true); sf.guards(true);
        }
        std::printf("batch_screen_case frame=%d eager_graph=1 retained_bits_exact=1 "
            "scalar_screen_equivalent=1 target_argmax_exact=1 guards=1\n",fixture);
    }
    for(int mode=0;mode<3;++mode) {
        CK(cudaGraphExecDestroy(executable[mode])); CK(cudaGraphDestroy(graphs[mode]));
    }
    CK(cudaStreamDestroy(stream));
    std::printf("batch_screen_all cases=20 rows=263 width=%d all_exact_argmax=1 immutable=1 guards=1\n",m);
}
