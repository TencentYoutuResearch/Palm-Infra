#include "kernels/cuda/matmul.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#define CK(v) do{auto e=(v);if(e!=cudaSuccess){std::fprintf(stderr,"line%d %s\n",__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
#define NEED(v) do{if(!(v)){std::fprintf(stderr,"scale lookup assertion line%d\n",__LINE__);std::exit(3);}}while(0)
struct Buffer {
    unsigned char *base,*p;size_t bytes;std::vector<unsigned char> original;
    explicit Buffer(size_t n):bytes(n){CK(cudaMalloc(&base,n+512));p=base+256;CK(cudaMemset(base,0x5a,n+512));CK(cudaStreamSynchronize(cudaStreamPerThread));}
    ~Buffer(){cudaFree(base);}
    template<class T>T* as(){return reinterpret_cast<T*>(p);}
    void put(const void* source){CK(cudaMemcpy(p,source,bytes,cudaMemcpyHostToDevice));CK(cudaStreamSynchronize(cudaStreamPerThread));original.assign(static_cast<const unsigned char*>(source),static_cast<const unsigned char*>(source)+bytes);}
    template<class T>std::vector<T> read(){NEED(bytes%sizeof(T)==0);std::vector<T> h(bytes/sizeof(T));CK(cudaMemcpy(h.data(),p,bytes,cudaMemcpyDeviceToHost));return h;}
    void guards(bool immutable=false){unsigned char a[256],b[256];CK(cudaMemcpy(a,base,256,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b,p+bytes,256,cudaMemcpyDeviceToHost));for(int j=0;j<256;++j)NEED(a[j]==0x5a&&b[j]==0x5a);if(immutable)NEED(read<unsigned char>()==original);}
};
// Preserve every finite bit, including signed zero. NaN payloads are not an API contract.
bool same(float a,float b){return(std::isnan(a)&&std::isnan(b))||!std::memcmp(&a,&b,4);}
int main(){
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    Buffer scales(65536*2*sizeof(float));
    NEED(mollm_cuda::launch_q4_compact_scale_check(nullptr,stream)==cudaErrorInvalidValue);
    CK(mollm_cuda::launch_q4_compact_scale_check(scales.as<float>(),stream));CK(cudaStreamSynchronize(stream));auto table=scales.read<float>();
    for(int c=0;c<65536;++c)NEED(same(table[c*2],table[c*2+1]));scales.guards();
    std::puts("scale_lookup_all65536 native_division_exact=1 guards=1");
    int cases=0;
    for(auto shape:{std::pair<int,int>{4129,32},{4129,2560},{4129,4096},{16448,2560},{16480,2560}})for(int weight_offset:{0,4}){
        const int n=shape.first,k=shape.second,modes=k==2560?3:2;
        std::vector<mollm_cuda::Q4CompactG32> weights(size_t((n+7)/8)*(k/32));std::vector<float> input(k);
        Buffer w(weights.size()*sizeof(weights[0])+weight_offset),x(k*4);std::unique_ptr<Buffer> y[3];for(int m=0;m<modes;++m)y[m]=std::make_unique<Buffer>(n*4);
        auto* weight=reinterpret_cast<mollm_cuda::Q4CompactG32*>(w.p+weight_offset);
        NEED(mollm_cuda::launch_q4_compact_scale_gemv_probe(x.as<float>()+1,weight,y[0]->as<float>(),n,k,true,stream)==cudaErrorInvalidValue);
        NEED(mollm_cuda::launch_q4_compact_scale_gemv_probe(x.as<float>(),weight,y[0]->as<float>(),4096,k,true,stream)==cudaErrorInvalidValue);
        NEED(mollm_cuda::launch_q4_compact_scale_gemv_probe(x.as<float>(),weight,y[0]->as<float>(),n,4097,true,stream)==cudaErrorInvalidValue);
        auto launch=[&](int mode){if(mode<2)CK(mollm_cuda::launch_q4_compact_scale_gemv_probe(x.as<float>(),weight,y[mode]->as<float>(),n,k,mode==1,stream));else {mollm_cuda::launch_q4_g32_compact_gemv(x.as<float>(),weight,y[mode]->as<float>(),n,k,stream);CK(cudaGetLastError());}};
        cudaGraph_t graphs[3];cudaGraphExec_t executable[3];for(int m=0;m<modes;++m){CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(m);CK(cudaStreamEndCapture(stream,&graphs[m]));CK(cudaGraphInstantiate(&executable[m],graphs[m],nullptr,nullptr,0));}
        for(int fixture=0;fixture<8;++fixture){
            for(size_t j=0;j<weights.size();++j){for(int r=0;r<8;++r){weights[j].scales[r]=fixture==2||fixture==3?uint16_t(j*8+r):fixture==6?0x8000u:fixture==7?0x7f7fu:uint16_t(0x3c00u+(j*17+r*23)%768);for(int q=0;q<16;++q)weights[j].q[r][q]=uint8_t(j*13+r*31+q*7);}}
            for(int j=0;j<k;++j)input[j]=fixture==1?0.f:std::sin(j*.31f)*.125f;
            if(fixture==2)for(auto& v:input)v*=1e16f;if(fixture==3)for(auto& v:input)v*=1e-40f;
            if(fixture==4)input[17]=NAN;if(fixture==5)input[31]=INFINITY;
            std::vector<unsigned char> storage(w.bytes,0x5a);
            std::memcpy(storage.data()+weight_offset,weights.data(),weights.size()*sizeof(weights[0]));
            w.put(storage.data());x.put(input.data());
            for(int phase=0;phase<2;++phase){for(int mode=0;mode<modes;++mode){CK(cudaMemsetAsync(y[mode]->p,0xa5,y[mode]->bytes,stream));if(phase)CK(cudaGraphLaunch(executable[mode],stream));else launch(mode);}CK(cudaStreamSynchronize(stream));auto ref=y[0]->read<float>();for(int mode=1;mode<modes;++mode){auto got=y[mode]->read<float>();for(int j=0;j<n;++j)NEED(same(ref[j],got[j]));}for(int mode=0;mode<modes;++mode)y[mode]->guards();w.guards(true);x.guards(true);}
            ++cases;std::printf("scale_lookup_case n=%d k=%d fixture=%d weight_offset=%d native_bits=1 mutable_graph=1 guards=1\n",n,k,fixture,weight_offset);
        }
        for(int m=0;m<modes;++m){CK(cudaGraphExecDestroy(executable[m]));CK(cudaGraphDestroy(graphs[m]));}
    }
    CK(cudaStreamDestroy(stream));std::printf("scale_lookup_all cases=%d exact_native=1 mutable_graphs=1 guards=1\n",cases);
}
