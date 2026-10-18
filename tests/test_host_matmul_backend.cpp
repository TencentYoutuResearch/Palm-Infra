#include "backends/factory.h"
#include "runtime/host_matmul_backend.h"
#include "runtime/threading.h"
#include "kernels/cpu/matmul/gguf_k.h"
#include "graph/graph.h"
#include <cstdio>
#include <cstring>
#include <vector>
#ifdef MOLLM_METAL
#include <sys/mman.h>
#include <unistd.h>
#endif
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while(0)
int main() {
    std::string reason;
    auto backend = create_metal_host_matmul_backend(reason);
#ifndef MOLLM_METAL
    CHECK(!backend && !reason.empty());
#else
    CHECK(backend && reason.empty());
    ThreadPool pool(4);
    uint32_t state=6173;
    auto random=[&]{state^=state<<13;state^=state>>17;state^=state<<5;return state;};
    const uint16_t scales[]={0,1,0x03ff,0x0400,0x3000,0xb000,0x4800,0xc800};
    for (int run=0;run<2;++run) {
        const int K=run?768:256,N=run?65:13;
        const size_t bytes=size_t(K/256)*N*176, page=getpagesize();
        const size_t delta=128,length=(bytes+delta+page-1)/page*page;
        void* mapping=mmap(nullptr,length,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
        CHECK(mapping!=MAP_FAILED);
        auto* w=static_cast<uint8_t*>(mapping)+delta;
        for(size_t i=0;i<bytes;++i)w[i]=uint8_t(random());
        for(size_t block=0;block<bytes/176;++block){
            const uint16_t d=scales[block%8],m=scales[(block+3)%8];
            std::memcpy(w+block*176,&d,2);std::memcpy(w+block*176+2,&m,2);
        }
        std::vector<float>x(K),y(N+16,-12345.f),ref(N);
        for(auto& v:x)v=float(int(random()%20001)-10000)*.0001f;
        Tensor a=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,K,1,1,1,x.data());
        Tensor b=Tensor::create(Precision::GGUF_Q5_K,MemoryType::EXTERNAL,N,K,1,1,w);
        Tensor c=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,N,1,1,1,y.data());
        Tensor r=c;r.data=ref.data();
        GraphNode node;node.op_type=OpType::MATMUL;
        std::vector<const Tensor*> inputs{&a,&b};
        CHECK(backend->dispatch(node,inputs,&c,&pool)==HostMatmulResult::Unsupported);
        CHECK(!backend->register_weight(b,mapping,delta+bytes-1));
        CHECK(!backend->register_weight(b,w,bytes)); // cannot prove page envelope
        CHECK(backend->register_weight(b,mapping,length));
        CHECK(backend->register_weight(b,mapping,length));
        kernel_matmul_gguf_k(a,b,r,&pool,Activation::NONE,0,-1);
        for(int repeat=0;repeat<3;++repeat){
            CHECK(backend->dispatch(node,inputs,&c,&pool)==HostMatmulResult::Completed);
            CHECK(std::memcmp(y.data(),ref.data(),N*sizeof(float))==0);
            for(int i=N;i<N+16;++i)CHECK(y[i]==-12345.f);
        }
        CHECK(backend->completed_dispatches()==3);
        a.device.buffer=reinterpret_cast<void*>(1);
        CHECK(backend->dispatch(node,inputs,&c,&pool)==HostMatmulResult::Unsupported);
        a.device.buffer=nullptr;
        a.prec=Precision::FP16;
        CHECK(backend->dispatch(node,inputs,&c,&pool)==HostMatmulResult::Unsupported);
        a.prec=Precision::FP32;
        a.stride[0]=8;CHECK(backend->dispatch(node,inputs,&c,&pool)==HostMatmulResult::Unsupported);a.compute_strides();
        a.shape[1]=2;CHECK(backend->dispatch(node,inputs,&c,&pool)==HostMatmulResult::Unsupported);a.shape[1]=1;
        b.shape[0]--;CHECK(backend->dispatch(node,inputs,&c,&pool)==HostMatmulResult::Unsupported);b.shape[0]++;
        node.params.i32={int(Activation::RELU)};CHECK(backend->dispatch(node,inputs,&c,&pool)==HostMatmulResult::Unsupported);
        CHECK(backend->completed_dispatches()==3);
        backend.reset(); // borrowed Metal views must die before mmap
        CHECK(munmap(mapping,length)==0);
        backend=create_metal_host_matmul_backend(reason);CHECK(backend);
    }
#endif
    std::puts("host matmul backend checks passed");
}
