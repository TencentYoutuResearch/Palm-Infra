#include "kernels/cuda/matmul.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) { if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status)); }
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
template<class T> struct Buffer {
    T* data=nullptr;size_t count;
    explicit Buffer(size_t n):count(n) { check(cudaMalloc(&data,n*sizeof(T))); }
    ~Buffer() { if(data)cudaFree(data); }
    std::vector<T> read(cudaStream_t stream) const {
        std::vector<T> result(count);check(cudaMemcpyAsync(result.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));return result;
    }
};
float float_bits(uint32_t bits) { float value;std::memcpy(&value,&bits,sizeof(value));return value; }
void run() {
    constexpr int inner=10240,down_rows=320,inject_rows=4,guard=32;
    check(mollm_cuda::configure_q4_g32_hc_adjacent_cache());
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::vector<Q4B8G32Block> down(size_t(down_rows/8)*(inner/32)),inject(inner/32);
    uint32_t state=813171;const auto random=[&]() mutable {state^=state<<13;state^=state>>17;state^=state<<5;return state;};
    // Every nibble pattern is valid for the existing biased CUDA BG32 layout.
    auto rng=random;
    for(auto* weights:{&down,&inject})for(auto& block:*weights)for(int row=0;row<8;++row) {
        block.scales[row]=float((rng()%1024)+1)*0.000013f;
        for(auto& code:block.q[row])code=static_cast<uint8_t>(rng());
    }
    Buffer<Q4B8G32Block> wd(down.size()),wi(inject.size());Buffer<float> x(inner+4),expected_down(down_rows+guard),expected_inject(inject_rows+guard),actual_down(down_rows+guard),actual_inject(inject_rows+guard);
    check(cudaMemcpyAsync(wd.data,down.data(),down.size()*sizeof(down[0]),cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(wi.data,inject.data(),inject.size()*sizeof(inject[0]),cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));
    const auto launch=[&](const float* input,const Q4B8G32Block* a,const Q4B8G32Block* b,float* y,float* z) {return mollm_cuda::launch_q4_g32_hc_adjacent_projections(input,a,b,y,z,stream);};
    cudaGraph_t empty;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    require(launch(nullptr,wd.data,wi.data,actual_down.data,actual_inject.data)==cudaErrorInvalidValue,"null input accepted");
    require(launch(x.data,nullptr,wi.data,actual_down.data,actual_inject.data)==cudaErrorInvalidValue,"null down weight accepted");
    require(launch(x.data,wd.data,nullptr,actual_down.data,actual_inject.data)==cudaErrorInvalidValue,"null inject weight accepted");
    require(launch(x.data,wd.data,wi.data,nullptr,actual_inject.data)==cudaErrorInvalidValue,"null down output accepted");
    require(launch(x.data,wd.data,wi.data,actual_down.data,nullptr)==cudaErrorInvalidValue,"null inject output accepted");
    require(launch(x.data+2,wd.data,wi.data,actual_down.data,actual_inject.data)==cudaErrorInvalidValue,"8-byte input accepted");
    require(launch(x.data,reinterpret_cast<const Q4B8G32Block*>(reinterpret_cast<const char*>(wd.data)+1),wi.data,actual_down.data,actual_inject.data)==cudaErrorInvalidValue,"unaligned weight accepted");
    require(launch(x.data,wd.data,wi.data,reinterpret_cast<float*>(reinterpret_cast<char*>(actual_down.data)+1),actual_inject.data)==cudaErrorInvalidValue,"unaligned output accepted");
    check(cudaStreamEndCapture(stream,&empty));size_t nodes=0;check(cudaGraphGetNodes(empty,nullptr,&nodes));require(nodes==0,"invalid calls enqueued work");check(cudaGraphDestroy(empty));
    cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));check(launch(x.data,wd.data,wi.data,actual_down.data,actual_inject.data));check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    std::vector<float> original(inner);for(float& value:original)value=float(int(rng()%2001)-1000)*0.00073f;
    for(int change=0;change<9;++change) {
        auto values=original;
        for(size_t i=0;i<values.size();++i) {
            if(change==1)values[i]=-values[i];
            else if(change==2)values[i]=values[i]*0.5f+float(int(i%13)-6)*0.00017f;
            else if(change==3)values[i]=0.f;
            else if(change==4)values[i]=float_bits(0x80000000u);
            else if(change==5)values[i]=(i%4==0)?float_bits(2):0.f;
        }
        if(change==6)values[13]=float_bits(0x7fc12345u);
        if(change==7)values[17]=float_bits(0x7f800000u);
        if(change==8){values[17]=float_bits(0x7f800000u);values[23]=float_bits(0xff800000u);}
        check(cudaMemcpyAsync(x.data,values.data(),inner*sizeof(float),cudaMemcpyHostToDevice,stream));
        for(auto* buffer:{&expected_down,&expected_inject,&actual_down,&actual_inject})check(cudaMemsetAsync(buffer->data,0x5a,buffer->count*sizeof(float),stream));
        // Independent existing 256-thread kernels supply the exact native reference.
        check(mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(x.data,wd.data,expected_down.data,down_rows,inner,1,true,stream));
        check(mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(x.data,wi.data,expected_inject.data,inject_rows,inner,1,true,stream));
        check(launch(x.data,wd.data,wi.data,actual_down.data,actual_inject.data));
        const auto reference_down=expected_down.read(stream),reference_inject=expected_inject.read(stream);
        auto result_down=actual_down.read(stream),result_inject=actual_inject.read(stream);
        require(!std::memcmp(reference_down.data(),result_down.data(),reference_down.size()*sizeof(float))&&!std::memcmp(reference_inject.data(),result_inject.data(),reference_inject.size()*sizeof(float)),"eager FP32/tree mismatch");
        check(cudaMemsetAsync(actual_down.data,0x5a,actual_down.count*sizeof(float),stream));check(cudaMemsetAsync(actual_inject.data,0x5a,actual_inject.count*sizeof(float),stream));check(cudaGraphLaunch(exec,stream));result_down=actual_down.read(stream);result_inject=actual_inject.read(stream);
        require(!std::memcmp(reference_down.data(),result_down.data(),reference_down.size()*sizeof(float))&&!std::memcmp(reference_inject.data(),result_inject.data(),reference_inject.size()*sizeof(float)),"changed-input graph/guard mismatch");
        for(size_t i=down_rows;i<result_down.size();++i){uint32_t bits;std::memcpy(&bits,&result_down[i],4);require(bits==0x5a5a5a5au,"down guard changed");}
        for(size_t i=inject_rows;i<result_inject.size();++i){uint32_t bits;std::memcpy(&bits,&result_inject[i],4);require(bits==0x5a5a5a5au,"inject guard changed");}
    }
    const auto final_down=wd.read(stream),final_inject=wi.read(stream);require(!std::memcmp(final_down.data(),down.data(),down.size()*sizeof(down[0]))&&!std::memcmp(final_inject.data(),inject.data(),inject.size()*sizeof(inject[0])),"weights changed");
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
}
}
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    try {run();std::puts("HC adjacent: native FP32/NaN/Inf/signed-zero/subnormal, changed graph, guards, immutable weights and empty invalid capture PASS");return 0;}
    catch(const std::exception& error){std::fprintf(stderr,"HC adjacent: %s\n",error.what());return 1;}
}
