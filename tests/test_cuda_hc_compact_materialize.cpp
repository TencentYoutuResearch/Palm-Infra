#include "kernels/cuda/elementwise.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t error) {
    if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(1);}
}
static void require(bool ok) {if(!ok)std::exit(2);}

int main(int argc,char** argv) {
    const bool small=argc==2&&std::strcmp(argv[1],"--small")==0;
    if(argc!=1&&!small)return 3;
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    int cases=0;
    const std::vector<int> geometries=small?std::vector<int>{1,7,17}:std::vector<int>{1,3,7,17,65,719,1024,1025,1743,2048};
    for(int rows:geometries) {
        const size_t count=size_t(rows)*10240,compact_bytes=count*2,logical_bytes=count*4;
        // Both sides of the two spans have guards, including the packed buffer's
        // formerly unused upper half. Captured replay changes the packed input.
        uint32_t* allocation=nullptr;uint16_t* staging=nullptr;
        check(cudaMalloc(&allocation,(count+16)*4));
        check(cudaMalloc(&staging,(count+16)*2));
        float* output=reinterpret_cast<float*>(allocation+8);
        uint16_t* scratch=staging+8;
        std::vector<uint16_t> packed(count),scratch_result(count+16);
        std::vector<uint32_t> result(count+16);
        cudaGraph_t graph;cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        require(mollm_cuda::launch_hc_compact_carry_materialize(output,scratch,compact_bytes,rows,stream));
        check(cudaStreamEndCapture(stream,&graph));
        check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int pass=0;pass<6;++pass) {
            // Rows>=7 exhaust all 65536 encodings, including NaN payloads.
            for(size_t i=0;i<count;++i)packed[i]=uint16_t(i*40503+pass*7919);
            check(cudaMemsetAsync(allocation,0x53,(count+16)*4,stream));
            check(cudaMemsetAsync(staging,0x5d,(count+16)*2,stream));
            check(cudaMemcpyAsync(output,packed.data(),compact_bytes,cudaMemcpyHostToDevice,stream));
            check(cudaGraphLaunch(executable,stream));
            check(cudaMemcpyAsync(result.data(),allocation,(count+16)*4,cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(scratch_result.data(),staging,(count+16)*2,cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            for(size_t i=0;i<count;++i){require(result[i+8]==uint32_t(packed[i])<<16);require(scratch_result[i+8]==packed[i]);}
            for(size_t i=0;i<8;++i){require(result[i]==0x53535353u&&result[count+8+i]==0x53535353u);
                require(scratch_result[i]==0x5d5d&&scratch_result[count+8+i]==0x5d5d);}
            ++cases;
        }
        check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
        // Every invalid request must reject before enqueueing anything. An empty
        // captured graph proves rejection did not secretly schedule copy/decode.
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        const auto reject=[&](float* out,uint16_t* tmp,size_t bytes,int n){
            require(!mollm_cuda::launch_hc_compact_carry_materialize(out,tmp,bytes,n,stream));};
        reject(nullptr,scratch,compact_bytes,rows);reject(output,nullptr,compact_bytes,rows);
        reject(output,scratch,compact_bytes,0);reject(output,scratch,compact_bytes,-1);
        reject(output,scratch,compact_bytes,2049);reject(output,scratch,compact_bytes-1,rows);
        reject(output,reinterpret_cast<uint16_t*>(output),compact_bytes,rows);
        // Scratch in the unused packed tail still aliases the logical FP32 span.
        reject(output,reinterpret_cast<uint16_t*>(output)+count,compact_bytes,rows);
        reject(output,reinterpret_cast<uint16_t*>(reinterpret_cast<uintptr_t>(output)-compact_bytes+2),compact_bytes,rows);
        reject(reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(output)+2),scratch,compact_bytes,rows);
        reject(output,reinterpret_cast<uint16_t*>(reinterpret_cast<uintptr_t>(scratch)+1),compact_bytes,rows);
        reject(reinterpret_cast<float*>(UINTPTR_MAX-3),scratch,compact_bytes,rows);
        reject(output,reinterpret_cast<uint16_t*>(UINTPTR_MAX-1),compact_bytes,rows);
        reject(output,scratch,SIZE_MAX,rows);
        // Even unused declared scratch capacity is forbidden to alias output.
        reject(output,reinterpret_cast<uint16_t*>(reinterpret_cast<uintptr_t>(output)+logical_bytes-2),compact_bytes,rows);
        check(cudaStreamEndCapture(stream,&graph));
        size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));require(nodes==0);
        check(cudaGraphDestroy(graph));
        // Also exercise non-captured execution and immediate adjacent scratch.
        uint16_t* adjacent=nullptr;
        check(cudaMalloc(&adjacent,logical_bytes+compact_bytes));
        float* adjacent_output=reinterpret_cast<float*>(adjacent);
        check(cudaMemcpyAsync(adjacent_output,packed.data(),compact_bytes,cudaMemcpyHostToDevice,stream));
        require(mollm_cuda::launch_hc_compact_carry_materialize(adjacent_output,adjacent+count*2,compact_bytes,rows,stream));
        check(cudaMemcpyAsync(result.data(),adjacent_output,logical_bytes,cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        for(size_t i=0;i<count;++i)require(result[i]==uint32_t(packed[i])<<16);
        ++cases;
        check(cudaFree(adjacent));check(cudaFree(allocation));check(cudaFree(staging));
    }
    check(cudaStreamDestroy(stream));
    std::printf("HC compact restoration: %d byte-exact cases, exhaustive bit patterns, changed capture replays, guards and empty-capture rejection PASS\n",cases);
}
