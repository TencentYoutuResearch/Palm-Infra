#include "kernels/cuda/moe.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void check(cudaError_t e) {if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
std::vector<uint8_t> read(const uint8_t* p,size_t n) {
    std::vector<uint8_t> v(n);check(cudaMemcpy(v.data(),p,n,cudaMemcpyDeviceToHost));return v;
}
}
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    constexpr size_t bytes=1280*160,guard=16;
    uint8_t *src=nullptr,*dst=nullptr;
    check(cudaMalloc(&src,bytes+2*guard));check(cudaMalloc(&dst,bytes+2*guard));
    check(cudaMemset(src,0x5a,bytes+2*guard));check(cudaMemset(dst,0x5a,bytes+2*guard));
    check(cudaDeviceSynchronize());
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    mollm_cuda::Nvfp4Projection source{};source.block_scales=src+guard;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<13;++i) {
        auto bad=source;auto* target=dst+guard;int rows=1280,inner=2560;
        switch(i) {
            case 0:bad.block_scales=nullptr;break;case 1:target=nullptr;break;
            case 2:rows=640;break;case 3:inner=1280;break;
            case 4:bad.mma_layout=true;break;case 5:bad.gate_scales_kmajor=true;break;
            case 6:++bad.block_scales;break;case 7:++target;break;
            case 8:target=src+guard;break;case 9:target=src+2*guard;break;
            case 10:bad.block_scales=dst+2*guard;break;
            case 11:bad.block_scales=reinterpret_cast<const uint8_t*>(UINTPTR_MAX-15);break;
            case 12:target=reinterpret_cast<uint8_t*>(UINTPTR_MAX-15);break;
        }
        if(mollm_cuda::launch_repack_nvfp4_gate_scales(bad,target,rows,inner,stream)!=cudaErrorInvalidValue)return 2;
    }
    cudaGraph_t empty;check(cudaStreamEndCapture(stream,&empty));size_t nodes=0;
    check(cudaGraphGetNodes(empty,nullptr,&nodes));if(nodes)return 3;check(cudaGraphDestroy(empty));
    const auto untouched=read(dst,bytes+2*guard);
    if(!std::all_of(untouched.begin(),untouched.end(),[](uint8_t b){return b==0x5a;}))return 4;
    cudaGraph_t graph;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    check(mollm_cuda::launch_repack_nvfp4_gate_scales(source,dst+guard,1280,2560,stream));
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphGetNodes(graph,nullptr,&nodes));if(nodes!=1)return 5;
    cudaGraphExec_t exec;check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    int host_tests=0;
    for(int variant=0;variant<8;++variant) {
        std::vector<uint8_t> input(bytes+2*guard,0x5a),expected(input);
        for(int row=0;row<1280;++row)for(int group=0;group<160;++group) {
            const uint32_t hash=uint32_t(row*160+group)*0x9e3779b9u+variant*0x85ebca6bu;
            const auto b=uint8_t(hash^(hash>>13)^(hash>>21));
            input[guard+size_t(row)*160+group]=b;
            expected[guard+(size_t(group/4)*1280+row)*4+group%4]=b;
        }
        check(cudaMemcpy(src,input.data(),input.size(),cudaMemcpyHostToDevice));
        check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
        if(read(dst,expected.size())!=expected||read(src,input.size())!=input)return 6;
        int pageable=0,tables=0;check(cudaDeviceGetAttribute(&pageable,cudaDevAttrPageableMemoryAccess,0));
        check(cudaDeviceGetAttribute(&tables,cudaDevAttrPageableMemoryAccessUsesHostPageTables,0));
        if(pageable&&tables) {
            auto staged=source;staged.block_scales=input.data()+guard;
            const auto snapshot=input;
            check(cudaMemsetAsync(dst,0x5a,expected.size(),stream));
            check(mollm_cuda::launch_repack_nvfp4_gate_scales(staged,dst+guard,1280,2560,stream));
            check(cudaStreamSynchronize(stream));
            if(read(dst,expected.size())!=expected||input!=snapshot)return 7;
            ++host_tests;
        }
    }
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    check(cudaFree(src));check(cudaFree(dst));
    std::printf("gate_scale_repack rejected=13 changed_replays=8 pageable_host_staging=%d byte_exact=1 source_immutable=1 guards=1\n",host_tests);
}
