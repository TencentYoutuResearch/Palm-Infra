#include "backends/cuda/prefill_batch_policy.h"
#include "backends/cpu/backend.h"
#include "engine/logical_prefill.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc,char** argv) {
    for(bool logical:{false,true}) {
        if(mollm_cuda::logical_prefill_option(nullptr,logical)!=logical||
           !mollm_cuda::logical_prefill_option("1",logical))return 27;
        for(const char* disabled:{"","0","2","true"," 1","1 "})
            if(mollm_cuda::logical_prefill_option(disabled,logical))return 28;
    }
    if(argc==2) {
        // Read ONLY the serialized prefill graph, not the weight payload.
        std::ifstream source(argv[1],std::ios::binary|std::ios::ate);
        if(!source)return 20;
        const auto length=source.tellg();source.seekg(0);
        uint32_t prefix[2]{};uint64_t sections[15]{};
        source.read(reinterpret_cast<char*>(prefix),sizeof(prefix));
        source.read(reinterpret_cast<char*>(sections),sizeof(sections));
        const auto offset=sections[6],bytes=sections[7];
        if(!source||prefix[0]!=0x4D4C4F4D||prefix[1]!=1||length<128||
            offset>uint64_t(length)||bytes>uint64_t(length)-offset||bytes>16ull*1024*1024)return 21;
        std::vector<char> data(bytes);source.seekg(offset);source.read(data.data(),bytes);
        if(!source)return 22;
        FILE* temporary=std::tmpfile();if(!temporary)return 23;
        if(std::fwrite(data.data(),1,data.size(),temporary)!=data.size()||std::fflush(temporary)){
            std::fclose(temporary);return 24;
        }
        char path[64];std::snprintf(path,sizeof(path),"/proc/self/fd/%d",fileno(temporary));
        Graph graph;const bool accepted=graph_load(graph,path)&&mollm_prefill::logical_graph_supported(graph);
        std::fclose(temporary);
        std::printf("logical graph package admission prefill_bytes=%llu nodes=%zu accepted=%d weights_read=0\n",
            static_cast<unsigned long long>(bytes),graph.nodes.size(),int(accepted));
        return accepted?0:25;
    }
    if(argc!=1)return 26;
    for(bool logical:{false,true})for(bool merged:{false,true}) {
        const mollm_cuda::Nvfp4PrefillBatchPolicy p{logical,merged};
        for(int rows:{-1,0,1,15,16,1024,1025,1743,2048,2049,4096}) {
            const bool large=logical||merged;
            if(p.fused_rows(rows)!=(rows>=16&&rows<=(large?2048:1024))||
                p.kmajor_rows(rows)!=(rows<=2048&&((logical&&rows>=16)||(merged&&rows>1024))))return 1;
        }
    }
    CPUBackend cpu;
    if(cpu.set_logical_prefill_batch(true)||!cpu.set_logical_prefill_batch(false)||
        cpu.logical_prefill_batch_enabled())return 2;
    EngineConfig c;c.device=Device::CUDA;c.offload_ple=true;c.cuda_prefill_reclaim=true;c.n_ctx=2048;
    if(!mollm_prefill::logical_config_supported(c))return 3;
    for(int context:{1,1024,2048,4096,16384,32768})for(int chunk:{0,1,128,512,957,1024,1743,2047,2048}) {
        auto supported=c;supported.n_ctx=context;supported.prefill_chunk_size=chunk;
        if(!mollm_prefill::logical_config_supported(supported)||
           mollm_prefill::logical_chunk_limit(supported)!=(chunk?chunk:2048))return 17;
    }
    for(int change=0;change<12;++change) {
        auto bad=c;
        switch(change) {
            case 0:bad.device=Device::CPU;break;
            case 1:bad.offload_ple=false;break;
            case 2:bad.cuda_prefill_reclaim=false;break;
            case 3:bad.weight_loading=WeightLoadingMode::MMAP;break;
            case 4:bad.moe_ssd_cache_bytes=1;break;
            case 5:bad.prefill_chunk_size=-1;break;
            case 6:bad.n_ctx=0;break;
            case 7:bad.prefill_chunk_size=2049;break;
            case 8:bad.static_padded=true;break;
            case 9:bad.use_decode_as_prefill=true;break;
            case 10:bad.mtp_draft_tokens=1;break;
            case 11:bad.mtp_companion_dir="test";break;
        }
        if(mollm_prefill::logical_config_supported(bad))return 4;
    }
    for(int context:{2048,16384,32768})for(int rows:{-1,0,1,128,2048,4096,16384})
        for(int past:{-1,0,2048,8192,16384,32768}) {
            const bool expected=rows>0&&past>=0&&past<=context&&rows<=context-past;
            if(mollm_prefill::logical_length_supported(rows,past,context)!=expected)return 18;
        }
    for(int rows:{-1,0,1,1024,1743,2048,2049})for(int past:{-1,0,305,2048}) {
        const bool expected=rows>0&&past>=0&&past<=2048&&rows<=2048-past;
        if(mollm_prefill::logical_length_supported(rows,past,2048)!=expected)return 5;
    }
    Graph graph;graph.nodes.resize(61);
    graph.nodes[0].op_type=OpType::INPUT;graph.nodes[0].params.str={"hidden"};graph.nodes[0].dim_expr[1].kind=DIM_SEQ;
    for(int i=1;i<=48;++i){graph.nodes[i].op_type=OpType::MOE;graph.nodes[i].params.i32={2560,512,10,640};}
    for(int i=49;i<61;++i){graph.nodes[i].op_type=OpType::SDPA;graph.nodes[i].params.i32={2,1,24,2,256,256};}
    if(!mollm_prefill::logical_graph_supported(graph))return 6;
    graph.nodes[1].params.i32[3]=641;if(mollm_prefill::logical_graph_supported(graph))return 7;
    graph.nodes[1].params.i32[3]=640;
    graph.nodes[49].params.i32[4]=128;if(mollm_prefill::logical_graph_supported(graph))return 8;
    graph.nodes[49].params.i32[4]=256;
    graph.nodes[0].dim_expr[1].kind=DIM_CONST;if(mollm_prefill::logical_graph_supported(graph))return 9;
    struct Mock:CPUBackend {
        bool active=false,reject=false;
        bool logical_prefill_batch_enabled() const override{return active;}
        bool set_logical_prefill_batch(bool value) override{if(reject)return false;active=value;return true;}
    } mock;
    {mollm_prefill::LogicalBatchScope s(&mock,true);if(!s.entered||!mock.active)return 10;}
    if(mock.active)return 11;
    mock.active=true;
    {mollm_prefill::LogicalBatchScope s(&mock,true);if(!s.entered||!mock.active)return 12;}
    if(!mock.active)return 13;
    mock.reject=true;
    {mollm_prefill::LogicalBatchScope s(&mock,true);if(s.entered)return 14;}
    {mollm_prefill::LogicalBatchScope s(&cpu,true);if(s.entered)return 15;}
    {mollm_prefill::LogicalBatchScope s(nullptr,false);if(!s.entered)return 16;}
    std::puts("logical/merged/default row boundaries and unsupported backend admission PASS");
}
