#pragma once
#include "engine/engine.h"
#include <cstdio>

namespace mollm_prefill {
// Bound transient per-batch work, not KV-cache capacity or total prompt length.
// Prefill splits long prompts and executes short/tail batches at their real size.
inline constexpr int logical_max_tokens=2048;
inline bool logical_config_supported(const EngineConfig& c) {
    return c.device==Device::CUDA&&c.offload_ple&&c.cuda_prefill_reclaim&&
        c.weight_loading==WeightLoadingMode::RESIDENT&&!c.moe_ssd_cache_bytes&&
        c.n_ctx>0&&c.prefill_chunk_size>=0&&c.prefill_chunk_size<=logical_max_tokens&&
        !c.static_padded&&!c.use_decode_as_prefill&&!c.mtp_draft_tokens&&c.mtp_companion_dir.empty();
}
inline bool logical_length_supported(int rows,int past,int context) {
    return rows>0&&past>=0&&context>0&&
        past<=context&&rows<=context-past;
}
inline int logical_chunk_limit(const EngineConfig& c) {
    return c.prefill_chunk_size>0?c.prefill_chunk_size:logical_max_tokens;
}
inline bool logical_graph_supported(const Graph& g) {
    int moe=0,attention=0;bool dynamic_hidden=false;
    for(const auto& n:g.nodes) {
        if(n.op_type==OpType::INPUT&&!n.params.str.empty()&&n.params.str[0]=="hidden")
            dynamic_hidden=n.dim_expr[1].kind==DIM_SEQ;
        if(n.op_type==OpType::MOE) {
            if(n.params.i32.size()<4||n.params.i32[0]!=2560||n.params.i32[1]!=512||
                n.params.i32[2]!=10||n.params.i32[3]!=640)return false;
            ++moe;
        }
        if(n.op_type==OpType::SDPA) {
            if(n.params.i32.size()<6||n.params.i32[4]!=256||n.params.i32[5]!=256)return false;
            ++attention;
        }
    }
    return dynamic_hidden&&moe==48&&attention==12;
}
class LogicalBatchScope {
    Backend* backend_=nullptr;
    bool previous_=false;
public:
    bool entered=true;
    LogicalBatchScope(Backend* backend,bool enabled) {
        if(!enabled)return;
        entered=false;if(!backend)return;
        previous_=backend->logical_prefill_batch_enabled();
        if(!backend->set_logical_prefill_batch(true))return;
        backend_=backend;entered=true;
    }
    ~LogicalBatchScope() {
        if(backend_&&!backend_->set_logical_prefill_batch(previous_))
            std::fprintf(stderr,"prefill: failed to restore backend logical batch policy\n");
    }
    LogicalBatchScope(const LogicalBatchScope&)=delete;
    LogicalBatchScope& operator=(const LogicalBatchScope&)=delete;
};
}
