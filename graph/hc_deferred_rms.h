#pragma once
#include "graph/graph.h"
#include <algorithm>
#include <unordered_map>

// Half-open byte ranges; overflow fails closed. Offset views must invalidate
// the same deferred representation as their full-sized owning tensors.
inline bool hc_byte_ranges_overlap(uintptr_t a,size_t a_bytes,uintptr_t b,size_t b_bytes) {
    if(!a||!b||!a_bytes||!b_bytes)return false;
    if(a_bytes>UINTPTR_MAX-a||b_bytes>UINTPTR_MAX-b)return true;
    return a<b+b_bytes&&b<a+a_bytes;
}

// Structural eligibility only. Device dispatch must also prove fresh half
// shadow, concrete shapes and retained input storage. No graph mutation.
// Three adjacent nodes; the executor must retain up inputs until reduce and
// materialize the original chain if concrete backend dispatch declines.
// Call-scoped index: rebuild after any graph mutation; never cache by pointer.
struct HcUpGraphIndex {
    std::unordered_map<uint32_t,size_t> position;
    std::unordered_map<uint32_t,size_t> users;
    bool valid = true;
    explicit HcUpGraphIndex(const Graph& graph) {
        position.reserve(graph.nodes.size());
        users.reserve(graph.nodes.size());
        for(size_t i=0;i<graph.nodes.size();++i) {
            if(!position.emplace(graph.nodes[i].id,i).second)valid=false;
            for(auto input:graph.nodes[i].inputs)++users[input];
        }
    }
    size_t use_count(uint32_t id) const {
        auto it=users.find(id);return it==users.end()?0:it->second;
    }
};

inline const GraphNode* hc_up_reduce_producer(const Graph& graph,const GraphNode& reduce,
                                             const HcUpGraphIndex& index) {
    if(!index.valid)return nullptr;
    const auto& position=index.position;
    auto at=[&](uint32_t id)->const GraphNode*{auto it=position.find(id);
        return it==position.end()?nullptr:&graph.nodes[it->second];};
    auto exposed=[&](uint32_t id){return std::find(graph.graph_outputs.begin(),graph.graph_outputs.end(),id)!=graph.graph_outputs.end();};
    const auto found=position.find(reduce.id);
    if(found==position.end()||&graph.nodes[found->second]!=&reduce||found->second<2||
        reduce.op_type!=OpType::GR_REDUCE||reduce.inputs.size()!=2||reduce.out_prec!=Precision::FP32||
        reduce.out_shape[0]!=2560||reduce.out_shape[2]!=1||reduce.out_shape[3]!=1||
        graph_params::get_i32(reduce.params,0,0)!=2560||graph_params::get_i32(reduce.params,1,4)!=4)return nullptr;
    const auto& sigmoid=graph.nodes[found->second-1];const auto& up=graph.nodes[found->second-2];
    if(sigmoid.op_type!=OpType::SIGMOID||sigmoid.inputs.size()!=1||sigmoid.inputs[0]!=up.id||
        reduce.inputs[1]!=sigmoid.id||up.op_type!=OpType::MATMUL||up.inputs.size()!=2||
        up.out_prec!=Precision::FP32||sigmoid.out_prec!=Precision::FP32||exposed(up.id)||exposed(sigmoid.id)||
        up.out_shape[0]!=10240||sigmoid.out_shape[0]!=10240||
        up.out_shape[1]!=reduce.out_shape[1]||sigmoid.out_shape[1]!=reduce.out_shape[1]||
        up.out_shape[2]!=1||up.out_shape[3]!=1||sigmoid.out_shape[2]!=1||sigmoid.out_shape[3]!=1||
        graph_params::get_i32(up.params,0,0)!=0||graph_params::get_i32(up.params,1,0)!=0||
        graph_params::get_i32(up.params,2,-1)!=-1)return nullptr;
    const auto* low=at(up.inputs[0]);const auto* weight=at(up.inputs[1]);const auto* norm=at(reduce.inputs[0]);
    if(!low||!weight||!norm||position.at(low->id)>=position.at(up.id)||position.at(weight->id)>=position.at(up.id)||
        position.at(norm->id)>=position.at(up.id)||low->out_prec!=Precision::FP32||low->out_shape[0]!=320||
        low->out_shape[1]!=reduce.out_shape[1]||low->out_shape[2]!=1||low->out_shape[3]!=1||
        weight->op_type!=OpType::CONSTANT||weight->out_prec!=Precision::FP16||
        weight->out_shape[0]!=10240||weight->out_shape[1]!=320||weight->out_shape[2]!=1||weight->out_shape[3]!=1||
        norm->out_prec!=Precision::FP32||norm->out_shape[0]!=10240||norm->out_shape[1]!=reduce.out_shape[1]||
        norm->out_shape[2]!=1||norm->out_shape[3]!=1)return nullptr;
    return index.use_count(up.id)==1&&index.use_count(sigmoid.id)==1?&up:nullptr;
}

inline const GraphNode* hc_up_reduce_producer(const Graph& graph,const GraphNode& reduce) {
    return hc_up_reduce_producer(graph,reduce,HcUpGraphIndex(graph));
}

// Consumer positions retain duplicate edges and graph order, including uses
// outside a proposed fusion. Owned only for one immutable preparation call.
struct HcDeferredGraphIndex : HcUpGraphIndex {
    std::unordered_map<uint32_t,std::vector<size_t>> consumers;
    explicit HcDeferredGraphIndex(const Graph& graph) : HcUpGraphIndex(graph) {
        consumers.reserve(graph.nodes.size());
        for(size_t i=0;i<graph.nodes.size();++i)
            for(auto input:graph.nodes[i].inputs)consumers[input].push_back(i);
    }
    const std::vector<size_t>& uses(uint32_t id) const {
        static const std::vector<size_t> empty;
        auto it=consumers.find(id);return it==consumers.end()?empty:it->second;
    }
};

// A branch-eliminating chain. Only constants/inputs may separate inject/RMS;
// no operator (including PLE/state mutation) may run between MoE and inject.
inline const GraphNode* hc_moe_inject_producer(const Graph& graph,
    const GraphNode& inject,const GraphNode& rms,const HcUpGraphIndex& index) {
    if(!index.valid)return nullptr;
    auto ip=index.position.find(inject.id),rp=index.position.find(rms.id);
    if(ip==index.position.end()||rp==index.position.end()||ip->second==0||
       &graph.nodes[ip->second]!=&inject||&graph.nodes[rp->second]!=&rms||rp->second<=ip->second)return nullptr;
    const auto& moe=graph.nodes[ip->second-1];
    const auto exposed=[&](uint32_t id){return std::find(graph.graph_outputs.begin(),graph.graph_outputs.end(),id)!=graph.graph_outputs.end();};
    if(moe.op_type!=OpType::MOE||moe.inputs.size()!=8||moe.out_prec!=Precision::FP32||
       moe.out_shape[0]!=2560||moe.out_shape[2]!=1||moe.out_shape[3]!=1||
       graph_params::get_i32(moe.params,0,0)!=2560||graph_params::get_i32(moe.params,2,0)!=10||
       exposed(moe.id)||index.use_count(moe.id)!=1||inject.op_type!=OpType::GR_INJECT||
       inject.inputs.size()!=3||inject.inputs[0]!=moe.id||inject.out_prec!=Precision::FP32||
       inject.out_shape[0]!=10240||inject.out_shape[1]!=moe.out_shape[1]||
       inject.out_shape[2]!=1||inject.out_shape[3]!=1||
       graph_params::get_i32(inject.params,0,0)!=2560||graph_params::get_i32(inject.params,1,4)!=4||
       rms.op_type!=OpType::GROUP_RMS_NORM||rms.inputs.size()!=2||rms.inputs[0]!=inject.id||
       rms.out_prec!=Precision::FP32||rms.out_shape[0]!=10240||rms.out_shape[1]!=moe.out_shape[1]||
       rms.out_shape[2]!=1||rms.out_shape[3]!=1||graph_params::get_i32(rms.params,0,0)!=2560)return nullptr;
    for(size_t i=ip->second+1;i<rp->second;++i)
        if(graph.nodes[i].op_type!=OpType::CONSTANT&&graph.nodes[i].op_type!=OpType::INPUT)return nullptr;
    return &moe;
}

inline const GraphNode* hc_deferred_rms_consumer(const Graph& graph,
    const GraphNode& rms,const GraphNode& first,const GraphNode& second,
    const HcDeferredGraphIndex& index) {
    if(!index.valid)return nullptr;
    const auto& position=index.position;
    auto find=[&](uint32_t id)->const GraphNode*{
        auto it=position.find(id);return it==position.end()?nullptr:&graph.nodes[it->second];
    };
    auto exposed=[&](uint32_t id){return std::find(graph.graph_outputs.begin(),graph.graph_outputs.end(),id)!=graph.graph_outputs.end();};
    if(find(rms.id)!=&rms||find(first.id)!=&first||find(second.id)!=&second||
       rms.op_type!=OpType::GROUP_RMS_NORM||rms.inputs.size()!=2||
       rms.out_prec!=Precision::FP32||rms.out_shape[0]!=10240||exposed(rms.id)||
       graph_params::get_i32(rms.params,0,0)!=2560||
       first.op_type!=OpType::MATMUL||second.op_type!=OpType::MATMUL||
       first.inputs.size()!=2||second.inputs.size()!=2||
       first.inputs[0]!=rms.id||second.inputs[0]!=rms.id||
       first.out_shape[0]!=320||second.out_shape[0]!=4||
       !(position.at(rms.id)<position.at(first.id)&&position.at(first.id)<position.at(second.id)))return nullptr;
    const auto* carry=find(rms.inputs[0]);const auto* gamma=find(rms.inputs[1]);
    if(!carry||!gamma||carry->op_type!=OpType::GR_INJECT||carry->inputs.size()!=3||
       carry->out_prec!=Precision::FP32||carry->out_shape[0]!=10240||
       graph_params::get_i32(carry->params,0,0)!=2560||graph_params::get_i32(carry->params,1,4)!=4||
       gamma->op_type!=OpType::CONSTANT||gamma->out_prec!=Precision::FP32||
       gamma->out_shape[0]!=10240||gamma->out_shape[1]!=1||
       position.at(carry->id)>=position.at(rms.id)||position.at(gamma->id)>=position.at(rms.id))return nullptr;
    const GraphNode* reduce=nullptr;int users=0;bool carry_retained=exposed(carry->id);
    for(auto i:index.uses(carry->id))if(i>position.at(second.id))carry_retained=true;
    for(auto i:index.uses(rms.id)){
        const auto& node=graph.nodes[i];
        ++users;
        if(&node==&first||&node==&second)continue;
        if(reduce||node.op_type!=OpType::GR_REDUCE||node.inputs.size()!=2||node.inputs[0]!=rms.id||
           node.out_prec!=Precision::FP32||node.out_shape[0]!=2560||
           graph_params::get_i32(node.params,0,0)!=2560||graph_params::get_i32(node.params,1,4)!=4||
           !(position.at(first.id)<i&&i<position.at(second.id)))return nullptr;
        reduce=&node;
    }
    for(size_t i=position.at(rms.id)+1;i<position.at(second.id);++i){
        const auto& node=graph.nodes[i];
        if(
           node.op_type!=OpType::CONSTANT&&node.op_type!=OpType::MATMUL&&
           node.op_type!=OpType::MUL&&node.op_type!=OpType::SILU&&
           node.op_type!=OpType::SIGMOID&&node.op_type!=OpType::GR_REDUCE)return nullptr;
    }
    if(users!=3||!reduce||!carry_retained)return nullptr;
    const auto* sigmoid=find(reduce->inputs[1]);
    if(!sigmoid||sigmoid->op_type!=OpType::SIGMOID||sigmoid->inputs.size()!=1||
       sigmoid->out_prec!=Precision::FP32||exposed(sigmoid->id)||
       !(position.at(first.id)<position.at(sigmoid->id)&&position.at(sigmoid->id)<position.at(reduce->id)))return nullptr;
    return index.use_count(sigmoid->id)==1?reduce:nullptr;
}

inline const GraphNode* hc_deferred_rms_consumer(const Graph& graph,
    const GraphNode& rms,const GraphNode& first,const GraphNode& second) {
    return hc_deferred_rms_consumer(graph,rms,first,second,HcDeferredGraphIndex(graph));
}
