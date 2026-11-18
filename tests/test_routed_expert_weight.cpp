#include "engine/routed_expert_weight.h"
#include <cstdio>
#include <cstdlib>

static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
int main() {
    Graph graph;
    for (int i=0;i<8;++i) {
        GraphNode node;node.id=i;node.op_type=OpType::CONSTANT;
        node.out_prec=(i==2||i==3)?Precision::FP16:Precision::FP32;
        node.out_shape[0]=i==2?512:256;node.out_shape[1]=128;
        node.out_shape[2]=node.out_shape[3]=1;graph.nodes.push_back(node);
    }
    GraphNode moe;moe.id=8;moe.op_type=OpType::MOE;moe.inputs={0,1,2,3,4,5,6,7};
    moe.params.i32={128,2,1,128};graph.nodes.push_back(moe);graph.graph_outputs={8};
    auto hints=mollm::routed_fp16_expert_rows(graph);
    require(hints.size()==2&&hints.at(2)==256&&hints.at(3)==128,"valid MoE roles");
    graph.nodes.push_back(moe);graph.nodes.back().id=9;
    require(mollm::routed_fp16_expert_rows(graph)==hints,"same boundary sharing");
    graph.nodes.back().params.i32[0]=256;
    require(mollm::routed_fp16_expert_rows(graph).empty(),"incompatible shared geometry");
    graph.nodes.pop_back();
    GraphNode alias;alias.id=9;alias.op_type=OpType::RESHAPE;alias.inputs={2};
    graph.nodes.push_back(alias);
    auto one=mollm::routed_fp16_expert_rows(graph);
    require(one.size()==1&&one.count(3),"non-MoE alias excludes gate");
    graph.nodes.pop_back();graph.graph_outputs.push_back(3);
    one=mollm::routed_fp16_expert_rows(graph);
    require(one.size()==1&&one.count(2),"exposed constant excluded");
    graph.graph_outputs={8};graph.graph_inputs={2,3};
    require(mollm::routed_fp16_expert_rows(graph).empty(),"external input aliases excluded");
    graph.graph_inputs.clear();graph.nodes[8].inputs.push_back(7);
    require(mollm::routed_fp16_expert_rows(graph).empty(),"unsupported MoE contract");
    graph.nodes[8]=moe;graph.nodes[8].params.i32[3]=INT32_MAX;
    require(mollm::routed_fp16_expert_rows(graph).empty(),"overflow geometry excluded");
    graph.nodes[8]=moe;graph.nodes[2].out_shape[2]=2;
    one=mollm::routed_fp16_expert_rows(graph);
    require(one.size()==1&&one.count(3),"batched constants excluded");
    std::puts("routed expert roles: passed");
}
