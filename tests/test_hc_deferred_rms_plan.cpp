#include "graph/hc_deferred_rms.h"
#include <cstdio>
#include <functional>
#include <vector>
#include <chrono>
#include <cstdlib>

// Frozen pre-index matcher: independent behavioral oracle for graph scans.
static const GraphNode* legacy_hc_deferred_rms_consumer(const Graph& graph,
    const GraphNode& rms,const GraphNode& first,const GraphNode& second) {
    std::unordered_map<uint32_t,size_t> position;
    for(size_t i=0;i<graph.nodes.size();++i)
        if(!position.emplace(graph.nodes[i].id,i).second)return nullptr;
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
       !(position[rms.id]<position[first.id]&&position[first.id]<position[second.id]))return nullptr;
    const auto* carry=find(rms.inputs[0]);const auto* gamma=find(rms.inputs[1]);
    if(!carry||!gamma||carry->op_type!=OpType::GR_INJECT||carry->inputs.size()!=3||
       carry->out_prec!=Precision::FP32||carry->out_shape[0]!=10240||
       graph_params::get_i32(carry->params,0,0)!=2560||graph_params::get_i32(carry->params,1,4)!=4||
       gamma->op_type!=OpType::CONSTANT||gamma->out_prec!=Precision::FP32||
       gamma->out_shape[0]!=10240||gamma->out_shape[1]!=1||
       position[carry->id]>=position[rms.id]||position[gamma->id]>=position[rms.id])return nullptr;
    const GraphNode* reduce=nullptr;int users=0;bool carry_retained=exposed(carry->id);
    for(size_t i=0;i<graph.nodes.size();++i){
        const auto& node=graph.nodes[i];
        for(auto input:node.inputs){
            if(input==carry->id&&i>position[second.id])carry_retained=true;
            if(input!=rms.id)continue;
            ++users;
            if(&node==&first||&node==&second)continue;
            if(reduce||node.op_type!=OpType::GR_REDUCE||node.inputs.size()!=2||node.inputs[0]!=rms.id||
               node.out_prec!=Precision::FP32||node.out_shape[0]!=2560||
               graph_params::get_i32(node.params,0,0)!=2560||graph_params::get_i32(node.params,1,4)!=4||
               !(position[first.id]<i&&i<position[second.id]))return nullptr;
            reduce=&node;
        }
        if(i>position[rms.id]&&i<position[second.id]&&
           node.op_type!=OpType::CONSTANT&&node.op_type!=OpType::MATMUL&&
           node.op_type!=OpType::MUL&&node.op_type!=OpType::SILU&&
           node.op_type!=OpType::SIGMOID&&node.op_type!=OpType::GR_REDUCE)return nullptr;
    }
    if(users!=3||!reduce||!carry_retained)return nullptr;
    const auto* sigmoid=find(reduce->inputs[1]);
    if(!sigmoid||sigmoid->op_type!=OpType::SIGMOID||sigmoid->inputs.size()!=1||
       sigmoid->out_prec!=Precision::FP32||exposed(sigmoid->id)||
       !(position[first.id]<position[sigmoid->id]&&position[sigmoid->id]<position[reduce->id]))return nullptr;
    int sigmoid_users=0;
    for(const auto& node:graph.nodes)for(auto id:node.inputs)if(id==sigmoid->id)++sigmoid_users;
    return sigmoid_users==1?reduce:nullptr;
}

static void fixture(Graph& g) {
    g.nodes.resize(15);
    for(int i=0;i<15;++i){g.nodes[i].id=i;g.nodes[i].out_prec=Precision::FP32;}
    auto node=[&](int id,OpType op,std::vector<uint32_t> inputs,int width){
        auto& n=g.nodes[id];n.op_type=op;n.inputs=inputs;n.out_shape[0]=width;
        n.out_shape[1]=17;n.out_shape[2]=n.out_shape[3]=1;
    };
    node(0,OpType::INPUT,{},2560);node(1,OpType::INPUT,{},10240);node(2,OpType::INPUT,{},4);
    node(3,OpType::GR_INJECT,{0,1,2},10240);g.nodes[3].params.i32={2560,4};
    node(4,OpType::CONSTANT,{},10240);g.nodes[4].out_shape[1]=1;
    node(5,OpType::GROUP_RMS_NORM,{3,4},10240);g.nodes[5].params.i32={2560};
    node(6,OpType::CONSTANT,{},320);
    node(7,OpType::MATMUL,{5,6},320);
    node(8,OpType::CONSTANT,{},10240);
    node(9,OpType::MATMUL,{7,8},10240);
    node(10,OpType::SIGMOID,{9},10240);
    node(11,OpType::GR_REDUCE,{5,10},2560);g.nodes[11].params.i32={2560,4};
    node(12,OpType::CONSTANT,{},4);
    node(13,OpType::MATMUL,{5,12},4);
    node(14,OpType::ADD,{3,3},10240);
    g.graph_outputs={7,11,13,14};
}
static bool moe_chain_matcher_checks(){
    Graph base;base.nodes.resize(12);
    for(size_t i=0;i<base.nodes.size();++i){auto& n=base.nodes[i];n.id=i;n.op_type=OpType::INPUT;
        n.out_prec=Precision::FP32;n.out_shape[0]=2560;n.out_shape[1]=-1;}
    auto& moe=base.nodes[8];moe.op_type=OpType::MOE;moe.inputs={0,1,2,3,4,5,6,7};moe.params.i32={2560,512,10};
    auto& inject=base.nodes[9];inject.op_type=OpType::GR_INJECT;inject.inputs={8,0,1};inject.out_shape[0]=10240;inject.params.i32={2560,4};
    base.nodes[10].op_type=OpType::CONSTANT;
    auto& rms=base.nodes[11];rms.op_type=OpType::GROUP_RMS_NORM;rms.inputs={9,10};rms.out_shape[0]=10240;rms.params.i32={2560};
    base.graph_outputs={9,11};
    const HcUpGraphIndex valid(base);
    if(hc_moe_inject_producer(base,inject,rms,valid)!=&moe)return false;
    GraphNode foreign=inject;if(hc_moe_inject_producer(base,foreign,rms,valid))return false;
    const std::vector<std::function<void(Graph&)>> mutations={
        [](Graph& g){g.graph_outputs.push_back(8);},
        [](Graph& g){g.nodes[11].inputs.push_back(8);},
        [](Graph& g){g.nodes[9].inputs.push_back(8);},
        [](Graph& g){g.nodes[0].id=8;},
        [](Graph& g){g.nodes[8].op_type=OpType::ADD;},
        [](Graph& g){g.nodes[8].inputs.pop_back();},
        [](Graph& g){g.nodes[8].params.i32[0]=2048;},
        [](Graph& g){g.nodes[8].params.i32[2]=8;},
        [](Graph& g){g.nodes[8].out_prec=Precision::FP16;},
        [](Graph& g){g.nodes[8].out_shape[2]=2;},
        [](Graph& g){g.nodes[9].inputs[0]=7;},
        [](Graph& g){g.nodes[9].params.i32[1]=8;},
        [](Graph& g){g.nodes[9].out_shape[1]=16;},
        [](Graph& g){g.nodes[9].out_shape[0]=2560;},
        [](Graph& g){g.nodes[9].out_prec=Precision::FP16;},
        [](Graph& g){g.nodes[10].op_type=OpType::PLE_LOOKUP;},
        [](Graph& g){g.nodes[10].op_type=OpType::RESHAPE;},
        [](Graph& g){g.nodes[11].inputs[0]=8;},
        [](Graph& g){g.nodes[11].params.i32[0]=1280;},
        [](Graph& g){g.nodes[11].out_shape[1]=16;},
        [](Graph& g){g.nodes[11].out_shape[3]=2;},
        [](Graph& g){g.nodes[11].out_prec=Precision::FP16;}
    };
    for(const auto& mutate:mutations){Graph g;g.nodes=base.nodes;g.graph_outputs=base.graph_outputs;mutate(g);
        if(hc_moe_inject_producer(g,g.nodes[9],g.nodes[11],HcUpGraphIndex(g)))return false;}
    std::printf("MoE HC graph matcher: unexposed single-use branch, retained carry, foreign identity and %zu mutations checked\n",mutations.size());
    return true;
}

int main() {
    if(!moe_chain_matcher_checks())return 1;
    auto up_fixture=[](Graph& g){fixture(g);g.nodes[8].out_prec=Precision::FP16;g.nodes[8].out_shape[1]=320;};
    const std::vector<std::function<void(Graph&)>> up_negative={
        [](Graph& g){g.graph_outputs.push_back(9);},
        [](Graph& g){g.graph_outputs.push_back(10);},
        [](Graph& g){g.nodes[14].inputs.push_back(9);},
        [](Graph& g){g.nodes[14].inputs.push_back(10);},
        [](Graph& g){g.nodes[9].params.i32={1};},
        [](Graph& g){g.nodes[9].params.i32={0,1};},
        [](Graph& g){g.nodes[9].params.i32={0,0,10};},
        [](Graph& g){g.nodes[9].out_shape[0]=10239;},
        [](Graph& g){g.nodes[7].out_shape[0]=319;},
        [](Graph& g){g.nodes[8].out_shape[1]=321;},
        [](Graph& g){g.nodes[8].out_prec=Precision::FP32;},
        [](Graph& g){g.nodes[8].op_type=OpType::INPUT;},
        [](Graph& g){g.nodes[7].out_prec=Precision::FP16;},
        [](Graph& g){g.nodes[9].inputs[0]=999;},
        [](Graph& g){g.nodes[11].inputs[0]=999;},
        [](Graph& g){g.nodes[11].params.i32={2560,8};},
        [](Graph& g){g.nodes[14].id=13;},
        [](Graph& g){g.nodes[10].out_shape[1]=18;},
        [](Graph& g){g.nodes[5].out_shape[3]=2;},
        [](Graph& g){g.nodes[9].out_shape[2]=2;},
        [](Graph& g){g.nodes[10].inputs[0]=8;},
        [](Graph& g){g.nodes[11].inputs={10,5};}
    };
    for(int mode=-1;mode<int(up_negative.size());++mode){Graph g;up_fixture(g);
        if(mode>=0)up_negative[mode](g);
        const auto* up=hc_up_reduce_producer(g,g.nodes[11]);
        const HcUpGraphIndex index(g);
        if(hc_up_reduce_producer(g,g.nodes[11],index)!=up) {
            std::fprintf(stderr,"HC up indexed structural case%d differs\n",mode);return 1;
        }
        if(mode<0?up!=&g.nodes[9]:up!=nullptr){std::fprintf(stderr,"HC up fusion structural case%d failed\n",mode);return 1;}
    }
    std::printf("HC up fusion chain: 1 eligible, %zu fail-closed cases passed; no executor lifetime proof\n",up_negative.size());
    {
        Graph g;up_fixture(g);
        for(auto& node:g.nodes) {
            node.id=node.id*3+7;
            for(auto& input:node.inputs)input=input*3+7;
        }
        for(auto& output:g.graph_outputs)output=output*3+7;
        const HcUpGraphIndex index(g);
        for(const auto& node:g.nodes)
            if(hc_up_reduce_producer(g,node,index)!=hc_up_reduce_producer(g,node))return 1;
        if(hc_up_reduce_producer(g,g.nodes[11],index)!=&g.nodes[9])return 1;
        GraphNode foreign=g.nodes[11];
        if(hc_up_reduce_producer(g,foreign,index))return 1;
    }
    if(std::getenv("MOLLM_HC_PLAN_BENCH")) {
        Graph g;
        constexpr int chains=192, repeats=12;
        for(int chain=0;chain<chains;++chain) {
            Graph part;up_fixture(part);
            const uint32_t base=static_cast<uint32_t>(g.nodes.size());
            for(auto node:part.nodes) {
                node.id+=base;
                for(auto& input:node.inputs)input+=base;
                g.nodes.push_back(std::move(node));
            }
            for(auto output:part.graph_outputs)g.graph_outputs.push_back(output+base);
        }
        for(int padding=0;padding<2500;++padding) {
            GraphNode node;node.id=static_cast<uint32_t>(g.nodes.size());
            node.op_type=OpType::CONSTANT;g.nodes.push_back(std::move(node));
        }
        auto timing=[&](bool shared) {
            const auto start=std::chrono::steady_clock::now();
            int matches=0;
            for(int repeat=0;repeat<repeats;++repeat) {
                if(shared) {
                    const HcUpGraphIndex index(g);
                    for(const auto& node:g.nodes)if(node.op_type==OpType::GR_REDUCE)
                        matches+=hc_up_reduce_producer(g,node,index)!=nullptr;
                } else {
                    for(const auto& node:g.nodes)if(node.op_type==OpType::GR_REDUCE)
                        matches+=hc_up_reduce_producer(g,node)!=nullptr;
                }
            }
            if(matches!=chains*repeats)std::exit(1);
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeats;
        };
        for(int round=0;round<4;++round) {
            double old_ms,new_ms;
            if(round%2){new_ms=timing(true);old_ms=timing(false);}
            else{old_ms=timing(false);new_ms=timing(true);}
            std::printf("hc-plan round=%d nodes=%zu chains=%d repeated_index_ms=%.6f shared_index_ms=%.6f\n",
                round,g.nodes.size(),chains,old_ms,new_ms);
        }
    }
    if(std::getenv("MOLLM_HC_DEFERRED_PLAN_BENCH")) {
        Graph g;constexpr int chains=192,repeats=12;
        for(int chain=0;chain<chains;++chain) {
            Graph part;fixture(part);const uint32_t base=uint32_t(g.nodes.size());
            for(auto node:part.nodes){node.id+=base;for(auto& input:node.inputs)input+=base;g.nodes.push_back(std::move(node));}
            for(auto output:part.graph_outputs)g.graph_outputs.push_back(output+base);
        }
        for(int padding=0;padding<2500;++padding){GraphNode node;node.id=uint32_t(g.nodes.size());
            node.op_type=OpType::CONSTANT;g.nodes.push_back(std::move(node));}
        auto timing=[&](bool shared) {
            const auto start=std::chrono::steady_clock::now();int matches=0;
            for(int repeat=0;repeat<repeats;++repeat) {
                if(shared) {
                    const HcDeferredGraphIndex index(g);
                    for(int chain=0;chain<chains;++chain){const int b=chain*15;
                        matches+=hc_deferred_rms_consumer(g,g.nodes[b+5],g.nodes[b+7],g.nodes[b+13],index)!=nullptr;}
                } else for(int chain=0;chain<chains;++chain){const int b=chain*15;
                    matches+=legacy_hc_deferred_rms_consumer(g,g.nodes[b+5],g.nodes[b+7],g.nodes[b+13])!=nullptr;}
            }
            if(matches!=chains*repeats)std::exit(1);
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeats;
        };
        for(int round=0;round<4;++round){double old_ms,new_ms;
            if(round%2){new_ms=timing(true);old_ms=timing(false);}else{old_ms=timing(false);new_ms=timing(true);}
            std::printf("hc-deferred-plan round=%d nodes=%zu chains=%d legacy_ms=%.6f shared_index_ms=%.6f\n",
                round,g.nodes.size(),chains,old_ms,new_ms);}
    }
    struct RangeCase {uintptr_t a;size_t as;uintptr_t b;size_t bs;bool expected;};
    const RangeCase ranges[]={
        {100,100,100,100,true},{100,100,140,4,true},{140,4,100,100,true},
        {100,100,0,100,false},{0,100,100,100,false},{100,0,100,100,false},
        {100,100,200,4,false},{200,4,100,100,false},{20,4,100,100,false},
        {100,100,96,8,true},{UINTPTR_MAX-1,4,100,100,true},
        {100,100,UINTPTR_MAX-1,4,true},{UINTPTR_MAX-3,3,UINTPTR_MAX-2,1,true},
        {100,100,196,4,true},{100,100,200,0,false}
    };
    for(const auto& r:ranges)if(hc_byte_ranges_overlap(r.a,r.as,r.b,r.bs)!=r.expected)return 1;
    const std::vector<std::function<void(Graph&)>> negative={
        [](Graph& g){g.graph_outputs.push_back(5);},
        [](Graph& g){g.graph_outputs.push_back(10);},
        [](Graph& g){g.nodes[14].inputs={5,5};},
        [](Graph& g){g.nodes[14].inputs={1,1};},
        [](Graph& g){g.nodes[9].op_type=OpType::SHORTCONV;},
        [](Graph& g){g.nodes[11].params.i32={2560,8};},
        [](Graph& g){g.nodes[11].inputs={10,5};},
        [](Graph& g){g.nodes[13].inputs={1,12};},
        [](Graph& g){g.nodes[10].op_type=OpType::RESHAPE;},
        [](Graph& g){g.nodes[9].inputs.push_back(10);},
        [](Graph& g){g.nodes[3].inputs.pop_back();},
        [](Graph& g){g.nodes[3].op_type=OpType::ADD;},
        [](Graph& g){g.nodes[4].op_type=OpType::INPUT;},
        [](Graph& g){g.nodes[4].out_prec=Precision::FP16;},
        [](Graph& g){g.nodes[5].out_prec=Precision::FP16;},
        [](Graph& g){g.nodes[5].inputs[1]=999;},
        [](Graph& g){g.nodes[14].id=13;},
        [](Graph& g){g.nodes[11].inputs={5,5};}
    };
    for(int mode=-2;mode<int(negative.size());++mode){
        Graph g;fixture(g);
        if(mode>=0)negative[mode](g);
        if(mode==-2){g.nodes[14].inputs={1,1};g.graph_outputs.push_back(3);}
        const auto* result=hc_deferred_rms_consumer(g,g.nodes[5],g.nodes[7],g.nodes[13]);
        const HcDeferredGraphIndex index(g);
        if(hc_deferred_rms_consumer(g,g.nodes[5],g.nodes[7],g.nodes[13],index)!=result)return 1;
        if(legacy_hc_deferred_rms_consumer(g,g.nodes[5],g.nodes[7],g.nodes[13])!=result)return 1;
        if(mode<0?result!=&g.nodes[11]:result!=nullptr){std::fprintf(stderr,"HC deferred eligibility case%d failed\n",mode);return 1;}
    }
    std::printf("HC deferred structural plan: 2 eligible, %zu fail-closed cases, %zu alias-range cases passed; no CUDA runtime proof\n",negative.size(),sizeof(ranges)/sizeof(ranges[0]));
    int mutations=0;
    for(int id=0;id<15;++id)for(int target=0;target<=15;++target) {
        Graph g;fixture(g);g.nodes[id].inputs.push_back(target==15?999:uint32_t(target));
        const HcDeferredGraphIndex index(g);
        if(hc_deferred_rms_consumer(g,g.nodes[5],g.nodes[7],g.nodes[13],index)!=
           legacy_hc_deferred_rms_consumer(g,g.nodes[5],g.nodes[7],g.nodes[13]))return 1;
        ++mutations;
    }
    {
        Graph g;fixture(g);
        for(auto& node:g.nodes){node.id=node.id*3+7;for(auto& input:node.inputs)input=input*3+7;}
        for(auto& output:g.graph_outputs)output=output*3+7;
        const HcDeferredGraphIndex index(g);
        if(hc_deferred_rms_consumer(g,g.nodes[5],g.nodes[7],g.nodes[13],index)!=&g.nodes[11])return 1;
        for(int which=0;which<3;++which) {
            GraphNode foreign=g.nodes[which==0?5:which==1?7:13];
            if(hc_deferred_rms_consumer(g,which==0?foreign:g.nodes[5],
                which==1?foreign:g.nodes[7],which==2?foreign:g.nodes[13],index))return 1;
        }
    }
    std::printf("HC deferred indexed oracle: %d extra-edge mutations, sparse IDs and foreign nodes passed\n",mutations);
}
