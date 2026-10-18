#include "backends/cpu/backend.h"
#include "graph/execute.h"
#include "kernels/cpu/matmul/matmul.h"
#include "runtime/cuda_host_memory_budget.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* label) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}

class TrackingCPU : public CPUBackend {
public:
    int executions = 0;
    int command_buffers = 0;
    void begin_execution() override {
        ++executions;
        CPUBackend::begin_execution();
    }
    void begin_graph() override { ++command_buffers; }
    void end_graph() override { ++command_buffers; }
};

class FusionCPU : public CPUBackend {
public:
    bool enabled = true, reject = false;
    int fused_calls = 0, sigmoid_calls = 0;
    bool supports_sigmoid_gr_reduce_fusion() const override { return enabled; }
    void dispatch(const GraphNode& node, const std::vector<const Tensor*>& inputs,
                  Tensor* output, ThreadPool* pool) override {
        if (node.op_type == OpType::SIGMOID) ++sigmoid_calls;
        CPUBackend::dispatch(node, inputs, output, pool);
    }
    bool dispatch_sigmoid_gr_reduce(const GraphNode& node,
        const std::vector<const Tensor*>& inputs, Tensor* output, ThreadPool* pool) override {
        if (reject) return false;
        check(inputs.size() == 2 && inputs[1] && inputs[1]->data,
              "deferred logits stay live until fused consumer");
        std::vector<float> gates(inputs[1]->nelements());
        Tensor gate = *inputs[1]; gate.data = gates.data();
        gate.mem_type = MemoryType::EXTERNAL;
        GraphNode sigmoid; sigmoid.op_type = OpType::SIGMOID;
        CPUBackend::dispatch(sigmoid, {inputs[1]}, &gate, pool);
        CPUBackend::dispatch(node, {inputs[0], &gate}, output, pool);
        ++fused_calls;
        return true;
    }
};

class HcFusionCPU : public FusionCPU {
public:
    bool hc_enabled=true,hc_reject=false;
    int hc_max_rows=1024;
    int hc_up_reduce_max_rows() const override{return hc_max_rows;}
    int attempts=0,hc_calls=0,up_calls=0;
    bool supports_hc_up_reduce_fusion() const override{return hc_enabled;}
    static void fill_up(const Tensor& low,Tensor& out){
        for(int64_t r=0;r<out.shape[1];++r)for(int d=0;d<10240;++d)
            out.ptr<float>()[r*10240+d]=low.ptr<float>()[r*320+d%320]*.125f+
                (d%13-6)*.03125f;
    }
    void dispatch(const GraphNode& node,const std::vector<const Tensor*>& inputs,
        Tensor* output,ThreadPool* pool) override{
        if(node.op_type==OpType::MATMUL){++up_calls;fill_up(*inputs[0],*output);return;}
        FusionCPU::dispatch(node,inputs,output,pool);
    }
    bool dispatch_hc_up_reduce(const GraphNode&,const GraphNode& reduce,
        const std::vector<const Tensor*>& inputs,Tensor* output,ThreadPool* pool) override{
        ++attempts;
        check(inputs.size()==3&&inputs[0]->data&&inputs[1]->data&&inputs[2]->data,
            "HC deferred low-rank input and weight remain live through reduce");
        if(hc_reject)return false;
        std::vector<float> raw(size_t(output->shape[1])*10240),gates(raw.size());
        Tensor up=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,10240,
            output->shape[1],1,1,raw.data());
        Tensor gate=up;gate.data=gates.data();fill_up(*inputs[1],up);
        GraphNode sigmoid;sigmoid.op_type=OpType::SIGMOID;
        CPUBackend::dispatch(sigmoid,{&up},&gate,pool);
        CPUBackend::dispatch(reduce,{inputs[0],&gate},output,pool);++hc_calls;return true;
    }
};

void test_hc_up_lifecycle(){
    for(int mode=0;mode<19;++mode){
        const bool borrowed=mode==8,extra=mode==7;
        Graph graph;graph.nodes.resize(extra?9:8);graph.runtime.tensors.resize(graph.nodes.size());
        for(size_t id=0;id<graph.nodes.size();++id){
            auto& n=graph.nodes[id];n.id=id;n.out_shape[0]=(id==1||id==3)?320:
                (id==6||id==7)?2560:10240;n.out_shape[1]=17;n.dim_expr[1].kind=DIM_SEQ;
            graph.runtime.tensors[id]=Tensor::create(Precision::FP32,MemoryType::NONE,
                n.out_shape[0],17,1,1,nullptr);
        }
        graph.nodes[0].op_type=graph.nodes[1].op_type=OpType::INPUT;
        auto& w=graph.nodes[2];w.op_type=OpType::CONSTANT;w.out_prec=Precision::FP16;
        w.out_shape[1]=320;w.dim_expr[1].kind=DIM_CONST;
        uint16_t dummy_weight=0;
        graph.runtime.tensors[2]=Tensor::create(Precision::FP16,MemoryType::EXTERNAL,
            10240,320,1,1,&dummy_weight); // mock never reads matrix payload
        graph.nodes[3].op_type=borrowed?OpType::RESHAPE:OpType::ADD;
        graph.nodes[3].inputs=borrowed?std::vector<uint32_t>{1}:std::vector<uint32_t>{1,1};
        graph.nodes[4].op_type=OpType::MATMUL;graph.nodes[4].inputs={3,2};
        graph.nodes[4].params.i32={0,0,-1};
        graph.nodes[5].op_type=OpType::SIGMOID;graph.nodes[5].inputs={4};
        graph.nodes[6].op_type=OpType::GR_REDUCE;graph.nodes[6].inputs={0,5};
        graph.nodes[6].params.i32={2560,4};
        graph.nodes[7].op_type=OpType::ADD;graph.nodes[7].inputs={6,6};
        graph.graph_inputs={0,1};graph.graph_outputs={7};
        if(mode==3)graph.graph_outputs.push_back(4);
        if(mode==4)graph.graph_outputs.push_back(5);
        if(extra){graph.nodes[8].op_type=OpType::ADD;graph.nodes[8].inputs={5,5};graph.graph_outputs.push_back(8);}
        HcFusionCPU backend;backend.hc_reject=mode==1||mode==10;backend.hc_enabled=mode!=2;
        if(mode==17)backend.hc_max_rows=2048;
        if(mode==10)backend.enabled=false; // reject new fusion without old two-node fallback
        BufferPool pool;ExecContext ctx;ctx.graph=&graph;ctx.backend=&backend;ctx.pool=&pool;
        ctx.profile_enabled=mode==5;ctx.reuse_same_shape_workspace=mode==9;
        ctx.segment_fusions_enabled=mode>=14&&mode<=16;
        prepare_execution(ctx);
        const bool eligible=mode==0||mode==1||mode==8||mode==9||mode==10||mode==14||mode==15||mode==17;
        for(int pass=0;pass<3;++pass){
            const int rows=mode>=17?(pass==0?1025:pass==1?1743:2048):mode==11?1:mode==9?17:pass==1?19:17;
            std::vector<float> norm(size_t(rows)*10240),low(size_t(rows)*320),prepared(low.size());
            for(size_t i=0;i<norm.size();++i)norm[i]=std::sin(i*.013f+pass);
            for(size_t i=0;i<low.size();++i){low[i]=std::cos(i*.017f+pass);prepared[i]=borrowed?low[i]:low[i]+low[i];}
            graph.runtime.tensors[0]=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,10240,rows,1,1,norm.data());
            graph.runtime.tensors[1]=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,320,rows,1,1,low.data());
            ctx.runtime_seq_len=rows;inject_runtime_shapes(ctx);
            if(mode>=12&&mode<=16){const int cut=mode==14?4:mode==15?7:mode==13?6:5;
                execute_graph_segment(ctx,0,cut);execute_graph_segment(ctx,cut,8);}
            else execute_graph(ctx,mode==6?4:-1);
            check(!ctx.execution_failed,"HC triple fusion lifecycle succeeds");
            std::vector<float> raw(size_t(rows)*10240),gates(raw.size()),expected(size_t(rows)*2560);
            Tensor a=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,320,rows,1,1,prepared.data());
            Tensor u=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,10240,rows,1,1,raw.data());
            Tensor s=u;s.data=gates.data();HcFusionCPU::fill_up(a,u);
            if(mode==6)check(std::memcmp(graph.runtime.tensors[4].data,raw.data(),raw.size()*4)==0,
                "partial execution preserves ordinary up output");
            else{
                Tensor o=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,2560,rows,1,1,expected.data());
                CPUBackend reference;reference.dispatch(graph.nodes[5],{&u},&s,nullptr);
                reference.dispatch(graph.nodes[6],{&graph.runtime.tensors[0],&s},&o,nullptr);
                for(auto& value:expected)value+=value;
                check(std::memcmp(graph.runtime.tensors[7].data,expected.data(),expected.size()*4)==0,
                    "HC fused/fallback dynamic result bytes match original graph");
            }
            const bool accepted=eligible&&mode!=1&&mode!=10;
            check(backend.hc_calls==(accepted?pass+1:0),"only eligible HC prefill chain fuses");
            check(backend.attempts==(eligible?pass+1:0),"HC diagnostics, exported intermediates and decode do not defer");
            check(backend.up_calls==(accepted?0:pass+1),"HC rejection reconstructs original up before sigmoid/reduce");
            if(accepted)check(!graph.runtime.tensors[4].data&&!graph.runtime.tensors[5].data,
                "successful HC fusion does not allocate wide intermediates");
            if(mode!=9)check(!graph.runtime.tensors[3].data,
                "deferred low-rank storage/view released at completed reduce");
        }
    }
    std::printf("HC triple execution lifecycle: 19 modes x3 passes; backend row capability, in-segment fusion/boundary fallback, mock only\n");
}

class InjectRmsCPU : public CPUBackend {
public:
    bool enabled = true, reject = false;
    int fused_calls = 0, inject_calls = 0, attempts = 0;
    bool supports_gr_inject_rms_fusion() const override { return enabled; }
    void dispatch(const GraphNode& node, const std::vector<const Tensor*>& inputs,
                  Tensor* output, ThreadPool* pool) override {
        if (node.op_type == OpType::GR_INJECT) ++inject_calls;
        CPUBackend::dispatch(node, inputs, output, pool);
    }
    bool dispatch_gr_inject_rms(const GraphNode& inject, const GraphNode& rms,
        const std::vector<const Tensor*>& inputs, Tensor* carry, Tensor* output,
        ThreadPool* pool) override {
        ++attempts;
        check(inputs.size() == 4 && std::all_of(inputs.begin(), inputs.end(),
                  [](const Tensor* t) { return t && t->data; }),
              "all deferred inject inputs remain live across constant nodes");
        check(carry && carry->data && output && output->data && carry->data != output->data,
              "dual-output fusion keeps separate carry and normalized storage");
        if (reject) return false;
        CPUBackend::dispatch(inject, {inputs[0], inputs[1], inputs[2]}, carry, pool);
        CPUBackend::dispatch(rms, {carry, inputs[3]}, output, pool);
        ++fused_calls;
        return true;
    }
};

class MoeInjectCPU : public InjectRmsCPU {
public:
    bool chain_enabled=true,chain_reject=false;int chain_calls=0,chain_attempts=0,moe_calls=0;
    bool supports_moe_gr_inject_rms_fusion() const override {return chain_enabled;}
    void dispatch(const GraphNode& node,const std::vector<const Tensor*>& inputs,Tensor* output,ThreadPool* pool) override {
        if(node.op_type==OpType::MOE){++moe_calls;check(inputs[0]&&inputs[0]->data,"mock MoE input retained");
            std::memcpy(output->data,inputs[0]->data,output->nbytes());return;}
        InjectRmsCPU::dispatch(node,inputs,output,pool);
    }
    bool dispatch_moe_gr_inject_rms(const GraphNode& moe,const GraphNode& inject,const GraphNode& rms,
        const std::vector<const Tensor*>& mi,const std::vector<const Tensor*>& hi,Tensor* branch,
        Tensor* carry,Tensor* output,ThreadPool* pool) override {
        ++chain_attempts;
        check(mi.size()==8&&std::all_of(mi.begin(),mi.end(),[](const Tensor* t){return t&&t->data;}),"all deferred MoE inputs retained through RMS");
        check(branch==hi[0]&&branch->data&&carry->data&&output->data,"chain outputs allocated independently");
        if(chain_reject)return false;
        // Deliberately never publish branch; use the mock MoE input directly.
        CPUBackend::dispatch(inject,{mi[0],hi[1],hi[2]},carry,pool);
        CPUBackend::dispatch(rms,{carry,hi[3]},output,pool);++chain_calls;return true;
    }
};

void test_moe_inject_rms_lifecycle(){
    constexpr int hidden=2560,wide=10240;
    for(int mode=0;mode<11;++mode){Graph graph;graph.nodes.resize(9);graph.runtime.tensors.resize(9);
        for(int i=0;i<9;++i){auto& n=graph.nodes[i];n.id=i;n.out_prec=Precision::FP32;
            n.out_shape[0]=(i==0||i==3||i==4||i==8)?hidden:i==2?4:wide;
            n.out_shape[1]=(i==6)?1:16;if(i!=6)n.dim_expr[1].kind=DIM_SEQ;
            graph.runtime.tensors[i]=Tensor::create(Precision::FP32,MemoryType::NONE,n.out_shape[0],n.out_shape[1],1,1);}
        for(int i:{0,1,2})graph.nodes[i].op_type=OpType::INPUT;
        graph.nodes[3].op_type=OpType::ADD;graph.nodes[3].inputs={0,0};
        graph.nodes[4].op_type=OpType::MOE;graph.nodes[4].inputs={3,0,0,0,0,0,0,0};graph.nodes[4].params.i32={2560,512,10};
        graph.nodes[5].op_type=OpType::GR_INJECT;graph.nodes[5].inputs={4,1,2};graph.nodes[5].params.i32={2560,4};
        graph.nodes[6].op_type=OpType::CONSTANT;
        graph.nodes[7].op_type=OpType::GROUP_RMS_NORM;graph.nodes[7].inputs={5,6};graph.nodes[7].params.i32={2560};graph.nodes[7].params.f32={1e-6f};
        graph.nodes[8].op_type=OpType::ADD;graph.nodes[8].inputs={4,4};
        // Keep a ninth input node normally: a second consumer only in mode4.
        if(mode!=4){graph.nodes[8].op_type=OpType::INPUT;graph.nodes[8].inputs.clear();}
        graph.graph_inputs={0,1,2};graph.graph_outputs={5,7};if(mode==3)graph.graph_outputs.push_back(4);
        std::vector<float>x(hidden*32),res(wide*32,.3f),g(4*32,.2f),gamma(wide,1.f);
        graph.runtime.tensors[6]=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,wide,1,1,1,gamma.data());
        MoeInjectCPU backend;backend.chain_reject=mode==1;backend.chain_enabled=mode!=2;
        BufferPool pool;ExecContext ctx;ctx.graph=&graph;ctx.pool=&pool;ctx.backend=&backend;
        ctx.profile_enabled=mode==5;ctx.reuse_same_shape_workspace=mode==8;ctx.segment_fusions_enabled=mode==9;
        prepare_execution(ctx);
        for(int pass=0;pass<3;++pass){const int rows=mode==6?1:mode==7?15:pass==1?32:16;
            for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i%71)*.1f+pass);
            graph.runtime.tensors[0]=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,hidden,rows,1,1,x.data());
            graph.runtime.tensors[1]=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,wide,rows,1,1,res.data());
            graph.runtime.tensors[2]=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,4,rows,1,1,g.data());
            if(mode!=4)graph.runtime.tensors[8]=graph.runtime.tensors[0];
            ctx.runtime_seq_len=rows;inject_runtime_shapes(ctx);
            if(mode==9){execute_graph_segment(ctx,0,5);execute_graph_segment(ctx,5,9);}
            else execute_graph(ctx,mode==10?5:-1);
            check(!ctx.execution_failed,"MoE HC mock execution succeeds");
            const bool attempted=mode==0||mode==1||mode==8;
            check(backend.chain_attempts==(attempted?pass+1:0),"single-use unexposed complete chain eligibility");
            check(backend.chain_calls==((attempted&&mode!=1)?pass+1:0),"chain false reconstructs all original stages");
            check(backend.moe_calls==((attempted&&mode!=1)?0:pass+1),"MoE dispatch suppressed only in accepted fusion");
            if(mode!=10){std::vector<float> doubled(hidden*rows),carry(wide*rows),norm(wide*rows);
                for(size_t i=0;i<doubled.size();++i)doubled[i]=x[i]+x[i];
                Tensor b=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,hidden,rows,1,1,doubled.data());
                Tensor c=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,wide,rows,1,1,carry.data());
                Tensor n=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,wide,rows,1,1,norm.data());
                CPUBackend reference;reference.dispatch(graph.nodes[5],{&b,&graph.runtime.tensors[1],&graph.runtime.tensors[2]},&c,nullptr);
                reference.dispatch(graph.nodes[7],{&c,&graph.runtime.tensors[6]},&n,nullptr);
                check(graph.runtime.tensors[7].data&&std::memcmp(norm.data(),graph.runtime.tensors[7].data,n.nbytes())==0,"chain/fallback normalized bytes match");
                if(mode!=8)check(!graph.runtime.tensors[3].data,"MoE transient input released only after completed chain");
            }
        }
    }
    std::puts("MoE HC chain mock lifecycle: 11 modes x3 dynamic passes");
}

void test_inject_rms_lifecycle() {
    constexpr int hidden = 19, streams = 4, wide = hidden * streams;
    for (int mode = 0; mode < 16; ++mode) {
        const bool stateful = mode >= 9;
        Graph graph; graph.nodes.resize(stateful ? 14 : 11); graph.runtime.tensors.resize(graph.nodes.size());
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            auto& node = graph.nodes[i]; node.id = i;
            node.out_shape[0] = i == 1 || i == 4 ? hidden : i == 3 || i == 6 ? streams : wide;
            node.out_shape[1] = i == 0 || i == 8 ? 1 : 3;
            if (i != 0 && i != 8) node.dim_expr[1].kind = DIM_SEQ;
            graph.runtime.tensors[i] = Tensor::create(Precision::FP32, MemoryType::NONE,
                node.out_shape[0], node.out_shape[1], 1, 1, nullptr);
        }
        graph.nodes[0].op_type = graph.nodes[8].op_type = OpType::CONSTANT;
        for (int i : {1, 2, 3}) graph.nodes[i].op_type = OpType::INPUT;
        for (int i : {4, 5, 6}) {
            graph.nodes[i].op_type = OpType::ADD;
            graph.nodes[i].inputs = {uint32_t(i - 3), uint32_t(i - 3)};
        }
        graph.nodes[7].op_type = OpType::GR_INJECT;
        graph.nodes[7].inputs = {4, 5, 6}; graph.nodes[7].params.i32 = {hidden, streams};
        graph.nodes[9].op_type = OpType::GROUP_RMS_NORM;
        graph.nodes[9].inputs = {7, 0}; graph.nodes[9].params.i32 = {hidden};
        graph.nodes[9].params.f32 = {1e-6f};
        graph.nodes[10].op_type = OpType::ADD; graph.nodes[10].inputs = {7, 7};
        if (mode == 7) {
            graph.nodes[8].op_type = OpType::ADD; graph.nodes[8].inputs = {7, 7};
            graph.nodes[8].out_shape[1] = 3; graph.nodes[8].dim_expr[1].kind = DIM_SEQ;
        }
        graph.graph_inputs = {1, 2, 3}; graph.graph_outputs = {7, 9, 10};
        if (mode == 8) graph.graph_outputs = {9, 10}; // carry still has a later user
        std::vector<float> conv_weight(wide * 4, .03f), state(wide * 3, .01f);
        if (stateful) {
            for (int id : {11, 12}) {
                graph.nodes[id].op_type = OpType::CONSTANT;
                for (auto& expr : graph.nodes[id].dim_expr) expr.kind = DIM_CONST;
            }
            graph.runtime.tensors[11] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                4, wide, 1, 1, conv_weight.data());
            graph.runtime.tensors[12] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                3, wide, 1, 1, state.data());
            graph.nodes[13].op_type = OpType::SHORTCONV;
            graph.nodes[13].inputs = {9, 11, 12}; graph.nodes[13].params.i32 = {4, 3};
            graph.graph_outputs.push_back(13);
        }
        std::vector<float> branch(hidden * 5), residual(wide * 5), gates(streams * 5), gamma(wide);
        for (int d = 0; d < wide; ++d) gamma[d] = 0.7f + float(d % 11) * .03f;
        for (int id : {0, 8}) if (graph.nodes[id].op_type == OpType::CONSTANT)
            graph.runtime.tensors[id] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                wide, 1, 1, 1, gamma.data());
        InjectRmsCPU backend; BufferPool pool; ExecContext ctx;
        backend.reject = mode == 1; backend.enabled = mode != 2;
        ctx.graph = &graph; ctx.pool = &pool; ctx.backend = &backend;
        ctx.profile_enabled = mode == 3;
        ctx.reuse_same_shape_workspace = mode == 6;
        ctx.conservative_stateful_liveness = mode == 10;
        ctx.segment_fusions_enabled = mode >= 13;
        prepare_execution(ctx);
        for (int pass = 0; pass < 4; ++pass) {
            const int rows = mode == 5 ? 1 : mode == 6 ? 3 : pass == 1 ? 5 : 3;
            for (size_t d = 0; d < branch.size(); ++d) branch[d] = std::sin(float(d) * .13f + pass);
            for (size_t d = 0; d < residual.size(); ++d) residual[d] = std::cos(float(d) * .07f + pass);
            for (size_t d = 0; d < gates.size(); ++d) gates[d] = .1f + float((d + pass) % 7) * .09f;
            graph.runtime.tensors[1] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                hidden, rows, 1, 1, branch.data());
            graph.runtime.tensors[2] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                wide, rows, 1, 1, residual.data());
            graph.runtime.tensors[3] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                streams, rows, 1, 1, gates.data());
            ctx.runtime_seq_len = rows; inject_runtime_shapes(ctx);
            std::vector<float> expected_state = state;
            if (mode >= 11) {
                const int cut = mode == 13 ? 7 : mode == 14 ? 10 : mode == 11 ? 8 : 9;
                execute_graph_segment(ctx, 0, cut);
                execute_graph_segment(ctx, cut, 14);
            } else execute_graph(ctx, mode == 4 ? 7 : -1);
            check(!ctx.execution_failed, "inject RMS execution succeeds");
            std::vector<float> doubled_branch(hidden * rows), doubled_residual(wide * rows),
                doubled_gates(streams * rows), expected_carry(wide * rows), expected_norm(wide * rows);
            for (size_t d = 0; d < doubled_branch.size(); ++d) doubled_branch[d] = branch[d] + branch[d];
            for (size_t d = 0; d < doubled_residual.size(); ++d) doubled_residual[d] = residual[d] + residual[d];
            for (size_t d = 0; d < doubled_gates.size(); ++d) doubled_gates[d] = gates[d] + gates[d];
            Tensor b = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hidden, rows, 1, 1, doubled_branch.data());
            Tensor r = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, wide, rows, 1, 1, doubled_residual.data());
            Tensor g = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, streams, rows, 1, 1, doubled_gates.data());
            Tensor c = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, wide, rows, 1, 1, expected_carry.data());
            Tensor n = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, wide, rows, 1, 1, expected_norm.data());
            CPUBackend reference;
            reference.dispatch(graph.nodes[7], {&b, &r, &g}, &c, nullptr);
            reference.dispatch(graph.nodes[9], {&c, &graph.runtime.tensors[0]}, &n, nullptr);
            if (stateful) {
                std::vector<float> expected_conv(wide * rows);
                Tensor s = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                    3, wide, 1, 1, expected_state.data());
                Tensor o = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                    wide, rows, 1, 1, expected_conv.data());
                reference.dispatch(graph.nodes[13], {&n, &graph.runtime.tensors[11], &s}, &o, nullptr);
                check(!std::memcmp(state.data(), expected_state.data(), state.size() * sizeof(float)),
                      "fused stateful graph preserves complete recurrent state bytes");
                check(graph.runtime.tensors[13].data &&
                    !std::memcmp(graph.runtime.tensors[13].data, expected_conv.data(), o.nbytes()),
                    "fused normalized output preserves downstream stateful output bytes");
            }
            if (mode != 8) check(graph.runtime.tensors[7].data &&
                !std::memcmp(graph.runtime.tensors[7].data, expected_carry.data(), c.nbytes()),
                "raw carry bytes match separate execution including partial stop");
            if (mode != 4) {
                check(graph.runtime.tensors[9].data &&
                    !std::memcmp(graph.runtime.tensors[9].data, expected_norm.data(), n.nbytes()),
                    "normalized bytes match separate execution");
                const auto* user = graph.runtime.tensors[10].ptr<float>();
                check(user != nullptr, "later carry user stays materialized");
                if (user) for (size_t d = 0; d < expected_carry.size(); ++d)
                    check(user[d] == expected_carry[d] + expected_carry[d],
                          "additional carry user sees original residual values");
            }
            const bool expect = mode == 0 || mode == 6 || mode == 8 || mode == 9 || mode == 10 ||
                mode == 13 || mode == 14;
            check(backend.fused_calls == (expect ? pass + 1 : 0), "inject RMS fusion scope");
            check(backend.inject_calls == (expect ? 0 : pass + 1), "inject RMS fallback dispatch count");
            if (mode == 1) check(backend.attempts == pass + 1, "rejected fusion uses ordinary two-op path");
        }
    }
}

void test_gate_fusion_lifecycle() {
    constexpr int hidden = 19, streams = 4, wide = hidden * streams;
    for (int mode = 0; mode < 19; ++mode) {
        const bool shared_gate = mode == 4;
        const bool borrowed_logits = mode == 9;
        const bool expect_fusion = mode == 0 || mode == 9 || mode == 10 || mode == 11 || mode == 13 ||
            mode == 16 || mode == 17;
        Graph graph; graph.nodes.resize(shared_gate ? 6 : 5);
        graph.runtime.tensors.resize(graph.nodes.size());
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            auto& node = graph.nodes[i]; node.id = i;
            node.out_shape[0] = i == 4 ? hidden : wide;
            node.out_shape[1] = 3; node.dim_expr[1].kind = DIM_SEQ;
            graph.runtime.tensors[i] = Tensor::create(Precision::FP32,
                MemoryType::NONE, node.out_shape[0], 3, 1, 1, nullptr);
        }
        graph.nodes[0].op_type = graph.nodes[1].op_type = OpType::INPUT;
        graph.nodes[2].op_type = OpType::ADD; graph.nodes[2].inputs = {1, 1};
        if (borrowed_logits) {
            graph.nodes[2].op_type = OpType::RESHAPE; graph.nodes[2].inputs = {1};
        }
        graph.nodes[3].op_type = OpType::SIGMOID; graph.nodes[3].inputs = {2};
        graph.nodes[4].op_type = OpType::GR_REDUCE;
        graph.nodes[4].inputs = {0, 3}; graph.nodes[4].params.i32 = {hidden, streams};
        if (shared_gate) { graph.nodes[5].op_type = OpType::ADD; graph.nodes[5].inputs = {3, 3}; }
        graph.graph_inputs = {0, 1}; graph.graph_outputs = {4};
        if (mode == 5) graph.graph_outputs.push_back(3);
        if (shared_gate) graph.graph_outputs.push_back(5);
        std::vector<float> x(wide * 5), logits(wide * 5);
        FusionCPU backend; BufferPool pool; ExecContext ctx;
        backend.reject = mode == 1; backend.enabled = mode != 8;
        ctx.graph = &graph; ctx.pool = &pool; ctx.backend = &backend;
        ctx.profile_enabled = mode == 2;
        ctx.reuse_same_shape_workspace = mode == 10;
        ctx.reuse_static_workspace = mode == 11;
        ctx.segment_fusions_enabled = mode >= 16;
        if (mode == 7) setenv("MOLLM_DUMP_NODES", "1", 1);
        prepare_execution(ctx);
        ctx.emulate_bf16_activations = mode == 12;
        for (int pass = 0; pass < 3; ++pass) {
            // Same-shape calls can switch back to liveness-managed outputs
            // after retaining an earlier execution's workspace.
            if (mode == 13) ctx.reuse_same_shape_workspace = pass < 2;
            const int rows = mode == 6 ? 1 : mode == 11 ? 3 :
                mode == 13 ? 3 : mode == 10 ? (pass < 2 ? 3 : 5) : (pass == 1 ? 5 : 3);
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = std::sin(float(i) * .17f + pass);
                logits[i] = std::cos(float(i) * .11f + pass);
            }
            graph.runtime.tensors[0] = Tensor::create(Precision::FP32,
                MemoryType::EXTERNAL, wide, rows, 1, 1, x.data());
            graph.runtime.tensors[1] = Tensor::create(Precision::FP32,
                MemoryType::EXTERNAL, wide, rows, 1, 1, logits.data());
            ctx.runtime_seq_len = mode == 11 ? -1 : rows;
            if (mode != 11) inject_runtime_shapes(ctx);
            if (mode >= 14) {
                const int cut = mode == 17 ? 2 : (mode == 14 || mode == 16) ? 3 : 4;
                execute_graph_segment(ctx, 0, cut);
                execute_graph_segment(ctx, cut, 5);
            } else execute_graph(ctx, mode == 3 ? 3 : -1);
            check(!ctx.execution_failed, "fusion lifecycle execution succeeds");
            if (mode == 13) {
                check((graph.runtime.tensors[2].data != nullptr) == (pass == 1),
                      "same-shape retention keeps producer only while reuse is enabled");
            }
            const int result_id = mode == 3 ? 3 : 4;
            const auto& result = graph.runtime.tensors[result_id];
            check(result.data != nullptr, "fusion/partial result materialized");
            if (mode == 3) {
                check(backend.fused_calls == 0 && backend.sigmoid_calls == pass + 1,
                      "early stop retains sigmoid semantics");
            } else {
                std::vector<float> doubled(wide * rows), gates(wide * rows), expected(hidden * rows);
                for (size_t i = 0; i < doubled.size(); ++i)
                    doubled[i] = borrowed_logits ? logits[i] : logits[i] + logits[i];
                Tensor raw = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, wide, rows, 1, 1, doubled.data());
                Tensor gate = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, wide, rows, 1, 1, gates.data());
                Tensor target = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hidden, rows, 1, 1, expected.data());
                CPUBackend reference; reference.dispatch(graph.nodes[3], {&raw}, &gate, nullptr);
                reference.dispatch(graph.nodes[4], {&graph.runtime.tensors[0], &gate}, &target, nullptr);
                check(result.data && std::memcmp(result.data, expected.data(), expected.size() * sizeof(float)) == 0,
                      "fused/fallback repeated dynamic output matches original graph");
                check(backend.fused_calls == (expect_fusion ? pass + 1 : 0),
                      "only eligible exclusive prefill fuses");
                check(backend.sigmoid_calls == (expect_fusion ? 0 : pass + 1),
                      "fallback/diagnostics/decode materialize gate");
                if (expect_fusion) check(!graph.runtime.tensors[3].data, "exclusive fused gate has no allocation");
            }
        }
        if (mode == 7) unsetenv("MOLLM_DUMP_NODES");
    }
}

void test_stateful_conservative_execution() {
    class ContractCPU : public CPUBackend {
    public:
        int state_index=2;
        mutable int queries=0;
        int independent_stateful_mutable_input(const GraphNode& node) const override {
            ++queries;
            return node.op_type==OpType::SHORTCONV && node.inputs.size()==3 ? state_index : -1;
        }
    };
    std::vector<std::vector<float>> reference_outputs, reference_states;
    for (int mode : {0,1,2,3,4,5}) {
        const bool planned=mode!=0 && mode!=5, contract=mode>=2, reclaim=mode==2;
        Graph graph; graph.nodes.resize(9); graph.runtime.tensors.resize(9);
        for (uint32_t i = 0; i < 9; ++i) {
            auto& node = graph.nodes[i]; node.id = i;
            node.out_shape[0] = 4; node.out_shape[1] = 3;
            node.dim_expr[1].kind = DIM_SEQ;
        }
        for (uint32_t i = 0; i < 3; ++i) graph.nodes[i].op_type = OpType::INPUT;
        graph.nodes[1].dim_expr[1].kind = DIM_CONST;
        graph.nodes[2].dim_expr[1].kind = DIM_CONST;
        graph.nodes[3].op_type = OpType::ADD; graph.nodes[3].inputs = {0, 0};
        graph.nodes[4].op_type = OpType::RESHAPE; graph.nodes[4].inputs = {3};
        graph.nodes[5].op_type = OpType::SHORTCONV;
        graph.nodes[5].inputs = {4, 1, 2}; graph.nodes[5].params.i32 = {4, 3};
        graph.nodes[6].op_type = OpType::ADD; graph.nodes[6].inputs = {0, 0};
        graph.nodes[7].op_type = OpType::ADD; graph.nodes[7].inputs = {6, 6};
        graph.nodes[8].op_type = OpType::ADD; graph.nodes[8].inputs = {5, 5};
        graph.graph_inputs = {0, 1, 2}; graph.graph_outputs = {8};
        std::vector<float> weights(16), state(12), input(20);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = .03f * (i + 1);
        for (size_t i = 0; i < state.size(); ++i) state[i] = .01f * i;
        ContractCPU backend; BufferPool pool; ExecContext ctx;
        backend.state_index=mode==3 ? 99 : mode==4 ? -1 : 2;
        ctx.graph = &graph; ctx.backend = &backend; ctx.pool = &pool;
        ctx.conservative_stateful_liveness = planned;
        ctx.stateful_readonly_last_use = contract;
        prepare_execution(ctx);
        check(backend.queries==(planned && contract ? 1 : 0),
              "state contract queried only under both explicit planner opt-ins");
        for (int pass = 0; pass < 3; ++pass) {
            const int rows = pass == 1 ? 5 : 3;
            for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(i * .2f + pass);
            graph.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                4, rows, 1, 1, input.data());
            graph.runtime.tensors[1] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                4, 4, 1, 1, weights.data());
            graph.runtime.tensors[2] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                3, 4, 1, 1, state.data());
            ctx.runtime_seq_len = rows; inject_runtime_shapes(ctx); execute_graph(ctx);
            check(!ctx.execution_failed, "stateful liveness repeated execution succeeds");
            const auto& output = graph.runtime.tensors[8];
            check(output.data != nullptr, "stateful liveness result stays materialized");
            if (!output.data) continue;
            std::vector<float> values(output.ptr<float>(), output.ptr<float>() + 4 * rows);
            if (mode==0) { reference_outputs.push_back(values); reference_states.push_back(state); }
            else {
                check(std::memcmp(values.data(), reference_outputs[pass].data(), values.size() * sizeof(float)) == 0,
                      "conservative stateful output bytes match deferred cleanup");
                check(std::memcmp(state.data(), reference_states[pass].data(), state.size() * sizeof(float)) == 0,
                      "persistent convolution state bytes match deferred cleanup");
                check(reclaim ? (!graph.runtime.tensors[3].data && !graph.runtime.tensors[4].data && !graph.runtime.tensors[5].data)
                               : (graph.runtime.tensors[3].data && graph.runtime.tensors[5].data),
                      "stateful contract releases read-only producer/view/output, default retains");
                check(planned ? (!graph.runtime.tensors[6].data && !graph.runtime.tensors[7].data)
                              : (graph.runtime.tensors[6].data && graph.runtime.tensors[7].data),
                      "default defers cleanup; planned unrelated materialized branch releases");
            }
        }
    }
}
}  // namespace

void test_segmented_execution() {
    class StatefulMock : public TrackingCPU {
    public:
        int state = 0, calls = 0;
        void dispatch(const GraphNode& node, const std::vector<const Tensor*>& inputs,
                      Tensor* output, ThreadPool* pool) override {
            ++calls;
            CPUBackend::dispatch(node, inputs, output, pool);
            if (node.id == 3) {
                ++state;
                for (int64_t i = 0; i < output->nelements(); ++i)
                    output->ptr<float>()[i] += state;
            }
        }
    };
    for (int mode = 0; mode < 3; ++mode) for (int cut = 1; cut < 6; ++cut) {
        Graph graph; graph.nodes.resize(6); graph.runtime.tensors.resize(6);
        for (int i = 0; i < 6; ++i) {
            auto& n = graph.nodes[i]; n.id = i; n.out_shape[0] = 4;
            n.out_shape[1] = 2; n.dim_expr[1].kind = DIM_SEQ;
            n.out_prec = Precision::FP32;
        }
        graph.nodes[0].op_type = OpType::INPUT;
        graph.nodes[1].op_type = OpType::ADD; graph.nodes[1].inputs = {0, 0};
        graph.nodes[2].op_type = OpType::RESHAPE; graph.nodes[2].inputs = {1};
        graph.nodes[3].op_type = OpType::ADD; graph.nodes[3].inputs = {2, 0};
        graph.nodes[4].op_type = OpType::ADD; graph.nodes[4].inputs = {0, 0};
        graph.nodes[5].op_type = OpType::ADD; graph.nodes[5].inputs = {3, 2};
        graph.graph_inputs = {0}; graph.graph_outputs = {5};
        std::vector<float> input(12);
        StatefulMock backend; BufferPool pool; ExecContext ctx;
        ctx.graph = &graph; ctx.backend = &backend; ctx.pool = &pool;
        ctx.reuse_static_workspace = mode == 1;
        ctx.reuse_same_shape_workspace = mode == 2;
        prepare_execution(ctx);
        for (int pass = 1; pass <= 3; ++pass) {
            ctx.runtime_seq_len = pass == 1 ? 2 : 3;
            std::fill(input.begin(), input.end(), float(pass));
            graph.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                4, ctx.runtime_seq_len, 1, 1, input.data());
            execute_graph_segment(ctx, 0, cut);
            check(!ctx.execution_failed && ctx.segment_next == cut,
                  "segment suspends at exact node boundary including INPUT");
            int before = backend.calls;
            execute_graph_segment(ctx, cut, 6);
            check(!ctx.execution_failed && ctx.segment_next == -1,
                  "segment completes without resetting live owners/views");
            const auto& out = graph.runtime.tensors[5];
            bool exact = out.data != nullptr;
            if (exact) for (int64_t i = 0; i < out.nelements(); ++i)
                exact &= out.ptr<float>()[i] == 6.f * pass;
            check(exact && backend.state == pass && backend.calls == 5 * pass,
                  "segmented changed-shape pass preserves output and executes side effect once");
            check(backend.calls >= before && backend.executions == 2 * pass,
                  "resume invalidates backend cache without replaying prefix");
        }
        execute_graph_segment(ctx, 0, 3);
        const int calls = backend.calls;
        execute_graph_segment(ctx, 2, 6);
        check(ctx.execution_failed && backend.calls == calls, "out-of-order resume rejects before dispatch");
        execute_graph(ctx); // Explicit fresh pass abandons suspended storage safely.
        execute_graph_segment(ctx, 0, 3);
        const int calls2 = backend.calls;
        ++ctx.runtime_seq_len;
        execute_graph_segment(ctx, 3, 6);
        check(ctx.execution_failed && backend.calls == calls2, "shape mutation rejects before dispatch");
        --ctx.runtime_seq_len;
        execute_graph(ctx);
        const int calls3 = backend.calls;
        execute_graph_segment(ctx, 3, 6);
        execute_graph_segment(ctx, 0, 7);
        check(ctx.execution_failed && backend.calls == calls3, "missing prefix and oversized range reject");
    }
}

void test_interleaved_segment_frames() {
    struct Frame {
        Graph graph;
        BufferPool pool;
        ExecContext ctx;
        std::vector<float> input = std::vector<float>(6);
        std::vector<float> weight = std::vector<float>(6, .25f);
        std::vector<float> state = std::vector<float>(4, .125f);
        std::vector<float> reference_state = state;
        std::vector<float> expected = std::vector<float>(6);
    };
    Frame a, b;
    TrackingCPU shared;
    for (Frame* f : {&a, &b}) {
        f->graph.nodes.resize(5); f->graph.runtime.tensors.resize(5);
        for (int i = 0; i < 5; ++i) {
            auto& n = f->graph.nodes[i]; n.id = i; n.out_prec = Precision::FP32;
            n.out_shape[0] = 2; n.out_shape[1] = 2;
            if (i == 0 || i >= 3) n.dim_expr[1].kind = DIM_SEQ;
        }
        f->graph.nodes[0].op_type = OpType::INPUT;
        f->graph.nodes[1].op_type = f->graph.nodes[2].op_type = OpType::CONSTANT;
        f->graph.nodes[1].out_shape[0] = 3;
        f->graph.nodes[3].op_type = OpType::SHORTCONV;
        f->graph.nodes[3].inputs = {0, 1, 2}; f->graph.nodes[3].params.i32 = {3, 2};
        f->graph.nodes[4].op_type = OpType::ADD; f->graph.nodes[4].inputs = {3, 3};
        f->graph.graph_inputs = {0}; f->graph.graph_outputs = {4};
        f->graph.runtime.tensors[1] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
            3, 2, 1, 1, f->weight.data());
        f->graph.runtime.tensors[2] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
            2, 2, 1, 1, f->state.data());
        f->ctx.graph = &f->graph; f->ctx.pool = &f->pool; f->ctx.backend = &shared;
        prepare_execution(f->ctx);
    }
    for (int pass = 0; pass < 3; ++pass) {
        const int rows = pass == 1 ? 3 : 2;
        for (Frame* f : {&a, &b}) {
            f->ctx.runtime_seq_len = rows;
            for (size_t i = 0; i < f->input.size(); ++i)
                f->input[i] = float(i + pass + (f == &a ? 1 : 10)) * .125f;
            auto& tensors = f->graph.runtime.tensors;
            tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                2, rows, 1, 1, f->input.data());
            inject_runtime_shapes(f->ctx);
            Tensor state = tensors[2]; state.data = f->reference_state.data();
            Tensor expected = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                2, rows, 1, 1, f->expected.data());
            CPUBackend reference;
            reference.dispatch(f->graph.nodes[3], {&tensors[0], &tensors[1], &state}, &expected, nullptr);
            for (int i = 0; i < rows * 2; ++i) f->expected[i] += f->expected[i];
        }
        execute_graph_segment(a.ctx, 0, 4);
        execute_graph_segment(b.ctx, 0, 4);
        execute_graph_segment(a.ctx, 4, 5);
        execute_graph_segment(b.ctx, 4, 5);
        for (Frame* f : {&a, &b}) {
            const Tensor& out = f->graph.runtime.tensors[4];
            check(!f->ctx.execution_failed && out.data &&
                std::memcmp(out.data, f->expected.data(), size_t(rows) * 2 * sizeof(float)) == 0,
                "interleaved independent runtimes sharing backend match sequential SHORTCONV outputs");
            check(f->state == f->reference_state,
                "interleaved frames preserve independent persistent SHORTCONV state");
        }
    }
    check(shared.executions == 12, "each interleaved segment invalidates shared backend cache");
}

void test_execution_frame_initialization() {
    Graph source, a, b;
    source.nodes.resize(4); source.runtime.tensors.resize(4);
    source.metadata["frame_test"] = "source";
    for (int i = 0; i < 4; ++i) {
        auto& n = source.nodes[i]; n.id = i; n.out_shape[0] = 4;
        n.out_prec = Precision::FP32;
    }
    source.nodes[0].op_type = OpType::INPUT;
    source.nodes[1].op_type = OpType::CONSTANT;
    source.nodes[2].op_type = OpType::ADD; source.nodes[2].inputs = {0, 1};
    source.nodes[3].op_type = OpType::RESHAPE; source.nodes[3].inputs = {2};
    source.graph_inputs = {0}; source.graph_outputs = {3};
    std::vector<float> input(4, 7.f), weight(4, 3.f), input_a(4, 2.f), input_b(4, 5.f);
    uint8_t quant_sidecar = 127;
    source.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1, input.data());
    source.runtime.tensors[1] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1, weight.data());
    source.runtime.tensors[1].e8m0_scales = &quant_sidecar;
    CPUBackend backend; ExecContext original;
    original.graph = &source; original.pool = &source.runtime.pool; original.backend = &backend;
    prepare_execution(original); execute_graph(original);
    check(source.runtime.tensors[2].data && source.runtime.tensors[3].data,
          "frame source has materialized owner and returned view");
    check(initialize_execution_frame(source, a, {0}) && initialize_execution_frame(source, b, {0}),
          "initialize separate frames from already executed source");
    check(!a.runtime.tensors[0].data && !a.runtime.tensors[2].data && !a.runtime.tensors[3].data &&
          !b.runtime.tensors[0].data && !b.runtime.tensors[2].data && !b.runtime.tensors[3].data,
          "private boundaries and all computed source storage are not copied");
    check(a.runtime.tensors[1].data == weight.data() && b.runtime.tensors[1].e8m0_scales == &quant_sidecar &&
          a.runtime.weights.empty() && a.runtime.tensors.data() != source.runtime.tensors.data(),
          "borrow immutable weights/sidecars without copying mapping ownership or tensor table");
    a.nodes[2].params.i32 = {1}; a.metadata["frame_test"] = "a";
    check(source.nodes[2].params.i32.empty() && b.nodes[2].params.i32.empty() &&
          source.metadata["frame_test"] == "source", "frame node params and metadata are independent");
    a.nodes[2].params.i32.clear();
    a.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1, input_a.data());
    b.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1, input_b.data());
    ExecContext ca, cb;
    ca.graph = &a; ca.pool = &a.runtime.pool; ca.backend = &backend;
    cb.graph = &b; cb.pool = &b.runtime.pool; cb.backend = &backend;
    prepare_execution(ca); prepare_execution(cb);
    check(ca.resolved_inputs[2][0] == &a.runtime.tensors[0] &&
          cb.resolved_inputs[2][0] == &b.runtime.tensors[0], "prepared inputs resolve to each frame, not source");
    execute_graph_segment(ca, 0, 3); execute_graph_segment(cb, 0, 3);
    execute_graph_segment(ca, 3, 4); execute_graph_segment(cb, 3, 4);
    check(!ca.execution_failed && !cb.execution_failed && a.runtime.tensors[3].ptr<float>()[0] == 5.f &&
          b.runtime.tensors[3].ptr<float>()[0] == 8.f && source.runtime.tensors[3].ptr<float>()[0] == 10.f,
          "interleaved cloned frames preserve private output and untouched source");
    for (const auto& ids : std::vector<std::vector<uint32_t>>{{2}, {1}, {8}, {0, 0}}) {
        Graph rejected;
        check(!initialize_execution_frame(source, rejected, ids) && rejected.nodes.empty() &&
            rejected.runtime.tensors.empty(), "invalid boundary list rejects without destination mutation");
    }
    check(!initialize_execution_frame(source, a, {0}) && a.runtime.tensors[3].ptr<float>()[0] == 5.f,
          "nonempty frame rejection preserves live output");
    check(!initialize_execution_frame(source, source, {0}), "in-place initialization rejected");
    Graph rejected;
    source.nodes[2].inputs = {3, 1};
    check(!initialize_execution_frame(source, rejected, {0}) && rejected.nodes.empty(), "non-topological source rejected");
    source.nodes[2].inputs = {0, 1}; source.nodes[3].id = 2;
    check(!initialize_execution_frame(source, rejected, {0}) && rejected.nodes.empty(), "duplicate node ID rejected");
    source.nodes[3].id = 3;
}

void test_paired_moe_execution() {
    class PairCPU : public CPUBackend {
    public:
        bool merge = false; int singles = 0, pairs = 0;
        void calculate(const std::vector<const Tensor*>& inputs, Tensor* out) {
            for (int64_t i = 0; i < out->nelements(); ++i)
                out->ptr<float>()[i] = inputs[0]->ptr<float>()[i] * 3.f;
        }
        void dispatch(const GraphNode& n, const std::vector<const Tensor*>& inputs,
                      Tensor* out, ThreadPool* pool) override {
            if (n.op_type == OpType::MOE) { ++singles; calculate(inputs, out); }
            else CPUBackend::dispatch(n, inputs, out, pool);
        }
        bool dispatch_moe_pair(const GraphNode&, const std::vector<const Tensor*>& a,
            const std::vector<const Tensor*>& b, Tensor* oa, Tensor* ob) override {
            if (!merge) return false;
            ++pairs; calculate(a, oa); calculate(b, ob); return true;
        }
    } backend;
    for (int mode : {0, 1, 2, 3}) {
        backend.merge = mode != 0; backend.singles = backend.pairs = 0;
        Graph source, a, b;
        source.nodes.resize(5); source.runtime.tensors.resize(5);
        for (int i = 0; i < 5; ++i) {
            auto& n = source.nodes[i]; n.id = i; n.out_shape[0] = 8; n.out_shape[1] = 17;
            n.dim_expr[1].kind = DIM_SEQ; n.op_type = OpType::ADD;
        }
        source.nodes[0].op_type = OpType::INPUT;
        source.nodes[1].inputs = {0, 0}; source.nodes[2].op_type = OpType::MOE;
        source.nodes[2].inputs = {1}; source.nodes[3].inputs = {2, 2};
        source.nodes[4].op_type = OpType::RESHAPE; source.nodes[4].inputs = {3};
        source.graph_inputs = {0}; source.graph_outputs = {4};
        check(initialize_execution_frame(source, a, {0}) && initialize_execution_frame(source, b, {0}),
              "paired MoE fixture clones independent runtime frames");
        ExecContext ca, cb;
        ca.graph = &a; ca.pool = &a.runtime.pool; ca.backend = &backend;
        cb.graph = &b; cb.pool = &b.runtime.pool; cb.backend = &backend;
        prepare_execution(ca); prepare_execution(cb);
        std::vector<float> va(8 * 17, .25f), vb(8 * 11, -.5f);
        a.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 8, 17, 1, 1, va.data());
        b.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 8, 11, 1, 1, vb.data());
        ca.runtime_seq_len = 17; cb.runtime_seq_len = 11;
        inject_runtime_shapes(ca); inject_runtime_shapes(cb);
        execute_graph_segment(ca, 0, 2); execute_graph_segment(cb, 0, 2);
        if (mode == 2) b.nodes[2].params.i32 = {123};
        const int result = mode == 3 ? execute_graph_moe_pair(ca, ca, 2) : execute_graph_moe_pair(ca, cb, 2);
        if (mode >= 2) {
            check(result == -1 && backend.singles == 0 && backend.pairs == 0,
                  "paired MoE rejects mismatched parameters/duplicate frame before dispatch");
            continue;
        }
        check(result == mode && !ca.execution_failed && !cb.execution_failed,
              "paired MoE supports merged and ordinary fallback paths");
        check(!a.runtime.tensors[1].data && !b.runtime.tensors[1].data,
              "paired MoE applies input-owner last-use reclamation to both frames");
        execute_graph_segment(ca, 3, 5); execute_graph_segment(cb, 3, 5);
        for (ExecContext* c : {&ca, &cb}) {
            const Tensor& out = c->graph->runtime.tensors[4];
            const float expected = c == &ca ? 3.f : -6.f;
            check(!c->execution_failed && out.data &&
                std::all_of(out.ptr<float>(), out.ptr<float>() + out.nelements(),
                            [&](float x) { return x == expected; }), "paired MoE suffix output exact");
            check(c->segment_next == -1 && !c->graph->runtime.tensors[2].data,
                  "paired MoE resumes correct cursor and reclaims independent output");
        }
        check(backend.pairs == mode && backend.singles == (mode ? 0 : 2),
              "materialized paired MoE is never dispatched a second time");
    }
}

int main() {
    test_paired_moe_execution();
    test_execution_frame_initialization();
    test_segmented_execution();
    test_interleaved_segment_frames();
    test_hc_up_lifecycle();
    {
        check(mollm_memory::fits_headroom(100, 60, 40), "memory headroom exact boundary");
        check(!mollm_memory::fits_headroom(100, 61, 40), "memory headroom rejects overshoot");
        check(!mollm_memory::fits_headroom(39, 0, 40), "memory headroom avoids unsigned underflow");
        check(!mollm_memory::fits_headroom(100, SIZE_MAX, 40), "memory headroom avoids overflow");
        size_t parsed_headroom = 17;
        check(mollm_memory::parse_cuda_host_headroom("24576", parsed_headroom) &&
                  parsed_headroom == size_t(24576) * 1024 * 1024,
              "headroom parses the test policy exactly");
        check(mollm_memory::parse_cuda_host_headroom("0", parsed_headroom) && parsed_headroom == 0,
              "explicit zero preserves opt-out");
        check(mollm_memory::parse_cuda_host_headroom("0001", parsed_headroom) &&
                  parsed_headroom == 1024 * 1024,
              "headroom accepts decimal leading zeros");
        check(!mollm_memory::parse_cuda_host_headroom(nullptr, parsed_headroom),
              "parser rejects null configuration");
        const size_t max_headroom_mb = SIZE_MAX / (1024 * 1024);
        const std::string max_headroom_text = std::to_string(max_headroom_mb);
        const std::string overflow_headroom_text = std::to_string(max_headroom_mb + 1);
        check(mollm_memory::parse_cuda_host_headroom(max_headroom_text.c_str(), parsed_headroom) &&
                  parsed_headroom == max_headroom_mb * (1024 * 1024),
              "headroom accepts maximum representable MiB");
        check(!mollm_memory::parse_cuda_host_headroom(overflow_headroom_text.c_str(), parsed_headroom) &&
                  parsed_headroom == 0,
              "headroom rejects one MiB beyond representable limit");
        const char* prior = std::getenv("MOLLM_CUDA_HOST_HEADROOM_MB");
        const bool had_prior = prior != nullptr;
        const std::string saved = prior ? prior : "";
        for (const char* invalid : {"", "-1", "-0", "+0", "+1", " 0", " -0", "1 ",
                                   "\t0", "0\n", "bad", "12junk", "18446744073709551615",
                                   "18446744073709551616"}) {
            check(!mollm_memory::parse_cuda_host_headroom(invalid, parsed_headroom) &&
                      parsed_headroom == 0,
                  "invalid headroom parser rejects without a partial result");
            setenv("MOLLM_CUDA_HOST_HEADROOM_MB", invalid, 1);
            check(!mollm_memory::cuda_host_allocation_allowed(1, "test-invalid"),
                  "configured invalid memory budget fails closed without CUDA allocation");
        }
        setenv("MOLLM_CUDA_HOST_HEADROOM_MB", "1", 1);
        check(!mollm_memory::cuda_host_allocation_allowed(SIZE_MAX, "test-oversize"),
              "host availability rejects oversized request without CUDA allocation");
        if (had_prior) setenv("MOLLM_CUDA_HOST_HEADROOM_MB", saved.c_str(), 1);
        else unsetenv("MOLLM_CUDA_HOST_HEADROOM_MB");
    }
    test_inject_rms_lifecycle();
    test_moe_inject_rms_lifecycle();
    test_stateful_conservative_execution();
    {
        Graph graph;
        graph.nodes.resize(7);
        graph.runtime.tensors.resize(7);
        for (uint32_t i = 0; i < 7; ++i) {
            graph.nodes[i].id = i;
            graph.nodes[i].out_shape[0] = 4;
            graph.nodes[i].out_shape[1] = 1;
        }
        graph.nodes[0].op_type = OpType::INPUT;
        graph.nodes[1].op_type = OpType::ADD; graph.nodes[1].inputs = {0, 0};
        graph.nodes[2].op_type = OpType::RESHAPE; graph.nodes[2].inputs = {1};
        graph.nodes[3].op_type = OpType::SDPA; graph.nodes[3].inputs = {2, 0};
        graph.nodes[4].op_type = OpType::ADD; graph.nodes[4].inputs = {0, 0};
        graph.nodes[5].op_type = OpType::ADD; graph.nodes[5].inputs = {4, 4};
        graph.nodes[6].op_type = OpType::ADD; graph.nodes[6].inputs = {3, 3};
        graph.graph_outputs = {6};
        CPUBackend backend; BufferPool pool; ExecContext ctx;
        ctx.graph = &graph; ctx.backend = &backend; ctx.pool = &pool;
        prepare_execution(ctx);
        check(std::all_of(ctx.release_queue.begin(), ctx.release_queue.end(),
                          [](const auto& q) { return q.empty(); }),
              "stateful default preserves end-of-call cleanup");
        ctx.conservative_stateful_liveness = true;
        prepare_execution(ctx);
        std::vector<uint32_t> released;
        for (const auto& q : ctx.release_queue)
            released.insert(released.end(), q.begin(), q.end());
        std::sort(released.begin(), released.end());
        check(released == std::vector<uint32_t>({4, 5}),
              "conservative planner retains stateful output, view and input owner");
    }
    for (int cap : {256, 512, 1024}) {
        Graph graph; graph.nodes.resize(1); graph.runtime.tensors.resize(1);
        graph.nodes[0].id = 0; graph.nodes[0].op_type = OpType::INPUT;
        CPUBackend backend; BufferPool pool; ExecContext ctx;
        ctx.graph = &graph; ctx.backend = &backend; ctx.pool = &pool;
        ctx.conservative_stateful_liveness = true;
        ctx.conservative_liveness_seq_limit = cap;
        ctx.runtime_seq_len = cap + 1;
        prepare_execution(ctx); execute_graph(ctx);
        check(ctx.execution_failed && !graph.runtime.tensors[0].data,
              "diagnostic per-call sequence cap rejects before tensor allocation");
    }
    test_gate_fusion_lifecycle();
    constexpr int k = 128, n = 8;
    std::vector<float> input(k, 1.f);
    std::vector<uint8_t> weights(n * k, 0x38);  // E4M3FN 1.0.
    uint8_t scale = 127;                       // E8M0 1.0.
    std::vector<int8_t> packed(pack_fp8_e4m3_q8dot_bytes(n, k));
    std::vector<float> packed_scales(n * k / 32);
    check(pack_fp8_e4m3_q8dot(weights.data(), &scale, n, k,
                              packed.data(), packed_scales.data()),
          "prepare FP8 sidecar for cached ARM GEMV path");

    Graph graph;
    graph.nodes.resize(3);
    graph.nodes[0].id = 0;
    graph.nodes[0].op_type = OpType::INPUT;
    graph.nodes[0].out_shape[0] = k;
    graph.nodes[1].id = 1;
    graph.nodes[1].op_type = OpType::CONSTANT;
    graph.nodes[1].out_shape[0] = n;
    graph.nodes[1].out_shape[1] = k;
    graph.nodes[2].id = 2;
    graph.nodes[2].op_type = OpType::MATMUL;
    graph.nodes[2].inputs = {0, 1};
    graph.nodes[2].out_shape[0] = n;
    graph.nodes[2].out_prec = Precision::FP32;
    graph.graph_inputs = {0};
    graph.graph_outputs = {2};
    graph.runtime.tensors.resize(3);
    auto& activation = graph.runtime.tensors[0];
    activation = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                k, 1, 1, 1, input.data());
    activation.storage_id = 7;
    auto& weight = graph.runtime.tensors[1];
    weight = Tensor::create(Precision::FP8_E4M3, MemoryType::EXTERNAL,
                            n, k, 1, 1, weights.data());
    weight.e8m0_scales = &scale;
    weight.group_size = 128;
    weight.groups_per_row = 1;
    weight.num_groups = 1;
    weight.is_fp8_block128 = true;
    if (mollm::cpu::capabilities().fp16_interleaved_weights) {
        weight.q8_repack_data = packed.data();
        weight.fp8_q8_scales = packed_scales.data();
    }
    TrackingCPU cpu, delegate;
    BufferPool pool;
    ExecContext ctx;
    ctx.graph = &graph;
    ctx.pool = &pool;
    ctx.backend = &cpu;
    ctx.reuse_static_workspace = true;
    prepare_execution(ctx);

    for (int pass = 1; pass <= 3; ++pass) {
        // Preserve the pointer and storage ID while changing its contents.
        std::fill(input.begin(), input.end(), static_cast<float>(pass));
        ctx.moe_backend = pass == 1 ? nullptr : (pass == 2 ? &cpu : &delegate);
        execute_graph(ctx);
        check(!ctx.execution_failed, "repeated FP8 execution succeeds");
        const auto& result = graph.runtime.tensors[2];
        bool correct = result.data != nullptr;
        if (correct)
            for (int i = 0; i < n; ++i)
                correct &= std::fabs(result.ptr<float>()[i] - pass * k) < 0.01f;
        check(correct, "FP8 activation cache does not survive an execution boundary");
        check(cpu.executions == pass, "primary backend starts once per pass");
    }
    check(delegate.executions == 1, "distinct delegated backend starts with the pass");
    execute_graph(ctx, 1);
    check(cpu.executions == 4 && delegate.executions == 2,
          "partial execution also starts a fresh backend pass");
    check(cpu.command_buffers == 0 && delegate.command_buffers == 0,
          "executor leaves device command-buffer ownership to its caller");
    if (!failures) std::puts("All execution lifecycle tests passed!");
    return failures ? 1 : 0;
}
