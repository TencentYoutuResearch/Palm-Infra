#include "backends/cuda/state_prefix_plan.h"
#include "backends/cpu/backend.h"
#include <climits>
#include <cstdio>
#include <limits>

using mollm::cuda_backend_detail::complete_state_prefix_plan;

struct Fixture {
    Graph graph;
    uint32_t state_id;
    explicit Fixture(OpType op) {
        const int count = op == OpType::GATED_DELTANET_PREFILL ? 8 : op == OpType::PLE_LOOKUP ? 6 : 3;
        graph.nodes.resize(count+1);
        graph.runtime.tensors.resize(count+1);
        for(int i=0;i<=count;++i) {
            graph.nodes[i].id=i;
            graph.nodes[i].op_type=OpType::INPUT;
            set(i,8,3);
        }
        auto& node=graph.nodes.back();
        node.op_type=op;
        for(int i=0;i<count;++i)node.inputs.push_back(i);
        if(op==OpType::GATED_DELTANET_PREFILL) {
            state_id=7;
            set(0,512,3);set(1,2,3);set(2,2,3);set(3,256,3);
            set(4,2);set(5,2);set(6,128);set(7,128,128,2);set(8,256,3);
            node.params.i32={1,128,128,3,1,4,3,2};
        } else if(op==OpType::PLE_LOOKUP) {
            state_id=1;
            set(0,3,1,1,Precision::INT32);set(1,3,1,1,Precision::INT32);
            set(4,16,1,1,Precision::INT32);set(5,16,1,1,Precision::INT32);
            node.params.i32={3,8,0,512,0,1234,0,3};
        } else {
            state_id=2;
            set(1,8,4);set(2,8,op==OpType::SHORTCONV?3:9);
            node.params.i32=op==OpType::SHORTCONV?std::vector<int32_t>{4,3}:std::vector<int32_t>{4,3,3};
        }
    }
    void set(int id,int64_t a,int64_t b=1,int64_t c=1,Precision prec=Precision::FP32) {
        // Opaque addresses only: this test never dispatches or touches CUDA.
        void* address=reinterpret_cast<void*>(uintptr_t(0x10000000)+uintptr_t(id)*0x1000000);
        auto& t=graph.runtime.tensors[id];
        t=Tensor::create(prec,MemoryType::EXTERNAL,a,b,c,1,address);
        t.device.buffer=address;
        graph.nodes[id].out_prec=prec;
        std::copy(t.shape,t.shape+4,graph.nodes[id].out_shape);
    }
    Tensor& state(){return graph.runtime.tensors[state_id];}
    GraphNode& writer(){return graph.nodes.back();}
    bool accepts(){return complete_state_prefix_plan(graph,{&state()},3);}
};

#define REQUIRE(condition) do { if(!(condition)){std::fprintf(stderr,"line%d: %s\n",__LINE__,#condition);return 1;} } while(0)

int main() {
    using mollm::cuda_backend_detail::retained_prefix_bank_limit_bytes;
    REQUIRE(retained_prefix_bank_limit_bytes(3)==size_t(512)*1024*1024);
    REQUIRE(retained_prefix_bank_limit_bytes(4)==size_t(512)*1024*1024);
    REQUIRE(retained_prefix_bank_limit_bytes(5)==size_t(1024)*1024*1024);
    REQUIRE(retained_prefix_bank_limit_bytes(2,true)==size_t(512)*1024*1024);
    REQUIRE(retained_prefix_bank_limit_bytes(5,true)==size_t(1024)*1024*1024);
    REQUIRE(mollm::cuda_backend_detail::verification_prefix_limit_bytes==size_t(2048)*1024*1024);
    for(int width:{6,7,8})REQUIRE(retained_prefix_bank_limit_bytes(width)==size_t(1024)*1024*1024);
    for(int width:{-1,0,1,2,9,INT_MAX})REQUIRE(retained_prefix_bank_limit_bytes(width)==0);
    REQUIRE(mollm::cuda_backend_detail::verification_workspace_budget_bytes(5)==size_t(1536)*1024*1024);
    REQUIRE(mollm::cuda_backend_detail::verification_prefix_budget_bytes(5)==size_t(2048)*1024*1024);
    REQUIRE(mollm::cuda_backend_detail::verification_workspace_budget_bytes(6)==size_t(3072)*1024*1024);
    REQUIRE(mollm::cuda_backend_detail::verification_prefix_budget_bytes(8)==size_t(4096)*1024*1024);
    for(OpType type:{OpType::SHORTCONV,OpType::GATED_DELTANET_PREFILL,
                     OpType::PLE_LOOKUP,OpType::PLE_DILATED_CONV}) {
        Fixture f(type);
        REQUIRE(f.accepts());
        REQUIRE(complete_state_prefix_plan(f.graph,{&f.state(),&f.state()},3));
        REQUIRE(!complete_state_prefix_plan(f.graph,{&f.state()},1));
        REQUIRE(!complete_state_prefix_plan(f.graph,{&f.state()},9));
        REQUIRE(!complete_state_prefix_plan(f.graph,{},3));
        const size_t real = type==OpType::SHORTCONV?1:type==OpType::GATED_DELTANET_PREFILL?6:
                            type==OpType::PLE_LOOKUP?7:2;
        const int saved_real=f.writer().params.i32[real];
        f.writer().params.i32[real]=2;
        REQUIRE(!f.accepts());
        f.writer().params.i32[real]=saved_real;
        REQUIRE(f.accepts());
        f.writer().op_type=OpType::RWKV7;
        REQUIRE(!f.accepts());
        f.writer().op_type=type;
        f.graph.nodes[f.state_id].dim_expr[0].kind=DIM_SEQ;
        REQUIRE(!f.accepts());
        f.graph.nodes[f.state_id].dim_expr[0].kind=DIM_CONST;
        Tensor clone=f.state();
        REQUIRE(!complete_state_prefix_plan(f.graph,{&clone},3));
        ++f.state().shape[0];f.state().compute_strides();
        REQUIRE(!f.accepts());
    }
    {
        Fixture f(OpType::SHORTCONV);
        GraphNode source;source.op_type=OpType::INPUT;
        Tensor source_tensor=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,16,3,1,1,
            reinterpret_cast<void*>(uintptr_t(0x80000000)));
        source_tensor.device.buffer=source_tensor.data;
        f.graph.nodes.insert(f.graph.nodes.begin(),source);
        f.graph.runtime.tensors.insert(f.graph.runtime.tensors.begin(),source_tensor);
        for(size_t i=1;i<f.graph.nodes.size();++i) {
            f.graph.nodes[i].id++;
            for(auto& input:f.graph.nodes[i].inputs)++input;
        }
        ++f.state_id;
        f.graph.nodes[1].op_type=OpType::SLICE;f.graph.nodes[1].inputs={0};
        auto& view=f.graph.runtime.tensors[1];
        view.data=nullptr;view.device={};view.stride[1]=16*4;
        REQUIRE(f.accepts());
        f.graph.nodes[1].inputs={f.state_id};
        REQUIRE(!f.accepts()); // State-derived view is not an independent input.
    }
    {
        Fixture f(OpType::GATED_DELTANET_PREFILL);
        f.graph.nodes[1].op_type=OpType::SLICE;f.graph.nodes[1].inputs={0};
        auto& view=f.graph.runtime.tensors[1];
        view.data=nullptr;view.device={};view.stride[1]=512*4;
        REQUIRE(f.accepts());
    }
    {
        Fixture f(OpType::SHORTCONV);
        f.graph.runtime.tensors[1].device={};f.graph.runtime.tensors[1].data=nullptr;
        REQUIRE(!f.accepts()); // A missing weight is not a borrowed view.
    }
    {
        Fixture f(OpType::SHORTCONV);
        f.graph.runtime.tensors.back().device=f.state().device;
        f.graph.runtime.tensors.back().device.offset=4;
        REQUIRE(!f.accepts()); // Partial overlap, not just pointer equality.
    }
    {
        Fixture f(OpType::SHORTCONV);
        GraphNode alias;alias.id=f.graph.nodes.size();alias.op_type=OpType::CONSTANT;
        Tensor host_alias=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,1,1,1,1,f.state().device.buffer);
        f.graph.nodes.push_back(alias);f.graph.runtime.tensors.push_back(host_alias);
        REQUIRE(!f.accepts()); // Host-visible alias of unified/device state.
    }
    {
        Fixture f(OpType::SHORTCONV);
        GraphNode packed;packed.id=f.graph.nodes.size();packed.op_type=OpType::CONSTANT;
        void* address=reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(f.state().device.buffer)-1500);
        Tensor weight=Tensor::create(Precision::INT4,MemoryType::EXTERNAL,2000,1,1,1,address);
        weight.device.buffer=address;
        f.graph.nodes.push_back(packed);f.graph.runtime.tensors.push_back(weight);
        REQUIRE(!f.accepts()); // Logical strides suggest an overlap.
        auto span=[&](const Tensor& t) {
            return &t==&f.graph.runtime.tensors.back()?size_t(1125):t.view_span_bytes();
        };
        REQUIRE(complete_state_prefix_plan(f.graph,{&f.state()},3,span));
        auto overlaps=[&](const Tensor& t) {
            return &t==&f.graph.runtime.tensors.back()?size_t(1600):t.view_span_bytes();
        };
        REQUIRE(!complete_state_prefix_plan(f.graph,{&f.state()},3,overlaps));
    }
    {
        Fixture f(OpType::GATED_DELTANET_PREFILL);
        f.graph.runtime.tensors[1].stride[0]=8;
        REQUIRE(!f.accepts());
    }
    {
        Fixture f(OpType::PLE_LOOKUP);
        f.writer().params.i32[1]=0;
        REQUIRE(!f.accepts());
    }
    {
        Fixture f(OpType::PLE_DILATED_CONV);
        f.writer().params.i32[1]=0;
        REQUIRE(!f.accepts());
    }
    {
        Fixture f(OpType::SHORTCONV);
        std::swap(f.writer().inputs[0],f.writer().inputs[2]);
        REQUIRE(!f.accepts());
    }
    {
        Fixture f(OpType::SHORTCONV);
        f.writer().inputs[2]=0;
        REQUIRE(!f.accepts()); // Missing writer.
    }
    {
        Fixture f(OpType::SHORTCONV);
        GraphNode view;
        view.id=f.graph.nodes.size();view.op_type=OpType::RESHAPE;view.inputs={f.state_id};
        f.graph.nodes.push_back(view);
        f.graph.runtime.tensors.push_back(f.graph.runtime.tensors[0]);
        REQUIRE(!f.accepts()); // Additional consumer/alias path.
    }
    {
        Fixture f(OpType::SHORTCONV);
        f.state().device.buffer=reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-16);
        REQUIRE(!f.accepts());
    }
    {
        Fixture first(OpType::SHORTCONV),second(OpType::SHORTCONV);
        for(size_t i=0;i<second.graph.nodes.size();++i) {
            GraphNode node=second.graph.nodes[i];node.id+=4;
            for(auto& input:node.inputs)input+=4;
            Tensor tensor=second.graph.runtime.tensors[i];
            tensor.device.buffer=reinterpret_cast<void*>(
                reinterpret_cast<uintptr_t>(tensor.device.buffer)+0x10000000);
            first.graph.nodes.push_back(node);first.graph.runtime.tensors.push_back(tensor);
        }
        std::vector<Tensor*> states{&first.graph.runtime.tensors[2],&first.graph.runtime.tensors[6]};
        REQUIRE(complete_state_prefix_plan(first.graph,states,3));
        states[1]->device=states[0]->device;
        states[1]->device.offset=4;
        REQUIRE(!complete_state_prefix_plan(first.graph,states,3));
    }
    {
        Fixture f(OpType::SHORTCONV);
        CPUBackend cpu;
        REQUIRE(!cpu.state_prefixes_cover_graph(f.graph,{&f.state()},3));
        REQUIRE(!cpu.state_prefixes_complete());
    }
    std::puts("CUDA prefix producer plans: known writers, geometry, alias rejection and default-backend fallback passed");
}
