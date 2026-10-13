#include "backends/cuda/backend.h"
#include "core/quant_layouts.h"
#include "graph/execute.h"
#include "graph/hc_prefill_limits.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int k = 10240;
struct Fixture {
    Graph graph;
    ExecContext context;
    CudaBackend& backend;
    std::vector<Q4B8G32Block> down, inject;
    std::vector<uint16_t> middle;
    std::vector<float> gamma;
    bool gated = false;
    bool deferred = false;
    explicit Fixture(CudaBackend& device, bool middle_gemm = false, bool rms = false, bool gated_input = false,bool deferred_input=false,int max_rows=1024)
        : backend(device), down(40 * k / 32), inject(k / 32), middle(middle_gemm?320*320:0,0x1400), gamma(rms?k:0) {
        gated = gated_input;
        deferred=deferred_input;
        graph.runtime.tensors.resize(deferred?16:gated?13:rms?9:(middle_gemm?7:6));
        graph.nodes.resize(6);
        for (int id = 0; id < 6; ++id) graph.nodes[id].id = id;
        graph.nodes[0].op_type = OpType::INPUT;
        graph.nodes[0].out_shape[0] = k;
        Tensor& input = graph.runtime.tensors[0];
        input = Tensor::create(Precision::FP32, MemoryType::NONE, k, max_rows);
        backend.alloc_output(input, input.nbytes(), nullptr);
        input.shape[1] = 1; input.compute_strides();
        const auto weight = [&](int id, int n, std::vector<Q4B8G32Block>& packed) {
            for (size_t block = 0; block < packed.size(); ++block)
                for (int row = 0; row < 8; ++row) {
                    packed[block].scales[row] = (1 + (block + row + id) % 17) / 37.f;
                    for (int byte = 0; byte < 16; ++byte)
                        packed[block].q[row][byte] = ((block * 19 + row * 7 + byte * 13 + id) & 255) ^ 0x88;
                }
            auto& node = graph.nodes[id];
            node.op_type = OpType::CONSTANT;
            node.out_shape[0] = n; node.out_shape[1] = k;
            node.out_prec = Precision::INT4;
            Tensor& tensor = graph.runtime.tensors[id];
            tensor = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, n, k, 1, 1, packed.data());
            tensor.rowmajor_data = packed.data(); tensor.q4_g32_data = packed.data();
            tensor.is_q4_g32_packed = true; tensor.group_size = 32; tensor.groups_per_row = k / 32;
            backend.wrap_weight_int4(tensor);
        };
        weight(1, 320, down); weight(2, 4, inject);
        for (int id : {3, 4, 5}) {
            auto& node = graph.nodes[id];
            node.op_type = id == 4 ? OpType::MUL : OpType::MATMUL;
            node.out_shape[0] = id == 5 ? 4 : 320;
            node.dim_expr[1].kind = DIM_SEQ;
        }
        graph.nodes[3].inputs = {0, 1};
        graph.nodes[4].inputs = {3, 3};
        graph.nodes[5].inputs = {0, 2};
        if (middle_gemm) {
            GraphNode constant;
            constant.id=6;constant.op_type=OpType::CONSTANT;constant.out_prec=Precision::FP16;
            constant.out_shape[0]=320;constant.out_shape[1]=320;
            auto& tensor=graph.runtime.tensors[6];
            tensor=Tensor::create(Precision::FP16,MemoryType::EXTERNAL,320,320,1,1,middle.data());
            backend.wrap_weight(tensor);
            graph.nodes[4].op_type=OpType::MATMUL;
            graph.nodes[4].inputs={3,6};
            graph.nodes.insert(graph.nodes.begin()+4,constant);
        }
        graph.graph_inputs = {0}; graph.graph_outputs = {3, 4, 5};
        if (rms) {
            for (int i=0;i<k;++i) gamma[i]=(i%31-15)/17.f;
            GraphNode weight,norm;
            weight.id=7;weight.op_type=OpType::CONSTANT;
            weight.out_shape[0]=k;weight.out_prec=Precision::FP32;
            auto& tensor=graph.runtime.tensors[7];
            tensor=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,k,1,1,1,gamma.data());
            backend.wrap_weight(tensor);
            norm.id=8;norm.op_type=OpType::GROUP_RMS_NORM;norm.inputs={0,7};
            norm.out_shape[0]=k;norm.dim_expr[1].kind=DIM_SEQ;
            norm.params.i32={2560,4};norm.params.f32={1e-6f};
            for (auto& node : graph.nodes)
                if (node.id==3 || node.id==5) node.inputs[0]=8;
            graph.nodes.insert(graph.nodes.begin()+3,weight);
            graph.nodes.insert(graph.nodes.begin()+4,norm);
            graph.graph_outputs.push_back(8);
        }
        if (gated) {
            GraphNode branch, gates, carry, user;
            branch.id=9; branch.op_type=OpType::INPUT; branch.out_shape[0]=2560;
            gates.id=10; gates.op_type=OpType::INPUT; gates.out_shape[0]=4;
            for (auto* node : {&branch,&gates}) {
                auto& t=graph.runtime.tensors[node->id];
                t=Tensor::create(Precision::FP32,MemoryType::NONE,node->out_shape[0],max_rows);
                backend.alloc_output(t,t.nbytes(),nullptr);
                t.shape[1]=1; t.compute_strides();
            }
            carry.id=11;carry.op_type=OpType::GR_INJECT;carry.inputs={9,0,10};
            carry.out_shape[0]=k;carry.dim_expr[1].kind=DIM_SEQ;carry.params.i32={2560,4};
            user.id=12;user.op_type=OpType::ADD;user.inputs={11,11};
            user.out_shape[0]=k;user.dim_expr[1].kind=DIM_SEQ;
            for (auto& node : graph.nodes) if (node.id==8) node.inputs[0]=11;
            graph.nodes.insert(graph.nodes.begin()+3,{branch,gates,carry});
            graph.nodes.push_back(user);
            graph.graph_inputs.insert(graph.graph_inputs.end(),{9,10});
            graph.graph_outputs.insert(graph.graph_outputs.end(),{11,12});
        }
        if(deferred) {
            GraphNode logits,sigmoid,reduce;
            logits.id=13;logits.op_type=OpType::INPUT;logits.out_shape[0]=k;
            auto& tensor=graph.runtime.tensors[13];
            tensor=Tensor::create(Precision::FP32,MemoryType::NONE,k,max_rows);
            backend.alloc_output(tensor,tensor.nbytes(),nullptr);
            tensor.shape[1]=1;tensor.compute_strides();
            sigmoid.id=14;sigmoid.op_type=OpType::SIGMOID;sigmoid.inputs={13};
            sigmoid.out_shape[0]=k;sigmoid.dim_expr[1].kind=DIM_SEQ;
            reduce.id=15;reduce.op_type=OpType::GR_REDUCE;reduce.inputs={8,14};
            reduce.out_shape[0]=2560;reduce.dim_expr[1].kind=DIM_SEQ;reduce.params.i32={2560,4};
            graph.nodes.insert(graph.nodes.begin(),logits);
            auto second=std::find_if(graph.nodes.begin(),graph.nodes.end(),[](const GraphNode& n){return n.id==5;});
            graph.nodes.insert(second,{sigmoid,reduce});
            graph.graph_inputs.push_back(13);
            graph.graph_outputs.erase(std::remove(graph.graph_outputs.begin(),graph.graph_outputs.end(),8),graph.graph_outputs.end());
            graph.graph_outputs.push_back(15);
        }
        context.graph = &graph; context.pool = &graph.runtime.pool; context.backend = &backend;
        context.runtime_seq_len = 1; context.reuse_same_shape_workspace = true;
        prepare_execution(context);
    }
    bool input(int m, int variant) {
        auto& tensor = graph.runtime.tensors[0];
        tensor.shape[1] = m; tensor.compute_strides();
        context.runtime_seq_len = m;
        if(deferred) {
            auto& raw=graph.runtime.tensors[13];raw.shape[1]=m;raw.compute_strides();
            std::vector<float> values(raw.nelements());
            for(size_t i=0;i<values.size();++i)values[i]=(int((i+variant)%41)-20)/7.f;
            if(!backend.copy_from_host(values.data(),raw,values.size()*sizeof(float)))return false;
        }
        std::vector<float> x(k * m);
        for (int i = 0; i < k * m; ++i) x[i] = (i % 41 - 20 + variant) / (37.f + variant);
        if (gated) for (int id : {9,10}) {
            auto& t=graph.runtime.tensors[id];t.shape[1]=m;t.compute_strides();
            std::vector<float> values(t.nelements());
            for (size_t i=0;i<values.size();++i)
                values[i]=id==9 ? (int(i%37)-18+variant)/(29.f+variant) : .1f+float((i+variant)%7)*.09f;
            if (!backend.copy_from_host(values.data(),t,values.size()*sizeof(float))) return false;
        }
        return backend.copy_from_host(x.data(), tensor, x.size() * sizeof(float));
    }
    bool run() {
        execute_graph(context);
        return !context.execution_failed && !backend.dispatch_failed();
    }
    bool dispatch_node(int id) {
        for (const auto& node : graph.nodes) if (node.id==static_cast<uint32_t>(id)) {
            std::vector<const Tensor*> inputs;
            for (auto input : node.inputs) inputs.push_back(&graph.runtime.tensors[input]);
            backend.dispatch(node,inputs,&graph.runtime.tensors[id],nullptr);
            return !backend.dispatch_failed();
        }
        return false;
    }
};
bool equal_nodes(Fixture& a, Fixture& b, const std::vector<int>& ids) {
    for (int id : ids) {
        const auto& expected = a.graph.runtime.tensors[id];
        const auto& actual = b.graph.runtime.tensors[id];
        std::vector<float> x(expected.nelements()), y(actual.nelements());
        if (x.size() != y.size() ||
            !a.backend.copy_to_host(expected, x.data(), x.size() * sizeof(float)) ||
            !b.backend.copy_to_host(actual, y.data(), y.size() * sizeof(float)) ||
            std::memcmp(x.data(), y.data(), x.size() * sizeof(float))) {
            std::fprintf(stderr, "HC projection mismatch at node %d\n", id); return false;
        }
    }
    return true;
}
bool equal(Fixture& a, Fixture& b) {
    return equal_nodes(a,b,std::vector<int>(a.graph.graph_outputs.begin(),a.graph.graph_outputs.end()));
}
} // namespace

int main(int argc,char** argv) {
    const bool compact=argc==2&&!std::strcmp(argv[1],"--compact");
    if(argc!=1&&!compact)return 2;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    // Deferred RMS is deliberately not a graph output. Its intermediate
    // allocation must be returned to a cache before captured execution.
    setenv("MOLLM_CUDA_OUTPUT_POOL_MB","1024",1);
    setenv("MOLLM_CUDA_HC_COMPACT_CARRY","0",1);
    setenv("MOLLM_CUDA_HC_PROJECTION_PAIR", "0", 1);
    setenv("MOLLM_CUDA_HC_PREFILL_ACTIVATION_REUSE", "0", 1);
    setenv("MOLLM_CUDA_HC_PREFILL_RMS_HALF", "0", 1);
    CudaBackend serial;
    setenv("MOLLM_CUDA_HC_PROJECTION_PAIR", "1", 1);
    setenv("MOLLM_CUDA_HC_PREFILL_ACTIVATION_REUSE", "1", 1);
    setenv("MOLLM_CUDA_HC_PREFILL_RMS_HALF", "1", 1);
    setenv("MOLLM_CUDA_GR_INJECT_RMS_FUSION", "1", 1);
    setenv("MOLLM_CUDA_SIGMOID_GR_REDUCE_FUSION", "1", 1);
    setenv("MOLLM_CUDA_HC_PREFILL_DEFERRED_RMS", "1", 1);
    setenv("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS", "1", 1);
    setenv("MOLLM_CUDA_HC_COMPACT_CARRY",compact?"1":"0",1);
    CudaBackend fused;
    if (!serial.available() || !fused.available()) return 1;
    serial.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    fused.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    Fixture reference(serial), candidate(fused);
    if (!reference.input(1, 0) || !candidate.input(1, 0) ||
        !reference.run() || !candidate.run() || !equal(reference, candidate)) return 1;
    // Capture actual backend dispatch + delayed D2D copy, not just the kernel.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess ||
        !candidate.run() || cudaStreamEndCapture(cudaStreamPerThread, &graph) != cudaSuccess ||
        cudaGraphInstantiateWithFlags(&executable, graph, 0) != cudaSuccess) return 1;
    for (int variant : {1, 2}) {
        if (!reference.input(1, variant) || !candidate.input(1, variant) || !reference.run() ||
            cudaGraphLaunch(executable, cudaStreamPerThread) != cudaSuccess ||
            cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess || !equal(reference, candidate)) return 1;
    }
    cudaGraphExecDestroy(executable); cudaGraphDestroy(graph);
    // A new partial pass must not reuse a prior pass's injection values.
    if (!reference.input(1, 3) || !candidate.input(1, 3) || !reference.run()) return 1;
    fused.begin_execution();
    fused.dispatch(candidate.graph.nodes[5],
        {&candidate.graph.runtime.tensors[0], &candidate.graph.runtime.tensors[2]},
        &candidate.graph.runtime.tensors[5], nullptr);
    std::vector<float> x(4), y(4);
    if (!serial.copy_to_host(reference.graph.runtime.tensors[5], x.data(), 16) ||
        !fused.copy_to_host(candidate.graph.runtime.tensors[5], y.data(), 16) ||
        std::memcmp(x.data(), y.data(), 16)) return 1;
    // Multi-token fallback followed by return to the fused M1 path.
    for (int m : {2, 1}) {
        if (!reference.input(m, 4) || !candidate.input(m, 4) ||
            !reference.run() || !candidate.run() || !equal(reference, candidate)) return 1;
    }
    // Prefill conversion reuse must remain fresh across shape changes and
    // graph replays. Compare entire FP32 projection outputs byte-for-byte.
    for (int m : {17,65,256,857,17}) {
        if (!reference.input(m,6) || !candidate.input(m,6) ||
            !reference.run() || !candidate.run() || !equal(reference,candidate)) return 1;
        cudaGraph_t prefill_graph=nullptr;
        cudaGraphExec_t prefill_exec=nullptr;
        if (cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)!=cudaSuccess ||
            !candidate.run() || cudaStreamEndCapture(cudaStreamPerThread,&prefill_graph)!=cudaSuccess ||
            cudaGraphInstantiateWithFlags(&prefill_exec,prefill_graph,0)!=cudaSuccess) return 1;
        for (int variant : {7,8,9}) {
            if (!reference.input(m,variant) || !candidate.input(m,variant) || !reference.run() ||
                cudaGraphLaunch(prefill_exec,cudaStreamPerThread)!=cudaSuccess ||
                cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess || !equal(reference,candidate)) return 1;
        }
        cudaGraphExecDestroy(prefill_exec);cudaGraphDestroy(prefill_graph);
    }
    // Re-registering the same graph replaces metadata without stale pointers.
    prepare_execution(candidate.context);
    if (!reference.input(1, 5) || !candidate.input(1, 5) ||
        !reference.run() || !candidate.run() || !equal(reference, candidate)) return 1;
    // An intervening FP16 GEMM overwrites ordinary activation scratch.
    // The original HC input conversion must survive in its dedicated buffer.
    Fixture middle_reference(serial,true),middle_candidate(fused,true);
    for (int m : {17,857}) {
        if (!middle_reference.input(m,10) || !middle_candidate.input(m,10) ||
            !middle_reference.run() || !middle_candidate.run() || !equal(middle_reference,middle_candidate)) return 1;
        cudaGraph_t middle_graph=nullptr;cudaGraphExec_t middle_exec=nullptr;
        if (cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)!=cudaSuccess ||
            !middle_candidate.run() || cudaStreamEndCapture(cudaStreamPerThread,&middle_graph)!=cudaSuccess ||
            cudaGraphInstantiateWithFlags(&middle_exec,middle_graph,0)!=cudaSuccess) return 1;
        for (int variant : {11,12}) {
            if (!middle_reference.input(m,variant) || !middle_candidate.input(m,variant) || !middle_reference.run() ||
                cudaGraphLaunch(middle_exec,cudaStreamPerThread)!=cudaSuccess ||
                cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess || !equal(middle_reference,middle_candidate)) return 1;
        }
        cudaGraphExecDestroy(middle_exec);cudaGraphDestroy(middle_graph);
    }
    Fixture rms_reference(serial,true,true),rms_candidate(fused,true,true);
    for (int m : {2,17,65,256,857,17}) {
        if (!rms_reference.input(m,13) || !rms_candidate.input(m,13) ||
            !rms_reference.run() || !rms_candidate.run() || !equal(rms_reference,rms_candidate)) return 1;
        cudaGraph_t rms_graph=nullptr;cudaGraphExec_t rms_exec=nullptr;
        if (cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)!=cudaSuccess ||
            !rms_candidate.run() || cudaStreamEndCapture(cudaStreamPerThread,&rms_graph)!=cudaSuccess ||
            cudaGraphInstantiateWithFlags(&rms_exec,rms_graph,0)!=cudaSuccess) return 1;
        for (int variant : {14,15,16}) {
            if (!rms_reference.input(m,variant) || !rms_candidate.input(m,variant) || !rms_reference.run() ||
                cudaGraphLaunch(rms_exec,cudaStreamPerThread)!=cudaSuccess ||
                cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess || !equal(rms_reference,rms_candidate)) return 1;
        }
        cudaGraphExecDestroy(rms_exec);cudaGraphDestroy(rms_graph);
    }
    // Producer-only state must not be mistaken for first-projection state.
    // Also exercise an execution boundary between producer and second use.
    for (bool new_epoch : {false,true}) {
        const int variant=new_epoch?18:17;
        if (!rms_reference.input(17,variant) || !rms_candidate.input(17,variant) ||
            !rms_reference.run()) return 1;
        fused.begin_execution();
        if (!rms_candidate.dispatch_node(8)) return 1;
        if (new_epoch) fused.begin_execution();
        if (!rms_candidate.dispatch_node(5) ||
            !equal_nodes(rms_reference,rms_candidate,{8,5})) return 1;
    }
    // Partial projection execution without a fresh producer must cast input.
    if (!rms_reference.input(17,19) || !rms_reference.run()) return 1;
    const auto& normalized=rms_reference.graph.runtime.tensors[8];
    std::vector<float> fresh(normalized.nelements());
    if (!serial.copy_to_host(normalized,fresh.data(),fresh.size()*sizeof(float)) ||
        !fused.copy_from_host(fresh.data(),rms_candidate.graph.runtime.tensors[8],fresh.size()*sizeof(float))) return 1;
    fused.begin_execution();
    if (!rms_candidate.dispatch_node(3) || !rms_candidate.dispatch_node(5) ||
        !equal_nodes(rms_reference,rms_candidate,{8,3,5})) return 1;
    prepare_execution(rms_candidate.context);
    if (!rms_reference.input(17,20) || !rms_candidate.input(17,20) ||
        !rms_reference.run() || !rms_candidate.run() || !equal(rms_reference,rms_candidate)) return 1;
    // Real dual-output fusion + HC half-shadow publication. The baseline has
    // HC reuse disabled and must execute the ordinary inject/RMS/projections.
    Fixture gated_reference(serial,true,true,true),gated_candidate(fused,true,true,true);
    for (int m : {1,2,17,65,256,17}) {
        const size_t before=fused.gr_inject_rms_fused_calls();
        if (!gated_reference.input(m,21) || !gated_candidate.input(m,21) ||
            !gated_reference.run() || !gated_candidate.run() ||
            !equal(gated_reference,gated_candidate) ||
            fused.gr_inject_rms_fused_calls() != before+(m>=16?1:0)) return 1;
        cudaGraph_t captured=nullptr;cudaGraphExec_t replay=nullptr;
        if (cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)!=cudaSuccess ||
            !gated_candidate.run() || cudaStreamEndCapture(cudaStreamPerThread,&captured)!=cudaSuccess ||
            cudaGraphInstantiateWithFlags(&replay,captured,0)!=cudaSuccess) return 1;
        for (int variant : {22,23,24}) {
            if (!gated_reference.input(m,variant) || !gated_candidate.input(m,variant) ||
                !gated_reference.run() || cudaGraphLaunch(replay,cudaStreamPerThread)!=cudaSuccess ||
                cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess ||
                !equal(gated_reference,gated_candidate)) return 1;
        }
        cudaGraphExecDestroy(replay);cudaGraphDestroy(captured);
    }
    Fixture deferred_reference(serial,true,true,true,true,mollm::hc::prefill_max_rows),deferred_candidate(fused,true,true,true,true,mollm::hc::prefill_max_rows);
    for(int m:{2,17,65,256,719,1024,1025,1743,2048,17}) {
        const auto before=fused.deferred_rms_producer_calls();
        if(!deferred_reference.input(m,25)||!deferred_candidate.input(m,25)||
           !deferred_reference.run()||!deferred_candidate.run()||!equal(deferred_reference,deferred_candidate)||
           fused.deferred_rms_producer_calls()!=before+(m>=16?1:0))return 1;
        // First changed-shape pass can recycle non-output RMS storage into
        // another graph output. A same-shape pass establishes all retained
        // workspace allocations before capturing their reuse.
        if(!deferred_candidate.run()||!equal(deferred_reference,deferred_candidate))return 1;
        cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;
        if(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)!=cudaSuccess||
           !deferred_candidate.run()||cudaStreamEndCapture(cudaStreamPerThread,&g)!=cudaSuccess||
           cudaGraphInstantiateWithFlags(&e,g,0)!=cudaSuccess)return 1;
        for(int variant:{26,27,28}) {
            if(!deferred_reference.input(m,variant)||!deferred_candidate.input(m,variant)||!deferred_reference.run()||
               cudaGraphLaunch(e,cudaStreamPerThread)!=cudaSuccess||cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess||
               !equal(deferred_reference,deferred_candidate))return 1;
        }
        cudaGraphExecDestroy(e);cudaGraphDestroy(g);
    }
    if(fused.deferred_rms_materializations()!=0)return 1;
    std::printf("Deferred HC RMS backend: producers=%zu normal_pipeline_materializations=0\n",fused.deferred_rms_producer_calls());
    auto node=[&](int id)->const GraphNode& {
        for(const auto& n:deferred_candidate.graph.nodes)if(n.id==uint32_t(id))return n;
        std::abort();
    };
    // Exercise the same lifetime/fallback contracts for both producers:
    // fused inject/RMS and an independent RMS following a scheduling cut.
    for(int producer=0;producer<2;++producer) for(int mode=0;mode<7;++mode) {
        unsetenv("MOLLM_TEST_CUDA_HC_STANDALONE_DEFERRED_RMS");
        if(!deferred_reference.input(17,40+mode)||!deferred_candidate.input(17,40+mode)||
           !deferred_reference.run()||!deferred_candidate.run())return 1;
        auto& t=deferred_candidate.graph.runtime.tensors;
        fused.begin_execution();
        const auto before=fused.deferred_rms_materializations();
        const auto producers_before=fused.deferred_rms_producer_calls();
        if(producer==0) {
            if(!fused.dispatch_gr_inject_rms(node(11),node(8),{&t[9],&t[0],&t[10],&t[7]},&t[11],&t[8],nullptr))return 1;
        } else {
            setenv("MOLLM_TEST_CUDA_HC_STANDALONE_DEFERRED_RMS","1",1);
            if(!deferred_candidate.dispatch_node(8))return 1;
        }
        if(fused.deferred_rms_producer_calls()!=producers_before+1)return 1;
        std::vector<float> normalized(t[8].nelements());
        if(!serial.copy_to_host(deferred_reference.graph.runtime.tensors[8],normalized.data(),normalized.size()*4))return 1;
        if(mode==0) {
            Tensor view=t[8];view.device.offset+=4;view.shape[0]=k-1;view.shape[1]=1;view.compute_strides();
            std::vector<float> actual(k-1);
            if(!fused.copy_to_host(view,actual.data(),actual.size()*4)||std::memcmp(actual.data(),normalized.data()+1,actual.size()*4))return 1;
        } else if(mode==1) {
            Tensor view=t[7];view.device.offset+=4;view.shape[0]=1;view.compute_strides();
            const float original=deferred_candidate.gamma[1],changed=1.75f;
            if(!fused.copy_from_host(&changed,view,4)||!fused.copy_from_host(&original,view,4))return 1;
        } else if(mode==2) {
            fused.begin_execution();
        } else if(mode==3) {
            prepare_execution(deferred_candidate.context);
        } else if(mode==4) {
            fused.free_output(t[11],deferred_candidate.context.pool);
            t[11].data=nullptr;t[11].device.buffer=nullptr;t[11].device.offset=0;
            t[11].mem_type=MemoryType::NONE;t[11].owner_id=t[11].storage_id=0;
        } else if(mode==5) {
            Tensor view=t[8];view.device.offset+=4;view.shape[0]=k-1;view.shape[1]=1;view.compute_strides();
            Tensor output=Tensor::create(Precision::FP32,MemoryType::NONE,k-1);
            if(!fused.alloc_output(output,output.nbytes(),nullptr))return 1;
            GraphNode add;add.op_type=OpType::ADD;
            fused.dispatch(add,{&view,&view},&output,nullptr);
            std::vector<float> actual(k-1);
            if(!fused.copy_to_host(output,actual.data(),actual.size()*4))return 1;
            for(int i=0;i<k-1;++i)if(actual[i]!=normalized[i+1]+normalized[i+1])return 1;
            fused.free_output(output,nullptr);
        } else {
            Tensor view=t[8],raw=t[13];
            view.device.offset+=k*4;raw.device.offset+=k*4;
            view.shape[1]=raw.shape[1]=16;view.compute_strides();raw.compute_strides();
            Tensor output=Tensor::create(Precision::FP32,MemoryType::NONE,2560,16);
            if(!fused.alloc_output(output,output.nbytes(),nullptr)||
               !fused.dispatch_sigmoid_gr_reduce(node(15),{&view,&raw},&output,nullptr))return 1;
            std::vector<float> actual(output.nelements()),expected(actual.size());
            if(!fused.copy_to_host(output,actual.data(),actual.size()*4)||
               !serial.copy_to_host(deferred_reference.graph.runtime.tensors[15],expected.data(),expected.size()*4,2560*4)||
               std::memcmp(actual.data(),expected.data(),actual.size()*4))return 1;
            fused.free_output(output,nullptr);
        }
        if(fused.deferred_rms_materializations()!=before+1)return 1;
        std::vector<float> actual(normalized.size());
        if(!fused.copy_to_host(t[8],actual.data(),actual.size()*4)||
           std::memcmp(actual.data(),normalized.data(),actual.size()*4))return 1;
    }
    unsetenv("MOLLM_TEST_CUDA_HC_STANDALONE_DEFERRED_RMS");
    if(compact) {
        const auto registrations=fused.compact_carry_allocation_registrations();
        for(int rows:{1,2,15,2049}) {
            Tensor excluded=Tensor::create(Precision::FP32,MemoryType::NONE,k,rows);
            if(!fused.alloc_output(excluded,excluded.nbytes(),nullptr))return 1;
            fused.free_output(excluded,nullptr);
        }
        for(int width:{4,320,2560}) {
            Tensor excluded=Tensor::create(Precision::FP32,MemoryType::NONE,width,17);
            if(!fused.alloc_output(excluded,excluded.nbytes(),nullptr))return 1;
            fused.free_output(excluded,nullptr);
        }
        if(fused.compact_carry_allocation_registrations()!=registrations)return 1;
        auto& t=deferred_candidate.graph.runtime.tensors;
        auto& r=deferred_reference.graph.runtime.tensors;
        for(int rows:{17,1743,2048}) {
        // Keep carry packed through an execution/scheduling boundary, then
        // consume it as the next residual without a full FP32 restoration.
        if(!deferred_reference.input(rows,61)||!deferred_candidate.input(rows,61)||
           !deferred_reference.run()||!deferred_candidate.run())return 1;
        // A changed-shape first pass releases non-output RMS storage. Establish
        // retained same-shape workspace before manually dispatching its node.
        if(!deferred_reference.run()||!deferred_candidate.run())return 1;
        fused.begin_execution();
        if(!fused.dispatch_gr_inject_rms(node(11),node(8),{&t[9],&t[0],&t[10],&t[7]},&t[11],&t[8],nullptr))return 1;
        std::printf("compact continuation rows=%d first_producer=%zu restores=%zu\n",rows,fused.compact_carry_producer_calls(),fused.compact_carry_restorations());
        fused.begin_execution();
        Tensor next=Tensor::create(Precision::FP32,MemoryType::NONE,k,rows),expected=next;
        if(!fused.alloc_output(next,next.nbytes(),nullptr)||!serial.alloc_output(expected,expected.nbytes(),nullptr))return 1;
        const auto reads=fused.compact_carry_residual_reads();
        if(!fused.dispatch_gr_inject_rms(node(11),node(8),{&t[9],&t[11],&t[10],&t[7]},&next,&t[8],nullptr)||
           fused.compact_carry_residual_reads()!=reads+1){std::fprintf(stderr,"compact continuation rows=%d residual reads=%zu expected=%zu\n",rows,fused.compact_carry_residual_reads(),reads+1);return 1;}
        serial.dispatch(node(11),{&r[9],&r[11],&r[10]},&expected,nullptr);
        serial.dispatch(node(8),{&expected,&r[7]},&r[8],nullptr);
        std::vector<float> x(next.nelements()),y(x.size());
        if(!serial.copy_to_host(expected,x.data(),x.size()*4)||!fused.copy_to_host(next,y.data(),y.size()*4)||
           std::memcmp(x.data(),y.data(),x.size()*4)||!equal_nodes(deferred_reference,deferred_candidate,{8}))return 1;
        // Free a newly packed owner while normalized data still depends on it.
        // The normalized result must survive; a recycled address must not carry
        // stale compact metadata into an ordinary FP32 zero/write/read.
        if(!fused.dispatch_gr_inject_rms(node(11),node(8),{&t[9],&t[11],&t[10],&t[7]},&next,&t[8],nullptr))return 1;
        void* recycled=next.device.buffer;
        fused.free_output(next,nullptr);serial.free_output(expected,nullptr);
        std::vector<Tensor> held;
        bool reused=false;const auto restores=fused.compact_carry_restorations();
        for(int attempt=0;attempt<64&&!reused;++attempt){
            Tensor replacement=Tensor::create(Precision::FP32,MemoryType::NONE,k,rows);
            if(!fused.alloc_output(replacement,replacement.nbytes(),nullptr))return 1;
            reused=replacement.device.buffer==recycled;
            if(!fused.zero_tensor(replacement,replacement.nbytes()))return 1;
            held.push_back(replacement);
        }
        if(!reused||fused.compact_carry_restorations()!=restores)return 1;
        if(!fused.copy_to_host(held.back(),y.data(),y.size()*4))return 1;
        for(float v:y)if(v!=0.f)return 1;
        for(auto& tensor:held)fused.free_output(tensor,nullptr);
        // Partial host reads/writes and zeroing operate on the whole logical
        // owner first, preserving untouched bytes and deferred normalized data.
        for(int mode=0;mode<3;++mode){
            if(!deferred_reference.input(rows,62+mode)||!deferred_candidate.input(rows,62+mode)||
               !deferred_reference.run()||!deferred_candidate.run())return 1;
            if(!serial.copy_to_host(r[11],x.data(),x.size()*4))return 1;
            fused.begin_execution();
            if(!fused.dispatch_gr_inject_rms(node(11),node(8),{&t[9],&t[0],&t[10],&t[7]},&t[11],&t[8],nullptr))return 1;
            Tensor view=t[11];view.device.offset+=4;view.shape[0]=8;view.shape[1]=1;view.compute_strides();
            if(mode==0){float part[8];if(!fused.copy_to_host(view,part,sizeof(part))||std::memcmp(part,x.data()+1,sizeof(part)))return 1;}
            else if(mode==1){const float change=1.234567f;if(!fused.copy_from_host(&change,view,4))return 1;x[1]=change;}
            else {if(!fused.zero_tensor(view,8))return 1;x[1]=x[2]=0.f;}
            if(!fused.copy_to_host(t[11],y.data(),y.size()*4)||std::memcmp(x.data(),y.data(),x.size()*4)||
               !equal_nodes(deferred_reference,deferred_candidate,{8}))return 1;
        }
        if(!fused.compact_carry_producer_calls()||!fused.compact_carry_consumer_calls()||!fused.compact_carry_restorations())return 1;
        std::printf("Compact HC backend rows=%d: residual continuation, packed-owner free/address reuse, partial host read/write/zero byte-exact PASS\n",rows);
        }
    }
    std::printf("Deferred HC RMS backend: two producers x fallback/view/overwrite/epoch/replacement/release cases=14 materializations=%zu exact=1\n",fused.deferred_rms_materializations());
    std::printf("GR inject RMS backend: dual-output/HC projections byte-exact, changed-input graph replay, shape fallback; handled_calls=%zu\n",fused.gr_inject_rms_fused_calls());
    std::puts("HC projection backend: bit-exact RMS shadow/projections, graph replays, partial-pass invalidation, M2 fallback and plan replacement passed");
    return 0;
}
