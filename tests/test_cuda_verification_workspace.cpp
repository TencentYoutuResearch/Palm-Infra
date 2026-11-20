#include "backends/cuda/state_prefix_plan.h"
#include "backends/cpu/backend.h"
#include "backends/cuda/backend.h"
#include "graph/execute.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
Tensor device_tensor(CudaBackend& backend, int64_t cols, int64_t rows = 1) {
    Tensor tensor = Tensor::create(Precision::FP32, MemoryType::NONE, cols, rows);
    require(backend.alloc_output(tensor, tensor.nbytes(), nullptr), "allocate fixture tensor");
    return tensor;
}
void upload(CudaBackend& backend, Tensor& tensor, const std::vector<float>& values) {
    require(values.size() * sizeof(float) == tensor.nbytes(), "upload extent");
    require(backend.copy_from_host(values.data(), tensor, tensor.nbytes()), "upload");
}
std::vector<float> download(CudaBackend& backend, const Tensor& tensor) {
    std::vector<float> values(tensor.nelements());
    require(backend.copy_to_host(tensor, values.data(), values.size() * sizeof(float)), "download");
    return values;
}
void exact(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size() &&
            !std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)),
            "output bits differ from independent native evaluation");
}
void history(const std::vector<float>& actual, const std::vector<float>& input, int kept) {
    require(actual.size() == 24, "history extent");
    for (int group = 0; group < 8; ++group) {
        for (int tap = 0; tap < 3; ++tap) {
            const int row = kept - 3 + tap;
            require(actual[group * 3 + tap] == (row < 0 ? .5f : input[row * 8 + group]),
                    "accepted history prefix differs");
        }
    }
}
struct Captures {
    std::map<int, cudaGraphExec_t> graphs;
    void clear() {
        for (auto& entry : graphs) cudaGraphExecDestroy(entry.second);
        graphs.clear();
    }
    ~Captures() { clear(); }
};
struct CaptureScope {
    bool active = true;
    CaptureScope() {
        check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    }
    cudaGraph_t finish() {
        cudaGraph_t graph = nullptr;
        const auto error = cudaStreamEndCapture(cudaStreamPerThread, &graph);
        active = false;
        check(error);
        return graph;
    }
    ~CaptureScope() {
        if (active) {
            cudaGraph_t graph = nullptr;
            cudaStreamEndCapture(cudaStreamPerThread, &graph);
            if (graph) cudaGraphDestroy(graph);
            cudaGetLastError();
        }
    }
};

int run() {
    unsetenv("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS");
    setenv("MOLLM_CUDA_MTP_STATE_PREFIXES", "1", 1);
    setenv("MOLLM_CUDA_OUTPUT_POOL_MB", "64", 1);
    CudaBackend backend;
    if (!backend.available()) return 77;
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    CudaBackend reference;
    reference.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    Tensor reference_input = device_tensor(reference, 8, 8);
    Tensor reference_weight = device_tensor(reference, 4, 8);
    Tensor reference_state = device_tensor(reference, 8, 3);
    Tensor reference_output = device_tensor(reference, 8, 8);
    upload(reference, reference_weight, std::vector<float>(32, .25f));

    Graph graph;
    graph.nodes.resize(5);
    graph.runtime.tensors.resize(5);
    auto& tensors = graph.runtime.tensors;
    tensors[0] = device_tensor(backend, 8, 8);
    tensors[1] = device_tensor(backend, 4, 8);
    tensors[2] = Tensor::create(Precision::FP32, MemoryType::NONE, 8, 3);
    backend.alloc_persistent(tensors[2], tensors[2].nbytes(), PersistentHostAccess::NONE);
    tensors[3] = Tensor::create(Precision::FP32, MemoryType::NONE, 8, 2);
    tensors[4] = Tensor::create(Precision::FP32, MemoryType::NONE, 8, 2);
    for (uint32_t i = 0; i < graph.nodes.size(); ++i) {
        auto& node = graph.nodes[i];
        node.id = i;
        node.op_type = OpType::INPUT;
        node.out_prec = Precision::FP32;
        std::copy(tensors[i].shape, tensors[i].shape + 4, node.out_shape);
    }
    graph.nodes[1].op_type = OpType::CONSTANT;
    auto& convolution = graph.nodes[3];
    convolution.op_type = OpType::SHORTCONV;
    convolution.inputs = {0, 1, 2};
    convolution.params.i32 = {4, 2};
    graph.nodes[0].dim_expr[1] = convolution.dim_expr[1] = {DIM_SEQ, 0};
    auto& view = graph.nodes[4];
    view.op_type = OpType::RESHAPE;
    view.inputs = {3};
    view.dim_expr[1] = {DIM_SEQ, 0};
    graph.graph_inputs = {0, 2};
    graph.graph_outputs = {4};
    ExecContext context;
    context.graph = &graph;
    context.pool = &graph.runtime.pool;
    context.backend = &backend;
    context.runtime_seq_len = 2;
    prepare_execution(context, false);
    upload(backend, tensors[1], std::vector<float>(32, .25f));
    Captures captures; // destroyed before Graph, including on test failure
    const std::vector<float> initial(24, .5f);
    std::map<int, void*> pointers;
    int passes = 0;
    auto forward = [&](int width, int salt) {
        require(backend.select_verification_workspace(graph, context, width), "select width");
        const bool prepared = context.workspace_shape_valid;
        std::vector<float> input(8 * width);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = float(int((i * 17 + salt) % 31) - 15) / 32.f;
        upload(backend, tensors[0], input);
        upload(backend, tensors[2], initial);
        require(backend.begin_state_prefixes({&tensors[2]}, width), "begin prefixes");
        require(backend.state_prefixes_cover_graph(graph, {&tensors[2]}, width) == prepared,
                "restored width must retain its own proven prefix plan");
        require(backend.set_retained_verification(true, prepared), "retain verifier");
        context.reuse_same_shape_workspace = context.workspace_shape_valid = true;
        context.workspace_runtime_seq_len = width;
        context.workspace_runtime_batch = context.runtime_batch;
        context.workspace_static_padded = false;
        context.workspace_padded_seq_len = -1;

        reference_input.shape[1] = reference_output.shape[1] = width;
        reference_input.compute_strides();
        reference_output.compute_strides();
        upload(reference, reference_input, input);
        upload(reference, reference_state, initial);
        require(reference.begin_state_prefixes({&reference_state}, width), "reference prefixes");
        reference.dispatch(convolution, {&reference_input, &reference_weight, &reference_state},
                           &reference_output, nullptr);
        reference.end_graph();
        require(!reference.dispatch_failed(), "reference dispatch");
        const auto expected = download(reference, reference_output);
        reference.end_state_prefixes();
        if (captures.graphs.count(width)) {
            check(cudaGraphLaunch(captures.graphs[width], cudaStreamPerThread));
            exact(download(backend, tensors[3]), expected);
            history(download(backend, tensors[2]), input, width);
            upload(backend, tensors[2], initial);
        }
        execute_graph(context);
        backend.end_graph();
        require(!context.execution_failed && !backend.dispatch_failed() &&
                backend.state_prefixes_complete(), "complete native prefixes");
        exact(download(backend, tensors[3]), expected);
        for (int kept = 1; kept <= width; ++kept) {
            require(backend.restore_state_prefix(kept), "restore prefix");
            history(download(backend, tensors[2]), input, kept);
        }
        if (!captures.graphs.count(width)) {
            CaptureScope scope;
            backend.dispatch(convolution, {&tensors[0], &tensors[1], &tensors[2]}, &tensors[3], nullptr);
            auto captured = scope.finish();
            cudaGraphExec_t executable = nullptr;
            const auto result = cudaGraphInstantiateWithFlags(&executable, captured, 0);
            cudaGraphDestroy(captured);
            check(result);
            captures.graphs[width] = executable;
        }
        require(!pointers.count(width) || pointers[width] == tensors[3].device.buffer,
                "per-width allocation address changed");
        pointers[width] = tensors[3].device.buffer;
        require(backend.set_retained_verification(false, false), "leave verifier");
        backend.end_state_prefixes();
        context.reuse_same_shape_workspace = false;
        ++passes;
    };
    const int max_width = backend.verification_workspace_max_rows();
    require(max_width == 5 || max_width == 8,"unexpected verifier capability");
    using namespace mollm::cuda_backend_detail;
    require(verification_workspace_budget_bytes(5)==size_t(1536)*1024*1024 &&
            verification_prefix_budget_bytes(5)==size_t(2048)*1024*1024 &&
            verification_workspace_budget_bytes(6)==size_t(3072)*1024*1024 &&
            verification_prefix_budget_bytes(8)==size_t(4096)*1024*1024,
            "narrow/wide live payload budgets changed");
    std::vector<int> widths;
    for(int width=2; width<=max_width; ++width)widths.push_back(width);
    for (int round = 0; round < 4; ++round) {
        for(int width:widths)forward(width,round*37+passes);
        for(auto it=widths.rbegin(); it!=widths.rend(); ++it)forward(*it,round*37+passes);
    }
    require(pointers.size() == widths.size(), "missing retained width");
    for (int width : widths)
        for (int other = width + 1; other <= max_width; ++other)
            require(pointers[width] != pointers[other], "widths share owned output");
    {
        Graph other;
        require(initialize_execution_frame(graph, other, {}), "independent graph owner");
        ExecContext other_context;
        other_context.graph = &other;
        other_context.backend = &backend;
        require(!backend.select_verification_workspace(other, other_context, 3), "foreign graph admitted");
        CPUBackend cpu;
        require(cpu.verification_workspace_max_rows()==0,"CPU advertised a CUDA workspace");
        other_context.backend = &cpu;
        require(cpu.select_verification_workspace(other, other_context, 3) &&
                cpu.release_verification_workspaces(other, other_context) &&
                !other_context.workspace_shape_valid, "unsupported backend changed workspace");
    }
    {
        auto check_unsupported = [&](CudaBackend& unsupported, int batch = -1) {
            ExecContext empty;
            empty.runtime_batch = batch;
            empty.graph = &graph;
            empty.backend = &unsupported;
            void* output = tensors[3].device.buffer;
            require(unsupported.select_verification_workspace(graph, empty, 3) &&
                    !empty.workspace_shape_valid && tensors[3].device.buffer == output,
                    "unsupported cache policy mutated graph storage");
        };
        setenv("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS", "1", 1);
        CudaBackend disabled_segments;
        unsetenv("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS");
        check_unsupported(disabled_segments);
        unsetenv("MOLLM_CUDA_MTP_STATE_PREFIXES");
        CudaBackend disabled_prefixes;
        check_unsupported(disabled_prefixes);
        setenv("MOLLM_CUDA_MTP_STATE_PREFIXES", "1", 1);
        CudaBackend multiple_batches;
        check_unsupported(multiple_batches, 2);
    }
    auto* before = tensors[3].device.buffer;
    const int prior = context.workspace_runtime_seq_len;
    {
        CaptureScope scope;
        require(!backend.select_verification_workspace(graph, context, prior) &&
                !backend.select_verification_workspace(graph, context, 2), "foreign capture admitted");
        auto captured = scope.finish();
        if (captured) cudaGraphDestroy(captured);
    }
    require(tensors[3].device.buffer == before && context.workspace_runtime_seq_len == prior,
            "capture rejection mutated workspace");
    forward(2, 999);
    void* state = tensors[2].device.buffer;
    captures.clear(); // caller-owned readers retire before backend-owned storage
    tensors[4] = tensors[2];
    tensors[4].mem_type = MemoryType::POOLED; // borrowed persistent owner in a partial pass
    require(backend.prepare_prefill_phase([] {}, &graph, &context) &&
            !tensors[3].data && !context.workspace_shape_valid && tensors[2].device.buffer == state,
            "ordinary prefill failed to retire cached storage");
    pointers.clear();
    forward(3, 1234);
    captures.clear();
    require(backend.release_verification_workspaces(graph, context), "explicit reset release");
    int releases = 0;
    require(backend.prepare_prefill_phase([&] { ++releases; }, &graph, &context) && releases == 1,
            "reset lost the pending pool-reclamation marker");
    pointers.clear();
    forward(3, 1235);
    captures.clear();
    Tensor rejected = Tensor::create(Precision::FP32, MemoryType::NONE, 32 * 1024 * 1024);
    require(!backend.alloc_output(rejected, rejected.nbytes(), nullptr) && backend.dispatch_failed(),
            "oversized request must be rejected before physical allocation");
    require(backend.release_verification_workspaces(graph, context) && backend.dispatch_failed(),
            "failed teardown must release storage without clearing the execution error");
    backend.clear_dispatch_error();
    require(!tensors[3].data && !context.workspace_shape_valid && tensors[2].device.buffer == state,
            "failure cleanup changed persistent owner");
    pointers.clear();
    forward(2, 11);
    forward(5, 12); // backend destructor sees a dormant bank after Graph destruction
    std::printf("verification_workspace transitions=%d exact outputs/history, captured replay, "
                "foreign owner/capture rejection, borrowed owners, allocation failure recovery, "
                "phase/reset cleanup PASS\n", passes);
    return 0;
}
} // namespace
int main() {
    try { return run(); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "verification_workspace: %s\n", error.what());
        return 1;
    }
}
