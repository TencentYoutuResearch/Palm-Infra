#include "backends/cpu/backend.h"
#include "graph/execute.h"
#include "kernels/cpu/matmul/matmul.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
}  // namespace

int main() {
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
