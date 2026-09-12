#include "backends/cpu/backend.h"
#include "backends/cpu/attention_ops.h"
#include "backends/cpu/deepseek_v4_ops.h"
#include "backends/cpu/layout_ops.h"
#include "backends/cpu/matmul_ops.h"
#include "backends/cpu/moe_ops.h"
#include "backends/cpu/normalization_ops.h"
#include "backends/cpu/recurrent_ops.h"
#include "graph/graph.h"
#include "kernels/cpu/elementwise.h"
#include "kernels/cpu/gated_residual.h"
#include "kernels/cpu/hyper_connection.h"
#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/ple.h"
#include "kernels/cpu/rope.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

void CPUBackend::begin_execution() {
    matmul_reset_activation_cache();
}

// ---------------------------------------------------------------------------
// CPUBackend::dispatch — CPU graph dispatcher.
//
// Routes OpType to the appropriate kernel. This is the only dispatch
// entry point for CPU; future NPU backend will have its own dispatch().
// ---------------------------------------------------------------------------

void CPUBackend::dispatch(const GraphNode& node,
                          const std::vector<const Tensor*>& inputs,
                          Tensor* output, ThreadPool* thread_pool) {
    const OpType op = node.op_type;
    const OpParams& params = node.params;
    auto reject = [&] {
        dispatch_failed_ = true;
    };
    auto has_inputs = [&](size_t count) {
        return inputs.size() >= count &&
               std::all_of(
                   inputs.begin(), inputs.begin() + count,
                   [](const Tensor* input) { return input != nullptr; });
    };
    switch (op) {
    case OpType::INPUT:
    case OpType::CONSTANT:
    case OpType::RESHAPE:
    case OpType::CONCAT:
    case OpType::SLICE:
    case OpType::TILE:
    case OpType::PERMUTE:
    case OpType::CONTIGUOUS:
        dispatch_cpu_layout(node, inputs, output);
        break;

    case OpType::MATMUL:
    case OpType::MATMUL_BATCH:
    case OpType::GEMV_SPARSE_A:
        dispatch_cpu_matmul(node, inputs, output, thread_pool);
        break;

    case OpType::RMS_NORM_ROPE:
    case OpType::QK_RMS_NORM_ROPE:
    case OpType::GROUP_RMS_NORM:
        if (!dispatch_cpu_normalization(
                node, inputs, output, thread_pool))
            reject();
        break;

    case OpType::SDPA:
    case OpType::SDPA_MLA:
        if (!dispatch_cpu_attention(node, inputs, output, thread_pool))
            reject();
        break;
    case OpType::GATED_DELTANET_PREFILL:
    case OpType::GATED_DELTANET_DECODE:
    case OpType::GATED_DELTANET_CONV_DECODE:
        dispatch_cpu_recurrent(node, inputs, output, thread_pool);
        break;
    case OpType::MOE:
        if (!dispatch_cpu_moe(node, inputs, output, thread_pool))
            reject();
        break;
    case OpType::HC_PRE:
        if (has_inputs(4) && output) {
            if (!kernel_hc_pre(
                *inputs[0], *inputs[1], *inputs[2], *inputs[3], *output,
                graph_params::get_i32(params, 0, 0),
                graph_params::get_i32(params, 1, 4),
                graph_params::get_i32(params, 2, 20),
                graph_params::get_f32(params, 0, 1e-6f),
                graph_params::get_f32(params, 1, 1e-6f), thread_pool))
                reject();
        } else {
            reject();
        }
        break;
    case OpType::HC_POST:
        if (has_inputs(3) && output) {
            if (!kernel_hc_post(
                *inputs[0], *inputs[1], *inputs[2], *output,
                graph_params::get_i32(params, 0, 0),
                graph_params::get_i32(params, 1, 4), thread_pool))
                reject();
        } else {
            reject();
        }
        break;
    case OpType::HC_HEAD:
        if (has_inputs(4) && output) {
            if (!kernel_hc_head(
                *inputs[0], *inputs[1], *inputs[2], *inputs[3], *output,
                graph_params::get_i32(params, 0, 0),
                graph_params::get_i32(params, 1, 4),
                graph_params::get_f32(params, 0, 1e-6f),
                graph_params::get_f32(params, 1, 1e-6f), thread_pool))
                reject();
        } else {
            reject();
        }
        break;
    case OpType::GR_REDUCE:
        if (has_inputs(2) && output) {
            if (!kernel_gr_reduce(
                    *inputs[0], *inputs[1], *output,
                    graph_params::get_i32(params, 0, 0),
                    graph_params::get_i32(params, 1, 4), thread_pool))
                reject();
        } else {
            reject();
        }
        break;
    case OpType::GR_INJECT:
        if (has_inputs(3) && output) {
            if (!kernel_gr_inject(
                    *inputs[0], *inputs[1], *inputs[2], *output,
                    graph_params::get_i32(params, 0, 0),
                    graph_params::get_i32(params, 1, 4), thread_pool))
                reject();
        } else {
            reject();
        }
        break;
    case OpType::PLE_LOOKUP:
        if (has_inputs(6) && output) {
            const uint64_t seed =
                static_cast<uint32_t>(graph_params::get_i32(params, 5, 1234)) |
                (static_cast<uint64_t>(static_cast<uint32_t>(
                    graph_params::get_i32(params, 6, 0))) << 32);
            if (!kernel_ple_lookup(
                    *inputs[0], const_cast<Tensor&>(*inputs[1]),
                    *inputs[2], *inputs[3], *inputs[4], *inputs[5], *output,
                    graph_params::get_i32(params, 0, 3),
                    graph_params::get_i32(params, 1, 8),
                    graph_params::get_i32(params, 2, 0),
                    graph_params::get_i32(params, 3, 0),
                    graph_params::get_i32(params, 4, 0), seed,
                    graph_params::get_i32(
                        params, 7,
                        static_cast<int>(inputs[0]->nelements())),
                    thread_pool))
                reject();
        } else {
            reject();
        }
        break;
    case OpType::PLE_GATE:
        if (has_inputs(3) && output) {
            if (!kernel_ple_gate(
                    *inputs[0], *inputs[1], *inputs[2], *output,
                    graph_params::get_i32(params, 0, 0),
                    graph_params::get_i32(params, 1, 4), thread_pool))
                reject();
        } else {
            reject();
        }
        break;
    case OpType::PLE_DILATED_CONV:
        if (has_inputs(3) && output) {
            if (!kernel_ple_dilated_conv(
                    *inputs[0], *inputs[1],
                    const_cast<Tensor&>(*inputs[2]), *output,
                    graph_params::get_i32(params, 0, 4),
                    graph_params::get_i32(params, 1, 3),
                    graph_params::get_i32(
                        params, 2, static_cast<int>(inputs[0]->shape[1])),
                    thread_pool))
                reject();
        } else {
            reject();
        }
        break;
    case OpType::DSV4_COMPRESSOR:
    case OpType::DSV4_INDEXER:
    case OpType::DSV4_SPARSE_ATTN:
    case OpType::DSV4_GROUPED_LINEAR:
        if (!dispatch_cpu_deepseek_v4(
                node, inputs, output, thread_pool))
            reject();
        break;
    case OpType::RWKV_TOKEN_SHIFT:
    case OpType::RWKV_MIX:
    case OpType::RWKV_L2_NORM:
    case OpType::RWKV_POST:
    case OpType::RWKV7:
    case OpType::SHORTCONV:
        dispatch_cpu_recurrent(node, inputs, output, thread_pool);
        break;
    case OpType::ROTARY_EMBED:
        if (inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] && output) {
            int rope_dim = graph_params::get_i32(params, 0, 64);
            bool interleave = graph_params::get_i32(params, 1, 1) != 0;
            kernel_rope(*inputs[0], *inputs[1], *inputs[2], rope_dim, interleave, *output);
        }
        break;

    case OpType::RMS_NORM:
    case OpType::ADD_RMS_NORM:
    case OpType::LAYER_NORM:
        if (!dispatch_cpu_normalization(
                node, inputs, output, thread_pool))
            reject();
        break;

    case OpType::ADD:
        kernel_elementwise(ElementwiseOp::ADD, inputs, output, thread_pool);
        break;
    case OpType::MUL:
        kernel_elementwise(ElementwiseOp::MUL, inputs, output, thread_pool);
        break;
    case OpType::SIGMOID_MUL:
        kernel_elementwise(ElementwiseOp::SIGMOID_MUL, inputs, output, thread_pool);
        break;
    case OpType::SILU:
        kernel_elementwise(ElementwiseOp::SILU, inputs, output, thread_pool);
        break;
    case OpType::GELU:
        kernel_elementwise(ElementwiseOp::GELU, inputs, output, thread_pool);
        break;
    case OpType::TANH:
        kernel_elementwise(ElementwiseOp::TANH, inputs, output, thread_pool);
        break;
    case OpType::SWIGLU:
        kernel_elementwise(ElementwiseOp::SWIGLU, inputs, output, thread_pool);
        break;
    case OpType::SIGMOID:
        kernel_elementwise(ElementwiseOp::SIGMOID, inputs, output, thread_pool);
        break;
    case OpType::SIGMOID_EXACT:
        kernel_elementwise(ElementwiseOp::SIGMOID_EXACT, inputs, output, thread_pool);
        break;
    case OpType::EXP:
        kernel_elementwise(ElementwiseOp::EXP, inputs, output, thread_pool);
        break;
    case OpType::EXP_EXACT:
        kernel_elementwise(ElementwiseOp::EXP_EXACT, inputs, output, thread_pool);
        break;
    case OpType::SOFTPLUS:
        kernel_elementwise(ElementwiseOp::SOFTPLUS, inputs, output, thread_pool);
        break;

    default:
        fprintf(stderr, "execute: unhandled op_type %u\n", (uint32_t)op);
        break;
    }
}
