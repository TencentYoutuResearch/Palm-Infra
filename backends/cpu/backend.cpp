#include "backends/cpu/backend.h"
#include "backends/cpu/deepseek_v4_ops.h"
#include "graph/graph.h"
#include "kernels/cpu/attention.h"
#include "kernels/cpu/elementwise.h"
#include "kernels/cpu/recurrent/gdn.h"
#include "kernels/cpu/gated_residual.h"
#include "kernels/cpu/hyper_connection.h"
#include "kernels/cpu/layout.h"
#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/moe/moe.h"
#include "kernels/cpu/norm.h"
#include "kernels/cpu/ple.h"
#include "kernels/cpu/rope.h"
#include "kernels/cpu/models/rwkv.h"
#include "kernels/cpu/recurrent/shortconv.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

GdnParams resolve_gdn_params(const OpParams& params) {
    GdnParams result;
    result.num_heads = graph_params::get_i32(params, 0, 16);
    result.k_head_dim = graph_params::get_i32(params, 1, 128);
    result.v_head_dim = graph_params::get_i32(params, 2, 128);
    result.seq_len = graph_params::get_i32(params, 3, 4);
    result.flags = graph_params::get_i32(params, 4, 1);
    result.conv_kernel = graph_params::get_i32(params, 5, 4);
    result.real_tokens = graph_params::get_i32(params, 6, result.seq_len);
    result.num_v_heads = graph_params::get_i32(params, 7, result.num_heads);
    result.rms_eps = graph_params::get_f32(params, 0, 1e-6f);
    result.l2norm_eps = graph_params::get_f32(params, 1, 1e-6f);
    result.scale = graph_params::get_f32(params, 2, 0.f);
    return result;
}

}  // namespace

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
        // no-op — data is already in the tensor
        break;

    case OpType::RESHAPE:
    case OpType::CONCAT:
    case OpType::SLICE:
    case OpType::TILE:
    case OpType::PERMUTE:
    case OpType::CONTIGUOUS: {
        if (!output || inputs.empty() || (op != OpType::CONCAT && !inputs[0]))
            break;
        LayoutParams layout;
        switch (op) {
        case OpType::RESHAPE:
            layout.op = LayoutOp::RESHAPE;
            for (int d = 0; d < 4; ++d) {
                layout.shape[d] = output->shape[d];
                if (params.i32.size() >= 4 && !node.dim_expr[d].is_dynamic())
                    layout.shape[d] = params.i32[d];
            }
            break;
        case OpType::CONCAT:
            layout.op = LayoutOp::CONCAT;
            layout.axis = graph_params::get_i32(params, 0, 0);
            break;
        case OpType::SLICE:
            layout.op = LayoutOp::SLICE;
            layout.axis = graph_params::get_i32(params, 0, 0);
            layout.offset = graph_params::get_i32(params, 1, 0);
            layout.size = graph_params::get_i32(params, 2,
                                               (int)output->shape[layout.axis]);
            break;
        case OpType::TILE:
            layout.op = LayoutOp::TILE;
            for (int d = 0; d < 4; ++d)
                layout.repeats[d] = graph_params::get_i32(params, d, 1);
            break;
        case OpType::PERMUTE:
            if (params.i32.size() < 4) return;
            layout.op = LayoutOp::PERMUTE;
            std::copy_n(params.i32.begin(), 4, layout.order.begin());
            break;
        case OpType::CONTIGUOUS:
            layout.op = LayoutOp::CONTIGUOUS;
            break;
        default:
            break;
        }
        kernel_layout(layout, inputs, output);
        break;
    }

    case OpType::MATMUL:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            // Fused activation: params.i32[0]=activation, [1]=act_n_begin, [2]=act_n_len.
            // Default: NONE, 0, -1 (whole output).
            Activation act = (Activation)graph_params::get_i32(params, 0, 0);
            int act_n_begin = graph_params::get_i32(params, 1, 0);
            int act_n_len = graph_params::get_i32(params, 2, -1);
            kernel_matmul_fp32(*inputs[0], *inputs[1], *output, thread_pool,
                                act, act_n_begin, act_n_len);
        }
        break;

    case OpType::MATMUL_BATCH:
        if (output)
            kernel_matmul_batch(inputs, *output, thread_pool);
        break;

    case OpType::GEMV_SPARSE_A:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            kernel_gemv_sparse_a(*inputs[0], *inputs[1], *output, thread_pool);
        }
        break;

    case OpType::RMS_NORM_ROPE:
        if (inputs.size() >= 4 && output) {
            const int dim = (int)output->shape[0];
            const int seq = (int)output->shape[1];
            const int heads = (int)output->shape[2];
            std::vector<float> normalized(
                (size_t)dim * (size_t)seq * (size_t)heads);
            Tensor tmp = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL,
                dim, seq, heads, 1, normalized.data());
            kernel_rms_norm(
                *inputs[0], *inputs[1],
                graph_params::get_f32(params, 0, 1e-6f), tmp);
            kernel_rope(
                tmp, *inputs[2], *inputs[3],
                graph_params::get_i32(params, 0, dim),
                graph_params::get_i32(params, 1, 1) != 0, *output);
        }
        break;

    case OpType::QK_RMS_NORM_ROPE:
        if (inputs.size() >= 6 && output) {
            const int dim = (int)output->shape[0];
            const int seq = (int)output->shape[1];
            const int total_heads = (int)output->shape[2];
            const int query_heads =
                graph_params::get_i32(params, 2, total_heads);
            const int key_heads = total_heads - query_heads;
            if (query_heads <= 0 || key_heads <= 0) {
                reject();
                break;
            }

            auto run = [&](const Tensor& x, const Tensor& weight,
                           int heads, Tensor& out) {
                std::vector<float> normalized(
                    (size_t)dim * (size_t)seq * (size_t)heads);
                Tensor tmp = Tensor::create(
                    Precision::FP32, MemoryType::EXTERNAL,
                    dim, seq, heads, 1, normalized.data());
                kernel_rms_norm(
                    x, weight,
                    graph_params::get_f32(params, 0, 1e-6f), tmp);
                kernel_rope(
                    tmp, *inputs[4], *inputs[5],
                    graph_params::get_i32(params, 0, dim),
                    graph_params::get_i32(params, 1, 1) != 0, out);
            };

            Tensor query_out = *output;
            query_out.shape[2] = query_heads;
            Tensor key_out = *output;
            key_out.shape[2] = key_heads;
            const size_t key_offset =
                (size_t)query_heads * output->stride[2];
            key_out.data =
                static_cast<char*>(output->data) + key_offset;
            key_out.device.offset = output->device.offset + key_offset;
            run(*inputs[0], *inputs[2], query_heads, query_out);
            run(*inputs[1], *inputs[3], key_heads, key_out);
        }
        break;
    case OpType::GROUP_RMS_NORM:
        if (has_inputs(2) && output) {
            if (!kernel_group_rms_norm(
                    *inputs[0], *inputs[1], *output,
                    graph_params::get_i32(params, 0, 0),
                    graph_params::get_f32(params, 0, 1e-6f), thread_pool))
                reject();
        } else {
            reject();
        }
        break;

    case OpType::SDPA:
    case OpType::SDPA_MLA: {
        const int cache_mode = graph_params::get_i32(params, 0, 2);
        if (cache_mode == 2 && inputs.size() > 5 && inputs[4] && inputs[5] &&
            inputs[4]->prec != inputs[5]->prec) {
            reject();
            break;
        }
        std::vector<Tensor*> sdpa_outs = { output };
        const SdpaParams sdpa{
            cache_mode,
            graph_params::get_i32(params, 1, 1),
            graph_params::get_i32(params, 2, 16),
            graph_params::get_i32(params, 3, 16),
            graph_params::get_i32(params, 4, 192),
            graph_params::get_i32(params, 5, 128),
            graph_params::get_f32(params, 0, 0.f)};
        kernel_sdpa(sdpa, inputs, sdpa_outs, thread_pool);
        break;
    }
    case OpType::GATED_DELTANET_PREFILL: {
        std::vector<Tensor*> gdn_outs = { output };
        kernel_gdn_prefill(resolve_gdn_params(params), inputs, gdn_outs, thread_pool);
        break;
    }
    case OpType::GATED_DELTANET_DECODE: {
        std::vector<Tensor*> gdn_outs = { output };
        kernel_gdn_decode(resolve_gdn_params(params), inputs, gdn_outs, thread_pool);
        break;
    }
    case OpType::GATED_DELTANET_CONV_DECODE: {
        std::vector<Tensor*> gdn_outs = { output };
        kernel_gdn_conv_decode(resolve_gdn_params(params), inputs, gdn_outs, thread_pool);
        break;
    }
    case OpType::MOE: {
        int hidden_size = graph_params::get_i32(params, 0, output ? (int)output->shape[0] : 0);
        int num_experts = graph_params::get_i32(params, 1, 0);
        int top_k = graph_params::get_i32(params, 2, 0);
        int intermediate_size = graph_params::get_i32(params, 3, 0);
        int shared_intermediate_size = graph_params::get_i32(params, 4, intermediate_size);
        int router_score_func = graph_params::get_i32(params, 5, 0);
        bool norm_topk_prob = graph_params::get_i32(params, 6, 1) != 0;
        bool has_shared_expert = graph_params::get_i32(params, 7, 1) != 0;
        int n_group = graph_params::get_i32(params, 8, 1);
        int topk_group = graph_params::get_i32(params, 9, 1);
        bool shared_expert_has_gate =
            graph_params::get_i32(params, 10, 1) != 0;
        int router_bias_input =
            graph_params::get_i32(
                params, 11, has_shared_expert ? 8 : -1);
        int token_ids_input = graph_params::get_i32(params, 12, -1);
        int hash_table_input = graph_params::get_i32(params, 13, -1);
        float routed_scaling_factor = graph_params::get_f32(params, 0, 1.0f);
        float swiglu_limit = graph_params::get_f32(params, 1, 0.0f);
        if (output) {
            if (!kernel_qwen3_moe(
                    inputs, *output, thread_pool,
                    hidden_size, num_experts, top_k,
                    intermediate_size, shared_intermediate_size,
                    router_score_func, norm_topk_prob,
                    has_shared_expert, n_group, topk_group,
                    routed_scaling_factor,
                    shared_expert_has_gate, router_bias_input,
                    token_ids_input, hash_table_input,
                    swiglu_limit)) {
                reject();
            }
        } else {
            reject();
        }
        break;
    }
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
    case OpType::RWKV_TOKEN_SHIFT: {
        RwkvTokenShiftParams config;
        config.hidden = graph_params::get_i32(params, 0, 0);
        config.seq_len = graph_params::get_i32(params, 1, 1);
        config.real_tokens = graph_params::get_i32(params, 2, config.seq_len);
        kernel_rwkv_token_shift(config, inputs, *output);
        break;
    }
    case OpType::RWKV_MIX:
        kernel_rwkv_mix(inputs, *output);
        break;
    case OpType::RWKV_L2_NORM:
        kernel_rwkv_l2_norm(
            RwkvL2NormParams{graph_params::get_i32(params, 0, 0),
                             graph_params::get_i32(params, 1, 0),
                             graph_params::get_f32(params, 0, 1e-12f)},
            inputs, *output);
        break;
    case OpType::RWKV_POST:
        kernel_rwkv_post(
            RwkvPostParams{graph_params::get_i32(params, 0, 0),
                           graph_params::get_i32(params, 1, 0),
                           graph_params::get_f32(params, 0, 64e-5f)},
            inputs, *output, thread_pool);
        break;
    case OpType::RWKV7: {
        Rwkv7Params config;
        config.num_heads = graph_params::get_i32(params, 0, 0);
        config.head_dim = graph_params::get_i32(params, 1, 0);
        config.seq_len = graph_params::get_i32(params, 2, 1);
        config.real_tokens = graph_params::get_i32(params, 3, config.seq_len);
        kernel_rwkv7(config, inputs, *output, thread_pool);
        break;
    }
    case OpType::SHORTCONV:
        if (output)
            kernel_shortconv(
                ShortConvParams{graph_params::get_i32(params, 0, 4),
                                graph_params::get_i32(params, 1, 0)},
                inputs, *output, thread_pool);
        break;
    case OpType::ROTARY_EMBED:
        if (inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] && output) {
            int rope_dim = graph_params::get_i32(params, 0, 64);
            bool interleave = graph_params::get_i32(params, 1, 1) != 0;
            kernel_rope(*inputs[0], *inputs[1], *inputs[2], rope_dim, interleave, *output);
        }
        break;

    case OpType::RMS_NORM:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            float eps = graph_params::get_f32(params, 0, 1e-6f);
            kernel_rms_norm(*inputs[0], *inputs[1], eps, *output);
        }
        break;

    case OpType::ADD_RMS_NORM:
        if (inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] &&
            output) {
            float eps = graph_params::get_f32(params, 0, 1e-6f);
            Tensor& residual = *const_cast<Tensor*>(inputs[0]);
            kernel_add_rms_norm(
                residual, *inputs[1], *inputs[2], eps, *output,
                thread_pool);
        }
        break;

    case OpType::LAYER_NORM:
        if (inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] && output) {
            float eps = graph_params::get_f32(params, 0, 1e-5f);
            kernel_layer_norm(*inputs[0], *inputs[1], *inputs[2], eps, *output,
                              thread_pool);
        }
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
