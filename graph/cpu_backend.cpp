#include "engine/backend.h"
#include "kernels/attention.h"
#include "kernels/elementwise.h"
#include "kernels/gdn.h"
#include "kernels/layout.h"
#include "kernels/matmul.h"
#include "kernels/moe.h"
#include "kernels/norm.h"
#include "kernels/rope.h"
#include "kernels/rwkv.h"
#include "kernels/shortconv.h"

#include <cstdio>

// ---------------------------------------------------------------------------
// CPUBackend::dispatch — kernel dispatcher for CPU (ARM NEON) backend.
//
// Routes OpType to the appropriate kernel. This is the only dispatch
// entry point for CPU; future NPU backend will have its own dispatch().
// ---------------------------------------------------------------------------

void CPUBackend::dispatch(const GraphNode& node,
                          const std::vector<const Tensor*>& inputs,
                          Tensor* output, ThreadPool* thread_pool) {
    const OpType op = node.op_type;
    const OpParams& params = node.params;
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
    case OpType::CONTIGUOUS:
        kernel_layout(node, inputs, output);
        break;

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

    case OpType::GEMV_SPARSE_A:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            kernel_gemv_sparse_a(*inputs[0], *inputs[1], *output, thread_pool);
        }
        break;

    case OpType::SDPA:
    case OpType::SDPA_MLA: {
        std::vector<Tensor*> sdpa_outs = { output };
        kernel_sdpa(params, inputs, sdpa_outs, thread_pool);
        break;
    }
    case OpType::GATED_DELTANET_PREFILL: {
        std::vector<Tensor*> gdn_outs = { output };
        kernel_gdn_prefill(params, inputs, gdn_outs, thread_pool);
        break;
    }
    case OpType::GATED_DELTANET_DECODE: {
        std::vector<Tensor*> gdn_outs = { output };
        kernel_gdn_decode(params, inputs, gdn_outs, thread_pool);
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
        float routed_scaling_factor = graph_params::get_f32(params, 0, 1.0f);
        if (output) {
            kernel_qwen3_moe(inputs, *output, thread_pool,
                             hidden_size, num_experts, top_k,
                             intermediate_size, shared_intermediate_size,
                             router_score_func, norm_topk_prob,
                             has_shared_expert, n_group, topk_group,
                             routed_scaling_factor);
        }
        break;
    }
    case OpType::RWKV_TOKEN_SHIFT:
        kernel_rwkv_token_shift(params, inputs, *output);
        break;
    case OpType::RWKV_MIX:
        kernel_rwkv_mix(params, inputs, *output);
        break;
    case OpType::RWKV_L2_NORM:
        kernel_rwkv_l2_norm(params, inputs, *output);
        break;
    case OpType::RWKV_POST:
        kernel_rwkv_post(params, inputs, *output, thread_pool);
        break;
    case OpType::RWKV7:
        kernel_rwkv7(params, inputs, *output, thread_pool);
        break;
    case OpType::SHORTCONV:
        if (output)
            kernel_shortconv(params, inputs, *output, thread_pool);
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

    case OpType::LAYER_NORM:
        if (inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] && output) {
            float eps = graph_params::get_f32(params, 0, 1e-5f);
            kernel_layer_norm(*inputs[0], *inputs[1], *inputs[2], eps, *output,
                              thread_pool);
        }
        break;


    case OpType::ADD:
    case OpType::MUL:
    case OpType::SILU:
    case OpType::TANH:
    case OpType::SWIGLU:
    case OpType::SIGMOID:
    case OpType::SIGMOID_EXACT:
    case OpType::EXP:
    case OpType::EXP_EXACT:
    case OpType::SOFTPLUS:
        kernel_elementwise(op, inputs, output, thread_pool);
        break;

    default:
        fprintf(stderr, "execute: unhandled op_type %u\n", (uint32_t)op);
        break;
    }
}
