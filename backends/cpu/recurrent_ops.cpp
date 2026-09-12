#include "backends/cpu/recurrent_ops.h"

#include "kernels/cpu/models/rwkv.h"
#include "kernels/cpu/recurrent/gdn.h"
#include "kernels/cpu/recurrent/shortconv.h"

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

void dispatch_cpu_recurrent(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* thread_pool) {
    const OpParams& params = node.params;
    switch (node.op_type) {
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
    default:
        break;
    }
}
