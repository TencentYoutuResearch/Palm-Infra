#include "engine/backend.h"
#include "kernels/attention.h"
#include "kernels/deepseek_v4_attention.h"
#include "kernels/elementwise.h"
#include "kernels/gdn.h"
#include "kernels/gated_residual.h"
#include "kernels/hyper_connection.h"
#include "kernels/layout.h"
#include "kernels/matmul.h"
#include "kernels/moe.h"
#include "kernels/norm.h"
#include "kernels/ple.h"
#include "kernels/rope.h"
#include "kernels/rwkv.h"
#include "kernels/shortconv.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

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
            key_out.device_offset = output->device_offset + key_offset;
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
    case OpType::GATED_DELTANET_CONV_DECODE: {
        std::vector<Tensor*> gdn_outs = { output };
        kernel_gdn_conv_decode(params, inputs, gdn_outs, thread_pool);
        break;
    }
    case OpType::GATED_DELTANET_CONV_VERIFY: {
        const int seq_len = graph_params::get_i32(params, 3, 0);
        const int confirmed_prefix = graph_params::get_i32(params, 8, 0);
        const bool valid_inputs = inputs.size() == 12 &&
            std::all_of(inputs.begin(), inputs.end(), [](const Tensor* input) {
                return input != nullptr && input->data != nullptr;
            });
        const bool valid_state = valid_inputs &&
            inputs[7]->prec == Precision::FP32 &&
            inputs[9]->prec == Precision::FP32 &&
            inputs[10]->prec == Precision::FP32 &&
            inputs[11]->prec == Precision::FP32 &&
            inputs[7]->nbytes() == inputs[10]->nbytes() &&
            inputs[9]->nbytes() == inputs[11]->nbytes();
        if (!valid_inputs || !output || !valid_state || seq_len <= 1 ||
            confirmed_prefix < 1 || confirmed_prefix >= seq_len) {
            std::fprintf(
                stderr,
                "CPUBackend: invalid GATED_DELTANET_CONV_VERIFY contract "
                "(inputs=%zu, seq_len=%d, confirmed_prefix=%d)\n",
                inputs.size(), seq_len, confirmed_prefix);
            reject();
            break;
        }
        std::vector<Tensor*> gdn_outs = { output };
        kernel_gdn_conv_verify(params, inputs, gdn_outs, thread_pool);
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
    case OpType::DSV4_COMPRESSOR: {
        if (!has_inputs(10) || !output) {
            reject();
            break;
        }
        Dsv4CompressorConfig config;
        config.hidden_size = graph_params::get_i32(params, 0, 4096);
        config.head_dim = graph_params::get_i32(params, 1, 512);
        config.ratio = graph_params::get_i32(params, 2, 4);
        config.overlap = graph_params::get_i32(params, 3, 1) != 0;
        config.rotate = graph_params::get_i32(params, 4, 0) != 0;
        config.rope.rope_dim = graph_params::get_i32(params, 5, 64);
        config.rope.original_context =
            graph_params::get_i32(params, 6, 65536);
        config.norm_eps = graph_params::get_f32(params, 0, 1e-6f);
        config.rope.theta = graph_params::get_f32(params, 1, 160000.0f);
        config.rope.factor = graph_params::get_f32(params, 2, 16.0f);
        config.rope.beta_fast = graph_params::get_f32(params, 3, 32.0f);
        config.rope.beta_slow = graph_params::get_f32(params, 4, 1.0f);
        const int start_pos = inputs[8] && inputs[8]->data
            ? inputs[8]->ptr<int32_t>()[0] : 0;
        const int n_tokens = inputs[9] && inputs[9]->data
            ? inputs[9]->ptr<int32_t>()[0]
            : static_cast<int>(inputs[0]->shape[1]);
        Tensor hidden = *inputs[0];
        hidden.shape[1] = std::min<int64_t>(
            hidden.shape[1], std::max(n_tokens, 0));
        hidden.compute_strides();
        const int emitted = kernel_dsv4_compressor(
            hidden, *inputs[1], *inputs[2], *inputs[3], *inputs[4],
            *const_cast<Tensor*>(inputs[5]),
            *const_cast<Tensor*>(inputs[6]),
            *const_cast<Tensor*>(inputs[7]), start_pos, config,
            thread_pool);
        if (emitted < 0) {
            reject();
            break;
        }
        output->ptr<float>()[0] = static_cast<float>(emitted);
        break;
    }
    case OpType::DSV4_INDEXER: {
        if (!has_inputs(13) || !output) {
            reject();
            break;
        }
        Dsv4IndexerConfig config;
        config.hidden_size = graph_params::get_i32(params, 0, 4096);
        config.q_lora_rank = graph_params::get_i32(params, 1, 1024);
        config.num_heads = graph_params::get_i32(params, 2, 64);
        config.head_dim = graph_params::get_i32(params, 3, 128);
        config.top_k = graph_params::get_i32(params, 4, 512);
        config.compressor.hidden_size = config.hidden_size;
        config.compressor.head_dim = config.head_dim;
        config.compressor.ratio = graph_params::get_i32(params, 5, 4);
        config.compressor.overlap =
            graph_params::get_i32(params, 6, 1) != 0;
        config.compressor.rotate =
            graph_params::get_i32(params, 7, 1) != 0;
        config.compressor.rope.rope_dim =
            graph_params::get_i32(params, 8, 64);
        config.compressor.rope.original_context =
            graph_params::get_i32(params, 9, 65536);
        config.compressor.norm_eps =
            graph_params::get_f32(params, 0, 1e-6f);
        config.compressor.rope.theta =
            graph_params::get_f32(params, 1, 160000.0f);
        config.compressor.rope.factor =
            graph_params::get_f32(params, 2, 16.0f);
        config.compressor.rope.beta_fast =
            graph_params::get_f32(params, 3, 32.0f);
        config.compressor.rope.beta_slow =
            graph_params::get_f32(params, 4, 1.0f);
        const int start_pos = inputs[11] && inputs[11]->data
            ? inputs[11]->ptr<int32_t>()[0] : 0;
        const int n_tokens = inputs[12] && inputs[12]->data
            ? inputs[12]->ptr<int32_t>()[0]
            : static_cast<int>(inputs[0]->shape[1]);
        const int sequence = std::min<int64_t>(
            inputs[0]->shape[1], std::max(n_tokens, 0));
        Tensor hidden = *inputs[0];
        Tensor q_lora = *inputs[1];
        Tensor indices = *output;
        hidden.shape[1] = sequence;
        q_lora.shape[1] = sequence;
        indices.shape[1] = sequence;
        hidden.compute_strides();
        q_lora.compute_strides();
        indices.compute_strides();
        std::fill(
            output->ptr<int32_t>(),
            output->ptr<int32_t>() + output->nelements(),
            static_cast<int32_t>(-1));
        if (!kernel_dsv4_indexer(
            hidden, q_lora, *inputs[2], *inputs[3], *inputs[4],
            *inputs[5], *inputs[6], *inputs[7],
            *const_cast<Tensor*>(inputs[8]),
            *const_cast<Tensor*>(inputs[9]),
            *const_cast<Tensor*>(inputs[10]), indices, start_pos,
            config, thread_pool)) {
            reject();
        }
        break;
    }
    case OpType::DSV4_SPARSE_ATTN: {
        // Inputs 4 and 5 are start_pos and n_real_tokens.
        if (!has_inputs(6) || !output) {
            reject();
            break;
        }
        Dsv4SparseAttentionConfig config;
        config.num_heads = graph_params::get_i32(params, 0, 64);
        config.head_dim = graph_params::get_i32(params, 1, 512);
        config.window_size = graph_params::get_i32(params, 2, 128);
        config.compress_ratio = graph_params::get_i32(params, 3, 0);
        config.compressed_top_k = graph_params::get_i32(params, 4, 512);
        config.rope.rope_dim = graph_params::get_i32(params, 5, 64);
        config.rope.original_context =
            graph_params::get_i32(params, 6, 65536);
        const int cache_input = graph_params::get_i32(params, 7, -1);
        const int indices_input = graph_params::get_i32(params, 8, -1);
        config.softmax_scale = graph_params::get_f32(params, 0, 0.0f);
        config.query_norm_eps = graph_params::get_f32(params, 1, 1e-6f);
        config.rope.theta = graph_params::get_f32(params, 2, 160000.0f);
        config.rope.factor = graph_params::get_f32(params, 3, 16.0f);
        config.rope.beta_fast = graph_params::get_f32(params, 4, 32.0f);
        config.rope.beta_slow = graph_params::get_f32(params, 5, 1.0f);
        const Tensor* compressed_cache =
            cache_input >= 0 &&
                    cache_input < static_cast<int>(inputs.size())
                ? inputs[cache_input] : nullptr;
        const Tensor* compressed_indices =
            indices_input >= 0 &&
                    indices_input < static_cast<int>(inputs.size())
                ? inputs[indices_input] : nullptr;
        const int start_pos = inputs[4] && inputs[4]->data
            ? inputs[4]->ptr<int32_t>()[0] : 0;
        const int n_tokens = inputs[5] && inputs[5]->data
            ? inputs[5]->ptr<int32_t>()[0]
            : static_cast<int>(inputs[0]->shape[1]);
        const int sequence = std::min<int64_t>(
            inputs[0]->shape[1], std::max(n_tokens, 0));
        Tensor query = *inputs[0];
        Tensor current_kv = *inputs[1];
        Tensor attention_output = *output;
        query.shape[1] = sequence;
        current_kv.shape[1] = sequence;
        attention_output.shape[1] = sequence;
        query.compute_strides();
        current_kv.compute_strides();
        attention_output.compute_strides();
        if (output->data)
            std::memset(output->data, 0, output->nbytes());
        if (!kernel_dsv4_sparse_attention(
            query, current_kv, *inputs[2],
            *const_cast<Tensor*>(inputs[3]), compressed_cache,
            compressed_indices, start_pos, attention_output, config,
            thread_pool)) {
            reject();
        }
        break;
    }
    case OpType::DSV4_GROUPED_LINEAR:
        if (has_inputs(2) && output) {
            if (!kernel_dsv4_grouped_linear(
                *inputs[0], *inputs[1], *output,
                graph_params::get_i32(params, 0, 8), thread_pool)) {
                reject();
            }
        } else {
            reject();
        }
        break;
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
    case OpType::MUL:
    case OpType::SIGMOID_MUL:
    case OpType::SILU:
    case OpType::GELU:
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
