#include "backends/cpu/deepseek_v4_ops.h"

#include "kernels/cpu/models/deepseek_v4_attention.h"

#include <algorithm>
#include <cstring>

bool dispatch_cpu_deepseek_v4(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* thread_pool) {
    const OpParams& params = node.params;
    bool success = true;
    auto reject = [&] { success = false; };
    auto has_inputs = [&](size_t count) {
        return inputs.size() >= count &&
               std::all_of(
                   inputs.begin(), inputs.begin() + count,
                   [](const Tensor* input) { return input != nullptr; });
    };
    switch (node.op_type) {
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
    default:
        return false;
    }
    return success;
}
