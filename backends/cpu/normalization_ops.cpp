#include "backends/cpu/normalization_ops.h"

#include "graph/graph.h"

#include "kernels/cpu/gated_residual.h"
#include "kernels/cpu/norm.h"
#include "kernels/cpu/rope.h"

#include <algorithm>
#include <vector>

bool dispatch_cpu_normalization(
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


    default:
        return false;
    }
    return success;
}
