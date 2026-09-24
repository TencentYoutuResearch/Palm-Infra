#include "backends/cpu/attention_ops.h"

#include "graph/graph.h"

#include "kernels/cpu/attention.h"

bool dispatch_cpu_attention(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* thread_pool) {
    const OpParams& params = node.params;
    const int cache_mode = graph_params::get_i32(params, 0, 2);
    if (cache_mode == 2 && inputs.size() > 5 && inputs[4] && inputs[5] &&
        inputs[4]->prec != inputs[5]->prec) {
        return false;
    }

    std::vector<Tensor*> outputs = {output};
    SdpaParams sdpa{
        cache_mode,
        graph_params::get_i32(params, 1, 1),
        graph_params::get_i32(params, 2, 16),
        graph_params::get_i32(params, 3, 16),
        graph_params::get_i32(params, 4, 192),
        graph_params::get_i32(params, 5, 128),
        graph_params::get_f32(params, 0, 0.f)};
    // Optional per-head attention sink arrives as a 7th input tensor of
    // [num_heads] FP32 values; absent means no sink.
    if (inputs.size() > 6 && inputs[6] && inputs[6]->data) {
        if (inputs[6]->prec != Precision::FP32 ||
            inputs[6]->nelements() < sdpa.num_heads) {
            return false;
        }
        sdpa.sink = inputs[6]->ptr<float>();
    }
    kernel_sdpa(sdpa, inputs, outputs, thread_pool);
    return true;
}
