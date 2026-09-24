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
    // Optional inputs are located by the indices the builder recorded, since
    // an absent one simply shortens the list. -1 means "not present".
    auto optional_input = [&](int index) -> const Tensor* {
        if (index < 0 || static_cast<size_t>(index) >= inputs.size())
            return nullptr;
        const Tensor* tensor = inputs[static_cast<size_t>(index)];
        return (tensor && tensor->data) ? tensor : nullptr;
    };

    const Tensor* sink = optional_input(graph_params::get_i32(params, 6, -1));
    if (sink) {
        if (sink->prec != Precision::FP32 ||
            sink->nelements() < sdpa.num_heads) {
            return false;
        }
        sdpa.sink = sink->ptr<float>();
    }

    const Tensor* selection =
        optional_input(graph_params::get_i32(params, 7, -1));
    if (selection) {
        // Hy4 Gated DSA: attend only to the indexed keys. kernel_sdpa_sparse
        // shares the flash-attention kernels and falls back to the dense path
        // when the selection carries no sparsity.
        if (selection->prec != Precision::INT32) {
            return false;
        }
        kernel_sdpa_sparse(sdpa, inputs, outputs, thread_pool);
        return true;
    }

    kernel_sdpa(sdpa, inputs, outputs, thread_pool);
    return true;
}
