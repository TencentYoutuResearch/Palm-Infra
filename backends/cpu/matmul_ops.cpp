#include "backends/cpu/matmul_ops.h"

#include "graph/graph.h"

#include "kernels/cpu/matmul/matmul.h"

void dispatch_cpu_matmul(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* thread_pool) {
    const OpParams& params = node.params;
    switch (node.op_type) {
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

    default:
        break;
    }
}
