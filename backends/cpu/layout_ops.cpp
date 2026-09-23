#include "backends/cpu/layout_ops.h"

#include "graph/graph.h"

#include "kernels/cpu/layout.h"

#include <algorithm>

void dispatch_cpu_layout(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output) {
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

    default:
        break;
    }
}
