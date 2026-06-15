#include "graph/execute.h"
#include "kernels/matmul.h"
#include "kernels/norm.h"
#include "kernels/rope.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// prepare_execution
// ---------------------------------------------------------------------------

void prepare_execution(ExecContext& ctx) {
    auto& nodes = ctx.graph->nodes;
    const size_t N = nodes.size();

    ctx.release_queue.resize(N);

    // 1. use_count[i] = how many later nodes reference node i's output
    std::vector<int> use_count(N, 0);
    for (const auto& node : nodes) {
        for (uint32_t inp : node.inputs) {
            if (inp < N) use_count[inp]++;
        }
    }

    // 2. last_use[i] = index of the last node that uses node i
    std::vector<int> last_use(N, -1);
    for (size_t j = 0; j < N; j++) {
        for (uint32_t inp : nodes[j].inputs) {
            if (inp < N) last_use[inp] = (int)j;
        }
    }

    // 3. release_queue[j] = nodes whose last_use == j
    for (size_t i = 0; i < N; i++) {
        if (last_use[i] >= 0) {
            ctx.release_queue[last_use[i]].push_back((uint32_t)i);
        }
    }

    // 4. ensure runtime tensors array is sized
    ctx.graph->runtime.tensors.resize(N);
}

// ---------------------------------------------------------------------------
// kernel dispatcher (stub — real kernels will be filled in as we build them)
// ---------------------------------------------------------------------------

static void dispatch_kernel(OpType op, const OpParams& params,
                            const std::vector<const Tensor*>& inputs,
                            Tensor* output) {
    switch (op) {
    case OpType::INPUT:
    case OpType::CONSTANT:
        // no-op — data is already in the tensor
        break;

    case OpType::RESHAPE:
        // zero-copy: adjust shape + stride
        if (!inputs.empty() && inputs[0] && output) {
            *output = *inputs[0];
            output->shape[0] = params.i32.size() >= 4 ? (int64_t)params.i32[0] : output->shape[0];
            output->shape[1] = params.i32.size() >= 4 ? (int64_t)params.i32[1] : 1;
            output->shape[2] = params.i32.size() >= 4 ? (int64_t)params.i32[2] : 1;
            output->shape[3] = params.i32.size() >= 4 ? (int64_t)params.i32[3] : 1;
        }
        break;

    case OpType::MATMUL:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            kernel_matmul_fp32(*inputs[0], *inputs[1], *output);
        }
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

    case OpType::PERMUTE:
        if (!inputs.empty() && inputs[0] && output && params.i32.size() >= 4) {
            *output = inputs[0]->permute(params.i32[0], params.i32[1],
                                         params.i32[2], params.i32[3]);
        }
        break;

    default:
        fprintf(stderr, "execute: unhandled op_type %u (kernel not yet implemented)\n",
                (uint32_t)op);
        break;
    }
}

// ---------------------------------------------------------------------------
// execute_graph
// ---------------------------------------------------------------------------

void execute_graph(ExecContext& ctx) {
    auto& nodes  = ctx.graph->nodes;
    auto& tensors = ctx.graph->runtime.tensors;
    auto* pool   = ctx.pool;

    for (size_t i = 0; i < nodes.size(); i++) {
        auto& node = nodes[i];

        // skip inputs / constants — data is pre-set by the caller
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) {
            continue;
        }

        auto& out = tensors[node.id];

        // ---- allocate output if needed ----
        if (out.data == nullptr && out.nbytes() > 0) {
            // compute required size from output shape + precision
            size_t nbytes = out.nbytes();
            if (nbytes == 0) {
                // tensor shape not set yet — infer from node definition
                out.shape[0] = node.out_shape[0];
                out.shape[1] = node.out_shape[1];
                out.shape[2] = node.out_shape[2];
                out.shape[3] = node.out_shape[3];
                out.prec     = node.out_prec;
                out.compute_strides();
                nbytes = out.nbytes();
            }

            if (nbytes > 0) {
                void* buf = pool->acquire(nbytes);
                if (!buf) {
                    fprintf(stderr, "execute: pool acquire failed for node %u (%zu bytes)\n",
                            node.id, nbytes);
                    return;
                }
                out.data     = buf;
                out.mem_type = MemoryType::POOLED;
            }
        }

        // ---- collect inputs ----
        std::vector<const Tensor*> inputs;
        inputs.reserve(node.inputs.size());
        for (uint32_t inp_id : node.inputs) {
            inputs.push_back(&tensors[inp_id]);
        }

        // ---- dispatch ----
        dispatch_kernel(node.op_type, node.params, inputs, &out);

        // ---- release completed tensors ----
        for (uint32_t rel_id : ctx.release_queue[i]) {
            auto& t = tensors[rel_id];
            if (t.data && t.mem_type == MemoryType::POOLED) {
                pool->release(t.data, t.nbytes());
                t.data     = nullptr;
                t.mem_type = MemoryType::NONE;
            }
        }
    }
}
