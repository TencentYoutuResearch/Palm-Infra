#include "graph/execute.h"
#include "kernels/matmul.h"
#include "kernels/norm.h"
#include "kernels/rope.h"
#include "kernels/attention.h"
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
    // Phase 1: simple approach — don't free anything.
    // Memory is released when the BufferPool is destroyed.
    // Proper liveness analysis with view-awareness will be added later.
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

    case OpType::SDPA:
    case OpType::SDPA_MLA: {
        // SDPA needs multi-output — create a local output vector from graph tensors
        // This is a simplified dispatch; the real engine will handle this properly
        std::vector<Tensor*> sdpa_outs = { output };
        kernel_sdpa(params, inputs, sdpa_outs);
        break;
    }
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

    case OpType::CONCAT:
        // zero-copy: concat produces a view? No — concat copies data.
        // Placeholder: skip for now (the model uses concat, needs kernel)
        break;

    case OpType::ADD:
        // element-wise add: output = a + b
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            const Tensor& a = *inputs[0];
            const Tensor& b = *inputs[1];
            int N = (int)output->nelements();
            // simple scalar add for now — works for broadcast or same-shape
            float* o = output->ptr<float>();
            const float* ap = a.ptr<float>();
            const float* bp = b.ptr<float>();
            if (a.nelements() == b.nelements()) {
                for (int i = 0; i < N; i++) o[i] = ap[i] + bp[i];
            } else if (b.nelements() == 1) {
                float bv = bp[0];
                for (int i = 0; i < N; i++) o[i] = ap[i] + bv;
            } else {
                for (int i = 0; i < N; i++) o[i] = ap[i] + bp[i];
            }
        }
        break;

    case OpType::MUL:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            const Tensor& a = *inputs[0];
            const Tensor& b = *inputs[1];
            int N = (int)output->nelements();
            float* o = output->ptr<float>();
            const float* ap = a.ptr<float>();
            const float* bp = b.ptr<float>();
            for (int i = 0; i < N; i++) o[i] = ap[i] * bp[i];
        }
        break;

    case OpType::SILU:
        if (inputs.size() >= 1 && inputs[0] && output) {
            int N = (int)output->nelements();
            const float* x = inputs[0]->ptr<float>();
            float* o = output->ptr<float>();
            for (int i = 0; i < N; i++) {
                float sx = 1.f / (1.f + std::exp(-x[i]));
                o[i] = x[i] * sx;
            }
        }
        break;

    case OpType::SLICE:
        // zero-copy: slice produces a view of the parent
        if (!inputs.empty() && inputs[0] && output) {
            int dim   = graph_params::get_i32(params, 0, 0);
            int offset= graph_params::get_i32(params, 1, 0);
            int size  = graph_params::get_i32(params, 2, (int)output->shape[dim]);
            // For dim=0: view at byte offset
            if (dim == 0 && offset >= 0) {
                *output = *inputs[0];
                size_t byte_off = (size_t)offset * inputs[0]->stride[0];
                output->data = static_cast<char*>(inputs[0]->data) + byte_off;
                output->shape[0] = size;
                output->compute_strides();
            } else {
                *output = *inputs[0];
                output->shape[dim] = size;
                output->compute_strides();
            }
        }
        break;

    case OpType::TILE:
        // Tile: replicate data. Placeholder.
        if (!inputs.empty() && inputs[0] && output) {
            *output = *inputs[0];
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

        // ---- always initialise output shape from node definition ----
        out.shape[0] = node.out_shape[0];
        out.shape[1] = node.out_shape[1];
        out.shape[2] = node.out_shape[2];
        out.shape[3] = node.out_shape[3];
        out.prec     = node.out_prec;
        out.compute_strides();

        // ---- allocate output if needed ----
        if (out.data == nullptr) {
            size_t nbytes = out.nbytes();

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
            if (rel_id < nodes.size()) {
                auto op = nodes[rel_id].op_type;
                // Never release INPUT, CONSTANT, SLICE, RESHAPE, or PERMUTE nodes.
                // View ops share parent data — freeing the parent leaves views dangling.
                if (op == OpType::INPUT || op == OpType::CONSTANT ||
                    op == OpType::SLICE || op == OpType::RESHAPE ||
                    op == OpType::PERMUTE || op == OpType::TILE ||
                    op == OpType::CONCAT)
                    continue;
            }
            if (t.data && t.mem_type == MemoryType::POOLED && t.nbytes() > 0) {
                pool->release(t.data, t.nbytes());
                t.data     = nullptr;
                t.mem_type = MemoryType::NONE;
            }
        }
    }
}
