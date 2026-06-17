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
// kernel dispatcher
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
        // Reshape preserves logical element order. For contiguous inputs this is
        // a zero-copy metadata change; for non-contiguous views we materialize a
        // contiguous output.
        if (!inputs.empty() && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            int64_t new_shape[4] = { output->shape[0], output->shape[1], output->shape[2], output->shape[3] };
            if (params.i32.size() >= 4) {
                new_shape[0] = params.i32[0];
                new_shape[1] = params.i32[1];
                new_shape[2] = params.i32[2];
                new_shape[3] = params.i32[3];
            }

            if (src.is_contiguous()) {
                *output = src;
                output->shape[0] = new_shape[0];
                output->shape[1] = new_shape[1];
                output->shape[2] = new_shape[2];
                output->shape[3] = new_shape[3];
                output->compute_strides();
            } else {
                output->shape[0] = new_shape[0];
                output->shape[1] = new_shape[1];
                output->shape[2] = new_shape[2];
                output->shape[3] = new_shape[3];
                output->compute_strides();

                size_t es = src.element_size();
                char* dst = static_cast<char*>(output->data);
                int64_t flat = 0;
                for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
                    for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
                        for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                            for (int64_t i0 = 0; i0 < src.shape[0]; i0++) {
                                const char* src_ptr = static_cast<const char*>(src.data)
                                    + i0 * src.stride[0]
                                    + i1 * src.stride[1]
                                    + i2 * src.stride[2]
                                    + i3 * src.stride[3];
                                std::memcpy(dst + flat * es, src_ptr, es);
                                flat++;
                            }
                        }
                    }
                }
            }
        }
        break;

    case OpType::MATMUL:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            kernel_matmul_fp32(*inputs[0], *inputs[1], *output);
        }
        break;

    case OpType::SDPA:
    case OpType::SDPA_MLA: {
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
        // Concatenate inputs along specified dimension. Materializes output and
        // respects input strides, so view inputs remain correct.
        if (!inputs.empty() && output) {
            int dim = graph_params::get_i32(params, 0, 0);
            size_t es = output->element_size();
            int64_t dim_offset = 0;
            char* dst_base = static_cast<char*>(output->data);
            for (size_t i = 0; i < inputs.size(); i++) {
                if (!inputs[i] || !inputs[i]->data) continue;
                const Tensor& src = *inputs[i];
                const char* src_base = static_cast<const char*>(src.data);
                for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
                    for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
                        for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                            for (int64_t i0 = 0; i0 < src.shape[0]; i0++) {
                                int64_t o0 = i0;
                                int64_t o1 = i1;
                                int64_t o2 = i2;
                                int64_t o3 = i3;
                                if (dim == 0) o0 += dim_offset;
                                else if (dim == 1) o1 += dim_offset;
                                else if (dim == 2) o2 += dim_offset;
                                else if (dim == 3) o3 += dim_offset;

                                const char* src_ptr = src_base
                                    + i0 * src.stride[0]
                                    + i1 * src.stride[1]
                                    + i2 * src.stride[2]
                                    + i3 * src.stride[3];
                                char* dst_ptr = dst_base
                                    + o0 * output->stride[0]
                                    + o1 * output->stride[1]
                                    + o2 * output->stride[2]
                                    + o3 * output->stride[3];
                                std::memcpy(dst_ptr, src_ptr, es);
                            }
                        }
                    }
                }
                dim_offset += src.shape[dim];
            }
        }
        break;

    case OpType::ADD:
        if (inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
            const Tensor& a = *inputs[0];
            const Tensor& b = *inputs[1];
            int N = (int)a.nelements();
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
            int N = (int)a.nelements();
            float* o = output->ptr<float>();
            const float* ap = a.ptr<float>();
            const float* bp = inputs[1]->ptr<float>();
            for (int i = 0; i < N; i++) o[i] = ap[i] * bp[i];
        }
        break;

    case OpType::SILU:
        if (inputs.size() >= 1 && inputs[0] && output) {
            int N = (int)inputs[0]->nelements();
            const float* x = inputs[0]->ptr<float>();
            float* o = output->ptr<float>();
            for (int i = 0; i < N; i++) {
                float sx = 1.f / (1.f + std::exp(-x[i]));
                o[i] = x[i] * sx;
            }
        }
        break;

    case OpType::SLICE:
        // zero-copy: slice produces a view of the parent and must preserve the
        // parent's stride layout.
        if (!inputs.empty() && inputs[0] && output) {
            int dim    = graph_params::get_i32(params, 0, 0);
            int offset = graph_params::get_i32(params, 1, 0);
            int size   = graph_params::get_i32(params, 2, (int)output->shape[dim]);
            *output = *inputs[0];
            if (offset >= 0) {
                size_t byte_off = (size_t)offset * inputs[0]->stride[dim];
                output->data = static_cast<char*>(inputs[0]->data) + byte_off;
            }
            output->shape[dim] = size;
        }
        break;

    case OpType::TILE:
        // Tile: replicate input along each dimension by the given multipliers.
        if (!inputs.empty() && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            size_t es = src.element_size();
            int reps[4] = {1, 1, 1, 1};
            for (int d = 0; d < 4 && d < (int)params.i32.size(); d++) {
                reps[d] = params.i32[d];
            }
            int64_t total = output->nelements();
            char* dst = static_cast<char*>(output->data);
            const char* s = static_cast<const char*>(src.data);
            for (int64_t o = 0; o < total; o++) {
                int64_t idx = o;
                int64_t src_off = 0;
                for (int d = 0; d < 4; d++) {
                    int64_t dim_idx = idx % output->shape[d];
                    idx /= output->shape[d];
                    int64_t src_idx = dim_idx % src.shape[d];
                    src_off += src_idx * (int64_t)(src.stride[d] / es);
                }
                std::memcpy(dst + o * es, s + src_off * es, es);
            }
        }
        break;

    case OpType::PERMUTE:
        if (!inputs.empty() && inputs[0] && output && params.i32.size() >= 4) {
            *output = inputs[0]->permute(params.i32[0], params.i32[1],
                                         params.i32[2], params.i32[3]);
        }
        break;

    default:
        fprintf(stderr, "execute: unhandled op_type %u\n", (uint32_t)op);
        break;
    }
}

// ---------------------------------------------------------------------------
// execute_graph — pure executor, no shape inference
//
// With the multi-graph architecture, all shapes are static at graph-build
// time.  The prefill graph is built with seq_len=N, the decode graph with
// seq_len=1.  The only runtime-dynamic element is the KV cache, which uses
// embedded CacheMetadata for past_len tracking.
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

        // initialise output shape from node definition (static!)
        out.shape[0] = node.out_shape[0];
        out.shape[1] = node.out_shape[1];
        out.shape[2] = node.out_shape[2];
        out.shape[3] = node.out_shape[3];
        out.prec     = node.out_prec;
        out.compute_strides();

        // allocate output if needed
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

        // collect inputs
        std::vector<const Tensor*> inputs;
        inputs.reserve(node.inputs.size());
        for (uint32_t inp_id : node.inputs) {
            inputs.push_back(&tensors[inp_id]);
        }

        // dispatch
        dispatch_kernel(node.op_type, node.params, inputs, &out);

        // Debug: capture layer-0 post-attention outputs and final layer-1 input
        // on the first prefill run.
        static int post_attn_dbg = 0;
        static bool expect_layer0_residual = false;
        static bool expect_layer1_input = false;
        if (post_attn_dbg == 0 && node.op_type == OpType::MATMUL && node.inputs.size() >= 2 && out.shape[1] > 1) {
            const auto& wnode = nodes[node.inputs[1]];
            if (wnode.op_type == OpType::CONSTANT && !wnode.params.str.empty()) {
                if (wnode.params.str[0].find("model_layers_0_self_attn_o_proj_weight.weights") != std::string::npos) {
                    const float* r0 = out.ptr<float>();
                    const float* r1 = out.ptr<float>() + out.stride[1] / sizeof(float);
                    fprintf(stderr, "  POST-ATTN DBG: out_proj[0..2,0]=%.6f %.6f %.6f\n", r0[0], r0[1], r0[2]);
                    fprintf(stderr, "  POST-ATTN DBG: out_proj[0..2,1]=%.6f %.6f %.6f\n", r1[0], r1[1], r1[2]);
                    expect_layer0_residual = true;
                } else if (wnode.params.str[0].find("model_layers_0_mlp_down_proj_weight.weights") != std::string::npos) {
                    expect_layer1_input = true;
                }
            }
        } else if (post_attn_dbg == 0 && expect_layer0_residual && node.op_type == OpType::ADD && out.shape[1] > 1) {
            const float* r0 = out.ptr<float>();
            const float* r1 = out.ptr<float>() + out.stride[1] / sizeof(float);
            fprintf(stderr, "  POST-ATTN DBG: residual[0..2,0]=%.6f %.6f %.6f\n", r0[0], r0[1], r0[2]);
            fprintf(stderr, "  POST-ATTN DBG: residual[0..2,1]=%.6f %.6f %.6f\n", r1[0], r1[1], r1[2]);
            expect_layer0_residual = false;
        } else if (post_attn_dbg == 0 && expect_layer1_input && node.op_type == OpType::ADD && out.shape[1] > 1) {
            const float* r0 = out.ptr<float>();
            const float* r1 = out.ptr<float>() + out.stride[1] / sizeof(float);
            fprintf(stderr, "  LAYER1 INPUT DBG: x[0..2,0]=%.6f %.6f %.6f\n", r0[0], r0[1], r0[2]);
            fprintf(stderr, "  LAYER1 INPUT DBG: x[0..2,1]=%.6f %.6f %.6f\n", r1[0], r1[1], r1[2]);
            expect_layer1_input = false;
            post_attn_dbg++;
        }

        // release completed tensors
        for (uint32_t rel_id : ctx.release_queue[i]) {
            auto& t = tensors[rel_id];
            if (rel_id < nodes.size()) {
                auto op = nodes[rel_id].op_type;
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
