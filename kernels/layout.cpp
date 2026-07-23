#include "kernels/layout.h"

#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// materialize_strided
//
// Used by RESHAPE (non-contiguous input) and CONTIGUOUS ops. The generic
// 4-loop elementwise memcpy is slow when dim0 is contiguous (stride[0]==es):
// each iteration does a 4-byte memcpy with loop overhead. This helper detects
// the contiguous-inner-dim case and falls back to a single memcpy per
// (i1,i2,i3) row, cutting loop overhead by shape[0]×.
//
// For the fully contiguous case it collapses to a single memcpy.
static inline void materialize_strided(const Tensor& src, void* dst) {
    size_t es = src.element_size();
    char* dst_base = static_cast<char*>(dst);
    int64_t d0 = src.shape[0];
    int64_t row_bytes = d0 * es;

    // Fast path: dim0 contiguous → memcpy each row in one call.
    if (src.stride[0] == (size_t)es) {
        int64_t flat = 0;
        for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
            for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
                for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                    const char* src_ptr = static_cast<const char*>(src.data) +
                                          i1 * src.stride[1] +
                                          i2 * src.stride[2] +
                                          i3 * src.stride[3];
                    std::memcpy(dst_base + flat * es, src_ptr, row_bytes);
                    flat += d0;
                }
            }
        }
        return;
    }

    // Slow path: element-wise copy (dim0 not contiguous).
    int64_t flat = 0;
    for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
        for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
            for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                for (int64_t i0 = 0; i0 < d0; i0++) {
                    const char* src_ptr =
                        static_cast<const char*>(src.data) +
                        i0 * src.stride[0] + i1 * src.stride[1] +
                        i2 * src.stride[2] + i3 * src.stride[3];
                    std::memcpy(dst_base + flat * es, src_ptr, es);
                    flat++;
                }
            }
        }
    }
}

void kernel_layout(const GraphNode& node,
                   const std::vector<const Tensor*>& inputs, Tensor* output) {
    const OpType op = node.op_type;
    const OpParams& params = node.params;
    switch (op) {
    case OpType::RESHAPE:
        // Reshape preserves logical element order. For contiguous inputs this
        // is a zero-copy metadata change; for non-contiguous views we
        // materialize a contiguous output.
        //
        // Output shape priority for each dim d:
        //   - if node.dim_expr[d].is_dynamic(): use output->shape[d] (already
        //     filled by execute_graph main loop via eval_dim)
        //   - else: use params.i32[d] (build-time literal)
        if (!inputs.empty() && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            int64_t new_shape[4] = {output->shape[0], output->shape[1],
                                    output->shape[2], output->shape[3]};
            if (params.i32.size() >= 4) {
                for (int d = 0; d < 4; d++) {
                    if (!node.dim_expr[d].is_dynamic()) {
                        new_shape[d] = params.i32[d];
                    }
                    // dynamic dims: preserve output->shape[d] (runtime-filled)
                }
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
                materialize_strided(src, output->data);
            }
        }
        break;
    case OpType::CONCAT:
        // Concatenate inputs along specified dimension. Materializes output and
        // respects input strides, so view inputs remain correct.
        //
        // Fast path for dim==0: for each (i1, i2, i3), the i0 elements are
        // contiguous in both src (stride[0]==es) and dst (stride[0]==es).
        // So we can memcpy the whole i0 block at once per (i1,i2,i3),
        // reducing memcpy calls from shape[0]*shape[1]*shape[2]*shape[3]
        // to shape[1]*shape[2]*shape[3].
        //
        // This is correct regardless of whether the outer dims are contiguous
        // — we still iterate i1/i2/i3 with their respective strides.
        if (!inputs.empty() && output) {
            int dim = graph_params::get_i32(params, 0, 0);
            size_t es = output->element_size();
            int64_t dim_offset = 0;
            char* dst_base = static_cast<char*>(output->data);

            if (dim == 0) {
                // Fast path: dim=0 concat with per-(i1,i2,i3) block copy.
                // i0 is contiguous in both src and dst (stride[0] == es).
                for (size_t i = 0; i < inputs.size(); i++) {
                    if (!inputs[i] || !inputs[i]->data)
                        continue;
                    const Tensor& src = *inputs[i];
                    const char* src_base = static_cast<const char*>(src.data);
                    int64_t i0_bytes = (int64_t)src.shape[0] * es;
                    int64_t dst_o0_offset = dim_offset * output->stride[0];

                    for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
                        for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
                            for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                                const char* src_ptr =
                                    src_base + i1 * src.stride[1] +
                                    i2 * src.stride[2] + i3 * src.stride[3];
                                char* dst_ptr = dst_base + dst_o0_offset +
                                                i1 * output->stride[1] +
                                                i2 * output->stride[2] +
                                                i3 * output->stride[3];
                                std::memcpy(dst_ptr, src_ptr, i0_bytes);
                            }
                        }
                    }
                    dim_offset += src.shape[0];
                }
            } else {
                // General path: per-element copy for dim != 0
                for (size_t i = 0; i < inputs.size(); i++) {
                    if (!inputs[i] || !inputs[i]->data)
                        continue;
                    const Tensor& src = *inputs[i];
                    const char* src_base = static_cast<const char*>(src.data);
                    for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
                        for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
                            for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                                for (int64_t i0 = 0; i0 < src.shape[0]; i0++) {
                                    int64_t o0 = i0, o1 = i1, o2 = i2, o3 = i3;
                                    if (dim == 1)
                                        o1 += dim_offset;
                                    else if (dim == 2)
                                        o2 += dim_offset;
                                    else if (dim == 3)
                                        o3 += dim_offset;

                                    const char* src_ptr =
                                        src_base + i0 * src.stride[0] +
                                        i1 * src.stride[1] +
                                        i2 * src.stride[2] + i3 * src.stride[3];
                                    char* dst_ptr = dst_base +
                                                    o0 * output->stride[0] +
                                                    o1 * output->stride[1] +
                                                    o2 * output->stride[2] +
                                                    o3 * output->stride[3];
                                    std::memcpy(dst_ptr, src_ptr, es);
                                }
                            }
                        }
                    }
                    dim_offset += src.shape[dim];
                }
            }
        }
        break;
    case OpType::SLICE:
        // zero-copy: slice produces a view of the parent and must preserve the
        // parent's stride layout.
        if (!inputs.empty() && inputs[0] && output) {
            int dim = graph_params::get_i32(params, 0, 0);
            int offset = graph_params::get_i32(params, 1, 0);
            int size =
                graph_params::get_i32(params, 2, (int)output->shape[dim]);
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
        //
        // Fast path for dim=2-only tile (common MLA case:
        // k_rope [rope_dim, seq, 1] → [rope_dim, seq, num_heads]):
        // Each rep is a copy of src with the same (i0, i1) layout.
        // For each (i0, i1) pair, copy reps[2] times to dst.
        // This is correct regardless of stride layout.
        //
        // General path: per-element copy with div/mod.
        if (!inputs.empty() && inputs[0] && output) {
            const Tensor& src = *inputs[0];
            size_t es = src.element_size();
            int reps[4] = {1, 1, 1, 1};
            for (int d = 0; d < 4 && d < (int)params.i32.size(); d++) {
                reps[d] = params.i32[d];
            }
            char* dst = static_cast<char*>(output->data);
            const char* s = static_cast<const char*>(src.data);

            bool only_dim2 =
                (reps[0] == 1 && reps[1] == 1 && reps[3] == 1 && reps[2] > 1);

            if (only_dim2) {
                // For each (i0, i1), copy src[i0][i1] to dst[i0][i1][r] for
                // r=0..reps[2]-1. src is [shape0, shape1, 1, 1], dst is
                // [shape0, shape1, reps2, 1]. For fixed (i0, i1), src elements
                // are contiguous (stride[0]==es in inner dim). But i0 may not
                // be contiguous across i1. Safe approach: per (i0, i1) row,
                // copy shape0 elements reps[2] times.
                int64_t i0_bytes = (int64_t)src.shape[0] * es;
                for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                    const char* src_row = s + i1 * src.stride[1];
                    // In dst, [i0=0..shape0-1, i1, r, 0] — for fixed i1, r,
                    // i0 is contiguous (stride[0]==es).
                    char* dst_row_base = dst + i1 * output->stride[1];
                    for (int r = 0; r < reps[2]; r++) {
                        char* dst_row = dst_row_base + r * output->stride[2];
                        std::memcpy(dst_row, src_row, i0_bytes);
                    }
                }
            } else {
                // General path
                int64_t total = output->nelements();
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
        }
        break;

    case OpType::PERMUTE:
        if (!inputs.empty() && inputs[0] && output && params.i32.size() >= 4) {
            *output = inputs[0]->permute(params.i32[0], params.i32[1],
                                         params.i32[2], params.i32[3]);
        }
        break;

    case OpType::CONTIGUOUS:
        // Materialize to row-major contiguous buffer.
        if (!inputs.empty() && inputs[0] && inputs[0]->data && output &&
            output->data) {
            materialize_strided(*inputs[0], output->data);
        }
        break;
    default:
        std::fprintf(stderr, "layout: unsupported op_type %u\n",
                     static_cast<unsigned>(op));
        break;
    }
}
