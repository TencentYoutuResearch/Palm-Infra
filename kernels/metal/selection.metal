#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;
inline bool argmax_better(float value, uint index,
                          float best_value, uint best_index) {
    return value > best_value ||
           (value == best_value && index < best_index);
}

// Reduce a large FP32 logits vector to at most 256 partial maxima.  Each
// threadgroup walks a grid-stride slice so the launch size stays bounded even
// for 100k+ vocabularies.
kernel void argmax_f32_stage1(
    device const float* values [[buffer(0)]],
    device ArgMaxPair* partial [[buffer(1)]],
    constant ArgMaxParams& p [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    threadgroup float best_values[256];
    threadgroup uint best_indices[256];
    float best_value = -INFINITY;
    uint best_index = 0xffffffffu;
    const uint start = group * 256u + tid;
    const uint stride = p.group_count * 256u;
    for (uint i = start; i < p.count; i += stride) {
        const float value = values[i];
        if (argmax_better(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }
    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (tid < offset &&
            argmax_better(best_values[tid + offset],
                          best_indices[tid + offset],
                          best_values[tid], best_indices[tid])) {
            best_values[tid] = best_values[tid + offset];
            best_indices[tid] = best_indices[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partial[group].value = best_values[0];
        partial[group].index = best_indices[0];
    }
}

// The first pass deliberately emits no more than 256 pairs, so one group can
// finish the reduction and write only the token id to shared host memory.
kernel void argmax_f32_stage2(
    device const ArgMaxPair* partial [[buffer(0)]],
    device uint* result [[buffer(1)]],
    constant ArgMaxParams& p [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float best_values[256];
    threadgroup uint best_indices[256];
    float best_value = -INFINITY;
    uint best_index = 0xffffffffu;
    if (tid < p.group_count) {
        best_value = partial[tid].value;
        best_index = partial[tid].index;
    }
    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (tid < offset &&
            argmax_better(best_values[tid + offset],
                          best_indices[tid + offset],
                          best_values[tid], best_indices[tid])) {
            best_values[tid] = best_values[tid + offset];
            best_indices[tid] = best_indices[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u)
        result[0] = best_indices[0];
}

