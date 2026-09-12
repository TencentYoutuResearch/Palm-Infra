#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

// ---------------------------------------------------------------------------
// CONTIGUOUS: strided gather -> row-major. Reads via input TensorDesc strides
// and offset; writes densely to output. grid = (total_elements).
// ---------------------------------------------------------------------------
kernel void contiguous_f32(
    device const float*   IN     [[buffer(0)]],
    device float*         OUT    [[buffer(2)]],
    constant TensorDesc&  in     [[buffer(3)]],
    uint  gid                    [[thread_position_in_grid]])
{
    int s0 = in.shape[0], s1 = in.shape[1], s2 = in.shape[2], s3 = in.shape[3];
    int total = s0 * s1 * s2 * s3;
    if (int(gid) >= total) return;

    // Decompose flat index (row-major over shape) into 4-D coords.
    int idx = int(gid);
    int i0 = idx % s0; idx /= s0;
    int i1 = idx % s1; idx /= s1;
    int i2 = idx % s2; idx /= s2;
    int i3 = idx;

    uint src = in.offset
             + (uint)i0 * in.stride[0]
             + (uint)i1 * in.stride[1]
             + (uint)i2 * in.stride[2]
             + (uint)i3 * in.stride[3];
    OUT[gid] = IN[src];
}

// ---------------------------------------------------------------------------
// CONTIGUOUS (3D grid variant): coords come DIRECTLY from thread_position_in_grid
// so there are NO per-element integer div/mod (the cost in contiguous_f32). Used
// when the logical tensor collapses to <=3 dims (shape[3]==1). The host maps
// (dim0, dim1, dim2) to the grid; the dense output index is recomputed from
// coords. grid = (s0, s1, s2).
// ---------------------------------------------------------------------------
kernel void contiguous3d_f32(
    device const float*   IN     [[buffer(0)]],
    device float*         OUT    [[buffer(2)]],
    constant TensorDesc&  in     [[buffer(3)]],
    uint3 gid                    [[thread_position_in_grid]])
{
    int s0 = in.shape[0], s1 = in.shape[1], s2 = in.shape[2];
    int i0 = int(gid.x), i1 = int(gid.y), i2 = int(gid.z);
    if (i0 >= s0 || i1 >= s1 || i2 >= s2) return;

    uint src = in.offset
             + (uint)i0 * in.stride[0]
             + (uint)i1 * in.stride[1]
             + (uint)i2 * in.stride[2];
    // Dense (row-major over shape) output index = i0 + i1*s0 + i2*s0*s1.
    uint dst = (uint)i0 + (uint)i1 * (uint)s0 + (uint)i2 * (uint)s0 * (uint)s1;
    OUT[dst] = IN[src];
}

// ---------------------------------------------------------------------------
// TILE (dim-2 broadcast): [s0, s1, 1] -> [s0, s1, reps2] (MLA k_rope broadcast
// across heads). Source dim-2 index is always 0; source may be strided.
// in.shape = SOURCE [s0,s1,1]; grid = (s0, s1, reps2). Dense row-major output.
// ---------------------------------------------------------------------------
kernel void tile_dim2_f32(
    device const float*   IN     [[buffer(0)]],
    device float*         OUT    [[buffer(2)]],
    constant TensorDesc&  in     [[buffer(3)]],
    uint3 gid                    [[thread_position_in_grid]])
{
    int s0 = in.shape[0], s1 = in.shape[1], reps2 = in.shape[2];
    int i0 = int(gid.x), i1 = int(gid.y), r = int(gid.z);
    if (i0 >= s0 || i1 >= s1 || r >= reps2) return;

    uint src = in.offset + (uint)i0 * in.stride[0] + (uint)i1 * in.stride[1];
    uint dst = (uint)i0 + (uint)i1 * (uint)s0 + (uint)r * (uint)s0 * (uint)s1;
    OUT[dst] = IN[src];
}

// ---------------------------------------------------------------------------
// CONCAT along dim 0: write one (possibly strided) source into its dim-0 slab
// of a dense output [out_shape0, s1, s2] (shape[3]==1). grid = (s0, s1, s2) of
// the SOURCE. Dispatched once per concat input with its own dim_offset.
// ---------------------------------------------------------------------------
kernel void concat_dim0_f32(
    device const float*   IN     [[buffer(0)]],
    device float*         OUT    [[buffer(2)]],
    constant ConcatParams& p     [[buffer(3)]],
    uint3 gid                    [[thread_position_in_grid]])
{
    int s0 = p.shape[0], s1 = p.shape[1], s2 = p.shape[2];
    int i0 = int(gid.x), i1 = int(gid.y), i2 = int(gid.z);
    if (i0 >= s0 || i1 >= s1 || i2 >= s2) return;

    uint src = p.offset + (uint)i0 * p.stride[0]
             + (uint)i1 * p.stride[1] + (uint)i2 * p.stride[2];
    // dense output over [out_shape0, s1, s2], dim-0 shifted by dim_offset.
    uint o0 = (uint)(p.dim_offset + i0);
    uint dst = o0 + (uint)i1 * (uint)p.out_shape0
             + (uint)i2 * (uint)p.out_shape0 * (uint)s1;
    OUT[dst] = IN[src];
}
