#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

// ---------------------------------------------------------------------------
// RMS norm over dim0. One threadgroup per row; parallel sum-of-squares.
// weight is FP32 (per qwen3 norm weights).
// ---------------------------------------------------------------------------
kernel void rms_norm_f32(
    device const float*    X     [[buffer(0)]],
    device const float*    W     [[buffer(1)]],
    device float*          O     [[buffer(2)]],
    constant RmsNormParams& p    [[buffer(3)]],
    uint  row                    [[threadgroup_position_in_grid]],
    uint  tid                    [[thread_position_in_threadgroup]],
    uint  tcount                 [[threads_per_threadgroup]])
{
    if (int(row) >= p.rows) return;
    device const float* x = X + p.x_offset + (uint)row * p.x_row_stride;
    device float*       o = O + p.out_offset + (uint)row * p.out_row_stride;
    device const float* w = W + p.w_offset;

    // Sum of squares: vectorized float4 loads + two-level simd reduction (simd_sum
    // within each simdgroup, then one shared-mem pass across simdgroups, then a
    // final simd_sum) — far fewer threadgroup barriers than the 8-level tree.
    uint lane = tid & 31u;
    uint sg   = tid >> 5;
    uint n_sg = (tcount + 31u) / 32u;
    float partial = 0.0f;
    int d4 = p.dim0 & ~3;
    device const float4* x4 = (device const float4*)x;
    for (int q = int(tid); q < (d4 >> 2); q += int(tcount)) {
        float4 v = x4[q];
        partial += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (int i = d4 + int(tid); i < p.dim0; i += int(tcount)) {
        float v = x[i]; partial += v * v;
    }
    partial = simd_sum(partial);
    threadgroup float sh[32];       // one slot per simdgroup (<=32 SGs)
    if (lane == 0) sh[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every simdgroup reduces the per-SG partials the same way, so ALL threads
    // end up with the identical full total — compute scale locally, no broadcast.
    float total = simd_sum((lane < n_sg) ? sh[lane] : 0.0f);
    float scale = rsqrt(total / float(p.dim0) + p.eps);

    // Scaled output: vectorized float4 where possible.
    device float4* o4 = (device float4*)o;
    device const float4* w4 = (device const float4*)w;
    for (int q = int(tid); q < (d4 >> 2); q += int(tcount)) {
        float4 v = x4[q], wv = w4[q];
        o4[q] = v * scale * wv;
    }
    for (int i = d4 + int(tid); i < p.dim0; i += int(tcount)) {
        o[i] = x[i] * scale * w[i];
    }
}

// Q/K normalization and RoPE share the same per-(head,position) row. Reading
// the original strided view and writing the final dense SDPA layout here
// replaces RMSNorm, materialization, and RoPE with one dispatch.
kernel void rms_norm_rope_f32(
    device const float*          X    [[buffer(0)]],
    device const float*          W    [[buffer(1)]],
    device float*                O    [[buffer(2)]],
    constant RmsNormRopeParams&  p    [[buffer(3)]],
    device const float*          COS  [[buffer(4)]],
    device const float*          SIN  [[buffer(5)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint tcount                        [[threads_per_threadgroup]])
{
    if ((int)row >= p.rows) return;
    device const float* x =
        X + p.x_offset + row*(uint)p.x_row_stride;
    device const float* w = W + p.w_offset;
    device float* o =
        O + p.out_offset + row*(uint)p.out_row_stride;

    uint lane = tid & 31u;
    uint sg = tid >> 5;
    uint nsg = (tcount + 31u) >> 5;
    float partial = 0.0f;
    const int d4 = p.dim0 & ~3;
    device const float4* x4 = (device const float4*)x;
    for (int q = (int)tid; q < (d4 >> 2); q += (int)tcount) {
        const float4 v = x4[q];
        partial += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (int d = d4 + (int)tid; d < p.dim0; d += (int)tcount)
        partial += x[d]*x[d];
    partial = simd_sum(partial);
    threadgroup float sums[32];
    if (lane == 0) sums[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        simd_sum(lane < nsg ? sums[lane] : 0.0f);
    const float scale = rsqrt(total/(float)p.dim0 + p.eps);

    const int half_dim = p.rope_dim/2;
    const int pos = (int)row % p.seq_len;
    if ((int)tid < half_dim) {
        const int d0 = p.interleave ? 2*(int)tid : (int)tid;
        const int d1 = p.interleave ? d0+1 : d0+half_dim;
        const float x0 = x[d0]*scale*w[d0];
        const float x1 = x[d1]*scale*w[d1];
        const float c =
            COS[p.cos_offset + (uint)(pos*half_dim+(int)tid)];
        const float s =
            SIN[p.sin_offset + (uint)(pos*half_dim+(int)tid)];
        o[d0] = x0*c - x1*s;
        o[d1] = x1*c + x0*s;
    }
    for (int d = p.rope_dim + (int)tid;
         d < p.dim0; d += (int)tcount)
        o[d] = x[d]*scale*w[d];
}

// Q and K have the same head dimension and position-dependent RoPE tables.
// Run their independent rows in one grid to remove one command dispatch per
// transformer layer while preserving full row-level parallelism.
kernel void qk_rms_norm_rope_f32(
    device const float*            QUERY    [[buffer(0)]],
    device const float*            KEY      [[buffer(1)]],
    device const float*            QUERY_W  [[buffer(2)]],
    device const float*            KEY_W    [[buffer(3)]],
    device float*                  OUT      [[buffer(4)]],
    constant QkRmsNormRopeParams&  p        [[buffer(5)]],
    device const float*            COS      [[buffer(6)]],
    device const float*            SIN      [[buffer(7)]],
    uint row                                [[threadgroup_position_in_grid]],
    uint tid                                [[thread_position_in_threadgroup]],
    uint tcount                             [[threads_per_threadgroup]])
{
    if ((int)row >= p.rows) return;
    const int query_rows = p.seq_len * p.query_heads;
    const bool is_query = (int)row < query_rows;
    const int source_row = is_query ? (int)row : (int)row - query_rows;
    device const float* x =
        (is_query ? QUERY + p.query_x_offset : KEY + p.key_x_offset) +
        (uint)source_row *
            (uint)(is_query ? p.query_x_row_stride : p.key_x_row_stride);
    device const float* w =
        is_query ? QUERY_W + p.query_w_offset : KEY_W + p.key_w_offset;
    device float* o =
        OUT + p.out_offset + row*(uint)p.out_row_stride;

    uint lane = tid & 31u;
    uint sg = tid >> 5;
    uint nsg = (tcount + 31u) >> 5;
    float partial = 0.0f;
    const int d4 = p.dim0 & ~3;
    device const float4* x4 = (device const float4*)x;
    for (int q = (int)tid; q < (d4 >> 2); q += (int)tcount) {
        const float4 v = x4[q];
        partial += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (int d = d4 + (int)tid; d < p.dim0; d += (int)tcount)
        partial += x[d]*x[d];
    partial = simd_sum(partial);
    float total;
    if (tcount == 32u) {
        total = partial;
    } else {
        threadgroup float sums[32];
        if (lane == 0) sums[sg] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = simd_sum(
            lane < nsg ? sums[lane] : 0.0f);
    }
    const float scale = rsqrt(total/(float)p.dim0 + p.eps);

    const int half_dim = p.rope_dim/2;
    const int pos = source_row % p.seq_len;
    for (int pair = (int)tid;
         pair < half_dim;
         pair += (int)tcount) {
        const int d0 = p.interleave ? 2*pair : pair;
        const int d1 = p.interleave ? d0+1 : d0+half_dim;
        const float x0 = x[d0]*scale*w[d0];
        const float x1 = x[d1]*scale*w[d1];
        const float c =
            COS[p.cos_offset + (uint)(pos*half_dim+pair)];
        const float s =
            SIN[p.sin_offset + (uint)(pos*half_dim+pair)];
        o[d0] = x0*c - x1*s;
        o[d1] = x1*c + x0*s;
    }
    for (int d = p.rope_dim + (int)tid;
         d < p.dim0; d += (int)tcount)
        o[d] = x[d]*scale*w[d];
}

// Residual add + RMSNorm. The residual stream is updated in place and the
// normalized value is written separately for the following projection.
kernel void add_rms_norm_f32(
    device float*                 RESIDUAL [[buffer(0)]],
    device const float*           UPDATE   [[buffer(1)]],
    device float*                 O        [[buffer(2)]],
    constant AddRmsNormParams&    p        [[buffer(3)]],
    device const float*           W        [[buffer(4)]],
    uint row                               [[threadgroup_position_in_grid]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint tcount                            [[threads_per_threadgroup]])
{
    if (int(row) >= p.rows) return;
    device float* residual =
        RESIDUAL + p.residual_offset +
        row * (uint)p.residual_row_stride;
    device const float* update =
        UPDATE + p.update_offset + row * (uint)p.update_row_stride;
    device float* out =
        O + p.out_offset + row * (uint)p.out_row_stride;

    float partial = 0.0f;
    const int d4 = p.dim0 & ~3;
    device float4* residual4 = (device float4*)residual;
    device const float4* update4 = (device const float4*)update;
    for (int q = int(tid); q < (d4 >> 2); q += int(tcount)) {
        const float4 value = residual4[q] + update4[q];
        residual4[q] = value;
        partial += dot(value, value);
    }
    for (int i = d4 + int(tid); i < p.dim0; i += int(tcount)) {
        const float value = residual[i] + update[i];
        residual[i] = value;
        partial += value * value;
    }

    const uint lane = tid & 31u;
    const uint sg = tid >> 5;
    const uint nsg = (tcount + 31u) >> 5;
    partial = simd_sum(partial);
    threadgroup float sh[32];
    if (lane == 0) sh[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        simd_sum(lane < nsg ? sh[lane] : 0.0f);
    const float scale =
        rsqrt(total / float(p.dim0) + p.eps);

    device float4* out4 = (device float4*)out;
    device const float4* weight4 = (device const float4*)W;
    for (int q = int(tid); q < (d4 >> 2); q += int(tcount))
        out4[q] = residual4[q] * scale * weight4[q];
    for (int i = d4 + int(tid); i < p.dim0; i += int(tcount))
        out[i] = residual[i] * scale * W[i];
}

// Layer norm over dim0. W and bias are bound at their byte offsets, so the
// parameter block only carries activation/output offsets.
kernel void layer_norm_f32(
    device const float*      X     [[buffer(0)]],
    device const float*      W     [[buffer(1)]],
    device float*            O     [[buffer(2)]],
    constant LayerNormParams& p    [[buffer(3)]],
    device const float*      B     [[buffer(4)]],
    uint  row                       [[threadgroup_position_in_grid]],
    uint  tid                       [[thread_position_in_threadgroup]],
    uint  tcount                    [[threads_per_threadgroup]])
{
    if (int(row) >= p.rows) return;
    device const float* x = X + p.x_offset + row * (uint)p.x_row_stride;
    device float* o = O + p.out_offset + row * (uint)p.out_row_stride;

    uint lane = tid & 31u;
    uint sg = tid >> 5;
    uint n_sg = (tcount + 31u) / 32u;
    threadgroup float sh[32];

    float partial = 0.0f;
    for (int i = int(tid); i < p.dim0; i += int(tcount)) partial += x[i];
    partial = simd_sum(partial);
    if (lane == 0) sh[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = simd_sum((lane < n_sg) ? sh[lane] : 0.0f) / float(p.dim0);
    // All SIMD groups must finish consuming the mean partials before any
    // group reuses sh[] for variance partials.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    partial = 0.0f;
    for (int i = int(tid); i < p.dim0; i += int(tcount)) {
        float z = x[i] - mean;
        partial += z * z;
    }
    partial = simd_sum(partial);
    if (lane == 0) sh[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float variance =
        simd_sum((lane < n_sg) ? sh[lane] : 0.0f) / float(p.dim0);
    float scale = rsqrt(variance + p.eps);

    for (int i = int(tid); i < p.dim0; i += int(tcount))
        o[i] = (x[i] - mean) * scale * W[i] + B[i];
}
