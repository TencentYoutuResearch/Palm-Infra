#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

// ---------------------------------------------------------------------------
// RoPE, interleave=false (Qwen3). x layout [head_dim, seq, heads].
// cos/sin are [rope_dim/2, seq] (per-position). One thread per (pair, pos, head).
// grid = (rope_dim/2, seq_len, heads)
// ---------------------------------------------------------------------------
kernel void rope_f32(
    device float*          X     [[buffer(0)]],
    device const float*    COS   [[buffer(1)]],
    device const float*    SIN   [[buffer(2)]],
    constant RopeParams&   p     [[buffer(3)]],
    uint3 gid                    [[thread_position_in_grid]])
{
    int half_dim = p.rope_dim / 2;
    int i   = int(gid.x);   // pair index [0, half_dim)
    int pos = int(gid.y);
    int h   = int(gid.z);
    if (i >= half_dim || pos >= p.seq_len || h >= p.heads) return;

    device float* base = X + p.x_offset
                       + (uint)pos * p.x_stride_pos
                       + (uint)h   * p.x_stride_head;

    float c = COS[p.cos_offset + (uint)pos * half_dim + i];
    float s = SIN[p.sin_offset + (uint)pos * half_dim + i];

    int i0 = p.interleave ? 2 * i : i;
    int i1 = p.interleave ? 2 * i + 1 : i + half_dim;
    float x0 = base[i0];
    float x1 = base[i1];
    base[i0] = x0 * c - x1 * s;
    base[i1] = x1 * c + x0 * s;
}
