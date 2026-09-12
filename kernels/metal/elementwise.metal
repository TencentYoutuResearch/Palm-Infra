#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Elementwise add (contiguous), with scalar-b broadcast.
// ---------------------------------------------------------------------------
kernel void add_f32(
    device const float*   A      [[buffer(0)]],
    device const float*   B      [[buffer(1)]],
    device float*         O      [[buffer(2)]],
    constant EwiseParams& p      [[buffer(3)]],
    uint  gid                    [[thread_position_in_grid]])
{
    if (int(gid) >= p.n) return;
    uint q = gid;
    uint i0 = q % (uint)p.shape[0]; q /= (uint)p.shape[0];
    uint i1 = q % (uint)p.shape[1]; q /= (uint)p.shape[1];
    uint i2 = q % (uint)p.shape[2]; q /= (uint)p.shape[2];
    uint i3 = q;
    uint ai = p.a_offset + i0*(uint)p.a_stride[0] +
              i1*(uint)p.a_stride[1] + i2*(uint)p.a_stride[2] +
              i3*(uint)p.a_stride[3];
    uint bi = p.b_offset + i0*(uint)p.b_stride[0] +
              i1*(uint)p.b_stride[1] + i2*(uint)p.b_stride[2] +
              i3*(uint)p.b_stride[3];
    uint oi = p.out_offset + i0*(uint)p.out_stride[0] +
              i1*(uint)p.out_stride[1] + i2*(uint)p.out_stride[2] +
              i3*(uint)p.out_stride[3];
    O[oi] = A[ai] + B[bi];
}

// Elementwise multiply (contiguous), with scalar-b broadcast.
kernel void mul_f32(
    device const float*   A      [[buffer(0)]],
    device const float*   B      [[buffer(1)]],
    device float*         O      [[buffer(2)]],
    constant EwiseParams& p      [[buffer(3)]],
    uint  gid                    [[thread_position_in_grid]])
{
    if (int(gid) >= p.n) return;
    uint q = gid;
    uint i0 = q % (uint)p.shape[0]; q /= (uint)p.shape[0];
    uint i1 = q % (uint)p.shape[1]; q /= (uint)p.shape[1];
    uint i2 = q % (uint)p.shape[2]; q /= (uint)p.shape[2];
    uint i3 = q;
    uint ai = p.a_offset + i0*(uint)p.a_stride[0] +
              i1*(uint)p.a_stride[1] + i2*(uint)p.a_stride[2] +
              i3*(uint)p.a_stride[3];
    uint bi = p.b_offset + i0*(uint)p.b_stride[0] +
              i1*(uint)p.b_stride[1] + i2*(uint)p.b_stride[2] +
              i3*(uint)p.b_stride[3];
    uint oi = p.out_offset + i0*(uint)p.out_stride[0] +
              i1*(uint)p.out_stride[1] + i2*(uint)p.out_stride[2] +
              i3*(uint)p.out_stride[3];
    O[oi] = A[ai] * B[bi];
}

// Fused output gate: value * sigmoid(gate).
kernel void sigmoid_mul_f32(
    device const float*   VALUE  [[buffer(0)]],
    device const float*   GATE   [[buffer(1)]],
    device float*         O      [[buffer(2)]],
    constant EwiseParams& p      [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]])
{
    if (int(gid) >= p.n) return;
    uint q = gid;
    uint i0 = q % (uint)p.shape[0]; q /= (uint)p.shape[0];
    uint i1 = q % (uint)p.shape[1]; q /= (uint)p.shape[1];
    uint i2 = q % (uint)p.shape[2]; q /= (uint)p.shape[2];
    uint i3 = q;
    uint vi = p.a_offset + i0*(uint)p.a_stride[0] +
              i1*(uint)p.a_stride[1] + i2*(uint)p.a_stride[2] +
              i3*(uint)p.a_stride[3];
    uint gi = p.b_offset + i0*(uint)p.b_stride[0] +
              i1*(uint)p.b_stride[1] + i2*(uint)p.b_stride[2] +
              i3*(uint)p.b_stride[3];
    uint oi = p.out_offset + i0*(uint)p.out_stride[0] +
              i1*(uint)p.out_stride[1] + i2*(uint)p.out_stride[2] +
              i3*(uint)p.out_stride[3];
    const float gate = GATE[gi];
    O[oi] = VALUE[vi] / (1.0f + exp(-gate));
}

// SILU: x * sigmoid(x) (contiguous).
kernel void silu_f32(
    device const float*   X      [[buffer(0)]],
    device float*         O      [[buffer(2)]],
    constant EwiseParams& p      [[buffer(3)]],
    uint  gid                    [[thread_position_in_grid]])
{
    if (int(gid) >= p.n) return;
    float v = X[p.a_offset + gid];
    O[p.out_offset + gid] = v / (1.0f + exp(-v));
}

// SIGMOID: 1 / (1 + exp(-x)) (contiguous).
kernel void sigmoid_f32(
    device const float*   X      [[buffer(0)]],
    device float*         O      [[buffer(2)]],
    constant EwiseParams& p      [[buffer(3)]],
    uint  gid                    [[thread_position_in_grid]])
{
    if (int(gid) >= p.n) return;
    float v = X[p.a_offset + gid];
    O[p.out_offset + gid] = 1.0f / (1.0f + exp(-v));
}

kernel void gelu_f32(
    device const float* X [[buffer(0)]], device float* O [[buffer(2)]],
    constant EwiseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.n) return;
    uint col = gid % (uint)p.shape0, row = gid / (uint)p.shape0;
    float v = X[p.a_offset + row * (uint)p.a_row_stride + col];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    O[p.out_offset + row * (uint)p.out_row_stride + col] =
        0.5f * v * (1.0f + precise::tanh(inner));
}

kernel void tanh_f32(
    device const float* X [[buffer(0)]], device float* O [[buffer(2)]],
    constant EwiseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.n) return;
    uint col = gid % (uint)p.shape0, row = gid / (uint)p.shape0;
    float v = X[p.a_offset + row * (uint)p.a_row_stride + col];
    O[p.out_offset + row * (uint)p.out_row_stride + col] = precise::tanh(v);
}

kernel void exp_f32(
    device const float* X [[buffer(0)]], device float* O [[buffer(2)]],
    constant EwiseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.n) return;
    uint col = gid % (uint)p.shape0, row = gid / (uint)p.shape0;
    float v = X[p.a_offset + row * (uint)p.a_row_stride + col];
    O[p.out_offset + row * (uint)p.out_row_stride + col] = precise::exp(v);
}

kernel void softplus_f32(
    device const float* X [[buffer(0)]], device float* O [[buffer(2)]],
    constant EwiseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.n) return;
    uint col = gid % (uint)p.shape0, row = gid / (uint)p.shape0;
    float v = X[p.a_offset + row * (uint)p.a_row_stride + col];
    float y = v > 20.0f ? v :
              (v < -20.0f ? precise::exp(v) :
                            precise::log(1.0f + precise::exp(v)));
    O[p.out_offset + row * (uint)p.out_row_stride + col] = y;
}

// Fused SwiGLU: reads gate/up halves from a single merged [2I, rows] buffer
// (gate = row[0..I), up = row[I..2I)), writes dense out[row*I + i] =
// silu(gate[i]) * up[i]. Splits internally with the merged row stride so it does
// NOT depend on stride-aware slice views (the dense mul_f32/silu_f32 can't handle
// those). grid = p.n (1D).
kernel void swiglu_f32(
    device const float*   M      [[buffer(0)]],
    device float*         O      [[buffer(2)]],
    constant SwigluParams& p     [[buffer(3)]],
    uint  gid                    [[thread_position_in_grid]])
{
    if (int(gid) >= p.n) return;
    int I = p.I;
    int row = int(gid) / I;
    int i   = int(gid) % I;
    uint base = p.merged_offset + (uint)row * (uint)p.merged_row_stride;
    float g = M[base + (uint)i];
    float u = M[base + (uint)(I + i)];
    O[p.out_offset + gid] = (g / (1.0f + exp(-g))) * u;
}
