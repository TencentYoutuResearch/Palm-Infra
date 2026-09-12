#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

inline float apply_activation(float v, int act);
inline float apply_activation_at(float v, int act, int n, int begin, int len);

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

kernel void matmul_cast_f32_to_f16(
    device const float* A [[buffer(0)]],
    device half* AH [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint count = (uint)p.M * (uint)p.K;
    const uint base = gid * 4;
    if (base >= count) return;
    const uint m = base / (uint)p.K;
    const uint k = base - m * (uint)p.K;
    if (k + 4 <= (uint)p.K) {
        const device float4* src = (const device float4*)(
            A + p.a_offset + m * (uint)p.a_row_stride + k);
        *((device half4*)(AH + base)) = half4(*src);
    } else {
        // Generic tail for odd K. It may cross a logical row boundary, so
        // derive the row/column again for every remaining scalar.
        for (uint i = 0; i < 4 && base + i < count; ++i) {
            const uint linear = base + i;
            const uint row = linear / (uint)p.K;
            const uint col = linear - row * (uint)p.K;
            AH[linear] =
                half(A[p.a_offset + row * (uint)p.a_row_stride + col]);
        }
    }
}

#ifdef MOLLM_METAL_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// FP16 tensor GEMM with both operands read directly from device memory.
// MPP understands the strided [K,N] weight view, avoiding K32 threadgroup
// staging and its barriers.
kernel void gemm_tensor_direct_f32a_f16b_f32c(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const int TN = 64, TM = 128, TK = 32;
    const int n0 = (int)tgpig.y * TN;
    const int m0 = (int)tgpig.x * TM;
    auto tA = tensor((device float*)(A + p.a_offset),
                     dextents<int32_t,2>(p.K, p.M),
                     array<int,2>({1, p.a_row_stride}));
    auto tW = tensor((device half*)(B + p.b_offset),
                     dextents<int32_t,2>(p.K, p.N),
                     array<int,2>({1, p.b_row_stride}));
    matmul2d<matmul2d_descriptor(TM, TN, TK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto acc =
        mm.get_destination_cooperative_tensor<decltype(tA), decltype(tW),
                                              float>();
    for (int k0 = 0; k0 < p.K; k0 += TK) {
        auto a = tA.slice(k0, m0);
        auto w = tW.slice(k0, n0);
        mm.run(a, w, acc);
    }
    auto out = tensor(C + p.c_offset,
                      dextents<int32_t,2>(p.N, p.M),
                      array<int,2>({1, p.c_row_stride}));
    acc.store(out.slice(n0, m0));
}

// Direct-device counterpart for activations already cast to FP16. This avoids
// weight staging and its K32 barriers while retaining FP32 accumulation.
kernel void gemm_tensor_direct_f16a_f16b_f32c(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const int TN = 64, TM = 128, TK = 32;
    const int n0 = (int)tgpig.y * TN;
    const int m0 = (int)tgpig.x * TM;
    auto tA = tensor((device half*)(A + p.a_offset),
                     dextents<int32_t,2>(p.K, p.M),
                     array<int,2>({1, p.a_row_stride}));
    auto tW = tensor((device half*)(B + p.b_offset),
                     dextents<int32_t,2>(p.K, p.N),
                     array<int,2>({1, p.b_row_stride}));
    matmul2d<matmul2d_descriptor(TM, TN, TK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto acc =
        mm.get_destination_cooperative_tensor<decltype(tA), decltype(tW),
                                              float>();
    for (int k0 = 0; k0 < p.K; k0 += TK) {
        auto a = tA.slice(k0, m0);
        auto w = tW.slice(k0, n0);
        mm.run(a, w, acc);
    }
    auto out = tensor(C + p.c_offset,
                      dextents<int32_t,2>(p.N, p.M),
                      array<int,2>({1, p.c_row_stride}));
    acc.store(out.slice(n0, m0));
}

// Small-output direct GEMM for MoE routers. A 128-column router launches only
// four threadgroups with the generic M128xN64 tile at S=256; M64xN32 exposes
// four times as much parallelism while keeping the same K32 FP32 accumulation.
kernel void gemm_tensor_router_f32a_f16b_f32c(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const int TN = 32, TM = 64, TK = 32;
    const int n0 = (int)tgpig.y * TN;
    const int m0 = (int)tgpig.x * TM;
    auto tA = tensor(
        (device float*)(A + p.a_offset),
        dextents<int32_t,2>(p.K, p.M),
        array<int,2>({1, p.a_row_stride}));
    auto tW = tensor(
        (device half*)(B + p.b_offset),
        dextents<int32_t,2>(p.K, p.N),
        array<int,2>({1, p.b_row_stride}));
    matmul2d<
        matmul2d_descriptor(
            TM, TN, TK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto acc =
        mm.get_destination_cooperative_tensor<
            decltype(tA), decltype(tW), float>();
    for (int k0 = 0; k0 < p.K; k0 += TK) {
        auto activation = tA.slice(k0, m0);
        auto weight = tW.slice(k0, n0);
        mm.run(activation, weight, acc);
    }
    auto output = tensor(
        C + p.c_offset,
        dextents<int32_t,2>(p.N, p.M),
        array<int,2>({1, p.c_row_stride}));
    acc.store(output.slice(n0, m0));
}

// W8A16 GEMM: identical to gemm_tensor but weights are int8 + per-group scale,
// dequantized to half during staging (sa[...] = half(int8 * scale)). The input
// activation matrix was cast to half once by matmul_cast_f32_to_f16. Handles
// any group size via the per-(row,group) scale lookup during staging.
kernel void gemm_tensor_w8_f16a_i8b_f32c(
    device const half*     A      [[buffer(0)]],
    device const int8_t*   B      [[buffer(1)]],
    device float*          C      [[buffer(2)]],
    device const float*    SCALES [[buffer(4)]],
    constant MatmulW8Params& p    [[buffer(3)]],
    threadgroup half*      shmem  [[threadgroup(0)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]])
{
    const int NRA = 64, NRB = 128, NK = 32, NUM_THREADS = 128;
    const int M = p.M, N = p.N, K = p.K;
    const int gpr = p.groups_per_row, gs = p.group_size;
    const int ra = (int)tgpig.y * NRA;
    const int rb = (int)tgpig.x * NRB;

    threadgroup half* sa = shmem;
    auto tA = tensor(sa, dextents<int32_t,2>(NK, NRA));
    device const half* ptrA = A + p.a_offset;
    auto tB = tensor((device half*)ptrA, dextents<int32_t,2>(K, M),
                     array<int,2>({1, p.a_row_stride}));
    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    const int UNROLL = 16;
    const int A_WORK = NRA * (NK / UNROLL);
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < A_WORK; work += NUM_THREADS) {
            int nl = work / (NK / UNROLL), sub = work % (NK / UNROLL);
            int kbase = sub * UNROLL, gn = ra + nl, gk0 = loop_k + kbase;
            threadgroup half* dst = sa + nl*NK + kbase;
            if (gn < N) {
                device const int8_t* wrow = B + (uint)gn * (uint)K + gk0;
                device const float*  srow = SCALES + (uint)gn * (uint)gpr;
                if (gpr == 1 && gk0 + UNROLL <= K) {
                    // W8PC common path: one channel scale and four contiguous
                    // vector loads cover the 16 staged weights.
                    const float sc = srow[0];
                    device const char4* src4 =
                        (device const char4*)wrow;
                    threadgroup half4* dst4 =
                        (threadgroup half4*)dst;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        dst4[i] = half4(float4(src4[i]) * sc);
                } else {
                    #pragma unroll
                    for (int i = 0; i < UNROLL; ++i) {
                        int k = gk0 + i;
                        dst[i] = (k < K)
                            ? (half)((float)wrow[i] * srow[k / gs])
                            : (half)0;
                    }
                }
            } else {
                #pragma unroll
                for (int i = 0; i < UNROLL; ++i) dst[i] = (half)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dstC = C + p.c_offset;
    auto tD = tensor(dstC, dextents<int32_t,2>(N, M), array<int,2>({1, p.c_row_stride}));
    cT.store(tD.slice(ra, rb));
}

// Smaller M tile for projection shapes where the M128 cooperative accumulator
// loses occupancy. Keep this as a separate entry point: making NRB a runtime
// branch changes cooperative-tensor register allocation for both paths.
kernel void gemm_tensor_w8_f16a_i8b_f32c_m64(
    device const half*     A      [[buffer(0)]],
    device const int8_t*   B      [[buffer(1)]],
    device float*          C      [[buffer(2)]],
    device const float*    SCALES [[buffer(4)]],
    constant MatmulW8Params& p    [[buffer(3)]],
    threadgroup half*      shmem  [[threadgroup(0)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]])
{
    const int NRA = 64, NRB = 64, NK = 32, NUM_THREADS = 128;
    const int M = p.M, N = p.N, K = p.K;
    const int gpr = p.groups_per_row, gs = p.group_size;
    const int ra = (int)tgpig.y * NRA;
    const int rb = (int)tgpig.x * NRB;

    threadgroup half* sa = shmem;
    auto tA = tensor(sa, dextents<int32_t,2>(NK, NRA));
    device const half* ptrA = A + p.a_offset;
    auto tB = tensor((device half*)ptrA, dextents<int32_t,2>(K, M),
                     array<int,2>({1, p.a_row_stride}));
    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto cT =
        mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    const int UNROLL = 16;
    const int A_WORK = NRA * (NK / UNROLL);
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < A_WORK; work += NUM_THREADS) {
            int nl = work / (NK / UNROLL), sub = work % (NK / UNROLL);
            int kbase = sub * UNROLL, gn = ra + nl, gk0 = loop_k + kbase;
            threadgroup half* dst = sa + nl*NK + kbase;
            if (gn < N) {
                device const int8_t* wrow = B + (uint)gn * (uint)K + gk0;
                device const float* srow =
                    SCALES + (uint)gn * (uint)gpr;
                if (gpr == 1 && gk0 + UNROLL <= K) {
                    const float sc = srow[0];
                    device const char4* src4 = (device const char4*)wrow;
                    threadgroup half4* dst4 = (threadgroup half4*)dst;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        dst4[i] = half4(float4(src4[i]) * sc);
                } else {
                    #pragma unroll
                    for (int i = 0; i < UNROLL; ++i) {
                        int k = gk0 + i;
                        dst[i] = (k < K)
                            ? (half)((float)wrow[i] * srow[k / gs])
                            : (half)0;
                    }
                }
            } else {
                #pragma unroll
                for (int i = 0; i < UNROLL; ++i) dst[i] = (half)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dstC = C + p.c_offset;
    auto tD = tensor(dstC, dextents<int32_t,2>(N, M),
                     array<int,2>({1, p.c_row_stride}));
    cT.store(tD.slice(ra, rb));
}

inline void stage_w4_16(
    device const uint8_t* packed_weights,
    threadgroup half* staged_weights,
    float scale)
{
    const uint2 raw =
        *((device const uint2*)packed_weights);
    const uint4 shifts0(0, 4, 8, 12);
    const uint4 shifts1(16, 20, 24, 28);
    const float4 scale4(scale);
    *((threadgroup half4*)(staged_weights + 0)) =
        half4(float4(
            int4((uint4(raw.x) >> shifts0) & 0x0f) - 8) *
            scale4);
    *((threadgroup half4*)(staged_weights + 4)) =
        half4(float4(
            int4((uint4(raw.x) >> shifts1) & 0x0f) - 8) *
            scale4);
    *((threadgroup half4*)(staged_weights + 8)) =
        half4(float4(
            int4((uint4(raw.y) >> shifts0) & 0x0f) - 8) *
            scale4);
    *((threadgroup half4*)(staged_weights + 12)) =
        half4(float4(
            int4((uint4(raw.y) >> shifts1) & 0x0f) - 8) *
            scale4);
}

constant bool FC_W4_A16_G128 [[function_constant(10)]];

// W4A16 GEMM: unpack per-group int4 weights to half while staging. Keep
// activations in FP32; casting them to FP16 measurably changes model PPL.
// Four adjacent K32 weight tiles share one staging/barrier pair.
kernel void gemm_tensor_w4_f32a_i4b_f32c(
    device const float* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const float* SCALES [[buffer(4)]],
    constant MatmulW8Params& p [[buffer(3)]],
    threadgroup half* shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]])
{
    const int NRA = 64, NRB = 128, NK = 32, NUM_THREADS = 128;
    const int M = p.M, N = p.N, K = p.K;
    const int gpr = p.groups_per_row, gs = p.group_size;
    const int scale_row_stride =
        FC_W4_A16_G128 ? K / 128 : gpr;
    const int ra = (int)tgpig.y * NRA;
    const int rb = (int)tgpig.x * NRB;

    threadgroup half* sa = shmem;
    auto tA0 = tensor(sa, dextents<int32_t,2>(NK, NRA));
    auto tA1 =
        tensor(sa + NRA * NK, dextents<int32_t,2>(NK, NRA));
    auto tA2 =
        tensor(sa + 2 * NRA * NK,
               dextents<int32_t,2>(NK, NRA));
    auto tA3 =
        tensor(sa + 3 * NRA * NK,
               dextents<int32_t,2>(NK, NRA));
    device const float* ptrA = A + p.a_offset;
    auto tB = tensor((device float*)ptrA, dextents<int32_t,2>(K, M),
                     array<int,2>({1, p.a_row_stride}));
    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto cT =
        mm.get_destination_cooperative_tensor<
            decltype(tB), decltype(tA0), float>();

    const int UNROLL = 16;
    const int A_WORK = NRA * (NK / UNROLL);
    const ulong row_bytes = (ulong)K / 2;
    for (int loop_k = 0; loop_k < K; loop_k += 4 * NK) {
        const bool full_g128 =
            FC_W4_A16_G128 ||
            (gs == 128 && loop_k + 4 * NK <= K);
        if (full_g128) {
            const int nl =
                (int)tiitg / (NK / UNROLL);
            const int sub =
                (int)tiitg % (NK / UNROLL);
            const int kbase = sub * UNROLL;
            const int gn = ra + nl;
            const float scale =
                gn < N
                    ? SCALES[
                          (ulong)gn *
                              (ulong)scale_row_stride +
                          (ulong)(
                              FC_W4_A16_G128
                                  ? loop_k / 128
                                  : loop_k / gs)]
                    : 0.0f;
            threadgroup half* dst =
                sa + nl * NK + kbase;
            if (gn < N) {
                device const uint8_t* src =
                    B + (ulong)gn * row_bytes +
                    (ulong)((loop_k + kbase) / 2);
                #pragma unroll
                for (int slice = 0; slice < 4; ++slice) {
                    stage_w4_16(
                        src + slice * (NK / 2),
                        dst + slice * NRA * NK, scale);
                }
            } else {
                #pragma unroll
                for (int slice = 0; slice < 4; ++slice) {
                    threadgroup half* slice_dst =
                        dst + slice * NRA * NK;
                    #pragma unroll
                    for (int i = 0; i < UNROLL; ++i)
                        slice_dst[i] = (half)0;
                }
            }
        } else {
            for (int work = tiitg; work < 4 * A_WORK;
                 work += NUM_THREADS) {
                const int slice = work / A_WORK;
                const int local = work - slice * A_WORK;
                const int nl =
                    local / (NK / UNROLL);
                const int sub =
                    local % (NK / UNROLL);
                const int kbase = sub * UNROLL;
                const int gn = ra + nl;
                const int gk0 =
                    loop_k + slice * NK + kbase;
                threadgroup half* dst =
                    sa + slice * NRA * NK +
                    nl * NK + kbase;
                if (gn < N) {
                    device const uint8_t* wr =
                        B + (ulong)gn * row_bytes +
                        (ulong)(gk0 / 2);
                    device const float* scales =
                        SCALES +
                        (ulong)gn * (ulong)gpr;
                    if (gk0 + UNROLL <= K &&
                        gs >= UNROLL &&
                        (gk0 % gs) + UNROLL <= gs) {
                        stage_w4_16(
                            wr, dst, scales[gk0 / gs]);
                    } else {
                        #pragma unroll
                        for (int i = 0;
                             i < UNROLL; i += 2) {
                            const int k = gk0 + i;
                            const uint8_t packed =
                                k < K ? wr[i / 2] : 0;
                            const int lo =
                                (packed & 0x0f) - 8;
                            const int hi =
                                (packed >> 4) - 8;
                            dst[i] = k < K
                                ? (half)(
                                      (float)lo *
                                      scales[k / gs])
                                : (half)0;
                            dst[i + 1] = k + 1 < K
                                ? (half)(
                                      (float)hi *
                                      scales[(k + 1) / gs])
                                : (half)0;
                        }
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < UNROLL; ++i)
                        dst[i] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA0 = tA0.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA0, cT);
        if (FC_W4_A16_G128 || loop_k + NK < K) {
            auto mA1 = tA1.slice(0, 0);
            auto mB1 = tB.slice(loop_k + NK, rb);
            mm.run(mB1, mA1, cT);
        }
        if (FC_W4_A16_G128 || loop_k + 2 * NK < K) {
            auto mA2 = tA2.slice(0, 0);
            auto mB2 = tB.slice(loop_k + 2 * NK, rb);
            mm.run(mB2, mA2, cT);
        }
        if (FC_W4_A16_G128 || loop_k + 3 * NK < K) {
            auto mA3 = tA3.slice(0, 0);
            auto mB3 = tB.slice(loop_k + 3 * NK, rb);
            mm.run(mB3, mA3, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dstC = C + p.c_offset;
    auto tD =
        tensor(dstC, dextents<int32_t,2>(N, M),
               array<int,2>({1, p.c_row_stride}));
    cT.store(tD.slice(ra, rb));
}

// Smaller output tile for W4 projection shapes where M128 loses occupancy.
// This remains an explicit entry point so each cooperative accumulator has a
// compile-time shape and independent register allocation.
kernel void gemm_tensor_w4_f32a_i4b_f32c_m64(
    device const float* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const float* SCALES [[buffer(4)]],
    constant MatmulW8Params& p [[buffer(3)]],
    threadgroup half* shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]])
{
    const int NRA = 64, NRB = 64, NK = 32, NUM_THREADS = 128;
    const int M = p.M, N = p.N, K = p.K;
    const int gpr = p.groups_per_row, gs = p.group_size;
    const int scale_row_stride =
        FC_W4_A16_G128 ? K / 128 : gpr;
    const int ra = (int)tgpig.y * NRA;
    const int rb = (int)tgpig.x * NRB;

    threadgroup half* sa = shmem;
    auto tA0 = tensor(sa, dextents<int32_t,2>(NK, NRA));
    auto tA1 =
        tensor(sa + NRA * NK, dextents<int32_t,2>(NK, NRA));
    auto tA2 =
        tensor(sa + 2 * NRA * NK,
               dextents<int32_t,2>(NK, NRA));
    auto tA3 =
        tensor(sa + 3 * NRA * NK,
               dextents<int32_t,2>(NK, NRA));
    device const float* ptrA = A + p.a_offset;
    auto tB = tensor((device float*)ptrA, dextents<int32_t,2>(K, M),
                     array<int,2>({1, p.a_row_stride}));
    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto cT =
        mm.get_destination_cooperative_tensor<
            decltype(tB), decltype(tA0), float>();

    const int UNROLL = 16;
    const int A_WORK = NRA * (NK / UNROLL);
    const ulong row_bytes = (ulong)K / 2;
    for (int loop_k = 0; loop_k < K; loop_k += 4 * NK) {
        const bool full_g128 =
            FC_W4_A16_G128 ||
            (gs == 128 && loop_k + 4 * NK <= K);
        if (full_g128) {
            const int nl =
                (int)tiitg / (NK / UNROLL);
            const int sub =
                (int)tiitg % (NK / UNROLL);
            const int kbase = sub * UNROLL;
            const int gn = ra + nl;
            const float scale =
                gn < N
                    ? SCALES[
                          (ulong)gn *
                              (ulong)scale_row_stride +
                          (ulong)(
                              FC_W4_A16_G128
                                  ? loop_k / 128
                                  : loop_k / gs)]
                    : 0.0f;
            threadgroup half* dst =
                sa + nl * NK + kbase;
            if (gn < N) {
                device const uint8_t* src =
                    B + (ulong)gn * row_bytes +
                    (ulong)((loop_k + kbase) / 2);
                #pragma unroll
                for (int slice = 0; slice < 4; ++slice) {
                    stage_w4_16(
                        src + slice * (NK / 2),
                        dst + slice * NRA * NK, scale);
                }
            } else {
                #pragma unroll
                for (int slice = 0; slice < 4; ++slice) {
                    threadgroup half* slice_dst =
                        dst + slice * NRA * NK;
                    #pragma unroll
                    for (int i = 0; i < UNROLL; ++i)
                        slice_dst[i] = (half)0;
                }
            }
        } else {
            for (int work = tiitg; work < 4 * A_WORK;
                 work += NUM_THREADS) {
                const int slice = work / A_WORK;
                const int local = work - slice * A_WORK;
                const int nl =
                    local / (NK / UNROLL);
                const int sub =
                    local % (NK / UNROLL);
                const int kbase = sub * UNROLL;
                const int gn = ra + nl;
                const int gk0 =
                    loop_k + slice * NK + kbase;
                threadgroup half* dst =
                    sa + slice * NRA * NK +
                    nl * NK + kbase;
                if (gn < N) {
                    device const uint8_t* wr =
                        B + (ulong)gn * row_bytes +
                        (ulong)(gk0 / 2);
                    device const float* scales =
                        SCALES +
                        (ulong)gn * (ulong)gpr;
                    if (gk0 + UNROLL <= K &&
                        gs >= UNROLL &&
                        (gk0 % gs) + UNROLL <= gs) {
                        stage_w4_16(
                            wr, dst, scales[gk0 / gs]);
                    } else {
                        #pragma unroll
                        for (int i = 0;
                             i < UNROLL; i += 2) {
                            const int k = gk0 + i;
                            const uint8_t packed =
                                k < K ? wr[i / 2] : 0;
                            const int lo =
                                (packed & 0x0f) - 8;
                            const int hi =
                                (packed >> 4) - 8;
                            dst[i] = k < K
                                ? (half)(
                                      (float)lo *
                                      scales[k / gs])
                                : (half)0;
                            dst[i + 1] = k + 1 < K
                                ? (half)(
                                      (float)hi *
                                      scales[(k + 1) / gs])
                                : (half)0;
                        }
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < UNROLL; ++i)
                        dst[i] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA0 = tA0.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA0, cT);
        if (FC_W4_A16_G128 || loop_k + NK < K) {
            auto mA1 = tA1.slice(0, 0);
            auto mB1 = tB.slice(loop_k + NK, rb);
            mm.run(mB1, mA1, cT);
        }
        if (FC_W4_A16_G128 || loop_k + 2 * NK < K) {
            auto mA2 = tA2.slice(0, 0);
            auto mB2 = tB.slice(loop_k + 2 * NK, rb);
            mm.run(mB2, mA2, cT);
        }
        if (FC_W4_A16_G128 || loop_k + 3 * NK < K) {
            auto mA3 = tA3.slice(0, 0);
            auto mB3 = tB.slice(loop_k + 3 * NK, rb);
            mm.run(mB3, mA3, cT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dstC = C + p.c_offset;
    auto tD =
        tensor(dstC, dextents<int32_t,2>(N, M),
               array<int,2>({1, p.c_row_stride}));
    cT.store(tD.slice(ra, rb));
}

// Post-pass for tensor W8 GEMM, whose cooperative tensor store cannot apply a
// per-column fused activation. Operates in place on row-strided C.
kernel void matmul_w8_activation_range_f32(
    device float* C [[buffer(2)]],
    constant MatmulW8Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.M * p.N) return;
    int m = int(gid) / p.N, n = int(gid) % p.N;
    if (p.act_n_len == 0 ||
        (p.act_n_len > 0 && (n < p.act_n_begin || n >= p.act_n_begin + p.act_n_len))) return;
    uint idx = p.c_offset + (uint)m * (uint)p.c_row_stride + (uint)n;
    C[idx] = apply_activation(C[idx], p.activation);
}

// Post-pass for FP16 GEMM tensor/tiled paths. Keeping activation separate lets
// fused graph nodes use the high-throughput matrix kernels instead of falling
// back to one scalar thread per output element.
kernel void matmul_activation_range_f32(
    device float* C [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.M * p.N) return;
    int m = int(gid) / p.N, n = int(gid) % p.N;
    if (p.act_n_len == 0 ||
        (p.act_n_len > 0 &&
         (n < p.act_n_begin || n >= p.act_n_begin + p.act_n_len))) return;
    uint idx = p.c_offset + (uint)m * (uint)p.c_row_stride + (uint)n;
    C[idx] = apply_activation(C[idx], p.activation);
}

// --- W8A8 int8xint8->int32 GEMM -------------------------------------------
// C[M,N] = A_i8[M,K] * W_i8[N,K]^T, then dequant: out = int32*scale_a[m]*scale_w[n].
// Same tiling as the fp16 gemm_tensor: weights staged into threadgroup as int8,
// activations read as a device tensor, matmul2d accumulates in int32. The int32
// tile is stored to threadgroup, then threads write the dequantized fp32 output.
// Manual per-thread fragment packing is intentionally avoided: the cooperative
// tensor register layout is compiler-dependent and only the tensor .load/.store
// path is portable across the offline (metallib) and online compilers.
kernel void gemm_w8a8_i8a_i8b_f32c(
    device const int8_t*  A       [[buffer(0)]],   // int8 activations [M,K]
    device const int8_t*  B       [[buffer(1)]],   // int8 weights [N,K]
    device float*         C       [[buffer(2)]],   // fp32 out [M,N] ([N,M]-strided)
    device const float*   SCALE_A [[buffer(4)]],   // per-(token, weight group)
    device const float*   SCALE_W [[buffer(5)]],   // per-channel [N]
    constant MatmulW8A8Params& p  [[buffer(3)]],
    threadgroup int8_t*   shmem   [[threadgroup(0)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]])
{
    const int NRA = 64;    // weights tile (our N), staged into threadgroup
    const int NRB = 64;    // activations tile (our M)
    const int NK  = 32;    // K chunk
    const int NUM_THREADS = 128;
    const int M = p.M, N = p.N, K = p.K;
    const int ra = (int)tgpig.y * NRA;   // first weight row (our N)
    const int rb = (int)tgpig.x * NRB;   // first activation row (our M)

    threadgroup int8_t* sa = shmem;      // staged weights [NK, NRA]
    auto tA = tensor(sa, dextents<int32_t,2>(NK, NRA));
    // A as device tensor [K, M]: element (m,k) at m*K + k -> strides {1, K}.
    auto tB = tensor((device int8_t*)A, dextents<int32_t,2>(K, M),
                     array<int,2>({1, K}));

    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), int32_t>();

    const int UNROLL = 16;
    const int A_WORK = NRA * (NK / UNROLL);
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < A_WORK; work += NUM_THREADS) {
            int nl = work / (NK / UNROLL), sub = work % (NK / UNROLL);
            int kbase = sub * UNROLL, gn = ra + nl, gk0 = loop_k + kbase;
            threadgroup int8_t* dst = sa + nl*NK + kbase;
            if (gn < N) {
                device const int8_t* wrow = B + (ulong)gn * (ulong)K + gk0;
                #pragma unroll
                for (int i = 0; i < UNROLL; ++i)
                    dst[i] = (gk0 + i < K) ? wrow[i] : (int8_t)0;
            } else {
                #pragma unroll
                for (int i = 0; i < UNROLL; ++i) dst[i] = (int8_t)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store int32 accumulators to threadgroup [NRA(N), NRB(M)], then dequant.
    threadgroup int32_t* si = (threadgroup int32_t*)shmem;   // reuse shmem (NRA*NRB int32)
    auto tI = tensor(si, dextents<int32_t,2>(NRB, NRA));      // [M-tile, N-tile]
    cT.store(tI);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* Cout = C + p.c_offset;
    for (int idx = tiitg; idx < NRA * NRB; idx += NUM_THREADS) {
        int nl = idx / NRB;          // weight/N within tile
        int ml = idx % NRB;          // activation/M within tile
        int ni = ra + nl, mi = rb + ml;
        if (mi < M && ni < N) {
            int32_t acc = si[ml * NRA + nl];   // tI is [NRB, NRA] row-major
            float v = (float)acc * SCALE_A[mi] * SCALE_W[ni];
            Cout[mi * p.c_row_stride + ni] = apply_activation_at(
                v, p.activation, ni, p.act_n_begin, p.act_n_len);
        }
    }
}

// --- W4A8 int8 x per-group int4 GEMM --------------------------------------
// C[M,N] = A_i8[M,K] * W_i4[N,K/2], per-group symmetric int4 weights. Same
// int8xint8->int32 matmul2d as W8A8, but weights are unpacked from nibbles at
// staging and the accumulator is flushed per weight-scale group. Weight nibble:
// byte = W[n*(K/2)+k/2]; low nibble = even k, high = odd k; w = nibble-16 if >=8.
// Group-major: each group (group_size K) accumulates into its own int32 tile,
// then flushes (int32 * scale_w[n,group]) into an fp32 tile; final *scale_a[m].
kernel void gemm_w4a8_i8a_i4b_f32c(
    device const int8_t*  A       [[buffer(0)]],   // int8 activations [M,K]
    device const uint8_t* B       [[buffer(1)]],   // int4 weights [N,K/2]
    device float*         C       [[buffer(2)]],   // fp32 out [M,N] ([N,M]-strided)
    device const float*   SCALE_A [[buffer(4)]],   // per-token [M]
    device const float*   SCALE_W [[buffer(5)]],   // per-group [N, groups_per_row]
    constant MatmulW4A8Params& p  [[buffer(3)]],
    threadgroup int8_t*   shmem   [[threadgroup(0)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]])
{
    const int NRA = 16, NRB = 64, NK = 32, NUM_THREADS = 128;
    const int M = p.M, N = p.N, K = p.K;
    const int GS = p.group_size, GPR = p.groups_per_row;
    const int ra = (int)tgpig.y * NRA;   // first weight row (our N)
    const int rb = (int)tgpig.x * NRB;   // first activation row (our M)

    // threadgroup: [ facc: NRB*NRA float (persists across groups) | scratch:
    // staged int8 weights / int32 store tile (reused, never simultaneously) ].
    threadgroup float*   facc = (threadgroup float*)shmem;
    threadgroup int8_t*  sa   = (threadgroup int8_t*)(shmem + NRA*NRB*sizeof(float));
    threadgroup int32_t* si   = (threadgroup int32_t*)sa;
    for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) facc[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tA = tensor(sa, dextents<int32_t,2>(NK, NRA));
    auto tB = tensor((device int8_t*)A, dextents<int32_t,2>(K, M), array<int,2>({1, K}));
    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;

    const int UNROLL = 16, A_WORK = NRA * (NK / UNROLL);
    for (int g0 = 0; g0 < K; g0 += GS) {
        int g = g0 / GS;
        int gend = min(g0 + GS, K);
        auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), int32_t>();
        for (int loop_k = g0; loop_k < gend; loop_k += NK) {
            for (int work = tiitg; work < A_WORK; work += NUM_THREADS) {
                int nl = work / (NK / UNROLL), sub = work % (NK / UNROLL);
                int kbase = sub * UNROLL, gn = ra + nl, gk0 = loop_k + kbase;
                threadgroup int8_t* dst = sa + nl*NK + kbase;
                if (gn < N) {
                    device const uint8_t* wr = B + (ulong)gn * ((ulong)K/2) + (gk0/2);
                    #pragma unroll
                    for (int i = 0; i < UNROLL; i += 2) {
                        uint8_t byte = ((gk0 + i) < K) ? wr[i/2] : 0;
                        int lo = (byte & 0x0F) - 8;
                        int hi = (byte >> 4) - 8;
                        dst[i]   = (int8_t)(((gk0 + i)   < K) ? lo : 0);
                        dst[i+1] = (int8_t)(((gk0 + i+1) < K) ? hi : 0);
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < UNROLL; ++i) dst[i] = 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto mA = tA.slice(0, 0);
            auto mB = tB.slice(loop_k, rb);
            mm.run(mB, mA, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        cT.store(tensor(si, dextents<int32_t,2>(NRA, NRB)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) {
            int ml = idx / NRA, nl = idx % NRA, ni = ra + nl;
            int mi = rb + ml;
            if (mi < M && ni < N)
                facc[idx] +=
                    (float)si[idx] *
                    SCALE_A[(ulong)mi * (ulong)GPR + (ulong)g] *
                    SCALE_W[(ulong)ni * (ulong)GPR + (ulong)g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* Cout = C + p.c_offset;
    for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) {
        int ml = idx / NRA, nl = idx % NRA;
        int ni = ra + nl, mi = rb + ml;
        if (mi < M && ni < N) {
            Cout[mi * p.c_row_stride + ni] = apply_activation_at(
                facc[idx], p.activation, ni,
                p.act_n_begin, p.act_n_len);
        }
    }
}

// K128 W4 path reading the package-native Q4B8G128 layout directly. One block
// stores scales[8] followed by q[4][8][16], so the 8 output channels consumed
// together by the kernel are contiguous instead of K/2 bytes apart.
kernel void gemm_w4a8_bg128_i8a_i4b_f32c(
    device const int8_t*  A       [[buffer(0)]],
    device const uint8_t* B       [[buffer(1)]],
    device float*         C       [[buffer(2)]],
    device const float*   SCALE_A [[buffer(4)]],
    constant MatmulW4A8Params& p  [[buffer(3)]],
    threadgroup int8_t*   shmem   [[threadgroup(0)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]])
{
    const int NRA = 16, NRB = 64, NK = 32, NUM_THREADS = 128;
    const int BLOCK_BYTES = 544, SCALE_BYTES = 32;
    const int M = p.M, N = p.N, K = p.K;
    const int GS = p.group_size, GPR = p.groups_per_row;
    const int ra = (int)tgpig.y * NRA;
    const int rb = (int)tgpig.x * NRB;

    threadgroup float* facc = (threadgroup float*)shmem;
    threadgroup int8_t* sw =
        (threadgroup int8_t*)(shmem + NRA*NRB*sizeof(float));
    threadgroup int32_t* si = (threadgroup int32_t*)sw;
    for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS)
        facc[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tW = tensor(sw, dextents<int32_t,2>(NK, NRA));
    auto tA = tensor((device int8_t*)A, dextents<int32_t,2>(K, M),
                     array<int,2>({1, K}));
    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;

    const int UNROLL = 16, W_WORK = NRA * (NK / UNROLL);
    for (int g0 = 0; g0 < K; g0 += GS) {
        const int g = g0 / GS;
        const int gend = min(g0 + GS, K);
        auto dotT =
            mm.get_destination_cooperative_tensor<decltype(tA), decltype(tW),
                                                  int32_t>();
        for (int k0 = g0; k0 < gend; k0 += NK) {
            const int qgi = (k0 - g0) / NK;
            for (int work = tiitg; work < W_WORK; work += NUM_THREADS) {
                const int nl = work / (NK / UNROLL);
                const int sub = work % (NK / UNROLL);
                const int gn = ra + nl;
                threadgroup int8_t* dst =
                    sw + nl*NK + sub*UNROLL;
                if (gn < N) {
                    const ulong block =
                        ((ulong)(gn / 8) * (ulong)GPR + (ulong)g) *
                        (ulong)BLOCK_BYTES;
                    device const ulong* src =
                        (device const ulong*)(B + block + SCALE_BYTES +
                            qgi*8*16 + (gn & 7)*16 + sub*8);
                    const ulong packed8 = *src;
                    #pragma unroll
                    for (int i = 0; i < UNROLL; i += 2) {
                        const uint nibs =
                            (uint)((packed8 >> (4*i)) & 0xff);
                        const int lo4 = (int)(nibs & 0x0f);
                        const int hi4 = (int)(nibs >> 4);
                        dst[i] = (int8_t)(lo4 >= 8 ? lo4 - 16 : lo4);
                        dst[i+1] = (int8_t)(hi4 >= 8 ? hi4 - 16 : hi4);
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < UNROLL; ++i) dst[i] = 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto mA = tA.slice(k0, rb);
            auto mW = tW.slice(0, 0);
            mm.run(mA, mW, dotT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        dotT.store(tensor(si, dextents<int32_t,2>(NRA, NRB)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) {
            const int ml = idx / NRA, nl = idx % NRA;
            const int ni = ra + nl, mi = rb + ml;
            if (mi < M && ni < N) {
                const ulong block =
                    ((ulong)(ni / 8) * (ulong)GPR + (ulong)g) *
                    (ulong)BLOCK_BYTES;
                device const float* scales =
                    (device const float*)(B + block);
                facc[idx] +=
                    (float)si[idx] *
                    SCALE_A[(ulong)mi * (ulong)GPR + (ulong)g] *
                    scales[ni & 7];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* out = C + p.c_offset;
    for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) {
        const int ml = idx / NRA, nl = idx % NRA;
        const int ni = ra + nl, mi = rb + ml;
        if (mi < M && ni < N)
            out[mi * p.c_row_stride + ni] = apply_activation_at(
                facc[idx], p.activation, ni,
                p.act_n_begin, p.act_n_len);
    }
}

// CPU-equivalent W4 prefill path. Activations use an independent Q8 scale for
// every 32-element K block (the same quantization granularity as
// quantize_a_q8_blocks_a4 on the CPU). Each tensor-MMA accumulation therefore
// covers exactly one activation-scale block before it is converted to fp32 and
// multiplied by the activation and weight scales.
kernel void gemm_w4a8_block32_i8a_i4b_f32c(
    device const int8_t*  A       [[buffer(0)]],
    device const uint8_t* B       [[buffer(1)]],
    device float*         C       [[buffer(2)]],
    device const float*   SCALE_A [[buffer(4)]], // [M, ceil(K/32)]
    device const float*   SCALE_W [[buffer(5)]],
    constant MatmulW4A8Params& p  [[buffer(3)]],
    threadgroup int8_t*   shmem   [[threadgroup(0)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]])
{
    const int NRA = 16, NRB = 64, NK = 32, NUM_THREADS = 128;
    const int M = p.M, N = p.N, K = p.K;
    const int GS = p.group_size, GPR = p.groups_per_row;
    const int ABS = 32;
    const int ABPR = (K + ABS - 1) / ABS;
    const int ra = (int)tgpig.y * NRA;
    const int rb = (int)tgpig.x * NRB;

    threadgroup float*   facc = (threadgroup float*)shmem;
    threadgroup int8_t*  sa =
        (threadgroup int8_t*)(shmem + NRA*NRB*sizeof(float));
    threadgroup int32_t* si =
        (threadgroup int32_t*)(sa + NRA*NK);
    for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS)
        facc[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tA = tensor(sa, dextents<int32_t,2>(NK, NRA));
    auto tB = tensor((device int8_t*)A, dextents<int32_t,2>(K, M),
                     array<int,2>({1, K}));
    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;

    const int UNROLL = 16, A_WORK = NRA * (NK / UNROLL);
    for (int ab0 = 0; ab0 < K; ab0 += ABS) {
        auto cT =
            mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA),
                                                  int32_t>();
        const int ab_end = min(ab0 + ABS, K);
        for (int loop_k = ab0; loop_k < ab_end; loop_k += NK) {
            for (int work = tiitg; work < A_WORK; work += NUM_THREADS) {
                int nl = work / (NK / UNROLL), sub = work % (NK / UNROLL);
                int kbase = sub * UNROLL, gn = ra + nl;
                int gk0 = loop_k + kbase;
                threadgroup int8_t* dst = sa + nl*NK + kbase;
                if (gn < N) {
                    device const uint8_t* wr =
                        B + (ulong)gn * ((ulong)K/2) + (gk0/2);
                    const ulong packed8 = *((device const ulong*)wr);
                    #pragma unroll
                    for (int i = 0; i < UNROLL; i += 2) {
                        uint8_t byte =
                            (uint8_t)(packed8 >> (4 * i));
                        dst[i] = (int8_t)(((gk0 + i) < K)
                                             ? (int(byte & 0x0f) - 8) : 0);
                        dst[i+1] = (int8_t)(((gk0 + i + 1) < K)
                                               ? (int(byte >> 4) - 8) : 0);
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < UNROLL; ++i) dst[i] = 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto mA = tA.slice(0, 0);
            auto mB = tB.slice(loop_k, rb);
            mm.run(mB, mA, cT);
        }
        cT.store(tensor(si, dextents<int32_t,2>(NRA, NRB)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int wg = ab0 / GS;
        for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) {
            int ml = idx / NRA, nl = idx % NRA;
            int ni = ra + nl, mi = rb + ml;
            if (mi < M && ni < N) {
                facc[idx] +=
                    (float)si[idx] *
                    SCALE_A[(ulong)mi * (ulong)ABPR +
                            (ulong)(ab0 / ABS)] *
                    SCALE_W[(ulong)ni * (ulong)GPR + (ulong)wg];
            }
        }
    }

    device float* Cout = C + p.c_offset;
    for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) {
        int ml = idx / NRA, nl = idx % NRA;
        int ni = ra + nl, mi = rb + ml;
        if (mi < M && ni < N) {
            Cout[mi * p.c_row_stride + ni] = apply_activation_at(
                facc[idx], p.activation, ni,
                p.act_n_begin, p.act_n_len);
        }
    }
}

// Exact K32 activation-quantized path over package-native BG128 weights.
kernel void gemm_w4a8_block32_bg128_i8a_i4b_f32c(
    device const int8_t*  A       [[buffer(0)]],
    device const uint8_t* B       [[buffer(1)]],
    device float*         C       [[buffer(2)]],
    device const float*   SCALE_A [[buffer(4)]],
    constant MatmulW4A8Params& p  [[buffer(3)]],
    threadgroup int8_t*   shmem   [[threadgroup(0)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]])
{
    const int NRA = 16, NRB = 64, NK = 32, NUM_THREADS = 128;
    const int BLOCK_BYTES = 544, SCALE_BYTES = 32;
    const int M = p.M, N = p.N, K = p.K;
    const int GS = p.group_size, GPR = p.groups_per_row;
    const int ABPR = (K + NK - 1) / NK;
    const int ra = (int)tgpig.y * NRA;
    const int rb = (int)tgpig.x * NRB;

    threadgroup float* facc = (threadgroup float*)shmem;
    threadgroup int8_t* sw =
        (threadgroup int8_t*)(shmem + NRA*NRB*sizeof(float));
    threadgroup int32_t* si =
        (threadgroup int32_t*)(sw + NRA*NK);
    for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS)
        facc[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tW = tensor(sw, dextents<int32_t,2>(NK, NRA));
    auto tA = tensor((device int8_t*)A, dextents<int32_t,2>(K, M),
                     array<int,2>({1, K}));
    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;

    const int UNROLL = 16, W_WORK = NRA * (NK / UNROLL);
    for (int k0 = 0, ab = 0; k0 < K; k0 += NK, ++ab) {
        const int g = k0 / GS;
        const int qgi = (k0 % GS) / NK;
        for (int work = tiitg; work < W_WORK; work += NUM_THREADS) {
            const int nl = work / (NK / UNROLL);
            const int sub = work % (NK / UNROLL);
            const int gn = ra + nl;
            threadgroup int8_t* dst =
                sw + nl*NK + sub*UNROLL;
            if (gn < N) {
                const ulong block =
                    ((ulong)(gn / 8) * (ulong)GPR + (ulong)g) *
                    (ulong)BLOCK_BYTES;
                device const ulong* src =
                    (device const ulong*)(B + block + SCALE_BYTES +
                        qgi*8*16 + (gn & 7)*16 + sub*8);
                const ulong packed8 = *src;
                #pragma unroll
                for (int i = 0; i < UNROLL; i += 2) {
                    const uint nibs =
                        (uint)((packed8 >> (4*i)) & 0xff);
                    const int lo4 = (int)(nibs & 0x0f);
                    const int hi4 = (int)(nibs >> 4);
                    dst[i] = (int8_t)(lo4 >= 8 ? lo4 - 16 : lo4);
                    dst[i+1] = (int8_t)(hi4 >= 8 ? hi4 - 16 : hi4);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < UNROLL; ++i) dst[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto dotT =
            mm.get_destination_cooperative_tensor<decltype(tA), decltype(tW),
                                                  int32_t>();
        auto mA = tA.slice(k0, rb);
        auto mW = tW.slice(0, 0);
        mm.run(mA, mW, dotT);
        dotT.store(tensor(si, dextents<int32_t,2>(NRA, NRB)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) {
            const int ml = idx / NRA, nl = idx % NRA;
            const int ni = ra + nl, mi = rb + ml;
            if (mi < M && ni < N) {
                const ulong block =
                    ((ulong)(ni / 8) * (ulong)GPR + (ulong)g) *
                    (ulong)BLOCK_BYTES;
                device const float* scales =
                    (device const float*)(B + block);
                facc[idx] +=
                    (float)si[idx] *
                    SCALE_A[(ulong)mi * (ulong)ABPR + (ulong)ab] *
                    scales[ni & 7];
            }
        }
    }

    device float* out = C + p.c_offset;
    for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) {
        const int ml = idx / NRA, nl = idx % NRA;
        const int ni = ra + nl, mi = rb + ml;
        if (mi < M && ni < N)
            out[mi * p.c_row_stride + ni] = apply_activation_at(
                facc[idx], p.activation, ni,
                p.act_n_begin, p.act_n_len);
    }
}

// Small-K K64 path. Keeping FP32 partials in threadgroup memory avoids the
// register pressure of the large-K specialization on K=1024 models.
kernel void gemm_w4a8_block64_bg128_smallk_i8a_i4b_f32c(
    device const int8_t*  A       [[buffer(0)]],
    device const uint8_t* B       [[buffer(1)]],
    device float*         C       [[buffer(2)]],
    device const float*   SCALE_A [[buffer(4)]],
    constant MatmulW4A8Params& p  [[buffer(3)]],
    threadgroup int8_t*   shmem   [[threadgroup(0)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]])
{
    const int NRA = 16, NRB = 64, NK = 32, ABS = 64, NUM_THREADS = 128;
    const int BLOCK_BYTES = 544, SCALE_BYTES = 32;
    const int M = p.M, N = p.N, K = p.K;
    const int GS = p.group_size, GPR = p.groups_per_row;
    const int ABPR = (K + ABS - 1) / ABS;
    const int ra = (int)tgpig.y * NRA;
    const int rb = (int)tgpig.x * NRB;

    threadgroup float* facc = (threadgroup float*)shmem;
    threadgroup int8_t* sw =
        shmem + NRA*NRB*sizeof(float);
    threadgroup int32_t* si =
        (threadgroup int32_t*)(sw + NRA*ABS);
    for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS)
        facc[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tW0 = tensor(sw, dextents<int32_t,2>(NK, NRA));
    auto tW1 = tensor(sw + NRA*NK, dextents<int32_t,2>(NK, NRA));
    auto tA = tensor((device int8_t*)A, dextents<int32_t,2>(K, M),
                     array<int,2>({1, K}));
    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;

    const int UNROLL = 16, W_WORK = NRA * (NK / UNROLL);
    for (int ab0 = 0, ab = 0; ab0 < K; ab0 += ABS, ++ab) {
        for (int work = tiitg; work < 2*W_WORK; work += NUM_THREADS) {
            const int half_idx = work / W_WORK;
            const int local = work - half_idx*W_WORK;
            const int nl = local / (NK / UNROLL);
            const int sub = local % (NK / UNROLL);
            const int k0 = ab0 + half_idx*NK;
            const int g = k0 / GS;
            const int qgi = (k0 % GS) / NK;
            const int gn = ra + nl;
            threadgroup int8_t* dst =
                sw + half_idx*NRA*NK + nl*NK + sub*UNROLL;
            if (gn < N && k0 < K) {
                const ulong block =
                    ((ulong)(gn / 8) * (ulong)GPR + (ulong)g) *
                    (ulong)BLOCK_BYTES;
                device const ulong* src =
                    (device const ulong*)(B + block + SCALE_BYTES +
                        qgi*8*16 + (gn & 7)*16 + sub*8);
                const ulong packed8 = *src;
                #pragma unroll
                for (int i = 0; i < UNROLL; i += 2) {
                    const uint nibs =
                        (uint)((packed8 >> (4*i)) & 0xff);
                    const int lo4 = (int)(nibs & 0x0f);
                    const int hi4 = (int)(nibs >> 4);
                    dst[i] = (int8_t)(lo4 >= 8 ? lo4 - 16 : lo4);
                    dst[i+1] = (int8_t)(hi4 >= 8 ? hi4 - 16 : hi4);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < UNROLL; ++i) dst[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto dotT =
            mm.get_destination_cooperative_tensor<
                decltype(tA), decltype(tW0), int32_t>();
        auto mA0 = tA.slice(ab0, rb);
        auto mW0 = tW0.slice(0, 0);
        mm.run(mA0, mW0, dotT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ab0 + NK < K) {
            auto mA1 = tA.slice(ab0 + NK, rb);
            auto mW1 = tW1.slice(0, 0);
            mm.run(mA1, mW1, dotT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        dotT.store(tensor(si, dextents<int32_t,2>(NRA, NRB)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = ab0 / GS;
        for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) {
            const int ml = idx / NRA, nl = idx % NRA;
            const int ni = ra + nl, mi = rb + ml;
            if (mi < M && ni < N) {
                const ulong block =
                    ((ulong)(ni / 8) * (ulong)GPR + (ulong)g) *
                    (ulong)BLOCK_BYTES;
                device const float* scales =
                    (device const float*)(B + block);
                facc[idx] +=
                    (float)si[idx] *
                    SCALE_A[(ulong)mi * (ulong)ABPR + (ulong)ab] *
                    scales[ni & 7];
            }
        }
    }

    device float* out = C + p.c_offset;
    for (int idx = tiitg; idx < NRA*NRB; idx += NUM_THREADS) {
        const int ml = idx / NRA, nl = idx % NRA;
        const int ni = ra + nl, mi = rb + ml;
        if (mi < M && ni < N)
            out[mi * p.c_row_stride + ni] = apply_activation_at(
                facc[idx], p.activation, ni,
                p.act_n_begin, p.act_n_len);
    }
}

// Large-K K64 path over package-native BG128 weights. Keep weights packed in
// threadgroup memory and consume the cooperative accumulator in registers:
// BG128 already stores signed int4 nibbles in the layout expected by
// int4b_format, so unpacking to int8 and round-tripping the int32 tile through
// threadgroup memory only wastes bandwidth and occupancy.
kernel void gemm_w4a8_block64_bg128_i8a_i4b_f32c(
    device const int8_t*  A       [[buffer(0)]],
    device const uint8_t* B       [[buffer(1)]],
    device float*         C       [[buffer(2)]],
    device const float*   SCALE_A [[buffer(4)]],
    constant MatmulW4A8Params& p  [[buffer(3)]],
    threadgroup int8_t*   shmem   [[threadgroup(0)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]])
{
    const int NRA = 16, NRB = 64, NK = 32, ABS = 64, NUM_THREADS = 128;
    const int BLOCK_BYTES = 544, SCALE_BYTES = 32;
    const int M = p.M, N = p.N, K = p.K;
    const int GS = p.group_size, GPR = p.groups_per_row;
    const int ABPR = (K + ABS - 1) / ABS;
    const int ra = (int)tgpig.y * NRA;
    const int rb = (int)tgpig.x * NRB;

    constexpr int PACKED_WEIGHT_BYTES = NRA * ABS / 2;
    threadgroup uchar* sw = (threadgroup uchar*)shmem;
    threadgroup float* weight_scales =
        (threadgroup float*)(sw + PACKED_WEIGHT_BYTES);
    threadgroup float* activation_scales = weight_scales + NRA;
    constexpr int ACC_PER_THREAD =
        (NRA * NRB + NUM_THREADS - 1) / NUM_THREADS;
    float facc[ACC_PER_THREAD];
    #pragma unroll
    for (int slot = 0; slot < ACC_PER_THREAD; ++slot)
        facc[slot] = 0.0f;

    auto tW0 =
        tensor<threadgroup metal::int4b_format,
               dextents<int32_t,2>, tensor_inline>(
            sw, dextents<int32_t,2>(NK, NRA));
    auto tW1 =
        tensor<threadgroup metal::int4b_format,
               dextents<int32_t,2>, tensor_inline>(
            sw + NRA * NK / 2,
            dextents<int32_t,2>(NK, NRA));
    auto tA = tensor((device int8_t*)A, dextents<int32_t,2>(K, M),
                     array<int,2>({1, K}));
    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;

    constexpr int W_ROW_WORK = NRA;
    for (int ab0 = 0, ab = 0; ab0 < K; ab0 += ABS, ++ab) {
        const int scale_group = ab0 / GS;
        // Each output row contributes 16 packed bytes to each K32 slice.
        for (int work = tiitg;
             work < 2 * W_ROW_WORK;
             work += NUM_THREADS) {
            const int half_idx = work / W_ROW_WORK;
            const int nl = work - half_idx * W_ROW_WORK;
            const int k0 = ab0 + half_idx*NK;
            const int g = k0 / GS;
            const int qgi = (k0 % GS) / NK;
            const int gn = ra + nl;
            threadgroup ulong* dst =
                (threadgroup ulong*)(
                    sw + half_idx * NRA * (NK / 2) +
                    nl * (NK / 2));
            if (gn < N && k0 < K) {
                const ulong block =
                    ((ulong)(gn / 8) * (ulong)GPR + (ulong)g) *
                    (ulong)BLOCK_BYTES;
                device const ulong* src =
                    (device const ulong*)(B + block + SCALE_BYTES +
                        qgi*8*16 + (gn & 7)*16);
                dst[0] = src[0];
                dst[1] = src[1];
            } else {
                dst[0] = 0;
                dst[1] = 0;
            }
        }
        if (tiitg < NRA) {
            const int ni = ra + int(tiitg);
            if (ni < N) {
                const ulong block =
                    ((ulong)(ni / 8) * (ulong)GPR +
                     (ulong)scale_group) *
                    (ulong)BLOCK_BYTES;
                device const float* scales =
                    (device const float*)(B + block);
                weight_scales[tiitg] = scales[ni & 7];
            } else {
                weight_scales[tiitg] = 0.0f;
            }
        }
        if (tiitg < NRB) {
            const int mi = rb + int(tiitg);
            activation_scales[tiitg] =
                mi < M
                    ? SCALE_A[
                          (ulong)mi * (ulong)ABPR +
                          (ulong)ab]
                    : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto dotT =
            mm.get_destination_cooperative_tensor<
                decltype(tA), decltype(tW0), int32_t>();
        auto mA0 = tA.slice(ab0, rb);
        auto mW0 = tW0.slice(0, 0);
        mm.run(mA0, mW0, dotT);
        if (ab0 + NK < K) {
            auto mA1 = tA.slice(ab0 + NK, rb);
            auto mW1 = tW1.slice(0, 0);
            mm.run(mA1, mW1, dotT);
        }
        #pragma unroll
        for (int slot = 0; slot < ACC_PER_THREAD; ++slot) {
            const auto index =
                dotT.get_multidimensional_index(slot);
            const int nl = index[0];
            const int ml = index[1];
            facc[slot] +=
                (float)dotT[slot] *
                activation_scales[ml] *
                weight_scales[nl];
        }
        if (ab0 + ABS >= K) {
            device float* out = C + p.c_offset;
            #pragma unroll
            for (int slot = 0; slot < ACC_PER_THREAD; ++slot) {
                const auto index =
                    dotT.get_multidimensional_index(slot);
                const int nl = index[0];
                const int ml = index[1];
                const int ni = ra + nl;
                const int mi = rb + ml;
                if (mi < M && ni < N)
                    out[mi * p.c_row_stride + ni] =
                        apply_activation_at(
                            facc[slot], p.activation, ni,
                            p.act_n_begin, p.act_n_len);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One independent M=1 W4 tensor GEMM per routed (token,top-k) selection.
// z selects the activation row and expert; y tiles that expert's output rows.
kernel void gemm_selected_w4a8_i8a_i4b_f32c(
    device const int8_t* A [[buffer(0)]], device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]], constant SelectedW4A8Params& p [[buffer(3)]],
    device const float* SCALE_A [[buffer(4)]], device const float* SCALE_W [[buffer(5)]],
    device const int* expert_idx [[buffer(6)]], threadgroup int8_t* shmem [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const int NRA=64,NRB=64,NK=32,NT=128;int sel=(int)tg.z;if(sel>=p.selections)return;
    int ra=(int)tg.y*NRA,expert=expert_idx[sel];
    threadgroup float* facc=(threadgroup float*)shmem;
    threadgroup int8_t* sa=shmem+NRA*NRB*sizeof(float);threadgroup int32_t* si=(threadgroup int32_t*)sa;
    for(int i=tid;i<NRA*NRB;i+=NT)facc[i]=0.0f;threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tA=tensor(sa,dextents<int32_t,2>(NK,NRA));
    auto tB=tensor((device int8_t*)A,dextents<int32_t,2>(p.K,p.activation_rows),array<int,2>({1,p.K}));
    matmul2d<matmul2d_descriptor(NRB,NRA,NK,false,true,true,
        matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<4>> mm;
    const int UNROLL=16,A_WORK=NRA*(NK/UNROLL);ulong expert_row=(ulong)expert*p.rows_per_expert;
    const ulong BG128_BYTES=544;
    for(int g0=0;g0<p.K;g0+=p.group_size){int g=g0/p.group_size,gend=min(g0+p.group_size,p.K);
        auto ct=mm.get_destination_cooperative_tensor<decltype(tB),decltype(tA),int32_t>();
        for(int lk=g0;lk<gend;lk+=NK){for(int work=tid;work<A_WORK;work+=NT){int nl=work/(NK/UNROLL);
            int sub=work%(NK/UNROLL),kb=sub*UNROLL,gn=ra+nl,gk=lk+kb;threadgroup int8_t* dst=sa+nl*NK+kb;
            if(gn<p.N){ulong flatrow=expert_row+gn,nt=flatrow/8;int ch=(int)(flatrow%8);
                int qgi=(gk%128)/32;device const uint8_t* block=B+(nt*p.groups_per_row+g)*BG128_BYTES;
                device const uint8_t* wr=block+32+qgi*128+ch*16+(gk%32)/2;
                #pragma unroll
                for(int i=0;i<UNROLL;i+=2){uint8_t q=(gk+i<p.K)?wr[i/2]:0;int lo=q&15,hi=q>>4;
                    if(lo>=8)lo-=16;if(hi>=8)hi-=16;dst[i]=(int8_t)lo;dst[i+1]=(int8_t)hi;}}
            else for(int i=0;i<UNROLL;i++)dst[i]=0;}
            threadgroup_barrier(mem_flags::mem_threadgroup);auto ma=tA.slice(0,0);
            int ar=sel/max(p.activation_repeat,1);auto mb=tB.slice(lk,ar);
            mm.run(mb,ma,ct);threadgroup_barrier(mem_flags::mem_threadgroup);}
        ct.store(tensor(si,dextents<int32_t,2>(NRB,NRA)));threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int i=tid;i<NRA;i+=NT){int gn=ra+i;if(gn<p.N){ulong flatrow=expert_row+gn,nt=flatrow/8;
            int ch=(int)(flatrow%8);device const float* bsc=(device const float*)(B+(nt*p.groups_per_row+g)*BG128_BYTES);
            facc[i]+=(float)si[i]*bsc[ch];}}threadgroup_barrier(mem_flags::mem_threadgroup);}
    int ar=sel/max(p.activation_repeat,1);
    for(int i=tid;i<NRA;i+=NT){int gn=ra+i;if(gn<p.N)C[p.c_offset+(ulong)sel*p.c_row_stride+gn]=facc[i]*SCALE_A[ar];}
}

// Resident-package native BG32 selected-expert GEMV. A BG32 block stores the
// FP32 scales for eight output channels followed by 16 packed signed-int4
// bytes for each channel. One SIMD group evaluates those eight channels
// together and shares each activation byte across them.
kernel void gemv_selected_experts_bg32_i8a_i4b_f32c(
    device const int8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant SelectedW4A8Params& p [[buffer(3)]],
    device const float* SCALE_A [[buffer(4)]],
    device const int* expert_idx [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    const int sel = (int)tg.z;
    const int row0 =
        ((int)tg.x * (int)nsg + (int)sg) * 8;
    if (sel >= p.selections || row0 >= p.N) return;

    const int repeat = max(p.activation_repeat, 1);
    const int activation_row = sel / repeat;
    const int expert = expert_idx[sel];
    const ulong expert_row_tile =
        (ulong)expert * (ulong)(p.rows_per_expert / 8);
    const ulong row_tile = (ulong)(row0 / 8);
    device const int8_t* activation =
        A + (ulong)activation_row * (ulong)p.K;
    float lane_sums[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f};

    // Eight four-lane teams process eight independent G32 blocks at once.
    // Each lane consumes four packed bytes (eight K values), matching the
    // vectorized BG128 kernel's work per lane while applying the finer G32
    // scale before the final, single SIMD reduction.
    const int group_lane = (int)lane >> 2;
    const int lane_in_group = (int)lane & 3;
    for (int group_base = 0;
         group_base < p.groups_per_row;
         group_base += 8) {
        const int group = group_base + group_lane;
        if (group >= p.groups_per_row) continue;
        const int k = group * 32 + lane_in_group * 8;
        const char4 av0 =
            *(device const char4*)(activation + k);
        const char4 av1 =
            *(device const char4*)(activation + k + 4);
        const int4 ae =
            int4(av0.x, av0.z, av1.x, av1.z);
        const int4 ao =
            int4(av0.y, av0.w, av1.y, av1.w);
        device const uint8_t* block =
            B + ((expert_row_tile + row_tile) *
                     (ulong)p.groups_per_row +
                 (ulong)group) *
                    160ul;
        const float activation_scale =
            SCALE_A[(ulong)activation_row *
                        (ulong)p.groups_per_row +
                    (ulong)group];
        device const float* weight_scales =
            (device const float*)block;
        #pragma unroll
        for (int channel = 0; channel < 8; ++channel) {
            const int row = row0 + channel;
            if (row < p.N) {
                const uchar4 packed =
                    *(device const uchar4*)(
                        block + 32 + channel * 16 +
                        lane_in_group * 4);
                const int4 lo =
                    int4((packed & uchar4(15)) ^ uchar4(8)) - 8;
                const int4 hi =
                    int4((packed >> 4) ^ uchar4(8)) - 8;
                const int4 products = ae * lo + ao * hi;
                lane_sums[channel] +=
                    (float)(products.x + products.y +
                            products.z + products.w) *
                    weight_scales[channel] * activation_scale;
            }
        }
    }

    const float4 accum0 = simd_sum(float4(
        lane_sums[0], lane_sums[1], lane_sums[2], lane_sums[3]));
    const float4 accum1 = simd_sum(float4(
        lane_sums[4], lane_sums[5], lane_sums[6], lane_sums[7]));

    if (lane == 0) {
        device float* out =
            C + p.c_offset +
            (ulong)sel * (ulong)p.c_row_stride + (ulong)row0;
        if (row0 + 8 <= p.N) {
            *(device float4*)(out + 0) = accum0;
            *(device float4*)(out + 4) = accum1;
        } else {
            #pragma unroll
            for (int channel = 0; channel < 8; ++channel) {
                if (row0 + channel < p.N)
                    out[channel] =
                        (channel < 4
                             ? accum0[channel]
                             : accum1[channel - 4]);
            }
        }
    }
}

// Resident W8PC selected-expert GEMV. One SIMD group evaluates eight output
// rows while reusing each int8 activation vector across all rows. Weight and
// activation scales are both per row, so the whole K reduction stays int32.
kernel void gemv_selected_experts_w8_i8a_i8b_f32c(
    device const int8_t* A [[buffer(0)]],
    device const int8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant SelectedW4A8Params& p [[buffer(3)]],
    device const float* SCALE_A [[buffer(4)]],
    device const float* SCALE_W [[buffer(5)]],
    device const int* expert_idx [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    const int selection = (int)tg.z;
    const int row0 =
        ((int)tg.x * (int)nsg + (int)sg) * 8;
    if (selection >= p.selections || row0 >= p.N) return;

    const int activation_row =
        selection / max(p.activation_repeat, 1);
    const int expert = expert_idx[selection];
    device const int8_t* activation =
        A + (ulong)activation_row * p.K;
    int sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int vector_end = p.K & ~3;
    for (int k = (int)lane * 4;
         k + 3 < p.K; k += 128) {
        const int4 av = int4(
            *(device const char4*)(activation + k));
        #pragma unroll
        for (int channel = 0; channel < 8; ++channel) {
            const int row = row0 + channel;
            if (row >= p.N) continue;
            const ulong flatrow =
                (ulong)expert * p.rows_per_expert + row;
            const int4 weight = int4(
                *(device const char4*)(B + flatrow * p.K + k));
            const int4 product = av * weight;
            sums[channel] +=
                product.x + product.y + product.z + product.w;
        }
    }
    for (int k = vector_end + (int)lane;
         k < p.K; k += 32) {
        const int av = (int)activation[k];
        #pragma unroll
        for (int channel = 0; channel < 8; ++channel) {
            const int row = row0 + channel;
            if (row >= p.N) continue;
            const ulong flatrow =
                (ulong)expert * p.rows_per_expert + row;
            sums[channel] += av * (int)B[flatrow * p.K + k];
        }
    }
    const int4 reduced0 = simd_sum(int4(
        sums[0], sums[1], sums[2], sums[3]));
    const int4 reduced1 = simd_sum(int4(
        sums[4], sums[5], sums[6], sums[7]));
    if (lane == 0) {
        const float activation_scale = SCALE_A[activation_row];
        device float* output =
            C + p.c_offset +
            (ulong)selection * p.c_row_stride + row0;
        #pragma unroll
        for (int channel = 0; channel < 8; ++channel) {
            const int row = row0 + channel;
            if (row >= p.N) continue;
            const ulong flatrow =
                (ulong)expert * p.rows_per_expert + row;
            const int dot = channel < 4
                ? reduced0[channel]
                : reduced1[channel - 4];
            output[channel] =
                (float)dot * activation_scale * SCALE_W[flatrow];
        }
    }
}

// Decode-specialized native BG128 GEMV. Four lanes cooperate on one
// quantization group, so a SIMD group evaluates eight groups concurrently.
// Each lane reads one contiguous 32-value qgi slice and reuses its activation
// values across all eight output channels in the native block.
kernel void gemv_selected_slots_bg128_i8a_i4b_f32c(
    device const int8_t* A [[buffer(0)]], device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]], constant SelectedW4A8Params& p [[buffer(3)]],
    device const float* SCALE_A [[buffer(4)]],
    device const ulong* weight_offsets [[buffer(6)]],
    device const uint* selection_indices [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int sel = (int)tg.z;
    const int row0 = (int)tg.x * 32 + (int)sg * 8;
    if (sel >= p.selections || row0 >= p.N) return;

    const int output_sel = (int)selection_indices[sel];
    const int ar = output_sel / max(p.activation_repeat, 1);
    device const int8_t* activation = A + (ulong)ar * p.K;
    device const uint8_t* weight = B + weight_offsets[sel];
    const int row_tile = row0 >> 3;
    const int group_lane = (int)lane >> 2;
    const int lane_in_group = (int)lane & 3;
    const ulong BG128_BYTES = 544;
    float lane_sums[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f};

    for (int group_base = 0; group_base < p.groups_per_row;
         group_base += 8) {
        const int g = group_base + group_lane;
        if (g >= p.groups_per_row) continue;
        const int k0 = g * 128 + lane_in_group * 32;
        device const uint8_t* block =
            weight + ((ulong)row_tile * p.groups_per_row + g) *
                         BG128_BYTES;
        int partials[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int segment = 0; segment < 4; ++segment) {
            const char4 av0 =
                *(device const char4*)(activation + k0 + segment * 8);
            const char4 av1 =
                *(device const char4*)(activation + k0 + segment * 8 + 4);
            const int4 ae = int4(av0.x, av0.z, av1.x, av1.z);
            const int4 ao = int4(av0.y, av0.w, av1.y, av1.w);
            #pragma unroll
            for (int channel = 0; channel < 8; ++channel) {
                const uchar4 q = *(device const uchar4*)(
                    block + 32 + lane_in_group * 128 +
                    channel * 16 + segment * 4);
                // Sign-extend a 4-bit two's-complement value without a
                // compare/select pair: (nibble XOR 8) - 8.
                const int4 lo =
                    int4((q & uchar4(15)) ^ uchar4(8)) - 8;
                const int4 hi =
                    int4((q >> 4) ^ uchar4(8)) - 8;
                const int4 products = ae * lo + ao * hi;
                partials[channel] +=
                    products.x + products.y + products.z + products.w;
            }
        }
        device const float* weight_scales =
            (device const float*)block;
        #pragma unroll
        for (int channel = 0; channel < 8; ++channel)
            lane_sums[channel] +=
                (float)partials[channel] * weight_scales[channel];
    }
    const float activation_scale = SCALE_A[ar];
    #pragma unroll
    for (int channel = 0; channel < 8; ++channel) {
        const float sum = simd_sum(lane_sums[channel]);
        if (lane == 0) {
            const int row = row0 + channel;
            if (row < p.N)
                C[p.c_offset + (ulong)output_sel * p.c_row_stride + row] =
                    sum * activation_scale;
        }
    }
}

constant char kMxfp4Coefficients[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12,
};

inline int mxfp4_coefficient(uint nibble) {
    return (int)kMxfp4Coefficients[nibble & 15u];
}

inline int4 mxfp4_coefficient4(uint4 nibble) {
    const int4 magnitude = int4(nibble & uint4(7u));
    int4 coefficient = magnitude;
    coefficient += select(int4(0), int4(1), magnitude >= int4(5));
    coefficient += select(int4(0), int4(1), magnitude >= int4(6));
    coefficient += select(int4(0), int4(3), magnitude >= int4(7));
    return select(
        coefficient, -coefficient, (nibble & uint4(8u)) != uint4(0u));
}

inline float decode_e8m0_metal(uint code) {
    const uint bits = code == 0u ? (1u << 22) : (code << 23);
    return as_type<float>(bits);
}

inline float round_bf16_metal(float value) {
    uint bits = as_type<uint>(value);
    const uint exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) return value;
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

// Fast MXFP4 path: one K32 activation scale exactly matches one E8M0 weight
// scale. The SIMD group forms eight signed integer dots in parallel and only
// converts the eight reduced sums to FP32.
kernel void gemv_selected_slots_mxfp4_i8a_f32c(
    device const int8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant SelectedMxfp4Params& p [[buffer(3)]],
    device const float* activation_scales [[buffer(4)]],
    device const int8_t* residual_A [[buffer(5)]],
    device const ulong* weight_offsets [[buffer(6)]],
    device const uint* selection_indices [[buffer(7)]],
    device const float* residual_scales [[buffer(8)]],
    threadgroup int8_t* activation_cache [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int ordered_selection = (int)tg.z;
    const int row0 = (int)tg.x * 64 + (int)sg * 16;
    if (ordered_selection >= p.selections || row0 >= p.N) return;
    // Two adjacent lanes cooperate on one output row. Each lane consumes 16
    // adjacent K values (eight packed FP4 bytes), so every weight byte is read
    // exactly once and a K32 block needs only one shuffle.
    const int row_in_simdgroup = (int)lane >> 1;
    const int lane_in_row = (int)lane & 1;
    const int row = row0 + row_in_simdgroup;
    const int weight_row = min(row, p.N - 1);
    const int selection = (int)selection_indices[ordered_selection];
    const int activation_row =
        selection / max(p.activation_repeat, 1);
    device const int8_t* activation_source =
        A + (ulong)activation_row * p.K;
    device const int8_t* residual_source =
        residual_A + (ulong)activation_row * p.K;
    threadgroup int8_t* activation = activation_cache;
    threadgroup int8_t* residual_activation =
        activation_cache + p.K;
    threadgroup float* cached_scales =
        (threadgroup float*)(activation_cache + 2 * p.K);
    threadgroup float* cached_residual_scales =
        cached_scales + p.groups_per_row;
    for (uint index = tid; index < (uint)p.K; index += threads.x) {
        activation[index] = activation_source[index];
        residual_activation[index] = residual_source[index];
    }
    for (uint index = tid; index < (uint)p.groups_per_row;
         index += threads.x) {
        cached_scales[index] = activation_scales[
            (ulong)activation_row * p.groups_per_row + index];
        cached_residual_scales[index] = residual_scales[
            (ulong)activation_row * p.groups_per_row + index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uint8_t* weight =
        B + weight_offsets[ordered_selection];
    const ulong data_bytes = (ulong)p.N * (ulong)(p.K / 2);
    device const uint8_t* scales = weight + data_bytes;
    float sum = 0.0f;

    for (int group = 0; group < p.groups_per_row; ++group) {
        const int k0 = group * 32 + lane_in_row * 16;
        const ulong packed_offset =
            (ulong)weight_row * (p.K / 2) + (ulong)(k0 >> 1);
        const uchar4 packed0 =
            *(device const uchar4*)(weight + packed_offset);
        const uchar4 packed1 =
            *(device const uchar4*)(weight + packed_offset + 4u);
        const int4 low0 = mxfp4_coefficient4(uint4(packed0 & uchar4(15u)));
        const int4 high0 = mxfp4_coefficient4(uint4(packed0 >> 4));
        const int4 low1 = mxfp4_coefficient4(uint4(packed1 & uchar4(15u)));
        const int4 high1 = mxfp4_coefficient4(uint4(packed1 >> 4));

        const char4 a0 = *(threadgroup const char4*)(activation + k0);
        const char4 a1 = *(threadgroup const char4*)(activation + k0 + 4);
        const char4 a2 = *(threadgroup const char4*)(activation + k0 + 8);
        const char4 a3 = *(threadgroup const char4*)(activation + k0 + 12);
        const int4 even0 = int4(char4(a0.xz, a1.xz));
        const int4 odd0 = int4(char4(a0.yw, a1.yw));
        const int4 even1 = int4(char4(a2.xz, a3.xz));
        const int4 odd1 = int4(char4(a2.yw, a3.yw));

        const char4 r0 =
            *(threadgroup const char4*)(residual_activation + k0);
        const char4 r1 =
            *(threadgroup const char4*)(residual_activation + k0 + 4);
        const char4 r2 =
            *(threadgroup const char4*)(residual_activation + k0 + 8);
        const char4 r3 =
            *(threadgroup const char4*)(residual_activation + k0 + 12);
        const int4 residual_even0 = int4(char4(r0.xz, r1.xz));
        const int4 residual_odd0 = int4(char4(r0.yw, r1.yw));
        const int4 residual_even1 = int4(char4(r2.xz, r3.xz));
        const int4 residual_odd1 = int4(char4(r2.yw, r3.yw));

        const int4 primary_products =
            even0 * low0 + odd0 * high0 +
            even1 * low1 + odd1 * high1;
        const int4 residual_products =
            residual_even0 * low0 + residual_odd0 * high0 +
            residual_even1 * low1 + residual_odd1 * high1;
        int dot = primary_products.x + primary_products.y +
                  primary_products.z + primary_products.w;
        int residual_dot =
            residual_products.x + residual_products.y +
            residual_products.z + residual_products.w;
        dot += simd_shuffle_down(dot, 1);
        residual_dot += simd_shuffle_down(residual_dot, 1);
        if (lane_in_row == 0 && row < p.N) {
            const float a_scale = cached_scales[group];
            const float residual_scale = cached_residual_scales[group];
            const uint scale_code = scales[
                (ulong)weight_row * p.groups_per_row + group];
            sum +=
                ((float)dot * a_scale +
                 (float)residual_dot * residual_scale) *
                (0.5f * decode_e8m0_metal(scale_code));
        }
    }
    if (lane_in_row == 0 && row < p.N)
        C[(ulong)selection * p.c_row_stride + row] =
            round_bf16_metal(sum);
}

// DeepSeek-V4 applies route weights before the down projection and rounds the
// routed intermediate to BF16. Keep those semantics in one lightweight pass.
kernel void moe_swiglu_route_bf16(
    device float* merged [[buffer(0)]],
    constant MoeW4Params& p [[buffer(3)]],
    device const float* topw [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint selection = index / (uint)p.intermediate;
    const uint column = index - selection * (uint)p.intermediate;
    if (selection >= (uint)(p.seq_len * p.top_k)) return;
    const ulong base =
        (ulong)selection * (ulong)(2 * p.intermediate);
    float gate = round_bf16_metal(merged[base + column]);
    float up = round_bf16_metal(
        merged[base + (uint)p.intermediate + column]);
    if (p.swiglu_limit > 0.0f) {
        gate = min(gate, p.swiglu_limit);
        up = clamp(up, -p.swiglu_limit, p.swiglu_limit);
    }
    const float activated =
        (gate / (1.0f + exp(-gate))) * up * topw[selection];
    merged[base + column] = round_bf16_metal(activated);
}

// Resident-package variant of the native BG128 selected GEMV. The whole
// aggregate expert tensor is one contiguous [expert,row-tile,group] block, so
// derive the selected expert's byte offset directly from the GPU router output
// instead of requiring host-generated SSD slot offsets.
inline void accumulate_expert_bg128(
    device const int8_t* activation,
    device const uint8_t* weight,
    device const float* activation_scales,
    int groups_per_row,
    int row_tile,
    float multiplier,
    ushort lane,
    thread float* lane_sums) {
    const int group_lane = (int)lane >> 2;
    const int lane_in_group = (int)lane & 3;
    const ulong BG128_BYTES = 544;
    for (int group_base = 0; group_base < groups_per_row;
         group_base += 8) {
        const int g = group_base + group_lane;
        if (g >= groups_per_row) continue;
        const int k0 = g * 128 + lane_in_group * 32;
        device const uint8_t* block =
            weight + ((ulong)row_tile * groups_per_row + g) *
                         BG128_BYTES;
        int partials[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int segment = 0; segment < 4; ++segment) {
            const char4 av0 =
                *(device const char4*)(
                    activation + k0 + segment * 8);
            const char4 av1 =
                *(device const char4*)(
                    activation + k0 + segment * 8 + 4);
            const int4 ae =
                int4(av0.x, av0.z, av1.x, av1.z);
            const int4 ao =
                int4(av0.y, av0.w, av1.y, av1.w);
            #pragma unroll
            for (int channel = 0; channel < 8; ++channel) {
                const uchar4 q = *(device const uchar4*)(
                    block + 32 + lane_in_group * 128 +
                    channel * 16 + segment * 4);
                // Sign-extend a 4-bit two's-complement value without a
                // compare/select pair: (nibble XOR 8) - 8.
                const int4 lo =
                    int4((q & uchar4(15)) ^ uchar4(8)) - 8;
                const int4 hi =
                    int4((q >> 4) ^ uchar4(8)) - 8;
                const int4 products = ae * lo + ao * hi;
                partials[channel] +=
                    products.x + products.y + products.z + products.w;
            }
        }
        device const float* weight_scales =
            (device const float*)block;
        const float scale = activation_scales[g] * multiplier;
        #pragma unroll
        for (int channel = 0; channel < 8; ++channel)
            lane_sums[channel] +=
                (float)partials[channel] *
                weight_scales[channel] * scale;
    }
}

kernel void gemv_selected_experts_bg128_i8a_i4b_f32c(
    device const int8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant SelectedW4A8Params& p [[buffer(3)]],
    device const float* SCALE_A [[buffer(4)]],
    device const int* expert_idx [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    const int sel = (int)tg.z;
    const int row0 =
        (int)tg.x * (int)nsg * 8 + (int)sg * 8;
    if (sel >= p.selections || row0 >= p.N) return;

    const int groups = p.groups_per_row;
    const int repeat = max(p.activation_repeat, 1);
    const int ar = sel / repeat;
    const int expert = expert_idx[sel];
    device const int8_t* activation = A + (ulong)ar * p.K;
    const ulong BG128_BYTES = 544;
    const ulong expert_bytes =
        (ulong)((p.rows_per_expert + 7) / 8) *
        (ulong)groups * BG128_BYTES;
    device const uint8_t* weight =
        B + (ulong)expert * expert_bytes;
    const int row_tile = row0 >> 3;
    float lane_sums[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f};
    accumulate_expert_bg128(
        activation, weight,
        SCALE_A + (ulong)ar * groups,
        groups, row_tile, 1.0f,
        lane, lane_sums);
    const float4 sums0 = simd_sum(
        float4(
            lane_sums[0], lane_sums[1],
            lane_sums[2], lane_sums[3]));
    const float4 sums1 = simd_sum(
        float4(
            lane_sums[4], lane_sums[5],
            lane_sums[6], lane_sums[7]));
    if (lane == 0) {
        device float* out =
            C + p.c_offset +
            (ulong)sel * p.c_row_stride + row0;
        if (row0 + 8 <= p.N) {
            *(device float4*)(out + 0) = sums0;
            *(device float4*)(out + 4) = sums1;
        } else {
            #pragma unroll
            for (int channel = 0; channel < 8; ++channel) {
                if (row0 + channel < p.N)
                    out[channel] =
                        channel < 4
                            ? sums0[channel]
                            : sums1[channel - 4];
            }
        }
    }
}

// Build a fixed-stride route list for each expert. top-k selection guarantees
// that an expert appears at most once per token, so max_routes=seq_len is a
// strict bound and no overflow path is needed.
kernel void moe_reset_expert_counts(
    device atomic_uint* counts [[buffer(0)]],
    device atomic_uint* grouped_job_counts [[buffer(1)]],
    constant MoeW4Params& p [[buffer(3)]],
    uint expert [[thread_position_in_grid]]) {
    if (expert < (uint)p.experts)
        atomic_store_explicit(
            counts + expert, 0u, memory_order_relaxed);
    if (expert == 0)
        for (uint queue = 0; queue < 2; ++queue)
            atomic_store_explicit(
                grouped_job_counts + queue, 0u,
                memory_order_relaxed);
}

// Keep every expert's routes in canonical selection order. One threadgroup
// scans 128 selections at a time for one expert, using SIMD prefix sums to
// compact matches in stable order. This avoids both the serial expert scan and
// the nondeterministic row permutation caused by atomic append.
kernel void moe_build_expert_routes(
    device const int* expert_idx [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    device int* routes [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    uint expert [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (expert >= (uint)p.experts) return;
    constexpr uint THREADS = 128;
    constexpr uint SIMD_GROUPS = THREADS / 32;
    threadgroup uint simd_counts[SIMD_GROUPS];
    threadgroup uint simd_offsets[SIMD_GROUPS];
    threadgroup uint batch_count;

    const uint selections = (uint)(p.seq_len * p.top_k);
    uint count = 0;
    for (uint batch = 0; batch < selections; batch += THREADS) {
        const uint selection = batch + (uint)tid;
        const uint match =
            selection < selections &&
            expert_idx[selection] == (int)expert;
        const uint local_offset =
            simd_prefix_exclusive_sum(match);
        const uint local_count = simd_sum(match);
        if (lane == 0)
            simd_counts[sg] = local_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            const uint value =
                lane < SIMD_GROUPS ? simd_counts[lane] : 0;
            const uint offset =
                simd_prefix_exclusive_sum(value);
            if (lane < SIMD_GROUPS)
                simd_offsets[lane] = offset;
            if (lane == SIMD_GROUPS - 1)
                batch_count = offset + value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (match != 0) {
            const uint output =
                count + simd_offsets[sg] + local_offset;
            if (output < (uint)p.seq_len)
                routes[(ulong)expert * (ulong)p.seq_len + output] =
                    (int)selection;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        count += batch_count;
    }
    if (tid == 0)
        atomic_store_explicit(
            counts + expert, min(count, (uint)p.seq_len),
            memory_order_relaxed);
}

kernel void moe_build_grouped_jobs(
    device const atomic_uint* expert_counts [[buffer(0)]],
    device uint2* grouped_jobs_small [[buffer(1)]],
    device atomic_uint* grouped_job_count [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device uint2* grouped_jobs_large [[buffer(4)]],
    uint expert [[thread_position_in_grid]]) {
    if (expert >= (uint)p.experts) return;
    const uint route_count = atomic_load_explicit(
        expert_counts + expert, memory_order_relaxed);
    const uint small_jobs =
        (route_count + MOLLM_GROUPED_MOE_ROUTE_TILE_SMALL - 1) /
        MOLLM_GROUPED_MOE_ROUTE_TILE_SMALL;
    const uint large_jobs =
        (route_count + MOLLM_GROUPED_MOE_ROUTE_TILE_LARGE - 1) /
        MOLLM_GROUPED_MOE_ROUTE_TILE_LARGE;
    if (small_jobs != 0) {
        const uint first = atomic_fetch_add_explicit(
            grouped_job_count, small_jobs,
            memory_order_relaxed);
        for (uint job = 0; job < small_jobs; ++job) {
            grouped_jobs_small[first + job] =
                uint2(
                    expert,
                    job * MOLLM_GROUPED_MOE_ROUTE_TILE_SMALL);
        }
    }
    if (large_jobs != 0) {
        const uint first = atomic_fetch_add_explicit(
            grouped_job_count + 1, large_jobs,
            memory_order_relaxed);
        for (uint job = 0; job < large_jobs; ++job) {
            grouped_jobs_large[first + job] =
                uint2(
                    expert,
                    job * MOLLM_GROUPED_MOE_ROUTE_TILE_LARGE);
        }
    }
}

// Four MTLDispatchThreadgroupsIndirectArguments records:
// gate/up route16, gate/up route32, down route16, and down route32.
kernel void moe_finalize_grouped_dispatch(
    device const atomic_uint* grouped_job_count [[buffer(0)]],
    device uint* indirect_args [[buffer(1)]],
    constant MoeW4Params& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const uint jobs_small = atomic_load_explicit(
        grouped_job_count, memory_order_relaxed);
    const uint jobs_large = atomic_load_explicit(
        grouped_job_count + 1, memory_order_relaxed);
    // Route32 has higher register pressure and wins only when its larger
    // route tile removes enough jobs. Select one tile for the whole layer so
    // the GPU sees one full occupancy tail instead of two partial queues.
    const bool use_large =
        jobs_small != 0 &&
        jobs_large * 5u <= jobs_small * 3u;
    const uint dispatch_small = use_large ? 0u : jobs_small;
    const uint dispatch_large = use_large ? jobs_large : 0u;
    const uint gate_tiles =
        ((uint)p.intermediate +
         MOLLM_GROUPED_MOE_GATE_UP_OUTPUT_TILE - 1) /
        MOLLM_GROUPED_MOE_GATE_UP_OUTPUT_TILE;
    const uint down_tiles =
        ((uint)p.hidden +
         MOLLM_GROUPED_MOE_DOWN_OUTPUT_TILE - 1) /
        MOLLM_GROUPED_MOE_DOWN_OUTPUT_TILE;
    indirect_args[0] = dispatch_small;
    indirect_args[1] = gate_tiles;
    indirect_args[2] = 1;
    indirect_args[3] = dispatch_large;
    indirect_args[4] = gate_tiles;
    indirect_args[5] = 1;
    indirect_args[6] = dispatch_small;
    indirect_args[7] = down_tiles;
    indirect_args[8] = 1;
    indirect_args[9] = dispatch_large;
    indirect_args[10] = down_tiles;
    indirect_args[11] = 1;
}

// Batched expert GEMM over the fixed-stride route lists above. One threadgroup
// reuses one expert weight tile across 16 or 32 routed tokens instead of
// launching an independent M=1 GEMM for every [token, top-k] pair. The common
// implementation is instantiated with independent output tiles: paired
// gate/up favors occupancy, while the single down projection benefits from a
// wider tile.
template<int NRA, int NRB, bool PAIRED_GATE_UP>
inline void gemm_grouped_experts_w8_impl(
    device const int8_t* A,
    device const int8_t* B,
    device float* C,
    constant GroupedW4A8Params& p,
    device const float* SCALE_A,
    device const float* SCALE_W,
    device const atomic_uint* expert_counts,
    device const int* expert_routes,
    device const uint2* grouped_jobs,
    threadgroup int8_t* shmem,
    uint3 tg,
    ushort tid) {
    constexpr int NK = 32;
    constexpr int NUM_THREADS =
        32 * MOLLM_GROUPED_MOE_SIMDGROUPS;
    constexpr int projections = PAIRED_GATE_UP ? 2 : 1;
    constexpr int projected_rows = projections * NRA;

    const uint2 grouped_job = grouped_jobs[tg.x];
    const int expert = (int)grouped_job.x;
    if (expert >= p.experts) return;
    const int route_begin = (int)grouped_job.y;
    const int route_count = (int)atomic_load_explicit(
        expert_counts + expert, memory_order_relaxed);
    if (route_begin >= route_count) return;
    const int row_begin = (int)tg.y * NRA;
    if (row_begin >= p.N) return;

    threadgroup int8_t* staged_w = shmem;
    threadgroup int8_t* staged_a =
        staged_w + projected_rows * NK;
    auto tW = tensor(
        staged_w, dextents<int32_t,2>(NK, projected_rows));
    auto tA = tensor(
        staged_a, dextents<int32_t,2>(NK, NRB));
    matmul2d<
        matmul2d_descriptor(
            NRB, projected_rows, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<MOLLM_GROUPED_MOE_SIMDGROUPS>> mm;
    constexpr int ACC_PER_THREAD =
        (projected_rows * NRB + NUM_THREADS - 1) / NUM_THREADS;
    float accum[ACC_PER_THREAD];
    #pragma unroll
    for (int slot = 0; slot < ACC_PER_THREAD; ++slot)
        accum[slot] = 0.0f;
    auto dot =
        mm.template get_destination_cooperative_tensor<
            decltype(tA), decltype(tW), int32_t>();

    constexpr int UNROLL = 16;
    constexpr int weight_work =
        projected_rows * (NK / UNROLL);
    constexpr int activation_work =
        NRB * (NK / UNROLL);
    const int activation_group_size =
        (p.K + p.groups_per_row - 1) / p.groups_per_row;
    for (int k0 = 0; k0 < p.K; k0 += NK) {
        const int activation_group = k0 / activation_group_size;
        if (k0 == activation_group * activation_group_size) {
            #pragma unroll
            for (int slot = 0; slot < ACC_PER_THREAD; ++slot)
                dot[slot] = 0;
        }
        for (int work = (int)tid;
             work < weight_work; work += NUM_THREADS) {
            const int projected_row = work / (NK / UNROLL);
            const int ksub = work % (NK / UNROLL);
            const int projection = projected_row / NRA;
            const int local_row = projected_row % NRA;
            const int output_row = row_begin + local_row;
            const int weight_row =
                projection * p.N + output_row;
            threadgroup int8_t* destination =
                staged_w + projected_row * NK + ksub * UNROLL;
            if (output_row < p.N) {
                device const int8_t* source =
                    B +
                    ((ulong)expert * (ulong)p.rows_per_expert +
                     (ulong)weight_row) * (ulong)p.K +
                    (ulong)k0 + (ulong)ksub * UNROLL;
                *((threadgroup ulong*)destination) =
                    *((device const ulong*)source);
                *((threadgroup ulong*)(destination + 8)) =
                    *((device const ulong*)(source + 8));
            } else {
                *((threadgroup ulong*)destination) = 0;
                *((threadgroup ulong*)(destination + 8)) = 0;
            }
        }
        for (int work = (int)tid;
             work < activation_work; work += NUM_THREADS) {
            const int local_route = work / (NK / UNROLL);
            const int ksub = work % (NK / UNROLL);
            const int route_slot = route_begin + local_route;
            const bool valid_route = route_slot < route_count;
            const int selection = valid_route
                ? expert_routes[
                      (ulong)expert * (ulong)p.max_routes +
                      (ulong)route_slot]
                : 0;
            const int activation_row = p.activation_by_token
                ? selection / p.top_k
                : selection;
            threadgroup int8_t* destination =
                staged_a + local_route * NK + ksub * UNROLL;
            if (valid_route) {
                device const int8_t* source =
                    A + (ulong)activation_row * (ulong)p.K +
                    (ulong)k0 + (ulong)ksub * UNROLL;
                *((threadgroup ulong*)destination) =
                    *((device const ulong*)source);
                *((threadgroup ulong*)(destination + 8)) =
                    *((device const ulong*)(source + 8));
            } else {
                *((threadgroup ulong*)destination) = 0;
                *((threadgroup ulong*)(destination + 8)) = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto activation_tile = tA.slice(0, 0);
        auto weight_tile = tW.slice(0, 0);
        mm.run(activation_tile, weight_tile, dot);

        const int group_end = min(
            p.K, (activation_group + 1) * activation_group_size);
        if (k0 + NK >= group_end) {
            #pragma unroll
            for (int slot = 0; slot < ACC_PER_THREAD; ++slot) {
                const auto index =
                    dot.get_multidimensional_index(slot);
                const int output_local = index[0];
                const int route_local = index[1];
                const int route_slot = route_begin + route_local;
                if (route_slot >= route_count ||
                    row_begin + output_local % NRA >= p.N)
                    continue;
                const int selection = expert_routes[
                    (ulong)expert * (ulong)p.max_routes +
                    (ulong)route_slot];
                const int activation_row = p.activation_by_token
                    ? selection / p.top_k
                    : selection;
                const float activation_scale = SCALE_A[
                    (ulong)activation_row *
                        (ulong)p.groups_per_row +
                    (ulong)activation_group];
                const int projection = output_local / NRA;
                const int local_row = output_local % NRA;
                const int weight_row =
                    projection * p.N + row_begin + local_row;
                const ulong scale_base =
                    (ulong)expert * (ulong)p.rows_per_expert;
                accum[slot] +=
                    (float)dot[slot] * activation_scale *
                    SCALE_W[scale_base + (ulong)weight_row];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if constexpr (PAIRED_GATE_UP) {
        // The staged tiles are dead after the final K32 block. Reuse their
        // threadgroup storage to pair gate/up accumulators for SwiGLU.
        threadgroup float* paired =
            (threadgroup float*)shmem;
        #pragma unroll
        for (int slot = 0; slot < ACC_PER_THREAD; ++slot) {
            const auto index =
                dot.get_multidimensional_index(slot);
            paired[index[0] * NRB + index[1]] =
                accum[slot];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int work = (int)tid;
             work < NRA * NRB; work += NUM_THREADS) {
            const int route_local = work / NRA;
            const int output_local = work % NRA;
            const int route_slot = route_begin + route_local;
            const int output_row = row_begin + output_local;
            if (route_slot >= route_count || output_row >= p.N)
                continue;
            const int selection = expert_routes[
                (ulong)expert * (ulong)p.max_routes +
                (ulong)route_slot];
            const float gate =
                paired[output_local * NRB + route_local];
            const float up =
                paired[(NRA + output_local) * NRB + route_local];
            C[(ulong)selection * (ulong)p.c_row_stride +
              (ulong)output_row] =
                (gate / (1.0f + exp(-gate))) * up;
        }
    } else {
        #pragma unroll
        for (int slot = 0; slot < ACC_PER_THREAD; ++slot) {
            const auto index =
                dot.get_multidimensional_index(slot);
            const int output_local = index[0];
            const int route_local = index[1];
            const int route_slot = route_begin + route_local;
            const int output_row = row_begin + output_local;
            if (route_slot >= route_count || output_row >= p.N)
                continue;
            const int selection = expert_routes[
                (ulong)expert * (ulong)p.max_routes +
                (ulong)route_slot];
            C[(ulong)selection * (ulong)p.c_row_stride +
              (ulong)output_row] = accum[slot];
        }
    }
}

#define MOLLM_DEFINE_GROUPED_W8_KERNEL(NAME, NRA, NRB, PAIRED)          \
kernel void NAME(                                                       \
    device const int8_t* A [[buffer(0)]],                               \
    device const int8_t* B [[buffer(1)]],                               \
    device float* C [[buffer(2)]],                                      \
    constant GroupedW4A8Params& p [[buffer(3)]],                        \
    device const float* SCALE_A [[buffer(4)]],                          \
    device const float* SCALE_W [[buffer(5)]],                          \
    device const atomic_uint* expert_counts [[buffer(6)]],              \
    device const int* expert_routes [[buffer(7)]],                      \
    device const uint2* grouped_jobs [[buffer(8)]],                     \
    threadgroup int8_t* shmem [[threadgroup(0)]],                       \
    uint3 tg [[threadgroup_position_in_grid]],                           \
    ushort tid [[thread_index_in_threadgroup]]) {                       \
    gemm_grouped_experts_w8_impl<NRA, NRB, PAIRED>(                    \
        A, B, C, p, SCALE_A, SCALE_W, expert_counts, expert_routes,    \
        grouped_jobs, shmem, tg, tid);                                  \
}

MOLLM_DEFINE_GROUPED_W8_KERNEL(
    gemm_grouped_experts_w8_gate_up_r16, 32, 16, true)
MOLLM_DEFINE_GROUPED_W8_KERNEL(
    gemm_grouped_experts_w8_gate_up_r32, 32, 32, true)
MOLLM_DEFINE_GROUPED_W8_KERNEL(
    gemm_grouped_experts_w8_down_r16, 64, 16, false)
MOLLM_DEFINE_GROUPED_W8_KERNEL(
    gemm_grouped_experts_w8_down_r32, 64, 32, false)

#undef MOLLM_DEFINE_GROUPED_W8_KERNEL

template<int NRA, int NRB, bool PAIRED_GATE_UP>
inline void gemm_grouped_experts_bg128_impl(
    device const int8_t* A,
    device const uint8_t* B,
    device float* C,
    constant GroupedW4A8Params& p,
    device const float* SCALE_A,
    device const atomic_uint* expert_counts,
    device const int* expert_routes,
    device const uint2* grouped_jobs,
    threadgroup int8_t* shmem,
    uint3 tg,
    ushort tid) {
    constexpr int NK = 32;
    constexpr int NUM_THREADS =
        32 * MOLLM_GROUPED_MOE_SIMDGROUPS;
    constexpr int BLOCK_BYTES = 544;
    constexpr int SCALE_BYTES = 32;
    constexpr int projections = PAIRED_GATE_UP ? 2 : 1;
    constexpr int projected_rows = projections * NRA;

    const uint2 grouped_job = grouped_jobs[tg.x];
    const int expert = (int)grouped_job.x;
    if (expert >= p.experts) return;
    const int route_begin = (int)grouped_job.y;
    const int route_count = (int)atomic_load_explicit(
        expert_counts + expert, memory_order_relaxed);
    if (route_begin >= route_count) return;
    const int row_begin = (int)tg.y * NRA;
    if (row_begin >= p.N) return;

    constexpr int packed_weight_bytes =
        projections * 4 * NRA * NK / 2;
    threadgroup int8_t* staged_w = shmem;
    threadgroup int8_t* staged_a =
        staged_w + packed_weight_bytes;
    threadgroup float* weight_scales =
        (threadgroup float*)(staged_a + 4 * NRB * NK);
    threadgroup float* activation_scales =
        weight_scales + projected_rows;

    constexpr int ACC_PER_THREAD =
        (projected_rows * NRB + NUM_THREADS - 1) / NUM_THREADS;
    float accum[ACC_PER_THREAD];
    #pragma unroll
    for (int slot = 0; slot < ACC_PER_THREAD; ++slot)
        accum[slot] = 0.0f;

    auto tW0 =
        tensor<threadgroup metal::int4b_format,
               dextents<int32_t,2>, tensor_inline>(
                   (threadgroup uchar*)staged_w,
                   dextents<int32_t,2>(NK, projected_rows));
    auto tW1 =
        tensor<threadgroup metal::int4b_format,
               dextents<int32_t,2>, tensor_inline>(
                   (threadgroup uchar*)(
                       staged_w + projected_rows * NK / 2),
                   dextents<int32_t,2>(NK, projected_rows));
    auto tW2 =
        tensor<threadgroup metal::int4b_format,
               dextents<int32_t,2>, tensor_inline>(
                   (threadgroup uchar*)(
                       staged_w + 2 * projected_rows * NK / 2),
                   dextents<int32_t,2>(NK, projected_rows));
    auto tW3 =
        tensor<threadgroup metal::int4b_format,
               dextents<int32_t,2>, tensor_inline>(
                   (threadgroup uchar*)(
                       staged_w + 3 * projected_rows * NK / 2),
                   dextents<int32_t,2>(NK, projected_rows));
    auto tA0 =
        tensor(staged_a, dextents<int32_t,2>(NK, NRB));
    auto tA1 =
        tensor(staged_a + NRB * NK,
               dextents<int32_t,2>(NK, NRB));
    auto tA2 =
        tensor(staged_a + 2 * NRB * NK,
               dextents<int32_t,2>(NK, NRB));
    auto tA3 =
        tensor(staged_a + 3 * NRB * NK,
               dextents<int32_t,2>(NK, NRB));
    matmul2d<
        matmul2d_descriptor(
            NRB, projected_rows, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<MOLLM_GROUPED_MOE_SIMDGROUPS>> mm;

    const ulong expert_bytes =
        (ulong)((p.rows_per_expert + 7) / 8) *
        (ulong)p.groups_per_row * (ulong)BLOCK_BYTES;
    device const uint8_t* expert_weights =
        B + (ulong)expert * expert_bytes;
    constexpr int UNROLL = 16;
    constexpr int A_HALF_WORK = NRB * (NK / UNROLL);

    for (int group = 0; group < p.groups_per_row; ++group) {
        auto dot =
            mm.template get_destination_cooperative_tensor<
                decltype(tA0), decltype(tW0), int32_t>();

        // Stage the complete BG128 group as four packed K32 slices before
        // issuing native int8 x int4 tensor operations. Paired gate/up stages
        // both weight projections alongside the shared activation tile and
        // pays one barrier pair for all eight tensor operations.
        const int projected_row = (int)tid;
        const bool valid_projected_row =
            projected_row < projected_rows;
        const int projection = projected_row / NRA;
        const int row = projected_row % NRA;
        const int local_output_row = row_begin + row;
        const int weight_row =
            projection * p.N + local_output_row;
        const bool valid_weight_row =
            valid_projected_row &&
            local_output_row < p.N;
        const ulong weight_block =
            ((ulong)(weight_row / 8) *
                 (ulong)p.groups_per_row +
             (ulong)group) *
            (ulong)BLOCK_BYTES;
        if (valid_projected_row) {
            #pragma unroll
            for (int qgi = 0; qgi < 4; ++qgi) {
                threadgroup ulong* destination =
                    (threadgroup ulong*)(
                        staged_w +
                        qgi * projected_rows * (NK / 2) +
                        projected_row * (NK / 2));
                if (valid_weight_row) {
                    device const ulong* source =
                        (device const ulong*)(
                            expert_weights + weight_block +
                            SCALE_BYTES + qgi * 8 * 16 +
                            (weight_row & 7) * 16);
                    destination[0] = source[0];
                    destination[1] = source[1];
                } else {
                    destination[0] = 0;
                    destination[1] = 0;
                }
            }
        }

        if constexpr (
            NRB == MOLLM_GROUPED_MOE_ROUTE_TILE_SMALL) {
            // Route16's exact two-slices-per-thread mapping is faster than the
            // generic work loop on Apple GPUs.
            const int first_slice =
                (int)tid / A_HALF_WORK;
            const int activation_local =
                (int)tid - first_slice * A_HALF_WORK;
            const int activation_route =
                activation_local / (NK / UNROLL);
            const int activation_sub =
                activation_local % (NK / UNROLL);
            const int activation_slot =
                route_begin + activation_route;
            const bool valid_activation =
                activation_slot < route_count;
            const int activation_selection =
                valid_activation
                    ? expert_routes[
                          (ulong)expert * (ulong)p.max_routes +
                          (ulong)activation_slot]
                    : 0;
            const int activation_row =
                p.activation_by_token
                    ? activation_selection / p.top_k
                    : activation_selection;
            #pragma unroll
            for (int slice = first_slice;
                 slice < 4;
                 slice += NUM_THREADS / A_HALF_WORK) {
                threadgroup int8_t* destination =
                    staged_a + slice * NRB * NK +
                    activation_route * NK +
                    activation_sub * UNROLL;
                if (valid_activation) {
                    device const int8_t* source =
                        A +
                        (ulong)activation_row * (ulong)p.K +
                        (ulong)(
                            group * 128 + slice * NK +
                            activation_sub * UNROLL);
                    *((threadgroup ulong*)(destination)) =
                        *((device const ulong*)(source));
                    *((threadgroup ulong*)(destination + 8)) =
                        *((device const ulong*)(source + 8));
                } else {
                    *((threadgroup ulong*)(destination)) = 0;
                    *((threadgroup ulong*)(destination + 8)) = 0;
                }
            }
        } else {
            constexpr int ACTIVATION_STAGE_WORK =
                4 * A_HALF_WORK;
            for (int work = (int)tid;
                 work < ACTIVATION_STAGE_WORK;
                 work += NUM_THREADS) {
                const int slice = work / A_HALF_WORK;
                const int activation_local =
                    work - slice * A_HALF_WORK;
                const int activation_route =
                    activation_local / (NK / UNROLL);
                const int activation_sub =
                    activation_local % (NK / UNROLL);
                const int activation_slot =
                    route_begin + activation_route;
                const bool valid_activation =
                    activation_slot < route_count;
                const int activation_selection =
                    valid_activation
                        ? expert_routes[
                              (ulong)expert *
                                  (ulong)p.max_routes +
                              (ulong)activation_slot]
                        : 0;
                const int activation_row =
                    p.activation_by_token
                        ? activation_selection / p.top_k
                        : activation_selection;
                threadgroup int8_t* destination =
                    staged_a + slice * NRB * NK +
                    activation_route * NK +
                    activation_sub * UNROLL;
                if (valid_activation) {
                    device const int8_t* source =
                        A +
                        (ulong)activation_row * (ulong)p.K +
                        (ulong)(
                            group * 128 + slice * NK +
                            activation_sub * UNROLL);
                    *((threadgroup ulong*)(destination)) =
                        *((device const ulong*)(source));
                    *((threadgroup ulong*)(destination + 8)) =
                        *((device const ulong*)(source + 8));
                } else {
                    *((threadgroup ulong*)(destination)) = 0;
                    *((threadgroup ulong*)(destination + 8)) = 0;
                }
            }
        }

        if (projected_row < projected_rows) {
            if (valid_weight_row) {
                device const float* scales =
                    (device const float*)(
                        expert_weights + weight_block);
                weight_scales[tid] =
                    scales[weight_row & 7];
            } else {
                weight_scales[tid] = 0.0f;
            }
        }

        if (tid < NRB) {
            const int route_slot = route_begin + (int)tid;
            if (route_slot < route_count) {
                const int selection =
                    expert_routes[
                        (ulong)expert * (ulong)p.max_routes +
                        (ulong)route_slot];
                const int scale_activation_row =
                    p.activation_by_token
                        ? selection / p.top_k
                        : selection;
                activation_scales[tid] =
                    SCALE_A[
                        (ulong)scale_activation_row *
                            (ulong)p.groups_per_row +
                        (ulong)group];
            } else {
                activation_scales[tid] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto activation0 = tA0.slice(0, 0);
        auto weight0 = tW0.slice(0, 0);
        mm.run(activation0, weight0, dot);
        auto activation1 = tA1.slice(0, 0);
        auto weight1 = tW1.slice(0, 0);
        mm.run(activation1, weight1, dot);
        auto activation2 = tA2.slice(0, 0);
        auto weight2 = tW2.slice(0, 0);
        mm.run(activation2, weight2, dot);
        auto activation3 = tA3.slice(0, 0);
        auto weight3 = tW3.slice(0, 0);
        mm.run(activation3, weight3, dot);

        #pragma unroll
        for (int slot = 0; slot < ACC_PER_THREAD; ++slot) {
            const auto index =
                dot.get_multidimensional_index(slot);
            const int output_local = index[0];
            const int route_local = index[1];
            const float activation_scale =
                activation_scales[route_local];
            accum[slot] +=
                (float)dot[slot] *
                activation_scale *
                weight_scales[output_local];
        }
        if (group + 1 == p.groups_per_row) {
            if constexpr (PAIRED_GATE_UP) {
                // The staged weights/activations are dead after the final
                // group. Reuse their threadgroup storage to pair each gate
                // accumulator with its up accumulator and materialize only
                // SwiGLU, rather than a 2*N FP32 intermediate.
                threadgroup_barrier(
                    mem_flags::mem_threadgroup);
                threadgroup float* paired =
                    (threadgroup float*)shmem;
                #pragma unroll
                for (int slot = 0;
                     slot < ACC_PER_THREAD; ++slot) {
                    const auto index =
                        dot.get_multidimensional_index(slot);
                    paired[index[0] * NRB + index[1]] =
                        accum[slot];
                }
                threadgroup_barrier(
                    mem_flags::mem_threadgroup);
                for (int work = (int)tid;
                     work < NRA * NRB;
                     work += NUM_THREADS) {
                    const int route_local = work / NRA;
                    const int output_local =
                        work - route_local * NRA;
                    const int route_slot =
                        route_begin + route_local;
                    const int output_row =
                        row_begin + output_local;
                    if (route_slot < route_count &&
                        output_row < p.N) {
                        const float gate =
                            paired[
                                output_local * NRB +
                                route_local];
                        const float up =
                            paired[
                                (NRA + output_local) *
                                    NRB +
                                route_local];
                        const int selection =
                            expert_routes[
                                (ulong)expert *
                                    (ulong)p.max_routes +
                                (ulong)route_slot];
                        C[(ulong)selection *
                              (ulong)p.c_row_stride +
                          (ulong)output_row] =
                            (gate /
                             (1.0f + exp(-gate))) * up;
                    }
                }
            } else {
                #pragma unroll
                for (int slot = 0;
                     slot < ACC_PER_THREAD; ++slot) {
                    const auto index =
                        dot.get_multidimensional_index(slot);
                    const int output_local = index[0];
                    const int route_local = index[1];
                    const int route_slot =
                        route_begin + route_local;
                    const int output_row =
                        row_begin + output_local;
                    if (route_slot < route_count &&
                        output_row < p.N) {
                        const int selection =
                            expert_routes[
                                (ulong)expert *
                                    (ulong)p.max_routes +
                                (ulong)route_slot];
                        C[(ulong)selection *
                              (ulong)p.c_row_stride +
                          (ulong)output_row] =
                            accum[slot];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gemm_grouped_experts_bg128_gate_up_r16(
    device const int8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GroupedW4A8Params& p [[buffer(3)]],
    device const float* SCALE_A [[buffer(4)]],
    device const atomic_uint* expert_counts [[buffer(5)]],
    device const int* expert_routes [[buffer(6)]],
    device const uint2* grouped_jobs [[buffer(7)]],
    threadgroup int8_t* shmem [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    gemm_grouped_experts_bg128_impl<
        MOLLM_GROUPED_MOE_GATE_UP_OUTPUT_TILE,
        MOLLM_GROUPED_MOE_ROUTE_TILE_SMALL, true>(
            A, B, C, p, SCALE_A, expert_counts, expert_routes,
            grouped_jobs,
            shmem, tg, tid);
}

kernel void gemm_grouped_experts_bg128_gate_up_r32(
    device const int8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GroupedW4A8Params& p [[buffer(3)]],
    device const float* SCALE_A [[buffer(4)]],
    device const atomic_uint* expert_counts [[buffer(5)]],
    device const int* expert_routes [[buffer(6)]],
    device const uint2* grouped_jobs [[buffer(7)]],
    threadgroup int8_t* shmem [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    gemm_grouped_experts_bg128_impl<
        MOLLM_GROUPED_MOE_GATE_UP_OUTPUT_TILE,
        MOLLM_GROUPED_MOE_ROUTE_TILE_LARGE, true>(
            A, B, C, p, SCALE_A, expert_counts, expert_routes,
            grouped_jobs,
            shmem, tg, tid);
}

kernel void gemm_grouped_experts_bg128_down_r16(
    device const int8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GroupedW4A8Params& p [[buffer(3)]],
    device const float* SCALE_A [[buffer(4)]],
    device const atomic_uint* expert_counts [[buffer(5)]],
    device const int* expert_routes [[buffer(6)]],
    device const uint2* grouped_jobs [[buffer(7)]],
    threadgroup int8_t* shmem [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    gemm_grouped_experts_bg128_impl<
        MOLLM_GROUPED_MOE_DOWN_OUTPUT_TILE,
        MOLLM_GROUPED_MOE_ROUTE_TILE_SMALL, false>(
            A, B, C, p, SCALE_A, expert_counts, expert_routes,
            grouped_jobs,
            shmem, tg, tid);
}

kernel void gemm_grouped_experts_bg128_down_r32(
    device const int8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GroupedW4A8Params& p [[buffer(3)]],
    device const float* SCALE_A [[buffer(4)]],
    device const atomic_uint* expert_counts [[buffer(5)]],
    device const int* expert_routes [[buffer(6)]],
    device const uint2* grouped_jobs [[buffer(7)]],
    threadgroup int8_t* shmem [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    gemm_grouped_experts_bg128_impl<
        MOLLM_GROUPED_MOE_DOWN_OUTPUT_TILE,
        MOLLM_GROUPED_MOE_ROUTE_TILE_LARGE, false>(
            A, B, C, p, SCALE_A, expert_counts, expert_routes,
            grouped_jobs,
            shmem, tg, tid);
}

// Native BG32 counterpart of the grouped BG128 path. Each weight group is
// exactly one K32 tensor tile, so a routed-token batch pays one package-block
// load per output tile and reuses it across 16 or 32 activations.
template<int NRA, int NRB, bool PAIRED_GATE_UP>
inline void gemm_grouped_experts_bg32_impl(
    device const int8_t* A,
    device const uint8_t* B,
    device float* C,
    constant GroupedW4A8Params& p,
    device const float* SCALE_A,
    device const atomic_uint* expert_counts,
    device const int* expert_routes,
    device const uint2* grouped_jobs,
    threadgroup int8_t* shmem,
    uint3 tg,
    ushort tid) {
    constexpr int NK = 32;
    constexpr int NUM_THREADS =
        32 * MOLLM_GROUPED_MOE_SIMDGROUPS;
    constexpr int BLOCK_BYTES = 160;
    constexpr int SCALE_BYTES = 32;
    constexpr int projections = PAIRED_GATE_UP ? 2 : 1;
    constexpr int projected_rows = projections * NRA;

    const uint2 grouped_job = grouped_jobs[tg.x];
    const int expert = (int)grouped_job.x;
    if (expert >= p.experts) return;
    const int route_begin = (int)grouped_job.y;
    const int route_count = (int)atomic_load_explicit(
        expert_counts + expert, memory_order_relaxed);
    if (route_begin >= route_count) return;
    const int row_begin = (int)tg.y * NRA;
    if (row_begin >= p.N) return;

    constexpr int packed_weight_bytes =
        projected_rows * NK / 2;
    threadgroup int8_t* staged_w = shmem;
    threadgroup int8_t* staged_a =
        staged_w + packed_weight_bytes;
    threadgroup float* weight_scales =
        (threadgroup float*)(staged_a + NRB * NK);
    threadgroup float* activation_scales =
        weight_scales + projected_rows;

    constexpr int ACC_PER_THREAD =
        (projected_rows * NRB + NUM_THREADS - 1) /
        NUM_THREADS;
    float accum[ACC_PER_THREAD];
    #pragma unroll
    for (int slot = 0; slot < ACC_PER_THREAD; ++slot)
        accum[slot] = 0.0f;

    auto tW =
        tensor<threadgroup metal::int4b_format,
               dextents<int32_t,2>, tensor_inline>(
                   (threadgroup uchar*)staged_w,
                   dextents<int32_t,2>(NK, projected_rows));
    auto tA =
        tensor(staged_a, dextents<int32_t,2>(NK, NRB));
    matmul2d<
        matmul2d_descriptor(
            NRB, projected_rows, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<MOLLM_GROUPED_MOE_SIMDGROUPS>> mm;

    const ulong expert_bytes =
        (ulong)((p.rows_per_expert + 7) / 8) *
        (ulong)p.groups_per_row * (ulong)BLOCK_BYTES;
    device const uint8_t* expert_weights =
        B + (ulong)expert * expert_bytes;
    constexpr int UNROLL = 16;
    constexpr int ACTIVATION_WORK = NRB * (NK / UNROLL);

    for (int group = 0; group < p.groups_per_row; ++group) {
        auto dot =
            mm.template get_destination_cooperative_tensor<
                decltype(tA), decltype(tW), int32_t>();

        const int projected_row = (int)tid;
        const bool valid_projected_row =
            projected_row < projected_rows;
        const int projection = projected_row / NRA;
        const int row = projected_row % NRA;
        const int local_output_row = row_begin + row;
        const int weight_row =
            projection * p.N + local_output_row;
        const bool valid_weight_row =
            valid_projected_row && local_output_row < p.N;
        const ulong weight_block =
            ((ulong)(weight_row / 8) *
                 (ulong)p.groups_per_row +
             (ulong)group) *
            (ulong)BLOCK_BYTES;
        if (valid_projected_row) {
            threadgroup ulong* destination =
                (threadgroup ulong*)(
                    staged_w + projected_row * (NK / 2));
            if (valid_weight_row) {
                device const ulong* source =
                    (device const ulong*)(
                        expert_weights + weight_block +
                        SCALE_BYTES + (weight_row & 7) * 16);
                destination[0] = source[0];
                destination[1] = source[1];
                device const float* scales =
                    (device const float*)(
                        expert_weights + weight_block);
                weight_scales[projected_row] =
                    scales[weight_row & 7];
            } else {
                destination[0] = 0;
                destination[1] = 0;
                weight_scales[projected_row] = 0.0f;
            }
        }

        for (int work = (int)tid;
             work < ACTIVATION_WORK;
             work += NUM_THREADS) {
            const int activation_route =
                work / (NK / UNROLL);
            const int activation_sub =
                work % (NK / UNROLL);
            const int activation_slot =
                route_begin + activation_route;
            const bool valid_activation =
                activation_slot < route_count;
            const int activation_selection =
                valid_activation
                    ? expert_routes[
                          (ulong)expert * (ulong)p.max_routes +
                          (ulong)activation_slot]
                    : 0;
            const int activation_row =
                p.activation_by_token
                    ? activation_selection / p.top_k
                    : activation_selection;
            threadgroup int8_t* destination =
                staged_a + activation_route * NK +
                activation_sub * UNROLL;
            if (valid_activation) {
                device const int8_t* source =
                    A + (ulong)activation_row * (ulong)p.K +
                    (ulong)(group * NK +
                            activation_sub * UNROLL);
                *((threadgroup ulong*)destination) =
                    *((device const ulong*)source);
                *((threadgroup ulong*)(destination + 8)) =
                    *((device const ulong*)(source + 8));
            } else {
                *((threadgroup ulong*)destination) = 0;
                *((threadgroup ulong*)(destination + 8)) = 0;
            }
        }

        if (tid < NRB) {
            const int route_slot = route_begin + (int)tid;
            if (route_slot < route_count) {
                const int selection =
                    expert_routes[
                        (ulong)expert * (ulong)p.max_routes +
                        (ulong)route_slot];
                const int scale_activation_row =
                    p.activation_by_token
                        ? selection / p.top_k
                        : selection;
                activation_scales[tid] =
                    SCALE_A[
                        (ulong)scale_activation_row *
                            (ulong)p.groups_per_row +
                        (ulong)group];
            } else {
                activation_scales[tid] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto activation = tA.slice(0, 0);
        auto weight = tW.slice(0, 0);
        mm.run(activation, weight, dot);

        #pragma unroll
        for (int slot = 0; slot < ACC_PER_THREAD; ++slot) {
            const auto index =
                dot.get_multidimensional_index(slot);
            accum[slot] +=
                (float)dot[slot] *
                activation_scales[index[1]] *
                weight_scales[index[0]];
        }

        if (group + 1 == p.groups_per_row) {
            if constexpr (PAIRED_GATE_UP) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                threadgroup float* paired =
                    (threadgroup float*)shmem;
                #pragma unroll
                for (int slot = 0;
                     slot < ACC_PER_THREAD; ++slot) {
                    const auto index =
                        dot.get_multidimensional_index(slot);
                    paired[index[0] * NRB + index[1]] =
                        accum[slot];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int work = (int)tid;
                     work < NRA * NRB;
                     work += NUM_THREADS) {
                    const int route_local = work / NRA;
                    const int output_local =
                        work - route_local * NRA;
                    const int route_slot =
                        route_begin + route_local;
                    const int output_row =
                        row_begin + output_local;
                    if (route_slot < route_count &&
                        output_row < p.N) {
                        const float gate =
                            paired[
                                output_local * NRB +
                                route_local];
                        const float up =
                            paired[
                                (NRA + output_local) *
                                    NRB +
                                route_local];
                        const int selection =
                            expert_routes[
                                (ulong)expert *
                                    (ulong)p.max_routes +
                                (ulong)route_slot];
                        C[(ulong)selection *
                              (ulong)p.c_row_stride +
                          (ulong)output_row] =
                            (gate /
                             (1.0f + exp(-gate))) * up;
                    }
                }
            } else {
                #pragma unroll
                for (int slot = 0;
                     slot < ACC_PER_THREAD; ++slot) {
                    const auto index =
                        dot.get_multidimensional_index(slot);
                    const int route_local = index[1];
                    const int route_slot =
                        route_begin + route_local;
                    const int output_row =
                        row_begin + index[0];
                    if (route_slot < route_count &&
                        output_row < p.N) {
                        const int selection =
                            expert_routes[
                                (ulong)expert *
                                    (ulong)p.max_routes +
                                (ulong)route_slot];
                        C[(ulong)selection *
                              (ulong)p.c_row_stride +
                          (ulong)output_row] =
                            accum[slot];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define MOLLM_BG32_GROUPED_KERNEL(NAME, NRA, NRB, PAIRED)                 \
kernel void NAME(                                                         \
    device const int8_t* A [[buffer(0)]],                                 \
    device const uint8_t* B [[buffer(1)]],                                \
    device float* C [[buffer(2)]],                                        \
    constant GroupedW4A8Params& p [[buffer(3)]],                          \
    device const float* SCALE_A [[buffer(4)]],                            \
    device const atomic_uint* expert_counts [[buffer(5)]],                \
    device const int* expert_routes [[buffer(6)]],                        \
    device const uint2* grouped_jobs [[buffer(7)]],                       \
    threadgroup int8_t* shmem [[threadgroup(0)]],                         \
    uint3 tg [[threadgroup_position_in_grid]],                            \
    ushort tid [[thread_index_in_threadgroup]]) {                         \
    gemm_grouped_experts_bg32_impl<NRA, NRB, PAIRED>(                     \
        A, B, C, p, SCALE_A, expert_counts, expert_routes,                \
        grouped_jobs, shmem, tg, tid);                                    \
}

MOLLM_BG32_GROUPED_KERNEL(
    gemm_grouped_experts_bg32_gate_up_r16,
    MOLLM_GROUPED_MOE_GATE_UP_OUTPUT_TILE,
    MOLLM_GROUPED_MOE_ROUTE_TILE_SMALL, true)
MOLLM_BG32_GROUPED_KERNEL(
    gemm_grouped_experts_bg32_gate_up_r32,
    MOLLM_GROUPED_MOE_GATE_UP_OUTPUT_TILE,
    MOLLM_GROUPED_MOE_ROUTE_TILE_LARGE, true)
MOLLM_BG32_GROUPED_KERNEL(
    gemm_grouped_experts_bg32_down_r16,
    MOLLM_GROUPED_MOE_DOWN_OUTPUT_TILE,
    MOLLM_GROUPED_MOE_ROUTE_TILE_SMALL, false)
MOLLM_BG32_GROUPED_KERNEL(
    gemm_grouped_experts_bg32_down_r32,
    MOLLM_GROUPED_MOE_DOWN_OUTPUT_TILE,
    MOLLM_GROUPED_MOE_ROUTE_TILE_LARGE, false)

#undef MOLLM_BG32_GROUPED_KERNEL

#endif // MOLLM_METAL_TENSOR

// ===========================================================================
// mollm Metal compute kernels — phase 1 (Qwen3 dense, FP16 weights)
//
// Convention mirrors the CPU path:
//   - activations / hidden state are FP32
//   - weights (matmul B) are FP16
//   - accumulation is FP32 (correctness first; PPL should be <= CPU FP16 path)
// Buffers are bound at their tensor's device_offset; the *_offset fields in the
// params structs are additional ELEMENT offsets from that bound base (usually 0
// for freshly-allocated outputs, non-zero for views).
// ===========================================================================

inline float apply_activation(float v, int act) {
    if (act == 1) {  // SILU: x * sigmoid(x)
        return v / (1.0f + exp(-v));
    }
    if (act == 2) {  // GELU tanh approximation
        float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
        return 0.5f * v * (1.0f + precise::tanh(inner));
    }
    if (act == 3) return max(v, 0.0f);
    if (act == 4) {
        float y = max(v, 0.0f);
        return y * y;
    }
    return v;
}

inline float apply_activation_at(float v, int act, int n, int begin, int len) {
    bool in_range = len < 0 || (len > 0 && n >= begin && n < begin + len);
    return in_range ? apply_activation(v, act) : v;
}

// ---------------------------------------------------------------------------
// Tiled GEMM (simdgroup 8x8 matrix instructions) — fallback when the tensor
// path is unavailable. C[M,N] = A[M,K] (fp32) * B[N,K]^T (fp16).
//   A element (m,k) at a_offset + m*a_row_stride + k
//   B element (n,k) at b_offset + n*b_row_stride + k
//   C element (m,n) at c_offset + m*c_row_stride + n
// 32x32 output tile / threadgroup, 4 simdgroups (2x2 grid of simdgroup_float8x8
// accumulators each), K streamed in chunks of 8. A/B tiles staged as half
// (activations downcast on load) with fp32 accumulation; B loaded transposed.
// B is bound at its 64-bit byte offset (p.b_offset==0) to avoid uint32
// element-offset overflow over the multi-GB weight region.
// ---------------------------------------------------------------------------
kernel void gemm_tiled_f32a_f16b_f32c(
    device const float*   A      [[buffer(0)]],
    device const half*    B      [[buffer(1)]],
    device float*         C      [[buffer(2)]],
    constant MatmulParams& p     [[buffer(3)]],
    threadgroup half*     shmem  [[threadgroup(0)]],
    uint3  tgpig                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]])
{
    const int TM = 32, TN = 32, TK = 8;    // TK=8 empirically best (TK=32 lowers occupancy)
    threadgroup half* sa = shmem;            // A tile [TM][TK] = 256 halves
    threadgroup half* sb = shmem + TM*TK;    // B tile [TN][TK] = 256 halves

    int m0 = (int)tgpig.y * TM;
    int n0 = (int)tgpig.x * TN;

    int sm = (sgitg & 1) * 16;    // M sub-origin (0 or 16)
    int sn = (sgitg >> 1) * 16;   // N sub-origin (0 or 16)

    simdgroup_float8x8 acc[4];
    for (int i = 0; i < 4; i++) acc[i] = make_filled_simdgroup_matrix<float,8>(0.0f);

    for (int k0 = 0; k0 < p.K; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = tiitg; t < TM*TK; t += 128) {
            int r = t / TK, c = t % TK;
            int gm = m0 + r, gk = k0 + c;
            sa[r*TK + c] = (gm < p.M && gk < p.K)
                ? (half)A[p.a_offset + (uint)gm * p.a_row_stride + gk] : (half)0;
        }
        for (int t = tiitg; t < TN*TK; t += 128) {
            int r = t / TK, c = t % TK;
            int gn = n0 + r, gk = k0 + c;
            sb[r*TK + c] = (gn < p.N && gk < p.K)
                ? B[p.b_offset + (uint)gn * p.b_row_stride + gk] : (half)0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner K-loop: TK/8 simdgroup MMA steps per staged chunk.
        simdgroup_half8x8 ma, mb;
        for (int kk = 0; kk < TK; kk += 8) {
            for (int mi = 0; mi < 2; mi++) {
                for (int ni = 0; ni < 2; ni++) {
                    simdgroup_load(ma, sa + (sm + 8*mi)*TK + kk, TK, 0, false);
                    simdgroup_load(mb, sb + (sn + 8*ni)*TK + kk, TK, 0, true);
                    simdgroup_multiply_accumulate(acc[mi*2+ni], ma, mb, acc[mi*2+ni]);
                }
            }
        }
    }

    threadgroup float* scratch = (threadgroup float*)shmem;
    for (int mi = 0; mi < 2; mi++) {
        for (int ni = 0; ni < 2; ni++) {
            int cm = m0 + sm + 8*mi;
            int cn = n0 + sn + 8*ni;
            if (cm >= p.M || cn >= p.N) continue;
            if (cm + 8 <= p.M && cn + 8 <= p.N) {
                simdgroup_store(acc[mi*2+ni],
                    C + p.c_offset + (uint)cm * p.c_row_stride + cn,
                    p.c_row_stride, 0, false);
            } else {
                threadgroup float* sc = scratch + sgitg*64;
                simdgroup_store(acc[mi*2+ni], sc, 8, 0, false);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if ((tiitg & 31) == 0) {
                    for (int rr = 0; rr < 8 && cm+rr < p.M; rr++)
                        for (int cc = 0; cc < 8 && cn+cc < p.N; cc++)
                            C[p.c_offset + (uint)(cm+rr) * p.c_row_stride + (cn+cc)]
                                = sc[rr*8 + cc];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tiled GEMM, 32(M) x 64(N) half-staged — WIDER N strip per simdgroup to raise
// the MMA:load ratio (the 32x32 kernel is MMA-throughput/occupancy bound at only
// ~16% of FP16 peak). Each of 4 simdgroups owns 16(M) x 32(N) = 2 M-blocks x 4
// N-blocks = acc[8]. Per K-step a sg loads 2 A-blocks + 4 B-blocks (6 loads) and
// issues 8 MMAs (ratio 1.33 vs 1.0 for the 2x2 kernel), better hiding load
// latency. A/B staged as half; fp32 accumulate. N is the big (weight) dim.
// threadgroup(0): (32*8 + 64*8) halves = 768 halves = 1.5KB; FP32 edge scratch
// (4 sg * 8 blocks * 64 floats... reuse per-sg slot of 64 floats) fits.
// grid: threadgroups = (ceil(N/64), ceil(M/32)); threads/tg = 128.
// ---------------------------------------------------------------------------
kernel void gemm_tiledN64_f32a_f16b_f32c(
    device const float*   A       [[buffer(0)]],
    device const half*    B       [[buffer(1)]],
    device float*         C       [[buffer(2)]],
    constant MatmulParams& p      [[buffer(3)]],
    threadgroup half*     shmem   [[threadgroup(0)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]],
    ushort sgitg                  [[simdgroup_index_in_threadgroup]])
{
    const int TM = 32, TN = 64, TK = 8;
    threadgroup half* sa = shmem;            // A tile [TM][TK] = 256 halves
    threadgroup half* sb = shmem + TM*TK;    // B tile [TN][TK] = 512 halves

    int m0 = (int)tgpig.y * TM;
    int n0 = (int)tgpig.x * TN;

    int sm = (sgitg & 1) * 16;    // M sub-origin (0 or 16) — 2 M-blocks
    int sn = (sgitg >> 1) * 32;   // N sub-origin (0 or 32) — 4 N-blocks

    simdgroup_float8x8 acc[8];    // 2(M) x 4(N)
    for (int i = 0; i < 8; i++) acc[i] = make_filled_simdgroup_matrix<float,8>(0.0f);

    for (int k0 = 0; k0 < p.K; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = tiitg; t < TM*TK; t += 128) {
            int r = t / TK, c = t % TK;
            int gm = m0 + r, gk = k0 + c;
            sa[r*TK + c] = (gm < p.M && gk < p.K)
                ? (half)A[p.a_offset + (uint)gm * p.a_row_stride + gk] : (half)0;
        }
        for (int t = tiitg; t < TN*TK; t += 128) {
            int r = t / TK, c = t % TK;
            int gn = n0 + r, gk = k0 + c;
            sb[r*TK + c] = (gn < p.N && gk < p.K)
                ? B[p.b_offset + (uint)gn * p.b_row_stride + gk] : (half)0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 ma[2], mb[4];
        for (int mi = 0; mi < 2; mi++)
            simdgroup_load(ma[mi], sa + (sm + 8*mi)*TK, TK, 0, false);
        for (int ni = 0; ni < 4; ni++)
            simdgroup_load(mb[ni], sb + (sn + 8*ni)*TK, TK, 0, true);
        for (int mi = 0; mi < 2; mi++)
            for (int ni = 0; ni < 4; ni++)
                simdgroup_multiply_accumulate(acc[mi*4+ni], ma[mi], mb[ni], acc[mi*4+ni]);
    }

    threadgroup float* scratch = (threadgroup float*)shmem;
    for (int mi = 0; mi < 2; mi++) {
        for (int ni = 0; ni < 4; ni++) {
            int cm = m0 + sm + 8*mi;
            int cn = n0 + sn + 8*ni;
            if (cm >= p.M || cn >= p.N) continue;
            if (cm + 8 <= p.M && cn + 8 <= p.N) {
                simdgroup_store(acc[mi*4+ni],
                    C + p.c_offset + (uint)cm * p.c_row_stride + cn,
                    p.c_row_stride, 0, false);
            } else {
                threadgroup float* sc = scratch + sgitg*64;
                simdgroup_store(acc[mi*4+ni], sc, 8, 0, false);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if ((tiitg & 31) == 0) {
                    for (int rr = 0; rr < 8 && cm+rr < p.M; rr++)
                        for (int cc = 0; cc < 8 && cn+cc < p.N; cc++)
                            C[p.c_offset + (uint)(cm+rr) * p.c_row_stride + (cn+cc)]
                                = sc[rr*8 + cc];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tiled GEMM, 64(M) x 32(N) tile — larger tile for higher prefill throughput.
// Same math as the 32x32 kernel:
// C[M,N] = A[M,K](fp32) * B[N,K]^T(fp16). 4 simdgroups, 128 threads. Each sg
// owns a 32(M) x 16(N) region = 4x2 grid of simdgroup_float8x8 accumulators
// (8 total per sg). A tile [64 M x 8 K], B tile [32 N x 8 K] staged as float.
//
// Store: to avoid the M-edge bug that sank the earlier attempt, ALL 8 blocks
// are handled uniformly — full 8x8 blocks go straight to C via simdgroup_store;
// partial (edge) blocks go through a per-simdgroup private scratch slot with a
// single simdgroup_barrier reached by all lanes, then a bounds-checked copy.
// The scratch is a SEPARATE threadgroup buffer (buffer index 1), never aliasing
// the staging region.
// threadgroup(0): staging = (64*8 + 32*8) floats = 768 floats = 3KB.
// threadgroup(1): store scratch = 4 sg * 64 floats = 256 floats = 1KB.
// grid: threadgroups = (ceil(N/32), ceil(M/64)); threads/tg = 128.
// ---------------------------------------------------------------------------
kernel void gemm_tiled64_f32a_f16b_f32c(
    device const float*   A       [[buffer(0)]],
    device const half*    B       [[buffer(1)]],
    device float*         C       [[buffer(2)]],
    constant MatmulParams& p      [[buffer(3)]],
    threadgroup float*    shmem   [[threadgroup(0)]],
    threadgroup float*    scratch [[threadgroup(1)]],
    uint3  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiitg                  [[thread_index_in_threadgroup]],
    ushort sgitg                  [[simdgroup_index_in_threadgroup]])
{
    const int TM = 64, TN = 32, TK = 8;
    threadgroup float* sa = shmem;            // A tile [TM][TK] = 512 floats
    threadgroup float* sb = shmem + TM*TK;    // B tile [TN][TK] = 256 floats

    int m0 = (int)tgpig.y * TM;
    int n0 = (int)tgpig.x * TN;

    // Each sg owns 32 M-rows (4 blocks of 8) x 16 N-cols (2 blocks of 8).
    int sm = (sgitg & 1) * 32;    // M sub-origin (0 or 32)
    int sn = (sgitg >> 1) * 16;   // N sub-origin (0 or 16)

    simdgroup_float8x8 acc[8];    // 4 (M) x 2 (N)
    for (int i = 0; i < 8; i++) acc[i] = make_filled_simdgroup_matrix<float,8>(0.0f);

    for (int k0 = 0; k0 < p.K; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = tiitg; t < TM*TK; t += 128) {
            int r = t / TK, c = t % TK;
            int gm = m0 + r, gk = k0 + c;
            sa[r*TK + c] = (gm < p.M && gk < p.K)
                ? A[p.a_offset + (uint)gm * p.a_row_stride + gk] : 0.0f;
        }
        for (int t = tiitg; t < TN*TK; t += 128) {
            int r = t / TK, c = t % TK;
            int gn = n0 + r, gk = k0 + c;
            sb[r*TK + c] = (gn < p.N && gk < p.K)
                ? float(B[p.b_offset + (uint)gn * p.b_row_stride + gk]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 ma, mb;
        for (int mi = 0; mi < 4; mi++) {
            simdgroup_load(ma, sa + (sm + 8*mi)*TK, TK, 0, false);
            for (int ni = 0; ni < 2; ni++) {
                simdgroup_load(mb, sb + (sn + 8*ni)*TK, TK, 0, true);
                simdgroup_multiply_accumulate(acc[mi*2+ni], ma, mb, acc[mi*2+ni]);
            }
        }
    }

    for (int mi = 0; mi < 4; mi++) {
        for (int ni = 0; ni < 2; ni++) {
            int cm = m0 + sm + 8*mi;
            int cn = n0 + sn + 8*ni;
            if (cm >= p.M || cn >= p.N) continue;
            if (cm + 8 <= p.M && cn + 8 <= p.N) {
                simdgroup_store(acc[mi*2+ni],
                    C + p.c_offset + (uint)cm * p.c_row_stride + cn,
                    p.c_row_stride, 0, false);
            } else {
                threadgroup float* sc = scratch + sgitg*64;
                simdgroup_store(acc[mi*2+ni], sc, 8, 0, false);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if ((tiitg & 31) == 0) {
                    for (int rr = 0; rr < 8 && cm+rr < p.M; rr++)
                        for (int cc = 0; cc < 8 && cn+cc < p.N; cc++)
                            C[p.c_offset + (uint)(cm+rr) * p.c_row_stride + (cn+cc)]
                                = sc[rr*8 + cc];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GEMM: C[M,N] = A[M,K] (fp32) * B[N,K]^T (fp16)   (B row-major [N,K])
// Naive one-thread-per-output version (correctness/parity first; M5 tiles it).
// grid = (N, M)
// ---------------------------------------------------------------------------
kernel void gemm_f32a_f16b_f32c(
    device const float*   A      [[buffer(0)]],
    device const half*    B      [[buffer(1)]],
    device float*         C      [[buffer(2)]],
    constant MatmulParams& p     [[buffer(3)]],
    uint  gid                    [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= p.M * p.N) return;
    int m = idx / p.N;
    int n = idx % p.N;

    device const float* a_row = A + p.a_offset + (uint)m * p.a_row_stride;
    device const half4* b4 = (device const half4*)(B + p.b_offset + (uint)n * p.b_row_stride);
    device const float4* a4 = (device const float4*)a_row;

    float acc = 0.0f;
    int K4 = p.K >> 2;
    for (int q = 0; q < K4; q++) {
        float4 av = a4[q];
        half4  bv = b4[q];
        acc += av.x*float(bv.x) + av.y*float(bv.y) + av.z*float(bv.z) + av.w*float(bv.w);
    }
    for (int k = (K4<<2); k < p.K; k++) acc += a_row[k] * float(B[p.b_offset + (uint)n*p.b_row_stride + k]);
    acc = apply_activation_at(acc, p.activation, n, p.act_n_begin, p.act_n_len);
    C[p.c_offset + (uint)m * p.c_row_stride + n] = acc;
}

// ---------------------------------------------------------------------------
// GEMV v2: M==1 decode fast path. C[1,N] = A[1,K] * B[N,K]^T.
//   - Each threadgroup owns NR0=2 output rows; the activation slice each lane
//     loads is reused across both rows (halves activation reads).
//   - NSG simdgroups split K into strided chunks and accumulate partial dots,
//     combined by a cross-SG reduction (simd_sum + shmem). This spreads large K
//     (e.g. down_proj K=9728) across up to NSG*32 lanes instead of 32.
//   - Weight and activation both read straight from device (no staging).
// grid: (ceil(N/NR0), 1, 1); threads/tg = (32, NSG, 1); shmem NR0*32 floats.
// NR0 (output rows per threadgroup) is a function constant (default 2); the host
// picks it per GPU/model. Max supported = 8 (register array sumf[NR0MAX]).
// ---------------------------------------------------------------------------
constant int FC_GEMV_NR0 [[function_constant(5)]];
constant bool FC_GEMV_HAS_NR0 = is_function_constant_defined(FC_GEMV_NR0);

kernel void gemv2_f32a_f16b_f32c(
    device const float*   A      [[buffer(0)]],
    device const half*    B      [[buffer(1)]],
    device float*         C      [[buffer(2)]],
    constant MatmulParams& p     [[buffer(3)]],
    threadgroup float*    shmem  [[threadgroup(0)]],
    uint  tgx                    [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]],
    ushort nsg                   [[simdgroups_per_threadgroup]])
{
    const short NR0 = FC_GEMV_HAS_NR0 ? (short)FC_GEMV_NR0 : (short)2;
    const short NR0MAX = 8;
    const short NW  = 32;
    const short NF  = 8;             // elements consumed per lane per step

    device const float* a = A + p.a_offset;
    const int r0 = (int)tgx * NR0;

    // Row base pointers into B (weight), row stride = b_row_stride (=K).
    device const half* bx[NR0MAX];
    for (short row = 0; row < NR0; ++row)
        bx[row] = B + p.b_offset + (uint)(r0 + row) * (uint)p.b_row_stride;

    float sumf[NR0MAX];
    for (short row = 0; row < NR0; ++row) sumf[row] = 0.0f;

    // Each lane owns NF contiguous elements within a 32*NF-wide chunk; SGs stride
    // over chunks. The activation slice av[] is loaded once and reused across all
    // NR0 rows (the memory-bandwidth win).
    const int NB = NW * NF;                 // elements per chunk (=256)
    const int nb = p.K / NB;                // whole chunks
    device const float* ay = a + (int)lane * NF;
    for (int ib = (int)sgitg; ib < nb; ib += nsg) {
        int base = ib * NB;
        // Vectorized loads: activation as 2x float4, weight as half8 (NF=8).
        // Addresses are 16-byte aligned (lane*8 halves = lane*16 bytes; base*2,
        // b_row_stride*2 both mult of 16), so vector loads are safe.
        float4 av0 = *(device const float4*)(ay + base);
        float4 av1 = *(device const float4*)(ay + base + 4);
        for (short row = 0; row < NR0; ++row) {
            device const half* by = bx[row] + base + (int)lane * NF;
            half4 b0 = *(device const half4*)(by);
            half4 b1 = *(device const half4*)(by + 4);
            float4 p0 = av0 * float4(b0);
            float4 p1 = av1 * float4(b1);
            sumf[row] += (p0.x+p0.y+p0.z+p0.w) + (p1.x+p1.y+p1.z+p1.w);
        }
    }
    // Tail (K not a multiple of NB): each thread strides remaining elements.
    for (int k = nb * NB + (int)sgitg * NW + (int)lane; k < p.K; k += NW * nsg) {
        float aval = a[k];
        for (short row = 0; row < NR0; ++row)
            sumf[row] += aval * float(bx[row][k]);
    }

    // ---- cross-simdgroup reduction: combine each SG's partial dots ----
    // shmem laid out [NR0][NW]; sg0 reduces the per-SG partials for each row.
    for (short row = 0; row < NR0; ++row) {
        if (sgitg == 0) shmem[row * NW + lane] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row)
        if (lane == 0) shmem[row * NW + sgitg] = sumf[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        for (short row = 0; row < NR0; ++row) {
            float tot = simd_sum(shmem[row * NW + lane]);
            if (lane == 0 && r0 + row < p.N)
                C[p.c_offset + r0 + row] = apply_activation_at(
                    tot, p.activation, r0 + row, p.act_n_begin, p.act_n_len);
        }
    }
}

// Small-M FP16-weight projection.  Used primarily by MoE routers during
// speculative verification, where M=2..4 is too small for a 64-row tensor
// tile.  Each SIMD group owns one output row and reuses every weight vector
// load across all activation rows.
constant int FC_SMALL_M [[function_constant(12)]];
constant bool FC_SMALL_M_DEFINED =
    is_function_constant_defined(FC_SMALL_M);

kernel void gemv_small_m_f32a_f16b_f32c(
    device const float* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint tgx [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]])
{
    const short MMAX = 4;
    const short M = FC_SMALL_M_DEFINED
        ? (short)FC_SMALL_M : (short)p.M;
    const int row = (int)tgx * (int)nsg + (int)sgitg;
    if (row >= p.N) return;
    device const half* weights =
        B + p.b_offset + (uint)row * (uint)p.b_row_stride;
    float sums[MMAX];
    for (short m = 0; m < M; ++m) sums[m] = 0.0f;

    constexpr int VEC = 4;
    const int vector_end = p.K & ~(VEC - 1);
    for (int k = (int)lane * VEC; k < vector_end; k += 32 * VEC) {
        const half4 weight = *((device const half4*)(weights + k));
        for (short m = 0; m < M; ++m) {
            device const float* activation =
                A + p.a_offset + (uint)m * (uint)p.a_row_stride;
            sums[m] += dot(
                *((device const float4*)(activation + k)),
                float4(weight));
        }
    }
    for (int k = vector_end + (int)lane; k < p.K; k += 32) {
        const float weight = (float)weights[k];
        for (short m = 0; m < M; ++m) {
            device const float* activation =
                A + p.a_offset + (uint)m * (uint)p.a_row_stride;
            sums[m] += activation[k] * weight;
        }
    }
    for (short m = 0; m < M; ++m) {
        const float total = simd_sum(sums[m]);
        if (lane == 0) {
            C[p.c_offset + (uint)m * (uint)p.c_row_stride + (uint)row] =
                apply_activation_at(
                    total, p.activation, row,
                    p.act_n_begin, p.act_n_len);
        }
    }
}

// ---------------------------------------------------------------------------
// W8 decode GEMV: C[1,N] = (A[1,K] fp32) * (W[N,K] int8) with per-group weight
// scale. int8 weight × float activation (no activation quant); the group scale
// is applied in the K accumulation so any group_size works. Same threadgroup
// structure as gemv2 (NR0 rows/tg, NSG simdgroups split K, shmem reduce).
// grid: (ceil(N/NR0),1,1); threads (32, NSG, 1); shmem NR0*32 floats.
// ---------------------------------------------------------------------------
kernel void gemv_w8_f32a_i8b_f32c(
    device const float*    A      [[buffer(0)]],
    device const int8_t*   B      [[buffer(1)]],
    device float*          C      [[buffer(2)]],
    device const float*    SCALES [[buffer(4)]],
    constant MatmulW8Params& p    [[buffer(3)]],
    threadgroup float*     shmem  [[threadgroup(0)]],
    uint  tgx                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort sgitg                  [[simdgroup_index_in_threadgroup]],
    ushort nsg                    [[simdgroups_per_threadgroup]])
{
    const short NR0 = 2, NR0MAX = 2, NW = 32;
    device const float* a = A + p.a_offset;
    const int r0 = (int)tgx * NR0;
    const int gpr = p.groups_per_row, gs = p.group_size;

    device const int8_t* bx[NR0MAX];
    device const float*  sc[NR0MAX];
    for (short row = 0; row < NR0; ++row) {
        bx[row] = B + (uint)(r0 + row) * (uint)p.K;               // int8 weight row
        sc[row] = SCALES + (uint)(r0 + row) * (uint)gpr;          // per-row group scales
    }
    float sumf[NR0MAX]; for (short r=0;r<NR0;++r) sumf[r]=0.0f;

    // Each lane strides K; scale applied per group boundary. w8pc (gpr==1) →
    // one scale/row, factored out at the end. W8G128 gets its own vector path:
    // one SIMD group covers one complete 128-value quantization group, avoiding
    // a scalar weight load and a runtime integer division for every element.
    if (gpr == 1) {
        const int NF = 8;
        const int NB = NW * NF;
        const int nb = p.K / NB;
        for (int ib = (int)sgitg; ib < nb; ib += nsg) {
            int k = ib * NB + (int)lane * NF;
            float4 av0 = *(device const float4*)(a + k);
            float4 av1 = *(device const float4*)(a + k + 4);
            for (short r=0;r<NR0;++r) {
                char4 w0 = *(device const char4*)(bx[r] + k);
                char4 w1 = *(device const char4*)(bx[r] + k + 4);
                sumf[r] += dot(av0, float4(w0)) + dot(av1, float4(w1));
            }
        }
        for (int k = nb * NB + (int)sgitg*NW + (int)lane;
             k < p.K; k += NW*(int)nsg) {
            float av = a[k];
            for (short r=0;r<NR0;++r)
                sumf[r] += av * (float)bx[r][k];
        }
        for (short r=0;r<NR0;++r) sumf[r] *= sc[r][0];            // one scale per row
    } else if (gs == 128 && (p.K & 127) == 0) {
        for (int g = (int)sgitg; g < gpr; g += (int)nsg) {
            const int k = g * 128 + (int)lane * 4;
            const float4 av =
                *(device const float4*)(a + k);
            for (short r = 0; r < NR0; ++r) {
                const char4 wv =
                    *(device const char4*)(bx[r] + k);
                sumf[r] += dot(av, float4(wv)) * sc[r][g];
            }
        }
    } else {
        for (int k = (int)sgitg*NW + (int)lane; k < p.K; k += NW*(int)nsg) {
            float av = a[k]; int g = k / gs;
            for (short r=0;r<NR0;++r) sumf[r] += av * ((float)bx[r][k]) * sc[r][g];
        }
    }

    // cross-simdgroup reduction (same as gemv2)
    for (short r=0;r<NR0;++r){ if(sgitg==0) shmem[r*NW+lane]=0.0f; sumf[r]=simd_sum(sumf[r]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short r=0;r<NR0;++r) if(lane==0) shmem[r*NW+sgitg]=sumf[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg==0) {
        for (short r=0;r<NR0;++r){
            float tot = simd_sum(shmem[r*NW+lane]);
            if (lane==0 && r0+r < p.N) C[p.c_offset + r0+r] = apply_activation_at(
                tot, p.activation, r0+r, p.act_n_begin, p.act_n_len);
        }
    }
}

// Small-M W8 projection for incremental prefill/speculative verification.
// One SIMD group owns one output row and reuses every int8 weight load across
// up to four activation rows. This avoids a mostly-empty tensor-GEMM tile for
// M=2..4 while scanning the weight matrix only once for the token batch.
kernel void gemv_w8_small_m_f32a_i8b_f32c(
    device const float*    A      [[buffer(0)]],
    device const int8_t*   B      [[buffer(1)]],
    device float*          C      [[buffer(2)]],
    device const float*    SCALES [[buffer(4)]],
    constant MatmulW8Params& p    [[buffer(3)]],
    uint  tgx                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort sgitg                  [[simdgroup_index_in_threadgroup]],
    ushort nsg                    [[simdgroups_per_threadgroup]])
{
    const short MMAX = 4;
    const short M = FC_SMALL_M_DEFINED
        ? (short)FC_SMALL_M : (short)p.M;
    const int row = (int)tgx * (int)nsg + (int)sgitg;
    if (row >= p.N) return;

    device const int8_t* weights = B + (uint)row * (uint)p.K;
    device const float* scales =
        SCALES + (uint)row * (uint)p.groups_per_row;
    float sums[MMAX];
    for (short m = 0; m < M; ++m) sums[m] = 0.0f;

    if (p.group_size == 128 && (p.K & 127) == 0) {
        // One lane owns four adjacent weights in each G128 block.  Iterating
        // by group makes the scale index explicit and lets the compiler remove
        // the division from this bandwidth-sensitive verification path.
        for (int g = 0; g < p.groups_per_row; ++g) {
            const int k = g * 128 + (int)lane * 4;
            const char4 weight =
                *((device const char4*)(weights + k));
            const float4 dequantized = float4(weight) * scales[g];
            for (short m = 0; m < M; ++m) {
                device const float* activation =
                    A + p.a_offset +
                    (uint)m * (uint)p.a_row_stride;
                sums[m] += dot(
                    *((device const float4*)(activation + k)),
                    dequantized);
            }
        }
    } else {
        constexpr int VEC = 4;
        const int vector_end = p.K & ~(VEC - 1);
        for (int k = (int)lane * VEC;
             k < vector_end; k += 32 * VEC) {
            const char4 weight =
                *((device const char4*)(weights + k));
            const float scale = scales[k / p.group_size];
            const float4 dequantized = float4(weight) * scale;
            for (short m = 0; m < M; ++m) {
                device const float* activation =
                    A + p.a_offset +
                    (uint)m * (uint)p.a_row_stride;
                sums[m] += dot(
                    *((device const float4*)(activation + k)),
                    dequantized);
            }
        }
        for (int k = vector_end + (int)lane;
             k < p.K; k += 32) {
            const float weight =
                (float)weights[k] * scales[k / p.group_size];
            for (short m = 0; m < M; ++m) {
                device const float* activation =
                    A + p.a_offset +
                    (uint)m * (uint)p.a_row_stride;
                sums[m] += activation[k] * weight;
            }
        }
    }

    for (short m = 0; m < M; ++m) {
        const float total = simd_sum(sums[m]);
        if (lane == 0) {
            C[p.c_offset + (uint)m * (uint)p.c_row_stride + (uint)row] =
                apply_activation_at(
                    total, p.activation, row,
                    p.act_n_begin, p.act_n_len);
        }
    }
}

// W4 decode GEMV: C[1,N] = A[1,K] (fp32) * W_i4[N,K/2] (per-group symmetric int4).
// Nibble: byte = B[n*(K/2)+k/2]; low = even k, high = odd k; w = nibble-16 if>=8.
// Per-group weight scale scale_w[n*gpr + k/gs] applied inside the group sum.
constant int FC_GEMV_W4_NR0 [[function_constant(6)]];
constant bool FC_GEMV_W4_HAS_NR0 =
    is_function_constant_defined(FC_GEMV_W4_NR0);
kernel void gemv_w4_f32a_i4b_f32c(
    device const float*    A      [[buffer(0)]],
    device const uint8_t*  B      [[buffer(1)]],
    device float*          C      [[buffer(2)]],
    device const float*    SCALES [[buffer(4)]],
    constant MatmulW8Params& p    [[buffer(3)]],
    threadgroup float*     shmem  [[threadgroup(0)]],
    uint  tgx                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort sgitg                  [[simdgroup_index_in_threadgroup]],
    ushort nsg                    [[simdgroups_per_threadgroup]])
{
    const short NR0 =
        FC_GEMV_W4_HAS_NR0 ? (short)FC_GEMV_W4_NR0 : (short)2;
    const short NR0MAX = 8, NW = 32;
    device const float* a = A + p.a_offset;
    // Match llama.cpp's row-parallel scheduling: every SIMD group owns an
    // independent NR0-row tile and scans the full K dimension. This avoids
    // cross-SIMD reductions/barriers and exposes NSG*NR0 rows per threadgroup.
    const int r0 = ((int)tgx * (int)nsg + (int)sgitg) * NR0;
    const int gpr = p.groups_per_row, gs = p.group_size;
    const uint row_bytes = (uint)p.K / 2;   // K/2 bytes per weight row

    device const uint8_t* bx[NR0MAX];
    device const float*   sc[NR0MAX];
    for (short row = 0; row < NR0; ++row) {
        bx[row] = B + (uint)(r0 + row) * row_bytes;
        sc[row] = SCALES + (uint)(r0 + row) * (uint)gpr;
    }
    float sumf[NR0MAX]; for (short r=0;r<NR0;++r) sumf[r]=0.0f;

    if (gs == 128) {
        // Block-oriented G128 path. Eight lanes cooperate on one quantization
        // group, so a SIMD group covers four independent groups at once. Each
        // lane keeps a contiguous 16-value activation slice in registers and
        // reuses it across the output-row tile.
        const int group_lane = (int)lane >> 3;
        const int lane_in_group = (int)lane & 7;
        for (int g=group_lane; g<gpr; g+=4) {
            const int k0=g*128+lane_in_group*16;
            const int kb=g*64+lane_in_group*8;
            const float4 av0=*((device const float4*)(a+k0));
            const float4 av1=*((device const float4*)(a+k0+4));
            const float4 av2=*((device const float4*)(a+k0+8));
            const float4 av3=*((device const float4*)(a+k0+12));
            const float4 ae0=float4(av0.xz,av1.xz);
            const float4 ao0=float4(av0.yw,av1.yw);
            const float4 ae1=float4(av2.xz,av3.xz);
            const float4 ao1=float4(av2.yw,av3.yw);
            const float activation_sum=
                dot(av0,float4(1.0f))+dot(av1,float4(1.0f))+
                dot(av2,float4(1.0f))+dot(av3,float4(1.0f));
            for (short r=0;r<NR0;++r) {
                const uchar4 q0=*((device const uchar4*)(bx[r]+kb));
                const uchar4 q1=*((device const uchar4*)(bx[r]+kb+4));
                const int4 lo0=int4(q0&uchar4(15));
                const int4 hi0=int4(q0>>4);
                const int4 lo1=int4(q1&uchar4(15));
                const int4 hi1=int4(q1>>4);
                const float dotv=dot(ae0,float4(lo0))+dot(ao0,float4(hi0))+
                                 dot(ae1,float4(lo1))+dot(ao1,float4(hi1))-
                                 8.0f*activation_sum;
                sumf[r] += dotv*sc[r][g];
            }
        }
    } else if (gs == 32) {
        // Eight four-lane teams cover eight independent G32 groups. Each lane
        // handles four packed bytes and reuses the group scale for eight K
        // values, avoiding the generic path's division and scalar loads.
        const int group_lane = (int)lane >> 2;
        const int lane_in_group = (int)lane & 3;
        for (int g = group_lane; g < gpr; g += 8) {
            const int ke = g * 32 + lane_in_group * 8;
            const int kb = g * 16 + lane_in_group * 4;
            const float4 av0 =
                *((device const float4*)(a + ke));
            const float4 av1 =
                *((device const float4*)(a + ke + 4));
            const float4 ae = float4(av0.xz, av1.xz);
            const float4 ao = float4(av0.yw, av1.yw);
            const float activation_sum =
                dot(av0, float4(1.0f)) +
                dot(av1, float4(1.0f));
            for (short r = 0; r < NR0; ++r) {
                const uchar4 q =
                    *((device const uchar4*)(bx[r] + kb));
                const float4 lo =
                    float4(int4(q & uchar4(15)));
                const float4 hi =
                    float4(int4(q >> 4));
                sumf[r] +=
                    (dot(ae, lo) + dot(ao, hi) -
                     8.0f * activation_sum) *
                    sc[r][g];
            }
        }
    } else {
        // Generic even-sized groups.
        for (int kb = (int)lane; kb < (int)row_bytes; kb += NW) {
            int ke = kb*2;
            float ae = a[ke], ao = a[ke+1];
            int g = kb / (gs / 2);
            for (short r=0;r<NR0;++r) {
                int byte = (int)bx[r][kb];
                int lo = (byte&15)-8, hi=((byte>>4)&15)-8;
                sumf[r] += (ae*(float)lo + ao*(float)hi)*sc[r][g];
            }
        }
    }

    for (short r=0;r<NR0;++r) {
        float tot = simd_sum(sumf[r]);
        if (lane==0 && r0+r < p.N) C[p.c_offset + r0+r] = apply_activation_at(
            tot, p.activation, r0+r, p.act_n_begin, p.act_n_len);
    }
}

// Small-M W4 projection for incremental prefill/speculative verification.
// A tensor-GEMM tile has 64 token rows, which is wasteful for M=2..4.  This
// kernel keeps the row-parallel GEMV scheduling but accumulates up to four
// activation rows while each packed weight byte and scale are resident in
// registers.  Thus the weight matrix is scanned once for the whole token
// batch instead of once per token.
kernel void gemv_w4_small_m_f32a_i4b_f32c(
    device const float*    A      [[buffer(0)]],
    device const uint8_t*  B      [[buffer(1)]],
    device float*          C      [[buffer(2)]],
    device const float*    SCALES [[buffer(4)]],
    constant MatmulW8Params& p    [[buffer(3)]],
    uint  tgx                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort sgitg                  [[simdgroup_index_in_threadgroup]],
    ushort nsg                    [[simdgroups_per_threadgroup]])
{
    const short MMAX = 4, NW = 32;
    const short M = FC_SMALL_M_DEFINED
        ? (short)FC_SMALL_M : (short)p.M;
    const int row = (int)tgx * (int)nsg + (int)sgitg;
    if (row >= p.N) return;
    const int gpr = p.groups_per_row, gs = p.group_size;
    const uint row_bytes = (uint)p.K / 2;
    device const uint8_t* weights = B + (uint)row * row_bytes;
    device const float* scales = SCALES + (uint)row * (uint)gpr;
    float sumf[MMAX];
    for (short m = 0; m < M; ++m)
        sumf[m] = 0.0f;

    if (gs == 128) {
        const int group_lane = (int)lane >> 3;
        const int lane_in_group = (int)lane & 7;
        for (int g = group_lane; g < gpr; g += 4) {
            const int k0 = g * 128 + lane_in_group * 16;
            const int kb = g * 64 + lane_in_group * 8;

            const uchar4 q0 =
                *((device const uchar4*)(weights + kb));
            const uchar4 q1 =
                *((device const uchar4*)(weights + kb + 4));
            const float4 lo0 = float4(int4(q0 & uchar4(15)));
            const float4 hi0 = float4(int4(q0 >> 4));
            const float4 lo1 = float4(int4(q1 & uchar4(15)));
            const float4 hi1 = float4(int4(q1 >> 4));
            const float scale = scales[g];
            for (short m = 0; m < M; ++m) {
                    device const float* a =
                        A + p.a_offset + (uint)m * (uint)p.a_row_stride;
                    const float4 av0 =
                        *((device const float4*)(a + k0));
                    const float4 av1 =
                        *((device const float4*)(a + k0 + 4));
                    const float4 av2 =
                        *((device const float4*)(a + k0 + 8));
                    const float4 av3 =
                        *((device const float4*)(a + k0 + 12));
                    const float4 ae0 = float4(av0.xz, av1.xz);
                    const float4 ao0 = float4(av0.yw, av1.yw);
                    const float4 ae1 = float4(av2.xz, av3.xz);
                    const float4 ao1 = float4(av2.yw, av3.yw);
                    const float activation_sum =
                        dot(av0, float4(1.0f)) +
                        dot(av1, float4(1.0f)) +
                        dot(av2, float4(1.0f)) +
                        dot(av3, float4(1.0f));
                    const float dotv =
                        dot(ae0, lo0) + dot(ao0, hi0) +
                        dot(ae1, lo1) + dot(ao1, hi1) -
                        8.0f * activation_sum;
                    sumf[m] += dotv * scale;
            }
        }
    } else if (gs == 32) {
        const int group_lane = (int)lane >> 2;
        const int lane_in_group = (int)lane & 3;
        for (int g = group_lane; g < gpr; g += 8) {
            const int ke = g * 32 + lane_in_group * 8;
            const int kb = g * 16 + lane_in_group * 4;
            const uchar4 q =
                *((device const uchar4*)(weights + kb));
            const float4 lo =
                float4(int4(q & uchar4(15)));
            const float4 hi =
                float4(int4(q >> 4));
            const float scale = scales[g];
            for (short m = 0; m < M; ++m) {
                device const float* a =
                    A + p.a_offset +
                    (uint)m * (uint)p.a_row_stride;
                const float4 av0 =
                    *((device const float4*)(a + ke));
                const float4 av1 =
                    *((device const float4*)(a + ke + 4));
                const float4 ae = float4(av0.xz, av1.xz);
                const float4 ao = float4(av0.yw, av1.yw);
                const float activation_sum =
                    dot(av0, float4(1.0f)) +
                    dot(av1, float4(1.0f));
                sumf[m] +=
                    (dot(ae, lo) + dot(ao, hi) -
                     8.0f * activation_sum) * scale;
            }
        }
    } else {
        for (int kb = (int)lane; kb < (int)row_bytes; kb += NW) {
            const int ke = kb * 2;
            const int g = kb / (gs / 2);
            const int byte = (int)weights[kb];
            const float lo = (float)((byte & 15) - 8);
            const float hi = (float)(((byte >> 4) & 15) - 8);
            const float scale = scales[g];
            for (short m = 0; m < M; ++m) {
                device const float* a =
                    A + p.a_offset +
                    (uint)m * (uint)p.a_row_stride;
                sumf[m] +=
                    (a[ke] * lo + a[ke + 1] * hi) * scale;
            }
        }
    }

    for (short m = 0; m < M; ++m) {
        const float total = simd_sum(sumf[m]);
        if (lane == 0) {
                C[p.c_offset + (uint)m * (uint)p.c_row_stride +
                  (uint)row] =
                    apply_activation_at(
                        total, p.activation, row,
                        p.act_n_begin, p.act_n_len);
        }
    }
}

inline void moe_select_sigmoid_token(
    device const float* lp, device int* top_idx, device float* top_w,
    constant MoeW4Params& p, device const float* bias, uint t) {
    float chosen[16];int indices[16];for(int k=0;k<p.top_k;k++){chosen[k]=-INFINITY;indices[k]=0;}
    const int epg=p.experts/max(p.n_group,1);float b0[16],b1[16];
    for(int g=0;g<p.n_group;g++){b0[g]=-INFINITY;b1[g]=-INFINITY;}
    for(int e=0;e<p.experts;e++){float s=1.0f/(1.0f+exp(-lp[e]))+bias[e];int g=e/max(epg,1);
        if(s>b0[g]){b1[g]=b0[g];b0[g]=s;}else if(s>b1[g])b1[g]=s;}
    bool keep[16];for(int g=0;g<p.n_group;g++)keep[g]=false;
    for(int n=0;n<p.topk_group;n++){int bg=0;float bv=-INFINITY;for(int g=0;g<p.n_group;g++)
        if(!keep[g]&&b0[g]+b1[g]>bv){bv=b0[g]+b1[g];bg=g;}keep[bg]=true;}
    for(int e=0;e<p.experts;e++){if(!keep[e/max(epg,1)])continue;float c=1.0f/(1.0f+exp(-lp[e]))+bias[e];
        for(int k=0;k<p.top_k;k++)if(c>chosen[k]){for(int j=p.top_k-1;j>k;j--){chosen[j]=chosen[j-1];indices[j]=indices[j-1];}
            chosen[k]=c;indices[k]=e;break;}}
    float sum=0.0f;for(int k=0;k<p.top_k;k++){chosen[k]=1.0f/(1.0f+exp(-lp[indices[k]]));sum+=chosen[k];}
    float mul=p.routed_scale*((p.norm_topk&&sum>0.0f)?1.0f/sum:1.0f);
    for(int k=0;k<p.top_k;k++){top_idx[t*p.top_k+k]=indices[k];top_w[t*p.top_k+k]=chosen[k]*mul;}
}

inline void moe_select_softmax_token(
    device const float* lp, device int* top_idx, device float* top_w,
    constant MoeW4Params& p, uint t) {
    float chosen[16];
    int indices[16];
    for (int k = 0; k < p.top_k; ++k) {
        chosen[k] = -INFINITY;
        indices[k] = 0;
    }
    for (int e = 0; e < p.experts; ++e) {
        const float value = lp[e];
        for (int k = 0; k < p.top_k; ++k) {
            if (value > chosen[k]) {
                for (int j = p.top_k - 1; j > k; --j) {
                    chosen[j] = chosen[j - 1];
                    indices[j] = indices[j - 1];
                }
                chosen[k] = value;
                indices[k] = e;
                break;
            }
        }
    }
    const float maximum = chosen[0];
    float sum = 0.0f;
    for (int k = 0; k < p.top_k; ++k) {
        chosen[k] = exp(chosen[k] - maximum);
        sum += chosen[k];
    }
    const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int k = 0; k < p.top_k; ++k) {
        top_idx[t * p.top_k + k] = indices[k];
        top_w[t * p.top_k + k] = chosen[k] * inverse;
    }
}

// Resident-MoE decode preparation. Router GEMV and BG128 input quantization
// are independent, so schedule both kinds of work in one dispatch. Router
// threadgroups retain the same 8-SIMD-group parallelism as gemv2; the trailing
// threadgroups quantize two 128-value blocks each. Selection remains a separate
// dispatch because it depends on all router logits.
kernel void moe_router_quantize_bg128(
    device const float* x [[buffer(0)]],
    device const half* router [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device int8_t* quantized [[buffer(4)]],
    device float* scales [[buffer(5)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    constexpr uint rows_per_group = 2;
    const uint router_groups =
        ((uint)p.experts + rows_per_group - 1u) / rows_per_group;
    device const float* input = x + p.hidden_offset;

    if (group < router_groups) {
        const uint row0 = group * rows_per_group;
        float sums[rows_per_group] = {0.0f, 0.0f};
        constexpr uint values_per_lane = 8;
        constexpr uint values_per_simd = 32 * values_per_lane;
        const uint chunks = (uint)p.hidden / values_per_simd;
        device const float* activation =
            input + (uint)lane * values_per_lane;
        for (uint chunk = (uint)sg; chunk < chunks; chunk += (uint)nsg) {
            const uint base = chunk * values_per_simd;
            const float4 av0 =
                *(device const float4*)(activation + base);
            const float4 av1 =
                *(device const float4*)(activation + base + 4);
            #pragma unroll
            for (uint r = 0; r < rows_per_group; ++r) {
                const uint row = row0 + r;
                if (row >= (uint)p.experts) continue;
                device const half* weight =
                    router + (ulong)row * (ulong)p.hidden + base +
                    (uint)lane * values_per_lane;
                const half4 w0 = *(device const half4*)weight;
                const half4 w1 = *(device const half4*)(weight + 4);
                const float4 product0 = av0 * float4(w0);
                const float4 product1 = av1 * float4(w1);
                sums[r] +=
                    (product0.x + product0.y +
                     product0.z + product0.w) +
                    (product1.x + product1.y +
                     product1.z + product1.w);
            }
        }
        for (uint k = chunks * values_per_simd +
                      (uint)sg * 32u + (uint)lane;
             k < (uint)p.hidden; k += 32u * (uint)nsg) {
            const float value = input[k];
            #pragma unroll
            for (uint r = 0; r < rows_per_group; ++r) {
                const uint row = row0 + r;
                if (row < (uint)p.experts)
                    sums[r] +=
                        value *
                        (float)router[
                            (ulong)row * (ulong)p.hidden + k];
            }
        }
        #pragma unroll
        for (uint r = 0; r < rows_per_group; ++r) {
            if (sg == 0) scratch[r * 32u + (uint)lane] = 0.0f;
            sums[r] = simd_sum(sums[r]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint r = 0; r < rows_per_group; ++r) {
            if (lane == 0)
                scratch[r * 32u + (uint)sg] = sums[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            #pragma unroll
            for (uint r = 0; r < rows_per_group; ++r) {
                const float value =
                    simd_sum(scratch[r * 32u + (uint)lane]);
                if (lane == 0 && row0 + r < (uint)p.experts)
                    logits[row0 + r] = value;
            }
        }
        return;
    }

    // Eight SIMD groups split into two independent four-group quantizers.
    const uint pair = (uint)sg >> 2;
    const uint quant_sg = (uint)sg & 3u;
    const uint block =
        (group - router_groups) * 2u + pair;
    const uint blocks = ((uint)p.hidden + 127u) / 128u;
    const uint k = block * 128u + quant_sg * 32u + (uint)lane;
    const float value =
        block < blocks && k < (uint)p.hidden ? input[k] : 0.0f;
    float amax = simd_max(fabs(value));
    if (lane == 0) scratch[(uint)sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (quant_sg == 0) {
        amax = simd_max(
            lane < 4u ? scratch[pair * 4u + (uint)lane] : 0.0f);
        if (lane == 0) scratch[pair * 4u] = amax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = scratch[pair * 4u];
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (quant_sg == 0 && lane == 0 && block < blocks)
        scales[block] = amax / 127.0f;
    if (block < blocks && k < (uint)p.hidden)
        quantized[k] =
            (int8_t)clamp((int)rint(value * inv), -127, 127);
}

// M=2..4 counterpart of moe_router_quantize_bg128.  Sixteen SIMD groups
// evaluate sixteen FP16 router rows while reusing each weight load across all
// token rows.  Trailing threadgroups use four SIMD groups per BG128 activation
// block, matching quantize_act_i8_blocks exactly.  Combining the independent
// jobs removes one dispatch from every resident W4 MoE layer.
kernel void moe_router_quantize_bg128_small_m(
    device const float* x [[buffer(0)]],
    device const half* router [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device int8_t* quantized [[buffer(4)]],
    device float* scales [[buffer(5)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint router_simdgroups = 16;
    constexpr uint quant_simdgroups = 4;
    const short M = FC_SMALL_M_DEFINED
        ? (short)FC_SMALL_M : (short)p.seq_len;
    const uint router_groups =
        ((uint)p.experts + router_simdgroups - 1u) /
        router_simdgroups;

    if (group < router_groups) {
        const uint row = group * router_simdgroups + (uint)sg;
        if (row >= (uint)p.experts) return;
        device const half* weights =
            router + (ulong)row * (ulong)p.hidden;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        constexpr uint vector_width = 4;
        const uint vector_end = (uint)p.hidden & ~(vector_width - 1u);
        for (uint k = (uint)lane * vector_width;
             k < vector_end; k += 32u * vector_width) {
            const half4 weight =
                *(device const half4*)(weights + k);
            #pragma unroll
            for (short token = 0; token < M; ++token) {
                device const float* activation =
                    x + p.hidden_offset +
                    (ulong)token * p.hidden_row_stride;
                sums[token] += dot(
                    *(device const float4*)(activation + k),
                    float4(weight));
            }
        }
        for (uint k = vector_end + (uint)lane;
             k < (uint)p.hidden; k += 32u) {
            const float weight = (float)weights[k];
            #pragma unroll
            for (short token = 0; token < M; ++token)
                sums[token] +=
                    x[p.hidden_offset +
                      (ulong)token * p.hidden_row_stride + k] * weight;
        }
        #pragma unroll
        for (short token = 0; token < M; ++token) {
            const float total = simd_sum(sums[token]);
            if (lane == 0)
                logits[(ulong)token * p.experts + row] = total;
        }
        return;
    }

    const uint quant_group = group - router_groups;
    const uint block_in_group = (uint)sg / quant_simdgroups;
    const uint quant_sg = (uint)sg & (quant_simdgroups - 1u);
    const uint blocks_per_group =
        router_simdgroups / quant_simdgroups;
    const uint flat_block =
        quant_group * blocks_per_group + block_in_group;
    const uint blocks = (uint)p.gu_groups_per_row;
    const uint token = flat_block / blocks;
    const uint block = flat_block - token * blocks;
    const uint k = block * 128u + quant_sg * 32u + (uint)lane;
    float value = 0.0f;
    if (token < (uint)M && block < blocks && k < (uint)p.hidden)
        value = x[p.hidden_offset +
                  (ulong)token * p.hidden_row_stride + k];
    float amax = simd_max(fabs(value));
    const uint scratch_base = block_in_group * quant_simdgroups;
    if (lane == 0) scratch[scratch_base + quant_sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (quant_sg == 0) {
        amax = simd_max(
            lane < quant_simdgroups
                ? scratch[scratch_base + (uint)lane]
                : 0.0f);
        if (lane == 0) scratch[scratch_base] = amax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = scratch[scratch_base];
    const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (quant_sg == 0 && lane == 0 &&
        token < (uint)M && block < blocks)
        scales[(ulong)token * blocks + block] = amax / 127.0f;
    if (token < (uint)M && block < blocks && k < (uint)p.hidden)
        quantized[(ulong)token * p.hidden + k] =
            (int8_t)clamp((int)rint(value * inverse), -127, 127);
}

kernel void moe_select_sigmoid(
    device const float* logits [[buffer(0)]], device int* top_idx [[buffer(1)]],
    device float* top_w [[buffer(2)]], constant MoeW4Params& p [[buffer(3)]],
    device const float* bias [[buffer(4)]], uint t [[thread_position_in_grid]]) {
    if(t>=(uint)p.seq_len)return;
    moe_select_sigmoid_token(
        logits+(ulong)t*p.experts, top_idx, top_w, p, bias, t);
}

// Parallel grouped-sigmoid or softmax routing for one token per workgroup.
// One 128/256-thread workgroup reduces all expert scores and repeats the max
// reduction for the (at most 16) selected experts. Grouped sigmoid routing
// filters groups from shared scores first. Ties retain the lower expert index,
// matching the serial insertion order.
constant bool FC_MOE_SELECT_SIGMOID [[function_constant(7)]];
constant bool FC_MOE_SELECT_GROUPED [[function_constant(8)]];

kernel void moe_select_parallel(
    device const float* logits [[buffer(0)]],
    device int* top_idx [[buffer(1)]],
    device float* top_w [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float group_scores[8];
    threadgroup uint group_indices[8];
    threadgroup float weight_values[256];
    threadgroup float ranking_values[256];
    threadgroup uint keep_groups[16];
    threadgroup uint winners[16];
    threadgroup float winner_values[16];
    threadgroup float local_winner_scores[32];
    threadgroup uint local_winner_indices[32];
    if (token >= (uint)p.seq_len) return;
    device const float* token_logits =
        logits + (ulong)token * (ulong)p.experts;
    device int* token_indices =
        top_idx + (ulong)token * (ulong)p.top_k;
    device float* token_weights =
        top_w + (ulong)token * (ulong)p.top_k;

    const uint expert = (uint)tid;
    float weight_value = 0.0f;
    float score = -INFINITY;
    uint expert_index = UINT_MAX;
    if (expert < (uint)p.experts) {
        if (FC_MOE_SELECT_SIGMOID) {
            weight_value =
                1.0f / (1.0f + exp(-token_logits[expert]));
            score = weight_value + bias[expert];
        } else {
            weight_value = token_logits[expert];
            score = weight_value;
        }
        expert_index = expert;
    }
    weight_values[expert] = weight_value;
    if (FC_MOE_SELECT_GROUPED)
        ranking_values[expert] = score;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (FC_MOE_SELECT_GROUPED) {
        if (tid == 0) {
            const int experts_per_group =
                p.experts / p.n_group;
            float best_group_scores[16];
            for (int group = 0; group < p.n_group; ++group) {
                float best0 = -INFINITY;
                float best1 = -INFINITY;
                const int begin = group * experts_per_group;
                const int end = begin + experts_per_group;
                for (int e = begin; e < end; ++e) {
                    const float value = ranking_values[e];
                    if (value > best0) {
                        best1 = best0;
                        best0 = value;
                    } else if (value > best1) {
                        best1 = value;
                    }
                }
                best_group_scores[group] = best0 + best1;
                keep_groups[group] = 0;
            }
            for (int rank = 0;
                 rank < p.topk_group; ++rank) {
                int best_group = 0;
                float best_score = -INFINITY;
                for (int group = 0;
                     group < p.n_group; ++group) {
                    if (!keep_groups[group] &&
                        best_group_scores[group] >
                            best_score) {
                        best_score =
                            best_group_scores[group];
                        best_group = group;
                    }
                }
                keep_groups[best_group] = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (expert < (uint)p.experts) {
            const int experts_per_group =
                p.experts / p.n_group;
            const int group =
                (int)expert / max(experts_per_group, 1);
            if (!keep_groups[group]) {
                score = -INFINITY;
                expert_index = UINT_MAX;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.experts <= 128 && p.top_k <= 8) {
        float local_score = score;
        uint local_index = expert_index;
        for (int rank = 0; rank < p.top_k; ++rank) {
            const float selected_score =
                simd_max(local_score);
            const uint selected_index =
                simd_min(
                    local_score == selected_score
                        ? local_index
                        : UINT_MAX);
            if (lane == 0) {
                const uint candidate =
                    (uint)sg * (uint)p.top_k +
                    (uint)rank;
                local_winner_scores[candidate] =
                    selected_score;
                local_winner_indices[candidate] =
                    selected_index;
            }
            if (local_index == selected_index) {
                local_score = -INFINITY;
                local_index = UINT_MAX;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            const uint candidate_count =
                (uint)nsg * (uint)p.top_k;
            float candidate_score =
                (uint)lane < candidate_count
                    ? local_winner_scores[lane]
                    : -INFINITY;
            uint candidate_index =
                (uint)lane < candidate_count
                    ? local_winner_indices[lane]
                    : UINT_MAX;
            for (int rank = 0; rank < p.top_k; ++rank) {
                const float final_score =
                    simd_max(candidate_score);
                const uint final_index =
                    simd_min(
                        candidate_score == final_score
                            ? candidate_index
                            : UINT_MAX);
                if (lane == 0) {
                    winners[rank] = final_index;
                    winner_values[rank] =
                        weight_values[final_index];
                }
                if (candidate_index == final_index) {
                    candidate_score = -INFINITY;
                    candidate_index = UINT_MAX;
                }
            }
        }
    } else {
        for (int rank = 0; rank < p.top_k; ++rank) {
            const float simd_score = simd_max(score);
            const uint simd_index = simd_min(
                score == simd_score ? expert_index : UINT_MAX);
            if (lane == 0) {
                group_scores[sg] = simd_score;
                group_indices[sg] = simd_index;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0) {
                const float candidate_score =
                    lane < nsg ? group_scores[lane] : -INFINITY;
                const float final_score =
                    simd_max(candidate_score);
                const uint final_index = simd_min(
                    candidate_score == final_score && lane < nsg
                        ? group_indices[lane]
                        : UINT_MAX);
                if (lane == 0) {
                    winners[rank] = final_index;
                    winner_values[rank] =
                        weight_values[final_index];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (expert == winners[rank]) {
                score = -INFINITY;
                expert_index = UINT_MAX;
            }
        }
    }

    if (tid == 0) {
        if (FC_MOE_SELECT_SIGMOID) {
            float sum = 0.0f;
            for (int rank = 0; rank < p.top_k; ++rank)
                sum += winner_values[rank];
            const float multiplier =
                p.routed_scale *
                ((p.norm_topk && sum > 0.0f)
                     ? 1.0f / sum
                     : 1.0f);
            for (int rank = 0; rank < p.top_k; ++rank) {
                token_indices[rank] = (int)winners[rank];
                token_weights[rank] =
                    winner_values[rank] * multiplier;
            }
        } else {
            const float maximum = winner_values[0];
            float sum = 0.0f;
            for (int rank = 0; rank < p.top_k; ++rank) {
                winner_values[rank] =
                    exp(winner_values[rank] - maximum);
                sum += winner_values[rank];
            }
            const float inverse =
                sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int rank = 0; rank < p.top_k; ++rank) {
                token_indices[rank] = (int)winners[rank];
                token_weights[rank] =
                    winner_values[rank] * inverse;
            }
        }
    }
}

kernel void moe_select_softmax(
    device const float* logits [[buffer(0)]], device int* top_idx [[buffer(1)]],
    device float* top_w [[buffer(2)]], constant MoeW4Params& p [[buffer(3)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= (uint)p.seq_len) return;
    moe_select_softmax_token(
        logits + (ulong)t * p.experts, top_idx, top_w, p, t);
}

// Legacy fused router retained as a simple reference implementation.
kernel void moe_route_sigmoid_f16(
    device const float* x [[buffer(0)]], device const half* router [[buffer(1)]],
    device int* top_idx [[buffer(2)]], device float* top_w [[buffer(4)]],
    device const float* bias [[buffer(5)]], constant MoeW4Params& p [[buffer(3)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= (uint)p.seq_len) return;
    device const float* xt = x + p.hidden_offset + (ulong)t*p.hidden_row_stride;
    float chosen[16]; int indices[16];
    for (int k=0;k<p.top_k;k++){ chosen[k]=-INFINITY; indices[k]=0; }
    const int epg = p.experts / max(p.n_group, 1);
    float group_best0[16], group_best1[16];
    for(int g=0;g<p.n_group;g++){group_best0[g]=-INFINITY;group_best1[g]=-INFINITY;}
    // First pass obtains the two best biased scores in every group.
    for (int e=0;e<p.experts;e++) {
        float v=0.0f; device const half* wr=router+(ulong)e*p.hidden;
        for(int h=0;h<p.hidden;h++) v += xt[h]*(float)wr[h];
        float s=1.0f/(1.0f+exp(-v)); float c=s+bias[e]; int g=e/max(epg,1);
        if(c>group_best0[g]){group_best1[g]=group_best0[g];group_best0[g]=c;}
        else if(c>group_best1[g]) group_best1[g]=c;
    }
    bool keep[16]; for(int g=0;g<p.n_group;g++) keep[g]=false;
    for(int n=0;n<p.topk_group;n++) { int bg=0; float bv=-INFINITY;
        for(int g=0;g<p.n_group;g++) if(!keep[g] && group_best0[g]+group_best1[g]>bv)
            {bv=group_best0[g]+group_best1[g];bg=g;} keep[bg]=true; }
    // Recompute scores and select experts from retained groups.
    for (int e=0;e<p.experts;e++) { if(!keep[e/max(epg,1)]) continue;
        float v=0.0f; device const half* wr=router+(ulong)e*p.hidden;
        for(int h=0;h<p.hidden;h++) v += xt[h]*(float)wr[h];
        float s=1.0f/(1.0f+exp(-v)); float c=s+bias[e];
        for(int k=0;k<p.top_k;k++) if(c>chosen[k]) { for(int j=p.top_k-1;j>k;j--)
            {chosen[j]=chosen[j-1];indices[j]=indices[j-1];} chosen[k]=c;indices[k]=e;break; }
    }
    float sum=0.0f;
    for(int k=0;k<p.top_k;k++){ int e=indices[k]; float v=0.0f;
        device const half* wr=router+(ulong)e*p.hidden;
        for(int h=0;h<p.hidden;h++) v+=xt[h]*(float)wr[h];
        chosen[k]=1.0f/(1.0f+exp(-v)); sum+=chosen[k]; }
    float mul=p.routed_scale*((p.norm_topk && sum>0.0f)?1.0f/sum:1.0f);
    for(int k=0;k<p.top_k;k++){top_idx[t*p.top_k+k]=indices[k];top_w[t*p.top_k+k]=chosen[k]*mul;}
}

inline float moe_w4_dot(device const float* a, device const uchar* w,
                        device const float* scales, int K, int gpr,
                        ushort lane, ushort sg, ushort nsg) {
    float sum=0.0f;
    for(int kb=(int)sg*32+(int)lane;kb<K/2;kb+=(int)nsg*32){uchar q=w[kb];int lo=q&15,hi=q>>4;
        if(lo>=8)lo-=16;if(hi>=8)hi-=16;int k=kb*2;
        sum += a[k]*(float)lo*scales[k/128] + a[k+1]*(float)hi*scales[(k+1)/128];}
    return simd_sum(sum);
}

inline float moe_w4_dot_i8_offset_binary(
    device const int8_t* a, device const uchar* w,
    device const float* scales, int K, ushort lane) {
    float sum = 0.0f;
    for (int kb = (int)lane; kb < (K + 1) / 2; kb += 32) {
        const uchar q = w[kb];
        const int k = kb * 2;
        const float weight_scale = scales[k / 128];
        sum += (float)((int)a[k] * ((int)(q & 15) - 8)) *
               weight_scale;
        if (k + 1 < K)
            sum += (float)((int)a[k + 1] * ((int)(q >> 4) - 8)) *
                   weight_scale;
    }
    return simd_sum(sum);
}

kernel void moe_shared_gate_up_w4_i8(
    device const int8_t* x [[buffer(0)]],
    device const uchar* gate_w [[buffer(1)]],
    device float* inter [[buffer(2)]],
    constant MoeSharedW4Params& p [[buffer(3)]],
    device const float* gate_scales [[buffer(4)]],
    device const uchar* up_w [[buffer(5)]],
    device const float* up_scales [[buffer(6)]],
    device const float* x_scale [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (row >= (uint)p.intermediate) return;
    float gate = moe_w4_dot_i8_offset_binary(
        x, gate_w + (ulong)row * (p.hidden / 2),
        gate_scales + (ulong)row * p.gate_groups_per_row,
        p.hidden, lane);
    float up = moe_w4_dot_i8_offset_binary(
        x, up_w + (ulong)row * (p.hidden / 2),
        up_scales + (ulong)row * p.up_groups_per_row,
        p.hidden, lane);
    if (lane == 0) {
        gate *= x_scale[0];
        up *= x_scale[0];
        inter[row] = (gate / (1.0f + exp(-gate))) * up;
    }
}

kernel void moe_shared_scale_f16(
    device const float* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant MoeSharedW4Params& p [[buffer(3)]],
    ushort lane [[thread_index_in_simdgroup]]) {
    float sum = 0.0f;
    device const float* hidden = x + p.hidden_offset;
    for (int k = lane; k < p.hidden; k += 32)
        sum += hidden[k] * (float)weight[k];
    sum = simd_sum(sum);
    if (lane == 0)
        scale[0] = 1.0f / (1.0f + exp(-sum));
}

kernel void moe_shared_down_w4_i8(
    device const int8_t* inter [[buffer(0)]],
    device const uchar* down_w [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MoeSharedW4Params& p [[buffer(3)]],
    device const float* down_scales [[buffer(4)]],
    device const float* scale [[buffer(5)]],
    device const float* inter_scale [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (row >= (uint)p.hidden) return;
    float value = moe_w4_dot_i8_offset_binary(
        inter, down_w + (ulong)row * (p.intermediate / 2),
        down_scales + (ulong)row * p.down_groups_per_row,
        p.intermediate, lane);
    if (lane == 0)
        output[p.output_offset + row] =
            value * inter_scale[0] * scale[0];
}

kernel void add_inplace_f32(
    device float* target [[buffer(0)]],
    device const float* update [[buffer(1)]],
    constant uint& count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) target[i] += update[i];
}

kernel void moe_gate_up_w4(
    device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* merged [[buffer(2)]], constant MoeW4Params& p [[buffer(3)]],
    device const float* scales [[buffer(4)]], device const int* idx [[buffer(5)]],
    threadgroup float* sh [[threadgroup(0)]], uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    int row0=(int)tg.x*4,ktop=(int)tg.y,t=(int)tg.z;
    int e=idx[t*p.top_k+ktop];
    for(int rr=0;rr<4;rr++){int row=row0+rr;if(row>=2*p.intermediate)break;
        ulong flatrow=(ulong)e*(2*p.intermediate)+row;
        float v=moe_w4_dot(x+p.hidden_offset+(ulong)t*p.hidden_row_stride,
            w+flatrow*(p.hidden/2),scales+flatrow*p.gu_groups_per_row,p.hidden,p.gu_groups_per_row,lane,sg,nsg);
        if(lane==0)sh[sg]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sg==0){float z=lane<nsg?sh[lane]:0.0f;z=simd_sum(z);if(lane==0)
            merged[((ulong)t*p.top_k+ktop)*(2*p.intermediate)+row]=z;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Resident W8 selected-expert projection. One SIMD group owns eight output
// rows and reuses every activation float4 across them. Four independent SIMD
// groups therefore cover 32 rows per threadgroup without shared memory or
// cross-SIMD barriers.
kernel void moe_gate_up_w8(
    device const float* x [[buffer(0)]],
    device const int8_t* w [[buffer(1)]],
    device float* merged [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    device const int* idx [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    const int row0 = ((int)tg.x * (int)nsg + (int)sg) * 8;
    const int ktop = (int)tg.y;
    const int t = (int)tg.z;
    if (row0 >= 2 * p.intermediate) return;
    const int expert = idx[t * p.top_k + ktop];
    device const float* activation =
        x + p.hidden_offset + (ulong)t * p.hidden_row_stride;
    float4 total0 = 0.0f;
    float4 total1 = 0.0f;
    for (int group = 0; group < p.gu_groups_per_row; ++group) {
        const int begin = group * p.gu_group_size;
        const int end = min(begin + p.gu_group_size, p.hidden);
        const int vector_end = end & ~3;
        float lane_sums[8] = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f};
        for (int k = begin + (int)lane * 4;
             k + 3 < end; k += 128) {
            const float4 av =
                *(device const float4*)(activation + k);
            #pragma unroll
            for (int channel = 0; channel < 8; ++channel) {
                const int row = row0 + channel;
                if (row >= 2 * p.intermediate) continue;
                const ulong flatrow =
                    (ulong)expert * (2 * p.intermediate) + row;
                const char4 wv = *(device const char4*)(
                    w + flatrow * p.hidden + k);
                lane_sums[channel] += dot(av, float4(wv));
            }
        }
        for (int k = vector_end + (int)lane;
             k < end; k += 32) {
            const float av = activation[k];
            #pragma unroll
            for (int channel = 0; channel < 8; ++channel) {
                const int row = row0 + channel;
                if (row >= 2 * p.intermediate) continue;
                const ulong flatrow =
                    (ulong)expert * (2 * p.intermediate) + row;
                lane_sums[channel] +=
                    av * (float)w[flatrow * p.hidden + k];
            }
        }
        const float4 reduced0 = simd_sum(float4(
            lane_sums[0], lane_sums[1],
            lane_sums[2], lane_sums[3]));
        const float4 reduced1 = simd_sum(float4(
            lane_sums[4], lane_sums[5],
            lane_sums[6], lane_sums[7]));
        float4 weight_scale0 = 0.0f;
        float4 weight_scale1 = 0.0f;
        #pragma unroll
        for (int channel = 0; channel < 8; ++channel) {
            const int row = row0 + channel;
            if (row >= 2 * p.intermediate) continue;
            const ulong flatrow =
                (ulong)expert * (2 * p.intermediate) + row;
            if (channel < 4)
                weight_scale0[channel] =
                    scales[flatrow * p.gu_groups_per_row + group];
            else
                weight_scale1[channel - 4] =
                    scales[flatrow * p.gu_groups_per_row + group];
        }
        total0 += reduced0 * weight_scale0;
        total1 += reduced1 * weight_scale1;
    }
    if (lane == 0) {
        device float* output =
            merged + ((ulong)t * p.top_k + ktop) *
                         (2 * p.intermediate) + row0;
        if (row0 + 8 <= 2 * p.intermediate) {
            *(device float4*)(output + 0) = total0;
            *(device float4*)(output + 4) = total1;
        } else {
            #pragma unroll
            for (int channel = 0; channel < 8; ++channel)
                if (row0 + channel < 2 * p.intermediate)
                    output[channel] = channel < 4
                        ? total0[channel]
                        : total1[channel - 4];
        }
    }
}

// Tiny-batch paired gate/up projection. A SIMD group evaluates four gate rows
// and their four matching up rows together, reusing each activation load and
// writing SwiGLU directly into the first half of the existing scratch row.
// The down projection already consumes that first half, so no layout or graph
// contract changes are required.
kernel void moe_gate_up_swiglu_w8_r4(
    device const float* x [[buffer(0)]],
    device const int8_t* w [[buffer(1)]],
    device float* merged [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    device const int* idx [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    const int row0 = ((int)tg.x * (int)nsg + (int)sg) * 4;
    const int ktop = (int)tg.y;
    const int t = (int)tg.z;
    if (row0 >= p.intermediate) return;
    const int expert = idx[t * p.top_k + ktop];
    const ulong expert_base =
        (ulong)expert * (2 * p.intermediate);
    device const float* activation =
        x + p.hidden_offset + (ulong)t * p.hidden_row_stride;
    float4 gate_total = 0.0f;
    float4 up_total = 0.0f;
    for (int group = 0; group < p.gu_groups_per_row; ++group) {
        const int begin = group * p.gu_group_size;
        const int end = min(begin + p.gu_group_size, p.hidden);
        const int vector_end = end & ~3;
        float4 gate_lane_sums = 0.0f;
        float4 up_lane_sums = 0.0f;
        for (int k = begin + (int)lane * 4;
             k + 3 < end; k += 128) {
            const float4 av =
                *(device const float4*)(activation + k);
            #pragma unroll
            for (int channel = 0; channel < 4; ++channel) {
                const int row = row0 + channel;
                if (row >= p.intermediate) continue;
                const char4 gate_w = *(device const char4*)(
                    w + (expert_base + row) * p.hidden + k);
                const char4 up_w = *(device const char4*)(
                    w + (expert_base + p.intermediate + row) *
                            p.hidden + k);
                gate_lane_sums[channel] +=
                    dot(av, float4(gate_w));
                up_lane_sums[channel] +=
                    dot(av, float4(up_w));
            }
        }
        for (int k = vector_end + (int)lane;
             k < end; k += 32) {
            const float av = activation[k];
            #pragma unroll
            for (int channel = 0; channel < 4; ++channel) {
                const int row = row0 + channel;
                if (row >= p.intermediate) continue;
                gate_lane_sums[channel] += av * (float)w[
                    (expert_base + row) * p.hidden + k];
                up_lane_sums[channel] += av * (float)w[
                    (expert_base + p.intermediate + row) *
                        p.hidden + k];
            }
        }
        const float4 gate_reduced = simd_sum(gate_lane_sums);
        const float4 up_reduced = simd_sum(up_lane_sums);
        float4 gate_scale = 0.0f;
        float4 up_scale = 0.0f;
        #pragma unroll
        for (int channel = 0; channel < 4; ++channel) {
            const int row = row0 + channel;
            if (row >= p.intermediate) continue;
            gate_scale[channel] = scales[
                (expert_base + row) * p.gu_groups_per_row + group];
            up_scale[channel] = scales[
                (expert_base + p.intermediate + row) *
                    p.gu_groups_per_row + group];
        }
        gate_total += gate_reduced * gate_scale;
        up_total += up_reduced * up_scale;
    }
    if (lane == 0) {
        device float* output =
            merged + ((ulong)t * p.top_k + ktop) *
                         (2 * p.intermediate) + row0;
        #pragma unroll
        for (int channel = 0; channel < 4; ++channel) {
            if (row0 + channel >= p.intermediate) continue;
            const float gate = gate_total[channel];
            output[channel] =
                (gate / (1.0f + exp(-gate))) * up_total[channel];
        }
    }
}

inline float moe_w8_dot_precise(
    device const float* activation,
    device const int8_t* weight,
    float weight_scale, int K,
    ushort lane, ushort sg, ushort nsg) {
    float sum = 0.0f;
    for (int k = (int)sg * 32 + (int)lane;
         k < K; k += (int)nsg * 32)
        sum += activation[k] * (float)weight[k];
    return simd_sum(sum) * weight_scale;
}

// Decode retains the original two-level reduction and four-row scheduling
// exactly so autoregressive numerics remain unchanged.
kernel void moe_gate_up_w8_precise(
    device const float* x [[buffer(0)]],
    device const int8_t* w [[buffer(1)]],
    device float* merged [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    device const int* idx [[buffer(5)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    const int row0 = (int)tg.x * 4;
    const int ktop = (int)tg.y;
    const int t = (int)tg.z;
    if (row0 >= 2 * p.intermediate) return;
    const int expert = idx[t * p.top_k + ktop];
    device const float* activation =
        x + p.hidden_offset + (ulong)t * p.hidden_row_stride;
    for (int channel = 0; channel < 4; ++channel) {
        const int row = row0 + channel;
        if (row >= 2 * p.intermediate) break;
        const ulong flatrow =
            (ulong)expert * (2 * p.intermediate) + row;
        const float value = moe_w8_dot_precise(
            activation, w + flatrow * p.hidden,
            scales[flatrow * p.gu_groups_per_row],
            p.hidden, lane, sg, nsg);
        if (lane == 0) scratch[sg] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float reduced = lane < nsg ? scratch[lane] : 0.0f;
            reduced = simd_sum(reduced);
            if (lane == 0)
                merged[((ulong)t * p.top_k + ktop) *
                           (2 * p.intermediate) + row] = reduced;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void moe_swiglu_selected(
    device float* merged [[buffer(0)]], constant MoeW4Params& p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    uint block=i/(uint)p.intermediate,j=i%(uint)p.intermediate;
    if(block>=(uint)(p.seq_len*p.top_k))return;ulong base=(ulong)block*(2*p.intermediate);
    float g=merged[base+j],u=merged[base+p.intermediate+j];
    merged[base+j]=(g/(1.0f+exp(-g)))*u;
}

// W8PC decode fusion. Compute one selected expert's SwiGLU row, derive a
// single activation scale, and write int8 directly. This preserves the W8PC
// per-row quantization contract while avoiding an FP32 activated round trip.
kernel void moe_swiglu_quantize_row(
    device const float* merged [[buffer(0)]],
    device int8_t* quantized [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device float* scales [[buffer(4)]],
    uint selection [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]],
    threadgroup float* shmem [[threadgroup(0)]]) {
    const uint selections = (uint)(p.seq_len * p.top_k);
    if (selection >= selections) return;
    const ulong base =
        (ulong)selection * (ulong)(2 * p.intermediate);
    float local_amax = 0.0f;
    for (uint j = (uint)sg * 32u + (uint)lane;
         j < (uint)p.intermediate;
         j += (uint)nsg * 32u) {
        const float gate = merged[base + j];
        const float up = merged[base + (uint)p.intermediate + j];
        const float value =
            (gate / (1.0f + exp(-gate))) * up;
        local_amax = max(local_amax, fabs(value));
    }
    local_amax = simd_max(local_amax);
    if (lane == 0) shmem[sg] = local_amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float value = lane < nsg ? shmem[lane] : 0.0f;
        value = simd_max(value);
        if (lane == 0) shmem[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float amax = shmem[0];
    const float scale = amax / 127.0f;
    const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (sg == 0 && lane == 0) scales[selection] = scale;
    device int8_t* output =
        quantized + (ulong)selection * (ulong)p.intermediate;
    for (uint j = (uint)sg * 32u + (uint)lane;
         j < (uint)p.intermediate;
         j += (uint)nsg * 32u) {
        const float gate = merged[base + j];
        const float up = merged[base + (uint)p.intermediate + j];
        const float value =
            (gate / (1.0f + exp(-gate))) * up;
        output[j] = (int8_t)clamp(
            (int)rint(value * inverse), -127, 127);
    }
}

// Resident BG128 decode fusion: compute SwiGLU and immediately quantize each
// 128-value intermediate block. Four SIMD groups cover one block, so the
// activation never needs to be materialized as FP32 between two dispatches.
kernel void moe_swiglu_quantize_blocks(
    device const float* merged [[buffer(0)]],
    device int8_t* quantized [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device float* scales [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    threadgroup float* shmem [[threadgroup(0)]]) {
    const uint blocks = ((uint)p.intermediate + 127u) / 128u;
    const uint selection = group / blocks;
    const uint block = group - selection * blocks;
    const uint selections = (uint)(p.seq_len * p.top_k);
    if (selection >= selections) return;

    const uint j =
        block * 128u + (uint)sg * 32u + (uint)lane;
    const ulong base =
        (ulong)selection * (ulong)(2 * p.intermediate);
    float value = 0.0f;
    if (j < (uint)p.intermediate) {
        const float gate = merged[base + j];
        const float up =
            merged[base + (uint)p.intermediate + j];
        value = (gate / (1.0f + exp(-gate))) * up;
    }

    float amax = simd_max(fabs(value));
    if (lane == 0) shmem[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < 4 ? shmem[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) shmem[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    amax = shmem[0];
    const float scale = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (sg == 0 && lane == 0)
        scales[(ulong)selection * blocks + block] = scale;
    if (j < (uint)p.intermediate)
        quantized[
            (ulong)selection * p.intermediate + j] =
            (int8_t)clamp(
                (int)rint(value * inv), -127, 127);
}

// Native BG32 decode fusion. Each SIMD group owns one K32 block and combines
// SwiGLU with the exact per-block activation quantization consumed by the
// down projection. Gate/up dot products remain in their established kernel.
kernel void moe_swiglu_quantize_block32(
    device const float* merged [[buffer(0)]],
    device int8_t* quantized [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device float* scales [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint blocks =
        ((uint)p.intermediate + 31u) / 32u;
    const uint block_groups =
        (blocks + (uint)nsg - 1u) / (uint)nsg;
    const uint selection = group / block_groups;
    const uint block =
        (group - selection * block_groups) *
            (uint)nsg +
        (uint)sg;
    const uint selections =
        (uint)(p.seq_len * p.top_k);
    if (selection >= selections || block >= blocks) return;

    const uint j = block * 32u + (uint)lane;
    const ulong base =
        (ulong)selection *
        (ulong)(2 * p.intermediate);
    float value = 0.0f;
    if (j < (uint)p.intermediate) {
        const float gate = merged[base + j];
        const float up =
            merged[base + (uint)p.intermediate + j];
        value = (gate / (1.0f + exp(-gate))) * up;
    }
    const float amax = simd_max(fabs(value));
    const float scale = amax / 127.0f;
    const float inv =
        amax > 0.0f ? 127.0f / amax : 0.0f;
    if (lane == 0)
        scales[(ulong)selection * blocks + block] =
            scale;
    if (j < (uint)p.intermediate)
        quantized[
            (ulong)selection * p.intermediate + j] =
            (int8_t)clamp(
                (int)rint(value * inv), -127, 127);
}

// Grouped prefill gate/up already materializes SwiGLU. This pass only performs
// the identical FP32 block-128 activation quantization needed by native BG128
// down projection.
kernel void moe_quantize_selected_blocks(
    device const float* activated [[buffer(0)]],
    device int8_t* quantized [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device float* scales [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint blocks = ((uint)p.intermediate + 127u) / 128u;
    const uint selection = group / blocks;
    const uint block = group - selection * blocks;
    const uint selections = (uint)(p.seq_len * p.top_k);
    if (selection >= selections) return;

    const uint j0 = block * 128u + (uint)lane;
    const ulong base =
        (ulong)selection * (ulong)p.intermediate;
    float values[4];
    float local_amax = 0.0f;
    #pragma unroll
    for (uint quarter = 0; quarter < 4; ++quarter) {
        const uint j = j0 + quarter * 32u;
        const float value =
            j < (uint)p.intermediate
                ? activated[base + j]
                : 0.0f;
        values[quarter] = value;
        local_amax = max(local_amax, fabs(value));
    }
    const float amax = simd_max(local_amax);
    const float scale = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (lane == 0)
        scales[(ulong)selection * blocks + block] = scale;
    #pragma unroll
    for (uint quarter = 0; quarter < 4; ++quarter) {
        const uint j = j0 + quarter * 32u;
        if (j < (uint)p.intermediate)
            quantized[base + j] =
                (int8_t)clamp(
                    (int)rint(values[quarter] * inv),
                    -127, 127);
    }
}

kernel void moe_combine_selected(
    device const float* selected [[buffer(0)]], device float* out [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]], device const float* topw [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
    uint d=pos.x,t=pos.y;if(d>=(uint)p.hidden||t>=(uint)p.seq_len)return;float v=0.0f;
    for(int k=0;k<p.top_k;k++)v+=selected[((ulong)t*p.top_k+k)*p.hidden+d]*topw[t*p.top_k+k];
    out[p.output_offset+(ulong)t*p.output_row_stride+d]=v;
}

kernel void moe_down_combine_w4(
    device const float* merged [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant MoeW4Params& p [[buffer(3)]],
    device const float* scales [[buffer(4)]], device const int* idx [[buffer(5)]],
    device const float* topw [[buffer(6)]], threadgroup float* sh [[threadgroup(0)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort nsg [[simdgroups_per_threadgroup]]) {
    int d0=(int)tg.x*4,t=(int)tg.y;
    for(int dd=0;dd<4;dd++){int d=d0+dd;if(d>=p.hidden)break;float total=0.0f;
      for(int q=0;q<p.top_k;q++){int e=idx[t*p.top_k+q];ulong base=((ulong)t*p.top_k+q)*(2*p.intermediate);
        float local=0.0f;ulong row=(ulong)e*p.hidden+d;device const uchar* wr=w+row*(p.intermediate/2);
        device const float* sc=scales+row*p.down_groups_per_row;
        for(int kb=(int)sg*32+(int)lane;kb<p.intermediate/2;kb+=(int)nsg*32){uchar z=wr[kb];int lo=z&15,hi=z>>4;
            if(lo>=8)lo-=16;if(hi>=8)hi-=16;int j=kb*2;
            float a=merged[base+j],a1=merged[base+j+1];
            local+=a*(float)lo*sc[j/128]+a1*(float)hi*sc[(j+1)/128];}
        local=simd_sum(local);if(lane==0)sh[sg]=local;threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sg==0){float z=lane<nsg?sh[lane]:0.0f;z=simd_sum(z);if(lane==0)sh[0]=z;}threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sg==0&&lane==0)total+=sh[0]*topw[t*p.top_k+q];threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if(sg==0&&lane==0)out[p.output_offset+(ulong)t*p.output_row_stride+d]=total;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void moe_down_combine_w8(
    device const float* merged [[buffer(0)]],
    device const int8_t* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    device const int* idx [[buffer(5)]],
    device const float* topw [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    const int d0 = ((int)tg.x * (int)nsg + (int)sg) * 8;
    const int t = (int)tg.y;
    if (d0 >= p.hidden) return;
    float4 total0 = 0.0f;
    float4 total1 = 0.0f;
    for (int q = 0; q < p.top_k; ++q) {
        const int expert = idx[t * p.top_k + q];
        const ulong selection = (ulong)t * p.top_k + q;
        device const float* activation =
            merged + selection * (2 * p.intermediate);
        float4 expert0 = 0.0f;
        float4 expert1 = 0.0f;
        for (int group = 0; group < p.down_groups_per_row; ++group) {
            const int begin = group * p.down_group_size;
            const int end = min(
                begin + p.down_group_size, p.intermediate);
            const int vector_end = end & ~3;
            float lane_sums[8] = {
                0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f};
            for (int k = begin + (int)lane * 4;
                 k + 3 < end; k += 128) {
                const float4 av =
                    *(device const float4*)(activation + k);
                #pragma unroll
                for (int channel = 0; channel < 8; ++channel) {
                    const int d = d0 + channel;
                    if (d >= p.hidden) continue;
                    const ulong row =
                        (ulong)expert * p.hidden + d;
                    const char4 wv = *(device const char4*)(
                        w + row * p.intermediate + k);
                    lane_sums[channel] += dot(av, float4(wv));
                }
            }
            for (int k = vector_end + (int)lane;
                 k < end; k += 32) {
                const float av = activation[k];
                #pragma unroll
                for (int channel = 0; channel < 8; ++channel) {
                    const int d = d0 + channel;
                    if (d >= p.hidden) continue;
                    const ulong row =
                        (ulong)expert * p.hidden + d;
                    lane_sums[channel] +=
                        av * (float)w[row * p.intermediate + k];
                }
            }
            const float4 reduced0 = simd_sum(float4(
                lane_sums[0], lane_sums[1],
                lane_sums[2], lane_sums[3]));
            const float4 reduced1 = simd_sum(float4(
                lane_sums[4], lane_sums[5],
                lane_sums[6], lane_sums[7]));
            float4 weight_scale0 = 0.0f;
            float4 weight_scale1 = 0.0f;
            #pragma unroll
            for (int channel = 0; channel < 8; ++channel) {
                const int d = d0 + channel;
                if (d >= p.hidden) continue;
                const ulong row =
                    (ulong)expert * p.hidden + d;
                if (channel < 4)
                    weight_scale0[channel] =
                        scales[row * p.down_groups_per_row + group];
                else
                    weight_scale1[channel - 4] =
                        scales[row * p.down_groups_per_row + group];
            }
            expert0 += reduced0 * weight_scale0;
            expert1 += reduced1 * weight_scale1;
        }
        const float route_weight = topw[selection];
        total0 += expert0 * route_weight;
        total1 += expert1 * route_weight;
    }
    if (lane == 0) {
        device float* output =
            out + p.output_offset +
            (ulong)t * p.output_row_stride + d0;
        if (d0 + 8 <= p.hidden) {
            *(device float4*)(output + 0) = total0;
            *(device float4*)(output + 4) = total1;
        } else {
            #pragma unroll
            for (int channel = 0; channel < 8; ++channel)
                if (d0 + channel < p.hidden)
                    output[channel] = channel < 4
                        ? total0[channel]
                        : total1[channel - 4];
        }
    }
}

kernel void moe_down_combine_w8_r4(
    device const float* merged [[buffer(0)]],
    device const int8_t* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    device const int* idx [[buffer(5)]],
    device const float* topw [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    const int d0 = ((int)tg.x * (int)nsg + (int)sg) * 4;
    const int t = (int)tg.y;
    if (d0 >= p.hidden) return;
    float4 total = 0.0f;
    for (int q = 0; q < p.top_k; ++q) {
        const int expert = idx[t * p.top_k + q];
        const ulong selection = (ulong)t * p.top_k + q;
        device const float* activation =
            merged + selection * (2 * p.intermediate);
        float4 expert_value = 0.0f;
        for (int group = 0; group < p.down_groups_per_row; ++group) {
            const int begin = group * p.down_group_size;
            const int end = min(
                begin + p.down_group_size, p.intermediate);
            const int vector_end = end & ~3;
            float4 lane_sums = 0.0f;
            for (int k = begin + (int)lane * 4;
                 k + 3 < end; k += 128) {
                const float4 av =
                    *(device const float4*)(activation + k);
                #pragma unroll
                for (int channel = 0; channel < 4; ++channel) {
                    const int d = d0 + channel;
                    if (d >= p.hidden) continue;
                    const ulong row =
                        (ulong)expert * p.hidden + d;
                    const char4 wv = *(device const char4*)(
                        w + row * p.intermediate + k);
                    lane_sums[channel] += dot(av, float4(wv));
                }
            }
            for (int k = vector_end + (int)lane;
                 k < end; k += 32) {
                const float av = activation[k];
                #pragma unroll
                for (int channel = 0; channel < 4; ++channel) {
                    const int d = d0 + channel;
                    if (d >= p.hidden) continue;
                    const ulong row =
                        (ulong)expert * p.hidden + d;
                    lane_sums[channel] +=
                        av * (float)w[row * p.intermediate + k];
                }
            }
            const float4 reduced = simd_sum(lane_sums);
            float4 weight_scale = 0.0f;
            #pragma unroll
            for (int channel = 0; channel < 4; ++channel) {
                const int d = d0 + channel;
                if (d >= p.hidden) continue;
                const ulong row = (ulong)expert * p.hidden + d;
                weight_scale[channel] =
                    scales[row * p.down_groups_per_row + group];
            }
            expert_value += reduced * weight_scale;
        }
        total += expert_value * topw[selection];
    }
    if (lane == 0) {
        device float* output =
            out + p.output_offset +
            (ulong)t * p.output_row_stride + d0;
        if (d0 + 4 <= p.hidden)
            *(device float4*)output = total;
        else
            for (int channel = 0; channel < 4; ++channel)
                if (d0 + channel < p.hidden)
                    output[channel] = total[channel];
    }
}

kernel void moe_down_combine_w8_precise(
    device const float* merged [[buffer(0)]],
    device const int8_t* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MoeW4Params& p [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    device const int* idx [[buffer(5)]],
    device const float* topw [[buffer(6)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]]) {
    const int d0 = (int)tg.x * 4;
    const int t = (int)tg.y;
    if (d0 >= p.hidden) return;
    for (int channel = 0; channel < 4; ++channel) {
        const int d = d0 + channel;
        if (d >= p.hidden) break;
        float total = 0.0f;
        for (int q = 0; q < p.top_k; ++q) {
            const int expert = idx[t * p.top_k + q];
            const ulong selection = (ulong)t * p.top_k + q;
            const ulong row = (ulong)expert * p.hidden + d;
            const float value = moe_w8_dot_precise(
                merged + selection * (2 * p.intermediate),
                w + row * p.intermediate,
                scales[row * p.down_groups_per_row],
                p.intermediate, lane, sg, nsg);
            if (lane == 0) scratch[sg] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                float reduced = lane < nsg ? scratch[lane] : 0.0f;
                reduced = simd_sum(reduced);
                if (lane == 0)
                    total += reduced * topw[selection];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (sg == 0 && lane == 0)
            out[p.output_offset +
                (ulong)t * p.output_row_stride + d] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Per-token int8 quantization of activations for the W8A8 GEMM.
//   in : fp32 A[M,K], element (m,k) at a_offset + m*a_row_stride + k
//   out: int8 A_i8[M,K] contiguous (a_i8[m*K + k]) + fp32 scale_a[M]
// One threadgroup per row m: absmax reduce over K, scale = absmax/127, then
// round each element. Rows with absmax==0 get scale 0 (all zeros out).
kernel void quantize_act_i8(
    device const float*   A       [[buffer(0)]],
    device int8_t*        A_I8     [[buffer(2)]],
    device float*         SCALE_A  [[buffer(4)]],
    constant QuantActParams& p     [[buffer(3)]],
    threadgroup float*    shmem    [[threadgroup(0)]],
    uint  m                        [[threadgroup_position_in_grid]],
    ushort lane                    [[thread_index_in_simdgroup]],
    ushort sgitg                   [[simdgroup_index_in_threadgroup]],
    ushort nsg                     [[simdgroups_per_threadgroup]])
{
    const int K = p.K;
    device const float* a = A + p.a_offset + (uint)m * (uint)p.a_row_stride;
    device int8_t*      o = A_I8 + (uint)m * (uint)K;

    // 1) absmax over the row (two-level: simd_sum-style max via simd_max).
    float amax = 0.0f;
    for (int k = (int)sgitg*32 + (int)lane; k < K; k += 32*(int)nsg)
        amax = fmax(amax, fabs(a[k]));
    amax = simd_max(amax);
    if (lane == 0) shmem[sgitg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        float v = (lane < nsg) ? shmem[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) shmem[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = shmem[0];

    float scale = amax / 127.0f;
    float inv   = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    if (sgitg == 0 && lane == 0) SCALE_A[m] = scale;

    // 2) quantize (round-to-nearest, clamp to int8 range).
    for (int k = (int)sgitg*32 + (int)lane; k < K; k += 32*(int)nsg) {
        int q = (int)rint(a[k] * inv);
        q = clamp(q, -127, 127);
        o[k] = (int8_t)q;
    }
}

// Q8 activation quantization at the CPU W4 kernel's 32-element granularity.
// One SIMD group owns one (row, K-block), so no threadgroup memory or barrier
// is required.
kernel void quantize_act_i8_block32(
    device const float* A        [[buffer(0)]],
    device int8_t*      A_I8     [[buffer(2)]],
    device float*       SCALE_A  [[buffer(4)]],
    constant QuantActParams& p   [[buffer(3)]],
    uint group                    [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]],
    ushort nsg                   [[simdgroups_per_threadgroup]])
{
    const uint blocks = ((uint)p.K + 31u) / 32u;
    const uint block_groups = (blocks + (uint)nsg - 1u) / (uint)nsg;
    const uint m = group / block_groups;
    const uint b =
        (group - m * block_groups) * (uint)nsg + (uint)sgitg;
    if (m >= (uint)p.M || b >= blocks) return;
    const uint k = b * 32u + (uint)lane;
    device const float* row =
        A + p.a_offset + m * (uint)p.a_row_stride;
    float v = k < (uint)p.K ? row[k] : 0.0f;
    float amax = simd_max(fabs(v));
    float scale = amax / 127.0f;
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (lane == 0) SCALE_A[m * blocks + b] = scale;
    if (k < (uint)p.K) {
        int q = clamp((int)rint(v * inv), -127, 127);
        A_I8[m * (uint)p.K + k] = (int8_t)q;
    }
}

inline float round_fp8_e4m3fn_metal(float value) {
    const float sign = value < 0.0f ? -1.0f : 1.0f;
    const float magnitude = min(fabs(value), 448.0f);
    if (magnitude < (1.0f / 64.0f))
        return sign * min(rint(magnitude * 512.0f), 8.0f) / 512.0f;
    const int exponent = (int)((as_type<uint>(magnitude) >> 23) & 0xffu) - 127;
    int significand = (int)rint(ldexp(magnitude, 3 - exponent));
    int rounded_exponent = exponent;
    if (significand == 16) {
        ++rounded_exponent;
        significand = 8;
    }
    return sign * min(
        ldexp((float)significand, rounded_exponent - 3), 448.0f);
}

// DeepSeek-V4 quantizes MXFP4 matmul activations through a 128-value
// UE8M0-scaled E4M3FN block before the internal K32 Q8 representation. One
// 128-thread group owns exactly one such block; its four SIMD groups then
// quantize their K32 slices independently, matching the CPU path.
kernel void quantize_act_fp8_i8_block32(
    device const float* A        [[buffer(0)]],
    device int8_t*      A_I8     [[buffer(2)]],
    device float*       SCALE_A  [[buffer(4)]],
    device int8_t*      RESIDUAL_I8 [[buffer(5)]],
    device float*       RESIDUAL_SCALE [[buffer(6)]],
    constant QuantActParams& p   [[buffer(3)]],
    uint group                    [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]],
    ushort nsg                   [[simdgroups_per_threadgroup]])
{
    const uint fp8_blocks = ((uint)p.K + 127u) / 128u;
    const uint q8_blocks = ((uint)p.K + 31u) / 32u;
    const uint m = group / fp8_blocks;
    const uint fp8_block = group - m * fp8_blocks;
    if (m >= (uint)p.M) return;
    const uint k = fp8_block * 128u + (uint)sgitg * 32u + (uint)lane;
    device const float* row =
        A + p.a_offset + m * (uint)p.a_row_stride;
    const float input = k < (uint)p.K ? row[k] : 0.0f;

    threadgroup float maxima[4];
    float maximum = simd_max(fabs(input));
    if (lane == 0) maxima[sgitg] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        maximum = lane < nsg ? maxima[lane] : 0.0f;
        maximum = simd_max(maximum);
        if (lane == 0) maxima[0] = max(maximum, 1.0e-4f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint bits = as_type<uint>(maxima[0]);
    const int unbiased = (int)((bits >> 23) & 0xffu) - 127;
    const uint mantissa = bits & 0x7fffffu;
    const int scale_exponent = clamp(
        unbiased - 8 + (mantissa > 0x600000u ? 1 : 0), -127, 127);
    const uint scale_bits = scale_exponent == -127
        ? (1u << 22)
        : (uint)(scale_exponent + 127) << 23;
    const float fp8_scale = as_type<float>(scale_bits);
    const float fp8_value =
        round_fp8_e4m3fn_metal(input / fp8_scale) * fp8_scale;

    const float q8_maximum = simd_max(fabs(fp8_value));
    const float q8_scale = q8_maximum > 0.0f
        ? q8_maximum / 127.0f : fp8_scale;
    const uint q8_block = fp8_block * 4u + (uint)sgitg;
    if (lane == 0 && q8_block < q8_blocks)
        SCALE_A[m * q8_blocks + q8_block] = q8_scale;
    const float inverse = q8_maximum > 0.0f
        ? 127.0f / q8_maximum : 0.0f;
    const int primary =
        clamp((int)rint(fp8_value * inverse), -127, 127);
    const float residual = fp8_value - (float)primary * q8_scale;
    const float residual_maximum = simd_max(fabs(residual));
    const float residual_scale = residual_maximum > 0.0f
        ? residual_maximum / 127.0f : 1.0f;
    if (lane == 0 && q8_block < q8_blocks)
        RESIDUAL_SCALE[m * q8_blocks + q8_block] = residual_scale;
    if (k < (uint)p.K) {
        A_I8[m * (uint)p.K + k] = (int8_t)primary;
        const float residual_inverse = residual_maximum > 0.0f
            ? 127.0f / residual_maximum : 0.0f;
        RESIDUAL_I8[m * (uint)p.K + k] = (int8_t)clamp(
            (int)rint(residual * residual_inverse), -127, 127);
    }
}

// Four independent K64 activation blocks per 128-thread group.  Each SIMD
// lane owns two values, so the absmax and quantization need no cross-SIMD
// synchronization.
kernel void quantize_act_i8_block64(
    device const float* A        [[buffer(0)]],
    device int8_t*      A_I8     [[buffer(2)]],
    device float*       SCALE_A  [[buffer(4)]],
    constant QuantActParams& p   [[buffer(3)]],
    uint group                    [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]],
    ushort nsg                   [[simdgroups_per_threadgroup]])
{
    const uint blocks = ((uint)p.K + 63u) / 64u;
    const uint block_groups = (blocks + (uint)nsg - 1u) / (uint)nsg;
    const uint m = group / block_groups;
    const uint b =
        (group - m * block_groups) * (uint)nsg + (uint)sgitg;
    if (m >= (uint)p.M || b >= blocks) return;
    const uint k0 = b * 64u + (uint)lane;
    const uint k1 = k0 + 32u;
    device const float* row =
        A + p.a_offset + m * (uint)p.a_row_stride;
    const float v0 = k0 < (uint)p.K ? row[k0] : 0.0f;
    const float v1 = k1 < (uint)p.K ? row[k1] : 0.0f;
    const float amax = simd_max(fmax(fabs(v0), fabs(v1)));
    const float scale = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (lane == 0) SCALE_A[m * blocks + b] = scale;
    if (k0 < (uint)p.K)
        A_I8[m * (uint)p.K + k0] =
            (int8_t)clamp((int)rint(v0 * inv), -127, 127);
    if (k1 < (uint)p.K)
        A_I8[m * (uint)p.K + k1] =
            (int8_t)clamp((int)rint(v1 * inv), -127, 127);
}

// Block-granular variant used by W4 prefill. The block size normally matches
// the weight quantization group (128), which preserves local activation range
// while allowing one integer MMA accumulation per weight group.
kernel void quantize_act_i8_blocks(
    device const float* A        [[buffer(0)]],
    device int8_t*      A_I8     [[buffer(2)]],
    device float*       SCALE_A  [[buffer(4)]],
    constant QuantActParams& p   [[buffer(3)]],
    uint group                    [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]],
    ushort sgitg                 [[simdgroup_index_in_threadgroup]],
    ushort nsg                   [[simdgroups_per_threadgroup]],
    threadgroup float* shmem     [[threadgroup(0)]])
{
    const uint bs = (uint)p.block_size;
    const uint blocks = ((uint)p.K + bs - 1u) / bs;
    const uint m = group / blocks;
    const uint b = group - m * blocks;
    if (m >= (uint)p.M) return;
    const uint k0 = b * bs;
    const uint kend = min(k0 + bs, (uint)p.K);
    device const float* row =
        A + p.a_offset + m * (uint)p.a_row_stride;

    float amax = 0.0f;
    for (uint k = k0 + (uint)sgitg * 32u + (uint)lane;
         k < kend; k += (uint)nsg * 32u)
        amax = fmax(amax, fabs(row[k]));
    amax = simd_max(amax);
    if (lane == 0) shmem[sgitg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        float v = lane < nsg ? shmem[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) shmem[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = shmem[0];
    const float scale = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (sgitg == 0 && lane == 0) SCALE_A[m * blocks + b] = scale;
    for (uint k = k0 + (uint)sgitg * 32u + (uint)lane;
         k < kend; k += (uint)nsg * 32u) {
        int q = clamp((int)rint(row[k] * inv), -127, 127);
        A_I8[m * (uint)p.K + k] = (int8_t)q;
    }
}

// ---------------------------------------------------------------------------
// GEMV: M==1 decode fast path. C[1,N] = A[1,K] * B[N,K]^T
// One SIMD-group (32 lanes) computes one output row using simd_sum (no
// threadgroup barriers). Multiple SIMD-groups per threadgroup for occupancy.
// A[k] (fp32 activation) is staged into threadgroup memory once per threadgroup
// so all rows reuse it from fast on-chip memory instead of re-reading device.
// grid: threadgroups = ceil(N / rows_per_tg); threads/tg = rows_per_tg * 32.
// ---------------------------------------------------------------------------
kernel void gemv_f32a_f16b_f32c(
    device const float*   A      [[buffer(0)]],
    device const half*    B      [[buffer(1)]],
    device float*         C      [[buffer(2)]],
    constant MatmulParams& p     [[buffer(3)]],
    uint  tgid                   [[threadgroup_position_in_grid]],
    uint  tid                    [[thread_position_in_threadgroup]],
    uint  lane                   [[thread_index_in_simdgroup]],
    uint  sg                     [[simdgroup_index_in_threadgroup]],
    uint  n_sg                   [[simdgroups_per_threadgroup]])
{
    device const float* a = A + p.a_offset;

    // Stage A into threadgroup memory when it fits (K <= AS_CAP); otherwise read
    // A directly from device. down_proj has K=intermediate (e.g. 9728) which
    // exceeds the staging buffer — must NOT stage or we'd read past it.
    const int AS_CAP = 4096;
    threadgroup float as[4096];
    bool staged = (p.K <= AS_CAP);
    if (staged) {
        for (uint k = tid; k < (uint)p.K; k += n_sg * 32u) as[k] = a[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n = tgid * n_sg + sg;    // output row handled by this SIMD-group
    if (n >= (uint)p.N) return;
    device const half* b = B + p.b_offset + n * (uint)p.b_row_stride;

    // Vectorized reduction: each lane consumes half4 chunks, stride 32 across
    // the SIMD group. half4 loads improve weight-read bandwidth. K is a multiple
    // of 4 for all Qwen3 weight dims; handle a scalar tail just in case.
    float partial = 0.0f;
    int K4 = p.K & ~3;
    device const half4*  b4 = (device const half4*)b;
    device const float4* a4 = (device const float4*)a;
    threadgroup float4*  as4 = (threadgroup float4*)as;
    for (int q = int(lane); q < (K4 >> 2); q += 32) {
        float4 av = staged ? as4[q] : a4[q];
        half4  bv = b4[q];
        partial += av.x*float(bv.x) + av.y*float(bv.y)
                 + av.z*float(bv.z) + av.w*float(bv.w);
    }
    for (int k = K4 + int(lane); k < p.K; k += 32)
        partial += (staged ? as[k] : a[k]) * float(b[k]);
    float dot = simd_sum(partial);
    if (lane == 0) {
        C[p.c_offset + n] = apply_activation_at(
            dot, p.activation, n, p.act_n_begin, p.act_n_len);
    }
}

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

// ---------------------------------------------------------------------------
// SDPA KV-cache append: write FP32 K_cur/V_cur rows into the FP16 cache at
// position (past + s), per kv-head. grid = (head_dim, cur_seqlen, num_kv_heads).
// ---------------------------------------------------------------------------
kernel void sdpa_append_kv_f32_to_f16(
    device const float* K_CUR [[buffer(0)]],
    device const float* V_CUR [[buffer(1)]],
    device half* K_CACHE [[buffer(2)]],
    constant SdpaAppendKvParams& p [[buffer(3)]],
    device half* V_CACHE [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint position = gid.y;
    const uint head = gid.z;
    if (position >= (uint)p.cur_seqlen ||
        head >= (uint)p.num_kv_heads)
        return;
    const uint cache_position =
        (uint)p.past_seqlen + position;
    if (d < (uint)p.k_dim) {
        const uint source =
            p.k_cur_offset + head * (uint)p.k_stride_head +
            position * (uint)p.k_stride_pos + d;
        const uint destination =
            p.k_cache_offset +
            head * ((uint)p.k_dim * (uint)p.max_seq_len) +
            cache_position * (uint)p.k_dim + d;
        K_CACHE[destination] = half(K_CUR[source]);
    }
    if (d < (uint)p.v_dim) {
        const uint source =
            p.v_cur_offset + head * (uint)p.v_stride_head +
            position * (uint)p.v_stride_pos + d;
        const uint destination =
            p.v_cache_offset +
            head * ((uint)p.v_dim * (uint)p.max_seq_len) +
            cache_position * (uint)p.v_dim + d;
        V_CACHE[destination] = half(V_CUR[source]);
    }
}

// ---------------------------------------------------------------------------
// SDPA prefill (src_seqlen > 1): one SIMD group (32 lanes) per (query pos, head).
// Lanes cooperate: the QK dot for each key is a simd_sum over head_dim; each
// lane then owns a strided subset of the v_head_dim output accumulator. Single-
// pass online softmax. Fallback when the tensor FA path (sdpa_prefill_fa2_f32)
// is unavailable.
// grid: threadgroups = ceil(num_heads*src_seqlen / (TG/32)); TG=128 (4 groups).
// ---------------------------------------------------------------------------
kernel void sdpa_prefill_f32(
    device const float*   Q       [[buffer(0)]],
    device const half*    KC      [[buffer(1)]],
    device const half*    VC      [[buffer(2)]],
    device float*         O       [[buffer(4)]],
    device const float*   MASK    [[buffer(5)]],
    constant SdpaParams&  p       [[buffer(3)]],
    uint  tgid                    [[threadgroup_position_in_grid]],
    uint  lane                    [[thread_index_in_simdgroup]],
    uint  sg                      [[simdgroup_index_in_threadgroup]],
    uint  n_sg                    [[simdgroups_per_threadgroup]])
{
    // Global query index handled by this SIMD group.
    uint qidx = tgid * n_sg + sg;
    uint total = (uint)p.num_heads * (uint)p.src_seqlen;
    if (qidx >= total) return;
    int h = (int)(qidx / (uint)p.src_seqlen);
    int s = (int)(qidx % (uint)p.src_seqlen);

    int heads_per_group = p.num_heads / p.num_kv_heads;
    int kv_h = h / heads_per_group;

    device const float* q = Q + p.q_offset + (uint)h * p.q_stride_head + (uint)s * p.q_stride_pos;
    device const half*  Kh = KC + p.k_cache_offset + (uint)kv_h * ((uint)p.head_dim * (uint)p.max_seq_len);
    device const half*  Vh = VC + p.v_cache_offset + (uint)kv_h * ((uint)p.v_head_dim * (uint)p.max_seq_len);

    int key_limit = p.dst_seqlen;
    if (p.causal != 0) key_limit = p.past_seqlen + s + 1;
    if (key_limit > p.dst_seqlen) key_limit = p.dst_seqlen;

    // Each lane owns v-dims {lane, lane+32, ...}; up to 4 for v_head_dim<=128.
    float acc[4];
    for (int i = 0; i < 4; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (int j = 0; j < key_limit; j++) {
        device const half* k = Kh + (uint)j * (uint)p.head_dim;
        // Cooperative QK dot: each lane sums a strided slice of head_dim.
        float partial = 0.0f;
        for (int d = int(lane); d < p.head_dim; d += 32) partial += q[d] * float(k[d]);
        float score = simd_sum(partial) * p.scale;
        if (p.has_mask != 0)
            score += MASK[p.mask_offset + (uint)s * (uint)p.mask_stride_row + (uint)j];
        float m_new = max(m, score);
        float corr = exp(m - m_new);
        float w = exp(score - m_new);
        l = l * corr + w;
        device const half* v = Vh + (uint)j * (uint)p.v_head_dim;
        for (int i = 0; i < 4; i++) {
            int d = int(lane) + 32*i;
            if (d < p.v_head_dim) acc[i] = acc[i] * corr + w * float(v[d]);
        }
        m = m_new;
    }

    device float* o = O + p.o_offset + (uint)h * p.o_stride_head + (uint)s * p.o_stride_pos;
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int i = 0; i < 4; i++) {
        int d = int(lane) + 32*i;
        if (d < p.v_head_dim) o[d] = acc[i] * inv;
    }
}

// ---------------------------------------------------------------------------
// Flash-attention prefill. grid ((S+Q-1)/Q, num_heads), threads (32, NSG): one
// threadgroup handles a tile of Q=8 or Q=16 query rows for one head. DK, DV,
// Q, and the SIMD-group count are function constants, so the host can choose
// the best compile-time QK/PV split for the GPU.
//   - K/V read straight from device into the simdgroup MMA (FP16 cache rows are
//     contiguous), so no threadgroup staging or per-block barrier.
//   - QK/PV matmuls are cooperative over all Q queries; KV columns are split
//     across simdgroups for the QK GEMM, each writing its stripe into `ss`.
//   - Online-softmax state (M/S) is split by query row (SG sgitg owns rows
//     {sgitg, sgitg+NSG, ...}), so there is no cross-simdgroup merge.
//   - O accumulator lives in threadgroup `so` (float) and is rescaled elementwise.
// Threadgroup memory: sq[Q*DK] half + so[Q*PV] float + ss[Q*SH] float
//   (C=64, SH=C+40, PV=PAD(DV,64)); 9.25/18.5 KB for Q=8/16 at
//   DK=DV=128.
//
// dk/dv/NSG/QT are function constants: the specialized pipeline fully unrolls
// the MMA loops and bakes in the threadgroup work split.
constant int FC_SDPA_DK [[function_constant(0)]];
constant int FC_SDPA_DV [[function_constant(1)]];
constant int FC_SDPA_NSG [[function_constant(9)]];
constant int FC_SDPA_QT [[function_constant(11)]];
constant bool FC_SDPA_HAS_DK = is_function_constant_defined(FC_SDPA_DK);
constant bool FC_SDPA_HAS_DV = is_function_constant_defined(FC_SDPA_DV);

kernel void sdpa_prefill_fa2_f32(
    device const float*   Q       [[buffer(0)]],
    device const half*    KC      [[buffer(1)]],
    device const half*    VC      [[buffer(2)]],
    device float*         O       [[buffer(4)]],
    device const float*   MASK    [[buffer(5)]],
    constant SdpaParams&  p       [[buffer(3)]],
    threadgroup half*     shmem   [[threadgroup(0)]],
    uint2  tgpig                  [[threadgroup_position_in_grid]],
    ushort tiisg                  [[thread_index_in_simdgroup]],
    ushort sgitg                  [[simdgroup_index_in_threadgroup]])
{
    const short QT  = (short)FC_SDPA_QT;
    const short C   = 64;             // KV columns per block
    const short NSG = (short)FC_SDPA_NSG;
    const short NW  = 32;             // simd width
    const short NQ = QT / NSG;        // query rows owned by each simdgroup
    const short SH  = C + 40;         // avoid power-of-two threadgroup bank stride

    const short DK  = FC_SDPA_HAS_DK ? (short)FC_SDPA_DK : (short)p.head_dim;
    const short DV  = FC_SDPA_HAS_DV ? (short)FC_SDPA_DV : (short)p.v_head_dim;
    const short DK8 = DK / 8;
    const short PV  = ((DV + 63) / 64) * 64;   // PAD(DV,64)
    const short PV4 = PV / 4;
    const short PV8 = PV / 8;
    const short DV4 = DV / 4;

    const short h  = (short)tgpig.y;
    const short q0 = (short)tgpig.x * QT;       // first query row of this tile
    if (q0 >= p.src_seqlen) return;

    const short heads_per_group = (short)(p.num_heads / p.num_kv_heads);
    const short kv_h = h / heads_per_group;

    // Direct-global K/V base pointers for this kv-head. Cache is laid out per
    // kv-head [kv_head, position, feature] FP16, so rows are contiguous with
    // row stride = head_dim (K) / v_head_dim (V), per-head stride = dim*max_seq_len.
    device const half* Kbase = KC + p.k_cache_offset + (uint)kv_h * ((uint)DK * (uint)p.max_seq_len);
    device const half* Vbase = VC + p.v_cache_offset + (uint)kv_h * ((uint)DV * (uint)p.max_seq_len);

    // ---- Threadgroup layout ----
    // sq [QT*DK] half | so [QT*PV] float (O accumulator, matches o_t=float) |
    // ss [QT*SH] float. so/ss are float so simdgroup_load/store of float8x8 work.
    threadgroup half*   sq  = shmem;
    threadgroup float*  so  = (threadgroup float*)(sq + QT * DK);
    threadgroup float4* so4 = (threadgroup float4*)so;
    threadgroup float*  ss  = so + QT * PV;
    threadgroup float2* ss2 = (threadgroup float2*)ss;

    const short tiitg = sgitg * NW + tiisg;   // 0..127

    // Stage Q tile into sq (all threads cooperate), cast FP32 -> half.
    for (short idx = tiitg; idx < QT * DK; idx += NSG * NW) {
        short row = idx / DK, d = idx % DK;
        short s = q0 + row;
        sq[idx] = (s < p.src_seqlen)
            ? (half)Q[p.q_offset + (uint)h * p.q_stride_head + (uint)s * p.q_stride_pos + d]
            : (half)0;
    }
    // Zero O accumulator and score buffer.
    for (short idx = tiitg; idx < QT * PV4; idx += NSG * NW) so4[idx] = float4(0);
    for (short idx = tiitg; idx < QT * SH; idx += NSG * NW) ss[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-simdgroup online-softmax state (registers), one per owned query row.
    float S[4], M[4];                 // max NQ at NSG=2
    for (short jj = 0; jj < NQ; jj++) { S[jj] = 0.0f; M[jj] = -INFINITY; }

    // Causal upper bound on keys for this whole tile (last query row in the tile).
    int tile_key_limit = p.dst_seqlen;
    if (p.causal != 0) {
        int last_s = min((int)q0 + QT - 1, p.src_seqlen - 1);
        tile_key_limit = p.past_seqlen + last_s + 1;
        if (tile_key_limit > p.dst_seqlen) tile_key_limit = p.dst_seqlen;
    }

    for (int ic0 = 0; ic0 * C < tile_key_limit; ++ic0) {
        const int ic = ic0 * C;

        // ---- Q * K^T : direct-global, KV columns split across simdgroups ----
        // Each SG computes NC = (C/8)/NSG column-blocks of 8 keys; stripe stride 8*NSG.
        {
            device const half*  pk = Kbase + (uint)ic * (uint)DK + (uint)sgitg * (8u * (uint)DK);
            threadgroup float*  ps = ss + sgitg * 8;
            const short NC = (C / 8) / NSG;   // = 1
            for (short cc = 0; cc < NC; ++cc) {
                simdgroup_float8x8 mqk[2];
                #pragma unroll
                for (short qb = 0; qb < QT / 8; ++qb)
                    mqk[qb] =
                        make_filled_simdgroup_matrix<float, 8>(
                            0.0f);
                for (short i = 0; i < DK8; ++i) {
                    simdgroup_half8x8 mk;
                    simdgroup_load(mk, pk + 8 * i, DK, 0, true);   // transpose -> K^T
                    #pragma unroll
                    for (short qb = 0;
                         qb < QT / 8; ++qb) {
                        simdgroup_half8x8 mq;
                        simdgroup_load(
                            mq,
                            sq + qb * 8 * DK + 8 * i,
                            DK, 0, false);
                        simdgroup_multiply_accumulate(
                            mqk[qb], mq, mk, mqk[qb]);
                    }
                }
                #pragma unroll
                for (short qb = 0; qb < QT / 8; ++qb) {
                    simdgroup_store(
                        mqk[qb], ps + qb * 8 * SH,
                        SH, 0, false);
                }
                pk += 8u * (uint)NSG * (uint)DK;
                ps += 8 * NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- online softmax over this block's C columns (query-row split) ----
        // Each SG owns rows j = jj*NSG + sgitg. Lane tiisg owns two score cols.
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const short s = q0 + j;
            const float m_old = M[jj];

            float2 s2 = ss2[(j * SH) / 2 + tiisg] * p.scale;

            // Causal + optional additive mask on the two cols this lane owns.
            short col0 = 2 * tiisg;
            for (short e = 0; e < 2; ++e) {
                short col = col0 + e;
                int key = ic + col;
                bool valid = (col < C) && (key < p.dst_seqlen)
                          && ((p.causal == 0) || (key <= p.past_seqlen + s));
                if (!valid) {
                    s2[e] = -INFINITY;
                } else if (p.has_mask != 0) {
                    s2[e] += MASK[p.mask_offset + (uint)s * (uint)p.mask_stride_row + (uint)key];
                }
            }

            M[jj] = simd_max(max(m_old, max(s2[0], s2[1])));
            const float  ms  = (m_old == -INFINITY) ? 0.0f : exp(m_old - M[jj]);
            float2 vs2;
            vs2[0] = (s2[0] == -INFINITY) ? 0.0f : exp(s2[0] - M[jj]);
            vs2[1] = (s2[1] == -INFINITY) ? 0.0f : exp(s2[1] - M[jj]);
            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[(j * SH) / 2 + tiisg] = vs2;   // P matrix

            // rescale threadgroup-O for this row (all DV lanes cooperate within SG).
            for (short i = tiisg; i < DV4; i += NW) so4[j * PV4 + i] *= ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O += P * V : direct-global V read ----
        // so holds [QT x PV] O accumulators. Split the PV8 output 8-col blocks
        // across simdgroups. P (=ss, float) loads as float8x8; V loads as half8x8
        // straight from device; the MMA takes float(P) x half(V) -> float(O).
        {
            const short NO = PV8 / NSG;   // output 8x8 col blocks per SG
            simdgroup_float8x8 lo[8];     // (QT/8)*NO <= 8
            #pragma unroll
            for (short qb = 0; qb < QT / 8; ++qb) {
                threadgroup float* sot =
                    so + qb * 8 * PV + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(
                        lo[qb * NO + ii],
                        sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
            // pv columns for this SG's output stripe: V[key, dcol]; V row stride = DV.
            device const half* pv = Vbase + (uint)ic * (uint)DV + (uint)(8 * sgitg);
            for (short cc = 0; cc < C / 8; ++cc) {
                simdgroup_float8x8 vs[2];
                #pragma unroll
                for (short qb = 0;
                     qb < QT / 8; ++qb) {
                    simdgroup_load(
                        vs[qb],
                        ss + qb * 8 * SH + 8 * cc,
                        SH, 0, false);
                }
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, pv + (uint)(8 * NSG * ii), DV, 0, false); // V[cc-block, dcol]
                    #pragma unroll
                    for (short qb = 0;
                         qb < QT / 8; ++qb) {
                        simdgroup_multiply_accumulate(
                            lo[qb * NO + ii],
                            vs[qb], mv,
                            lo[qb * NO + ii]);
                    }
                }
                pv += (uint)8 * (uint)DV;   // advance 8 keys
            }
            #pragma unroll
            for (short qb = 0; qb < QT / 8; ++qb) {
                threadgroup float* sot =
                    so + qb * 8 * PV + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(
                        lo[qb * NO + ii],
                        sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalized store — honor mollm o_offset / o_stride (NOT contiguous).
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        const short s = q0 + j;
        if (s >= p.src_seqlen) continue;
        const float inv = (S[jj] > 0.0f) ? (1.0f / S[jj]) : 0.0f;
        device float* o = O + p.o_offset + (uint)h * p.o_stride_head + (uint)s * p.o_stride_pos;
        for (short i = tiisg; i < DV; i += NW) o[i] = so[j * PV + i] * inv;
    }
}

// ---------------------------------------------------------------------------
// SDPA decode fast path for conventional 128-wide Q/K/V heads. This mirrors
// the 192/128 online-softmax kernel below while removing one QK float4 per
// half-warp. It avoids the generic kernel's full score buffer and three
// separate passes over the sequence.
// ---------------------------------------------------------------------------
kernel void sdpa_decode_128_128_f32(
    device const float*   Q       [[buffer(0)]],
    device const half*    KC      [[buffer(1)]],
    device const half*    VC      [[buffer(2)]],
    device float*         O       [[buffer(4)]],
    device const float*   MASK    [[buffer(5)]],
    constant SdpaParams&  p       [[buffer(3)]],
    uint   h                      [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort sg                     [[simdgroup_index_in_threadgroup]])
{
    constexpr short DK = 128;
    constexpr short DV = 128;
    constexpr short C = 32;
    constexpr short NL = 16;
    constexpr short NSG = 8;

    const short tx = lane & 15;
    const short ty = lane >> 4;
    const int heads_per_group = p.num_heads / p.num_kv_heads;
    const int kv_h = int(h) / heads_per_group;
    const int key_limit = min(
        p.dst_seqlen,
        p.causal ? p.past_seqlen + 1 : p.dst_seqlen);

    device const float* q =
        Q + p.q_offset + h * p.q_stride_head;
    device const half* Kh = KC + p.k_cache_offset
        + (uint)kv_h * ((uint)DK * (uint)p.max_seq_len);
    device const half* Vh = VC + p.v_cache_offset
        + (uint)kv_h * ((uint)DV * (uint)p.max_seq_len);

    threadgroup float4 q4[DK / 4];
    threadgroup float scores[8 * C];
    threadgroup float4 og[8 * (DV / 4)];
    threadgroup float state_m[8];
    threadgroup float state_s[8];

    const short ti = sg * 32 + lane;
    if (ti < DK / 4)
        q4[ti] = ((device const float4*)q)[ti];
    for (short i = ti; i < NSG * (DV / 4); i += NSG * 32)
        og[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;
    for (int ic = int(sg) * C; ic < key_limit; ic += NSG * C) {
        #pragma unroll
        for (short cc = 0; cc < C / 2; ++cc) {
            const int key = ic + 2 * cc + ty;
            float dotqk = 0.0f;
            if (key < key_limit) {
                #pragma unroll
                for (short ii = 0; ii < DK / 4 / NL; ++ii) {
                    const short d4 = ii * NL + tx;
                    device const half4* k4 =
                        (device const half4*)(
                            Kh + (uint)key * (uint)DK);
                    dotqk += dot(q4[d4], float4(k4[d4]));
                }
            }
            const float even_sum =
                simd_sum(ty == 0 ? dotqk : 0.0f);
            const float odd_sum =
                simd_sum(ty == 1 ? dotqk : 0.0f);
            if (lane == 0)
                scores[sg * C + 2 * cc] = even_sum;
            if (lane == 16)
                scores[sg * C + 2 * cc + 1] = odd_sum;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const int key = ic + lane;
        float score = key < key_limit
            ? scores[sg * C + lane] * p.scale
            : -INFINITY;
        if (p.has_mask && key < key_limit)
            score += MASK[p.mask_offset + (uint)key];

        const float Mnew = simd_max(max(M, score));
        const float correction = exp(M - Mnew);
        const float weight = exp(score - Mnew);
        S = S * correction + simd_sum(weight);
        M = Mnew;
        scores[sg * C + lane] = weight;

        if (ty == 0) {
            og[sg * (DV / 4) + tx] *= correction;
            og[sg * (DV / 4) + NL + tx] *= correction;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo0 = 0.0f;
        float4 lo1 = 0.0f;
        #pragma unroll
        for (short cc = 0; cc < C / 2; ++cc) {
            const int value_key = ic + 2 * cc + ty;
            if (value_key < key_limit) {
                const float w = scores[sg * C + 2 * cc + ty];
                device const half4* v4 =
                    (device const half4*)(
                        Vh + (uint)value_key * (uint)DV);
                lo0 += float4(v4[tx]) * w;
                lo1 += float4(v4[NL + tx]) * w;
            }
        }
        lo0.x += simd_shuffle_down(lo0.x, 16);
        lo0.y += simd_shuffle_down(lo0.y, 16);
        lo0.z += simd_shuffle_down(lo0.z, 16);
        lo0.w += simd_shuffle_down(lo0.w, 16);
        lo1.x += simd_shuffle_down(lo1.x, 16);
        lo1.y += simd_shuffle_down(lo1.y, 16);
        lo1.z += simd_shuffle_down(lo1.z, 16);
        lo1.w += simd_shuffle_down(lo1.w, 16);
        if (ty == 0) {
            og[sg * (DV / 4) + tx] += lo0;
            og[sg * (DV / 4) + NL + tx] += lo1;
        }
    }

    if (lane == 0) {
        state_m[sg] = M;
        state_s[sg] = S;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        const float mlane =
            lane < NSG ? state_m[lane] : -INFINITY;
        const float maximum = simd_max(mlane);
        const float factor =
            lane < NSG ? exp(state_m[lane] - maximum) : 0.0f;
        const float denominator = simd_sum(
            lane < NSG ? state_s[lane] * factor : 0.0f);
        const float inverse =
            denominator > 0.0f ? 1.0f / denominator : 0.0f;
        device float4* out4 = (device float4*)(
            O + p.o_offset + h * p.o_stride_head);
        for (short d4 = lane; d4 < DV / 4; d4 += 32) {
            float4 value = 0.0f;
            #pragma unroll
            for (short s = 0; s < NSG; ++s)
                value +=
                    og[s * (DV / 4) + d4] *
                    exp(state_m[s] - maximum);
            out4[d4] = value * inverse;
        }
    }
}

// ---------------------------------------------------------------------------
// SDPA decode fast path for MLA-expanded heads (DK=192, DV=128).
//
// Eight SIMD groups split the KV cache into 32-row tiles.  Each group computes
// QK, online softmax, and P*V in one pass, keeping its 128-wide output
// accumulator in threadgroup memory.  The online-softmax states are merged
// once at the end.  This avoids the generic decode kernel's full score buffer,
// three global passes over the sequence, and threadgroup-wide barriers between
// those passes.
//
// grid: threadgroups = num_heads, threads/tg = 256 (8 SIMD groups).
// ---------------------------------------------------------------------------
kernel void sdpa_decode_192_128_f32(
    device const float*   Q       [[buffer(0)]],
    device const half*    KC      [[buffer(1)]],
    device const half*    VC      [[buffer(2)]],
    device float*         O       [[buffer(4)]],
    device const float*   MASK    [[buffer(5)]],
    constant SdpaParams&  p       [[buffer(3)]],
    uint   h                      [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort sg                     [[simdgroup_index_in_threadgroup]])
{
    constexpr short DK = 192;
    constexpr short DV = 128;
    constexpr short C = 32;
    constexpr short NL = 16;  // lanes collaborating on one KV row
    constexpr short NSG = 8;

    const short tx = lane & 15;
    const short ty = lane >> 4;
    const int heads_per_group = p.num_heads / p.num_kv_heads;
    const int kv_h = int(h) / heads_per_group;
    const int key_limit = min(p.dst_seqlen, p.causal ? p.past_seqlen + 1 : p.dst_seqlen);

    device const float* q = Q + p.q_offset + h * p.q_stride_head;
    device const half* Kh = KC + p.k_cache_offset
        + (uint)kv_h * ((uint)DK * (uint)p.max_seq_len);
    device const half* Vh = VC + p.v_cache_offset
        + (uint)kv_h * ((uint)DV * (uint)p.max_seq_len);

    threadgroup float4 q4[DK / 4];
    threadgroup float scores[NSG * C];
    threadgroup float4 og[NSG * (DV / 4)];
    threadgroup float state_m[NSG];
    threadgroup float state_s[NSG];

    // SIMD groups cooperate on staging Q and clearing the accumulators.
    const short ti = sg * 32 + lane;
    if (ti < DK / 4) {
        device const float4* srcq = (device const float4*)q;
        q4[ti] = srcq[ti];
    }
    for (short i = ti; i < NSG * (DV / 4); i += NSG * 32) og[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;

    for (int ic = int(sg) * C; ic < key_limit; ic += NSG * C) {
        // Lanes [0,15] and [16,31] handle the even and odd keys respectively.
        // Mask the other half to zero before each full-SIMD reduction. This
        // retains two-key parallelism without relying on unsupported 16-lane
        // subgroup shuffle semantics.
        #pragma unroll
        for (short cc = 0; cc < C / 2; ++cc) {
            const int kcc = ic + 2 * cc + ty;
            float dotqk = 0.0f;
            if (kcc < key_limit) {
                device const half4* k4 =
                    (device const half4*)(Kh + (uint)kcc * (uint)DK);
                #pragma unroll
                for (short ii = 0; ii < DK / 4 / NL; ++ii) {
                    const short d4 = ii * NL + tx;
                    dotqk += dot(q4[d4], float4(k4[d4]));
                }
            }
            const float even_sum = simd_sum(ty == 0 ? dotqk : 0.0f);
            const float odd_sum  = simd_sum(ty == 1 ? dotqk : 0.0f);
            if (lane == 0)
                scores[sg * C + 2 * cc] = even_sum;
            if (lane == 16)
                scores[sg * C + 2 * cc + 1] = odd_sum;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int key = ic + lane;
        float score_for_lane = key < key_limit
            ? scores[sg * C + lane] * p.scale
            : -INFINITY;
        if (p.has_mask && key < key_limit)
            score_for_lane += MASK[p.mask_offset + (uint)key];

        const float Mnew = simd_max(max(M, score_for_lane));
        const float corr = exp(M - Mnew);
        const float weight = exp(score_for_lane - Mnew);
        S = S * corr + simd_sum(weight);
        M = Mnew;
        scores[sg * C + lane] = weight;

        // Scale the prior online-softmax accumulator before adding this tile.
        if (ty == 0) {
            og[sg * (DV / 4) + tx] *= corr;
            og[sg * (DV / 4) + NL + tx] *= corr;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo0 = 0.0f;
        float4 lo1 = 0.0f;
        #pragma unroll
        for (short cc = 0; cc < C / 2; ++cc) {
            const int vkey = ic + 2 * cc + ty;
            if (vkey < key_limit) {
                device const half4* v4 =
                    (device const half4*)(Vh + (uint)vkey * (uint)DV);
                const float w = scores[sg * C + 2 * cc + ty];
                lo0 += float4(v4[tx]) * w;
                lo1 += float4(v4[NL + tx]) * w;
            }
        }
        lo0.x += simd_shuffle_down(lo0.x, 16);
        lo0.y += simd_shuffle_down(lo0.y, 16);
        lo0.z += simd_shuffle_down(lo0.z, 16);
        lo0.w += simd_shuffle_down(lo0.w, 16);
        lo1.x += simd_shuffle_down(lo1.x, 16);
        lo1.y += simd_shuffle_down(lo1.y, 16);
        lo1.z += simd_shuffle_down(lo1.z, 16);
        lo1.w += simd_shuffle_down(lo1.w, 16);
        if (ty == 0) {
            og[sg * (DV / 4) + tx] += lo0;
            og[sg * (DV / 4) + NL + tx] += lo1;
        }
    }

    if (lane == 0) {
        state_m[sg] = M;
        state_s[sg] = S;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Merge all independent online-softmax states and store the result.
    if (sg == 0) {
        const float mlane = lane < NSG ? state_m[lane] : -INFINITY;
        const float Mf = simd_max(mlane);
        const float alane = lane < NSG ? exp(state_m[lane] - Mf) : 0.0f;
        const float denom = simd_sum(lane < NSG ? state_s[lane] * alane : 0.0f);
        const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        device float4* out4 = (device float4*)(
            O + p.o_offset + h * p.o_stride_head);
        for (short d4 = lane; d4 < DV / 4; d4 += 32) {
            float4 acc = 0.0f;
            #pragma unroll
            for (short s = 0; s < NSG; ++s)
                acc += og[s * (DV / 4) + d4] * exp(state_m[s] - Mf);
            out4[d4] = acc * inv;
        }
    }
}

// ---------------------------------------------------------------------------
// Long-context companion to sdpa_decode_192_128_f32.  Thirty-two independent
// workgroups per head split the KV blocks and write online-softmax partials;
// sdpa_decode_192_128_reduce_f32 combines them.  This exposes enough parallelism
// once a single 8-SIMD-group workgroup has multiple KV tiles per SIMD group.
// ---------------------------------------------------------------------------
kernel void sdpa_decode_192_128_partial_f32(
    device const float*   Q       [[buffer(0)]],
    device const half*    KC      [[buffer(1)]],
    device const half*    VC      [[buffer(2)]],
    device const float*   MASK    [[buffer(5)]],
    device float*         PARTIAL [[buffer(7)]],
    constant SdpaParams&  p       [[buffer(3)]],
    constant int&         nparts  [[buffer(6)]],
    uint2  tg                     [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]])
{
    constexpr short DK = 192;
    constexpr short DV = 128;
    constexpr short C = 32;
    constexpr short NL = 16;

    const short tx = lane & 15;
    const short ty = lane >> 4;
    const int part = int(tg.x);
    const int h = int(tg.y);
    const int heads_per_group = p.num_heads / p.num_kv_heads;
    const int kv_h = h / heads_per_group;
    const int key_limit = min(p.dst_seqlen, p.causal ? p.past_seqlen + 1 : p.dst_seqlen);

    device const float4* q4 =
        (device const float4*)(Q + p.q_offset + (uint)h * p.q_stride_head);
    device const half* Kh = KC + p.k_cache_offset
        + (uint)kv_h * ((uint)DK * (uint)p.max_seq_len);
    device const half* Vh = VC + p.v_cache_offset
        + (uint)kv_h * ((uint)DV * (uint)p.max_seq_len);

    threadgroup float scores[C];
    float4 out0 = 0.0f;
    float4 out1 = 0.0f;
    float M = -INFINITY;
    float S = 0.0f;

    for (int ic = part * C; ic < key_limit; ic += nparts * C) {
        #pragma unroll
        for (short cc = 0; cc < C / 2; ++cc) {
            const int key = ic + 2 * cc + ty;
            float dotqk = 0.0f;
            if (key < key_limit) {
                device const half4* k4 =
                    (device const half4*)(Kh + (uint)key * (uint)DK);
                #pragma unroll
                for (short ii = 0; ii < DK / 4 / NL; ++ii) {
                    const short d4 = ii * NL + tx;
                    dotqk += dot(q4[d4], float4(k4[d4]));
                }
            }
            const float even_sum = simd_sum(ty == 0 ? dotqk : 0.0f);
            const float odd_sum  = simd_sum(ty == 1 ? dotqk : 0.0f);
            if (lane == 0)
                scores[2 * cc] = even_sum;
            if (lane == 16)
                scores[2 * cc + 1] = odd_sum;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int key = ic + lane;
        float score = key < key_limit
            ? scores[lane] * p.scale
            : -INFINITY;
        if (p.has_mask && key < key_limit)
            score += MASK[p.mask_offset + (uint)key];

        const float Mnew = simd_max(max(M, score));
        const float corr = exp(M - Mnew);
        const float weight = exp(score - Mnew);
        S = S * corr + simd_sum(weight);
        M = Mnew;
        scores[lane] = weight;
        if (ty == 0) {
            out0 *= corr;
            out1 *= corr;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo0 = 0.0f;
        float4 lo1 = 0.0f;
        #pragma unroll
        for (short cc = 0; cc < C / 2; ++cc) {
            const int vkey = ic + 2 * cc + ty;
            if (vkey < key_limit) {
                device const half4* v4 =
                    (device const half4*)(Vh + (uint)vkey * (uint)DV);
                const float w = scores[2 * cc + ty];
                lo0 += float4(v4[tx]) * w;
                lo1 += float4(v4[NL + tx]) * w;
            }
        }
        lo0.x += simd_shuffle_down(lo0.x, 16);
        lo0.y += simd_shuffle_down(lo0.y, 16);
        lo0.z += simd_shuffle_down(lo0.z, 16);
        lo0.w += simd_shuffle_down(lo0.w, 16);
        lo1.x += simd_shuffle_down(lo1.x, 16);
        lo1.y += simd_shuffle_down(lo1.y, 16);
        lo1.z += simd_shuffle_down(lo1.z, 16);
        lo1.w += simd_shuffle_down(lo1.w, 16);
        if (ty == 0) {
            out0 += lo0;
            out1 += lo1;
        }
    }

    const uint base = ((uint)h * (uint)nparts + (uint)part) * (DV + 2);
    if (ty == 0) {
        device float4* po = (device float4*)(PARTIAL + base);
        po[tx] = out0;
        po[NL + tx] = out1;
    }
    if (lane == 0) {
        PARTIAL[base + DV] = M;
        PARTIAL[base + DV + 1] = S;
    }
}

kernel void sdpa_decode_192_128_reduce_f32(
    device const float*   PARTIAL [[buffer(7)]],
    device float*         O       [[buffer(4)]],
    constant SdpaParams&  p       [[buffer(3)]],
    constant int&         nparts  [[buffer(6)]],
    uint   h                      [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]])
{
    constexpr short DV = 128;
    const uint stride = DV + 2;
    const float mlane = lane < nparts
        ? PARTIAL[((uint)h * (uint)nparts + lane) * stride + DV]
        : -INFINITY;
    const float Mf = simd_max(mlane);
    const float slane = lane < nparts
        ? PARTIAL[((uint)h * (uint)nparts + lane) * stride + DV + 1]
            * exp(mlane - Mf)
        : 0.0f;
    const float inv = 1.0f / simd_sum(slane);

    device float4* out4 =
        (device float4*)(O + p.o_offset + h * p.o_stride_head);
    const short d4 = lane;
    float4 acc = 0.0f;
    for (int part = 0; part < nparts; ++part) {
        const uint base = ((uint)h * (uint)nparts + (uint)part) * stride;
        const float mp = PARTIAL[base + DV];
        acc += ((device const float4*)(PARTIAL + base))[d4] * exp(mp - Mf);
    }
    out4[d4] = acc * inv;
}

// ---------------------------------------------------------------------------
// SDPA decode fast path for Qwen3.5 full-attention heads (DK=DV=256).
//
// This is the 256-wide counterpart of sdpa_decode_192_128_f32. Eight SIMD
// groups independently traverse 32-key tiles and keep an online-softmax output
// accumulator. The final merge touches only eight partial states, avoiding the
// generic kernel's full score buffer and separate score/softmax/value passes.
//
// grid: threadgroups = num_heads, threads/tg = 256 (8 SIMD groups).
// ---------------------------------------------------------------------------
kernel void sdpa_decode_256_256_f32(
    device const float*   Q       [[buffer(0)]],
    device const half*    KC      [[buffer(1)]],
    device const half*    VC      [[buffer(2)]],
    device float*         O       [[buffer(4)]],
    device const float*   MASK    [[buffer(5)]],
    constant SdpaParams&  p       [[buffer(3)]],
    uint   h                      [[threadgroup_position_in_grid]],
    ushort lane                   [[thread_index_in_simdgroup]],
    ushort sg                     [[simdgroup_index_in_threadgroup]])
{
    constexpr short DK = 256;
    constexpr short DV = 256;
    constexpr short C = 32;
    constexpr short NL = 16;
    constexpr short NSG = 8;

    const short tx = lane & 15;
    const short ty = lane >> 4;
    const int heads_per_group = p.num_heads / p.num_kv_heads;
    const int kv_h = int(h) / heads_per_group;
    const int key_limit =
        min(p.dst_seqlen,
            p.causal ? p.past_seqlen + 1 : p.dst_seqlen);

    device const float* q =
        Q + p.q_offset + h * p.q_stride_head;
    device const half* Kh = KC + p.k_cache_offset
        + (uint)kv_h * ((uint)DK * (uint)p.max_seq_len);
    device const half* Vh = VC + p.v_cache_offset
        + (uint)kv_h * ((uint)DV * (uint)p.max_seq_len);

    threadgroup float4 q4[DK / 4];
    threadgroup float scores[NSG * C];
    threadgroup float4 og[NSG * (DV / 4)];
    threadgroup float state_m[NSG];
    threadgroup float state_s[NSG];

    const short ti = sg * 32 + lane;
    if (ti < DK / 4)
        q4[ti] = ((device const float4*)q)[ti];
    for (short i = ti; i < NSG * (DV / 4); i += NSG * 32)
        og[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;

    for (int ic = int(sg) * C; ic < key_limit; ic += NSG * C) {
        #pragma unroll
        for (short cc = 0; cc < C / 2; ++cc) {
            const int key = ic + 2 * cc + ty;
            float dotqk = 0.0f;
            if (key < key_limit) {
                device const half4* k4 =
                    (device const half4*)(Kh + (uint)key * (uint)DK);
                #pragma unroll
                for (short ii = 0; ii < DK / 4 / NL; ++ii) {
                    const short d4 = ii * NL + tx;
                    dotqk += dot(q4[d4], float4(k4[d4]));
                }
            }
            const float even_sum =
                simd_sum(ty == 0 ? dotqk : 0.0f);
            const float odd_sum =
                simd_sum(ty == 1 ? dotqk : 0.0f);
            if (lane == 0)
                scores[sg * C + 2 * cc] = even_sum;
            if (lane == 16)
                scores[sg * C + 2 * cc + 1] = odd_sum;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const int key = ic + lane;
        float score = key < key_limit
            ? scores[sg * C + lane] * p.scale
            : -INFINITY;
        if (p.has_mask && key < key_limit)
            score += MASK[p.mask_offset + (uint)key];

        const float Mnew = simd_max(max(M, score));
        const float corr = exp(M - Mnew);
        const float weight = exp(score - Mnew);
        S = S * corr + simd_sum(weight);
        M = Mnew;
        scores[sg * C + lane] = weight;

        if (ty == 0) {
            #pragma unroll
            for (short block = 0; block < DV / 4 / NL; ++block)
                og[sg * (DV / 4) + block * NL + tx] *= corr;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo[DV / 4 / NL];
        #pragma unroll
        for (short block = 0; block < DV / 4 / NL; ++block)
            lo[block] = 0.0f;
        #pragma unroll
        for (short cc = 0; cc < C / 2; ++cc) {
            const int vkey = ic + 2 * cc + ty;
            if (vkey < key_limit) {
                device const half4* v4 =
                    (device const half4*)(Vh + (uint)vkey * (uint)DV);
                const float w = scores[sg * C + 2 * cc + ty];
                #pragma unroll
                for (short block = 0; block < DV / 4 / NL; ++block)
                    lo[block] +=
                        float4(v4[block * NL + tx]) * w;
            }
        }
        #pragma unroll
        for (short block = 0; block < DV / 4 / NL; ++block) {
            lo[block].x += simd_shuffle_down(lo[block].x, 16);
            lo[block].y += simd_shuffle_down(lo[block].y, 16);
            lo[block].z += simd_shuffle_down(lo[block].z, 16);
            lo[block].w += simd_shuffle_down(lo[block].w, 16);
            if (ty == 0)
                og[sg * (DV / 4) + block * NL + tx] += lo[block];
        }
    }

    if (lane == 0) {
        state_m[sg] = M;
        state_s[sg] = S;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        const float mlane =
            lane < NSG ? state_m[lane] : -INFINITY;
        const float Mf = simd_max(mlane);
        const float alane =
            lane < NSG ? exp(state_m[lane] - Mf) : 0.0f;
        const float denom = simd_sum(
            lane < NSG ? state_s[lane] * alane : 0.0f);
        const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        device float4* out4 = (device float4*)(
            O + p.o_offset + h * p.o_stride_head);
        for (short d4 = lane; d4 < DV / 4; d4 += 32) {
            float4 acc = 0.0f;
            #pragma unroll
            for (short s = 0; s < NSG; ++s)
                acc += og[s * (DV / 4) + d4]
                     * exp(state_m[s] - Mf);
            out4[d4] = acc * inv;
        }
    }
}

// ---------------------------------------------------------------------------
// SDPA decode (src_seqlen == 1): one THREADGROUP per head, TG threads split the
// key loop for GPU parallelism (the per-thread sdpa_f32 above uses only
// num_heads threads for decode, starving the GPU as context grows).
// Pass 1: each thread computes scores for its key subset -> threadgroup max+sum
//         (two-pass stable softmax). Pass 2: weighted V accumulation reduced
//         across threads into the output.
// grid: threadgroups = num_heads, threads/tg = TG (256).
// ---------------------------------------------------------------------------
kernel void sdpa_decode_f32(
    device const float*   Q       [[buffer(0)]],
    device const half*    KC      [[buffer(1)]],
    device const half*    VC      [[buffer(2)]],
    device float*         O       [[buffer(4)]],
    device const float*   MASK    [[buffer(5)]],
    constant SdpaParams&  p       [[buffer(3)]],
    uint  h                       [[threadgroup_position_in_grid]],
    uint  tid                     [[thread_position_in_threadgroup]],
    uint  tcount                  [[threads_per_threadgroup]])
{
    int heads_per_group = p.num_heads / p.num_kv_heads;
    int kv_h = int(h) / heads_per_group;

    device const float* q = Q + p.q_offset + (uint)h * p.q_stride_head;
    device const half*  Kh = KC + p.k_cache_offset + (uint)kv_h * ((uint)p.head_dim * (uint)p.max_seq_len);
    device const half*  Vh = VC + p.v_cache_offset + (uint)kv_h * ((uint)p.v_head_dim * (uint)p.max_seq_len);

    int key_limit = p.causal ? (p.past_seqlen + 1) : p.dst_seqlen;  // s==0 for decode
    if (key_limit > p.dst_seqlen) key_limit = p.dst_seqlen;

    uint lane = tid & 31u, sg = tid >> 5, n_sg = (tcount + 31u) / 32u;

    // Stage q (head_dim floats) in threadgroup memory for fast reuse.
    threadgroup float qs[256];
    for (int d = int(tid); d < p.head_dim; d += int(tcount)) qs[d] = q[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 1: compute the QK score for each key ONCE, store to wbuf. Also track
    // the running max — avoids recomputing the head_dim dot a second time (the
    // old kernel did the full QK reduction twice). head_dim%4==0 (=128) so the
    // dot is float4-vectorized.
    threadgroup float wbuf[2048];  // score then weight per key (past+1 <= 2048)
    threadgroup const float4* qs4 = (threadgroup const float4*)qs;
    int hd4 = p.head_dim >> 2;
    float local_max = -INFINITY;
    for (int j = int(tid); j < key_limit; j += int(tcount)) {
        device const half4* k4 = (device const half4*)(Kh + (uint)j * (uint)p.head_dim);
        float acc = 0.0f;
        for (int d = 0; d < hd4; d++) { float4 kv = float4(k4[d]); acc += dot(qs4[d], kv); }
        float score = acc * p.scale;
        if (p.has_mask) score += MASK[p.mask_offset + (uint)j];
        wbuf[j] = score;
        local_max = max(local_max, score);
    }
    // Two-level simd reduction for max (few barriers vs the old 8-level tree).
    threadgroup float red[32];
    local_max = simd_max(local_max);
    if (lane == 0) red[sg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = simd_max((lane < n_sg) ? red[lane] : -INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: exp in place (read stored score, no QK recompute) + sum.
    float local_sum = 0.0f;
    for (int j = int(tid); j < key_limit; j += int(tcount)) {
        float w = exp(wbuf[j] - gmax);
        wbuf[j] = w;
        local_sum += w;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) red[sg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gsum = simd_sum((lane < n_sg) ? red[lane] : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = (gsum > 0.0f) ? (1.0f/gsum) : 0.0f;

    // Pass 3: weighted-V, parallel over output dims (each thread owns v-dims,
    // loops all keys reading the precomputed weights — no atomics).
    device float* o = O + p.o_offset + (uint)h * p.o_stride_head;
    for (int d = int(tid); d < p.v_head_dim; d += int(tcount)) {
        float acc = 0.0f;
        for (int j = 0; j < key_limit; j++)
            acc += wbuf[j] * float(Vh[(uint)j * (uint)p.v_head_dim + d]);
        o[d] = acc * inv;
    }
}
