#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

inline float apply_activation(float v, int act);
inline float apply_activation_at(float v, int act, int n, int begin, int len);

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

#endif // MOLLM_METAL_TENSOR

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
