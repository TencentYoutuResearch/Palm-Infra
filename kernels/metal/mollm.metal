#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

#ifdef MOLLM_METAL_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// ---------------------------------------------------------------------------
// Tensor-API GEMM (Metal 4, mpp::tensor_ops) — the fast path on M5/A19+.
// C[M,N] = A[M,K] (fp32 activations) * B[N,K]^T (fp16 weights).
// Adapted from llama.cpp kernel_mul_mm (tensor path). Activations are staged in
// threadgroup memory as half [NRA(M) x NK]; weights are read directly from
// device as a 2D tensor [K x N] (row n at b_offset + n*b_row_stride, K contig,
// so as a [K,N] tensor the stride is {1, b_row_stride}). matmul2d does the MMA
// on the M5 tensor units. Output C is [N(inner), M]: element (m,n) at
// c_offset + m*c_row_stride + n → as an [M,N]-logical tensor with strides
// {c_row_stride, 1} (row=m stride c_row_stride, col=n stride 1).
//
// Tile (from llama): NRA=64 (M), NRB=128 (N), NK=32. 4 simdgroups, 128 threads.
// threadgroup(0): A staging = NRA*NK halves = 64*32 = 2048 halves = 4KB.
// grid: threadgroups = (ceil(N/128), ceil(M/64)); threads/tg = 128.
// ---------------------------------------------------------------------------
kernel void gemm_tensor_f32a_f16b_f32c(
    device const float*   A      [[buffer(0)]],
    device const half*    B      [[buffer(1)]],
    device float*         C      [[buffer(2)]],
    constant MatmulParams& p     [[buffer(3)]],
    threadgroup half*     shmem  [[threadgroup(0)]],
    uint3  tgpig                 [[threadgroup_position_in_grid]],
    ushort tiitg                 [[thread_index_in_threadgroup]])
{
    // Mirror llama's tensor kernel roles EXACTLY (this is the key fix):
    //   - llama STAGES srcA=weights into threadgroup `sa`; wraps srcB=activations
    //     as the device tensor tB. Output dst[ne0=weightN, ne1=tokenM].
    // Map to us: stage WEIGHTS (our N dim) into sa (tile NRA=64), wrap
    // ACTIVATIONS (our M dim) as device tB (dim NRB tile=128). descriptor
    // (NRB, NRA, NK, false, true) with run(mB=act, mA=wt). Output [ourN, ourM].
    const int NRA = 64;    // weights tile (our N)  — staged
    const int NRB = 128;   // activations tile (our M) — device
    const int NK  = 32;    // K chunk
    const int NUM_THREADS = 128;

    const int M = p.M, N = p.N, K = p.K;
    const int ra = (int)tgpig.y * NRA;   // first weight-row (our N) of tile
    const int rb = (int)tgpig.x * NRB;   // first activation-row (our M) of tile

    // Weights staged as tensor [NK, NRA] (default col-major: physical k + n*NK).
    // Stage sa[k + n_local*NK] = weight(ra + n_local, loop_k + k).
    threadgroup half* sa = shmem;
    auto tA = tensor(sa, dextents<int32_t,2>(NK, NRA));

    // Activations as device tensor [K, M]: element (k,m) = act(m,k) at
    // a_offset + m*a_row_stride + k → strides {1 (k), a_row_stride (m)}.
    // (fp32 activations read directly; the API converts to the compute type.)
    device const float* ptrA = A + p.a_offset;
    auto tB = tensor((device float*)ptrA, dextents<int32_t,2>(K, M),
                     array<int,2>({1, p.a_row_stride}));

    matmul2d<matmul2d_descriptor(NRB, NRA, NK, false, true, true,
             matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // A-staging unroll factor: each work-item loads UNROLL contiguous K values
    // (llama's FOR_UNROLL(16) pattern). NRA*NK/UNROLL = 2048/16 = 128 work-items
    // == one per thread, no re-stride loop, contiguous vectorizable device reads.
    const int UNROLL = 16;
    const int A_WORK = NRA * (NK / UNROLL);   // 64 * 2 = 128

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        // Stage weights: work-item -> (weight row nl, K sub-chunk). Each writes
        // UNROLL contiguous K halves: sa[k + nl*NK] = weight(ra+nl, loop_k+k).
        for (int work = tiitg; work < A_WORK; work += NUM_THREADS) {
            int nl   = work / (NK / UNROLL);        // weight row within tile (our N)
            int sub  = work % (NK / UNROLL);        // K sub-chunk index
            int kbase = sub * UNROLL;               // K offset within chunk
            int gn = ra + nl;
            threadgroup half* dst = sa + nl*NK + kbase;
            int gk0 = loop_k + kbase;
            if (gn < N && gk0 + UNROLL <= K) {
                // Fast path: full in-bounds UNROLL span, vectorized half4 loads.
                device const half4* wrow4 =
                    (device const half4*)(B + p.b_offset + (uint)gn * p.b_row_stride + gk0);
                threadgroup half4* dst4 = (threadgroup half4*)dst;
                #pragma unroll
                for (int i = 0; i < UNROLL/4; ++i) dst4[i] = wrow4[i];
            } else if (gn < N) {
                // Edge: K tail — scalar with bounds check.
                device const half* wrow = B + p.b_offset + (uint)gn * p.b_row_stride + gk0;
                #pragma unroll
                for (int i = 0; i < UNROLL; ++i)
                    dst[i] = (gk0 + i < K) ? wrow[i] : (half)0;
            } else {
                #pragma unroll
                for (int i = 0; i < UNROLL; ++i) dst[i] = (half)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);        // weights staged chunk
        auto mB = tB.slice(loop_k, rb);  // activations [K-off, M-off]
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Output dst[ourN, ourM] (llama's [ne0,ne1]). Our C element (m,n) at
    // m*c_row_stride + n → as [ourN, ourM] tensor: strides {1 (n), c_row_stride (m)}.
    // slice (ra = weight/N-off, rb = act/M-off).
    device float* dstC = C + p.c_offset;
    auto tD = tensor(dstC, dextents<int32_t,2>(N, M), array<int,2>({1, p.c_row_stride}));
    cT.store(tD.slice(ra, rb));
}
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
    return v;
}

// ---------------------------------------------------------------------------
// Tiled GEMM using Apple simdgroup 8x8 matrix instructions.
// C[M,N] = A[M,K] (fp32) * B[N,K]^T (fp16).
//   A (activations) element (m,k) at A[a_offset + m*a_row_stride + k]  (K contig)
//   B (weights)     element (n,k) at B[b_offset + n*b_row_stride + k]  (K contig, [N,K])
//   C element (m,n) at C[c_offset + m*c_row_stride + n]                (N contig)
//
// 32(M) x 32(N) output tile / threadgroup, 4 simdgroups, each a 2x2 grid of
// simdgroup_float8x8 accumulators (16x16 sub-tile). K streamed in chunks of 8.
//
// HALF-STAGING (llama.cpp mul_mm technique): A and B tiles are staged in
// threadgroup memory as HALF (activations downcast fp32->fp16 on load), and the
// MMA uses simdgroup_half8x8 with fp32 accumulation. This halves threadgroup
// bandwidth and uses the faster half MMA path. Matches llama.cpp's f16 numerics;
// PPL/coherence validated. C = A · B^T via loading the B tile transposed.
//
// Weight buffer B is bound at its 64-bit byte offset by the dispatcher
// (p.b_offset==0) to avoid uint32 element-offset overflow for the 8.8GB region.
// grid: threadgroups = (ceil(N/32), ceil(M/32)); threads/tg = 128.
// threadgroup(0): staging = (32*8 + 32*8) halves = 512 halves = 1KB; the FP32
// edge-store scratch (256 floats = 1KB) reuses the same region after compute.
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
// Tiled GEMM, 64(M) x 32(N) tile — larger tile for higher prefill throughput
// (llama.cpp mul_mm uses a comparable 64x32). Same math as the 32x32 kernel:
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
    acc = apply_activation(acc, p.activation);
    C[p.c_offset + (uint)m * p.c_row_stride + n] = acc;
}

// ---------------------------------------------------------------------------
// GEMV v2: M==1 decode fast path. C[1,N] = A[1,K] * B[N,K]^T. Mirrors llama.cpp
// `kernel_mul_mv_t_t`:
//   - Each threadgroup owns NR0=2 output rows (r0 = tgpig.x*NR0). The activation
//     slice each lane loads is REUSED across both rows (halves activation reads).
//   - NSG simdgroups split K: SG sgitg processes NF-element chunks strided by
//     NSG*NF*32, accumulating a partial dot per row. A cross-SG reduction
//     (simd_sum + shmem) combines the NSG partials. This parallelizes large K
//     (e.g. down_proj K=9728) across up to NSG*32 lanes instead of 32.
//   - Weight (B, half) read directly from device; activation (A, f32) read
//     directly from device (no staging — the reuse across NR0 rows is the win).
// grid: threadgroups = (ceil(N/NR0), 1, 1); threads/tg = (32, NSG, 1).
// shmem: NR0 * 32 floats for the cross-SG reduction.
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

    // ---- cross-simdgroup reduction (llama helper_mv_reduce_and_write) ----
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
                C[p.c_offset + r0 + row] = apply_activation(tot, p.activation);
        }
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
        C[p.c_offset + n] = apply_activation(dot, p.activation);
    }
}

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

    float x0 = base[i];
    float x1 = base[i + half_dim];
    base[i]            = x0 * c - x1 * s;
    base[i + half_dim] = x1 * c + x0 * s;
}

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
    float a = A[p.a_offset + gid];
    float b = p.broadcast_b ? B[p.b_offset] : B[p.b_offset + gid];
    O[p.out_offset + gid] = a + b;
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
    float a = A[p.a_offset + gid];
    float b = p.broadcast_b ? B[p.b_offset] : B[p.b_offset + gid];
    O[p.out_offset + gid] = a * b;
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

// ---------------------------------------------------------------------------
// ShortConv: depth-wise causal conv1d + silu, one thread per group. Mirrors the
// CPU kernel (execute.cpp OpType::SHORTCONV). Each group is independent (own
// state, own output rows), so one thread walks that group's sequence序 serially.
//   window[i+k] = k<(ksize-1) ? state[i+k-...]  actually: prepend state then x.
//   For position i in [0,seq): sum = Σ_{k=0..ksize} win[i+k]*w[g,k], out=silu(sum),
//   where win = [state(ksize-1) | x(seq)]. state updated to last (ksize-1) x vals.
// grid = ceil(groups) threadgroups; ksize<=4 (KMAX). Persistent state buffer is
// read AND written in-place (like the KV cache append).
// ---------------------------------------------------------------------------
kernel void shortconv_f32(
    device const float*      X     [[buffer(0)]],
    device const float*      W     [[buffer(1)]],
    device float*            STATE [[buffer(2)]],
    device float*            O     [[buffer(4)]],
    constant ShortConvParams& p    [[buffer(3)]],
    uint  gid                      [[thread_position_in_grid]])
{
    const int KMAX = 4;
    int g = int(gid);
    if (g >= p.groups) return;
    int ks = p.kernel_size;          // <= KMAX
    int pre = ks - 1;                // state window length
    int seq = p.seq;
    int nreal = (p.n_real > 0 && p.n_real < seq) ? p.n_real : seq;

    device const float* w = W + p.w_offset + (uint)g * (uint)ks;
    device float* cs = STATE + p.state_offset + (uint)g * (uint)pre;
    device const float* x = X + p.x_offset;      // layout [seq, groups]: x[s*groups+g]
    device float* o = O + p.out_offset + (uint)g * (uint)seq;

    // Load state prefix into a register window [win(pre)]. At position i the
    // conv window is [win(pre) , x[i]] (win holds the previous `pre` values).
    float win[KMAX];                 // last `pre` values seen (pre <= KMAX-1)
    float st0[KMAX];                 // snapshot of the incoming state (for edge case)
    for (int p_ = 0; p_ < pre; p_++) { win[p_] = cs[p_]; st0[p_] = cs[p_]; }

    for (int i = 0; i < seq; i++) {
        float xi = x[(uint)i * (uint)p.groups + (uint)g];
        float sum = 0.0f;
        for (int k = 0; k < pre; k++) sum += win[k] * w[k];
        sum += xi * w[pre];
        o[i] = (i < nreal) ? (sum / (1.0f + exp(-sum))) : 0.0f;   // silu on real positions
        // shift window left, append xi
        for (int k = 0; k < pre - 1; k++) win[k] = win[k + 1];
        if (pre > 0) win[pre - 1] = xi;
    }

    // New state = last `pre` elements of the window [old_state(pre) | x(nreal)].
    //   index j in [nreal, nreal+pre): if j < pre -> old_state[j], else x[j-pre].
    for (int p_ = 0; p_ < pre; p_++) {
        int j = nreal + p_;                 // position in the [state|x] window
        cs[p_] = (j < pre) ? st0[j]
                           : x[(uint)(j - pre) * (uint)p.groups + (uint)g];
    }
}

// ---------------------------------------------------------------------------
// GDN decode (seq=1): fused Gated Delta Rule + RMSNormGated for one token.
// One THREADGROUP per value head; V_dim threads (thread dv owns state column
// [*,dv], kv_mem[dv], attn_out[dv]). Mirrors kernels/gdn_neon.h gdn_recurrence.
// State [num_v_heads, k_dim, v_dim] read+written in place (device buffer).
//   grid: threadgroups = num_v_heads; threads = v_dim (<=256).
// Threadgroup mem: sq[k_dim], sk[k_dim] (staged+L2-normed q/k), plus reduction.
// ---------------------------------------------------------------------------
inline float gdn_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log(1.0f + exp(x));
}

kernel void gdn_decode_f32(
    device const float*   QKV    [[buffer(0)]],
    device const float*   A      [[buffer(1)]],
    device const float*   B      [[buffer(2)]],
    device const float*   Z      [[buffer(5)]],
    device const float*   ALOG   [[buffer(6)]],
    device const float*   DTB    [[buffer(7)]],
    device const float*   NORMW  [[buffer(8)]],
    device float*         STATE  [[buffer(9)]],
    device float*         O      [[buffer(4)]],
    constant GdnParams&   p      [[buffer(3)]],
    threadgroup float*    sh     [[threadgroup(0)]],
    uint  vh                     [[threadgroup_position_in_grid]],
    uint  dv                     [[thread_position_in_threadgroup]],
    uint  nthreads               [[threads_per_threadgroup]])
{
    int K = p.k_dim, V = p.v_dim;
    if ((int)vh >= p.num_v_heads) return;
    int repeat = p.num_v_heads / p.num_heads;
    int kh = (int)vh / repeat;

    int qkv_dim = p.num_heads * K;                 // q block width (= k block width)
    uint q_base = p.qkv_offset + (uint)(kh * K);
    uint k_base = p.qkv_offset + (uint)qkv_dim + (uint)(kh * K);
    uint v_base = p.qkv_offset + (uint)(2 * qkv_dim) + (uint)((int)vh * V);

    // Threadgroup layout: sq[K] | sk[K] | red[nthreads]
    threadgroup float* sq  = sh;
    threadgroup float* sk  = sq + K;
    threadgroup float* red = sk + K;

    // Stage q,k; L2-normalize (cooperative). Each thread handles strided dims.
    for (int d = (int)dv; d < K; d += (int)nthreads) { sq[d] = QKV[q_base + d]; sk[d] = QKV[k_base + d]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // L2 norm of q and k: reduce sum of squares over K.
    float qss = 0.0f, kss = 0.0f;
    for (int d = (int)dv; d < K; d += (int)nthreads) { qss += sq[d]*sq[d]; kss += sk[d]*sk[d]; }
    // reduce qss then kss via shared mem.
    red[dv] = qss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads/2; s>0; s>>=1){ if(dv<s) red[dv]+=red[dv+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}    
    float q_inv = 1.0f / sqrt(red[0] + p.l2_eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[dv] = kss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads/2; s>0; s>>=1){ if(dv<s) red[dv]+=red[dv+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}    
    float k_inv = 1.0f / sqrt(red[0] + p.l2_eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = (int)dv; d < K; d += (int)nthreads) { sq[d] *= q_inv; sk[d] *= k_inv; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gate scalars (per value head).
    float a_h = A[p.a_offset + vh];
    float b_h = B[p.b_offset + vh];
    float sp = gdn_softplus(a_h + DTB[p.dtb_offset + vh]);
    float g_exp = exp((-exp(ALOG[p.Alog_offset + vh])) * sp);
    float beta  = 1.0f / (1.0f + exp(-b_h));       // sigmoid(b)

    // NOTE: host dispatches exactly V threads, so dv < V for all threads (all
    // must reach the RMSNorm barriers below — no early return allowed).
    device float* state_h = STATE + p.state_offset + (uint)((int)vh * K * V);
    float vv = QKV[v_base + dv];

    // Pass 1: decay state rows + kv_mem[dv] = Σ_dk state[dk,dv]*k[dk].
    float kv = 0.0f;
    for (int dk = 0; dk < K; dk++) {
        float r = state_h[dk*V + dv] * g_exp;
        state_h[dk*V + dv] = r;
        kv += r * sk[dk];
    }
    float delta = (vv - kv) * beta;
    // Pass 2: state[dk,dv] += k[dk]*delta ; attn_out[dv] = Σ state[dk,dv]*q[dk].
    float attn = 0.0f;
    for (int dk = 0; dk < K; dk++) {
        float r = state_h[dk*V + dv] + sk[dk]*delta;
        state_h[dk*V + dv] = r;
        attn += r * sq[dk];
    }
    attn *= p.scale;

    // RMSNormGated: rms over v_dim (reduction), then *norm_w * silu(z).
    red[dv] = attn*attn; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads/2; s>0; s>>=1){ if(dv<s) red[dv]+=red[dv+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}    
    float rms = 1.0f / sqrt(red[0] / (float)V + p.rms_eps);
    float normed = attn * rms * NORMW[p.norm_offset + dv];
    float z = Z[p.z_offset + (uint)((int)vh * V) + dv];
    float silu_z = z / (1.0f + exp(-z));
    O[p.out_offset + (uint)((int)vh * V) + dv] = normed * silu_z;
}

// ---------------------------------------------------------------------------
// GDN prefill (seq>1): same recurrence as decode, looped sequentially over the
// token dim (state is recurrent). One THREADGROUP per value head; V_dim threads.
// LAYOUT DIFFERS FROM DECODE:
//   qkv is [qkv_total, seq] (dim-major): qkv[(base+d)*seq + t]
//   a/b/z/out are [seq, dim] (seq-major): a[t*num_v_heads+vh], out[t*zdim+vh*V+dv]
// Correctness-first (serial over t); optimize later. grid: num_v_heads tg, V thr.
// Threadgroup mem: sq[K] + sk[K] + red[V].
// ---------------------------------------------------------------------------
kernel void gdn_prefill_f32(
    device const float*   QKV    [[buffer(0)]],
    device const float*   A      [[buffer(1)]],
    device const float*   B      [[buffer(2)]],
    device const float*   Z      [[buffer(5)]],
    device const float*   ALOG   [[buffer(6)]],
    device const float*   DTB    [[buffer(7)]],
    device const float*   NORMW  [[buffer(8)]],
    device float*         STATE  [[buffer(9)]],
    device float*         O      [[buffer(4)]],
    constant GdnParams&   p      [[buffer(3)]],
    threadgroup float*    sh     [[threadgroup(0)]],
    uint  vh                     [[threadgroup_position_in_grid]],
    uint  dv                     [[thread_position_in_threadgroup]],
    uint  nthreads               [[threads_per_threadgroup]])
{
    int K = p.k_dim, V = p.v_dim, S = p.seq_len;
    if ((int)vh >= p.num_v_heads) return;
    int nreal = (p.n_real > 0 && p.n_real < S) ? p.n_real : S;
    int repeat = p.num_v_heads / p.num_heads;
    int kh = (int)vh / repeat;
    int qkv_dim = p.num_heads * K;
    int zdim = p.num_v_heads * V;

    threadgroup float* sq  = sh;
    threadgroup float* sk  = sq + K;
    threadgroup float* red = sk + K;

    device float* state_h = STATE + p.state_offset + (uint)((int)vh * K * V);
    float neg_exp_A = -exp(ALOG[p.Alog_offset + vh]);

    for (int t = 0; t < S; t++) {
        if (t >= nreal) {   // zero padding output rows
            O[p.out_offset + (uint)(t*zdim) + (uint)((int)vh*V) + dv] = 0.0f;
            continue;
        }
        // Stage q,k for token t: qkv[(base+d)*seq + t]  (dim-major layout).
        uint qb = p.qkv_offset + (uint)(kh * K);
        uint kb = p.qkv_offset + (uint)qkv_dim + (uint)(kh * K);
        for (int d = (int)dv; d < K; d += (int)nthreads) {
            sq[d] = QKV[(qb + (uint)d) * (uint)S + (uint)t];
            sk[d] = QKV[(kb + (uint)d) * (uint)S + (uint)t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qss=0.0f, kss=0.0f;
        for (int d=(int)dv; d<K; d+=(int)nthreads){ qss+=sq[d]*sq[d]; kss+=sk[d]*sk[d]; }
        red[dv]=qss; threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint s=nthreads/2;s>0;s>>=1){ if(dv<s) red[dv]+=red[dv+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
        float qi=1.0f/sqrt(red[0]+p.l2_eps); threadgroup_barrier(mem_flags::mem_threadgroup);
        red[dv]=kss; threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint s=nthreads/2;s>0;s>>=1){ if(dv<s) red[dv]+=red[dv+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
        float ki=1.0f/sqrt(red[0]+p.l2_eps); threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int d=(int)dv; d<K; d+=(int)nthreads){ sq[d]*=qi; sk[d]*=ki; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // gate scalars for (t, vh): a/b are [seq, num_v_heads].
        float a_h = A[p.a_offset + (uint)(t*p.num_v_heads) + vh];
        float b_h = B[p.b_offset + (uint)(t*p.num_v_heads) + vh];
        float sp = gdn_softplus(a_h + DTB[p.dtb_offset + vh]);
        float g_exp = exp(neg_exp_A * sp);
        float beta  = 1.0f/(1.0f+exp(-b_h));
        float vv = QKV[(p.qkv_offset + (uint)(2*qkv_dim) + (uint)((int)vh*V) + dv) * (uint)S + (uint)t];

        // Pass1
        float kv = 0.0f;
        for (int dk=0; dk<K; dk++){ float r=state_h[dk*V+dv]*g_exp; state_h[dk*V+dv]=r; kv+=r*sk[dk]; }
        float delta = (vv - kv) * beta;
        // Pass2
        float attn = 0.0f;
        for (int dk=0; dk<K; dk++){ float r=state_h[dk*V+dv]+sk[dk]*delta; state_h[dk*V+dv]=r; attn+=r*sq[dk]; }
        attn *= p.scale;
        // RMSNormGated
        red[dv]=attn*attn; threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint s=nthreads/2;s>0;s>>=1){ if(dv<s) red[dv]+=red[dv+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
        float rms=1.0f/sqrt(red[0]/(float)V + p.rms_eps);
        float normed = attn*rms*NORMW[p.norm_offset + dv];
        float z = Z[p.z_offset + (uint)(t*zdim) + (uint)((int)vh*V) + dv];
        float silu_z = z/(1.0f+exp(-z));
        O[p.out_offset + (uint)(t*zdim) + (uint)((int)vh*V) + dv] = normed * silu_z;
        threadgroup_barrier(mem_flags::mem_threadgroup);  // state consistent before next t
    }
}

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
// SDPA KV-cache append: write FP32 K_cur/V_cur rows into the FP16 cache at
// position (past + s), per kv-head. grid = (head_dim, cur_seqlen, num_kv_heads).
// ---------------------------------------------------------------------------
kernel void sdpa_append_f32_to_f16(
    device const float*      CUR   [[buffer(0)]],
    device half*             CACHE [[buffer(2)]],
    constant SdpaAppendParams& p   [[buffer(3)]],
    uint3 gid                      [[thread_position_in_grid]])
{
    int d = int(gid.x);
    int s = int(gid.y);
    int g = int(gid.z);
    if (d >= p.head_dim || s >= p.cur_seqlen || g >= p.num_kv_heads) return;

    uint src = p.cur_offset + (uint)g * p.cur_stride_head + (uint)s * p.cur_stride_pos + d;
    // cache row-major per head: [head, position, feature]
    uint dst = p.cache_offset
             + (uint)g * ((uint)p.head_dim * (uint)p.max_seq_len)
             + (uint)(p.past_seqlen + s) * (uint)p.head_dim
             + (uint)d;
    CACHE[dst] = half(CUR[src]);
}

// ---------------------------------------------------------------------------
// SDPA prefill (src_seqlen > 1): one SIMD group (32 lanes) per (query pos, head).
// Lanes cooperate: the QK dot for each key is a simd_sum over head_dim; each
// lane then owns a strided subset of the v_head_dim output accumulator. Single-
// pass online softmax. Fallback when the tensor FA path (sdpa_prefill_fa2_f32)
// is unavailable, or forced via MOLLM_METAL_SDPA_SIMPLE=1.
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
// (Removed: old KV-split flash-attention `sdpa_prefill_fa_f32`. Superseded by
// the llama-style query-split `sdpa_prefill_fa2_f32` below, which is faster and
// simpler. History in git.)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// SDPA prefill v2 — llama.cpp `kernel_flash_attn_ext` structure.
//   grid: ((S+Q-1)/Q, num_heads), threads (32, NSG). One threadgroup handles a
//   tile of Q=8 query rows for one head, with NSG=4 simdgroups (128 threads).
//
// Key structural choices (vs the old sdpa_prefill_fa_f32):
//   - K/V read DIRECTLY from device global memory into the simdgroup MMA (the
//     FP16 cache rows are contiguous with stride = head_dim / v_head_dim), so
//     NO threadgroup staging of K/V and NO per-block staging barrier.
//   - QK / PV matmuls are cooperative over all Q queries; the KV *columns* are
//     split across simdgroups for the QK GEMM (pk += sgitg*8*DK), each SG storing
//     its column stripe into the shared score buffer ss.
//   - The online-softmax STATE (M/S) is split by QUERY ROW: simdgroup sgitg owns
//     rows {sgitg, sgitg+NSG, ...} (NQ = Q/NSG rows). No cross-simdgroup merge.
//   - O accumulator lives in threadgroup memory `so` (half), rescaled ELEMENTWISE
//     (so4[...] *= ms), not via a diagonal-matrix MMA.
//
// Threadgroup memory: sq[Q*DK] half(2B) + so[Q*PV] float(4B) + ss[Q*SH] float(4B),
//   SH=2*C, PV=PAD(DV,64). For DK=DV=128: 8*128*2 + 8*128*4 + 8*128*4 =
//   2048 + 4096 + 4096 = 10240 bytes = 10 KB. Fits the 32 KB limit.
//   so is float (matches llama o_t=float) so simdgroup_load/store of float8x8
//   accumulators work directly. dk/dv runtime (Phase A); Q/C/NSG compile-time.
// ---------------------------------------------------------------------------
// Function constants (Phase B): specialize dk/dv at pipeline-compile time so the
// QK/PV MMA loop bounds (DK8, DV8, NO) are compile-time constants → fully
// unrolled, no runtime-variable loop overhead. When the host does not set them
// (index not defined), DK/DV fall back to the runtime SdpaParams values (Phase A
// behavior). mollm's function-constant namespace was previously empty.
constant int FC_SDPA_DK [[function_constant(0)]];
constant int FC_SDPA_DV [[function_constant(1)]];
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
    const short QT  = 8;              // queries per threadgroup tile (llama nqptg)
    const short C   = 64;             // KV columns per block         (llama ncpsg)
    const short NSG = 4;              // simdgroups per threadgroup
    const short NW  = 32;             // simd width
    const short NQ  = QT / NSG;       // query rows owned by each simdgroup (=2)
    const short SH  = 2 * C;          // score buffer stride per query (float)

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
    float S[NQ], M[NQ];
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
            const short NC = (C / 8) / NSG;   // = 2
            for (short cc = 0; cc < NC; ++cc) {
                simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                for (short i = 0; i < DK8; ++i) {
                    simdgroup_half8x8 mq, mk;
                    simdgroup_load(mq, sq + 8 * i, DK, 0, false);
                    simdgroup_load(mk, pk + 8 * i, DK, 0, true);   // transpose -> K^T
                    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                }
                simdgroup_store(mqk, ps, SH, 0, false);
                pk += 8u * (uint)NSG * (uint)DK;
                ps += 8 * NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- online softmax over this block's C columns (query-row split) ----
        // Each SG owns rows j = jj*NSG + sgitg. Lane tiisg owns score cols {2t, 2t+1}.
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const short s = q0 + j;
            const float m_old = M[jj];

            float2 s2 = ss2[(j * SH) / 2 + tiisg] * p.scale;

            // causal + optional additive mask on the two cols this lane owns.
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
            ss2[(j * SH) / 2 + tiisg] = vs2;   // P matrix (float; read as half for PV MMA)

            // rescale threadgroup-O for this row (all DV lanes cooperate within SG).
            for (short i = tiisg; i < DV4; i += NW) so4[j * PV4 + i] *= ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O += P * V : direct-global V read ----
        // so holds [QT x PV] O accumulators (half in threadgroup mem). Split the
        // PV8 output 8-col blocks across simdgroups (stripe stride 8*NSG). P (=ss,
        // float) loads as simdgroup_float8x8; V loads as simdgroup_half8x8 direct
        // from global. Metal MMA accepts float(P) x half(V) -> float(O). Mirrors
        // llama:6316-6384.
        {
            const short NO = PV8 / NSG;   // output 8x8 col blocks per SG
            simdgroup_float8x8 lo[8];     // NO <= 8 (PV<=256, NSG=4)
            {
                threadgroup float* sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
            // pv columns for this SG's output stripe: V[key, dcol]; V row stride = DV.
            device const half* pv = Vbase + (uint)ic * (uint)DV + (uint)(8 * sgitg);
            for (short cc = 0; cc < C / 8; ++cc) {
                simdgroup_float8x8 vs;                       // P block [QT x 8] (=ss)
                simdgroup_load(vs, ss + 8 * cc, SH, 0, false);
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, pv + (uint)(8 * NSG * ii), DV, 0, false); // V[cc-block, dcol]
                    simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                }
                pv += (uint)8 * (uint)DV;   // advance 8 keys
            }
            {
                threadgroup float* sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, PV, 0, false);
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
