#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

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
