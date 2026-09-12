#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

// RWKV previous-token shift. Each thread owns one hidden channel and walks the
// sequence serially, which preserves the recurrent state update order.
kernel void rwkv_token_shift_f32(
    device const float* X       [[buffer(0)]],
    device void* STATE          [[buffer(1)]],
    device float* O             [[buffer(2)]],
    constant RwkvTokenShiftParams& p [[buffer(3)]],
    uint d                      [[thread_position_in_grid]])
{
    if (int(d) >= p.hidden) return;
    float previous = p.state_fp16
        ? float(((device half*)STATE)[p.state_offset + d])
        : ((device float*)STATE)[p.state_offset + d];
    for (int t = 0; t < p.real; ++t) {
        uint i = (uint)t * (uint)p.hidden + d;
        float current = X[p.x_offset + i];
        O[p.out_offset + i] = previous - current;
        // Persistent shift state is FP16 in RWKV packages. Match repeated
        // decode exactly by applying the FP16 round-trip after every token,
        // not only when the whole prefill sequence finishes.
        previous = p.state_fp16 ? float(half(current)) : current;
    }
    for (int t = p.real; t < p.seq; ++t)
        O[p.out_offset + (uint)t * (uint)p.hidden + d] = 0.0f;
    if (p.state_fp16)
        ((device half*)STATE)[p.state_offset + d] = half(previous);
    else
        ((device float*)STATE)[p.state_offset + d] = previous;
}

kernel void rwkv_mix_f32(
    device const float* X       [[buffer(0)]],
    device const float* SHIFT   [[buffer(1)]],
    device float* O             [[buffer(2)]],
    constant RwkvMixParams& p   [[buffer(3)]],
    device const float* MIX     [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]])
{
    if (int(gid) >= p.total) return;
    uint d = gid % (uint)p.hidden;
    O[p.out_offset + gid] =
        X[p.x_offset + gid] +
        SHIFT[p.shift_offset + gid] * MIX[p.mix_offset + d];
}

// One threadgroup per (token, head), normalizing the head vector by its L2
// magnitude. This mirrors the CPU definition 1 / (sqrt(sum_sq) + eps).
kernel void rwkv_l2_norm_f32(
    device const float* X          [[buffer(0)]],
    device float* O                [[buffer(2)]],
    constant RwkvL2NormParams& p   [[buffer(3)]],
    uint group                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]])
{
    if (int(group) >= p.groups) return;
    uint base = group * (uint)p.head_size;
    float partial = 0.0f;
    for (int i = int(tid); i < p.head_size; i += 256) {
        float v = X[p.x_offset + base + (uint)i];
        partial += v * v;
    }
    partial = simd_sum(partial);
    threadgroup float sh[8];
    uint lane = tid & 31u, sg = tid >> 5;
    if (lane == 0) sh[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = simd_sum(lane < 8 ? sh[lane] : 0.0f);
    float scale = 1.0f / (precise::sqrt(sum) + p.eps);
    for (int i = int(tid); i < p.head_size; i += 256)
        O[p.out_offset + base + (uint)i] =
            X[p.x_offset + base + (uint)i] * scale;
}

// Fused per-head GroupNorm + RWKV bonus + gate. One threadgroup owns one
// (token, head) vector.
kernel void rwkv_post_f32(
    device const float* RAW       [[buffer(0)]],
    device const float* R         [[buffer(1)]],
    device float* O               [[buffer(2)]],
    constant RwkvPostParams& p    [[buffer(3)]],
    device const float* K         [[buffer(4)]],
    device const float* V         [[buffer(5)]],
    device const float* RK        [[buffer(6)]],
    device const float* W         [[buffer(7)]],
    device const float* BIAS      [[buffer(8)]],
    device const float* GATE      [[buffer(9)]],
    uint group                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]])
{
    if (int(group) >= p.groups) return;
    uint h = group % (uint)p.heads;
    uint base = group * (uint)p.head_size;
    uint wb = h * (uint)p.head_size;
    uint lane = tid & 31u, sg = tid >> 5;
    threadgroup float sum_sh[8];
    threadgroup float bonus_sh[8];

    float sum = 0.0f, bonus = 0.0f;
    for (int j = int(tid); j < p.head_size; j += 256) {
        uint i = base + (uint)j;
        sum += RAW[p.raw_offset + i];
        bonus += R[p.r_offset + i] * K[p.k_offset + i] *
                 RK[p.rk_offset + wb + (uint)j];
    }
    sum = simd_sum(sum);
    bonus = simd_sum(bonus);
    if (lane == 0) {
        sum_sh[sg] = sum;
        bonus_sh[sg] = bonus;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = simd_sum(lane < 8 ? sum_sh[lane] : 0.0f) /
                 float(p.head_size);
    float bonus_total = simd_sum(lane < 8 ? bonus_sh[lane] : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float variance = 0.0f;
    for (int j = int(tid); j < p.head_size; j += 256) {
        float z = RAW[p.raw_offset + base + (uint)j] - mean;
        variance += z * z;
    }
    variance = simd_sum(variance);
    if (lane == 0) sum_sh[sg] = variance;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    variance = simd_sum(lane < 8 ? sum_sh[lane] : 0.0f) /
               float(p.head_size);
    float inv = rsqrt(variance + p.eps);

    for (int j = int(tid); j < p.head_size; j += 256) {
        uint i = base + (uint)j;
        uint wi = wb + (uint)j;
        float normalized =
            (RAW[p.raw_offset + i] - mean) * inv * W[p.weight_offset + wi] +
            BIAS[p.bias_offset + wi];
        O[p.out_offset + i] =
            (normalized + bonus_total * V[p.v_offset + i]) *
            GATE[p.gate_offset + i];
    }
}

// RWKV-7 recurrence. A 32-thread SIMD group owns a row tile within one head;
// every lane owns one state-matrix row. Rows are independent, so splitting a
// 64-row head across two threadgroups improves occupancy without introducing
// synchronization. Tokens remain serial within each row.
kernel void rwkv7_f32(
    device const float* R          [[buffer(0)]],
    device const float* DECAY      [[buffer(1)]],
    device float* O                [[buffer(2)]],
    constant Rwkv7Params& p        [[buffer(3)]],
    device const float* K          [[buffer(4)]],
    device const float* V          [[buffer(5)]],
    device const float* A          [[buffer(6)]],
    device const float* B          [[buffer(7)]],
    device void* STATE             [[buffer(8)]],
    uint group                     [[threadgroup_position_in_grid]],
    uint lane                      [[thread_position_in_threadgroup]])
{
    const uint rows_per_group = 32;
    const uint groups_per_head =
        ((uint)p.head_size + rows_per_group - 1) / rows_per_group;
    uint head = group / groups_per_head;
    uint row = (group % groups_per_head) * rows_per_group + lane;
    if (int(head) >= p.heads || int(row) >= p.head_size) return;
    uint hidden = (uint)p.heads * (uint)p.head_size;
    uint state_head =
        p.state_offset + head * (uint)p.head_size * (uint)p.head_size;
    uint state_row = state_head + row * (uint)p.head_size;

    if (p.state_fp16 && (p.head_size % 4) == 0) {
        device half4* state4 =
            (device half4*)STATE + state_row / 4;
        for (int t = 0; t < p.real; ++t) {
            uint base =
                (uint)t * hidden + head * (uint)p.head_size;
            float state_a = 0.0f;
            for (int j = 0; j < p.head_size; j += 4) {
                const float4 sv = float4(state4[j / 4]);
                const float4 av =
                    *((device const float4*)(
                        A + p.a_offset + base + (uint)j));
                state_a += sv.x * av.x;
                state_a += sv.y * av.y;
                state_a += sv.z * av.z;
                state_a += sv.w * av.w;
            }

            float result = 0.0f;
            const float value = V[p.v_offset + base + row];
            for (int j = 0; j < p.head_size; j += 4) {
                float4 sv = float4(state4[j / 4]);
                const float4 decay =
                    *((device const float4*)(
                        DECAY + p.decay_offset + base + (uint)j));
                const float4 key =
                    *((device const float4*)(
                        K + p.k_offset + base + (uint)j));
                const float4 bv =
                    *((device const float4*)(
                        B + p.b_offset + base + (uint)j));
                const float4 rv =
                    *((device const float4*)(
                        R + p.r_offset + base + (uint)j));
                sv = sv * decay + value * key + state_a * bv;
                state4[j / 4] = half4(sv);
                result += sv.x * rv.x;
                result += sv.y * rv.y;
                result += sv.z * rv.z;
                result += sv.w * rv.w;
            }
            O[p.out_offset + base + row] = result;
        }
        for (int t = p.real; t < p.seq; ++t) {
            uint base =
                (uint)t * hidden + head * (uint)p.head_size;
            O[p.out_offset + base + row] = 0.0f;
        }
        return;
    }

    for (int t = 0; t < p.real; ++t) {
        uint base = (uint)t * hidden + head * (uint)p.head_size;
        float state_a = 0.0f;
        for (int j = 0; j < p.head_size; ++j) {
            float state_value = p.state_fp16
                ? float(((device half*)STATE)[state_row + (uint)j])
                : ((device float*)STATE)[state_row + (uint)j];
            state_a += state_value * A[p.a_offset + base + (uint)j];
        }

        float result = 0.0f;
        float value = V[p.v_offset + base + row];
        for (int j = 0; j < p.head_size; ++j) {
            uint sj = state_row + (uint)j;
            float state_value = p.state_fp16
                ? float(((device half*)STATE)[sj])
                : ((device float*)STATE)[sj];
            state_value =
                state_value * DECAY[p.decay_offset + base + (uint)j] +
                value * K[p.k_offset + base + (uint)j] +
                state_a * B[p.b_offset + base + (uint)j];
            if (p.state_fp16)
                ((device half*)STATE)[sj] = half(state_value);
            else
                ((device float*)STATE)[sj] = state_value;
            result += state_value * R[p.r_offset + base + (uint)j];
        }
        O[p.out_offset + base + row] = result;
    }
    for (int t = p.real; t < p.seq; ++t) {
        uint base = (uint)t * hidden + head * (uint)p.head_size;
        O[p.out_offset + base + row] = 0.0f;
    }
}

// Head-size-64 RWKV7 prefill. One workgroup owns an entire head and keeps its
// FP32 state matrix in threadgroup memory across the sequence. The five
// per-token input vectors are loaded once per head instead of once per state
// row. Decode retains rwkv7_f32 to avoid the two per-token barriers here.
kernel void rwkv7_h64_tgstate_fp32(
    device const float* R          [[buffer(0)]],
    device const float* DECAY      [[buffer(1)]],
    device float* O                [[buffer(2)]],
    constant Rwkv7Params& p        [[buffer(3)]],
    device const float* K          [[buffer(4)]],
    device const float* V          [[buffer(5)]],
    device const float* A          [[buffer(6)]],
    device const float* B          [[buffer(7)]],
    device float* STATE            [[buffer(8)]],
    uint head                      [[threadgroup_position_in_grid]],
    uint row                       [[thread_position_in_threadgroup]])
{
    constexpr uint HS = 64;
    threadgroup float state[HS * HS];
    threadgroup float sr[HS];
    threadgroup float sw[HS];
    threadgroup float sk[HS];
    threadgroup float sa[HS];
    threadgroup float sb[HS];

    if (int(head) >= p.heads) return;
    const uint hidden = (uint)p.heads * HS;
    const uint state_head = p.state_offset + head * HS * HS;
    threadgroup float* state_row = state + row * HS;
    device const float* device_state_row = STATE + state_head + row * HS;
    #pragma unroll
    for (uint j = 0; j < HS; j += 4)
        *((threadgroup float4*)(state_row + j)) =
            *((device const float4*)(device_state_row + j));

    for (int t = 0; t < p.real; ++t) {
        const uint base = (uint)t * hidden + head * HS;
        sr[row] = R[p.r_offset + base + row];
        sw[row] = DECAY[p.decay_offset + base + row];
        sk[row] = K[p.k_offset + base + row];
        sa[row] = A[p.a_offset + base + row];
        sb[row] = B[p.b_offset + base + row];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float state_a = 0.0f;
        #pragma unroll
        for (uint j = 0; j < HS; j += 4) {
            const float4 sv = *((threadgroup float4*)(state_row + j));
            state_a += sv.x * sa[j];
            state_a += sv.y * sa[j + 1];
            state_a += sv.z * sa[j + 2];
            state_a += sv.w * sa[j + 3];
        }

        const float value = V[p.v_offset + base + row];
        float result = 0.0f;
        #pragma unroll
        for (uint j = 0; j < HS; j += 4) {
            float4 sv = *((threadgroup float4*)(state_row + j));
            const float4 decay(sw[j], sw[j + 1], sw[j + 2], sw[j + 3]);
            const float4 key(sk[j], sk[j + 1], sk[j + 2], sk[j + 3]);
            const float4 bv(sb[j], sb[j + 1], sb[j + 2], sb[j + 3]);
            const float4 rv(sr[j], sr[j + 1], sr[j + 2], sr[j + 3]);
            sv = sv * decay + value * key + state_a * bv;
            *((threadgroup float4*)(state_row + j)) = sv;
            result += sv.x * rv.x;
            result += sv.y * rv.y;
            result += sv.z * rv.z;
            result += sv.w * rv.w;
        }
        O[p.out_offset + base + row] = result;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint j = 0; j < HS; j += 4)
        *((device float4*)(STATE + state_head + row * HS + j)) =
            *((threadgroup float4*)(state_row + j));
    for (int t = p.real; t < p.seq; ++t) {
        const uint base = (uint)t * hidden + head * HS;
        O[p.out_offset + base + row] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// ShortConv: depth-wise causal conv1d + silu, one thread per group (groups are
// independent, so each thread walks its own sequence). For position i:
// out[i] = silu(Σ_k win[i+k]*w[g,k]) over win = [state(ksize-1) | x(seq)]; the
// persistent state is then updated in place to the last (ksize-1) x values.
// ksize<=4 (KMAX). Matches the CPU SHORTCONV kernel.
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
    device const float* x = X + p.x_offset;
    device float* o = O + p.out_offset + (uint)g * (uint)seq;

    // Load state prefix into a register window [win(pre)]. At position i the
    // conv window is [win(pre) , x[i]] (win holds the previous `pre` values).
    float win[KMAX];                 // last `pre` values seen (pre <= KMAX-1)
    float st0[KMAX];                 // snapshot of the incoming state (for edge case)
    for (int p_ = 0; p_ < pre; p_++) { win[p_] = cs[p_]; st0[p_] = cs[p_]; }

    for (int i = 0; i < seq; i++) {
        float xi = x[(uint)i * p.x_row_stride + (uint)g];
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
                           : x[(uint)(j - pre) * p.x_row_stride + (uint)g];
    }
}

// ---------------------------------------------------------------------------
// GDN decode (seq=1): fused Gated Delta Rule + RMSNormGated for one token.
// One threadgroup per value head; v_dim threads (thread dv owns state column
// [*,dv], kv_mem[dv], attn_out[dv]). State [num_v_heads, k_dim, v_dim] is read
// and written in place. Threadgroup mem holds the staged, L2-normed q/k plus a
// reduction scratch. Matches the CPU GDN kernel.
// ---------------------------------------------------------------------------
inline float gdn_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log(1.0f + exp(x));
}

// Decode-only ShortConv + GDN fusion. A threadgroup owns one key head and all
// value heads that share it (one for 0.8B, two for 4B). This ownership is
// important: q/k convolution state is shared by those value heads and must be
// advanced exactly once.
kernel void gdn_conv_decode_f32(
    device const float*   QKV    [[buffer(0)]],
    device const float*   A      [[buffer(1)]],
    device const float*   B      [[buffer(2)]],
    device const float*   Z      [[buffer(5)]],
    device const float*   ALOG   [[buffer(6)]],
    device const float*   DTB    [[buffer(7)]],
    device const float*   NORMW  [[buffer(8)]],
    device float*         STATE  [[buffer(9)]],
    device const float*   CONVW  [[buffer(10)]],
    device float*         CONVS  [[buffer(11)]],
    device float*         O      [[buffer(4)]],
    constant GdnParams&   p      [[buffer(3)]],
    threadgroup float*    sh     [[threadgroup(0)]],
    uint  kh                     [[threadgroup_position_in_grid]],
    uint  tid                    [[thread_position_in_threadgroup]])
{
    const int K = p.k_dim, V = p.v_dim;
    const int repeat = p.num_v_heads / p.num_heads;
    const int qkv_dim = p.num_heads * K;
    const int qkv_total = 2*qkv_dim + p.num_v_heads*V;
    const int pre = p.conv_kernel - 1;
    const uint lane = tid & 31u;
    const uint qk_nsg = ((uint)K + 31u) >> 5;
    const uint v_nsg = ((uint)V + 31u) >> 5;

    // sq/sk are shared by every value head attached to this key head.
    threadgroup float* sq = sh;
    threadgroup float* sk = sq + K;
    threadgroup float* qred = sk + K;
    threadgroup float* kred = qred + qk_nsg;
    threadgroup float* ared = kred + qk_nsg;

    // Q and K channels have one owner each. Decode ShortConv is the dot of
    // three saved samples plus the current sample, followed by SiLU.
    if ((int)tid < K) {
        const int d = (int)tid;
        const int qch = (int)kh*K + d;
        const int kch = qkv_dim + (int)kh*K + d;
        float qsum = 0.0f, ksum = 0.0f;
        for (int i = 0; i < pre; ++i) {
            qsum += CONVS[p.conv_state_offset + (uint)(qch*pre+i)] *
                    CONVW[p.conv_weight_offset +
                          (uint)(qch*p.conv_kernel+i)];
            ksum += CONVS[p.conv_state_offset + (uint)(kch*pre+i)] *
                    CONVW[p.conv_weight_offset +
                          (uint)(kch*p.conv_kernel+i)];
        }
        const float qx = QKV[p.qkv_offset + (uint)qch];
        const float kx = QKV[p.qkv_offset + (uint)kch];
        qsum += qx * CONVW[p.conv_weight_offset +
                           (uint)(qch*p.conv_kernel+pre)];
        ksum += kx * CONVW[p.conv_weight_offset +
                           (uint)(kch*p.conv_kernel+pre)];
        sq[d] = qsum / (1.0f + exp(-qsum));
        sk[d] = ksum / (1.0f + exp(-ksum));
        for (int i = 0; i < pre-1; ++i) {
            CONVS[p.conv_state_offset + (uint)(qch*pre+i)] =
                CONVS[p.conv_state_offset + (uint)(qch*pre+i+1)];
            CONVS[p.conv_state_offset + (uint)(kch*pre+i)] =
                CONVS[p.conv_state_offset + (uint)(kch*pre+i+1)];
        }
        if (pre > 0) {
            CONVS[p.conv_state_offset + (uint)(qch*pre+pre-1)] = qx;
            CONVS[p.conv_state_offset + (uint)(kch*pre+pre-1)] = kx;
        }
    }

    // Each remaining thread owns one value channel and its convolution state.
    const int rv = (int)tid / V;
    const int dv = (int)tid - rv*V;
    const int vh = (int)kh*repeat + rv;
    const int vch = 2*qkv_dim + vh*V + dv;
    float vsum = 0.0f;
    for (int i = 0; i < pre; ++i)
        vsum += CONVS[p.conv_state_offset + (uint)(vch*pre+i)] *
                CONVW[p.conv_weight_offset +
                      (uint)(vch*p.conv_kernel+i)];
    const float vx = QKV[p.qkv_offset + (uint)vch];
    vsum += vx * CONVW[p.conv_weight_offset +
                       (uint)(vch*p.conv_kernel+pre)];
    const float vv = vsum / (1.0f + exp(-vsum));
    for (int i = 0; i < pre-1; ++i)
        CONVS[p.conv_state_offset + (uint)(vch*pre+i)] =
            CONVS[p.conv_state_offset + (uint)(vch*pre+i+1)];
    if (pre > 0)
        CONVS[p.conv_state_offset + (uint)(vch*pre+pre-1)] = vx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize q/k using only the first K threads. The reduction order is the
    // same as the standalone decode kernel for K=128.
    float qss = 0.0f, kss = 0.0f;
    if ((int)tid < K) {
        qss = sq[tid]*sq[tid];
        kss = sk[tid]*sk[tid];
    }
    qss = simd_sum(qss);
    kss = simd_sum(kss);
    if ((int)tid < K && lane == 0) {
        const uint sg = tid >> 5;
        qred[sg] = qss;
        kred[sg] = kss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float qtotal = simd_sum(lane < qk_nsg ? qred[lane] : 0.0f);
    const float ktotal = simd_sum(lane < qk_nsg ? kred[lane] : 0.0f);
    float qinv = 0.0f, kinv = 0.0f;
    if (lane == 0) {
        qinv = 1.0f / sqrt(qtotal + p.l2_eps);
        kinv = 1.0f / sqrt(ktotal + p.l2_eps);
    }
    qinv = simd_shuffle(qinv, 0);
    kinv = simd_shuffle(kinv, 0);
    if ((int)tid < K) {
        sq[tid] *= qinv;
        sk[tid] *= kinv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gexp = 0.0f, beta = 0.0f;
    if (lane == 0) {
        const float ah = A[p.a_offset + (uint)vh];
        const float bh = B[p.b_offset + (uint)vh];
        const float sp =
            gdn_softplus(ah + DTB[p.dtb_offset + (uint)vh]);
        gexp = exp((-exp(ALOG[p.Alog_offset + (uint)vh])) * sp);
        beta = 1.0f / (1.0f + exp(-bh));
    }
    gexp = simd_shuffle(gexp, 0);
    beta = simd_shuffle(beta, 0);
    device float* state_h =
        STATE + p.state_offset + (uint)(vh*K*V);

    float kv = 0.0f, attn_decay = 0.0f, qk = 0.0f;
    for (int dk = 0; dk < K; ++dk) {
        const float r = state_h[dk*V + dv] * gexp;
        kv += r * sk[dk];
        attn_decay += r * sq[dk];
        qk += sk[dk] * sq[dk];
    }
    const float delta = (vv - kv) * beta;
    for (int dk = 0; dk < K; ++dk)
        state_h[dk*V + dv] =
            state_h[dk*V + dv] * gexp + sk[dk] * delta;
    const float attn = (attn_decay + delta*qk) * p.scale;

    const float part = simd_sum(attn*attn);
    const uint vsg = (uint)dv >> 5;
    if (lane == 0)
        ared[(uint)rv*v_nsg + vsg] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = simd_sum(
        lane < v_nsg ? ared[(uint)rv*v_nsg + lane] : 0.0f);
    float rms = 0.0f;
    if (lane == 0)
        rms = 1.0f / sqrt(total/(float)V + p.rms_eps);
    rms = simd_shuffle(rms, 0);
    const float normed = attn*rms*NORMW[p.norm_offset + (uint)dv];
    const float z = Z[p.z_offset + (uint)(vh*V+dv)];
    O[p.out_offset + (uint)(vh*V+dv)] =
        normed * (z/(1.0f + exp(-z)));
    (void)qkv_total;
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
    uint lane = dv & 31u;
    uint sg = dv >> 5;
    uint nsg = (nthreads + 31u) >> 5;

    // Stage q,k; L2-normalize (cooperative). Each thread handles strided dims.
    for (int d = (int)dv; d < K; d += (int)nthreads) { sq[d] = QKV[q_base + d]; sk[d] = QKV[k_base + d]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // L2 norm of q and k: reduce sum of squares over K.
    float qss = 0.0f, kss = 0.0f;
    for (int d = (int)dv; d < K; d += (int)nthreads) {
        qss += sq[d]*sq[d];
        kss += sk[d]*sk[d];
    }
    // Two-level SIMD reduction avoids the log2(V) threadgroup-barrier tree.
    qss = simd_sum(qss);
    kss = simd_sum(kss);
    if (lane == 0) {
        red[sg] = qss;
        red[nsg + sg] = kss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = simd_sum(lane < nsg ? red[lane] : 0.0f);
    float k_total = simd_sum(lane < nsg ? red[nsg + lane] : 0.0f);
    float q_inv = 1.0f / sqrt(q_total + p.l2_eps);
    float k_inv = 1.0f / sqrt(k_total + p.l2_eps);
    for (int d = (int)dv; d < K; d += (int)nthreads) {
        sq[d] *= q_inv;
        sk[d] *= k_inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gate scalars (per value head).
    float a_h = A[p.a_offset + vh];
    float b_h = B[p.b_offset + vh];
    float sp = gdn_softplus(a_h + DTB[p.dtb_offset + vh]);
    float g_exp = exp((-exp(ALOG[p.Alog_offset + vh])) * sp);
    float beta = 1.0f / (1.0f + exp(-b_h));

    // NOTE: host dispatches exactly V threads, so dv < V for all threads (all
    // must reach the RMSNorm barriers below — no early return allowed).
    device float* state_h = STATE + p.state_offset + (uint)((int)vh * K * V);
    float vv = QKV[v_base + dv];

    // Pass 1 reads the old state without writing it. Besides kv, accumulate
    // the decayed-state contribution to q and q·k. This lets pass 2 perform
    // the only state write and avoids storing then immediately reloading the
    // decayed matrix.
    float kv = 0.0f, attn_decay = 0.0f, qk = 0.0f;
    for (int dk = 0; dk < K; dk++) {
        float r = state_h[dk*V + dv] * g_exp;
        kv += r * sk[dk];
        attn_decay += r * sq[dk];
        qk += sk[dk] * sq[dk];
    }
    float delta = (vv - kv) * beta;
    // Pass 2 writes final state. attn(state_new,q) is reconstructed from the
    // two dot products above, so this traversal does no extra state readback
    // for the output reduction.
    for (int dk = 0; dk < K; dk++) {
        float r = state_h[dk*V + dv] * g_exp + sk[dk] * delta;
        state_h[dk*V + dv] = r;
    }
    float attn = (attn_decay + delta * qk) * p.scale;

    // RMSNormGated: rms over v_dim, then model-selected output gate.
    float attn_sq = simd_sum(attn * attn);
    if (lane == 0) red[sg] = attn_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float attn_total = simd_sum(lane < nsg ? red[lane] : 0.0f);
    float rms = 1.0f / sqrt(attn_total / (float)V + p.rms_eps);
    float normed = attn * rms * NORMW[p.norm_offset + dv];
    float z = Z[p.z_offset + (uint)((int)vh * V) + dv];
    float sigmoid_z = 1.0f / (1.0f + exp(-z));
    float gate = p.output_gate_type == 1 ? sigmoid_z : z * sigmoid_z;
    O[p.out_offset + (uint)((int)vh * V) + dv] = normed * gate;
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
    uint lane = dv & 31u;
    uint sg = dv >> 5;
    uint nsg = (nthreads + 31u) >> 5;

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
        float qss=0.0f,kss=0.0f;
        for(int d=(int)dv;d<K;d+=(int)nthreads){
            qss+=sq[d]*sq[d];
            kss+=sk[d]*sk[d];
        }
        qss=simd_sum(qss);
        kss=simd_sum(kss);
        if(lane==0) {
            red[sg]=qss;
            red[nsg+sg]=kss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qt=simd_sum(lane<nsg?red[lane]:0.0f);
        float kt=simd_sum(lane<nsg?red[nsg+lane]:0.0f);
        float qi=1.0f/sqrt(qt+p.l2_eps);
        float ki=1.0f/sqrt(kt+p.l2_eps);
        for(int d=(int)dv; d<K; d+=(int)nthreads) {
            sq[d]*=qi;
            sk[d]*=ki;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // gate scalars for (t, vh): a/b are [seq, num_v_heads].
        float a_h =
            A[p.a_offset + (uint)t*(uint)p.a_row_stride + vh];
        float b_h =
            B[p.b_offset + (uint)t*(uint)p.b_row_stride + vh];
        float sp = gdn_softplus(a_h + DTB[p.dtb_offset + vh]);
        float g_exp = exp(neg_exp_A * sp);
        float beta = 1.0f/(1.0f+exp(-b_h));
        float vv = QKV[(p.qkv_offset + (uint)(2*qkv_dim) + (uint)((int)vh*V) + dv) * (uint)S + (uint)t];

        // Read old state once for the three reductions; write only the final
        // state in pass 2.
        float kv=0.0f,attn_decay=0.0f,qk=0.0f;
        for(int dk=0;dk<K;dk++){
            float r=state_h[dk*V+dv]*g_exp;
            kv+=r*sk[dk];
            attn_decay+=r*sq[dk];
            qk+=sk[dk]*sq[dk];
        }
        float delta = (vv - kv) * beta;
        for(int dk=0;dk<K;dk++){
            float r=state_h[dk*V+dv]*g_exp+sk[dk]*delta;
            state_h[dk*V+dv]=r;
        }
        float attn=(attn_decay+delta*qk)*p.scale;
        // RMSNormGated
        float ats=simd_sum(attn*attn);
        if(lane==0) red[sg]=ats;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float at=simd_sum(lane<nsg?red[lane]:0.0f);
        float rms=1.0f/sqrt(at/(float)V + p.rms_eps);
        float normed = attn*rms*NORMW[p.norm_offset + dv];
        float z =
            Z[p.z_offset + (uint)t*(uint)p.z_row_stride +
              (uint)((int)vh*V) + dv];
        float sigmoid_z = 1.0f/(1.0f+exp(-z));
        float gate = p.output_gate_type == 1 ? sigmoid_z : z * sigmoid_z;
        O[p.out_offset + (uint)(t*zdim) + (uint)((int)vh*V) + dv] = normed * gate;
        threadgroup_barrier(mem_flags::mem_threadgroup);  // state consistent before next t
    }
}

// K-parallel GDN prefill. Threads cooperate on each state column,
// while q·k is reduced once per head/token instead of redundantly in every
// value-column thread. Token recurrence remains serial and exact.
kernel void gdn_prefill_kparallel_f32(
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
    uint  tid                    [[thread_position_in_threadgroup]],
    uint  nthreads               [[threads_per_threadgroup]])
{
    const int K = p.k_dim, V = p.v_dim, S = p.seq_len;
    const int SPLITS = 4;
    if ((int)vh >= p.num_v_heads) return;
    const int nreal =
        (p.n_real > 0 && p.n_real < S) ? p.n_real : S;
    const int repeat = p.num_v_heads / p.num_heads;
    const int kh = (int)vh / repeat;
    const int qkv_dim = p.num_heads * K;
    const int zdim = p.num_v_heads * V;
    const int split = (int)tid / V;
    const int dv = (int)tid - split * V;
    const uint lane = tid & 31u;
    const uint sg = tid >> 5;
    const uint nsg = (nthreads + 31u) >> 5;

    threadgroup float* sq = sh;
    threadgroup float* sk = sq + K;
    threadgroup float* red_q = sk + K;       // nsg
    threadgroup float* red_k = red_q + nsg;  // nsg
    threadgroup float* red_qk = red_k + nsg; // nsg
    threadgroup float* partial_kv = red_qk + nsg;
    threadgroup float* partial_attn = partial_kv + SPLITS*V;
    threadgroup float* delta = partial_attn + SPLITS*V;
    threadgroup float* attn_s = delta + V;             // V
    threadgroup float* gate = attn_s + V;              // 2

    device float* state_h =
        STATE + p.state_offset + (uint)((int)vh * K * V);
    const float neg_exp_A = -exp(ALOG[p.Alog_offset + vh]);

    for (int t = 0; t < S; ++t) {
        if (t >= nreal) {
            if (split == 0 && dv < V)
                O[p.out_offset + (uint)(t*zdim) +
                  (uint)((int)vh*V + dv)] = 0.0f;
            continue;
        }

        const uint qb = p.qkv_offset + (uint)(kh*K);
        const uint kb = p.qkv_offset + (uint)qkv_dim + (uint)(kh*K);
        for (int d = (int)tid; d < K; d += (int)nthreads) {
            sq[d] = QKV[(qb + (uint)d)*(uint)S + (uint)t];
            sk[d] = QKV[(kb + (uint)d)*(uint)S + (uint)t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float qss = 0.0f, kss = 0.0f, qks = 0.0f;
        for (int d = (int)tid; d < K; d += (int)nthreads) {
            const float q = sq[d], k = sk[d];
            qss += q*q;
            kss += k*k;
            qks += q*k;
        }
        qss = simd_sum(qss);
        kss = simd_sum(kss);
        qks = simd_sum(qks);
        if (lane == 0) {
            red_q[sg] = qss;
            red_k[sg] = kss;
            red_qk[sg] = qks;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float qt = simd_sum(lane < nsg ? red_q[lane] : 0.0f);
        const float kt = simd_sum(lane < nsg ? red_k[lane] : 0.0f);
        const float qkt = simd_sum(lane < nsg ? red_qk[lane] : 0.0f);
        const float qi = 1.0f / sqrt(qt + p.l2_eps);
        const float ki = 1.0f / sqrt(kt + p.l2_eps);
        const float qk = qkt * qi * ki;
        for (int d = (int)tid; d < K; d += (int)nthreads) {
            sq[d] *= qi;
            sk[d] *= ki;
        }
        if (tid == 0) {
            const float a_h =
                A[p.a_offset + (uint)t*(uint)p.a_row_stride + vh];
            const float b_h =
                B[p.b_offset + (uint)t*(uint)p.b_row_stride + vh];
            gate[0] =
                exp(neg_exp_A *
                    gdn_softplus(a_h + DTB[p.dtb_offset + vh]));
            gate[1] = 1.0f / (1.0f + exp(-b_h));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float kv = 0.0f, attn_decay = 0.0f;
        if (dv < V) {
            for (int dk = split; dk < K; dk += SPLITS) {
                const float r =
                    state_h[dk*V + dv] * gate[0];
                kv += r * sk[dk];
                attn_decay += r * sq[dk];
            }
            partial_kv[split*V + dv] = kv;
            partial_attn[split*V + dv] = attn_decay;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (split == 0 && dv < V) {
            const float vv =
                QKV[(p.qkv_offset + (uint)(2*qkv_dim) +
                     (uint)((int)vh*V + dv)) * (uint)S + (uint)t];
            float kv_total = 0.0f, attn_total = 0.0f;
            #pragma unroll
            for (int s = 0; s < SPLITS; ++s) {
                kv_total += partial_kv[s*V + dv];
                attn_total += partial_attn[s*V + dv];
            }
            const float d = (vv - kv_total) * gate[1];
            delta[dv] = d;
            attn_s[dv] = (attn_total + d*qk) * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (dv < V) {
            const float d = delta[dv];
            for (int dk = split; dk < K; dk += SPLITS)
                state_h[dk*V + dv] =
                    state_h[dk*V + dv] * gate[0] + sk[dk] * d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float ats =
            (split == 0 && dv < V) ? attn_s[dv] * attn_s[dv] : 0.0f;
        ats = simd_sum(ats);
        if (lane == 0) red_q[sg] = ats;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float at =
            simd_sum(lane < nsg ? red_q[lane] : 0.0f);
        if (split == 0 && dv < V) {
            const float rms =
                1.0f / sqrt(at / (float)V + p.rms_eps);
            const float z =
                Z[p.z_offset + (uint)t*(uint)p.z_row_stride +
                  (uint)((int)vh*V + dv)];
            const float sigmoid_z = 1.0f / (1.0f + exp(-z));
            const float gate = p.output_gate_type == 1
                ? sigmoid_z : z * sigmoid_z;
            O[p.out_offset + (uint)(t*zdim) +
              (uint)((int)vh*V + dv)] =
                attn_s[dv] * rms * NORMW[p.norm_offset + dv] * gate;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Prepare normalized Q/K for the row-parallel recurrent kernel. One
// threadgroup owns one (token, key-head).
kernel void gdn_prepare_qk_f32(
    device const float* QKV [[buffer(0)]],
    device float* QN [[buffer(10)]],
    device float* KN [[buffer(11)]],
    constant GdnParams& p [[buffer(3)]],
    threadgroup float* red [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]])
{
    const int K = p.k_dim, S = p.seq_len;
    const int kh = (int)group / S;
    const int t = (int)group - kh*S;
    if (kh >= p.num_heads) return;
    const uint qb = p.qkv_offset + (uint)(kh*K);
    const uint kb =
        p.qkv_offset + (uint)(p.num_heads*K + kh*K);
    float qss = 0.0f, kss = 0.0f;
    for (int d = (int)tid; d < K; d += (int)nthreads) {
        const float q = QKV[(qb + (uint)d)*(uint)S + (uint)t];
        const float k = QKV[(kb + (uint)d)*(uint)S + (uint)t];
        QN[((kh*S + t)*K) + d] = q;
        KN[((kh*S + t)*K) + d] = k;
        qss += q*q;
        kss += k*k;
    }
    const uint lane = tid & 31u;
    const uint sg = tid >> 5;
    const uint nsg = (nthreads + 31u) >> 5;
    qss = simd_sum(qss);
    kss = simd_sum(kss);
    if (lane == 0) {
        red[sg] = qss;
        red[nsg + sg] = kss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float qt = simd_sum(lane < nsg ? red[lane] : 0.0f);
    const float kt = simd_sum(lane < nsg ? red[nsg + lane] : 0.0f);
    const float qi = 1.0f / sqrt(qt + p.l2_eps);
    const float ki = 1.0f / sqrt(kt + p.l2_eps);
    for (int d = (int)tid; d < K; d += (int)nthreads) {
        QN[((kh*S + t)*K) + d] *= qi;
        KN[((kh*S + t)*K) + d] *= ki;
    }
}

// Precompute the two scalar gates once per (token, value-head).
kernel void gdn_prepare_gates_f32(
    device const float* A [[buffer(1)]],
    device const float* B [[buffer(2)]],
    device const float* ALOG [[buffer(6)]],
    device const float* DTB [[buffer(7)]],
    device float* GEXP [[buffer(12)]],
    device float* BETA [[buffer(13)]],
    constant GdnParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const int total = p.seq_len * p.num_v_heads;
    if ((int)gid >= total) return;
    const int vh = (int)gid % p.num_v_heads;
    const int t = (int)gid / p.num_v_heads;
    const float a =
        A[p.a_offset + (uint)t*(uint)p.a_row_stride + (uint)vh];
    const float b =
        B[p.b_offset + (uint)t*(uint)p.b_row_stride + (uint)vh];
    GEXP[gid] =
        exp(-exp(ALOG[p.Alog_offset + vh]) *
            gdn_softplus(a + DTB[p.dtb_offset + vh]));
    BETA[gid] = 1.0f / (1.0f + exp(-b));
}

// One SIMD group per (value-head, value-row). Each lane keeps K/32 state
// values in registers for the whole sequence, so state is loaded and stored
// once per prefill instead of once per token.
kernel void gdn_recurrence_rows_f32(
    device const float* QKV [[buffer(0)]],
    device float* STATE [[buffer(9)]],
    device const float* QN [[buffer(10)]],
    device const float* KN [[buffer(11)]],
    device const float* GEXP [[buffer(12)]],
    device const float* BETA [[buffer(13)]],
    device float* RAW [[buffer(14)]],
    constant GdnParams& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    const int K = 128, V = p.v_dim, S = p.seq_len;
    const int dv = (int)group.x*4 + (int)sg;
    const int vh = (int)group.y;
    if (dv >= V || vh >= p.num_v_heads) return;
    const int repeat = p.num_v_heads / p.num_heads;
    const int kh = vh / repeat;
    const int qkv_dim = p.num_heads*K;
    const int nreal =
        (p.n_real > 0 && p.n_real < S) ? p.n_real : S;
    device float* state =
        STATE + p.state_offset + (ulong)vh*K*V;
    float ls[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int dk = (int)lane + j*32;
        ls[j] = state[dk*V + dv];
    }

    for (int t = 0; t < nreal; ++t) {
        const int gate_idx = t*p.num_v_heads + vh;
        const float decay = GEXP[gate_idx];
        float sk = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int dk = (int)lane + j*32;
            ls[j] *= decay;
            sk += ls[j] * KN[(kh*S + t)*K + dk];
        }
        sk = simd_sum(sk);
        const float vv =
            QKV[(p.qkv_offset + (uint)(2*qkv_dim + vh*V + dv)) *
                (uint)S + (uint)t];
        const float delta = (vv - sk) * BETA[gate_idx];
        float y = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int dk = (int)lane + j*32;
            ls[j] += KN[(kh*S + t)*K + dk] * delta;
            y += ls[j] * QN[(kh*S + t)*K + dk];
        }
        y = simd_sum(y);
        if (lane == 0)
            RAW[(t*p.num_v_heads + vh)*V + dv] = y * p.scale;
    }
    if (lane == 0) {
        for (int t = nreal; t < S; ++t)
            RAW[(t*p.num_v_heads + vh)*V + dv] = 0.0f;
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int dk = (int)lane + j*32;
        state[dk*V + dv] = ls[j];
    }
}

// RMSNormGated over the recurrent output, one threadgroup per
// (token, value-head).
kernel void gdn_post_f32(
    device const float* Z [[buffer(5)]],
    device const float* NORMW [[buffer(8)]],
    device const float* RAW [[buffer(14)]],
    device float* O [[buffer(4)]],
    constant GdnParams& p [[buffer(3)]],
    threadgroup float* red [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]])
{
    const int V = p.v_dim;
    const int vh = (int)group % p.num_v_heads;
    const int t = (int)group / p.num_v_heads;
    if (t >= p.seq_len) return;
    const int base = (t*p.num_v_heads + vh)*V;
    float ss = 0.0f;
    if ((int)tid < V) {
        const float v = RAW[base + tid];
        ss = v*v;
    }
    const uint lane = tid & 31u;
    const uint sg = tid >> 5;
    const uint nsg = (nthreads + 31u) >> 5;
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        simd_sum(lane < nsg ? red[lane] : 0.0f);
    if ((int)tid < V) {
        const float rms =
            1.0f / sqrt(total/(float)V + p.rms_eps);
        const float z =
            Z[p.z_offset + (uint)t*(uint)p.z_row_stride +
              (uint)(vh*V) + tid];
        const float sigmoid_z = 1.0f/(1.0f + exp(-z));
        const float gate = p.output_gate_type == 1
            ? sigmoid_z : z * sigmoid_z;
        O[p.out_offset + (uint)base + tid] =
            RAW[base + tid] * rms *
            NORMW[p.norm_offset + tid] * gate;
    }
}
