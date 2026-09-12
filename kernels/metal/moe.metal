#include <metal_stdlib>
#include "metal_common.h"

using namespace metal;

#ifdef MOLLM_METAL_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

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

