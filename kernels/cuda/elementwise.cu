#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/reduction.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

__device__ float round_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__global__ void round_to_bf16_cuda(float* values, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    values[index] = round_bf16(values[index]);
}

template<bool PublishHalf = false, bool Deferred = false>
__global__ void group_rms_norm_cuda(
    const float* input, const float* weight, float* output,
    int group_size, int groups, float epsilon, __half* half = nullptr, float* inverses = nullptr) {
    const size_t base = static_cast<size_t>(blockIdx.x) * group_size;
    const int weight_base = (blockIdx.x % groups) * group_size;
    float sum = 0.0f;
    for (int dim = threadIdx.x; dim < group_size; dim += blockDim.x) {
        const float value = input[base + dim];
        sum += value * value;
    }
    __shared__ float reduction[256];
    const float total = mollm_cuda::detail::block_reduce_sum<256>(sum, reduction);
    const float inverse = rsqrtf(total / group_size + epsilon);
    if constexpr (Deferred) if (threadIdx.x == 0) inverses[blockIdx.x] = inverse;
    for (int dim = threadIdx.x; dim < group_size; dim += blockDim.x) {
        const float value = input[base + dim] * inverse * weight[weight_base + dim];
        if constexpr (!Deferred) output[base + dim] = value;
        if constexpr (PublishHalf) half[base + dim] = __float2half(value);
    }
}

template<bool PublishHalf = false>
__global__ void group_rms_norm_cached_cuda(
    const float* input, const float* weight, float* output,
    int groups, float epsilon, __half* half = nullptr) {
    constexpr int width = 2560;
    const size_t base = static_cast<size_t>(blockIdx.x) * width;
    const int weight_base = (blockIdx.x % groups) * width;
    float values[10];
    float sum = 0.0f;
#pragma unroll
    for (int step = 0; step < 10; ++step) {
        values[step] = input[base + threadIdx.x + step * 256];
        sum += values[step] * values[step];
    }
    __shared__ float reduction[256];
    const float total = mollm_cuda::detail::block_reduce_sum<256>(sum, reduction);
    const float inverse = rsqrtf(total / width + epsilon);
#pragma unroll
    for (int step = 0; step < 10; ++step) {
        const int dim = threadIdx.x + step * 256;
        const float value = values[step] * inverse * weight[weight_base + dim];
        output[base + dim] = value;
        if constexpr (PublishHalf) half[base + dim] = __float2half(value);
    }
}

template<bool SigmoidLogits = false>
__global__ void gr_reduce_cuda(
    const float* input, const float* gates, float* output,
    int hidden, int streams, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t base = (index / hidden) * hidden * streams + index % hidden;
    float sum = 0.0f;
    for (int stream = 0; stream < streams; ++stream) {
        const size_t source = base + static_cast<size_t>(stream) * hidden;
        const float gate = SigmoidLogits
            ? 1.0f / (1.0f + expf(-gates[source])) : gates[source];
        sum += gate * input[source];
    }
    output[index] = round_bf16(sum * (1.0f / streams));
}

__global__ void gr_reduce_sigmoid_vector4_cuda(
    const float4* input, const float4* logits, float4* output, int vectors) {
    const size_t dim = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (dim >= size_t(vectors)) return;
    const size_t row = blockIdx.y;
    const size_t base = row * size_t(vectors) * 4 + dim;
    float4 sum{0.f,0.f,0.f,0.f};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const size_t source = base + size_t(s) * vectors;
        const float4 x = input[source], g = logits[source];
        sum.x += (1.f / (1.f + expf(-g.x))) * x.x;
        sum.y += (1.f / (1.f + expf(-g.y))) * x.y;
        sum.z += (1.f / (1.f + expf(-g.z))) * x.z;
        sum.w += (1.f / (1.f + expf(-g.w))) * x.w;
    }
    output[row * vectors + dim] = float4{round_bf16(sum.x * .25f),
        round_bf16(sum.y * .25f),round_bf16(sum.z * .25f),round_bf16(sum.w * .25f)};
}

template<int StaticStreams = 0>
__global__ void gr_reduce_sigmoid_tiled_cuda(
    const float* input, const float* logits, float* output,
    int hidden, int streams) {
    const size_t dim = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (dim >= size_t(hidden)) return;
    const size_t row = blockIdx.y;
    const size_t base = row * size_t(hidden) * streams + dim;
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < (StaticStreams ? StaticStreams : streams); ++s) {
        const size_t source = base + size_t(s) * hidden;
        const float gate = 1.0f / (1.0f + expf(-logits[source]));
        sum += gate * input[source];
    }
    output[row * hidden + dim] = round_bf16(sum * (1.0f / streams));
}

__global__ void gr_inject_cuda(
    const float* branch, const float* residual, const float* gates,
    float* output, int hidden, int streams, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t row = index / (static_cast<size_t>(hidden) * streams);
    const int stream = (index / hidden) % streams;
    output[index] = round_bf16(residual[index] +
        gates[row * streams + stream] * branch[row * hidden + index % hidden]);
}

__global__ void gr_inject_tiled_cuda(
    const float* branch, const float* residual, const float* gates,
    float* output, int hidden, int streams) {
    const size_t dim = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (dim >= static_cast<size_t>(hidden)) return;
    const size_t row = blockIdx.z;
    const size_t stream = blockIdx.y;
    const size_t index = (row * streams + stream) * hidden + dim;
    output[index] = round_bf16(residual[index] +
        gates[row * streams + stream] * branch[row * hidden + dim]);
}

template<bool DeferredNormalized=false,bool CompactCarry=false,bool CompactResidual=false>
__global__ void gr_inject_group_rms_half_cuda(const float* branch,
    const float* residual, const float* gates, const float* weight,
    float* carry, float* normalized, __half* half, float epsilon,
    float* inverses=nullptr) {
    constexpr int width = 2560, streams = 4;
    const size_t base = size_t(blockIdx.x) * width;
    const size_t row = blockIdx.x / streams;
    const int weight_base = (blockIdx.x % streams) * width;
    const float gate = gates[blockIdx.x];
    float values[10], sum = 0.f;
#pragma unroll
    for (int step = 0; step < 10; ++step) {
        const int dim = threadIdx.x + step * 256;
        float residual_value;
        if constexpr(CompactResidual)
            residual_value=__uint_as_float(uint32_t(reinterpret_cast<const uint16_t*>(residual)[base+dim])<<16);
        else residual_value=residual[base+dim];
        const float value = round_bf16(residual_value + gate * branch[row * width + dim]);
        values[step] = value;
        if constexpr(CompactCarry)
            reinterpret_cast<uint16_t*>(carry)[base+dim]=uint16_t(__float_as_uint(value)>>16);
        else carry[base + dim] = value;
        sum += value * value;
    }
    __shared__ float reduction[256];
    const float total = mollm_cuda::detail::block_reduce_sum<256>(sum, reduction);
    const float inverse = rsqrtf(total / width + epsilon);
    if constexpr (DeferredNormalized) {
        if(threadIdx.x==0)inverses[blockIdx.x]=inverse;
    }
#pragma unroll
    for (int step = 0; step < 10; ++step) {
        const int dim = threadIdx.x + step * 256;
        const float value = values[step] * inverse * weight[weight_base + dim];
        if constexpr (!DeferredNormalized) normalized[base + dim] = value;
        half[base + dim] = __float2half(value);
    }
}

template<bool PublishBranch>
__global__ void moe_combine_gr_inject_rms_deferred_cuda(const uint16_t* routed,
    const float* shared,const float* scale,const float* residual,const float* gates,
    const float* gamma,float* branch_output,float* carry_output,__half* half,
    float* inverses,float epsilon) {
    constexpr int width=2560;
    const int stream=threadIdx.x/256,t=threadIdx.x%256,row=blockIdx.x;
    __shared__ float branch[width],reduction[1024];
    if(stream==0){
        const float multiplier=1.f/(1.f+expf(-round_bf16(scale[row])));
#pragma unroll
        for(int step=0;step<10;++step){const int dim=t+step*256;float sum=0.f;
#pragma unroll
            for(int route=0;route<10;++route)
                sum+=__uint_as_float(unsigned(routed[(size_t(row)*10+route)*width+dim])<<16);
            const float value=__fadd_rn(sum,__fmul_rn(round_bf16(shared[size_t(row)*width+dim]),multiplier));
            branch[dim]=value;
            if constexpr(PublishBranch)branch_output[size_t(row)*width+dim]=value;
        }
    }
    __syncthreads();
    const size_t base=(size_t(row)*4+stream)*width;
    const auto* compact_residual=reinterpret_cast<const uint16_t*>(residual);
    auto* compact_carry=reinterpret_cast<uint16_t*>(carry_output);
    const float gate=gates[row*4+stream];float values[10],sum=0.f;
#pragma unroll
    for(int step=0;step<10;++step){const int dim=t+step*256;
        const float r=__uint_as_float(unsigned(compact_residual[base+dim])<<16);
        const float v=round_bf16(r+gate*branch[dim]);values[step]=v;
        compact_carry[base+dim]=uint16_t(__float_as_uint(v)>>16);sum+=v*v;
    }
    reduction[threadIdx.x]=sum;__syncthreads();
#pragma unroll
    for(int offset=128;offset>=64;offset/=2){
        if(t<offset)reduction[stream*256+t]+=reduction[stream*256+t+offset];
        __syncthreads();
    }
    if(t<32){float v=reduction[stream*256+t]+reduction[stream*256+t+32];
        v=mollm_cuda::detail::warp_reduce_sum(v);if(t==0)reduction[stream*256]=v;
    }
    __syncthreads();
    const float inverse=rsqrtf(reduction[stream*256]/width+epsilon);
    if(t==0)inverses[row*4+stream]=inverse;
#pragma unroll
    for(int step=0;step<10;++step){const int dim=t+step*256;
        half[base+dim]=__float2half(values[step]*inverse*gamma[stream*width+dim]);
    }
}

template<int InputPitch=320,int Columns=16,bool CompactCarry=false>
__global__ void hc_up_reduce_fused_cuda(const __half* input,const __half* weight,
    const float* carry,const float* gamma,const float* inv,float* output,int rows) {
    using namespace nvcuda;
    constexpr int hidden=2560,k=320;
    const int first_token=blockIdx.y*16,first_dim=blockIdx.x*Columns;
    const int s=threadIdx.x/32;
    __shared__ __align__(32) __half x[16*InputPitch];
    __shared__ __align__(32) float logits[4][16*Columns];
    for(int i=threadIdx.x;i<16*k;i+=128){const int token=first_token+i/k;
        if constexpr(InputPitch==320)
            x[i]=token<rows?input[size_t(token)*k+i%k]:__float2half(0.f);
        else x[(i/k)*InputPitch+i%k]=token<rows?input[size_t(token)*k+i%k]:__float2half(0.f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,float> c;
    wmma::fragment<wmma::accumulator,16,16,16,float> c_second;
    wmma::fill_fragment(c,0.f);
    if constexpr(Columns==32)wmma::fill_fragment(c_second,0.f);
    for(int j=0;j<k;j+=16){
        wmma::load_matrix_sync(a,x+j,InputPitch);
        wmma::load_matrix_sync(b,weight+size_t(s*hidden+first_dim)*k+j,k);
        wmma::mma_sync(c,a,b,c);
        if constexpr(Columns==32){
            wmma::load_matrix_sync(b,weight+size_t(s*hidden+first_dim+16)*k+j,k);
            wmma::mma_sync(c_second,a,b,c_second);
        }
    }
    wmma::store_matrix_sync(logits[s],c,Columns,wmma::mem_row_major);
    if constexpr(Columns==32)wmma::store_matrix_sync(logits[s]+16,c_second,Columns,wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<16*Columns;i+=128){
        const int row=first_token+i/Columns,dim=first_dim+i%Columns;
        if(row>=rows)continue;
        float sum=0.f;
#pragma unroll
        for(int stream=0;stream<4;++stream){
            const size_t source=(size_t(row)*4+stream)*hidden+dim;
            float value;
            if constexpr(CompactCarry)value=__uint_as_float(uint32_t(reinterpret_cast<const uint16_t*>(carry)[source])<<16);
            else value=carry[source];
            const float normalized=(value*inv[row*4+stream])*gamma[stream*hidden+dim];
            const float gate=1.f/(1.f+expf(-logits[stream][i]));
            sum+=gate*normalized;
        }
        output[size_t(row)*hidden+dim]=round_bf16(sum*.25f);
    }
}

// Diagnostic consumer for a deferred RMS value, not a backend dispatch path.
// Reconstruct each FP32 normalized value in the original multiply order.
template<bool CompactCarry=false>
__global__ void gr_reduce_sigmoid_deferred_rms_cuda(const float* carry,
    const float* logits,const float* weight,const float* inverses,
    float* output,size_t count) {
    constexpr int hidden=2560,streams=4;
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const size_t row=index/hidden,dim=index%hidden;
    float sum=0.f;
#pragma unroll
    for(int s=0;s<streams;++s) {
        const size_t source=(row*streams+s)*hidden+dim;
        float value;
        if constexpr(CompactCarry)value=__uint_as_float(uint32_t(reinterpret_cast<const uint16_t*>(carry)[source])<<16);
        else value=carry[source];
        const float normalized=(value*inverses[row*streams+s])*weight[s*hidden+dim];
        const float gate=1.f/(1.f+expf(-logits[source]));
        sum+=gate*normalized;
    }
    output[index]=round_bf16(sum*.25f);
}

template<bool CompactCarry=false>
__global__ void gr_rms_deferred_materialize_cuda(const float* carry,
    const float* weight,const float* inverses,float* output,size_t count) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count){
        float value;
        if constexpr(CompactCarry)value=__uint_as_float(uint32_t(reinterpret_cast<const uint16_t*>(carry)[index])<<16);
        else value=carry[index];
        output[index]=(value*inverses[index/2560])*weight[index%10240];
    }
}

__global__ void hc_compact_carry_materialize_cuda(const uint16_t* scratch,
    float* output,size_t count) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    // Integer stores preserve every bit pattern, including signaling NaNs.
    if(index<count)reinterpret_cast<uint32_t*>(output)[index]=uint32_t(scratch[index])<<16;
}

__device__ float cuda_activation(float value, int kind) {
    switch (kind) {
    case 1: return value / (1.0f + expf(-value));
    case 2: {
        const float inner = 0.7978845608f *
            (value + 0.044715f * value * value * value);
        return 0.5f * value * (1.0f + tanhf(inner));
    }
    case 3: return fmaxf(value, 0.0f);
    case 4: {
        const float positive = fmaxf(value, 0.0f);
        return positive * positive;
    }
    default: return value;
    }
}

__global__ void apply_activation_cuda(float* values, int rows, int columns,
                                      int kind, int begin, int end) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(rows) * columns;
    if (index >= count)
        return;
    const int column = static_cast<int>(index % columns);
    if (column >= begin && column < end)
        values[index] = cuda_activation(values[index], kind);
}

__global__ void binary_cuda(const float* lhs, const float* rhs, float* output,
                            size_t count, bool multiply,
                            bool lhs_scalar, bool rhs_scalar) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) {
        const float a = lhs[lhs_scalar ? 0 : index];
        const float b = rhs[rhs_scalar ? 0 : index];
        output[index] = multiply ? a * b : a + b;
    }
}

__global__ void sigmoid_mul_cuda(const float* value, const float* gate,
                                 float* output, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        output[index] = value[index] / (1.0f + expf(-gate[index]));
}

__global__ void sigmoid_mul_strided_cuda(
    const float* value, const float* gate, float* output, size_t count,
    int64_t d0, int64_t d1, int64_t d2,
    size_t value_s0, size_t value_s1, size_t value_s2, size_t value_s3,
    size_t gate_s0, size_t gate_s1, size_t gate_s2, size_t gate_s3) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t remaining = index;
    const size_t i0 = remaining % static_cast<size_t>(d0);
    remaining /= static_cast<size_t>(d0);
    const size_t i1 = remaining % static_cast<size_t>(d1);
    remaining /= static_cast<size_t>(d1);
    const size_t i2 = remaining % static_cast<size_t>(d2);
    const size_t i3 = remaining / static_cast<size_t>(d2);
    const float v = value[i0 * value_s0 + i1 * value_s1 + i2 * value_s2 +
                          i3 * value_s3];
    const float g = gate[i0 * gate_s0 + i1 * gate_s1 + i2 * gate_s2 +
                         i3 * gate_s3];
    output[index] = v / (1.0f + expf(-g));
}

template <bool Strided = false>
__global__ void unary_cuda(const float* input, float* output, size_t count,
                           int operation, int64_t d0 = 1, int64_t d1 = 1,
                           int64_t d2 = 1, size_t s0 = 1, size_t s1 = 0,
                           size_t s2 = 0, size_t s3 = 0) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t source_index = index;
    if constexpr (Strided) {
        size_t remaining = index;
        const size_t i0 = remaining % d0;
        remaining /= d0;
        const size_t i1 = remaining % d1;
        remaining /= d1;
        const size_t i2 = remaining % d2;
        const size_t i3 = remaining / d2;
        source_index = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;
    }
    const float value = input[source_index];
    switch (operation) {
    case 0: output[index] = value / (1.0f + expf(-value)); break;
    case 1: {
        const float inner = 0.7978845608f *
            (value + 0.044715f * value * value * value);
        output[index] = 0.5f * value * (1.0f + tanhf(inner));
        break;
    }
    case 2: output[index] = tanhf(value); break;
    case 3: output[index] = 1.0f / (1.0f + expf(-value)); break;
    case 4: output[index] = expf(value); break;
    case 5:
        output[index] = value > 20.0f ? value : log1pf(expf(value));
        break;
    }
}

__global__ void swiglu_cuda(const float* input, float* output,
                            size_t output_count, size_t half) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < output_count) {
        const size_t row = index / half;
        const size_t column = index % half;
        const size_t base = row * half * 2;
        const float gate = input[base + column];
        output[index] = gate / (1.0f + expf(-gate)) *
            input[base + half + column];
    }
}

__device__ __forceinline__ bool argmax_better(
    const mollm_cuda::ArgMaxPair& candidate,
    const mollm_cuda::ArgMaxPair& current) {
    return candidate.index >= 0 &&
        (current.index < 0 || candidate.value > current.value ||
         (candidate.value == current.value &&
          candidate.index < current.index));
}

__device__ __forceinline__ mollm_cuda::ArgMaxPair warp_argmax(
    mollm_cuda::ArgMaxPair best) {
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        mollm_cuda::ArgMaxPair candidate;
        candidate.value = __shfl_down_sync(0xffffffffu, best.value, offset);
        candidate.index = __shfl_down_sync(0xffffffffu, best.index, offset);
        if (lane + offset < warpSize && argmax_better(candidate, best))
            best = candidate;
    }
    return best;
}

__device__ __forceinline__ void block_argmax(
    mollm_cuda::ArgMaxPair best, mollm_cuda::ArgMaxPair* warp_best,
    mollm_cuda::ArgMaxPair* output) {
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    best = warp_argmax(best);
    if (lane == 0)
        warp_best[warp] = best;
    __syncthreads();
    if (warp == 0) {
        const int warps = static_cast<int>(blockDim.x) / warpSize;
        best = lane < warps
            ? warp_best[lane]
            : mollm_cuda::ArgMaxPair{-FLT_MAX, -1};
        best = warp_argmax(best);
        if (lane == 0)
            *output = best;
    }
}

__global__ void argmax_stage1_cuda(
    const float* input, int count, mollm_cuda::ArgMaxPair* partial) {
    mollm_cuda::ArgMaxPair best{-FLT_MAX, -1};
    for (int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int>(gridDim.x) * blockDim.x) {
        const mollm_cuda::ArgMaxPair candidate{input[index], index};
        if (argmax_better(candidate, best))
            best = candidate;
    }
    __shared__ mollm_cuda::ArgMaxPair warp_best[32];
    block_argmax(best, warp_best, partial + blockIdx.x);
}

__global__ void argmax_stage2_cuda(
    const mollm_cuda::ArgMaxPair* partial, int count,
    mollm_cuda::ArgMaxPair* result) {
    mollm_cuda::ArgMaxPair best{-FLT_MAX, -1};
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        const mollm_cuda::ArgMaxPair candidate = partial[index];
        if (argmax_better(candidate, best))
            best = candidate;
    }
    __shared__ mollm_cuda::ArgMaxPair warp_best[32];
    block_argmax(best, warp_best, result);
}

__global__ void rms_norm_cuda(const float* input, const float* weight,
                              float* output, int width, int rows,
                              float epsilon) {
    const int row = blockIdx.x;
    if (row >= rows)
        return;
    float sum = 0.0f;
    const float* source = input + static_cast<size_t>(row) * width;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = source[column];
        sum += value * value;
    }
    __shared__ float reduction[256];
    const float block_sum =
        mollm_cuda::detail::block_reduce_sum<256>(sum, reduction);
    const float inverse = rsqrtf(block_sum / width + epsilon);
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        output[static_cast<size_t>(row) * width + column] =
            source[column] * inverse * weight[column];
}

template <int Threads>
__global__ void add_rms_norm_cuda(
    float* residual, const float* update, const float* weight, float* output,
    int width, int rows, size_t residual_row_stride,
    size_t update_row_stride, size_t output_row_stride, float epsilon) {
    const int row = blockIdx.x;
    if (row >= rows)
        return;
    float sum = 0.0f;
    float* residual_row = residual + static_cast<size_t>(row) *
        residual_row_stride;
    const float* update_row = update + static_cast<size_t>(row) *
        update_row_stride;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = residual_row[column] + update_row[column];
        residual_row[column] = value;
        sum += value * value;
    }
    __shared__ float reduction[Threads];
    const float block_sum =
        mollm_cuda::detail::block_reduce_sum<Threads>(sum, reduction);
    const float inverse = rsqrtf(block_sum / width + epsilon);
    float* output_row = output + static_cast<size_t>(row) *
        output_row_stride;
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        output_row[column] = residual_row[column] * inverse * weight[column];
}

__global__ void contiguous_cuda(
    const float* input, float* output, size_t count,
    int64_t d0, int64_t d1, int64_t d2,
    size_t s0, size_t s1, size_t s2, size_t s3) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t remaining = index;
    const size_t i0 = remaining % static_cast<size_t>(d0);
    remaining /= static_cast<size_t>(d0);
    const size_t i1 = remaining % static_cast<size_t>(d1);
    remaining /= static_cast<size_t>(d1);
    const size_t i2 = remaining % static_cast<size_t>(d2);
    const size_t i3 = remaining / static_cast<size_t>(d2);
    output[index] = input[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3];
}

__global__ void tile_cuda(const float* input, float* output, size_t count,
    int64_t o0, int64_t o1, int64_t o2,
    int64_t i0, int64_t i1, int64_t i2, int64_t i3,
    size_t s0, size_t s1, size_t s2, size_t s3) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    size_t remaining = index;
    const size_t x = (remaining % o0) % i0;
    remaining /= o0;
    const size_t y = (remaining % o1) % i1;
    remaining /= o1;
    const size_t z = (remaining % o2) % i2;
    const size_t w = (remaining / o2) % i3;
    output[index] = input[x * s0 + y * s1 + z * s2 + w * s3];
}

__global__ void ple_gate_cuda(const float* query, const float* key,
    const float* value, float* output, int hidden, int streams) {
    const int row = blockIdx.x / streams;
    const size_t base = static_cast<size_t>(blockIdx.x) * hidden;
    float dot = 0;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        dot = fmaf(query[base + i], key[base + i], dot);
    __shared__ float reduction[256];
    dot = mollm_cuda::detail::block_reduce_sum<256>(dot, reduction);
    float score = dot / sqrtf(static_cast<float>(hidden));
    score = (score < 0 ? -1.f : 1.f) * sqrtf(fmaxf(fabsf(score), 1e-6f));
    const float gate = 1.f / (1.f + expf(-score));
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        output[base + i] = gate * value[static_cast<size_t>(row) * hidden + i];
}

template<bool StorePrefixes = false>
__global__ void ple_dilated_conv_cuda(const float* input, const float* weight,
    float* state, float* output, int channels, int tokens, int kernel_size,
    int dilation, int real_tokens, float* prefixes = nullptr) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const int state_len = (kernel_size - 1) * dilation;
    float* history = state + static_cast<size_t>(channel) * state_len;
    const float* kernel = weight + static_cast<size_t>(channel) * kernel_size;
    for (int token = 0; token < tokens; ++token) {
        float sum = 0;
        if (token < real_tokens) {
            for (int tap = 0; tap < kernel_size; ++tap) {
                const int source = token + tap * dilation;
                const float x = source < state_len ? history[source] :
                    input[static_cast<size_t>(source - state_len) * channels + channel];
                // Match the scalar CPU convolution's multiply/add boundaries.
                sum = __fadd_rn(sum, __fmul_rn(x, kernel[tap]));
            }
        }
        output[static_cast<size_t>(token) * channels + channel] =
            token < real_tokens ? sum / (1.f + expf(-sum)) : 0.f;
        if constexpr (StorePrefixes) {
            if(token<real_tokens)for(int j=0;j<state_len;++j) {
                const int source=j+token+1;
                prefixes[(static_cast<size_t>(token)*channels+channel)*state_len+j]=source<state_len
                    ? history[source] : input[static_cast<size_t>(source-state_len)*channels+channel];
            }
        }
    }
    // Left shift in increasing order preserves every yet-unread old value.
    for (int j = 0; j < state_len; ++j) {
        const int source = j + real_tokens;
        history[j] = source < state_len ? history[source] :
            input[static_cast<size_t>(source - state_len) * channels + channel];
    }
}

}  // namespace

namespace mollm_cuda {

void launch_ple_gate(const float* query, const float* key, const float* value,
                     float* output, int hidden, int streams, int rows) {
    ple_gate_cuda<<<rows * streams, 256>>>(query, key, value, output, hidden, streams);
}

void launch_ple_dilated_conv(const float* input, const float* weight,
    float* state, float* output, int channels, int tokens, int kernel_size,
    int dilation, int real_tokens) {
    ple_dilated_conv_cuda<><<<(channels + 255) / 256, 256>>>(input, weight,
        state, output, channels, tokens, kernel_size, dilation, real_tokens);
}

bool launch_ple_dilated_conv_prefixes(const float* input,const float* weight,
    float* state,float* output,float* prefixes,int channels,int tokens,int kernel_size,
    int dilation,int real_tokens) {
    if(!input||!weight||!state||!output||!prefixes||channels<=0||channels>INT32_MAX-255||
        tokens<1||tokens>8||kernel_size<2||dilation<=0||
        int64_t(kernel_size-1)*dilation>INT32_MAX-8||real_tokens<1||real_tokens>tokens)return false;
    ple_dilated_conv_cuda<true><<<(channels+255)/256,256>>>(input,weight,state,output,
        channels,tokens,kernel_size,dilation,real_tokens,prefixes);
    return cudaGetLastError()==cudaSuccess;
}

bool launch_round_to_bf16(float* values, size_t count) {
    if (count == 0)
        return true;
    constexpr unsigned threads = 256;
    round_to_bf16_cuda<<<
        static_cast<unsigned>((count + threads - 1) / threads), threads>>>(
        values, count);
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess)
        return true;
    std::fprintf(stderr, "CudaBackend: round_to_bf16_cuda failed: %s\n",
                 cudaGetErrorString(error));
    return false;
}

void launch_apply_activation(float* values, int rows, int columns, int kind,
                             int begin, int end) {
    constexpr int threads = 256;
    const size_t count = static_cast<size_t>(rows) * columns;
    apply_activation_cuda<<<
        static_cast<unsigned>((count + threads - 1) / threads), threads>>>(
        values, rows, columns, kind, begin, end);
}

void launch_binary(const float* lhs, const float* rhs, float* output,
                   size_t count, bool multiply,
                   bool lhs_scalar, bool rhs_scalar) {
    constexpr int threads = 256;
    binary_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                  threads>>>(lhs, rhs, output, count, multiply,
                             lhs_scalar, rhs_scalar);
}

void launch_sigmoid_mul(const float* value, const float* gate, float* output,
                        size_t count) {
    constexpr int threads = 256;
    sigmoid_mul_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                       threads>>>(value, gate, output, count);
}

void launch_sigmoid_mul_strided(
    const float* value, const float* gate, float* output, size_t count,
    int64_t d0, int64_t d1, int64_t d2, size_t value_s0, size_t value_s1,
    size_t value_s2, size_t value_s3, size_t gate_s0, size_t gate_s1,
    size_t gate_s2, size_t gate_s3) {
    constexpr int threads = 256;
    sigmoid_mul_strided_cuda<<<
        static_cast<unsigned>((count + threads - 1) / threads), threads>>>(
        value, gate, output, count, d0, d1, d2, value_s0, value_s1, value_s2,
        value_s3, gate_s0, gate_s1, gate_s2, gate_s3);
}

void launch_unary(const float* input, float* output, size_t count,
                  int operation) {
    constexpr int threads = 256;
    unary_cuda<false><<<static_cast<unsigned>((count + threads - 1) / threads),
                 threads>>>(input, output, count, operation);
}

void launch_unary_strided(
    const float* input, float* output, size_t count, int operation,
    int64_t d0, int64_t d1, int64_t d2,
    size_t s0, size_t s1, size_t s2, size_t s3) {
    constexpr int threads = 256;
    unary_cuda<true><<<static_cast<unsigned>((count + threads - 1) / threads),
                      threads>>>(input, output, count, operation,
                                 d0, d1, d2, s0, s1, s2, s3);
}

void launch_swiglu(const float* input, float* output, size_t output_count,
                   size_t half) {
    constexpr int threads = 256;
    swiglu_cuda<<<
        static_cast<unsigned>((output_count + threads - 1) / threads),
        threads>>>(input, output, output_count, half);
}

void launch_rms_norm(const float* input, const float* weight, float* output,
                     int width, int rows, float epsilon) {
    constexpr int threads = 256;
    rms_norm_cuda<<<rows, threads>>>(
        input, weight, output, width, rows, epsilon);
}

void launch_group_rms_norm(const float* input, const float* weight,
                           float* output, int group_size, int groups,
                           int rows, float epsilon) {
    // Single-token 4-group RMS uses the static unroll; other shapes fall back.
    if (rows == 1 && groups == 4 &&
        launch_group_rms_norm_cached(input, weight, output, group_size,
                                     groups, rows, epsilon)) return;
    group_rms_norm_cuda<false><<<rows * groups, 256>>>(
        input, weight, output, group_size, groups, epsilon);
}

bool launch_group_rms_norm_half(const float* input, const float* weight,
    float* output, uint16_t* half, int group_size, int groups, int rows,
    float epsilon, bool cache_input) {
    if (!input || !weight || !output || !half || group_size <= 0 ||
        groups <= 0 || rows <= 0 || rows > 2147483647 / groups) return false;
    if (cache_input) {
        if (group_size != 2560) return false;
        group_rms_norm_cached_cuda<true><<<rows * groups,256>>>(input,weight,output,
            groups,epsilon,reinterpret_cast<__half*>(half));
        return true;
    }
    group_rms_norm_cuda<true><<<rows * groups, 256>>>(input, weight, output,
        group_size, groups, epsilon, reinterpret_cast<__half*>(half));
    return true;
}

bool launch_group_rms_norm_deferred(const float* input, const float* weight,
    uint16_t* half, float* inverses, int rows, float epsilon, cudaStream_t stream) {
    if (!input || !weight || !half || !inverses || rows < 16 || rows > mollm::hc::prefill_max_rows ||
        !std::isfinite(epsilon) || epsilon <= 0.f) return false;
    if ((reinterpret_cast<uintptr_t>(input) & 3) || (reinterpret_cast<uintptr_t>(weight) & 3) ||
        (reinterpret_cast<uintptr_t>(half) & 1) || (reinterpret_cast<uintptr_t>(inverses) & 3)) return false;
    struct Span { const void* pointer; size_t bytes; };
    const size_t count = size_t(rows) * 10240;
    const Span inputs[] = {{input, count * 4}, {weight, 10240 * 4}};
    const Span outputs[] = {{half, count * 2}, {inverses, size_t(rows) * 4 * 4}};
    auto overlap = [](Span a, Span b) {
        const uintptr_t x = reinterpret_cast<uintptr_t>(a.pointer), y = reinterpret_cast<uintptr_t>(b.pointer);
        if (a.bytes > UINTPTR_MAX - x || b.bytes > UINTPTR_MAX - y) return true;
        return x < y + b.bytes && y < x + a.bytes;
    };
    for (int i = 0; i < 2; ++i) {
        for (auto input_span : inputs) if (overlap(outputs[i], input_span)) return false;
        for (int j = 0; j < i; ++j) if (overlap(outputs[i], outputs[j])) return false;
    }
    group_rms_norm_cuda<true, true><<<rows * 4, 256, 0, stream>>>(input, weight, nullptr,
        2560, 4, epsilon, reinterpret_cast<__half*>(half), inverses);
    return true;
}

bool launch_group_rms_norm_cached(const float* input, const float* weight,
    float* output, int group_size, int groups, int rows, float epsilon,
    cudaStream_t stream) {
    if (!input || !weight || !output || group_size != 2560 ||
        groups <= 0 || rows <= 0 || rows > 2147483647 / groups) return false;
    group_rms_norm_cached_cuda<false><<<rows * groups, 256, 0, stream>>>(
        input, weight, output, groups, epsilon);
    return true;
}

bool launch_gr_reduce_sigmoid_vector4(const float* input, const float* logits,
                                    float* output, int hidden, int streams,
                                    int rows, cudaStream_t stream) {
    if (!input || !logits || !output || hidden <= 0 || hidden % 4 || streams != 4 ||
        rows <= 0 || rows > 65535 ||
        (reinterpret_cast<uintptr_t>(input) & 15) ||
        (reinterpret_cast<uintptr_t>(logits) & 15) ||
        (reinterpret_cast<uintptr_t>(output) & 15)) return false;
    if (size_t(hidden) > SIZE_MAX / sizeof(float) / 4 / size_t(rows)) return false;
    const dim3 grid(unsigned((size_t(hidden) / 4 + 255) / 256),unsigned(rows));
    gr_reduce_sigmoid_vector4_cuda<<<grid,256,0,stream>>>(
        reinterpret_cast<const float4*>(input),reinterpret_cast<const float4*>(logits),
        reinterpret_cast<float4*>(output),hidden / 4);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_gr_reduce_sigmoid_tiled(const float* input, const float* logits,
                                  float* output, int hidden, int streams,
                                  int rows, cudaStream_t stream, bool specialize_four) {
    if (!input || !logits || !output || hidden <= 0 || streams <= 0 ||
        rows <= 0 || rows > 65535) return false;
    if (size_t(hidden) > SIZE_MAX / sizeof(float) / size_t(streams) / size_t(rows)) return false;
    const dim3 grid(unsigned((size_t(hidden) + 255) / 256), unsigned(rows));
    if (specialize_four && streams == 4)
        gr_reduce_sigmoid_tiled_cuda<4><<<grid,256,0,stream>>>(input,logits,output,hidden,streams);
    else gr_reduce_sigmoid_tiled_cuda<0><<<grid,256,0,stream>>>(input,logits,output,hidden,streams);
    return cudaGetLastError() == cudaSuccess;
}

void launch_gr_reduce(const float* input, const float* gates, float* output,
                       int hidden, int streams, int rows) {
    const size_t count = static_cast<size_t>(hidden) * rows;
    gr_reduce_cuda<false><<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        input, gates, output, hidden, streams, count);
}

void launch_gr_reduce_sigmoid(const float* input, const float* logits,
                              float* output, int hidden, int streams,
                              int rows, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(hidden) * rows;
    gr_reduce_cuda<true><<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        input, logits, output, hidden, streams, count);
}

void launch_gr_inject(const float* branch, const float* residual,
                       const float* gates, float* output,
                       int hidden, int streams, int rows) {
    const size_t count = static_cast<size_t>(hidden) * streams * rows;
    gr_inject_cuda<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        branch, residual, gates, output, hidden, streams, count);
}

bool launch_gr_inject_group_rms_half(const float* branch, const float* residual,
    const float* gates, const float* weight, float* carry, float* normalized,
    uint16_t* half, int hidden, int streams, int rows, float epsilon,
    cudaStream_t stream) {
    if (!branch || !residual || !gates || !weight || !carry || !normalized || !half ||
        hidden != 2560 || streams != 4 || rows <= 0 || rows > 4096) return false;
    const size_t count = size_t(hidden) * streams * rows;
    struct Span { const void* pointer; size_t bytes; };
    const Span inputs[] = {{branch, size_t(hidden)*rows*4}, {residual, count*4},
                           {gates, size_t(streams)*rows*4}, {weight, size_t(hidden)*streams*4}};
    const Span outputs[] = {{carry, count*4}, {normalized, count*4}, {half, count*2}};
    const auto overlap = [](Span a, Span b) {
        const uintptr_t x = reinterpret_cast<uintptr_t>(a.pointer), y = reinterpret_cast<uintptr_t>(b.pointer);
        if (a.bytes > UINTPTR_MAX-x || b.bytes > UINTPTR_MAX-y) return true;
        return x < y+b.bytes && y < x+a.bytes;
    };
    for (int i=0;i<3;++i) {
        for (auto input : inputs) if (overlap(outputs[i], input)) return false;
        for (int j=0;j<i;++j) if (overlap(outputs[i], outputs[j])) return false;
    }
    gr_inject_group_rms_half_cuda<false><<<rows*streams,256,0,stream>>>(branch,residual,
        gates,weight,carry,normalized,reinterpret_cast<__half*>(half),epsilon);
    return true;
}

bool launch_gr_inject_rms_deferred(const float* branch,const float* residual,
    const float* gates,const float* weight,float* carry,uint16_t* half,
    float* inverses,int rows,float epsilon,cudaStream_t stream,bool compact_carry,bool compact_residual) {
    if(!branch||!residual||!gates||!weight||!carry||!half||!inverses||rows<=0||rows>4096)return false;
    if(compact_residual&&!compact_carry)return false;
    if(compact_carry&&(rows<16||rows>mollm::hc::prefill_max_rows||!std::isfinite(epsilon)||epsilon<=0||
        (reinterpret_cast<uintptr_t>(branch)&3)||(reinterpret_cast<uintptr_t>(residual)&3)||
        (reinterpret_cast<uintptr_t>(gates)&3)||(reinterpret_cast<uintptr_t>(weight)&3)||
        (reinterpret_cast<uintptr_t>(carry)&3)||(reinterpret_cast<uintptr_t>(half)&1)||
        (reinterpret_cast<uintptr_t>(inverses)&3)))return false;
    struct Span {const void* pointer;size_t bytes;};
    const size_t count=size_t(rows)*10240;
    const Span inputs[]={{branch,size_t(rows)*2560*4},{residual,count*4},
        {gates,size_t(rows)*4*4},{weight,10240*4}};
    const Span outputs[]={{carry,count*4},{half,count*2},{inverses,size_t(rows)*4*4}};
    const auto overlap=[](Span a,Span b){
        const uintptr_t x=reinterpret_cast<uintptr_t>(a.pointer),y=reinterpret_cast<uintptr_t>(b.pointer);
        if(a.bytes>UINTPTR_MAX-x||b.bytes>UINTPTR_MAX-y)return true;
        return x<y+b.bytes&&y<x+a.bytes;
    };
    for(int i=0;i<3;++i){
        for(auto input:inputs)if(overlap(outputs[i],input))return false;
        for(int j=0;j<i;++j)if(overlap(outputs[i],outputs[j]))return false;
    }
    if(compact_residual)gr_inject_group_rms_half_cuda<true,true,true><<<rows*4,256,0,stream>>>(branch,residual,
        gates,weight,carry,nullptr,reinterpret_cast<__half*>(half),epsilon,inverses);
    else if(compact_carry)gr_inject_group_rms_half_cuda<true,true,false><<<rows*4,256,0,stream>>>(branch,residual,
        gates,weight,carry,nullptr,reinterpret_cast<__half*>(half),epsilon,inverses);
    else gr_inject_group_rms_half_cuda<true><<<rows*4,256,0,stream>>>(branch,residual,
        gates,weight,carry,nullptr,reinterpret_cast<__half*>(half),epsilon,inverses);
    return cudaGetLastError()==cudaSuccess;
}

bool launch_moe_combine_gr_inject_rms_deferred(const uint16_t* routed,
    const float* shared,const float* scale,const float* residual,const float* gates,
    const float* weight,float* branch,float* carry,uint16_t* half,float* inverses,
    int rows,float epsilon,cudaStream_t stream,bool publish_branch) {
    if(rows<16||rows>2048||!std::isfinite(epsilon)||epsilon<=0)return false;
    struct Span {const void* pointer;size_t bytes;size_t alignment;};
    const size_t n=size_t(rows)*2560,w=n*4;
    const Span inputs[]={{routed,n*10*2,2},{shared,n*4,4},{scale,size_t(rows)*4,4},
        {residual,w*4,4},{gates,size_t(rows)*4*4,4},{weight,10240*4,4}};
    const Span outputs[]={{branch,n*4,4},{carry,w*4,4},{half,w*2,2},{inverses,size_t(rows)*4*4,4}};
    const auto valid=[](Span a){const auto p=reinterpret_cast<uintptr_t>(a.pointer);
        return p&&p%a.alignment==0&&a.bytes<=UINTPTR_MAX-p;};
    const auto overlap=[](Span a,Span b){const auto p=reinterpret_cast<uintptr_t>(a.pointer),q=reinterpret_cast<uintptr_t>(b.pointer);
        return p<q+b.bytes&&q<p+a.bytes;};
    for(auto a:inputs)if(!valid(a))return false;
    for(auto a:outputs)if(!valid(a))return false;
    for(int i=0;i<4;++i){for(auto a:inputs)if(overlap(outputs[i],a))return false;
        for(int j=0;j<i;++j)if(overlap(outputs[i],outputs[j]))return false;}
    if(publish_branch)moe_combine_gr_inject_rms_deferred_cuda<true><<<rows,1024,0,stream>>>(routed,shared,scale,
        residual,gates,weight,branch,carry,reinterpret_cast<__half*>(half),inverses,epsilon);
    else moe_combine_gr_inject_rms_deferred_cuda<false><<<rows,1024,0,stream>>>(routed,shared,scale,
        residual,gates,weight,branch,carry,reinterpret_cast<__half*>(half),inverses,epsilon);
    return cudaGetLastError()==cudaSuccess;
}

template<bool CompactCarry>
static bool launch_hc_up_reduce_fused_impl(const uint16_t* activation,const uint16_t* weight,
    const float* carry,const float* gamma,const float* inverses,float* output,
    int rows,cudaStream_t stream,int input_pitch,int tile_columns) {
    if(input_pitch!=320&&input_pitch!=328&&input_pitch!=336)return false;
    if(tile_columns!=16&&tile_columns!=32)return false;
    if(!activation||!weight||!carry||!gamma||!inverses||!output||rows<16||rows>mollm::hc::prefill_max_rows||
        (reinterpret_cast<uintptr_t>(activation)&31)||(reinterpret_cast<uintptr_t>(weight)&31)||
        (reinterpret_cast<uintptr_t>(carry)&3)||(reinterpret_cast<uintptr_t>(gamma)&3)||
        (reinterpret_cast<uintptr_t>(inverses)&3)||(reinterpret_cast<uintptr_t>(output)&3))return false;
    const size_t output_bytes=size_t(rows)*2560*sizeof(float);
    const uintptr_t start=reinterpret_cast<uintptr_t>(output);
    if(output_bytes>UINTPTR_MAX-start)return false;
    struct Span{const void* pointer;size_t bytes;};
    const Span inputs[]={{activation,size_t(rows)*320*sizeof(uint16_t)},
        {weight,10240ull*320*sizeof(uint16_t)},{carry,size_t(rows)*10240*sizeof(float)},
        {gamma,10240*sizeof(float)},{inverses,size_t(rows)*4*sizeof(float)}};
    for(const auto& input:inputs){const uintptr_t p=reinterpret_cast<uintptr_t>(input.pointer);
        if(input.bytes>UINTPTR_MAX-p||(start<p+input.bytes&&p<start+output_bytes))return false;}
    if(tile_columns==32&&input_pitch==328)hc_up_reduce_fused_cuda<328,32,CompactCarry><<<dim3(80,(rows+15)/16),128,0,stream>>>(
        reinterpret_cast<const __half*>(activation),reinterpret_cast<const __half*>(weight),
        carry,gamma,inverses,output,rows);
    else if(tile_columns==32&&input_pitch==336)hc_up_reduce_fused_cuda<336,32,CompactCarry><<<dim3(80,(rows+15)/16),128,0,stream>>>(
        reinterpret_cast<const __half*>(activation),reinterpret_cast<const __half*>(weight),
        carry,gamma,inverses,output,rows);
    else if(tile_columns==32)hc_up_reduce_fused_cuda<320,32,CompactCarry><<<dim3(80,(rows+15)/16),128,0,stream>>>(
        reinterpret_cast<const __half*>(activation),reinterpret_cast<const __half*>(weight),
        carry,gamma,inverses,output,rows);
    else if(input_pitch==328)hc_up_reduce_fused_cuda<328,16,CompactCarry><<<dim3(160,(rows+15)/16),128,0,stream>>>(
        reinterpret_cast<const __half*>(activation),reinterpret_cast<const __half*>(weight),
        carry,gamma,inverses,output,rows);
    else if(input_pitch==336)hc_up_reduce_fused_cuda<336,16,CompactCarry><<<dim3(160,(rows+15)/16),128,0,stream>>>(
        reinterpret_cast<const __half*>(activation),reinterpret_cast<const __half*>(weight),
        carry,gamma,inverses,output,rows);
    else hc_up_reduce_fused_cuda<320,16,CompactCarry><<<dim3(160,(rows+15)/16),128,0,stream>>>(
        reinterpret_cast<const __half*>(activation),reinterpret_cast<const __half*>(weight),
        carry,gamma,inverses,output,rows);
    return cudaGetLastError()==cudaSuccess;
}

bool launch_hc_up_reduce_fused(const uint16_t* activation,const uint16_t* weight,
    const float* carry,const float* gamma,const float* inverses,float* output,
    int rows,cudaStream_t stream,int input_pitch,int tile_columns,bool compact_carry) {
    if(compact_carry)return launch_hc_up_reduce_fused_impl<true>(activation,weight,carry,gamma,inverses,output,rows,stream,input_pitch,tile_columns);
    return launch_hc_up_reduce_fused_impl<false>(activation,weight,carry,gamma,inverses,output,rows,stream,input_pitch,tile_columns);
}

bool launch_gr_reduce_sigmoid_deferred_rms(const float* carry,const float* logits,
    const float* weight,const float* inverses,float* output,int rows,cudaStream_t stream,bool compact_carry) {
    if(!carry||!logits||!weight||!inverses||!output||rows<=0||rows>4096)return false;
    if(compact_carry&&(rows<16||rows>mollm::hc::prefill_max_rows||(reinterpret_cast<uintptr_t>(carry)&3)||
        (reinterpret_cast<uintptr_t>(logits)&3)||(reinterpret_cast<uintptr_t>(weight)&3)||
        (reinterpret_cast<uintptr_t>(inverses)&3)||(reinterpret_cast<uintptr_t>(output)&3)))return false;
    const size_t count=size_t(rows)*2560;
    struct Span {const void* pointer;size_t bytes;};
    const Span inputs[]={{carry,count*4*4},{logits,count*4*4},
        {weight,10240*4},{inverses,size_t(rows)*4*4}};
    const uintptr_t start=reinterpret_cast<uintptr_t>(output);
    if(count*4>UINTPTR_MAX-start)return false;
    for(auto input:inputs){
        const uintptr_t p=reinterpret_cast<uintptr_t>(input.pointer);
        if(input.bytes>UINTPTR_MAX-p||(start<p+input.bytes&&p<start+count*4))return false;
    }
    if(compact_carry)gr_reduce_sigmoid_deferred_rms_cuda<true><<<(count+255)/256,256,0,stream>>>(
        carry,logits,weight,inverses,output,count);
    else gr_reduce_sigmoid_deferred_rms_cuda<false><<<(count+255)/256,256,0,stream>>>(
        carry,logits,weight,inverses,output,count);
    return cudaGetLastError()==cudaSuccess;
}

bool launch_hc_compact_carry_materialize(float* logical_buffer,uint16_t* scratch,
    size_t scratch_bytes,int rows,cudaStream_t stream) {
    if(!logical_buffer||!scratch||rows<1||rows>mollm::hc::prefill_max_rows)return false;
    const uintptr_t output=reinterpret_cast<uintptr_t>(logical_buffer);
    const uintptr_t staging=reinterpret_cast<uintptr_t>(scratch);
    const size_t count=size_t(rows)*10240;
    const size_t logical_bytes=count*sizeof(float),compact_bytes=count*sizeof(uint16_t);
    if((output&3)||(staging&1)||scratch_bytes<compact_bytes||
       logical_bytes>UINTPTR_MAX-output||scratch_bytes>UINTPTR_MAX-staging||
       (output<staging+scratch_bytes&&staging<output+logical_bytes))return false;
    // Never expand directly in place: one CTA could overwrite another CTA's
    // unread packed data. The ordered D2D copy completes before all decode CTAs.
    if(cudaMemcpyAsync(scratch,logical_buffer,compact_bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)
        return false;
    hc_compact_carry_materialize_cuda<<<(count+255)/256,256,0,stream>>>(scratch,logical_buffer,count);
    return cudaGetLastError()==cudaSuccess;
}

bool launch_gr_rms_deferred_materialize(const float* carry,const float* weight,
    const float* inverses,float* output,int rows,cudaStream_t stream,bool compact_carry) {
    if(!carry||!weight||!inverses||!output||rows<=0||rows>4096)return false;
    if(compact_carry&&(rows<16||rows>mollm::hc::prefill_max_rows||(reinterpret_cast<uintptr_t>(carry)&3)||
        (reinterpret_cast<uintptr_t>(weight)&3)||(reinterpret_cast<uintptr_t>(inverses)&3)||
        (reinterpret_cast<uintptr_t>(output)&3)))return false;
    const size_t count=size_t(rows)*10240;
    struct Span {const void* pointer;size_t bytes;};
    const Span inputs[]={{carry,count*4},{weight,10240*4},{inverses,size_t(rows)*4*4}};
    const uintptr_t start=reinterpret_cast<uintptr_t>(output);
    if(count*4>UINTPTR_MAX-start)return false;
    for(auto input:inputs){
        const uintptr_t p=reinterpret_cast<uintptr_t>(input.pointer);
        if(input.bytes>UINTPTR_MAX-p||(start<p+input.bytes&&p<start+count*4))return false;
    }
    if(compact_carry)gr_rms_deferred_materialize_cuda<true><<<(count+255)/256,256,0,stream>>>(carry,weight,inverses,output,count);
    else gr_rms_deferred_materialize_cuda<false><<<(count+255)/256,256,0,stream>>>(carry,weight,inverses,output,count);
    return cudaGetLastError()==cudaSuccess;
}

bool launch_gr_inject_tiled(const float* branch, const float* residual,
    const float* gates, float* output, int hidden, int streams, int rows,
    cudaStream_t stream) {
    if (!branch || !residual || !gates || !output || hidden <= 0 ||
        streams <= 0 || streams > 65535 || rows <= 0 || rows > 65535) return false;
    if (static_cast<size_t>(hidden) * streams * rows > SIZE_MAX / sizeof(float)) return false;
    const dim3 grid((static_cast<unsigned>(hidden) + 255) / 256,
        static_cast<unsigned>(streams), static_cast<unsigned>(rows));
    gr_inject_tiled_cuda<<<grid, 256, 0, stream>>>(
        branch, residual, gates, output, hidden, streams);
    return true;
}

void launch_add_rms_norm(
    float* residual, const float* update, const float* weight, float* output,
    int width, int rows, size_t residual_row_stride,
    size_t update_row_stride, size_t output_row_stride, float epsilon) {
    // Decode has one wide row and benefits from exposing one column per
    // thread. Keep smaller tensors and multi-row prefill on the lower-cost
    // 256-thread specialization.
    if (rows == 1 && width >= 1024) {
        add_rms_norm_cuda<1024><<<rows, 1024>>>(
            residual, update, weight, output, width, rows,
            residual_row_stride, update_row_stride, output_row_stride,
            epsilon);
    } else {
        add_rms_norm_cuda<256><<<rows, 256>>>(
            residual, update, weight, output, width, rows,
            residual_row_stride, update_row_stride, output_row_stride,
            epsilon);
    }
}

void launch_contiguous(
    const float* input, float* output, size_t count, int64_t d0, int64_t d1,
    int64_t d2, size_t s0, size_t s1, size_t s2, size_t s3) {
    constexpr int threads = 256;
    contiguous_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                      threads>>>(
        input, output, count, d0, d1, d2, s0, s1, s2, s3);
}

void launch_tile(const float* input, float* output, size_t count,
    int64_t o0, int64_t o1, int64_t o2,
    int64_t i0, int64_t i1, int64_t i2, int64_t i3,
    size_t s0, size_t s1, size_t s2, size_t s3) {
    tile_cuda<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        input, output, count, o0, o1, o2, i0, i1, i2, i3, s0, s1, s2, s3);
}

void launch_argmax(const float* input, int count, ArgMaxPair* partial,
                   int groups, ArgMaxPair* result) {
    constexpr int threads = 256;
    argmax_stage1_cuda<<<groups, threads>>>(input, count, partial);
    argmax_stage2_cuda<<<1, threads>>>(partial, groups, result);
}

}  // namespace mollm_cuda
