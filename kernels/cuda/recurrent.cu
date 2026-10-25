#include "kernels/cuda/recurrent.h"
#include "kernels/cuda/reduction.cuh"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int shortconv_max_kernel = 8;
constexpr int shortconv_parallel_min_sequence = 8;
constexpr int gdn_head_dim = 128;
constexpr int gdn_warp_size = 32;
constexpr int gdn_state_values_per_lane = gdn_head_dim / gdn_warp_size;
// Two independent value rows per block balance grid-level parallelism against
// launch and scheduling overhead across both small and large hybrid models.
constexpr int gdn_value_warps_per_block = 2;
static_assert(gdn_head_dim % gdn_warp_size == 0);

template<bool StorePrefixes = false>
__global__ void shortconv_serial_cuda(
    const float* input, const float* weight, float* state, float* output,
    int groups, int sequence_length, int kernel_size, int real_tokens,
    size_t input_feature_stride, size_t input_row_stride, float* prefixes = nullptr) {
    const int group = static_cast<int>(blockIdx.x) * blockDim.x +
        static_cast<int>(threadIdx.x);
    if (group >= groups)
        return;

    const int prefix_length = kernel_size - 1;
    float window[shortconv_max_kernel - 1];
    float* group_state = state +
        static_cast<size_t>(group) * prefix_length;
#pragma unroll
    for (int index = 0; index < shortconv_max_kernel - 1; ++index)
        if (index < prefix_length)
            window[index] = group_state[index];

    const float* group_weight = weight +
        static_cast<size_t>(group) * kernel_size;
    float* group_output = output +
        static_cast<size_t>(group) * sequence_length;
    for (int token = 0; token < real_tokens; ++token) {
        const float value = input[
            static_cast<size_t>(token) * input_row_stride +
            static_cast<size_t>(group) * input_feature_stride];
        float sum = value * group_weight[prefix_length];
#pragma unroll
        for (int index = 0; index < shortconv_max_kernel - 1; ++index)
            if (index < prefix_length)
                sum = fmaf(window[index], group_weight[index], sum);
        group_output[token] = sum / (1.0f + expf(-sum));
#pragma unroll
        for (int index = 0; index < shortconv_max_kernel - 2; ++index)
            if (index + 1 < prefix_length)
                window[index] = window[index + 1];
        if (prefix_length > 0)
            window[prefix_length - 1] = value;
        if constexpr (StorePrefixes) {
#pragma unroll
            for(int index=0;index<shortconv_max_kernel-1;++index)
                if(index<prefix_length)
                    prefixes[(static_cast<size_t>(token)*groups+group)*prefix_length+index]=window[index];
        }
    }
    for (int token = real_tokens; token < sequence_length; ++token)
        group_output[token] = 0.0f;
#pragma unroll
    for (int index = 0; index < shortconv_max_kernel - 1; ++index)
        if (index < prefix_length)
            group_state[index] = window[index];
}

// Causal convolution outputs are independent once the incoming history is
// fixed. Prefill therefore parallelizes over both tokens and groups, then a
// separate ordered pass commits the final history. Decode stays on the serial
// kernel above so its output and state update share one launch.
template<bool TokenMajor = false, bool TransposeOutput = false>
__global__ void shortconv_prefill_cuda(
    const float* input, const float* weight, const float* state,
    float* output, int groups, int sequence_length, int kernel_size,
    int real_tokens, size_t input_feature_stride, size_t input_row_stride) {
    const int group = TokenMajor ? static_cast<int>(blockIdx.y)
        : static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    const int token = TokenMajor
        ? static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x)
        : TransposeOutput ? static_cast<int>(blockIdx.y) * 8 + static_cast<int>(threadIdx.y)
        : static_cast<int>(blockIdx.y);
    if (!TransposeOutput && (group >= groups || token >= sequence_length))
        return;
    const size_t output_index =
        static_cast<size_t>(group) * sequence_length + token;
    if (!TransposeOutput && token >= real_tokens) {
        output[output_index] = 0.0f;
        return;
    }

    const int prefix_length = kernel_size - 1;
    const float* group_weight = weight +
        static_cast<size_t>(group) * kernel_size;
    const bool valid = group < groups && token < real_tokens && token < sequence_length;
    const float current = valid ? input[
        static_cast<size_t>(token) * input_row_stride +
        static_cast<size_t>(group) * input_feature_stride] : 0.f;
    float sum = valid ? current * group_weight[prefix_length] : 0.f;
#pragma unroll
    for (int tap = 0; tap < shortconv_max_kernel - 1; ++tap) {
        if (valid && tap < prefix_length) {
            const int source_token = token + tap - prefix_length;
            const float value = source_token < 0
                ? state[static_cast<size_t>(group) * prefix_length +
                        static_cast<size_t>(source_token + prefix_length)]
                : input[static_cast<size_t>(source_token) * input_row_stride +
                        static_cast<size_t>(group) * input_feature_stride];
            sum = fmaf(value, group_weight[tap], sum);
        }
    }
    const float result = valid ? sum / (1.0f + expf(-sum)) : 0.f;
    if constexpr (TransposeOutput) {
        __shared__ float tile[8][33];
        tile[threadIdx.y][threadIdx.x] = result;
        __syncthreads();
        const int linear = static_cast<int>(threadIdx.y) * 32 + threadIdx.x;
        const int write_group = static_cast<int>(blockIdx.x) * 32 + linear / 8;
        const int write_token = static_cast<int>(blockIdx.y) * 8 + linear % 8;
        if (write_group < groups && write_token < sequence_length)
            output[static_cast<size_t>(write_group) * sequence_length + write_token] = tile[linear % 8][linear / 8];
    } else output[output_index] = result;
}

__global__ void shortconv_commit_state_cuda(
    const float* input, float* state, int groups, int kernel_size,
    int real_tokens, size_t input_feature_stride, size_t input_row_stride) {
    const int group = static_cast<int>(blockIdx.x) * blockDim.x +
        static_cast<int>(threadIdx.x);
    if (group >= groups)
        return;
    const int prefix_length = kernel_size - 1;
    float* group_state = state +
        static_cast<size_t>(group) * prefix_length;
    // source_state is always ahead of destination for real_tokens > 0, so
    // ascending in-place copies preserve the still-needed incoming history.
#pragma unroll
    for (int index = 0; index < shortconv_max_kernel - 1; ++index) {
        if (index < prefix_length) {
            const int source_token = real_tokens - prefix_length + index;
            group_state[index] = source_token < 0
                ? group_state[source_token + prefix_length]
                : input[static_cast<size_t>(source_token) * input_row_stride +
                        static_cast<size_t>(group) * input_feature_stride];
        }
    }
}

__device__ __forceinline__ float gdn_softplus(float value) {
    if (value > 20.0f)
        return value;
    if (value < -20.0f)
        return expf(value);
    return log1pf(expf(value));
}

// Prefill exposes the independent value columns of the recurrent state as
// separate warps. Q/K normalization and output RMSNorm remain separate so the
// recurrence can keep one state column in registers for the entire sequence.
__global__ void gdn_prepare_qk_128_cuda(
    const float* qkv, float* normalized_query, float* normalized_key,
    float* query_key_dot, int num_heads, int sequence_length,
    bool normalize_qk, float l2_epsilon) {
    const int row = static_cast<int>(blockIdx.x);
    const int dimension = static_cast<int>(threadIdx.x);
    const int key_head = row / sequence_length;
    const int token = row - key_head * sequence_length;
    if (key_head >= num_heads)
        return;

    const int qkv_key_width = num_heads * gdn_head_dim;
    const size_t query_index =
        (static_cast<size_t>(key_head) * gdn_head_dim + dimension) *
            sequence_length + token;
    const size_t key_index =
        (static_cast<size_t>(qkv_key_width) +
         static_cast<size_t>(key_head) * gdn_head_dim + dimension) *
            sequence_length + token;
    const float query = qkv[query_index];
    const float key = qkv[key_index];
    float query_inverse = 1.0f;
    float key_inverse = 1.0f;
    if (normalize_qk) {
        __shared__ float reduction[gdn_head_dim];
        const float query_sum =
            mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
                query * query, reduction);
        const float key_sum =
            mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
                key * key, reduction);
        query_inverse = rsqrtf(query_sum + l2_epsilon);
        key_inverse = rsqrtf(key_sum + l2_epsilon);
    }
    const size_t destination =
        (static_cast<size_t>(key_head) * sequence_length + token) *
            gdn_head_dim + dimension;
    const float normalized_query_value = query * query_inverse;
    const float normalized_key_value = key * key_inverse;
    normalized_query[destination] = normalized_query_value;
    normalized_key[destination] = normalized_key_value;
    __shared__ float dot_reduction[gdn_head_dim];
    const float dot = mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
        normalized_query_value * normalized_key_value, dot_reduction);
    if (dimension == 0)
        query_key_dot[row] = dot;
}

__global__ void gdn_prepare_gates_cuda(
    const float* a, const float* b, const float* a_log,
    const float* dt_bias, float* decay, float* beta,
    int num_value_heads, int sequence_length, size_t a_row_stride,
    size_t b_row_stride) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x +
        static_cast<int>(threadIdx.x);
    const int count = sequence_length * num_value_heads;
    if (index >= count)
        return;
    const int token = index / num_value_heads;
    const int value_head = index - token * num_value_heads;
    const float a_value =
        a[static_cast<size_t>(token) * a_row_stride + value_head];
    const float b_value =
        b[static_cast<size_t>(token) * b_row_stride + value_head];
    decay[index] = expf(
        -expf(a_log[value_head]) *
        gdn_softplus(a_value + dt_bias[value_head]));
    beta[index] = 1.0f / (1.0f + expf(-b_value));
}

template<bool PrefetchQK = false, int ValueWarps = 2, bool ButterflyKey = false, bool BroadcastGates = false>
__global__ void gdn_recurrence_rows_128_cuda(
    const float* qkv, const float* normalized_query,
    const float* normalized_key, const float* query_key_dot,
    const float* decay, const float* beta, float* state, float* raw_output,
    int num_heads, int num_value_heads, int sequence_length,
    int real_tokens, float scale) {
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int value_dimension =
        static_cast<int>(blockIdx.x) * ValueWarps + warp;
    const int value_head = static_cast<int>(blockIdx.y);
    if (value_dimension >= gdn_head_dim || value_head >= num_value_heads)
        return;

    const int repeat = num_value_heads / num_heads;
    const int key_head = value_head / repeat;
    const int qkv_key_width = num_heads * gdn_head_dim;
    float* head_state = state +
        static_cast<size_t>(value_head) * gdn_head_dim * gdn_head_dim;
    // A warp owns one value-state column. Each lane keeps its four strided
    // key dimensions in registers, which makes Q/K loads coalesced and lets
    // the two dot products use a five-step warp reduction without shared
    // memory or a serial lane-zero fold.
    float local_state[gdn_state_values_per_lane];
#pragma unroll
    for (int chunk = 0; chunk < gdn_state_values_per_lane; ++chunk) {
        const int key_dimension = lane + chunk * warpSize;
        local_state[chunk] = head_state[
            static_cast<size_t>(key_dimension) * gdn_head_dim +
            value_dimension];
    }

    float cached_key[4]{}, cached_query[4]{};
    if constexpr (PrefetchQK) {
        if (real_tokens > 0) {
            const size_t base = static_cast<size_t>(key_head) * sequence_length * gdn_head_dim;
#pragma unroll
            for (int chunk = 0; chunk < 4; ++chunk) {
                cached_key[chunk] = normalized_key[base + lane + chunk * warpSize];
                cached_query[chunk] = normalized_query[base + lane + chunk * warpSize];
            }
        }
    }
    for (int token = 0; token < real_tokens; ++token) {
        const int gate_index = token * num_value_heads + value_head;
        float token_decay = 0.0f;
        if constexpr (BroadcastGates) {
            if (lane == 0) token_decay = decay[gate_index];
            token_decay = __shfl_sync(0xffffffffu, token_decay, 0);
        }
        const size_t qk_base =
            (static_cast<size_t>(key_head) * sequence_length + token) *
            gdn_head_dim;
        float next_key[4]{}, next_query[4]{};
        if constexpr (PrefetchQK) {
            if (token + 1 < real_tokens) {
#pragma unroll
                for (int chunk = 0; chunk < 4; ++chunk) {
                    const size_t next = qk_base + gdn_head_dim + lane + chunk * warpSize;
                    next_key[chunk] = normalized_key[next];
                    next_query[chunk] = normalized_query[next];
                }
            }
        }
        float key_projection = 0.0f;
        float attention_decay = 0.0f;
#pragma unroll
        for (int chunk = 0; chunk < gdn_state_values_per_lane; ++chunk) {
            const int key_dimension = lane + chunk * warpSize;
            local_state[chunk] *= BroadcastGates ? token_decay : decay[gate_index];
            key_projection = fmaf(
                local_state[chunk],
                PrefetchQK ? cached_key[chunk] : normalized_key[qk_base + key_dimension], key_projection);
            attention_decay = fmaf(
                local_state[chunk],
                PrefetchQK ? cached_query[chunk] : normalized_query[qk_base + key_dimension], attention_decay);
        }
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            if constexpr (ButterflyKey)
                key_projection += __shfl_xor_sync(0xffffffffu,key_projection,offset);
            else
                key_projection += __shfl_down_sync(0xffffffffu,key_projection,offset);
            attention_decay += __shfl_down_sync(
                0xffffffffu, attention_decay, offset);
        }
        // XOR reduction distributes the same finite-input tree result to
        // every lane, avoiding the final lane-zero broadcast. Diagnostic only.
        if constexpr (!ButterflyKey)
            key_projection = __shfl_sync(0xffffffffu, key_projection, 0);
        const size_t value_index =
            (static_cast<size_t>(2 * qkv_key_width) +
             static_cast<size_t>(value_head) * gdn_head_dim +
             value_dimension) * sequence_length + token;
        float token_value = 0.0f, token_beta = 0.0f;
        if constexpr (BroadcastGates) {
            if (lane == 0) {
                token_value = qkv[value_index];
                token_beta = beta[gate_index];
            }
            token_value = __shfl_sync(0xffffffffu, token_value, 0);
            token_beta = __shfl_sync(0xffffffffu, token_beta, 0);
        }
        const float delta =
            ((BroadcastGates ? token_value : qkv[value_index]) - key_projection) *
            (BroadcastGates ? token_beta : beta[gate_index]);
#pragma unroll
        for (int chunk = 0; chunk < gdn_state_values_per_lane; ++chunk) {
            const int key_dimension = lane + chunk * warpSize;
            local_state[chunk] = fmaf(
                PrefetchQK ? cached_key[chunk] : normalized_key[qk_base + key_dimension], delta,
                local_state[chunk]);
        }
        if (lane == 0) {
            const size_t output_index =
                (static_cast<size_t>(token) * num_value_heads + value_head) *
                    gdn_head_dim + value_dimension;
            raw_output[output_index] =
                (attention_decay + delta * query_key_dot[
                    static_cast<size_t>(key_head) * sequence_length + token]) *
                scale;
        }
        if constexpr (PrefetchQK) {
#pragma unroll
            for (int chunk = 0; chunk < 4; ++chunk) {
                cached_key[chunk] = next_key[chunk];
                cached_query[chunk] = next_query[chunk];
            }
        }
    }
    if (lane == 0) {
        for (int token = real_tokens; token < sequence_length; ++token) {
            const size_t output_index =
                (static_cast<size_t>(token) * num_value_heads + value_head) *
                    gdn_head_dim + value_dimension;
            raw_output[output_index] = 0.0f;
        }
    }
#pragma unroll
    for (int chunk = 0; chunk < gdn_state_values_per_lane; ++chunk) {
        const int key_dimension = lane + chunk * warpSize;
        head_state[static_cast<size_t>(key_dimension) * gdn_head_dim +
                   value_dimension] = local_state[chunk];
    }
}

// Diagnostic: two independent state columns share each warp's Q/K loads.
// Each column preserves the original four-FMA and down-shuffle trees.
__global__ void gdn_recurrence_pairs_128_cuda(
    const float* qkv, const float* query, const float* key, const float* dot,
    const float* decay, const float* beta, float* state, float* output,
    int heads, int value_heads, int sequence, int real_tokens, float scale) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int first_value = (blockIdx.x * 2 + warp) * 2;
    const int value_head = blockIdx.y, key_head = value_head / (value_heads / heads);
    float* head_state = state + size_t(value_head) * 128 * 128;
    float local[2][4];
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        const auto values = *reinterpret_cast<const float2*>(
            head_state + size_t(lane + chunk * 32) * 128 + first_value);
        local[0][chunk] = values.x; local[1][chunk] = values.y;
    }
    for (int token = 0; token < real_tokens; ++token) {
        const size_t qk_base = (size_t(key_head) * sequence + token) * 128;
        const int gate_index = token * value_heads + value_head;
        const float gate_decay = decay[gate_index];
        float current_key[4], key_projection[2] = {}, attention[2] = {};
#pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            current_key[chunk] = key[qk_base + lane + chunk * 32];
            const float query_value = query[qk_base + lane + chunk * 32];
#pragma unroll
            for (int column = 0; column < 2; ++column) {
                local[column][chunk] *= gate_decay;
                key_projection[column] = fmaf(local[column][chunk], current_key[chunk], key_projection[column]);
                attention[column] = fmaf(local[column][chunk], query_value, attention[column]);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
#pragma unroll
            for (int column = 0; column < 2; ++column) {
                key_projection[column] += __shfl_down_sync(0xffffffffu, key_projection[column], offset);
                attention[column] += __shfl_down_sync(0xffffffffu, attention[column], offset);
            }
        }
        float delta[2];
#pragma unroll
        for (int column = 0; column < 2; ++column) {
            key_projection[column] = __shfl_sync(0xffffffffu, key_projection[column], 0);
            const size_t value_index = (size_t(2 * heads * 128) + size_t(value_head) * 128 +
                first_value + column) * sequence + token;
            delta[column] = (qkv[value_index] - key_projection[column]) * beta[gate_index];
        }
#pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk)
#pragma unroll
            for (int column = 0; column < 2; ++column)
                local[column][chunk] = fmaf(current_key[chunk], delta[column], local[column][chunk]);
        if (lane == 0) {
            const size_t index = (size_t(token) * value_heads + value_head) * 128 + first_value;
            const float qk_dot = dot[size_t(key_head) * sequence + token];
            output[index] = (attention[0] + delta[0] * qk_dot) * scale;
            output[index + 1] = (attention[1] + delta[1] * qk_dot) * scale;
        }
    }
    if (lane == 0)
        for (int token = real_tokens; token < sequence; ++token) {
            const size_t index = (size_t(token) * value_heads + value_head) * 128 + first_value;
            output[index] = 0.f; output[index + 1] = 0.f;
        }
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk)
        *reinterpret_cast<float2*>(head_state + size_t(lane + chunk * 32) * 128 + first_value) =
            make_float2(local[0][chunk], local[1][chunk]);
}

// Four-column reuse; retained paired kernel stays unchanged.
__global__ void gdn_recurrence_quads_128_cuda(
    const float* qkv,const float* query,const float* key,const float* dot,
    const float* decay,const float* beta,float* state,float* output,
    int heads,int value_heads,int sequence,int real_tokens,float scale) {
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const int first_value=(blockIdx.x*2+warp)*4;
    const int value_head=blockIdx.y,key_head=value_head/(value_heads/heads);
    float* head_state=state+size_t(value_head)*128*128;
    float local[4][4];
#pragma unroll
    for(int chunk=0;chunk<4;++chunk) {
        const auto values=*reinterpret_cast<const float4*>(head_state+size_t(lane+chunk*32)*128+first_value);
        local[0][chunk]=values.x;local[1][chunk]=values.y;
        local[2][chunk]=values.z;local[3][chunk]=values.w;
    }
    for(int token=0;token<real_tokens;++token) {
        const size_t qk_base=(size_t(key_head)*sequence+token)*128;
        const int gate_index=token*value_heads+value_head;
        const float gate_decay=decay[gate_index];
        float current_key[4],key_projection[4]={},attention[4]={};
#pragma unroll
        for(int chunk=0;chunk<4;++chunk) {
            current_key[chunk]=key[qk_base+lane+chunk*32];
            const float query_value=query[qk_base+lane+chunk*32];
#pragma unroll
            for(int column=0;column<4;++column) {
                local[column][chunk]*=gate_decay;
                key_projection[column]=fmaf(local[column][chunk],current_key[chunk],key_projection[column]);
                attention[column]=fmaf(local[column][chunk],query_value,attention[column]);
            }
        }
#pragma unroll
        for(int offset=16;offset>0;offset/=2)
#pragma unroll
            for(int column=0;column<4;++column) {
                key_projection[column]+=__shfl_down_sync(0xffffffffu,key_projection[column],offset);
                attention[column]+=__shfl_down_sync(0xffffffffu,attention[column],offset);
            }
        float delta[4];
#pragma unroll
        for(int column=0;column<4;++column) {
            key_projection[column]=__shfl_sync(0xffffffffu,key_projection[column],0);
            const size_t value_index=(size_t(2*heads*128)+size_t(value_head)*128+first_value+column)*sequence+token;
            delta[column]=(qkv[value_index]-key_projection[column])*beta[gate_index];
        }
#pragma unroll
        for(int chunk=0;chunk<4;++chunk)
#pragma unroll
            for(int column=0;column<4;++column)
                local[column][chunk]=fmaf(current_key[chunk],delta[column],local[column][chunk]);
        if(lane==0) {
            const size_t index=(size_t(token)*value_heads+value_head)*128+first_value;
            const float qk_dot=dot[size_t(key_head)*sequence+token];
#pragma unroll
            for(int column=0;column<4;++column)output[index+column]=(attention[column]+delta[column]*qk_dot)*scale;
        }
    }
    if(lane==0)for(int token=real_tokens;token<sequence;++token) {
        const size_t index=(size_t(token)*value_heads+value_head)*128+first_value;
#pragma unroll
        for(int column=0;column<4;++column)output[index+column]=0.f;
    }
#pragma unroll
    for(int chunk=0;chunk<4;++chunk)
        *reinterpret_cast<float4*>(head_state+size_t(lane+chunk*32)*128+first_value)=
            make_float4(local[0][chunk],local[1][chunk],local[2][chunk],local[3][chunk]);
}

__global__ void gdn_post_128_cuda(
    const float* z, const float* norm_weight, const float* raw_output,
    float* output, int num_value_heads, int sequence_length,
    float rms_epsilon, size_t z_row_stride, bool sigmoid_output_gate) {
    const int row = static_cast<int>(blockIdx.x);
    const int dimension = static_cast<int>(threadIdx.x);
    const int value_head = row % num_value_heads;
    const int token = row / num_value_heads;
    if (token >= sequence_length)
        return;
    const size_t base = static_cast<size_t>(row) * gdn_head_dim;
    const float value = raw_output[base + dimension];
    __shared__ float reduction[gdn_head_dim];
    const float square_sum =
        mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
            value * value, reduction);
    const float inverse_rms =
        rsqrtf(square_sum / gdn_head_dim + rms_epsilon);
    const float z_value = z[
        static_cast<size_t>(token) * z_row_stride +
        static_cast<size_t>(value_head) * gdn_head_dim + dimension];
    output[base + dimension] = value * inverse_rms *
        norm_weight[dimension] *
        ((sigmoid_output_gate ? 1.0f : z_value) / (1.0f + expf(-z_value)));
}

// Qwen3.5's recurrent dimensions are 128-wide. One block owns one value head
// so every state column is independent across threads while the token axis is
// traversed in recurrence order. Q/K and RMS reductions stay on chip.
__device__ __forceinline__ float2 gdn_reduce_pair_128(float2 value, float2* scratch) {
    const int thread = threadIdx.x;
    scratch[thread] = value;
    __syncthreads();
    if (thread < 64) {
        scratch[thread].x += scratch[thread + 64].x;
        scratch[thread].y += scratch[thread + 64].y;
    }
    __syncthreads();
    if (thread < 32) {
        value.x = scratch[thread].x + scratch[thread + 32].x;
        value.y = scratch[thread].y + scratch[thread + 32].y;
        value.x = mollm_cuda::detail::warp_reduce_sum(value.x);
        value.y = mollm_cuda::detail::warp_reduce_sum(value.y);
        if (thread == 0) scratch[0] = value;
    }
    __syncthreads();
    const float2 result = scratch[0];
    __syncthreads();
    return result;
}

template <bool CacheState, bool StorePrefixes = false, bool RegisterState = false,
          bool PairedQK = false>
__global__ void gdn_128_cuda(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, int num_heads, int num_value_heads,
    int sequence_length, int real_tokens, bool normalize_qk, float rms_epsilon,
    float l2_epsilon, float scale, size_t a_row_stride,
    size_t b_row_stride, size_t z_row_stride, bool sigmoid_output_gate,
    float* prefixes = nullptr) {
    const int value_head = static_cast<int>(blockIdx.x);
    const int dimension = static_cast<int>(threadIdx.x);
    if (value_head >= num_value_heads)
        return;
    const int repeat = num_value_heads / num_heads;
    const int key_head = value_head / repeat;
    const int qkv_key_width = num_heads * gdn_head_dim;
    const int output_width = num_value_heads * gdn_head_dim;
    float* head_state = state +
        static_cast<size_t>(value_head) * gdn_head_dim * gdn_head_dim;
    extern __shared__ float state_cache[];
    float register_state[RegisterState ? gdn_head_dim : 1];

    __shared__ float query[gdn_head_dim];
    __shared__ float key[gdn_head_dim];
    __shared__ float reduction[gdn_head_dim];
    __shared__ float gate[2];
    __shared__ float2 paired_reduction[PairedQK ? gdn_head_dim : 1];

    if constexpr (RegisterState) {
#pragma unroll
        for (int row = 0; row < gdn_head_dim; ++row)
            register_state[row] = head_state[static_cast<size_t>(row) * gdn_head_dim + dimension];
    } else if constexpr (CacheState) {
        for (int state_row = 0; state_row < gdn_head_dim; ++state_row) {
            const size_t index =
                static_cast<size_t>(state_row) * gdn_head_dim + dimension;
            state_cache[index] = head_state[index];
        }
        __syncthreads();
    }

    for (int token = 0; token < sequence_length; ++token) {
        if (token >= real_tokens) {
            output[static_cast<size_t>(token) * output_width +
                   static_cast<size_t>(value_head) * gdn_head_dim +
                   dimension] = 0.0f;
            continue;
        }

        const size_t query_index =
            (static_cast<size_t>(key_head) * gdn_head_dim + dimension) *
                sequence_length + token;
        const size_t key_index =
            (static_cast<size_t>(qkv_key_width) +
             static_cast<size_t>(key_head) * gdn_head_dim + dimension) *
                sequence_length + token;
        query[dimension] = qkv[query_index];
        key[dimension] = qkv[key_index];
        float query_inverse = 1.0f;
        float key_inverse = 1.0f;
        if (normalize_qk) {
            float query_sum, key_sum;
            if constexpr (PairedQK) {
                const float2 sums = gdn_reduce_pair_128(make_float2(
                    query[dimension] * query[dimension], key[dimension] * key[dimension]), paired_reduction);
                query_sum = sums.x;
                key_sum = sums.y;
            } else {
                query_sum =
                    mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
                        query[dimension] * query[dimension], reduction);
                key_sum =
                    mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
                        key[dimension] * key[dimension], reduction);
            }
            query_inverse = rsqrtf(query_sum + l2_epsilon);
            key_inverse = rsqrtf(key_sum + l2_epsilon);
        }
        query[dimension] *= query_inverse;
        key[dimension] *= key_inverse;
        __syncthreads();
        const float query_key =
            mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
                query[dimension] * key[dimension], reduction);

        if (dimension == 0) {
            const float a_value =
                a[static_cast<size_t>(token) * a_row_stride + value_head];
            const float b_value =
                b[static_cast<size_t>(token) * b_row_stride + value_head];
            gate[0] = expf(
                -expf(a_log[value_head]) *
                gdn_softplus(a_value + dt_bias[value_head]));
            gate[1] = 1.0f / (1.0f + expf(-b_value));
        }
        __syncthreads();

        float key_value = 0.0f;
        float attention_decay = 0.0f;
#pragma unroll (RegisterState ? 128 : 1)
        for (int state_row = 0; state_row < gdn_head_dim; ++state_row) {
            const size_t index =
                static_cast<size_t>(state_row) * gdn_head_dim + dimension;
            float state_value;
            if constexpr (RegisterState)
                state_value = register_state[state_row];
            else if constexpr (CacheState)
                state_value = state_cache[index];
            else
                state_value = head_state[index];
            const float decayed = state_value * gate[0];
            if constexpr (RegisterState)
                register_state[state_row] = decayed;
            else if constexpr (CacheState)
                state_cache[index] = decayed;
            const float key_element = RegisterState ? static_cast<volatile float*>(key)[state_row] : key[state_row];
            const float query_element = RegisterState ? static_cast<volatile float*>(query)[state_row] : query[state_row];
            key_value = fmaf(decayed, key_element, key_value);
            attention_decay =
                fmaf(decayed, query_element, attention_decay);
        }
        const size_t value_index =
            (static_cast<size_t>(2 * qkv_key_width) +
             static_cast<size_t>(value_head) * gdn_head_dim + dimension) *
                sequence_length + token;
        const float delta = (qkv[value_index] - key_value) * gate[1];
#pragma unroll (RegisterState ? 128 : 1)
        for (int state_row = 0; state_row < gdn_head_dim; ++state_row) {
            const size_t index =
                static_cast<size_t>(state_row) * gdn_head_dim + dimension;
            if constexpr (RegisterState) {
                register_state[state_row] += static_cast<volatile float*>(key)[state_row] * delta;
            } else if constexpr (CacheState) {
                state_cache[index] += key[state_row] * delta;
            } else {
                head_state[index] =
                    head_state[index] * gate[0] + key[state_row] * delta;
            }
        }
        const float attention =
            (attention_decay + delta * query_key) * scale;
        const float square_sum =
            mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
                attention * attention, reduction);
        const float inverse_rms =
            rsqrtf(square_sum / gdn_head_dim + rms_epsilon);
        const float z_value = z[
            static_cast<size_t>(token) * z_row_stride +
            static_cast<size_t>(value_head) * gdn_head_dim + dimension];
        output[static_cast<size_t>(token) * output_width +
               static_cast<size_t>(value_head) * gdn_head_dim + dimension] =
            attention * inverse_rms * norm_weight[dimension] *
            ((sigmoid_output_gate ? 1.0f : z_value) /
             (1.0f + expf(-z_value)));
        if constexpr (StorePrefixes) {
            float* snapshot=prefixes+(static_cast<size_t>(token)*num_value_heads+value_head)*gdn_head_dim*gdn_head_dim;
            for(int row=0;row<gdn_head_dim;++row) {
                const size_t index=static_cast<size_t>(row)*gdn_head_dim+dimension;
                if constexpr (RegisterState) snapshot[index]=register_state[row];
                else snapshot[index]=CacheState?state_cache[index]:head_state[index];
            }
        }
        __syncthreads();
    }
    if constexpr (RegisterState) {
#pragma unroll
        for (int row = 0; row < gdn_head_dim; ++row)
            head_state[static_cast<size_t>(row) * gdn_head_dim + dimension] = register_state[row];
    } else if constexpr (CacheState) {
        for (int state_row = 0; state_row < gdn_head_dim; ++state_row) {
            const size_t index =
                static_cast<size_t>(state_row) * gdn_head_dim + dimension;
            head_state[index] = state_cache[index];
        }
    }
}

template<bool StorePrefixes = false>
bool configure_gdn_shared_state() {
    constexpr size_t state_cache_bytes =
        gdn_head_dim * gdn_head_dim * sizeof(float);
    int device = 0;
    int max_shared_bytes = 0;
    cudaFuncAttributes attributes{};
    const cudaError_t device_status = cudaGetDevice(&device);
    const cudaError_t limit_status = device_status == cudaSuccess
        ? cudaDeviceGetAttribute(
              &max_shared_bytes,
              cudaDevAttrMaxSharedMemoryPerBlockOptin,
              device)
        : device_status;
    const cudaError_t function_status = limit_status == cudaSuccess
        ? cudaFuncGetAttributes(&attributes, gdn_128_cuda<true,StorePrefixes>)
        : limit_status;
    if (function_status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (state_cache_bytes + attributes.sharedSizeBytes >
        static_cast<size_t>(max_shared_bytes))
        return false;

    const cudaError_t configure_status = cudaFuncSetAttribute(
        gdn_128_cuda<true,StorePrefixes>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(state_cache_bytes));
    if (configure_status != cudaSuccess)
        cudaGetLastError();
    return configure_status == cudaSuccess;
}

}  // namespace

namespace mollm_cuda {

bool launch_gdn_recurrence_probe(
    const float* qkv, const float* query, const float* key, const float* dot,
    const float* decay, const float* beta, float* state, float* output,
    int heads, int value_heads, int sequence, int real_tokens, float scale,
    bool prefetch_qk, int value_warps, bool butterfly_key, bool broadcast_gates, bool paired_values, bool quad_values) {
    if (!qkv || !query || !key || !dot || !decay || !beta || !state || !output ||
        heads <= 0 || heads > 8388607 || value_heads <= 0 || value_heads > 65535 ||
        value_heads % heads || sequence <= 0 || sequence > 2147483647 / value_heads ||
        real_tokens < 0 || real_tokens > sequence ||
        (value_warps != 2 && value_warps != 4 && value_warps != 8)) return false;
    if (butterfly_key && (prefetch_qk || value_warps != 2)) return false;
    if (broadcast_gates && (prefetch_qk || butterfly_key || value_warps != 2)) return false;
    if (quad_values) {
        if(prefetch_qk || butterfly_key || broadcast_gates || paired_values || value_warps!=2 ||
            (reinterpret_cast<uintptr_t>(state)&15))return false;
        gdn_recurrence_quads_128_cuda<<<dim3(16,static_cast<unsigned>(value_heads)),64>>>(
            qkv,query,key,dot,decay,beta,state,output,heads,value_heads,sequence,real_tokens,scale);
        return cudaGetLastError()==cudaSuccess;
    }
    if (paired_values) {
        if (prefetch_qk || butterfly_key || broadcast_gates || value_warps != 2 ||
            (reinterpret_cast<uintptr_t>(state) & 7)) return false;
        gdn_recurrence_pairs_128_cuda<<<dim3(32, static_cast<unsigned>(value_heads)), 64>>>(
            qkv, query, key, dot, decay, beta, state, output,
            heads, value_heads, sequence, real_tokens, scale);
        return cudaGetLastError() == cudaSuccess;
    }
    const dim3 grid(128 / value_warps, static_cast<unsigned>(value_heads));
    const auto kernel = broadcast_gates ? gdn_recurrence_rows_128_cuda<false,2,false,true>
        : butterfly_key ? gdn_recurrence_rows_128_cuda<false,2,true> : value_warps == 8
        ? (prefetch_qk ? gdn_recurrence_rows_128_cuda<true, 8> : gdn_recurrence_rows_128_cuda<false, 8>)
        : value_warps == 4
        ? (prefetch_qk ? gdn_recurrence_rows_128_cuda<true, 4> : gdn_recurrence_rows_128_cuda<false, 4>)
        : (prefetch_qk ? gdn_recurrence_rows_128_cuda<true, 2> : gdn_recurrence_rows_128_cuda<false, 2>);
    kernel<<<grid, value_warps * 32>>>(qkv, query, key, dot, decay, beta, state, output,
        heads, value_heads, sequence, real_tokens, scale);
    return cudaGetLastError() == cudaSuccess;
}

size_t gdn_prefill_scratch_bytes(
    int num_heads, int num_value_heads, int key_dimension,
    int value_dimension, int sequence_length) {
    if (num_heads <= 0 || num_value_heads <= 0 || key_dimension != 128 ||
        value_dimension != 128 || sequence_length <= 1)
        return 0;
    const size_t qk_elements = static_cast<size_t>(sequence_length) *
        num_heads * key_dimension;
    const size_t qk_rows =
        static_cast<size_t>(sequence_length) * num_heads;
    const size_t gate_elements = static_cast<size_t>(sequence_length) *
        num_value_heads;
    const size_t output_elements = gate_elements * value_dimension;
    return (2 * qk_elements + qk_rows + 2 * gate_elements + output_elements) *
        sizeof(float);
}

bool launch_shortconv(
    const float* input, const float* weight, float* state, float* output,
    int groups, int sequence_length, int kernel_size, int real_tokens,
    size_t input_feature_stride, size_t input_row_stride, int output_mapping) {
    if (!input || !weight || !state || !output || groups <= 0 || groups > INT32_MAX - 255 ||
        sequence_length <= 0 || sequence_length > INT32_MAX - 255 || kernel_size <= 0 ||
        kernel_size > shortconv_max_kernel || output_mapping < 0 || output_mapping > 2 ||
        (output_mapping == 1 && groups > 65535))
        return false;
    if (sequence_length >= shortconv_parallel_min_sequence &&
        (output_mapping == 2 ? (sequence_length + 7) / 8 > 65535
                            : output_mapping == 0 && sequence_length > 65535)) return false;
    const int process_length =
        real_tokens > 0 && real_tokens < sequence_length
        ? real_tokens : sequence_length;
    constexpr int threads = 256;
    const int group_blocks = (groups + threads - 1) / threads;
    if (sequence_length >= shortconv_parallel_min_sequence) {
        const dim3 grid(group_blocks, sequence_length);
        if (output_mapping == 2) {
            shortconv_prefill_cuda<false, true><<<dim3((groups + 31) / 32, (sequence_length + 7) / 8), dim3(32, 8)>>>(
                input, weight, state, output, groups, sequence_length,
                kernel_size, process_length, input_feature_stride, input_row_stride);
        } else if (output_mapping == 1) {
            shortconv_prefill_cuda<true><<<dim3((sequence_length + threads - 1) / threads, groups), threads>>>(
                input, weight, state, output, groups, sequence_length,
                kernel_size, process_length, input_feature_stride, input_row_stride);
        } else {
            shortconv_prefill_cuda<false><<<grid, threads>>>(
                input, weight, state, output, groups, sequence_length,
                kernel_size, process_length, input_feature_stride, input_row_stride);
        }
        if (kernel_size > 1)
            shortconv_commit_state_cuda<<<group_blocks, threads>>>(
                input, state, groups, kernel_size, process_length,
                input_feature_stride, input_row_stride);
    } else {
        shortconv_serial_cuda<><<<group_blocks, threads>>>(
            input, weight, state, output, groups, sequence_length,
            kernel_size, process_length, input_feature_stride,
            input_row_stride);
    }
    return true;
}

bool launch_shortconv_prefixes(
    const float* input, const float* weight, float* state, float* output,
    float* prefixes, int groups, int sequence_length, int kernel_size,
    int real_tokens, size_t input_feature_stride, size_t input_row_stride) {
    if(!input||!weight||!state||!output||!prefixes||groups<=0||
        groups>INT32_MAX-255||sequence_length<1||sequence_length>8||
        kernel_size<2||kernel_size>shortconv_max_kernel||real_tokens<1||
        real_tokens>sequence_length)return false;
    shortconv_serial_cuda<true><<<(groups+255)/256,256>>>(input,weight,state,
        output,groups,sequence_length,kernel_size,real_tokens,input_feature_stride,
        input_row_stride,prefixes);
    return cudaGetLastError()==cudaSuccess;
}

bool launch_gdn(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, int num_heads, int num_value_heads,
    int key_dimension, int value_dimension, int sequence_length,
    int real_tokens, int flags, float rms_epsilon, float l2_epsilon,
    float scale, size_t a_row_stride, size_t b_row_stride,
    size_t z_row_stride, void* prefill_scratch,
    size_t prefill_scratch_bytes) {
    const bool normalize_qk = (flags & 1) != 0;
    const bool sigmoid_output_gate = (flags & 2) != 0;
    if (!qkv || !a || !b || !z || !a_log || !dt_bias || !norm_weight ||
        !state || !output || num_heads <= 0 || num_value_heads <= 0 ||
        num_value_heads % num_heads != 0 ||
        key_dimension != gdn_head_dim || value_dimension != gdn_head_dim ||
        sequence_length <= 0)
        return false;
    const int process_length =
        real_tokens > 0 && real_tokens < sequence_length
        ? real_tokens : sequence_length;
    const size_t required_scratch = gdn_prefill_scratch_bytes(
        num_heads, num_value_heads, key_dimension, value_dimension,
        sequence_length);
    // Diagnostic: keep short verification batches on decode arithmetic.
    // Leave longer prompt prefill and the default dispatch unchanged.
    static const bool serial_small_batch = [] {
        const char* value = std::getenv("MOLLM_CUDA_GDN_SMALL_BATCH_SERIAL");
        return value && std::strcmp(value, "1") == 0;
    }();
    const bool use_serial_small_batch =
        serial_small_batch && sequence_length > 1 && sequence_length <= 8;
    // Diagnostic reference: run arbitrary prefill lengths through the original
    // sequential shared-state arithmetic. Default and small-batch policy stay
    // unchanged; this is not a performance selector.
    static const bool serial_prefill = [] {
        const char* value = std::getenv("MOLLM_CUDA_GDN_PREFILL_SERIAL");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!serial_prefill && !use_serial_small_batch && required_scratch > 0 && prefill_scratch &&
        prefill_scratch_bytes >= required_scratch) {
        auto* scratch = static_cast<float*>(prefill_scratch);
        const size_t qk_elements =
            static_cast<size_t>(sequence_length) * num_heads * key_dimension;
        const size_t qk_rows =
            static_cast<size_t>(sequence_length) * num_heads;
        const size_t gate_elements =
            static_cast<size_t>(sequence_length) * num_value_heads;
        float* normalized_query = scratch;
        float* normalized_key = normalized_query + qk_elements;
        float* query_key_dot = normalized_key + qk_elements;
        float* decay = query_key_dot + qk_rows;
        float* beta = decay + gate_elements;
        float* raw_output = beta + gate_elements;
        gdn_prepare_qk_128_cuda<<<sequence_length * num_heads, 128>>>(
            qkv, normalized_query, normalized_key, query_key_dot,
            num_heads, sequence_length, normalize_qk, l2_epsilon);
        constexpr int gate_threads = 256;
        const int gate_count = sequence_length * num_value_heads;
        gdn_prepare_gates_cuda<<<
            (gate_count + gate_threads - 1) / gate_threads, gate_threads>>>(
            a, b, a_log, dt_bias, decay, beta, num_value_heads,
            sequence_length, a_row_stride, b_row_stride);
        const dim3 recurrence_grid(
            (value_dimension + gdn_value_warps_per_block - 1) /
                gdn_value_warps_per_block,
            num_value_heads);
        constexpr int recurrence_threads =
            gdn_value_warps_per_block * gdn_warp_size;
        const auto recurrence_kernel = gdn_recurrence_rows_128_cuda<false,2,false>;
        // Qualified prefill policy: quad values first, paired values as the
        // 8-byte-aligned fallback. Both preserve per-column arithmetic order.
        constexpr bool paired_values_prefill=true;
        constexpr bool quad_values_prefill=true;
        if(quad_values_prefill && sequence_length>=16 &&
            !(reinterpret_cast<uintptr_t>(state)&15))
            gdn_recurrence_quads_128_cuda<<<dim3(16,num_value_heads),recurrence_threads>>>(
                qkv,normalized_query,normalized_key,query_key_dot,decay,beta,state,raw_output,
                num_heads,num_value_heads,sequence_length,process_length,scale);
        else if (paired_values_prefill && sequence_length >= 16 &&
            !(reinterpret_cast<uintptr_t>(state) & 7))
            gdn_recurrence_pairs_128_cuda<<<dim3(32, num_value_heads), recurrence_threads>>>(
                qkv, normalized_query, normalized_key, query_key_dot, decay,
                beta, state, raw_output, num_heads, num_value_heads,
                sequence_length, process_length, scale);
        else
            recurrence_kernel<<<recurrence_grid, recurrence_threads>>>(
                qkv, normalized_query, normalized_key, query_key_dot, decay,
                beta, state, raw_output, num_heads, num_value_heads,
                sequence_length, process_length, scale);
        gdn_post_128_cuda<<<sequence_length * num_value_heads, 128>>>(
            z, norm_weight, raw_output, output, num_value_heads,
            sequence_length, rms_epsilon, z_row_stride, sigmoid_output_gate);
        return true;
    }
    constexpr size_t state_cache_bytes =
        gdn_head_dim * gdn_head_dim * sizeof(float);
    static const bool shared_state_configured = configure_gdn_shared_state<>();
    // Single-token decode keeps the 128-value state in registers.
    if (sequence_length == 1) {
        gdn_128_cuda<true,false,true><<<num_value_heads,gdn_head_dim>>>(
            qkv, a, b, z, a_log, dt_bias, norm_weight, state, output,
            num_heads, num_value_heads, sequence_length, process_length,
            normalize_qk, rms_epsilon, l2_epsilon, scale, a_row_stride,
            b_row_stride, z_row_stride, sigmoid_output_gate);
    } else if (shared_state_configured) {
        gdn_128_cuda<true><<<
            num_value_heads, gdn_head_dim, state_cache_bytes>>>(
            qkv, a, b, z, a_log, dt_bias, norm_weight, state, output,
            num_heads, num_value_heads, sequence_length, process_length,
            normalize_qk, rms_epsilon, l2_epsilon, scale, a_row_stride,
            b_row_stride, z_row_stride, sigmoid_output_gate);
    } else {
        gdn_128_cuda<false><<<num_value_heads, gdn_head_dim>>>(
            qkv, a, b, z, a_log, dt_bias, norm_weight, state, output,
            num_heads, num_value_heads, sequence_length, process_length,
            normalize_qk, rms_epsilon, l2_epsilon, scale, a_row_stride,
            b_row_stride, z_row_stride, sigmoid_output_gate);
    }
    return true;
}

bool launch_gdn_prefixes(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, float* prefixes, int num_heads, int num_value_heads,
    int sequence_length, int real_tokens, int flags, float rms_epsilon,
    float l2_epsilon, float scale, size_t a_row_stride, size_t b_row_stride,
    size_t z_row_stride) {
    if(!qkv||!a||!b||!z||!a_log||!dt_bias||!norm_weight||!state||!output||!prefixes||
        num_heads<=0||num_value_heads<=0||num_value_heads%num_heads||
        sequence_length<1||sequence_length>8||real_tokens<1||real_tokens>sequence_length)
        return false;
    static const bool cached=configure_gdn_shared_state<true>();
    if(cached)
        gdn_128_cuda<true,true><<<num_value_heads,128,128*128*sizeof(float)>>>(
            qkv,a,b,z,a_log,dt_bias,norm_weight,state,output,num_heads,num_value_heads,
            sequence_length,real_tokens,(flags&1)!=0,rms_epsilon,l2_epsilon,scale,
            a_row_stride,b_row_stride,z_row_stride,(flags&2)!=0,prefixes);
    else
        gdn_128_cuda<false,true><<<num_value_heads,128>>>(
            qkv,a,b,z,a_log,dt_bias,norm_weight,state,output,num_heads,num_value_heads,
            sequence_length,real_tokens,(flags&1)!=0,rms_epsilon,l2_epsilon,scale,
            a_row_stride,b_row_stride,z_row_stride,(flags&2)!=0,prefixes);
    return cudaGetLastError()==cudaSuccess;
}

}  // namespace mollm_cuda
