#include "kernels/cuda/attention.h"
#include <cstdio>
#include "kernels/cuda/reduction.cuh"

#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <type_traits>

namespace {

__global__ void sdpa_gemm_bind_heads_cuda(float** pointers, const float* query,
    float* key, float* value, float* scores, float* output,
    mollm_cuda::SdpaGemmPrefillParams p) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= p.heads) return;
    const int kv_head = head / (p.heads / p.kv_heads);
    pointers[head] = key + static_cast<size_t>(kv_head) * p.keys * p.dimension;
    pointers[p.heads + head] = const_cast<float*>(query + static_cast<size_t>(head) * p.query_head_stride);
    pointers[2 * p.heads + head] = scores + static_cast<size_t>(head) * p.queries * p.keys;
    pointers[3 * p.heads + head] = value + static_cast<size_t>(kv_head) * p.keys * p.dimension;
    pointers[4 * p.heads + head] = output + static_cast<size_t>(head) * p.output_head_stride;
}

__global__ void sdpa_expand_fp16_cache_cuda(const __half* key, const __half* value,
    float* key32, float* value32, size_t count, int keys, int capacity, int dimension) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t head_span = static_cast<size_t>(keys) * dimension;
    const size_t source = (index / head_span) * capacity * dimension + index % head_span;
    key32[index] = __half2float(key[source]);
    value32[index] = __half2float(value[source]);
}

__global__ void sdpa_gemm_softmax_cuda(float* scores, const float* mask,
    int queries, int keys, int past, bool causal, float scale,
    size_t mask_column_stride, size_t mask_row_stride) {
    const int query = blockIdx.x;
    float* row = scores + (static_cast<size_t>(blockIdx.y) * queries + query) * keys;
    float maximum = -FLT_MAX;
    for (int k = threadIdx.x; k < keys; k += 256) {
        float score = row[k] * scale;
        if (mask) score += mask[static_cast<size_t>(query) * mask_row_stride + static_cast<size_t>(k) * mask_column_stride];
        else if (causal && k > past + query) score = -FLT_MAX;
        row[k] = score;
        maximum = fmaxf(maximum, score);
    }
    __shared__ float reduction[256];
    maximum = mollm_cuda::detail::block_reduce_max<256>(maximum, reduction);
    float sum = 0.f;
    for (int k = threadIdx.x; k < keys; k += 256) {
        const float value = expf(row[k] - maximum);
        row[k] = value; sum += value;
    }
    sum = mollm_cuda::detail::block_reduce_sum<256>(sum, reduction);
    const float inverse = sum > 0.f ? 1.f / sum : 0.f;
    for (int k = threadIdx.x; k < keys; k += 256) row[k] *= inverse;
}

constexpr int staged_sdpa_head_dim = 128;
constexpr int staged_sdpa_pairs = staged_sdpa_head_dim / 2;
constexpr int staged_sdpa_threads = 256;
constexpr int staged_sdpa_score_lanes_per_key = 4;
constexpr int staged_sdpa_value_pairs_per_block = 8;
constexpr int prefill_sdpa_threads = 256;
static_assert(staged_sdpa_pairs % staged_sdpa_value_pairs_per_block == 0);
static_assert(prefill_sdpa_threads % staged_sdpa_pairs == 0);

__global__ void rope_cuda(
    const float* input, const float* cosine, const float* sine,
    float* output, int feature_dim, int sequence_length, int channels,
    int shape2, int rope_dim, bool interleave,
    size_t x_s0, size_t x_s1, size_t x_s2, size_t x_s3,
    size_t c_s0, size_t c_s1, size_t s_s0, size_t s_s1,
    size_t o_s0, size_t o_s1, size_t o_s2, size_t o_s3) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(feature_dim) *
        sequence_length * channels;
    if (index >= count)
        return;
    size_t remaining = index;
    const int dimension = static_cast<int>(remaining % feature_dim);
    remaining /= feature_dim;
    const int position = static_cast<int>(remaining % sequence_length);
    const int channel = static_cast<int>(remaining / sequence_length);
    const int channel2 = channel % shape2;
    const int channel3 = channel / shape2;
    const size_t input_base = static_cast<size_t>(position) * x_s1 +
        static_cast<size_t>(channel2) * x_s2 +
        static_cast<size_t>(channel3) * x_s3;
    const size_t output_index = static_cast<size_t>(dimension) * o_s0 +
        static_cast<size_t>(position) * o_s1 +
        static_cast<size_t>(channel2) * o_s2 +
        static_cast<size_t>(channel3) * o_s3;
    if (dimension >= rope_dim) {
        output[output_index] =
            input[input_base + static_cast<size_t>(dimension) * x_s0];
        return;
    }

    const int half = rope_dim / 2;
    const int pair = interleave ? dimension / 2 : dimension % half;
    const int first = interleave ? pair * 2 : pair;
    const int second = interleave ? first + 1 : pair + half;
    const float x0 = input[input_base + static_cast<size_t>(first) * x_s0];
    const float x1 = input[input_base + static_cast<size_t>(second) * x_s0];
    const float c = cosine[static_cast<size_t>(position) * c_s1 +
                           static_cast<size_t>(pair) * c_s0];
    const float s = sine[static_cast<size_t>(position) * s_s1 +
                         static_cast<size_t>(pair) * s_s0];
    const bool first_component = interleave
        ? (dimension & 1) == 0 : dimension < half;
    output[output_index] = first_component
        ? x0 * c - x1 * s : x0 * s + x1 * c;
}

__global__ void qk_rms_norm_rope_cuda(
    const float* query, const float* key, const float* query_weight,
    const float* key_weight, const float* cosine, const float* sine,
    float* output, int feature_dim, int sequence_length, int query_heads,
    int total_heads, int rope_dim, bool interleave,
    size_t query_feature_stride, size_t query_row_stride,
    size_t key_feature_stride, size_t key_row_stride,
    size_t cosine_feature_stride, size_t cosine_position_stride,
    size_t sine_feature_stride, size_t sine_position_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride, float epsilon) {
    const int row = static_cast<int>(blockIdx.x);
    const int rows = sequence_length * total_heads;
    if (row >= rows)
        return;

    const int query_rows = sequence_length * query_heads;
    const bool is_query = row < query_rows;
    const int source_row = is_query ? row : row - query_rows;
    const float* source = (is_query ? query : key) +
        static_cast<size_t>(source_row) *
            (is_query ? query_row_stride : key_row_stride);
    const size_t source_feature_stride = is_query
        ? query_feature_stride : key_feature_stride;
    const float* weight = is_query ? query_weight : key_weight;

    float sum = 0.0f;
    for (int dimension = threadIdx.x; dimension < feature_dim;
         dimension += blockDim.x) {
        const float value = source[
            static_cast<size_t>(dimension) * source_feature_stride];
        sum += value * value;
    }
    __shared__ float reduction[256];
    const float block_sum =
        mollm_cuda::detail::block_reduce_sum<256>(sum, reduction);
    const float inverse = rsqrtf(block_sum / feature_dim + epsilon);

    const int position = source_row % sequence_length;
    const int head = row / sequence_length;
    float* destination = output +
        static_cast<size_t>(position) * output_position_stride +
        static_cast<size_t>(head) * output_head_stride;
    const int half = rope_dim / 2;
    for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
        const int first = interleave ? pair * 2 : pair;
        const int second = interleave ? first + 1 : pair + half;
        const float x0 = source[
            static_cast<size_t>(first) * source_feature_stride] * inverse *
            weight[first];
        const float x1 = source[
            static_cast<size_t>(second) * source_feature_stride] * inverse *
            weight[second];
        const float c = cosine[
            static_cast<size_t>(pair) * cosine_feature_stride +
            static_cast<size_t>(position) * cosine_position_stride];
        const float s = sine[
            static_cast<size_t>(pair) * sine_feature_stride +
            static_cast<size_t>(position) * sine_position_stride];
        destination[static_cast<size_t>(first) * output_feature_stride] =
            x0 * c - x1 * s;
        destination[static_cast<size_t>(second) * output_feature_stride] =
            x1 * c + x0 * s;
    }
    for (int dimension = rope_dim + threadIdx.x; dimension < feature_dim;
         dimension += blockDim.x) {
        destination[static_cast<size_t>(dimension) * output_feature_stride] =
            source[static_cast<size_t>(dimension) * source_feature_stride] *
            inverse * weight[dimension];
    }
}

__device__ float load_kv_cache_value(
    const void* cache, size_t index, bool fp16_cache) {
    return fp16_cache
        ? __half2float(static_cast<const __half*>(cache)[index])
        : static_cast<const float*>(cache)[index];
}

__device__ void store_kv_cache_value(
    void* cache, size_t index, float value, bool fp16_cache) {
    if (fp16_cache)
        static_cast<__half*>(cache)[index] = __float2half_rn(value);
    else
        static_cast<float*>(cache)[index] = value;
}

template<bool Pipeline = false>
__device__ __forceinline__ float query_key_dot(
    const float* query, const void* key, bool cached, bool fp16_cache,
    int key_dim, size_t cached_key_base, size_t current_key_base,
    size_t query_feature_stride, size_t key_feature_stride) {
    float dot = 0.0f;
    const bool half2_aligned =
        (reinterpret_cast<uintptr_t>(query) & (alignof(float2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(key) & (alignof(__half2) - 1)) == 0;
    if (cached && fp16_cache && query_feature_stride == 1 &&
        key_dim % 2 == 0 && half2_aligned) {
        const auto* query2 = reinterpret_cast<const float2*>(query);
        const auto* key2 = reinterpret_cast<const __half2*>(key) +
            cached_key_base / 2;
        if constexpr (Pipeline) {
            if (key_dim <= 0) return dot;
            float2 q = query2[0], k = __half22float2(key2[0]);
            for (int pair = 0; pair < key_dim / 2; ++pair) {
                float2 next_q{}, next_k{};
                if (pair + 1 < key_dim / 2) {
                    next_q = query2[pair + 1];
                    next_k = __half22float2(key2[pair + 1]);
                }
                dot = fmaf(q.x, k.x, dot);
                dot = fmaf(q.y, k.y, dot);
                q = next_q; k = next_k;
            }
            return dot;
        }
        for (int pair = 0; pair < key_dim / 2; ++pair) {
            const float2 q = query2[pair];
            const float2 k = __half22float2(key2[pair]);
            dot = fmaf(q.x, k.x, dot);
            dot = fmaf(q.y, k.y, dot);
        }
        return dot;
    }
    for (int dimension = 0; dimension < key_dim; ++dimension) {
        const float key_element = cached
            ? load_kv_cache_value(
                  key, cached_key_base + dimension, fp16_cache)
            : static_cast<const float*>(key)[
                  current_key_base + static_cast<size_t>(dimension) *
                      key_feature_stride];
        dot += query[static_cast<size_t>(dimension) *
                     query_feature_stride] * key_element;
    }
    return dot;
}

__device__ __forceinline__ float attention_value_dot(
    const float* probabilities, const void* value, int key_head,
    int key_length, int value_dim, int dimension, int first_position,
    int position_stride, int value_capacity, bool cached, bool fp16_cache,
    size_t value_feature_stride, size_t value_position_stride,
    size_t value_head_stride) {
    float result = 0.0f;
    for (int position = first_position; position < key_length;
         position += position_stride) {
        const size_t cached_index =
            (static_cast<size_t>(key_head) * value_capacity + position) *
                value_dim + dimension;
        const size_t current_index =
            static_cast<size_t>(key_head) * value_head_stride +
            static_cast<size_t>(position) * value_position_stride +
            static_cast<size_t>(dimension) * value_feature_stride;
        const float value_element = cached
            ? load_kv_cache_value(value, cached_index, fp16_cache)
            : static_cast<const float*>(value)[current_index];
        result = fmaf(probabilities[position], value_element, result);
    }
    return result;
}

__device__ __forceinline__ float2 attention_value_dot_fp16_pair(
    const float* probabilities, const void* value, int key_head,
    int key_length, int value_dim, int dimension_pair, int first_position,
    int position_stride, int value_capacity) {
    float2 result{0.0f, 0.0f};
    for (int position = first_position; position < key_length;
         position += position_stride) {
        const size_t index =
            (static_cast<size_t>(key_head) * value_capacity + position) *
                value_dim + dimension_pair * 2;
        const float2 value_pair = __half22float2(
            reinterpret_cast<const __half2*>(value)[index / 2]);
        const float probability = probabilities[position];
        result.x = fmaf(probability, value_pair.x, result.x);
        result.y = fmaf(probability, value_pair.y, result.y);
    }
    return result;
}

__global__ void append_kv_cuda(
    const float* key, const float* value, void* key_cache,
    void* value_cache, bool fp16_cache, int num_kv_heads,
    int current_length, int past_length, int max_length,
    int key_dim, int value_dim,
    size_t key_position_stride, size_t key_head_stride,
    size_t value_position_stride, size_t value_head_stride) {
    const int maximum_dim = max(key_dim, value_dim);
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(num_kv_heads) *
        current_length * maximum_dim;
    if (index >= count)
        return;
    size_t remaining = index;
    const int dimension = static_cast<int>(remaining % maximum_dim);
    remaining /= maximum_dim;
    const int position = static_cast<int>(remaining % current_length);
    const int head = static_cast<int>(remaining / current_length);
    if (dimension < key_dim) {
        const size_t destination =
            (static_cast<size_t>(head) * max_length + past_length +
             position) * key_dim + dimension;
        store_kv_cache_value(
            key_cache, destination,
            key[static_cast<size_t>(head) * key_head_stride +
                static_cast<size_t>(position) * key_position_stride +
                dimension],
            fp16_cache);
    }
    if (dimension < value_dim) {
        const size_t destination =
            (static_cast<size_t>(head) * max_length + past_length +
             position) * value_dim + dimension;
        store_kv_cache_value(
            value_cache, destination,
            value[static_cast<size_t>(head) * value_head_stride +
                  static_cast<size_t>(position) * value_position_stride +
                  dimension],
            fp16_cache);
    }
}

__global__ void sdpa_scores_cuda(
    const float* query, const void* key, float* scores,
    const float* mask, int num_heads, int num_kv_heads,
    int query_length, int key_length, int past_length, int key_dim,
    int key_capacity, bool cached, bool fp16_cache, bool causal, float scale,
    size_t query_feature_stride, size_t query_position_stride,
    size_t query_head_stride, size_t key_feature_stride,
    size_t key_position_stride, size_t key_head_stride,
    size_t mask_column_stride, size_t mask_row_stride) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(num_heads) * query_length *
        key_length;
    if (index >= count)
        return;
    size_t remaining = index;
    const int key_position = static_cast<int>(remaining % key_length);
    remaining /= key_length;
    const int query_position = static_cast<int>(remaining % query_length);
    const int head = static_cast<int>(remaining / query_length);
    const int key_head = head / (num_heads / num_kv_heads);
    const float* query_row = query +
        static_cast<size_t>(head) * query_head_stride +
        static_cast<size_t>(query_position) * query_position_stride;
    const size_t cached_key_base =
        (static_cast<size_t>(key_head) * key_capacity + key_position) *
        key_dim;
    const size_t current_key_base =
        static_cast<size_t>(key_head) * key_head_stride +
        static_cast<size_t>(key_position) * key_position_stride;
    const float dot = query_key_dot(
        query_row, key, cached, fp16_cache, key_dim, cached_key_base,
        current_key_base, query_feature_stride, key_feature_stride);
    float score = dot * scale;
    if (mask) {
        score += mask[static_cast<size_t>(query_position) * mask_row_stride +
                      static_cast<size_t>(key_position) *
                          mask_column_stride];
    } else if (causal && key_position > past_length + query_position) {
        score = -FLT_MAX;
    }
    scores[index] = score;
}

__global__ void sdpa_output_cuda(
    const float* scores, const void* value, float* output,
    int num_heads, int num_kv_heads, int query_length, int key_length,
    int value_dim, int value_capacity, bool cached, bool fp16_cache,
    size_t value_feature_stride, size_t value_position_stride,
    size_t value_head_stride, size_t output_feature_stride,
    size_t output_position_stride, size_t output_head_stride) {
    const int row = blockIdx.x;
    const int head = row / query_length;
    const int query_position = row % query_length;
    if (head >= num_heads)
        return;
    const int key_head = head / (num_heads / num_kv_heads);
    const float* score_row = scores + static_cast<size_t>(row) * key_length;
    __shared__ float reduction[256];
    float local_maximum = -FLT_MAX;
    for (int position = threadIdx.x; position < key_length;
         position += blockDim.x)
        local_maximum = fmaxf(local_maximum, score_row[position]);
    const float maximum =
        mollm_cuda::detail::block_reduce_max<256>(
            local_maximum, reduction);
    float local_sum = 0.0f;
    for (int position = threadIdx.x; position < key_length;
         position += blockDim.x)
        local_sum += expf(score_row[position] - maximum);
    const float block_sum =
        mollm_cuda::detail::block_reduce_sum<256>(local_sum, reduction);
    const float inverse_sum = block_sum > 0.0f
        ? 1.0f / block_sum : 0.0f;
    for (int dimension = threadIdx.x; dimension < value_dim;
         dimension += blockDim.x) {
        float result = 0.0f;
        for (int position = 0; position < key_length; ++position) {
            const float probability =
                expf(score_row[position] - maximum) * inverse_sum;
            const size_t cached_index =
                (static_cast<size_t>(key_head) * value_capacity + position) *
                    value_dim + dimension;
            const auto* current_value = static_cast<const float*>(value);
            const size_t current_index =
                static_cast<size_t>(key_head) * value_head_stride +
                static_cast<size_t>(position) * value_position_stride +
                static_cast<size_t>(dimension) * value_feature_stride;
            const float value_element = cached
                ? load_kv_cache_value(value, cached_index, fp16_cache)
                : current_value[current_index];
            result += probability * value_element;
        }
        output[static_cast<size_t>(head) * output_head_stride +
               static_cast<size_t>(query_position) *
                   output_position_stride +
               static_cast<size_t>(dimension) * output_feature_stride] =
            result;
    }
}

// Decode has one query position, so the score and softmax/value passes can be
// owned by one block per query head. Short contexts keep the temporary score
// row in shared memory; long contexts retain the backend scratch buffer to
// avoid a context-length shared-memory limit. A single launch removes the
// inter-kernel boundary, and softmax exponentials are computed once per key
// instead of once per output dimension.
template <int Threads, bool SharedScores, bool PipelineQK = false>
__global__ void sdpa_decode_cuda(
    const float* query, const void* key, const void* value, float* scores,
    float* output, const float* mask, int num_heads, int num_kv_heads,
    int key_length, int past_length, int key_dim, int value_dim,
    int key_capacity, bool cached, bool fp16_cache, bool causal, float scale,
    size_t query_feature_stride, size_t query_head_stride,
    size_t key_feature_stride, size_t key_position_stride,
    size_t key_head_stride, size_t value_feature_stride,
    size_t value_position_stride, size_t value_head_stride,
    size_t mask_column_stride, size_t output_feature_stride,
    size_t output_head_stride) {
    const int head = blockIdx.x;
    if (head >= num_heads)
        return;
    const int key_head = head / (num_heads / num_kv_heads);
    const float* query_row = query +
        static_cast<size_t>(head) * query_head_stride;
    // The 512-thread launch guarantees key_length <= Threads, so its complete
    // score row fits on chip. Long-context instances retain the global row.
    __shared__ float score_cache[SharedScores ? Threads : 1];
    float* score_row;
    if constexpr (SharedScores)
        score_row = score_cache;
    else
        score_row = scores + static_cast<size_t>(head) * key_length;
    __shared__ float reduction[Threads];
    __shared__ float2 pair_reduction[Threads];

    float local_maximum = -FLT_MAX;
    constexpr int cooperative_lanes_per_key = Threads == 512 ? 2 : 4;
    const bool cooperative_shape =
        (Threads == 512 && key_dim == 256) ||
        (Threads == 1024 && key_dim == 128);
    const bool cooperative_qk = cooperative_shape && cached && fp16_cache &&
        query_feature_stride == 1 &&
        (reinterpret_cast<uintptr_t>(query_row) &
         (alignof(float2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(key) &
         (alignof(__half2) - 1)) == 0;
    if (cooperative_qk) {
        // Two or four adjacent lanes cooperate on one key. Their half2 reads
        // cover a contiguous span instead of unrelated cache rows while still
        // exposing 256 independent keys per block.
        constexpr int lanes_per_key = cooperative_lanes_per_key;
        const int lane = threadIdx.x % lanes_per_key;
        const int first_key = threadIdx.x / lanes_per_key;
        const auto* query2 = reinterpret_cast<const float2*>(query_row);
        const auto* key2 = reinterpret_cast<const __half2*>(key);
        for (int key_position = first_key; key_position < key_length;
             key_position += Threads / lanes_per_key) {
            const size_t cached_key_base =
                (static_cast<size_t>(key_head) * key_capacity +
                 key_position) * key_dim;
            float dot = 0.0f;
            for (int pair = lane; pair < key_dim / 2;
                 pair += lanes_per_key) {
                const float2 q = query2[pair];
                const float2 k = __half22float2(
                    key2[cached_key_base / 2 + pair]);
                dot = fmaf(q.x, k.x, dot);
                dot = fmaf(q.y, k.y, dot);
            }
            const unsigned active = __activemask();
            for (int offset = lanes_per_key / 2; offset > 0; offset /= 2)
                dot += __shfl_down_sync(
                    active, dot, offset, lanes_per_key);
            if (lane == 0) {
                float score = dot * scale;
                if (mask)
                    score += mask[static_cast<size_t>(key_position) *
                                  mask_column_stride];
                else if (causal && key_position > past_length)
                    score = -FLT_MAX;
                score_row[key_position] = score;
                local_maximum = fmaxf(local_maximum, score);
            }
        }
    } else {
        for (int key_position = threadIdx.x; key_position < key_length;
             key_position += blockDim.x) {
            const size_t cached_key_base =
                (static_cast<size_t>(key_head) * key_capacity + key_position) *
                key_dim;
            const size_t current_key_base =
                static_cast<size_t>(key_head) * key_head_stride +
                static_cast<size_t>(key_position) * key_position_stride;
            const float dot = query_key_dot<PipelineQK>(
                query_row, key, cached, fp16_cache, key_dim, cached_key_base,
                current_key_base, query_feature_stride, key_feature_stride);
            float score = dot * scale;
            if (mask)
                score += mask[static_cast<size_t>(key_position) *
                              mask_column_stride];
            else if (causal && key_position > past_length)
                score = -FLT_MAX;
            score_row[key_position] = score;
            local_maximum = fmaxf(local_maximum, score);
        }
    }
    const float maximum =
        mollm_cuda::detail::block_reduce_max<Threads>(
            local_maximum, reduction);

    float local_sum = 0.0f;
    for (int key_position = threadIdx.x; key_position < key_length;
         key_position += blockDim.x) {
        const float numerator = expf(score_row[key_position] - maximum);
        score_row[key_position] = numerator;
        local_sum += numerator;
    }
    const float block_sum =
        mollm_cuda::detail::block_reduce_sum<Threads>(local_sum, reduction);
    const float inverse_sum = block_sum > 0.0f
        ? 1.0f / block_sum : 0.0f;

    // Normalize each score once. The value pass reuses every probability for
    // all output dimensions, so applying inverse_sum there would repeat the
    // same multiply value_dim times.
    for (int key_position = threadIdx.x; key_position < key_length;
         key_position += blockDim.x)
        score_row[key_position] *= inverse_sum;
    __syncthreads();

    // Pairing adjacent FP16 dimensions lets a full block split the position
    // axis more finely without leaving threads idle: eight groups in the
    // 512-thread kernel and sixteen in the 1024-thread specialization.
    const bool vector_value = cached && fp16_cache && value_dim > 0 &&
        value_dim % 2 == 0 && value_dim / 2 <= Threads &&
        output_feature_stride == 1 &&
        (reinterpret_cast<uintptr_t>(value) &
         (alignof(__half2) - 1)) == 0;
    if (vector_value) {
        const int pairs = value_dim / 2;
        constexpr int max_pair_groups = Threads == 512 ? 8 : 16;
        const int output_groups = pairs > 0
            ? min(max_pair_groups, Threads / pairs) : 0;
        const int output_threads = output_groups * pairs;
        if (threadIdx.x < output_threads) {
            const int pair = threadIdx.x % pairs;
            const int group = threadIdx.x / pairs;
            pair_reduction[threadIdx.x] = attention_value_dot_fp16_pair(
                score_row, value, key_head, key_length, value_dim, pair,
                group, output_groups, key_capacity);
        }
        __syncthreads();
        if (threadIdx.x < pairs) {
            float2 result = pair_reduction[threadIdx.x];
            for (int group = 1; group < output_groups; ++group) {
                const float2 partial =
                    pair_reduction[group * pairs + threadIdx.x];
                result.x += partial.x;
                result.y += partial.y;
            }
            const size_t output_base =
                static_cast<size_t>(head) * output_head_stride +
                static_cast<size_t>(threadIdx.x) * 2;
            output[output_base] = result.x;
            output[output_base + 1] = result.y;
        }
    } else {
        // Split the value accumulation across otherwise idle threads. Decode
        // commonly has value_dim << Threads (128 vs. 512/1024), so assigning
        // one thread per dimension leaves most of the block unused.
        const int output_groups = value_dim > 0
            ? min(8, Threads / value_dim) : 0;
        const int output_threads = output_groups * value_dim;
        if (output_groups > 1 && threadIdx.x < output_threads) {
            const int dimension = threadIdx.x % value_dim;
            const int group = threadIdx.x / value_dim;
            reduction[threadIdx.x] = attention_value_dot(
                score_row, value, key_head, key_length, value_dim, dimension,
                group, output_groups, key_capacity, cached, fp16_cache,
                value_feature_stride, value_position_stride,
                value_head_stride);
        }
        __syncthreads();
        if (output_groups > 1) {
            if (threadIdx.x < value_dim) {
                float result = reduction[threadIdx.x];
                for (int group = 1; group < output_groups; ++group)
                    result += reduction[group * value_dim + threadIdx.x];
                output[static_cast<size_t>(head) * output_head_stride +
                       static_cast<size_t>(threadIdx.x) *
                           output_feature_stride] = result;
            }
        } else {
            for (int dimension = threadIdx.x; dimension < value_dim;
                 dimension += blockDim.x) {
                const float result = attention_value_dot(
                    score_row, value, key_head, key_length, value_dim,
                    dimension, 0, 1, key_capacity, cached, fp16_cache,
                    value_feature_stride, value_position_stride,
                    value_head_stride);
                output[static_cast<size_t>(head) * output_head_stride +
                       static_cast<size_t>(dimension) *
                           output_feature_stride] = result;
            }
        }
    }
}

// FP16 decode benefits from exposing work from the key and value axes as
// independent blocks. A single fused block per query head leaves most SMs idle
// on large GPUs; these three stages keep the same SDPA semantics while
// increasing grid-level parallelism for the common 128-wide Qwen layout.
template <bool AppendCurrent>
__global__ void sdpa_decode_scores_fp16_cuda(
    const float* query, const float* current_key,
    const float* current_value, const __half* key, __half* key_append,
    __half* value_append, float* scores,
    const float* mask, int num_heads, int num_kv_heads, int key_length,
    int past_length, int key_capacity, bool causal, float scale,
    size_t query_head_stride, size_t current_key_head_stride,
    size_t current_value_head_stride, size_t mask_column_stride) {
    constexpr int lanes_per_key = staged_sdpa_score_lanes_per_key;
    constexpr int keys_per_block = staged_sdpa_threads / lanes_per_key;
    const int head = static_cast<int>(blockIdx.x);
    const int key_head = head / (num_heads / num_kv_heads);
    const int heads_per_key = num_heads / num_kv_heads;
    const int lane = static_cast<int>(threadIdx.x) % lanes_per_key;
    const int key_position = static_cast<int>(blockIdx.y) * keys_per_block +
        static_cast<int>(threadIdx.x) / lanes_per_key;

    // The score launch is already ordered before softmax/value. Let one block
    // for each KV head append the one-token K/V payload while every score
    // block reads the current K directly. This removes the separate append
    // launch without introducing an inter-block synchronization dependency.
    if constexpr (AppendCurrent) {
        if (blockIdx.y == 0 && head % heads_per_key == 0) {
            const size_t cache_pair_base =
                (static_cast<size_t>(key_head) * key_capacity + past_length) *
                staged_sdpa_pairs;
            auto* key_cache2 =
                reinterpret_cast<__half2*>(key_append) + cache_pair_base;
            auto* value_cache2 =
                reinterpret_cast<__half2*>(value_append) + cache_pair_base;
            const auto* current_key2 = reinterpret_cast<const float2*>(
                current_key + static_cast<size_t>(key_head) *
                    current_key_head_stride);
            const auto* current_value2 = reinterpret_cast<const float2*>(
                current_value + static_cast<size_t>(key_head) *
                    current_value_head_stride);
            for (int pair = threadIdx.x; pair < staged_sdpa_pairs;
                 pair += staged_sdpa_threads) {
                const float2 current_k = current_key2[pair];
                const float2 current_v = current_value2[pair];
                key_cache2[pair] =
                    __floats2half2_rn(current_k.x, current_k.y);
                value_cache2[pair] =
                    __floats2half2_rn(current_v.x, current_v.y);
            }
        }
    }
    if (key_position >= key_length)
        return;

    const auto* query2 = reinterpret_cast<const float2*>(
        query + static_cast<size_t>(head) * query_head_stride);
    const size_t key_base =
        (static_cast<size_t>(key_head) * key_capacity + key_position) *
        staged_sdpa_head_dim;
    const auto* key2 = reinterpret_cast<const __half2*>(key) + key_base / 2;
    float dot = 0.0f;
#pragma unroll
    for (int pair = lane; pair < staged_sdpa_pairs;
         pair += lanes_per_key) {
        const float2 q = query2[pair];
        float2 k;
        if constexpr (AppendCurrent) {
            if (key_position == past_length) {
                const float2 current = reinterpret_cast<const float2*>(
                    current_key + static_cast<size_t>(key_head) *
                        current_key_head_stride)[pair];
                k = __half22float2(
                    __floats2half2_rn(current.x, current.y));
            } else {
                k = __half22float2(key2[pair]);
            }
        } else {
            k = __half22float2(key2[pair]);
        }
        dot = fmaf(q.x, k.x, dot);
        dot = fmaf(q.y, k.y, dot);
    }
    const unsigned active = __activemask();
#pragma unroll
    for (int offset = lanes_per_key / 2; offset > 0; offset /= 2)
        dot += __shfl_down_sync(active, dot, offset, lanes_per_key);
    if (lane != 0)
        return;
    float score = dot * scale;
    if (mask)
        score += mask[static_cast<size_t>(key_position) *
                      mask_column_stride];
    else if (causal && key_position > past_length)
        score = -FLT_MAX;
    scores[static_cast<size_t>(head) * key_length + key_position] = score;
}

template <int Threads>
__global__ void sdpa_decode_softmax_cuda(float* scores, int key_length) {
    const int head = static_cast<int>(blockIdx.x);
    float* row = scores + static_cast<size_t>(head) * key_length;
    __shared__ float reduction[Threads];
    float local_maximum = -FLT_MAX;
    for (int position = threadIdx.x; position < key_length;
         position += Threads)
        local_maximum = fmaxf(local_maximum, row[position]);
    const float maximum = mollm_cuda::detail::block_reduce_max<Threads>(
        local_maximum, reduction);
    float local_sum = 0.0f;
    for (int position = threadIdx.x; position < key_length;
         position += Threads) {
        const float numerator = expf(row[position] - maximum);
        row[position] = numerator;
        local_sum += numerator;
    }
    const float sum = mollm_cuda::detail::block_reduce_sum<Threads>(
        local_sum, reduction);
    const float inverse_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int position = threadIdx.x; position < key_length;
         position += Threads)
        row[position] *= inverse_sum;
}

__global__ void sdpa_decode_value_fp16_cuda(
    const float* probabilities, const __half* value, float* output,
    int num_heads, int num_kv_heads, int key_length, int value_capacity,
    size_t output_head_stride) {
    constexpr int pairs_per_block = staged_sdpa_value_pairs_per_block;
    constexpr int position_lanes = staged_sdpa_threads / pairs_per_block;
    const int head = static_cast<int>(blockIdx.x);
    const int key_head = head / (num_heads / num_kv_heads);
    const int local_pair =
        static_cast<int>(threadIdx.x) % pairs_per_block;
    const int pair = static_cast<int>(blockIdx.y) * pairs_per_block +
        local_pair;
    const int position_lane =
        static_cast<int>(threadIdx.x) / pairs_per_block;
    const float* score_row =
        probabilities + static_cast<size_t>(head) * key_length;
    const auto* value2 = reinterpret_cast<const __half2*>(value) +
        static_cast<size_t>(key_head) * value_capacity * staged_sdpa_pairs;
    float2 sum{0.0f, 0.0f};
    for (int position = position_lane; position < key_length;
         position += position_lanes) {
        const float2 element = __half22float2(
            value2[static_cast<size_t>(position) * staged_sdpa_pairs + pair]);
        const float probability = score_row[position];
        sum.x = fmaf(probability, element.x, sum.x);
        sum.y = fmaf(probability, element.y, sum.y);
    }
    // A warp contains four position lanes for each of its eight adjacent
    // value pairs. Fold those four first, then combine the eight warp totals
    // through a compact shared-memory slab.
    const unsigned active = __activemask();
    sum.x += __shfl_down_sync(active, sum.x, 16);
    sum.y += __shfl_down_sync(active, sum.y, 16);
    sum.x += __shfl_down_sync(active, sum.x, 8);
    sum.y += __shfl_down_sync(active, sum.y, 8);
    constexpr int warps_per_block = staged_sdpa_threads / 32;
    __shared__ float2 partial[warps_per_block * pairs_per_block];
    const int position_lane_in_warp = position_lane % 4;
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    if (position_lane_in_warp == 0)
        partial[warp * pairs_per_block + local_pair] = sum;
    __syncthreads();
    if (threadIdx.x < pairs_per_block) {
        float2 result = partial[local_pair];
#pragma unroll
        for (int lane = 1; lane < warps_per_block; ++lane) {
            const float2 next =
                partial[lane * pairs_per_block + local_pair];
            result.x += next.x;
            result.y += next.y;
        }
        const size_t output_base =
            static_cast<size_t>(head) * output_head_stride + pair * 2;
        output[output_base] = result.x;
        output[output_base + 1] = result.y;
    }
}

template<bool ValidateV = false,bool ExactMask = false>
__global__ void validate_sdpa_prefix_cuda(const float* query,const __half* key,
    const __half* value,const float* mask,uint32_t* valid,mollm_cuda::SdpaGemmPrefillParams p) {
    const size_t qc=size_t(p.heads)*p.queries*256;
    const size_t kc=size_t(p.kv_heads)*p.keys*256;
    const size_t mc=size_t(p.queries)*p.keys;
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<qc+kc+mc+(ValidateV?kc:0);
         i+=size_t(gridDim.x)*blockDim.x) {
        bool ok;
        if (i<qc) {
            const size_t row=i/256,dim=i%256;
            const float q=query[(row/p.queries)*p.query_head_stride+
                (row%p.queries)*p.query_position_stride+dim];
            ok=isfinite(q) && fabsf(q)<=1e20f;
        } else if (i<qc+kc) {
            const size_t j=i-qc,row=j/256;
            const float k=__half2float(key[(row/p.keys)*size_t(p.capacity)*256+
                (row%p.keys)*256+j%256]);
            ok=isfinite(k);
        } else if (i<qc+kc+mc) {
            const size_t j=i-qc-kc,row=j/p.keys,col=j%p.keys;
            const float expected=col<=size_t(p.past)+row?0.f:-1e38f;
            const float actual=mask[row*p.mask_row_stride+col*p.mask_column_stride];
            if constexpr(ExactMask)ok=__float_as_uint(actual)==__float_as_uint(expected);
            else ok=actual==expected;
        } else {
            const size_t j=i-qc-kc-mc,row=j/256;
            ok=isfinite(__half2float(value[(row/p.keys)*size_t(p.capacity)*256+
                (row%p.keys)*256+j%256]));
        }
        if (!ok) atomicExch(valid,0u);
    }
}



// Warp-head layout: retain each head's original256-thread reduction
// tree while eight-lane groups of different heads share a physical warp.
template<bool Sum>
__device__ __forceinline__ float sdpa_head_lane_reduce(float value,float* reduction,int thread) {
    const auto combine=[] (float a,float b) {return Sum?a+b:fmaxf(a,b);};
    reduction[thread]=value;
    __syncthreads();
#pragma unroll
    for(int offset=128;offset>=64;offset>>=1) {
        if(thread<offset)reduction[thread]=combine(reduction[thread],reduction[thread+offset]);
        __syncthreads();
    }
    if(thread<32)reduction[thread]=combine(reduction[thread],reduction[thread+32]);
    __syncthreads();
    if(thread<8) {
        // Original warp offsets16 and8 cross physical warps after remapping.
        const float low=combine(reduction[thread],reduction[thread+16]);
        const float high=combine(reduction[thread+8],reduction[thread+24]);
        value=combine(low,high);
        const unsigned active=__activemask();
#pragma unroll
        for(int offset=4;offset>0;offset>>=1)
            value=combine(value,__shfl_down_sync(active,value,offset,8));
        if(thread==0)reduction[0]=value;
    }
    __syncthreads();
    const float result=reduction[0];
    __syncthreads();
    return result;
}

template <int HeadDim, bool SharedScores, bool SkipCausalQK = false, bool ValidatedPrefix = false, bool SkipZeroPV = false, bool UnrollPV = false, bool PairPV = false, bool ProbabilitiesOnly = false, int HeadLanes = 1, bool KnownMask = false>
__global__ void sdpa_prefill_fp16_cuda(
    const float* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, float* __restrict__ scores,
    float* __restrict__ output, const float* __restrict__ mask, int num_heads,
    int num_kv_heads, int query_length, int key_length, int past_length,
    int key_capacity, bool causal, float scale,
    size_t query_position_stride, size_t query_head_stride,
    size_t mask_column_stride, size_t mask_row_stride,
    size_t output_position_stride, size_t output_head_stride,
    const uint32_t* valid = nullptr) {
    static_assert(HeadDim == 128 || HeadDim == 256);
    static_assert(!ProbabilitiesOnly || (ValidatedPrefix && !SharedScores));
    static_assert(HeadLanes==1||HeadLanes==2);
    static_assert(HeadLanes==1||(HeadDim==256&&ProbabilitiesOnly&&PairPV));
    static_assert(!KnownMask||ProbabilitiesOnly);
    constexpr int pairs = HeadDim / 2;
    // Keep sixteen FP16 pairs per lane for both supported widths. This exposes
    // enough keys per block while preserving a fixed reduction tree.
    constexpr int lanes_per_key =
        HeadDim == 256 ? 8 : staged_sdpa_score_lanes_per_key;
    constexpr int keys_per_iteration = prefill_sdpa_threads / lanes_per_key;
    const int query_position = static_cast<int>(blockIdx.x);
    const int local_head=HeadLanes==1?0:(threadIdx.x/8)%HeadLanes;
    const int thread=HeadLanes==1?int(threadIdx.x):int(threadIdx.x/(8*HeadLanes))*8+int(threadIdx.x%8);
    const int head = static_cast<int>(blockIdx.y)*HeadLanes+local_head;
    const int key_head = head / (num_heads / num_kv_heads);
    const int lane = thread % lanes_per_key;
    const int first_key = thread / lanes_per_key;
    const auto* query2 = reinterpret_cast<const float2*>(
        query + static_cast<size_t>(head) * query_head_stride +
        static_cast<size_t>(query_position) * query_position_stride);
    const auto* key2 = reinterpret_cast<const __half2*>(key) +
        static_cast<size_t>(key_head) * key_capacity * pairs;
    extern __shared__ float shared_scores[];
    float* score_row = SharedScores
        ? shared_scores
        : scores +
            (static_cast<size_t>(head) * query_length + query_position) *
                key_length;
    float local_maximum = -FLT_MAX;
    const bool validated=ValidatedPrefix && valid && *valid;
    const int dot_key_length = validated || (SkipCausalQK && !mask && causal)
        ? min(key_length, past_length + query_position + 1) : key_length;
    int key_position = first_key;
    for (; key_position < dot_key_length;
         key_position += keys_per_iteration) {
        float dot = 0.0f;
#pragma unroll
        for (int pair = lane; pair < pairs;
             pair += lanes_per_key) {
            const float2 q = query2[pair];
            const float2 k = __half22float2(
                key2[static_cast<size_t>(key_position) * pairs +
                     pair]);
            dot = fmaf(q.x, k.x, dot);
            dot = fmaf(q.y, k.y, dot);
        }
        const unsigned active = __activemask();
#pragma unroll
        for (int offset = lanes_per_key / 2; offset > 0; offset /= 2)
            dot += __shfl_down_sync(active, dot, offset, lanes_per_key);
        if (lane == 0) {
            float score = dot * scale;
            if (mask) {
                // ExactMask validation distinguishes +0 from -0. Preserve
                // the original positive-zero addition, not just its value:
                // removing it could change a negative-zero score.
                if constexpr(KnownMask) {
                    if(validated)score=__fadd_rn(score,0.f);
                    else score+=mask[size_t(query_position)*mask_row_stride+size_t(key_position)*mask_column_stride];
                } else score += mask[size_t(query_position)*mask_row_stride+size_t(key_position)*mask_column_stride];
            }
            else if (causal &&
                     key_position > past_length + query_position)
                score = -FLT_MAX;
            score_row[key_position] = score;
            local_maximum = fmaxf(local_maximum, score);
        }
    }
    if constexpr (SkipCausalQK || ValidatedPrefix) {
        if (lane == 0)
            for (; key_position < key_length; key_position += keys_per_iteration)
                score_row[key_position] = validated ? -1e38f : -FLT_MAX;
    }

    __shared__ float reductions[prefill_sdpa_threads*HeadLanes];
    float* reduction=reductions+local_head*prefill_sdpa_threads;
    const float maximum = [&] {
        if constexpr(HeadLanes==1)return mollm_cuda::detail::block_reduce_max<prefill_sdpa_threads>(local_maximum,reduction);
        else return sdpa_head_lane_reduce<false>(local_maximum,reduction,thread);
    }();
    float local_sum = 0.0f;
    for (int key_position = thread; key_position < key_length;
         key_position += prefill_sdpa_threads) {
        const float numerator = expf(score_row[key_position] - maximum);
        score_row[key_position] = numerator;
        local_sum += numerator;
    }
    const float block_sum = [&] {
        if constexpr(HeadLanes==1)return mollm_cuda::detail::block_reduce_sum<prefill_sdpa_threads>(local_sum,reduction);
        else return sdpa_head_lane_reduce<true>(local_sum,reduction,thread);
    }();
    const float inverse_sum = block_sum > 0.0f
        ? 1.0f / block_sum : 0.0f;
    for (int key_position = thread; key_position < key_length;
         key_position += prefill_sdpa_threads)
        score_row[key_position] *= inverse_sum;
    __syncthreads();

    if constexpr(ProbabilitiesOnly)if(validated)return;
    if constexpr (HeadDim == 256 && PairPV) {
        // Two independent ordered FMA chains per active lane.
        // QK and softmax retain their original 256-thread reduction trees.
        if (thread < HeadDim / 2) {
            const int pair = thread;
            const int pv_length = SkipZeroPV && validated ? dot_key_length : key_length;
            const auto* value2 = reinterpret_cast<const __half2*>(value) +
                size_t(key_head) * key_capacity * pairs;
            float2 result{0.f, 0.f};
            for (int position = 0; position < pv_length; ++position) {
                const float2 v = __half22float2(value2[size_t(position) * pairs + pair]);
                const float probability = score_row[position];
                result.x = fmaf(probability, v.x, result.x);
                result.y = fmaf(probability, v.y, result.y);
            }
            if constexpr (SkipZeroPV) {
                const bool tail_x = (__float_as_uint(result.x) & 0x7f800000u) == 0u;
                const bool tail_y = (__float_as_uint(result.y) & 0x7f800000u) == 0u;
                if (tail_x || tail_y)
                    for (int position = pv_length; position < key_length; ++position) {
                        const float2 v = __half22float2(value2[size_t(position) * pairs + pair]);
                        const float probability = score_row[position];
                        if (tail_x) result.x = fmaf(probability, v.x, result.x);
                        if (tail_y) result.y = fmaf(probability, v.y, result.y);
                    }
            }
            float* destination = output + size_t(head) * output_head_stride +
                size_t(query_position) * output_position_stride + pair * 2;
            destination[0] = result.x;
            destination[1] = result.y;
        }
    } else if constexpr (HeadDim == 256) {
        const int dimension = static_cast<int>(threadIdx.x);
        float result = 0.0f;
        // Validated future scores are -1e38 while the finite maximum has
        // magnitude <1.7e27, so expf(future - maximum) is exactly +0.
        // The PV omission specialization additionally validates finite V.
        const int pv_length = SkipZeroPV && validated ? dot_key_length : key_length;
        if constexpr (UnrollPV) {
#pragma unroll 4
            for (int key_position = 0; key_position < pv_length; ++key_position) {
                const size_t value_index =
                    (static_cast<size_t>(key_head) * key_capacity + key_position) * HeadDim + dimension;
                result = fmaf(score_row[key_position], __half2float(value[value_index]), result);
            }
        } else for (int key_position = 0; key_position < pv_length;
             ++key_position) {
            const size_t value_index =
                (static_cast<size_t>(key_head) * key_capacity +
                 key_position) * HeadDim + dimension;
            result = fmaf(
                score_row[key_position], __half2float(value[value_index]),
                result);
        }
        if constexpr (SkipZeroPV) {
            // Zero/subnormal results retain all future FMAs: preserve signed
            // zero and any build-specific flush-to-zero behavior. Nonzero
            // normal results are unchanged by zero times finite V.
            if ((__float_as_uint(result) & 0x7f800000u) == 0u)
                for (int key_position = pv_length; key_position < key_length; ++key_position) {
                    const size_t index = (static_cast<size_t>(key_head)*key_capacity+
                        key_position)*HeadDim+dimension;
                    result = fmaf(score_row[key_position],__half2float(value[index]),result);
                }
        }
        output[static_cast<size_t>(head) * output_head_stride +
               static_cast<size_t>(query_position) * output_position_stride +
               dimension] = result;
    } else {
        constexpr int value_groups = prefill_sdpa_threads / pairs;
        const int pair = static_cast<int>(threadIdx.x) % pairs;
        const int group = static_cast<int>(threadIdx.x) / pairs;
        const auto* value2 = reinterpret_cast<const __half2*>(value) +
            static_cast<size_t>(key_head) * key_capacity * pairs;
        float2 value_sum{0.0f, 0.0f};
        for (int key_position = group; key_position < key_length;
             key_position += value_groups) {
            const float2 element = __half22float2(
                value2[static_cast<size_t>(key_position) * pairs + pair]);
            const float probability = score_row[key_position];
            value_sum.x = fmaf(probability, element.x, value_sum.x);
            value_sum.y = fmaf(probability, element.y, value_sum.y);
        }
        __shared__ float2 value_partial[prefill_sdpa_threads];
        value_partial[threadIdx.x] = value_sum;
        __syncthreads();
        if (threadIdx.x < pairs) {
            float2 result = value_partial[pair];
#pragma unroll
            for (int value_group = 1; value_group < value_groups;
                 ++value_group) {
                const float2 next =
                    value_partial[value_group * pairs + pair];
                result.x += next.x;
                result.y += next.y;
            }
            const size_t output_base =
                static_cast<size_t>(head) * output_head_stride +
                static_cast<size_t>(query_position) *
                    output_position_stride +
                pair * 2;
            output[output_base] = result.x;
            output[output_base + 1] = result.y;
        }
    }
}

// Two adjacent queries retain independent original QK/softmax reductions and
// ordered PV FMA chains, while sharing each value-cache load in the PV pass.
template<bool SharedScores, bool ReuseKey = false, bool ReloadQuery = false, bool ValidatedPrefix = false, int QuerySlots = 2, bool HeadSlots = false>
__global__ void sdpa_prefill_pair_queries_cuda(
    const float* __restrict__ query, const __half* __restrict__ key, const __half* __restrict__ value, float* __restrict__ scores,
    float* __restrict__ output, const float* __restrict__ mask, int heads, int kv_heads, int queries,
    int keys, int past, int capacity, bool causal, float scale,
    size_t q_position_stride, size_t q_head_stride, size_t mask_column_stride,
    size_t mask_row_stride, size_t output_position_stride, size_t output_head_stride,
    const uint32_t* __restrict__ valid = nullptr) {
    constexpr int threads = 256, pairs = 128, lanes_per_key = 8;
    static_assert(!HeadSlots || (ValidatedPrefix && !ReuseKey),"head grouping requires ordered validated path");
    const int first_query = static_cast<int>(blockIdx.x) * (HeadSlots ? 1 : QuerySlots);
    const int slots = HeadSlots ? QuerySlots : min(QuerySlots, queries - first_query);
    const int head = blockIdx.y * (HeadSlots ? QuerySlots : 1), key_head = head / (heads / kv_heads);
    const size_t score_slot_stride = size_t(keys) * (SharedScores || !HeadSlots ? 1 : queries);
    const int lane = threadIdx.x % lanes_per_key;
    const bool validated=ValidatedPrefix&&valid&&*valid;
    const int first_key = threadIdx.x / lanes_per_key;
    const auto* key2 = reinterpret_cast<const __half2*>(key) + static_cast<size_t>(key_head) * capacity * pairs;
    extern __shared__ float shared_scores[];
    float* row0 = SharedScores ? shared_scores : scores + (static_cast<size_t>(head) * queries + first_query) * keys;
    __shared__ float reduction[threads];
    float paired_maximum[2]{-FLT_MAX,-FLT_MAX};
    if constexpr (ReuseKey) {
        const auto* query0 = reinterpret_cast<const float2*>(query + static_cast<size_t>(head) * q_head_stride +
            static_cast<size_t>(first_query) * q_position_stride);
        const auto* query1 = reinterpret_cast<const float2*>(query + static_cast<size_t>(head) * q_head_stride +
            static_cast<size_t>(first_query + (slots == 2)) * q_position_stride);
        for (int key_position = first_key; key_position < keys; key_position += threads / lanes_per_key) {
            // Diagnostic compiler fence: immutable queries may be reloaded
            // from L1 instead of occupying registers across K iterations.
            if constexpr (ReloadQuery) asm volatile("" ::: "memory");
            float dot0 = 0.f, dot1 = 0.f;
#pragma unroll
            for (int pair = lane; pair < pairs; pair += lanes_per_key) {
                const float2 k = __half22float2(key2[static_cast<size_t>(key_position) * pairs + pair]);
                const float2 q0 = query0[pair], q1 = query1[pair];
                dot0 = fmaf(q0.x,k.x,dot0); dot0 = fmaf(q0.y,k.y,dot0);
                dot1 = fmaf(q1.x,k.x,dot1); dot1 = fmaf(q1.y,k.y,dot1);
            }
            const unsigned active = __activemask();
#pragma unroll
            for (int offset = lanes_per_key / 2; offset > 0; offset /= 2) {
                dot0 += __shfl_down_sync(active,dot0,offset,lanes_per_key);
                dot1 += __shfl_down_sync(active,dot1,offset,lanes_per_key);
            }
            if (lane == 0) {
                float score0 = dot0 * scale, score1 = dot1 * scale;
                if (mask) {
                    score0 += mask[static_cast<size_t>(first_query)*mask_row_stride+static_cast<size_t>(key_position)*mask_column_stride];
                    if (slots == 2) score1 += mask[static_cast<size_t>(first_query+1)*mask_row_stride+static_cast<size_t>(key_position)*mask_column_stride];
                } else if (causal) {
                    if (key_position > past + first_query) score0 = -FLT_MAX;
                    if (key_position > past + first_query + 1) score1 = -FLT_MAX;
                }
                row0[key_position] = score0;
                paired_maximum[0] = fmaxf(paired_maximum[0],score0);
                if (slots == 2) {
                    row0[keys+key_position] = score1;
                    paired_maximum[1] = fmaxf(paired_maximum[1],score1);
                }
            }
        }
    }
    for (int slot = 0; slot < slots; ++slot) {
        const int query_position = first_query + (HeadSlots ? 0 : slot);
        const auto* query2 = reinterpret_cast<const float2*>(query + static_cast<size_t>(head + (HeadSlots ? slot : 0)) * q_head_stride +
            static_cast<size_t>(query_position) * q_position_stride);
        float* row = row0 + static_cast<size_t>(slot) * score_slot_stride;
        float local_maximum = ReuseKey ? paired_maximum[slot] : -FLT_MAX;
        const int dot_keys=validated?min(keys,past+query_position+1):keys;
        int key_position=first_key;
        if constexpr (!ReuseKey) for (; key_position < dot_keys; key_position += threads / lanes_per_key) {
            float dot = 0.f;
#pragma unroll
            for (int pair = lane; pair < pairs; pair += lanes_per_key) {
                const float2 q = query2[pair];
                const float2 k = __half22float2(key2[static_cast<size_t>(key_position) * pairs + pair]);
                dot = fmaf(q.x,k.x,dot);
                dot = fmaf(q.y,k.y,dot);
            }
            const unsigned active = __activemask();
#pragma unroll
            for (int offset = lanes_per_key / 2; offset > 0; offset /= 2)
                dot += __shfl_down_sync(active,dot,offset,lanes_per_key);
            if (lane == 0) {
                float score = dot * scale;
                if (mask) score += mask[static_cast<size_t>(query_position) * mask_row_stride + static_cast<size_t>(key_position) * mask_column_stride];
                else if (causal && key_position > past + query_position) score = -FLT_MAX;
                row[key_position] = score;
                local_maximum = fmaxf(local_maximum,score);
            }
        }
        if constexpr(ValidatedPrefix)if(lane==0)
            for(;key_position<keys;key_position+=threads/lanes_per_key)row[key_position]=-1e38f;
        const float maximum = mollm_cuda::detail::block_reduce_max<threads>(local_maximum,reduction);
        float local_sum = 0.f;
        for (int key_position = threadIdx.x; key_position < keys; key_position += threads) {
            const float numerator = expf(row[key_position] - maximum);
            row[key_position] = numerator; local_sum += numerator;
        }
        const float sum = mollm_cuda::detail::block_reduce_sum<threads>(local_sum,reduction);
        const float inverse = sum > 0.f ? 1.f / sum : 0.f;
        for (int key_position = threadIdx.x; key_position < keys; key_position += threads) row[key_position] *= inverse;
        __syncthreads();
    }
    if constexpr(ValidatedPrefix) {
        if(threadIdx.x<128) {
            const int pair=threadIdx.x;
            const auto* value2=reinterpret_cast<const __half2*>(value)+size_t(key_head)*capacity*128;
            const int end0=validated?min(keys,past+first_query+1):keys;
            int ends[QuerySlots];float2 result[QuerySlots]{};
#pragma unroll
            for(int slot=0;slot<QuerySlots;++slot)ends[slot]=validated?min(keys,past+first_query+(HeadSlots?0:slot)+1):keys;
            for(int position=0;position<end0;++position) {
                const float2 v=__half22float2(value2[size_t(position)*128+pair]);
#pragma unroll
                for(int slot=0;slot<QuerySlots;++slot)if(slot<slots) {
                    const float probability=row0[size_t(slot)*score_slot_stride+position];
                    result[slot].x=fmaf(probability,v.x,result[slot].x);
                    result[slot].y=fmaf(probability,v.y,result[slot].y);
                }
            }
            const int end_last=validated?min(keys,past+first_query+(HeadSlots?1:slots)):keys;
            for(int position=end0;position<end_last;++position) {
                const float2 v=__half22float2(value2[size_t(position)*128+pair]);
#pragma unroll
                for(int slot=0;slot<QuerySlots;++slot)if(slot<slots&&position<ends[slot]) {
                    const float probability=row0[size_t(slot)*score_slot_stride+position];
                    result[slot].x=fmaf(probability,v.x,result[slot].x);
                    result[slot].y=fmaf(probability,v.y,result[slot].y);
                }
            }
            // Keep the original zero/subnormal tail FMAs for each component.
            // They preserve signed zero and denormal rounding even though the
            // future probabilities are +0. Do not skip based on probability.
            bool tx[QuerySlots],ty[QuerySlots],tail=false;
#pragma unroll
            for(int slot=0;slot<QuerySlots;++slot) {
                tx[slot]=slot<slots&&(__float_as_uint(result[slot].x)&0x7f800000u)==0;
                ty[slot]=slot<slots&&(__float_as_uint(result[slot].y)&0x7f800000u)==0;
                tail|=tx[slot]||ty[slot];
            }
            if(tail)for(int position=end0;position<keys;++position) {
                const float2 v=__half22float2(value2[size_t(position)*128+pair]);
#pragma unroll
                for(int slot=0;slot<QuerySlots;++slot)if(slot<slots&&position>=ends[slot]) {
                    const float probability=row0[size_t(slot)*score_slot_stride+position];
                    if(tx[slot])result[slot].x=fmaf(probability,v.x,result[slot].x);
                    if(ty[slot])result[slot].y=fmaf(probability,v.y,result[slot].y);
                }
            }
#pragma unroll
            for(int slot=0;slot<QuerySlots;++slot)if(slot<slots) {
                float* destination=output+size_t(head+(HeadSlots?slot:0))*output_head_stride+size_t(first_query+(HeadSlots?0:slot))*output_position_stride+pair*2;
                destination[0]=result[slot].x;destination[1]=result[slot].y;
            }
        }
        return;
    }
    float first = 0.f, second = 0.f;
    for (int key_position = 0; key_position < keys; ++key_position) {
        const float v = __half2float(value[(static_cast<size_t>(key_head) * capacity + key_position) * 256 + threadIdx.x]);
        first = fmaf(row0[key_position],v,first);
        if (slots == 2) second = fmaf(row0[keys + key_position],v,second);
    }
    output[static_cast<size_t>(head) * output_head_stride + static_cast<size_t>(first_query) * output_position_stride + threadIdx.x] = first;
    if (slots == 2) output[static_cast<size_t>(head) * output_head_stride + static_cast<size_t>(first_query + 1) * output_position_stride + threadIdx.x] = second;
}

__global__ void sdpa_pv_expand_cuda(const __half* value,float* expanded,
    mollm_cuda::SdpaGemmPrefillParams p) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(p.kv_heads)*p.keys*p.dimension)return;
    const size_t row=i/p.dimension;
    expanded[i]=__half2float(value[(row/p.keys*size_t(p.capacity)+row%p.keys)*p.dimension+i%p.dimension]);
}
__global__ void sdpa_pv_bind_cuda(float** pointers,const float* value,
    const float* probabilities,float* output,mollm_cuda::SdpaGemmPrefillParams p) {
    const int h=blockIdx.x*blockDim.x+threadIdx.x;if(h>=p.heads)return;
    pointers[h]=const_cast<float*>(value)+size_t(h/(p.heads/p.kv_heads))*p.keys*p.dimension;
    pointers[p.heads+h]=const_cast<float*>(probabilities)+size_t(h)*p.queries*p.keys;
    pointers[2*p.heads+h]=output+size_t(h)*p.output_head_stride;
}
__global__ void sdpa_pv_copy_cuda(const float* source,float* destination,
    const uint32_t* valid,mollm_cuda::SdpaGemmPrefillParams p) {
    if(!*valid)return;
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(p.heads)*p.queries*p.dimension)return;
    const size_t row=i/p.dimension;
    const size_t offset=row/p.queries*p.output_head_stride+row%p.queries*p.output_position_stride+i%p.dimension;
    destination[offset]=source[offset];
}

// Diagnostic stage isolation: keep GEMM QK/softmax, but accumulate PV in
// ascending key order with FP32 FMAs. Never selected by production dispatch.
__global__ void sdpa_gemm_ordered_pv_cuda(const float* probabilities,
    const float* value,float* output,mollm_cuda::SdpaGemmPrefillParams p) {
    const int d=threadIdx.x,q=blockIdx.x,head=blockIdx.y;
    if(d>=p.dimension)return;
    const int kv_head=head/(p.heads/p.kv_heads);
    const float* row=probabilities+(size_t(head)*p.queries+q)*p.keys;
    const float* values=value+size_t(kv_head)*p.keys*p.dimension+d;
    float sum=0.f;
    for(int key=0;key<p.keys;++key)sum=fmaf(row[key],values[size_t(key)*p.dimension],sum);
    output[size_t(head)*p.output_head_stride+size_t(q)*p.output_position_stride+d]=sum;
}

}  // namespace

namespace mollm_cuda {

static cublasStatus_t sdpa_prefill_fp32_pv_impl(cublasHandle_t handle,const void* value,
    const float* probabilities,float* output,float* value_fp32,float** pointers,
    const SdpaGemmPrefillParams& p,cudaStream_t stream,bool values_ready) {
    if(!handle||!value||!probabilities||!output||!value_fp32||!pointers||
       p.dimension!=256||p.heads<1||p.heads>64||p.kv_heads<1||p.heads%p.kv_heads||
       p.queries<2||p.queries>2048||p.keys<p.queries||p.keys>32768||p.capacity<p.keys||
       p.past<0||p.past>p.keys-p.queries||p.output_position_stride<256||
       p.output_position_stride>INT32_MAX||p.output_head_stride<256||
       p.output_head_stride>SIZE_MAX/p.heads/8||p.output_position_stride>SIZE_MAX/p.queries/8||
       reinterpret_cast<uintptr_t>(pointers)%alignof(float*))return CUBLAS_STATUS_INVALID_VALUE;
    // Canonical head-major or position-major non-overlapping output matrices.
    if(p.heads>1&&p.output_head_stride<size_t(p.queries)*p.output_position_stride&&
       p.output_position_stride<size_t(p.heads)*p.output_head_stride)return CUBLAS_STATUS_INVALID_VALUE;
    cudaStream_t hs;auto status=cublasGetStream(handle,&hs);if(status!=CUBLAS_STATUS_SUCCESS)return status;
    cublasPointerMode_t mode;status=cublasGetPointerMode(handle,&mode);if(status!=CUBLAS_STATUS_SUCCESS)return status;
    if(hs!=stream||mode!=CUBLAS_POINTER_MODE_HOST)return CUBLAS_STATUS_INVALID_VALUE;
    const size_t n=size_t(p.kv_heads)*p.keys*p.dimension;
    if(!values_ready) {
        sdpa_pv_expand_cuda<<<unsigned((n+255)/256),256,0,stream>>>(static_cast<const __half*>(value),value_fp32,p);
        if(cudaGetLastError()!=cudaSuccess)return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    sdpa_pv_bind_cuda<<<(p.heads+127)/128,128,0,stream>>>(pointers,value_fp32,probabilities,output,p);
    if(cudaGetLastError()!=cudaSuccess)return CUBLAS_STATUS_EXECUTION_FAILED;
    const float one=1.f,zero=0.f;
    return cublasGemmBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,p.dimension,p.queries,p.keys,
        &one,reinterpret_cast<const void* const*>(pointers),CUDA_R_32F,p.dimension,
        reinterpret_cast<const void* const*>(pointers+p.heads),CUDA_R_32F,p.keys,
        &zero,reinterpret_cast<void* const*>(pointers+2*p.heads),CUDA_R_32F,int(p.output_position_stride),p.heads,
        CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT);
}
cublasStatus_t launch_sdpa_prefill_fp32_pv(cublasHandle_t handle,const void* value,
    const float* probabilities,float* output,float* value_fp32,float** pointers,
    const SdpaGemmPrefillParams& p,cudaStream_t stream) {
    return sdpa_prefill_fp32_pv_impl(handle,value,probabilities,output,value_fp32,pointers,p,stream,false);
}
void launch_sdpa_validated_pv_copy(const float* source,float* destination,
    const uint32_t* valid,const SdpaGemmPrefillParams& p,cudaStream_t stream) {
    const size_t n=size_t(p.heads)*p.queries*p.dimension;
    sdpa_pv_copy_cuda<<<unsigned((n+255)/256),256,0,stream>>>(source,destination,valid,p);
}

cublasStatus_t launch_sdpa_prefill_fp32_gemm(cublasHandle_t handle,
    const float* query, const void* key, const void* value, float* scores,
    float* output, const float* mask, float* key_fp32, float* value_fp32,
    const SdpaGemmPrefillParams& p, cudaStream_t stream, float** pointer_workspace,bool ordered_pv) {
    if (!handle || !query || !key || !value || !scores || !output || !key_fp32 || !value_fp32 ||
        p.heads <= 0 || p.heads > 65535 || p.kv_heads <= 0 || p.heads % p.kv_heads ||
        p.queries <= 1 || p.keys < p.queries || p.keys > INT32_MAX - 255 || p.capacity < p.keys || p.past < 0 ||
        p.past > p.keys - p.queries || (p.dimension != 128 && p.dimension != 256) ||
        p.query_position_stride < static_cast<size_t>(p.dimension) || p.query_position_stride > INT32_MAX ||
        p.output_position_stride < static_cast<size_t>(p.dimension) || p.output_position_stride > INT32_MAX ||
        p.query_head_stride < static_cast<size_t>(p.dimension) || p.output_head_stride < static_cast<size_t>(p.dimension) ||
        static_cast<size_t>(p.queries) > SIZE_MAX / static_cast<size_t>(p.keys) / p.heads / sizeof(float) ||
        p.query_head_stride > SIZE_MAX / p.heads / 8 || p.output_head_stride > SIZE_MAX / p.heads / 8 ||
        p.query_position_stride > SIZE_MAX / p.queries / 8 || p.output_position_stride > SIZE_MAX / p.queries / 8 ||
        (mask && (p.mask_column_stride > SIZE_MAX / p.keys / 8 || p.mask_row_stride > SIZE_MAX / p.queries / 8)))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (pointer_workspace && reinterpret_cast<uintptr_t>(pointer_workspace) % alignof(float*))
        return CUBLAS_STATUS_INVALID_VALUE;
    cudaStream_t handle_stream;
    auto status = cublasGetStream(handle, &handle_stream);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    if (handle_stream != stream) return CUBLAS_STATUS_INVALID_VALUE;
    cublasPointerMode_t pointer_mode;
    status = cublasGetPointerMode(handle, &pointer_mode);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    if (pointer_mode != CUBLAS_POINTER_MODE_HOST) return CUBLAS_STATUS_INVALID_VALUE;
    const size_t kv_count = static_cast<size_t>(p.kv_heads) * p.keys * p.dimension;
    if ((kv_count + 255) / 256 > INT32_MAX) return CUBLAS_STATUS_INVALID_VALUE;
    sdpa_expand_fp16_cache_cuda<<<static_cast<unsigned>((kv_count + 255) / 256),256,0,stream>>>(
        static_cast<const __half*>(key), static_cast<const __half*>(value), key_fp32, value_fp32,
        kv_count, p.keys, p.capacity, p.dimension);
    if (cudaGetLastError() != cudaSuccess) return CUBLAS_STATUS_EXECUTION_FAILED;
    const float one = 1.f, zero = 0.f;
    if (pointer_workspace) {
        sdpa_gemm_bind_heads_cuda<<<(p.heads+127)/128,128,0,stream>>>(pointer_workspace,query,key_fp32,value_fp32,scores,output,p);
        if (cudaGetLastError() != cudaSuccess) return CUBLAS_STATUS_EXECUTION_FAILED;
        status = cublasGemmBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,p.keys,p.queries,p.dimension,
            &one,reinterpret_cast<const void* const*>(pointer_workspace),CUDA_R_32F,p.dimension,
            reinterpret_cast<const void* const*>(pointer_workspace+p.heads),CUDA_R_32F,static_cast<int>(p.query_position_stride),
            &zero,reinterpret_cast<void* const*>(pointer_workspace+2*p.heads),CUDA_R_32F,p.keys,p.heads,
            CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    } else for (int head = 0; head < p.heads; ++head) {
        const int kv_head = head / (p.heads / p.kv_heads);
        status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, p.keys, p.queries, p.dimension,
            &one, key_fp32 + static_cast<size_t>(kv_head) * p.keys * p.dimension, CUDA_R_32F, p.dimension,
            query + static_cast<size_t>(head) * p.query_head_stride, CUDA_R_32F, static_cast<int>(p.query_position_stride),
            &zero, scores + static_cast<size_t>(head) * p.queries * p.keys, CUDA_R_32F, p.keys,
            CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    sdpa_gemm_softmax_cuda<<<dim3(p.queries,p.heads),256,0,stream>>>(scores,mask,
        p.queries,p.keys,p.past,p.causal,p.scale,p.mask_column_stride,p.mask_row_stride);
    if (cudaGetLastError() != cudaSuccess) return CUBLAS_STATUS_EXECUTION_FAILED;
    if(ordered_pv) {
        sdpa_gemm_ordered_pv_cuda<<<dim3(p.queries,p.heads),256,0,stream>>>(scores,value_fp32,output,p);
        return cudaGetLastError()==cudaSuccess?CUBLAS_STATUS_SUCCESS:CUBLAS_STATUS_EXECUTION_FAILED;
    }
    if (pointer_workspace) {
        status = cublasGemmBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,p.dimension,p.queries,p.keys,
            &one,reinterpret_cast<const void* const*>(pointer_workspace+3*p.heads),CUDA_R_32F,p.dimension,
            reinterpret_cast<const void* const*>(pointer_workspace+2*p.heads),CUDA_R_32F,p.keys,
            &zero,reinterpret_cast<void* const*>(pointer_workspace+4*p.heads),CUDA_R_32F,static_cast<int>(p.output_position_stride),p.heads,
            CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    } else for (int head = 0; head < p.heads; ++head) {
        const int kv_head = head / (p.heads / p.kv_heads);
        status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, p.dimension, p.queries, p.keys,
            &one, value_fp32 + static_cast<size_t>(kv_head) * p.keys * p.dimension, CUDA_R_32F, p.dimension,
            scores + static_cast<size_t>(head) * p.queries * p.keys, CUDA_R_32F, p.keys,
            &zero, output + static_cast<size_t>(head) * p.output_head_stride, CUDA_R_32F, static_cast<int>(p.output_position_stride),
            CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    return CUBLAS_STATUS_SUCCESS;
}

void launch_rope(
    const float* input, const float* cosine, const float* sine, float* output,
    int feature_dim, int sequence_length, int channels, int shape2,
    int rope_dim, bool interleave, size_t x_s0, size_t x_s1, size_t x_s2,
    size_t x_s3, size_t c_s0, size_t c_s1, size_t s_s0, size_t s_s1,
    size_t o_s0, size_t o_s1, size_t o_s2, size_t o_s3) {
    constexpr int threads = 256;
    const size_t count = static_cast<size_t>(feature_dim) *
        sequence_length * channels;
    rope_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                threads>>>(
        input, cosine, sine, output, feature_dim, sequence_length, channels,
        shape2, rope_dim, interleave, x_s0, x_s1, x_s2, x_s3, c_s0, c_s1,
        s_s0, s_s1, o_s0, o_s1, o_s2, o_s3);
}

void launch_qk_rms_norm_rope(
    const float* query, const float* key, const float* query_weight,
    const float* key_weight, const float* cosine, const float* sine,
    float* output, int feature_dim, int sequence_length, int query_heads,
    int total_heads, int rope_dim, bool interleave,
    size_t query_feature_stride, size_t query_row_stride,
    size_t key_feature_stride, size_t key_row_stride,
    size_t cosine_feature_stride, size_t cosine_position_stride,
    size_t sine_feature_stride, size_t sine_position_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride, float epsilon) {
    constexpr int threads = 256;
    qk_rms_norm_rope_cuda<<<sequence_length * total_heads, threads>>>(
        query, key, query_weight, key_weight, cosine, sine, output,
        feature_dim, sequence_length, query_heads, total_heads, rope_dim,
        interleave, query_feature_stride, query_row_stride,
        key_feature_stride, key_row_stride, cosine_feature_stride,
        cosine_position_stride, sine_feature_stride, sine_position_stride,
        output_feature_stride, output_position_stride, output_head_stride,
        epsilon);
}

void launch_append_kv(
    const float* key, const float* value, void* key_cache,
    void* value_cache, bool fp16_cache, int num_kv_heads,
    int current_length, int past_length, int max_length,
    int key_dim, int value_dim, size_t key_position_stride,
    size_t key_head_stride, size_t value_position_stride,
    size_t value_head_stride) {
    constexpr int threads = 256;
    const int maximum_dim = key_dim > value_dim ? key_dim : value_dim;
    const size_t count = static_cast<size_t>(num_kv_heads) * current_length *
        maximum_dim;
    append_kv_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                     threads>>>(
        key, value, key_cache, value_cache, fp16_cache, num_kv_heads,
        current_length, past_length, max_length, key_dim, value_dim,
        key_position_stride, key_head_stride, value_position_stride,
        value_head_stride);
}

void launch_sdpa_scores(
    const float* query, const void* key, float* scores, const float* mask,
    int num_heads, int num_kv_heads, int query_length, int key_length,
    int past_length, int key_dim, int key_capacity, bool cached,
    bool fp16_cache, bool causal, float scale,
    size_t query_feature_stride, size_t query_position_stride,
    size_t query_head_stride, size_t key_feature_stride,
    size_t key_position_stride, size_t key_head_stride,
    size_t mask_column_stride, size_t mask_row_stride) {
    constexpr int threads = 256;
    const size_t count = static_cast<size_t>(num_heads) * query_length *
        key_length;
    sdpa_scores_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                       threads>>>(
        query, key, scores, mask, num_heads, num_kv_heads, query_length,
        key_length, past_length, key_dim, key_capacity, cached, fp16_cache,
        causal, scale, query_feature_stride, query_position_stride,
        query_head_stride, key_feature_stride, key_position_stride,
        key_head_stride, mask_column_stride, mask_row_stride);
}

void launch_sdpa_output(
    const float* scores, const void* value, float* output, int num_heads,
    int num_kv_heads, int query_length, int key_length, int value_dim,
    int value_capacity, bool cached, bool fp16_cache,
    size_t value_feature_stride, size_t value_position_stride,
    size_t value_head_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride) {
    constexpr int threads = 256;
    sdpa_output_cuda<<<num_heads * query_length, threads>>>(
        scores, value, output, num_heads, num_kv_heads, query_length,
        key_length, value_dim, value_capacity, cached, fp16_cache,
        value_feature_stride, value_position_stride, value_head_stride,
        output_feature_stride, output_position_stride, output_head_stride);
}

bool try_launch_sdpa_prefill(
    const float* query, const void* key, const void* value, float* scores,
    float* output, const float* mask, int num_heads, int num_kv_heads,
    int query_length, int key_length, int past_length, int key_dim,
    int value_dim, int key_capacity, bool cached, bool fp16_cache,
    bool causal, float scale, size_t query_feature_stride,
    size_t query_position_stride, size_t query_head_stride,
    size_t mask_column_stride, size_t mask_row_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride, bool skip_causal_qk, bool paired_queries, bool paired_qk, bool reload_paired_query) {
    const bool supported = query && key && value && scores && output &&
        query_length > 1 && cached && fp16_cache &&
        (key_dim == 128 || key_dim == 256) && value_dim == key_dim &&
        query_feature_stride == 1 &&
        query_position_stride % 2 == 0 && query_head_stride % 2 == 0 &&
        output_feature_stride == 1 &&
        (reinterpret_cast<uintptr_t>(query) &
         (alignof(float2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(key) &
         (alignof(__half2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(value) &
         (alignof(__half2) - 1)) == 0;
    if (!supported)
        return false;
    if (paired_qk && !paired_queries) return false;
    if (reload_paired_query && !paired_qk) return false;
    if (paired_queries && (num_heads <= 0 || num_heads > 65535 || num_kv_heads <= 0 ||
            num_heads % num_kv_heads || query_length > INT32_MAX - 1 ||
            key_length <= 0 || key_length > INT32_MAX - 255 || key_capacity < key_length ||
            past_length < 0 || past_length > INT32_MAX - query_length)) return false;
    if (paired_queries && key_dim == 256) {
        const dim3 paired_grid((query_length + 1) / 2, num_heads);
        if (key_length <= 1024) {
            const auto kernel = paired_qk
                ? (reload_paired_query ? sdpa_prefill_pair_queries_cuda<true,true,true> : sdpa_prefill_pair_queries_cuda<true,true,false>)
                : sdpa_prefill_pair_queries_cuda<true,false>;
            kernel<<<paired_grid,prefill_sdpa_threads,size_t(key_length)*2*sizeof(float)>>>(
                query,static_cast<const __half*>(key),static_cast<const __half*>(value),scores,output,mask,
                num_heads,num_kv_heads,query_length,key_length,past_length,key_capacity,causal,scale,
                query_position_stride,query_head_stride,mask_column_stride,mask_row_stride,output_position_stride,output_head_stride,nullptr);
        } else {
            const auto kernel = paired_qk
                ? (reload_paired_query ? sdpa_prefill_pair_queries_cuda<false,true,true> : sdpa_prefill_pair_queries_cuda<false,true,false>)
                : sdpa_prefill_pair_queries_cuda<false,false>;
            kernel<<<paired_grid,prefill_sdpa_threads>>>(
                query,static_cast<const __half*>(key),static_cast<const __half*>(value),scores,output,mask,
                num_heads,num_kv_heads,query_length,key_length,past_length,key_capacity,causal,scale,
                query_position_stride,query_head_stride,mask_column_stride,mask_row_stride,output_position_stride,output_head_stride,nullptr);
        }
        return true;
    }
    const dim3 grid(
        static_cast<unsigned>(query_length),
        static_cast<unsigned>(num_heads));
    // Small score rows are cheaper to keep on chip. Above 1024 entries the
    // dynamic shared-memory footprint reduces occupancy enough that the
    // global scratch row is faster.
    constexpr int shared_score_limit = 1024;
    if (key_length <= shared_score_limit) {
        const size_t shared_bytes =
            static_cast<size_t>(key_length) * sizeof(float);
        if (key_dim == 256) {
            const auto kernel = skip_causal_qk
                ? sdpa_prefill_fp16_cuda<256, true, true>
                : sdpa_prefill_fp16_cuda<256, true, false>;
            kernel
                <<<grid, prefill_sdpa_threads, shared_bytes>>>(
                    query, static_cast<const __half*>(key),
                    static_cast<const __half*>(value), scores, output, mask,
                    num_heads, num_kv_heads, query_length, key_length,
                    past_length, key_capacity, causal, scale,
                    query_position_stride, query_head_stride,
                    mask_column_stride, mask_row_stride,
                    output_position_stride, output_head_stride, nullptr);
        } else
            sdpa_prefill_fp16_cuda<128, true>
                <<<grid, prefill_sdpa_threads, shared_bytes>>>(
                    query, static_cast<const __half*>(key),
                    static_cast<const __half*>(value), scores, output, mask,
                    num_heads, num_kv_heads, query_length, key_length,
                    past_length, key_capacity, causal, scale,
                    query_position_stride, query_head_stride,
                    mask_column_stride, mask_row_stride,
                    output_position_stride, output_head_stride);
    } else {
        if (key_dim == 256) {
            const auto kernel = skip_causal_qk
                ? sdpa_prefill_fp16_cuda<256, false, true>
                : sdpa_prefill_fp16_cuda<256, false, false>;
            kernel
                <<<grid, prefill_sdpa_threads>>>(
                    query, static_cast<const __half*>(key),
                    static_cast<const __half*>(value), scores, output, mask,
                    num_heads, num_kv_heads, query_length, key_length,
                    past_length, key_capacity, causal, scale,
                    query_position_stride, query_head_stride,
                    mask_column_stride, mask_row_stride,
                    output_position_stride, output_head_stride, nullptr);
        } else
            sdpa_prefill_fp16_cuda<128, false>
                <<<grid, prefill_sdpa_threads>>>(
                    query, static_cast<const __half*>(key),
                    static_cast<const __half*>(value), scores, output, mask,
                    num_heads, num_kv_heads, query_length, key_length,
                    past_length, key_capacity, causal, scale,
                    query_position_stride, query_head_stride,
                    mask_column_stride, mask_row_stride,
                    output_position_stride, output_head_stride);
    }
    return true;
}

// These phases are TU-private: only the guarded composite below can reuse a
// freshly validated parent flag, with unchanged inputs on the same stream.
enum class SdpaPrefixPhase { VALIDATE_AND_EXECUTE, VALIDATE_ONLY, EXECUTE_VALIDATED };
static bool sdpa_prefill_prefix_phase(const float* query,
    const void* key,const void* value,float* scores,float* output,
    const float* mask,uint32_t* valid,const SdpaGemmPrefillParams& p,cudaStream_t stream,
    cudaError_t* launch_error,int shared_score_limit,bool skip_zero_pv,bool unroll_pv,bool pair_pv,bool pair_queries,int query_group,bool grouped_long_shared,bool grouped_heads,bool probabilities_only,int head_lanes,bool elide_validated_mask,SdpaPrefixPhase phase) {
    if (launch_error) *launch_error=cudaSuccess;
    const auto fail=[&](cudaError_t error) {
        if (launch_error) *launch_error=error;
        return false;
    };
    if(phase!=SdpaPrefixPhase::VALIDATE_AND_EXECUTE&&phase!=SdpaPrefixPhase::VALIDATE_ONLY&&
       phase!=SdpaPrefixPhase::EXECUTE_VALIDATED)return false;
    if(phase!=SdpaPrefixPhase::VALIDATE_AND_EXECUTE&&(!probabilities_only||!elide_validated_mask))return false;
    // Grouped score rows multiply shared storage by QuerySlots. Keep the same
    // total32KiB payload bound as the original single-query threshold8192.
    // Validate group BEFORE division; old callers retain the1024 restriction.
    if(probabilities_only&&(p.keys<=shared_score_limit||pair_queries||grouped_heads||!skip_zero_pv))return false;
    if(head_lanes!=1&&(head_lanes!=2||!probabilities_only||!pair_pv||
        p.kv_heads<=0||p.heads<=0||p.heads%p.kv_heads||(p.heads/p.kv_heads)%head_lanes))return false;
    if(elide_validated_mask&&!probabilities_only)return false;
    if(grouped_long_shared&&!pair_queries)return false;
    if(grouped_heads&&(!pair_queries||query_group!=4||pair_pv||grouped_long_shared||
        p.kv_heads<=0||p.heads<=0||p.heads%p.kv_heads||(p.heads/p.kv_heads)%4))return false;
    if(pair_queries&&((query_group!=2&&query_group!=4)||!skip_zero_pv||unroll_pv||
        shared_score_limit>(grouped_long_shared?8192/query_group:1024)))return false;
    if ((pair_pv && unroll_pv) || (unroll_pv && !skip_zero_pv) || shared_score_limit < 1 || shared_score_limit > 8192 ||
        !query || !key || !value || !scores || !output || !mask || !valid ||
        p.dimension!=256 || p.heads<=0 || p.heads>65535 || p.kv_heads<=0 ||
        p.heads%p.kv_heads || p.queries<2 || p.queries>65535 ||
        p.keys<=0 || p.keys>65535 || p.capacity<p.keys || p.past<0 ||
        p.past>INT32_MAX-p.queries || p.query_position_stride<256 ||
        p.query_head_stride<256 || p.query_position_stride%2 || p.query_head_stride%2 ||
        p.output_position_stride<256 || p.output_head_stride<256 ||
        !p.mask_column_stride || !p.mask_row_stride ||
        (reinterpret_cast<uintptr_t>(query)&7) || (reinterpret_cast<uintptr_t>(key)&3) ||
        (reinterpret_cast<uintptr_t>(value)&3) || (reinterpret_cast<uintptr_t>(valid)&3)) return false;
    const auto fits=[](int a,int b,size_t sa,size_t sb,size_t tail,size_t bytes) {
        const size_t limit=SIZE_MAX/bytes-tail;
        if (sa && size_t(a-1)>limit/sa) return false;
        const size_t first=size_t(a-1)*sa;
        return !sb || size_t(b-1)<=(limit-first)/sb;
    };
    if (!fits(p.heads,p.queries,p.query_head_stride,p.query_position_stride,256,4) ||
        !fits(p.heads,p.queries,p.output_head_stride,p.output_position_stride,256,4) ||
        !fits(p.queries,p.keys,p.mask_row_stride,p.mask_column_stride,1,4)) return false;
    if(pair_queries||probabilities_only) {
        // Match the canonical kernel's non-alias contract explicitly. Without
        // restrict, stores to global scores can inhibit Q-load hoisting.
        const size_t kv_span=(size_t(p.kv_heads-1)*p.capacity+p.keys)*256*2;
        const void* pointers[]={query,key,value,mask,scores,output,valid};
        const size_t bytes[]={
            (size_t(p.heads-1)*p.query_head_stride+size_t(p.queries-1)*p.query_position_stride+256)*4,
            kv_span,kv_span,(size_t(p.queries-1)*p.mask_row_stride+size_t(p.keys-1)*p.mask_column_stride+1)*4,
            phase==SdpaPrefixPhase::VALIDATE_ONLY?0:size_t(p.heads)*p.queries*p.keys*4,
            phase==SdpaPrefixPhase::VALIDATE_ONLY?0:(size_t(p.heads-1)*p.output_head_stride+size_t(p.queries-1)*p.output_position_stride+256)*4,4};
        for(int i=0;i<7;++i) {
            if(!bytes[i])continue;
            const uintptr_t current=reinterpret_cast<uintptr_t>(pointers[i]);
            if(bytes[i]>UINTPTR_MAX-current)return false;
            for(int j=0;j<i;++j) {
                if(!bytes[j])continue;
                const uintptr_t previous=reinterpret_cast<uintptr_t>(pointers[j]);
                if((i>=4||j>=4)&&current<previous+bytes[j]&&previous<current+bytes[i])return false;
            }
        }
    }
    // With finite FP16 K and abs(Q)<=1e20, all head256 dots remain <1.7e27
    // in magnitude when abs(scale)<=1. Adding them to -1e38 rounds back to
    // exactly -1e38 (ULP~1e31); no softmax/PV operations are omitted.
    const bool bounded_scale=std::isfinite(p.scale) && std::fabs(p.scale)<=1.f;
    if(phase!=SdpaPrefixPhase::EXECUTE_VALIDATED) {
    const auto reset=cudaMemsetAsync(valid,bounded_scale?1:0,sizeof(*valid),stream);
    if (reset!=cudaSuccess) return fail(reset);
    const size_t count=size_t(p.heads)*p.queries*256+size_t(p.kv_heads)*p.keys*256*(skip_zero_pv?2:1)+size_t(p.queries)*p.keys;
    const unsigned blocks=static_cast<unsigned>(std::min<size_t>(65535,(count+255)/256));
    if (skip_zero_pv&&elide_validated_mask)
        validate_sdpa_prefix_cuda<true,true><<<blocks,256,0,stream>>>(query,static_cast<const __half*>(key),
            static_cast<const __half*>(value),mask,valid,p);
    else if (skip_zero_pv)
        validate_sdpa_prefix_cuda<true><<<blocks,256,0,stream>>>(query,static_cast<const __half*>(key),
            static_cast<const __half*>(value),mask,valid,p);
    else validate_sdpa_prefix_cuda<false><<<blocks,256,0,stream>>>(query,static_cast<const __half*>(key),
        static_cast<const __half*>(value),mask,valid,p);
    const auto validation=cudaGetLastError();
    if (validation!=cudaSuccess) return fail(validation);
    }
    if(phase==SdpaPrefixPhase::VALIDATE_ONLY)return true;
    const dim3 grid(p.queries,p.heads);
    const auto launch=[&](auto shared,auto skip) {
        constexpr bool cached_scores=decltype(shared)::value;
        constexpr bool omit_zero_pv=decltype(skip)::value;
        if(probabilities_only) {
            if constexpr(!cached_scores) {
              if(head_lanes==2&&elide_validated_mask)
                sdpa_prefill_fp16_cuda<256,false,false,true,true,false,true,true,2,true><<<dim3(p.queries,p.heads/2),prefill_sdpa_threads*2,0,stream>>>(query,static_cast<const __half*>(key),
                    static_cast<const __half*>(value),scores,output,mask,p.heads,p.kv_heads,p.queries,p.keys,
                    p.past,p.capacity,p.causal,p.scale,p.query_position_stride,p.query_head_stride,
                    p.mask_column_stride,p.mask_row_stride,p.output_position_stride,p.output_head_stride,valid);
              else if(head_lanes==2)
                sdpa_prefill_fp16_cuda<256,false,false,true,true,false,true,true,2><<<dim3(p.queries,p.heads/2),prefill_sdpa_threads*2,0,stream>>>(query,static_cast<const __half*>(key),
                    static_cast<const __half*>(value),scores,output,mask,p.heads,p.kv_heads,p.queries,p.keys,
                    p.past,p.capacity,p.causal,p.scale,p.query_position_stride,p.query_head_stride,
                    p.mask_column_stride,p.mask_row_stride,p.output_position_stride,p.output_head_stride,valid);
              else if(elide_validated_mask)
                sdpa_prefill_fp16_cuda<256,false,false,true,true,false,true,true,1,true><<<grid,prefill_sdpa_threads,0,stream>>>(query,static_cast<const __half*>(key),
                    static_cast<const __half*>(value),scores,output,mask,p.heads,p.kv_heads,p.queries,p.keys,
                    p.past,p.capacity,p.causal,p.scale,p.query_position_stride,p.query_head_stride,
                    p.mask_column_stride,p.mask_row_stride,p.output_position_stride,p.output_head_stride,valid);
              else
                sdpa_prefill_fp16_cuda<256,false,false,true,true,false,true,true><<<grid,prefill_sdpa_threads,0,stream>>>(query,static_cast<const __half*>(key),
                    static_cast<const __half*>(value),scores,output,mask,p.heads,p.kv_heads,p.queries,p.keys,
                    p.past,p.capacity,p.causal,p.scale,p.query_position_stride,p.query_head_stride,
                    p.mask_column_stride,p.mask_row_stride,p.output_position_stride,p.output_head_stride,valid);
            }
        } else if(grouped_heads)
            sdpa_prefill_pair_queries_cuda<cached_scores,false,false,true,4,true><<<dim3(p.queries,p.heads/4),prefill_sdpa_threads,
                cached_scores?size_t(p.keys)*4*sizeof(float):0,stream>>>(query,static_cast<const __half*>(key),
                static_cast<const __half*>(value),scores,output,mask,p.heads,p.kv_heads,p.queries,p.keys,
                p.past,p.capacity,p.causal,p.scale,p.query_position_stride,p.query_head_stride,
                p.mask_column_stride,p.mask_row_stride,p.output_position_stride,p.output_head_stride,valid);
        else if(pair_queries&&query_group==4)
            sdpa_prefill_pair_queries_cuda<cached_scores,false,false,true,4><<<dim3((p.queries+3)/4,p.heads),prefill_sdpa_threads,
                cached_scores?size_t(p.keys)*4*sizeof(float):0,stream>>>(query,static_cast<const __half*>(key),
                static_cast<const __half*>(value),scores,output,mask,p.heads,p.kv_heads,p.queries,p.keys,
                p.past,p.capacity,p.causal,p.scale,p.query_position_stride,p.query_head_stride,
                p.mask_column_stride,p.mask_row_stride,p.output_position_stride,p.output_head_stride,valid);
        else if(pair_queries)
            sdpa_prefill_pair_queries_cuda<cached_scores,false,false,true><<<dim3((p.queries+1)/2,p.heads),prefill_sdpa_threads,
                cached_scores?size_t(p.keys)*2*sizeof(float):0,stream>>>(query,static_cast<const __half*>(key),
                static_cast<const __half*>(value),scores,output,mask,p.heads,p.kv_heads,p.queries,p.keys,
                p.past,p.capacity,p.causal,p.scale,p.query_position_stride,p.query_head_stride,
                p.mask_column_stride,p.mask_row_stride,p.output_position_stride,p.output_head_stride,valid);
        else if (pair_pv)
            sdpa_prefill_fp16_cuda<256,cached_scores,false,true,omit_zero_pv,false,true><<<grid,prefill_sdpa_threads,
                cached_scores?size_t(p.keys)*sizeof(float):0,stream>>>(query,static_cast<const __half*>(key),
                static_cast<const __half*>(value),scores,output,mask,p.heads,p.kv_heads,p.queries,p.keys,
                p.past,p.capacity,p.causal,p.scale,p.query_position_stride,p.query_head_stride,
                p.mask_column_stride,p.mask_row_stride,p.output_position_stride,p.output_head_stride,valid);
        else if (unroll_pv)
            sdpa_prefill_fp16_cuda<256,cached_scores,false,true,omit_zero_pv,true><<<grid,prefill_sdpa_threads,
                cached_scores?size_t(p.keys)*sizeof(float):0,stream>>>(query,static_cast<const __half*>(key),
                static_cast<const __half*>(value),scores,output,mask,p.heads,p.kv_heads,p.queries,p.keys,
                p.past,p.capacity,p.causal,p.scale,p.query_position_stride,p.query_head_stride,
                p.mask_column_stride,p.mask_row_stride,p.output_position_stride,p.output_head_stride,valid);
        else sdpa_prefill_fp16_cuda<256,cached_scores,false,true,omit_zero_pv><<<grid,prefill_sdpa_threads,
            cached_scores?size_t(p.keys)*sizeof(float):0,stream>>>(query,static_cast<const __half*>(key),
            static_cast<const __half*>(value),scores,output,mask,p.heads,p.kv_heads,p.queries,p.keys,
            p.past,p.capacity,p.causal,p.scale,p.query_position_stride,p.query_head_stride,
            p.mask_column_stride,p.mask_row_stride,p.output_position_stride,p.output_head_stride,valid);
    };
    const auto select=[&](auto skip) {
        if (p.keys<=shared_score_limit) launch(std::true_type{},skip);
        else launch(std::false_type{},skip);
    };
    if (skip_zero_pv) select(std::true_type{}); else select(std::false_type{});
    const auto attention=cudaGetLastError();
    return attention==cudaSuccess ? true : fail(attention);
}

bool try_launch_sdpa_prefill_validated_prefix(const float* query,
    const void* key,const void* value,float* scores,float* output,
    const float* mask,uint32_t* valid,const SdpaGemmPrefillParams& p,cudaStream_t stream,
    cudaError_t* launch_error,int shared_score_limit,bool skip_zero_pv,bool unroll_pv,bool pair_pv,
    bool pair_queries,int query_group,bool grouped_long_shared,bool grouped_heads,bool probabilities_only,
    int head_lanes,bool elide_validated_mask) {
    return sdpa_prefill_prefix_phase(query,key,value,scores,output,mask,valid,p,stream,launch_error,
        shared_score_limit,skip_zero_pv,unroll_pv,pair_pv,pair_queries,query_group,grouped_long_shared,
        grouped_heads,probabilities_only,head_lanes,elide_validated_mask,SdpaPrefixPhase::VALIDATE_AND_EXECUTE);
}

bool try_launch_sdpa_prefill_query_slices(cublasHandle_t handle,const float* query,
    const void* key,const void* value,float* scores,float* output,const float* mask,
    uint32_t* valid,float* value_fp32,float* pv_output,float** pointers,
    const SdpaGemmPrefillParams& p,int query_tile,int head_lanes,
    cudaStream_t stream,cudaError_t* launch_error) {
    if(launch_error)*launch_error=cudaSuccess;
    const auto fail=[&](cudaError_t error){if(launch_error)*launch_error=error;return false;};
    if(!handle||!query||!key||!value||!scores||!output||!mask||!valid||!value_fp32||!pv_output||!pointers||
       query_tile<256||query_tile>2048||p.queries<2||p.queries>2048||p.heads<1||p.heads>64||
       p.kv_heads<1||p.heads%p.kv_heads||p.dimension!=256||p.keys<p.queries||p.keys>16384||
       p.capacity<p.keys||p.capacity>32768||p.past<0||p.past>p.keys-p.queries||
       p.query_position_stride<256||p.query_head_stride<256||p.output_position_stride<256||
       p.output_position_stride>INT32_MAX||p.output_head_stride<256||!p.mask_column_stride||!p.mask_row_stride||
       p.output_head_stride>SIZE_MAX/p.heads/8||p.output_position_stride>SIZE_MAX/p.queries/8||
       (p.heads>1&&p.output_head_stride<size_t(p.queries)*p.output_position_stride&&
        p.output_position_stride<size_t(p.heads)*p.output_head_stride)||
       (head_lanes!=1&&head_lanes!=2)||reinterpret_cast<uintptr_t>(pointers)%alignof(float*))return false;
    const auto span=[](int a,int b,size_t sa,size_t sb,size_t tail,size_t unit,size_t& bytes) {
        const size_t limit=SIZE_MAX/unit-tail;
        if(sa&&size_t(a-1)>limit/sa)return false;
        const size_t first=size_t(a-1)*sa;
        if(sb&&size_t(b-1)>(limit-first)/sb)return false;
        bytes=(first+size_t(b-1)*sb+tail)*unit;return true;
    };
    const void* buffers[]={query,key,value,mask,scores,output,valid,value_fp32,pv_output,pointers};
    size_t bytes[10]{};
    if(!span(p.heads,p.queries,p.query_head_stride,p.query_position_stride,256,4,bytes[0])||
       !span(p.queries,p.keys,p.mask_row_stride,p.mask_column_stride,1,4,bytes[3])||
       !span(p.heads,p.queries,p.output_head_stride,p.output_position_stride,256,4,bytes[5]))return false;
    bytes[1]=bytes[2]=size_t(p.kv_heads)*p.capacity*256*2;
    bytes[4]=size_t(p.heads)*std::min(query_tile,p.queries)*p.keys*4;
    bytes[6]=4;bytes[7]=size_t(p.kv_heads)*p.keys*256*4;bytes[8]=bytes[5];bytes[9]=size_t(p.heads)*3*sizeof(float*);
    for(int i=0;i<10;++i) {
        const uintptr_t current=reinterpret_cast<uintptr_t>(buffers[i]);
        if(i>=3&&i<=8&&(current&3))return false;
        if(bytes[i]>UINTPTR_MAX-current)return false;
        for(int j=0;j<i;++j) {
            const uintptr_t previous=reinterpret_cast<uintptr_t>(buffers[j]);
            if((i>=4||j>=4)&&current<previous+bytes[j]&&previous<current+bytes[i])return false;
        }
    }
    cudaStream_t hs;cublasPointerMode_t mode;
    if(cublasGetStream(handle,&hs)!=CUBLAS_STATUS_SUCCESS||hs!=stream||
       cublasGetPointerMode(handle,&mode)!=CUBLAS_STATUS_SUCCESS||mode!=CUBLAS_POINTER_MODE_HOST)return false;
    if(!sdpa_prefill_prefix_phase(query,key,value,scores,output,mask,valid,p,stream,launch_error,
       1024,true,false,true,false,2,false,false,true,head_lanes,true,SdpaPrefixPhase::VALIDATE_ONLY))return false;
    for(int offset=0;offset<p.queries;) {
        int rows=std::min(query_tile,p.queries-offset);
        // Avoid tiny-tail cuBLAS variants, which can use a different K-sum
        // order even in pedantic mode. Splitting never creates a tail <128.
        const int tail=p.queries-offset-rows;
        if(tail>0&&tail<128)rows-=128-tail;
        auto part=p;part.queries=rows;part.past+=offset;
        const auto* part_query=query+size_t(offset)*p.query_position_stride;
        const auto* part_mask=mask+size_t(offset)*p.mask_row_stride;
        auto* part_output=output+size_t(offset)*p.output_position_stride;
        auto* part_pv=pv_output+size_t(offset)*p.output_position_stride;
        if(!sdpa_prefill_prefix_phase(part_query,key,value,scores,part_output,part_mask,valid,part,
           stream,launch_error,1024,true,false,true,false,2,false,false,true,head_lanes,true,
           SdpaPrefixPhase::EXECUTE_VALIDATED))return false;
        const auto status=sdpa_prefill_fp32_pv_impl(handle,value,scores,part_pv,value_fp32,pointers,part,stream,offset!=0);
        if(status!=CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr,"query-sliced PV cuBLAS status=%d\n",int(status));return fail(cudaErrorUnknown);
        }
        launch_sdpa_validated_pv_copy(part_pv,part_output,valid,part,stream);
        const auto copied=cudaGetLastError();if(copied!=cudaSuccess)return fail(copied);
        offset+=rows;
    }
    return true;
}

bool try_launch_sdpa_decode_fp16_cached(
    const float* query, const float* current_key,
    const float* current_value, void* key_cache, void* value_cache,
    float* scores, float* output, const float* mask, int num_heads,
    int num_kv_heads, int key_length, int past_length, int key_dim,
    int value_dim, int key_capacity, bool causal, float scale,
    size_t query_feature_stride, size_t query_head_stride,
    size_t current_key_feature_stride, size_t current_key_head_stride,
    size_t current_value_feature_stride, size_t current_value_head_stride,
    size_t mask_column_stride, size_t output_feature_stride,
    size_t output_head_stride) {
    const bool supported = query && current_key && current_value &&
        key_cache && value_cache && scores && output && num_heads > 0 &&
        num_kv_heads > 0 && num_heads % num_kv_heads == 0 &&
        key_length == past_length + 1 &&
        key_dim == staged_sdpa_head_dim &&
        value_dim == staged_sdpa_head_dim && query_feature_stride == 1 &&
        current_key_feature_stride == 1 &&
        current_value_feature_stride == 1 && output_feature_stride == 1 &&
        query_head_stride % 2 == 0 && current_key_head_stride % 2 == 0 &&
        current_value_head_stride % 2 == 0 &&
        (reinterpret_cast<uintptr_t>(query) &
         (alignof(float2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(current_key) &
         (alignof(float2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(current_value) &
         (alignof(float2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(key_cache) &
         (alignof(__half2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(value_cache) &
         (alignof(__half2) - 1)) == 0;
    if (!supported)
        return false;

    constexpr int keys_per_score_block =
        staged_sdpa_threads / staged_sdpa_score_lanes_per_key;
    const dim3 score_grid(
        static_cast<unsigned>(num_heads),
        static_cast<unsigned>(
            (key_length + keys_per_score_block - 1) /
            keys_per_score_block));
    sdpa_decode_scores_fp16_cuda<true>
        <<<score_grid, staged_sdpa_threads>>>(
            query, current_key, current_value,
            static_cast<const __half*>(key_cache),
            static_cast<__half*>(key_cache),
            static_cast<__half*>(value_cache), scores, mask, num_heads,
            num_kv_heads, key_length, past_length, key_capacity, causal,
            scale, query_head_stride, current_key_head_stride,
            current_value_head_stride, mask_column_stride);
    sdpa_decode_softmax_cuda<staged_sdpa_threads>
        <<<num_heads, staged_sdpa_threads>>>(scores, key_length);
    const dim3 value_grid(
        static_cast<unsigned>(num_heads),
        staged_sdpa_pairs / staged_sdpa_value_pairs_per_block);
    sdpa_decode_value_fp16_cuda<<<value_grid, staged_sdpa_threads>>>(
        scores, static_cast<const __half*>(value_cache), output, num_heads,
        num_kv_heads, key_length, key_capacity, output_head_stride);
    return true;
}

void launch_sdpa_decode(
    const float* query, const void* key, const void* value, float* scores,
    float* output, const float* mask, int num_heads, int num_kv_heads,
    int key_length, int past_length, int key_dim, int value_dim,
    int key_capacity, bool cached, bool fp16_cache, bool causal, float scale,
    size_t query_feature_stride, size_t query_head_stride,
    size_t key_feature_stride, size_t key_position_stride,
    size_t key_head_stride, size_t value_feature_stride,
    size_t value_position_stride, size_t value_head_stride,
    size_t mask_column_stride, size_t output_feature_stride,
    size_t output_head_stride) {
    const bool staged_fp16_128 = cached && fp16_cache &&
        key_dim == staged_sdpa_head_dim &&
        value_dim == staged_sdpa_head_dim && query_feature_stride == 1 &&
        query_head_stride % 2 == 0 && output_feature_stride == 1 &&
        (reinterpret_cast<uintptr_t>(query) &
         (alignof(float2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(key) &
         (alignof(__half2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(value) &
         (alignof(__half2) - 1)) == 0;
    if (staged_fp16_128) {
        constexpr int keys_per_score_block =
            staged_sdpa_threads / staged_sdpa_score_lanes_per_key;
        const dim3 score_grid(
            static_cast<unsigned>(num_heads),
            static_cast<unsigned>(
                (key_length + keys_per_score_block - 1) /
                keys_per_score_block));
        sdpa_decode_scores_fp16_cuda<false>
            <<<score_grid, staged_sdpa_threads>>>(
                query, nullptr, nullptr, static_cast<const __half*>(key),
                nullptr, nullptr, scores, mask, num_heads, num_kv_heads,
                key_length, past_length, key_capacity, causal, scale,
                query_head_stride, 0, 0, mask_column_stride);
        sdpa_decode_softmax_cuda<staged_sdpa_threads>
            <<<num_heads, staged_sdpa_threads>>>(scores, key_length);
        const dim3 value_grid(
            static_cast<unsigned>(num_heads),
            staged_sdpa_pairs / staged_sdpa_value_pairs_per_block);
        sdpa_decode_value_fp16_cuda<<<value_grid, staged_sdpa_threads>>>(
            scores, static_cast<const __half*>(value), output, num_heads,
            num_kv_heads, key_length, key_capacity, output_head_stride);
        return;
    }

    // A decode head owns only one block, so expose as many independent key
    // dot products as practical. 512 threads avoids an unnecessary reduction
    // stage for short contexts; longer contexts benefit from the 1024-thread
    // specialization despite its larger block.
    if (key_length <= 512) {
        sdpa_decode_cuda<512, true><<<num_heads, 512>>>(
            query, key, value, scores, output, mask, num_heads, num_kv_heads,
            key_length, past_length, key_dim, value_dim, key_capacity,
            cached, fp16_cache, causal, scale, query_feature_stride,
            query_head_stride, key_feature_stride, key_position_stride,
            key_head_stride, value_feature_stride, value_position_stride,
            value_head_stride, mask_column_stride, output_feature_stride,
            output_head_stride);
    } else {
        sdpa_decode_cuda<1024, false><<<num_heads, 1024>>>(
            query, key, value, scores, output, mask, num_heads, num_kv_heads,
            key_length, past_length, key_dim, value_dim, key_capacity,
            cached, fp16_cache, causal, scale, query_feature_stride,
            query_head_stride, key_feature_stride, key_position_stride,
            key_head_stride, value_feature_stride, value_position_stride,
            value_head_stride, mask_column_stride, output_feature_stride,
            output_head_stride);
    }
}

}  // namespace mollm_cuda
