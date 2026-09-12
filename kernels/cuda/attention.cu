#include "kernels/cuda/attention.h"
#include "kernels/cuda/reduction.cuh"

#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
#include <cmath>

namespace {

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
template <int Threads, bool SharedScores>
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
            const float dot = query_key_dot(
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

template <int HeadDim, bool SharedScores>
__global__ void sdpa_prefill_fp16_cuda(
    const float* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, float* __restrict__ scores,
    float* __restrict__ output, const float* __restrict__ mask, int num_heads,
    int num_kv_heads, int query_length, int key_length, int past_length,
    int key_capacity, bool causal, float scale,
    size_t query_position_stride, size_t query_head_stride,
    size_t mask_column_stride, size_t mask_row_stride,
    size_t output_position_stride, size_t output_head_stride) {
    static_assert(HeadDim == 128 || HeadDim == 256);
    constexpr int pairs = HeadDim / 2;
    // Keep sixteen FP16 pairs per lane for both supported widths. This exposes
    // enough keys per block while preserving a fixed reduction tree.
    constexpr int lanes_per_key =
        HeadDim == 256 ? 8 : staged_sdpa_score_lanes_per_key;
    constexpr int keys_per_iteration = prefill_sdpa_threads / lanes_per_key;
    const int query_position = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int key_head = head / (num_heads / num_kv_heads);
    const int lane = static_cast<int>(threadIdx.x) % lanes_per_key;
    const int first_key = static_cast<int>(threadIdx.x) / lanes_per_key;
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
    for (int key_position = first_key; key_position < key_length;
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
            if (mask)
                score += mask[
                    static_cast<size_t>(query_position) * mask_row_stride +
                    static_cast<size_t>(key_position) * mask_column_stride];
            else if (causal &&
                     key_position > past_length + query_position)
                score = -FLT_MAX;
            score_row[key_position] = score;
            local_maximum = fmaxf(local_maximum, score);
        }
    }

    __shared__ float reduction[prefill_sdpa_threads];
    const float maximum =
        mollm_cuda::detail::block_reduce_max<prefill_sdpa_threads>(
            local_maximum, reduction);
    float local_sum = 0.0f;
    for (int key_position = threadIdx.x; key_position < key_length;
         key_position += prefill_sdpa_threads) {
        const float numerator = expf(score_row[key_position] - maximum);
        score_row[key_position] = numerator;
        local_sum += numerator;
    }
    const float block_sum =
        mollm_cuda::detail::block_reduce_sum<prefill_sdpa_threads>(
            local_sum, reduction);
    const float inverse_sum = block_sum > 0.0f
        ? 1.0f / block_sum : 0.0f;
    for (int key_position = threadIdx.x; key_position < key_length;
         key_position += prefill_sdpa_threads)
        score_row[key_position] *= inverse_sum;
    __syncthreads();

    if constexpr (HeadDim == 256) {
        const int dimension = static_cast<int>(threadIdx.x);
        float result = 0.0f;
        for (int key_position = 0; key_position < key_length;
             ++key_position) {
            const size_t value_index =
                (static_cast<size_t>(key_head) * key_capacity +
                 key_position) * HeadDim + dimension;
            result = fmaf(
                score_row[key_position], __half2float(value[value_index]),
                result);
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

}  // namespace

namespace mollm_cuda {

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
    size_t output_head_stride) {
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
        if (key_dim == 256)
            sdpa_prefill_fp16_cuda<256, true>
                <<<grid, prefill_sdpa_threads, shared_bytes>>>(
                    query, static_cast<const __half*>(key),
                    static_cast<const __half*>(value), scores, output, mask,
                    num_heads, num_kv_heads, query_length, key_length,
                    past_length, key_capacity, causal, scale,
                    query_position_stride, query_head_stride,
                    mask_column_stride, mask_row_stride,
                    output_position_stride, output_head_stride);
        else
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
        if (key_dim == 256)
            sdpa_prefill_fp16_cuda<256, false>
                <<<grid, prefill_sdpa_threads>>>(
                    query, static_cast<const __half*>(key),
                    static_cast<const __half*>(value), scores, output, mask,
                    num_heads, num_kv_heads, query_length, key_length,
                    past_length, key_capacity, causal, scale,
                    query_position_stride, query_head_stride,
                    mask_column_stride, mask_row_stride,
                    output_position_stride, output_head_stride);
        else
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
