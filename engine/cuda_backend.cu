#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "engine/cuda_backend.h"
#include "engine/engine.h"
#include "graph/mmap_file.h"

#include "kernels/activations.h"
#include "kernels/moe_ssd.h"
#include "kernels/quant_layouts.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct CudaTensorLayout {
    int64_t shape[4];
    size_t stride[4];
};

__device__ void logical_coordinates(size_t index,
                                    const CudaTensorLayout& layout,
                                    size_t coordinates[4]) {
    for (int dimension = 0; dimension < 4; ++dimension) {
        coordinates[dimension] =
            index % static_cast<size_t>(layout.shape[dimension]);
        index /= static_cast<size_t>(layout.shape[dimension]);
    }
}

__device__ size_t layout_offset(const size_t coordinates[4],
                                const CudaTensorLayout& layout,
                                bool broadcast) {
    size_t offset = 0;
    for (int dimension = 0; dimension < 4; ++dimension) {
        const size_t coordinate = broadcast && layout.shape[dimension] == 1
            ? 0 : coordinates[dimension];
        offset += coordinate * layout.stride[dimension];
    }
    return offset;
}

const char* cuda_error(cudaError_t error) {
    return cudaGetErrorString(error);
}

bool report_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess)
        return true;
    std::fprintf(stderr, "CudaBackend: %s failed: %s\n", operation,
                 cuda_error(error));
    return false;
}

bool report_cublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return true;
    std::fprintf(stderr, "CudaBackend: %s failed (cuBLAS status %d)\n",
                 operation, static_cast<int>(status));
    return false;
}

__global__ void fp32_to_fp16(const float* source, __half* destination,
                             size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        destination[index] = __float2half(source[index]);
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

__device__ float cuda_round_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__device__ float cuda_decode_e8m0(uint8_t value) {
    if (value == 0xff)
        return __uint_as_float(0x7fffffffu);
    const uint32_t bits = value == 0
        ? (1u << 22) : static_cast<uint32_t>(value) << 23;
    return __uint_as_float(bits);
}

__device__ float cuda_decode_fp8_e4m3fn(uint8_t value) {
    const bool negative = (value & 0x80u) != 0;
    const int exponent = (value >> 3) & 0x0f;
    const int mantissa = value & 0x07;
    float decoded = 0.0f;
    if (exponent == 0x0f && mantissa == 0x07)
        decoded = __uint_as_float(0x7fffffffu);
    else if (exponent == 0)
        decoded = ldexpf(static_cast<float>(mantissa), -9);
    else
        decoded = ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                         exponent - 7);
    return negative ? -decoded : decoded;
}

__device__ uint8_t cuda_encode_fp8_e4m3fn(float value) {
    if (isnan(value))
        return 0x7f;
    const uint8_t sign = signbit(value) ? 0x80u : 0u;
    const float magnitude = fminf(fabsf(value), 448.0f);
    if (magnitude < (1.0f / 64.0f)) {
        const int code = max(0, min(8,
            static_cast<int>(nearbyintf(magnitude * 512.0f))));
        return static_cast<uint8_t>(sign | code);
    }
    const uint32_t bits = __float_as_uint(magnitude);
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
    int significand = static_cast<int>(
        nearbyintf(ldexpf(magnitude, 3 - exponent)));
    if (significand == 16) {
        ++exponent;
        significand = 8;
    }
    const int code = max(0, min(126, (exponent + 6) * 8 + significand));
    return static_cast<uint8_t>(sign | code);
}

__device__ float cuda_decode_mxfp4_e2m1(uint8_t value) {
    constexpr float magnitudes[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = magnitudes[value & 0x07u];
    return value & 0x08u ? -magnitude : magnitude;
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

__global__ void round_bf16_cuda(float* values, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        values[index] = cuda_round_to_bf16(values[index]);
}

__global__ void quantize_activation_fp8_cuda(
    const float* input, int lda, float* output, int rows, int inner) {
    const int block = blockIdx.x;
    const int blocks_per_row = (inner + 127) / 128;
    const int row = block / blocks_per_row;
    const int group = block % blocks_per_row;
    if (row >= rows || threadIdx.x != 0)
        return;
    const int begin = group * 128;
    const int end = min(inner, begin + 128);
    const float* input_row = input + static_cast<size_t>(row) * lda;
    float maximum = 1.0e-4f;
    for (int k = begin; k < end; ++k)
        maximum = fmaxf(maximum, fabsf(input_row[k]));
    const int exponent = static_cast<int>(ceilf(log2f(maximum / 448.0f)));
    const float scale = ldexpf(1.0f, exponent);
    float* output_row = output + static_cast<size_t>(row) * inner;
    for (int k = begin; k < end; ++k) {
        const float scaled = fminf(448.0f,
            fmaxf(-448.0f, input_row[k] / scale));
        output_row[k] = cuda_decode_fp8_e4m3fn(
            cuda_encode_fp8_e4m3fn(scaled)) * scale;
    }
}

__global__ void microscaled_matmul_cuda(
    const float* activation, int lda, const uint8_t* weight,
    const uint8_t* scales, int layout, float* output, int ldc,
    int rows, int columns, int inner, int groups_per_row) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(rows) * columns;
    if (index >= count)
        return;
    const int row = static_cast<int>(index / columns);
    const int column = static_cast<int>(index % columns);
    const float* input = activation + static_cast<size_t>(row) * lda;
    float sum = 0.0f;
    // Layout 5 is row-major E4M3 with one E8M0 scale per 128x128 tile.
    if (layout == 5) {
        const uint8_t* weight_row =
            weight + static_cast<size_t>(column) * inner;
        for (int group = 0; group < groups_per_row; ++group) {
            const int begin = group * 128;
            const int end = min(inner, begin + 128);
            float block = 0.0f;
            for (int k = begin; k < end; ++k)
                block += input[k] *
                    cuda_decode_fp8_e4m3fn(weight_row[k]);
            sum += block * cuda_decode_e8m0(
                scales[static_cast<size_t>(column / 128) *
                           groups_per_row + group]);
        }
    } else {
        // Layout 6 is row-major packed E2M1 with one E8M0 scale per K32.
        const uint8_t* weight_row =
            weight + static_cast<size_t>(column) * (inner / 2);
        for (int group = 0; group < groups_per_row; ++group) {
            float block = 0.0f;
            const int begin = group * 32;
            for (int k = begin; k < begin + 32; ++k) {
                const uint8_t packed = weight_row[k / 2];
                const uint8_t nibble = k & 1 ? packed >> 4
                                              : packed & 0x0f;
                block += input[k] * cuda_decode_mxfp4_e2m1(nibble);
            }
            sum += block * cuda_decode_e8m0(
                scales[static_cast<size_t>(column) * groups_per_row + group]);
        }
    }
    output[static_cast<size_t>(row) * ldc + column] = sum;
}

__global__ void dsv4_grouped_fp8_linear_cuda(
    const float* input, size_t input_stride, const uint8_t* weight,
    const uint8_t* scales, float* output, size_t output_stride,
    int tokens, int groups, int group_width, int rank,
    int k_blocks) {
    const int output_width = groups * rank;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(tokens) * output_width;
    if (index >= count)
        return;
    const int token = static_cast<int>(index / output_width);
    const int row = static_cast<int>(index % output_width);
    const int group = row / rank;
    const float* group_input = input +
        static_cast<size_t>(token) * input_stride +
        static_cast<size_t>(group) * group_width;
    const uint8_t* weight_row =
        weight + static_cast<size_t>(row) * group_width;
    float sum = 0.0f;
    for (int k = 0; k < group_width; ++k) {
        const float weight_value = cuda_round_to_bf16(
            cuda_decode_fp8_e4m3fn(weight_row[k]) *
            cuda_decode_e8m0(
                scales[static_cast<size_t>(row / 128) * k_blocks +
                       k / 128]));
        sum += group_input[k] * weight_value;
    }
    output[static_cast<size_t>(token) * output_stride + row] = sum;
}

__device__ float dsv4_yarn_frequency_cuda(
    int pair, int rope_dim, float theta, int original_context,
    float factor, float beta_fast, float beta_slow) {
    constexpr float pi = 3.14159265358979323846f;
    float frequency = 1.0f / powf(
        theta, 2.0f * static_cast<float>(pair) /
                   static_cast<float>(rope_dim));
    if (original_context <= 0 || factor <= 1.0f)
        return frequency;
    const float fast_dim = static_cast<float>(rope_dim) *
        logf(static_cast<float>(original_context) /
             (beta_fast * 2.0f * pi)) /
        (2.0f * logf(theta));
    const float slow_dim = static_cast<float>(rope_dim) *
        logf(static_cast<float>(original_context) /
             (beta_slow * 2.0f * pi)) /
        (2.0f * logf(theta));
    const int low = max(0, static_cast<int>(floorf(fast_dim)));
    const int high = min(
        rope_dim - 1, static_cast<int>(ceilf(slow_dim)));
    const float denominator = high == low
        ? 0.001f : static_cast<float>(high - low);
    const float ramp = fminf(1.0f, fmaxf(
        0.0f, (static_cast<float>(pair) - low) / denominator));
    const float smooth = 1.0f - ramp;
    return frequency / factor * (1.0f - smooth) + frequency * smooth;
}

__device__ void dsv4_apply_rope_cuda(
    float* values, int width, int position, int rope_dim, float theta,
    int original_context, float factor, float beta_fast, float beta_slow) {
    if (rope_dim <= 0 || rope_dim > width || (rope_dim & 1))
        return;
    float* rotary = values + width - rope_dim;
    for (int pair = 0; pair < rope_dim / 2; ++pair) {
        const float angle = static_cast<float>(position) *
            dsv4_yarn_frequency_cuda(
                pair, rope_dim, theta, original_context, factor,
                beta_fast, beta_slow);
        float sine = 0.0f;
        float cosine = 0.0f;
        sincosf(angle, &sine, &cosine);
        const float real = rotary[pair * 2];
        const float imaginary = rotary[pair * 2 + 1];
        rotary[pair * 2] = real * cosine - imaginary * sine;
        rotary[pair * 2 + 1] = real * sine + imaginary * cosine;
    }
}

__device__ void dsv4_hadamard_rotate_cuda(float* values, int width) {
    if (!values || width <= 0 || (width & (width - 1)) != 0)
        return;
    for (int span = 1; span < width; span *= 2) {
        for (int base = 0; base < width; base += span * 2) {
            for (int offset = 0; offset < span; ++offset) {
                const float left = values[base + offset];
                const float right = values[base + span + offset];
                values[base + offset] = left + right;
                values[base + span + offset] = left - right;
            }
        }
    }
    const float scale = rsqrtf(static_cast<float>(width));
    for (int index = 0; index < width; ++index)
        values[index] *= scale;
}

__device__ float dsv4_nearest_e2m1_cuda(float value) {
    constexpr float magnitudes[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float sign = signbit(value) ? -1.0f : 1.0f;
    const float magnitude = fabsf(value);
    float nearest = magnitudes[0];
    float distance = fabsf(magnitude - nearest);
    int nearest_code = 0;
    for (int code = 1; code < 8; ++code) {
        const float candidate_distance =
            fabsf(magnitude - magnitudes[code]);
        if (candidate_distance < distance ||
            (candidate_distance == distance && (code & 1) == 0 &&
             (nearest_code & 1) != 0)) {
            distance = candidate_distance;
            nearest = magnitudes[code];
            nearest_code = code;
        }
    }
    return sign * nearest;
}

__device__ void dsv4_fp4_simulate_cuda(
    float* values, int width, int group_size) {
    if (!values || width <= 0 || group_size <= 0 ||
        width % group_size != 0)
        return;
    for (int begin = 0; begin < width; begin += group_size) {
        float maximum = 0.0f;
        for (int index = 0; index < group_size; ++index)
            maximum = fmaxf(maximum, fabsf(values[begin + index]));
        maximum = fmaxf(maximum, 6.0f * ldexpf(1.0f, -126));
        const float scale = exp2f(ceilf(log2f(maximum / 6.0f)));
        for (int index = 0; index < group_size; ++index) {
            const float normalized = fminf(
                6.0f, fmaxf(-6.0f, values[begin + index] / scale));
            values[begin + index] =
                dsv4_nearest_e2m1_cuda(normalized) * scale;
        }
    }
}

__device__ void dsv4_fp8_simulate_cuda(
    float* values, int width, int group_size) {
    if (!values || width <= 0 || group_size <= 0 ||
        width % group_size != 0)
        return;
    for (int begin = 0; begin < width; begin += group_size) {
        float maximum = 0.0f;
        for (int index = 0; index < group_size; ++index)
            maximum = fmaxf(maximum, fabsf(values[begin + index]));
        maximum = fmaxf(maximum, 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(maximum / 448.0f)));
        for (int index = 0; index < group_size; ++index) {
            const float normalized = fminf(
                448.0f,
                fmaxf(-448.0f, values[begin + index] / scale));
            values[begin + index] = cuda_decode_fp8_e4m3fn(
                cuda_encode_fp8_e4m3fn(normalized)) * scale;
        }
    }
}

__global__ void dsv4_compressor_state_cuda(
    const float* kv_values, const float* gate_values, const float* ape,
    const float* norm, float* kv_state, float* score_state, float* cache,
    const int32_t* start_position, const int32_t* real_tokens,
    float* compressed, float* emitted_output, int sequence,
    int head_dim, int ratio, bool overlap, bool rotate,
    int cache_capacity, int rope_dim, int original_context, float norm_eps,
    float rope_theta, float rope_factor, float beta_fast, float beta_slow) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const int start_pos = start_position ? start_position[0] : 0;
    const int tokens = real_tokens
        ? min(sequence, max(real_tokens[0], 0)) : sequence;
    const int coff = overlap ? 2 : 1;
    const int projected = coff * head_dim;
    const int state_rows = coff * ratio;
    if (start_pos < 0 || tokens <= 0) {
        emitted_output[0] = -1.0f;
        return;
    }
    if (start_pos == 0) {
        for (int index = 0; index < projected * state_rows; ++index) {
            kv_state[index] = 0.0f;
            score_state[index] = -INFINITY;
        }
        for (int index = 0; index < head_dim * cache_capacity; ++index)
            cache[index] = 0.0f;
    }
    int emitted = 0;
    for (int token = 0; token < tokens; ++token) {
        const int position = start_pos + token;
        const int within = position % ratio;
        const int destination_row = overlap ? ratio + within : within;
        const float* token_kv =
            kv_values + static_cast<size_t>(token) * projected;
        const float* token_gate =
            gate_values + static_cast<size_t>(token) * projected;
        const float* position_ape =
            ape + static_cast<size_t>(within) * projected;
        float* state_kv =
            kv_state + static_cast<size_t>(destination_row) * projected;
        float* state_score =
            score_state + static_cast<size_t>(destination_row) * projected;
        for (int dim = 0; dim < projected; ++dim) {
            state_kv[dim] = token_kv[dim];
            state_score[dim] = token_gate[dim] + position_ape[dim];
        }
        if ((position + 1) % ratio != 0)
            continue;
        const int cache_index = position / ratio;
        if (cache_index < 0 || cache_index >= cache_capacity) {
            emitted_output[0] = -1.0f;
            return;
        }
        const int row_count = overlap ? ratio * 2 : ratio;
        for (int dim = 0; dim < head_dim; ++dim) {
            float maximum = -INFINITY;
            for (int row = 0; row < row_count; ++row) {
                const int source_dim = overlap && row >= ratio
                    ? head_dim + dim : dim;
                maximum = fmaxf(
                    maximum,
                    score_state[static_cast<size_t>(row) * projected +
                                source_dim]);
            }
            float denominator = 0.0f;
            float numerator = 0.0f;
            for (int row = 0; row < row_count; ++row) {
                const int source_dim = overlap && row >= ratio
                    ? head_dim + dim : dim;
                const size_t index =
                    static_cast<size_t>(row) * projected + source_dim;
                const float probability = expf(score_state[index] - maximum);
                denominator += probability;
                numerator += probability * kv_state[index];
            }
            compressed[dim] = cuda_round_to_bf16(
                denominator > 0.0f ? numerator / denominator : 0.0f);
        }
        float mean_square = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim)
            mean_square += compressed[dim] * compressed[dim];
        mean_square /= static_cast<float>(head_dim);
        const float inverse = rsqrtf(mean_square + norm_eps);
        for (int dim = 0; dim < head_dim; ++dim)
            compressed[dim] = cuda_round_to_bf16(
                compressed[dim] * inverse * norm[dim]);
        const int compressed_position = position + 1 - ratio;
        dsv4_apply_rope_cuda(
            compressed, head_dim, compressed_position, rope_dim, rope_theta,
            original_context, rope_factor, beta_fast, beta_slow);
        for (int dim = 0; dim < head_dim; ++dim)
            compressed[dim] = cuda_round_to_bf16(compressed[dim]);
        if (rotate) {
            dsv4_hadamard_rotate_cuda(compressed, head_dim);
            for (int dim = 0; dim < head_dim; ++dim)
                compressed[dim] = cuda_round_to_bf16(compressed[dim]);
            dsv4_fp4_simulate_cuda(compressed, head_dim, 32);
        } else {
            const int non_rotary = head_dim - rope_dim;
            if (non_rotary > 0)
                dsv4_fp8_simulate_cuda(compressed, non_rotary, 64);
        }
        for (int dim = 0; dim < head_dim; ++dim)
            cache[static_cast<size_t>(cache_index) * head_dim + dim] =
                compressed[dim];
        ++emitted;
        if (overlap) {
            const int half = ratio * projected;
            for (int index = 0; index < half; ++index) {
                kv_state[index] = kv_state[half + index];
                score_state[index] = score_state[half + index];
                score_state[half + index] = -INFINITY;
            }
        }
    }
    emitted_output[0] = static_cast<float>(emitted);
}

__global__ void dsv4_indexer_prepare_cuda(
    float* query, float* head_weights, const int32_t* start_position,
    const int32_t* real_tokens, int sequence, int num_heads, int head_dim,
    int rope_dim, int original_context, float rope_theta,
    float rope_factor, float beta_fast, float beta_slow) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const int start_pos = start_position ? start_position[0] : 0;
    const int tokens = real_tokens
        ? min(sequence, max(real_tokens[0], 0)) : sequence;
    if (start_pos < 0 || tokens <= 0)
        return;
    const int query_width = num_heads * head_dim;
    const float weight_scale = 1.0f /
        sqrtf(static_cast<float>(head_dim * num_heads));
    for (int token = 0; token < tokens; ++token) {
        float* token_query =
            query + static_cast<size_t>(token) * query_width;
        float* token_weights =
            head_weights + static_cast<size_t>(token) * num_heads;
        for (int index = 0; index < query_width; ++index)
            token_query[index] = cuda_round_to_bf16(token_query[index]);
        for (int head = 0; head < num_heads; ++head) {
            token_weights[head] = cuda_round_to_bf16(
                cuda_round_to_bf16(token_weights[head]) * weight_scale);
            float* head_query =
                token_query + static_cast<size_t>(head) * head_dim;
            dsv4_apply_rope_cuda(
                head_query, head_dim, start_pos + token, rope_dim,
                rope_theta, original_context, rope_factor, beta_fast,
                beta_slow);
            for (int dimension = 0; dimension < head_dim; ++dimension)
                head_query[dimension] =
                    cuda_round_to_bf16(head_query[dimension]);
            dsv4_hadamard_rotate_cuda(head_query, head_dim);
            for (int dimension = 0; dimension < head_dim; ++dimension)
                head_query[dimension] =
                    cuda_round_to_bf16(head_query[dimension]);
            dsv4_fp4_simulate_cuda(head_query, head_dim, 32);
        }
    }
}

__global__ void dsv4_indexer_scores_cuda(
    const float* query, const float* head_weights, const float* cache,
    const int32_t* start_position, const int32_t* real_tokens,
    float* scores, int sequence, int num_heads, int head_dim, int ratio,
    int cache_capacity) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(sequence) * cache_capacity;
    if (index >= count)
        return;
    const int token = static_cast<int>(index / cache_capacity);
    const int cache_index = static_cast<int>(index % cache_capacity);
    const int start_pos = start_position ? start_position[0] : 0;
    const int tokens = real_tokens
        ? min(sequence, max(real_tokens[0], 0)) : sequence;
    const int available = token < tokens && start_pos >= 0
        ? min(cache_capacity, (start_pos + token + 1) / ratio) : 0;
    if (cache_index >= available) {
        scores[index] = -INFINITY;
        return;
    }
    const int query_width = num_heads * head_dim;
    const float* token_query =
        query + static_cast<size_t>(token) * query_width;
    const float* token_weights =
        head_weights + static_cast<size_t>(token) * num_heads;
    const float* key =
        cache + static_cast<size_t>(cache_index) * head_dim;
    float score = 0.0f;
    for (int head = 0; head < num_heads; ++head) {
        float dot = 0.0f;
        const float* head_query =
            token_query + static_cast<size_t>(head) * head_dim;
        for (int dimension = 0; dimension < head_dim; ++dimension)
            dot += head_query[dimension] * key[dimension];
        dot = cuda_round_to_bf16(dot);
        score += cuda_round_to_bf16(
            fmaxf(dot, 0.0f) * token_weights[head]);
    }
    scores[index] = cuda_round_to_bf16(score);
}

__global__ void dsv4_indexer_select_cuda(
    float* scores, const int32_t* start_position,
    const int32_t* real_tokens, int32_t* indices, int sequence,
    int ratio, int cache_capacity, int top_k) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= sequence || threadIdx.x != 0)
        return;
    const int start_pos = start_position ? start_position[0] : 0;
    const int tokens = real_tokens
        ? min(sequence, max(real_tokens[0], 0)) : sequence;
    if (token >= tokens || start_pos < 0)
        return;
    const int available = min(
        cache_capacity, (start_pos + token + 1) / ratio);
    const int selected = min(top_k, available);
    float* token_scores =
        scores + static_cast<size_t>(token) * cache_capacity;
    int32_t* token_indices =
        indices + static_cast<size_t>(token) * top_k;
    for (int rank = 0; rank < selected; ++rank) {
        int best = -1;
        float best_score = -INFINITY;
        for (int candidate = 0; candidate < available; ++candidate) {
            const float candidate_score = token_scores[candidate];
            if (isnan(candidate_score))
                continue;
            if (best < 0 || candidate_score > best_score) {
                best = candidate;
                best_score = candidate_score;
            }
        }
        if (best < 0)
            break;
        token_indices[rank] = best;
        token_scores[best] = NAN;
    }
}

__global__ void dsv4_sparse_prepare_cuda(
    const float* query_input, const float* kv_input,
    const int32_t* start_position, const int32_t* real_tokens,
    float* query, float* kv, int sequence, int num_heads, int head_dim,
    int rope_dim, int original_context, float query_norm_eps,
    float rope_theta, float rope_factor, float beta_fast, float beta_slow) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const int start_pos = start_position ? start_position[0] : 0;
    const int tokens = real_tokens
        ? min(sequence, max(real_tokens[0], 0)) : sequence;
    if (start_pos < 0 || tokens <= 0)
        return;
    const int query_width = num_heads * head_dim;
    for (int token = 0; token < tokens; ++token) {
        const int position = start_pos + token;
        float* token_kv = kv + static_cast<size_t>(token) * head_dim;
        const float* source_kv =
            kv_input + static_cast<size_t>(token) * head_dim;
        for (int dimension = 0; dimension < head_dim; ++dimension)
            token_kv[dimension] = source_kv[dimension];
        dsv4_apply_rope_cuda(
            token_kv, head_dim, position, rope_dim, rope_theta,
            original_context, rope_factor, beta_fast, beta_slow);
        for (int dimension = 0; dimension < head_dim; ++dimension)
            token_kv[dimension] = cuda_round_to_bf16(token_kv[dimension]);
        dsv4_fp8_simulate_cuda(
            token_kv, head_dim - rope_dim, 64);

        float* token_query =
            query + static_cast<size_t>(token) * query_width;
        const float* source_query =
            query_input + static_cast<size_t>(token) * query_width;
        for (int index = 0; index < query_width; ++index)
            token_query[index] = source_query[index];
        for (int head = 0; head < num_heads; ++head) {
            float* head_query =
                token_query + static_cast<size_t>(head) * head_dim;
            float square_sum = 0.0f;
            for (int dimension = 0; dimension < head_dim; ++dimension) {
                square_sum += cuda_round_to_bf16(
                    head_query[dimension] * head_query[dimension]);
            }
            const float mean = cuda_round_to_bf16(
                square_sum / static_cast<float>(head_dim));
            const float variance = cuda_round_to_bf16(
                mean + query_norm_eps);
            const float inverse = cuda_round_to_bf16(
                1.0f / sqrtf(variance));
            for (int dimension = 0; dimension < head_dim; ++dimension) {
                head_query[dimension] = cuda_round_to_bf16(
                    head_query[dimension] * inverse);
            }
            dsv4_apply_rope_cuda(
                head_query, head_dim, position, rope_dim, rope_theta,
                original_context, rope_factor, beta_fast, beta_slow);
            for (int dimension = 0; dimension < head_dim; ++dimension)
                head_query[dimension] =
                    cuda_round_to_bf16(head_query[dimension]);
        }
    }
}

__device__ float dsv4_sparse_dot_cuda(
    const float* query, const float* key, int head_dim) {
    float sum = 0.0f;
    for (int dimension = 0; dimension < head_dim; ++dimension)
        sum += query[dimension] * key[dimension];
    return sum;
}

__global__ void dsv4_sparse_attention_cuda(
    const float* query, const float* current_kv, const float* sink,
    const float* window, const float* compressed,
    const int32_t* selected, const int32_t* start_position,
    const int32_t* real_tokens, float* output, int sequence,
    int num_heads, int head_dim, int window_size, int compress_ratio,
    int compressed_capacity, int selected_width, float softmax_scale,
    int rope_dim, int original_context, float rope_theta,
    float rope_factor, float beta_fast, float beta_slow) {
    const int task = static_cast<int>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const int tasks = sequence * num_heads;
    if (task >= tasks)
        return;
    const int token = task / num_heads;
    const int head = task % num_heads;
    const int start_pos = start_position ? start_position[0] : 0;
    const int tokens = real_tokens
        ? min(sequence, max(real_tokens[0], 0)) : sequence;
    if (start_pos < 0 || token >= tokens)
        return;
    const int position = start_pos + token;
    const int first_position = max(0, position - window_size + 1);
    const int query_width = num_heads * head_dim;
    const float* head_query =
        query + static_cast<size_t>(token) * query_width +
        static_cast<size_t>(head) * head_dim;
    float maximum = -INFINITY;
    for (int key_position = first_position; key_position <= position;
         ++key_position) {
        const float* key = key_position >= start_pos
            ? current_kv +
                static_cast<size_t>(key_position - start_pos) * head_dim
            : window +
                static_cast<size_t>(key_position % window_size) * head_dim;
        maximum = fmaxf(
            maximum,
            dsv4_sparse_dot_cuda(head_query, key, head_dim) *
                softmax_scale);
    }
    if (compress_ratio > 0 && compressed) {
        if (selected) {
            const int32_t* token_indices =
                selected + static_cast<size_t>(token) * selected_width;
            for (int rank = 0; rank < selected_width; ++rank) {
                const int cache_index = token_indices[rank];
                if (cache_index < 0 || cache_index >= compressed_capacity)
                    continue;
                const float* key = compressed +
                    static_cast<size_t>(cache_index) * head_dim;
                maximum = fmaxf(
                    maximum,
                    dsv4_sparse_dot_cuda(head_query, key, head_dim) *
                        softmax_scale);
            }
        } else {
            const int available = min(
                compressed_capacity, (position + 1) / compress_ratio);
            for (int cache_index = 0; cache_index < available;
                 ++cache_index) {
                const float* key = compressed +
                    static_cast<size_t>(cache_index) * head_dim;
                maximum = fmaxf(
                    maximum,
                    dsv4_sparse_dot_cuda(head_query, key, head_dim) *
                        softmax_scale);
            }
        }
    }

    float* head_output =
        output + static_cast<size_t>(token) * query_width +
        static_cast<size_t>(head) * head_dim;
    float denominator = expf(sink[head] - maximum);
    for (int key_position = first_position; key_position <= position;
         ++key_position) {
        const float* key = key_position >= start_pos
            ? current_kv +
                static_cast<size_t>(key_position - start_pos) * head_dim
            : window +
                static_cast<size_t>(key_position % window_size) * head_dim;
        const float probability = expf(
            dsv4_sparse_dot_cuda(head_query, key, head_dim) *
                softmax_scale - maximum);
        denominator += probability;
        const float value_probability = cuda_round_to_bf16(probability);
        for (int dimension = 0; dimension < head_dim; ++dimension)
            head_output[dimension] += value_probability * key[dimension];
    }
    if (compress_ratio > 0 && compressed) {
        if (selected) {
            const int32_t* token_indices =
                selected + static_cast<size_t>(token) * selected_width;
            for (int rank = 0; rank < selected_width; ++rank) {
                const int cache_index = token_indices[rank];
                if (cache_index < 0 || cache_index >= compressed_capacity)
                    continue;
                const float* key = compressed +
                    static_cast<size_t>(cache_index) * head_dim;
                const float probability = expf(
                    dsv4_sparse_dot_cuda(head_query, key, head_dim) *
                        softmax_scale - maximum);
                denominator += probability;
                const float value_probability =
                    cuda_round_to_bf16(probability);
                for (int dimension = 0; dimension < head_dim; ++dimension)
                    head_output[dimension] +=
                        value_probability * key[dimension];
            }
        } else {
            const int available = min(
                compressed_capacity, (position + 1) / compress_ratio);
            for (int cache_index = 0; cache_index < available;
                 ++cache_index) {
                const float* key = compressed +
                    static_cast<size_t>(cache_index) * head_dim;
                const float probability = expf(
                    dsv4_sparse_dot_cuda(head_query, key, head_dim) *
                        softmax_scale - maximum);
                denominator += probability;
                const float value_probability =
                    cuda_round_to_bf16(probability);
                for (int dimension = 0; dimension < head_dim; ++dimension)
                    head_output[dimension] +=
                        value_probability * key[dimension];
            }
        }
    }
    if (denominator > 0.0f) {
        const float inverse = 1.0f / denominator;
        for (int dimension = 0; dimension < head_dim; ++dimension)
            head_output[dimension] *= inverse;
    }
    for (int dimension = 0; dimension < head_dim; ++dimension)
        head_output[dimension] = cuda_round_to_bf16(
            head_output[dimension]);
    dsv4_apply_rope_cuda(
        head_output, head_dim, -position, rope_dim, rope_theta,
        original_context, rope_factor, beta_fast, beta_slow);
    for (int dimension = 0; dimension < head_dim; ++dimension)
        head_output[dimension] = cuda_round_to_bf16(
            head_output[dimension]);
}

__global__ void dsv4_sparse_update_window_cuda(
    const float* current_kv, const int32_t* start_position,
    const int32_t* real_tokens, float* window, int sequence,
    int head_dim, int window_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const int start_pos = start_position ? start_position[0] : 0;
    const int tokens = real_tokens
        ? min(sequence, max(real_tokens[0], 0)) : sequence;
    if (start_pos < 0 || tokens <= 0)
        return;
    if (start_pos == 0) {
        for (int index = 0; index < head_dim * window_size; ++index)
            window[index] = 0.0f;
    }
    for (int token = 0; token < tokens; ++token) {
        const int position = start_pos + token;
        float* destination = window +
            static_cast<size_t>(position % window_size) * head_dim;
        const float* source =
            current_kv + static_cast<size_t>(token) * head_dim;
        for (int dimension = 0; dimension < head_dim; ++dimension)
            destination[dimension] = source[dimension];
    }
}

__global__ void binary_cuda(const float* lhs, const float* rhs, float* output,
                            size_t count, bool multiply) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        output[index] = multiply ? lhs[index] * rhs[index]
                                 : lhs[index] + rhs[index];
}

__global__ void binary_strided_cuda(
    const float* lhs, const float* rhs, float* output, size_t count,
    CudaTensorLayout lhs_layout, CudaTensorLayout rhs_layout,
    CudaTensorLayout output_layout, bool multiply) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t coordinates[4];
    logical_coordinates(index, output_layout, coordinates);
    const float lhs_value = lhs[layout_offset(
        coordinates, lhs_layout, true)];
    const float rhs_value = rhs[layout_offset(
        coordinates, rhs_layout, true)];
    output[layout_offset(coordinates, output_layout, false)] = multiply
        ? lhs_value * rhs_value : lhs_value + rhs_value;
}

__device__ void copy_storage_element(
    const uint8_t* input, size_t input_index, uint8_t* output,
    size_t output_index, size_t element_size) {
    if (element_size == sizeof(uint32_t)) {
        reinterpret_cast<uint32_t*>(output)[output_index] =
            reinterpret_cast<const uint32_t*>(input)[input_index];
    } else if (element_size == sizeof(uint16_t)) {
        reinterpret_cast<uint16_t*>(output)[output_index] =
            reinterpret_cast<const uint16_t*>(input)[input_index];
    } else {
        output[output_index] = input[input_index];
    }
}

__global__ void tile_cuda(
    const uint8_t* input, uint8_t* output, size_t count,
    CudaTensorLayout input_layout, CudaTensorLayout output_layout,
    size_t element_size) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t coordinates[4];
    logical_coordinates(index, output_layout, coordinates);
    size_t input_coordinates[4];
    for (int dimension = 0; dimension < 4; ++dimension)
        input_coordinates[dimension] = coordinates[dimension] %
            static_cast<size_t>(input_layout.shape[dimension]);
    copy_storage_element(
        input, layout_offset(input_coordinates, input_layout, false), output,
        layout_offset(coordinates, output_layout, false), element_size);
}

__global__ void concat_cuda(
    const uint8_t* input, uint8_t* output, size_t count,
    CudaTensorLayout input_layout, CudaTensorLayout output_layout,
    int concat_dimension, int64_t concat_offset, size_t element_size) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t coordinates[4];
    logical_coordinates(index, input_layout, coordinates);
    const size_t input_offset = layout_offset(
        coordinates, input_layout, false);
    coordinates[concat_dimension] += static_cast<size_t>(concat_offset);
    copy_storage_element(
        input, input_offset, output,
        layout_offset(coordinates, output_layout, false), element_size);
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

__device__ float unary_cuda_value(float value, int operation) {
    switch (operation) {
    case 0: return value / (1.0f + expf(-value));
    case 1: {
        const float inner = 0.7978845608f *
            (value + 0.044715f * value * value * value);
        return 0.5f * value * (1.0f + tanhf(inner));
    }
    case 2: return tanhf(value);
    case 3: return 1.0f / (1.0f + expf(-value));
    case 4: return expf(value);
    case 5:
        return value > 20.0f ? value : log1pf(expf(value));
    default: return value;
    }
}

__global__ void unary_cuda(const float* input, float* output, size_t count,
                           int operation) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        output[index] = unary_cuda_value(input[index], operation);
}

__global__ void unary_strided_cuda(
    const float* input, float* output, size_t count,
    CudaTensorLayout input_layout, CudaTensorLayout output_layout,
    int operation) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t coordinates[4];
    logical_coordinates(index, output_layout, coordinates);
    output[layout_offset(coordinates, output_layout, false)] =
        unary_cuda_value(
            input[layout_offset(coordinates, input_layout, false)],
            operation);
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

__global__ void rms_norm_cuda(const float* input, const float* weight,
                              float* output, int width, int rows,
                              size_t input_row_stride,
                              size_t output_row_stride,
                              float epsilon) {
    const int row = blockIdx.x;
    if (row >= rows)
        return;
    float sum = 0.0f;
    const float* source = input + static_cast<size_t>(row) *
        input_row_stride;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = source[column];
        sum += value * value;
    }
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduction[0] / width + epsilon);
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        output[static_cast<size_t>(row) * output_row_stride + column] =
            source[column] * inverse * weight[column];
}

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
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduction[0] / width + epsilon);
    float* output_row = output + static_cast<size_t>(row) *
        output_row_stride;
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        output_row[column] = residual_row[column] * inverse * weight[column];
}

__global__ void layer_norm_cuda(
    const float* input, const float* weight, const float* bias, float* output,
    int width, int rows, size_t input_row_stride,
    size_t output_row_stride, float epsilon) {
    const int row = blockIdx.x;
    if (row >= rows)
        return;
    const float* source = input + static_cast<size_t>(row) *
        input_row_stride;
    __shared__ float reduction[256];
    float sum = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        sum += source[column];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = reduction[0] / width;
    float variance = 0.0f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float centered = source[column] - mean;
        variance += centered * centered;
    }
    reduction[threadIdx.x] = variance;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduction[0] / width + epsilon);
    float* destination = output + static_cast<size_t>(row) *
        output_row_stride;
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        destination[column] =
            (source[column] - mean) * inverse * weight[column] + bias[column];
}

__global__ void rwkv_token_shift_cuda(
    const float* input, void* state, float* output, int hidden,
    int sequence_length, int real_length, bool state_fp16,
    size_t input_row_stride, size_t output_row_stride) {
    const int dimension = static_cast<int>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (dimension >= hidden)
        return;
    __half* state16 = static_cast<__half*>(state);
    float* state32 = static_cast<float*>(state);
    float previous = state_fp16
        ? __half2float(state16[dimension]) : state32[dimension];
    for (int token = 0; token < sequence_length; ++token) {
        float result = 0.0f;
        if (token < real_length) {
            const float current = input[
                static_cast<size_t>(token) * input_row_stride + dimension];
            result = previous - current;
            if (state_fp16) {
                const __half rounded = __float2half(current);
                previous = __half2float(rounded);
                state16[dimension] = rounded;
            } else {
                previous = current;
                state32[dimension] = current;
            }
        }
        output[static_cast<size_t>(token) * output_row_stride + dimension] =
            result;
    }
}

__global__ void rwkv_mix_cuda(
    const float* input, const float* shift, const float* mix, float* output,
    size_t count, int hidden) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        output[index] = input[index] + shift[index] * mix[index % hidden];
}

__global__ void rwkv_l2_norm_cuda(
    const float* input, float* output, int head_size, int groups,
    float epsilon) {
    const int group = blockIdx.x;
    if (group >= groups)
        return;
    const size_t base = static_cast<size_t>(group) * head_size;
    float sum = 0.0f;
    for (int dimension = threadIdx.x; dimension < head_size;
         dimension += blockDim.x) {
        const float value = input[base + dimension];
        sum += value * value;
    }
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = 1.0f / (sqrtf(reduction[0]) + epsilon);
    for (int dimension = threadIdx.x; dimension < head_size;
         dimension += blockDim.x)
        output[base + dimension] = input[base + dimension] * inverse;
}

__device__ float rwkv_block_sum(float value, float* reduction) {
    reduction[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = reduction[0];
    __syncthreads();
    return result;
}

__global__ void rwkv_post_cuda(
    const float* raw, const float* receptance, const float* key,
    const float* value, const float* receptance_key, const float* weight,
    const float* bias, const float* gate, float* output, int heads,
    int head_size, int groups, float epsilon) {
    const int group = blockIdx.x;
    if (group >= groups)
        return;
    const int head = group % heads;
    const size_t base = static_cast<size_t>(group) * head_size;
    const size_t weight_base = static_cast<size_t>(head) * head_size;
    extern __shared__ float reduction[];
    float sum = 0.0f;
    float bonus = 0.0f;
    for (int dimension = threadIdx.x; dimension < head_size;
         dimension += blockDim.x) {
        sum += raw[base + dimension];
        bonus += receptance[base + dimension] * key[base + dimension] *
            receptance_key[weight_base + dimension];
    }
    const float mean = rwkv_block_sum(sum, reduction) / head_size;
    const float bonus_total = rwkv_block_sum(bonus, reduction);
    float variance = 0.0f;
    for (int dimension = threadIdx.x; dimension < head_size;
         dimension += blockDim.x) {
        const float centered = raw[base + dimension] - mean;
        variance += centered * centered;
    }
    const float inverse = rsqrtf(
        rwkv_block_sum(variance, reduction) / head_size + epsilon);
    for (int dimension = threadIdx.x; dimension < head_size;
         dimension += blockDim.x) {
        const size_t index = base + dimension;
        const size_t parameter_index = weight_base + dimension;
        const float normalized = (raw[index] - mean) * inverse *
            weight[parameter_index] + bias[parameter_index];
        output[index] =
            (normalized + bonus_total * value[index]) * gate[index];
    }
}

__global__ void rwkv7_cuda(
    const float* receptance, const float* decay, const float* key,
    const float* value, const float* a, const float* b, void* state,
    float* output, int heads, int head_size, int sequence_length,
    int real_length, bool state_fp16) {
    const int head = blockIdx.x;
    const int row = threadIdx.x;
    if (head >= heads || row >= head_size)
        return;
    const int hidden = heads * head_size;
    const size_t state_head_base =
        static_cast<size_t>(head) * head_size * head_size;
    const size_t state_row_base = state_head_base +
        static_cast<size_t>(row) * head_size;
    __half* state16 = static_cast<__half*>(state);
    float* state32 = static_cast<float*>(state);
    for (int token = 0; token < sequence_length; ++token) {
        const size_t base = static_cast<size_t>(token) * hidden +
            static_cast<size_t>(head) * head_size;
        if (token >= real_length) {
            output[base + row] = 0.0f;
            continue;
        }
        float state_a = 0.0f;
        for (int column = 0; column < head_size; ++column) {
            const size_t state_index = state_row_base + column;
            const float state_value = state_fp16
                ? __half2float(state16[state_index])
                : state32[state_index];
            state_a += state_value * a[base + column];
        }
        float result = 0.0f;
        for (int column = 0; column < head_size; ++column) {
            const size_t state_index = state_row_base + column;
            const float state_value = state_fp16
                ? __half2float(state16[state_index])
                : state32[state_index];
            const float updated = state_value * decay[base + column] +
                value[base + row] * key[base + column] +
                state_a * b[base + column];
            if (state_fp16)
                state16[state_index] = __float2half(updated);
            else
                state32[state_index] = updated;
            result += updated * receptance[base + column];
        }
        output[base + row] = result;
    }
}

__device__ int moe_signed_nibble(uint8_t value) {
    const int nibble = value & 0x0f;
    return nibble >= 8 ? nibble - 16 : nibble;
}

__device__ float moe_weight_value_cuda(
    const void* data, const float* scales, const uint8_t* e8m0_scales,
    bool dense_fp32, int layout, int group_size, int groups_per_row,
    int row, int column, int width) {
    if (layout == 0) {
        const size_t index = static_cast<size_t>(row) * width + column;
        return dense_fp32
            ? static_cast<const float*>(data)[index]
            : __half2float(static_cast<const __half*>(data)[index]);
    }
    if (layout == 1) {
        const int group = min(column / group_size, groups_per_row - 1);
        return static_cast<float>(
                   static_cast<const int8_t*>(data)[
                       static_cast<size_t>(row) * width + column]) *
            scales[static_cast<size_t>(row) * groups_per_row + group];
    }
    if (layout == 2) {
        const int groups = width / 32;
        const auto& block = static_cast<const Q4B8G32Block*>(data)[
            static_cast<size_t>(row / 8) * groups + column / 32];
        const int lane = row & 7;
        const int inner = column & 31;
        const uint8_t packed = block.q[lane][inner / 2];
        return static_cast<float>(moe_signed_nibble(
                   inner & 1 ? packed >> 4 : packed)) *
            block.scales[lane];
    }
    if (layout == 3) {
        const int groups = width / 128;
        const auto& block = static_cast<const Q4B8G128Block*>(data)[
            static_cast<size_t>(row / 8) * groups + column / 128];
        const int lane = row & 7;
        const int inner = column & 127;
        const int subgroup = inner / 32;
        const int subgroup_inner = inner & 31;
        const uint8_t packed = block.q[subgroup][lane][subgroup_inner / 2];
        return static_cast<float>(moe_signed_nibble(
                   subgroup_inner & 1 ? packed >> 4 : packed)) *
            block.scales[lane];
    }
    if (layout == 5) {
        const uint8_t encoded = static_cast<const uint8_t*>(data)[
            static_cast<size_t>(row) * width + column];
        return cuda_decode_fp8_e4m3fn(encoded) * cuda_decode_e8m0(
            e8m0_scales[static_cast<size_t>(row / 128) * groups_per_row +
                         column / 128]);
    }
    const uint8_t packed = static_cast<const uint8_t*>(data)[
        static_cast<size_t>(row) * (width / 2) + column / 2];
    const uint8_t nibble = column & 1 ? packed >> 4 : packed & 0x0f;
    return cuda_decode_mxfp4_e2m1(nibble) * cuda_decode_e8m0(
        e8m0_scales[static_cast<size_t>(row) * groups_per_row +
                     column / 32]);
}

__global__ void dequantize_q8_dense_weight_cuda(
    const int8_t* weight, const float* scales, int group_size,
    int groups_per_row, __half* output, size_t count, int width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    const int group = min(column / group_size, groups_per_row - 1);
    output[index] = __float2half(
        static_cast<float>(weight[index]) *
        scales[static_cast<size_t>(row) * groups_per_row + group]);
}

__global__ void dequantize_q4_g32_dense_weight_cuda(
    const Q4B8G32Block* weight, __half2* output, size_t packed_count,
    int rows, int groups) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= packed_count)
        return;
    constexpr int packed_per_block = 8 * 16;
    const size_t block_index = index / packed_per_block;
    const int packed_index = static_cast<int>(index % packed_per_block);
    const int lane = packed_index / 16;
    const int byte = packed_index % 16;
    const int group = static_cast<int>(block_index % groups);
    const int row = static_cast<int>(block_index / groups) * 8 + lane;
    if (row >= rows)
        return;
    const auto& block = weight[block_index];
    const uint8_t packed = block.q[lane][byte];
    const float scale = block.scales[lane];
    output[(static_cast<size_t>(row) * groups + group) * 16 + byte] =
        __halves2half2(
            __float2half(moe_signed_nibble(packed) * scale),
            __float2half(moe_signed_nibble(packed >> 4) * scale));
}

__global__ void dequantize_q4_g128_dense_weight_cuda(
    const Q4B8G128Block* weight, __half2* output, size_t packed_count,
    int rows, int groups) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= packed_count)
        return;
    constexpr int packed_per_block = 4 * 8 * 16;
    const size_t block_index = index / packed_per_block;
    int packed_index = static_cast<int>(index % packed_per_block);
    const int subgroup = packed_index / (8 * 16);
    packed_index %= 8 * 16;
    const int lane = packed_index / 16;
    const int byte = packed_index % 16;
    const int group = static_cast<int>(block_index % groups);
    const int row = static_cast<int>(block_index / groups) * 8 + lane;
    if (row >= rows)
        return;
    const auto& block = weight[block_index];
    const uint8_t packed = block.q[subgroup][lane][byte];
    const float scale = block.scales[lane];
    output[((static_cast<size_t>(row) * groups + group) * 4 + subgroup) *
               16 +
           byte] = __halves2half2(
        __float2half(moe_signed_nibble(packed) * scale),
        __float2half(moe_signed_nibble(packed >> 4) * scale));
}

__global__ void q8_dense_gemv_cuda(
    const float* activation, const int8_t* weight, const float* scales,
    int group_size, int groups_per_row, float* output, int columns,
    int inner) {
    constexpr int warps_per_block = 4;
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int column = static_cast<int>(blockIdx.x) * warps_per_block + warp;
    if (column >= columns)
        return;
    float sum = 0.0f;
    const int8_t* row = weight + static_cast<size_t>(column) * inner;
    const float* row_scales =
        scales + static_cast<size_t>(column) * groups_per_row;
    for (int group = 0; group < groups_per_row; ++group) {
        const int begin = group * group_size;
        const int end = min(inner, begin + group_size);
        const float scale = row_scales[group];
        for (int k = begin + lane; k < end; k += warpSize)
            sum += activation[k] * static_cast<float>(row[k]) * scale;
    }
    for (int offset = warpSize / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[column] = sum;
}

__global__ void q4_g32_dense_gemv_cuda(
    const float* activation, const Q4B8G32Block* weight, float* output,
    int columns, int inner) {
    constexpr int warps_per_block = 4;
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int column = static_cast<int>(blockIdx.x) * warps_per_block + warp;
    if (column >= columns)
        return;
    const int groups = inner / 32;
    const int row_lane = column & 7;
    float sum = 0.0f;
    for (int group = 0; group < groups; ++group) {
        const auto& block = weight[
            static_cast<size_t>(column / 8) * groups + group];
        float scale = lane == 0 ? block.scales[row_lane] : 0.0f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const uint8_t packed = block.q[row_lane][lane / 2];
        const int k = group * 32 + lane;
        sum += activation[k] * moe_signed_nibble(
            lane & 1 ? packed >> 4 : packed) * scale;
    }
    for (int offset = warpSize / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[column] = sum;
}

__global__ void q4_g128_dense_gemv_cuda(
    const float* activation, const Q4B8G128Block* weight, float* output,
    int columns, int inner) {
    constexpr int warps_per_block = 4;
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int column = static_cast<int>(blockIdx.x) * warps_per_block + warp;
    if (column >= columns)
        return;
    const int groups = inner / 128;
    const int row_lane = column & 7;
    float sum = 0.0f;
    for (int group = 0; group < groups; ++group) {
        const auto& block = weight[
            static_cast<size_t>(column / 8) * groups + group];
        float scale = lane == 0 ? block.scales[row_lane] : 0.0f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        for (int subgroup = 0; subgroup < 4; ++subgroup) {
            const uint8_t packed =
                block.q[subgroup][row_lane][lane / 2];
            const int k = group * 128 + subgroup * 32 + lane;
            sum += activation[k] * moe_signed_nibble(
                lane & 1 ? packed >> 4 : packed) * scale;
        }
    }
    for (int offset = warpSize / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[column] = sum;
}

__device__ float moe_router_score_cuda(float value, int score_function) {
    if (score_function == 1)
        return 1.0f / (1.0f + expf(-value));
    if (score_function == 2) {
        const float softplus = value > 0.0f
            ? value + log1pf(expf(-value)) : log1pf(expf(value));
        return sqrtf(softplus);
    }
    return value;
}

__global__ void moe_select_routes_cuda(
    const float* logits, const float* bias, int* route_indices,
    float* route_weights, int tokens, int num_experts, int top_k,
    int score_function, bool normalize_topk, int num_groups,
    int topk_groups, float scaling_factor) {
    constexpr int maximum_top_k = 64;
    constexpr int maximum_groups = 64;
    const int token = blockIdx.x;
    if (token >= tokens || threadIdx.x != 0 || top_k > maximum_top_k ||
        num_groups > maximum_groups)
        return;
    const float* token_logits = logits +
        static_cast<size_t>(token) * num_experts;
    float selected_values[maximum_top_k];
    int selected_indices[maximum_top_k];
    unsigned char keep_group[maximum_groups];
    for (int index = 0; index < top_k; ++index) {
        selected_values[index] = -FLT_MAX;
        selected_indices[index] = 0;
    }
    for (int group = 0; group < num_groups; ++group)
        keep_group[group] = 1;

    const int experts_per_group = num_experts / num_groups;
    if (score_function != 0 && num_groups > 1 && experts_per_group > 0 &&
        topk_groups < num_groups) {
        float selected_group_scores[maximum_groups];
        int selected_groups[maximum_groups];
        for (int index = 0; index < topk_groups; ++index) {
            selected_group_scores[index] = -FLT_MAX;
            selected_groups[index] = 0;
        }
        for (int group = 0; group < num_groups; ++group) {
            float best0 = -FLT_MAX;
            float best1 = -FLT_MAX;
            const int begin = group * experts_per_group;
            const int end = group == num_groups - 1
                ? num_experts : begin + experts_per_group;
            for (int expert = begin; expert < end; ++expert) {
                float choice = moe_router_score_cuda(
                    token_logits[expert], score_function);
                if (bias && score_function != 0)
                    choice += bias[expert];
                if (choice > best0) {
                    best1 = best0;
                    best0 = choice;
                } else if (choice > best1) {
                    best1 = choice;
                }
            }
            const float group_score = best1 > -FLT_MAX
                ? best0 + best1 : best0;
            for (int index = 0; index < topk_groups; ++index) {
                if (group_score > selected_group_scores[index]) {
                    for (int shift = topk_groups - 1; shift > index;
                         --shift) {
                        selected_group_scores[shift] =
                            selected_group_scores[shift - 1];
                        selected_groups[shift] = selected_groups[shift - 1];
                    }
                    selected_group_scores[index] = group_score;
                    selected_groups[index] = group;
                    break;
                }
            }
        }
        for (int group = 0; group < num_groups; ++group)
            keep_group[group] = 0;
        for (int index = 0; index < topk_groups; ++index)
            keep_group[selected_groups[index]] = 1;
    }

    for (int expert = 0; expert < num_experts; ++expert) {
        const int group = experts_per_group > 0
            ? min(expert / experts_per_group, num_groups - 1) : 0;
        if (!keep_group[group])
            continue;
        const float score = moe_router_score_cuda(
            token_logits[expert], score_function);
        const float choice = score_function == 0
            ? token_logits[expert] : score + (bias ? bias[expert] : 0.0f);
        for (int index = 0; index < top_k; ++index) {
            if (choice > selected_values[index]) {
                for (int shift = top_k - 1; shift > index; --shift) {
                    selected_values[shift] = selected_values[shift - 1];
                    selected_indices[shift] = selected_indices[shift - 1];
                }
                selected_values[index] = choice;
                selected_indices[index] = expert;
                break;
            }
        }
    }

    float sum = 0.0f;
    if (score_function == 0) {
        const float maximum = selected_values[0];
        for (int index = 0; index < top_k; ++index) {
            selected_values[index] = expf(selected_values[index] - maximum);
            sum += selected_values[index];
        }
        const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int index = 0; index < top_k; ++index)
            selected_values[index] *= inverse;
    } else {
        for (int index = 0; index < top_k; ++index) {
            selected_values[index] = moe_router_score_cuda(
                token_logits[selected_indices[index]], score_function);
            sum += selected_values[index];
        }
        const float multiplier = normalize_topk && sum > 0.0f
            ? scaling_factor / sum : scaling_factor;
        for (int index = 0; index < top_k; ++index)
            selected_values[index] *= multiplier;
    }

    // The CPU reference accumulates experts in ascending expert-id order.
    // Preserve that FP32 summation order after score-based selection.
    for (int index = 1; index < top_k; ++index) {
        int selected_index = selected_indices[index];
        float selected_weight = selected_values[index];
        int position = index;
        while (position > 0 &&
               selected_indices[position - 1] > selected_index) {
            selected_indices[position] = selected_indices[position - 1];
            selected_values[position] = selected_values[position - 1];
            --position;
        }
        selected_indices[position] = selected_index;
        selected_values[position] = selected_weight;
    }
    for (int index = 0; index < top_k; ++index) {
        const size_t output_index =
            static_cast<size_t>(token) * top_k + index;
        route_indices[output_index] = selected_indices[index];
        route_weights[output_index] = selected_values[index];
    }
}

__global__ void moe_select_hash_routes_cuda(
    const float* logits, const int32_t* token_ids,
    const int32_t* token_to_experts, int* route_indices,
    float* route_weights, int* invalid_route, int tokens, int num_experts,
    int top_k, int vocab_size, bool normalize_topk, float scaling_factor) {
    constexpr int maximum_top_k = 64;
    const int token = blockIdx.x;
    if (token >= tokens || threadIdx.x != 0 || top_k > maximum_top_k)
        return;

    const int token_id = token_ids[token];
    if (token_id < 0 || token_id >= vocab_size) {
        atomicExch(invalid_route, 1);
        return;
    }

    int selected_indices[maximum_top_k];
    float selected_weights[maximum_top_k];
    float sum = 0.0f;
    for (int index = 0; index < top_k; ++index) {
        const int expert = token_to_experts[
            static_cast<size_t>(token_id) * top_k + index];
        if (expert < 0 || expert >= num_experts) {
            atomicExch(invalid_route, 1);
            return;
        }
        selected_indices[index] = expert;
        selected_weights[index] = moe_router_score_cuda(
            logits[static_cast<size_t>(token) * num_experts + expert], 2);
        sum += selected_weights[index];
    }

    const float multiplier = normalize_topk && sum > 0.0f
        ? scaling_factor / sum : scaling_factor;
    for (int index = 0; index < top_k; ++index)
        selected_weights[index] *= multiplier;

    // The CPU implementation groups routed work by ascending expert id.
    // Match its FP32 accumulation order while retaining each hash weight.
    for (int index = 1; index < top_k; ++index) {
        const int selected_index = selected_indices[index];
        const float selected_weight = selected_weights[index];
        int position = index;
        while (position > 0 &&
               selected_indices[position - 1] > selected_index) {
            selected_indices[position] = selected_indices[position - 1];
            selected_weights[position] = selected_weights[position - 1];
            --position;
        }
        selected_indices[position] = selected_index;
        selected_weights[position] = selected_weight;
    }
    for (int index = 0; index < top_k; ++index) {
        const size_t output_index =
            static_cast<size_t>(token) * top_k + index;
        route_indices[output_index] = selected_indices[index];
        route_weights[output_index] = selected_weights[index];
    }
}

__global__ void hc_project_cuda(
    const float* input, size_t input_stride, const float* weight,
    float* projected, int tokens, int rows, int width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(tokens) * rows;
    if (index >= count)
        return;
    const int token = static_cast<int>(index / rows);
    const int row = static_cast<int>(index % rows);
    const float* token_input = input +
        static_cast<size_t>(token) * input_stride;
    const float* row_weight = weight + static_cast<size_t>(row) * width;
    float sum = 0.0f;
    for (int column = 0; column < width; ++column)
        sum += token_input[column] * row_weight[column];
    projected[index] = sum;
}

__global__ void hc_pre_cuda(
    const float* input, size_t input_stride, const float* projected,
    const float* scale, const float* base, float* output,
    size_t output_stride, int tokens, int hidden_size, int hc_mult,
    int sinkhorn_iters, float norm_eps, float sinkhorn_eps) {
    constexpr int maximum_hc = 16;
    const int token = blockIdx.x;
    if (token >= tokens || threadIdx.x != 0 || hc_mult > maximum_hc)
        return;
    const int wide = hidden_size * hc_mult;
    const int mix_size = (2 + hc_mult) * hc_mult;
    const float* token_input = input +
        static_cast<size_t>(token) * input_stride;
    const float* mixes = projected + static_cast<size_t>(token) * mix_size;
    float* token_output = output +
        static_cast<size_t>(token) * output_stride;
    float* post_output = token_output + hidden_size;
    float* combination_output = post_output + hc_mult;

    float square_sum = 0.0f;
    for (int index = 0; index < wide; ++index)
        square_sum += token_input[index] * token_input[index];
    const float inverse_rms = 1.0f / sqrtf(
        square_sum / static_cast<float>(wide) + norm_eps);

    float pre[maximum_hc];
    float combination[maximum_hc * maximum_hc];
    float row_sums[maximum_hc];
    float column_sums[maximum_hc];
    for (int h = 0; h < hc_mult; ++h) {
        pre[h] = 1.0f /
            (1.0f + expf(-(mixes[h] * inverse_rms * scale[0] + base[h]))) +
            sinkhorn_eps;
        post_output[h] = 2.0f /
            (1.0f + expf(-(
                mixes[hc_mult + h] * inverse_rms * scale[1] +
                base[hc_mult + h])));
    }
    for (int row = 0; row < hc_mult; ++row) {
        float maximum = -FLT_MAX;
        for (int column = 0; column < hc_mult; ++column) {
            const int mix_index =
                2 * hc_mult + row * hc_mult + column;
            const float value =
                mixes[mix_index] * inverse_rms * scale[2] +
                base[mix_index];
            combination[row * hc_mult + column] = value;
            maximum = fmaxf(maximum, value);
        }
        float sum = 0.0f;
        for (int column = 0; column < hc_mult; ++column) {
            float& value = combination[row * hc_mult + column];
            value = expf(value - maximum);
            sum += value;
        }
        for (int column = 0; column < hc_mult; ++column)
            combination[row * hc_mult + column] =
                combination[row * hc_mult + column] / sum + sinkhorn_eps;
    }
    for (int iteration = 0; iteration < sinkhorn_iters; ++iteration) {
        if (iteration != 0) {
            for (int row = 0; row < hc_mult; ++row)
                row_sums[row] = 0.0f;
            for (int row = 0; row < hc_mult; ++row)
                for (int column = 0; column < hc_mult; ++column)
                    row_sums[row] +=
                        combination[row * hc_mult + column];
            for (int row = 0; row < hc_mult; ++row)
                for (int column = 0; column < hc_mult; ++column)
                    combination[row * hc_mult + column] /=
                        row_sums[row] + sinkhorn_eps;
        }
        for (int column = 0; column < hc_mult; ++column)
            column_sums[column] = 0.0f;
        for (int row = 0; row < hc_mult; ++row)
            for (int column = 0; column < hc_mult; ++column)
                column_sums[column] +=
                    combination[row * hc_mult + column];
        for (int row = 0; row < hc_mult; ++row)
            for (int column = 0; column < hc_mult; ++column)
                combination[row * hc_mult + column] /=
                    column_sums[column] + sinkhorn_eps;
    }
    for (int dimension = 0; dimension < hidden_size; ++dimension) {
        float sum = 0.0f;
        for (int h = 0; h < hc_mult; ++h)
            sum += pre[h] *
                token_input[static_cast<size_t>(h) * hidden_size +
                            dimension];
        token_output[dimension] = cuda_round_to_bf16(sum);
    }
    for (int index = 0; index < hc_mult * hc_mult; ++index)
        combination_output[index] = combination[index];
}

__global__ void hc_post_cuda(
    const float* branch, size_t branch_stride, const float* residual,
    size_t residual_stride, const float* packed, size_t packed_stride,
    float* output, size_t output_stride, size_t count, int hidden_size,
    int hc_mult) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    const int wide = hidden_size * hc_mult;
    const int token = static_cast<int>(index / wide);
    const int inner = static_cast<int>(index % wide);
    const int output_h = inner / hidden_size;
    const int dimension = inner % hidden_size;
    const float* token_branch = branch +
        static_cast<size_t>(token) * branch_stride;
    const float* token_residual = residual +
        static_cast<size_t>(token) * residual_stride;
    const float* post = packed +
        static_cast<size_t>(token) * packed_stride + hidden_size;
    const float* combination = post + hc_mult;
    float residual_sum = 0.0f;
    for (int input_h = 0; input_h < hc_mult; ++input_h) {
        residual_sum += combination[input_h * hc_mult + output_h] *
            token_residual[input_h * hidden_size + dimension];
    }
    output[static_cast<size_t>(token) * output_stride + inner] =
        cuda_round_to_bf16(
            post[output_h] * token_branch[dimension] + residual_sum);
}

__global__ void hc_head_cuda(
    const float* input, size_t input_stride, const float* projected,
    const float* scale, const float* base, float* output,
    size_t output_stride, int tokens, int hidden_size, int hc_mult,
    float norm_eps, float hc_eps) {
    constexpr int maximum_hc = 16;
    const int token = blockIdx.x;
    if (token >= tokens || threadIdx.x != 0 || hc_mult > maximum_hc)
        return;
    const int wide = hidden_size * hc_mult;
    const float* token_input = input +
        static_cast<size_t>(token) * input_stride;
    const float* token_projected = projected +
        static_cast<size_t>(token) * hc_mult;
    float square_sum = 0.0f;
    for (int index = 0; index < wide; ++index)
        square_sum += token_input[index] * token_input[index];
    const float inverse_rms = 1.0f / sqrtf(
        square_sum / static_cast<float>(wide) + norm_eps);
    float pre[maximum_hc];
    for (int h = 0; h < hc_mult; ++h) {
        pre[h] = 1.0f /
            (1.0f + expf(-(
                token_projected[h] * inverse_rms * scale[0] + base[h]))) +
            hc_eps;
    }
    float* token_output = output +
        static_cast<size_t>(token) * output_stride;
    for (int dimension = 0; dimension < hidden_size; ++dimension) {
        float sum = 0.0f;
        for (int h = 0; h < hc_mult; ++h)
            sum += pre[h] *
                token_input[static_cast<size_t>(h) * hidden_size +
                            dimension];
        token_output[dimension] = cuda_round_to_bf16(sum);
    }
}

__global__ void moe_gate_up_cuda(
    const float* hidden, size_t hidden_stride, const int* route_indices,
    const float* route_weights, const void* gate_up,
    const void* const* gate_up_expert_data, bool gate_up_fp32,
    int gate_up_layout, const float* gate_up_scales,
    const void* const* gate_up_expert_scales, int gate_up_group_size,
    int gate_up_groups_per_row, const uint8_t* gate_up_e8m0_scales,
    float* intermediate, int tokens, int top_k, int hidden_size,
    int intermediate_size, float swiglu_limit, bool bf16_activations) {
    const size_t count = static_cast<size_t>(tokens) * top_k *
        intermediate_size;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    const int dimension = index % intermediate_size;
    const int route = index / intermediate_size;
    const int token = route / top_k;
    const int expert = route_indices[route];
    const bool indirect = gate_up_expert_data != nullptr;
    const void* expert_data = indirect
        ? gate_up_expert_data[expert] : gate_up;
    const void* expert_scale_data = gate_up_expert_scales
        ? gate_up_expert_scales[expert] : nullptr;
    const float* expert_scales = expert_scale_data
        ? static_cast<const float*>(expert_scale_data) : gate_up_scales;
    const uint8_t* expert_e8m0_scales = expert_scale_data
        ? static_cast<const uint8_t*>(expert_scale_data)
        : gate_up_e8m0_scales;
    const float* input = hidden + static_cast<size_t>(token) * hidden_stride;
    const int gate_row = indirect
        ? dimension : expert * 2 * intermediate_size + dimension;
    const int up_row = gate_row + intermediate_size;
    float gate = 0.0f;
    float up = 0.0f;
    if (bf16_activations) {
        const auto* packed = static_cast<const uint8_t*>(expert_data);
        const uint8_t* gate_row_data =
            packed + static_cast<size_t>(gate_row) * (hidden_size / 2);
        const uint8_t* up_row_data =
            packed + static_cast<size_t>(up_row) * (hidden_size / 2);
        for (int group = 0; group < gate_up_groups_per_row; ++group) {
            float gate_block = 0.0f;
            float up_block = 0.0f;
            const int begin = group * 32;
            for (int inner = begin; inner < begin + 32; ++inner) {
                const uint8_t gate_packed = gate_row_data[inner / 2];
                const uint8_t up_packed = up_row_data[inner / 2];
                const uint8_t gate_nibble = inner & 1
                    ? gate_packed >> 4 : gate_packed & 0x0f;
                const uint8_t up_nibble = inner & 1
                    ? up_packed >> 4 : up_packed & 0x0f;
                gate_block += input[inner] *
                    cuda_decode_mxfp4_e2m1(gate_nibble);
                up_block += input[inner] *
                    cuda_decode_mxfp4_e2m1(up_nibble);
            }
            gate += gate_block * cuda_decode_e8m0(
                expert_e8m0_scales[
                    static_cast<size_t>(gate_row) *
                        gate_up_groups_per_row + group]);
            up += up_block * cuda_decode_e8m0(
                expert_e8m0_scales[
                    static_cast<size_t>(up_row) *
                        gate_up_groups_per_row + group]);
        }
    } else {
        for (int inner = 0; inner < hidden_size; ++inner) {
            const float value = input[inner];
            gate += value * moe_weight_value_cuda(
                expert_data, expert_scales, expert_e8m0_scales,
                gate_up_fp32, gate_up_layout, gate_up_group_size,
                gate_up_groups_per_row,
                gate_row, inner, hidden_size);
            up += value * moe_weight_value_cuda(
                expert_data, expert_scales, expert_e8m0_scales,
                gate_up_fp32, gate_up_layout, gate_up_group_size,
                gate_up_groups_per_row,
                up_row, inner, hidden_size);
        }
    }
    if (bf16_activations) {
        gate = cuda_round_to_bf16(gate);
        up = cuda_round_to_bf16(up);
    }
    if (swiglu_limit > 0.0f) {
        gate = fminf(gate, swiglu_limit);
        up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    float value = gate / (1.0f + expf(-gate)) * up;
    if (bf16_activations)
        value = cuda_round_to_bf16(value * route_weights[route]);
    intermediate[index] = value;
}

__global__ void moe_down_cuda(
    const int* route_indices, const float* route_weights,
    const float* intermediate, const void* down,
    const void* const* down_expert_data, bool down_fp32,
    int down_layout, const float* down_scales,
    const void* const* down_expert_scales, int down_group_size,
    int down_groups_per_row, const uint8_t* down_e8m0_scales, float* output,
    size_t output_stride, int tokens, int top_k, int hidden_size,
    int intermediate_size, bool bf16_activations) {
    const size_t count = static_cast<size_t>(tokens) * hidden_size;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    const int dimension = index % hidden_size;
    const int token = index / hidden_size;
    float result = 0.0f;
    for (int route_index = 0; route_index < top_k; ++route_index) {
        const int route = token * top_k + route_index;
        const int expert = route_indices[route];
        const bool indirect = down_expert_data != nullptr;
        const void* expert_data = indirect
            ? down_expert_data[expert] : down;
        const void* expert_scale_data = down_expert_scales
            ? down_expert_scales[expert] : nullptr;
        const float* expert_scales = expert_scale_data
            ? static_cast<const float*>(expert_scale_data) : down_scales;
        const uint8_t* expert_e8m0_scales = expert_scale_data
            ? static_cast<const uint8_t*>(expert_scale_data)
            : down_e8m0_scales;
        const int row = indirect
            ? dimension : expert * hidden_size + dimension;
        float contribution = 0.0f;
        const float* route_input = intermediate +
            static_cast<size_t>(route) * intermediate_size;
        if (bf16_activations) {
            const auto* packed = static_cast<const uint8_t*>(expert_data) +
                static_cast<size_t>(row) * (intermediate_size / 2);
            for (int group = 0; group < down_groups_per_row; ++group) {
                float block = 0.0f;
                const int begin = group * 32;
                for (int inner = begin; inner < begin + 32; ++inner) {
                    const uint8_t byte = packed[inner / 2];
                    const uint8_t nibble = inner & 1
                        ? byte >> 4 : byte & 0x0f;
                    block += route_input[inner] *
                        cuda_decode_mxfp4_e2m1(nibble);
                }
                contribution += block * cuda_decode_e8m0(
                    expert_e8m0_scales[
                        static_cast<size_t>(row) *
                            down_groups_per_row + group]);
            }
        } else {
            for (int inner = 0; inner < intermediate_size; ++inner)
                contribution += route_input[inner] *
                    moe_weight_value_cuda(
                        expert_data, expert_scales, expert_e8m0_scales,
                        down_fp32, down_layout, down_group_size,
                        down_groups_per_row,
                        row, inner, intermediate_size);
        }
        if (bf16_activations) {
            contribution = cuda_round_to_bf16(contribution);
            result += contribution;
        } else {
            result += route_weights[route] * contribution;
        }
    }
    output[static_cast<size_t>(token) * output_stride + dimension] = result;
}

__global__ void moe_swiglu_cuda(
    const float* gate, const float* up, float* output, size_t count,
    float swiglu_limit, bool bf16_activations) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    float gate_value = gate[index];
    float up_value = up[index];
    if (bf16_activations) {
        gate_value = cuda_round_to_bf16(gate_value);
        up_value = cuda_round_to_bf16(up_value);
    }
    if (swiglu_limit > 0.0f) {
        gate_value = fminf(gate_value, swiglu_limit);
        up_value = fminf(fmaxf(up_value, -swiglu_limit), swiglu_limit);
    }
    const float value = gate_value / (1.0f + expf(-gate_value)) * up_value;
    output[index] = bf16_activations ? cuda_round_to_bf16(value) : value;
}

__global__ void moe_add_shared_cuda(
    float* output, size_t output_stride, const float* shared,
    const float* shared_scale, int tokens, int hidden_size,
    bool has_scale) {
    const size_t count = static_cast<size_t>(tokens) * hidden_size;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    const int dimension = index % hidden_size;
    const int token = index / hidden_size;
    const float scale = has_scale
        ? 1.0f / (1.0f + expf(-shared_scale[token])) : 1.0f;
    output[static_cast<size_t>(token) * output_stride + dimension] +=
        shared[index] * scale;
}

__global__ void contiguous_cuda(
    const uint8_t* input, uint8_t* output, size_t count,
    CudaTensorLayout input_layout, size_t element_size) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t coordinates[4];
    logical_coordinates(index, input_layout, coordinates);
    copy_storage_element(
        input, layout_offset(coordinates, input_layout, false), output,
        index, element_size);
}

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
    const auto* current_key = static_cast<const float*>(key);
    const size_t current_key_base =
        static_cast<size_t>(key_head) * key_head_stride +
        static_cast<size_t>(key_position) * key_position_stride;
    float dot = 0.0f;
    for (int dimension = 0; dimension < key_dim; ++dimension) {
        const float key_value = cached
            ? load_kv_cache_value(
                  key, cached_key_base + dimension, fp16_cache)
            : current_key[current_key_base +
                          static_cast<size_t>(dimension) *
                              key_feature_stride];
        dot += query_row[static_cast<size_t>(dimension) *
                         query_feature_stride] * key_value;
    }
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
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = reduction[0];
    float local_sum = 0.0f;
    for (int position = threadIdx.x; position < key_length;
         position += blockDim.x)
        local_sum += expf(score_row[position] - maximum);
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_sum = reduction[0] > 0.0f
        ? 1.0f / reduction[0] : 0.0f;
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

__global__ void shortconv_cuda(
    const float* input, const float* weight, float* state, float* output,
    int groups, int sequence_length, int kernel_size, int real_length,
    size_t input_row_stride) {
    const int group = static_cast<int>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (group >= groups)
        return;
    const int prefix_length = kernel_size - 1;
    const int process_length = real_length > 0 && real_length < sequence_length
        ? real_length : sequence_length;
    const float* group_weight = weight +
        static_cast<size_t>(group) * kernel_size;
    float* group_state = state +
        static_cast<size_t>(group) * prefix_length;
    float* group_output = output +
        static_cast<size_t>(group) * sequence_length;

    for (int token = 0; token < sequence_length; ++token) {
        float sum = 0.0f;
        for (int inner = 0; inner < kernel_size; ++inner) {
            const int window_position = token + inner;
            const float value = window_position < prefix_length
                ? group_state[window_position]
                : input[static_cast<size_t>(window_position - prefix_length) *
                            input_row_stride + group];
            sum += value * group_weight[inner];
        }
        group_output[token] = token < process_length
            ? sum / (1.0f + expf(-sum)) : 0.0f;
    }

    for (int inner = 0; inner < prefix_length; ++inner) {
        const int window_position = process_length + inner;
        group_state[inner] = window_position < prefix_length
            ? group_state[window_position]
            : input[static_cast<size_t>(window_position - prefix_length) *
                        input_row_stride + group];
    }
}

__device__ float gdn_block_sum(float value, float* reduction) {
    reduction[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = reduction[0];
    __syncthreads();
    return result;
}

__device__ float gdn_softplus_cuda(float value) {
    if (value > 20.0f)
        return value;
    if (value < -20.0f)
        return expf(value);
    return log1pf(expf(value));
}

__global__ void gated_deltanet_cuda(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, int num_heads, int key_dim, int value_dim,
    int sequence_length, int real_length, int num_value_heads,
    bool use_l2_norm, size_t a_row_stride, size_t b_row_stride,
    size_t z_row_stride, size_t output_row_stride, float rms_epsilon,
    float l2_epsilon, float scale) {
    const int value_head = blockIdx.x;
    if (value_head >= num_value_heads)
        return;
    const int repeat = num_value_heads / num_heads;
    const int key_head = value_head / repeat;
    const int qkv_dim = num_heads * key_dim;
    const int process_length = real_length > 0 &&
            real_length < sequence_length
        ? real_length : sequence_length;
    extern __shared__ float shared[];
    float* query = shared;
    float* key = query + key_dim;
    float* reduction = key + key_dim;
    float* gate = reduction + blockDim.x;
    float* state_head = state +
        static_cast<size_t>(value_head) * key_dim * value_dim;

    for (int token = 0; token < sequence_length; ++token) {
        if (token >= process_length) {
            for (int dimension = threadIdx.x; dimension < value_dim;
                 dimension += blockDim.x)
                output[static_cast<size_t>(token) * output_row_stride +
                       static_cast<size_t>(value_head) * value_dim +
                       dimension] = 0.0f;
            __syncthreads();
            continue;
        }

        for (int dimension = threadIdx.x; dimension < key_dim;
             dimension += blockDim.x) {
            query[dimension] = qkv[
                static_cast<size_t>(key_head * key_dim + dimension) *
                    sequence_length + token];
            key[dimension] = qkv[
                static_cast<size_t>(qkv_dim + key_head * key_dim +
                                    dimension) * sequence_length + token];
        }
        __syncthreads();

        float query_sum = 0.0f;
        float key_sum = 0.0f;
        for (int dimension = threadIdx.x; dimension < key_dim;
             dimension += blockDim.x) {
            query_sum += query[dimension] * query[dimension];
            key_sum += key[dimension] * key[dimension];
        }
        const float query_total = gdn_block_sum(query_sum, reduction);
        const float key_total = gdn_block_sum(key_sum, reduction);
        const float query_inverse = use_l2_norm
            ? rsqrtf(query_total + l2_epsilon) : 1.0f;
        const float key_inverse = use_l2_norm
            ? rsqrtf(key_total + l2_epsilon) : 1.0f;
        for (int dimension = threadIdx.x; dimension < key_dim;
             dimension += blockDim.x) {
            query[dimension] *= query_inverse;
            key[dimension] *= key_inverse;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            const float a_value =
                a[static_cast<size_t>(token) * a_row_stride + value_head];
            const float b_value =
                b[static_cast<size_t>(token) * b_row_stride + value_head];
            const float softplus = gdn_softplus_cuda(
                a_value + dt_bias[value_head]);
            gate[0] = expf(-expf(a_log[value_head]) * softplus);
            gate[1] = 1.0f / (1.0f + expf(-b_value));
        }
        __syncthreads();
        const float decay = gate[0];
        const float beta = gate[1];

        float attention = 0.0f;
        if (threadIdx.x < value_dim) {
            const int value_column = threadIdx.x;
            float memory_value = 0.0f;
            for (int key_row = 0; key_row < key_dim; ++key_row) {
                const float decayed =
                    state_head[static_cast<size_t>(key_row) * value_dim +
                               value_column] * decay;
                memory_value += decayed * key[key_row];
            }
            const float current_value = qkv[
                static_cast<size_t>(2 * qkv_dim +
                                    value_head * value_dim + value_column) *
                    sequence_length + token];
            const float delta = (current_value - memory_value) * beta;
            float updated_attention = 0.0f;
            for (int key_row = 0; key_row < key_dim; ++key_row) {
                const size_t state_index =
                    static_cast<size_t>(key_row) * value_dim + value_column;
                const float updated =
                    state_head[state_index] * decay + key[key_row] * delta;
                state_head[state_index] = updated;
                updated_attention += updated * query[key_row];
            }
            attention = updated_attention * scale;
        }
        const float attention_sum = gdn_block_sum(
            threadIdx.x < value_dim ? attention * attention : 0.0f,
            reduction);
        const float rms_inverse =
            rsqrtf(attention_sum / value_dim + rms_epsilon);
        if (threadIdx.x < value_dim) {
            const int value_column = threadIdx.x;
            const float z_value = z[
                static_cast<size_t>(token) * z_row_stride +
                static_cast<size_t>(value_head) * value_dim + value_column];
            const float silu = z_value / (1.0f + expf(-z_value));
            output[static_cast<size_t>(token) * output_row_stride +
                   static_cast<size_t>(value_head) * value_dim +
                   value_column] = attention * rms_inverse *
                norm_weight[value_column] * silu;
        }
        __syncthreads();
    }
}

}  // namespace

struct CudaBackend::Impl {
    enum class WeightLayout : uint8_t {
        Dense = 0,
        Q8RowMajor,
        Q4Bg32,
        Q4Bg128,
        Int32Lookup,
        Fp8Block128,
        Mxfp4RowMajor,
    };

    struct DeviceWeight {
        void* data = nullptr;
        float* scales = nullptr;
        uint8_t* e8m0_scales = nullptr;
        cudaDataType type = CUDA_R_16F;
        int n = 0;
        int k = 0;
        int group_size = 0;
        int groups_per_row = 0;
        WeightLayout layout = WeightLayout::Dense;
    };

    struct BoundaryBuffer {
        void* data = nullptr;
        size_t capacity = 0;
    };

    struct PersistentHostMirror {
        explicit PersistentHostMirror(size_t bytes)
            : size(bytes), words((bytes + sizeof(uint64_t) - 1) /
                                 sizeof(uint64_t)) {}

        uint8_t* data() {
            return reinterpret_cast<uint8_t*>(words.data());
        }

        size_t size = 0;
        std::vector<uint64_t> words;
    };

    struct MoeDeviceCacheEntry {
        const MoeSsdTensorSource* gate_up = nullptr;
        const MoeSsdTensorSource* down = nullptr;
        int expert = -1;
        void* allocation = nullptr;
        size_t allocation_bytes = 0;
        const void* gate_up_data = nullptr;
        const void* gate_up_scales = nullptr;
        const void* down_data = nullptr;
        const void* down_scales = nullptr;
        uint64_t used_at = 0;
        bool pinned = false;
        bool ready = false;
    };

    bool ok = false;
    bool failed = false;
    cublasHandle_t cublas = nullptr;
    CPUBackend cpu;
    std::unordered_map<const void*, DeviceWeight> weights;
    std::unordered_map<const void*, const DeviceWeight*> weights_by_device;
    std::vector<void*> device_allocations;
    std::vector<void*> pooled_allocations;
    std::unordered_map<void*, size_t> pooled_sizes;
    std::multimap<size_t, void*> free_pooled;
    std::vector<void*> managed_allocations;
    std::unordered_map<void*, PersistentHostMirror> persistent_host_mirrors;
    std::unordered_map<std::string, BoundaryBuffer> boundary_buffers;
    std::vector<MoeDeviceCacheEntry> moe_device_cache;
    DeviceMoeCacheStats moe_device_cache_stats;
    uint64_t moe_device_cache_clock = 0;
    std::unordered_map<uint32_t, uint64_t> native_ops;
    std::unordered_map<uint32_t, uint64_t> fallback_ops;
    OperatorFallbackPolicy operator_fallback =
        OperatorFallbackPolicy::ALLOW_REFERENCE;
    void* activation = nullptr;
    size_t activation_bytes = 0;
    void* activation_fp16 = nullptr;
    size_t activation_fp16_bytes = 0;
    void* output = nullptr;
    size_t output_bytes = 0;
    void* attention_scores = nullptr;
    size_t attention_scores_bytes = 0;
    void* norm_scratch = nullptr;
    size_t norm_scratch_bytes = 0;
    void* gdn_qkv_scratch = nullptr;
    size_t gdn_qkv_scratch_bytes = 0;

    ~Impl() {
        if (cublas)
            cublasDestroy(cublas);
        for (auto& entry : moe_device_cache)
            if (entry.allocation)
                cudaFree(entry.allocation);
        for (void* allocation : device_allocations)
            cudaFree(allocation);
        for (void* allocation : pooled_allocations)
            cudaFree(allocation);
        for (void* allocation : managed_allocations)
            cudaFree(allocation);
        for (auto& entry : boundary_buffers)
            if (entry.second.data)
                cudaFree(entry.second.data);
        if (activation)
            cudaFree(activation);
        if (activation_fp16)
            cudaFree(activation_fp16);
        if (output)
            cudaFree(output);
        if (attention_scores)
            cudaFree(attention_scores);
        if (norm_scratch)
            cudaFree(norm_scratch);
        if (gdn_qkv_scratch)
            cudaFree(gdn_qkv_scratch);
        if (std::getenv("MOLLM_CUDA_PROFILE")) {
            std::fprintf(stderr, "\nCudaBackend operator coverage:\n");
            for (const auto& entry : native_ops)
                std::fprintf(stderr, "  native   %-24s %llu\n",
                             op_type_name(static_cast<OpType>(entry.first)),
                             static_cast<unsigned long long>(entry.second));
            for (const auto& entry : fallback_ops)
                std::fprintf(stderr, "  fallback %-24s %llu\n",
                             op_type_name(static_cast<OpType>(entry.first)),
                             static_cast<unsigned long long>(entry.second));
        }
    }

    void* acquire_pooled(size_t bytes) {
        auto found = free_pooled.lower_bound(bytes);
        if (found != free_pooled.end()) {
            void* pointer = found->second;
            free_pooled.erase(found);
            return pointer;
        }
        void* pointer = nullptr;
        if (!report_cuda(cudaMalloc(&pointer, bytes),
                         "cudaMalloc output"))
            return nullptr;
        pooled_allocations.push_back(pointer);
        pooled_sizes.emplace(pointer, bytes);
        return pointer;
    }

    void release_pooled(void* pointer) {
        const auto found = pooled_sizes.find(pointer);
        if (found != pooled_sizes.end())
            free_pooled.emplace(found->second, pointer);
    }

    bool reserve(void*& pointer, size_t& capacity, size_t requested) {
        if (capacity >= requested)
            return true;
        if (pointer)
            cudaFree(pointer);
        pointer = nullptr;
        capacity = 0;
        if (!report_cuda(cudaMalloc(&pointer, requested), "cudaMalloc scratch"))
            return false;
        capacity = requested;
        return true;
    }

    void* scratch(const char* name, size_t requested) {
        auto& buffer = boundary_buffers[name];
        return reserve(buffer.data, buffer.capacity, requested)
            ? buffer.data : nullptr;
    }

    void clear_moe_device_cache() {
        for (auto& entry : moe_device_cache)
            if (entry.allocation)
                cudaFree(entry.allocation);
        moe_device_cache.clear();
        moe_device_cache_stats.resident_bytes = 0;
        moe_device_cache_clock = 0;
    }

    bool configure_moe_device_cache(size_t capacity_bytes) {
        clear_moe_device_cache();
        moe_device_cache_stats = {};
        moe_device_cache_stats.capacity_bytes = capacity_bytes;
        return true;
    }

    void prepare_moe_device_experts(
        const MoeSsdTensorSource* gate_up,
        const MoeSsdTensorSource* down,
        const std::vector<int>& experts) {
        for (auto& entry : moe_device_cache)
            entry.pinned = false;
        for (int expert : experts) {
            for (auto& entry : moe_device_cache) {
                if (entry.gate_up != gate_up || entry.down != down ||
                    entry.expert != expert || !entry.ready)
                    continue;
                entry.pinned = true;
                entry.used_at = ++moe_device_cache_clock;
                ++moe_device_cache_stats.hits;
                break;
            }
        }
    }

    MoeDeviceCacheEntry* find_pinned_moe_device_expert(
        const MoeSsdTensorSource* gate_up,
        const MoeSsdTensorSource* down, int expert) {
        for (auto& entry : moe_device_cache) {
            if (entry.gate_up == gate_up && entry.down == down &&
                entry.expert == expert && entry.pinned) {
                return &entry;
            }
        }
        return nullptr;
    }

    bool has_moe_device_expert(
        const MoeSsdTensorSource* gate_up,
        const MoeSsdTensorSource* down, int expert) const {
        return std::any_of(
            moe_device_cache.begin(), moe_device_cache.end(),
            [&](const MoeDeviceCacheEntry& entry) {
                return entry.gate_up == gate_up && entry.down == down &&
                    entry.expert == expert && entry.ready;
            });
    }

    MoeDeviceCacheEntry* reserve_moe_device_expert(
        const MoeSsdTensorSource* gate_up,
        const MoeSsdTensorSource* down, int expert) {
        ++moe_device_cache_stats.misses;
        const size_t capacity = moe_device_cache_stats.capacity_bytes;
        if (capacity == 0)
            return nullptr;

        constexpr size_t alignment = 256;
        auto align_offset = [=](size_t value) {
            return (value + alignment - 1) & ~(alignment - 1);
        };
        const size_t gate_data_offset = 0;
        const size_t gate_scales_offset = align_offset(
            gate_up->spec.data_bytes);
        const size_t down_data_offset = align_offset(
            gate_scales_offset + gate_up->spec.scales_bytes);
        const size_t down_scales_offset = align_offset(
            down_data_offset + down->spec.data_bytes);
        const size_t allocation_bytes = align_offset(
            down_scales_offset + down->spec.scales_bytes);
        if (allocation_bytes > capacity)
            return nullptr;

        while (moe_device_cache_stats.resident_bytes >
               capacity - allocation_bytes) {
            auto victim = moe_device_cache.end();
            for (auto candidate = moe_device_cache.begin();
                 candidate != moe_device_cache.end(); ++candidate) {
                if (!candidate->pinned &&
                    (victim == moe_device_cache.end() ||
                     candidate->used_at < victim->used_at))
                    victim = candidate;
            }
            if (victim == moe_device_cache.end())
                return nullptr;
            if (!report_cuda(cudaFree(victim->allocation),
                             "cudaFree MOE device cache entry")) {
                failed = true;
                return nullptr;
            }
            moe_device_cache_stats.resident_bytes -=
                victim->allocation_bytes;
            ++moe_device_cache_stats.evictions;
            moe_device_cache.erase(victim);
        }

        void* allocation = nullptr;
        if (!report_cuda(cudaMalloc(&allocation, allocation_bytes),
                         "cudaMalloc MOE device cache entry")) {
            failed = true;
            return nullptr;
        }
        auto* base = static_cast<uint8_t*>(allocation);
        moe_device_cache.push_back({
            gate_up,
            down,
            expert,
            allocation,
            allocation_bytes,
            base + gate_data_offset,
            gate_up->spec.scales_bytes ? base + gate_scales_offset : nullptr,
            base + down_data_offset,
            down->spec.scales_bytes ? base + down_scales_offset : nullptr,
            ++moe_device_cache_clock,
            true,
            false,
        });
        moe_device_cache_stats.resident_bytes += allocation_bytes;
        return &moe_device_cache.back();
    }

    bool fill_moe_device_expert(
        MoeDeviceCacheEntry& entry,
        const MoeSsdTensorSource* gate_up,
        const MoeSsdTensorSource* down,
        const Tensor& gate_view, const Tensor& down_view) {
        const void* gate_host_scales = gate_view.e8m0_scales
            ? static_cast<const void*>(gate_view.e8m0_scales)
            : static_cast<const void*>(gate_view.scales);
        const void* down_host_scales = down_view.e8m0_scales
            ? static_cast<const void*>(down_view.e8m0_scales)
            : static_cast<const void*>(down_view.scales);
        if (!gate_view.data || !down_view.data ||
            (gate_up->spec.scales_bytes != 0 && !gate_host_scales) ||
            (down->spec.scales_bytes != 0 && !down_host_scales)) {
            failed = true;
            return false;
        }
        auto copy_component = [](void* destination, const void* source,
                                 size_t bytes, const char* label) {
            return bytes == 0 || report_cuda(
                cudaMemcpy(destination, source, bytes,
                           cudaMemcpyHostToDevice), label);
        };
        if (!copy_component(
                const_cast<void*>(entry.gate_up_data), gate_view.data,
                gate_up->spec.data_bytes,
                "cudaMemcpy cached MOE gate/up data") ||
            !copy_component(
                const_cast<void*>(entry.gate_up_scales), gate_host_scales,
                gate_up->spec.scales_bytes,
                "cudaMemcpy cached MOE gate/up scales") ||
            !copy_component(
                const_cast<void*>(entry.down_data), down_view.data,
                down->spec.data_bytes,
                "cudaMemcpy cached MOE down data") ||
            !copy_component(
                const_cast<void*>(entry.down_scales), down_host_scales,
                down->spec.scales_bytes,
                "cudaMemcpy cached MOE down scales")) {
            failed = true;
            return false;
        }
        entry.ready = true;
        moe_device_cache_stats.host_to_device_bytes +=
            gate_up->spec.data_bytes + gate_up->spec.scales_bytes +
            down->spec.data_bytes + down->spec.scales_bytes;
        return true;
    }

    const DeviceWeight* find_weight(const Tensor& tensor) const {
        if (!tensor.device_data)
            return nullptr;
        const auto found = weights_by_device.find(tensor.device_data);
        return found == weights_by_device.end() ? nullptr : found->second;
    }

    bool upload_weight(Tensor& tensor, const void* cache_key,
                       const void* source, size_t bytes, cudaDataType type,
                       int n, int k,
                       WeightLayout layout = WeightLayout::Dense) {
        if (!cache_key || !source || bytes == 0)
            return false;
        auto found = weights.find(cache_key);
        if (found == weights.end()) {
            void* device = nullptr;
            if (!report_cuda(cudaMalloc(&device, bytes),
                             "cudaMalloc weight") ||
                !report_cuda(cudaMemcpy(device, source, bytes,
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy weight")) {
                if (device)
                    cudaFree(device);
                return false;
            }
            device_allocations.push_back(device);
            found = weights.emplace(
                cache_key,
                DeviceWeight{
                    device, nullptr, nullptr, type, n, k, 0, 0, layout})
                        .first;
            weights_by_device.emplace(device, &found->second);
        }
        tensor.device_data = found->second.data;
        tensor.device_offset = 0;
        return true;
    }

    bool upload_quantized_weight(
        Tensor& tensor, const void* cache_key, const void* source,
        size_t bytes, const float* host_scales, size_t scale_count,
        int n, int k, int group_size, int groups_per_row,
        WeightLayout layout) {
        if (!host_scales || scale_count == 0 || group_size <= 0 ||
            groups_per_row <= 0 ||
            !upload_weight(tensor, cache_key, source, bytes, CUDA_R_8I,
                           n, k, layout))
            return false;
        auto found = weights.find(cache_key);
        if (found == weights.end())
            return false;
        DeviceWeight& prepared = found->second;
        if (!prepared.scales) {
            float* device_scales = nullptr;
            const size_t scale_bytes = scale_count * sizeof(float);
            if (!report_cuda(cudaMalloc(&device_scales, scale_bytes),
                             "cudaMalloc weight scales") ||
                !report_cuda(cudaMemcpy(device_scales, host_scales,
                                        scale_bytes,
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy weight scales")) {
                if (device_scales)
                    cudaFree(device_scales);
                return false;
            }
            device_allocations.push_back(device_scales);
            prepared.scales = device_scales;
        }
        prepared.group_size = group_size;
        prepared.groups_per_row = groups_per_row;
        return true;
    }

    bool upload_microscaled_weight(
        Tensor& tensor, const void* cache_key, const void* source,
        size_t bytes, const uint8_t* host_scales, size_t scale_count,
        int n, int k, int group_size, int groups_per_row,
        WeightLayout layout) {
        if (!host_scales || scale_count == 0 || group_size <= 0 ||
            groups_per_row <= 0 ||
            !upload_weight(tensor, cache_key, source, bytes, CUDA_R_8U,
                           n, k, layout))
            return false;
        auto found = weights.find(cache_key);
        if (found == weights.end())
            return false;
        DeviceWeight& prepared = found->second;
        if (!prepared.e8m0_scales) {
            uint8_t* device_scales = nullptr;
            if (!report_cuda(cudaMalloc(&device_scales, scale_count),
                             "cudaMalloc E8M0 weight scales") ||
                !report_cuda(cudaMemcpy(device_scales, host_scales,
                                        scale_count,
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy E8M0 weight scales")) {
                if (device_scales)
                    cudaFree(device_scales);
                return false;
            }
            device_allocations.push_back(device_scales);
            prepared.e8m0_scales = device_scales;
        }
        prepared.group_size = group_size;
        prepared.groups_per_row = groups_per_row;
        return true;
    }

    bool run_matmul_device(const float* device_a, int lda,
                           const Tensor& weight, float* device_c, int ldc,
                           int m, int n, int k,
                           Activation activation_kind, int act_begin,
                           int act_len) {
        const DeviceWeight* prepared = find_weight(weight);
        if (!prepared || prepared->n != n || prepared->k != k || !device_a ||
            !device_c || ldc < n)
            return false;
        if (ldc != n && activation_kind != Activation::NONE && act_len != 0)
            return false;

        const bool valid_q8 =
            prepared->layout == WeightLayout::Q8RowMajor &&
            prepared->scales && prepared->group_size > 0 &&
            prepared->groups_per_row ==
                (k + prepared->group_size - 1) / prepared->group_size;
        const bool valid_q4_g32 =
            prepared->layout == WeightLayout::Q4Bg32 && k % 32 == 0;
        const bool valid_q4_g128 =
            prepared->layout == WeightLayout::Q4Bg128 && k % 128 == 0;
        const bool valid_quantized =
            valid_q8 || valid_q4_g32 || valid_q4_g128;

        if (valid_quantized && m == 1) {
            constexpr int warps_per_block = 4;
            constexpr int quantized_threads = warps_per_block * 32;
            const unsigned blocks = static_cast<unsigned>(
                (n + warps_per_block - 1) / warps_per_block);
            const char* label = nullptr;
            if (valid_q8) {
                q8_dense_gemv_cuda<<<blocks, quantized_threads>>>(
                    device_a, static_cast<const int8_t*>(prepared->data),
                    prepared->scales, prepared->group_size,
                    prepared->groups_per_row, device_c, n, k);
                label = "q8_dense_gemv_cuda";
            } else if (valid_q4_g32) {
                q4_g32_dense_gemv_cuda<<<blocks, quantized_threads>>>(
                    device_a,
                    static_cast<const Q4B8G32Block*>(prepared->data),
                    device_c, n, k);
                label = "q4_g32_dense_gemv_cuda";
            } else {
                q4_g128_dense_gemv_cuda<<<blocks, quantized_threads>>>(
                    device_a,
                    static_cast<const Q4B8G128Block*>(prepared->data),
                    device_c, n, k);
                label = "q4_g128_dense_gemv_cuda";
            }
            if (!report_cuda(cudaGetLastError(), label))
                return false;
        } else if (prepared->layout == WeightLayout::Fp8Block128 ||
            prepared->layout == WeightLayout::Mxfp4RowMajor) {
            const bool valid_fp8 =
                prepared->layout == WeightLayout::Fp8Block128 &&
                prepared->e8m0_scales && prepared->group_size == 128 &&
                prepared->groups_per_row == (k + 127) / 128;
            const bool valid_mxfp4 =
                prepared->layout == WeightLayout::Mxfp4RowMajor &&
                prepared->e8m0_scales && prepared->group_size == 32 &&
                k % 32 == 0 && prepared->groups_per_row == k / 32;
            if (!valid_fp8 && !valid_mxfp4)
                return false;
            constexpr int threads = 256;
            const bool quantize_activation = valid_mxfp4 ||
                (valid_fp8 && k >= 32 && k % 32 == 0);
            const float* matmul_activation = device_a;
            int matmul_lda = lda;
            if (quantize_activation) {
                auto* quantized = static_cast<float*>(scratch(
                    "microscaled_activation",
                    static_cast<size_t>(m) * k * sizeof(float)));
                if (!quantized)
                    return false;
                const int blocks = m * ((k + 127) / 128);
                quantize_activation_fp8_cuda<<<blocks, 1>>>(
                    device_a, lda, quantized, m, k);
                if (!report_cuda(cudaGetLastError(),
                                 "quantize_activation_fp8_cuda"))
                    return false;
                matmul_activation = quantized;
                matmul_lda = k;
            }
            const size_t count = static_cast<size_t>(m) * n;
            microscaled_matmul_cuda<<<
                static_cast<unsigned>((count + threads - 1) / threads),
                threads>>>(
                matmul_activation, matmul_lda,
                static_cast<const uint8_t*>(prepared->data),
                prepared->e8m0_scales,
                static_cast<int>(prepared->layout), device_c, ldc,
                m, n, k, prepared->groups_per_row);
            if (!report_cuda(cudaGetLastError(),
                             "microscaled_matmul_cuda"))
                return false;
        } else {
            const void* linear_weight = prepared->data;
            cudaDataType linear_weight_type = prepared->type;
            if (valid_quantized) {
                const size_t weight_elements =
                    static_cast<size_t>(n) * k;
                auto* dequantized = static_cast<__half*>(scratch(
                    "dense_quantized_weight",
                    weight_elements * sizeof(__half)));
                if (!dequantized)
                    return false;
                constexpr int dequantize_threads = 256;
                const char* dequantize_label = nullptr;
                if (valid_q8) {
                    dequantize_q8_dense_weight_cuda<<<
                        static_cast<unsigned>(
                            (weight_elements + dequantize_threads - 1) /
                            dequantize_threads),
                        dequantize_threads>>>(
                        static_cast<const int8_t*>(prepared->data),
                        prepared->scales, prepared->group_size,
                        prepared->groups_per_row, dequantized,
                        weight_elements, k);
                    dequantize_label = "dequantize_q8_dense_weight_cuda";
                } else if (valid_q4_g32) {
                    const size_t packed_count =
                        static_cast<size_t>((n + 7) / 8) * (k / 32) *
                        8 * 16;
                    dequantize_q4_g32_dense_weight_cuda<<<
                        static_cast<unsigned>(
                            (packed_count + dequantize_threads - 1) /
                            dequantize_threads),
                        dequantize_threads>>>(
                        static_cast<const Q4B8G32Block*>(prepared->data),
                        reinterpret_cast<__half2*>(dequantized),
                        packed_count, n, k / 32);
                    dequantize_label = "dequantize_q4_g32_dense_weight_cuda";
                } else {
                    const size_t packed_count =
                        static_cast<size_t>((n + 7) / 8) * (k / 128) *
                        4 * 8 * 16;
                    dequantize_q4_g128_dense_weight_cuda<<<
                        static_cast<unsigned>(
                            (packed_count + dequantize_threads - 1) /
                            dequantize_threads),
                        dequantize_threads>>>(
                        static_cast<const Q4B8G128Block*>(prepared->data),
                        reinterpret_cast<__half2*>(dequantized),
                        packed_count, n, k / 128);
                    dequantize_label =
                        "dequantize_q4_g128_dense_weight_cuda";
                }
                if (!report_cuda(cudaGetLastError(), dequantize_label))
                    return false;
                linear_weight = dequantized;
                linear_weight_type = CUDA_R_16F;
            } else if (prepared->layout != WeightLayout::Dense) {
                return false;
            }
            if (ldc != n)
                return false;

            const size_t a_elements = static_cast<size_t>(m) * lda;
            const void* gemm_activation = device_a;
            cudaDataType activation_type = CUDA_R_32F;
            if (linear_weight_type == CUDA_R_16F) {
                if (!reserve(activation_fp16, activation_fp16_bytes,
                             a_elements * sizeof(__half)))
                    return false;
                constexpr int threads = 256;
                fp32_to_fp16<<<
                    static_cast<unsigned>(
                        (a_elements + threads - 1) / threads),
                    threads>>>(device_a,
                               static_cast<__half*>(activation_fp16),
                               a_elements);
                if (!report_cuda(cudaGetLastError(), "fp32_to_fp16"))
                    return false;
                gemm_activation = activation_fp16;
                activation_type = CUDA_R_16F;
            }

            const float alpha = 1.0f;
            const float beta = 0.0f;
            // Row-major C[M,N] = A[M,K] * W[N,K]^T is the equivalent
            // column-major operation C_col[N,M] = W_col[K,N]^T * A_col[K,M].
            if (!report_cublas(
                    cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                        linear_weight, linear_weight_type, k, gemm_activation,
                        activation_type, lda, &beta, device_c, CUDA_R_32F, n,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "cublasGemmEx"))
                return false;
        }

        if (activation_kind != Activation::NONE && act_len != 0) {
            const int begin = std::max(0, act_begin);
            const int end = act_len < 0 ? n : std::min(n, begin + act_len);
            const size_t count = static_cast<size_t>(m) * n;
            constexpr int threads = 256;
            apply_activation_cuda<<<
                static_cast<unsigned>((count + threads - 1) / threads),
                threads>>>(device_c, m, n,
                           static_cast<int>(activation_kind), begin, end);
            if (!report_cuda(cudaGetLastError(), "apply_activation_cuda"))
                return false;
        }
        return true;
    }

    bool run_matmul(const float* host_a, int lda, const Tensor& weight,
                    float* host_c, int ldc, int m, int n, int k,
                    Activation activation_kind, int act_begin, int act_len) {
        const size_t a_bytes = static_cast<size_t>(m) * lda * sizeof(float);
        const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
        if (!host_a || !host_c || ldc != n ||
            !reserve(activation, activation_bytes, a_bytes) ||
            !reserve(output, output_bytes, c_bytes) ||
            !report_cuda(cudaMemcpy(activation, host_a, a_bytes,
                                    cudaMemcpyHostToDevice),
                         "cudaMemcpy activation") ||
            !run_matmul_device(
                static_cast<const float*>(activation), lda, weight,
                static_cast<float*>(output), ldc, m, n, k, activation_kind,
                act_begin, act_len) ||
            !report_cuda(cudaMemcpy(host_c, output, c_bytes,
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy output"))
            return false;
        return true;
    }
};

namespace {

template <typename T>
T* device_pointer(const Tensor& tensor) {
    if (!tensor.device_data)
        return nullptr;
    return reinterpret_cast<T*>(
        static_cast<uint8_t*>(tensor.device_data) + tensor.device_offset);
}

template <typename T>
const T* device_pointer_const(const Tensor& tensor) {
    if (!tensor.device_data)
        return nullptr;
    return reinterpret_cast<const T*>(
        static_cast<const uint8_t*>(tensor.device_data) +
        tensor.device_offset);
}

CudaTensorLayout cuda_layout(const Tensor& tensor, size_t element_size) {
    CudaTensorLayout layout{};
    for (int dimension = 0; dimension < 4; ++dimension) {
        layout.shape[dimension] = tensor.shape[dimension];
        layout.stride[dimension] =
            tensor.stride[dimension] / element_size;
    }
    return layout;
}

CudaTensorLayout cuda_layout(const Tensor& tensor) {
    return cuda_layout(tensor, sizeof(float));
}

bool supported_copy_element_size(size_t element_size) {
    return element_size == sizeof(uint8_t) ||
        element_size == sizeof(uint16_t) ||
        element_size == sizeof(uint32_t);
}

bool fp32_contiguous(const Tensor& tensor) {
    return tensor.prec == Precision::FP32 && tensor.is_contiguous();
}

bool broadcasts_to(const Tensor& source, const Tensor& destination) {
    for (int dimension = 0; dimension < 4; ++dimension)
        if (source.shape[dimension] != 1 &&
            source.shape[dimension] != destination.shape[dimension])
            return false;
    return true;
}

bool same_shape(const Tensor& lhs, const Tensor& rhs) {
    for (int dimension = 0; dimension < 4; ++dimension)
        if (lhs.shape[dimension] != rhs.shape[dimension])
            return false;
    return true;
}

}  // namespace

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {
    int count = 0;
    if (!report_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount") ||
        count <= 0)
        return;
    if (!report_cuda(cudaSetDevice(0), "cudaSetDevice") ||
        !report_cublas(cublasCreate(&impl_->cublas), "cublasCreate"))
        return;
    impl_->ok = true;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess)
        std::fprintf(stderr, "CudaBackend: using %s (sm_%d%d)\n",
                     properties.name, properties.major, properties.minor);
}

CudaBackend::~CudaBackend() = default;

bool CudaBackend::available() const { return impl_ && impl_->ok; }

bool CudaBackend::configure_moe_device_cache(size_t capacity_bytes) {
    return available() && impl_->configure_moe_device_cache(capacity_bytes);
}

DeviceMoeCacheStats CudaBackend::moe_device_cache_stats() const {
    return impl_ ? impl_->moe_device_cache_stats : DeviceMoeCacheStats{};
}

BackendOperatorStats CudaBackend::operator_stats() const {
    BackendOperatorStats stats;
    stats.tracked = impl_ != nullptr;
    if (!impl_)
        return stats;
    for (const auto& entry : impl_->native_ops)
        stats.native_calls += entry.second;
    for (const auto& entry : impl_->fallback_ops)
        stats.fallback_calls += entry.second;
    return stats;
}

bool CudaBackend::set_operator_fallback_policy(
    OperatorFallbackPolicy policy) {
    if (!impl_)
        return false;
    impl_->operator_fallback = policy;
    return true;
}

void CudaBackend::clear_dispatch_error() {
    impl_->failed = false;
    impl_->cpu.clear_dispatch_error();
}

bool CudaBackend::dispatch_failed() const {
    return impl_->failed || impl_->cpu.dispatch_failed();
}

bool CudaBackend::register_weight_region(void*, size_t) { return true; }

void CudaBackend::wrap_weight(Tensor& tensor) {
    if (!available() || !tensor.data || tensor.shape[0] <= 0 ||
        tensor.shape[1] <= 0)
        return;
    const int n = static_cast<int>(tensor.shape[0]);
    const int k = static_cast<int>(tensor.shape[1]);
    if (tensor.prec == Precision::FP16) {
        impl_->upload_weight(tensor, tensor.data, tensor.data,
                             static_cast<size_t>(n) * k * sizeof(__half),
                             CUDA_R_16F, n, k);
    } else if (tensor.prec == Precision::FP32) {
        impl_->upload_weight(tensor, tensor.data, tensor.data,
                             static_cast<size_t>(n) * k * sizeof(float),
                             CUDA_R_32F, n, k);
    } else if (tensor.prec == Precision::INT32) {
        impl_->upload_weight(
            tensor, tensor.data, tensor.data,
            static_cast<size_t>(n) * k * sizeof(int32_t), CUDA_R_32I,
            n, k, Impl::WeightLayout::Int32Lookup);
    }
}

void CudaBackend::wrap_weight_int4(Tensor& tensor,
                                   bool keep_native_experts) {
    if (!available() || tensor.shape[0] <= 0 ||
        tensor.shape[1] <= 0)
        return;
    const int n = static_cast<int>(tensor.shape[0]);
    const int k = static_cast<int>(tensor.shape[1]);
    const void* rowmajor_source = tensor.rowmajor_data
        ? tensor.rowmajor_data : tensor.data;
    if (tensor.prec == Precision::FP8_E4M3) {
        const int groups_per_row = (k + 127) / 128;
        const size_t scale_count =
            static_cast<size_t>((n + 127) / 128) * groups_per_row;
        if (rowmajor_source && tensor.e8m0_scales &&
            tensor.is_fp8_block128 && tensor.group_size == 128 &&
            tensor.groups_per_row == static_cast<uint32_t>(groups_per_row)) {
            impl_->upload_microscaled_weight(
                tensor, rowmajor_source, rowmajor_source,
                static_cast<size_t>(n) * k, tensor.e8m0_scales,
                scale_count, n, k, 128, groups_per_row,
                Impl::WeightLayout::Fp8Block128);
        }
        return;
    }
    if (tensor.prec == Precision::MXFP4) {
        if (rowmajor_source && tensor.e8m0_scales && k % 32 == 0 &&
            tensor.group_size == 32 &&
            tensor.groups_per_row == static_cast<uint32_t>(k / 32)) {
            impl_->upload_microscaled_weight(
                tensor, rowmajor_source, rowmajor_source,
                static_cast<size_t>(n) * k / 2, tensor.e8m0_scales,
                static_cast<size_t>(n) * (k / 32), n, k, 32, k / 32,
                Impl::WeightLayout::Mxfp4RowMajor);
        }
        return;
    }
    (void)keep_native_experts;
    if (tensor.prec == Precision::INT8) {
        const void* source = tensor.rowmajor_data
            ? tensor.rowmajor_data : tensor.data;
        const int group_size = static_cast<int>(tensor.group_size);
        const int groups_per_row =
            static_cast<int>(tensor.groups_per_row);
        if (source && tensor.scales && group_size > 0 &&
            groups_per_row == (k + group_size - 1) / group_size) {
            impl_->upload_quantized_weight(
                tensor, source, source,
                static_cast<size_t>(n) * k, tensor.scales,
                static_cast<size_t>(n) * groups_per_row, n, k,
                group_size, groups_per_row,
                Impl::WeightLayout::Q8RowMajor);
        }
        return;
    }
    if (tensor.prec != Precision::INT4)
        return;
    if (tensor.is_q4_g32_packed && tensor.q4_g32_data && k % 32 == 0) {
        const size_t bytes = static_cast<size_t>((n + 7) / 8) * (k / 32) *
            sizeof(Q4B8G32Block);
        impl_->upload_weight(
            tensor, tensor.q4_g32_data, tensor.q4_g32_data, bytes,
            CUDA_R_8I, n, k, Impl::WeightLayout::Q4Bg32);
    } else if (tensor.is_q4_g128_packed && tensor.q4_g128_data &&
               k % 128 == 0) {
        const size_t bytes = static_cast<size_t>((n + 7) / 8) * (k / 128) *
            sizeof(Q4B8G128Block);
        impl_->upload_weight(
            tensor, tensor.q4_g128_data, tensor.q4_g128_data, bytes,
            CUDA_R_8I, n, k, Impl::WeightLayout::Q4Bg128);
    }
}

void* CudaBackend::alloc_output(Tensor& output, size_t nbytes, BufferPool*) {
    if (!available() || nbytes == 0)
        return nullptr;
    void* pointer = impl_->acquire_pooled(nbytes);
    if (!pointer) {
        impl_->failed = true;
        return nullptr;
    }
    output.data = pointer;
    output.device_data = pointer;
    output.device_offset = 0;
    output.mem_type = MemoryType::POOLED;
    output.owner_id = 0;
    output.storage_id = 0;
    return pointer;
}

void CudaBackend::free_output(Tensor& tensor, BufferPool*) {
    if (tensor.device_data)
        impl_->release_pooled(tensor.device_data);
}

bool CudaBackend::copy_to_host(const Tensor& source, void* destination,
                               size_t nbytes, size_t source_offset) {
    if (!destination || source_offset > source.view_span_bytes() ||
        nbytes > source.view_span_bytes() - source_offset) {
        impl_->failed = true;
        return false;
    }
    const auto* device = device_pointer_const<uint8_t>(source);
    if (!device) {
        if (!source.data) {
            impl_->failed = true;
            return false;
        }
        std::memcpy(
            destination,
            static_cast<const uint8_t*>(source.data) + source_offset,
            nbytes);
        return true;
    }
    if (!report_cuda(
            cudaMemcpy(destination, device + source_offset, nbytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy tensor to host")) {
        impl_->failed = true;
        return false;
    }
    return true;
}

bool CudaBackend::copy_from_host(const void* source, Tensor& destination,
                                 size_t nbytes,
                                 size_t destination_offset) {
    if (!source || destination_offset > destination.view_span_bytes() ||
        nbytes > destination.view_span_bytes() - destination_offset) {
        impl_->failed = true;
        return false;
    }
    auto* device = device_pointer<uint8_t>(destination);
    if (!device) {
        if (!destination.data) {
            impl_->failed = true;
            return false;
        }
        std::memcpy(
            static_cast<uint8_t*>(destination.data) + destination_offset,
            source, nbytes);
        return true;
    }
    if (!report_cuda(
            cudaMemcpy(device + destination_offset, source, nbytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy tensor from host")) {
        impl_->failed = true;
        return false;
    }
    const auto mirror = impl_->persistent_host_mirrors.find(
        destination.device_data);
    const size_t absolute_offset =
        destination.device_offset + destination_offset;
    if (mirror != impl_->persistent_host_mirrors.end() &&
        absolute_offset < mirror->second.size) {
        const size_t mirror_bytes = std::min(
            nbytes, mirror->second.size - absolute_offset);
        std::memcpy(
            mirror->second.data() + absolute_offset, source, mirror_bytes);
    }
    return true;
}

bool CudaBackend::zero_tensor(Tensor& tensor, size_t nbytes,
                              size_t destination_offset) {
    if (destination_offset > tensor.view_span_bytes() ||
        nbytes > tensor.view_span_bytes() - destination_offset) {
        impl_->failed = true;
        return false;
    }
    auto* device = device_pointer<uint8_t>(tensor);
    if (!device) {
        if (!tensor.data) {
            impl_->failed = true;
            return false;
        }
        std::memset(
            static_cast<uint8_t*>(tensor.data) + destination_offset,
            0, nbytes);
        return true;
    }
    if (!report_cuda(
            cudaMemset(device + destination_offset, 0, nbytes),
            "cudaMemset tensor")) {
        impl_->failed = true;
        return false;
    }
    const auto mirror = impl_->persistent_host_mirrors.find(
        tensor.device_data);
    const size_t absolute_offset = tensor.device_offset + destination_offset;
    if (mirror != impl_->persistent_host_mirrors.end() &&
        absolute_offset < mirror->second.size) {
        const size_t mirror_bytes = std::min(
            nbytes, mirror->second.size - absolute_offset);
        std::memset(
            mirror->second.data() + absolute_offset, 0, mirror_bytes);
    }
    return true;
}

bool CudaBackend::round_to_bf16(Tensor& tensor) {
    float* values = device_pointer<float>(tensor);
    if (!values || tensor.prec != Precision::FP32) {
        impl_->failed = true;
        return false;
    }
    const size_t count = static_cast<size_t>(tensor.nelements());
    if (count == 0)
        return true;
    constexpr int threads = 256;
    round_bf16_cuda<<<
        static_cast<unsigned>((count + threads - 1) / threads), threads>>>(
        values, count);
    if (!report_cuda(cudaGetLastError(), "round_bf16_cuda")) {
        impl_->failed = true;
        return false;
    }
    return true;
}

void CudaBackend::synchronize_for_host_read() {
    if (!report_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"))
        impl_->failed = true;
}

void CudaBackend::begin_graph() {}

void CudaBackend::end_graph() { synchronize_for_host_read(); }

void CudaBackend::alloc_persistent(
    Tensor& tensor, size_t nbytes, PersistentHostAccess host_access,
    size_t host_prefix_bytes) {
    void* storage = nullptr;
    if (!available() || nbytes == 0 ||
        (host_access == PersistentHostAccess::MIRRORED_PREFIX &&
         (host_prefix_bytes == 0 || host_prefix_bytes > nbytes))) {
        impl_->failed = true;
        return;
    }
    if (host_access == PersistentHostAccess::FULL) {
        if (!report_cuda(cudaMallocManaged(&storage, nbytes),
                         "cudaMallocManaged host-coherent persistent")) {
            impl_->failed = true;
            return;
        }
        impl_->managed_allocations.push_back(storage);
        std::memset(storage, 0, nbytes);
        tensor.data = storage;
    } else {
        if (!report_cuda(cudaMalloc(&storage, nbytes),
                         "cudaMalloc persistent") ||
            !report_cuda(cudaMemset(storage, 0, nbytes),
                         "cudaMemset persistent")) {
            if (storage)
                cudaFree(storage);
            impl_->failed = true;
            return;
        }
        impl_->device_allocations.push_back(storage);
        if (host_access == PersistentHostAccess::MIRRORED_PREFIX) {
            auto [entry, inserted] =
                impl_->persistent_host_mirrors.emplace(
                    storage, Impl::PersistentHostMirror(host_prefix_bytes));
            (void)inserted;
            tensor.data = entry->second.data();
        } else {
            tensor.data = storage;
        }
    }
    tensor.device_data = storage;
    tensor.device_offset = 0;
    tensor.mem_type = MemoryType::EXTERNAL;
}

void CudaBackend::upload_input(Tensor& tensor, const std::string& key,
                               const void* host_source, size_t nbytes) {
    if (!available() || key.empty() || !host_source || nbytes == 0)
        return;
    auto& buffer = impl_->boundary_buffers[key];
    if (buffer.capacity < nbytes) {
        if (buffer.data)
            cudaFree(buffer.data);
        buffer.data = nullptr;
        buffer.capacity = 0;
        if (!report_cuda(cudaMalloc(&buffer.data, nbytes),
                         "cudaMalloc input")) {
            impl_->failed = true;
            return;
        }
        buffer.capacity = nbytes;
    }
    if (!report_cuda(cudaMemcpy(buffer.data, host_source, nbytes,
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy input")) {
        impl_->failed = true;
        return;
    }
    tensor.device_data = buffer.data;
    tensor.device_offset = 0;
}

bool CudaBackend::supports_lm_head(const Tensor& weight) const {
    return impl_->find_weight(weight) != nullptr;
}

void CudaBackend::dispatch(const GraphNode& node,
                           const std::vector<const Tensor*>& inputs,
                           Tensor* output, ThreadPool* thread_pool) {
    auto record_native = [&]() {
        ++impl_->native_ops[static_cast<uint32_t>(node.op_type)];
    };
    constexpr int threads = 256;

    if (node.op_type == OpType::INPUT ||
        node.op_type == OpType::CONSTANT) {
        record_native();
        return;
    }

    if (node.op_type == OpType::RESHAPE && !inputs.empty() && inputs[0] &&
        output && inputs[0]->is_contiguous()) {
        const int64_t shape[4] = {output->shape[0], output->shape[1],
                                  output->shape[2], output->shape[3]};
        *output = *inputs[0];
        for (int dimension = 0; dimension < 4; ++dimension)
            output->shape[dimension] = shape[dimension];
        output->compute_strides();
        record_native();
        return;
    }

    if (node.op_type == OpType::PERMUTE && !inputs.empty() && inputs[0] &&
        output) {
        const Tensor& source = *inputs[0];
        const int axis[4] = {
            graph_params::get_i32(node.params, 0, 0),
            graph_params::get_i32(node.params, 1, 1),
            graph_params::get_i32(node.params, 2, 2),
            graph_params::get_i32(node.params, 3, 3),
        };
        Tensor view = source;
        for (int dimension = 0; dimension < 4; ++dimension) {
            view.shape[axis[dimension]] = source.shape[dimension];
            view.stride[axis[dimension]] = source.stride[dimension];
        }
        *output = view;
        record_native();
        return;
    }

    if (node.op_type == OpType::SLICE && !inputs.empty() && inputs[0] &&
        output) {
        const Tensor& source = *inputs[0];
        const int dimension = graph_params::get_i32(node.params, 0, 0);
        const int offset = graph_params::get_i32(node.params, 1, 0);
        const int size = graph_params::get_i32(
            node.params, 2, static_cast<int>(source.shape[dimension]));
        *output = source;
        output->device_offset = source.device_offset +
            static_cast<size_t>(offset) * source.stride[dimension];
        output->shape[dimension] = size;
        record_native();
        return;
    }

    if (node.op_type == OpType::TILE && !inputs.empty() && inputs[0] &&
        output && inputs[0]->prec == output->prec) {
        const Tensor& source = *inputs[0];
        const size_t element_size = source.element_size();
        const uint8_t* input = device_pointer_const<uint8_t>(source);
        uint8_t* destination = device_pointer<uint8_t>(*output);
        bool valid = input && destination && source.nelements() > 0 &&
            output->nelements() > 0 &&
            supported_copy_element_size(element_size);
        for (int dimension = 0; dimension < 4; ++dimension) {
            const int repeat = graph_params::get_i32(
                node.params, dimension, 1);
            valid = valid && repeat > 0 &&
                output->shape[dimension] ==
                    source.shape[dimension] * repeat;
        }
        if (valid) {
            const size_t count = static_cast<size_t>(output->nelements());
            tile_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                              threads), threads>>>(
                input, destination, count,
                cuda_layout(source, element_size),
                cuda_layout(*output, element_size),
                element_size);
            if (!report_cuda(cudaGetLastError(), "tile_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::CONCAT && !inputs.empty() && output &&
        supported_copy_element_size(output->element_size())) {
        const int dimension = graph_params::get_i32(node.params, 0, 0);
        const size_t element_size = output->element_size();
        uint8_t* destination = device_pointer<uint8_t>(*output);
        int64_t concatenated = 0;
        bool valid = destination && dimension >= 0 && dimension < 4;
        for (const Tensor* source : inputs) {
            valid = valid && source && source->prec == output->prec &&
                device_pointer_const<uint8_t>(*source) &&
                source->nelements() > 0;
            if (!valid)
                break;
            for (int other = 0; other < 4; ++other)
                if (other != dimension)
                    valid = valid &&
                        source->shape[other] == output->shape[other];
            concatenated += source->shape[dimension];
        }
        valid = valid && concatenated == output->shape[dimension];
        if (valid) {
            int64_t offset = 0;
            for (const Tensor* source : inputs) {
                const size_t count =
                    static_cast<size_t>(source->nelements());
                concat_cuda<<<
                    static_cast<unsigned>((count + threads - 1) / threads),
                    threads>>>(
                    device_pointer_const<uint8_t>(*source), destination,
                    count, cuda_layout(*source, element_size),
                    cuda_layout(*output, element_size), dimension, offset,
                    element_size);
                if (!report_cuda(cudaGetLastError(), "concat_cuda")) {
                    impl_->failed = true;
                    return;
                }
                offset += source->shape[dimension];
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::CONTIGUOUS ||
         node.op_type == OpType::RESHAPE) &&
        !inputs.empty() && inputs[0] && output &&
        inputs[0]->prec == output->prec && output->is_contiguous() &&
        supported_copy_element_size(output->element_size())) {
        const Tensor& source = *inputs[0];
        const size_t element_size = source.element_size();
        const uint8_t* input = device_pointer_const<uint8_t>(source);
        uint8_t* destination = device_pointer<uint8_t>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (input && destination && source.nelements() == output->nelements()) {
            contiguous_cuda<<<
                static_cast<unsigned>((count + threads - 1) / threads),
                threads>>>(
                input, destination, count,
                cuda_layout(source, element_size),
                element_size);
            if (!report_cuda(cudaGetLastError(), "contiguous_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::ROTARY_EMBED && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& source = *inputs[0];
        const Tensor& cosine = *inputs[1];
        const Tensor& sine = *inputs[2];
        const float* input = device_pointer_const<float>(source);
        const float* cos_data = device_pointer_const<float>(cosine);
        const float* sin_data = device_pointer_const<float>(sine);
        float* destination = device_pointer<float>(*output);
        const int feature_dim = static_cast<int>(source.shape[0]);
        const int sequence_length = static_cast<int>(source.shape[1]);
        const int channels = static_cast<int>(source.shape[2] * source.shape[3]);
        const int rope_dim = graph_params::get_i32(node.params, 0, 64);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        const size_t count = static_cast<size_t>(source.nelements());
        if (input && cos_data && sin_data && destination &&
            rope_dim > 0 && rope_dim <= feature_dim && rope_dim % 2 == 0 &&
            cosine.shape[0] >= rope_dim / 2 &&
            sine.shape[0] >= rope_dim / 2) {
            rope_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                              threads), threads>>>(
                input, cos_data, sin_data, destination, feature_dim,
                sequence_length, channels, static_cast<int>(source.shape[2]),
                rope_dim, interleave,
                source.stride[0] / sizeof(float),
                source.stride[1] / sizeof(float),
                source.stride[2] / sizeof(float),
                source.stride[3] / sizeof(float),
                cosine.stride[0] / sizeof(float),
                cosine.stride[1] / sizeof(float),
                sine.stride[0] / sizeof(float),
                sine.stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                output->stride[3] / sizeof(float));
            if (!report_cuda(cudaGetLastError(), "rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::SDPA ||
         node.op_type == OpType::SDPA_MLA) &&
        inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] &&
        output && inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        inputs[2]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& query = *inputs[0];
        const Tensor& current_key = *inputs[1];
        const Tensor& current_value = *inputs[2];
        const Tensor* mask = inputs.size() > 3 && inputs[3] &&
                inputs[3]->data && inputs[3]->nelements() > 0
            ? inputs[3] : nullptr;
        const Tensor* key_cache = inputs.size() > 4 && inputs[4] &&
                inputs[4]->data
            ? inputs[4] : nullptr;
        const Tensor* value_cache = inputs.size() > 5 && inputs[5] &&
                inputs[5]->data
            ? inputs[5] : nullptr;
        const int cache_mode = graph_params::get_i32(node.params, 0, 2);
        const bool causal = graph_params::get_i32(node.params, 1, 1) != 0;
        const int num_heads = graph_params::get_i32(
            node.params, 2, static_cast<int>(query.shape[2]));
        const int num_kv_heads = graph_params::get_i32(
            node.params, 3, static_cast<int>(current_key.shape[2]));
        const int key_dim = graph_params::get_i32(
            node.params, 4, static_cast<int>(query.shape[0]));
        const int value_dim = graph_params::get_i32(
            node.params, 5, static_cast<int>(current_value.shape[0]));
        float scale = graph_params::get_f32(node.params, 0, 0.0f);
        if (scale == 0.0f)
            scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
        const int query_length = static_cast<int>(query.shape[1]);
        const int current_length = static_cast<int>(current_key.shape[1]);
        int past_length = 0;
        int key_capacity = current_length;
        bool cached = false;
        const float* current_key_data =
            device_pointer_const<float>(current_key);
        const float* current_value_data =
            device_pointer_const<float>(current_value);
        const void* key_data = current_key_data;
        const void* value_data = current_value_data;
        void* key_cache_data = nullptr;
        void* value_cache_data = nullptr;
        bool fp16_cache = false;
        if (cache_mode == 2 && key_cache && value_cache &&
            key_cache->prec == value_cache->prec &&
            (key_cache->prec == Precision::FP16 ||
             key_cache->prec == Precision::FP32) &&
            key_cache->device_data && value_cache->device_data) {
            const auto* metadata = cache_meta(
                static_cast<const uint8_t*>(key_cache->data) +
                key_cache->device_offset);
            past_length = static_cast<int>(metadata->current_seq_len);
            key_capacity = static_cast<int>(metadata->max_seq_len);
            key_cache_data =
                static_cast<uint8_t*>(key_cache->device_data) +
                key_cache->device_offset + CacheMetadata::SIZE;
            value_cache_data =
                static_cast<uint8_t*>(value_cache->device_data) +
                value_cache->device_offset + CacheMetadata::SIZE;
            fp16_cache = key_cache->prec == Precision::FP16;
            cached = true;
        }
        const int key_length = past_length + current_length;
        const float* query_data = device_pointer_const<float>(query);
        float* output_data = device_pointer<float>(*output);
        const float* mask_data = mask
            ? device_pointer_const<float>(*mask) : nullptr;
        const bool valid = query_data && current_key_data &&
            current_value_data &&
            output_data && num_heads > 0 && num_kv_heads > 0 &&
            num_heads % num_kv_heads == 0 && query_length > 0 &&
            current_length > 0 && key_dim > 0 && value_dim > 0 &&
            query.shape[0] >= key_dim && current_key.shape[0] >= key_dim &&
            current_value.shape[0] >= value_dim &&
            output->shape[0] >= value_dim &&
            (cache_mode == 0 || cached) &&
            (!cached || (key_cache_data && value_cache_data &&
                         key_length <= key_capacity)) &&
            (!mask || mask_data);
        if (valid) {
            if (cached) {
                const int maximum_dim = std::max(key_dim, value_dim);
                const size_t append_count =
                    static_cast<size_t>(num_kv_heads) * current_length *
                    maximum_dim;
                append_kv_cuda<<<
                    static_cast<unsigned>((append_count + threads - 1) /
                                          threads), threads>>>(
                    current_key_data, current_value_data,
                    key_cache_data, value_cache_data,
                    fp16_cache, num_kv_heads, current_length, past_length,
                    key_capacity, key_dim, value_dim,
                    current_key.stride[1] / sizeof(float),
                    current_key.stride[2] / sizeof(float),
                    current_value.stride[1] / sizeof(float),
                    current_value.stride[2] / sizeof(float));
                if (!report_cuda(cudaGetLastError(), "append_kv_cuda")) {
                    impl_->failed = true;
                    return;
                }
                key_data = key_cache_data;
                value_data = value_cache_data;
            }
            const size_t score_count = static_cast<size_t>(num_heads) *
                query_length * key_length;
            const size_t score_bytes = score_count * sizeof(float);
            if (!impl_->reserve(impl_->attention_scores,
                                impl_->attention_scores_bytes,
                                score_bytes)) {
                impl_->failed = true;
                return;
            }
            auto* scores = static_cast<float*>(impl_->attention_scores);
            sdpa_scores_cuda<<<
                static_cast<unsigned>((score_count + threads - 1) / threads),
                threads>>>(
                query_data, key_data, scores, mask_data, num_heads,
                num_kv_heads, query_length, key_length, past_length, key_dim,
                key_capacity, cached, fp16_cache, causal, scale,
                query.stride[0] / sizeof(float),
                query.stride[1] / sizeof(float),
                query.stride[2] / sizeof(float),
                cached ? 1 : current_key.stride[0] / sizeof(float),
                cached ? static_cast<size_t>(key_dim)
                       : current_key.stride[1] / sizeof(float),
                cached ? static_cast<size_t>(key_capacity) * key_dim
                       : current_key.stride[2] / sizeof(float),
                mask ? mask->stride[0] / sizeof(float) : 0,
                mask ? mask->stride[1] / sizeof(float) : 0);
            if (!report_cuda(cudaGetLastError(), "sdpa_scores_cuda")) {
                impl_->failed = true;
                return;
            }
            sdpa_output_cuda<<<num_heads * query_length, threads>>>(
                scores, value_data, output_data, num_heads, num_kv_heads,
                query_length, key_length, value_dim, key_capacity, cached,
                fp16_cache,
                cached ? 1 : current_value.stride[0] / sizeof(float),
                cached ? static_cast<size_t>(value_dim)
                       : current_value.stride[1] / sizeof(float),
                cached ? static_cast<size_t>(key_capacity) * value_dim
                       : current_value.stride[2] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float));
            if (!report_cuda(cudaGetLastError(), "sdpa_output_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::GATED_DELTANET_PREFILL ||
         node.op_type == OpType::GATED_DELTANET_DECODE ||
         node.op_type == OpType::GATED_DELTANET_CONV_DECODE) &&
        inputs.size() >= 8 && output) {
        const bool conv_decode =
            node.op_type == OpType::GATED_DELTANET_CONV_DECODE;
        const int num_heads = graph_params::get_i32(node.params, 0, 16);
        const int key_dim = graph_params::get_i32(node.params, 1, 128);
        const int value_dim = graph_params::get_i32(node.params, 2, 128);
        const int sequence_length = conv_decode ? 1 :
            graph_params::get_i32(node.params, 3, 1);
        const bool use_l2_norm =
            graph_params::get_i32(node.params, 4, 1) != 0;
        const int real_length = conv_decode ? 1 :
            graph_params::get_i32(node.params, 6, sequence_length);
        const int num_value_heads = graph_params::get_i32(
            node.params, 7, num_heads);
        const int qkv_total =
            2 * num_heads * key_dim + num_value_heads * value_dim;
        float rms_epsilon = graph_params::get_f32(
            node.params, 0, 1e-6f);
        float l2_epsilon = graph_params::get_f32(
            node.params, 1, 1e-6f);
        float scale = graph_params::get_f32(node.params, 2, 0.0f);
        if (scale == 0.0f)
            scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
        const float* qkv_data = device_pointer_const<float>(*inputs[0]);
        const float* a_data = device_pointer_const<float>(*inputs[1]);
        const float* b_data = device_pointer_const<float>(*inputs[2]);
        const float* z_data = device_pointer_const<float>(*inputs[3]);
        const float* a_log_data = device_pointer_const<float>(*inputs[4]);
        const float* dt_bias_data = device_pointer_const<float>(*inputs[5]);
        const float* norm_data = device_pointer_const<float>(*inputs[6]);
        float* state_data = device_pointer<float>(*inputs[7]);
        float* output_data = device_pointer<float>(*output);

        bool valid = qkv_data && a_data && b_data && z_data && a_log_data &&
            dt_bias_data && norm_data && state_data && output_data &&
            num_heads > 0 && key_dim > 0 && value_dim > 0 &&
            value_dim <= threads && sequence_length > 0 &&
            num_value_heads >= num_heads &&
            num_value_heads % num_heads == 0 &&
            output->nelements() >=
                static_cast<int64_t>(sequence_length) * num_value_heads *
                    value_dim;
        if (conv_decode) {
            valid = valid && inputs.size() >= 10 && inputs[8] && inputs[9];
            if (valid) {
                const int kernel_size = graph_params::get_i32(
                    node.params, 5, 4);
                const float* conv_weight =
                    device_pointer_const<float>(*inputs[8]);
                float* conv_state = device_pointer<float>(*inputs[9]);
                const size_t scratch_bytes =
                    static_cast<size_t>(qkv_total) * sizeof(float);
                valid = conv_weight && conv_state && kernel_size > 0 &&
                    inputs[8]->nelements() >=
                        static_cast<int64_t>(qkv_total) * kernel_size &&
                    inputs[9]->nelements() >=
                        static_cast<int64_t>(qkv_total) * (kernel_size - 1) &&
                    impl_->reserve(impl_->gdn_qkv_scratch,
                                   impl_->gdn_qkv_scratch_bytes,
                                   scratch_bytes);
                if (valid) {
                    auto* convolved =
                        static_cast<float*>(impl_->gdn_qkv_scratch);
                    shortconv_cuda<<<
                        static_cast<unsigned>((qkv_total + threads - 1) /
                                              threads), threads>>>(
                        qkv_data, conv_weight, conv_state, convolved,
                        qkv_total, 1, kernel_size, 1,
                        inputs[0]->stride[1] / sizeof(float));
                    if (!report_cuda(cudaGetLastError(),
                                     "gdn shortconv_cuda")) {
                        impl_->failed = true;
                        return;
                    }
                    qkv_data = convolved;
                }
            }
        }
        if (valid) {
            const size_t shared_bytes =
                static_cast<size_t>(2 * key_dim + threads + 2) *
                sizeof(float);
            gated_deltanet_cuda<<<num_value_heads, threads, shared_bytes>>>(
                qkv_data, a_data, b_data, z_data, a_log_data, dt_bias_data,
                norm_data, state_data, output_data, num_heads, key_dim,
                value_dim, sequence_length, real_length, num_value_heads,
                use_l2_norm, inputs[1]->stride[1] / sizeof(float),
                inputs[2]->stride[1] / sizeof(float),
                inputs[3]->stride[1] / sizeof(float),
                output->stride[1] / sizeof(float), rms_epsilon, l2_epsilon,
                scale);
            if (!report_cuda(cudaGetLastError(), "gated_deltanet_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SHORTCONV && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& input = *inputs[0];
        const int kernel_size = graph_params::get_i32(node.params, 0, 4);
        const int groups = static_cast<int>(input.shape[0]);
        const int sequence_length = static_cast<int>(input.shape[1]);
        const int real_length = graph_params::get_i32(
            node.params, 1, sequence_length);
        const float* input_data = device_pointer_const<float>(input);
        const float* weight_data = device_pointer_const<float>(*inputs[1]);
        float* state_data = device_pointer<float>(*inputs[2]);
        float* output_data = device_pointer<float>(*output);
        if (input_data && weight_data && state_data && output_data &&
            groups > 0 && sequence_length > 0 && kernel_size > 0 &&
            inputs[1]->nelements() >=
                static_cast<int64_t>(groups) * kernel_size &&
            inputs[2]->nelements() >=
                static_cast<int64_t>(groups) * (kernel_size - 1) &&
            output->nelements() >=
                static_cast<int64_t>(groups) * sequence_length) {
            shortconv_cuda<<<
                static_cast<unsigned>((groups + threads - 1) / threads),
                threads>>>(
                input_data, weight_data, state_data, output_data, groups,
                sequence_length, kernel_size, real_length,
                input.stride[1] / sizeof(float));
            if (!report_cuda(cudaGetLastError(), "shortconv_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RWKV_TOKEN_SHIFT && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*output) &&
        (inputs[1]->prec == Precision::FP16 ||
         inputs[1]->prec == Precision::FP32)) {
        const int hidden = graph_params::get_i32(
            node.params, 0, static_cast<int>(inputs[0]->shape[0]));
        const int sequence_length = graph_params::get_i32(
            node.params, 1, static_cast<int>(inputs[0]->shape[1]));
        int real_length = graph_params::get_i32(
            node.params, 2, sequence_length);
        if (real_length <= 0 || real_length > sequence_length)
            real_length = sequence_length;
        const float* input = device_pointer_const<float>(*inputs[0]);
        void* state = device_pointer<uint8_t>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        if (input && state && destination && hidden > 0 &&
            sequence_length > 0 && inputs[0]->shape[0] == hidden &&
            inputs[0]->shape[1] >= sequence_length &&
            inputs[1]->nelements() >= hidden &&
            output->nelements() >=
                static_cast<int64_t>(hidden) * sequence_length) {
            rwkv_token_shift_cuda<<<
                static_cast<unsigned>((hidden + threads - 1) / threads),
                threads>>>(
                input, state, destination, hidden, sequence_length,
                real_length, inputs[1]->prec == Precision::FP16,
                inputs[0]->stride[1] / sizeof(float),
                output->stride[1] / sizeof(float));
            if (!report_cuda(cudaGetLastError(),
                             "rwkv_token_shift_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RWKV_MIX && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        fp32_contiguous(*inputs[2]) && fp32_contiguous(*output) &&
        inputs[0]->nelements() == inputs[1]->nelements() &&
        inputs[0]->nelements() == output->nelements()) {
        const int hidden = static_cast<int>(inputs[2]->nelements());
        const size_t count = static_cast<size_t>(output->nelements());
        const float* input = device_pointer_const<float>(*inputs[0]);
        const float* shift = device_pointer_const<float>(*inputs[1]);
        const float* mix = device_pointer_const<float>(*inputs[2]);
        float* destination = device_pointer<float>(*output);
        if (input && shift && mix && destination && hidden > 0 &&
            count % static_cast<size_t>(hidden) == 0) {
            rwkv_mix_cuda<<<
                static_cast<unsigned>((count + threads - 1) / threads),
                threads>>>(input, shift, mix, destination, count, hidden);
            if (!report_cuda(cudaGetLastError(), "rwkv_mix_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RWKV_L2_NORM && !inputs.empty() &&
        inputs[0] && output && fp32_contiguous(*inputs[0]) &&
        fp32_contiguous(*output) &&
        inputs[0]->nelements() == output->nelements()) {
        const int heads = graph_params::get_i32(node.params, 0, 0);
        const int head_size = graph_params::get_i32(node.params, 1, 0);
        const int hidden = heads * head_size;
        const int groups = hidden > 0
            ? static_cast<int>(inputs[0]->nelements() / head_size) : 0;
        const float* input = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        if (input && destination && heads > 0 && head_size > 0 &&
            inputs[0]->nelements() % hidden == 0 && groups > 0) {
            rwkv_l2_norm_cuda<<<groups, threads>>>(
                input, destination, head_size, groups,
                graph_params::get_f32(node.params, 0, 1e-12f));
            if (!report_cuda(cudaGetLastError(), "rwkv_l2_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RWKV_POST && inputs.size() >= 8 &&
        inputs[0] && inputs[1] && inputs[2] && inputs[3] && inputs[4] &&
        inputs[5] && inputs[6] && inputs[7] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        fp32_contiguous(*inputs[2]) && fp32_contiguous(*inputs[3]) &&
        fp32_contiguous(*inputs[4]) && fp32_contiguous(*inputs[5]) &&
        fp32_contiguous(*inputs[6]) && fp32_contiguous(*inputs[7]) &&
        fp32_contiguous(*output)) {
        const int heads = graph_params::get_i32(node.params, 0, 0);
        const int head_size = graph_params::get_i32(node.params, 1, 0);
        const int hidden = heads * head_size;
        const int groups = hidden > 0
            ? static_cast<int>(inputs[0]->nelements() / head_size) : 0;
        bool valid = heads > 0 && head_size > 0 && head_size <= threads &&
            inputs[0]->nelements() % hidden == 0 && groups > 0 &&
            inputs[0]->nelements() == inputs[1]->nelements() &&
            inputs[0]->nelements() == inputs[2]->nelements() &&
            inputs[0]->nelements() == inputs[3]->nelements() &&
            inputs[0]->nelements() == inputs[7]->nelements() &&
            inputs[0]->nelements() == output->nelements() &&
            inputs[4]->nelements() >= hidden &&
            inputs[5]->nelements() >= hidden &&
            inputs[6]->nelements() >= hidden;
        const float* raw = device_pointer_const<float>(*inputs[0]);
        const float* receptance = device_pointer_const<float>(*inputs[1]);
        const float* key = device_pointer_const<float>(*inputs[2]);
        const float* value = device_pointer_const<float>(*inputs[3]);
        const float* receptance_key =
            device_pointer_const<float>(*inputs[4]);
        const float* weight = device_pointer_const<float>(*inputs[5]);
        const float* bias = device_pointer_const<float>(*inputs[6]);
        const float* gate = device_pointer_const<float>(*inputs[7]);
        float* destination = device_pointer<float>(*output);
        valid = valid && raw && receptance && key && value && receptance_key &&
            weight && bias && gate && destination;
        if (valid) {
            rwkv_post_cuda<<<groups, threads,
                             threads * sizeof(float)>>>(
                raw, receptance, key, value, receptance_key, weight, bias,
                gate, destination, heads, head_size, groups,
                graph_params::get_f32(node.params, 0, 64e-5f));
            if (!report_cuda(cudaGetLastError(), "rwkv_post_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RWKV7 && inputs.size() == 7 && inputs[0] &&
        inputs[1] && inputs[2] && inputs[3] && inputs[4] && inputs[5] &&
        inputs[6] && output && fp32_contiguous(*inputs[0]) &&
        fp32_contiguous(*inputs[1]) && fp32_contiguous(*inputs[2]) &&
        fp32_contiguous(*inputs[3]) && fp32_contiguous(*inputs[4]) &&
        fp32_contiguous(*inputs[5]) && fp32_contiguous(*output) &&
        (inputs[6]->prec == Precision::FP16 ||
         inputs[6]->prec == Precision::FP32)) {
        const int heads = graph_params::get_i32(node.params, 0, 0);
        const int head_size = graph_params::get_i32(node.params, 1, 0);
        const int sequence_length = graph_params::get_i32(node.params, 2, 1);
        int real_length = graph_params::get_i32(
            node.params, 3, sequence_length);
        if (real_length <= 0 || real_length > sequence_length)
            real_length = sequence_length;
        const int hidden = heads * head_size;
        const int64_t elements =
            static_cast<int64_t>(hidden) * sequence_length;
        bool valid = heads > 0 && head_size > 0 && head_size <= threads &&
            sequence_length > 0 && inputs[0]->nelements() == elements &&
            inputs[1]->nelements() == elements &&
            inputs[2]->nelements() == elements &&
            inputs[3]->nelements() == elements &&
            inputs[4]->nelements() == elements &&
            inputs[5]->nelements() == elements &&
            output->nelements() == elements &&
            inputs[6]->nelements() >=
                static_cast<int64_t>(heads) * head_size * head_size;
        const float* receptance =
            device_pointer_const<float>(*inputs[0]);
        const float* decay = device_pointer_const<float>(*inputs[1]);
        const float* key = device_pointer_const<float>(*inputs[2]);
        const float* value = device_pointer_const<float>(*inputs[3]);
        const float* a = device_pointer_const<float>(*inputs[4]);
        const float* b = device_pointer_const<float>(*inputs[5]);
        void* state = device_pointer<uint8_t>(*inputs[6]);
        float* destination = device_pointer<float>(*output);
        valid = valid && receptance && decay && key && value && a && b &&
            state && destination;
        if (valid) {
            rwkv7_cuda<<<heads, threads>>>(
                receptance, decay, key, value, a, b, state, destination,
                heads, head_size, sequence_length, real_length,
                inputs[6]->prec == Precision::FP16);
            if (!report_cuda(cudaGetLastError(), "rwkv7_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::HC_PRE ||
         node.op_type == OpType::HC_HEAD) &&
        inputs.size() == 4 && inputs[0] && inputs[1] && inputs[2] &&
        inputs[3] && output) {
        const int hidden_size = graph_params::get_i32(node.params, 0, 0);
        const int hc_mult = graph_params::get_i32(node.params, 1, 4);
        const int tokens = static_cast<int>(inputs[0]->shape[1]);
        const int wide = hidden_size * hc_mult;
        const int rows = node.op_type == OpType::HC_PRE
            ? (2 + hc_mult) * hc_mult : hc_mult;
        const int output_width = node.op_type == OpType::HC_PRE
            ? hidden_size + hc_mult + hc_mult * hc_mult : hidden_size;
        const auto* fn = impl_->find_weight(*inputs[1]);
        const auto* scale = impl_->find_weight(*inputs[2]);
        const auto* base = impl_->find_weight(*inputs[3]);
        auto valid_fp32_weight = [](const Impl::DeviceWeight* weight,
                                    int n, int k) {
            return weight && weight->layout == Impl::WeightLayout::Dense &&
                weight->type == CUDA_R_32F && weight->n == n &&
                weight->k == k;
        };
        const int scale_rows = node.op_type == OpType::HC_PRE ? 3 : 1;
        bool valid = hidden_size > 0 && hc_mult > 0 && hc_mult <= 16 &&
            tokens > 0 && inputs[0]->shape[0] == wide &&
            output->shape[0] == output_width &&
            output->shape[1] == tokens && fp32_contiguous(*inputs[0]) &&
            fp32_contiguous(*output) &&
            valid_fp32_weight(fn, rows, wide) &&
            valid_fp32_weight(scale, scale_rows, 1) &&
            valid_fp32_weight(base, rows, 1);
        const int sinkhorn_iters = node.op_type == OpType::HC_PRE
            ? graph_params::get_i32(node.params, 2, 20) : 0;
        valid = valid &&
            (node.op_type != OpType::HC_PRE || sinkhorn_iters > 0);
        const float* input = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        auto* projected = static_cast<float*>(impl_->scratch(
            "hc_projected", static_cast<size_t>(tokens) * rows *
                sizeof(float)));
        valid = valid && input && destination && projected;
        if (valid) {
            const size_t projection_count =
                static_cast<size_t>(tokens) * rows;
            hc_project_cuda<<<
                static_cast<unsigned>((projection_count + threads - 1) /
                                      threads), threads>>>(
                input, inputs[0]->stride[1] / sizeof(float),
                static_cast<const float*>(fn->data), projected,
                tokens, rows, wide);
            if (node.op_type == OpType::HC_PRE) {
                hc_pre_cuda<<<tokens, 1>>>(
                    input, inputs[0]->stride[1] / sizeof(float), projected,
                    static_cast<const float*>(scale->data),
                    static_cast<const float*>(base->data), destination,
                    output->stride[1] / sizeof(float), tokens, hidden_size,
                    hc_mult, sinkhorn_iters,
                    graph_params::get_f32(node.params, 0, 1e-6f),
                    graph_params::get_f32(node.params, 1, 1e-6f));
            } else {
                hc_head_cuda<<<tokens, 1>>>(
                    input, inputs[0]->stride[1] / sizeof(float), projected,
                    static_cast<const float*>(scale->data),
                    static_cast<const float*>(base->data), destination,
                    output->stride[1] / sizeof(float), tokens, hidden_size,
                    hc_mult,
                    graph_params::get_f32(node.params, 0, 1e-6f),
                    graph_params::get_f32(node.params, 1, 1e-6f));
            }
            if (!report_cuda(cudaGetLastError(), "CUDA Hyper-Connection")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::HC_POST && inputs.size() == 3 && inputs[0] &&
        inputs[1] && inputs[2] && output) {
        const int hidden_size = graph_params::get_i32(node.params, 0, 0);
        const int hc_mult = graph_params::get_i32(node.params, 1, 4);
        const int tokens = static_cast<int>(output->shape[1]);
        const int wide = hidden_size * hc_mult;
        const int packed_size = hidden_size + hc_mult + hc_mult * hc_mult;
        bool valid = hidden_size > 0 && hc_mult > 0 && tokens > 0 &&
            inputs[0]->shape[0] == hidden_size &&
            inputs[0]->shape[1] == tokens &&
            inputs[1]->shape[0] == wide && inputs[1]->shape[1] == tokens &&
            inputs[2]->shape[0] == packed_size &&
            inputs[2]->shape[1] == tokens && output->shape[0] == wide &&
            fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
            fp32_contiguous(*inputs[2]) && fp32_contiguous(*output);
        const float* branch = device_pointer_const<float>(*inputs[0]);
        const float* residual = device_pointer_const<float>(*inputs[1]);
        const float* packed = device_pointer_const<float>(*inputs[2]);
        float* destination = device_pointer<float>(*output);
        valid = valid && branch && residual && packed && destination;
        if (valid) {
            const size_t count = static_cast<size_t>(tokens) * wide;
            hc_post_cuda<<<
                static_cast<unsigned>((count + threads - 1) / threads),
                threads>>>(
                branch, inputs[0]->stride[1] / sizeof(float), residual,
                inputs[1]->stride[1] / sizeof(float), packed,
                inputs[2]->stride[1] / sizeof(float), destination,
                output->stride[1] / sizeof(float), count, hidden_size,
                hc_mult);
            if (!report_cuda(cudaGetLastError(), "hc_post_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    auto run_dsv4_compressor = [&](
            const GraphNode& compressor_node,
            const std::vector<const Tensor*>& compressor_inputs,
            Tensor* compressor_output) {
        if (compressor_inputs.size() < 10 || !compressor_output)
            return false;
        for (int index = 0; index < 10; ++index)
            if (!compressor_inputs[index])
                return false;
        const int hidden_size = graph_params::get_i32(
            compressor_node.params, 0, 4096);
        const int head_dim = graph_params::get_i32(
            compressor_node.params, 1, 512);
        const int ratio = graph_params::get_i32(
            compressor_node.params, 2, 4);
        const bool overlap = graph_params::get_i32(
            compressor_node.params, 3, 1) != 0;
        const bool rotate = graph_params::get_i32(
            compressor_node.params, 4, 0) != 0;
        const int rope_dim = graph_params::get_i32(
            compressor_node.params, 5, 64);
        const int original_context = graph_params::get_i32(
            compressor_node.params, 6, 65536);
        const int sequence = static_cast<int>(
            compressor_inputs[0]->shape[1]);
        const int coff = overlap ? 2 : 1;
        const int projected = coff * head_dim;
        const int state_rows = coff * ratio;
        const int cache_capacity = static_cast<int>(
            compressor_inputs[7]->shape[1]);
        const auto* wkv = impl_->find_weight(*compressor_inputs[1]);
        const auto* wgate = impl_->find_weight(*compressor_inputs[2]);
        const auto* ape = impl_->find_weight(*compressor_inputs[3]);
        const auto* norm = impl_->find_weight(*compressor_inputs[4]);
        auto fp32_weight = [](const Impl::DeviceWeight* weight,
                              int n, int k) {
            return weight && weight->layout == Impl::WeightLayout::Dense &&
                weight->type == CUDA_R_32F && weight->n == n &&
                weight->k == k && weight->data;
        };
        bool valid = hidden_size > 0 && head_dim > 0 && ratio > 0 &&
            sequence > 0 && projected > 0 && state_rows > 0 &&
            cache_capacity > 0 && rope_dim >= 0 && rope_dim <= head_dim &&
            compressor_inputs[0]->shape[0] == hidden_size &&
            compressor_inputs[0]->shape[2] == 1 &&
            compressor_inputs[0]->shape[3] == 1 &&
            fp32_contiguous(*compressor_inputs[0]) &&
            fp32_weight(wkv, projected, hidden_size) &&
            fp32_weight(wgate, projected, hidden_size) &&
            fp32_weight(ape, projected, ratio) &&
            fp32_weight(norm, head_dim, 1) &&
            compressor_inputs[5]->shape[0] == projected &&
            compressor_inputs[5]->shape[1] == state_rows &&
            compressor_inputs[6]->shape[0] == projected &&
            compressor_inputs[6]->shape[1] == state_rows &&
            compressor_inputs[7]->shape[0] == head_dim &&
            fp32_contiguous(*compressor_inputs[5]) &&
            fp32_contiguous(*compressor_inputs[6]) &&
            fp32_contiguous(*compressor_inputs[7]) &&
            compressor_inputs[8]->prec == Precision::INT32 &&
            compressor_inputs[8]->nelements() >= 1 &&
            compressor_inputs[8]->is_contiguous() &&
            compressor_inputs[9]->prec == Precision::INT32 &&
            compressor_inputs[9]->nelements() >= 1 &&
            compressor_inputs[9]->is_contiguous() &&
            fp32_contiguous(*compressor_output) &&
            compressor_output->nelements() >= 1;
        const float* hidden =
            device_pointer_const<float>(*compressor_inputs[0]);
        float* kv_state = device_pointer<float>(*compressor_inputs[5]);
        float* score_state =
            device_pointer<float>(*compressor_inputs[6]);
        float* cache = device_pointer<float>(*compressor_inputs[7]);
        const int32_t* start_position =
            device_pointer_const<int32_t>(*compressor_inputs[8]);
        const int32_t* real_tokens =
            device_pointer_const<int32_t>(*compressor_inputs[9]);
        float* emitted = device_pointer<float>(*compressor_output);
        auto* kv_values = static_cast<float*>(impl_->scratch(
            "dsv4_compressor_kv",
            static_cast<size_t>(sequence) * projected * sizeof(float)));
        auto* gate_values = static_cast<float*>(impl_->scratch(
            "dsv4_compressor_gate",
            static_cast<size_t>(sequence) * projected * sizeof(float)));
        auto* compressed = static_cast<float*>(impl_->scratch(
            "dsv4_compressor_vector",
            static_cast<size_t>(head_dim) * sizeof(float)));
        valid = valid && hidden && kv_state && score_state && cache &&
            start_position && real_tokens && emitted && kv_values &&
            gate_values && compressed;
        if (valid &&
            impl_->run_matmul_device(
                hidden,
                compressor_inputs[0]->stride[1] / sizeof(float),
                *compressor_inputs[1], kv_values, projected, sequence,
                projected, hidden_size, Activation::NONE, 0, 0) &&
            impl_->run_matmul_device(
                hidden,
                compressor_inputs[0]->stride[1] / sizeof(float),
                *compressor_inputs[2], gate_values, projected, sequence,
                projected, hidden_size, Activation::NONE, 0, 0)) {
            dsv4_compressor_state_cuda<<<1, 1>>>(
                kv_values, gate_values,
                static_cast<const float*>(ape->data),
                static_cast<const float*>(norm->data), kv_state,
                score_state, cache, start_position, real_tokens,
                compressed, emitted, sequence, head_dim, ratio, overlap,
                rotate, cache_capacity, rope_dim, original_context,
                graph_params::get_f32(
                    compressor_node.params, 0, 1e-6f),
                graph_params::get_f32(
                    compressor_node.params, 1, 160000.0f),
                graph_params::get_f32(
                    compressor_node.params, 2, 16.0f),
                graph_params::get_f32(
                    compressor_node.params, 3, 32.0f),
                graph_params::get_f32(
                    compressor_node.params, 4, 1.0f));
            if (!report_cuda(
                    cudaGetLastError(), "dsv4_compressor_state_cuda")) {
                impl_->failed = true;
                return false;
            }
            return true;
        }
        return false;
    };

    if (node.op_type == OpType::DSV4_COMPRESSOR &&
        run_dsv4_compressor(node, inputs, output)) {
        record_native();
        return;
    }

    if (node.op_type == OpType::DSV4_INDEXER && inputs.size() >= 13 &&
        inputs[0] && inputs[1] && inputs[2] && inputs[3] && inputs[4] &&
        inputs[5] && inputs[6] && inputs[7] && inputs[8] && inputs[9] &&
        inputs[10] && inputs[11] && inputs[12] && output) {
        const int hidden_size = graph_params::get_i32(
            node.params, 0, 4096);
        const int q_lora_rank = graph_params::get_i32(
            node.params, 1, 1024);
        const int num_heads = graph_params::get_i32(
            node.params, 2, 64);
        const int head_dim = graph_params::get_i32(
            node.params, 3, 128);
        const int top_k = graph_params::get_i32(node.params, 4, 512);
        const int ratio = graph_params::get_i32(node.params, 5, 4);
        const bool overlap =
            graph_params::get_i32(node.params, 6, 1) != 0;
        const bool rotate =
            graph_params::get_i32(node.params, 7, 1) != 0;
        const int rope_dim = graph_params::get_i32(
            node.params, 8, 64);
        const int original_context = graph_params::get_i32(
            node.params, 9, 65536);
        const int sequence = static_cast<int>(inputs[0]->shape[1]);
        const int query_width = num_heads * head_dim;
        const int cache_capacity = static_cast<int>(inputs[10]->shape[1]);
        const auto* wq_b = impl_->find_weight(*inputs[2]);
        const auto* weights_projection = impl_->find_weight(*inputs[3]);
        bool valid = hidden_size > 0 && q_lora_rank > 0 &&
            num_heads > 0 && head_dim > 0 && top_k > 0 && ratio > 0 &&
            sequence > 0 && query_width > 0 && cache_capacity > 0 &&
            rope_dim >= 0 && rope_dim <= head_dim &&
            inputs[0]->shape[0] == hidden_size &&
            inputs[0]->shape[1] == sequence &&
            inputs[1]->shape[0] == q_lora_rank &&
            inputs[1]->shape[1] == sequence &&
            fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
            wq_b && wq_b->n == query_width && wq_b->k == q_lora_rank &&
            weights_projection && weights_projection->n == num_heads &&
            weights_projection->k == hidden_size &&
            inputs[10]->shape[0] == head_dim &&
            fp32_contiguous(*inputs[10]) &&
            inputs[11]->prec == Precision::INT32 &&
            inputs[11]->nelements() >= 1 && inputs[11]->is_contiguous() &&
            inputs[12]->prec == Precision::INT32 &&
            inputs[12]->nelements() >= 1 && inputs[12]->is_contiguous() &&
            output->prec == Precision::INT32 && output->is_contiguous() &&
            output->shape[0] == top_k && output->shape[1] == sequence;
        const float* hidden = device_pointer_const<float>(*inputs[0]);
        const float* q_lora = device_pointer_const<float>(*inputs[1]);
        const int32_t* start_position =
            device_pointer_const<int32_t>(*inputs[11]);
        const int32_t* real_tokens =
            device_pointer_const<int32_t>(*inputs[12]);
        const float* cache = device_pointer_const<float>(*inputs[10]);
        int32_t* destination = device_pointer<int32_t>(*output);
        auto* query = static_cast<float*>(impl_->scratch(
            "dsv4_indexer_query",
            static_cast<size_t>(sequence) * query_width * sizeof(float)));
        auto* head_weights = static_cast<float*>(impl_->scratch(
            "dsv4_indexer_head_weights",
            static_cast<size_t>(sequence) * num_heads * sizeof(float)));
        auto* scores = static_cast<float*>(impl_->scratch(
            "dsv4_indexer_scores",
            static_cast<size_t>(sequence) * cache_capacity *
                sizeof(float)));
        auto* emitted = static_cast<float*>(impl_->scratch(
            "dsv4_indexer_emitted", sizeof(float)));
        valid = valid && hidden && q_lora && start_position && real_tokens &&
            cache && destination && query && head_weights && scores &&
            emitted;
        if (valid &&
            impl_->run_matmul_device(
                q_lora, inputs[1]->stride[1] / sizeof(float), *inputs[2],
                query, query_width, sequence, query_width, q_lora_rank,
                Activation::NONE, 0, 0) &&
            impl_->run_matmul_device(
                hidden, inputs[0]->stride[1] / sizeof(float), *inputs[3],
                head_weights, num_heads, sequence, num_heads, hidden_size,
                Activation::NONE, 0, 0) &&
            report_cuda(
                cudaMemset(destination, 0xff, output->nbytes()),
                "cudaMemset DSV4 indexer output")) {
            dsv4_indexer_prepare_cuda<<<1, 1>>>(
                query, head_weights, start_position, real_tokens, sequence,
                num_heads, head_dim, rope_dim, original_context,
                graph_params::get_f32(node.params, 1, 160000.0f),
                graph_params::get_f32(node.params, 2, 16.0f),
                graph_params::get_f32(node.params, 3, 32.0f),
                graph_params::get_f32(node.params, 4, 1.0f));
            if (!report_cuda(
                    cudaGetLastError(), "dsv4_indexer_prepare_cuda")) {
                impl_->failed = true;
                return;
            }
            GraphNode compressor_node;
            compressor_node.op_type = OpType::DSV4_COMPRESSOR;
            compressor_node.params.i32 = {
                hidden_size, head_dim, ratio, overlap ? 1 : 0,
                rotate ? 1 : 0, rope_dim, original_context};
            compressor_node.params.f32 = node.params.f32;
            std::vector<const Tensor*> compressor_inputs = {
                inputs[0], inputs[4], inputs[5], inputs[6], inputs[7],
                inputs[8], inputs[9], inputs[10], inputs[11], inputs[12]};
            Tensor compressor_output = Tensor::create(
                Precision::FP32, MemoryType::NONE, 1, 1, 1, 1);
            compressor_output.device_data = emitted;
            if (!run_dsv4_compressor(
                    compressor_node, compressor_inputs,
                    &compressor_output)) {
                if (impl_->failed)
                    return;
            } else {
                const size_t score_count =
                    static_cast<size_t>(sequence) * cache_capacity;
                dsv4_indexer_scores_cuda<<<
                    static_cast<unsigned>(
                        (score_count + threads - 1) / threads),
                    threads>>>(
                    query, head_weights, cache, start_position, real_tokens,
                    scores, sequence, num_heads, head_dim, ratio,
                    cache_capacity);
                dsv4_indexer_select_cuda<<<sequence, 1>>>(
                    scores, start_position, real_tokens, destination,
                    sequence, ratio, cache_capacity, top_k);
                if (!report_cuda(
                        cudaGetLastError(), "DSV4 indexer score/select")) {
                    impl_->failed = true;
                    return;
                }
                record_native();
                return;
            }
        }
    }

    if (node.op_type == OpType::DSV4_SPARSE_ATTN && inputs.size() >= 6 &&
        inputs[0] && inputs[1] && inputs[2] && inputs[3] && inputs[4] &&
        inputs[5] && output) {
        const int num_heads = graph_params::get_i32(
            node.params, 0, 64);
        const int head_dim = graph_params::get_i32(
            node.params, 1, 512);
        const int window_size = graph_params::get_i32(
            node.params, 2, 128);
        const int compress_ratio = graph_params::get_i32(
            node.params, 3, 0);
        const int rope_dim = graph_params::get_i32(
            node.params, 5, 64);
        const int original_context = graph_params::get_i32(
            node.params, 6, 65536);
        const int cache_input = graph_params::get_i32(
            node.params, 7, -1);
        const int indices_input = graph_params::get_i32(
            node.params, 8, -1);
        const int sequence = static_cast<int>(inputs[0]->shape[1]);
        const int query_width = num_heads * head_dim;
        const Tensor* compressed_cache =
            cache_input >= 0 && cache_input < static_cast<int>(inputs.size())
                ? inputs[cache_input] : nullptr;
        const Tensor* compressed_indices =
            indices_input >= 0 &&
                    indices_input < static_cast<int>(inputs.size())
                ? inputs[indices_input] : nullptr;
        int compressed_capacity = compressed_cache
            ? static_cast<int>(compressed_cache->shape[1]) : 0;
        int selected_width = compressed_indices
            ? static_cast<int>(compressed_indices->shape[0]) : 0;
        bool valid = num_heads > 0 && head_dim > 0 && window_size > 0 &&
            sequence > 0 && query_width > 0 && rope_dim >= 0 &&
            rope_dim <= head_dim && inputs[0]->shape[0] == query_width &&
            inputs[1]->shape[0] == head_dim &&
            inputs[1]->shape[1] == sequence &&
            fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
            inputs[2]->prec == Precision::FP32 &&
            inputs[2]->nelements() >= static_cast<size_t>(num_heads) &&
            inputs[2]->is_contiguous() &&
            inputs[3]->shape[0] == head_dim &&
            inputs[3]->shape[1] == window_size &&
            fp32_contiguous(*inputs[3]) &&
            inputs[4]->prec == Precision::INT32 &&
            inputs[4]->nelements() >= 1 && inputs[4]->is_contiguous() &&
            inputs[5]->prec == Precision::INT32 &&
            inputs[5]->nelements() >= 1 && inputs[5]->is_contiguous() &&
            output->shape[0] == query_width &&
            output->shape[1] == sequence && fp32_contiguous(*output);
        if (compress_ratio > 0) {
            valid = valid && compressed_cache &&
                compressed_capacity > 0 &&
                compressed_cache->shape[0] == head_dim &&
                fp32_contiguous(*compressed_cache);
            if (compressed_indices) {
                valid = valid && selected_width > 0 &&
                    compressed_indices->prec == Precision::INT32 &&
                    compressed_indices->shape[1] == sequence &&
                    compressed_indices->is_contiguous();
            }
        }
        const float* query_input =
            device_pointer_const<float>(*inputs[0]);
        const float* kv_input = device_pointer_const<float>(*inputs[1]);
        const float* sink = device_pointer_const<float>(*inputs[2]);
        float* window = device_pointer<float>(*inputs[3]);
        const int32_t* start_position =
            device_pointer_const<int32_t>(*inputs[4]);
        const int32_t* real_tokens =
            device_pointer_const<int32_t>(*inputs[5]);
        const float* compressed = compressed_cache
            ? device_pointer_const<float>(*compressed_cache) : nullptr;
        const int32_t* selected = compressed_indices
            ? device_pointer_const<int32_t>(*compressed_indices) : nullptr;
        float* destination = device_pointer<float>(*output);
        auto* rotated_query = static_cast<float*>(impl_->scratch(
            "dsv4_sparse_query",
            static_cast<size_t>(sequence) * query_width * sizeof(float)));
        auto* rotated_kv = static_cast<float*>(impl_->scratch(
            "dsv4_sparse_kv",
            static_cast<size_t>(sequence) * head_dim * sizeof(float)));
        valid = valid && query_input && kv_input && sink && window &&
            start_position && real_tokens && destination && rotated_query &&
            rotated_kv && (compress_ratio <= 0 || compressed) &&
            (!compressed_indices || selected);
        if (valid && report_cuda(
                cudaMemset(destination, 0, output->nbytes()),
                "cudaMemset DSV4 sparse output")) {
            dsv4_sparse_prepare_cuda<<<1, 1>>>(
                query_input, kv_input, start_position, real_tokens,
                rotated_query, rotated_kv, sequence, num_heads, head_dim,
                rope_dim, original_context,
                graph_params::get_f32(node.params, 1, 1e-6f),
                graph_params::get_f32(node.params, 2, 160000.0f),
                graph_params::get_f32(node.params, 3, 16.0f),
                graph_params::get_f32(node.params, 4, 32.0f),
                graph_params::get_f32(node.params, 5, 1.0f));
            if (!report_cuda(
                    cudaGetLastError(), "dsv4_sparse_prepare_cuda")) {
                impl_->failed = true;
                return;
            }
            float softmax_scale = graph_params::get_f32(
                node.params, 0, 0.0f);
            if (softmax_scale <= 0.0f)
                softmax_scale = 1.0f /
                    std::sqrt(static_cast<float>(head_dim));
            const int tasks = sequence * num_heads;
            dsv4_sparse_attention_cuda<<<
                static_cast<unsigned>((tasks + threads - 1) / threads),
                threads>>>(
                rotated_query, rotated_kv, sink, window, compressed,
                selected, start_position, real_tokens, destination,
                sequence, num_heads, head_dim, window_size, compress_ratio,
                compressed_capacity, selected_width, softmax_scale,
                rope_dim, original_context,
                graph_params::get_f32(node.params, 2, 160000.0f),
                graph_params::get_f32(node.params, 3, 16.0f),
                graph_params::get_f32(node.params, 4, 32.0f),
                graph_params::get_f32(node.params, 5, 1.0f));
            dsv4_sparse_update_window_cuda<<<1, 1>>>(
                rotated_kv, start_position, real_tokens, window, sequence,
                head_dim, window_size);
            if (!report_cuda(
                    cudaGetLastError(), "DSV4 sparse attention/update")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::DSV4_GROUPED_LINEAR &&
        inputs.size() >= 2 && inputs[0] && inputs[1] && output) {
        const int groups = graph_params::get_i32(node.params, 0, 8);
        const auto* prepared = impl_->find_weight(*inputs[1]);
        const int input_width = static_cast<int>(inputs[0]->shape[0]);
        const int tokens = static_cast<int>(inputs[0]->shape[1]);
        const int output_width = static_cast<int>(output->shape[0]);
        const int group_width = groups > 0 ? input_width / groups : 0;
        const int rank = groups > 0 ? output_width / groups : 0;
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const bool valid = groups > 0 && input_width > 0 && tokens > 0 &&
            input_width % groups == 0 && output_width > 0 &&
            output_width % groups == 0 &&
            inputs[0]->prec == Precision::FP32 &&
            inputs[0]->stride[0] == sizeof(float) &&
            inputs[0]->shape[2] == 1 && inputs[0]->shape[3] == 1 &&
            inputs[0]->stride[1] >=
                static_cast<size_t>(input_width) * sizeof(float) &&
            output->prec == Precision::FP32 &&
            output->stride[0] == sizeof(float) &&
            output->shape[1] == tokens && output->shape[2] == 1 &&
            output->shape[3] == 1 && output->stride[1] >=
                static_cast<size_t>(output_width) * sizeof(float) &&
            prepared &&
            prepared->layout == Impl::WeightLayout::Fp8Block128 &&
            prepared->e8m0_scales && prepared->group_size == 128 &&
            prepared->n == output_width && prepared->k == group_width &&
            prepared->groups_per_row == (group_width + 127) / 128 &&
            inputs[1]->shape[0] == output_width &&
            inputs[1]->shape[1] == group_width && source && destination;
        if (valid) {
            const size_t count =
                static_cast<size_t>(tokens) * output_width;
            dsv4_grouped_fp8_linear_cuda<<<
                static_cast<unsigned>((count + threads - 1) / threads),
                threads>>>(
                source, inputs[0]->stride[1] / sizeof(float),
                static_cast<const uint8_t*>(prepared->data),
                prepared->e8m0_scales, destination,
                output->stride[1] / sizeof(float), tokens, groups,
                group_width, rank, prepared->groups_per_row);
            if (!report_cuda(
                    cudaGetLastError(), "DSV4 grouped FP8 linear")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::MOE && inputs.size() >= 4 && inputs[0] &&
        inputs[1] && inputs[2] && inputs[3] && output) {
        const int hidden_size = graph_params::get_i32(
            node.params, 0, static_cast<int>(output->shape[0]));
        const int num_experts = graph_params::get_i32(node.params, 1, 0);
        const int top_k = graph_params::get_i32(node.params, 2, 0);
        const int intermediate_size =
            graph_params::get_i32(node.params, 3, 0);
        const int shared_intermediate_size =
            graph_params::get_i32(node.params, 4, intermediate_size);
        const int score_function =
            graph_params::get_i32(node.params, 5, 0);
        const bool normalize_topk =
            graph_params::get_i32(node.params, 6, 1) != 0;
        const bool has_shared =
            graph_params::get_i32(node.params, 7, 1) != 0;
        const int num_groups = std::max(
            1, graph_params::get_i32(node.params, 8, 1));
        int topk_groups = graph_params::get_i32(
            node.params, 9, num_groups);
        if (topk_groups <= 0 || topk_groups > num_groups)
            topk_groups = num_groups;
        const bool shared_has_gate =
            graph_params::get_i32(node.params, 10, 1) != 0;
        const int router_bias_input = graph_params::get_i32(
            node.params, 11, has_shared ? 8 : -1);
        const int token_ids_input =
            graph_params::get_i32(node.params, 12, -1);
        const int hash_table_input =
            graph_params::get_i32(node.params, 13, -1);
        const float routed_scaling_factor =
            graph_params::get_f32(node.params, 0, 1.0f);
        const float swiglu_limit =
            graph_params::get_f32(node.params, 1, 0.0f);
        const int tokens = static_cast<int>(inputs[0]->shape[1]);

        const auto* router = impl_->find_weight(*inputs[1]);
        const auto* gate_up = impl_->find_weight(*inputs[2]);
        const auto* down = impl_->find_weight(*inputs[3]);
        const auto* gate_up_source =
            static_cast<const MoeSsdTensorSource*>(
                inputs[2]->moe_ssd_source);
        const auto* down_source =
            static_cast<const MoeSsdTensorSource*>(
                inputs[3]->moe_ssd_source);
        const bool streamed_experts = gate_up_source || down_source;
        auto valid_microscaled_weight = [](const Impl::DeviceWeight* weight) {
            if (!weight || !weight->e8m0_scales)
                return false;
            if (weight->layout == Impl::WeightLayout::Fp8Block128)
                return weight->group_size == 128 &&
                    weight->groups_per_row == (weight->k + 127) / 128;
            if (weight->layout == Impl::WeightLayout::Mxfp4RowMajor)
                return weight->group_size == 32 && weight->k % 32 == 0 &&
                    weight->groups_per_row == weight->k / 32;
            return false;
        };
        auto valid_matmul_weight = [&](const Impl::DeviceWeight* weight) {
            return weight &&
                (weight->layout == Impl::WeightLayout::Dense ||
                 weight->layout == Impl::WeightLayout::Q8RowMajor ||
                 weight->layout == Impl::WeightLayout::Q4Bg32 ||
                 weight->layout == Impl::WeightLayout::Q4Bg128 ||
                 valid_microscaled_weight(weight));
        };
        auto valid_expert_weight = [](const Impl::DeviceWeight* weight) {
            if (!weight)
                return false;
            if (weight->layout == Impl::WeightLayout::Q8RowMajor)
                return weight->scales && weight->group_size > 0 &&
                    weight->groups_per_row > 0 &&
                    weight->group_size * weight->groups_per_row >= weight->k;
            if (weight->layout == Impl::WeightLayout::Mxfp4RowMajor)
                return weight->e8m0_scales && weight->group_size == 32 &&
                    weight->k % 32 == 0 &&
                    weight->groups_per_row == weight->k / 32;
            if (weight->layout == Impl::WeightLayout::Fp8Block128)
                return weight->e8m0_scales && weight->group_size == 128 &&
                    weight->groups_per_row == (weight->k + 127) / 128;
            return weight->layout == Impl::WeightLayout::Dense ||
                weight->layout == Impl::WeightLayout::Q4Bg32 ||
                weight->layout == Impl::WeightLayout::Q4Bg128;
        };
        const Tensor* router_bias =
            router_bias_input >= 0 &&
                    static_cast<size_t>(router_bias_input) < inputs.size()
                ? inputs[router_bias_input]
                : nullptr;
        const bool hash_routing =
            token_ids_input >= 0 && hash_table_input >= 0;
        const Tensor* token_ids =
            hash_routing &&
                    static_cast<size_t>(token_ids_input) < inputs.size()
                ? inputs[token_ids_input]
                : nullptr;
        const Tensor* hash_table =
            hash_routing &&
                    static_cast<size_t>(hash_table_input) < inputs.size()
                ? inputs[hash_table_input]
                : nullptr;
        const auto* prepared_hash_table = hash_table
            ? impl_->find_weight(*hash_table) : nullptr;
        const int hash_vocab_size = hash_table
            ? static_cast<int>(hash_table->shape[1]) : 0;
        const bool valid_hash_routing = !hash_routing ||
            (score_function == 2 && token_ids && hash_table &&
             token_ids->prec == Precision::INT32 &&
             token_ids->nelements() >= tokens &&
             hash_table->prec == Precision::INT32 &&
             hash_table->shape[0] == top_k && hash_vocab_size > 0 &&
             prepared_hash_table &&
             prepared_hash_table->layout ==
                 Impl::WeightLayout::Int32Lookup &&
             prepared_hash_table->n == top_k &&
             prepared_hash_table->k == hash_vocab_size &&
             device_pointer_const<int32_t>(*token_ids));
        const bool fp32_row_major_hidden =
            inputs[0]->prec == Precision::FP32 &&
            inputs[0]->stride[0] == sizeof(float) &&
            inputs[0]->shape[2] == 1 && inputs[0]->shape[3] == 1 &&
            inputs[0]->stride[1] >=
                static_cast<size_t>(hidden_size) * sizeof(float);
        auto streamed_weight_layout = [](const MoeSsdTensorSource* source) {
            if (!source)
                return -1;
            const auto& spec = source->spec;
            if (spec.precision == Precision::FP16 ||
                spec.precision == Precision::FP32)
                return static_cast<int>(Impl::WeightLayout::Dense);
            if (spec.precision == Precision::INT8)
                return static_cast<int>(Impl::WeightLayout::Q8RowMajor);
            if (spec.precision == Precision::INT4) {
                if ((spec.flags & MappedFile::FLAG_INT4_BG32) != 0)
                    return static_cast<int>(Impl::WeightLayout::Q4Bg32);
                if ((spec.flags & MappedFile::FLAG_INT4_BG128) != 0)
                    return static_cast<int>(Impl::WeightLayout::Q4Bg128);
            }
            if (spec.precision == Precision::FP8_E4M3)
                return static_cast<int>(
                    Impl::WeightLayout::Fp8Block128);
            if (spec.precision == Precision::MXFP4)
                return static_cast<int>(
                    Impl::WeightLayout::Mxfp4RowMajor);
            return -1;
        };
        auto valid_streamed_weight = [&](const MoeSsdTensorSource* source,
                                         int rows, int cols) {
            if (!source || source->spec.rows != rows ||
                source->spec.cols != cols || rows <= 0 || cols <= 0)
                return false;
            const auto& spec = source->spec;
            const int layout = streamed_weight_layout(source);
            if (layout == static_cast<int>(Impl::WeightLayout::Dense)) {
                const uint64_t element_bytes =
                    spec.precision == Precision::FP32 ? 4 : 2;
                return (spec.precision == Precision::FP16 ||
                        spec.precision == Precision::FP32) &&
                    spec.flags == 0 && spec.group_size == 0 &&
                    spec.groups_per_row == 0 && spec.scales_bytes == 0 &&
                    spec.data_bytes ==
                        static_cast<uint64_t>(rows) * cols * element_bytes;
            }
            if (layout == static_cast<int>(
                              Impl::WeightLayout::Q8RowMajor)) {
                return spec.group_size > 0 && spec.groups_per_row > 0 &&
                    static_cast<uint64_t>(spec.group_size) *
                            spec.groups_per_row >=
                        static_cast<uint64_t>(cols) &&
                    spec.data_bytes ==
                        static_cast<uint64_t>(rows) * cols &&
                    spec.scales_bytes == static_cast<uint64_t>(rows) *
                        spec.groups_per_row * sizeof(float);
            }
            if (layout == static_cast<int>(Impl::WeightLayout::Q4Bg32)) {
                return rows % 8 == 0 && cols % 32 == 0 &&
                    spec.group_size == 32 &&
                    spec.groups_per_row == static_cast<uint32_t>(cols / 32) &&
                    spec.data_bytes == static_cast<uint64_t>(rows / 8) *
                        (cols / 32) * sizeof(Q4B8G32Block);
            }
            if (layout == static_cast<int>(Impl::WeightLayout::Q4Bg128)) {
                return rows % 8 == 0 && cols % 128 == 0 &&
                    spec.group_size == 128 &&
                    spec.groups_per_row ==
                        static_cast<uint32_t>(cols / 128) &&
                    spec.data_bytes == static_cast<uint64_t>(rows / 8) *
                        (cols / 128) * sizeof(Q4B8G128Block);
            }
            if (layout == static_cast<int>(
                              Impl::WeightLayout::Fp8Block128)) {
                const uint64_t groups_per_row =
                    (static_cast<uint64_t>(cols) + 127) / 128;
                return spec.flags == MappedFile::FLAG_FP8_BLOCK128 &&
                    rows % 128 == 0 && spec.group_size == 128 &&
                    spec.groups_per_row == groups_per_row &&
                    spec.data_bytes ==
                        static_cast<uint64_t>(rows) * cols &&
                    spec.scales_bytes ==
                        static_cast<uint64_t>(rows / 128) *
                            groups_per_row;
            }
            if (layout == static_cast<int>(
                              Impl::WeightLayout::Mxfp4RowMajor)) {
                return cols % 32 == 0 && spec.group_size == 32 &&
                    spec.groups_per_row == static_cast<uint32_t>(cols / 32) &&
                    spec.data_bytes ==
                        static_cast<uint64_t>(rows) * cols / 2 &&
                    spec.scales_bytes == static_cast<uint64_t>(rows) *
                        spec.groups_per_row;
            }
            return false;
        };
        const bool valid_streamed_experts = streamed_experts &&
            gate_up_source && down_source && gate_up_source->cache &&
            gate_up_source->cache == down_source->cache &&
            gate_up_source->spec.num_experts == num_experts &&
            down_source->spec.num_experts == num_experts &&
            valid_streamed_weight(
                gate_up_source, 2 * intermediate_size, hidden_size) &&
            valid_streamed_weight(
                down_source, hidden_size, intermediate_size);
        const bool valid_expert_pair = streamed_experts
            ? valid_streamed_experts
            : valid_expert_weight(gate_up) && valid_expert_weight(down);
        bool valid = fp32_row_major_hidden && fp32_contiguous(*output) &&
            router && valid_expert_pair &&
            hidden_size > 0 && num_experts > 0 && num_experts >= top_k &&
            top_k > 0 && top_k <= 64 && intermediate_size > 0 &&
            num_groups <= 64 && tokens > 0 && score_function >= 0 &&
            score_function <= 2 && valid_hash_routing &&
            inputs[0]->shape[0] == hidden_size &&
            output->shape[0] == hidden_size && output->shape[1] == tokens &&
            router->n == num_experts && router->k == hidden_size &&
            valid_matmul_weight(router) &&
            (streamed_experts ||
             (gate_up->n == num_experts * 2 * intermediate_size &&
              gate_up->k == hidden_size &&
              down->n == num_experts * hidden_size &&
              down->k == intermediate_size));
        const bool gate_up_mxfp4 = streamed_experts
            ? gate_up_source &&
                gate_up_source->spec.precision == Precision::MXFP4
            : gate_up &&
                gate_up->layout == Impl::WeightLayout::Mxfp4RowMajor;
        const bool down_mxfp4 = streamed_experts
            ? down_source && down_source->spec.precision == Precision::MXFP4
            : down && down->layout == Impl::WeightLayout::Mxfp4RowMajor;
        const bool gate_up_fp8 = streamed_experts
            ? gate_up_source &&
                gate_up_source->spec.precision == Precision::FP8_E4M3
            : gate_up && gate_up->layout == Impl::WeightLayout::Fp8Block128;
        const bool down_fp8 = streamed_experts
            ? down_source &&
                down_source->spec.precision == Precision::FP8_E4M3
            : down && down->layout == Impl::WeightLayout::Fp8Block128;
        const bool bf16_activations = gate_up_mxfp4 && down_mxfp4;
        const bool quantize_gate_up_activation = gate_up_mxfp4 || gate_up_fp8;
        const bool quantize_down_activation = down_mxfp4 || down_fp8;
        valid = valid && gate_up_mxfp4 == down_mxfp4 &&
            (!gate_up_fp8 || (2 * intermediate_size) % 128 == 0) &&
            (!down_fp8 || hidden_size % 128 == 0);
        if (router_bias) {
            valid = valid && router_bias->prec == Precision::FP32 &&
                router_bias->nelements() >= num_experts &&
                device_pointer_const<float>(*router_bias);
        }

        const Tensor* shared_gate = nullptr;
        const Tensor* shared_up = nullptr;
        const Tensor* shared_down = nullptr;
        const Tensor* shared_scale_weight = nullptr;
        if (has_shared) {
            valid = valid && shared_intermediate_size > 0 &&
                inputs.size() >= 7 && inputs[4] && inputs[5] && inputs[6];
            if (valid) {
                shared_gate = inputs[4];
                shared_up = inputs[5];
                shared_down = inputs[6];
                const auto* prepared_gate = impl_->find_weight(*shared_gate);
                const auto* prepared_up = impl_->find_weight(*shared_up);
                const auto* prepared_down = impl_->find_weight(*shared_down);
                valid = valid_matmul_weight(prepared_gate) &&
                    valid_matmul_weight(prepared_up) &&
                    valid_matmul_weight(prepared_down) &&
                    prepared_gate->n == shared_intermediate_size &&
                    prepared_gate->k == hidden_size &&
                    prepared_up->n == shared_intermediate_size &&
                    prepared_up->k == hidden_size &&
                    prepared_down->n == hidden_size &&
                    prepared_down->k == shared_intermediate_size;
                if (shared_has_gate) {
                    valid = valid && inputs.size() >= 8 && inputs[7];
                    if (valid) {
                        shared_scale_weight = inputs[7];
                        const auto* prepared_scale =
                            impl_->find_weight(*shared_scale_weight);
                        valid = valid_matmul_weight(prepared_scale) &&
                            prepared_scale->n == 1 &&
                            prepared_scale->k == hidden_size;
                    }
                }
            }
        }

        const float* hidden = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        valid = valid && hidden && destination;
        if (valid) {
            const size_t route_count =
                static_cast<size_t>(tokens) * top_k;
            auto* logits = static_cast<float*>(impl_->scratch(
                "moe_logits", static_cast<size_t>(tokens) * num_experts *
                    sizeof(float)));
            auto* route_indices = static_cast<int*>(impl_->scratch(
                "moe_route_indices", route_count * sizeof(int)));
            auto* route_weights = static_cast<float*>(impl_->scratch(
                "moe_route_weights", route_count * sizeof(float)));
            auto* routed_intermediate = static_cast<float*>(impl_->scratch(
                "moe_routed_intermediate",
                route_count * intermediate_size * sizeof(float)));
            auto* routed_hidden_fp8 = quantize_gate_up_activation
                ? static_cast<float*>(impl_->scratch(
                      "moe_routed_hidden_fp8",
                      static_cast<size_t>(tokens) * hidden_size *
                          sizeof(float)))
                : nullptr;
            auto* routed_intermediate_fp8 = quantize_down_activation
                ? static_cast<float*>(impl_->scratch(
                      "moe_routed_intermediate_fp8",
                      route_count * intermediate_size * sizeof(float)))
                : nullptr;
            const void* gate_up_data = gate_up ? gate_up->data : nullptr;
            const float* gate_up_scales = gate_up ? gate_up->scales : nullptr;
            const uint8_t* gate_up_e8m0_scales =
                gate_up ? gate_up->e8m0_scales : nullptr;
            int gate_up_layout = gate_up
                ? static_cast<int>(gate_up->layout)
                : streamed_weight_layout(gate_up_source);
            int gate_up_group_size = gate_up
                ? gate_up->group_size
                : (gate_up_source ? gate_up_source->spec.group_size : 0);
            int gate_up_groups_per_row = gate_up
                ? gate_up->groups_per_row
                : (gate_up_source ? gate_up_source->spec.groups_per_row : 0);
            const void* down_data = down ? down->data : nullptr;
            const float* down_scales = down ? down->scales : nullptr;
            const uint8_t* down_e8m0_scales =
                down ? down->e8m0_scales : nullptr;
            int down_layout = down
                ? static_cast<int>(down->layout)
                : streamed_weight_layout(down_source);
            int down_group_size = down
                ? down->group_size
                : (down_source ? down_source->spec.group_size : 0);
            int down_groups_per_row = down
                ? down->groups_per_row
                : (down_source ? down_source->spec.groups_per_row : 0);
            const bool gate_up_dense_fp32 = gate_up
                ? gate_up->type == CUDA_R_32F
                : gate_up_source &&
                    gate_up_source->spec.precision == Precision::FP32;
            const bool down_dense_fp32 = down
                ? down->type == CUDA_R_32F
                : down_source &&
                    down_source->spec.precision == Precision::FP32;
            const void* const* gate_up_expert_data = nullptr;
            const void* const* gate_up_expert_scales = nullptr;
            const void* const* down_expert_data = nullptr;
            const void* const* down_expert_scales = nullptr;
            valid = logits && route_indices && route_weights &&
                routed_intermediate &&
                (!quantize_gate_up_activation || routed_hidden_fp8) &&
                (!quantize_down_activation || routed_intermediate_fp8) &&
                impl_->run_matmul_device(
                    hidden,
                    static_cast<int>(inputs[0]->stride[1] / sizeof(float)),
                    *inputs[1], logits, num_experts, tokens, num_experts,
                    hidden_size, Activation::NONE, 0, -1);
            if (valid) {
                if (hash_routing) {
                    auto* invalid_route = static_cast<int*>(impl_->scratch(
                        "moe_invalid_hash_route", sizeof(int)));
                    int host_invalid_route = 0;
                    valid = invalid_route &&
                        report_cuda(cudaMemset(invalid_route, 0, sizeof(int)),
                                    "cudaMemset MOE hash route status");
                    if (valid) {
                        moe_select_hash_routes_cuda<<<tokens, 1>>>(
                            logits, device_pointer_const<int32_t>(*token_ids),
                            static_cast<const int32_t*>(
                                prepared_hash_table->data),
                            route_indices, route_weights, invalid_route,
                            tokens, num_experts, top_k, hash_vocab_size,
                            normalize_topk, routed_scaling_factor);
                        valid = report_cuda(
                                    cudaGetLastError(),
                                    "CUDA MOE hash routing") &&
                            report_cuda(
                                cudaMemcpy(&host_invalid_route, invalid_route,
                                           sizeof(int),
                                           cudaMemcpyDeviceToHost),
                                "cudaMemcpy MOE hash route status") &&
                            host_invalid_route == 0;
                    }
                } else {
                    moe_select_routes_cuda<<<tokens, 1>>>(
                        logits,
                        router_bias
                            ? device_pointer_const<float>(*router_bias)
                            : nullptr,
                        route_indices, route_weights, tokens, num_experts,
                        top_k, score_function, normalize_topk, num_groups,
                        topk_groups, routed_scaling_factor);
                }
            }
            if (valid && streamed_experts) {
                std::vector<int> host_routes(route_count);
                valid = report_cuda(
                    cudaMemcpy(host_routes.data(), route_indices,
                               route_count * sizeof(int),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy MOE SSD routes");
                std::vector<int> selected_experts;
                std::vector<bool> selected_flags(
                    static_cast<size_t>(num_experts), false);
                if (valid) {
                    selected_experts.reserve(route_count);
                    for (int expert : host_routes) {
                        if (expert < 0 || expert >= num_experts) {
                            valid = false;
                            break;
                        }
                        if (!selected_flags[static_cast<size_t>(expert)]) {
                            selected_flags[static_cast<size_t>(expert)] = true;
                            selected_experts.push_back(expert);
                        }
                    }
                }
                std::vector<int> fallback_slots(
                    static_cast<size_t>(num_experts), -1);
                size_t fallback_count = 0;
                if (valid) {
                    impl_->prepare_moe_device_experts(
                        gate_up_source, down_source, selected_experts);
                    std::vector<int> host_requests;
                    host_requests.reserve(selected_experts.size());
                    for (int expert : selected_experts) {
                        if (!impl_->has_moe_device_expert(
                                gate_up_source, down_source, expert)) {
                            host_requests.push_back(expert);
                        }
                    }
                    for (int expert : host_requests) {
                        if (impl_->moe_device_cache_stats.capacity_bytes ==
                                0 ||
                            !impl_->reserve_moe_device_expert(
                                gate_up_source, down_source, expert)) {
                            fallback_slots[static_cast<size_t>(expert)] =
                                static_cast<int>(fallback_count++);
                        }
                    }
                    valid = !impl_->failed &&
                        (host_requests.empty() ||
                         gate_up_source->cache->request_many(
                             gate_up_source, down_source, host_requests));
                }
                const size_t selected_count = selected_experts.size();
                const size_t pair_payload_bytes =
                    gate_up_source->spec.data_bytes +
                    gate_up_source->spec.scales_bytes +
                    down_source->spec.data_bytes +
                    down_source->spec.scales_bytes;
                impl_->moe_device_cache_stats.peak_selected_bytes = std::max(
                    impl_->moe_device_cache_stats.peak_selected_bytes,
                    selected_count * pair_payload_bytes);
                impl_->moe_device_cache_stats.fallback_scratch_bytes =
                    std::max(
                        impl_->moe_device_cache_stats.fallback_scratch_bytes,
                        fallback_count * pair_payload_bytes);
                auto* compact_gate_up = valid
                    && fallback_count != 0
                    ? static_cast<uint8_t*>(impl_->scratch(
                          "moe_ssd_gate_up_data",
                          fallback_count * gate_up_source->spec.data_bytes))
                    : nullptr;
                auto* compact_gate_up_scales =
                    valid && fallback_count != 0 &&
                            gate_up_source->spec.scales_bytes != 0
                    ? static_cast<uint8_t*>(impl_->scratch(
                          "moe_ssd_gate_up_scales",
                          fallback_count *
                              gate_up_source->spec.scales_bytes))
                    : nullptr;
                auto* compact_down = valid
                    && fallback_count != 0
                    ? static_cast<uint8_t*>(impl_->scratch(
                          "moe_ssd_down_data",
                          fallback_count * down_source->spec.data_bytes))
                    : nullptr;
                auto* compact_down_scales =
                    valid && fallback_count != 0 &&
                            down_source->spec.scales_bytes != 0
                    ? static_cast<uint8_t*>(impl_->scratch(
                          "moe_ssd_down_scales",
                          fallback_count * down_source->spec.scales_bytes))
                    : nullptr;
                const size_t pointer_table_bytes =
                    static_cast<size_t>(num_experts) * sizeof(void*);
                auto* device_gate_up_data = valid
                    ? static_cast<const void**>(impl_->scratch(
                          "moe_ssd_gate_up_ptrs", pointer_table_bytes))
                    : nullptr;
                auto* device_gate_up_scales =
                    valid && gate_up_source->spec.scales_bytes != 0
                    ? static_cast<const void**>(impl_->scratch(
                          "moe_ssd_gate_up_scale_ptrs",
                          pointer_table_bytes))
                    : nullptr;
                auto* device_down_data = valid
                    ? static_cast<const void**>(impl_->scratch(
                          "moe_ssd_down_ptrs", pointer_table_bytes))
                    : nullptr;
                auto* device_down_scales =
                    valid && down_source->spec.scales_bytes != 0
                    ? static_cast<const void**>(impl_->scratch(
                          "moe_ssd_down_scale_ptrs",
                          pointer_table_bytes))
                    : nullptr;
                valid = valid &&
                    (fallback_count == 0 ||
                     (compact_gate_up && compact_down)) &&
                    (gate_up_source->spec.scales_bytes == 0 ||
                     (device_gate_up_scales &&
                      (fallback_count == 0 || compact_gate_up_scales))) &&
                    (down_source->spec.scales_bytes == 0 ||
                     (device_down_scales &&
                      (fallback_count == 0 || compact_down_scales))) &&
                    device_gate_up_data && device_down_data;
                std::vector<const void*> host_gate_up_data(
                    static_cast<size_t>(num_experts), nullptr);
                std::vector<const void*> host_gate_up_scales(
                    static_cast<size_t>(num_experts), nullptr);
                std::vector<const void*> host_down_data(
                    static_cast<size_t>(num_experts), nullptr);
                std::vector<const void*> host_down_scales(
                    static_cast<size_t>(num_experts), nullptr);
                for (size_t slot = 0; valid && slot < selected_count; ++slot) {
                    const int expert = selected_experts[slot];
                    auto* cached = impl_->find_pinned_moe_device_expert(
                        gate_up_source, down_source, selected_experts[slot]);
                    Tensor gate_view;
                    Tensor down_view;
                    if (!cached || !cached->ready) {
                        valid = gate_up_source->cache->acquire(
                            gate_up_source, down_source,
                            selected_experts[slot], gate_view, down_view);
                        valid = valid && gate_view.data && down_view.data &&
                            (gate_up_source->spec.scales_bytes == 0 ||
                             gate_view.scales || gate_view.e8m0_scales) &&
                            (down_source->spec.scales_bytes == 0 ||
                             down_view.scales || down_view.e8m0_scales);
                        if (valid && cached) {
                            valid = impl_->fill_moe_device_expert(
                                *cached, gate_up_source, down_source,
                                gate_view, down_view);
                        }
                    }
                    if (valid) {
                        if (cached && cached->ready) {
                            host_gate_up_data[expert] = cached->gate_up_data;
                            host_gate_up_scales[expert] =
                                cached->gate_up_scales;
                            host_down_data[expert] = cached->down_data;
                            host_down_scales[expert] = cached->down_scales;
                            impl_->moe_device_cache_stats
                                .direct_expert_bytes +=
                                gate_up_source->spec.data_bytes +
                                gate_up_source->spec.scales_bytes +
                                down_source->spec.data_bytes +
                                down_source->spec.scales_bytes;
                            continue;
                        }
                        const int fallback_slot = fallback_slots[
                            static_cast<size_t>(expert)];
                        valid = fallback_slot >= 0;
                        if (!valid)
                            break;
                        uint8_t* gate_data = compact_gate_up +
                            static_cast<size_t>(fallback_slot) *
                                gate_up_source->spec.data_bytes;
                        uint8_t* gate_scales = compact_gate_up_scales
                            ? compact_gate_up_scales +
                                static_cast<size_t>(fallback_slot) *
                                    gate_up_source->spec.scales_bytes
                            : nullptr;
                        uint8_t* down_data_pointer = compact_down +
                            static_cast<size_t>(fallback_slot) *
                                down_source->spec.data_bytes;
                        uint8_t* down_scales_pointer = compact_down_scales
                            ? compact_down_scales +
                                static_cast<size_t>(fallback_slot) *
                                    down_source->spec.scales_bytes
                            : nullptr;
                        valid = report_cuda(
                            cudaMemcpy(gate_data, gate_view.data,
                                       gate_up_source->spec.data_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy MOE SSD gate/up data");
                        const void* gate_host_scales = gate_view.e8m0_scales
                            ? static_cast<const void*>(gate_view.e8m0_scales)
                            : static_cast<const void*>(gate_view.scales);
                        if (valid && gate_up_source->spec.scales_bytes != 0) {
                            valid = report_cuda(
                                cudaMemcpy(gate_scales, gate_host_scales,
                                           gate_up_source->spec.scales_bytes,
                                           cudaMemcpyHostToDevice),
                                "cudaMemcpy MOE SSD gate/up scales");
                        }
                        if (valid) {
                            valid = report_cuda(
                                cudaMemcpy(down_data_pointer, down_view.data,
                                           down_source->spec.data_bytes,
                                           cudaMemcpyHostToDevice),
                                "cudaMemcpy MOE SSD down data");
                        }
                        const void* down_host_scales = down_view.e8m0_scales
                            ? static_cast<const void*>(down_view.e8m0_scales)
                            : static_cast<const void*>(down_view.scales);
                        if (valid && down_source->spec.scales_bytes != 0) {
                            valid = report_cuda(
                                cudaMemcpy(down_scales_pointer,
                                           down_host_scales,
                                           down_source->spec.scales_bytes,
                                           cudaMemcpyHostToDevice),
                                "cudaMemcpy MOE SSD down scales");
                        }
                        host_gate_up_data[expert] = gate_data;
                        host_gate_up_scales[expert] = gate_scales;
                        host_down_data[expert] = down_data_pointer;
                        host_down_scales[expert] = down_scales_pointer;
                        if (valid) {
                            impl_->moe_device_cache_stats
                                .fallback_host_to_device_bytes +=
                                pair_payload_bytes;
                        }
                    }
                }
                if (valid) {
                    valid = report_cuda(
                        cudaMemcpy(device_gate_up_data,
                                   host_gate_up_data.data(),
                                   pointer_table_bytes,
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy MOE SSD gate/up pointers") &&
                        report_cuda(
                            cudaMemcpy(device_down_data,
                                       host_down_data.data(),
                                       pointer_table_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy MOE SSD down pointers");
                }
                if (valid && gate_up_source->spec.scales_bytes != 0) {
                    valid = report_cuda(
                        cudaMemcpy(device_gate_up_scales,
                                   host_gate_up_scales.data(),
                                   pointer_table_bytes,
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy MOE SSD gate/up scale pointers");
                }
                if (valid && down_source->spec.scales_bytes != 0) {
                    valid = report_cuda(
                        cudaMemcpy(device_down_scales,
                                   host_down_scales.data(),
                                   pointer_table_bytes,
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy MOE SSD down scale pointers");
                }
                if (valid) {
                    gate_up_expert_data = device_gate_up_data;
                    gate_up_expert_scales = device_gate_up_scales;
                    down_expert_data = device_down_data;
                    down_expert_scales = device_down_scales;
                    gate_up_source->cache->retain_for_next_forward(
                        gate_up_source, down_source, selected_experts);
                }
            }
            if (valid) {
                const float* routed_hidden = hidden;
                size_t routed_hidden_stride =
                    inputs[0]->stride[1] / sizeof(float);
                if (quantize_gate_up_activation) {
                    const int hidden_blocks =
                        tokens * ((hidden_size + 127) / 128);
                    quantize_activation_fp8_cuda<<<hidden_blocks, 1>>>(
                        hidden,
                        static_cast<int>(routed_hidden_stride),
                        routed_hidden_fp8, tokens, hidden_size);
                    valid = report_cuda(
                        cudaGetLastError(),
                        "CUDA MOE FP8 routed activation");
                    routed_hidden = routed_hidden_fp8;
                    routed_hidden_stride = hidden_size;
                }
                const size_t intermediate_count =
                    route_count * intermediate_size;
                if (valid) {
                    moe_gate_up_cuda<<<
                        static_cast<unsigned>(
                            (intermediate_count + threads - 1) / threads),
                        threads>>>(
                        routed_hidden, routed_hidden_stride, route_indices,
                        route_weights, gate_up_data, gate_up_expert_data,
                        gate_up_dense_fp32,
                        gate_up_layout, gate_up_scales,
                        gate_up_expert_scales,
                        gate_up_group_size, gate_up_groups_per_row,
                        gate_up_e8m0_scales, routed_intermediate, tokens,
                        top_k, hidden_size, intermediate_size, swiglu_limit,
                        bf16_activations);
                }
                const float* down_activation = routed_intermediate;
                if (valid && quantize_down_activation) {
                    const int intermediate_blocks = static_cast<int>(
                        route_count * ((intermediate_size + 127) / 128));
                    quantize_activation_fp8_cuda<<<
                        intermediate_blocks, 1>>>(
                        routed_intermediate, intermediate_size,
                        routed_intermediate_fp8,
                        static_cast<int>(route_count), intermediate_size);
                    valid = report_cuda(
                        cudaGetLastError(),
                        "CUDA MOE FP8 down activation");
                    down_activation = routed_intermediate_fp8;
                }
                const size_t output_count =
                    static_cast<size_t>(tokens) * hidden_size;
                if (valid) {
                    moe_down_cuda<<<
                        static_cast<unsigned>(
                            (output_count + threads - 1) / threads),
                        threads>>>(
                        route_indices, route_weights, down_activation,
                        down_data, down_expert_data,
                        down_dense_fp32,
                        down_layout, down_scales, down_expert_scales,
                        down_group_size, down_groups_per_row,
                        down_e8m0_scales, destination,
                        output->stride[1] / sizeof(float), tokens, top_k,
                        hidden_size, intermediate_size, bf16_activations);
                    valid = report_cuda(
                        cudaGetLastError(), "CUDA MOE routed");
                }
            }

            if (valid && has_shared) {
                const size_t shared_count =
                    static_cast<size_t>(tokens) * shared_intermediate_size;
                auto* shared_gate_output = static_cast<float*>(impl_->scratch(
                    "moe_shared_gate", shared_count * sizeof(float)));
                auto* shared_up_output = static_cast<float*>(impl_->scratch(
                    "moe_shared_up", shared_count * sizeof(float)));
                auto* shared_intermediate = static_cast<float*>(
                    impl_->scratch("moe_shared_intermediate",
                                   shared_count * sizeof(float)));
                auto* shared_output = static_cast<float*>(impl_->scratch(
                    "moe_shared_output",
                    static_cast<size_t>(tokens) * hidden_size *
                        sizeof(float)));
                auto* shared_scale = shared_has_gate
                    ? static_cast<float*>(impl_->scratch(
                          "moe_shared_scale",
                          static_cast<size_t>(tokens) * sizeof(float)))
                    : nullptr;
                valid = shared_gate_output && shared_up_output &&
                    shared_intermediate && shared_output &&
                    (!shared_has_gate || shared_scale) &&
                    impl_->run_matmul_device(
                        hidden,
                        static_cast<int>(inputs[0]->stride[1] /
                                         sizeof(float)),
                        *shared_gate, shared_gate_output,
                        shared_intermediate_size, tokens,
                        shared_intermediate_size, hidden_size,
                        Activation::NONE, 0, -1) &&
                    impl_->run_matmul_device(
                        hidden,
                        static_cast<int>(inputs[0]->stride[1] /
                                         sizeof(float)),
                        *shared_up, shared_up_output,
                        shared_intermediate_size, tokens,
                        shared_intermediate_size, hidden_size,
                        Activation::NONE, 0, -1);
                if (valid) {
                    moe_swiglu_cuda<<<
                        static_cast<unsigned>((shared_count + threads - 1) /
                                              threads), threads>>>(
                        shared_gate_output, shared_up_output,
                        shared_intermediate, shared_count, swiglu_limit,
                        bf16_activations);
                    valid = report_cuda(
                        cudaGetLastError(), "CUDA MOE shared SwiGLU") &&
                        impl_->run_matmul_device(
                            shared_intermediate, shared_intermediate_size,
                            *shared_down, shared_output, hidden_size, tokens,
                            hidden_size, shared_intermediate_size,
                            Activation::NONE, 0, -1);
                    if (valid && bf16_activations) {
                        const size_t shared_output_count =
                            static_cast<size_t>(tokens) * hidden_size;
                        round_bf16_cuda<<<
                            static_cast<unsigned>(
                                (shared_output_count + threads - 1) /
                                threads), threads>>>(
                            shared_output, shared_output_count);
                        valid = report_cuda(
                            cudaGetLastError(),
                            "CUDA MOE shared BF16 output");
                    }
                }
                if (valid && shared_has_gate) {
                    valid = impl_->run_matmul_device(
                        hidden,
                        static_cast<int>(inputs[0]->stride[1] /
                                         sizeof(float)),
                        *shared_scale_weight, shared_scale, 1, tokens, 1,
                        hidden_size, Activation::NONE, 0, -1);
                    if (valid && bf16_activations) {
                        round_bf16_cuda<<<
                            static_cast<unsigned>((tokens + threads - 1) /
                                                  threads), threads>>>(
                            shared_scale, tokens);
                        valid = report_cuda(
                            cudaGetLastError(),
                            "CUDA MOE shared BF16 scale");
                    }
                }
                if (valid) {
                    const size_t output_count =
                        static_cast<size_t>(tokens) * hidden_size;
                    moe_add_shared_cuda<<<
                        static_cast<unsigned>((output_count + threads - 1) /
                                              threads), threads>>>(
                        destination, output->stride[1] / sizeof(float),
                        shared_output, shared_scale, tokens, hidden_size,
                        shared_has_gate);
                    valid = report_cuda(
                        cudaGetLastError(), "CUDA MOE shared output");
                }
            }
            if (!valid) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::MATMUL_BATCH && !inputs.empty() && output &&
        (inputs.size() & 1) == 0 && output->prec == Precision::FP32 &&
        output->is_contiguous()) {
        const size_t batch = inputs.size() / 2;
        const int n = static_cast<int>(output->shape[0]);
        const int m = static_cast<int>(output->shape[1]);
        bool valid = batch > 0 && output->shape[2] ==
            static_cast<int64_t>(batch) && output->shape[3] == 1;
        for (size_t index = 0; index < batch && valid; ++index) {
            const Tensor* activation = inputs[index * 2];
            const Tensor* weight = inputs[index * 2 + 1];
            const Impl::DeviceWeight* prepared = weight
                ? impl_->find_weight(*weight) : nullptr;
            valid = activation && weight && prepared &&
                fp32_contiguous(*activation) &&
                activation->shape[1] == m && weight->shape[0] == n &&
                activation->shape[0] == weight->shape[1] &&
                prepared->n == n && prepared->k == weight->shape[1];
        }
        float* destination = device_pointer<float>(*output);
        valid = valid && destination;
        if (valid) {
            for (size_t index = 0; index < batch; ++index) {
                const Tensor& activation = *inputs[index * 2];
                const Tensor& weight = *inputs[index * 2 + 1];
                const float* source =
                    device_pointer_const<float>(activation);
                float* slice = destination +
                    index * (output->stride[2] / sizeof(float));
                if (!impl_->run_matmul_device(
                        source,
                        static_cast<int>(activation.stride[1] /
                                         sizeof(float)),
                        weight, slice,
                        static_cast<int>(output->stride[1] / sizeof(float)),
                        m, n, static_cast<int>(activation.shape[0]),
                        Activation::NONE, 0, -1)) {
                    impl_->failed = true;
                    return;
                }
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::MATMUL ||
         node.op_type == OpType::GEMV_SPARSE_A) &&
        inputs.size() >= 2 && inputs[0] && inputs[1] && output &&
        inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 &&
        impl_->find_weight(*inputs[1])) {
        const Tensor& a = *inputs[0];
        const Tensor& weight = *inputs[1];
        const int m = static_cast<int>(a.shape[1]);
        const int k = static_cast<int>(a.shape[0]);
        const int n = static_cast<int>(weight.shape[0]);
        const int lda = static_cast<int>(a.stride[1] / sizeof(float));
        const int ldc = static_cast<int>(output->stride[1] / sizeof(float));
        const Activation activation_kind = node.op_type == OpType::MATMUL
            ? static_cast<Activation>(
                  graph_params::get_i32(node.params, 0, 0))
            : Activation::NONE;
        const float* device_a = device_pointer_const<float>(a);
        float* device_c = device_pointer<float>(*output);
        const bool ok = device_a && device_c
            ? impl_->run_matmul_device(
                  device_a, lda, weight, device_c, ldc, m, n, k,
                  activation_kind,
                  graph_params::get_i32(node.params, 1, 0),
                  graph_params::get_i32(node.params, 2, -1))
            : impl_->run_matmul(
                  a.ptr<float>(), lda, weight, output->ptr<float>(), ldc,
                  m, n, k, activation_kind,
                  graph_params::get_i32(node.params, 1, 0),
                  graph_params::get_i32(node.params, 2, -1));
        if (ok) {
            record_native();
            return;
        }
        impl_->failed = true;
        return;
    }

    if (node.op_type == OpType::QK_RMS_NORM_ROPE &&
        inputs.size() >= 6 && inputs[0] && inputs[1] && inputs[2] &&
        inputs[3] && inputs[4] && inputs[5] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        inputs[2]->prec == Precision::FP32 &&
        inputs[3]->prec == Precision::FP32 &&
        inputs[4]->prec == Precision::FP32 &&
        inputs[5]->prec == Precision::FP32 && fp32_contiguous(*output)) {
        const int width = static_cast<int>(output->shape[0]);
        const int sequence_length = static_cast<int>(output->shape[1]);
        const int total_heads = static_cast<int>(output->shape[2]);
        const int query_heads = graph_params::get_i32(
            node.params, 2, total_heads);
        const int key_heads = total_heads - query_heads;
        const int rope_dim = graph_params::get_i32(
            node.params, 0, width);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        const size_t query_count = static_cast<size_t>(width) *
            sequence_length * query_heads;
        const size_t key_count = static_cast<size_t>(width) *
            sequence_length * key_heads;
        const float* query = device_pointer_const<float>(*inputs[0]);
        const float* key = device_pointer_const<float>(*inputs[1]);
        const float* query_weight =
            device_pointer_const<float>(*inputs[2]);
        const float* key_weight = device_pointer_const<float>(*inputs[3]);
        const float* cosine = device_pointer_const<float>(*inputs[4]);
        const float* sine = device_pointer_const<float>(*inputs[5]);
        float* destination = device_pointer<float>(*output);
        auto* normalized = static_cast<float*>(impl_->scratch(
            "qk_rms_norm_rope", (query_count + key_count) * sizeof(float)));
        const bool valid = width > 0 && width <= threads &&
            sequence_length > 0 && query_heads > 0 && key_heads > 0 &&
            rope_dim > 0 && rope_dim <= width && rope_dim % 2 == 0 &&
            inputs[0]->nelements() == static_cast<int64_t>(query_count) &&
            inputs[1]->nelements() == static_cast<int64_t>(key_count) &&
            inputs[2]->nelements() >= width &&
            inputs[3]->nelements() >= width &&
            inputs[4]->shape[0] >= rope_dim / 2 &&
            inputs[5]->shape[0] >= rope_dim / 2 && query && key &&
            query_weight && key_weight && cosine && sine && destination &&
            normalized;
        if (valid) {
            const float epsilon = graph_params::get_f32(
                node.params, 0, 1e-6f);
            const int query_rows = sequence_length * query_heads;
            const int key_rows = sequence_length * key_heads;
            float* normalized_key = normalized + query_count;
            rms_norm_cuda<<<query_rows, threads>>>(
                query, query_weight, normalized, width, query_rows,
                inputs[0]->stride[1] / sizeof(float), width, epsilon);
            rms_norm_cuda<<<key_rows, threads>>>(
                key, key_weight, normalized_key, width, key_rows,
                inputs[1]->stride[1] / sizeof(float), width, epsilon);
            rope_cuda<<<
                static_cast<unsigned>((query_count + threads - 1) / threads),
                threads>>>(
                normalized, cosine, sine, destination, width,
                sequence_length, query_heads, query_heads, rope_dim,
                interleave, 1, width,
                static_cast<size_t>(width) * sequence_length,
                query_count, inputs[4]->stride[0] / sizeof(float),
                inputs[4]->stride[1] / sizeof(float),
                inputs[5]->stride[0] / sizeof(float),
                inputs[5]->stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                output->stride[3] / sizeof(float));
            rope_cuda<<<
                static_cast<unsigned>((key_count + threads - 1) / threads),
                threads>>>(
                normalized_key, cosine, sine, destination + query_count,
                width, sequence_length, key_heads, key_heads, rope_dim,
                interleave, 1, width,
                static_cast<size_t>(width) * sequence_length,
                key_count, inputs[4]->stride[0] / sizeof(float),
                inputs[4]->stride[1] / sizeof(float),
                inputs[5]->stride[0] / sizeof(float),
                inputs[5]->stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                output->stride[3] / sizeof(float));
            if (!report_cuda(cudaGetLastError(),
                             "qk_rms_norm_rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RMS_NORM_ROPE && inputs.size() >= 4 &&
        inputs[0] && inputs[1] && inputs[2] && inputs[3] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        inputs[2]->prec == Precision::FP32 &&
        inputs[3]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& source = *inputs[0];
        const Tensor& weight = *inputs[1];
        const Tensor& cosine = *inputs[2];
        const Tensor& sine = *inputs[3];
        const float* source_data = device_pointer_const<float>(source);
        const float* weight_data = device_pointer_const<float>(weight);
        const float* cosine_data = device_pointer_const<float>(cosine);
        const float* sine_data = device_pointer_const<float>(sine);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(source.shape[0]);
        const int sequence_length = static_cast<int>(output->shape[1]);
        const int channels =
            static_cast<int>(output->shape[2] * output->shape[3]);
        const int rows = width > 0
            ? static_cast<int>(source.nelements() / width) : 0;
        const int rope_dim = graph_params::get_i32(node.params, 0, width);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        const size_t count = static_cast<size_t>(source.nelements());
        const size_t scratch_bytes = count * sizeof(float);
        if (source_data && weight_data && cosine_data && sine_data &&
            destination && width > 0 && rows > 0 && rope_dim > 0 &&
            rope_dim <= width && rope_dim % 2 == 0 &&
            rows == sequence_length * channels &&
            cosine.shape[0] >= rope_dim / 2 &&
            sine.shape[0] >= rope_dim / 2 &&
            impl_->reserve(impl_->norm_scratch,
                           impl_->norm_scratch_bytes, scratch_bytes)) {
            auto* normalized = static_cast<float*>(impl_->norm_scratch);
            rms_norm_cuda<<<rows, threads>>>(
                source_data, weight_data, normalized, width, rows,
                static_cast<size_t>(width), static_cast<size_t>(width),
                graph_params::get_f32(node.params, 0, 1e-6f));
            rope_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                              threads), threads>>>(
                normalized, cosine_data, sine_data, destination, width,
                sequence_length, channels, static_cast<int>(output->shape[2]),
                rope_dim, interleave, 1, static_cast<size_t>(width),
                static_cast<size_t>(width) * sequence_length,
                static_cast<size_t>(width) * sequence_length *
                    output->shape[2],
                cosine.stride[0] / sizeof(float),
                cosine.stride[1] / sizeof(float),
                sine.stride[0] / sizeof(float),
                sine.stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                output->stride[3] / sizeof(float));
            if (!report_cuda(cudaGetLastError(), "rms_norm_rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::ADD_RMS_NORM && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        fp32_contiguous(*inputs[2]) && output->prec == Precision::FP32 &&
        inputs[0]->stride[0] == sizeof(float) &&
        inputs[1]->stride[0] == sizeof(float) &&
        output->stride[0] == sizeof(float) &&
        inputs[0]->shape[2] == 1 && inputs[0]->shape[3] == 1 &&
        inputs[1]->shape[2] == 1 && inputs[1]->shape[3] == 1 &&
        output->shape[2] == 1 && output->shape[3] == 1 &&
        inputs[0]->nelements() == inputs[1]->nelements() &&
        inputs[0]->nelements() == output->nelements()) {
        float* residual = device_pointer<float>(*inputs[0]);
        const float* update = device_pointer_const<float>(*inputs[1]);
        const float* weight = device_pointer_const<float>(*inputs[2]);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(inputs[0]->shape[0]);
        const int rows = static_cast<int>(inputs[0]->shape[1]);
        if (residual && update && weight && destination && width > 0 &&
            rows > 0) {
            add_rms_norm_cuda<<<rows, threads>>>(
                residual, update, weight, destination, width, rows,
                inputs[0]->stride[1] / sizeof(float),
                inputs[1]->stride[1] / sizeof(float),
                output->stride[1] / sizeof(float),
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!report_cuda(cudaGetLastError(), "add_rms_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::LAYER_NORM && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        fp32_contiguous(*inputs[1]) && fp32_contiguous(*inputs[2]) &&
        output->prec == Precision::FP32 &&
        inputs[0]->shape[2] == 1 && inputs[0]->shape[3] == 1 &&
        inputs[0]->stride[0] == sizeof(float) &&
        output->stride[0] == sizeof(float) &&
        same_shape(*inputs[0], *output)) {
        const int width = static_cast<int>(inputs[0]->shape[0]);
        const int rows = static_cast<int>(inputs[0]->shape[1]);
        const float* input = device_pointer_const<float>(*inputs[0]);
        const float* weight = device_pointer_const<float>(*inputs[1]);
        const float* bias = device_pointer_const<float>(*inputs[2]);
        float* destination = device_pointer<float>(*output);
        if (input && weight && bias && destination && width > 0 && rows > 0 &&
            inputs[1]->nelements() >= width &&
            inputs[2]->nelements() >= width) {
            layer_norm_cuda<<<rows, threads>>>(
                input, weight, bias, destination, width, rows,
                inputs[0]->stride[1] / sizeof(float),
                output->stride[1] / sizeof(float),
                graph_params::get_f32(node.params, 0, 1e-5f));
            if (!report_cuda(cudaGetLastError(), "layer_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RMS_NORM && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output &&
        inputs[0]->prec == Precision::FP32 &&
        fp32_contiguous(*inputs[1]) && output->prec == Precision::FP32 &&
        inputs[0]->shape[2] == 1 && inputs[0]->shape[3] == 1 &&
        inputs[0]->stride[0] == sizeof(float) &&
        output->stride[0] == sizeof(float) &&
        same_shape(*inputs[0], *output)) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        const float* weight = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(inputs[0]->shape[0]);
        const int rows = static_cast<int>(inputs[0]->shape[1]);
        if (source && weight && destination && width > 0 && rows > 0 &&
            inputs[1]->nelements() >= width) {
            rms_norm_cuda<<<rows, threads>>>(
                source, weight, destination, width, rows,
                inputs[0]->stride[1] / sizeof(float),
                output->stride[1] / sizeof(float),
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!report_cuda(cudaGetLastError(), "rms_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::ADD || node.op_type == OpType::MUL) &&
        inputs.size() >= 2 && inputs[0] && inputs[1] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 &&
        same_shape(*inputs[0], *output) &&
        broadcasts_to(*inputs[1], *inputs[0])) {
        const float* lhs = device_pointer_const<float>(*inputs[0]);
        const float* rhs = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (lhs && rhs && destination) {
            if (fp32_contiguous(*inputs[0]) &&
                fp32_contiguous(*inputs[1]) &&
                fp32_contiguous(*output) &&
                same_shape(*inputs[0], *inputs[1])) {
                binary_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                                    threads), threads>>>(
                    lhs, rhs, destination, count,
                    node.op_type == OpType::MUL);
            } else {
                binary_strided_cuda<<<
                    static_cast<unsigned>((count + threads - 1) / threads),
                    threads>>>(
                    lhs, rhs, destination, count, cuda_layout(*inputs[0]),
                    cuda_layout(*inputs[1]), cuda_layout(*output),
                    node.op_type == OpType::MUL);
            }
            if (!report_cuda(cudaGetLastError(), "binary_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SIGMOID_MUL && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 && fp32_contiguous(*output) &&
        same_shape(*inputs[0], *inputs[1]) &&
        inputs[0]->nelements() == output->nelements()) {
        const float* value = device_pointer_const<float>(*inputs[0]);
        const float* gate = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (value && gate && destination) {
            if (inputs[0]->is_contiguous() && inputs[1]->is_contiguous()) {
                sigmoid_mul_cuda<<<
                    static_cast<unsigned>((count + threads - 1) / threads),
                    threads>>>(value, gate, destination, count);
            } else {
                sigmoid_mul_strided_cuda<<<
                    static_cast<unsigned>((count + threads - 1) / threads),
                    threads>>>(
                    value, gate, destination, count, inputs[0]->shape[0],
                    inputs[0]->shape[1], inputs[0]->shape[2],
                    inputs[0]->stride[0] / sizeof(float),
                    inputs[0]->stride[1] / sizeof(float),
                    inputs[0]->stride[2] / sizeof(float),
                    inputs[0]->stride[3] / sizeof(float),
                    inputs[1]->stride[0] / sizeof(float),
                    inputs[1]->stride[1] / sizeof(float),
                    inputs[1]->stride[2] / sizeof(float),
                    inputs[1]->stride[3] / sizeof(float));
            }
            if (!report_cuda(cudaGetLastError(), "sigmoid_mul_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    int unary_operation = -1;
    switch (node.op_type) {
    case OpType::SILU: unary_operation = 0; break;
    case OpType::GELU: unary_operation = 1; break;
    case OpType::TANH: unary_operation = 2; break;
    case OpType::SIGMOID:
    case OpType::SIGMOID_EXACT: unary_operation = 3; break;
    case OpType::EXP:
    case OpType::EXP_EXACT: unary_operation = 4; break;
    case OpType::SOFTPLUS: unary_operation = 5; break;
    default: break;
    }
    if (unary_operation >= 0 && !inputs.empty() && inputs[0] && output &&
        inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 &&
        same_shape(*inputs[0], *output)) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (source && destination) {
            if (fp32_contiguous(*inputs[0]) && fp32_contiguous(*output)) {
                unary_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                                   threads), threads>>>(
                    source, destination, count, unary_operation);
            } else {
                unary_strided_cuda<<<
                    static_cast<unsigned>((count + threads - 1) / threads),
                    threads>>>(
                    source, destination, count, cuda_layout(*inputs[0]),
                    cuda_layout(*output), unary_operation);
            }
            if (!report_cuda(cudaGetLastError(), "unary_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SWIGLU && !inputs.empty() && inputs[0] &&
        output && fp32_contiguous(*inputs[0]) && fp32_contiguous(*output) &&
        inputs[0]->shape[0] == output->shape[0] * 2) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (source && destination) {
            swiglu_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                               threads), threads>>>(
                source, destination, count,
                static_cast<size_t>(output->shape[0]));
            if (!report_cuda(cudaGetLastError(), "swiglu_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (impl_->operator_fallback ==
        OperatorFallbackPolicy::REQUIRE_NATIVE) {
        std::fprintf(
            stderr,
            "CudaBackend: native-only mode rejected %s operator fallback\n",
            op_type_name(node.op_type));
        impl_->failed = true;
        return;
    }

    synchronize_for_host_read();
    if (impl_->failed)
        return;
    std::vector<Tensor> host_tensors;
    std::vector<const Tensor*> host_inputs;
    std::vector<std::vector<uint32_t>> host_storage;
    std::vector<int> staged_storage(inputs.size(), -1);
    host_tensors.reserve(inputs.size());
    host_inputs.reserve(inputs.size());
    host_storage.reserve(inputs.size());
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        const Tensor* input = inputs[input_index];
        if (!input) {
            host_inputs.push_back(nullptr);
            continue;
        }
        host_tensors.push_back(*input);
        Tensor& host = host_tensors.back();
        // Constants retain their package-native host view for CPU reference
        // kernels. Device activations and persistent state are staged
        // explicitly; Tensor::data may be an opaque cudaMalloc address.
        const bool device_weight = impl_->find_weight(*input) != nullptr;
        if (input->device_data && !device_weight) {
            const size_t bytes = input->view_span_bytes();
            host_storage.emplace_back((bytes + sizeof(uint32_t) - 1) /
                                      sizeof(uint32_t));
            staged_storage[input_index] =
                static_cast<int>(host_storage.size() - 1);
            void* destination = host_storage.back().data();
            const void* source = device_pointer_const<uint8_t>(*input);
            if (!source || !report_cuda(
                    cudaMemcpy(destination, source, bytes,
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy fallback input")) {
                impl_->failed = true;
                return;
            }
            host.data = destination;
        } else if (device_weight && !host.data) {
            // CUDA can execute device-only weights natively, but a CPU
            // fallback needs the package-native host view.
            impl_->failed = true;
            return;
        }
        host.device_data = nullptr;
        host.device_offset = 0;
        host_inputs.push_back(&host);
    }
    if (!output || !output->device_data) {
        impl_->failed = true;
        return;
    }
    const size_t output_bytes = output->nbytes();
    std::vector<uint32_t> host_output_storage(
        (output_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    Tensor host_output = *output;
    host_output.data = host_output_storage.data();
    host_output.device_data = nullptr;
    host_output.device_offset = 0;
    impl_->cpu.clear_dispatch_error();
    impl_->cpu.dispatch(node, host_inputs, &host_output, thread_pool);
    if (impl_->cpu.dispatch_failed()) {
        impl_->failed = true;
        return;
    }
    // Reference kernels may update cache/recurrent inputs in place. Copy all
    // staged tensors back so the bridge preserves those semantics; copying a
    // read-only activation is harmless and keeps this generic.
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        const int storage_index = staged_storage[input_index];
        if (storage_index < 0)
            continue;
        const Tensor& input = *inputs[input_index];
        const size_t bytes = input.view_span_bytes();
        if (!report_cuda(
                cudaMemcpy(
                    device_pointer<uint8_t>(input),
                    host_storage[static_cast<size_t>(storage_index)].data(),
                    bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy fallback updated input")) {
            impl_->failed = true;
            return;
        }
    }
    if (host_output.data != host_output_storage.data() ||
        !report_cuda(
            cudaMemcpy(device_pointer<uint8_t>(*output), host_output.data,
                       output_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy fallback output")) {
        impl_->failed = true;
        return;
    }
    ++impl_->fallback_ops[static_cast<uint32_t>(node.op_type)];
}

void CudaBackend::lm_head_gemv(const float* activation_host,
                               const Tensor& weight, float* output_host,
                               int n, int k, int activation) {
    if (!impl_->run_matmul(
            activation_host, k, weight, output_host, n, 1, n, k,
            static_cast<Activation>(activation), 0, -1))
        impl_->failed = true;
}

void CudaBackend::lm_head_gemv_device_and_end_graph(
    const Tensor& activation, size_t activation_element_offset,
    const Tensor& weight, float* output_host, int n, int k,
    int activation_kind) {
    const float* activation_base = device_pointer_const<float>(activation);
    const float* source = activation_base
        ? activation_base + activation_element_offset : nullptr;
    const size_t output_size = static_cast<size_t>(n) * sizeof(float);
    if (!source ||
        !impl_->reserve(impl_->output, impl_->output_bytes, output_size) ||
        !impl_->run_matmul_device(
            source, k, weight, static_cast<float*>(impl_->output), n,
            1, n, k, static_cast<Activation>(activation_kind), 0, -1) ||
        !report_cuda(cudaMemcpy(output_host, impl_->output, output_size,
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy lm_head output")) {
        impl_->failed = true;
        return;
    }
    ++impl_->native_ops[static_cast<uint32_t>(OpType::MATMUL)];
}
