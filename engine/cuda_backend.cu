#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "engine/cuda_backend.h"
#include "engine/engine.h"

#include "kernels/activations.h"
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

__global__ void tile_cuda(
    const float* input, float* output, size_t count,
    CudaTensorLayout input_layout, CudaTensorLayout output_layout) {
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
    output[layout_offset(coordinates, output_layout, false)] =
        input[layout_offset(input_coordinates, input_layout, false)];
}

__global__ void concat_cuda(
    const float* input, float* output, size_t count,
    CudaTensorLayout input_layout, CudaTensorLayout output_layout,
    int concat_dimension, int64_t concat_offset) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t coordinates[4];
    logical_coordinates(index, input_layout, coordinates);
    const size_t input_offset = layout_offset(
        coordinates, input_layout, false);
    coordinates[concat_dimension] += static_cast<size_t>(concat_offset);
    output[layout_offset(coordinates, output_layout, false)] =
        input[input_offset];
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
    const void* data, const float* scales, bool dense_fp32, int layout,
    int group_size, int groups_per_row, int row, int column, int width) {
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

__global__ void moe_gate_up_cuda(
    const float* hidden, size_t hidden_stride, const int* route_indices,
    const void* gate_up, bool gate_up_fp32, int gate_up_layout,
    const float* gate_up_scales, int gate_up_group_size,
    int gate_up_groups_per_row, float* intermediate, int tokens, int top_k,
    int hidden_size, int intermediate_size, float swiglu_limit) {
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
    const float* input = hidden + static_cast<size_t>(token) * hidden_stride;
    const int gate_row = expert * 2 * intermediate_size + dimension;
    const int up_row = gate_row + intermediate_size;
    float gate = 0.0f;
    float up = 0.0f;
    for (int inner = 0; inner < hidden_size; ++inner) {
        const float value = input[inner];
        gate += value * moe_weight_value_cuda(
            gate_up, gate_up_scales, gate_up_fp32, gate_up_layout,
            gate_up_group_size, gate_up_groups_per_row, gate_row, inner,
            hidden_size);
        up += value * moe_weight_value_cuda(
            gate_up, gate_up_scales, gate_up_fp32, gate_up_layout,
            gate_up_group_size, gate_up_groups_per_row, up_row, inner,
            hidden_size);
    }
    if (swiglu_limit > 0.0f) {
        gate = fminf(gate, swiglu_limit);
        up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    intermediate[index] = gate / (1.0f + expf(-gate)) * up;
}

__global__ void moe_down_cuda(
    const int* route_indices, const float* route_weights,
    const float* intermediate, const void* down, bool down_fp32,
    int down_layout, const float* down_scales, int down_group_size,
    int down_groups_per_row, float* output, size_t output_stride,
    int tokens, int top_k, int hidden_size, int intermediate_size) {
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
        const int row = expert * hidden_size + dimension;
        float contribution = 0.0f;
        for (int inner = 0; inner < intermediate_size; ++inner)
            contribution += intermediate[
                static_cast<size_t>(route) * intermediate_size + inner] *
                moe_weight_value_cuda(
                    down, down_scales, down_fp32, down_layout,
                    down_group_size, down_groups_per_row, row, inner,
                    intermediate_size);
        result += route_weights[route] * contribution;
    }
    output[static_cast<size_t>(token) * output_stride + dimension] = result;
}

__global__ void moe_swiglu_cuda(
    const float* gate, const float* up, float* output, size_t count,
    float swiglu_limit) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    float gate_value = gate[index];
    float up_value = up[index];
    if (swiglu_limit > 0.0f) {
        gate_value = fminf(gate_value, swiglu_limit);
        up_value = fminf(fmaxf(up_value, -swiglu_limit), swiglu_limit);
    }
    output[index] = gate_value / (1.0f + expf(-gate_value)) * up_value;
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

__global__ void append_kv_cuda(
    const float* key, const float* value, float* key_cache,
    float* value_cache, int num_kv_heads, int current_length,
    int past_length, int max_length, int key_dim, int value_dim,
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
        key_cache[(static_cast<size_t>(head) * max_length + past_length +
                   position) * key_dim + dimension] =
            key[static_cast<size_t>(head) * key_head_stride +
                static_cast<size_t>(position) * key_position_stride +
                dimension];
    }
    if (dimension < value_dim) {
        value_cache[(static_cast<size_t>(head) * max_length + past_length +
                     position) * value_dim + dimension] =
            value[static_cast<size_t>(head) * value_head_stride +
                  static_cast<size_t>(position) * value_position_stride +
                  dimension];
    }
}

__global__ void sdpa_scores_cuda(
    const float* query, const float* key, float* scores,
    const float* mask, int num_heads, int num_kv_heads,
    int query_length, int key_length, int past_length, int key_dim,
    int key_capacity, bool cached, bool causal, float scale,
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
    const float* key_row = cached
        ? key + (static_cast<size_t>(key_head) * key_capacity +
                 key_position) * key_dim
        : key + static_cast<size_t>(key_head) * key_head_stride +
            static_cast<size_t>(key_position) * key_position_stride;
    float dot = 0.0f;
    for (int dimension = 0; dimension < key_dim; ++dimension)
        dot += query_row[static_cast<size_t>(dimension) *
                         query_feature_stride] *
            key_row[static_cast<size_t>(dimension) * key_feature_stride];
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
    const float* scores, const float* value, float* output,
    int num_heads, int num_kv_heads, int query_length, int key_length,
    int value_dim, int value_capacity, bool cached,
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
            const float* value_row = cached
                ? value + (static_cast<size_t>(key_head) * value_capacity +
                           position) * value_dim
                : value + static_cast<size_t>(key_head) * value_head_stride +
                    static_cast<size_t>(position) * value_position_stride;
            result += probability *
                value_row[static_cast<size_t>(dimension) *
                          value_feature_stride];
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

int signed_nibble(uint8_t value) {
    const int nibble = value & 0x0f;
    return nibble >= 8 ? nibble - 16 : nibble;
}

}  // namespace

struct CudaBackend::Impl {
    enum class WeightLayout : uint8_t {
        Dense = 0,
        Q8RowMajor,
        Q4Bg32,
        Q4Bg128,
    };

    struct DeviceWeight {
        void* data = nullptr;
        float* scales = nullptr;
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

    bool ok = false;
    bool failed = false;
    cublasHandle_t cublas = nullptr;
    CPUBackend cpu;
    std::unordered_map<const void*, DeviceWeight> weights;
    std::unordered_map<const void*, const DeviceWeight*> weights_by_device;
    std::vector<void*> device_allocations;
    std::vector<void*> managed_allocations;
    std::unordered_map<void*, size_t> managed_sizes;
    std::multimap<size_t, void*> free_managed;
    std::unordered_map<std::string, BoundaryBuffer> boundary_buffers;
    std::unordered_map<uint32_t, uint64_t> native_ops;
    std::unordered_map<uint32_t, uint64_t> fallback_ops;
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
        for (void* allocation : device_allocations)
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

    void* acquire_managed(size_t bytes) {
        auto found = free_managed.lower_bound(bytes);
        if (found != free_managed.end()) {
            void* pointer = found->second;
            free_managed.erase(found);
            return pointer;
        }
        void* pointer = nullptr;
        if (!report_cuda(cudaMallocManaged(&pointer, bytes),
                         "cudaMallocManaged output"))
            return nullptr;
        managed_allocations.push_back(pointer);
        managed_sizes.emplace(pointer, bytes);
        return pointer;
    }

    void release_managed(void* pointer) {
        const auto found = managed_sizes.find(pointer);
        if (found != managed_sizes.end())
            free_managed.emplace(found->second, pointer);
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
                DeviceWeight{device, nullptr, type, n, k, 0, 0, layout})
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

    bool run_matmul_device(const float* device_a, int lda,
                           const Tensor& weight, float* device_c, int ldc,
                           int m, int n, int k,
                           Activation activation_kind, int act_begin,
                           int act_len) {
        const DeviceWeight* prepared = find_weight(weight);
        if (!prepared || prepared->n != n || prepared->k != k || !device_a ||
            !device_c || ldc != n ||
            prepared->layout != WeightLayout::Dense)
            return false;

        const size_t a_elements = static_cast<size_t>(m) * lda;
        const void* gemm_activation = device_a;
        cudaDataType activation_type = CUDA_R_32F;
        if (prepared->type == CUDA_R_16F) {
            if (!reserve(activation_fp16, activation_fp16_bytes,
                         a_elements * sizeof(__half)))
                return false;
            constexpr int threads = 256;
            fp32_to_fp16<<<
                static_cast<unsigned>((a_elements + threads - 1) / threads),
                threads>>>(device_a,
                           static_cast<__half*>(activation_fp16), a_elements);
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
                    prepared->data, prepared->type, k, gemm_activation,
                    activation_type, lda, &beta, device_c, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx"))
            return false;

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
        if (!host_a || !host_c ||
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

CudaTensorLayout cuda_layout(const Tensor& tensor) {
    CudaTensorLayout layout{};
    for (int dimension = 0; dimension < 4; ++dimension) {
        layout.shape[dimension] = tensor.shape[dimension];
        layout.stride[dimension] =
            tensor.stride[dimension] / sizeof(float);
    }
    return layout;
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
    }
}

void CudaBackend::wrap_weight_int4(Tensor& tensor,
                                   bool keep_native_experts) {
    if (!available() || tensor.shape[0] <= 0 ||
        tensor.shape[1] <= 0)
        return;
    const int n = static_cast<int>(tensor.shape[0]);
    const int k = static_cast<int>(tensor.shape[1]);
    if (keep_native_experts) {
        if (tensor.prec == Precision::INT8) {
            const void* source = tensor.rowmajor_data
                ? tensor.rowmajor_data : tensor.data;
            const int group_size = static_cast<int>(tensor.group_size);
            const int groups_per_row =
                static_cast<int>(tensor.groups_per_row);
            if (source && tensor.scales && group_size > 0 &&
                groups_per_row > 0) {
                impl_->upload_quantized_weight(
                    tensor, source, source,
                    static_cast<size_t>(n) * k, tensor.scales,
                    static_cast<size_t>(n) * groups_per_row, n, k,
                    group_size, groups_per_row,
                    Impl::WeightLayout::Q8RowMajor);
            }
            return;
        }
        if (tensor.prec != Precision::INT4 || n % 8 != 0)
            return;
        if (tensor.is_q4_g32_packed && tensor.q4_g32_data &&
            k % 32 == 0) {
            const size_t bytes = static_cast<size_t>(n / 8) * (k / 32) *
                sizeof(Q4B8G32Block);
            impl_->upload_weight(
                tensor, tensor.q4_g32_data, tensor.q4_g32_data, bytes,
                CUDA_R_8I, n, k, Impl::WeightLayout::Q4Bg32);
        } else if (tensor.is_q4_g128_packed && tensor.q4_g128_data &&
                   k % 128 == 0) {
            const size_t bytes = static_cast<size_t>(n / 8) * (k / 128) *
                sizeof(Q4B8G128Block);
            impl_->upload_weight(
                tensor, tensor.q4_g128_data, tensor.q4_g128_data, bytes,
                CUDA_R_8I, n, k, Impl::WeightLayout::Q4Bg128);
        }
        return;
    }
    std::vector<__half> dequantized(static_cast<size_t>(n) * k);

    if (tensor.prec == Precision::INT8) {
        const auto* quantized = static_cast<const int8_t*>(
            tensor.rowmajor_data ? tensor.rowmajor_data : tensor.data);
        if (!quantized || !tensor.scales || tensor.group_size == 0 ||
            tensor.groups_per_row == 0)
            return;
        const int group_size = static_cast<int>(tensor.group_size);
        const int groups_per_row =
            static_cast<int>(tensor.groups_per_row);
        for (int row = 0; row < n; ++row)
            for (int inner = 0; inner < k; ++inner) {
                const int group = std::min(
                    inner / group_size, groups_per_row - 1);
                const float scale = tensor.scales[
                    static_cast<size_t>(row) * groups_per_row + group];
                dequantized[static_cast<size_t>(row) * k + inner] =
                    __float2half(
                        static_cast<float>(quantized[
                            static_cast<size_t>(row) * k + inner]) * scale);
            }
        const void* cache_key = tensor.rowmajor_data
            ? tensor.rowmajor_data : tensor.data;
        impl_->upload_weight(
            tensor, cache_key, dequantized.data(),
            dequantized.size() * sizeof(__half), CUDA_R_16F, n, k);
        return;
    }
    if (tensor.prec != Precision::INT4)
        return;

    if (tensor.is_q4_g32_packed && tensor.q4_g32_data && k % 32 == 0) {
        const auto* blocks =
            static_cast<const Q4B8G32Block*>(tensor.q4_g32_data);
        const int groups = k / 32;
        for (int row = 0; row < n; ++row) {
            for (int group = 0; group < groups; ++group) {
                const auto& block =
                    blocks[static_cast<size_t>(row / 8) * groups + group];
                const int lane = row % 8;
                for (int inner = 0; inner < 32; ++inner) {
                    const uint8_t packed = block.q[lane][inner / 2];
                    const int value = signed_nibble(
                        inner & 1 ? packed >> 4 : packed);
                    dequantized[static_cast<size_t>(row) * k +
                                group * 32 + inner] =
                        __float2half(static_cast<float>(value) *
                                     block.scales[lane]);
                }
            }
        }
    } else if (tensor.is_q4_g128_packed && tensor.q4_g128_data &&
               k % 128 == 0) {
        const auto* blocks =
            static_cast<const Q4B8G128Block*>(tensor.q4_g128_data);
        const int groups = k / 128;
        for (int row = 0; row < n; ++row) {
            for (int group = 0; group < groups; ++group) {
                const auto& block =
                    blocks[static_cast<size_t>(row / 8) * groups + group];
                const int lane = row % 8;
                for (int inner = 0; inner < 128; ++inner) {
                    const int qgroup = inner / 32;
                    const int qinner = inner % 32;
                    const uint8_t packed =
                        block.q[qgroup][lane][qinner / 2];
                    const int value = signed_nibble(
                        qinner & 1 ? packed >> 4 : packed);
                    dequantized[static_cast<size_t>(row) * k +
                                group * 128 + inner] =
                        __float2half(static_cast<float>(value) *
                                     block.scales[lane]);
                }
            }
        }
    } else {
        return;
    }

    const void* cache_key = tensor.rowmajor_data
        ? tensor.rowmajor_data : tensor.data;
    impl_->upload_weight(tensor, cache_key, dequantized.data(),
                         dequantized.size() * sizeof(__half), CUDA_R_16F,
                         n, k);
}

void* CudaBackend::alloc_output(Tensor& output, size_t nbytes, BufferPool*) {
    if (!available() || nbytes == 0)
        return nullptr;
    void* pointer = impl_->acquire_managed(nbytes);
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
        impl_->release_managed(tensor.device_data);
}

void CudaBackend::synchronize_for_host_read() {
    if (!report_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"))
        impl_->failed = true;
}

void CudaBackend::begin_graph() {}

void CudaBackend::end_graph() { synchronize_for_host_read(); }

void CudaBackend::alloc_persistent(Tensor& tensor, size_t nbytes) {
    void* storage = nullptr;
    if (!available() || nbytes == 0 ||
        !report_cuda(cudaMallocManaged(&storage, nbytes),
                     "cudaMallocManaged persistent")) {
        impl_->failed = true;
        return;
    }
    impl_->managed_allocations.push_back(storage);
    std::memset(storage, 0, nbytes);
    tensor.data = storage;
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
        output && inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& source = *inputs[0];
        const float* input = device_pointer_const<float>(source);
        float* destination = device_pointer<float>(*output);
        bool valid = input && destination && source.nelements() > 0 &&
            output->nelements() > 0;
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
                input, destination, count, cuda_layout(source),
                cuda_layout(*output));
            if (!report_cuda(cudaGetLastError(), "tile_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::CONCAT && !inputs.empty() && output &&
        output->prec == Precision::FP32) {
        const int dimension = graph_params::get_i32(node.params, 0, 0);
        float* destination = device_pointer<float>(*output);
        int64_t concatenated = 0;
        bool valid = destination && dimension >= 0 && dimension < 4;
        for (const Tensor* source : inputs) {
            valid = valid && source && source->prec == Precision::FP32 &&
                device_pointer_const<float>(*source) &&
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
                concat_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                                    threads), threads>>>(
                    device_pointer_const<float>(*source), destination, count,
                    cuda_layout(*source), cuda_layout(*output), dimension,
                    offset);
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
        inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 && output->is_contiguous()) {
        const Tensor& source = *inputs[0];
        const float* input = device_pointer_const<float>(source);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (input && destination && source.nelements() == output->nelements()) {
            contiguous_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                                    threads), threads>>>(
                input, destination, count, source.shape[0], source.shape[1],
                source.shape[2], source.stride[0] / sizeof(float),
                source.stride[1] / sizeof(float),
                source.stride[2] / sizeof(float),
                source.stride[3] / sizeof(float));
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
        const float* key_data = device_pointer_const<float>(current_key);
        const float* value_data = device_pointer_const<float>(current_value);
        float* key_cache_data = nullptr;
        float* value_cache_data = nullptr;
        if (cache_mode == 2 && key_cache && value_cache &&
            key_cache->prec == Precision::FP32 &&
            value_cache->prec == Precision::FP32 &&
            key_cache->device_data && value_cache->device_data) {
            const auto* metadata = cache_meta(
                static_cast<const uint8_t*>(key_cache->data) +
                key_cache->device_offset);
            past_length = static_cast<int>(metadata->current_seq_len);
            key_capacity = static_cast<int>(metadata->max_seq_len);
            key_cache_data = reinterpret_cast<float*>(
                static_cast<uint8_t*>(key_cache->device_data) +
                key_cache->device_offset + CacheMetadata::SIZE);
            value_cache_data = reinterpret_cast<float*>(
                static_cast<uint8_t*>(value_cache->device_data) +
                value_cache->device_offset + CacheMetadata::SIZE);
            cached = true;
        }
        const int key_length = past_length + current_length;
        const float* query_data = device_pointer_const<float>(query);
        float* output_data = device_pointer<float>(*output);
        const float* mask_data = mask
            ? device_pointer_const<float>(*mask) : nullptr;
        const bool valid = query_data && key_data && value_data &&
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
                    key_data, value_data, key_cache_data, value_cache_data,
                    num_kv_heads, current_length, past_length, key_capacity,
                    key_dim, value_dim,
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
                key_capacity, cached, causal, scale,
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
        auto valid_expert_weight = [](const Impl::DeviceWeight* weight) {
            return weight &&
                (weight->layout != Impl::WeightLayout::Q8RowMajor ||
                 (weight->scales && weight->group_size > 0 &&
                  weight->groups_per_row > 0 &&
                  weight->group_size * weight->groups_per_row >= weight->k));
        };
        const Tensor* router_bias =
            router_bias_input >= 0 &&
                    static_cast<size_t>(router_bias_input) < inputs.size()
                ? inputs[router_bias_input]
                : nullptr;
        const bool hash_routing =
            token_ids_input >= 0 && hash_table_input >= 0;
        bool valid = fp32_contiguous(*inputs[0]) &&
            fp32_contiguous(*output) && router &&
            valid_expert_weight(gate_up) && valid_expert_weight(down) &&
            hidden_size > 0 && num_experts > 0 && num_experts >= top_k &&
            top_k > 0 && top_k <= 64 && intermediate_size > 0 &&
            num_groups <= 64 && tokens > 0 && score_function >= 0 &&
            score_function <= 2 && !hash_routing &&
            inputs[0]->shape[0] == hidden_size &&
            output->shape[0] == hidden_size && output->shape[1] == tokens &&
            router->n == num_experts && router->k == hidden_size &&
            router->layout == Impl::WeightLayout::Dense &&
            gate_up->n == num_experts * 2 * intermediate_size &&
            gate_up->k == hidden_size &&
            down->n == num_experts * hidden_size &&
            down->k == intermediate_size;
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
                valid = prepared_gate && prepared_up && prepared_down &&
                    prepared_gate->layout == Impl::WeightLayout::Dense &&
                    prepared_up->layout == Impl::WeightLayout::Dense &&
                    prepared_down->layout == Impl::WeightLayout::Dense &&
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
                        valid = prepared_scale &&
                            prepared_scale->layout ==
                                Impl::WeightLayout::Dense &&
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
            valid = logits && route_indices && route_weights &&
                routed_intermediate &&
                impl_->run_matmul_device(
                    hidden,
                    static_cast<int>(inputs[0]->stride[1] / sizeof(float)),
                    *inputs[1], logits, num_experts, tokens, num_experts,
                    hidden_size, Activation::NONE, 0, -1);
            if (valid) {
                moe_select_routes_cuda<<<tokens, 1>>>(
                    logits,
                    router_bias
                        ? device_pointer_const<float>(*router_bias) : nullptr,
                    route_indices, route_weights, tokens, num_experts, top_k,
                    score_function, normalize_topk, num_groups, topk_groups,
                    routed_scaling_factor);
                const size_t intermediate_count =
                    route_count * intermediate_size;
                moe_gate_up_cuda<<<
                    static_cast<unsigned>((intermediate_count + threads - 1) /
                                          threads), threads>>>(
                    hidden,
                    inputs[0]->stride[1] / sizeof(float), route_indices,
                    gate_up->data, gate_up->type == CUDA_R_32F,
                    static_cast<int>(gate_up->layout), gate_up->scales,
                    gate_up->group_size, gate_up->groups_per_row,
                    routed_intermediate, tokens, top_k, hidden_size,
                    intermediate_size, swiglu_limit);
                const size_t output_count =
                    static_cast<size_t>(tokens) * hidden_size;
                moe_down_cuda<<<
                    static_cast<unsigned>((output_count + threads - 1) /
                                          threads), threads>>>(
                    route_indices, route_weights, routed_intermediate,
                    down->data, down->type == CUDA_R_32F,
                    static_cast<int>(down->layout), down->scales,
                    down->group_size, down->groups_per_row, destination,
                    output->stride[1] / sizeof(float), tokens, top_k,
                    hidden_size, intermediate_size);
                valid = report_cuda(cudaGetLastError(), "CUDA MOE routed");
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
                        shared_intermediate, shared_count, swiglu_limit);
                    valid = report_cuda(
                        cudaGetLastError(), "CUDA MOE shared SwiGLU") &&
                        impl_->run_matmul_device(
                            shared_intermediate, shared_intermediate_size,
                            *shared_down, shared_output, hidden_size, tokens,
                            hidden_size, shared_intermediate_size,
                            Activation::NONE, 0, -1);
                }
                if (valid && shared_has_gate) {
                    valid = impl_->run_matmul_device(
                        hidden,
                        static_cast<int>(inputs[0]->stride[1] /
                                         sizeof(float)),
                        *shared_scale_weight, shared_scale, 1, tokens, 1,
                        hidden_size, Activation::NONE, 0, -1);
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

    synchronize_for_host_read();
    if (impl_->failed)
        return;
    std::vector<Tensor> host_tensors;
    std::vector<const Tensor*> host_inputs;
    host_tensors.reserve(inputs.size());
    host_inputs.reserve(inputs.size());
    for (const Tensor* input : inputs) {
        if (!input) {
            host_inputs.push_back(nullptr);
            continue;
        }
        host_tensors.push_back(*input);
        Tensor& host = host_tensors.back();
        if (host.data && host.device_offset)
            host.data = static_cast<uint8_t*>(host.data) +
                host.device_offset;
        host.device_data = nullptr;
        host.device_offset = 0;
        host_inputs.push_back(&host);
    }
    impl_->cpu.clear_dispatch_error();
    impl_->cpu.dispatch(node, host_inputs, output, thread_pool);
    if (impl_->cpu.dispatch_failed()) {
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
