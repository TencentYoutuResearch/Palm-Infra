#include "kernels/cuda/matmul.h"

#include <cstdio>

namespace {

__device__ int signed_nibble(uint8_t value) {
    const int nibble = value & 0x0f;
    return nibble >= 8 ? nibble - 16 : nibble;
}

__device__ int biased_q4_g32_nibble(uint8_t value) {
    return static_cast<int>(value & 0x0f) - 8;
}

struct alignas(8) Half2Pair {
    __half2 first;
    __half2 second;
};

constexpr int q4_g32_pairs_per_storage_block = 8 * 8;
constexpr int q4_g32_rows_per_storage_block = 8;
// Four rows per warp need enough inner work to amortize their longer lane
// loop and a wide output grid to retain enough resident blocks. Below either
// point the two-row kernel's extra row parallelism wins instead.
constexpr int q4_g32_quad_rows_target_blocks_per_sm = 6;
constexpr int q4_g32_quad_rows_fallback_min_columns = 16384;
constexpr int q4_g32_quad_rows_min_groups = 64;
// Caching avoids repeated L1 activation reads for profitable middle-width
// output grids. Shorter pair-row dots do not amortize the preload, while the
// very wide vocabulary grid is already bandwidth-bound and only adds traffic.
// Keep pair and quad limits independent: their row reuse and occupancy differ.
constexpr int q4_g32_pair_activation_cache_min_inner = 4096;
constexpr int q4_g32_pair_activation_cache_max_inner = 9216;
constexpr int q4_g32_pair_activation_cache_max_columns = 4096;
constexpr int q4_g32_quad_activation_cache_max_inner = 4096;
constexpr int q4_g32_quad_activation_cache_max_columns = 65536;
// Four canonical storage blocks exactly fill one 256-thread CUDA block.
constexpr int q4_g32_storage_blocks_per_cuda_block = 4;
constexpr int q4_g32_dequant_threads =
    q4_g32_pairs_per_storage_block *
    q4_g32_storage_blocks_per_cuda_block;

int cuda_multiprocessor_count() {
    static const int count = [] {
        int device = 0;
        int multiprocessors = 0;
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(
                &multiprocessors, cudaDevAttrMultiProcessorCount,
                device) == cudaSuccess)
            return multiprocessors;
        cudaGetLastError();
        return 0;
    }();
    return count;
}

__global__ void fp32_to_fp16(const float* source, __half* destination,
                             size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        destination[index] = __float2half(source[index]);
}

__global__ void fp32_to_fp16_vectorized(
    const float2* source, __half2* destination, size_t pair_count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < pair_count) {
        const float2 value = source[index];
        destination[index] = __floats2half2_rn(value.x, value.y);
    }
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

__global__ void bias_q4_g32_weight_cuda(
    Q4B8G32Block* weight, size_t byte_count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= byte_count)
        return;
    constexpr size_t bytes_per_block = 8 * 16;
    const size_t block = index / bytes_per_block;
    const size_t byte = index % bytes_per_block;
    reinterpret_cast<uint8_t*>(weight[block].q)[byte] ^= 0x88;
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

// W8PC is the common decode layout: every output row has one scale and the
// inner dimension is naturally vector aligned.  Assign four adjacent values
// to each lane so both the int8 weights and FP32 activation are fetched with
// vector loads.  The generic kernel above remains the single source of truth
// for grouped and ragged layouts.
__global__ void q8_per_channel_dense_gemv_cuda(
    const float* __restrict__ activation,
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    float* __restrict__ output, int columns, int inner) {
    constexpr int warps_per_block = 8;
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int column = static_cast<int>(blockIdx.x) * warps_per_block + warp;
    if (column >= columns)
        return;

    const auto* activation4 = reinterpret_cast<const float4*>(activation);
    const auto* row4 = reinterpret_cast<const char4*>(
        weight + static_cast<size_t>(column) * inner);
    const int vectors = inner / 4;
    const float scale = scales[column];
    float sum = 0.0f;
    for (int index = lane; index < vectors; index += warpSize) {
        const float4 a = activation4[index];
        const char4 w = row4[index];
        sum = fmaf(a.x, static_cast<float>(w.x) * scale, sum);
        sum = fmaf(a.y, static_cast<float>(w.y) * scale, sum);
        sum = fmaf(a.z, static_cast<float>(w.z) * scale, sum);
        sum = fmaf(a.w, static_cast<float>(w.w) * scale, sum);
    }
    for (int offset = warpSize / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[column] = sum;
}

template <bool VectorizedActivation>
__global__ void prepare_q4_g32_dense_gemm_cuda(
    const Q4B8G32Block* weight, __half2* weight_output,
    const float* activation, __half* activation_output,
    size_t activation_count, int rows, int groups) {
    // A canonical block has sixteen packed bytes per row. Convert two bytes
    // per thread so the expanded FP16 weight uses one naturally aligned
    // 64-bit store while sharing the scale and index arithmetic.
    const int local_block =
        static_cast<int>(threadIdx.x) /
            q4_g32_pairs_per_storage_block;
    const int packed_index =
        static_cast<int>(threadIdx.x) %
            q4_g32_pairs_per_storage_block;
    const int group = static_cast<int>(blockIdx.x) *
        q4_g32_storage_blocks_per_cuda_block + local_block;
    const int lane = packed_index / 8;
    const int pair = packed_index % 8;
    const int row_block = static_cast<int>(blockIdx.y);
    const int row = row_block * q4_g32_rows_per_storage_block + lane;
    if (group < groups && row < rows) {
        const size_t block_index =
            static_cast<size_t>(row_block) * groups + group;
        const auto& block = weight[block_index];
        const uint8_t first = block.q[lane][pair * 2];
        const uint8_t second = block.q[lane][pair * 2 + 1];
        const float scale = block.scales[lane];
        reinterpret_cast<Half2Pair*>(weight_output)[
            (static_cast<size_t>(row) * groups + group) * 8 + pair] = {
                __halves2half2(
                    __float2half(biased_q4_g32_nibble(first) * scale),
                    __float2half(
                        biased_q4_g32_nibble(first >> 4) * scale)),
                __halves2half2(
                    __float2half(biased_q4_g32_nibble(second) * scale),
                    __float2half(
                        biased_q4_g32_nibble(second >> 4) * scale))};
    }

    const size_t block = static_cast<size_t>(blockIdx.y) * gridDim.x +
        blockIdx.x;
    const size_t thread = block * blockDim.x + threadIdx.x;
    const size_t thread_count = static_cast<size_t>(gridDim.x) * gridDim.y *
        blockDim.x;
    if constexpr (VectorizedActivation) {
        const size_t pair_count = activation_count / 2;
        const auto* source = reinterpret_cast<const float2*>(activation);
        auto* destination = reinterpret_cast<__half2*>(activation_output);
        for (size_t index = thread; index < pair_count;
             index += thread_count) {
            const float2 value = source[index];
            destination[index] = __floats2half2_rn(value.x, value.y);
        }
    } else {
        for (size_t index = thread; index < activation_count;
             index += thread_count)
            activation_output[index] = __float2half(activation[index]);
    }
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
        __float2half(signed_nibble(packed) * scale),
        __float2half(signed_nibble(packed >> 4) * scale));
}

__global__ void q4_g32_dense_gemv_cuda(
    const float* __restrict__ activation,
    const Q4B8G32Block* __restrict__ weight,
    float* __restrict__ output, int columns, int inner) {
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
        const auto& block =
            weight[static_cast<size_t>(column / 8) * groups + group];
        // Every lane requests the same address, so the uniform load is
        // broadcast by the GPU without an explicit shuffle.
        const float scale = block.scales[row_lane];
        const uint8_t packed = block.q[row_lane][lane / 2];
        const int k = group * 32 + lane;
        sum += activation[k] *
            biased_q4_g32_nibble(lane & 1 ? packed >> 4 : packed) * scale;
    }
    for (int offset = warpSize / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[column] = sum;
}

template <bool CacheActivation>
__global__ void q4_g32_pair_rows_gemv_cuda(
    const float* __restrict__ activation,
    const Q4B8G32Block* __restrict__ weight,
    float* __restrict__ output, int columns, int inner) {
    // Four warps cover one canonical eight-row storage tile. Each half-warp
    // owns one row and consumes both nibbles of a packed byte, halving the
    // number of warps and reductions without serializing a row below sixteen
    // lanes.
    constexpr int lanes_per_row = 16;
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int row = warp * 2 + lane / lanes_per_row;
    const int pair = lane % lanes_per_row;
    const int column = static_cast<int>(blockIdx.x) *
        q4_g32_rows_per_storage_block + row;
    const int groups = inner / 32;
    extern __shared__ float activation_cache[];
    if constexpr (CacheActivation) {
        for (int index = static_cast<int>(threadIdx.x); index < inner;
             index += static_cast<int>(blockDim.x))
            activation_cache[index] = activation[index];
        __syncthreads();
        activation = activation_cache;
    }
    float sum = 0.0f;
    for (int group = 0; group < groups; ++group) {
        const auto& block =
            weight[static_cast<size_t>(blockIdx.x) * groups + group];
        const uint8_t packed = block.q[row][pair];
        const float2 value = reinterpret_cast<const float2*>(
            activation + static_cast<size_t>(group) * 32)[pair];
        const float scale = block.scales[row];
        // Both nibbles share one group scale. Form their two-element dot
        // first so the scale is applied once instead of multiplying both
        // weights separately.
        const float quantized_dot = fmaf(
            value.x,
            static_cast<float>(biased_q4_g32_nibble(packed)),
            value.y * static_cast<float>(
                biased_q4_g32_nibble(packed >> 4)));
        sum = fmaf(quantized_dot, scale, sum);
    }
    for (int offset = lanes_per_row / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(
            0xffffffffu, sum, offset, lanes_per_row);
    if (lane % lanes_per_row == 0 && column < columns)
        output[column] = sum;
}

template <bool CacheActivation>
__global__ void q4_g32_quad_rows_gemv_cuda(
    const float* __restrict__ activation,
    const Q4B8G32Block* __restrict__ weight,
    float* __restrict__ output, int columns, int inner) {
    constexpr int lanes_per_row = 8;
    constexpr int rows_per_warp = 4;
    constexpr int storage_blocks_per_cuda_block = 2;
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int row = warp * rows_per_warp + lane / lanes_per_row;
    const int pair = lane % lanes_per_row;
    const int storage_block = static_cast<int>(blockIdx.x) *
        storage_blocks_per_cuda_block +
        row / q4_g32_rows_per_storage_block;
    const int block_row = row % q4_g32_rows_per_storage_block;
    const int column = static_cast<int>(blockIdx.x) *
        storage_blocks_per_cuda_block * q4_g32_rows_per_storage_block + row;
    extern __shared__ float activation_cache[];
    if constexpr (CacheActivation) {
        for (int index = static_cast<int>(threadIdx.x); index < inner;
             index += static_cast<int>(blockDim.x))
            activation_cache[index] = activation[index];
        __syncthreads();
        activation = activation_cache;
    }
    if (storage_block * q4_g32_rows_per_storage_block >= columns)
        return;
    const int groups = inner / 32;
    float sum = 0.0f;
    for (int group = 0; group < groups; ++group) {
        const auto& block =
            weight[static_cast<size_t>(storage_block) * groups + group];
        const uint8_t first = block.q[block_row][pair];
        const uint8_t second = block.q[block_row][pair + lanes_per_row];
        const auto* group_activation = reinterpret_cast<const float2*>(
            activation + static_cast<size_t>(group) * 32);
        const float2 first_value = group_activation[pair];
        const float2 second_value = group_activation[pair + lanes_per_row];
        const float scale = block.scales[block_row];
        float quantized_dot = fmaf(
            first_value.x,
            static_cast<float>(biased_q4_g32_nibble(first)),
            first_value.y * static_cast<float>(
                biased_q4_g32_nibble(first >> 4)));
        quantized_dot = fmaf(
            second_value.x,
            static_cast<float>(biased_q4_g32_nibble(second)),
            quantized_dot);
        quantized_dot = fmaf(
            second_value.y,
            static_cast<float>(biased_q4_g32_nibble(second >> 4)),
            quantized_dot);
        sum = fmaf(quantized_dot, scale, sum);
    }
    for (int offset = lanes_per_row / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(
            0xffffffffu, sum, offset, lanes_per_row);
    if (lane % lanes_per_row == 0 && column < columns)
        output[column] = sum;
}

__global__ void q4_g128_dense_gemv_cuda(
    const float* __restrict__ activation,
    const Q4B8G128Block* __restrict__ weight,
    float* __restrict__ output, int columns, int inner) {
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
        const auto& block =
            weight[static_cast<size_t>(column / 8) * groups + group];
        // Every lane requests the same address, so the uniform load is
        // broadcast by the GPU without an explicit shuffle.
        const float scale = block.scales[row_lane];
        for (int subgroup = 0; subgroup < 4; ++subgroup) {
            const uint8_t packed =
                block.q[subgroup][row_lane][lane / 2];
            const int k = group * 128 + subgroup * 32 + lane;
            sum += activation[k] *
                signed_nibble(lane & 1 ? packed >> 4 : packed) * scale;
        }
    }
    for (int offset = warpSize / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[column] = sum;
}

}  // namespace

namespace mollm_cuda {

void launch_fp32_to_fp16(const float* source, __half* destination,
                         size_t count) {
    constexpr int threads = 256;
    const bool vector_aligned = count % 2 == 0 &&
        reinterpret_cast<uintptr_t>(source) % alignof(float2) == 0 &&
        reinterpret_cast<uintptr_t>(destination) % alignof(__half2) == 0;
    if (vector_aligned) {
        const size_t pair_count = count / 2;
        fp32_to_fp16_vectorized<<<
            static_cast<unsigned>((pair_count + threads - 1) / threads),
            threads>>>(
                reinterpret_cast<const float2*>(source),
                reinterpret_cast<__half2*>(destination), pair_count);
    } else {
        fp32_to_fp16<<<
            static_cast<unsigned>((count + threads - 1) / threads),
            threads>>>(source, destination, count);
    }
}

void launch_dequantize_q8_dense_weight(
    const int8_t* weight, const float* scales, int group_size,
    int groups_per_row, __half* output, size_t count, int width) {
    constexpr int threads = 256;
    dequantize_q8_dense_weight_cuda<<<
        static_cast<unsigned>((count + threads - 1) / threads), threads>>>(
        weight, scales, group_size, groups_per_row, output, count, width);
}

void launch_bias_q4_g32_weight(
    Q4B8G32Block* weight, size_t block_count) {
    constexpr size_t bytes_per_block = 8 * 16;
    constexpr int threads = 256;
    const size_t byte_count = block_count * bytes_per_block;
    bias_q4_g32_weight_cuda<<<
        static_cast<unsigned>((byte_count + threads - 1) / threads),
        threads>>>(weight, byte_count);
}

void launch_q8_dense_gemv(
    const float* activation, const int8_t* weight, const float* scales,
    int group_size, int groups_per_row, float* output, int columns,
    int inner) {
    const bool vector_aligned =
        reinterpret_cast<uintptr_t>(activation) % alignof(float4) == 0 &&
        reinterpret_cast<uintptr_t>(weight) % alignof(char4) == 0;
    if (groups_per_row == 1 && group_size >= inner && inner % 4 == 0 &&
        vector_aligned) {
        // The vector path has enough independent row work to amortize a
        // larger block; keep the ragged/grouped fallback at its lower
        // register-pressure configuration below.
        constexpr int warps_per_block = 8;
        constexpr int threads = warps_per_block * 32;
        const unsigned blocks = static_cast<unsigned>(
            (columns + warps_per_block - 1) / warps_per_block);
        q8_per_channel_dense_gemv_cuda<<<blocks, threads>>>(
            activation, weight, scales, output, columns, inner);
    } else {
        constexpr int warps_per_block = 4;
        constexpr int threads = warps_per_block * 32;
        const unsigned blocks = static_cast<unsigned>(
            (columns + warps_per_block - 1) / warps_per_block);
        q8_dense_gemv_cuda<<<blocks, threads>>>(
            activation, weight, scales, group_size, groups_per_row, output,
            columns, inner);
    }
}

void launch_prepare_q4_g32_dense_gemm(
    const Q4B8G32Block* weight, __half2* weight_output,
    const float* activation, __half* activation_output,
    size_t activation_count, int rows, int groups) {
    const dim3 grid(
        (groups + q4_g32_storage_blocks_per_cuda_block - 1) /
            q4_g32_storage_blocks_per_cuda_block,
        (rows + q4_g32_rows_per_storage_block - 1) /
            q4_g32_rows_per_storage_block);
    const bool vector_aligned = activation_count % 2 == 0 &&
        reinterpret_cast<uintptr_t>(activation) % alignof(float2) == 0 &&
        reinterpret_cast<uintptr_t>(activation_output) %
            alignof(__half2) == 0;
    if (vector_aligned) {
        prepare_q4_g32_dense_gemm_cuda<true><<<
            grid, q4_g32_dequant_threads>>>(
            weight, weight_output, activation, activation_output,
            activation_count, rows, groups);
    } else {
        prepare_q4_g32_dense_gemm_cuda<false><<<
            grid, q4_g32_dequant_threads>>>(
            weight, weight_output, activation, activation_output,
            activation_count, rows, groups);
    }
}

void launch_dequantize_q4_g128_dense_weight(
    const Q4B8G128Block* weight, __half2* output, size_t packed_count,
    int rows, int groups) {
    constexpr int threads = 256;
    dequantize_q4_g128_dense_weight_cuda<<<
        static_cast<unsigned>((packed_count + threads - 1) / threads),
        threads>>>(weight, output, packed_count, rows, groups);
}

void launch_q4_g32_dense_gemv(
    const float* activation, const Q4B8G32Block* weight, float* output,
    int columns, int inner) {
    constexpr int warps_per_block = 4;
    constexpr int threads = warps_per_block * 32;
    if (reinterpret_cast<uintptr_t>(activation) % alignof(float2) == 0) {
        constexpr int quad_storage_blocks_per_cuda_block = 2;
        const unsigned quad_blocks = static_cast<unsigned>(
            (columns + quad_storage_blocks_per_cuda_block *
                           q4_g32_rows_per_storage_block - 1) /
            (quad_storage_blocks_per_cuda_block *
             q4_g32_rows_per_storage_block));
        const int multiprocessors = cuda_multiprocessor_count();
        const bool enough_grid = multiprocessors > 0
            ? quad_blocks >= static_cast<unsigned>(
                  multiprocessors *
                  q4_g32_quad_rows_target_blocks_per_sm)
            : columns >= q4_g32_quad_rows_fallback_min_columns;
        const bool use_quad_rows =
            inner / 32 >= q4_g32_quad_rows_min_groups && enough_grid;
        const bool cache_pair_activation =
            inner >= q4_g32_pair_activation_cache_min_inner &&
            inner <= q4_g32_pair_activation_cache_max_inner &&
            columns <= q4_g32_pair_activation_cache_max_columns;
        const bool cache_quad_activation =
            inner <= q4_g32_quad_activation_cache_max_inner &&
            columns < q4_g32_quad_activation_cache_max_columns;
        if (use_quad_rows) {
            if (cache_quad_activation)
                q4_g32_quad_rows_gemv_cuda<true><<<
                    quad_blocks, threads,
                    static_cast<size_t>(inner) * sizeof(float)>>>(
                        activation, weight, output, columns, inner);
            else
                q4_g32_quad_rows_gemv_cuda<false><<<quad_blocks, threads>>>(
                    activation, weight, output, columns, inner);
        } else {
            const unsigned blocks = static_cast<unsigned>(
                (columns + q4_g32_rows_per_storage_block - 1) /
                q4_g32_rows_per_storage_block);
            if (cache_pair_activation)
                q4_g32_pair_rows_gemv_cuda<true><<<
                    blocks, threads,
                    static_cast<size_t>(inner) * sizeof(float)>>>(
                        activation, weight, output, columns, inner);
            else
                q4_g32_pair_rows_gemv_cuda<false><<<blocks, threads>>>(
                    activation, weight, output, columns, inner);
        }
    } else {
        const unsigned blocks = static_cast<unsigned>(
            (columns + warps_per_block - 1) / warps_per_block);
        q4_g32_dense_gemv_cuda<<<blocks, threads>>>(
            activation, weight, output, columns, inner);
    }
}

void launch_q4_g128_dense_gemv(
    const float* activation, const Q4B8G128Block* weight, float* output,
    int columns, int inner) {
    constexpr int warps_per_block = 4;
    constexpr int threads = warps_per_block * 32;
    const unsigned blocks = static_cast<unsigned>(
        (columns + warps_per_block - 1) / warps_per_block);
    q4_g128_dense_gemv_cuda<<<blocks, threads>>>(
        activation, weight, output, columns, inner);
}

bool run_dense_matmul(
    cublasHandle_t cublas, const void* weight, cudaDataType weight_type,
    const void* activation, cudaDataType activation_type, float* output,
    int m, int n, int k, int lda) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C[M,N] = A[M,K] * W[N,K]^T is the equivalent
    // column-major operation C_col[N,M] = W_col[K,N]^T * A_col[K,M].
    const cublasStatus_t status = cublasGemmEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, weight,
        weight_type, k, activation, activation_type, lda, &beta, output,
        CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_SUCCESS)
        return true;
    std::fprintf(stderr,
                 "CudaBackend: cublasGemmEx failed (cuBLAS status %d)\n",
                 static_cast<int>(status));
    return false;
}

}  // namespace mollm_cuda
