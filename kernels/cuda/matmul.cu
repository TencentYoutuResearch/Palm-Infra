#include "kernels/cuda/matmul.h"
#include "kernels/cuda/reduction.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <algorithm>

namespace {

__device__ int signed_nibble(uint8_t value) {
    const int nibble = value & 0x0f;
    return nibble >= 8 ? nibble - 16 : nibble;
}

__device__ float biased_q4_g32_nibble(uint8_t value) {
    // At this exponent one mantissa unit is exactly 1.0f, so this
    // constructs all sixteen values [-8,7] without integer-to-float conversion.
    return __uint_as_float(0x4b400000u | (value & 0x0fu)) - 12582920.0f;
}

struct alignas(8) Half2Pair {
    __half2 first;
    __half2 second;
};

constexpr int q4_g32_pairs_per_storage_block = 8 * 8;
constexpr int q4_g32_rows_per_storage_block = 8;
// Eight rows per warp need enough inner work to amortize their longer lane
// loop and a wide output grid to retain enough resident blocks. Below either
// point the two-row kernel's extra row parallelism wins instead.
constexpr int q4_g32_quad_rows_target_blocks_per_sm = 6;
constexpr int q4_g32_quad_rows_fallback_min_columns = 16384;
constexpr int q4_g32_quad_rows_min_groups = 64;
// Caching avoids repeated L1 activation reads for profitable middle-width
// output grids. Shorter pair-row dots do not amortize the preload, while the
// very wide vocabulary grid is already bandwidth-bound and only adds traffic.
// Keep pair and wide-row limits independent: their row reuse and occupancy differ.
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

bool cuda_q4_prefetch_enabled() {
    static const int override_mode = [] {
        const char* option = std::getenv("MOLLM_CUDA_Q4_PREFETCH");
        if (option && std::strcmp(option, "1") == 0) return 1;
        if (option && std::strcmp(option, "0") == 0) return 0;
        return -1;
    }();
    if (override_mode >= 0) return override_mode != 0;
    // Only the SM121 K2560 path has full-model performance evidence.
    // Cache by current device, not by the first GPU seen by the process.
    thread_local int cached_device = -1;
    thread_local bool supported = false;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (device != cached_device) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        supported = major == 12 && minor == 1;
        cached_device = device;
    }
    return supported;
}

__global__ void fp16_to_fp32_reference(const __half* source, float* destination, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        destination[i] = __half2float(source[i]);
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

__global__ void swizzle_q4_g32_weight_cuda(Q4B8G32Block* weight, size_t rows) {
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    auto* packet = reinterpret_cast<uint4*>(weight[row / 8].q[row % 8]);
    const uint4 original = *packet;
    // Lane p gets canonical half p in the low word and half p+4 in
    // the high word. No additional weight payload or scales are needed.
    *packet = make_uint4((original.x & 0xffffu) | (original.z << 16),
        (original.x >> 16) | (original.z & 0xffff0000u),
        (original.y & 0xffffu) | (original.w << 16),
        (original.y >> 16) | (original.w & 0xffff0000u));
}

// W8PC is the common decode layout: every output row has one scale and the
// inner dimension is naturally vector aligned.  Assign four adjacent values
// to each lane so both the int8 weights and FP32 activation are fetched with
// vector loads.  The generic kernel above remains the single source of truth
// for grouped and ragged layouts.
template<int PrefetchDistance = 0, int Warps = 8, bool CacheActivation = false, int Pipeline = 0>
__global__ void q8_per_channel_dense_gemv_cuda(
    const float* __restrict__ activation,
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    float* __restrict__ output, int columns, int inner) {
    constexpr int warps_per_block = Warps;
    __shared__ float4 cached_activation[CacheActivation ? 640 : 1];
    if constexpr (CacheActivation) {
        for (int i = threadIdx.x; i < inner / 4; i += blockDim.x)
            cached_activation[i] = reinterpret_cast<const float4*>(activation)[i];
        __syncthreads();
    }
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
    float4 pending_a{};
    char4 pending_w{};
    if constexpr (Pipeline) if (lane < vectors) {
        if constexpr (Pipeline == 2) pending_a = activation4[lane];
        pending_w = row4[lane];
    }
    for (int index = lane; index < vectors; index += warpSize) {
        float4 incoming_a{};
        char4 incoming_w{};
        if constexpr (Pipeline) if (index + warpSize < vectors) {
            if constexpr (Pipeline == 2) incoming_a = activation4[index + warpSize];
            incoming_w = row4[index + warpSize];
        }
        if constexpr (PrefetchDistance > 0) {
            if (index + warpSize * PrefetchDistance < vectors)
                asm volatile("prefetch.global.L2 [%0];" ::
                    "l"(row4 + index + warpSize * PrefetchDistance));
        }
        const float4 a = Pipeline == 2 ? pending_a : CacheActivation ? cached_activation[index] : activation4[index];
        const char4 w = Pipeline ? pending_w : row4[index];
        sum = fmaf(a.x, static_cast<float>(w.x) * scale, sum);
        sum = fmaf(a.y, static_cast<float>(w.y) * scale, sum);
        sum = fmaf(a.z, static_cast<float>(w.z) * scale, sum);
        sum = fmaf(a.w, static_cast<float>(w.w) * scale, sum);
        if constexpr (Pipeline) {
            if constexpr (Pipeline == 2) pending_a = incoming_a;
            pending_w = incoming_w;
        }
    }
    for (int offset = warpSize / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[column] = sum;
}

// Diagnostic exact short-batch W8PC path: reuse each char4 across tokens,
// preserving the serial GEMV's scaled-weight FMA order and warp reduction.
template<int Batch>
__global__ void q8_small_batch_exact_cuda(const float* activation,
    const int8_t* weight, const float* scales, float* output,
    int tokens, int columns, int inner, int lda, int ldc) {
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.x * 8 + threadIdx.x / 32;
    if (column >= columns) return;
    const auto* row = reinterpret_cast<const char4*>(weight + static_cast<size_t>(column)*inner);
    const float scale = scales[column];
    float sums[Batch] = {};
    for (int index=lane; index<inner/4; index+=32) {
        const char4 w = row[index];
        const float wx=static_cast<float>(w.x)*scale, wy=static_cast<float>(w.y)*scale;
        const float wz=static_cast<float>(w.z)*scale, ww=static_cast<float>(w.w)*scale;
#pragma unroll
        for(int token=0;token<Batch;++token) {
            if(token>=tokens)continue;
            const float4 a=reinterpret_cast<const float4*>(activation+static_cast<size_t>(token)*lda)[index];
            sums[token]=fmaf(a.x,wx,sums[token]);
            sums[token]=fmaf(a.y,wy,sums[token]);
            sums[token]=fmaf(a.z,wz,sums[token]);
            sums[token]=fmaf(a.w,ww,sums[token]);
        }
    }
#pragma unroll
    for(int token=0;token<Batch;++token) {
        if(token>=tokens)continue;
        float sum=sums[token];
        for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
        if(!lane)output[static_cast<size_t>(token)*ldc+column]=sum;
    }
}

template<bool Swizzled>
__device__ uint8_t q4_g32_byte(const uint8_t* row, int byte) {
    if constexpr (Swizzled) byte = ((byte & 6) << 1) | ((byte & 8) >> 2) | (byte & 1);
    return row[byte];
}

template <bool VectorizedActivation, class Block = Q4B8G32Block>
__global__ void prepare_q4_g32_dense_gemm_cuda(
    const Block* weight, __half2* weight_output,
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
        float scale;
        if constexpr (std::is_same<Block, mollm_cuda::Q4CompactG32>::value)
            scale = __fdiv_rn(__uint_as_float(static_cast<uint32_t>(block.scales[lane]) << 16), 7.f);
        else
            scale = block.scales[lane];
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

template<bool Swizzled = false>
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
        const uint8_t packed = q4_g32_byte<Swizzled>(block.q[row_lane], lane / 2);
        const int k = group * 32 + lane;
        sum += activation[k] *
            biased_q4_g32_nibble(lane & 1 ? packed >> 4 : packed) * scale;
    }
    for (int offset = warpSize / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[column] = sum;
}

__global__ void repack_q4_row_g32_cuda(const Q4B8G32Block* source,
    mollm_cuda::Q4RowG32* target, int columns, int groups) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(columns) * groups) return;
    const int row = index / groups;
    const int group = index % groups;
    const auto& block = source[static_cast<size_t>(row / 8) * groups + group];
    target[index].scale = block.scales[row % 8];
#pragma unroll
    for (int i = 0; i < 16; ++i) target[index].q[i] = block.q[row % 8][i];
}

__global__ void q4_row_g32_gemv_cuda(const float* activation,
    const mollm_cuda::Q4RowG32* weight, float* output, int inner) {
    constexpr int threads = 256;
    const int row = blockIdx.x;
    const int groups = inner / 32;
    float sum = 0;
    for (int pair = threadIdx.x; pair < inner / 2; pair += threads) {
        const auto& block = weight[static_cast<size_t>(row) * groups + pair / 16];
        const uint8_t packed = block.q[pair % 16];
        const float2 x = reinterpret_cast<const float2*>(activation)[pair];
        const float dot = fmaf(x.x, biased_q4_g32_nibble(packed),
            x.y * biased_q4_g32_nibble(packed >> 4));
        sum = fmaf(dot, block.scale, sum);
    }
    __shared__ float reduction[threads];
    const float total = mollm_cuda::detail::block_reduce_sum<threads>(sum, reduction);
    if (threadIdx.x == 0) output[row] = total;
}

template<int Batch, bool ParallelReduction = false>
__global__ void q4_g32_small_batch_cuda(const float* activation,
    const Q4B8G32Block* weight, float* output, int tokens, int inner, int lda, int ldc) {
    constexpr int threads = 256;
    const int row = blockIdx.x;
    const int groups = inner / 32;
    float sums[Batch] = {};
    for (int pair = threadIdx.x; pair < inner / 2; pair += threads) {
        const auto& block = weight[static_cast<size_t>(row / 8) * groups + pair / 16];
        const uint8_t packed = block.q[row % 8][pair % 16];
        const float low = biased_q4_g32_nibble(packed);
        const float high = biased_q4_g32_nibble(packed >> 4);
        const float scale = block.scales[row % 8];
#pragma unroll
        for (int token = 0; token < Batch; ++token) {
            if (token >= tokens) continue;
            const float* x = activation + static_cast<size_t>(token) * lda + 2 * pair;
            const float dot = fmaf(x[0], low, x[1] * high);
            sums[token] = fmaf(dot, scale, sums[token]);
        }
    }
    __shared__ float reduction[ParallelReduction ? Batch * threads : threads];
    if constexpr (ParallelReduction) {
#pragma unroll
        for (int token = 0; token < Batch; ++token)
            reduction[token * threads + threadIdx.x] = sums[token];
        __syncthreads();
        if (threadIdx.x < 32) {
#pragma unroll
            for (int token = 0; token < Batch; ++token) {
                if (token >= tokens) continue;
                const float* r = reduction + token * threads + threadIdx.x;
                // Same 128,64,32,16,... tree as block_reduce_sum<256>,
                // with all tokens stored before the single barrier.
                const float first = (r[0] + r[128]) + (r[64] + r[192]);
                const float second = (r[32] + r[160]) + (r[96] + r[224]);
                const float total = mollm_cuda::detail::warp_reduce_sum(first + second);
                if (threadIdx.x == 0) output[static_cast<size_t>(token) * ldc + row] = total;
            }
        }
    } else {
#pragma unroll
        for (int token = 0; token < Batch; ++token) {
            if (token >= tokens) continue;
            const float total = mollm_cuda::detail::block_reduce_sum<threads>(sums[token], reduction);
            if (threadIdx.x == 0) output[static_cast<size_t>(token) * ldc + row] = total;
        }
    }
}

template<int Batch, class Block = Q4B8G32Block>
__global__ void q4_g32_small_batch_wide_cuda(const float* activation,
    const Block* weight, float* output, int tokens, int columns, int inner, int lda, int ldc) {
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.x * 32 + threadIdx.x / 4;
    const int storage_block = column / 8;
    if (storage_block * 8 >= columns) return;
    const int row = column % 8, pair = lane % 4, groups = inner / 32;
    float sums[Batch] = {}, seconds[Batch] = {};
    for (int group = 0; group < groups; ++group) {
        const auto& block = weight[static_cast<size_t>(storage_block) * groups + group];
        const uint32_t packed = reinterpret_cast<const uint32_t*>(block.q[row])[pair];
        const uint32_t first = __shfl_sync(0xffffffffu, packed, pair / 2, 4) >> ((pair & 1) * 16);
        const uint32_t second = __shfl_sync(0xffffffffu, packed, pair / 2 + 2, 4) >> ((pair & 1) * 16);
        float scale;
        if constexpr (std::is_same<Block, mollm_cuda::Q4CompactG32>::value)
            scale = __fdiv_rn(__uint_as_float(static_cast<unsigned>(block.scales[row]) << 16), 7.f);
        else scale = block.scales[row];
#pragma unroll
        for (int token = 0; token < Batch; ++token) {
            if (token >= tokens) continue;
            const auto* x = reinterpret_cast<const float4*>(activation + static_cast<size_t>(token) * lda + group * 32);
            const float4 a = x[pair], b = x[pair + 4];
            float dot = fmaf(a.x, biased_q4_g32_nibble(first), a.y * biased_q4_g32_nibble(first >> 4));
            dot = fmaf(a.z, biased_q4_g32_nibble(first >> 8), dot);
            dot = fmaf(a.w, biased_q4_g32_nibble(first >> 12), dot);
            sums[token] = fmaf(dot, scale, sums[token]);
            dot = fmaf(b.x, biased_q4_g32_nibble(second), b.y * biased_q4_g32_nibble(second >> 4));
            dot = fmaf(b.z, biased_q4_g32_nibble(second >> 8), dot);
            dot = fmaf(b.w, biased_q4_g32_nibble(second >> 12), dot);
            seconds[token] = fmaf(dot, scale, seconds[token]);
        }
    }
#pragma unroll
    for (int token = 0; token < Batch; ++token) {
        if (token >= tokens) continue;
        float sum = sums[token] + seconds[token];
        sum += __shfl_down_sync(0xffffffffu, sum, 2, 4);
        sum += __shfl_down_sync(0xffffffffu, sum, 1, 4);
        if (pair == 0 && column < columns) output[static_cast<size_t>(token) * ldc + column] = sum;
    }
}

template <int Threads, bool TerminalReduction = false, int PrefetchDistance = 0,
          class Block = Q4B8G32Block, bool SingleBarrier = false>
__device__ __forceinline__ void q4_g32_row_compute(
    const float* __restrict__ activation, const Block* __restrict__ weight,
    float* __restrict__ output, int inner, int row) {
    const int groups = inner / 32;
    float sum = 0.0f;
    for (int pair = threadIdx.x; pair < inner / 2; pair += Threads) {
        if constexpr (PrefetchDistance > 0) {
            if (pair + Threads * PrefetchDistance < inner / 2) {
                const auto& next = weight[static_cast<size_t>(row / 8) * groups +
                    (pair + Threads * PrefetchDistance) / 16];
                asm volatile("prefetch.global.L2 [%0];" :: "l"(&next.q[row % 8][pair % 16]));
                asm volatile("prefetch.global.L2 [%0];" :: "l"(&next.scales[row % 8]));
            }
        }
        const auto& block = weight[static_cast<size_t>(row / 8) * groups + pair / 16];
        const uint8_t packed = block.q[row % 8][pair % 16];
        const float2 x = reinterpret_cast<const float2*>(activation)[pair];
        const float dot = fmaf(x.x, static_cast<float>(biased_q4_g32_nibble(packed)),
            x.y * static_cast<float>(biased_q4_g32_nibble(packed >> 4)));
        float scale;
        if constexpr (std::is_same<Block, mollm_cuda::Q4CompactG32>::value) {
            const unsigned code = block.scales[row % 8];
            const unsigned exponent = (code >> 7) & 255u;
            const unsigned mantissa = code & 127u;
            if (exponent <= 3u || exponent == 255u)
                scale = __fdiv_rn(__uint_as_float(code << 16), 7.f);
            else {
                const unsigned numerator = (128u + mantissa) << (mantissa >= 96u ? 18 : 19);
                scale = __uint_as_float(((code & 0x8000u) << 16) |
                    ((exponent - (mantissa >= 96u ? 2u : 3u)) << 23) |
                    (((numerator + 3u) / 7u) & 0x7fffffu));
            }
        } else scale = block.scales[row % 8];
        sum = fmaf(dot, scale, sum);
    }
    __shared__ float reduction[Threads];
    if constexpr (SingleBarrier) {
        static_assert(Threads == 256);
        reduction[threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.x < 32) {
            const float* r = reduction + threadIdx.x;
            const float first = (r[0] + r[128]) + (r[64] + r[192]);
            const float second = (r[32] + r[160]) + (r[96] + r[224]);
            const float total = mollm_cuda::detail::warp_reduce_sum(first + second);
            if (threadIdx.x == 0) output[row] = total;
        }
        return;
    }
    if constexpr (TerminalReduction) {
        reduction[threadIdx.x] = sum;
        __syncthreads();
#pragma unroll
        for (int offset = Threads / 2; offset >= 64; offset >>= 1) {
            if (threadIdx.x < offset)
                reduction[threadIdx.x] += reduction[threadIdx.x + offset];
            __syncthreads();
        }
        if (threadIdx.x < 32) {
            float total = reduction[threadIdx.x] + reduction[threadIdx.x + 32];
            total = mollm_cuda::detail::warp_reduce_sum(total);
            if (threadIdx.x == 0) output[row] = total;
        }
        return;
    }
    const float total = mollm_cuda::detail::block_reduce_sum<Threads>(sum, reduction);
    if (threadIdx.x == 0) output[row] = total;
}

template <int Threads, bool TerminalReduction = false, int PrefetchDistance = 0,
          class Block = Q4B8G32Block, bool SingleBarrier = false>
__global__ void q4_g32_row_parallel_gemv_cuda(
    const float* __restrict__ activation, const Block* __restrict__ weight,
    float* __restrict__ output, int inner) {
    q4_g32_row_compute<Threads, TerminalReduction, PrefetchDistance, Block, SingleBarrier>(activation,
        weight, output, inner, static_cast<int>(blockIdx.x));
}

__global__ void q4_g32_paired_projections_cuda(const float* activation,
    const Q4B8G32Block* first_weight, const Q4B8G32Block* second_weight,
    float* first_output, float* second_output, int inner) {
    q4_g32_row_compute<256, false, 0>(activation,
        blockIdx.y ? second_weight : first_weight,
        blockIdx.y ? second_output : first_output, inner, static_cast<int>(blockIdx.x));
}

__device__ __forceinline__ float q4_shared_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__device__ __forceinline__ float q4_shared_reduce_256(const float* values) {
    const int t = threadIdx.x;
    // Exactly the original stride128 -> stride64 -> stride32 tree.
    float first = values[t] + values[t + 128];
    first += values[t + 64] + values[t + 192];
    float second = values[t + 32] + values[t + 160];
    second += values[t + 96] + values[t + 224];
    return mollm_cuda::detail::warp_reduce_sum(first + second);
}

__global__ void q4_g32_shared_activation_cuda(const float* __restrict__ activation,
    const Q4B8G32Block* __restrict__ gate_weight,
    const Q4B8G32Block* __restrict__ up_weight, float* __restrict__ output, int inner) {
    const int row = blockIdx.x, groups = inner / 32;
    float gate = 0.f, up = 0.f;
    for (int pair = threadIdx.x; pair < inner / 2; pair += 256) {
        const size_t index = static_cast<size_t>(row / 8) * groups + pair / 16;
        const auto& g = gate_weight[index];
        const auto& u = up_weight[index];
        const uint8_t gp = g.q[row % 8][pair % 16], upack = u.q[row % 8][pair % 16];
        const float2 x = reinterpret_cast<const float2*>(activation)[pair];
        const float gd = fmaf(x.x, static_cast<float>(biased_q4_g32_nibble(gp)),
            x.y * static_cast<float>(biased_q4_g32_nibble(gp >> 4)));
        const float ud = fmaf(x.x, static_cast<float>(biased_q4_g32_nibble(upack)),
            x.y * static_cast<float>(biased_q4_g32_nibble(upack >> 4)));
        gate = fmaf(gd, g.scales[row % 8], gate);
        up = fmaf(ud, u.scales[row % 8], up);
    }
    __shared__ float partial[2][256];
    partial[0][threadIdx.x] = gate;
    partial[1][threadIdx.x] = up;
    __syncthreads();
    if (threadIdx.x < 32) {
        gate = q4_shared_bf16(q4_shared_reduce_256(partial[0]));
        up = q4_shared_bf16(q4_shared_reduce_256(partial[1]));
        if (threadIdx.x == 0)
            output[row] = q4_shared_bf16((gate * (1.f / (1.f + expf(-gate)))) * up);
    }
}

// Map the original 256 logical lanes onto 128 physical threads. Keep the
// logical lanes' independent FMA chains, then perform the original first
// reduction (stride128) before the unchanged remaining reduction tree.
__global__ void q4_g32_dual_lane_gemv_cuda(const float* __restrict__ activation,
    const Q4B8G32Block* __restrict__ weight, float* __restrict__ output, int inner) {
    const int row = blockIdx.x;
    const int groups = inner / 32;
    float sums[2] = {};
    for (int pair = threadIdx.x; pair < inner / 2; pair += 256) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int logical_pair = pair + half * 128;
            if (logical_pair < inner / 2) {
                const auto& block = weight[static_cast<size_t>(row / 8) * groups + logical_pair / 16];
                const uint8_t packed = block.q[row % 8][logical_pair % 16];
                const float2 x = reinterpret_cast<const float2*>(activation)[logical_pair];
                const float dot = fmaf(x.x, static_cast<float>(biased_q4_g32_nibble(packed)),
                    x.y * static_cast<float>(biased_q4_g32_nibble(packed >> 4)));
                sums[half] = fmaf(dot, block.scales[row % 8], sums[half]);
            }
        }
    }
    __shared__ float reduction[128];
    const float total = mollm_cuda::detail::block_reduce_sum<128>(sums[0] + sums[1], reduction);
    if (threadIdx.x == 0) output[row] = total;
}

struct Q4NarrowPacket {
    unsigned codes;
    float scale;
    float2 input;
};

template<bool PipelineInputs, class Block = Q4B8G32Block>
__device__ __forceinline__ Q4NarrowPacket q4_narrow_packet(const float* activation,
    const Block* weight, int row, int inner, int pair) {
    if (pair >= inner / 2) return {};
    const auto& block = weight[static_cast<size_t>(row / 8) * (inner / 32) + pair / 16];
    Q4NarrowPacket result{block.q[row % 8][pair % 16], 0.f, {}};
    if constexpr (std::is_same<Block,mollm_cuda::Q4CompactG32>::value) {
        const unsigned code=block.scales[row%8], exponent=(code>>7)&255u, mantissa=code&127u;
        if(exponent<=3u || exponent==255u)
            result.scale=__fdiv_rn(__uint_as_float(code<<16),7.f);
        else {
            const unsigned numerator=(128u+mantissa)<<(mantissa>=96u?18:19);
            result.scale=__uint_as_float(((code&0x8000u)<<16)|
                ((exponent-(mantissa>=96u?2u:3u))<<23)|(((numerator+3u)/7u)&0x7fffffu));
        }
    } else result.scale=block.scales[row%8];
    if constexpr (PipelineInputs) result.input = reinterpret_cast<const float2*>(activation)[pair];
    return result;
}

__global__ void q4_g32_dual_lane_pipeline_cuda(const float* activation,
    const Q4B8G32Block* weight,float* output,int inner) {
    const int row=blockIdx.x;
    Q4NarrowPacket pending[2];
    float sums[2]={};
#pragma unroll
    for(int half=0;half<2;++half)
        pending[half]=q4_narrow_packet<true>(activation,weight,row,inner,threadIdx.x+128*half);
    for(int pair=threadIdx.x;pair<inner/2;pair+=256) {
        Q4NarrowPacket incoming[2];
#pragma unroll
        for(int half=0;half<2;++half)
            incoming[half]=q4_narrow_packet<true>(activation,weight,row,inner,pair+128*half+256);
#pragma unroll
        for(int half=0;half<2;++half) {
            if(pair+128*half<inner/2) {
                const auto packet=pending[half];
                const float dot=fmaf(packet.input.x,biased_q4_g32_nibble(packet.codes),
                    packet.input.y*biased_q4_g32_nibble(packet.codes>>4));
                sums[half]=fmaf(dot,packet.scale,sums[half]);
            }
            pending[half]=incoming[half];
        }
    }
    __shared__ float reduction[128];
    const float total=mollm_cuda::detail::block_reduce_sum<128>(sums[0]+sums[1],reduction);
    if(!threadIdx.x)output[row]=total;
}

template<int Depth, bool PipelineInputs, class Block = Q4B8G32Block, bool SingleBarrier = false>
__device__ __forceinline__ void q4_g32_pipelined_narrow_compute(const float* activation,
    const Block* weight, float* output, int inner, int row) {
    Q4NarrowPacket pending[Depth];
#pragma unroll
    for (int i = 0; i < Depth; ++i)
        pending[i] = q4_narrow_packet<PipelineInputs>(activation, weight, row, inner, threadIdx.x + 256 * i);
    float sum = 0.f;
    for (int pair = threadIdx.x; pair < inner / 2; pair += 256) {
        const auto incoming = q4_narrow_packet<PipelineInputs>(activation, weight, row, inner, pair + 256 * Depth);
        const auto current = pending[0];
        const float2 x = PipelineInputs ? current.input : reinterpret_cast<const float2*>(activation)[pair];
        const float dot = fmaf(x.x, static_cast<float>(biased_q4_g32_nibble(current.codes)),
            x.y * static_cast<float>(biased_q4_g32_nibble(current.codes >> 4)));
        sum = fmaf(dot, current.scale, sum);
#pragma unroll
        for (int i = 0; i < Depth - 1; ++i) pending[i] = pending[i + 1];
        pending[Depth - 1] = incoming;
    }
    __shared__ float reduction[256];
    if constexpr (SingleBarrier) {
        reduction[threadIdx.x]=sum;
        __syncthreads();
        if(threadIdx.x<32) {
            const float total=q4_shared_reduce_256(reduction);
            if(!threadIdx.x)output[row]=total;
        }
        return;
    }
    const float total = mollm_cuda::detail::block_reduce_sum<256>(sum, reduction);
    if (!threadIdx.x) output[row] = total;
}

template<int Depth, bool PipelineInputs, class Block = Q4B8G32Block, bool SingleBarrier = false>
__global__ void q4_g32_pipelined_narrow_cuda(const float* __restrict__ activation,
    const Block* __restrict__ weight, float* __restrict__ output, int inner) {
    q4_g32_pipelined_narrow_compute<Depth, PipelineInputs, Block, SingleBarrier>(activation, weight, output, inner, blockIdx.x);
}

struct CompactPacket { uint8_t codes; uint16_t scale; float2 input; };
__device__ __forceinline__ CompactPacket compact_packet(const float* x,
    const mollm_cuda::Q4CompactG32* weight, int row, int inner, int pair) {
    if (pair >= inner / 2) return {};
    const auto& block = weight[size_t(row / 8) * (inner / 32) + pair / 16];
    return {block.q[row % 8][pair % 16], block.scales[row % 8],
            reinterpret_cast<const float2*>(x)[pair]};
}
__global__ void compact_warp_table_cuda(const float* x,
    const mollm_cuda::Q4CompactG32* weight, float* output, int inner) {
    const int row = blockIdx.x, lane = threadIdx.x % 32;
    unsigned table[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned m = lane + i * 32;
        table[i] = ((((128u + m) << (m >= 96 ? 18 : 19)) + 3u) / 7u) & 0x7fffffu;
    }
    float sum = 0;
    auto pending = compact_packet(x, weight, row, inner, threadIdx.x);
    // API requires inner%512==0, so all lanes enter every shuffle together.
    for (int pair = threadIdx.x; pair < inner / 2; pair += 256) {
        const auto incoming = compact_packet(x, weight, row, inner, pair + 256);
        const unsigned code = pending.scale, m = code & 127, e = (code >> 7) & 255;
        unsigned values[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            values[i] = __shfl_sync(0xffffffffu, table[i], m & 31);
        const unsigned fraction = m < 32 ? values[0] : m < 64 ? values[1] : m < 96 ? values[2] : values[3];
        const float scale = e <= 3 || e == 255 ? __fdiv_rn(__uint_as_float(code << 16), 7.f)
            : __uint_as_float(((code & 0x8000u) << 16) | ((e - (m >= 96 ? 2 : 3)) << 23) | fraction);
        const float dot = fmaf(pending.input.x, float(biased_q4_g32_nibble(pending.codes)),
            pending.input.y * float(biased_q4_g32_nibble(pending.codes >> 4)));
        sum = fmaf(dot, scale, sum);
        pending = incoming;
    }
    __shared__ float reduction[256];
    const float total = mollm_cuda::detail::block_reduce_sum<256>(sum, reduction);
    if (!threadIdx.x) output[row] = total;
}

__global__ void q4_g32_unequal_projections_cuda(const float* activation,
    const Q4B8G32Block* first_weight, const Q4B8G32Block* second_weight,
    float* first_output, float* second_output, int first_columns, int inner) {
    const int row = blockIdx.x;
    // CTA-uniform selection: independent weights/outputs, original reduction.
    const bool first = row < first_columns;
    q4_g32_pipelined_narrow_compute<1, true>(activation,
        first ? first_weight : second_weight, first ? first_output : second_output,
        inner, first ? row : row - first_columns);
}

template<int Rows>
__global__ void q4_g32_spatial_narrow_cuda(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner) {
    constexpr int physical = 256 / Rows;
    constexpr int stride = physical + 32 / Rows;
    const int local_row = threadIdx.x % Rows;
    const int logical_lane = threadIdx.x / Rows;
    const int row = blockIdx.x * Rows + local_row;
    const int groups = inner / 32;
    float sums[Rows] = {};
    for (int base = logical_lane; base < inner / 2; base += 256) {
#pragma unroll
        for (int chain = 0; chain < Rows; ++chain) {
            const int pair = base + chain * physical;
            if (row < columns && pair < inner / 2) {
                const auto& block = weight[static_cast<size_t>(row / 8) * groups + pair / 16];
                const uint8_t packed = block.q[row % 8][pair % 16];
                const float2 x = reinterpret_cast<const float2*>(activation)[pair];
                const float dot = fmaf(x.x, static_cast<float>(biased_q4_g32_nibble(packed)),
                    x.y * static_cast<float>(biased_q4_g32_nibble(packed >> 4)));
                sums[chain] = fmaf(dot, block.scales[row % 8], sums[chain]);
            }
        }
    }
    // Fold logical stride128/64/32 stages without changing their tree.
#pragma unroll
    for (int offset = Rows / 2; offset; offset /= 2) {
#pragma unroll
        for (int chain = 0; chain < offset; ++chain)
            sums[chain] += sums[chain + offset];
    }
    __shared__ float partial[Rows][stride];
    partial[local_row][logical_lane] = sums[0];
    __syncthreads();
    const int output_row = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (output_row < Rows) {
        float total = partial[output_row][lane];
        if constexpr (physical == 128) {
            total += partial[output_row][lane + 64];
            total += partial[output_row][lane + 32] + partial[output_row][lane + 96];
        } else if constexpr (physical == 64) {
            total += partial[output_row][lane + 32];
        }
        total = mollm_cuda::detail::warp_reduce_sum(total);
        if (!lane && blockIdx.x * Rows + output_row < columns)
            output[blockIdx.x * Rows + output_row] = total;
    }
}

template<int Rows>
__global__ void q4_g32_grouped_narrow_cuda(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner) {
    constexpr int threads = 256;
    const int first_row = blockIdx.x * Rows;
    const int groups = inner / 32;
    float sums[Rows] = {};
    for (int pair = threadIdx.x; pair < inner / 2; pair += threads) {
        const auto& block = weight[static_cast<size_t>(first_row / 8) * groups + pair / 16];
        const float2 x = reinterpret_cast<const float2*>(activation)[pair];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const int lane = first_row % 8 + r;
            const uint8_t packed = block.q[lane][pair % 16];
            const float dot = fmaf(x.x, static_cast<float>(biased_q4_g32_nibble(packed)),
                x.y * static_cast<float>(biased_q4_g32_nibble(packed >> 4)));
            sums[r] = fmaf(dot, block.scales[lane], sums[r]);
        }
    }
    __shared__ float partial[Rows][threads];
#pragma unroll
    for (int r = 0; r < Rows; ++r) partial[r][threadIdx.x] = sums[r];
    __syncthreads();
#pragma unroll
    for (int offset = threads / 2; offset >= 64; offset >>= 1) {
        if (threadIdx.x < offset) {
#pragma unroll
            for (int r = 0; r < Rows; ++r)
                partial[r][threadIdx.x] += partial[r][threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const float value = mollm_cuda::detail::warp_reduce_sum(
                partial[r][threadIdx.x] + partial[r][threadIdx.x + 32]);
            if (threadIdx.x == 0 && first_row + r < columns) output[first_row + r] = value;
        }
    }
}

template<int Splits>
__global__ void q4_g32_split_pair_gemv_cuda(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner) {
    const int split = threadIdx.x / 128;
    const int row = (threadIdx.x % 128) / 16;
    const int pair = threadIdx.x % 16;
    const int groups = inner / 32;
    float sum = 0;
    for (int group = split; group < groups; group += Splits) {
        const auto& block = weight[static_cast<size_t>(blockIdx.x) * groups + group];
        const uint8_t packed = block.q[row][pair];
        const float2 x = reinterpret_cast<const float2*>(activation + group * 32)[pair];
        const float dot = fmaf(x.x, biased_q4_g32_nibble(packed),
            x.y * biased_q4_g32_nibble(packed >> 4));
        sum = fmaf(dot, block.scales[row], sum);
    }
    for (int offset = 8; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset, 16);
    __shared__ float partial[Splits][8];
    if (pair == 0) partial[split][row] = sum;
    __syncthreads();
    if (threadIdx.x < 8 && blockIdx.x * 8 + threadIdx.x < columns) {
        float total = partial[0][threadIdx.x];
#pragma unroll
        for (int i = 1; i < Splits; ++i) total += partial[i][threadIdx.x];
        output[blockIdx.x * 8 + threadIdx.x] = total;
    }
}

template <bool CacheActivation, bool Swizzled = false, int Pipeline = 0, class Block = Q4B8G32Block, bool PredecodeScale = false>
__global__ void q4_g32_pair_rows_gemv_cuda(
    const float* __restrict__ activation,
    const Block* __restrict__ weight,
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
    if constexpr (PredecodeScale) {
        static_assert(!CacheActivation && std::is_same<Block,mollm_cuda::Q4CompactG32>::value);
        for(int index=threadIdx.x;index<groups*8;index+=blockDim.x) {
            const auto& block=weight[static_cast<size_t>(blockIdx.x)*groups+index/8];
            activation_cache[index]=__fdiv_rn(__uint_as_float(static_cast<uint32_t>(block.scales[index%8])<<16),7.f);
        }
        __syncthreads();
    }
    if constexpr (CacheActivation) {
        for (int index = static_cast<int>(threadIdx.x); index < inner;
             index += static_cast<int>(blockDim.x))
            activation_cache[index] = activation[index];
        __syncthreads();
        activation = activation_cache;
    }
    struct Packet { uint8_t packed; float scale; float2 input; };
    const auto load = [&](int group) {
        Packet packet{};
        if (group >= groups || column >= columns) return packet;
        const auto& block = weight[static_cast<size_t>(blockIdx.x) * groups + group];
        packet.packed = q4_g32_byte<Swizzled>(block.q[row], pair);
        if constexpr (PredecodeScale) packet.scale=activation_cache[group*8+row];
        else if constexpr (std::is_same<Block,mollm_cuda::Q4CompactG32>::value)
            packet.scale = q4_narrow_packet<false>(activation,weight,column,inner,group*16+pair).scale;
        else packet.scale = block.scales[row];
        if constexpr (Pipeline == 2)
            packet.input = reinterpret_cast<const float2*>(activation + static_cast<size_t>(group) * 32)[pair];
        return packet;
    };
    Packet pending{};
    if constexpr (Pipeline) pending = load(0);
    float sum = 0.0f;
    for (int group = 0; group < groups; ++group) {
        Packet incoming{};
        if constexpr (Pipeline) incoming = load(group + 1);
        const auto& block =
            weight[static_cast<size_t>(blockIdx.x) * groups + group];
        const uint8_t packed = Pipeline ? pending.packed : q4_g32_byte<Swizzled>(block.q[row], pair);
        const float2 value = Pipeline == 2 ? pending.input : reinterpret_cast<const float2*>(
            activation + static_cast<size_t>(group) * 32)[pair];
        float scale;
        if constexpr (PredecodeScale) scale=activation_cache[group*8+row];
        else if constexpr (std::is_same<Block,mollm_cuda::Q4CompactG32>::value)
            scale = Pipeline ? pending.scale : q4_narrow_packet<false>(activation,weight,column,inner,group*16+pair).scale;
        else scale = Pipeline ? pending.scale : block.scales[row];
        // Both nibbles share one group scale. Form their two-element dot
        // first so the scale is applied once instead of multiplying both
        // weights separately.
        const float quantized_dot = fmaf(
            value.x,
            static_cast<float>(biased_q4_g32_nibble(packed)),
            value.y * static_cast<float>(
                biased_q4_g32_nibble(packed >> 4)));
        sum = fmaf(quantized_dot, scale, sum);
        if constexpr (Pipeline) pending = incoming;
    }
    for (int offset = lanes_per_row / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(
            0xffffffffu, sum, offset, lanes_per_row);
    if (lane % lanes_per_row == 0 && column < columns)
        output[column] = sum;
}

template<int Hint> __device__ uint32_t q4_load_packet(const void* pointer) {
    uint32_t value;
    if constexpr (Hint == -1)
        value = __ldcg(static_cast<const uint32_t*>(pointer));
    else if constexpr (Hint == -2)
        value = __ldcs(static_cast<const uint32_t*>(pointer));
    else if constexpr (Hint == 64)
        asm("ld.global.nc.L2::64B.u32 %0, [%1];" : "=r"(value) : "l"(pointer));
    else if constexpr (Hint == 128)
        asm("ld.global.nc.L2::128B.u32 %0, [%1];" : "=r"(value) : "l"(pointer));
    else if constexpr (Hint == 256) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm("ld.global.nc.L2::256B.u32 %0, [%1];" : "=r"(value) : "l"(pointer));
#else
        asm("ld.global.nc.L2::128B.u32 %0, [%1];" : "=r"(value) : "l"(pointer));
#endif
    }
    else
        asm("ld.global.nc.u32 %0, [%1];" : "=r"(value) : "l"(pointer));
    return value;
}

__device__ void q4_wait_staged_weights() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

__constant__ uint32_t q4_constant_scale_mantissas[128] = {
1198373u,1273271u,1348169u,1423067u,1497966u,1572864u,1647762u,1722661u,1797559u,1872457u,1947355u,2022254u,2097152u,2172050u,2246949u,2321847u,2396745u,2471643u,2546542u,2621440u,2696338u,2771237u,2846135u,2921033u,2995931u,3070830u,3145728u,3220626u,3295525u,3370423u,3445321u,3520219u,3595118u,3670016u,3744914u,3819813u,3894711u,3969609u,4044507u,4119406u,4194304u,4269202u,4344101u,4418999u,4493897u,4568795u,4643694u,4718592u,4793490u,4868389u,4943287u,5018185u,5093083u,5167982u,5242880u,5317778u,5392677u,5467575u,5542473u,5617371u,5692270u,5767168u,5842066u,5916965u,5991863u,6066761u,6141659u,6216558u,6291456u,6366354u,6441253u,6516151u,6591049u,6665947u,6740846u,6815744u,6890642u,6965541u,7040439u,7115337u,7190235u,7265134u,7340032u,7414930u,7489829u,7564727u,7639625u,7714523u,7789422u,7864320u,7939218u,8014117u,8089015u,8163913u,8238811u,8313710u,0u,37449u,74898u,112347u,149797u,187246u,224695u,262144u,299593u,337042u,374491u,411941u,449390u,486839u,524288u,561737u,599186u,636635u,674085u,711534u,748983u,786432u,823881u,861330u,898779u,936229u,973678u,1011127u,1048576u,1086025u,1123474u,1160923u
};

__device__ __align__(128) const uint32_t q4_global_scale_mantissas[128] = {
1198373u,1273271u,1348169u,1423067u,1497966u,1572864u,1647762u,1722661u,1797559u,1872457u,1947355u,2022254u,2097152u,2172050u,2246949u,2321847u,2396745u,2471643u,2546542u,2621440u,2696338u,2771237u,2846135u,2921033u,2995931u,3070830u,3145728u,3220626u,3295525u,3370423u,3445321u,3520219u,3595118u,3670016u,3744914u,3819813u,3894711u,3969609u,4044507u,4119406u,4194304u,4269202u,4344101u,4418999u,4493897u,4568795u,4643694u,4718592u,4793490u,4868389u,4943287u,5018185u,5093083u,5167982u,5242880u,5317778u,5392677u,5467575u,5542473u,5617371u,5692270u,5767168u,5842066u,5916965u,5991863u,6066761u,6141659u,6216558u,6291456u,6366354u,6441253u,6516151u,6591049u,6665947u,6740846u,6815744u,6890642u,6965541u,7040439u,7115337u,7190235u,7265134u,7340032u,7414930u,7489829u,7564727u,7639625u,7714523u,7789422u,7864320u,7939218u,8014117u,8089015u,8163913u,8238811u,8313710u,0u,37449u,74898u,112347u,149797u,187246u,224695u,262144u,299593u,337042u,374491u,411941u,449390u,486839u,524288u,561737u,599186u,636635u,674085u,711534u,748983u,786432u,823881u,861330u,898779u,936229u,973678u,1011127u,1048576u,1086025u,1123474u,1160923u
};

template <bool CacheActivation, bool Swizzled = false, int PrefetchDistance = 0,
          class Block = Q4B8G32Block, int LoadHint = 0, bool AsyncWeights = false,
          bool PrefetchScales = false, bool RegisterLookahead = false, bool PrefetchL1 = false,
          bool DirectCompactScale = false, bool CombineInitBarrier = false,
          int Warps = 4, bool GlobalCompactTable = false, bool ConstantCompactTable = false>
__global__ void q4_g32_oct_rows_gemv_cuda(
    const float* __restrict__ activation,
    const Block* __restrict__ weight,
    float* __restrict__ output, int columns, int inner) {
    constexpr int lanes_per_row = 4;
    constexpr int rows_per_warp = 8;
    constexpr int storage_blocks_per_cuda_block = Warps;
    static_assert(!AsyncWeights || Warps == 4);
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
    constexpr bool compact = std::is_same<Block, mollm_cuda::Q4CompactG32>::value;
    __shared__ uint32_t scale_mantissas[compact && !DirectCompactScale && !GlobalCompactTable ? 128 : 1];
    __shared__ __align__(16) Q4B8G32Block staged_weights[AsyncWeights ? 2 : 1][AsyncWeights ? 4 : 1][AsyncWeights ? 4 : 1];
    if constexpr (compact && !DirectCompactScale && !GlobalCompactTable) {
        for (unsigned table_index = threadIdx.x; table_index < 128; table_index += blockDim.x) {
            // Exact rounded significand of (1 + m/128)/7. Division by
            // seven cannot tie, so integer rounding needs no FP division.
            const unsigned m = table_index;
            const unsigned numerator = (128u + m) << (m >= 96 ? 18 : 19);
            scale_mantissas[m] = ((numerator + 3u) / 7u) & 0x7fffffu;
        }
        if constexpr (!CombineInitBarrier || !CacheActivation) __syncthreads();
    }
    if constexpr (CacheActivation) {
        for (int index = static_cast<int>(threadIdx.x); index < inner;
             index += static_cast<int>(blockDim.x))
            activation_cache[index] = activation[index];
        __syncthreads();
        activation = activation_cache;
    }
    if (!AsyncWeights && storage_block * q4_g32_rows_per_storage_block >= columns)
        return;
    const int groups = inner / 32;
    const auto stage = [&](int buffer, int first_group) {
        for (int ordinal = threadIdx.x; ordinal < 160; ordinal += blockDim.x) {
            const int local_block = ordinal / 40;
            const int group = first_group + (ordinal % 40) / 10;
            const int packet = ordinal % 10;
            const int source_block = blockIdx.x * 4 + local_block;
            const bool valid = source_block * 8 < columns && group < groups;
            const auto* source = reinterpret_cast<const uint8_t*>(weight) +
                (valid ? (static_cast<size_t>(source_block) * groups + group) * sizeof(Block) + packet * 16 : 0);
            auto* destination = reinterpret_cast<uint8_t*>(&staged_weights[buffer][local_block][(ordinal % 40) / 10]) + packet * 16;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
            const int bytes = valid ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                "r"(address), "l"(source), "r"(bytes) : "memory");
#else
            *reinterpret_cast<uint4*>(destination) = valid
                ? *reinterpret_cast<const uint4*>(source) : make_uint4(0, 0, 0, 0);
#endif
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;" ::: "memory");
#endif
    };
    if constexpr (AsyncWeights) {
        static_assert(!compact && !Swizzled && !LoadHint && !PrefetchDistance);
        stage(0, 0);
        q4_wait_staged_weights();
        __syncthreads();
    }
    float sum = 0.0f;
    float second_sum = 0.0f;
    const auto load_lookahead = [&](int group) {
        if (group >= groups) return make_uint2(0, 0);
        const auto& next = weight[static_cast<size_t>(storage_block) * groups + group];
        uint32_t scale_bits;
        if constexpr (compact) scale_bits = next.scales[block_row];
        else scale_bits = __float_as_uint(next.scales[block_row]);
        return make_uint2(reinterpret_cast<const uint32_t*>(next.q[block_row])[pair], scale_bits);
    };
    uint2 pending{};
    if constexpr (RegisterLookahead) {
        static_assert(!Swizzled && !AsyncWeights && !LoadHint);
        pending = load_lookahead(0);
    }
    for (int group = 0; group < groups; ++group) {
        uint2 incoming{};
        if constexpr (RegisterLookahead) incoming = load_lookahead(group + 1);
        if constexpr (AsyncWeights) {
            if (group % 4 == 0 && group + 4 < groups)
                stage((group / 4 + 1) % 2, group + 4);
        }
        if constexpr (PrefetchDistance > 0) {
            if (group + PrefetchDistance < groups) {
                const auto* next = &weight[static_cast<size_t>(storage_block) * groups + group + PrefetchDistance];
                if constexpr (PrefetchL1)
                    asm volatile("prefetch.global.L1 [%0];" :: "l"(&next->q[block_row][pair * 4]));
                else asm volatile("prefetch.global.L2 [%0];" :: "l"(&next->q[block_row][pair * 4]));
                if constexpr (PrefetchScales)
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(&next->scales[block_row]));
            }
        }
        const Block* block_pointer;
        if constexpr (AsyncWeights) block_pointer = &staged_weights[(group / 4) % 2][warp][group % 4];
        else block_pointer = &weight[static_cast<size_t>(storage_block) * groups + group];
        const auto& block = *block_pointer;
        uint32_t packed;
        if constexpr (RegisterLookahead) packed = pending.x;
        else packed = LoadHint
            ? q4_load_packet<LoadHint>(&reinterpret_cast<const uint32_t*>(block.q[block_row])[pair])
            : reinterpret_cast<const uint32_t*>(block.q[block_row])[pair];
        float scale;
        if constexpr (RegisterLookahead && !compact) scale = __uint_as_float(pending.y);
        else if constexpr (compact) {
            const uint16_t code = RegisterLookahead ? pending.y : block.scales[block_row];
            const uint32_t mantissa = code & 127u, exponent = (code >> 7) & 255u;
            uint32_t fractional;
            if constexpr (DirectCompactScale) {
                const unsigned numerator = (128u + mantissa) << (mantissa >= 96 ? 18 : 19);
                fractional = ((numerator + 3u) / 7u) & 0x7fffffu;
            } else if constexpr (ConstantCompactTable) fractional = q4_constant_scale_mantissas[mantissa];
            else if constexpr (GlobalCompactTable) fractional = __ldg(q4_global_scale_mantissas+mantissa);
            else fractional = scale_mantissas[mantissa];
            // Subnormal maxima and zero use the exact generic decoder.
            scale = exponent <= 3 || exponent == 255
                ? __fdiv_rn(__uint_as_float(static_cast<uint32_t>(code) << 16), 7.f)
                : __uint_as_float((static_cast<uint32_t>(code & 0x8000u) << 16) |
                    ((exponent - (mantissa >= 96 ? 2 : 3)) << 23) | fractional);
        } else {
            if constexpr (LoadHint) scale = __uint_as_float(q4_load_packet<LoadHint>(&block.scales[block_row]));
            else scale = block.scales[block_row];
        }
        // Redistribute the coalesced 32-bit packets into the original eight
        // lane dot products. Keep two accumulators per lane so the old FP32
        // reduction order is preserved despite using half as many lanes.
        uint16_t first_half, second_half;
        if constexpr (Swizzled) {
            first_half = packed & 0xffffu;
            second_half = packed >> 16;
        } else {
            const uint32_t first_packet = __shfl_sync(0xffffffffu, packed, pair / 2, lanes_per_row);
            const uint32_t second_packet = __shfl_sync(0xffffffffu, packed, pair / 2 + 2, lanes_per_row);
            first_half = first_packet >> ((pair & 1) * 16);
            second_half = second_packet >> ((pair & 1) * 16);
        }
        const uint8_t first = first_half & 255;
        const uint8_t second = first_half >> 8;
        const auto* group_activation = reinterpret_cast<const float4*>(
            activation + static_cast<size_t>(group) * 32);
        const float4 value = group_activation[pair];
        float quantized_dot = fmaf(
            value.x,
            static_cast<float>(biased_q4_g32_nibble(first)),
            value.y * static_cast<float>(
                biased_q4_g32_nibble(first >> 4)));
        quantized_dot = fmaf(
            value.z,
            static_cast<float>(biased_q4_g32_nibble(second)),
            quantized_dot);
        quantized_dot = fmaf(
            value.w,
            static_cast<float>(biased_q4_g32_nibble(second >> 4)),
            quantized_dot);
        sum = fmaf(quantized_dot, scale, sum);
        const float4 more = group_activation[pair + 4];
        const uint8_t third = second_half & 255;
        const uint8_t fourth = second_half >> 8;
        float second_dot = fmaf(more.x, static_cast<float>(biased_q4_g32_nibble(third)),
            more.y * static_cast<float>(biased_q4_g32_nibble(third >> 4)));
        second_dot = fmaf(more.z, static_cast<float>(biased_q4_g32_nibble(fourth)), second_dot);
        second_dot = fmaf(more.w, static_cast<float>(biased_q4_g32_nibble(fourth >> 4)), second_dot);
        second_sum = fmaf(second_dot, scale, second_sum);
        if constexpr (RegisterLookahead) pending = incoming;
        if constexpr (AsyncWeights) {
            if (group % 4 == 3 || group + 1 == groups) {
                q4_wait_staged_weights();
                __syncthreads();
            }
        }
    }
    sum += second_sum;
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

void launch_swizzle_q4_g32_weight(Q4B8G32Block* weight, size_t block_count) {
    constexpr int threads = 256;
    const size_t rows = block_count * 8;
    swizzle_q4_g32_weight_cuda<<<(rows + threads - 1) / threads, threads>>>(weight, rows);
}

void launch_q4_g32_swizzled_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, cudaStream_t stream) {
    if (reinterpret_cast<uintptr_t>(activation) % alignof(float4)) {
        if (reinterpret_cast<uintptr_t>(activation) % alignof(float2) == 0)
            q4_g32_pair_rows_gemv_cuda<false, true><<<(columns + 7) / 8, 128, 0, stream>>>(
                activation, weight, output, columns, inner);
        else
            q4_g32_dense_gemv_cuda<true><<<(columns + 3) / 4, 128, 0, stream>>>(
                activation, weight, output, columns, inner);
        return;
    }
    const unsigned blocks = (columns + 31) / 32;
    if (inner <= q4_g32_quad_activation_cache_max_inner &&
        columns < q4_g32_quad_activation_cache_max_columns)
        q4_g32_oct_rows_gemv_cuda<true, true><<<blocks, 128,
            static_cast<size_t>(inner) * sizeof(float), stream>>>(activation, weight, output, columns, inner);
    else
        q4_g32_oct_rows_gemv_cuda<false, true><<<blocks, 128, 0, stream>>>(
            activation, weight, output, columns, inner);
}

void launch_q4_g32_compact_gemv(const float* activation,
    const Q4CompactG32* weight, float* output, int columns, int inner, cudaStream_t stream) {
    if(columns==10240 && inner==320 && reinterpret_cast<uintptr_t>(activation)%alignof(float2)==0) {
        q4_g32_pair_rows_gemv_cuda<false,false,0,Q4CompactG32,true><<<1280,128,320,stream>>>(activation,weight,output,columns,inner);
        return;
    }
    if (columns <= 4096 && inner >= 2048 &&
        reinterpret_cast<uintptr_t>(activation) % alignof(float2) == 0) {
        q4_g32_row_parallel_gemv_cuda<256><<<columns, 256, 0, stream>>>(
            activation, weight, output, inner);
        return;
    }
    q4_g32_oct_rows_gemv_cuda<true, false, 4><<<(columns + 31) / 32, 128,
        static_cast<size_t>(inner) * sizeof(float), stream>>>(activation, weight, output, columns, inner);
}

cudaError_t launch_q4_compact_pair_probe(const float* activation,
    const Q4CompactG32* weight,float* output,int columns,int inner,int pipeline,cudaStream_t stream) {
    if(!activation||!weight||!output||columns<=0||columns>INT32_MAX-7||inner<=0||inner>INT32_MAX-32||inner%32||pipeline<0||pipeline>3||(pipeline==3&&inner>4096)||
        reinterpret_cast<uintptr_t>(activation)%alignof(float2)||reinterpret_cast<uintptr_t>(weight)%alignof(Q4CompactG32))return cudaErrorInvalidValue;
    if(pipeline==3)q4_g32_pair_rows_gemv_cuda<false,false,0,Q4CompactG32,true><<<(columns+7)/8,128,size_t(inner/32)*8*sizeof(float),stream>>>(activation,weight,output,columns,inner);
    else if(pipeline==2)q4_g32_pair_rows_gemv_cuda<false,false,2,Q4CompactG32><<<(columns+7)/8,128,0,stream>>>(activation,weight,output,columns,inner);
    else if(pipeline==1)q4_g32_pair_rows_gemv_cuda<false,false,1,Q4CompactG32><<<(columns+7)/8,128,0,stream>>>(activation,weight,output,columns,inner);
    else q4_g32_pair_rows_gemv_cuda<false,false,0,Q4CompactG32><<<(columns+7)/8,128,0,stream>>>(activation,weight,output,columns,inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_compact_cache_probe(const float* activation,
    const Q4CompactG32* weight, float* output, int columns, int inner,
    bool integer_scale, int hint, cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 4096 || columns > INT32_MAX - 31 ||
        inner <= 0 || inner > 4096 || inner % 32 || hint < 1 || hint > 9 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float4) ||
        reinterpret_cast<uintptr_t>(weight) % alignof(uint32_t)) return cudaErrorInvalidValue;
    const dim3 grid((columns + 31) / 32);
    const size_t shared = static_cast<size_t>(inner) * sizeof(float);
    if (hint >= 4 && integer_scale) return cudaErrorInvalidValue;
    if (hint == 9)
        q4_g32_oct_rows_gemv_cuda<true,false,4,Q4CompactG32,0,false,false,false,false,false,false,4,true,true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner);
    else if (hint == 8)
        q4_g32_oct_rows_gemv_cuda<true,false,4,Q4CompactG32,0,false,false,false,false,false,false,4,true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner);
    else if (hint == 7)
        q4_g32_oct_rows_gemv_cuda<true,false,0,Q4CompactG32,0,false,false,true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner);
    else if (hint == 6)
        q4_g32_oct_rows_gemv_cuda<true,false,4,Q4CompactG32,0,false,false,true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner);
    else if (hint == 4)
        q4_g32_oct_rows_gemv_cuda<true,false,4,Q4CompactG32,0,false,false,false,false,false,false,2><<<(columns+15)/16,64,shared,stream>>>(activation,weight,output,columns,inner);
    else if (hint == 5)
        q4_g32_oct_rows_gemv_cuda<true,false,4,Q4CompactG32,0,false,false,false,false,false,false,8><<<(static_cast<uint64_t>(columns)+63)/64,256,shared,stream>>>(activation,weight,output,columns,inner);
    else if (hint == 3 && !integer_scale)
        q4_g32_oct_rows_gemv_cuda<true,false,4,Q4CompactG32,0,false,false,false,false,false,true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner);
    else if (hint == 3) return cudaErrorInvalidValue;
    else if (integer_scale && hint == 1)
        q4_g32_oct_rows_gemv_cuda<true,false,4,Q4CompactG32,-1,false,false,false,false,true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner);
    else if (integer_scale)
        q4_g32_oct_rows_gemv_cuda<true,false,4,Q4CompactG32,-2,false,false,false,false,true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner);
    else if (hint == 1)
        q4_g32_oct_rows_gemv_cuda<true,false,4,Q4CompactG32,-1><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner);
    else
        q4_g32_oct_rows_gemv_cuda<true,false,4,Q4CompactG32,-2><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_compact_integer_gemv(const float* activation,
    const Q4CompactG32* weight, float* output, int columns, int inner, cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 0 || columns > INT32_MAX - 31 ||
        inner <= 0 || inner % 32) return cudaErrorInvalidValue;
    q4_g32_oct_rows_gemv_cuda<true, false, 4, Q4CompactG32, 0, false, false, false, false, true><<<
        (columns + 31) / 32, 128, static_cast<size_t>(inner) * sizeof(float), stream>>>(
        activation, weight, output, columns, inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_compact_warp_table_gemv(const float* activation,
    const Q4CompactG32* weight, float* output, int columns, int inner, cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 0 || columns > 4096 ||
        inner <= 0 || inner > INT32_MAX - 512 || inner % 512 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float2) ||
        reinterpret_cast<uintptr_t>(weight) % alignof(Q4CompactG32)) return cudaErrorInvalidValue;
    compact_warp_table_cuda<<<columns, 256, 0, stream>>>(activation, weight, output, inner);
    return cudaGetLastError();
}

void launch_q4_g32_grouped_narrow_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int rows, cudaStream_t stream) {
    switch (rows) {
    case 2: q4_g32_grouped_narrow_cuda<2><<<(columns + 1) / 2, 256, 0, stream>>>(activation, weight, output, columns, inner); break;
    case 4: q4_g32_grouped_narrow_cuda<4><<<(columns + 3) / 4, 256, 0, stream>>>(activation, weight, output, columns, inner); break;
    case 8: q4_g32_grouped_narrow_cuda<8><<<(columns + 7) / 8, 256, 0, stream>>>(activation, weight, output, columns, inner); break;
    }
}

cudaError_t launch_q8_pipeline_gemv(const float* activation, const int8_t* weight,
    const float* scales, float* output, int columns, int inner, int mode, cudaStream_t stream) {
    if (!activation || !weight || !scales || !output || columns <= 0 ||
        columns > 2147483631 || inner <= 0 || inner > 2147483519 || inner % 4 || (mode != 1 && mode != 2) ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float4) ||
        reinterpret_cast<uintptr_t>(weight) % alignof(char4)) return cudaErrorInvalidValue;
    if (mode == 1)
        q8_per_channel_dense_gemv_cuda<0, 8, false, 1><<<(columns + 7) / 8, 256, 0, stream>>>(activation, weight, scales, output, columns, inner);
    else
        q8_per_channel_dense_gemv_cuda<0, 8, false, 2><<<(columns + 7) / 8, 256, 0, stream>>>(activation, weight, scales, output, columns, inner);
    return cudaGetLastError();
}

cudaError_t launch_q8_tuned_gemv(const float* activation, const int8_t* weight,
    const float* scales, float* output, int columns, int inner, int warps,
    bool cache_activation, cudaStream_t stream) {
    if (!activation || !weight || !scales || !output || columns <= 0 ||
        columns > 2147483616 || inner <= 0 || inner % 4 ||
        (cache_activation && inner > 2560) ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float4) ||
        reinterpret_cast<uintptr_t>(weight) % alignof(char4)) return cudaErrorInvalidValue;
    if (cache_activation && warps == 8)
        q8_per_channel_dense_gemv_cuda<0, 8, true><<<(columns + 7) / 8, 256, 0, stream>>>(activation, weight, scales, output, columns, inner);
    else if (cache_activation) return cudaErrorInvalidValue;
    else if (warps == 1)
        q8_per_channel_dense_gemv_cuda<0, 1><<<columns, 32, 0, stream>>>(activation, weight, scales, output, columns, inner);
    else if (warps == 2)
        q8_per_channel_dense_gemv_cuda<0, 2><<<(columns + 1) / 2, 64, 0, stream>>>(activation, weight, scales, output, columns, inner);
    else if (warps == 4)
        q8_per_channel_dense_gemv_cuda<0, 4><<<(columns + 3) / 4, 128, 0, stream>>>(activation, weight, scales, output, columns, inner);
    else if (warps == 8)
        q8_per_channel_dense_gemv_cuda<><<<(columns + 7) / 8, 256, 0, stream>>>(activation, weight, scales, output, columns, inner);
    else if (warps == 16)
        q8_per_channel_dense_gemv_cuda<0, 16><<<(columns + 15) / 16, 512, 0, stream>>>(activation, weight, scales, output, columns, inner);
    else if (warps == 32)
        q8_per_channel_dense_gemv_cuda<0, 32><<<(columns + 31) / 32, 1024, 0, stream>>>(activation, weight, scales, output, columns, inner);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}

void launch_q8_prefetch_gemv(const float* activation, const int8_t* weight,
    const float* scales, float* output, int columns, int inner, int distance) {
    const int blocks = (columns + 7) / 8;
    switch (distance) {
    case 1: q8_per_channel_dense_gemv_cuda<1><<<blocks, 256>>>(activation, weight, scales, output, columns, inner); break;
    case 2: q8_per_channel_dense_gemv_cuda<2><<<blocks, 256>>>(activation, weight, scales, output, columns, inner); break;
    case 4: q8_per_channel_dense_gemv_cuda<4><<<blocks, 256>>>(activation, weight, scales, output, columns, inner); break;
    default: q8_per_channel_dense_gemv_cuda<><<<blocks, 256>>>(activation, weight, scales, output, columns, inner); break;
    }
}

cudaError_t launch_q8_small_batch_exact(const float* activation,
    const int8_t* weight, const float* scales, float* output, int tokens,
    int columns, int inner, int lda, int ldc, cudaStream_t stream) {
    if(!activation||!weight||!scales||!output||tokens<2||tokens>8||
        columns<=0||columns>INT32_MAX-7||inner<=0||inner%4||lda<inner||
        lda%4||ldc<columns||reinterpret_cast<uintptr_t>(activation)%alignof(float4)||
        reinterpret_cast<uintptr_t>(weight)%alignof(char4))return cudaErrorInvalidValue;
    const int blocks=(columns+7)/8;
    if(tokens==2)q8_small_batch_exact_cuda<2><<<blocks,256,0,stream>>>(activation,weight,scales,output,tokens,columns,inner,lda,ldc);
    else if(tokens<=4)q8_small_batch_exact_cuda<4><<<blocks,256,0,stream>>>(activation,weight,scales,output,tokens,columns,inner,lda,ldc);
    else q8_small_batch_exact_cuda<8><<<blocks,256,0,stream>>>(activation,weight,scales,output,tokens,columns,inner,lda,ldc);
    return cudaGetLastError();
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
        q8_per_channel_dense_gemv_cuda<><<<blocks, threads>>>(
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

void launch_prepare_q4_compact_g32_dense_gemm(
    const Q4CompactG32* weight, __half2* weight_output,
    const float* activation, __half* activation_output,
    size_t activation_count, int rows, int groups) {
    const dim3 grid((groups + q4_g32_storage_blocks_per_cuda_block - 1) /
        q4_g32_storage_blocks_per_cuda_block,
        (rows + q4_g32_rows_per_storage_block - 1) / q4_g32_rows_per_storage_block);
    // Scalar activation conversion also handles odd counts and unaligned views.
    prepare_q4_g32_dense_gemm_cuda<false><<<grid, q4_g32_dequant_threads>>>(
        weight, weight_output, activation, activation_output,
        activation_count, rows, groups);
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
    int columns, int inner, cudaStream_t stream) {
    constexpr int warps_per_block = 4;
    constexpr int threads = warps_per_block * 32;
    if (reinterpret_cast<uintptr_t>(activation) % alignof(float2) == 0) {
        // Narrow projections otherwise expose only 1--40 blocks and keep
        // hundreds of K groups serial in each lane. Parallelize the dot
        // within each row rather than wasting the rest of the GPU.
        if (columns <= 4096 && inner >= 2048) {
            static const bool grouped_narrow = [] {
                const char* option = std::getenv("MOLLM_CUDA_Q4_GROUPED_NARROW");
                return option && std::strcmp(option, "1") == 0;
            }();
            if (grouped_narrow && columns == 320 && inner == 10240) {
                launch_q4_g32_grouped_narrow_gemv(activation, weight, output, columns, inner, 2, stream);
                return;
            }
            static const bool narrow_prefetch = [] {
                const char* option = std::getenv("MOLLM_CUDA_Q4_NARROW_PREFETCH");
                return option && std::strcmp(option, "1") == 0;
            }();
            constexpr bool narrow_pipeline=true;
            const bool pipeline_shape = (columns == 2560 && inner == 6144) ||
                (columns == 320 && inner == 10240) || (columns == 640 && inner == 2560);
            if (narrow_pipeline && pipeline_shape &&
                !(narrow_prefetch && columns == 2560 && inner == 6144)) {
                q4_g32_pipelined_narrow_cuda<1,true><<<columns,256,0,stream>>>(activation,weight,output,inner);
                return;
            }
            if (narrow_prefetch && columns == 2560 && inner == 6144)
                q4_g32_row_parallel_gemv_cuda<256, false, 4><<<columns, 256, 0, stream>>>(
                    activation, weight, output, inner);
            else
                q4_g32_row_parallel_gemv_cuda<256><<<columns, 256, 0, stream>>>(
                    activation, weight, output, inner);
            return;
        }
        constexpr int quad_threads = 128;
        constexpr int quad_storage_blocks_per_cuda_block = quad_threads / 32;
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
        if (use_quad_rows && reinterpret_cast<uintptr_t>(activation) % alignof(float4) == 0) {
            if (cache_quad_activation && inner == 2560 && cuda_q4_prefetch_enabled())
                q4_g32_oct_rows_gemv_cuda<true, false, 4><<<quad_blocks, quad_threads,
                    static_cast<size_t>(inner) * sizeof(float), stream>>>(
                        activation, weight, output, columns, inner);
            else if (cache_quad_activation)
                q4_g32_oct_rows_gemv_cuda<true><<<quad_blocks, quad_threads,
                    static_cast<size_t>(inner) * sizeof(float), stream>>>(
                        activation, weight, output, columns, inner);
            else
                q4_g32_oct_rows_gemv_cuda<false><<<quad_blocks, quad_threads, 0, stream>>>(
                    activation, weight, output, columns, inner);
        } else {
            const unsigned blocks = static_cast<unsigned>(
                (columns + q4_g32_rows_per_storage_block - 1) /
                q4_g32_rows_per_storage_block);
            if (cache_pair_activation)
                q4_g32_pair_rows_gemv_cuda<true><<<
                    blocks, threads,
                    static_cast<size_t>(inner) * sizeof(float), stream>>>(
                        activation, weight, output, columns, inner);
            else
                q4_g32_pair_rows_gemv_cuda<false><<<blocks, threads, 0, stream>>>(
                    activation, weight, output, columns, inner);
        }
    } else {
        const unsigned blocks = static_cast<unsigned>(
            (columns + warps_per_block - 1) / warps_per_block);
        q4_g32_dense_gemv_cuda<><<<blocks, threads, 0, stream>>>(
            activation, weight, output, columns, inner);
    }
}

cudaError_t launch_q4_g32_small_batch(const float* activation,
    const Q4B8G32Block* weight, float* output, int tokens, int columns,
    int inner, int lda, int ldc, cudaStream_t stream) {
    if (!activation || !weight || !output || tokens < 2 || tokens > 8 ||
        columns <= 0 || inner <= 0 || inner % 32 || lda < inner || ldc < columns)
        return cudaErrorInvalidValue;
    static const bool wide = [] {
        const char* value = std::getenv("MOLLM_CUDA_Q4_SMALL_BATCH_WIDE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (wide && columns > 4096 && inner >= 2048 && lda % 4 == 0 &&
        reinterpret_cast<uintptr_t>(activation) % alignof(float4) == 0)
        return launch_q4_g32_small_batch_wide(activation, weight, output, tokens, columns, inner, lda, ldc, stream);
    static const bool reduced = [] {
        const char* value = std::getenv("MOLLM_CUDA_Q4_SMALL_BATCH_REDUCE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (reduced && tokens <= 4)
        return launch_q4_g32_small_batch_reduced(activation, weight, output, tokens, columns, inner, lda, ldc, stream);
    if (tokens == 2)
        q4_g32_small_batch_cuda<2><<<columns, 256, 0, stream>>>(activation, weight, output, tokens, inner, lda, ldc);
    else if (tokens <= 4)
        q4_g32_small_batch_cuda<4><<<columns, 256, 0, stream>>>(activation, weight, output, tokens, inner, lda, ldc);
    else
        q4_g32_small_batch_cuda<8><<<columns, 256, 0, stream>>>(activation, weight, output, tokens, inner, lda, ldc);
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_small_batch_wide(const float* activation,
    const Q4B8G32Block* weight, float* output, int tokens, int columns,
    int inner, int lda, int ldc, cudaStream_t stream) {
    if (!activation || !weight || !output || tokens < 2 || tokens > 8 ||
        columns <= 0 || columns > INT32_MAX - 31 || inner <= 0 || inner % 32 ||
        lda < inner || ldc < columns || lda % 4 || reinterpret_cast<uintptr_t>(activation) % alignof(float4))
        return cudaErrorInvalidValue;
    const int blocks = (columns + 31) / 32;
    if (tokens == 2)
        q4_g32_small_batch_wide_cuda<2><<<blocks,128,0,stream>>>(activation,weight,output,tokens,columns,inner,lda,ldc);
    else if (tokens <= 4)
        q4_g32_small_batch_wide_cuda<4><<<blocks,128,0,stream>>>(activation,weight,output,tokens,columns,inner,lda,ldc);
    else
        q4_g32_small_batch_wide_cuda<8><<<blocks,128,0,stream>>>(activation,weight,output,tokens,columns,inner,lda,ldc);
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_small_batch_exact(const float* activation,
    const Q4B8G32Block* weight, float* output, int tokens, int columns,
    int inner, int lda, int ldc, cudaStream_t stream) {
    if (!activation || !weight || !output || tokens < 2 || tokens > 8 ||
        columns <= 0 || columns > INT32_MAX - 31 || inner <= 0 || inner % 32 ||
        lda < inner || ldc < columns) return cudaErrorInvalidValue;
    const bool aligned_pair = lda % 2 == 0 &&
        reinterpret_cast<uintptr_t>(activation) % alignof(float2) == 0;
    // Experimental grouped narrow decode can have a different reduction tree.
    const char* grouped = std::getenv("MOLLM_CUDA_Q4_GROUPED_NARROW");
    if (aligned_pair && columns <= 4096 && inner >= 2048 &&
        !(grouped && std::strcmp(grouped, "1") == 0)) {
        if (tokens <= 4)
            return launch_q4_g32_small_batch_reduced(activation, weight, output,
                tokens, columns, inner, lda, ldc, stream);
        // The single-barrier variant regresses at M8. The original row tree
        // still matches decode and reuses weights without that bottleneck.
        q4_g32_small_batch_cuda<8><<<columns,256,0,stream>>>(
            activation,weight,output,tokens,inner,lda,ldc);
        return cudaGetLastError();
    }
    const unsigned blocks = static_cast<unsigned>((columns + 31) / 32);
    const int sms = cuda_multiprocessor_count();
    const bool enough_grid = sms > 0 ?
        blocks >= static_cast<unsigned>(sms * q4_g32_quad_rows_target_blocks_per_sm) :
        columns >= q4_g32_quad_rows_fallback_min_columns;
    if (columns > 4096 && inner / 32 >= q4_g32_quad_rows_min_groups && enough_grid &&
        lda % 4 == 0 && reinterpret_cast<uintptr_t>(activation) % alignof(float4) == 0)
        return launch_q4_g32_small_batch_wide(activation, weight, output,
            tokens, columns, inner, lda, ldc, stream);
    for (int token = 0; token < tokens; ++token)
        launch_q4_g32_dense_gemv(activation + static_cast<size_t>(token) * lda,
            weight, output + static_cast<size_t>(token) * ldc, columns, inner, stream);
    return cudaGetLastError();
}

cudaError_t launch_q4_compact_small_batch_exact(const float* activation,
    const Q4CompactG32* weight, float* output, int tokens, int columns,
    int inner, int lda, int ldc, bool integer_scale, cudaStream_t stream) {
    if (!activation || !weight || !output || tokens < 2 || tokens > 8 ||
        columns <= 0 || columns > INT32_MAX - 31 || inner <= 0 || inner % 32 ||
        lda < inner || ldc < columns) return cudaErrorInvalidValue;
    // Only use the same four-lane row tree as the compact wide decode path.
    if ((columns == 13312 || columns == 16480) && inner == 2560 &&
        lda % 4 == 0 && reinterpret_cast<uintptr_t>(activation) % alignof(float4) == 0) {
        const int blocks = (columns + 31) / 32;
        if (tokens == 2)
            q4_g32_small_batch_wide_cuda<2,Q4CompactG32><<<blocks,128,0,stream>>>(activation,weight,output,tokens,columns,inner,lda,ldc);
        else if (tokens <= 4)
            q4_g32_small_batch_wide_cuda<4,Q4CompactG32><<<blocks,128,0,stream>>>(activation,weight,output,tokens,columns,inner,lda,ldc);
        else
            q4_g32_small_batch_wide_cuda<8,Q4CompactG32><<<blocks,128,0,stream>>>(activation,weight,output,tokens,columns,inner,lda,ldc);
    } else {
        for (int token = 0; token < tokens; ++token) {
            if (integer_scale) {
                const cudaError_t status = launch_q4_compact_integer_gemv(
                    activation + static_cast<size_t>(token)*lda,
                    weight, output + static_cast<size_t>(token)*ldc, columns, inner, stream);
                if (status != cudaSuccess) return status;
            }
            else launch_q4_g32_compact_gemv(activation + static_cast<size_t>(token)*lda,
                    weight, output + static_cast<size_t>(token)*ldc, columns, inner, stream);
        }
    }
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_small_batch_reduced(const float* activation,
    const Q4B8G32Block* weight, float* output, int tokens, int columns,
    int inner, int lda, int ldc, cudaStream_t stream) {
    if (!activation || !weight || !output || tokens < 2 || tokens > 8 ||
        columns <= 0 || inner <= 0 || inner % 32 || lda < inner || ldc < columns)
        return cudaErrorInvalidValue;
    if (tokens == 2)
        q4_g32_small_batch_cuda<2,true><<<columns,256,0,stream>>>(activation,weight,output,tokens,inner,lda,ldc);
    else if (tokens <= 4)
        q4_g32_small_batch_cuda<4,true><<<columns,256,0,stream>>>(activation,weight,output,tokens,inner,lda,ldc);
    else
        q4_g32_small_batch_cuda<8,true><<<columns,256,0,stream>>>(activation,weight,output,tokens,inner,lda,ldc);
    return cudaGetLastError();
}

void launch_q4_row_g32_gemv(const float* activation, const Q4RowG32* weight,
    float* output, int columns, int inner, cudaStream_t stream) {
    q4_row_g32_gemv_cuda<<<columns, 256, 0, stream>>>(activation, weight, output, inner);
}

void launch_repack_q4_row_g32(const Q4B8G32Block* source, Q4RowG32* target,
    int columns, int inner) {
    const size_t count = static_cast<size_t>(columns) * (inner / 32);
    repack_q4_row_g32_cuda<<<(count + 255) / 256, 256>>>(source, target, columns, inner / 32);
}

void launch_q4_g32_terminal_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner) {
    q4_g32_row_parallel_gemv_cuda<256, true><<<columns, 256>>>(
        activation, weight, output, inner);
}

cudaError_t launch_q4_g32_paired_projections(const float* activation,
    const Q4B8G32Block* first_weight, const Q4B8G32Block* second_weight,
    float* first_output, float* second_output, int columns, int inner,
    cudaStream_t stream) {
    if (!activation || !first_weight || !second_weight || !first_output || !second_output ||
        columns <= 0 || columns > 4096 || inner < 2048 || inner % 32 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float2)) return cudaErrorInvalidValue;
    q4_g32_paired_projections_cuda<<<dim3(columns, 2), 256, 0, stream>>>(activation,
        first_weight, second_weight, first_output, second_output, inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_shared_activation(const float* activation,
    const Q4B8G32Block* gate_weight, const Q4B8G32Block* up_weight,
    float* output, int columns, int inner, cudaStream_t stream) {
    if (!activation || !gate_weight || !up_weight || !output || columns <= 0 ||
        columns > 4096 || inner < 2048 || inner % 32 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float2)) return cudaErrorInvalidValue;
    q4_g32_shared_activation_cuda<<<columns,256,0,stream>>>(activation,
        gate_weight,up_weight,output,inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_dual_lane_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    cudaStream_t stream, bool pipeline_inputs) {
    if (!activation || !weight || !output || columns <= 0 || inner <= 0 ||
        inner % 32 || reinterpret_cast<uintptr_t>(activation) % alignof(float2))
        return cudaErrorInvalidValue;
    if(pipeline_inputs)
        q4_g32_dual_lane_pipeline_cuda<<<columns,128,0,stream>>>(activation,weight,output,inner);
    else q4_g32_dual_lane_gemv_cuda<<<columns, 128, 0, stream>>>(activation, weight, output, inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_single_barrier_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 0 || columns > 4096 ||
        inner < 2048 || inner % 32 || reinterpret_cast<uintptr_t>(activation) % alignof(float2))
        return cudaErrorInvalidValue;
    q4_g32_row_parallel_gemv_cuda<256, false, 0, Q4B8G32Block, true><<<columns, 256, 0, stream>>>(
        activation, weight, output, inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_unequal_projections(const float* activation,
    const Q4B8G32Block* first_weight, const Q4B8G32Block* second_weight,
    float* first_output, float* second_output, int first_columns, int second_columns,
    int inner, cudaStream_t stream) {
    if (!activation || !first_weight || !second_weight || !first_output || !second_output ||
        first_columns <= 0 || first_columns > 4096 || second_columns <= 0 || second_columns > 4096 ||
        inner < 2048 || inner > INT32_MAX - 2048 || inner % 32 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float2)) return cudaErrorInvalidValue;
    q4_g32_unequal_projections_cuda<<<first_columns + second_columns, 256, 0, stream>>>(
        activation, first_weight, second_weight, first_output, second_output, first_columns, inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_pipelined_narrow_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int depth,
    bool pipeline_inputs, cudaStream_t stream, bool single_barrier) {
    if (!activation || !weight || !output || columns <= 0 || columns > 4096 ||
        inner < 2048 || inner > INT32_MAX - 2048 || inner % 32 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float2)) return cudaErrorInvalidValue;
    if(single_barrier) {
        if(!pipeline_inputs || depth!=1)return cudaErrorInvalidValue;
        q4_g32_pipelined_narrow_cuda<1,true,Q4B8G32Block,true><<<columns,256,0,stream>>>(activation,weight,output,inner);
    } else if (pipeline_inputs) {
        switch (depth) {
        case 1: q4_g32_pipelined_narrow_cuda<1,true><<<columns,256,0,stream>>>(activation,weight,output,inner); break;
        case 2: q4_g32_pipelined_narrow_cuda<2,true><<<columns,256,0,stream>>>(activation,weight,output,inner); break;
        case 4: q4_g32_pipelined_narrow_cuda<4,true><<<columns,256,0,stream>>>(activation,weight,output,inner); break;
        default: return cudaErrorInvalidValue;
        }
    } else {
        switch (depth) {
        case 1: q4_g32_pipelined_narrow_cuda<1,false><<<columns,256,0,stream>>>(activation,weight,output,inner); break;
        case 2: q4_g32_pipelined_narrow_cuda<2,false><<<columns,256,0,stream>>>(activation,weight,output,inner); break;
        case 4: q4_g32_pipelined_narrow_cuda<4,false><<<columns,256,0,stream>>>(activation,weight,output,inner); break;
        default: return cudaErrorInvalidValue;
        }
    }
    return cudaGetLastError();
}

cudaError_t launch_q4_compact_pipelined_narrow_gemv(const float* activation,
    const Q4CompactG32* weight,float* output,int columns,int inner,cudaStream_t stream) {
    if(!activation||!weight||!output||columns<=0||columns>4096||inner<2048||
        inner>INT32_MAX-2048||inner%32||reinterpret_cast<uintptr_t>(activation)%alignof(float2))
        return cudaErrorInvalidValue;
    q4_g32_pipelined_narrow_cuda<1,true,Q4CompactG32><<<columns,256,0,stream>>>(activation,weight,output,inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_spatial_narrow_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int rows_per_cta,
    cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 0 || columns > 4096 ||
        inner < 2048 || inner % 32 || reinterpret_cast<uintptr_t>(activation) % alignof(float2))
        return cudaErrorInvalidValue;
    switch (rows_per_cta) {
    case 2: q4_g32_spatial_narrow_cuda<2><<<(columns + 1) / 2, 256, 0, stream>>>(activation, weight, output, columns, inner); break;
    case 4: q4_g32_spatial_narrow_cuda<4><<<(columns + 3) / 4, 256, 0, stream>>>(activation, weight, output, columns, inner); break;
    case 8: q4_g32_spatial_narrow_cuda<8><<<(columns + 7) / 8, 256, 0, stream>>>(activation, weight, output, columns, inner); break;
    default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

void launch_q4_g32_pair_narrow_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, bool cache, cudaStream_t stream) {
    if (cache)
        q4_g32_pair_rows_gemv_cuda<true><<<(columns + 7) / 8, 128, inner * sizeof(float), stream>>>(
            activation, weight, output, columns, inner);
    else
        q4_g32_pair_rows_gemv_cuda<false><<<(columns + 7) / 8, 128, 0, stream>>>(
            activation, weight, output, columns, inner);
}

cudaError_t launch_q4_g32_l1_prefetch_wide_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    int distance, cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 0 || columns > INT32_MAX - 31 ||
        inner <= 0 || inner > 2560 || inner % 32 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float4)) return cudaErrorInvalidValue;
    const dim3 grid((columns + 31) / 32);
    const size_t shared = static_cast<size_t>(inner) * sizeof(float);
    switch (distance) {
    case 1: q4_g32_oct_rows_gemv_cuda<true, false, 1, Q4B8G32Block, 0, false, false, false, true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner); break;
    case 2: q4_g32_oct_rows_gemv_cuda<true, false, 2, Q4B8G32Block, 0, false, false, false, true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner); break;
    case 4: q4_g32_oct_rows_gemv_cuda<true, false, 4, Q4B8G32Block, 0, false, false, false, true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner); break;
    case 8: q4_g32_oct_rows_gemv_cuda<true, false, 8, Q4B8G32Block, 0, false, false, false, true><<<grid,128,shared,stream>>>(activation,weight,output,columns,inner); break;
    default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_pipelined_pair_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    bool pipeline_inputs, cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 0 || columns > INT32_MAX - 7 ||
        inner <= 0 || inner > 2560 || inner % 32 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float2)) return cudaErrorInvalidValue;
    if (pipeline_inputs)
        q4_g32_pair_rows_gemv_cuda<false, false, 2><<<(columns + 7) / 8, 128, 0, stream>>>(activation, weight, output, columns, inner);
    else
        q4_g32_pair_rows_gemv_cuda<false, false, 1><<<(columns + 7) / 8, 128, 0, stream>>>(activation, weight, output, columns, inner);
    return cudaGetLastError();
}

void launch_q4_g32_split_pair_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int splits) {
    switch (splits) {
    case 2: q4_g32_split_pair_gemv_cuda<2><<<(columns + 7) / 8, 256>>>(activation, weight, output, columns, inner); break;
    case 4: q4_g32_split_pair_gemv_cuda<4><<<(columns + 7) / 8, 512>>>(activation, weight, output, columns, inner); break;
    case 8: q4_g32_split_pair_gemv_cuda<8><<<(columns + 7) / 8, 1024>>>(activation, weight, output, columns, inner); break;
    }
}

void launch_q4_g32_prefetch_narrow_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int distance) {
    switch (distance) {
    case 1: q4_g32_row_parallel_gemv_cuda<256, false, 1><<<columns, 256>>>(activation, weight, output, inner); break;
    case 2: q4_g32_row_parallel_gemv_cuda<256, false, 2><<<columns, 256>>>(activation, weight, output, inner); break;
    case 4: q4_g32_row_parallel_gemv_cuda<256, false, 4><<<columns, 256>>>(activation, weight, output, inner); break;
    default: q4_g32_row_parallel_gemv_cuda<256><<<columns, 256>>>(activation, weight, output, inner); break;
    }
}

void launch_q4_g32_uncached_wide_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner) {
    q4_g32_oct_rows_gemv_cuda<false><<<(columns + 31) / 32, 128>>>(
        activation, weight, output, columns, inner);
}

cudaError_t launch_q4_g32_async_wide_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 0 || columns > 2147483615 ||
        inner <= 0 || inner > 2560 || inner % 32 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float4) ||
        reinterpret_cast<uintptr_t>(weight) % 16) return cudaErrorInvalidValue;
    q4_g32_oct_rows_gemv_cuda<true, false, 0, Q4B8G32Block, 0, true><<<(columns + 31) / 32, 128,
        static_cast<size_t>(inner) * sizeof(float), stream>>>(activation, weight, output, columns, inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_load_hint_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int hint,
    cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 0 || columns > 2147483615 ||
        inner <= 0 || inner > 2560 || inner % 32 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float4)) return cudaErrorInvalidValue;
    const unsigned blocks = (columns + 31) / 32;
    const size_t shared = static_cast<size_t>(inner) * sizeof(float);
    switch (hint) {
    case 32: q4_g32_oct_rows_gemv_cuda<true, false, 4, Q4B8G32Block, 32><<<blocks, 128, shared, stream>>>(activation, weight, output, columns, inner); break;
    case 64: q4_g32_oct_rows_gemv_cuda<true, false, 4, Q4B8G32Block, 64><<<blocks, 128, shared, stream>>>(activation, weight, output, columns, inner); break;
    case 128: q4_g32_oct_rows_gemv_cuda<true, false, 4, Q4B8G32Block, 128><<<blocks, 128, shared, stream>>>(activation, weight, output, columns, inner); break;
    case 256: q4_g32_oct_rows_gemv_cuda<true, false, 4, Q4B8G32Block, 256><<<blocks, 128, shared, stream>>>(activation, weight, output, columns, inner); break;
    default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

void launch_q4_g32_prefetch_wide_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int distance,
    cudaStream_t stream) {
    const unsigned blocks = (columns + 31) / 32;
    const size_t shared = inner * sizeof(float);
    switch (distance) {
    case 4: q4_g32_oct_rows_gemv_cuda<true, false, 4><<<blocks, 128, shared, stream>>>(activation, weight, output, columns, inner); break;
    case 8: q4_g32_oct_rows_gemv_cuda<true, false, 8><<<blocks, 128, shared, stream>>>(activation, weight, output, columns, inner); break;
    case 16: q4_g32_oct_rows_gemv_cuda<true, false, 16><<<blocks, 128, shared, stream>>>(activation, weight, output, columns, inner); break;
    }
}

cudaError_t launch_q4_g32_pipelined_wide_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 0 || columns > INT32_MAX - 31 ||
        inner <= 0 || inner > 2560 || inner % 32 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float4)) return cudaErrorInvalidValue;
    q4_g32_oct_rows_gemv_cuda<true, false, 4, Q4B8G32Block, 0, false, false, true>
        <<<(columns + 31) / 32, 128, static_cast<size_t>(inner) * sizeof(float), stream>>>(
            activation, weight, output, columns, inner);
    return cudaGetLastError();
}

cudaError_t launch_q4_g32_prefetch_scale_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    int distance, cudaStream_t stream) {
    if (!activation || !weight || !output || columns <= 0 || columns > 2147483615 ||
        inner <= 0 || inner > 2560 || inner % 32 ||
        reinterpret_cast<uintptr_t>(activation) % alignof(float4)) return cudaErrorInvalidValue;
    const unsigned blocks = (columns + 31) / 32;
    const size_t shared = static_cast<size_t>(inner) * sizeof(float);
    if (distance == 4)
        q4_g32_oct_rows_gemv_cuda<true, false, 4, Q4B8G32Block, 0, false, true><<<blocks,128,shared,stream>>>(activation,weight,output,columns,inner);
    else if (distance == 8)
        q4_g32_oct_rows_gemv_cuda<true, false, 8, Q4B8G32Block, 0, false, true><<<blocks,128,shared,stream>>>(activation,weight,output,columns,inner);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
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

void launch_fp16_to_fp32(const __half* source, float* destination, size_t count) {
    if (!count) return;
    const size_t blocks = (count + 255) / 256;
    fp16_to_fp32_reference<<<static_cast<unsigned>(blocks > 65535 ? 65535 : blocks), 256>>>(source, destination, count);
}

bool run_dense_matmul(
    cublasHandle_t cublas, const void* weight, cudaDataType weight_type,
    const void* activation, cudaDataType activation_type, float* output,
    int m, int n, int k, int lda, bool pedantic, int hc_projection_row_tile) {
    // Diagnostic operator-local tiling, not a graph/prompt chunk policy.
    // Unsupported geometries and decode retain the original single call.
    if (hc_projection_row_tile != 0 && hc_projection_row_tile != 1024) return false;
    if (hc_projection_row_tile == 1024 && m > 1024 && m <= 2048 &&
        (n == 4 || n == 320) && k == 10240 && lda >= k && !pedantic &&
        weight_type == CUDA_R_16F && activation_type == CUDA_R_16F) {
        for (int offset = 0; offset < m; offset += 1024) {
            if (!run_dense_matmul(cublas, weight, weight_type,
                    static_cast<const __half*>(activation) + size_t(offset) * lda,
                    activation_type, output + size_t(offset) * n,
                    std::min(1024, m - offset), n, k, lda, false, 0))
                return false;
        }
        return true;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C[M,N] = A[M,K] * W[N,K]^T is the equivalent
    // column-major operation C_col[N,M] = W_col[K,N]^T * A_col[K,M].
    const cublasStatus_t status = cublasGemmEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, weight,
        weight_type, k, activation, activation_type, lda, &beta, output,
        CUDA_R_32F, n, pedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,
        pedantic ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_SUCCESS)
        return true;
    std::fprintf(stderr,
                 "CudaBackend: cublasGemmEx failed (cuBLAS status %d)\n",
                 static_cast<int>(status));
    return false;
}

}  // namespace mollm_cuda
