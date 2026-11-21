#include "kernels/cuda/draft_compact_head.h"
#include "kernels/cuda/device_span.h"
#include "kernels/cuda/reduction.cuh"

#include <climits>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <math_constants.h>

namespace {

// Round FP32 bits to the nearest-even BF16 bits. NaN inputs are not
// special-cased; callers that must preserve NaN check for it first.
__device__ __forceinline__ uint16_t bf16_bits_round_nearest_even(unsigned bits) {
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

__global__ void draft_probability_partial(
    const float* logits, int count, const mollm_cuda::DraftHeadResult* result,
    float* partial) {
    __shared__ float reduction[256];
    const float maximum = result->top1.value;
    float sum = 0.0f;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size_t(count); i += size_t(gridDim.x) * blockDim.x)
        sum += expf(logits[i] - maximum);
    const float value = mollm_cuda::detail::block_reduce_sum<256>(sum, reduction);
    if (!threadIdx.x) partial[blockIdx.x] = value;
}

__global__ void draft_probability_finish(
    const float* partial, int groups, mollm_cuda::DraftHeadResult* result) {
    __shared__ float reduction[256];
    const float value = threadIdx.x < groups ? partial[threadIdx.x] : 0.0f;
    const float denominator = mollm_cuda::detail::block_reduce_sum<256>(value, reduction);
    if (!threadIdx.x) {
        const float probability = 1.0f / denominator;
        result->probability = isfinite(probability) && probability > 0.0f &&
            probability <= 1.0f ? probability : -1.0f;
        result->reserved = 0.0f;
    }
}

__global__ void pack_q8_draft_head_compact_cuda(
    const int8_t* weights, const float* scales,
    mollm_cuda::Q4CompactG32* output, int rows, int inner) {
    const int groups = inner / 32;
    const int group = blockIdx.x % groups;
    const int row = (blockIdx.x / groups) * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;

    const int value = weights[size_t(row) * inner + group * 32 + lane];
    int maximum = abs(value);
    for (int offset = 16; offset; offset /= 2)
        maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    maximum = __shfl_sync(0xffffffffu, maximum, 0);
    int code = maximum ? __float2int_rn(
        __fmul_rn(float(value), __fdiv_rn(7.f, float(maximum)))) : 0;
    code = max(-7, min(7, code)) + 8;
    const int next = __shfl_down_sync(0xffffffffu, code, 1);
    auto& block = output[size_t(row / 8) * groups + group];
    if (!(lane & 1))
        block.q[row % 8][lane / 2] = uint8_t(code | (next << 4));
    if (!lane) {
        const unsigned bits = __float_as_uint(float(maximum) * scales[row]);
        block.scales[row % 8] = bf16_bits_round_nearest_even(bits);
    }
}

// Fit the existing signed Q4 levels once at load. Integer dot products make
// each K32 scale fit independent of the warp reduction order.
__global__ void pack_q8_draft_head_compact_fitted_cuda(
    const int8_t* weights, const float* scales,
    mollm_cuda::Q4CompactG32* output, int rows, int inner) {
    const int groups = inner / 32;
    const int group = blockIdx.x % groups;
    const int row = (blockIdx.x / groups) * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;
    const int value = weights[size_t(row) * inner + group * 32 + lane];
    int maximum = abs(value);
    for (int distance = 16; distance; distance /= 2)
        maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, distance));
    maximum = __shfl_sync(0xffffffffu, maximum, 0);
    float step = __fdiv_rn(float(maximum), 7.f);
    int quantized = 0;
    // Four updates, then quantize once more and fit the final stored scale.
    for (int iteration = 0; iteration < 5; ++iteration) {
        quantized = step > 0.f ? max(-7, min(7,
            __float2int_rn(__fdiv_rn(float(value), step)))) : 0;
        int numerator = value * quantized;
        int denominator = quantized * quantized;
        for (int distance = 16; distance; distance /= 2) {
            numerator += __shfl_down_sync(0xffffffffu, numerator, distance);
            denominator += __shfl_down_sync(0xffffffffu, denominator, distance);
        }
        numerator = __shfl_sync(0xffffffffu, numerator, 0);
        denominator = __shfl_sync(0xffffffffu, denominator, 0);
        step = denominator ? __fdiv_rn(float(numerator), float(denominator)) : 0.f;
    }
    const int code = quantized + 8;
    const int next = __shfl_down_sync(0xffffffffu, code, 1);
    auto& block = output[size_t(row / 8) * groups + group];
    if (!(lane & 1)) block.q[row % 8][lane / 2] = uint8_t(code | (next << 4));
    if (!lane) {
        const unsigned bits = __float_as_uint(
            __fmul_rn(__fmul_rn(7.f, step), scales[row]));
        // Canonical GPU NaN bits can overflow the usual integer BF16 rounding.
        // Preserve NaN semantics rather than wrapping to negative zero.
        block.scales[row % 8] =
            (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x7fffffu)
            ? uint16_t(0x7fc0u)
            : bf16_bits_round_nearest_even(bits);
    }
}

constexpr int screen_rows = 248320;
constexpr int screen_inner = 2560;
constexpr int screen_candidates = 1024;

// Fixed-size scratch spans are checked before any kernel is submitted.
bool screen_scratch_valid(const mollm_cuda::DraftHeadScreenScratch& scratch, bool temporary) {
    constexpr size_t bytes = size_t(screen_rows) * 4;
    const mollm_cuda::DeviceSpan spans[] = {{scratch.coarse, bytes}, {scratch.sorted, bytes},
        {scratch.ids, bytes}, {scratch.sorted_ids, bytes}, {scratch.groups, 32 * sizeof(int)},
        {scratch.sort_temp, scratch.sort_bytes}};
    return mollm_cuda::device_spans_disjoint(spans, temporary ? 6 : 5, 4) &&
        (!temporary || (scratch.sort_bytes && uintptr_t(scratch.sort_temp) % 256 == 0));
}

__global__ void q4_draft_screen_initialize_cuda(int* ids, float* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < screen_rows) {
        if (ids) ids[i] = i;
        if (output) output[i] = -CUDART_INF_F;
    }
}

__global__ void q4_draft_screen_groups_cuda(const float* input, int* output) {
    using Sort = cub::BlockRadixSort<float, 128, 1, int>;
    __shared__ typename Sort::TempStorage temporary;
    const int id = threadIdx.x;
    float energy = -CUDART_INF_F;
    if (id < 80) {
        energy = 0.f;
        for (int j = 0; j < 32; ++j) {
            const float value = input[id * 32 + j];
            energy = __fadd_rn(energy, __fmul_rn(value, value));
        }
        // Float radix ordering distinguishes NaN bit patterns. Keep every
        // real group above the -infinity padding, including non-finite input,
        // so the coarse pass can never index a padding group beyond K.
        if (isnan(energy)) energy = CUDART_INF_F;
    }
    float key[1] = {energy};
    int group[1] = {id};
    Sort(temporary).SortDescending(key, group);
    if (threadIdx.x < 32) output[threadIdx.x] = group[0];
}

// Four physical lanes retain the original eight logical FP32 chains: each
// lane owns two sums, merges them, then uses the same stride-2/1 shuffle tree.
// Only the coarse pass changes the K32 groups; selected rescoring uses all 80.
template<bool Selected, int Groups = 32>
__global__ void q4_draft_screen_dot_cuda(const float* x,
    const mollm_cuda::Q4CompactG32* weights, float* output,
    const int* ids, const int* chosen) {
    __shared__ __align__(16) float input[screen_inner];
    for (int j = threadIdx.x; j < screen_inner; j += blockDim.x) input[j] = x[j];
    __syncthreads();
    const int ordinal = (blockIdx.x * blockDim.x + threadIdx.x) / 4;
    const int lane = threadIdx.x % 4;
    if (ordinal >= (Selected ? screen_candidates : screen_rows)) return;
    const int row = Selected ? ids[ordinal] : ordinal;
    float first = 0.f, second = 0.f;
    for (int it = 0; it < (Selected ? 80 : Groups); ++it) {
        const int group = Selected ? it : chosen[it];
        const auto& block = weights[size_t(row / 8) * 80 + group];
        const float scale = __fdiv_rn(__uint_as_float(uint32_t(block.scales[row % 8]) << 16), 7.f);
        const uint16_t a = reinterpret_cast<const uint16_t*>(block.q[row % 8])[lane];
        const uint16_t b = reinterpret_cast<const uint16_t*>(block.q[row % 8])[lane + 4];
        const float4 u = reinterpret_cast<const float4*>(input + group * 32)[lane];
        const float4 v = reinterpret_cast<const float4*>(input + group * 32)[lane + 4];
        float d = fmaf(u.x, float(int(a & 15) - 8), u.y * float(int((a >> 4) & 15) - 8));
        d = fmaf(u.z, float(int((a >> 8) & 15) - 8), d);
        d = fmaf(u.w, float(int(a >> 12) - 8), d);
        float e = fmaf(v.x, float(int(b & 15) - 8), v.y * float(int((b >> 4) & 15) - 8));
        e = fmaf(v.z, float(int((b >> 8) & 15) - 8), e);
        e = fmaf(v.w, float(int(b >> 12) - 8), e);
        first = fmaf(d, scale, first);
        second = fmaf(e, scale, second);
    }
    first += second;
    for (int offset = 2; offset; offset /= 2)
        first += __shfl_down_sync(0xffffffffu, first, offset, 4);
    if (!lane) output[row] = first;
}

}  // namespace

cudaError_t mollm_cuda::launch_draft_head_probability(
    const float* logits, int count, DraftHeadResult* result,
    float* scratch, int groups, cudaStream_t stream) {
    if (!logits || !result || !scratch || count <= 0 || groups < 1 || groups > 256)
        return cudaErrorInvalidValue;
    draft_probability_partial<<<groups, 256, 0, stream>>>(logits, count, result, scratch);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    draft_probability_finish<<<1, 256, 0, stream>>>(scratch, groups, result);
    return cudaGetLastError();
}

cudaError_t mollm_cuda::pack_q8_draft_head_compact(
    const int8_t* weights, const float* scales, Q4CompactG32* output,
    int rows, int inner, cudaStream_t stream) {
    if (!weights || !scales || !output || rows <= 0 || rows > INT_MAX - 7 ||
        inner <= 0 || inner % 32)
        return cudaErrorInvalidValue;
    const size_t groups = inner / 32;
    const size_t blocks = size_t((rows + 3) / 4) * groups;
    if (blocks > INT_MAX) return cudaErrorInvalidValue;
    const auto status = cudaMemsetAsync(
        output, 0, size_t((rows + 7) / 8) * groups * sizeof(Q4CompactG32), stream);
    if (status != cudaSuccess) return status;
    pack_q8_draft_head_compact_cuda<<<unsigned(blocks), 128, 0, stream>>>(
        weights, scales, output, rows, inner);
    return cudaGetLastError();
}

cudaError_t mollm_cuda::pack_q8_draft_head_compact_fitted(
    const int8_t* weights, const float* scales, Q4CompactG32* output,
    int rows, int inner, cudaStream_t stream) {
    if (!weights || !scales || !output || rows <= 0 || rows > INT_MAX - 7 ||
        inner <= 0 || inner % 32 || uintptr_t(scales) % alignof(float) ||
        uintptr_t(output) % alignof(Q4CompactG32)) return cudaErrorInvalidValue;
    const size_t groups = inner / 32;
    const size_t blocks = size_t((rows + 3) / 4) * groups;
    if (blocks > INT_MAX) return cudaErrorInvalidValue;
    const auto status = cudaMemsetAsync(
        output, 0, size_t((rows + 7) / 8) * groups * sizeof(Q4CompactG32), stream);
    if (status != cudaSuccess) return status;
    pack_q8_draft_head_compact_fitted_cuda<<<unsigned(blocks), 128, 0, stream>>>(
        weights, scales, output, rows, inner);
    return cudaGetLastError();
}

cudaError_t mollm_cuda::draft_head_screen_temp_bytes(
    const DraftHeadScreenScratch& scratch, size_t& bytes) {
    bytes = 0;
    if (!screen_scratch_valid(scratch, false)) return cudaErrorInvalidValue;
    return cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes,
        scratch.coarse, scratch.sorted, scratch.ids, scratch.sorted_ids,
        screen_rows, 0, 32, cudaStreamPerThread);
}

cudaError_t mollm_cuda::initialize_draft_head_screen(
    const DraftHeadScreenScratch& scratch, cudaStream_t stream) {
    if (!screen_scratch_valid(scratch, true)) return cudaErrorInvalidValue;
    size_t required = 0;
    auto status = draft_head_screen_temp_bytes(scratch, required);
    if (status != cudaSuccess) return status;
    if (scratch.sort_bytes < required) return cudaErrorInvalidValue;
    q4_draft_screen_initialize_cuda<<<(screen_rows + 255) / 256, 256, 0, stream>>>(scratch.ids, nullptr);
    return cudaGetLastError();
}

cudaError_t mollm_cuda::launch_draft_head_screen(const float* input,
    const Q4CompactG32* weights, float* output, int rows, int inner,
    const DraftHeadScreenScratch& scratch, cudaStream_t stream, int group_count) {
    if ((group_count != 32 && group_count != 24 && group_count != 16) || rows != screen_rows ||
        inner != screen_inner || uintptr_t(input) % 16 ||
        !screen_scratch_valid(scratch, true)) return cudaErrorInvalidValue;
    constexpr size_t bytes = size_t(screen_rows) * 4;
    const mollm_cuda::DeviceSpan spans[] = {{input, screen_inner * sizeof(float)},
        {weights, size_t(screen_rows / 8) * (screen_inner / 32) * sizeof(Q4CompactG32)},
        {output, bytes}, {scratch.coarse, bytes}, {scratch.sorted, bytes},
        {scratch.ids, bytes}, {scratch.sorted_ids, bytes}, {scratch.groups, 32 * sizeof(int)},
        {scratch.sort_temp, scratch.sort_bytes}};
    if (!mollm_cuda::device_spans_disjoint(spans, 9, 4)) return cudaErrorInvalidValue;
    size_t required = 0;
    auto status = draft_head_screen_temp_bytes(scratch, required);
    if (status != cudaSuccess) return status;
    if (scratch.sort_bytes < required) return cudaErrorInvalidValue;
    q4_draft_screen_groups_cuda<<<1, 128, 0, stream>>>(input, scratch.groups);
    if (group_count == 16)
        q4_draft_screen_dot_cuda<false, 16><<<(screen_rows + 31) / 32, 128, 0, stream>>>(
            input, weights, scratch.coarse, nullptr, scratch.groups);
    else if (group_count == 24)
        q4_draft_screen_dot_cuda<false, 24><<<(screen_rows + 31) / 32, 128, 0, stream>>>(
            input, weights, scratch.coarse, nullptr, scratch.groups);
    else
        q4_draft_screen_dot_cuda<false><<<(screen_rows + 31) / 32, 128, 0, stream>>>(
            input, weights, scratch.coarse, nullptr, scratch.groups);
    size_t temporary_bytes = scratch.sort_bytes;
    status = cub::DeviceRadixSort::SortPairsDescending(scratch.sort_temp, temporary_bytes,
        scratch.coarse, scratch.sorted, scratch.ids, scratch.sorted_ids, screen_rows, 0, 32, stream);
    if (status != cudaSuccess) return status;
    q4_draft_screen_initialize_cuda<<<(screen_rows + 255) / 256, 256, 0, stream>>>(nullptr, output);
    q4_draft_screen_dot_cuda<true><<<(screen_candidates + 31) / 32, 128, 0, stream>>>(
        input, weights, output, scratch.sorted_ids, nullptr);
    return cudaGetLastError();
}
