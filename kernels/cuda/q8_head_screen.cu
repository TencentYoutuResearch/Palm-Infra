#include "kernels/cuda/q8_head_screen.h"
#include "kernels/cuda/device_span.h"

#include <climits>
#include <cmath>

namespace {

// Radius scale factors in units of |scale| * L1(input); see the derivations
// above q8_head_radius6() and q8_head_radius4().
constexpr double q6_radius_factor = 2.04;
constexpr double q4_radius_factor = 7.51;

// Scalar Q4 packs four 4-bit codes per 16-bit packet: K / 2 = 1280 bytes per row.
constexpr size_t q4_packed_row_bytes = 320 * 4;

// Only this vocabulary/SM121 combination has matched-model performance
// evidence. Cache by current device so switching GPUs cannot reuse its policy.
// Query before launching any kernels and propagate errors without clearing them.
cudaError_t q8_head_prefetch_enabled(int rows, bool& enabled) {
    enabled = false;
    if (rows != 248320) return cudaSuccess;
    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    thread_local int cached_device = -1;
    thread_local bool supported = false;
    if (device != cached_device) {
        int major = 0;
        int minor = 0;
        status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        if (status != cudaSuccess) return status;
        status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        if (status != cudaSuccess) return status;
        supported = major == 12 && minor == 1;
        cached_device = device;
    }
    enabled = supported;
    return cudaSuccess;
}

// Q6 packing: four signed Q8 codes -> four 6-bit codes in 3 bytes.
__global__ void q8_head_pack6_cuda(const int8_t* w, uint8_t* out, size_t groups) {
    for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < groups;
         g += size_t(gridDim.x) * blockDim.x) {
        unsigned bits = 0;
        for (int j = 0; j < 4; ++j) {
            int q = max(-32, min(31, __float2int_rn(float(w[g * 4 + j]) / 4.0625f)));
            bits |= unsigned(q + 32) << (j * 6);
        }
        out[g * 3] = bits;
        out[g * 3 + 1] = bits >> 8;
        out[g * 3 + 2] = bits >> 16;
    }
}

template <bool Batched>
__global__ void q8_head_input_norm_cuda(const float* x, double* norm, int k) {
    if constexpr (Batched) {
        x += size_t(blockIdx.x) * k;
        norm += blockIdx.x;
    }
    __shared__ double sums[256];
    double v = 0;
    for (int j = threadIdx.x; j < k; j += 256) v += fabs(double(x[j]));
    sums[threadIdx.x] = v;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) *norm = isfinite(sums[0]) ? sums[0] * (1. + 1e-12) : INFINITY;
}

// One warp per row, eight rows per 256-thread block.
template <int PrefetchDistance = 0>
__global__ void q8_head_approximate6_cuda(const float* x, const uint8_t* packed,
    const float* scales, float* out, int n, int k) {
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= n) return;
    auto* words = reinterpret_cast<const unsigned*>(packed + size_t(row) * (k / 4) * 3);
    float sum = 0;
    float scale = scales[row];
    for (int j = lane; j < k / 4; j += 32) {
        if constexpr (PrefetchDistance > 0) {
            if (j + 32 * PrefetchDistance < k / 4)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(words + (j + 32 * PrefetchDistance) * 3 / 4));
        }
        int b = j * 3;
        int shift = (b % 4) * 8;
        unsigned low = words[b / 4];
        unsigned high = shift > 8 ? words[b / 4 + 1] : 0;
        unsigned bits = low >> shift;
        if (shift > 8) bits |= high << (32 - shift);
        float4 a = reinterpret_cast<const float4*>(x)[j];
        float w0 = float(int(bits & 63) - 32) * 4.0625f * scale;
        float w1 = float(int((bits >> 6) & 63) - 32) * 4.0625f * scale;
        float w2 = float(int((bits >> 12) & 63) - 32) * 4.0625f * scale;
        float w3 = float(int((bits >> 18) & 63) - 32) * 4.0625f * scale;
        sum = fmaf(a.x, w0, sum);
        sum = fmaf(a.y, w1, sum);
        sum = fmaf(a.z, w2, sum);
        sum = fmaf(a.w, w3, sum);
    }
    for (int off = 16; off; off /= 2) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (!lane) out[row] = sum;
}

// Q6 approximation for Queries contiguous K=2560 inputs (1920 packed bytes per
// row). Output is query-major: out[t * n + row].
template <int Queries>
__global__ void q8_head_approximate6_batch_cuda(const float* x, const uint8_t* packed,
    const float* scales, float* out, int n) {
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= n) return;
    const auto* words = reinterpret_cast<const unsigned*>(packed + size_t(row) * 1920);
    float sums[Queries] = {};
    float scale = scales[row];
    for (int j = lane; j < 640; j += 32) {
        if (j + 64 < 640) asm volatile("prefetch.global.L2 [%0];" :: "l"(words + (j + 64) * 3 / 4));
        int b = j * 3;
        int shift = (b % 4) * 8;
        unsigned low = words[b / 4];
        unsigned high = shift > 8 ? words[b / 4 + 1] : 0;
        unsigned bits = low >> shift;
        if (shift > 8) bits |= high << (32 - shift);
        const float w0 = float(int(bits & 63) - 32) * 4.0625f * scale;
        const float w1 = float(int((bits >> 6) & 63) - 32) * 4.0625f * scale;
        const float w2 = float(int((bits >> 12) & 63) - 32) * 4.0625f * scale;
        const float w3 = float(int((bits >> 18) & 63) - 32) * 4.0625f * scale;
#pragma unroll
        for (int t = 0; t < Queries; ++t) {
            float4 a = reinterpret_cast<const float4*>(x + size_t(t) * 2560)[j];
            sums[t] = fmaf(a.x, w0, sums[t]);
            sums[t] = fmaf(a.y, w1, sums[t]);
            sums[t] = fmaf(a.z, w2, sums[t]);
            sums[t] = fmaf(a.w, w3, sums[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < Queries; ++t) {
        float sum = sums[t];
        for (int off = 16; off; off /= 2) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (!lane) out[size_t(t) * n + row] = sum;
    }
}

// K=2560: 80 FMAs/lane, then five additions, plus weight scaling.
// Relative rounding is bounded by gamma(86). Across the two paths,
// (128 + 130) * gamma(86) < 0.00133 in units of |scale| * L1.
// Reconstruction error <= 2.03125; 2.04 bounds their sum conservatively.
// Eligible L1 <= 1e20, |scale| <= 1e10 guarantee no overflow. Even with
// flush-to-zero, weight underflow contributes < 2 * FLT_MIN * 1e20 and
// accumulation underflow < 5200 * FLT_MIN; 1e-5 exceeds both bounds.
// Nonfinite/out-of-range rows are always verified using original Q8.
__device__ double q8_head_radius6(float scale, double norm) {
    return q6_radius_factor * fabs(double(scale)) * norm + 1e-5;
}

__device__ bool q8_head_eligible(float scale, double norm) {
    return isfinite(norm) && norm <= 1e20 && isfinite(scale) && fabsf(scale) <= 1e10f;
}

// Per-block maximum of the Q6 lower bounds; best[0..255] holds partial maxima.
template <bool Batched>
__global__ void q8_head_threshold_cuda(const float* a, const float* scales, const double* norm,
    float* best, int n) {
    if constexpr (Batched) {
        a += size_t(blockIdx.y) * n;
        norm += blockIdx.y;
        best += size_t(blockIdx.y) * 257;
    }
    __shared__ float values[256];
    float v = -INFINITY;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        if (isfinite(a[i]) && q8_head_eligible(scales[i], *norm))
            v = fmaxf(v, __double2float_rd(double(a[i]) - q8_head_radius6(scales[i], *norm)));
    }
    values[threadIdx.x] = v;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s) values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + s]);
        __syncthreads();
    }
    if (!threadIdx.x) best[blockIdx.x] = values[0];
}

// Reduces best[0..255] into best[256].
template <bool Batched>
__global__ void q8_head_finish_threshold_cuda(float* best) {
    if constexpr (Batched) best += size_t(blockIdx.x) * 257;
    __shared__ float values[256];
    values[threadIdx.x] = best[threadIdx.x];
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s) values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + s]);
        __syncthreads();
    }
    if (!threadIdx.x) best[256] = values[0];
}

// Clears output to -infinity and appends every row whose upper bound reaches
// the threshold (or that is not eligible for screening) to indices.
template <bool Batched>
__global__ void q8_head_select_rows_cuda(const float* a, const float* scales, const double* norm,
    const float* best, unsigned* count, int* indices, float* output, int n) {
    if constexpr (Batched) {
        a += size_t(blockIdx.y) * n;
        norm += blockIdx.y;
        best += size_t(blockIdx.y) * 257;
        count += blockIdx.y;
        indices += size_t(blockIdx.y) * n;
        output += size_t(blockIdx.y) * n;
    }
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    output[r] = -INFINITY;
    if (!isfinite(a[r]) || !q8_head_eligible(scales[r], *norm) ||
        __double2float_ru(double(a[r]) + q8_head_radius6(scales[r], *norm)) >= *best)
        indices[atomicAdd(count, 1u)] = r;
}

// Rescores the selected rows with the original Q8 weights, one warp per row.
template <bool Batched>
__global__ void q8_head_verify_cuda(const float* x, const int8_t* w, const float* scales,
    const unsigned* count, const int* indices, float* out, const double* norm, int k, int n) {
    if constexpr (Batched) {
        x += size_t(blockIdx.y) * k;
        count += blockIdx.y;
        indices += size_t(blockIdx.y) * n;
        out += size_t(blockIdx.y) * n;
        norm += blockIdx.y;
    }
    int lane = threadIdx.x % 32;
    for (unsigned index = blockIdx.x * 4 + threadIdx.x / 32; index < *count; index += gridDim.x * 4) {
        int row = indices[index];
        float scale = scales[row];
        float sum = 0;
        if (*norm == 0 && isfinite(scale) && fabsf(scale) <= 1e10f) {
            if (!lane) out[row] = 0;
            continue;
        }
        for (int j = lane; j < k / 4; j += 32) {
            float4 a = reinterpret_cast<const float4*>(x)[j];
            char4 b = reinterpret_cast<const char4*>(w + size_t(row) * k)[j];
            sum = fmaf(a.x, float(b.x) * scale, sum);
            sum = fmaf(a.y, float(b.y) * scale, sum);
            sum = fmaf(a.z, float(b.z) * scale, sum);
            sum = fmaf(a.w, float(b.w) * scale, sum);
        }
        for (int off = 16; off; off /= 2) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (!lane) out[row] = sum;
    }
}

// Q4 packing: four signed Q8 codes -> four 4-bit codes in 2 bytes.
__global__ void q8_head_pack4_cuda(const int8_t* w, uint8_t* out, size_t groups) {
    for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < groups;
         g += size_t(gridDim.x) * blockDim.x) {
        unsigned bits = 0;
        for (int j = 0; j < 4; ++j) bits |= unsigned((int(w[g * 4 + j]) + 128) >> 4) << (4 * j);
        out[g * 2] = bits;
        out[g * 2 + 1] = bits >> 8;
    }
}

// One warp per row, eight rows per 256-thread block.
template <int PrefetchDistance = 0>
__global__ void q8_head_approximate4_cuda(const float* x, const uint8_t* packed,
    const float* scales, float* out, int n, int k) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= n) return;
    const auto* packets = reinterpret_cast<const uint16_t*>(packed + size_t(row) * (k / 2));
    float sum = 0;
    float scale = scales[row];
    for (int j = lane; j < k / 4; j += 32) {
        if constexpr (PrefetchDistance > 0) {
            if (j + 32 * PrefetchDistance < k / 4)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(packets + j + 32 * PrefetchDistance));
        }
        const unsigned bits = packets[j];
        const float4 a = reinterpret_cast<const float4*>(x)[j];
        const float w0 = float(int(bits & 15) * 16 - 120.5f) * scale;
        const float w1 = float(int((bits >> 4) & 15) * 16 - 120.5f) * scale;
        const float w2 = float(int((bits >> 8) & 15) * 16 - 120.5f) * scale;
        const float w3 = float(int((bits >> 12) & 15) * 16 - 120.5f) * scale;
        sum = fmaf(a.x, w0, sum);
        sum = fmaf(a.y, w1, sum);
        sum = fmaf(a.z, w2, sum);
        sum = fmaf(a.w, w3, sum);
    }
    for (int off = 16; off; off /= 2) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (!lane) out[row] = sum;
}

// Q4 retains 80 FP32 FMAs/lane and the same five-add reduction. Reconstruction
// uses exact half-integers, so all signed Q8 codes have error <= 7.5.
// Including weight scaling and dot rounding: 7.5 + 248.5 * gamma(86) < 7.51.
// Keep the inflated norm, directed interval endpoints and underflow allowance.
__device__ double q8_head_radius4(float scale, double norm) {
    return q4_radius_factor * fabs(double(scale)) * norm + 1e-5;
}

// Preserve the original max-lower-bound anchor and add the largest coarse
// center. Both become exact original-Q8 row scores. Taking their maximum
// cannot lower the original threshold or increase its retained candidate set.
// indices[0..511] are stream-ordered temporary IDs; requires rows >= 512.
__global__ void q8_head_anchor_threshold4_cuda(const float* a, const float* scales,
    const double* norm, float* best, int* best_ids, int n) {
    __shared__ float lower_values[256];
    __shared__ float centers[256];
    __shared__ int lower_rows[256];
    __shared__ int center_rows[256];
    int t = threadIdx.x;
    float lower = -INFINITY;
    float center = -INFINITY;
    int lower_row = INT_MAX;
    int center_row = INT_MAX;
    for (int i = blockIdx.x * 256 + t; i < n; i += gridDim.x * 256) {
        if (isfinite(a[i]) && q8_head_eligible(scales[i], *norm)) {
            float v = __double2float_rd(double(a[i]) - q8_head_radius4(scales[i], *norm));
            if (v > lower || (v == lower && i < lower_row)) {
                lower = v;
                lower_row = i;
            }
            v = a[i];
            if (v > center || (v == center && i < center_row)) {
                center = v;
                center_row = i;
            }
        }
    }
    lower_values[t] = lower;
    centers[t] = center;
    lower_rows[t] = lower_row;
    center_rows[t] = center_row;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) {
            float v = lower_values[t + stride];
            int id = lower_rows[t + stride];
            if (v > lower_values[t] || (v == lower_values[t] && id < lower_rows[t])) {
                lower_values[t] = v;
                lower_rows[t] = id;
            }
            v = centers[t + stride];
            id = center_rows[t + stride];
            if (v > centers[t] || (v == centers[t] && id < center_rows[t])) {
                centers[t] = v;
                center_rows[t] = id;
            }
        }
        __syncthreads();
    }
    if (!t) {
        best[blockIdx.x] = lower_values[0];
        best_ids[blockIdx.x] = lower_rows[0];
        best_ids[256 + blockIdx.x] = center_rows[0];
    }
}

// Reduces the per-block anchors, rescores both anchor rows exactly with the
// original Q8 weights (warp 0: lower-bound anchor, warp 1: center anchor) and
// stores the final threshold in best[256].
__global__ void q8_head_anchor_finish_cuda(const float* x, const int8_t* w, const float* scales,
    const float* approximate, float* best, const int* best_ids, int n) {
    __shared__ float lower_values[256];
    __shared__ float centers[256];
    __shared__ float exact[2];
    __shared__ int lower_rows[256];
    __shared__ int center_rows[256];
    int t = threadIdx.x;
    lower_values[t] = best[t];
    lower_rows[t] = best_ids[t];
    int center_row = best_ids[256 + t];
    center_rows[t] = center_row;
    centers[t] = center_row >= 0 && center_row < n ? approximate[center_row] : -INFINITY;
    if (t < 2) exact[t] = -INFINITY;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) {
            float v = lower_values[t + stride];
            int id = lower_rows[t + stride];
            if (v > lower_values[t] || (v == lower_values[t] && id < lower_rows[t])) {
                lower_values[t] = v;
                lower_rows[t] = id;
            }
            v = centers[t + stride];
            id = center_rows[t + stride];
            if (v > centers[t] || (v == centers[t] && id < center_rows[t])) {
                centers[t] = v;
                center_rows[t] = id;
            }
        }
        __syncthreads();
    }
    int warp = t / 32;
    int lane = t % 32;
    int row = warp == 0 ? lower_rows[0] : center_rows[0];
    if (t < 64 && row >= 0 && row < n && (warp == 0 || row != lower_rows[0])) {
        float sum = 0;
        float scale = scales[row];
        for (int j = lane; j < 640; j += 32) {
            float4 a = reinterpret_cast<const float4*>(x)[j];
            char4 q = reinterpret_cast<const char4*>(w + size_t(row) * 2560)[j];
            sum = fmaf(a.x, float(q.x) * scale, sum);
            sum = fmaf(a.y, float(q.y) * scale, sum);
            sum = fmaf(a.z, float(q.z) * scale, sum);
            sum = fmaf(a.w, float(q.w) * scale, sum);
        }
        for (int offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (!lane) exact[warp] = sum;
    }
    __syncthreads();
    if (!t) best[256] = fmaxf(lower_values[0], fmaxf(exact[0], exact[1]));
}

// Same selection as q8_head_select_rows_cuda, using the Q4 radius.
__global__ void q8_head_select_rows4_cuda(const float* a, const float* scales, const double* norm,
    const float* best, unsigned* count, int* indices, float* output, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    output[row] = -INFINITY;
    if (!isfinite(a[row]) || !q8_head_eligible(scales[row], *norm) ||
        __double2float_ru(double(a[row]) + q8_head_radius4(scales[row], *norm)) >= *best)
        indices[atomicAdd(count, 1u)] = row;
}

// q8_head_threshold_cuda() strides its int row index by 65536; keep the final
// step in range.
constexpr int max_screen_rows = INT32_MAX - 65536;

}  // namespace

namespace mollm_cuda {

cudaError_t pack_q8_head_screen(const int8_t* weights, uint8_t* packed, int rows, cudaStream_t stream) {
    if (!weights || !packed || rows <= 0 || rows > max_screen_rows || uintptr_t(packed) % 4)
        return cudaErrorInvalidValue;
    q8_head_pack6_cuda<<<256, 256, 0, stream>>>(weights, packed, size_t(rows) * 2560 / 4);
    return cudaGetLastError();
}

cudaError_t launch_q8_head_screen(const float* input, const int8_t* weights,
    const uint8_t* packed, const float* scales, float* output, int rows,
    Q8HeadScreenScratch s, cudaStream_t stream) {
    if (!input || !weights || !packed || !scales || !output || rows <= 0 || rows > max_screen_rows ||
        uintptr_t(input) % 16 || uintptr_t(weights) % 4 || uintptr_t(packed) % 4 ||
        !s.approximate || !s.lower || !s.norm || !s.count || !s.indices)
        return cudaErrorInvalidValue;
    bool prefetch = false;
    cudaError_t status = q8_head_prefetch_enabled(rows, prefetch);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(s.count, 0, sizeof(unsigned), stream);
    if (status != cudaSuccess) return status;
    q8_head_input_norm_cuda<false><<<1, 256, 0, stream>>>(input, s.norm, 2560);
    if (prefetch) {
        q8_head_approximate6_cuda<2><<<(rows + 7) / 8, 256, 0, stream>>>(
            input, packed, scales, s.approximate, rows, 2560);
    } else {
        q8_head_approximate6_cuda<0><<<(rows + 7) / 8, 256, 0, stream>>>(
            input, packed, scales, s.approximate, rows, 2560);
    }
    q8_head_threshold_cuda<false><<<256, 256, 0, stream>>>(s.approximate, scales, s.norm, s.lower, rows);
    q8_head_finish_threshold_cuda<false><<<1, 256, 0, stream>>>(s.lower);
    q8_head_select_rows_cuda<false><<<(rows + 255) / 256, 256, 0, stream>>>(
        s.approximate, scales, s.norm, s.lower + 256, s.count, s.indices, output, rows);
    q8_head_verify_cuda<false><<<128, 128, 0, stream>>>(
        input, weights, scales, s.count, s.indices, output, s.norm, 2560, rows);
    return cudaGetLastError();
}

cudaError_t pack_q8_head_screen_q4(const int8_t* weights, uint8_t* packed,
    int rows, cudaStream_t stream) {
    if (rows < 512 || rows > max_screen_rows || uintptr_t(weights) % 4 || uintptr_t(packed) % 4)
        return cudaErrorInvalidValue;
    const mollm_cuda::DeviceSpan spans[] = {
        {weights, size_t(rows) * 2560},
        {packed, size_t(rows) * q4_packed_row_bytes},
    };
    if (!mollm_cuda::device_spans_disjoint(spans, 2)) return cudaErrorInvalidValue;
    q8_head_pack4_cuda<<<256, 256, 0, stream>>>(weights, packed, size_t(rows) * 2560 / 4);
    return cudaGetLastError();
}

cudaError_t launch_q8_head_screen_q4(const float* input, const int8_t* weights,
    const uint8_t* packed, const float* scales, float* output, int rows,
    Q8HeadScreenScratch s, cudaStream_t stream) {
    // Anchor IDs occupy indices[0..511].
    if (rows < 512 || rows > max_screen_rows || uintptr_t(input) % 16 || uintptr_t(weights) % 4 ||
        uintptr_t(packed) % 4 || uintptr_t(scales) % 4 || uintptr_t(output) % 4 ||
        uintptr_t(s.approximate) % 4 || uintptr_t(s.lower) % 4 || uintptr_t(s.norm) % 8 ||
        uintptr_t(s.count) % 4 || uintptr_t(s.indices) % 4)
        return cudaErrorInvalidValue;
    const size_t bytes = size_t(rows) * 4;
    const mollm_cuda::DeviceSpan spans[] = {
        {input, 2560 * sizeof(float)},
        {weights, size_t(rows) * 2560},
        {packed, size_t(rows) * q4_packed_row_bytes},
        {scales, bytes},
        {output, bytes},
        {s.approximate, bytes},
        {s.lower, 257 * sizeof(float)},
        {s.norm, sizeof(double)},
        {s.count, sizeof(unsigned)},
        {s.indices, bytes},
    };
    if (!mollm_cuda::device_spans_disjoint(spans, 10)) return cudaErrorInvalidValue;
    bool prefetch = false;
    cudaError_t status = q8_head_prefetch_enabled(rows, prefetch);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(s.count, 0, sizeof(unsigned), stream);
    if (status != cudaSuccess) return status;
    q8_head_input_norm_cuda<false><<<1, 256, 0, stream>>>(input, s.norm, 2560);
    if (prefetch) {
        q8_head_approximate4_cuda<2><<<(rows + 7) / 8, 256, 0, stream>>>(
            input, packed, scales, s.approximate, rows, 2560);
    } else {
        q8_head_approximate4_cuda<0><<<(rows + 7) / 8, 256, 0, stream>>>(
            input, packed, scales, s.approximate, rows, 2560);
    }
    q8_head_anchor_threshold4_cuda<<<256, 256, 0, stream>>>(
        s.approximate, scales, s.norm, s.lower, s.indices, rows);
    q8_head_anchor_finish_cuda<<<1, 256, 0, stream>>>(
        input, weights, scales, s.approximate, s.lower, s.indices, rows);
    q8_head_select_rows4_cuda<<<(rows + 255) / 256, 256, 0, stream>>>(
        s.approximate, scales, s.norm, s.lower + 256, s.count, s.indices, output, rows);
    q8_head_verify_cuda<false><<<128, 128, 0, stream>>>(
        input, weights, scales, s.count, s.indices, output, s.norm, 2560, rows);
    return cudaGetLastError();
}

cudaError_t launch_q8_head_screen_batch3(const float* input, const int8_t* weights,
    const uint8_t* packed, const float* scales, float* output, int rows,
    Q8HeadScreenBatch3Scratch s, cudaStream_t stream) {
    if (!input || !weights || !packed || !scales || !output || rows <= 0 || rows > max_screen_rows ||
        uintptr_t(input) % 16 || uintptr_t(weights) % 4 || uintptr_t(packed) % 4 ||
        !s.approximate || !s.lower || !s.norm || !s.count || !s.indices)
        return cudaErrorInvalidValue;
    cudaError_t status = cudaMemsetAsync(s.count, 0, 3 * sizeof(unsigned), stream);
    if (status != cudaSuccess) return status;
    q8_head_input_norm_cuda<true><<<3, 256, 0, stream>>>(input, s.norm, 2560);
    q8_head_approximate6_batch_cuda<3><<<(rows + 7) / 8, 256, 0, stream>>>(
        input, packed, scales, s.approximate, rows);
    q8_head_threshold_cuda<true><<<dim3(256, 3), 256, 0, stream>>>(
        s.approximate, scales, s.norm, s.lower, rows);
    q8_head_finish_threshold_cuda<true><<<3, 256, 0, stream>>>(s.lower);
    q8_head_select_rows_cuda<true><<<dim3((rows + 255) / 256, 3), 256, 0, stream>>>(
        s.approximate, scales, s.norm, s.lower + 256, s.count, s.indices, output, rows);
    q8_head_verify_cuda<true><<<dim3(128, 3), 128, 0, stream>>>(
        input, weights, scales, s.count, s.indices, output, s.norm, 2560, rows);
    return cudaGetLastError();
}

cudaError_t launch_q8_head_screen_batch(const float* input, const int8_t* weights,
    const uint8_t* packed, const float* scales, float* output, int queries, int rows,
    Q8HeadScreenBatchScratch s, cudaStream_t stream) {
    if ((queries < 4 || queries > 8) || rows <= 0 || rows > max_screen_rows ||
        !input || !weights || !packed || !scales || !output ||
        uintptr_t(input) % 16 || uintptr_t(weights) % 4 || uintptr_t(packed) % 4 ||
        uintptr_t(scales) % 4 || uintptr_t(output) % 4 ||
        !s.approximate || !s.lower || !s.norm || !s.count || !s.indices ||
        uintptr_t(s.approximate) % 4 || uintptr_t(s.lower) % 4 || uintptr_t(s.norm) % 8 ||
        uintptr_t(s.count) % 4 || uintptr_t(s.indices) % 4 ||
        s.approximate_capacity < size_t(queries) * rows || s.lower_capacity < size_t(queries) * 257 ||
        s.norm_capacity < size_t(queries) || s.count_capacity < size_t(queries) ||
        s.indices_capacity < size_t(queries) * rows)
        return cudaErrorInvalidValue;
    cudaError_t status = cudaMemsetAsync(s.count, 0, size_t(queries) * sizeof(unsigned), stream);
    if (status != cudaSuccess) return status;
    q8_head_input_norm_cuda<true><<<queries, 256, 0, stream>>>(input, s.norm, 2560);
    const int blocks = (rows + 7) / 8;
    if (queries == 4)
        q8_head_approximate6_batch_cuda<4><<<blocks, 256, 0, stream>>>(
            input, packed, scales, s.approximate, rows);
    else if (queries == 5)
        q8_head_approximate6_batch_cuda<5><<<blocks, 256, 0, stream>>>(
            input, packed, scales, s.approximate, rows);
    else if (queries == 6)
        q8_head_approximate6_batch_cuda<6><<<blocks, 256, 0, stream>>>(
            input, packed, scales, s.approximate, rows);
    else if (queries == 7)
        q8_head_approximate6_batch_cuda<7><<<blocks, 256, 0, stream>>>(
            input, packed, scales, s.approximate, rows);
    else
        q8_head_approximate6_batch_cuda<8><<<blocks, 256, 0, stream>>>(
            input, packed, scales, s.approximate, rows);
    q8_head_threshold_cuda<true><<<dim3(256, queries), 256, 0, stream>>>(
        s.approximate, scales, s.norm, s.lower, rows);
    q8_head_finish_threshold_cuda<true><<<queries, 256, 0, stream>>>(s.lower);
    q8_head_select_rows_cuda<true><<<dim3((rows + 255) / 256, queries), 256, 0, stream>>>(
        s.approximate, scales, s.norm, s.lower + 256, s.count, s.indices, output, rows);
    q8_head_verify_cuda<true><<<dim3(128, queries), 128, 0, stream>>>(
        input, weights, scales, s.count, s.indices, output, s.norm, 2560, rows);
    return cudaGetLastError();
}

}  // namespace mollm_cuda
