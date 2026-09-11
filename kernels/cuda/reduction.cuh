#pragma once

// CUDA block-reduction helpers shared by GPU operator kernels.

#include <cuda_runtime.h>

namespace mollm_cuda::detail {

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        value = fmaxf(
            value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

// Keep the cross-warp tree explicit through stride 32, then finish the first
// warp with shuffles. Threads is compile-time fixed so every specialization
// has the same reduction order as the equivalent shared-memory tree.
template <int Threads>
__device__ __forceinline__ float block_reduce_sum(
    float value, float* reduction) {
    static_assert(Threads >= 64 && Threads <= 1024);
    static_assert((Threads & (Threads - 1)) == 0);
    const int thread = static_cast<int>(threadIdx.x);
    reduction[thread] = value;
    __syncthreads();
#pragma unroll
    for (int offset = Threads / 2; offset >= 64; offset >>= 1) {
        if (thread < offset)
            reduction[thread] += reduction[thread + offset];
        __syncthreads();
    }
    if (thread < 32) {
        value = reduction[thread] + reduction[thread + 32];
        value = warp_reduce_sum(value);
        if (thread == 0)
            reduction[0] = value;
    }
    __syncthreads();
    const float result = reduction[0];
    // Callers commonly reuse the same shared-memory array for consecutive
    // reductions. Ensure every thread has consumed the result before any
    // thread can enter the next reduction and overwrite reduction[0].
    __syncthreads();
    return result;
}

template <int Threads>
__device__ __forceinline__ float block_reduce_max(
    float value, float* reduction) {
    static_assert(Threads >= 64 && Threads <= 1024);
    static_assert((Threads & (Threads - 1)) == 0);
    const int thread = static_cast<int>(threadIdx.x);
    reduction[thread] = value;
    __syncthreads();
#pragma unroll
    for (int offset = Threads / 2; offset >= 64; offset >>= 1) {
        if (thread < offset)
            reduction[thread] = fmaxf(
                reduction[thread], reduction[thread + offset]);
        __syncthreads();
    }
    if (thread < 32) {
        value = fmaxf(reduction[thread], reduction[thread + 32]);
        value = warp_reduce_max(value);
        if (thread == 0)
            reduction[0] = value;
    }
    __syncthreads();
    const float result = reduction[0];
    __syncthreads();
    return result;
}

}  // namespace mollm_cuda::detail
