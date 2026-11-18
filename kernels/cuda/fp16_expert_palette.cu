#include "kernels/cuda/fp16_expert_palette.h"
#include "kernels/cuda/moe.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace {
__global__ void fp16_palette_projections_cuda(
    const mollm_cuda::Nvfp4Projection* jobs, int rows, int inner) {
    const auto job = jobs[blockIdx.y];
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const auto* weight = job.weight + size_t(row) * inner;
    const auto* palette = reinterpret_cast<const __half*>(job.block_scales) +
                          size_t(row / 128) * (inner / 128) * 256;
    float sum = 0;
    for (int k = lane; k < inner; k += 32)
        sum = fmaf(__half2float(palette[size_t(k / 128) * 256 + weight[k]]), job.input[k], sum);
    for (int distance = 16; distance; distance >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, distance);
    if (!lane) job.output[row] = sum;
}
}

cudaError_t mollm_cuda::launch_fp16_palette_projections(
    const Nvfp4Projection* jobs, int count, int rows, int inner, cudaStream_t stream) {
    if (count < 0 || count > 65535 || rows <= 0 || inner <= 0 ||
        rows % 128 || inner % 128 || (count && (!jobs ||
        reinterpret_cast<uintptr_t>(jobs) % alignof(Nvfp4Projection))))
        return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    fp16_palette_projections_cuda<<<dim3(rows / 8, count), 256, 0, stream>>>(jobs, rows, inner);
    return cudaGetLastError();
}
