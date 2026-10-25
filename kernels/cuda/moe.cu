#include "kernels/cuda/moe.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <cub/device/device_radix_sort.cuh>
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif
#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif

namespace {

__device__ __forceinline__ size_t nvfp4_block_scale_offset(
    const mollm_cuda::Nvfp4Projection& job,int row,int groups,int group) {
    return job.gate_scales_kmajor?
        (size_t(group/4)*1280+row)*4+group%4:size_t(row)*groups+group;
}

__device__ void nvfp4_transpose_gate_scales(const uint8_t* source,uint8_t* target) {
    __shared__ uint32_t tile[32][33];
    const int x=threadIdx.x, y=threadIdx.y;
    const int row=blockIdx.x*32+y, group=blockIdx.y*32+x;
    const auto* src=reinterpret_cast<const uint32_t*>(source);
    auto* dst=reinterpret_cast<uint32_t*>(target);
    for(int j=0;j<32;j+=8)
        if(group<40)tile[y+j][x]=src[size_t(row+j)*40+group];
    __syncthreads();
    const int out_row=blockIdx.x*32+x, out_group=blockIdx.y*32+y;
    for(int j=0;j<32;j+=8)
        if(out_group+j<40)dst[size_t(out_group+j)*1280+out_row]=tile[x][y+j];
}

__global__ void nvfp4_gate_scale_view_cuda(
    const mollm_cuda::Nvfp4Projection* source,
    mollm_cuda::Nvfp4Projection* target, uint8_t* scales) {
    const int expert=blockIdx.z;
    auto* dst=scales+size_t(expert)*204800;
    nvfp4_transpose_gate_scales(source[expert].block_scales,dst);
    if(!blockIdx.x&&!blockIdx.y&&!threadIdx.x&&!threadIdx.y) {
        target[expert].block_scales=dst;
        target[expert].gate_scales_kmajor=true;
    }
}
__global__ void repack_nvfp4_gate_scales_cuda(const uint8_t* source,uint8_t* target) {
    nvfp4_transpose_gate_scales(source,target);
}

__device__ float bf16(float value);
template<bool Packed> __device__ __forceinline__ void nvfp4_store_output(
    float* output,int row,float value) {
    if constexpr(Packed)reinterpret_cast<uint16_t*>(output)[row]=
        static_cast<uint16_t>(__float_as_uint(bf16(value))>>16);
    else output[row]=value;
}

__global__ void fp16_expert_projections_cuda(
    const mollm_cuda::Nvfp4Projection* jobs, int rows, int inner) {
    const auto job = jobs[blockIdx.y];
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const auto* weight = reinterpret_cast<const __half*>(job.weight) +
                         static_cast<size_t>(row) * inner;
    float sum = 0;
    for (int k = lane; k < inner; k += 32)
        sum = fmaf(__half2float(weight[k]), job.input[k], sum);
    for (int distance = 16; distance; distance >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, distance);
    if (lane == 0) job.output[row] = sum;
}

template<bool Selective = false, bool CanonicalWeights = false, bool Prequantized = false, bool KmajorCodes = false>
__global__ void nvfp4_native_mma_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int rows, int inner, int split_row, float* partial,
    const mollm_cuda::Nvfp4MmaBatch* batches = nullptr, int minimum = 1,
    const uint8_t* prepared = nullptr, const uint8_t* prepared_scales = nullptr) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    static_assert(!KmajorCodes || CanonicalWeights);
    __shared__ uint32_t shared_packed[320];
    __shared__ uint32_t shared_scales[40];
    if constexpr (Selective) if (batches[blockIdx.y].size >= minimum) return;
    const auto job = jobs[blockIdx.y];
    const int cta_row = blockIdx.x * (blockDim.x / 32) * 16;
    const float global = split_row && cta_row >= split_row ? job.input_global_second : job.input_global;
    auto* packed = reinterpret_cast<uint8_t*>(shared_packed);
    auto* scales = reinterpret_cast<uint8_t*>(shared_scales);
    if constexpr (!Prequantized) {
    for (int ordinal = threadIdx.x; ; ordinal += blockDim.x) {
        const int group = (blockIdx.z + (ordinal / 4) * gridDim.z) * 4 + ordinal % 4;
        if (group >= inner / 16) break;
        float maximum = 0;
        for (int i = 0; i < 16; ++i)
            maximum = fmaxf(maximum, fabsf(job.input[group * 16 + i]));
        const uint8_t code = __nv_cvt_float_to_fp8(__fdiv_rn(maximum, global * 6.f),
            __NV_SATFINITE, __NV_E4M3);
        scales[group] = code;
        const float block_scale = __half2float(static_cast<__half>(
            __nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
        const float inverse = block_scale > 0 ? __fdiv_rn(1.f, block_scale * global) : 0;
        for (int i = 0; i < 8; ++i) {
            const int offset = group * 16 + i * 2;
            packed[group * 8 + i] = __nv_cvt_float2_to_fp4x2(
                make_float2(job.input[offset] * inverse, job.input[offset + 1] * inverse),
                __NV_E2M1, cudaRoundNearest);
        }
    }
    __syncthreads();
    }
    const size_t prepared_job = blockIdx.y +
        (split_row && cta_row >= split_row ? gridDim.y : 0);
    const uint8_t* input_packed = Prequantized ? prepared + prepared_job * (inner / 2) : packed;
    const uint8_t* input_scales = Prequantized ? prepared_scales + prepared_job * (inner / 16) : scales;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int base_row = cta_row + warp * 16;
    if (base_row >= rows) return;
    const int row = base_row + lane / 4, thread = lane % 4;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int k = blockIdx.z * 64; k < inner; k += gridDim.z * 64) {
        const size_t tile = static_cast<size_t>(base_row / 16) * (inner / 64) + k / 64;
        const auto* weight = reinterpret_cast<const uint32_t*>(job.weight) + tile * 128;
        const int col = k + thread * 8;
        uint32_t a0, a1, a2, a3, sa;
        if constexpr (CanonicalWeights) {
            const auto load = [&](int r, int c) {
                const size_t offset = KmajorCodes
                    ? size_t(c / 64) * 1280 * 32 + (job.pair_layout
                        ? size_t(r / 4) * 128 + (c % 64 / 16) * 32 + (r % 4) * 8 + (c % 16) / 2
                        : size_t(r) * 32 + (c % 64) / 2)
                    : job.pair_layout
                    ? (static_cast<size_t>(r / 4) * (inner / 16) + c / 16) * 32 +
                        (r % 4) * 8 + (c % 16) / 2
                    : static_cast<size_t>(r) * (inner / 2) + c / 2;
                return *reinterpret_cast<const uint32_t*>(job.weight + offset);
            };
            a0 = load(row, col); a1 = load(row + 8, col);
            a2 = load(row, col + 32); a3 = load(row + 8, col + 32);
            sa = *reinterpret_cast<const uint32_t*>(job.block_scales +
                nvfp4_block_scale_offset(job,row+(thread&1)*8,inner/16,k/16));
        } else {
            a0 = weight[lane]; a1 = weight[32 + lane];
            a2 = weight[64 + lane]; a3 = weight[96 + lane];
            sa = *reinterpret_cast<const uint32_t*>(job.block_scales + tile * 64 +
                (lane / 4) * 8 + (thread & 1) * 4);
        }
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(input_packed + col / 2);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(input_packed + (col + 32) / 2);
        const uint32_t sb = *reinterpret_cast<const uint32_t*>(input_scales + k / 16);
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10,{0,0}, %11,{0,0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    }
    if (thread == 0) {
        if (gridDim.z > 1) {
            float* destination = partial + (static_cast<size_t>(blockIdx.z) * gridDim.y + blockIdx.y) * rows;
            if (row < rows) destination[row] = d0;
            if (row + 8 < rows) destination[row + 8] = d2;
        } else {
            if (row < rows) job.output[row] = ((d0 * global) * job.row_scales[row]) * job.output_scale;
            if (row + 8 < rows) job.output[row + 8] = ((d2 * global) * job.row_scales[row + 8]) * job.output_scale;
        }
    }
#else
    asm volatile("trap;");
#endif
}

template<bool Prequantized = false, bool CanonicalWeights = false, bool Compact = false, bool Persistent = false, bool OutputBf16 = false, bool KmajorCodes = false>
__global__ void nvfp4_native_batched8_mma_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int rows, int inner, int split_row, const mollm_cuda::Nvfp4MmaBatch* batches,
    const uint8_t* quantized = nullptr, const uint8_t* quantized_scales = nullptr, int count = 0, int minimum = 1,
    const int* compact_count = nullptr, int plan_offset = 0) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    __shared__ uint32_t shared_packed[Prequantized ? 1 : 8][320];
    __shared__ uint32_t shared_scales[Prequantized ? 1 : 8][40];
    const int plan_count = Compact ? *compact_count : gridDim.y;
    for (int plan_index = blockIdx.y + (Persistent ? plan_offset : 0);
         plan_index < plan_count; plan_index += gridDim.y) {
    const int first_job = batches ? batches[plan_index].first : plan_index * 8;
    if (first_job < 0) return;
    const int batch_size = batches ? batches[plan_index].size : 8;
    if (batch_size < minimum) return;
    const auto job = jobs[first_job];
    const int cta_row = blockIdx.x * (blockDim.x / 32) * 16;
    auto* packed = reinterpret_cast<uint8_t*>(shared_packed);
    auto* scales = reinterpret_cast<uint8_t*>(shared_scales);
    if constexpr (!Prequantized) {
    for (int ordinal = threadIdx.x; ; ordinal += blockDim.x) {
        const int column = ordinal / (inner / 16);
        const int group = ordinal % (inner / 16);
        if (column >= 8) break;
        auto* column_packed = packed + column * 1280;
        auto* column_scales = scales + column * 160;
        if (column >= batch_size) {
            column_scales[group] = 0;
            for (int i = 0; i < 8; ++i) column_packed[group * 8 + i] = 0;
            continue;
        }
        const auto input_job = jobs[first_job + column];
        const float input_global = split_row && cta_row >= split_row ? input_job.input_global_second : input_job.input_global;
        float maximum = 0;
        for (int i = 0; i < 16; ++i)
            maximum = fmaxf(maximum, fabsf(input_job.input[group * 16 + i]));
        const uint8_t code = __nv_cvt_float_to_fp8(__fdiv_rn(maximum, input_global * 6.f),
            __NV_SATFINITE, __NV_E4M3);
        column_scales[group] = code;
        const float block_scale = __half2float(static_cast<__half>(
            __nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
        const float inverse = block_scale > 0 ? __fdiv_rn(1.f, block_scale * input_global) : 0;
        for (int i = 0; i < 8; ++i) {
            const int offset = group * 16 + i * 2;
            column_packed[group * 8 + i] = __nv_cvt_float2_to_fp4x2(
                make_float2(input_job.input[offset] * inverse, input_job.input[offset + 1] * inverse),
                __NV_E2M1, cudaRoundNearest);
        }
    }
    __syncthreads();
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int base_row = cta_row + warp * 16;
    if constexpr (!Persistent) { if (base_row >= rows) return; }
    if (base_row < rows) {
    const int row = base_row + lane / 4, thread = lane % 4;
    // Canonical K64 fragment addresses advance by a fixed byte increment.
    // Hoist row/pair-layout decoding without changing the MMA sequence.
    const uint8_t* canonical_codes = nullptr;
    const uint8_t* canonical_scales = nullptr;
    int canonical_k_step = 0, canonical_half_step = 0;
    if constexpr (CanonicalWeights) {
        const int col = thread * 8;
        const size_t offset = job.pair_layout
            ? static_cast<size_t>(row / 4) * (KmajorCodes ? 128 : (inner / 16) * 32) +
                (row % 4) * 8 + (col / 16) * 32 + (col % 16) / 2
            : static_cast<size_t>(row) * (KmajorCodes ? 32 : inner / 2) + col / 2;
        canonical_codes = job.weight + offset;
        canonical_scales = job.block_scales +
            nvfp4_block_scale_offset(job,row+(thread&1)*8,inner/16,0);
        canonical_k_step = KmajorCodes ? 1280 * 32 : job.pair_layout ? 128 : 32;
        canonical_half_step = job.pair_layout ? 64 : 16;
    }
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int k = 0; k < inner; k += 64) {
        const size_t tile = static_cast<size_t>(base_row / 16) * (inner / 64) + k / 64;
        const auto* weight = reinterpret_cast<const uint32_t*>(job.weight) + tile * 128;
        const int col = k + thread * 8;
        uint32_t a0, a1, a2, a3, sa;
        if constexpr (CanonicalWeights) {
            const uint8_t* codes = canonical_codes +
                static_cast<size_t>(k / 64) * canonical_k_step;
            a0 = *reinterpret_cast<const uint32_t*>(codes);
            a1 = *reinterpret_cast<const uint32_t*>(codes + (KmajorCodes ? 256 : 4 * inner));
            a2 = *reinterpret_cast<const uint32_t*>(codes + canonical_half_step);
            a3 = *reinterpret_cast<const uint32_t*>(codes + (KmajorCodes ? 256 : 4 * inner) + canonical_half_step);
            sa = *reinterpret_cast<const uint32_t*>(canonical_scales +
                (job.gate_scales_kmajor?size_t(k/64)*1280*4:size_t(k/16)));
        } else {
            a0 = weight[lane]; a1 = weight[32 + lane];
            a2 = weight[64 + lane]; a3 = weight[96 + lane];
            sa = *reinterpret_cast<const uint32_t*>(job.block_scales + tile * 64 +
                (lane / 4) * 8 + (thread & 1) * 4);
        }
        const int input_column = lane / 4;
        const int quantized_job = first_job + min(input_column, batch_size - 1) +
            (split_row && cta_row >= split_row ? count : 0);
        const uint8_t* column_packed = Prequantized ? quantized + static_cast<size_t>(quantized_job) * (inner / 2) : packed + input_column * 1280;
        const uint8_t* column_scales = Prequantized ? quantized_scales + static_cast<size_t>(quantized_job) * (inner / 16) : scales + input_column * 160;
        const uint32_t b0 = Prequantized && input_column >= batch_size ? 0 : *reinterpret_cast<const uint32_t*>(column_packed + col / 2);
        const uint32_t b1 = Prequantized && input_column >= batch_size ? 0 : *reinterpret_cast<const uint32_t*>(column_packed + (col + 32) / 2);
        const uint32_t sb = Prequantized && input_column >= batch_size ? 0 : *reinterpret_cast<const uint32_t*>(column_scales + k / 16);
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10,{0,0}, %11,{0,0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    }
    const int column = thread * 2;
    const auto first = jobs[first_job + min(column, batch_size - 1)];
    const auto second = jobs[first_job + min(column + 1, batch_size - 1)];
    const float g0 = split_row && cta_row >= split_row ? first.input_global_second : first.input_global;
    const float g1 = split_row && cta_row >= split_row ? second.input_global_second : second.input_global;
    if (row < rows) {
        if (column < batch_size) nvfp4_store_output<OutputBf16>(first.output,row,((d0 * g0) * job.row_scales[row]) * first.output_scale);
        if (column + 1 < batch_size) nvfp4_store_output<OutputBf16>(second.output,row,((d1 * g1) * job.row_scales[row]) * second.output_scale);
    }
    if (row + 8 < rows) {
        if (column < batch_size) nvfp4_store_output<OutputBf16>(first.output,row+8,((d2 * g0) * job.row_scales[row + 8]) * first.output_scale);
        if (column + 1 < batch_size) nvfp4_store_output<OutputBf16>(second.output,row+8,((d3 * g1) * job.row_scales[row + 8]) * second.output_scale);
    }
    }
    // Tail warps must participate before another iteration can overwrite the
    // shared quantization buffers; they cannot return in compact mode.
    if constexpr (Persistent && !Prequantized) __syncthreads();
    if constexpr (!Persistent) break;
    }
#else
    asm volatile("trap;");
#endif
}

template<int Columns = 16, bool CanonicalWeights = false, bool Compact = false, bool Persistent = false, bool KmajorCodes = false>
__global__ void nvfp4_prepared_batched_mma_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int rows, int inner, int split_row, const mollm_cuda::Nvfp4MmaBatch* batches,
    const uint8_t* packed, const uint8_t* scales, int count, int minimum,
    const int* compact_count = nullptr, int plan_offset = 0) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int cta_row = blockIdx.x * (blockDim.x / 32) * 16;
    const int base_row = cta_row + warp * 16;
    // No shared-buffer writes/barriers: inactive tail warps can always return.
    if (base_row >= rows) return;
    const int plan_count = Compact ? *compact_count : gridDim.y;
    for (int index = blockIdx.y + (Persistent ? plan_offset : 0);
         index < plan_count; index += gridDim.y) {
        const auto batch = batches[index];
        if (batch.first < 0 || batch.size < minimum) return;
        const int first_job = batch.first, batch_size = batch.size;
        const auto job = jobs[first_job];
        const int row = base_row + lane / 4, thread = lane % 4;
        // Canonical layouts have constant byte increments between K64 tiles.
        // Resolve the row/pair layout once instead of reconstructing four
        // fragment addresses on each MMA iteration. Both layouts advance
        // four*K bytes between rows r and r+8.
        const uint8_t* canonical_codes = nullptr;
        const uint8_t* canonical_scales = nullptr;
        int canonical_k_step = 0, canonical_half_step = 0;
        if constexpr (CanonicalWeights) {
            const int col = thread * 8;
            const size_t offset = job.pair_layout
                ? static_cast<size_t>(row / 4) * (KmajorCodes ? 128 : (inner / 16) * 32) +
                    (row % 4) * 8 + (col / 16) * 32 + (col % 16) / 2
                : static_cast<size_t>(row) * (KmajorCodes ? 32 : inner / 2) + col / 2;
            canonical_codes = job.weight + offset;
            canonical_scales = job.block_scales +
                nvfp4_block_scale_offset(job,row+(thread&1)*8,inner/16,0);
            canonical_k_step = KmajorCodes ? 1280 * 32 : job.pair_layout ? 128 : 32;
            canonical_half_step = job.pair_layout ? 64 : 16;
        }
        float accum[Columns / 8][4] = {};
        for (int k = 0; k < inner; k += 64) {
            const size_t tile = static_cast<size_t>(base_row / 16) * (inner / 64) + k / 64;
            const int col = k + thread * 8;
            uint32_t a0, a1, a2, a3, sa;
            if constexpr (CanonicalWeights) {
                const uint8_t* codes = canonical_codes +
                    static_cast<size_t>(k / 64) * canonical_k_step;
                a0 = *reinterpret_cast<const uint32_t*>(codes);
                a1 = *reinterpret_cast<const uint32_t*>(codes + (KmajorCodes ? 256 : 4 * inner));
                a2 = *reinterpret_cast<const uint32_t*>(codes + canonical_half_step);
                a3 = *reinterpret_cast<const uint32_t*>(codes + (KmajorCodes ? 256 : 4 * inner) + canonical_half_step);
                sa = *reinterpret_cast<const uint32_t*>(canonical_scales +
                    (job.gate_scales_kmajor?size_t(k/64)*1280*4:size_t(k/16)));
            } else {
                const auto* weight = reinterpret_cast<const uint32_t*>(job.weight) + tile * 128;
                a0 = weight[lane]; a1 = weight[32 + lane];
                a2 = weight[64 + lane]; a3 = weight[96 + lane];
                sa = *reinterpret_cast<const uint32_t*>(job.block_scales + tile * 64 +
                    (lane / 4) * 8 + (thread & 1) * 4);
            }
#pragma unroll
            for (int group = 0; group < Columns / 8; ++group) {
                if (group * 8 >= batch_size) continue;
                const int input_column = group * 8 + lane / 4;
                const int source = first_job + min(input_column, batch_size - 1) +
                    (split_row && cta_row >= split_row ? count : 0);
                const auto* column_packed = packed + static_cast<size_t>(source) * (inner / 2);
                const auto* column_scales = scales + static_cast<size_t>(source) * (inner / 16);
                const uint32_t b0 = input_column >= batch_size ? 0 :
                    *reinterpret_cast<const uint32_t*>(column_packed + col / 2);
                const uint32_t b1 = input_column >= batch_size ? 0 :
                    *reinterpret_cast<const uint32_t*>(column_packed + (col + 32) / 2);
                const uint32_t sb = input_column >= batch_size ? 0 :
                    *reinterpret_cast<const uint32_t*>(column_scales + k / 16);
                asm volatile(
                    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10,{0,0}, %11,{0,0};"
                    : "+f"(accum[group][0]), "+f"(accum[group][1]), "+f"(accum[group][2]), "+f"(accum[group][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
            }
        }
#pragma unroll
        for (int group = 0; group < Columns / 8; ++group) {
            const int column = group * 8 + thread * 2;
            const auto first = jobs[first_job + min(column, batch_size - 1)];
            const auto second = jobs[first_job + min(column + 1, batch_size - 1)];
            const float g0 = split_row && cta_row >= split_row ? first.input_global_second : first.input_global;
            const float g1 = split_row && cta_row >= split_row ? second.input_global_second : second.input_global;
            if (column < batch_size) {
                first.output[row] = ((accum[group][0] * g0) * job.row_scales[row]) * first.output_scale;
                first.output[row + 8] = ((accum[group][2] * g0) * job.row_scales[row + 8]) * first.output_scale;
            }
            if (column + 1 < batch_size) {
                second.output[row] = ((accum[group][1] * g1) * job.row_scales[row]) * second.output_scale;
                second.output[row + 8] = ((accum[group][3] * g1) * job.row_scales[row + 8]) * second.output_scale;
            }
        }
        if constexpr (!Persistent) break;
    }
#else
    asm volatile("trap;");
#endif
}

template<bool KmajorCodes = false>
__device__ __forceinline__ void nvfp4_stage_gate_weights(
    const mollm_cuda::Nvfp4Projection& job, uint8_t* stage, int cta_row, int k) {
    const int t=threadIdx.x;
    const size_t source = KmajorCodes ? size_t(k/64)*1280*32+
        (job.pair_layout ? size_t(cta_row/4+t/8)*128+(t%8)*16 : size_t(cta_row+t/2)*32+(t%2)*16)
        : job.pair_layout
        ? size_t(cta_row/4+t/8)*5120 + size_t(k)*2 + (t%8)*16
        : size_t(cta_row+t/2)*1280 + k/2 + (t%2)*16;
    const unsigned destination=static_cast<unsigned>(__cvta_generic_to_shared(stage+t*16));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
        "r"(destination),"l"(job.weight+source):"memory");
    if (t<64) {
        const unsigned scales=static_cast<unsigned>(__cvta_generic_to_shared(stage+2048+t*4));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::
            "r"(scales),"l"(job.block_scales+nvfp4_block_scale_offset(job,cta_row+t,160,k/16)):"memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Diagnostic, fixed production gate geometry. Canonical payload remains intact.
template<bool OutputBf16 = false, bool EarlyInputs = false, bool KmajorCodes = false>
__global__ void nvfp4_n32_async_weights_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int split_row, const mollm_cuda::Nvfp4MmaBatch* batches, const uint8_t* packed,
    const uint8_t* scales, int count, int minimum) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const auto batch=batches[blockIdx.y];
    if(batch.first<0 || batch.size<minimum)return; // CTA-uniform, before barriers
    const int first_job=batch.first,batch_size=batch.size;
    const auto job=jobs[first_job];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int cta_row=blockIdx.x*64, local_row=warp*16+lane/4,thread=lane%4;
    const int row=cta_row+local_row;
    __shared__ __align__(16) uint8_t staging[2][2304];
    nvfp4_stage_gate_weights<KmajorCodes>(job,staging[0],cta_row,0);
    float accum[4][4]={};
    for(int k=0;k<2560;k+=64) {
        uint32_t early_b0[4],early_b1[4],early_sb[4];
        if constexpr(EarlyInputs) {
#pragma unroll
            for(int group=0;group<4;++group) {
                if(group*8>=batch_size)continue;
                const int column=group*8+lane/4;
                const int source=first_job+min(column,batch_size-1)+
                    (split_row && cta_row>=split_row ? count : 0);
                const auto* p=packed+size_t(source)*1280;
                const auto* s=scales+size_t(source)*160;
                early_b0[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(p+k/2+thread*4);
                early_b1[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(p+k/2+thread*4+16);
                early_sb[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(s+k/16);
            }
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        const int current=(k/64)&1;
        if(k+64<2560)nvfp4_stage_gate_weights<KmajorCodes>(job,staging[current^1],cta_row,k+64);
        const int offset=job.pair_layout
            ? (local_row/4)*128+(local_row%4)*8+(thread/2)*32+(thread%2)*4
            : local_row*32+thread*4;
        const auto* words=reinterpret_cast<const uint32_t*>(staging[current]+offset);
        const int half_step=job.pair_layout?16:4;
        const uint32_t a0=words[0],a1=words[64],a2=words[half_step],a3=words[64+half_step];
        const uint32_t sa=*reinterpret_cast<const uint32_t*>(staging[current]+2048+
            (local_row+(thread&1)*8)*4);
#pragma unroll
        for(int group=0;group<4;++group) {
            if(group*8>=batch_size)continue;
            const int column=group*8+lane/4;
            const int source=first_job+min(column,batch_size-1)+
                (split_row && cta_row>=split_row ? count : 0);
            const auto* p=packed+size_t(source)*1280;
            const auto* s=scales+size_t(source)*160;
            const uint32_t b0=EarlyInputs?early_b0[group]:(column>=batch_size?0:*reinterpret_cast<const uint32_t*>(p+k/2+thread*4));
            const uint32_t b1=EarlyInputs?early_b1[group]:(column>=batch_size?0:*reinterpret_cast<const uint32_t*>(p+k/2+thread*4+16));
            const uint32_t sb=EarlyInputs?early_sb[group]:(column>=batch_size?0:*reinterpret_cast<const uint32_t*>(s+k/16));
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10,{0,0}, %11,{0,0};"
                : "+f"(accum[group][0]),"+f"(accum[group][1]),"+f"(accum[group][2]),"+f"(accum[group][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
        }
        __syncthreads(); // current stage readers finish before its next reuse
    }
#pragma unroll
    for(int group=0;group<4;++group) {
        const int column=group*8+thread*2;
        const auto first=jobs[first_job+min(column,batch_size-1)];
        const auto second=jobs[first_job+min(column+1,batch_size-1)];
        const float g0=split_row&&cta_row>=split_row?first.input_global_second:first.input_global;
        const float g1=split_row&&cta_row>=split_row?second.input_global_second:second.input_global;
        if(column<batch_size) {
            nvfp4_store_output<OutputBf16>(first.output,row,((accum[group][0]*g0)*job.row_scales[row])*first.output_scale);
            nvfp4_store_output<OutputBf16>(first.output,row+8,((accum[group][2]*g0)*job.row_scales[row+8])*first.output_scale);
        }
        if(column+1<batch_size) {
            nvfp4_store_output<OutputBf16>(second.output,row,((accum[group][1]*g1)*job.row_scales[row])*second.output_scale);
            nvfp4_store_output<OutputBf16>(second.output,row+8,((accum[group][3]*g1)*job.row_scales[row+8])*second.output_scale);
        }
    }
#else
    asm volatile("trap;");
#endif
}

__global__ void quantize_nvfp4_jobs_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int count, int inner, int variants, uint8_t* packed, uint8_t* scales,
    const mollm_cuda::Nvfp4MmaBatch* batches, int minimum) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = inner / 16;
    const int flat_job = ordinal / groups;
    if (flat_job >= count * variants) return;
    if (batches && batches[flat_job % count].size < minimum) return;
    const int group = ordinal % groups;
    const auto job = jobs[flat_job % count];
    const float global = flat_job >= count ? job.input_global_second : job.input_global;
    float maximum = 0;
    for (int i = 0; i < 16; ++i)
        maximum = fmaxf(maximum, fabsf(job.input[group * 16 + i]));
    const uint8_t code = __nv_cvt_float_to_fp8(__fdiv_rn(maximum, global * 6.f), __NV_SATFINITE, __NV_E4M3);
    scales[static_cast<size_t>(flat_job) * groups + group] = code;
    const float scale = __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
    const float inverse = scale > 0 ? __fdiv_rn(1.f, scale * global) : 0;
    for (int i = 0; i < 8; ++i) {
        const int offset = group * 16 + i * 2;
        packed[static_cast<size_t>(flat_job) * (inner / 2) + group * 8 + i] = __nv_cvt_float2_to_fp4x2(
            make_float2(job.input[offset] * inverse, job.input[offset + 1] * inverse), __NV_E2M1, cudaRoundNearest);
    }
#else
    asm volatile("trap;");
#endif
}

// Diagnostic vector I/O only. Scalar maximum/FP8/FP4 arithmetic is unchanged.
// Input and output storage must be disjoint, as required by the probe API.
template<bool KmajorScales=false,bool TokenInputs=false,bool PairWords=false,bool KmajorTokenCodes=false>
__global__ void quantize_nvfp4_jobs_vector_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int count,int inner,int variants,uint8_t* packed,uint8_t* scales,
    const mollm_cuda::Nvfp4MmaBatch* batches,int minimum,const float* token_input=nullptr,float token_global=1.f) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const int ordinal=blockIdx.x*blockDim.x+threadIdx.x,groups=inner/16;
    const int route_tiles=(count*variants+31)/32;
    const int flat_job=KmajorScales?(blockIdx.x%route_tiles)*32+threadIdx.x/4:ordinal/groups;
    if(flat_job>=count*variants)return;
    if(batches&&batches[flat_job%count].size<minimum)return;
    const int group=KmajorScales?(blockIdx.x/route_tiles)*4+threadIdx.x%4:ordinal%groups;
    const auto job=TokenInputs?mollm_cuda::Nvfp4Projection{}:jobs[flat_job%count];
    const float global=TokenInputs?token_global:flat_job>=count?job.input_global_second:job.input_global;
    const float* input=TokenInputs?token_input+size_t(flat_job)*inner+group*16:job.input+group*16;
    float values[16];
    if((reinterpret_cast<uintptr_t>(input)&15)==0) {
#pragma unroll
        for(int i=0;i<4;++i) {
            const float4 part=reinterpret_cast<const float4*>(input)[i];
            values[i*4]=part.x;values[i*4+1]=part.y;
            values[i*4+2]=part.z;values[i*4+3]=part.w;
        }
    } else {
#pragma unroll
        for(int i=0;i<16;++i)values[i]=input[i];
    }
    float maximum=0;
#pragma unroll
    for(int i=0;i<16;++i)maximum=fmaxf(maximum,fabsf(values[i]));
    const uint8_t code=__nv_cvt_float_to_fp8(__fdiv_rn(maximum,global*6.f),__NV_SATFINITE,__NV_E4M3);
    const size_t scale_index=KmajorScales?
        (size_t(group/4)*count*variants+flat_job)*4+group%4:size_t(flat_job)*groups+group;
    scales[scale_index]=code;
    const float scale=__half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code,__NV_E4M3)));
    const float inverse=scale>0?__fdiv_rn(1.f,scale*global):0;
    uint32_t words[2]={};
#pragma unroll
    for(int i=0;i<8;++i) {
        const uint8_t q=__nv_cvt_float2_to_fp4x2(make_float2(values[i*2]*inverse,
            values[i*2+1]*inverse),__NV_E2M1,cudaRoundNearest);
        words[i/4]|=uint32_t(q)<<((i%4)*8);
    }
    if constexpr(PairWords) {
        // Within each K64 row: [w0,w4,w1,w5,w2,w6,w3,w7].
        // One lane can fetch its two original B words with one aligned LDG64.
        auto* output=reinterpret_cast<uint32_t*>(packed+(KmajorTokenCodes?
            (size_t(group/4)*count+flat_job)*32:size_t(flat_job)*(inner/2)+(group/4)*32));
        const int word=(group%4)*2;
        output[(word%4)*2+word/4]=words[0];
        output[((word+1)%4)*2+(word+1)/4]=words[1];
    } else {
    uint8_t* output=packed+size_t(flat_job)*(inner/2)+group*8;
    if((reinterpret_cast<uintptr_t>(output)&3)==0) {
        reinterpret_cast<uint32_t*>(output)[0]=words[0];
        reinterpret_cast<uint32_t*>(output)[1]=words[1];
    } else {
#pragma unroll
        for(int i=0;i<8;++i)output[i]=uint8_t(words[i/4]>>((i%4)*8));
    }
    }
#else
    asm volatile("trap;");
#endif
}

// Diagnostic only: cache raw per-token maxima, not expert-dependent FP4
// bytes/scales. The original scalar quantizer and inference dispatch stay
// unchanged. Recompute on every invocation, including captured replays.
__global__ void nvfp4_input_block_maxima_cuda(const float* input, float* maxima,
    int rows, int inner) {
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= rows * (inner / 16)) return;
    float maximum = 0.f;
    for (int i = 0; i < 16; ++i)
        maximum = fmaxf(maximum, fabsf(input[size_t(ordinal) * 16 + i]));
    maxima[ordinal] = maximum;
}

__global__ void quantize_nvfp4_jobs_cached_max_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int count, int inner, int variants, uint8_t* packed, uint8_t* scales,
    const mollm_cuda::Nvfp4MmaBatch* batches, int minimum, const float* input,
    int input_rows, const float* maxima) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = inner / 16, flat_job = ordinal / groups;
    if (flat_job >= count * variants) return;
    if (batches && batches[flat_job % count].size < minimum) return;
    const int group = ordinal % groups;
    const auto job = jobs[flat_job % count];
    const uintptr_t base = reinterpret_cast<uintptr_t>(input);
    const uintptr_t address = reinterpret_cast<uintptr_t>(job.input);
    const unsigned row_bytes = unsigned(inner) * sizeof(float);
    const uintptr_t delta = address - base;
    float maximum = 0.f;
    // The host bounds guarantee a matrix extent below 2^30 bytes. Check the
    // full-width address range before narrowing, avoiding per-block uint64
    // division without accepting wrapped/out-of-matrix descriptor pointers.
    const bool in_matrix = address >= base && delta < size_t(input_rows) * row_bytes;
    const unsigned offset = in_matrix ? unsigned(delta) : 0;
    if (in_matrix && offset % row_bytes == 0)
        maximum = maxima[size_t(offset / row_bytes) * groups + group];
    else {
        // Valid descriptors outside the advertised matrix, or shifted views,
        // must keep the original scalar computation rather than use stale or
        // unrelated statistics. Descriptor input validity remains caller-owned.
        for (int i = 0; i < 16; ++i)
            maximum = fmaxf(maximum, fabsf(job.input[group * 16 + i]));
    }
    const float global = flat_job >= count ? job.input_global_second : job.input_global;
    const uint8_t code = __nv_cvt_float_to_fp8(__fdiv_rn(maximum, global * 6.f), __NV_SATFINITE, __NV_E4M3);
    scales[size_t(flat_job) * groups + group] = code;
    const float scale = __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
    const float inverse = scale > 0 ? __fdiv_rn(1.f, scale * global) : 0;
    for (int i = 0; i < 8; ++i) {
        const int offset = group * 16 + i * 2;
        packed[size_t(flat_job) * (inner / 2) + group * 8 + i] = __nv_cvt_float2_to_fp4x2(
            make_float2(job.input[offset] * inverse, job.input[offset + 1] * inverse), __NV_E2M1, cudaRoundNearest);
    }
#else
    asm volatile("trap;");
#endif
}

__device__ bool same_nvfp4_expert(const mollm_cuda::Nvfp4Projection& a,
    const mollm_cuda::Nvfp4Projection& b) {
    return a.weight == b.weight && a.block_scales == b.block_scales &&
        a.row_scales == b.row_scales && a.pair_layout == b.pair_layout &&
        a.mma_layout == b.mma_layout;
}

// Gate/up globals differ, but their input-block maxima are identical.
// Preserve the scalar max order and quantize each variant independently.
__global__ void quantize_nvfp4_jobs_dual_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int count,int inner,int variants,uint8_t* packed,uint8_t* scales,
    const mollm_cuda::Nvfp4MmaBatch* batches,int minimum) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const int ordinal=blockIdx.x*blockDim.x+threadIdx.x;
    const int groups=inner/16,job_index=ordinal/groups;
    if(job_index>=count || (batches && batches[job_index].size<minimum))return;
    const int group=ordinal%groups;
    const auto job=jobs[job_index];
    float maximum=0.f;
    for(int i=0;i<16;++i)maximum=fmaxf(maximum,fabsf(job.input[group*16+i]));
    for(int variant=0;variant<variants;++variant) {
        const int flat_job=job_index+variant*count;
        const float global=variant?job.input_global_second:job.input_global;
        const uint8_t code=__nv_cvt_float_to_fp8(__fdiv_rn(maximum,global*6.f),__NV_SATFINITE,__NV_E4M3);
        scales[size_t(flat_job)*groups+group]=code;
        const float scale=__half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code,__NV_E4M3)));
        const float inverse=scale>0?__fdiv_rn(1.f,scale*global):0.f;
        for(int i=0;i<8;++i) {
            const int offset=group*16+i*2;
            packed[size_t(flat_job)*(inner/2)+group*8+i]=__nv_cvt_float2_to_fp4x2(
                make_float2(job.input[offset]*inverse,job.input[offset+1]*inverse),__NV_E2M1,cudaRoundNearest);
        }
    }
#else
    asm volatile("trap;");
#endif
}

template<int Lanes = 16>
__global__ void quantize_nvfp4_jobs_subgroup_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int count, int inner, int variants, uint8_t* packed, uint8_t* scales,
    const mollm_cuda::Nvfp4MmaBatch* batches, int minimum) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const int ordinal = (size_t(blockIdx.x)*blockDim.x + threadIdx.x)/Lanes;
    const int groups = inner/16, flat_job = ordinal/groups;
    if (flat_job >= count*variants) return;
    if (batches && batches[flat_job%count].size < minimum) return;
    const int group = ordinal%groups, lane = threadIdx.x%Lanes;
    const auto job = jobs[flat_job%count];
    const float value = job.input[group*16+lane*(16/Lanes)];
    float second = 0.f;
    float maximum = fmaxf(0.f,fabsf(value));
    if constexpr (Lanes==8) {second=job.input[group*16+lane*2+1];maximum=fmaxf(maximum,fabsf(second));}
    const unsigned mask = __activemask();
#pragma unroll
    for (int offset=Lanes/2;offset;offset/=2)
        maximum = fmaxf(maximum,__shfl_down_sync(mask,maximum,offset,Lanes));
    float inverse = 0.f;
    if (lane==0) {
        const float global = flat_job>=count ? job.input_global_second : job.input_global;
        const uint8_t code = __nv_cvt_float_to_fp8(__fdiv_rn(maximum,global*6.f),__NV_SATFINITE,__NV_E4M3);
        scales[size_t(flat_job)*groups+group] = code;
        const float scale = __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code,__NV_E4M3)));
        inverse = scale>0 ? __fdiv_rn(1.f,scale*global) : 0.f;
    }
    inverse = __shfl_sync(mask,inverse,0,Lanes);
    if constexpr (Lanes==8)
        packed[size_t(flat_job)*(inner/2)+group*8+lane] = __nv_cvt_float2_to_fp4x2(
            make_float2(value*inverse,second*inverse),__NV_E2M1,cudaRoundNearest);
    else {
        const float next = __shfl_down_sync(mask,value,1,16);
        if (!(lane&1)) packed[size_t(flat_job)*(inner/2)+group*8+lane/2] = __nv_cvt_float2_to_fp4x2(
            make_float2(value*inverse,next*inverse),__NV_E2M1,cudaRoundNearest);
    }
#else
    asm volatile("trap;");
#endif
}

// Diagnostic prefill MMA + activation/quantization dataflow fusion.
template<bool SkewWeights = false,bool TransposedScales = false,bool KmajorCodes = false>
__device__ __forceinline__ void nvfp4_stage_fused_gate_weights(
    const mollm_cuda::Nvfp4Projection& job, uint8_t* stage, int cta_row, int k) {
    const int t=threadIdx.x%128;
    cta_row+=(threadIdx.x/128)*640;
    constexpr int weight_bytes=SkewWeights?3328:2304;
    constexpr int scale_offset=SkewWeights?3072:2048;
    stage+=(threadIdx.x/128)*weight_bytes;
    const size_t source = KmajorCodes ? size_t(k/64)*1280*32+
        (job.pair_layout?size_t(cta_row/4+t/8)*128+(t%8)*16:size_t(cta_row+t/2)*32+(t%2)*16) : job.pair_layout
        ? size_t(cta_row/4+t/8)*5120 + size_t(k)*2 + (t%8)*16
        : size_t(cta_row+t/2)*1280 + k/2 + (t%2)*16;
    const int destination_offset=SkewWeights&&job.pair_layout?(t/8)*192+(t%8)*16:t*16;
    const unsigned destination=static_cast<unsigned>(__cvta_generic_to_shared(stage+destination_offset));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
        "r"(destination),"l"(job.weight+source):"memory");
    if (t<64) {
        const unsigned scales=static_cast<unsigned>(__cvta_generic_to_shared(stage+scale_offset+t*4));
        const size_t sf_source=(TransposedScales||job.gate_scales_kmajor)?
            (size_t(k/64)*1280+cta_row+t)*4:size_t(cta_row+t)*160+k/16;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::
            "r"(scales),"l"(job.block_scales+sf_source):"memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Cooperatively copy one input tile per gate/up half instead of requesting it
// independently from each of the four output-row warps. Ragged columns use a
// valid source; their MMA registers are still explicitly zeroed below.
template<int InputPitch,bool KmajorInputScales = false>
__device__ __forceinline__ void nvfp4_stage_fused_gate_inputs(
    uint8_t* stage,const uint8_t* packed,const uint8_t* scales,
    int first,int batch_size,int count,int k) {
    const int t=threadIdx.x%128,half=threadIdx.x/128;
    constexpr int input_bytes=32*InputPitch+128;
    stage+=half*input_bytes;
    if(t<64) {
        const int column=t/2,source=first+min(column,batch_size-1)+half*count;
        const unsigned destination=static_cast<unsigned>(__cvta_generic_to_shared(stage+column*InputPitch+(t%2)*16));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
            "r"(destination),"l"(packed+size_t(source)*1280+k/2+(t%2)*16):"memory");
    }
    if(t<32) {
        const int source=first+min(t,batch_size-1)+half*count;
        const unsigned destination=static_cast<unsigned>(__cvta_generic_to_shared(stage+32*InputPitch+t*4));
        const size_t scale_source=KmajorInputScales?
            (size_t(k/64)*count*2+source)*4:size_t(source)*160+k/16;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::
            "r"(destination),"l"(scales+scale_source):"memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Diagnostic, fixed production gate geometry. Canonical payload remains intact.
template<bool Compact> __device__ __forceinline__ auto nvfp4_logit_code(float value) {
    if constexpr(Compact)return static_cast<uint16_t>(__float_as_uint(bf16(value))>>16);
    else return bf16(value);
}
// Caller guarantees each descriptor's input is a complete row of token_input.
// Rebuild every invocation: no stale map across capture replays/changed routes.
__global__ void nvfp4_token_input_map_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int count,const float* token_input,int* indices) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)indices[i]=static_cast<int>((jobs[i].input-token_input)/2560);
}
template<bool PublishActivated = true,bool StageInputs = false,int InputPitch = 32,bool PadLogits = false,bool SkewWeights = false,bool TransposedScales = false,bool KmajorInputScales = false,bool CompactLogits = false,bool TokenInputs = false,bool PairWords = false,bool KmajorCodes = false,bool KmajorTokenCodes = false,bool HoistTokenSources = false>
__global__ void nvfp4_n32_fused_activation_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    const mollm_cuda::Nvfp4Projection* down_jobs,
    const mollm_cuda::Nvfp4MmaBatch* batches, const uint8_t* packed,
    const uint8_t* scales, int count, float* activated, uint8_t* down_packed,
    uint8_t* down_scales,const int* token_indices=nullptr,int token_count=0) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const auto batch=batches[blockIdx.y];
    if(batch.first<0 || batch.size<1)return; // CTA-uniform, before barriers
    const int first_job=batch.first,batch_size=batch.size;
    const auto job=jobs[first_job];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int half=warp/4;
    const int cta_row=blockIdx.x*64, local_row=(warp%4)*16+lane/4,thread=lane%4;
    const int row=cta_row+local_row+half*640;
    constexpr int input_bytes=32*InputPitch+128;
    constexpr int weight_bytes=SkewWeights?3328:2304;
    constexpr int scale_offset=SkewWeights?3072:2048;
    __shared__ __align__(16) uint8_t staging[2][2*weight_bytes+(StageInputs?2*input_bytes:0)];
    using Logit=std::conditional_t<CompactLogits,uint16_t,float>;
    __shared__ Logit logits[2][64][PadLogits?33:32];
    nvfp4_stage_fused_gate_weights<SkewWeights,TransposedScales,KmajorCodes>(job,staging[0],cta_row,0);
    if constexpr(StageInputs)nvfp4_stage_fused_gate_inputs<InputPitch,KmajorInputScales>(staging[0]+2*weight_bytes,packed,scales,first_job,batch_size,count,0);
    float accum[4][4]={};
    int source_indices[4];
    if constexpr(HoistTokenSources) {
        static_assert(TokenInputs&&!StageInputs);
#pragma unroll
        for(int group=0;group<4;++group) {
            const int column=group*8+lane/4;
            source_indices[group]=group*8<batch_size?token_indices[first_job+min(column,batch_size-1)]:0;
        }
    }
    for(int k=0;k<2560;k+=64) {
        uint32_t early_b0[4],early_b1[4],early_sb[4];
        if constexpr(!StageInputs) {
#pragma unroll
            for(int group=0;group<4;++group) {
                if(group*8>=batch_size)continue;
                const int column=group*8+lane/4;
                const int source=HoistTokenSources?source_indices[group]:TokenInputs?token_indices[first_job+min(column,batch_size-1)]:
                    first_job+min(column,batch_size-1)+(half ? count : 0);
                const auto* p=packed+size_t(source)*1280;
                const auto* s=scales+size_t(source)*160;
                if constexpr(PairWords) {
                    const uint2 b=column>=batch_size?make_uint2(0,0):
                        *reinterpret_cast<const uint2*>(KmajorTokenCodes?
                            packed+(size_t(k/64)*token_count+source)*32+thread*8:p+k/2+thread*8);
                    early_b0[group]=b.x;early_b1[group]=b.y;
                } else {
                    early_b0[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(p+k/2+thread*4);
                    early_b1[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(p+k/2+thread*4+16);
                }
                if constexpr(KmajorInputScales) {
                    if constexpr(TokenInputs)early_sb[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(
                        scales+(size_t(k/64)*token_count+source)*4);
                    else early_sb[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(
                        scales+(size_t(k/64)*count*2+source)*4);
                }
                else early_sb[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(s+k/16);
            }
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        const int current=(k/64)&1;
        if(k+64<2560) {
            nvfp4_stage_fused_gate_weights<SkewWeights,TransposedScales,KmajorCodes>(job,staging[current^1],cta_row,k+64);
            if constexpr(StageInputs)nvfp4_stage_fused_gate_inputs<InputPitch,KmajorInputScales>(staging[current^1]+2*weight_bytes,
                packed,scales,first_job,batch_size,count,k+64);
        }
        if constexpr(StageInputs) {
#pragma unroll
            for(int group=0;group<4;++group) {
                if(group*8>=batch_size)continue;
                const int column=group*8+lane/4;
                const auto* p=staging[current]+2*weight_bytes+half*input_bytes+column*InputPitch;
                const auto* s=staging[current]+2*weight_bytes+half*input_bytes+32*InputPitch+column*4;
                early_b0[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(p+thread*4);
                early_b1[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(p+thread*4+16);
                early_sb[group]=column>=batch_size?0:*reinterpret_cast<const uint32_t*>(s);
            }
        }
        const int offset=job.pair_layout
            ? (local_row/4)*(SkewWeights?192:128)+(local_row%4)*8+(thread/2)*32+(thread%2)*4
            : local_row*32+thread*4;
        const auto* words=reinterpret_cast<const uint32_t*>(staging[current]+half*weight_bytes+offset);
        const int half_step=job.pair_layout?16:4;
        const int row_step=SkewWeights&&job.pair_layout?96:64;
        const uint32_t a0=words[0],a1=words[row_step],a2=words[half_step],a3=words[row_step+half_step];
        const uint32_t sa=*reinterpret_cast<const uint32_t*>(staging[current]+half*weight_bytes+scale_offset+
            (local_row+(thread&1)*8)*4);
#pragma unroll
        for(int group=0;group<4;++group) {
            if(group*8>=batch_size)continue;
            const int column=group*8+lane/4;
            const uint32_t b0=early_b0[group];
            const uint32_t b1=early_b1[group];
            const uint32_t sb=early_sb[group];
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10,{0,0}, %11,{0,0};"
                : "+f"(accum[group][0]),"+f"(accum[group][1]),"+f"(accum[group][2]),"+f"(accum[group][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
        }
        __syncthreads(); // current stage readers finish before its next reuse
    }
#pragma unroll
    for(int group=0;group<4;++group) {
        const int column=group*8+thread*2;
        const auto first=jobs[first_job+min(column,batch_size-1)];
        const auto second=jobs[first_job+min(column+1,batch_size-1)];
        const float g0=half?first.input_global_second:first.input_global;
        const float g1=half?second.input_global_second:second.input_global;
        if(column<batch_size) {
            logits[half][local_row][column]=nvfp4_logit_code<CompactLogits>(((accum[group][0]*g0)*job.row_scales[row])*first.output_scale);
            logits[half][local_row+8][column]=nvfp4_logit_code<CompactLogits>(((accum[group][2]*g0)*job.row_scales[row+8])*first.output_scale);
        }
        if(column+1<batch_size) {
            logits[half][local_row][column+1]=nvfp4_logit_code<CompactLogits>(((accum[group][1]*g1)*job.row_scales[row])*second.output_scale);
            logits[half][local_row+8][column+1]=nvfp4_logit_code<CompactLogits>(((accum[group][3]*g1)*job.row_scales[row+8])*second.output_scale);
        }
    }

    __syncthreads();
    for(int ordinal=threadIdx.x/16;ordinal<batch_size*4;ordinal+=16) {
        const int token=ordinal/4, group=ordinal%4, l=threadIdx.x%16;
        const int dimension=group*16+l;
        const auto down=down_jobs[first_job+token];
        const size_t route=(down.input-activated)/640;
        float g,u;
        if constexpr(CompactLogits) {
            g=__uint_as_float(static_cast<uint32_t>(logits[0][dimension][token])<<16);
            u=__uint_as_float(static_cast<uint32_t>(logits[1][dimension][token])<<16);
        } else {g=logits[0][dimension][token];u=logits[1][dimension][token];}
        const float value=bf16((g*(1.f/(1.f+expf(-g))))*u);
        if constexpr(PublishActivated)activated[route*640+cta_row+dimension]=value;
        float maximum=fmaxf(0.f,fabsf(value));
        const unsigned mask=__activemask();
#pragma unroll
        for(int offset=8;offset;offset/=2)
            maximum=fmaxf(maximum,__shfl_down_sync(mask,maximum,offset,16));
        float inverse=0;
        if(l==0) {
            const float global=down.input_global;
            const uint8_t code=__nv_cvt_float_to_fp8(__fdiv_rn(maximum,global*6.f),__NV_SATFINITE,__NV_E4M3);
            down_scales[size_t(first_job+token)*40+cta_row/16+group]=code;
            const float scale=__half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code,__NV_E4M3)));
            inverse=scale>0?__fdiv_rn(1.f,scale*global):0;
        }
        inverse=__shfl_sync(mask,inverse,0,16);
        const float next=__shfl_down_sync(mask,value,1,16);
        if(!(l&1))down_packed[size_t(first_job+token)*320+cta_row/2+group*8+l/2]=
            __nv_cvt_float2_to_fp4x2(make_float2(value*inverse,next*inverse),__NV_E2M1,cudaRoundNearest);
    }
#else
    asm volatile("trap;");
#endif
}

template<int Columns = 8>
__global__ void build_nvfp4_batches_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int count, mollm_cuda::Nvfp4MmaBatch* batches,
    mollm_cuda::Nvfp4MmaBatch* compact = nullptr, int minimum = 1) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || (i > 0 && same_nvfp4_expert(jobs[i - 1], jobs[i]))) return;
    int end = i + 1;
    while (end < count && same_nvfp4_expert(jobs[i], jobs[end])) ++end;
    int compact_base = 0;
    if (compact) {
        const int group_size = end - i;
        const int selected = group_size / Columns + (group_size % Columns >= minimum);
        // Reserve an entire expert run once. Appending individual batches
        // could interleave experts and discard the weight-cache locality that
        // stable grouping was meant to provide.
        if (selected) compact_base = atomicAdd(&compact[0].first, selected);
    }
    for (int first = i; first < end; first += Columns) {
        const int size = min(Columns, end - first);
        for (int j = first; j < first + size; ++j)
            batches[j] = {j == first ? first : -1, size};
        if (compact && size >= minimum) {
            compact[compact_base + (first - i) / Columns + 1] = {first, size};
        }
    }
}

// Diagnostic sparse planner: each lane owns one entry. Warp-cooperative
// backward search finds the initial run; ballots cover the next two warps.
// Arbitrary expert order (including A,B,A with equal weight keys) is supported.
template<int Columns>
__global__ void build_nvfp4_batches_warp_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int count,mollm_cuda::Nvfp4MmaBatch* batches) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x%32;
    const int start=i-lane;if(start>=count)return; // warp-uniform; no CTA barriers
    const bool active=i<count;
    const unsigned heads=__ballot_sync(0xffffffffu,active&&
        (i==0||!same_nvfp4_expert(jobs[i-1],jobs[i])));
    int carry=start;
    if(!(heads&1u)) {
        const auto first=jobs[start];
        for(int base=start-32;;base-=32){const int previous=base+lane;
            const unsigned different=__ballot_sync(0xffffffffu,previous<0||
                !same_nvfp4_expert(jobs[previous],first));
            if(different){carry=base+32-__clz(different);break;}
        }
    }
    const unsigned prior=heads&(0xffffffffu>>(31-lane));
    const int run_start=prior?start+31-__clz(prior):carry;
    const int next=i+32;
    const unsigned next_heads=__ballot_sync(0xffffffffu,next<count&&
        !same_nvfp4_expert(jobs[next-1],jobs[next]));
    const unsigned after=lane==31?0u:heads&(0xffffffffu<<(lane+1));
    const int run_end=after?start+__ffs(after)-1:
        next_heads?start+32+__ffs(next_heads)-1:count;
    if(active){const int first=run_start+((i-run_start)/Columns)*Columns;
        batches[i]={i==first?first:-1,min(Columns,min(count,run_end)-first)};}
}

__global__ void combine_nvfp4_native_parts_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    const float* partial, int rows, int count, int parts, int split_row) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(rows) * count) return;
    const int row = index % rows;
    const auto job = jobs[index / rows];
    float sum = 0;
    for (int part = 0; part < parts; ++part)
        sum += partial[static_cast<size_t>(part) * rows * count + index];
    const float global = split_row && row >= split_row ? job.input_global_second : job.input_global;
    job.output[row] = ((sum * global) * job.row_scales[row]) * job.output_scale;
}

__global__ void repack_nvfp4_gate_codes_cuda(const uint4* source, uint4* target,
    bool pair_layout) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr unsigned vectors = 1280 * 2560 / 32;
    if (index >= vectors) return;
    // Transpose [row-block][K64][vector-within-block] without changing any
    // bytes inside the canonical 32-byte ROW or 128-byte four-row PAIR tile.
    const unsigned block_vectors = pair_layout ? 8 : 2;
    const unsigned row_blocks = pair_layout ? 320 : 1280;
    const unsigned within = index % block_vectors;
    const unsigned row_block = (index / block_vectors) % row_blocks;
    const unsigned k64 = index / (block_vectors * row_blocks);
    target[index] = source[(row_block * 40 + k64) * block_vectors + within];
}

__global__ void repack_nvfp4_mma_cuda(mollm_cuda::Nvfp4Projection source,
    uint32_t* weights, uint8_t* scales, int rows, int inner, size_t tiles) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < tiles * 128; index += step) {
        const size_t tile = index / 128;
        const int reg = (index % 128) / 32, lane = index % 32;
        const int row = (tile / (inner / 64)) * 16 + lane / 4 + (reg % 2) * 8;
        const int col = (tile % (inner / 64)) * 64 + (lane % 4) * 8 + (reg / 2) * 32;
        uint32_t value = 0;
        if (row < rows) {
            const size_t offset = source.pair_layout
                ? (static_cast<size_t>(row / 4) * (inner / 16) + col / 16) * 32
                    + (row % 4) * 8 + (col % 16) / 2
                : static_cast<size_t>(row) * (inner / 2) + col / 2;
            value = *reinterpret_cast<const uint32_t*>(source.weight + offset);
        }
        weights[index] = value;
    }
    for (size_t index = first; index < tiles * 64; index += step) {
        const size_t tile = index / 64;
        const int local = index % 64;
        const int row = (tile / (inner / 64)) * 16 + local / 8 + ((local % 8) / 4) * 8;
        const int group = (tile % (inner / 64)) * 4 + local % 4;
        scales[index] = row < rows
            ? source.block_scales[nvfp4_block_scale_offset(source,row,inner/16,group)] : 0;
    }
}

__device__ float bf16(float value) {
    unsigned bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__global__ void moe_activation_cuda(
    const float* gate, const float* up, const float* weights, float* output,
    size_t count, int width, int stride) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const size_t row = i / width;
    const size_t src = row * stride + i % width;
    const float g = bf16(gate[src]);
    const float u = bf16(up[src]);
    const float activated = (g * (1.0f / (1.0f + expf(-g)))) * u;
    output[i] = bf16(activated * (weights ? weights[row] : 1.0f));
}

__global__ void nvfp4_activation_prepare_cuda(
    const mollm_cuda::Nvfp4Projection* jobs, int count, int inner,
    const float* gate_up, float* activated, uint8_t* packed, uint8_t* scales) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = inner / 16, job_index = ordinal / groups;
    if (job_index >= count) return;
    const int group = ordinal % groups;
    const auto job = jobs[job_index];
    const size_t route = (job.input - activated) / inner;
    const size_t source = route * (2 * inner) + group * 16;
    float values[16], maximum = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float g = bf16(gate_up[source + i]);
        const float u = bf16(gate_up[source + inner + i]);
        const float value = bf16((g * (1.0f / (1.0f + expf(-g)))) * u);
        values[i] = value;
        activated[route * inner + group * 16 + i] = value;
        maximum = fmaxf(maximum, fabsf(value));
    }
    const float global = job.input_global;
    const uint8_t code = __nv_cvt_float_to_fp8(__fdiv_rn(maximum, global * 6.f), __NV_SATFINITE, __NV_E4M3);
    scales[size_t(job_index) * groups + group] = code;
    const float scale = __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
    const float inverse = scale > 0 ? __fdiv_rn(1.f, scale * global) : 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        packed[size_t(job_index) * (inner / 2) + group * 8 + i] = __nv_cvt_float2_to_fp4x2(
            make_float2(values[i * 2] * inverse, values[i * 2 + 1] * inverse), __NV_E2M1, cudaRoundNearest);
#else
    asm volatile("trap;");
#endif
}

template<bool Packed> __device__ __forceinline__ float nvfp4_load_bf16(
    const float* source,size_t index) {
    if constexpr(Packed)return __uint_as_float(uint32_t(
        reinterpret_cast<const uint16_t*>(source)[index])<<16);
    else return bf16(source[index]);
}

template<bool PublishActivated = true, bool GateBf16 = false>
__global__ void nvfp4_activation_prepare_subgroup_cuda(
    const mollm_cuda::Nvfp4Projection* jobs, int count, int inner,
    const float* gate_up, float* activated, uint8_t* packed, uint8_t* scales) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const int ordinal = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 16;
    const int groups = inner / 16, job_index = ordinal / groups;
    if (job_index >= count) return;
    const int group = ordinal % groups, lane = threadIdx.x % 16;
    const auto job = jobs[job_index];
    const size_t route = (job.input - activated) / inner;
    const size_t source = route * (2 * inner) + group * 16 + lane;
    const float g = nvfp4_load_bf16<GateBf16>(gate_up,source);
    const float u = nvfp4_load_bf16<GateBf16>(gate_up,source+inner);
    const float value = bf16((g * (1.0f / (1.0f + expf(-g)))) * u);
    if constexpr (PublishActivated) activated[route * inner + group * 16 + lane] = value;
    // fmax(0,abs) matches the scalar baseline's NaN/zero handling. Only
    // maxima are reduced: activation evaluation and quantization stay intact.
    float maximum = fmaxf(0.f,fabsf(value));
    const unsigned mask = __activemask();
#pragma unroll
    for (int offset=8;offset;offset/=2)
        maximum = fmaxf(maximum,__shfl_down_sync(mask,maximum,offset,16));
    float inverse=0;
    if (lane==0) {
        const float global=job.input_global;
        const uint8_t code = __nv_cvt_float_to_fp8(__fdiv_rn(maximum,global*6.f),__NV_SATFINITE,__NV_E4M3);
        scales[size_t(job_index)*groups+group]=code;
        const float scale=__half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code,__NV_E4M3)));
        inverse=scale>0?__fdiv_rn(1.f,scale*global):0;
    }
    inverse=__shfl_sync(mask,inverse,0,16);
    const float next=__shfl_down_sync(mask,value,1,16);
    if (!(lane&1)) packed[size_t(job_index)*(inner/2)+group*8+lane/2]=__nv_cvt_float2_to_fp4x2(
        make_float2(value*inverse,next*inverse),__NV_E2M1,cudaRoundNearest);
#else
    asm volatile("trap;");
#endif
}

template<bool RoutedBf16=false>
__global__ void moe_combine_cuda(
    const float* routed, const float* shared, const float* shared_gate,
    float* output, size_t count, int hidden, int top_k) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const size_t row = i / hidden;
    float sum = 0.0f;
    for (int route = 0; route < top_k; ++route) {
        const size_t source=(row*top_k+route)*hidden+i%hidden;
        if constexpr(RoutedBf16)sum+=__uint_as_float(
            uint32_t(reinterpret_cast<const uint16_t*>(routed)[source])<<16);
        else sum+=bf16(routed[source]);
    }
    const float multiplier = 1.0f / (1.0f + expf(-bf16(shared_gate[row])));
    // Preserve the separate FP32 multiplication/addition used by the CPU
    // reference after rounding the shared down-projection to BF16.
    output[i] = __fadd_rn(sum, __fmul_rn(bf16(shared[i]), multiplier));
}

[[maybe_unused]] __device__ float e2m1_value(unsigned code) {
    // A constant-memory lookup serializes a warp when its 32 lanes decode
    // different nibbles. Construct the exact FP32 encoding with integer ops.
    const unsigned magnitude = code & 7;
    const unsigned bits = magnitude <= 1 ? magnitude * 0x3f000000u
        : ((magnitude / 2 + 126) << 23) | ((magnitude & 1) << 22);
    return __uint_as_float(bits | ((code & 8) << 28));
}


__device__ float e4m3_scale(uint8_t code) {
#if CUDART_VERSION >= 11080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
#else
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    float value = exponent == 0 ? mantissa * (1.0f / 512.0f)
        : __uint_as_float(static_cast<unsigned>(exponent + 120) << 23) *
          (1.0f + mantissa * 0.125f);
    if (exponent == 15 && mantissa == 7)
        value = __uint_as_float(0x7fc00000u);
    return code & 128 ? -value : value;
#endif
}

__device__ __forceinline__ float direct_e4m3_scale(uint8_t code) {
    const unsigned magnitude = code & 127u;
    const unsigned sign = static_cast<unsigned>(code & 128u) << 24;
    // Preserve the hardware NaN encoding, including its sign convention.
    if (magnitude == 127u) return e4m3_scale(code);
    const unsigned bits = magnitude < 8u
        ? __float_as_uint(static_cast<float>(magnitude) * (1.f / 512.f))
        : (magnitude << 20) + (120u << 23);
    return __uint_as_float(bits | sign);
}

__global__ void nvfp4_scale_decode_check_cuda(uint32_t* hardware, uint32_t* direct) {
    const unsigned code = threadIdx.x;
    hardware[code] = __float_as_uint(e4m3_scale(static_cast<uint8_t>(code)));
    direct[code] = __float_as_uint(direct_e4m3_scale(static_cast<uint8_t>(code)));
}

__device__ float2 e2m1_pair(uint8_t packed) {
#if CUDART_VERSION >= 12080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    return __half22float2(static_cast<__half2>(
        __nv_cvt_fp4x2_to_halfraw2(packed, __NV_E2M1)));
#else
    return make_float2(e2m1_value(packed & 15), e2m1_value(packed >> 4));
#endif
}

__device__ float direct_e2m1_value(unsigned code) {
    const unsigned magnitude = code & 7u;
    const unsigned bits = magnitude < 2 ? magnitude * 0x3f000000u
        : ((magnitude / 2 + 126u) << 23) | ((magnitude & 1u) << 22);
    return __uint_as_float(bits | ((code & 8u) << 28));
}

template<int Slots = 32, bool RegisterScores = false, bool CachedBest = false, bool PrefetchDescriptors = false, bool DownBf16 = false, bool GateBf16 = false>
__global__ void moe_resident_routes_cuda(
    const float* logits, int experts, int top_k, int score_func,
    bool normalize, float scale, const mollm_cuda::Nvfp4Projection* tables,
    const float* input, float* gate_up, float* activated, float* down,
    int hidden, int inter, mollm_cuda::Nvfp4Projection* gate_jobs,
    mollm_cuda::Nvfp4Projection* down_jobs, float* weights) {
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    float scores[Slots];
    auto score = [&](int j) {
        const int expert = lane + j * 32;
        float s = expert < experts ? logits[row * experts + expert] : -INFINITY;
        if (expert < experts && score_func == 1)
            s = 1.0f / (1.0f + expf(-s));
        else if (expert < experts && score_func == 2)
            s = sqrtf(s > 0 ? s + log1pf(expf(-s)) : log1pf(expf(s)));
        return s;
    };
    if constexpr (RegisterScores) {
#pragma unroll
        for (int j = 0; j < Slots; ++j) scores[j] = score(j);
    } else {
        for (int j = 0; j < Slots; ++j) scores[j] = score(j);
    }
    __shared__ int ids[32];
    __shared__ float values[32];
    auto local_max = [&](float& best, int& id) {
        auto consider = [&](int j) {
            const int candidate = lane + j * 32;
            if (scores[j] > best ||
                (scores[j] == best && candidate < id)) {
                best = scores[j]; id = candidate;
            }
        };
        if constexpr (RegisterScores) {
#pragma unroll
            for (int j = 0; j < Slots; ++j)
                if (j < (experts + 31) / 32) consider(j);
        } else {
            for (int j = 0; j < (experts + 31) / 32; ++j) consider(j);
        }
    };
    float cached_best = -INFINITY;
    int cached_id = 0x7fffffff;
    if constexpr (CachedBest) local_max(cached_best, cached_id);
    for (int k = 0; k < top_k; ++k) {
        float best = CachedBest ? cached_best : -INFINITY;
        int id = CachedBest ? cached_id : 0x7fffffff;
        if constexpr (!CachedBest) local_max(best, id);
        for (int offset = 16; offset; offset >>= 1) {
            const float other = __shfl_down_sync(0xffffffffu, best, offset);
            const int other_id = __shfl_down_sync(0xffffffffu, id, offset);
            if (lane + offset < 32 &&
                (other > best || (other == best && other_id < id))) {
                best = other; id = other_id;
            }
        }
        id = __shfl_sync(0xffffffffu, id, 0);
        // Malformed non-finite logits must not turn into an out-of-range
        // descriptor access. The CPU selector also defaults to expert zero
        // when no score beats its initial -infinity sentinel.
        if (id >= experts) id = 0;
        if (lane == 0) { ids[k] = id; values[k] = best; }
        if constexpr (RegisterScores) {
#pragma unroll
            for (int j = 0; j < Slots; ++j)
                if (id == lane + j * 32) scores[j] = -INFINITY;
        } else {
            if (lane == id % 32) scores[id / 32] = -INFINITY;
        }
        if constexpr (CachedBest) {
            if (lane == id % 32 && k + 1 < top_k) {
                cached_best = -INFINITY;
                cached_id = 0x7fffffff;
                local_max(cached_best, cached_id);
            }
        }
    }
    if constexpr (PrefetchDescriptors) {
        __syncwarp();
        if (lane < top_k) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(tables + ids[lane]));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(tables + experts + ids[lane]));
        }
    }
    if (lane == 0) {
        float sum = 0;
        const float maximum = values[0];
        for (int k = 0; k < top_k; ++k) {
            if (score_func == 0) values[k] = expf(values[k] - maximum);
            sum += values[k];
        }
        const float inverse = sum > 0 ? 1.0f / sum : 0.0f;
        for (int k = 0; k < top_k; ++k) {
            if (score_func == 0 || normalize) values[k] *= inverse;
            if (score_func != 0) values[k] *= scale;
        }
        // CPU accumulation visits experts in ascending ID order.
        for (int k = 1; k < top_k; ++k) {
            for (int j = k; j > 0 && ids[j] < ids[j - 1]; --j) {
                const int id = ids[j]; ids[j] = ids[j - 1]; ids[j - 1] = id;
                const float w = values[j]; values[j] = values[j - 1]; values[j - 1] = w;
            }
        }
    }
    __syncwarp();
    if (lane < top_k) {
        const int route = row * top_k + lane;
        auto gate = tables[ids[lane]], projection = tables[experts + ids[lane]];
        gate.input = input + static_cast<size_t>(row) * hidden;
        if constexpr(GateBf16) gate.output = reinterpret_cast<float*>(
            reinterpret_cast<uint16_t*>(gate_up) + static_cast<size_t>(route) * 2 * inter);
        else gate.output = gate_up + static_cast<size_t>(route) * 2 * inter;
        projection.input = activated + static_cast<size_t>(route) * inter;
        if constexpr(DownBf16) projection.output = reinterpret_cast<float*>(
            reinterpret_cast<uint16_t*>(down) + static_cast<size_t>(route) * hidden);
        else projection.output = down + static_cast<size_t>(route) * hidden;
        projection.output_scale = values[lane];
        gate_jobs[route] = gate;
        down_jobs[route] = projection;
        weights[route] = values[lane];
    }
}

// Diagnostic scalable-key variant: preserve the exact (weight, original slot)
// ranking, but use a fixed 1 KiB tile rather than count*8 shared bytes.
// This removes the shared-memory capacity boundary, not quadratic work.
template<bool PairRanks = false>
__global__ void group_nvfp4_jobs_tiled_cuda(
    const mollm_cuda::Nvfp4Projection* gate_jobs,
    const mollm_cuda::Nvfp4Projection* down_jobs,
    mollm_cuda::Nvfp4Projection* grouped_gate,
    mollm_cuda::Nvfp4Projection* grouped_down, int count,
    mollm_cuda::Nvfp4MmaBatch* pair_ranks) {
    __shared__ uintptr_t keys[128];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < count;
    const uintptr_t key = active ? reinterpret_cast<uintptr_t>(gate_jobs[i].weight) : 0;
    int rank = 0, before = 0, total = 0;
    bool compatible = true;
    for (int base = 0; base < count; base += 128) {
        const int source = base + threadIdx.x;
        keys[threadIdx.x] = source < count
            ? reinterpret_cast<uintptr_t>(gate_jobs[source].weight) : 0;
        __syncthreads();
        if (active) {
            const int available = count - base < 128 ? count - base : 128;
            for (int j = 0; j < available; ++j) {
                const int original = base + j;
                rank += keys[j] < key || (keys[j] == key && original < i);
                if constexpr (PairRanks) {
                    if (keys[j] == key) {
                        before += original < i;
                        ++total;
                        compatible &= same_nvfp4_expert(gate_jobs[i], gate_jobs[original]) &&
                            gate_jobs[i].pair_layout == gate_jobs[original].pair_layout &&
                            same_nvfp4_expert(down_jobs[i], down_jobs[original]) &&
                            down_jobs[i].pair_layout == down_jobs[original].pair_layout;
                    }
                }
            }
        }
        // Inactive tail threads must participate in both tile barriers.
        __syncthreads();
    }
    if (!active) return;
    grouped_gate[rank] = gate_jobs[i];
    grouped_down[rank] = down_jobs[i];
    if constexpr (PairRanks) {
        pair_ranks[rank] = !compatible ? mollm_cuda::Nvfp4MmaBatch{rank, 1}
            : before % 2 ? mollm_cuda::Nvfp4MmaBatch{-1, 0}
            : mollm_cuda::Nvfp4MmaBatch{rank, before + 1 < total ? 2 : 1};
    }
}

__global__ void nvfp4_radix_keys_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    uintptr_t* keys, int* indices, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        keys[i] = reinterpret_cast<uintptr_t>(jobs[i].weight);
        indices[i] = i;
    }
}

__global__ void nvfp4_radix_scatter_cuda(const mollm_cuda::Nvfp4Projection* gate,
    const mollm_cuda::Nvfp4Projection* down, mollm_cuda::Nvfp4Projection* grouped_gate,
    mollm_cuda::Nvfp4Projection* grouped_down, const int* indices, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int original = indices[i];
        grouped_gate[i] = gate[original];
        grouped_down[i] = down[original];
    }
}

template<bool PairRanks = false>
__global__ void group_nvfp4_jobs_cuda(
    const mollm_cuda::Nvfp4Projection* gate_jobs,
    const mollm_cuda::Nvfp4Projection* down_jobs,
    mollm_cuda::Nvfp4Projection* grouped_gate,
    mollm_cuda::Nvfp4Projection* grouped_down, int count,
    mollm_cuda::Nvfp4MmaBatch* pair_ranks = nullptr) {
    extern __shared__ uintptr_t keys[];
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        keys[i] = reinterpret_cast<uintptr_t>(gate_jobs[i].weight);
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uintptr_t key = keys[i];
    int rank = 0;
    for (int j = 0; j < count; ++j)
        rank += keys[j] < key || (keys[j] == key && j < i);
    grouped_gate[rank] = gate_jobs[i];
    grouped_down[rank] = down_jobs[i];
    if constexpr (PairRanks) {
        int before=0,total=0;
        bool compatible=true;
        for(int j=0;j<count;++j) if(keys[j]==key) {
            before += j<i; ++total;
            compatible &= same_nvfp4_expert(gate_jobs[i],gate_jobs[j]) &&
                gate_jobs[i].pair_layout==gate_jobs[j].pair_layout &&
                same_nvfp4_expert(down_jobs[i],down_jobs[j]) &&
                down_jobs[i].pair_layout==down_jobs[j].pair_layout;
        }
        pair_ranks[rank] = !compatible ? mollm_cuda::Nvfp4MmaBatch{rank,1}
            : before%2 ? mollm_cuda::Nvfp4MmaBatch{-1,0}
            : mollm_cuda::Nvfp4MmaBatch{rank,before+1<total?2:1};
    }
}

__global__ void nvfp4_pair_plan_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    mollm_cuda::Nvfp4MmaBatch* plan, int count) {
    if (threadIdx.x || blockIdx.x) return;
    int pairs = 0, singles = 0;
    for (int i = 0; i < count;) {
        const auto job = jobs[i];
        const bool paired = i + 1 < count && jobs[i + 1].weight == job.weight &&
            jobs[i + 1].block_scales == job.block_scales && jobs[i + 1].pair_layout == job.pair_layout;
        if (paired) {
            plan[pairs++] = {i, 2};
            i += 2;
        } else {
            plan[count - 1 - singles++] = {i++, 1};
        }
    }
    plan[count] = {pairs, singles};
}

struct Nvfp4SimtPacket {
    uint16_t packed;
    uint8_t scale;
    float4 input;
};
template<bool Inputs, bool Interleaved = false, int LoadHint = 0, bool WarpPacket = false, bool AlignedPackets = false, bool PairOnly = false, bool BroadcastScale = false, bool TiledScale = false, bool NativeMmaLayout = false, bool KmajorCodes = false>
__device__ __forceinline__ Nvfp4SimtPacket nvfp4_simt_packet(
    const mollm_cuda::Nvfp4Projection& job, int row, int inner, int quad) {
    if (quad >= inner / 4) return {};
    const int groups = inner / 16, group = quad / 4;
    static_assert(!NativeMmaLayout || (!Interleaved && !LoadHint && !WarpPacket && !AlignedPackets && !PairOnly && !BroadcastScale && !TiledScale));
    const int col = quad * 4;
    static_assert(!KmajorCodes || (!NativeMmaLayout && !Interleaved));
    const size_t native_tile = static_cast<size_t>(row / 16) * (inner / 64) + col / 64;
    const size_t native_offset = (native_tile * 128 +
        ((col % 64 / 32) * 2 + row % 16 / 8) * 32 +
        (row % 8) * 4 + col % 32 / 8) * 4 + (col % 8) / 2;
    const size_t tile = (static_cast<size_t>(row / 4) * groups + group) * 36;
    const size_t offset = KmajorCodes
        ? size_t(col / 64) * 1280 * 32 + (job.pair_layout
            ? size_t(row / 4) * 128 + (col % 64 / 16) * 32 + (row % 4) * 8 + (col % 16) / 2
            : size_t(row) * 32 + (col % 64) / 2)
        : NativeMmaLayout ? native_offset : Interleaved ? tile + (row % 4) * 8 + (quad % 4) * 2 : (PairOnly || job.pair_layout)
        ? (static_cast<size_t>(row / 4) * groups + group) * 32 + (row % 4) * 8 + (quad % 4) * 2
        : static_cast<size_t>(row) * (inner / 2) + quad * 2;
    const auto* codes = job.weight + offset;
    Nvfp4SimtPacket result{};
    const auto* packed = reinterpret_cast<const uint16_t*>(codes);
    if constexpr (!WarpPacket) result.packed = (AlignedPackets || reinterpret_cast<uintptr_t>(codes) % 2 == 0)
        ? (LoadHint == 1 || LoadHint == 6 ? __ldcg(packed) : LoadHint == 2 ? __ldcs(packed) : *packed)
        : uint16_t(codes[0]) | (uint16_t(codes[1]) << 8);
    if constexpr (LoadHint == 3 || LoadHint == 4) {
        if (!(reinterpret_cast<uintptr_t>(codes) & 1)) {
            unsigned short code;
            unsigned long long policy;
            if constexpr (LoadHint == 4) asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
            else asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
            asm volatile("ld.global.L1::no_allocate.L2::cache_hint.u16 %0, [%1], %2;" : "=h"(code) : "l"(codes), "l"(policy));
            result.packed=code;
        }
    }
    const auto* scale = NativeMmaLayout ? job.block_scales + native_tile * 64 + (row % 8) * 8 + (row % 16 / 8) * 4 + group % 4
        : Interleaved ? job.weight + tile + 32 + row % 4
        : job.block_scales + (TiledScale ? (static_cast<size_t>(row / 4) * groups + group) * 4 + row % 4
                                       : nvfp4_block_scale_offset(job,row,groups,group));
    if constexpr (!WarpPacket) {
        if constexpr (BroadcastScale) {
            unsigned code = (quad & 3) == 0 ? *scale : 0;
            result.scale = __shfl_sync(0xffffffffu, code, (threadIdx.x & 31) & ~3);
        } else result.scale = LoadHint == 1 || LoadHint == 5 ? __ldcg(scale) : LoadHint == 2 ? __ldcs(scale) : *scale;
    }
    if constexpr (LoadHint == 3 || LoadHint == 4) {
        unsigned code;
        unsigned long long policy;
        if constexpr (LoadHint == 4) asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
        else asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
        asm volatile("ld.global.L1::no_allocate.L2::cache_hint.u8 %0, [%1], %2;" : "=r"(code) : "l"(scale), "l"(policy));
        result.scale=code;
    }
    if constexpr (WarpPacket) {
        // Four neighbouring lanes consume one 8-byte FP4 group. Every lane
        // enters these shuffles; supported K values have no partial warp tail.
        const int slot=quad&3;
        const uint8_t* group_codes=codes-slot*2;
        uint64_t whole=0;
        unsigned scale_code=0;
        if(!slot) {
            if(!(reinterpret_cast<uintptr_t>(group_codes)&7)) whole=*reinterpret_cast<const uint64_t*>(group_codes);
            else for(int byte=0;byte<8;++byte)whole|=static_cast<uint64_t>(group_codes[byte])<<(byte*8);
            scale_code=*scale;
        }
        const int owner=(threadIdx.x&31)&~3;
        const uint32_t low=__shfl_sync(0xffffffffu,static_cast<uint32_t>(whole),owner);
        const uint32_t high=__shfl_sync(0xffffffffu,static_cast<uint32_t>(whole>>32),owner);
        result.packed=(slot<2?low:high)>>((slot&1)*16);
        result.scale=__shfl_sync(0xffffffffu,scale_code,owner);
    }
    if constexpr (Inputs) {
        const float* x = job.input + quad * 4;
        result.input = (AlignedPackets || reinterpret_cast<uintptr_t>(x) % alignof(float4) == 0)
            ? *reinterpret_cast<const float4*>(x) : make_float4(x[0], x[1], x[2], x[3]);
    }
    return result;
}

template<int Rows>
__global__ void nvfp4_route_sum_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int rows, int inner, int count, float* output) {
    const int warp=threadIdx.x/32, local_row=warp%Rows;
    const int row=blockIdx.x*Rows+local_row, route=warp/Rows, lane=threadIdx.x&31;
    const auto job=jobs[route];
    auto pending=row<rows?nvfp4_simt_packet<true>(job,row,inner,lane):Nvfp4SimtPacket{};
    float sum=0.0f;
    for(int quad=lane;quad<inner/4;quad+=32) {
        const auto next=row<rows?nvfp4_simt_packet<true>(job,row,inner,quad+32):Nvfp4SimtPacket{};
        const float scale=e4m3_scale(pending.scale);
        const float2 w0=e2m1_pair(pending.packed&255),w1=e2m1_pair(pending.packed>>8);
        const float4 a=pending.input;
        float dot=a.x*w0.x;
        dot=fmaf(a.y,w0.y,dot);dot=fmaf(a.z,w1.x,dot);dot=fmaf(a.w,w1.y,dot);
        sum=fmaf(dot,scale,sum);pending=next;
    }
    for(int offset=16;offset;offset/=2)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
    __shared__ float values[Rows][16];
    if(!lane)values[local_row][route]=row<rows?bf16(sum*job.row_scales[row]):0.f;
    __syncthreads();
    if(threadIdx.x<Rows && blockIdx.x*Rows+threadIdx.x<rows) {
        float total=0.0f;
        for(int i=0;i<count;++i)total+=values[threadIdx.x][i];
        output[blockIdx.x*Rows+threadIdx.x]=total;
    }
}

template<int Lanes, bool TailGuard = false>
__global__ void nvfp4_exact_subwarp_cuda(const mollm_cuda::Nvfp4Projection* jobs,int rows,int inner) {
    constexpr int chains=32/Lanes;
    const int lane=threadIdx.x%Lanes;
    const int row=(blockIdx.x*(blockDim.x/Lanes))+threadIdx.x/Lanes;
    const auto job=jobs[blockIdx.y];
    float sums[chains]{};
    Nvfp4SimtPacket pending[chains]{};
#pragma unroll
    for(int chain=0;chain<chains;++chain)
        if(row<rows)pending[chain]=nvfp4_simt_packet<true>(job,row,inner,lane+chain*Lanes);
    for(int first=lane;first<inner/4;first+=32) {
        Nvfp4SimtPacket next[chains]{};
#pragma unroll
        for(int chain=0;chain<chains;++chain)
            if(row<rows)next[chain]=nvfp4_simt_packet<true>(job,row,inner,first+chain*Lanes+32);
#pragma unroll
        for(int chain=0;chain<chains;++chain) {
            if constexpr(TailGuard) { if(first+chain*Lanes>=inner/4)continue; }
            const auto packet=pending[chain];
            const float2 low=e2m1_pair(packet.packed&255),high=e2m1_pair(packet.packed>>8);
            float dot=packet.input.x*low.x;
            dot=fmaf(packet.input.y,low.y,dot);dot=fmaf(packet.input.z,high.x,dot);dot=fmaf(packet.input.w,high.y,dot);
            sums[chain]=fmaf(dot,e4m3_scale(packet.scale),sums[chain]);
            pending[chain]=next[chain];
        }
    }
    float sum;
    if constexpr(chains==2)sum=sums[0]+sums[1];
    else sum=(sums[0]+sums[2])+(sums[1]+sums[3]);
    for(int offset=Lanes/2;offset;offset/=2)sum+=__shfl_down_sync(0xffffffffu,sum,offset,Lanes);
    if(!lane && row<rows)job.output[row]=sum*job.row_scales[row];
}

template<int RowGroup>
__global__ void nvfp4_row_reuse_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int rows, int inner) {
    const int base = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * RowGroup;
    if (base >= rows) return;
    const int lane = threadIdx.x % 32;
    const auto job = jobs[blockIdx.y];
    float sums[RowGroup]{};
    Nvfp4SimtPacket pending[RowGroup]{};
#pragma unroll
    for (int r=0;r<RowGroup;++r) if(base+r<rows) {
        if(r==0) pending[r]=nvfp4_simt_packet<true>(job,base+r,inner,lane);
        else pending[r]=nvfp4_simt_packet<false>(job,base+r,inner,lane);
    }
    for(int quad=lane;quad<inner/4;quad+=32) {
        Nvfp4SimtPacket incoming[RowGroup]{};
#pragma unroll
        for(int r=0;r<RowGroup;++r) if(base+r<rows) {
            if(r==0) incoming[r]=nvfp4_simt_packet<true>(job,base+r,inner,quad+32);
            else incoming[r]=nvfp4_simt_packet<false>(job,base+r,inner,quad+32);
        }
        const float4 a=pending[0].input;
#pragma unroll
        for(int r=0;r<RowGroup;++r) if(base+r<rows) {
            const float2 low=e2m1_pair(pending[r].packed&255);
            const float2 high=e2m1_pair(pending[r].packed>>8);
            float dot=a.x*low.x;
            dot=fmaf(a.y,low.y,dot); dot=fmaf(a.z,high.x,dot); dot=fmaf(a.w,high.y,dot);
            sums[r]=fmaf(dot,e4m3_scale(pending[r].scale),sums[r]);
        }
#pragma unroll
        for(int r=0;r<RowGroup;++r) pending[r]=incoming[r];
    }
#pragma unroll
    for(int r=0;r<RowGroup;++r) if(base+r<rows) {
        for(int offset=16;offset;offset>>=1)
            sums[r]+=__shfl_down_sync(0xffffffffu,sums[r],offset);
        if(!lane)job.output[base+r]=sums[r]*job.row_scales[base+r];
    }
}

// Gate and up retain independent lane accumulators and the original warp tree.
// A shared input packet removes intermediate writes without changing precision.
template<int LoadHint = 0>
__global__ void nvfp4_gate_activation_cuda(const mollm_cuda::Nvfp4Projection* jobs,
    int width, int inner, const float* route_weights, float* activated) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= width) return;
    const int lane = threadIdx.x % 32;
    const auto job = jobs[blockIdx.y];
    auto gate = nvfp4_simt_packet<true,false,LoadHint>(job, row, inner, lane);
    auto up = nvfp4_simt_packet<false,false,LoadHint>(job, row + width, inner, lane);
    float gate_sum = 0, up_sum = 0;
    for (int quad = lane; quad < inner / 4; quad += 32) {
        const auto next_gate = nvfp4_simt_packet<true,false,LoadHint>(job, row, inner, quad + 32);
        const auto next_up = nvfp4_simt_packet<false,false,LoadHint>(job, row + width, inner, quad + 32);
        const auto accumulate = [](const Nvfp4SimtPacket& packet, const float4& input, float sum) {
            const float2 low = e2m1_pair(packet.packed & 255);
            const float2 high = e2m1_pair(packet.packed >> 8);
            float dot = input.x * low.x;
            dot = fmaf(input.y, low.y, dot);
            dot = fmaf(input.z, high.x, dot);
            dot = fmaf(input.w, high.y, dot);
            return fmaf(dot, e4m3_scale(packet.scale), sum);
        };
        gate_sum = accumulate(gate, gate.input, gate_sum);
        up_sum = accumulate(up, gate.input, up_sum);
        gate = next_gate; up = next_up;
    }
    for (int distance = 16; distance; distance >>= 1) {
        gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, distance);
        up_sum += __shfl_down_sync(0xffffffffu, up_sum, distance);
    }
    if (!lane) {
        const float g = bf16(gate_sum * job.row_scales[row]);
        const float u = bf16(up_sum * job.row_scales[row + width]);
        const float value = (g * (1.f / (1.f + expf(-g)))) * u;
        activated[static_cast<size_t>(blockIdx.y) * width + row] =
            bf16(value * (route_weights ? route_weights[blockIdx.y] : 1.f));
    }
}

template<int PrefetchIterations = 0, bool NativeMmaLayout = false, bool PairedJobs = false,
         int PlannedSize = 0, int Pipeline = 0, int PipelineDepth = 1, bool Interleaved = false,
         bool CacheInput = false, int LoadHint = 0, bool DirectFp4 = false, bool InlinePairs = false,
         bool RankPairs = false, int FixedInner = 0, bool WarpPacket = false, bool AlignedPackets = false, bool DirectScale = false, bool RouteMajor = false, bool PairOnly = false, bool BroadcastScale = false, bool TiledScale = false, bool KmajorCodes = false>
__global__ void nvfp4_projections_cuda(
    const mollm_cuda::Nvfp4Projection* jobs, int rows, int dynamic_inner, int count = 0,
    const mollm_cuda::Nvfp4MmaBatch* plan = nullptr) {
    const int inner = FixedInner ? FixedInner : dynamic_inner;
    static_assert(!KmajorCodes || (!NativeMmaLayout && !Interleaved && !PairedJobs && !PrefetchIterations));
    if constexpr (PlannedSize != 0) {
        const int valid = PlannedSize == 2 ? plan[count].first : plan[count].size;
        if (blockIdx.y >= valid) return;
    }
    const int warps = blockDim.x / 32;
    static_assert(!Interleaved || (Pipeline == 2 && !NativeMmaLayout && !PairedJobs && !PlannedSize));
    constexpr int lanes = 32;
    const int row = (RouteMajor ? blockIdx.y : blockIdx.x) * warps + threadIdx.x / 32;
    if constexpr (!CacheInput) { if (row >= rows) return; }
    const int lane = threadIdx.x % lanes;
    int first = RouteMajor ? blockIdx.x : PlannedSize == 2 ? plan[blockIdx.y].first :
        PlannedSize == 1 ? plan[count - 1 - blockIdx.y].first :
        blockIdx.y * (PairedJobs ? 2 : 1);
    const auto same_group = [](const auto& a, const auto& b) {
        return same_nvfp4_expert(a,b) && a.pair_layout == b.pair_layout;
    };
    if constexpr (InlinePairs) {
        static_assert(PairedJobs && !PlannedSize && !NativeMmaLayout && !CacheInput);
        first = blockIdx.y;
        if constexpr (RankPairs) {
            if (plan[first].first < 0) return;
        } else {
            int begin = first;
            while (begin > 0 && same_group(jobs[begin-1],jobs[first])) --begin;
            if ((first-begin) & 1) return;
        }
    }
    auto job = jobs[first];
    float early_row_scale = 0.f;
    if constexpr (LoadHint == 8) {
        if (lane == 0 && row < rows)
            asm volatile("ld.global.f32 %0, [%1];" : "=f"(early_row_scale) : "l"(job.row_scales + row));
    }
    if constexpr (LoadHint == 7) {
        if (lane == 0 && row < rows)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(job.row_scales + row));
    }
    if constexpr (LoadHint == 9) {
        // Four adjacent scales fit an aligned16-byte segment. Otherwise
        // retain per-warp prefetch so the final scale is not overlooked.
        if (lane == 0 && row < rows &&
            (threadIdx.x == 0 || reinterpret_cast<uintptr_t>(job.row_scales) % 16 != 0))
            asm volatile("prefetch.global.L2 [%0];" :: "l"(job.row_scales + row));
    }
    if constexpr (CacheInput) {
        static_assert(Pipeline == 2 && !NativeMmaLayout && !PairedJobs && !PlannedSize);
        extern __shared__ float cached_input[];
        for (int i = threadIdx.x; i < inner; i += blockDim.x)
            cached_input[i] = job.input[i];
        __syncthreads();
        job.input = cached_input;
        if (row >= rows) return;
    }
    const bool second_valid = PairedJobs && first + 1 < count &&
        (!InlinePairs || same_group(jobs[first],jobs[first+1])) &&
        (!RankPairs || plan[first].size==2);
    const auto second = second_valid ? jobs[first + 1] : job;
    const bool shared = second.weight == job.weight &&
        second.block_scales == job.block_scales && second.pair_layout == job.pair_layout;
    const int groups = inner / 16;
    float sum = 0.0f;
    float second_sum = 0.0f;
    Nvfp4SimtPacket pending[PipelineDepth]{};
    if constexpr (Pipeline) {
        static_assert((!PairedJobs || InlinePairs) && !PlannedSize);
        static_assert(!NativeMmaLayout || (!PairedJobs && !Interleaved && !CacheInput && !LoadHint && !DirectFp4 && !FixedInner && !WarpPacket && !AlignedPackets && !DirectScale && !RouteMajor && !PairOnly && !BroadcastScale && !TiledScale));
        static_assert(PipelineDepth == 1 || PipelineDepth == 2 || PipelineDepth == 4);
#pragma unroll
        for (int i = 0; i < PipelineDepth; ++i)
            pending[i] = nvfp4_simt_packet<Pipeline == 2, Interleaved, LoadHint, WarpPacket, AlignedPackets, PairOnly, BroadcastScale, TiledScale, NativeMmaLayout, KmajorCodes>(job, row, inner, lane + lanes * i);
    }
    for (int quad = lane; quad < inner / 4; quad += lanes) {
        if constexpr (LoadHint == 10) {
            if (lane == 0 && quad == (inner / 4 / lanes / 2) * lanes)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(job.row_scales + row));
        }
        Nvfp4SimtPacket incoming{};
        if constexpr (Pipeline) incoming = nvfp4_simt_packet<Pipeline == 2, Interleaved, LoadHint, WarpPacket, AlignedPackets, PairOnly, BroadcastScale, TiledScale, NativeMmaLayout, KmajorCodes>(job, row, inner, quad + lanes * PipelineDepth);
        const int group = quad / 4;
        const int col = quad * 4;
        const size_t native_tile = static_cast<size_t>(row / 16) * (inner / 64) + col / 64;
        const size_t native_offset = (native_tile * 128 +
            ((col % 64 / 32) * 2 + row % 16 / 8) * 32 +
            (row % 8) * 4 + col % 32 / 8) * 4 + (col % 8) / 2;
        const size_t offset = KmajorCodes
            ? size_t(col / 64) * 1280 * 32 + (job.pair_layout
                ? size_t(row / 4) * 128 + (col % 64 / 16) * 32 + (row % 4) * 8 + (col % 16) / 2
                : size_t(row) * 32 + (col % 64) / 2)
            : NativeMmaLayout ? native_offset : job.pair_layout
            ? (static_cast<size_t>(row / 4) * groups + group) * 32 +
              (row % 4) * 8 + (quad % 4) * 2
            : static_cast<size_t>(row) * (inner / 2) + quad * 2;
        const uint8_t* codes = job.weight + offset;
        if constexpr (PrefetchIterations > 0) {
            if (!NativeMmaLayout && quad + lanes * PrefetchIterations < inner / 4) {
                const auto* next_codes = codes +
                    (job.pair_layout ? 256 : 64) * PrefetchIterations;
                const auto* next_scale = job.block_scales +
                    static_cast<size_t>(row) * groups + group + 8 * PrefetchIterations;
                asm volatile("prefetch.global.L2 [%0];" :: "l"(next_codes));
                asm volatile("prefetch.global.L2 [%0];" :: "l"(next_scale));
            }
        }
        const uint16_t packed = Pipeline ? pending[0].packed : reinterpret_cast<uintptr_t>(codes) % 2 == 0
            ? *reinterpret_cast<const uint16_t*>(codes)
            : static_cast<uint16_t>(codes[0]) | (static_cast<uint16_t>(codes[1]) << 8);
        const size_t scale_offset = NativeMmaLayout
            ? native_tile * 64 + (row % 8) * 8 + (row % 16 / 8) * 4 + group % 4
            : nvfp4_block_scale_offset(job,row,groups,group);
        const uint8_t scale_code = Pipeline ? pending[0].scale : job.block_scales[scale_offset];
        float scale;
        if constexpr (LoadHint == 11) {
            const unsigned scale_mask = __activemask();
            const float leader_scale = (lane % 4 == 0) ? e4m3_scale(scale_code) : 0.f;
            scale = __shfl_sync(scale_mask, leader_scale, lane & ~3);
        } else scale = DirectScale ? direct_e4m3_scale(scale_code) : e4m3_scale(scale_code);
        const float2 w0 = DirectFp4 ? make_float2(direct_e2m1_value(packed),direct_e2m1_value(packed >> 4)) : e2m1_pair(packed & 255);
        const float2 w1 = DirectFp4 ? make_float2(direct_e2m1_value(packed >> 8),direct_e2m1_value(packed >> 12)) : e2m1_pair(packed >> 8);
        const float* x = job.input + quad * 4;
        const float4 a = Pipeline == 2 ? pending[0].input : reinterpret_cast<uintptr_t>(x) % alignof(float4) == 0
            ? *reinterpret_cast<const float4*>(x)
            : make_float4(x[0], x[1], x[2], x[3]);
        float dot = a.x * w0.x;
        dot = fmaf(a.y, w0.y, dot);
        dot = fmaf(a.z, w1.x, dot);
        dot = fmaf(a.w, w1.y, dot);
        sum = fmaf(dot, scale, sum);
        if constexpr (Pipeline) {
#pragma unroll
            for (int i = 0; i + 1 < PipelineDepth; ++i) pending[i] = pending[i + 1];
            pending[PipelineDepth - 1] = incoming;
        }
        if constexpr (PairedJobs) {
            if (second_valid) {
                const size_t second_offset = NativeMmaLayout ? native_offset : second.pair_layout
                    ? (static_cast<size_t>(row / 4) * groups + group) * 32 +
                      (row % 4) * 8 + (quad % 4) * 2
                    : static_cast<size_t>(row) * (inner / 2) + quad * 2;
                const uint8_t* second_codes = second.weight + second_offset;
                const uint16_t second_packed = shared ? packed :
                    reinterpret_cast<uintptr_t>(second_codes) % 2 == 0
                    ? *reinterpret_cast<const uint16_t*>(second_codes)
                    : static_cast<uint16_t>(second_codes[0]) |
                      (static_cast<uint16_t>(second_codes[1]) << 8);
                const float second_scale = shared ? scale : e4m3_scale(second.block_scales[scale_offset]);
                const float2 v0 = shared ? w0 : e2m1_pair(second_packed & 255);
                const float2 v1 = shared ? w1 : e2m1_pair(second_packed >> 8);
                const float* y = second.input + quad * 4;
                const float4 b = reinterpret_cast<uintptr_t>(y) % alignof(float4) == 0
                    ? *reinterpret_cast<const float4*>(y)
                    : make_float4(y[0], y[1], y[2], y[3]);
                float second_dot = b.x * v0.x;
                second_dot = fmaf(b.y, v0.y, second_dot);
                second_dot = fmaf(b.z, v1.x, second_dot);
                second_dot = fmaf(b.w, v1.y, second_dot);
                second_sum = fmaf(second_dot, second_scale, second_sum);
            }
        }
    }
    const unsigned mask = __activemask();
    for (int offset = lanes / 2; offset; offset /= 2)
        sum += __shfl_down_sync(mask, sum, offset, lanes);
    if (lane == 0) job.output[row] = sum * (LoadHint == 8 ? early_row_scale : job.row_scales[row]);
    if constexpr (PairedJobs) {
        if (second_valid) {
            for (int offset = lanes / 2; offset; offset /= 2)
                second_sum += __shfl_down_sync(mask, second_sum, offset, lanes);
            if (lane == 0) second.output[row] = second_sum * second.row_scales[row];
        }
    }
}

}  // namespace

namespace mollm_cuda {

cudaError_t launch_repack_nvfp4_gate_scales(const Nvfp4Projection& source,
    uint8_t* target,int rows,int inner,cudaStream_t stream) {
    constexpr uintptr_t bytes=1280*2560/16;
    const auto src=reinterpret_cast<uintptr_t>(source.block_scales);
    const auto dst=reinterpret_cast<uintptr_t>(target);
    if(rows!=1280||inner!=2560||source.mma_layout||source.gate_scales_kmajor||
       !src||!dst||(src&3)||(dst&3)||src>UINTPTR_MAX-bytes||dst>UINTPTR_MAX-bytes||
       (src<dst+bytes&&dst<src+bytes))return cudaErrorInvalidValue;
    repack_nvfp4_gate_scales_cuda<<<dim3(40,2),dim3(32,8),0,stream>>>(source.block_scales,target);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_gate_scale_view(const Nvfp4Projection* source,
    Nvfp4Projection* target,uint8_t* scales,int experts,cudaStream_t stream) {
    if(!source||!target||!scales||experts<1||experts>1024||
        reinterpret_cast<uintptr_t>(source)%alignof(Nvfp4Projection)||
        reinterpret_cast<uintptr_t>(target)%alignof(Nvfp4Projection)||
        reinterpret_cast<uintptr_t>(scales)%4)return cudaErrorInvalidValue;
    const size_t table_bytes=size_t(experts)*2*sizeof(Nvfp4Projection);
    const size_t scale_bytes=size_t(experts)*204800;
    const auto overlap=[](const void* a,size_t an,const void* b,size_t bn){
        const auto ap=reinterpret_cast<uintptr_t>(a),bp=reinterpret_cast<uintptr_t>(b);
        return ap>UINTPTR_MAX-an||bp>UINTPTR_MAX-bn||
            (ap<bp+bn&&bp<ap+an);
    };
    if(overlap(source,table_bytes,target,table_bytes)||
        overlap(source,table_bytes,scales,scale_bytes)||
        overlap(target,table_bytes,scales,scale_bytes))return cudaErrorInvalidValue;
    auto status=cudaMemcpyAsync(target,source,table_bytes,cudaMemcpyDeviceToDevice,stream);
    if(status!=cudaSuccess)return status;
    nvfp4_gate_scale_view_cuda<<<dim3(40,2,experts),dim3(32,8),0,stream>>>(source,target,scales);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_scale_decode_check(uint32_t* hardware, uint32_t* direct,
    cudaStream_t stream) {
    if (!hardware || !direct) return cudaErrorInvalidValue;
    nvfp4_scale_decode_check_cuda<<<1,256,0,stream>>>(hardware,direct);
    return cudaGetLastError();
}

cudaError_t launch_fp16_expert_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, cudaStream_t stream) {
    if (count < 0 || rows <= 0 || rows > INT32_MAX - 7 || inner <= 0 || count > 65535 ||
        (count && !jobs)) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    fp16_expert_projections_cuda<<<dim3((rows + 7) / 8, count), 256, 0, stream>>>(
        jobs, rows, inner);
    return cudaGetLastError();
}
cudaError_t launch_nvfp4_mma_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int split_row, cudaStream_t stream, float* partial, int parts, bool canonical_weights,
    uint8_t* packed_inputs, uint8_t* input_scales, bool gate_codes_kmajor) {
    if (!jobs || count <= 0 || count > 65535 || rows <= 0 || rows > 2147483584 ||
        rows % 16 || inner <= 0 || inner > 2560 || inner % 64 ||
        split_row < 0 || split_row >= rows || split_row % 64 ||
        parts < 1 || parts > 8 || (parts > 1 && !partial) ||
        ((packed_inputs == nullptr) != (input_scales == nullptr)) || (packed_inputs && parts != 1))
        return cudaErrorInvalidValue;
    if (gate_codes_kmajor && (!canonical_weights || rows != 1280 || inner != 2560))
        return cudaErrorInvalidValue;
    static const bool wide_cta = [] {
        const char* option = std::getenv("MOLLM_CUDA_NVFP4_MMA_WARPS");
        return option && std::strcmp(option, "8") == 0;
    }();
    // Never share activation quantization across a gate/up global-scale
    // boundary. Eight warps halve redundant quantization without split-K.
    const int warps = (parts > 1 || wide_cta) && split_row % 128 == 0 ? 8 : 4;
    const int tile_rows = warps * 16;
    if (packed_inputs) {
        const int variants = split_row ? 2 : 1;
        quantize_nvfp4_jobs_cuda<<<(count * variants * (inner / 16) + 127) / 128,128,0,stream>>>(
            jobs,count,inner,variants,packed_inputs,input_scales,nullptr,1);
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        if (gate_codes_kmajor)
            nvfp4_native_mma_cuda<false,true,true,true><<<dim3((rows+tile_rows-1)/tile_rows,count),warps*32,0,stream>>>(
                jobs,rows,inner,split_row,nullptr,nullptr,1,packed_inputs,input_scales);
        else if (canonical_weights)
            nvfp4_native_mma_cuda<false,true,true><<<dim3((rows+tile_rows-1)/tile_rows,count),warps*32,0,stream>>>(
                jobs,rows,inner,split_row,nullptr,nullptr,1,packed_inputs,input_scales);
        else nvfp4_native_mma_cuda<false,false,true><<<dim3((rows+tile_rows-1)/tile_rows,count),warps*32,0,stream>>>(
                jobs,rows,inner,split_row,nullptr,nullptr,1,packed_inputs,input_scales);
    } else if (gate_codes_kmajor)
        nvfp4_native_mma_cuda<false,true,false,true><<<dim3((rows+tile_rows-1)/tile_rows,count,parts),warps*32,0,stream>>>(
            jobs,rows,inner,split_row,partial);
    else if (canonical_weights)
        nvfp4_native_mma_cuda<false, true><<<dim3((rows + tile_rows - 1) / tile_rows, count, parts), warps * 32, 0, stream>>>(
            jobs, rows, inner, split_row, partial);
    else
        nvfp4_native_mma_cuda<<<dim3((rows + tile_rows - 1) / tile_rows, count, parts), warps * 32, 0, stream>>>(
            jobs, rows, inner, split_row, partial);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (parts > 1) combine_nvfp4_native_parts_cuda<<<
        (static_cast<size_t>(rows) * count + 255) / 256, 256, 0, stream>>>(
        jobs, partial, rows, count, parts, split_row);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_mma_batched8(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int split_row, cudaStream_t stream) {
    if (!jobs || count <= 0 || count % 8 || count / 8 > 65535 ||
        rows <= 0 || rows > 2147483584 || rows % 16 || inner <= 0 ||
        inner > 2560 || inner % 64 || split_row < 0 || split_row >= rows || split_row % 64)
        return cudaErrorInvalidValue;
    nvfp4_native_batched8_mma_cuda<<<dim3((rows + 63) / 64, count / 8), 128, 0, stream>>>(
        jobs, rows, inner, split_row, nullptr);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_batch_plan_diagnostic(const Nvfp4Projection* jobs,int count,
    Nvfp4MmaBatch* batches,int columns,bool parallel,cudaStream_t stream) {
    if(!jobs||!batches||count<=0||count>65535||(columns!=8&&columns!=16&&columns!=32))return cudaErrorInvalidValue;
    auto status=cudaMemsetAsync(batches,0xff,size_t(count)*sizeof(*batches),stream);
    if(status!=cudaSuccess)return status;
    const auto launch=[&](auto width){constexpr int c=decltype(width)::value;
        if(parallel)build_nvfp4_batches_warp_cuda<c><<<(count+127)/128,128,0,stream>>>(jobs,count,batches);
        else build_nvfp4_batches_cuda<c><<<(count+127)/128,128,0,stream>>>(jobs,count,batches);
    };
    if(columns==32)launch(std::integral_constant<int,32>{});
    else if(columns==16)launch(std::integral_constant<int,16>{});
    else launch(std::integral_constant<int,8>{});
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_mma_activation_diagnostic(const Nvfp4Projection* jobs,
    const Nvfp4Projection* down_jobs,int count,Nvfp4MmaBatch* batches,
    uint8_t* packed_inputs,uint8_t* input_scales,float* activated,
    uint8_t* down_packed,uint8_t* down_scales,bool publish_activated,cudaStream_t stream,bool stage_inputs,int input_pitch,bool pad_logits,bool skew_weights,
    const Nvfp4MmaBatch* gate_plan,int gate_batches,bool transposed_scales,bool kmajor_input_scales,bool compact_logits,
    const float* token_input,int token_count,float token_global,int* token_indices,bool pair_words,bool weight_codes_kmajor,bool token_codes_kmajor,bool hoist_token_sources) {
    const bool token_inputs=token_input!=nullptr;
    if(!jobs||!down_jobs||!batches||!packed_inputs||!input_scales||!activated||
        !down_packed||!down_scales||count<=0||count>65535||
        (token_inputs&&(!compact_logits||token_count<1||token_count>2048||!token_indices||!std::isfinite(token_global)||token_global<=0))||
        (!token_inputs&&(token_count||token_indices||pair_words))||
        (weight_codes_kmajor&&!pair_words)||
        (token_codes_kmajor&&(!pair_words||!weight_codes_kmajor))||
        (hoist_token_sources&&!token_codes_kmajor)||
        (pair_words&&(reinterpret_cast<uintptr_t>(packed_inputs)&7))||
        (compact_logits&&(stage_inputs||pad_logits||skew_weights||gate_plan||!transposed_scales||!kmajor_input_scales))||
        (stage_inputs&&input_pitch!=32&&input_pitch!=48)||(pad_logits&&stage_inputs)||
        (skew_weights&&(pad_logits||stage_inputs))||gate_batches<0||gate_batches>count||
        ((gate_plan!=nullptr)!=(gate_batches>0))||
        (gate_plan&&(stage_inputs||pad_logits||skew_weights))||
        ((transposed_scales||kmajor_input_scales)&&(gate_plan||pad_logits||skew_weights))||
        (stage_inputs&&(transposed_scales||kmajor_input_scales)&&
            !(transposed_scales&&kmajor_input_scales&&input_pitch==32)))return cudaErrorInvalidValue;
    struct Span { uintptr_t begin,end;bool writable; } spans[10];
    const void* pointers[]={jobs,down_jobs,batches,packed_inputs,input_scales,
        activated,down_packed,down_scales,token_input,token_indices};
    const size_t lengths[]={size_t(count)*sizeof(*jobs),size_t(count)*sizeof(*down_jobs),
        size_t(count)*sizeof(*batches),token_inputs?size_t(token_count)*1280:size_t(count)*2560,
        token_inputs?size_t(token_count)*160:size_t(count)*320,
        size_t(count)*640*sizeof(float),size_t(count)*320,size_t(count)*40,
        token_inputs?size_t(token_count)*2560*sizeof(float):0,token_inputs?size_t(count)*sizeof(int):0};
    const size_t alignment[]={alignof(Nvfp4Projection),alignof(Nvfp4Projection),
        alignof(Nvfp4MmaBatch),4,4,alignof(float),1,1,16,alignof(int)};
    for(int i=0;i<10;++i){
        const uintptr_t begin=reinterpret_cast<uintptr_t>(pointers[i]);
        if(begin%alignment[i]||lengths[i]>UINTPTR_MAX-begin)return cudaErrorInvalidValue;
        spans[i]={begin,begin+lengths[i],i>=2&&i!=8};
        for(int j=0;j<i;++j)
            if(lengths[i]&&lengths[j]&&(spans[i].writable||spans[j].writable)&&begin<spans[j].end&&
                spans[j].begin<spans[i].end)return cudaErrorInvalidValue;
    }
    if(gate_plan) {
        const uintptr_t begin=reinterpret_cast<uintptr_t>(gate_plan);
        const size_t bytes=size_t(gate_batches)*sizeof(*gate_plan);
        if(begin%alignof(Nvfp4MmaBatch)||bytes>UINTPTR_MAX-begin)return cudaErrorInvalidValue;
        for(const auto& span:spans)if(span.writable&&begin<span.end&&span.begin<begin+bytes)
            return cudaErrorInvalidValue;
    }
    auto status=cudaMemsetAsync(batches,0xff,count*sizeof(*batches),stream);
    if(status!=cudaSuccess)return status;
    build_nvfp4_batches_warp_cuda<32><<<(count+127)/128,128,0,stream>>>(jobs,count,batches);
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    if(token_inputs) {
        nvfp4_token_input_map_cuda<<<(count+127)/128,128,0,stream>>>(jobs,count,token_input,token_indices);
        if(token_codes_kmajor)quantize_nvfp4_jobs_vector_cuda<true,true,true,true><<<40*((token_count+31)/32),128,0,stream>>>(
            nullptr,token_count,2560,1,packed_inputs,input_scales,nullptr,1,token_input,token_global);
        else if(pair_words)quantize_nvfp4_jobs_vector_cuda<true,true,true><<<40*((token_count+31)/32),128,0,stream>>>(
            nullptr,token_count,2560,1,packed_inputs,input_scales,nullptr,1,token_input,token_global);
        else quantize_nvfp4_jobs_vector_cuda<true,true><<<40*((token_count+31)/32),128,0,stream>>>(
            nullptr,token_count,2560,1,packed_inputs,input_scales,nullptr,1,token_input,token_global);
    }
    else if(kmajor_input_scales)quantize_nvfp4_jobs_vector_cuda<true><<<40*((count*2+31)/32),128,0,stream>>>(
        jobs,count,2560,2,packed_inputs,input_scales,batches,1);
    else quantize_nvfp4_jobs_vector_cuda<><<<(count*2*160+127)/128,128,0,stream>>>(
        jobs,count,2560,2,packed_inputs,input_scales,batches,1);
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    if(hoist_token_sources&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,false,false,true,true,true,true,true,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales,token_indices,token_count);
    else if(hoist_token_sources)nvfp4_n32_fused_activation_cuda<false,false,32,false,false,true,true,true,true,true,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales,token_indices,token_count);
    else if(token_codes_kmajor&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,false,false,true,true,true,true,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales,token_indices,token_count);
    else if(token_codes_kmajor)nvfp4_n32_fused_activation_cuda<false,false,32,false,false,true,true,true,true,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales,token_indices,token_count);
    else if(weight_codes_kmajor&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,false,false,true,true,true,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales,token_indices,token_count);
    else if(weight_codes_kmajor)nvfp4_n32_fused_activation_cuda<false,false,32,false,false,true,true,true,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales,token_indices,token_count);
    else if(pair_words&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,false,false,true,true,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales,token_indices,token_count);
    else if(pair_words)nvfp4_n32_fused_activation_cuda<false,false,32,false,false,true,true,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales,token_indices,token_count);
    else if(token_inputs&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,false,false,true,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales,token_indices,token_count);
    else if(token_inputs)nvfp4_n32_fused_activation_cuda<false,false,32,false,false,true,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales,token_indices,token_count);
    else if(compact_logits&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,false,false,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(compact_logits)nvfp4_n32_fused_activation_cuda<false,false,32,false,false,true,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(stage_inputs&&kmajor_input_scales&&transposed_scales&&publish_activated)nvfp4_n32_fused_activation_cuda<true,true,32,false,false,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(stage_inputs&&kmajor_input_scales&&transposed_scales)nvfp4_n32_fused_activation_cuda<false,true,32,false,false,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(kmajor_input_scales&&transposed_scales&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,false,false,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(kmajor_input_scales&&transposed_scales)nvfp4_n32_fused_activation_cuda<false,false,32,false,false,true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(kmajor_input_scales&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,false,false,false,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(kmajor_input_scales)nvfp4_n32_fused_activation_cuda<false,false,32,false,false,false,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(transposed_scales&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,false,false,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(transposed_scales)nvfp4_n32_fused_activation_cuda<false,false,32,false,false,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(gate_plan&&publish_activated)nvfp4_n32_fused_activation_cuda<true><<<dim3(10,gate_batches),256,0,stream>>>(
        jobs,down_jobs,gate_plan,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(gate_plan)nvfp4_n32_fused_activation_cuda<false><<<dim3(10,gate_batches),256,0,stream>>>(
        jobs,down_jobs,gate_plan,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(skew_weights&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,false,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(skew_weights)nvfp4_n32_fused_activation_cuda<false,false,32,false,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(pad_logits&&publish_activated)nvfp4_n32_fused_activation_cuda<true,false,32,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(pad_logits)nvfp4_n32_fused_activation_cuda<false,false,32,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(stage_inputs&&input_pitch==48&&publish_activated)nvfp4_n32_fused_activation_cuda<true,true,48><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(stage_inputs&&input_pitch==48)nvfp4_n32_fused_activation_cuda<false,true,48><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(stage_inputs&&publish_activated)nvfp4_n32_fused_activation_cuda<true,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(stage_inputs)nvfp4_n32_fused_activation_cuda<false,true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else if(publish_activated)nvfp4_n32_fused_activation_cuda<true><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    else nvfp4_n32_fused_activation_cuda<false><<<dim3(10,count),256,0,stream>>>(
        jobs,down_jobs,batches,packed_inputs,input_scales,count,activated,down_packed,down_scales);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_mma_grouped(const Nvfp4Projection* jobs, int count,
    Nvfp4MmaBatch* batches, int rows, int inner, int split_row, cudaStream_t stream,
    uint8_t* packed_inputs, uint8_t* input_scales, int minimum, bool canonical_weights,
    bool prepare_singles, Nvfp4MmaBatch* compact_batches, int compact_grid_y, int max_batch_size,
    bool specialized_prepared, bool inputs_prepared, bool async_weights,bool output_bf16,bool early_inputs,bool parallel_plan,bool vector_quantize,bool gate_codes_kmajor) {
    if (gate_codes_kmajor && (!canonical_weights || rows != 1280 || inner != 2560))
        return cudaErrorInvalidValue;
    if(vector_quantize&&(!output_bf16||!async_weights||rows!=1280||inner!=2560||
        split_row!=640||max_batch_size!=32||minimum!=1))return cudaErrorInvalidValue;
    if(parallel_plan&&(compact_batches||specialized_prepared))return cudaErrorInvalidValue;
    constexpr int requested_batch_warps=4;
    if(early_inputs&&(!output_bf16||!async_weights||rows!=1280||inner!=2560||
        split_row!=640||max_batch_size!=32||minimum!=1||requested_batch_warps!=4))return cudaErrorInvalidValue;
    if(output_bf16) {
        const bool down=rows==2560&&inner==640&&!split_row&&max_batch_size==8&&!async_weights;
        const bool gate=rows==1280&&inner==2560&&split_row==640&&max_batch_size==32&&async_weights;
        if(!canonical_weights||!packed_inputs||compact_batches||specialized_prepared||
            prepare_singles||minimum!=1||(!down&&!gate)||requested_batch_warps!=4)
            return cudaErrorInvalidValue;
    }
    if(async_weights && (!canonical_weights || !packed_inputs || compact_batches ||
        specialized_prepared || max_batch_size!=32 || rows!=1280 || inner!=2560))
        return cudaErrorInvalidValue;
    if (!jobs || !batches || count <= 0 || count > 65535 || rows <= 0 ||
        rows > 2147483584 || rows % 16 || inner <= 0 || inner > 2560 ||
        inner % 64 || split_row < 0 || split_row >= rows || split_row % 64 ||
        ((packed_inputs == nullptr) != (input_scales == nullptr)) ||
        (prepare_singles && !packed_inputs) ||
        (inputs_prepared && (!packed_inputs || split_row)) || minimum < 1 || minimum > 8 ||
        compact_grid_y < 0 || compact_grid_y > count ||
        (!compact_batches && compact_grid_y != 0) ||
        (max_batch_size != 8 && max_batch_size != 16 && max_batch_size != 32) ||
        ((max_batch_size > 8 || specialized_prepared) && !packed_inputs))
        return cudaErrorInvalidValue;
    const auto dispatch_code_layout = [&](auto code_layout) -> cudaError_t {
    constexpr bool kmajor_codes = decltype(code_layout)::value;
    auto status = cudaMemsetAsync(batches, 0xff, count * sizeof(*batches), stream);
    if (status != cudaSuccess) return status;
    if (compact_batches) {
        status = cudaMemsetAsync(compact_batches, 0, sizeof(*compact_batches), stream);
        if (status != cudaSuccess) return status;
    }
    if(parallel_plan){
        if(max_batch_size==32)build_nvfp4_batches_warp_cuda<32><<<(count+127)/128,128,0,stream>>>(jobs,count,batches);
        else if(max_batch_size==16)build_nvfp4_batches_warp_cuda<16><<<(count+127)/128,128,0,stream>>>(jobs,count,batches);
        else build_nvfp4_batches_warp_cuda<8><<<(count+127)/128,128,0,stream>>>(jobs,count,batches);
    } else if (max_batch_size == 32)
        build_nvfp4_batches_cuda<32><<<(count + 127) / 128, 128, 0, stream>>>(jobs, count, batches, compact_batches, minimum);
    else if (max_batch_size == 16)
        build_nvfp4_batches_cuda<16><<<(count + 127) / 128, 128, 0, stream>>>(jobs, count, batches, compact_batches, minimum);
    else build_nvfp4_batches_cuda<8><<<(count + 127) / 128, 128, 0, stream>>>(jobs, count, batches, compact_batches, minimum);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    constexpr int batch_warps=4;
    const int batch_tile_rows=batch_warps*16;
    const dim3 batch_grid((static_cast<uint64_t>(rows)+batch_tile_rows-1)/batch_tile_rows,
        compact_batches ? (compact_grid_y ? compact_grid_y : (count + max_batch_size - 1) / max_batch_size) : count);
    if (packed_inputs && !inputs_prepared) {
        const int variants = split_row ? 2 : 1;
        if(vector_quantize)quantize_nvfp4_jobs_vector_cuda<<<(count * variants * (inner / 16) + 127) / 128,128,0,stream>>>(
            jobs,count,inner,variants,packed_inputs,input_scales,batches,minimum);
        else quantize_nvfp4_jobs_cuda<<<(count * variants * (inner / 16) + 127) / 128,128,0,stream>>>(
            jobs, count, inner, variants, packed_inputs, input_scales, batches, prepare_singles ? 1 : minimum);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    if (max_batch_size > 8 || specialized_prepared) {
        const auto launch_prepared = [&](auto columns, auto compact, auto persistent, dim3 grid, int offset) {
            constexpr int width = decltype(columns)::value;
            constexpr bool grouped = decltype(compact)::value, loop = decltype(persistent)::value;
            const auto* plan = compact_batches ? compact_batches + 1 : batches;
            const int* plan_count = compact_batches ? &compact_batches[0].first : nullptr;
            if (canonical_weights)
                nvfp4_prepared_batched_mma_cuda<width, true, grouped, loop, kmajor_codes><<<grid, batch_warps*32, 0, stream>>>(
                    jobs, rows, inner, split_row, plan, packed_inputs, input_scales, count, minimum, plan_count, offset);
            else nvfp4_prepared_batched_mma_cuda<width, false, grouped, loop><<<grid, batch_warps*32, 0, stream>>>(
                    jobs, rows, inner, split_row, plan, packed_inputs, input_scales, count, minimum, plan_count, offset);
        };
        const auto launch_width = [&](auto columns) {
            if (compact_batches) {
                launch_prepared(columns, std::true_type{}, std::false_type{}, batch_grid, 0);
                auto result = cudaGetLastError();
                if (result != cudaSuccess) return result;
                launch_prepared(columns, std::true_type{}, std::true_type{},
                    dim3(batch_grid.x, batch_grid.y < 64 ? batch_grid.y : 64), batch_grid.y);
            } else launch_prepared(columns, std::false_type{}, std::false_type{}, batch_grid, 0);
            return cudaGetLastError();
        };
        if(async_weights) {
            if(early_inputs)nvfp4_n32_async_weights_cuda<true,true,kmajor_codes><<<batch_grid,128,0,stream>>>(
                jobs,split_row,batches,packed_inputs,input_scales,count,minimum);
            else if(output_bf16)nvfp4_n32_async_weights_cuda<true,false,kmajor_codes><<<batch_grid,128,0,stream>>>(
                jobs,split_row,batches,packed_inputs,input_scales,count,minimum);
            else nvfp4_n32_async_weights_cuda<false,false,kmajor_codes><<<batch_grid,128,0,stream>>>(
                jobs,split_row,batches,packed_inputs,input_scales,count,minimum);
            status=cudaGetLastError();
        } else status = max_batch_size == 32 ? launch_width(std::integral_constant<int, 32>{})
            : max_batch_size == 16 ? launch_width(std::integral_constant<int, 16>{})
            : launch_width(std::integral_constant<int, 8>{});
        if (status != cudaSuccess) return status;
    } else if(output_bf16) {
        nvfp4_native_batched8_mma_cuda<true,true,false,false,true,kmajor_codes><<<batch_grid,batch_warps*32,0,stream>>>(
            jobs,rows,inner,split_row,batches,packed_inputs,input_scales,count,minimum);
    } else if (compact_batches) {
        const auto* plan = compact_batches + 1;
        const int* plan_count = &compact_batches[0].first;
        const auto launch_compact = [&](auto persistent, dim3 grid, int offset) {
            constexpr bool loop = decltype(persistent)::value;
            if (packed_inputs && canonical_weights)
                nvfp4_native_batched8_mma_cuda<true, true, true, loop, false, kmajor_codes><<<grid, batch_warps*32, 0, stream>>>(
                    jobs, rows, inner, split_row, plan, packed_inputs, input_scales, count, minimum, plan_count, offset);
            else if (packed_inputs)
                nvfp4_native_batched8_mma_cuda<true, false, true, loop><<<grid, batch_warps*32, 0, stream>>>(
                    jobs, rows, inner, split_row, plan, packed_inputs, input_scales, count, minimum, plan_count, offset);
            else if (canonical_weights)
                nvfp4_native_batched8_mma_cuda<false, true, true, loop, false, kmajor_codes><<<grid, batch_warps*32, 0, stream>>>(
                    jobs, rows, inner, split_row, plan, nullptr, nullptr, 0, minimum, plan_count, offset);
            else nvfp4_native_batched8_mma_cuda<false, false, true, loop><<<grid, batch_warps*32, 0, stream>>>(
                    jobs, rows, inner, split_row, plan, nullptr, nullptr, 0, minimum, plan_count, offset);
        };
        launch_compact(std::false_type{}, batch_grid, 0);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        // Never assume the host hint bounds device batch counts. Most resident
        // calls fit the one-shot grid; arbitrary small groups use this fallback.
        launch_compact(std::true_type{}, dim3(batch_grid.x, batch_grid.y < 64 ? batch_grid.y : 64), batch_grid.y);
    } else if (packed_inputs) {
        if (canonical_weights)
            nvfp4_native_batched8_mma_cuda<true, true, false, false, false, kmajor_codes><<<batch_grid, batch_warps*32, 0, stream>>>(
                jobs, rows, inner, split_row, batches, packed_inputs, input_scales, count, minimum);
        else nvfp4_native_batched8_mma_cuda<true><<<batch_grid, batch_warps*32, 0, stream>>>(
            jobs, rows, inner, split_row, batches, packed_inputs, input_scales, count, minimum);
    } else if (canonical_weights)
        nvfp4_native_batched8_mma_cuda<false, true, false, false, false, kmajor_codes><<<batch_grid, batch_warps*32, 0, stream>>>(
            jobs, rows, inner, split_row, batches, nullptr, nullptr, 0, minimum);
    else nvfp4_native_batched8_mma_cuda<<<batch_grid, batch_warps*32, 0, stream>>>(
        jobs, rows, inner, split_row, batches, nullptr, nullptr, 0, minimum);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (minimum > 1) {
        if (prepare_singles && canonical_weights)
            nvfp4_native_mma_cuda<true, true, true, kmajor_codes><<<dim3((rows + 63) / 64, count), 128, 0, stream>>>(
                jobs, rows, inner, split_row, nullptr, batches, minimum, packed_inputs, input_scales);
        else if (prepare_singles)
            nvfp4_native_mma_cuda<true, false, true><<<dim3((rows + 63) / 64, count), 128, 0, stream>>>(
                jobs, rows, inner, split_row, nullptr, batches, minimum, packed_inputs, input_scales);
        else if (canonical_weights)
            nvfp4_native_mma_cuda<true, true, false, kmajor_codes><<<dim3((rows + 63) / 64, count), 128, 0, stream>>>(
                jobs, rows, inner, split_row, nullptr, batches, minimum);
        else nvfp4_native_mma_cuda<true><<<dim3((rows + 63) / 64, count), 128, 0, stream>>>(
            jobs, rows, inner, split_row, nullptr, batches, minimum);
    }
    return cudaGetLastError();
    };
    if (gate_codes_kmajor) return dispatch_code_layout(std::true_type{});
    return dispatch_code_layout(std::false_type{});
}

cudaError_t launch_repack_nvfp4_gate_codes(const Nvfp4Projection& source,
    uint8_t* target_weights, int rows, int inner, cudaStream_t stream) {
    constexpr uintptr_t bytes = 1280 * 2560 / 2;
    const uintptr_t src = reinterpret_cast<uintptr_t>(source.weight);
    const uintptr_t dst = reinterpret_cast<uintptr_t>(target_weights);
    if (rows != 1280 || inner != 2560 || source.mma_layout || !src || !dst ||
        (src & 15) || (dst & 15) || src > UINTPTR_MAX - bytes ||
        dst > UINTPTR_MAX - bytes || (src < dst + bytes && dst < src + bytes))
        return cudaErrorInvalidValue;
    repack_nvfp4_gate_codes_cuda<<<400, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(source.weight),
        reinterpret_cast<uint4*>(target_weights), source.pair_layout);
    return cudaGetLastError();
}

cudaError_t launch_repack_nvfp4_mma(const Nvfp4Projection& source,
    uint8_t* target_weights, uint8_t* target_scales, int rows, int inner,
    cudaStream_t stream) {
    if (rows <= 0 || rows > 2147483632 || inner <= 0 || inner % 64 ||
        !source.weight || !source.block_scales || !target_weights || !target_scales ||
        (reinterpret_cast<uintptr_t>(source.weight) & 3) ||
        (reinterpret_cast<uintptr_t>(target_weights) & 3)) return cudaErrorInvalidValue;
    const size_t tiles = static_cast<size_t>((rows + 15) / 16) * (inner / 64);
    const unsigned blocks = static_cast<unsigned>(
        tiles * 128 / 256 + (tiles * 128 % 256 != 0));
    repack_nvfp4_mma_cuda<<<blocks < 65535 ? blocks : 65535, 256, 0, stream>>>(
        source, reinterpret_cast<uint32_t*>(target_weights), target_scales, rows, inner, tiles);
    return cudaGetLastError();
}

void launch_group_nvfp4_jobs(const Nvfp4Projection* gate_jobs,
    const Nvfp4Projection* down_jobs, Nvfp4Projection* grouped_gate,
    Nvfp4Projection* grouped_down, int count, cudaStream_t stream, Nvfp4MmaBatch* pair_ranks, bool tiled_keys) {
    if (tiled_keys) {
        if (count <= 0 || count > 65535) return;
        if (pair_ranks) group_nvfp4_jobs_tiled_cuda<true><<<(count+127)/128,128,0,stream>>>(
            gate_jobs,down_jobs,grouped_gate,grouped_down,count,pair_ranks);
        else group_nvfp4_jobs_tiled_cuda<false><<<(count+127)/128,128,0,stream>>>(
            gate_jobs,down_jobs,grouped_gate,grouped_down,count,nullptr);
        return;
    }
    if(pair_ranks)group_nvfp4_jobs_cuda<true><<<(count+127)/128,128,count*sizeof(uintptr_t),stream>>>(
        gate_jobs,down_jobs,grouped_gate,grouped_down,count,pair_ranks);
    else group_nvfp4_jobs_cuda<false><<<(count + 127) / 128, 128, count * sizeof(uintptr_t), stream>>>(
        gate_jobs, down_jobs, grouped_gate, grouped_down, count);
}

cudaError_t nvfp4_radix_group_scratch_size(int count, size_t* bytes) {
    if (!bytes || count < 0 || count > 65535) return cudaErrorInvalidValue;
    *bytes = 0;
    if (!count) return cudaSuccess;
    cub::DoubleBuffer<uintptr_t> keys(nullptr, nullptr);
    cub::DoubleBuffer<int> indices(nullptr, nullptr);
    size_t temp_bytes = 0;
    const auto status = cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
        keys, indices, count);
    if (status != cudaSuccess) return status;
    const size_t key_stride = (static_cast<size_t>(count) * sizeof(uintptr_t) + 255) & ~size_t(255);
    const size_t index_stride = (static_cast<size_t>(count) * sizeof(int) + 255) & ~size_t(255);
    *bytes = 2 * (key_stride + index_stride) + temp_bytes;
    return cudaSuccess;
}

cudaError_t launch_group_nvfp4_jobs_radix(const Nvfp4Projection* gate_jobs,
    const Nvfp4Projection* down_jobs, Nvfp4Projection* grouped_gate,
    Nvfp4Projection* grouped_down, int count, void* scratch, size_t scratch_bytes,
    cudaStream_t stream) {
    if (count < 0 || count > 65535) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    if (!gate_jobs || !down_jobs || !grouped_gate || !grouped_down || !scratch ||
        reinterpret_cast<uintptr_t>(scratch) % 256) return cudaErrorInvalidValue;
    const size_t descriptor_bytes = static_cast<size_t>(count) * sizeof(Nvfp4Projection);
    const auto overlaps = [](const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
        const auto first = reinterpret_cast<uintptr_t>(a);
        const auto second = reinterpret_cast<uintptr_t>(b);
        return first <= second ? second - first < a_bytes : first - second < b_bytes;
    };
    if (overlaps(grouped_gate, descriptor_bytes, grouped_down, descriptor_bytes) ||
        overlaps(grouped_gate, descriptor_bytes, gate_jobs, descriptor_bytes) ||
        overlaps(grouped_gate, descriptor_bytes, down_jobs, descriptor_bytes) ||
        overlaps(grouped_down, descriptor_bytes, gate_jobs, descriptor_bytes) ||
        overlaps(grouped_down, descriptor_bytes, down_jobs, descriptor_bytes) ||
        overlaps(scratch, scratch_bytes, gate_jobs, descriptor_bytes) ||
        overlaps(scratch, scratch_bytes, down_jobs, descriptor_bytes) ||
        overlaps(scratch, scratch_bytes, grouped_gate, descriptor_bytes) ||
        overlaps(scratch, scratch_bytes, grouped_down, descriptor_bytes)) return cudaErrorInvalidValue;
    size_t required = 0;
    auto status = nvfp4_radix_group_scratch_size(count, &required);
    if (status != cudaSuccess) return status;
    if (scratch_bytes < required) return cudaErrorInvalidValue;
    auto* base = static_cast<uint8_t*>(scratch);
    const size_t key_stride = (static_cast<size_t>(count) * sizeof(uintptr_t) + 255) & ~size_t(255);
    const size_t index_stride = (static_cast<size_t>(count) * sizeof(int) + 255) & ~size_t(255);
    cub::DoubleBuffer<uintptr_t> keys(reinterpret_cast<uintptr_t*>(base),
        reinterpret_cast<uintptr_t*>(base + key_stride));
    cub::DoubleBuffer<int> indices(reinterpret_cast<int*>(base + 2 * key_stride),
        reinterpret_cast<int*>(base + 2 * key_stride + index_stride));
    const size_t header_bytes = 2 * (key_stride + index_stride);
    size_t temp_bytes = scratch_bytes - header_bytes;
    nvfp4_radix_keys_cuda<<<(count + 127) / 128, 128, 0, stream>>>(
        gate_jobs, keys.Current(), indices.Current(), count);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    status = cub::DeviceRadixSort::SortPairs(base + header_bytes, temp_bytes,
        keys, indices, count, 0, sizeof(uintptr_t) * 8, stream);
    if (status != cudaSuccess) return status;
    nvfp4_radix_scatter_cuda<<<(count + 127) / 128, 128, 0, stream>>>(
        gate_jobs, down_jobs, grouped_gate, grouped_down, indices.Current(), count);
    return cudaGetLastError();
}

void launch_moe_resident_routes(const float* logits, int rows, int experts,
    int top_k, int score_func, bool normalize, float scale,
    const Nvfp4Projection* tables, const float* input, float* gate_up,
    float* activated, float* down, int hidden, int inter,
    Nvfp4Projection* gate_jobs, Nvfp4Projection* down_jobs, float* weights, cudaStream_t stream,
    int cached_best_override,bool down_output_bf16,bool gate_output_bf16) {
    constexpr bool register_scores=false;
    const auto launch = [&](auto packed,auto gate_packed) {
    constexpr bool bf16_down=decltype(packed)::value;
    constexpr bool bf16_gate=decltype(gate_packed)::value;
    if (cached_best_override == 2 && experts <= 512)
        moe_resident_routes_cuda<16, false, false, false, bf16_down, bf16_gate><<<rows, 32, 0, stream>>>(logits, experts, top_k, score_func,
            normalize, scale, tables, input, gate_up, activated, down, hidden, inter,
            gate_jobs, down_jobs, weights);
    else if (cached_best_override == 1)
        moe_resident_routes_cuda<32, false, true, false, bf16_down, bf16_gate><<<rows, 32, 0, stream>>>(logits, experts, top_k, score_func,
            normalize, scale, tables, input, gate_up, activated, down, hidden, inter,
            gate_jobs, down_jobs, weights);
    else if (cached_best_override < 0 && register_scores && experts <= 512)
        moe_resident_routes_cuda<16, true, false, false, bf16_down, bf16_gate><<<rows, 32, 0, stream>>>(logits, experts, top_k, score_func,
            normalize, scale, tables, input, gate_up, activated, down, hidden, inter,
            gate_jobs, down_jobs, weights);
    else if (cached_best_override < 0 && register_scores)
        moe_resident_routes_cuda<32, true, false, false, bf16_down, bf16_gate><<<rows, 32, 0, stream>>>(logits, experts, top_k, score_func,
            normalize, scale, tables, input, gate_up, activated, down, hidden, inter,
            gate_jobs, down_jobs, weights);
    else
        moe_resident_routes_cuda<32, false, false, false, bf16_down, bf16_gate><<<rows, 32, 0, stream>>>(logits, experts, top_k, score_func,
        normalize, scale, tables, input, gate_up, activated, down, hidden, inter,
        gate_jobs, down_jobs, weights);
    };
    if(gate_output_bf16) {
        if(down_output_bf16)launch(std::true_type{},std::true_type{});
        else launch(std::false_type{},std::true_type{});
    } else {
        if(down_output_bf16)launch(std::true_type{},std::false_type{});
        else launch(std::false_type{},std::false_type{});
    }
}

void launch_nvfp4_projections(const Nvfp4Projection* jobs, int count,
                              int rows, int inner, cudaStream_t stream, bool mma_layout) {
    const dim3 grid((rows + 3) / 4, count);
    if (mma_layout)
        nvfp4_projections_cuda<0, true><<<grid, 128, 0, stream>>>(jobs, rows, inner);
    else
        nvfp4_projections_cuda<><<<grid, 128, 0, stream>>>(jobs, rows, inner);
}

cudaError_t launch_nvfp4_exact_subwarp(const Nvfp4Projection* jobs,int count,int rows,int inner,int lanes,cudaStream_t stream) {
    if(count<0||count>65535||rows<=0||rows>INT32_MAX-16||inner<=0||inner>INT32_MAX-128||inner%16||(lanes!=16&&lanes!=8)||(count&&!jobs))return cudaErrorInvalidValue;
    if(!count)return cudaSuccess;
    const int tile=128/lanes;
    if(lanes==16 && inner%128)nvfp4_exact_subwarp_cuda<16,true><<<dim3((rows+tile-1)/tile,count),128,0,stream>>>(jobs,rows,inner);
    else if(lanes==8 && inner%128)nvfp4_exact_subwarp_cuda<8,true><<<dim3((rows+tile-1)/tile,count),128,0,stream>>>(jobs,rows,inner);
    else if(lanes==16)nvfp4_exact_subwarp_cuda<16><<<dim3((rows+tile-1)/tile,count),128,0,stream>>>(jobs,rows,inner);
    else nvfp4_exact_subwarp_cuda<8><<<dim3((rows+tile-1)/tile,count),128,0,stream>>>(jobs,rows,inner);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_ranked_pairs(const Nvfp4Projection* jobs, int count,
    const Nvfp4MmaBatch* pair_ranks, int rows, int inner, cudaStream_t stream) {
    if(count<0||count>4096||rows<=0||rows>INT32_MAX-3||inner<=0||inner>INT32_MAX-128||inner%16||
        (count&&(!jobs||!pair_ranks)))return cudaErrorInvalidValue;
    if(!count)return cudaSuccess;
    nvfp4_projections_cuda<0,false,true,0,2,1,false,false,0,false,true,true><<<
        dim3((rows+3)/4,count),128,0,stream>>>(jobs,rows,inner,count,pair_ranks);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_inline_pairs(const Nvfp4Projection* jobs, int count,
    int rows, int inner, cudaStream_t stream) {
    if (count < 0 || count > 4096 || rows <= 0 || rows > INT32_MAX-3 ||
        inner <= 0 || inner > INT32_MAX-128 || inner % 16 || (count && !jobs)) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    nvfp4_projections_cuda<0,false,true,0,2,1,false,false,0,false,true><<<
        dim3((rows+3)/4,count),128,0,stream>>>(jobs,rows,inner,count);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_row_reuse(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int row_group, cudaStream_t stream) {
    if(count<0||count>65535||rows<=0||rows>INT32_MAX-16||inner<=0||
        inner>INT32_MAX-128||inner%16||(count&&!jobs)||(row_group!=2&&row_group!=4))
        return cudaErrorInvalidValue;
    if(!count)return cudaSuccess;
    const dim3 grid((rows+4*row_group-1)/(4*row_group),count);
    if(row_group==2)nvfp4_row_reuse_cuda<2><<<grid,128,0,stream>>>(jobs,rows,inner);
    else nvfp4_row_reuse_cuda<4><<<grid,128,0,stream>>>(jobs,rows,inner);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_gate_activation(const Nvfp4Projection* jobs, int count,
    int width, int inner, const float* route_weights, float* activated, cudaStream_t stream, int warps, int load_hint) {
    if (count < 0 || count > 65535 || width <= 0 || width > INT32_MAX / 2 - 4 ||
        width % 4 || inner <= 0 || inner > INT32_MAX - 128 || inner % 16 ||
        (count && (!jobs || !activated)) || (warps != 2 && warps != 4 && warps != 8) ||
        (load_hint != 0 && load_hint != 3)) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    if (load_hint == 3)
        nvfp4_gate_activation_cuda<3><<<dim3((width + warps - 1) / warps, count),warps*32,0,stream>>>(
            jobs,width,inner,route_weights,activated);
    else
        nvfp4_gate_activation_cuda<0><<<dim3((width + warps - 1) / warps, count),warps*32,0,stream>>>(
            jobs,width,inner,route_weights,activated);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_route_sum(const Nvfp4Projection* jobs,int count,int rows,
    int inner,float* output,cudaStream_t stream,int rows_per_cta) {
    if(!jobs||!output||count<1||count>16||rows<=0||rows>INT32_MAX-1||inner<=0||inner>INT32_MAX-128||inner%16||
        (rows_per_cta!=1 && rows_per_cta!=2))
        return cudaErrorInvalidValue;
    if(rows_per_cta==2)nvfp4_route_sum_cuda<2><<<(rows+1)/2,count*64,0,stream>>>(jobs,rows,inner,count,output);
    else nvfp4_route_sum_cuda<1><<<rows,count*32,0,stream>>>(jobs,rows,inner,count,output);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_tiled_scale_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, cudaStream_t stream) {
    if (count < 0 || count > 65535 || rows <= 0 || rows > INT32_MAX - 3 || rows % 4 ||
        inner <= 0 || inner > INT32_MAX - 128 || inner % 16 || (count && !jobs)) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,false,false,false,0,false,false,false,false,false,false,true>
        <<<dim3(rows/4,count),128,0,stream>>>(jobs,rows,inner);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_gate_code_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, bool pipeline_inputs, cudaStream_t stream) {
    if (!jobs || count < 1 || count > 65535 || rows != 1280 || inner != 2560 ||
        reinterpret_cast<uintptr_t>(jobs) % alignof(Nvfp4Projection))
        return cudaErrorInvalidValue;
    const dim3 grid(320, count);
    if (pipeline_inputs)
        nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,false,false,false,0,false,false,false,false,false,false,false,true>
            <<<grid,128,0,stream>>>(jobs,rows,inner);
    else
        nvfp4_projections_cuda<0,false,false,0,0,1,false,false,0,false,false,false,0,false,false,false,false,false,false,false,true>
            <<<grid,128,0,stream>>>(jobs,rows,inner);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_pipelined_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, bool pipeline_inputs, cudaStream_t stream, bool cache_input, int load_hint,
    bool direct_fp4, bool fixed_inner, bool warp_packet, bool aligned_packets, bool direct_scale, bool route_major, bool pair_only, bool broadcast_scale) {
    if (count < 0 || count > 65535 || rows <= 0 || rows > INT32_MAX - 3 ||
        inner <= 0 || inner > INT32_MAX - 128 || inner % 16 || (count && !jobs)) return cudaErrorInvalidValue;
    if (cache_input && (!pipeline_inputs || inner > 4096)) return cudaErrorInvalidValue;
    if (load_hint < 0 || load_hint > 11 || (load_hint && (!pipeline_inputs || cache_input))) return cudaErrorInvalidValue;
    if (direct_fp4 && (!pipeline_inputs || cache_input || load_hint)) return cudaErrorInvalidValue;
    if (fixed_inner && (!pipeline_inputs || cache_input || load_hint || direct_fp4 ||
        (inner != 640 && inner != 2560))) return cudaErrorInvalidValue;
    if(warp_packet&&(!pipeline_inputs||cache_input||load_hint||direct_fp4||fixed_inner||inner%128))
        return cudaErrorInvalidValue;
    if(aligned_packets&&(!pipeline_inputs||cache_input||load_hint||direct_fp4||fixed_inner||warp_packet))
        return cudaErrorInvalidValue;
    if(direct_scale&&(!pipeline_inputs||cache_input||load_hint||direct_fp4||fixed_inner||warp_packet||aligned_packets))
        return cudaErrorInvalidValue;
    if(route_major&&(!pipeline_inputs||cache_input||load_hint||direct_fp4||fixed_inner||warp_packet||aligned_packets||direct_scale||rows>65535*4))
        return cudaErrorInvalidValue;
    if(pair_only&&(!pipeline_inputs||cache_input||load_hint||direct_fp4||fixed_inner||warp_packet||aligned_packets||direct_scale||route_major))
        return cudaErrorInvalidValue;
    if(broadcast_scale && (!pipeline_inputs || cache_input || load_hint || direct_fp4 || fixed_inner ||
        warp_packet || aligned_packets || direct_scale || route_major || pair_only || inner%128)) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    const dim3 grid((rows + 3) / 4, count);
    if(broadcast_scale)
        nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,false,false,false,0,false,false,false,false,false,true><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if(pair_only)
        nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,false,false,false,0,false,false,false,false,true><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if(route_major)
        nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,false,false,false,0,false,false,false,true><<<dim3(count,(rows+3)/4),128,0,stream>>>(jobs,rows,inner);
    else if(direct_scale)
        nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,false,false,false,0,false,false,true><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if(aligned_packets)
        nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,false,false,false,0,false,true><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if(warp_packet)
        nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,false,false,false,0,true><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (fixed_inner && inner == 640)
        nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,false,false,false,640><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (fixed_inner)
        nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,false,false,false,2560><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (direct_fp4)
        nvfp4_projections_cuda<0,false,false,0,2,1,false,false,0,true><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 11)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 11><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 10)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 10><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 9)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 9><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 8)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 8><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 7)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 7><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 6)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 6><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 5)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 5><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 4)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 4><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 3)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 3><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 1)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 1><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (load_hint == 2)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, false, 2><<<grid,128,0,stream>>>(jobs,rows,inner);
    else if (cache_input)
        nvfp4_projections_cuda<0, false, false, 0, 2, 1, false, true><<<grid,128,inner*sizeof(float),stream>>>(jobs,rows,inner);
    else if (pipeline_inputs)
        nvfp4_projections_cuda<0, false, false, 0, 2><<<grid,128,0,stream>>>(jobs,rows,inner);
    else
        nvfp4_projections_cuda<0, false, false, 0, 1><<<grid,128,0,stream>>>(jobs,rows,inner);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_native_pipelined_projections(const Nvfp4Projection* jobs,
    int count, int rows, int inner, cudaStream_t stream) {
    if (count < 0 || count > 65535 || rows <= 0 || rows > INT32_MAX - 3 ||
        inner <= 0 || inner > INT32_MAX - 128 || inner % 64 || (count && !jobs))
        return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    nvfp4_projections_cuda<0, true, false, 0, 2><<<dim3((rows + 3) / 4, count),128,0,stream>>>(jobs,rows,inner);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_interleaved_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, cudaStream_t stream) {
    if (count < 0 || count > 65535 || rows <= 0 || rows > INT32_MAX - 3 ||
        inner <= 0 || inner > INT32_MAX - 128 || inner % 16 || (count && !jobs))
        return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    nvfp4_projections_cuda<0, false, false, 0, 2, 1, true><<<
        dim3((rows + 3) / 4, count), 128, 0, stream>>>(jobs, rows, inner);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_cta_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int warps, cudaStream_t stream) {
    if (count < 0 || count > 65535 || rows <= 0 || rows > INT32_MAX - 16 ||
        inner <= 0 || inner > INT32_MAX - 128 || inner % 16 || (count && !jobs) ||
        (warps != 1 && warps != 2 && warps != 4 && warps != 8 && warps != 16))
        return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    nvfp4_projections_cuda<0, false, false, 0, 2><<<
        dim3((rows + warps - 1) / warps, count), warps * 32, 0, stream>>>(jobs, rows, inner);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_deep_pipelined_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int depth, cudaStream_t stream) {
    if (count < 0 || count > 65535 || rows <= 0 || rows > INT32_MAX - 3 ||
        inner <= 0 || inner > INT32_MAX - 512 || inner % 16 || (count && !jobs) ||
        (depth != 2 && depth != 4)) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    const dim3 grid((rows + 3) / 4, count);
    if (depth == 2)
        nvfp4_projections_cuda<0, false, false, 0, 2, 2><<<grid,128,0,stream>>>(jobs,rows,inner);
    else
        nvfp4_projections_cuda<0, false, false, 0, 2, 4><<<grid,128,0,stream>>>(jobs,rows,inner);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_paired_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, cudaStream_t stream, bool mma_layout) {
    if (count < 0 || count > 65535 || rows <= 0 || rows > INT32_MAX - 3 ||
        inner <= 0 || inner % (mma_layout ? 64 : 16) || (count && !jobs))
        return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    const dim3 grid((rows + 3) / 4, (count + 1) / 2);
    if (mma_layout)
        nvfp4_projections_cuda<0, true, true><<<grid, 128, 0, stream>>>(jobs, rows, inner, count);
    else
        nvfp4_projections_cuda<0, false, true><<<grid, 128, 0, stream>>>(jobs, rows, inner, count);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_grouped_pairs(const Nvfp4Projection* jobs, int count,
    Nvfp4MmaBatch* plan, int rows, int inner, cudaStream_t stream, bool mma_layout) {
    if (count < 0 || count > 4096 || rows <= 0 || rows > INT32_MAX - 3 ||
        inner <= 0 || inner % (mma_layout ? 64 : 16) || (count && (!jobs || !plan)))
        return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    nvfp4_pair_plan_cuda<<<1, 32, 0, stream>>>(jobs, plan, count);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    const dim3 pair_grid((rows + 3) / 4, (count + 1) / 2);
    const dim3 grid((rows + 3) / 4, count);
    if (mma_layout)
        nvfp4_projections_cuda<0, true, true, 2><<<pair_grid, 128, 0, stream>>>(jobs, rows, inner, count, plan);
    else
        nvfp4_projections_cuda<0, false, true, 2><<<pair_grid, 128, 0, stream>>>(jobs, rows, inner, count, plan);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (mma_layout)
        nvfp4_projections_cuda<0, true, false, 1><<<grid, 128, 0, stream>>>(jobs, rows, inner, count, plan);
    else
        nvfp4_projections_cuda<0, false, false, 1><<<grid, 128, 0, stream>>>(jobs, rows, inner, count, plan);
    return cudaGetLastError();
}

void launch_nvfp4_prefetch_projections(const Nvfp4Projection* jobs, int count,
    int rows, int inner, int iterations, cudaStream_t stream) {
    const dim3 grid((rows + 3) / 4, count);
    switch (iterations) {
    case 1: nvfp4_projections_cuda<1><<<grid, 128, 0, stream>>>(jobs, rows, inner); break;
    case 2: nvfp4_projections_cuda<2><<<grid, 128, 0, stream>>>(jobs, rows, inner); break;
    case 4: nvfp4_projections_cuda<4><<<grid, 128, 0, stream>>>(jobs, rows, inner); break;
    }
}

void launch_moe_activation(const float* gate, const float* up,
                            const float* route_weights, float* output,
                            int rows, int width, int stride, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * width;
    moe_activation_cuda<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        gate, up, route_weights, output, count, width, stride);
}

cudaError_t launch_nvfp4_activation_prepare(const Nvfp4Projection* jobs,
    int count, int inner, const float* gate_up, float* activated,
    uint8_t* packed, uint8_t* scales, int fusion_mode, cudaStream_t stream,bool gate_input_bf16) {
    if(gate_input_bf16&&(inner!=640||(fusion_mode!=2&&fusion_mode!=3)||
        reinterpret_cast<uintptr_t>(gate_up)%2))return cudaErrorInvalidValue;
    if (count < 0 || count > 65535 || inner <= 0 || inner % 16 ||
        fusion_mode < 0 || fusion_mode > 3 ||
        inner > INT32_MAX / 2 ||
        uint64_t(count) * (inner / 16) > INT32_MAX - 127 ||
        (count && (!jobs || !gate_up || !activated || !packed || !scales)))
        return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    const int grid = (count * (inner / 16) + 127) / 128;
    if(gate_input_bf16) {
        if(fusion_mode==2)nvfp4_activation_prepare_subgroup_cuda<true,true><<<
            (count*(inner/16)+7)/8,128,0,stream>>>(jobs,count,inner,gate_up,activated,packed,scales);
        else nvfp4_activation_prepare_subgroup_cuda<false,true><<<
            (count*(inner/16)+7)/8,128,0,stream>>>(jobs,count,inner,gate_up,activated,packed,scales);
    } else if (fusion_mode==1) nvfp4_activation_prepare_cuda<<<grid,128,0,stream>>>(
        jobs,count,inner,gate_up,activated,packed,scales);
    else if (fusion_mode==2) nvfp4_activation_prepare_subgroup_cuda<true><<<
        (count*(inner/16)+7)/8,128,0,stream>>>(jobs,count,inner,gate_up,activated,packed,scales);
    else if (fusion_mode==3) nvfp4_activation_prepare_subgroup_cuda<false><<<
        (count*(inner/16)+7)/8,128,0,stream>>>(jobs,count,inner,gate_up,activated,packed,scales);
    else {
        launch_moe_activation(gate_up,gate_up+inner,nullptr,activated,count,inner,2*inner,stream);
        const auto status=cudaGetLastError();
        if (status!=cudaSuccess) return status;
        quantize_nvfp4_jobs_cuda<<<grid,128,0,stream>>>(jobs,count,inner,1,packed,scales,nullptr,1);
    }
    return cudaGetLastError();
}

void launch_moe_combine(const float* routed, const float* shared,
                         const float* shared_gate, float* output,
                         int rows, int hidden, int top_k, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * hidden;
    moe_combine_cuda<false><<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        routed, shared, shared_gate, output, count, hidden, top_k);
}

cudaError_t launch_moe_combine_bf16(const uint16_t* routed,const float* shared,
    const float* shared_gate,float* output,int rows,int hidden,int top_k,cudaStream_t stream) {
    if(!routed||!shared||!shared_gate||!output||rows<=0||rows>4096||hidden!=2560||
        top_k<=0||top_k>32||reinterpret_cast<uintptr_t>(routed)%2||
        (reinterpret_cast<uintptr_t>(shared)|reinterpret_cast<uintptr_t>(shared_gate)|
         reinterpret_cast<uintptr_t>(output))%4)return cudaErrorInvalidValue;
    const size_t count=size_t(rows)*hidden;
    const uintptr_t out=reinterpret_cast<uintptr_t>(output);
    if(out>UINTPTR_MAX-count*4)return cudaErrorInvalidValue;
    struct Span {const void* pointer;size_t bytes;};
    const Span inputs[]={{routed,count*top_k*2},{shared,count*4},{shared_gate,size_t(rows)*4}};
    for(auto input:inputs){const uintptr_t p=reinterpret_cast<uintptr_t>(input.pointer);
        if(p>UINTPTR_MAX-input.bytes||(out<p+input.bytes&&p<out+count*4))return cudaErrorInvalidValue;}
    moe_combine_cuda<true><<<(count+255)/256,256,0,stream>>>(
        reinterpret_cast<const float*>(routed),shared,shared_gate,output,count,hidden,top_k);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_quantize_probe(const Nvfp4Projection* jobs,int count,int inner,
    int variants,uint8_t* packed,uint8_t* scales,const Nvfp4MmaBatch* batches,
    int minimum,int mode,cudaStream_t stream) {
    if (!jobs || !packed || !scales || count<=0 || count>65535 || inner<=0 ||
        inner>2560 || inner%16 || (variants!=1 && variants!=2) || minimum<1 || minimum>8 || mode<0 || mode>4)
        return cudaErrorInvalidValue;
    const int groups = count*variants*(inner/16);
    if(mode==4)quantize_nvfp4_jobs_vector_cuda<<<(groups+127)/128,128,0,stream>>>(
        jobs,count,inner,variants,packed,scales,batches,minimum);
    else if (mode==3) quantize_nvfp4_jobs_dual_cuda<<<(count*(inner/16)+127)/128,128,0,stream>>>(
        jobs,count,inner,variants,packed,scales,batches,minimum);
    else if (mode==2) quantize_nvfp4_jobs_subgroup_cuda<8><<<(groups+15)/16,128,0,stream>>>(
        jobs,count,inner,variants,packed,scales,batches,minimum);
    else if (mode==1) quantize_nvfp4_jobs_subgroup_cuda<16><<<(groups+7)/8,128,0,stream>>>(
        jobs,count,inner,variants,packed,scales,batches,minimum);
    else quantize_nvfp4_jobs_cuda<<<(groups+127)/128,128,0,stream>>>(
        jobs,count,inner,variants,packed,scales,batches,minimum);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_quantize_cached_max_probe(const Nvfp4Projection* jobs,
    int count, int inner, int variants, uint8_t* packed, uint8_t* scales,
    const Nvfp4MmaBatch* batches, int minimum, const float* input,
    int input_rows, float* maxima, cudaStream_t stream) {
    if (!jobs || !packed || !scales || !input || !maxima || count <= 0 || count > 65535 ||
        inner <= 0 || inner > 2560 || inner % 16 || (variants != 1 && variants != 2) ||
        minimum < 1 || minimum > 8 || input_rows <= 0 || input_rows > 65535)
        return cudaErrorInvalidValue;
    const int groups = inner / 16;
    nvfp4_input_block_maxima_cuda<<<(input_rows * groups + 127) / 128, 128, 0, stream>>>(
        input, maxima, input_rows, inner);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    quantize_nvfp4_jobs_cached_max_cuda<<<(count * variants * groups + 127) / 128, 128, 0, stream>>>(
        jobs, count, inner, variants, packed, scales, batches, minimum, input, input_rows, maxima);
    return cudaGetLastError();
}


}  // namespace mollm_cuda
