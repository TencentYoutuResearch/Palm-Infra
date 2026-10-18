// Standalone experimental NVFP4 Tensor Core probe. Not an inference backend.
// Fragment and scale-selector contract:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
#include "kernels/cuda/moe.h"
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <fstream>
#include <string>
#include <sys/mman.h>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}
template<class T> T* device_allocate(size_t count) {
    T* result = nullptr;
    check(cudaMalloc(&result, count * sizeof(T)));
    return result;
}
struct Arena {
    uint8_t* data;
    size_t size;
    explicit Arena(size_t bytes) : size((bytes + (2u << 20) - 1) & ~size_t((2u << 20) - 1)) {
        constexpr size_t huge = 2u << 20;
        auto* raw = static_cast<uint8_t*>(mmap(nullptr, size + huge, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (raw == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
        const uintptr_t aligned = (reinterpret_cast<uintptr_t>(raw) + huge - 1) & ~(uintptr_t(huge) - 1);
        data = reinterpret_cast<uint8_t*>(aligned);
        const size_t prefix = data - raw;
        if (prefix) munmap(raw, prefix);
        if (huge > prefix) munmap(data + size, huge - prefix);
#ifdef MADV_HUGEPAGE
        madvise(data, size, MADV_HUGEPAGE);
#endif
        if (mlock(data, size)) { std::perror("mlock"); std::exit(1); }
    }
    ~Arena() { munlock(data, size); munmap(data, size); }
};

__global__ void route_sum_control(const float* projected,float* output,int rows,int routes) {
    const int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=rows)return;
    float total=0.f;
    for(int i=0;i<routes;++i)total+=__bfloat162float(__float2bfloat16_rn(projected[size_t(i)*rows+r]));
    output[r]=total;
}

__device__ uint32_t bf16_pair(float x, float y) {
    const __nv_bfloat162_raw raw = __floats2bfloat162_rn(x, y);
    return uint32_t(raw.x) | (uint32_t(raw.y) << 16);
}
__device__ uint32_t nvfp4_bf16_weight_pair(const mollm_cuda::Nvfp4Projection& job,
    int row, int col, int rows, int inner) {
    if (row >= rows) return 0;
    const int group = col / 16;
    const size_t offset = job.pair_layout
        ? (static_cast<size_t>(row / 4) * (inner / 16) + group) * 32 + (row % 4) * 8 + (col % 16) / 2
        : static_cast<size_t>(row) * (inner / 2) + col / 2;
    const float2 decoded = __half22float2(__nv_cvt_fp4x2_to_halfraw2(job.weight[offset], __NV_E2M1));
    const float scale = __half2float(__nv_cvt_fp8_to_halfraw(
        job.block_scales[static_cast<size_t>(row) * (inner / 16) + group], __NV_E4M3));
    return bf16_pair(decoded.x * scale, decoded.y * scale);
}
__device__ void nvfp4_bf16_weight_block(const mollm_cuda::Nvfp4Projection& job,
    int row, int group, int pair, int rows, int inner, uint32_t& low, uint32_t& high) {
    low = high = 0;
    if (row >= rows) return;
    const size_t offset = job.pair_layout
        ? (static_cast<size_t>(row / 4) * (inner / 16) + group) * 32 + (row % 4) * 8
        : static_cast<size_t>(row) * (inner / 2) + group * 8;
    const auto* source = job.weight + offset;
    uint8_t lo, hi;
    if ((reinterpret_cast<uintptr_t>(source) & 7) == 0) {
        const uint64_t packed = *reinterpret_cast<const uint64_t*>(source);
        lo = packed >> (pair * 8); hi = packed >> ((pair + 4) * 8);
    } else { lo = source[pair]; hi = source[pair + 4]; }
    const float scale = __half2float(__nv_cvt_fp8_to_halfraw(
        job.block_scales[static_cast<size_t>(row) * (inner / 16) + group], __NV_E4M3));
    const float2 x = __half22float2(__nv_cvt_fp4x2_to_halfraw2(lo, __NV_E2M1));
    const float2 y = __half22float2(__nv_cvt_fp4x2_to_halfraw2(hi, __NV_E2M1));
    low = bf16_pair(x.x * scale, x.y * scale);
    high = bf16_pair(y.x * scale, y.y * scale);
}
template<int Terms, bool CacheInput = false, bool BlockWeights = false>
__global__ void nvfp4_bf16_mma_probe(const mollm_cuda::Nvfp4Projection* jobs, int rows, int inner,
    float* partial = nullptr, int parts = 1) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int first_row = (blockIdx.x * (blockDim.x / 32) + warp) * 16;
    const auto job = jobs[blockIdx.y];
    extern __shared__ uint32_t activation[];
    if constexpr (CacheInput) {
        for (int i = threadIdx.x; i < inner / 2; i += blockDim.x) {
            float2 x = make_float2(job.input[i * 2], job.input[i * 2 + 1]);
#pragma unroll
            for (int term = 0; term < Terms; ++term) {
                activation[term * (inner / 2) + i] = bf16_pair(x.x, x.y);
                if constexpr (Terms > 1) {
                    const float2 high = __bfloat1622float2(__floats2bfloat162_rn(x.x, x.y));
                    x = make_float2(x.x - high.x, x.y - high.y);
                }
            }
        }
        __syncthreads();
    }
    // All threads participate in cache initialization before any warp exits.
    if (first_row >= rows) return;
    const int row = first_row + lane / 4, pair = lane % 4;
    float accumulator[4]{};
    const int groups = inner / 16;
    const int begin = groups * blockIdx.z / parts;
    const int end = groups * (blockIdx.z + 1) / parts;
    for (int group = begin; group < end; ++group) {
        const int col = group * 16 + pair * 2;
        uint32_t a[4];
        if constexpr (BlockWeights) {
            nvfp4_bf16_weight_block(job, row, group, pair, rows, inner, a[0], a[2]);
            nvfp4_bf16_weight_block(job, row + 8, group, pair, rows, inner, a[1], a[3]);
        } else {
            const uint32_t scalar[4] = {
            nvfp4_bf16_weight_pair(job, row, col, rows, inner),
            nvfp4_bf16_weight_pair(job, row + 8, col, rows, inner),
            nvfp4_bf16_weight_pair(job, row, col + 8, rows, inner),
            nvfp4_bf16_weight_pair(job, row + 8, col + 8, rows, inner)};
            for (int i = 0; i < 4; ++i) a[i] = scalar[i];
        }
        float2 x0{}, x1{};
        if constexpr (!CacheInput) {
            x0 = make_float2(job.input[col], job.input[col + 1]);
            x1 = make_float2(job.input[col + 8], job.input[col + 9]);
        }
#pragma unroll
        for (int term = 0; term < Terms; ++term) {
            const uint32_t b0 = CacheInput ? activation[term * (inner / 2) + col / 2] : bf16_pair(x0.x, x0.y);
            const uint32_t b1 = CacheInput ? activation[term * (inner / 2) + col / 2 + 4] : bf16_pair(x1.x, x1.y);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
            if constexpr (Terms > 1 && !CacheInput) {
                const float2 h0 = __bfloat1622float2(__floats2bfloat162_rn(x0.x, x0.y));
                const float2 h1 = __bfloat1622float2(__floats2bfloat162_rn(x1.x, x1.y));
                x0 = make_float2(x0.x - h0.x, x0.y - h0.y);
                x1 = make_float2(x1.x - h1.x, x1.y - h1.y);
            }
        }
    }
    if (!pair) {
        if (parts > 1) {
            float* destination = partial + (static_cast<size_t>(blockIdx.z) * gridDim.y + blockIdx.y) * rows;
            if (row < rows) destination[row] = accumulator[0];
            if (row + 8 < rows) destination[row + 8] = accumulator[2];
        } else {
            if (row < rows) job.output[row] = (accumulator[0] * job.row_scales[row]) * job.output_scale;
            if (row + 8 < rows) job.output[row + 8] = (accumulator[2] * job.row_scales[row + 8]) * job.output_scale;
        }
    }
}
__global__ void combine_bf16_parts(const mollm_cuda::Nvfp4Projection* jobs,
    const float* partial, int rows, int count, int parts) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0;
    for (int part = 0; part < parts; ++part)
        sum += partial[(static_cast<size_t>(part) * count + blockIdx.y) * rows + row];
    const auto job = jobs[blockIdx.y];
    job.output[row] = (sum * job.row_scales[row]) * job.output_scale;
}

struct SubwarpPacket { uint16_t packed; uint8_t scale; float4 input; };
// Read-cost diagnostic: same canonical per-quad weight/scale addresses, no
// activation/dequantization/dot. Unsigned checksum is not a model output.
template<bool WithScales = true>
__global__ void nvfp4_read_probe(const mollm_cuda::Nvfp4Projection* jobs, int rows, int inner) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x % 32;
    const auto job = jobs[blockIdx.y];
    uint32_t sum = 0;
    for (int quad = lane; quad < inner / 4; quad += 32) {
        const int group = quad / 4;
        const size_t offset = job.pair_layout
            ? (size_t(row / 4) * (inner / 16) + group) * 32 + (row % 4) * 8 + (quad % 4) * 2
            : size_t(row) * (inner / 2) + quad * 2;
        const uint16_t packet = *reinterpret_cast<const uint16_t*>(job.weight + offset);
        sum += packet;
        if constexpr (WithScales)
            sum += uint32_t(job.block_scales[size_t(row) * (inner / 16) + group]) << 16;
    }
    for (int distance = 16; distance; distance >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, distance);
    if (!lane) reinterpret_cast<uint32_t*>(job.output)[row] = sum;
}
template<bool Interleaved = false>
__device__ SubwarpPacket subwarp_packet(const mollm_cuda::Nvfp4Projection& job, int row, int inner, int quad) {
        if (quad >= inner / 4) return {};
        const int group = quad / 4;
        const size_t tile = (size_t(row / 4) * (inner / 16) + group) * 36;
        const size_t offset = Interleaved ? tile + (row % 4) * 8 + (quad % 4) * 2 : job.pair_layout
            ? (static_cast<size_t>(row / 4) * (inner / 16) + group) * 32 + (row % 4) * 8 + (quad % 4) * 2
            : static_cast<size_t>(row) * (inner / 2) + quad * 2;
        const auto* source = job.weight + offset;
        const uint16_t packed = (reinterpret_cast<uintptr_t>(source) & 1) == 0
            ? *reinterpret_cast<const uint16_t*>(source) : uint16_t(source[0]) | (uint16_t(source[1]) << 8);
        const auto* x = job.input + quad * 4;
        const float4 a = (reinterpret_cast<uintptr_t>(x) & 15) == 0
            ? *reinterpret_cast<const float4*>(x) : make_float4(x[0], x[1], x[2], x[3]);
        const uint8_t scale = Interleaved ? job.weight[tile + 32 + row % 4]
            : job.block_scales[static_cast<size_t>(row) * (inner / 16) + group];
        return {packed, scale, a};
}
template<int Lanes, bool Pipeline = false, bool IntegerScale = false, bool Interleaved = false>
__global__ void nvfp4_subwarp_probe(const mollm_cuda::Nvfp4Projection* jobs, int rows, int inner) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / Lanes;
    if (row >= rows) return;
    const int lane = threadIdx.x % Lanes;
    const auto job = jobs[blockIdx.y];
    float sum = 0;
    SubwarpPacket pending{};
    if constexpr (Pipeline) pending = subwarp_packet<Interleaved>(job, row, inner, lane);
    for (int quad = lane; quad < inner / 4; quad += Lanes) {
        SubwarpPacket incoming{};
        if constexpr (Pipeline) incoming = subwarp_packet<Interleaved>(job, row, inner, quad + Lanes);
        const auto packet = Pipeline ? pending : subwarp_packet<Interleaved>(job, row, inner, quad);
        const float2 w0 = __half22float2(__nv_cvt_fp4x2_to_halfraw2(packet.packed & 255, __NV_E2M1));
        const float2 w1 = __half22float2(__nv_cvt_fp4x2_to_halfraw2(packet.packed >> 8, __NV_E2M1));
        float scale;
        if constexpr (IntegerScale) {
            const unsigned magnitude = packet.scale & 127;
            if (magnitude == 127)
                scale = __half2float(__nv_cvt_fp8_to_halfraw(packet.scale, __NV_E4M3));
            else {
                const float value = magnitude < 8 ? magnitude * (1.f / 512.f)
                    : __uint_as_float((magnitude << 20) + 0x3c000000u);
                scale = __uint_as_float(__float_as_uint(value) | ((packet.scale & 128u) << 24));
            }
        } else scale = __half2float(__nv_cvt_fp8_to_halfraw(packet.scale, __NV_E4M3));
        const float4 a = packet.input;
        float dot = a.x * w0.x;
        dot = fmaf(a.y, w0.y, dot);
        dot = fmaf(a.z, w1.x, dot);
        dot = fmaf(a.w, w1.y, dot);
        sum = fmaf(dot, scale, sum);
        if constexpr (Pipeline) pending = incoming;
    }
    const unsigned mask = __activemask();
    for (int distance = Lanes / 2; distance; distance /= 2)
        sum += __shfl_down_sync(mask, sum, distance, Lanes);
    if (!lane) job.output[row] = (sum * job.row_scales[row]) * job.output_scale;
}

template<bool Packed>
__device__ uint32_t weight_fragment(const mollm_cuda::Nvfp4Projection& job,
                                  int row, int col, int rows, int inner) {
    if (row >= rows) return 0;
    if constexpr (Packed) {
        const size_t tile = static_cast<size_t>(row / 16) * (inner / 64) + col / 64;
        const int reg = (col % 64 / 32) * 2 + row % 16 / 8;
        const int lane = (row % 8) * 4 + col % 32 / 8;
        return reinterpret_cast<const uint32_t*>(job.weight)[tile * 128 + reg * 32 + lane];
    }
    const size_t offset = job.pair_layout
        ? (static_cast<size_t>(row / 4) * (inner / 16) + col / 16) * 32 +
            (row % 4) * 8 + (col % 16) / 2
        : static_cast<size_t>(row) * (inner / 2) + col / 2;
    return *reinterpret_cast<const uint32_t*>(job.weight + offset);
}

__device__ void quantize_activation_block(const float*, uint8_t*, uint8_t*, int, float);

template<bool Packed = false, int PrefetchDistance = 0, bool FusedQuantization = false>
__global__ void nvfp4_mma_probe(const mollm_cuda::Nvfp4Projection* jobs,
    const uint8_t* activation, const uint8_t* activation_scales,
    float activation_global, int rows, int inner, float* partial = nullptr) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    if constexpr (FusedQuantization) {
        // This diagnostic targets K <= 2560. Quantization is repeated per CTA
        // but avoids a separate launch and global intermediate allocations.
        __shared__ uint32_t shared_packed[320];
        __shared__ uint32_t shared_scales[40];
        if (inner > 2560) { asm volatile("trap;"); return; }
        auto* packed = reinterpret_cast<uint8_t*>(shared_packed);
        auto* scales = reinterpret_cast<uint8_t*>(shared_scales);
        for (int ordinal = threadIdx.x; ; ordinal += blockDim.x) {
            const int block = (blockIdx.z + (ordinal / 4) * gridDim.z) * 4 + ordinal % 4;
            if (block >= inner / 16) break;
            quantize_activation_block(jobs[blockIdx.y].input, packed, scales,
                block, activation_global);
        }
        // Must precede the ragged-output warp return below.
        __syncthreads();
        activation = packed;
        activation_scales = scales;
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int base_row = (blockIdx.x * (blockDim.x / 32) + warp) * 16;
    if (base_row >= rows) return;  // Uniform across the complete warp.
    const int group = lane / 4, thread = lane % 4;
    const int row = base_row + group;
    const auto job = jobs[blockIdx.y];
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int k = blockIdx.z * 64; k < inner; k += gridDim.z * 64) {
        if constexpr (Packed && PrefetchDistance > 0) {
            const int next_k = k + PrefetchDistance * gridDim.z * 64;
            if (next_k < inner) {
                const size_t tile = static_cast<size_t>(base_row / 16) * (inner / 64)
                    + next_k / 64;
                // Each warp covers the four contiguous 128-byte register
                // planes of its future MMA tile without changing arithmetic.
                for (int reg = 0; reg < 4; ++reg) {
                    const auto* address = job.weight + (tile * 128 + reg * 32 + lane) * 4;
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(address));
                }
            }
        }
        const int col = k + thread * 8;
        const uint32_t a0 = weight_fragment<Packed>(job, row, col, rows, inner);
        const uint32_t a1 = weight_fragment<Packed>(job, row + 8, col, rows, inner);
        const uint32_t a2 = weight_fragment<Packed>(job, row, col + 32, rows, inner);
        const uint32_t a3 = weight_fragment<Packed>(job, row + 8, col + 32, rows, inner);
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(activation + col / 2);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(activation + (col + 32) / 2);
        const int scale_row = row + (thread & 1) * 8;
        const size_t scale_offset = Packed
            ? (static_cast<size_t>(scale_row / 16) * (inner / 64) + k / 64) * 64 +
                (scale_row % 8) * 8 + (scale_row % 16 / 8) * 4
            : static_cast<size_t>(scale_row) * (inner / 16) + k / 16;
        const uint32_t sa = scale_row < rows ? *reinterpret_cast<const uint32_t*>(
            job.block_scales + scale_offset) : 0;
        const uint32_t sb = *reinterpret_cast<const uint32_t*>(activation_scales + k / 16);
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10,{0,0}, %11,{0,0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    }
    // B's eight columns duplicate one decode activation. Save column zero.
    if (thread == 0) {
        if (gridDim.z > 1) {
            float* destination = partial + (blockIdx.z * gridDim.y + blockIdx.y) * rows;
            if (row < rows) destination[row] = d0;
            if (row + 8 < rows) destination[row + 8] = d2;
        } else {
            if (row < rows) job.output[row] = (d0 * activation_global) * job.row_scales[row];
            if (row + 8 < rows) job.output[row + 8] =
                (d2 * activation_global) * job.row_scales[row + 8];
        }
    }
#else
    asm volatile("trap;");
#endif
}

__global__ void combine_mma_parts(const mollm_cuda::Nvfp4Projection* jobs,
    const float* partial, float activation_global, int rows, int experts, int parts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * experts) return;
    float sum = 0;
    for (int part = 0; part < parts; ++part) sum += partial[part * rows * experts + i];
    const auto job = jobs[i / rows];
    job.output[i % rows] = (sum * activation_global) * job.row_scales[i % rows];
}

float fp8_value(uint8_t code) {
    return __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
}
float fp4_value(int code) {
    constexpr float magnitude[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return code & 8 ? -magnitude[code & 7] : magnitude[code & 7];
}
struct QuantizedActivation {
    std::vector<uint8_t> data, scales;
    std::vector<float> decoded;
    float global = 1;
    // Zero selects the original dynamic-global diagnostic. A positive scale
    // holds the global dequantization scale fixed while each block's E4M3
    // scale is still computed from its current activation maximum. This is
    // not the reciprocal multiplier accepted by trtllm.fp4_quantize.
    explicit QuantizedActivation(const std::vector<float>& input, float fixed_global = 0)
        : data(input.size() / 2), scales(input.size() / 16), decoded(input.size()) {
        if (!std::isfinite(fixed_global) || fixed_global < 0 || input.size() % 16) {
            std::fprintf(stderr, "invalid NVFP4 activation scale/shape\n"); std::exit(1);
        }
        float maximum = 0;
        for (float x : input) maximum = std::max(maximum, std::abs(x));
        global = fixed_global > 0 ? fixed_global
            : (maximum > 0 ? maximum / (6.f * 448.f) : 1.f);
        for (size_t group = 0; group < scales.size(); ++group) {
            float block_max = 0;
            for (int i = 0; i < 16; ++i)
                block_max = std::max(block_max, std::abs(input[group * 16 + i]));
            scales[group] = __nv_cvt_float_to_fp8(block_max / (global * 6.f),
                __NV_SATFINITE, __NV_E4M3);
            const float scale = fp8_value(scales[group]);
            for (int i = 0; i < 8; ++i) {
                const int offset = group * 16 + i * 2;
                const float inverse = scale > 0 ? 1.f / (scale * global) : 0;
                const uint8_t packed = __nv_cvt_float2_to_fp4x2(
                    make_float2(input[offset] * inverse, input[offset + 1] * inverse),
                    __NV_E2M1, cudaRoundNearest);
                data[group * 8 + i] = packed;
                decoded[offset] = (fp4_value(packed & 15) * scale) * global;
                decoded[offset + 1] = (fp4_value(packed >> 4) * scale) * global;
            }
        }
    }
};
template<class F> double measure(F launch, int iterations = 128) {
    for (int i = 0; i < 16; ++i) launch(i);
    check(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    check(cudaEventRecord(start, cudaStreamPerThread));
    for (int i = 0; i < iterations; ++i) launch(i);
    check(cudaEventRecord(stop, cudaStreamPerThread)); check(cudaEventSynchronize(stop));
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, start, stop));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    return milliseconds * 1000. / iterations;
}

template<class F> double measure_graph(F launch, int iterations) {
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iterations; ++i) launch(i, stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 2; ++i) check(cudaGraphLaunch(executable, stream));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    check(cudaEventRecord(start, stream));
    for (int i = 0; i < 4; ++i) check(cudaGraphLaunch(executable, stream));
    check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));
    return ms * 1000. / (4 * iterations);
}

// Fixed-global activation quantization on the GPU. One thread owns a full
// 16-value block: no cross-block reduction or CPU activation round trip.
// Explicit round-to-nearest division mirrors the independent CPU reference.
__device__ void quantize_activation_block(const float* input,
    uint8_t* packed, uint8_t* scales, int group, float global) {
    float maximum = 0;
    for (int i = 0; i < 16; ++i)
        maximum = fmaxf(maximum, fabsf(input[group * 16 + i]));
    const uint8_t code = __nv_cvt_float_to_fp8(__fdiv_rn(maximum, global * 6.f),
        __NV_SATFINITE, __NV_E4M3);
    scales[group] = code;
    const float scale = __half2float(static_cast<__half>(
        __nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
    const float inverse = scale > 0 ? __fdiv_rn(1.f, scale * global) : 0;
    for (int i = 0; i < 8; ++i) {
        const int offset = group * 16 + i * 2;
        packed[group * 8 + i] = __nv_cvt_float2_to_fp4x2(
            make_float2(input[offset] * inverse, input[offset + 1] * inverse),
            __NV_E2M1, cudaRoundNearest);
    }
}

__global__ void quantize_activation_fixed_global(const float* input,
    uint8_t* packed, uint8_t* scales, int inner, float global) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group < inner / 16)
        quantize_activation_block(input, packed, scales, group, global);
}

void run(int rows, int inner, bool pair_layout, bool benchmark, float fixed_global = 0,
         const std::vector<int>* replay_counts = nullptr) {
    if(std::getenv("MOLLM_NVFP4_N32_ASYNC_WEIGHTS_PROBE") && rows!=1280)return;
    const bool prepared_timing=std::getenv("MOLLM_NVFP4_GROUPED_PREPARED_TIMING")!=nullptr;
    if(prepared_timing && rows!=2560)return;
    const int experts = replay_counts ? static_cast<int>(replay_counts->size()) : benchmark ? 10 : 2;
    const int padded_rows = ((rows + 3) / 4) * 4;
    const size_t weight_bytes = static_cast<size_t>(padded_rows) * inner / 2;
    const size_t scale_bytes = static_cast<size_t>(padded_rows) * inner / 16;
    const size_t stride = weight_bytes + scale_bytes + padded_rows * sizeof(float);
    // Exceed the target even for weight-only checksum controls. Scales, row
    // globals and other resident allocations cannot stand in for read traffic.
    const size_t projection_payload = weight_bytes * experts;
    const int sets = benchmark ? std::max<size_t>(replay_counts ? 2 : 8, (128ull << 20) / projection_payload + 1) : 1;
    Arena resident(stride * experts * sets);
    std::vector<float> original(inner);
    for (int k = 0; k < inner; ++k) original[k] =
        (static_cast<int>((k * 13 + 7) % 101) - 50) / 37.f;
    if (!benchmark && rows == 3) std::fill(original.begin(), original.end(), 0.f);
    QuantizedActivation quantized(original, fixed_global);
    if (fixed_global > 0 && quantized.global != fixed_global) {
        std::fprintf(stderr, "calibrated global scale was recomputed\n"); std::exit(1);
    }
    if (fixed_global > 0)
        std::printf("Fixed-global activation check: rows=%d K=%d pair=%d scale=%.9g\n",
            rows, inner, pair_layout, fixed_global);
    auto* input = device_allocate<float>(inner);
    auto* qinput = device_allocate<uint8_t>(inner / 2);
    auto* qscales = device_allocate<uint8_t>(inner / 16);
    auto* output = device_allocate<float>(static_cast<size_t>(rows) * experts + 8);
    auto* partial = device_allocate<float>(rows * experts * 8);
    check(cudaMemcpy(input, original.data(), inner * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(qinput, quantized.data.data(), inner / 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(qscales, quantized.scales.data(), inner / 16, cudaMemcpyHostToDevice));
    // Use GPU-produced activation buffers for the ensuing native MMA checks.
    // Dynamic-global tests use the CPU-selected global solely as a diagnostic;
    // the GPU kernel never recomputes a calibrated global scale.
    auto quantize_gpu = [&](int) {
        quantize_activation_fixed_global<<<(inner / 16 + 127) / 128, 128,
            0, cudaStreamPerThread>>>(input, qinput, qscales, inner, quantized.global);
    };
    quantize_gpu(0);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<uint8_t> gpu_packed(inner / 2), gpu_scales(inner / 16);
    check(cudaMemcpy(gpu_packed.data(), qinput, inner / 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(gpu_scales.data(), qscales, inner / 16, cudaMemcpyDeviceToHost));
    if (gpu_packed != quantized.data || gpu_scales != quantized.scales) {
        std::fprintf(stderr, "GPU activation quantization byte mismatch K=%d scale=%g\n",
            inner, quantized.global); std::exit(1);
    }
    if (benchmark)
        std::printf("GPU_activation_quantization K=%d global=%.9g us=%.3f bytes_exact=1\n",
            inner, quantized.global, measure(quantize_gpu));
    std::vector<mollm_cuda::Nvfp4Projection> jobs(experts * sets);
    constexpr uint8_t scale_codes[] = {0, 1, 7, 8, 48, 56, 64, 126};
    for (size_t job = 0; job < jobs.size(); ++job) {
        uint8_t* weights = resident.data + job * stride;
        uint8_t* scales = weights + weight_bytes;
        float* row_scales = reinterpret_cast<float*>(scales + scale_bytes);
        std::memset(weights, 0, weight_bytes);
        std::memset(scales, 0, scale_bytes);
        for (int row = 0; row < rows; ++row) {
            row_scales[row] = (1 + (row + job) % 7) / 64.f;
            for (int group = 0; group < inner / 16; ++group) {
                scales[row * (inner / 16) + group] = scale_codes[(row + group + job) % 8];
                for (int byte = 0; byte < 8; ++byte) {
                    const uint8_t code = (row * 29 + group * 17 + byte * 13 + job * 7) & 255;
                    const size_t offset = pair_layout
                        ? (static_cast<size_t>(row / 4) * (inner / 16) + group) * 32 + (row % 4) * 8 + byte
                        : static_cast<size_t>(row) * (inner / 2) + group * 8 + byte;
                    weights[offset] = code;
                }
            }
        }
        jobs[job] = {weights, scales, row_scales, input, output + (job % experts) * rows, pair_layout};
    }
    // Separate probe buffers isolate the benefit of MMA-native layout. A
    // production resident provider would prepare its single owned payload,
    // not retain both expert representations as this measurement harness does.
    const size_t native_weight_bytes = static_cast<size_t>((rows + 15) / 16) * 16 * inner / 2;
    const size_t native_scale_bytes = static_cast<size_t>((rows + 15) / 16) * 16 * inner / 16;
    Arena native_resident((native_weight_bytes + native_scale_bytes) * jobs.size());
    std::vector<mollm_cuda::Nvfp4Projection> native_jobs = jobs;
    for (size_t job = 0; job < jobs.size(); ++job) {
        auto* weights = native_resident.data + job * (native_weight_bytes + native_scale_bytes);
        auto* scales = weights + native_weight_bytes;
        std::memset(weights, 0, native_weight_bytes);
        std::memset(scales, 0, native_scale_bytes);
        for (int tile_row = 0; tile_row < (rows + 15) / 16; ++tile_row) {
            for (int tile_k = 0; tile_k < inner / 64; ++tile_k) {
                const size_t tile = static_cast<size_t>(tile_row) * (inner / 64) + tile_k;
                for (int reg = 0; reg < 4; ++reg) for (int lane = 0; lane < 32; ++lane) {
                    const int row = tile_row * 16 + lane / 4 + (reg % 2) * 8;
                    const int col = tile_k * 64 + lane % 4 * 8 + reg / 2 * 32;
                    if (row >= rows) continue;
                    const size_t offset = pair_layout
                        ? (static_cast<size_t>(row / 4) * (inner / 16) + col / 16) * 32 + (row % 4) * 8 + (col % 16) / 2
                        : static_cast<size_t>(row) * (inner / 2) + col / 2;
                    std::memcpy(weights + tile * 512 + reg * 128 + lane * 4,
                        jobs[job].weight + offset, 4);
                }
                for (int row = 0; row < 16 && tile_row * 16 + row < rows; ++row)
                    std::memcpy(scales + tile * 64 + (row % 8) * 8 + (row / 8) * 4,
                        jobs[job].block_scales + static_cast<size_t>(tile_row * 16 + row) * (inner / 16) + tile_k * 4, 4);
            }
        }
        native_jobs[job].weight = weights;
        native_jobs[job].block_scales = scales;
        // Keep only a single projection-sized CPU reference while replacing
        // the prepared payload through the reusable allocation-free launcher.
        const std::vector<uint8_t> expected_weights(weights, weights + native_weight_bytes);
        const std::vector<uint8_t> expected_scales(scales, scales + native_scale_bytes);
        check(mollm_cuda::launch_repack_nvfp4_mma(jobs[job], weights, scales,
            rows, inner, cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        if (std::memcmp(weights, expected_weights.data(), native_weight_bytes) ||
            std::memcmp(scales, expected_scales.data(), native_scale_bytes)) {
            std::fprintf(stderr, "GPU native-layout repacking byte mismatch\n"); std::exit(1);
        }
        if (job == 0 && !benchmark) {
            auto* device_payload = device_allocate<uint8_t>(native_weight_bytes + native_scale_bytes);
            check(mollm_cuda::launch_repack_nvfp4_mma(jobs[job], device_payload,
                device_payload + native_weight_bytes, rows, inner, cudaStreamPerThread));
            check(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<uint8_t> actual_payload(native_weight_bytes + native_scale_bytes);
            check(cudaMemcpy(actual_payload.data(), device_payload, actual_payload.size(), cudaMemcpyDeviceToHost));
            if (std::memcmp(actual_payload.data(), expected_weights.data(), native_weight_bytes) ||
                std::memcmp(actual_payload.data() + native_weight_bytes,
                    expected_scales.data(), native_scale_bytes)) {
                std::fprintf(stderr, "CUDA destination repacking byte mismatch\n"); std::exit(1);
            }
            check(cudaFree(device_payload));
            if (mollm_cuda::launch_repack_nvfp4_mma(jobs[job], weights, scales,
                    rows, 63) != cudaErrorInvalidValue ||
                mollm_cuda::launch_repack_nvfp4_mma(jobs[job], weights, scales,
                    0, inner) != cudaErrorInvalidValue ||
                mollm_cuda::launch_repack_nvfp4_mma(jobs[job], weights + 1, scales,
                    rows, inner) != cudaErrorInvalidValue) {
                std::fprintf(stderr, "invalid repacking arguments accepted\n"); std::exit(1);
            }
        }
    }
    if (std::getenv("MOLLM_NVFP4_NATIVE_PIPELINE_PROBE")) {
        uint8_t* device_canonical = nullptr;
        uint8_t* device_native = nullptr;
        const bool device_weights = std::getenv("MOLLM_NVFP4_NATIVE_PIPELINE_DEVICE") != nullptr;
        if (device_weights) {
            device_canonical = device_allocate<uint8_t>(resident.size);
            device_native = device_allocate<uint8_t>(native_resident.size);
            check(cudaMemcpy(device_canonical, resident.data, resident.size, cudaMemcpyHostToDevice));
            check(cudaMemcpy(device_native, native_resident.data, native_resident.size, cudaMemcpyHostToDevice));
            for (size_t i = 0; i < jobs.size(); ++i) {
                auto& canonical = jobs[i]; auto& native = native_jobs[i];
                canonical.weight = device_canonical + (canonical.weight - resident.data);
                canonical.block_scales = device_canonical + (canonical.block_scales - resident.data);
                canonical.row_scales = reinterpret_cast<float*>(device_canonical +
                    (reinterpret_cast<const uint8_t*>(canonical.row_scales) - resident.data));
                native.weight = device_native + (native.weight - native_resident.data);
                native.block_scales = device_native + (native.block_scales - native_resident.data);
                native.row_scales = canonical.row_scales;
            }
        }
        for(int set=0;set<sets;++set) {
        cudaGraph_t graph; cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        check(mollm_cuda::launch_nvfp4_native_pipelined_projections(native_jobs.data()+set*experts, experts, rows, inner, cudaStreamPerThread));
        check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        std::vector<float> reference(size_t(rows) * experts + 8), actual(reference.size());
        for (int replay = 0; replay < 4; ++replay) {
            for (int k = 0; k < inner; ++k) original[k] = (int((k * 13 + replay * 7) % 101) - 50) / 37.f;
            if(replay==2)std::fill(original.begin(),original.end(),0.f);
            if(replay==3)for(int k=0;k<inner;k+=17)original[k]=-0.f;
            check(cudaMemcpy(input, original.data(), inner * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemset(output, 0x7f, reference.size() * sizeof(float)));
            mollm_cuda::launch_nvfp4_projections(jobs.data()+set*experts, experts, rows, inner, cudaStreamPerThread);
            check(cudaStreamSynchronize(cudaStreamPerThread));
            check(cudaMemcpy(reference.data(), output, reference.size() * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemset(output, 0x7f, reference.size() * sizeof(float)));
            mollm_cuda::launch_nvfp4_projections(native_jobs.data()+set*experts, experts, rows, inner, cudaStreamPerThread, true);
            check(cudaStreamSynchronize(cudaStreamPerThread));
            check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(reference.data(), actual.data(), actual.size() * sizeof(float))) { std::fprintf(stderr, "native baseline mismatch\n"); std::exit(1); }
            check(cudaMemset(output, 0x7f, reference.size() * sizeof(float)));
            check(cudaGraphLaunch(executable, cudaStreamPerThread));
            check(cudaStreamSynchronize(cudaStreamPerThread));
            check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(reference.data(), actual.data(), actual.size() * sizeof(float))) { std::fprintf(stderr, "native pipeline mismatch\n"); std::exit(1); }
            const auto* guard=reinterpret_cast<const uint8_t*>(actual.data()+size_t(rows)*experts);
            if(!std::all_of(guard,guard+32,[](uint8_t b){return b==0x7f;}))std::exit(1);
        }
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        }
        std::printf("native_pipeline_exact rows=%d K=%d pair=%d device=%d replays=%d full_bytes_guards=true\n", rows, inner, pair_layout, device_weights,4*sets);
        if (benchmark) for (int repeat = 0; repeat < 7; ++repeat) for (int order = 0; order < 4; ++order) {
            int mode = (repeat + order) % 4;
            auto launch = [&](int i) {
                auto* selected = (mode < 2 ? native_jobs.data() : jobs.data()) + (i % sets) * experts;
                if (mode == 1) check(mollm_cuda::launch_nvfp4_native_pipelined_projections(selected, experts, rows, inner, cudaStreamPerThread));
                else if (mode == 3) check(mollm_cuda::launch_nvfp4_pipelined_projections(selected, experts, rows, inner, true, cudaStreamPerThread));
                else mollm_cuda::launch_nvfp4_projections(selected, experts, rows, inner, cudaStreamPerThread, mode == 0);
            };
            std::printf("native_pipeline_timing rows=%d K=%d pair=%d device=%d repeat=%d mode=%d us=%.3f\n", rows, inner, pair_layout, device_weights, repeat, mode, measure(launch, 128));
        }
        if (device_weights) { check(cudaFree(device_canonical)); check(cudaFree(device_native)); }
        check(cudaFree(input)); check(cudaFree(qinput)); check(cudaFree(qscales));
        check(cudaFree(output)); check(cudaFree(partial));
        return;
    }
    // CPU double accumulation over explicitly decoded W4/A4 is an independent
    // reference for the scale/fragment layout. Original FP32 activations give
    // a separate measure of activation quantization error, not a correctness gate.
    std::vector<double> expected(rows * experts), original_expected(rows * experts);
    for (int job = 0; job < experts; ++job) {
        for (int row = 0; row < rows; ++row) {
            double sum = 0, original_sum = 0;
            for (int k = 0; k < inner; ++k) {
                const size_t offset = pair_layout
                    ? (static_cast<size_t>(row / 4) * (inner / 16) + k / 16) * 32 + (row % 4) * 8 + (k % 16) / 2
                    : static_cast<size_t>(row) * (inner / 2) + k / 2;
                const uint8_t packed = jobs[job].weight[offset];
                const int code = k & 1 ? packed >> 4 : packed & 15;
                const double weight = fp4_value(code) * fp8_value(jobs[job].block_scales[row * (inner / 16) + k / 16]);
                sum += weight * quantized.decoded[k];
                original_sum += weight * original[k];
            }
            expected[job * rows + row] = sum * jobs[job].row_scales[row];
            original_expected[job * rows + row] = original_sum * jobs[job].row_scales[row];
        }
    }
    nvfp4_mma_probe<true, 0, true><<<dim3((rows + 15) / 16, experts), 32>>>(native_jobs.data(), qinput, qscales,
        quantized.global, rows, inner);
    check(cudaGetLastError());
    std::vector<float> actual(rows * experts);
    check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double max_error = 0, maximum = 0, quant_error = 0, original_energy = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) { std::fprintf(stderr, "nonfinite MMA output\n"); std::exit(1); }
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
        maximum = std::max(maximum, std::abs(expected[i]));
        quant_error += (expected[i] - original_expected[i]) * (expected[i] - original_expected[i]);
        original_energy += original_expected[i] * original_expected[i];
    }
    const double normalized_error = max_error / std::max(1., maximum);
    if (rows % 16 == 0) {
        // Exercise the reusable runtime kernel, including separate gate/up
        // globals, per-expert calibration and output-side route weighting.
        auto runtime_jobs = native_jobs;
        std::vector<double> runtime_expected(expected.size());
        const int split = rows >= 128 && (rows / 2) % 64 == 0 ? rows / 2 : 0;
        for (int job = 0; job < experts; ++job) {
            auto& descriptor = runtime_jobs[job];
            descriptor.mma_layout = true;
            descriptor.input_global = quantized.global * (1 + job * .125f);
            descriptor.input_global_second = descriptor.input_global * 1.75f;
            descriptor.output_scale = (job + 1) * .125f;
            QuantizedActivation first(original, descriptor.input_global);
            QuantizedActivation second(original, descriptor.input_global_second);
            for (int row = 0; row < rows; ++row) {
                double sum = 0;
                const auto& decoded = split && row >= split ? second.decoded : first.decoded;
                for (int k = 0; k < inner; ++k) {
                    const size_t offset = pair_layout
                        ? (static_cast<size_t>(row / 4) * (inner / 16) + k / 16) * 32 +
                            (row % 4) * 8 + (k % 16) / 2
                        : static_cast<size_t>(row) * (inner / 2) + k / 2;
                    const uint8_t code = jobs[job].weight[offset];
                    sum += fp4_value(k & 1 ? code >> 4 : code & 15) *
                        fp8_value(jobs[job].block_scales[row * (inner / 16) + k / 16]) * decoded[k];
                }
                runtime_expected[job * rows + row] = sum * jobs[job].row_scales[row] * descriptor.output_scale;
            }
        }
        for (int runtime_parts : {1, 2, 4, 8}) {
        check(mollm_cuda::launch_nvfp4_mma_projections(runtime_jobs.data(), experts, rows, inner, split,
            nullptr, partial, runtime_parts));
        check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double error = 0, magnitude = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i])) { std::fprintf(stderr, "nonfinite runtime MMA output\n"); std::exit(1); }
            error = std::max(error, std::abs(actual[i] - runtime_expected[i]));
            magnitude = std::max(magnitude, std::abs(runtime_expected[i]));
        }
        if (error / std::max(1., magnitude) > 2e-5) {
            std::fprintf(stderr, "runtime MMA mismatch rows=%d K=%d split=%d error=%g\n",
                rows, inner, split, error / std::max(1., magnitude)); std::exit(1);
        }
        std::printf("runtime_MMA rows=%d K=%d split=%d parts=%d distinct_globals_and_output_weights=true error=%g\n",
            rows, inner, split, runtime_parts, error / std::max(1., magnitude));
        const auto native_actual = actual;
        auto canonical_jobs = runtime_jobs;
        for (int j = 0; j < experts; ++j) {
            canonical_jobs[j].weight = jobs[j].weight;
            canonical_jobs[j].block_scales = jobs[j].block_scales;
            canonical_jobs[j].pair_layout = pair_layout;
            canonical_jobs[j].mma_layout = false;
        }
        check(mollm_cuda::launch_nvfp4_mma_projections(canonical_jobs.data(), experts,
            rows, inner, split, nullptr, partial, runtime_parts, true));
        check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (std::memcmp(actual.data(), native_actual.data(), actual.size() * sizeof(float))) {
            std::fprintf(stderr, "canonical/native MMA bit mismatch rows=%d K=%d pair=%d parts=%d\n",
                rows, inner, pair_layout, runtime_parts); std::exit(1);
        }
        std::printf("canonical_MMA rows=%d K=%d pair=%d parts=%d native_bit_exact=true\n",
            rows, inner, pair_layout, runtime_parts);
        if(runtime_parts==1 && std::getenv("MOLLM_NVFP4_SCALAR_PREPARE_PROBE")) {
            const size_t variants=split?2:1;
            auto* prepared_input=device_allocate<uint8_t>(experts*variants*inner/2);
            auto* prepared_scale=device_allocate<uint8_t>(experts*variants*inner/16);
            auto candidate=[&](int,cudaStream_t stream) {
                check(mollm_cuda::launch_nvfp4_mma_projections(canonical_jobs.data(),experts,
                    rows,inner,split,stream,nullptr,1,true,prepared_input,prepared_scale));
            };
            auto control=[&](int,cudaStream_t stream) {
                check(mollm_cuda::launch_nvfp4_mma_projections(canonical_jobs.data(),experts,
                    rows,inner,split,stream,nullptr,1,true));
            };
            for(int repeat=0;repeat<3;++repeat) {
                double candidate_us,control_us;
                if(repeat%2){candidate_us=measure_graph(candidate,32);control_us=measure_graph(control,32);}
                else {control_us=measure_graph(control,32);candidate_us=measure_graph(candidate,32);}
                // Validate the prepared path explicitly, regardless of timing order.
                candidate(0,nullptr);
                check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
                if(std::memcmp(actual.data(),native_actual.data(),actual.size()*sizeof(float))) {
                    std::fprintf(stderr,"prepared scalar MMA mismatch\n");std::exit(1);
                }
                std::printf("scalar_prepare rows=%d K=%d pair=%d repeat=%d original_us=%.3f prepared_us=%.3f bit_exact=true\n",
                    rows,inner,pair_layout,repeat,control_us,candidate_us);
            }
            check(cudaFree(prepared_input));check(cudaFree(prepared_scale));
        }
        }
    }
    if (!benchmark && rows % 16 == 0) {
        auto* batch_input = device_allocate<float>(8 * inner);
        auto* batch_output = device_allocate<float>(8 * rows);
        auto* batch_jobs = device_allocate<mollm_cuda::Nvfp4Projection>(8);
        std::vector<float> input_values(8 * inner), scalar(8 * rows), batched(8 * rows);
        std::vector<mollm_cuda::Nvfp4Projection> descriptors(8, native_jobs[0]);
        for (int column = 0; column < 8; ++column) {
            for (int k = 0; k < inner; ++k)
                input_values[column * inner + k] = original[k] * ((column + 1) / 8.f) +
                    ((k + column) % 13 - 6) * .01f;
            auto& job = descriptors[column];
            job.input = batch_input + column * inner;
            job.output = batch_output + column * rows;
            job.input_global = quantized.global * (1 + column * .125f);
            job.input_global_second = job.input_global * 1.75f;
            job.output_scale = (column - 3) * .125f;
        }
        check(cudaMemcpy(batch_input, input_values.data(), input_values.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(batch_jobs, descriptors.data(), 8 * sizeof(descriptors[0]), cudaMemcpyHostToDevice));
        const int split = rows >= 128 && (rows / 2) % 64 == 0 ? rows / 2 : 0;
        check(mollm_cuda::launch_nvfp4_mma_projections(batch_jobs, 8, rows, inner, split));
        check(cudaMemcpy(scalar.data(), batch_output, scalar.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(mollm_cuda::launch_nvfp4_mma_batched8(batch_jobs, 8, rows, inner, split));
        check(cudaMemcpy(batched.data(), batch_output, batched.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (std::memcmp(scalar.data(), batched.data(), scalar.size() * sizeof(float))) {
            std::fprintf(stderr, "batched N8 mismatch rows=%d K=%d split=%d\n", rows, inner, split); std::exit(1);
        }
        std::printf("batched_N8 rows=%d K=%d split=%d distinct_inputs_globals_routes=true bit_exact=true\n", rows, inner, split);
        auto* grouped_output = device_allocate<float>(17 * rows);
        auto* grouped_jobs = device_allocate<mollm_cuda::Nvfp4Projection>(17);
        auto* batches = device_allocate<mollm_cuda::Nvfp4MmaBatch>(17);
        auto* compact_batches = device_allocate<mollm_cuda::Nvfp4MmaBatch>(18);
        auto* prepared_input = device_allocate<uint8_t>(2 * 17 * inner / 2);
        auto* prepared_scales = device_allocate<uint8_t>(2 * 17 * inner / 16);
        std::vector<mollm_cuda::Nvfp4Projection> grouped_descriptors(17);
        std::vector<float> scalar_grouped(17 * rows), actual_grouped(17 * rows);
        for (int i = 0; i < 17; ++i) {
            grouped_descriptors[i] = descriptors[i % 8];
            if (i >= 8) {
                grouped_descriptors[i].weight = native_jobs[1].weight;
                grouped_descriptors[i].block_scales = native_jobs[1].block_scales;
                grouped_descriptors[i].row_scales = native_jobs[1].row_scales;
            }
            grouped_descriptors[i].output = grouped_output + (16 - i) * rows;
        }
        check(cudaMemcpy(grouped_jobs, grouped_descriptors.data(), 17 * sizeof(grouped_descriptors[0]), cudaMemcpyHostToDevice));
        for (int count : {1, 3, 7, 8, 9, 17}) {
            check(cudaMemset(grouped_output, 0, 17 * rows * sizeof(float)));
            check(mollm_cuda::launch_nvfp4_mma_projections(grouped_jobs, count, rows, inner, split));
            check(cudaMemcpy(scalar_grouped.data(), grouped_output, scalar_grouped.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (bool compact : {false, true}) {
            check(cudaMemset(grouped_output, 0, 17 * rows * sizeof(float)));
            check(mollm_cuda::launch_nvfp4_mma_grouped(grouped_jobs, count, batches, rows, inner, split,
                nullptr, nullptr, nullptr, 1, false, false, compact ? compact_batches : nullptr));
            check(cudaMemcpy(actual_grouped.data(), grouped_output, actual_grouped.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(scalar_grouped.data(), actual_grouped.data(), scalar_grouped.size() * sizeof(float))) {
                std::fprintf(stderr, "grouped N8 mismatch count=%d rows=%d K=%d\n", count, rows, inner); std::exit(1);
            }
            }
            for (bool compact : {false, true}) for (bool reuse : {false, true}) for (int minimum : {1, 2, 3, 4, 5, 6, 7, 8}) {
            check(cudaMemset(grouped_output, 0, 17 * rows * sizeof(float)));
            check(cudaMemset(prepared_input, 0xff, 2 * 17 * inner / 2));
            check(cudaMemset(prepared_scales, 0xff, 2 * 17 * inner / 16));
            check(mollm_cuda::launch_nvfp4_mma_grouped(grouped_jobs, count, batches, rows, inner, split,
                nullptr, prepared_input, prepared_scales, minimum, false, reuse, compact ? compact_batches : nullptr));
            check(cudaMemcpy(actual_grouped.data(), grouped_output, actual_grouped.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(scalar_grouped.data(), actual_grouped.data(), scalar_grouped.size() * sizeof(float))) {
                std::fprintf(stderr, "prequantized grouped mismatch count=%d rows=%d K=%d\n", count, rows, inner); std::exit(1);
            }
            if (!split) {
                check(cudaMemset(grouped_output,0,17*rows*sizeof(float)));
                check(mollm_cuda::launch_nvfp4_mma_grouped(grouped_jobs,count,batches,rows,inner,0,
                    nullptr,prepared_input,prepared_scales,minimum,false,reuse,
                    compact?compact_batches:nullptr,0,8,false,true));
                check(cudaMemcpy(actual_grouped.data(),grouped_output,actual_grouped.size()*sizeof(float),cudaMemcpyDeviceToHost));
                if (std::memcmp(scalar_grouped.data(),actual_grouped.data(),actual_grouped.size()*sizeof(float))) {
                    std::fprintf(stderr,"preprepared generic N8 mismatch\n");std::exit(1);
                }
            }
            }
        }
        for (int i = 0; i < 17; ++i) {
            const int expert = i >= 8 ? 1 : 0;
            grouped_descriptors[i].weight = jobs[expert].weight;
            grouped_descriptors[i].block_scales = jobs[expert].block_scales;
            grouped_descriptors[i].pair_layout = pair_layout;
            grouped_descriptors[i].mma_layout = false;
        }
        check(cudaMemcpy(grouped_jobs, grouped_descriptors.data(),
            17 * sizeof(grouped_descriptors[0]), cudaMemcpyHostToDevice));
        for (int count : {1, 3, 7, 8, 9, 17}) {
            check(cudaMemset(grouped_output, 0, 17 * rows * sizeof(float)));
            check(mollm_cuda::launch_nvfp4_mma_projections(grouped_jobs, count,
                rows, inner, split, nullptr, nullptr, 1, true));
            check(cudaMemcpy(scalar_grouped.data(), grouped_output,
                scalar_grouped.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (bool compact : {false, true}) for (bool prepared : {false, true}) for (bool reuse : {false, true})
            for (int minimum : {1, 2, 3, 4, 5, 6, 7, 8}) {
                if (reuse && !prepared) continue;
                check(cudaMemset(grouped_output, 0, 17 * rows * sizeof(float)));
                check(mollm_cuda::launch_nvfp4_mma_grouped(grouped_jobs, count,
                    batches, rows, inner, split, nullptr,
                    prepared ? prepared_input : nullptr,
                    prepared ? prepared_scales : nullptr, minimum, true, reuse, compact ? compact_batches : nullptr));
                check(cudaMemcpy(actual_grouped.data(), grouped_output,
                    actual_grouped.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (std::memcmp(scalar_grouped.data(), actual_grouped.data(),
                    scalar_grouped.size() * sizeof(float))) {
                    std::fprintf(stderr, "canonical grouped mismatch count=%d rows=%d K=%d pair=%d prepared=%d minimum=%d\n",
                        count, rows, inner, pair_layout, prepared, minimum);
                    std::exit(1);
                }
            }
        }
        grouped_descriptors[9].pair_layout = !pair_layout;
        check(cudaMemcpy(grouped_jobs, grouped_descriptors.data(),
            17 * sizeof(grouped_descriptors[0]), cudaMemcpyHostToDevice));
        check(mollm_cuda::launch_nvfp4_mma_projections(grouped_jobs, 17,
            rows, inner, split, nullptr, nullptr, 1, true));
        check(cudaMemcpy(scalar_grouped.data(), grouped_output,
            scalar_grouped.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (bool compact : {false, true}) for (bool prepared : {false, true}) for (bool reuse : {false, true})
        for (int minimum : {1, 2, 3, 4, 5, 6, 7, 8}) {
            if (reuse && !prepared) continue;
            check(cudaMemset(grouped_output, 0, 17 * rows * sizeof(float)));
            check(mollm_cuda::launch_nvfp4_mma_grouped(grouped_jobs, 17,
                batches, rows, inner, split, nullptr,
                prepared ? prepared_input : nullptr,
                prepared ? prepared_scales : nullptr, minimum, true, reuse, compact ? compact_batches : nullptr));
            check(cudaMemcpy(actual_grouped.data(), grouped_output,
                actual_grouped.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(scalar_grouped.data(), actual_grouped.data(),
                scalar_grouped.size() * sizeof(float))) {
                std::fprintf(stderr, "canonical mixed-layout grouping mismatch\n");
                std::exit(1);
            }
        }
        cudaStream_t grouped_stream;
        check(cudaStreamCreateWithFlags(&grouped_stream, cudaStreamNonBlocking));
        for (bool compact : {false, true}) for (bool prepared : {false, true}) for (bool reuse : {false, true})
        for (int minimum : {1, 4}) {
            if (reuse && !prepared) continue;
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(grouped_stream, cudaStreamCaptureModeThreadLocal));
            check(mollm_cuda::launch_nvfp4_mma_grouped(grouped_jobs, 17,
                batches, rows, inner, split, grouped_stream,
                prepared ? prepared_input : nullptr,
                prepared ? prepared_scales : nullptr, minimum, true, reuse, compact ? compact_batches : nullptr));
            check(cudaStreamEndCapture(grouped_stream, &graph));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 2; ++replay) {
                for (float& value : input_values) value = -value;
                // Alternate every descriptor, then restore two long runs.
                // Compact count transitions 17->3 at minimum1 and 0->2 at
                // minimum4. This also exercises repeated shared-buffer reuse
                // with inactive tail warps, without assuming a group bound.
                for (int i = 0; i < 17; ++i) {
                    const auto& source = jobs[replay == 0 ? i % 2 : int(i >= 8)];
                    grouped_descriptors[i].weight = source.weight;
                    grouped_descriptors[i].block_scales = source.block_scales;
                    grouped_descriptors[i].row_scales = source.row_scales;
                    grouped_descriptors[i].pair_layout = pair_layout;
                }
                check(cudaMemcpyAsync(grouped_jobs, grouped_descriptors.data(),
                    grouped_descriptors.size() * sizeof(grouped_descriptors[0]), cudaMemcpyHostToDevice, grouped_stream));
                check(cudaMemcpyAsync(batch_input, input_values.data(),
                    input_values.size() * sizeof(float), cudaMemcpyHostToDevice, grouped_stream));
                check(mollm_cuda::launch_nvfp4_mma_projections(grouped_jobs, 17,
                    rows, inner, split, grouped_stream, nullptr, 1, true));
                check(cudaMemcpyAsync(scalar_grouped.data(), grouped_output,
                    scalar_grouped.size() * sizeof(float), cudaMemcpyDeviceToHost, grouped_stream));
                check(cudaGraphLaunch(executable, grouped_stream));
                check(cudaMemcpyAsync(actual_grouped.data(), grouped_output,
                    actual_grouped.size() * sizeof(float), cudaMemcpyDeviceToHost, grouped_stream));
                check(cudaStreamSynchronize(grouped_stream));
                if (std::memcmp(scalar_grouped.data(), actual_grouped.data(),
                    scalar_grouped.size() * sizeof(float))) {
                    std::fprintf(stderr, "canonical grouped changed-input graph mismatch\n");
                    std::exit(1);
                }
            }
            check(cudaGraphExecDestroy(executable));
            check(cudaGraphDestroy(graph));
        }
        check(cudaStreamDestroy(grouped_stream));
        std::printf("canonical_grouped_N8 rows=%d K=%d pair=%d ragged_prepared_thresholds_mixed_layout_graph=true bit_exact=true\n",
            rows, inner, pair_layout);
        std::printf("grouped_N8 rows=%d K=%d ragged_two_experts_reverse_scatter=true bit_exact=true\n", rows, inner);
        {
            const bool shared_max_check=std::getenv("MOLLM_NVFP4_SHARED_MAX_CHECK")!=nullptr;
            const int total = shared_max_check?161:37;
            auto* wide_output = device_allocate<float>(total * rows);
            auto* wide_jobs = device_allocate<mollm_cuda::Nvfp4Projection>(total);
            auto* wide_plan = device_allocate<mollm_cuda::Nvfp4MmaBatch>(total);
            auto* wide_compact = device_allocate<mollm_cuda::Nvfp4MmaBatch>(total + 1);
            auto* wide_packed = device_allocate<uint8_t>(2 * total * inner / 2);
            auto* wide_scales = device_allocate<uint8_t>(2 * total * inner / 16);
            std::vector<mollm_cuda::Nvfp4Projection> wide_descriptors(total);
            std::vector<float> expected(total * rows), got(total * rows);
            for (bool canonical : {false, true}) {
                const auto set_descriptors = [&](int pattern) {
                    for (int i = 0; i < total; ++i) {
                        const int expert = pattern == 0 ? i % 2 : pattern == 1 ? 0 : int(i >= 23);
                        wide_descriptors[i] = (canonical ? jobs : native_jobs)[expert];
                        auto& job = wide_descriptors[i];
                        job.input = batch_input + (i % 8) * inner;
                        job.output = wide_output + (total - 1 - i) * rows;
                        job.input_global = quantized.global * (1 + i * .025f);
                        job.input_global_second = job.input_global * 1.75f;
                        job.output_scale = (i - 10) / 32.f;
                        if (pattern == 3 && i == 9)
                            job.row_scales = (canonical ? jobs : native_jobs)[1].row_scales;
                    }
                };
                set_descriptors(1);
                check(cudaMemcpy(wide_jobs, wide_descriptors.data(), total * sizeof(wide_descriptors[0]), cudaMemcpyHostToDevice));
                if (mollm_cuda::launch_nvfp4_mma_grouped(wide_jobs, total, wide_plan, rows, inner, split,
                        nullptr, nullptr, nullptr, 1, canonical, false, nullptr, 0, 16) != cudaErrorInvalidValue ||
                    mollm_cuda::launch_nvfp4_mma_grouped(wide_jobs, total, wide_plan, rows, inner, split,
                        nullptr, nullptr, nullptr, 1, canonical, false, nullptr, 0, 32) != cudaErrorInvalidValue ||
                    mollm_cuda::launch_nvfp4_mma_grouped(wide_jobs, total, wide_plan, rows, inner, split,
                        nullptr, wide_packed, wide_scales, 1, canonical, false, nullptr, 0, 64) != cudaErrorInvalidValue ||
                    mollm_cuda::launch_nvfp4_mma_grouped(wide_jobs, total, wide_plan, rows, inner, split,
                        nullptr, nullptr, nullptr, 1, canonical, false, nullptr, 0, 8, true) != cudaErrorInvalidValue) {
                    std::fprintf(stderr, "N16/N32 preparation/width guard failed\n"); std::exit(1);
                }
                if (mollm_cuda::launch_nvfp4_mma_grouped(wide_jobs,total,wide_plan,rows,inner,split,
                        nullptr,nullptr,nullptr,1,canonical,false,nullptr,0,8,false,true)!=cudaErrorInvalidValue ||
                    (split && mollm_cuda::launch_nvfp4_mma_grouped(wide_jobs,total,wide_plan,rows,inner,split,
                        nullptr,wide_packed,wide_scales,1,canonical,false,nullptr,0,8,false,true)!=cudaErrorInvalidValue)) {
                    std::fprintf(stderr,"already-prepared input guard failed\n");std::exit(1);
                }
                for (int pattern : {1, 2, 3}) {
                    set_descriptors(pattern);
                    check(cudaMemcpy(wide_jobs, wide_descriptors.data(), total * sizeof(wide_descriptors[0]), cudaMemcpyHostToDevice));
                    std::vector<int> counts={1,7,8,9,15,16,17,23,31,32,33,37};
                    if(shared_max_check) for(int count:{159,160,161})counts.push_back(count);
                    for (int count : counts) {
                        check(cudaMemset(wide_output, 0, expected.size() * sizeof(float)));
                        check(mollm_cuda::launch_nvfp4_mma_projections(wide_jobs, count, rows, inner, split, nullptr, nullptr, 1, canonical));
                        check(cudaMemcpy(expected.data(), wide_output, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
                        for (int width : {8, 16, 32}) for (bool compact : {false, true}) for (bool reuse : {false, true})
                        for (int minimum : {1, 2, 3, 4, 5, 6, 7, 8}) {
                            check(cudaMemset(wide_output, 0, got.size() * sizeof(float)));
                            check(cudaMemset(wide_packed, 0xff, 2 * total * inner / 2));
                            check(cudaMemset(wide_scales, 0xff, 2 * total * inner / 16));
                            check(mollm_cuda::launch_nvfp4_mma_grouped(wide_jobs, count, wide_plan, rows, inner, split,
                                nullptr, wide_packed, wide_scales, minimum, canonical, reuse,
                                compact ? wide_compact : nullptr, 0, width, true));
                            check(cudaMemcpy(got.data(), wide_output, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
                            if (std::memcmp(expected.data(), got.data(), got.size() * sizeof(float))) {
                                std::fprintf(stderr, "prepared MMA mismatch width=%d rows=%d K=%d canonical=%d pattern=%d count=%d minimum=%d compact=%d reuse=%d\n",
                                    width, rows, inner, canonical, pattern, count, minimum, compact, reuse); std::exit(1);
                            }
                            if (!split) {
                                check(cudaMemset(wide_output,0,got.size()*sizeof(float)));
                                // Fully prepared batches and prepared singleton
                                // fallbacks must not read the FP32 input at all.
                                const bool poison=pattern==1 && (count==1 || count==total) &&
                                    (minimum==1 || reuse);
                                if (poison) {
                                    auto descriptors=wide_descriptors;
                                    for (auto& job : descriptors) job.input=nullptr;
                                    check(cudaMemcpy(wide_jobs,descriptors.data(),total*sizeof(descriptors[0]),cudaMemcpyHostToDevice));
                                }
                                check(mollm_cuda::launch_nvfp4_mma_grouped(wide_jobs,count,wide_plan,rows,inner,0,
                                    nullptr,wide_packed,wide_scales,minimum,canonical,reuse,
                                    compact?wide_compact:nullptr,0,width,true,true));
                                if (poison)
                                    check(cudaMemcpy(wide_jobs,wide_descriptors.data(),total*sizeof(wide_descriptors[0]),cudaMemcpyHostToDevice));
                                check(cudaMemcpy(got.data(),wide_output,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
                                if (std::memcmp(expected.data(),got.data(),got.size()*sizeof(float))) {
                                    std::fprintf(stderr,"preprepared MMA mismatch width=%d\n",width);std::exit(1);
                                }
                            }
                        }
                    }
                }
                cudaStream_t wide_stream;
                check(cudaStreamCreateWithFlags(&wide_stream, cudaStreamNonBlocking));
                for (int width : {8, 16, 32}) for (bool compact : {false, true}) for (bool reuse : {false, true}) for (int minimum : {1, 4, 8}) {
                    cudaGraph_t wide_graph;
                    cudaGraphExec_t wide_exec;
                    check(cudaStreamBeginCapture(wide_stream, cudaStreamCaptureModeThreadLocal));
                    check(mollm_cuda::launch_nvfp4_mma_grouped(wide_jobs, total, wide_plan, rows, inner, split,
                        wide_stream, wide_packed, wide_scales, minimum, canonical, reuse,
                        compact ? wide_compact : nullptr, 0, width, true));
                    check(cudaStreamEndCapture(wide_stream, &wide_graph));
                    check(cudaGraphInstantiate(&wide_exec, wide_graph, nullptr, nullptr, 0));
                    for (int replay = 0; replay < 4; ++replay) {
                        set_descriptors(replay);
                        for (float& value : input_values) value = -value;
                        check(cudaMemcpyAsync(wide_jobs, wide_descriptors.data(), total * sizeof(wide_descriptors[0]), cudaMemcpyHostToDevice, wide_stream));
                        check(cudaMemcpyAsync(batch_input, input_values.data(), input_values.size() * sizeof(float), cudaMemcpyHostToDevice, wide_stream));
                        check(mollm_cuda::launch_nvfp4_mma_projections(wide_jobs, total, rows, inner, split, wide_stream, nullptr, 1, canonical));
                        check(cudaMemcpyAsync(expected.data(), wide_output, expected.size() * sizeof(float), cudaMemcpyDeviceToHost, wide_stream));
                        check(cudaMemsetAsync(wide_output, 0, got.size() * sizeof(float), wide_stream));
                        check(cudaGraphLaunch(wide_exec, wide_stream));
                        check(cudaMemcpyAsync(got.data(), wide_output, got.size() * sizeof(float), cudaMemcpyDeviceToHost, wide_stream));
                        check(cudaStreamSynchronize(wide_stream));
                        if (std::memcmp(expected.data(), got.data(), got.size() * sizeof(float))) {
                            std::fprintf(stderr, "prepared MMA changed-input/metadata graph mismatch width=%d rows=%d K=%d canonical=%d minimum=%d replay=%d\n",
                                width, rows, inner, canonical, minimum, replay); std::exit(1);
                        }
                    }
                    check(cudaGraphExecDestroy(wide_exec)); check(cudaGraphDestroy(wide_graph));
                }
                check(cudaStreamDestroy(wide_stream));
            }
            check(cudaFree(wide_output)); check(cudaFree(wide_jobs)); check(cudaFree(wide_plan));
            check(cudaFree(wide_compact)); check(cudaFree(wide_packed)); check(cudaFree(wide_scales));
            std::printf("prepared_grouped_N8_N16_N32 rows=%d K=%d pair=%d total=%d thresholds_ragged_reverse_scatter_changed_metadata_graph=true bit_exact=true\n",
                rows, inner, pair_layout,total);
        }
        check(cudaFree(grouped_output)); check(cudaFree(grouped_jobs)); check(cudaFree(batches));
        check(cudaFree(compact_batches));
        check(cudaFree(prepared_input)); check(cudaFree(prepared_scales));
        check(cudaFree(batch_input)); check(cudaFree(batch_output)); check(cudaFree(batch_jobs));
    }
    if (!benchmark) {
        std::vector<float> reference(static_cast<size_t>(experts)*rows), candidate(reference.size());
        check(mollm_cuda::launch_nvfp4_pipelined_projections(jobs.data(),experts,rows,inner));
        check(cudaMemcpy(reference.data(),output,reference.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(int row_group:{2,4}) {
            check(mollm_cuda::launch_nvfp4_row_reuse(jobs.data(),experts,rows,inner,row_group));
            check(cudaMemcpy(candidate.data(),output,candidate.size()*sizeof(float),cudaMemcpyDeviceToHost));
            if(std::memcmp(reference.data(),candidate.data(),candidate.size()*sizeof(float))) {
                std::fprintf(stderr,"row reuse shape mismatch rows=%d K=%d group=%d\n",rows,inner,row_group);
                std::exit(1);
            }
        }
    }
    if (!benchmark && rows % 8 == 0) {
        const int width = rows / 2;
        auto* fused_output = device_allocate<float>(static_cast<size_t>(experts) * width);
        auto* activation_reference = device_allocate<float>(static_cast<size_t>(experts) * width);
        auto* route_weights = device_allocate<float>(experts);
        std::vector<float> weights(experts), reference(static_cast<size_t>(experts) * width), fused(reference.size());
        for (int i = 0; i < experts; ++i) weights[i] = (i % 3 - 1) * .375f;
        check(cudaMemcpy(route_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (bool weighted : {false, true}) for (int hint : {0, 3}) {
            check(mollm_cuda::launch_nvfp4_pipelined_projections(jobs.data(), experts, rows, inner));
            mollm_cuda::launch_moe_activation(output, output + width,
                weighted ? route_weights : nullptr, activation_reference, experts, width, rows);
            check(cudaGetLastError());
            check(mollm_cuda::launch_nvfp4_gate_activation(jobs.data(), experts, width, inner,
                weighted ? route_weights : nullptr, fused_output, nullptr, 4, hint));
            check(cudaMemcpy(reference.data(), activation_reference, reference.size() * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(fused.data(), fused_output, fused.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(reference.data(), fused.data(), reference.size() * sizeof(float))) {
                std::fprintf(stderr, "gate activation fusion mismatch rows=%d K=%d pair=%d weighted=%d\n",
                    rows,inner,pair_layout,weighted);
                std::exit(1);
            }
        }
        std::printf("gate_activation_fusion rows=%d K=%d pair=%d hints=0,3 weighted_and_unweighted=true bit_exact=true\n",
            rows,inner,pair_layout);
        check(cudaFree(fused_output)); check(cudaFree(activation_reference)); check(cudaFree(route_weights));
    }
    if (normalized_error > 2e-5) {
        std::fprintf(stderr, "MMA mismatch rows=%d K=%d pair=%d normalized_max_error=%.9g\n", rows, inner, pair_layout, normalized_error);
        for (int i = 0; i < std::min<int>(32, actual.size()); ++i)
            std::fprintf(stderr, "%d actual=%g expected=%g\n", i, actual[i], expected[i]);
        std::exit(1);
    }
    std::printf("validated layout=%s rows=%d K=%d normalized_max_error=%.9g activation_quant_relative_l2=%.6f\n",
        pair_layout ? "pair" : "row", rows, inner, normalized_error,
        std::sqrt(quant_error / std::max(1e-30, original_energy)));
    // Optional allocation-path diagnostic; all copies happen before timing.
    // This does not propose a second model-sized expert copy in production.
    uint8_t* device_weights = nullptr;
    uint8_t* device_native_weights = nullptr;
    const char* pool_option = std::getenv("MOLLM_NVFP4_PROBE_DEVICE_POOL");
    const bool managed_pool = benchmark && pool_option && std::strcmp(pool_option, "managed") == 0;
    const bool pinned_pool = benchmark && pool_option && std::strcmp(pool_option, "pinned") == 0;
    const bool device_pool = benchmark && pool_option &&
        (std::strcmp(pool_option, "1") == 0 || managed_pool || pinned_pool);
    if (device_pool) {
        if (pinned_pool) {
            check(cudaHostAlloc(&device_weights, stride * jobs.size(), cudaHostAllocMapped));
            check(cudaHostAlloc(&device_native_weights,
                (native_weight_bytes + native_scale_bytes) * jobs.size(), cudaHostAllocMapped));
        } else if (managed_pool) {
            check(cudaMallocManaged(&device_weights, stride * jobs.size()));
            check(cudaMallocManaged(&device_native_weights,
                (native_weight_bytes + native_scale_bytes) * jobs.size()));
        } else {
            device_weights = device_allocate<uint8_t>(stride * jobs.size());
            device_native_weights = device_allocate<uint8_t>(
                (native_weight_bytes + native_scale_bytes) * jobs.size());
        }
        if (pinned_pool) {
            std::memcpy(device_weights, resident.data, stride * jobs.size());
            std::memcpy(device_native_weights, native_resident.data,
                (native_weight_bytes + native_scale_bytes) * jobs.size());
        } else {
            check(cudaMemcpy(device_weights, resident.data, stride * jobs.size(), cudaMemcpyHostToDevice));
            check(cudaMemcpy(device_native_weights, native_resident.data,
                (native_weight_bytes + native_scale_bytes) * jobs.size(), cudaMemcpyHostToDevice));
        }
        if (managed_pool) {
            int device = 0;
            check(cudaGetDevice(&device));
#if CUDART_VERSION >= 13000
            cudaMemLocation location{cudaMemLocationTypeDevice, device};
            check(cudaMemPrefetchAsync(device_weights, stride * jobs.size(), location, 0, cudaStreamPerThread));
            check(cudaMemPrefetchAsync(device_native_weights,
                (native_weight_bytes + native_scale_bytes) * jobs.size(), location, 0, cudaStreamPerThread));
#else
            check(cudaMemPrefetchAsync(device_weights, stride * jobs.size(), device, cudaStreamPerThread));
            check(cudaMemPrefetchAsync(device_native_weights,
                (native_weight_bytes + native_scale_bytes) * jobs.size(), device, cudaStreamPerThread));
#endif
            check(cudaStreamSynchronize(cudaStreamPerThread));
        }
        for (size_t i = 0; i < jobs.size(); ++i) {
            jobs[i].weight = device_weights + i * stride;
            jobs[i].block_scales = jobs[i].weight + weight_bytes;
            jobs[i].row_scales = reinterpret_cast<const float*>(jobs[i].block_scales + scale_bytes);
            native_jobs[i].weight = device_native_weights + i * (native_weight_bytes + native_scale_bytes);
            native_jobs[i].block_scales = native_jobs[i].weight + native_weight_bytes;
            native_jobs[i].row_scales = jobs[i].row_scales;
        }
    }
    if (benchmark && std::getenv("MOLLM_NVFP4_SCALAR_PREPARE_ROTATING")) {
        const int split = rows >= 128 && (rows / 2) % 64 == 0 ? rows / 2 : 0;
        auto descriptors = jobs;
        for(size_t i=0;i<descriptors.size();++i) {
            descriptors[i].input_global=quantized.global*(1+(i%experts)*.125f);
            descriptors[i].input_global_second=descriptors[i].input_global*1.75f;
            descriptors[i].output_scale=(1+i%experts)*.125f;
        }
        const size_t variants=split?2:1;
        auto* prepared_input=device_allocate<uint8_t>(experts*variants*inner/2);
        auto* prepared_scale=device_allocate<uint8_t>(experts*variants*inner/16);
        std::vector<float> reference(size_t(experts)*rows),actual(reference.size());
        auto launch=[&](int i,cudaStream_t stream,bool prepared) {
            check(mollm_cuda::launch_nvfp4_mma_projections(
                descriptors.data()+(i%sets)*experts,experts,rows,inner,split,
                stream,nullptr,1,true,prepared?prepared_input:nullptr,prepared?prepared_scale:nullptr));
        };
        for(int set=0;set<sets;++set) {
            launch(set,nullptr,false);
            check(cudaMemcpy(reference.data(),output,reference.size()*sizeof(float),cudaMemcpyDeviceToHost));
            launch(set,nullptr,true);
            check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
            if(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float))) {
                std::fprintf(stderr,"rotating prepared MMA mismatch set=%d\n",set);std::exit(1);
            }
        }
        constexpr int iterations=128;
        auto timing=[&](bool prepared) {
            const double us=measure_graph([&](int i,cudaStream_t stream){launch(i,stream,prepared);},iterations);
            check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
            launch(iterations-1,nullptr,false);
            check(cudaMemcpy(reference.data(),output,reference.size()*sizeof(float),cudaMemcpyDeviceToHost));
            if(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float))) {
                std::fprintf(stderr,"rotating prepared final graph mismatch\n");std::exit(1);
            }
            return us;
        };
        for(int repeat=0;repeat<3;++repeat) {
            double candidate,control;
            if(repeat%2){candidate=timing(true);control=timing(false);}
            else {control=timing(false);candidate=timing(true);}
            std::printf("scalar_prepare_rotating rows=%d K=%d pair=%d sets=%d weight_bytes=%zu repeat=%d original_us=%.3f prepared_us=%.3f bit_exact=true\n",
                rows,inner,pair_layout,sets,projection_payload*sets,repeat,control,candidate);
        }
        cudaStream_t changed_stream;cudaGraph_t changed_graph;cudaGraphExec_t changed_exec;
        check(cudaStreamCreateWithFlags(&changed_stream,cudaStreamNonBlocking));
        check(cudaStreamBeginCapture(changed_stream,cudaStreamCaptureModeThreadLocal));
        launch(sets-1,changed_stream,true);
        check(cudaStreamEndCapture(changed_stream,&changed_graph));
        check(cudaGraphInstantiateWithFlags(&changed_exec,changed_graph,0));
        auto changed=original;
        for(int replay=0;replay<2;++replay) {
            for(float& value:changed)value=-value;
            check(cudaMemcpyAsync(input,changed.data(),inner*sizeof(float),cudaMemcpyHostToDevice,changed_stream));
            check(cudaGraphLaunch(changed_exec,changed_stream));
            check(cudaMemcpyAsync(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost,changed_stream));
            launch(sets-1,changed_stream,false);
            check(cudaMemcpyAsync(reference.data(),output,reference.size()*sizeof(float),cudaMemcpyDeviceToHost,changed_stream));
            check(cudaStreamSynchronize(changed_stream));
            if(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float))) {
                std::fprintf(stderr,"rotating prepared changed input mismatch\n");std::exit(1);
            }
        }
        check(cudaGraphExecDestroy(changed_exec));check(cudaGraphDestroy(changed_graph));
        check(cudaStreamDestroy(changed_stream));
        check(cudaFree(prepared_input));check(cudaFree(prepared_scale));
        std::puts("scalar_prepare_rotating changed_input_graph_bit_exact=true");
    }
    if (std::getenv("MOLLM_NVFP4_READ_PROBE")) {
        std::vector<uint32_t> expected(size_t(rows) * jobs.size()), actual(size_t(rows) * experts + 8);
        std::vector<uint32_t> weights_expected(expected.size());
        for (size_t index = 0; index < jobs.size(); ++index) {
            const auto* weights = resident.data + index * stride;
            const auto* scales = weights + weight_bytes;
            for (int row = 0; row < rows; ++row) for (int quad = 0; quad < inner / 4; ++quad) {
                const int group = quad / 4;
                const size_t offset = pair_layout
                    ? (size_t(row / 4) * (inner / 16) + group) * 32 + (row % 4) * 8 + (quad % 4) * 2
                    : size_t(row) * (inner / 2) + quad * 2;
                const uint32_t packet = weights[offset] | (uint32_t(weights[offset + 1]) << 8);
                expected[index * rows + row] += packet + (uint32_t(scales[size_t(row) * (inner / 16) + group]) << 16);
                weights_expected[index * rows + row] += packet;
            }
        }
        auto verify = [&](int set, bool with_scales = true) {
            check(cudaMemcpy(actual.data(), output, actual.size() * 4, cudaMemcpyDeviceToHost));
            const auto& selected = with_scales ? expected : weights_expected;
            if (std::memcmp(actual.data(), selected.data() + size_t(set) * experts * rows, size_t(rows) * experts * 4)) {
                std::fprintf(stderr, "expert read checksum mismatch\n"); std::exit(1);
            }
            for (size_t i = size_t(rows) * experts; i < actual.size(); ++i)
                if (actual[i] != 0x7f7f7f7fu) { std::fprintf(stderr, "expert read guard corrupted\n"); std::exit(1); }
        };
        check(cudaMemset(output, 0x7f, actual.size() * 4));
        for (int set = 0; set < sets; ++set) {
            nvfp4_read_probe<><<<dim3((rows + 3) / 4, experts), 128>>>(jobs.data() + set * experts, rows, inner);
            check(cudaGetLastError()); verify(set);
            nvfp4_read_probe<false><<<dim3((rows + 3) / 4, experts), 128>>>(jobs.data() + set * experts, rows, inner);
            check(cudaGetLastError()); verify(set, false);
        }
        if (benchmark) for (int repeat = 0; repeat < 3; ++repeat) {
            const int iterations = std::max(sets, 256);
            auto read = [&] {
                const double us = measure_graph([&](int i, cudaStream_t stream) {
                    nvfp4_read_probe<><<<dim3((rows + 3) / 4, experts), 128, 0, stream>>>(jobs.data() + (i % sets) * experts, rows, inner);
                    check(cudaGetLastError());
                }, iterations);
                verify((iterations - 1) % sets);
                return us;
            };
            auto dot = [&] { return measure_graph([&](int i, cudaStream_t stream) {
                check(mollm_cuda::launch_nvfp4_pipelined_projections(jobs.data() + (i % sets) * experts, experts, rows, inner, true, stream));
            }, iterations); };
            double read_us, dot_us;
            if (repeat % 2) { read_us = read(); dot_us = dot(); }
            else { dot_us = dot(); read_us = read(); }
            std::printf("expert-read rows=%d K=%d pair=%d sets=%d repeat=%d read_us=%.3f projection_us=%.3f checksum_exact=1\n",
                        rows, inner, pair_layout, sets, repeat, read_us, dot_us);
            const double weights_us = measure_graph([&](int i, cudaStream_t stream) {
                nvfp4_read_probe<false><<<dim3((rows + 3) / 4, experts), 128, 0, stream>>>(jobs.data() + (i % sets) * experts, rows, inner);
                check(cudaGetLastError());
            }, iterations);
            verify((iterations - 1) % sets, false);
            std::printf("expert-weight-read rows=%d K=%d pair=%d repeat=%d weights_us=%.3f checksum_exact=1\n",
                        rows, inner, pair_layout, repeat, weights_us);
        }
        if (pinned_pool) { check(cudaFreeHost(device_weights)); check(cudaFreeHost(device_native_weights)); }
        else { check(cudaFree(device_weights)); check(cudaFree(device_native_weights)); }
        check(cudaFree(input)); check(cudaFree(qinput)); check(cudaFree(qscales));
        check(cudaFree(output)); check(cudaFree(partial));
        return;
    }
    if (std::getenv("MOLLM_NVFP4_BF16_MMA_PROBE") || std::getenv("MOLLM_NVFP4_SIMT_SUBWARP") || std::getenv("MOLLM_NVFP4_INTEGER_SCALE") || std::getenv("MOLLM_NVFP4_CTA_PROBE") || std::getenv("MOLLM_NVFP4_INTERLEAVED")) {
        std::vector<float> reference(static_cast<size_t>(experts) * rows), actual(reference.size());
        const std::vector<float> guards(8, -1234567.f);
        std::vector<float> actual_guards(8);
        check(cudaMemcpy(output + reference.size(), guards.data(), guards.size() * sizeof(float), cudaMemcpyHostToDevice));
        const bool cache_input = std::getenv("MOLLM_NVFP4_BF16_CACHE_INPUT") != nullptr;
        const bool block_weights = std::getenv("MOLLM_NVFP4_BF16_BLOCK_WEIGHTS") != nullptr;
        const bool split_k = std::getenv("MOLLM_NVFP4_BF16_SPLIT_K") != nullptr;
        const bool subwarp = std::getenv("MOLLM_NVFP4_SIMT_SUBWARP") != nullptr;
        const bool integer_scale = std::getenv("MOLLM_NVFP4_INTEGER_SCALE") != nullptr;
        const bool cta_probe = std::getenv("MOLLM_NVFP4_CTA_PROBE") != nullptr;
        const bool interleaved = std::getenv("MOLLM_NVFP4_INTERLEAVED") != nullptr;
        uint8_t* interleaved_weights = nullptr;
        auto packed_jobs = jobs;
        if (interleaved) {
            if (cta_probe || integer_scale || subwarp || split_k || cache_input || block_weights) {
                std::fprintf(stderr, "interleaved is a separate layout probe\n"); std::exit(1);
            }
            const size_t packed_stride = weight_bytes + scale_bytes;
            std::vector<uint8_t> payload(packed_stride * jobs.size(), 0);
            for (size_t j = 0; j < jobs.size(); ++j) {
                const auto* canonical = resident.data + j * stride;
                for (int row = 0; row < rows; ++row) for (int group = 0; group < inner / 16; ++group) {
                    const size_t tile = (size_t(row / 4) * (inner / 16) + group) * 36;
                    const size_t source = pair_layout
                        ? (size_t(row / 4) * (inner / 16) + group) * 32 + (row % 4) * 8
                        : size_t(row) * (inner / 2) + group * 8;
                    auto* destination = payload.data() + j * packed_stride;
                    std::memcpy(destination + tile + (row % 4) * 8, canonical + source, 8);
                    destination[tile + 32 + row % 4] = canonical[weight_bytes + size_t(row) * (inner / 16) + group];
                }
            }
            interleaved_weights = device_allocate<uint8_t>(payload.size());
            check(cudaMemcpy(interleaved_weights, payload.data(), payload.size(), cudaMemcpyHostToDevice));
            for (size_t j = 0; j < jobs.size(); ++j) {
                packed_jobs[j].weight = interleaved_weights + j * packed_stride;
                packed_jobs[j].block_scales = nullptr; // Must not read original scales in this candidate.
            }
            std::printf("interleaved diagnostic extra_payload_bytes=%zu tile_bytes=36\n", payload.size());
            if (std::getenv("MOLLM_NVFP4_INTERLEAVED_RUNTIME_PROBE")) {
                if (mollm_cuda::launch_nvfp4_interleaved_projections(nullptr, 1, rows, inner) != cudaErrorInvalidValue ||
                    mollm_cuda::launch_nvfp4_interleaved_projections(packed_jobs.data(), experts, 0, inner) != cudaErrorInvalidValue ||
                    mollm_cuda::launch_nvfp4_interleaved_projections(packed_jobs.data(), experts, rows, 17) != cudaErrorInvalidValue ||
                    mollm_cuda::launch_nvfp4_interleaved_projections(nullptr, 0, rows, inner) != cudaSuccess) {
                    std::fprintf(stderr, "interleaved argument validation failed\n"); std::exit(1);
                }
            }
        }
        if (cta_probe) {
            if (mollm_cuda::launch_nvfp4_cta_projections(nullptr, 1, rows, inner, 4) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_cta_projections(jobs.data(), experts, rows, inner, 3) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_cta_projections(jobs.data(), experts, rows, 17, 4) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_cta_projections(nullptr, 0, rows, inner, 4) != cudaSuccess) {
                std::fprintf(stderr, "CTA argument validation failed\n"); std::exit(1);
            }
        }
        if (cta_probe && (integer_scale || subwarp || split_k || cache_input || block_weights)) {
            std::fprintf(stderr, "CTA is a separate canonical pipeline probe\n"); std::exit(1);
        }
        if (integer_scale && (subwarp || split_k || cache_input || block_weights)) {
            std::fprintf(stderr, "integer scale is a separate 32-lane pipeline probe\n"); std::exit(1);
        }
        const bool subwarp_pipeline = subwarp && std::getenv("MOLLM_NVFP4_SUBWARP_PIPELINE");
        if (subwarp && (split_k || cache_input || block_weights)) { std::fprintf(stderr, "subwarp is a separate probe\n"); std::exit(1); }
        const char* probe = interleaved ? "interleaved" : cta_probe ? "cta" : integer_scale ? "integer-scale" : subwarp ? "subwarp" : "bf16";
        std::printf("projection_schedule probe=%s; %s\n", probe,
            interleaved ? "36-byte four-row weight+scale tiles, 32 lanes, depth1 pipeline" :
            cta_probe ? "canonical 32 lanes, depth1 pipeline, CTA warps1/2/4/8/16" :
            integer_scale ? "integer scale uses 32 lanes, depth1 input pipeline" : "subwarp terms1=16lanes terms2=8lanes");
        std::printf("projection_schedule subwarp_pipeline=%d\n", subwarp_pipeline);
        if (split_k && cache_input) { std::fprintf(stderr, "split-K probe excludes cached activation schedule\n"); std::exit(1); }
        const size_t partial_count = static_cast<size_t>(experts) * rows * 16;
        auto* bf16_partial = split_k ? device_allocate<float>(partial_count + 8) : nullptr;
        if (split_k) check(cudaMemcpy(bf16_partial + partial_count, guards.data(), guards.size() * sizeof(float), cudaMemcpyHostToDevice));
        int parts = 1;
        if (cache_input && block_weights) { std::fprintf(stderr, "test schedules separately\n"); std::exit(1); }
        std::printf("bf16_schedule cache_input=%d shared_max_bytes=%d\n", cache_input, cache_input ? inner * 4 : 0);
        std::printf("bf16_schedule block_weights=%d\n", block_weights);
        auto launch = [&](int terms, int warps, const mollm_cuda::Nvfp4Projection* selected, cudaStream_t stream) {
            const dim3 grid((rows + warps * 16 - 1) / (warps * 16), experts, parts);
            if (interleaved && std::getenv("MOLLM_NVFP4_INTERLEAVED_RUNTIME_PROBE"))
                check(mollm_cuda::launch_nvfp4_interleaved_projections(
                    packed_jobs.data() + (selected - jobs.data()), experts, rows, inner, stream));
            else if (interleaved) nvfp4_subwarp_probe<32, true, false, true><<<dim3((rows + warps - 1) / warps, experts), warps * 32, 0, stream>>>(
                    packed_jobs.data() + (selected - jobs.data()), rows, inner);
            else if (cta_probe) check(mollm_cuda::launch_nvfp4_cta_projections(selected, experts, rows, inner, warps, stream));
            else if (integer_scale) nvfp4_subwarp_probe<32, true, true><<<dim3((rows + warps - 1) / warps, experts), warps * 32, 0, stream>>>(selected, rows, inner);
            else if (subwarp_pipeline && terms == 1) nvfp4_subwarp_probe<16, true><<<dim3((rows + warps * 2 - 1) / (warps * 2), experts), warps * 32, 0, stream>>>(selected, rows, inner);
            else if (subwarp_pipeline) nvfp4_subwarp_probe<8, true><<<dim3((rows + warps * 4 - 1) / (warps * 4), experts), warps * 32, 0, stream>>>(selected, rows, inner);
            else if (subwarp && terms == 1) nvfp4_subwarp_probe<16><<<dim3((rows + warps * 2 - 1) / (warps * 2), experts), warps * 32, 0, stream>>>(selected, rows, inner);
            else if (subwarp) nvfp4_subwarp_probe<8><<<dim3((rows + warps * 4 - 1) / (warps * 4), experts), warps * 32, 0, stream>>>(selected, rows, inner);
            else if (split_k && block_weights && terms == 1) nvfp4_bf16_mma_probe<1, false, true><<<grid, warps * 32, 0, stream>>>(selected, rows, inner, bf16_partial, parts);
            else if (split_k && block_weights) nvfp4_bf16_mma_probe<2, false, true><<<grid, warps * 32, 0, stream>>>(selected, rows, inner, bf16_partial, parts);
            else if (split_k && terms == 1) nvfp4_bf16_mma_probe<1><<<grid, warps * 32, 0, stream>>>(selected, rows, inner, bf16_partial, parts);
            else if (split_k) nvfp4_bf16_mma_probe<2><<<grid, warps * 32, 0, stream>>>(selected, rows, inner, bf16_partial, parts);
            else if (block_weights && terms == 1) nvfp4_bf16_mma_probe<1, false, true><<<grid, warps * 32, 0, stream>>>(selected, rows, inner);
            else if (block_weights) nvfp4_bf16_mma_probe<2, false, true><<<grid, warps * 32, 0, stream>>>(selected, rows, inner);
            else if (cache_input && terms == 1) nvfp4_bf16_mma_probe<1, true><<<grid, warps * 32, inner * 2, stream>>>(selected, rows, inner);
            else if (cache_input) nvfp4_bf16_mma_probe<2, true><<<grid, warps * 32, inner * 4, stream>>>(selected, rows, inner);
            else if (terms == 1) nvfp4_bf16_mma_probe<1><<<grid, warps * 32, 0, stream>>>(selected, rows, inner);
            else nvfp4_bf16_mma_probe<2><<<grid, warps * 32, 0, stream>>>(selected, rows, inner);
            check(cudaGetLastError());
            if (parts > 1) {
                combine_bf16_parts<<<dim3((rows + 255) / 256, experts), 256, 0, stream>>>(selected, bf16_partial, rows, experts, parts);
                check(cudaGetLastError());
            }
        };
        for (int split : split_k ? std::vector<int>{1, 2, 4, 8, 16} : std::vector<int>{1}) {
        if (split > inner / 16) continue;
        parts = split;
        for (int terms : (integer_scale || cta_probe || interleaved) ? std::vector<int>{1} : std::vector<int>{1, 2}) for (int warps : cta_probe ? std::vector<int>{1, 2, 4, 8, 16} : (integer_scale || interleaved) ? std::vector<int>{4} : split_k ? std::vector<int>{1, 4} : std::vector<int>{1, 2, 4, 8}) {
            double worst = 0, difference = 0, energy = 0, cpu_error = 0, cpu_max = 0;
            for (int set = 0; set < sets; ++set) {
                const auto* selected = jobs.data() + set * experts;
                check(mollm_cuda::launch_nvfp4_pipelined_projections(selected, experts, rows, inner, true));
                check(cudaMemcpy(reference.data(), output, reference.size() * sizeof(float), cudaMemcpyDeviceToHost));
                check(cudaMemset(output, 0xff, actual.size() * sizeof(float)));
                const size_t used_partial = static_cast<size_t>(experts) * rows * parts;
                if (split_k) {
                    check(cudaMemset(bf16_partial, 0xff, used_partial * sizeof(float)));
                    check(cudaMemcpy(bf16_partial + used_partial, guards.data(), guards.size() * sizeof(float), cudaMemcpyHostToDevice));
                }
                launch(terms, warps, selected, cudaStreamPerThread);
                check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if ((integer_scale || cta_probe || interleaved) && std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float))) {
                    std::fprintf(stderr, "integer scale changed projection bits\n"); std::exit(1);
                }
                if ((cache_input || block_weights || subwarp_pipeline) && parts == 1) {
                    const dim3 grid((rows + warps * 16 - 1) / (warps * 16), experts);
                    if (subwarp && terms == 1) nvfp4_subwarp_probe<16><<<dim3((rows + warps * 2 - 1) / (warps * 2), experts), warps * 32>>>(selected, rows, inner);
                    else if (subwarp) nvfp4_subwarp_probe<8><<<dim3((rows + warps * 4 - 1) / (warps * 4), experts), warps * 32>>>(selected, rows, inner);
                    else if (terms == 1) nvfp4_bf16_mma_probe<1><<<grid, warps * 32>>>(selected, rows, inner);
                    else nvfp4_bf16_mma_probe<2><<<grid, warps * 32>>>(selected, rows, inner);
                    check(cudaGetLastError());
                    std::vector<float> uncached(actual.size());
                    check(cudaMemcpy(uncached.data(), output, uncached.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    if (std::memcmp(actual.data(), uncached.data(), actual.size() * sizeof(float))) {
                        std::fprintf(stderr, "BF16 schedule changed result\n"); std::exit(1);
                    }
                }
                check(cudaMemcpy(actual_guards.data(), output + reference.size(), guards.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (actual_guards != guards) { std::fprintf(stderr, "BF16 output guard corrupted\n"); std::exit(1); }
                if (split_k) {
                    check(cudaMemcpy(actual_guards.data(), bf16_partial + used_partial, guards.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    if (actual_guards != guards) { std::fprintf(stderr, "BF16 partial guard corrupted\n"); std::exit(1); }
                }
                double peak = 0, error = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(actual[i])) { std::fprintf(stderr, "nonfinite BF16 result\n"); std::exit(1); }
                    peak = std::max(peak, std::abs(double(reference[i])));
                    error = std::max(error, std::abs(double(actual[i]) - reference[i]));
                    difference += std::pow(double(actual[i]) - reference[i], 2);
                    energy += double(reference[i]) * reference[i];
                    if (!set) {
                        cpu_error = std::max(cpu_error, std::abs(actual[i] - original_expected[i]));
                        cpu_max = std::max(cpu_max, std::abs(original_expected[i]));
                    }
                }
                worst = std::max(worst, error / std::max(1., peak));
            }
            std::printf("%s_check rows=%d K=%d pair=%d terms=%d warps=%d parts=%d normalized_max=%.9g relative_l2=%.9g cpu_normalized_max=%.9g\n",
                probe, rows, inner, pair_layout, terms, warps, parts, worst,
                std::sqrt(difference / std::max(1e-30, energy)), cpu_error / std::max(1., cpu_max));
            const double tolerance = (subwarp || integer_scale || cta_probe || interleaved) ? .00001 : terms == 1 ? .01 : .0001;
            if (worst > tolerance || cpu_error / std::max(1., cpu_max) > tolerance) {
                std::fprintf(stderr, "BF16 numerical gate failed\n"); std::exit(1);
            }
            if (benchmark && !std::getenv("MOLLM_NVFP4_BF16_CHECK_ONLY")) for (int repeat = 0; repeat < 3; ++repeat) {
                auto candidate = [&] {
                    const int iterations = std::max(sets, 256);
                    const double time = measure_graph([&](int i, cudaStream_t stream) {
                    launch(terms, warps, jobs.data() + (i % sets) * experts, stream);
                    }, iterations);
                    check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    const auto* last = jobs.data() + ((iterations - 1) % sets) * experts;
                    check(mollm_cuda::launch_nvfp4_pipelined_projections(last, experts, rows, inner, true));
                    check(cudaMemcpy(reference.data(), output, reference.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    if ((integer_scale || cta_probe || interleaved) && std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float))) {
                        std::fprintf(stderr, "integer scale changed graph projection bits\n"); std::exit(1);
                    }
                    double peak = 0, error = 0;
                    for (size_t i = 0; i < actual.size(); ++i) {
                        if (!std::isfinite(actual[i])) { std::fprintf(stderr, "nonfinite BF16 graph output\n"); std::exit(1); }
                        peak = std::max(peak, std::abs(double(reference[i])));
                        error = std::max(error, std::abs(double(actual[i]) - reference[i]));
                    }
                    if (error / std::max(1., peak) > tolerance) {
                        std::fprintf(stderr, "BF16 graph result gate failed\n"); std::exit(1);
                    }
                    return time;
                };
                auto control = [&] { return measure_graph([&](int i, cudaStream_t stream) {
                    check(mollm_cuda::launch_nvfp4_pipelined_projections(jobs.data() + (i % sets) * experts,
                        experts, rows, inner, true, stream));
                }, std::max(sets, 256)); };
                double baseline, mma;
                double packet = 0;
                auto packet_control = [&] { return measure_graph([&](int i, cudaStream_t stream) {
                    nvfp4_subwarp_probe<32, true><<<dim3((rows + 3) / 4, experts), 128, 0, stream>>>(
                        jobs.data() + (i % sets) * experts, rows, inner);
                    check(cudaGetLastError());
                }, std::max(sets, 256)); };
                if ((integer_scale || interleaved) && repeat == 0) { baseline = control(); packet = packet_control(); mma = candidate(); }
                else if ((integer_scale || interleaved) && repeat == 1) { mma = candidate(); packet = packet_control(); baseline = control(); }
                else if (integer_scale || interleaved) { packet = packet_control(); baseline = control(); mma = candidate(); }
                else if (repeat % 2) { mma = candidate(); baseline = control(); }
                else { baseline = control(); mma = candidate(); }
                std::printf("%s_time rows=%d K=%d pair=%d terms=%d warps=%d parts=%d sets=%d repeat=%d pipeline_us=%.3f candidate_us=%.3f\n",
                    probe, rows, inner, pair_layout, terms, warps, parts, sets, repeat, baseline, mma);
                if (integer_scale) std::printf("integer-scale_packet_control rows=%d K=%d pair=%d repeat=%d intrinsic_us=%.3f integer_us=%.3f\n",
                    rows, inner, pair_layout, repeat, packet, mma);
                if (interleaved) std::printf("interleaved_packet_control rows=%d K=%d pair=%d repeat=%d original_us=%.3f packed_us=%.3f\n",
                    rows, inner, pair_layout, repeat, packet, mma);
            }
        }
        }
        if (bf16_partial) check(cudaFree(bf16_partial));
        if (interleaved_weights) check(cudaFree(interleaved_weights));
        if (pinned_pool) { check(cudaFreeHost(device_weights)); check(cudaFreeHost(device_native_weights)); }
        else { check(cudaFree(device_weights)); check(cudaFree(device_native_weights)); }
        check(cudaFree(input)); check(cudaFree(qinput)); check(cudaFree(qscales));
        check(cudaFree(output)); check(cudaFree(partial));
        return;
    }
    const bool paired_simt = std::getenv("MOLLM_NVFP4_PAIRED_SIMT_BENCH") != nullptr;
    if (benchmark && std::getenv("MOLLM_NVFP4_GATE_ACTIVATION_BENCH")) {
        const int fused_load_hint = std::getenv("MOLLM_NVFP4_GATE_ACTIVATION_L2_FIRST") ? 3 : 0;
        const char* warp_option = std::getenv("MOLLM_NVFP4_GATE_ACTIVATION_WARPS");
        const int fused_warps = warp_option && std::strcmp(warp_option,"2")==0 ? 2 :
            warp_option && std::strcmp(warp_option,"8")==0 ? 8 : 4;
        const int width = rows / 2;
        auto* fused_output = device_allocate<float>(static_cast<size_t>(experts) * width);
        auto* separate_output = device_allocate<float>(static_cast<size_t>(experts) * width);
        auto* route_weights = device_allocate<float>(experts);
        std::vector<float> weights(experts), expected(static_cast<size_t>(experts) * width), actual(expected.size());
        for (int i = 0; i < experts; ++i) weights[i] = (i % 5 - 2) * .125f;
        check(cudaMemcpy(route_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto launch = [&](bool fused, int set) {
            const auto* selected = jobs.data() + set * experts;
            if (fused) check(mollm_cuda::launch_nvfp4_gate_activation(selected, experts,
                width, inner, route_weights, fused_output, stream, fused_warps, fused_load_hint));
            else {
                check(mollm_cuda::launch_nvfp4_pipelined_projections(selected, experts,
                    rows, inner, true, stream));
                mollm_cuda::launch_moe_activation(output, output + width, route_weights,
                    separate_output, experts, width, rows, stream);
                check(cudaGetLastError());
            }
        };
        auto compare = [&](bool fused) {
            check(cudaMemcpyAsync(actual.data(), fused ? fused_output : separate_output,
                actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
        };
        for (int set = 0; set < sets; ++set) {
            launch(false,set); compare(false); expected = actual;
            launch(true,set); compare(true);
            if (std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float))) {
                std::fprintf(stderr,"fusion rotating-set mismatch set=%d\n",set); std::exit(1);
            }
        }
        auto measure = [&](bool fused) {
            const int iterations = std::max(sets,256), last = (iterations-1)%sets;
            cudaGraph_t graph; cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            for (int i=0;i<iterations;++i) launch(fused,i%sets);
            check(cudaStreamEndCapture(stream,&graph));
            check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            for (int i=0;i<2;++i) check(cudaGraphLaunch(executable,stream));
            cudaEvent_t start,stop;
            check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
            check(cudaEventRecord(start,stream));
            for(int i=0;i<4;++i) check(cudaGraphLaunch(executable,stream));
            check(cudaEventRecord(stop,stream)); check(cudaEventSynchronize(stop));
            float ms; check(cudaEventElapsedTime(&ms,start,stop));
            compare(fused); expected=actual;
            launch(!fused,last); compare(!fused);
            if(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float))) {
                std::fprintf(stderr,"fusion timed graph mismatch\n");std::exit(1);
            }
            auto changed=original;
            for(int replay=0;replay<2;++replay) {
                for(float& value:changed)value=-value;
                check(cudaMemcpyAsync(input,changed.data(),inner*sizeof(float),cudaMemcpyHostToDevice,stream));
                check(cudaGraphLaunch(executable,stream));compare(fused);expected=actual;
                launch(!fused,last);compare(!fused);
                if(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float))) {
                    std::fprintf(stderr,"fusion changed-input graph mismatch\n");std::exit(1);
                }
            }
            check(cudaEventDestroy(start));check(cudaEventDestroy(stop));
            check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
            return ms*1000./(4*iterations);
        };
        for(int repeat=0;repeat<3;++repeat) {
            double separate,fused;
            if(repeat%2){fused=measure(true);separate=measure(false);}
            else {separate=measure(false);fused=measure(true);}
            std::printf("gate_activation_timing rows=%d K=%d pair=%d sets=%d warps=%d hint=%d repeat=%d separate_us=%.3f fused_us=%.3f bit_exact_graph=true\n",
                rows,inner,pair_layout,sets,fused_warps,fused_load_hint,repeat,separate,fused);
        }
        check(cudaStreamDestroy(stream));
        check(cudaFree(fused_output));check(cudaFree(separate_output));check(cudaFree(route_weights));
        if(pinned_pool){check(cudaFreeHost(device_weights));check(cudaFreeHost(device_native_weights));}
        else {check(cudaFree(device_weights));check(cudaFree(device_native_weights));}
        check(cudaFree(input));check(cudaFree(qinput));check(cudaFree(qscales));
        check(cudaFree(output));check(cudaFree(partial));
        return;
    }
    if(benchmark && std::getenv("MOLLM_NVFP4_ROUTE_SUM_PROBE")) {
        const int row_tile=std::getenv("MOLLM_NVFP4_ROUTE_SUM_TWO_ROWS")?2:1;
        std::vector<float> reference(rows),actual(rows);
        auto launch=[&](int i,cudaStream_t stream,bool fused) {
            const auto* selected=jobs.data()+(i%sets)*experts;
            if(fused)check(mollm_cuda::launch_nvfp4_route_sum(selected,experts,rows,inner,partial,stream,row_tile));
            else {
                check(mollm_cuda::launch_nvfp4_pipelined_projections(selected,experts,rows,inner,true,stream));
                route_sum_control<<<(rows+255)/256,256,0,stream>>>(output,partial,rows,experts);
                check(cudaGetLastError());
            }
        };
        for(int set=0;set<sets;++set) {
            launch(set,nullptr,false);check(cudaMemcpy(reference.data(),partial,rows*sizeof(float),cudaMemcpyDeviceToHost));
            launch(set,nullptr,true);check(cudaMemcpy(actual.data(),partial,rows*sizeof(float),cudaMemcpyDeviceToHost));
            if(std::memcmp(reference.data(),actual.data(),rows*sizeof(float))) {std::fprintf(stderr,"route sum mismatch\n");std::exit(1);}
        }
        const int iterations=std::max(sets,256);
        auto timing=[&](bool fused) {
            const double us=measure_graph([&](int i,cudaStream_t stream){launch(i,stream,fused);},iterations);
            check(cudaMemcpy(actual.data(),partial,rows*sizeof(float),cudaMemcpyDeviceToHost));
            launch(iterations-1,nullptr,false);check(cudaMemcpy(reference.data(),partial,rows*sizeof(float),cudaMemcpyDeviceToHost));
            if(std::memcmp(reference.data(),actual.data(),rows*sizeof(float)))std::exit(1);
            return us;
        };
        for(int repeat=0;repeat<3;++repeat) {
            double fused,separate;
            if(repeat%2){separate=timing(false);fused=timing(true);}
            else{fused=timing(true);separate=timing(false);}
            std::printf("route-sum rows=%d K=%d pair=%d routes=%d tile=%d repeat=%d fused_us=%.3f separate_us=%.3f bit_exact=true\n",rows,inner,pair_layout,experts,row_tile,repeat,fused,separate);
        }
        if(pinned_pool){check(cudaFreeHost(device_weights));check(cudaFreeHost(device_native_weights));}
        else {check(cudaFree(device_weights));check(cudaFree(device_native_weights));}
        check(cudaFree(input));check(cudaFree(qinput));check(cudaFree(qscales));check(cudaFree(output));check(cudaFree(partial));
        return;
    }
    if (benchmark && std::getenv("MOLLM_NVFP4_PIPELINE_SIMT_BENCH")) {
        const bool pair_only_probe = std::getenv("MOLLM_NVFP4_PAIR_ONLY_PROBE") != nullptr;
        const bool exact_subwarp_probe = std::getenv("MOLLM_NVFP4_EXACT_SUBWARP_PROBE") != nullptr;
        const bool scale_broadcast_probe = std::getenv("MOLLM_NVFP4_SCALE_BROADCAST_PROBE") != nullptr;
        const bool tiled_scale_probe = std::getenv("MOLLM_NVFP4_TILED_SCALE_PROBE") != nullptr;
        const char* split_hint=std::getenv("MOLLM_NVFP4_SPLIT_CACHE_HINT");
        const bool row_scale_early = std::getenv("MOLLM_NVFP4_ROW_SCALE_EARLY_PROBE") != nullptr;
        const bool row_scale_cta = std::getenv("MOLLM_NVFP4_ROW_SCALE_CTA_PROBE") != nullptr;
        const bool row_scale_mid = std::getenv("MOLLM_NVFP4_ROW_SCALE_MID_PROBE") != nullptr;
        const bool decoded_scale_broadcast = std::getenv("MOLLM_NVFP4_DECODED_SCALE_BROADCAST_PROBE") != nullptr;
        const bool row_scale_prefetch_control = (row_scale_cta || row_scale_mid) &&
            std::getenv("MOLLM_NVFP4_ROW_SCALE_PREFETCH_CONTROL") != nullptr;
        const bool row_scale_prefetch = std::getenv("MOLLM_NVFP4_ROW_SCALE_PREFETCH_PROBE") != nullptr || row_scale_early || row_scale_cta || row_scale_mid || decoded_scale_broadcast;
        const int l2_probe_hint = split_hint ? std::atoi(split_hint) : decoded_scale_broadcast ? 11 : row_scale_mid ? 10 : row_scale_cta ? 9 : row_scale_early ? 8 : row_scale_prefetch ? 7 : std::getenv("MOLLM_NVFP4_L2_LAST_PROBE") ? 4 : 3;
        if(split_hint && l2_probe_hint!=5 && l2_probe_hint!=6){std::fprintf(stderr,"split cache hint must be5 or6\n");std::exit(1);}
        const bool l2_first_probe = std::getenv("MOLLM_NVFP4_L2_FIRST_PROBE") != nullptr || l2_probe_hint == 4 || split_hint || row_scale_prefetch;
        uint8_t* tiled_scales=nullptr;
        mollm_cuda::Nvfp4Projection* tiled_jobs=nullptr;
        if(tiled_scale_probe) {
            std::vector<uint8_t> reordered(scale_bytes*jobs.size());
            auto descriptors=jobs;
            check(cudaMalloc(&tiled_scales,reordered.size()));
            for(size_t j=0;j<jobs.size();++j) {
                const auto* source=resident.data+j*stride+weight_bytes;
                for(int r=0;r<rows;++r)for(int g=0;g<inner/16;++g)
                    reordered[j*scale_bytes+(size_t(r/4)*(inner/16)+g)*4+r%4]=source[size_t(r)*(inner/16)+g];
                descriptors[j].block_scales=tiled_scales+j*scale_bytes;
            }
            check(cudaMemcpy(tiled_scales,reordered.data(),reordered.size(),cudaMemcpyHostToDevice));
            check(cudaMalloc(&tiled_jobs,descriptors.size()*sizeof(descriptors[0])));
            check(cudaMemcpy(tiled_jobs,descriptors.data(),descriptors.size()*sizeof(descriptors[0]),cudaMemcpyHostToDevice));
        }
        uint32_t *hardware_codes=nullptr, *direct_codes=nullptr;
        check(cudaMalloc(&hardware_codes,256*sizeof(uint32_t)));
        check(cudaMalloc(&direct_codes,256*sizeof(uint32_t)));
        check(mollm_cuda::launch_nvfp4_scale_decode_check(hardware_codes,direct_codes));
        uint32_t hardware_bits[256],direct_bits[256];
        check(cudaMemcpy(hardware_bits,hardware_codes,sizeof(hardware_bits),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(direct_bits,direct_codes,sizeof(direct_bits),cudaMemcpyDeviceToHost));
        if(std::memcmp(hardware_bits,direct_bits,sizeof(hardware_bits))) {
            std::fprintf(stderr,"E4M3 direct decode mismatch\n");std::exit(1);
        }
        check(cudaFree(hardware_codes));check(cudaFree(direct_codes));
        std::printf("E4M3 all 256 codes bit_exact=true\n");
        std::vector<float> reference(static_cast<size_t>(experts) * rows), actual(reference.size());
        for (int set = 0; set < sets; ++set) {
            const auto* selected = jobs.data() + set * experts;
            mollm_cuda::launch_nvfp4_projections(selected, experts, rows, inner);
            check(cudaMemcpy(reference.data(), output, reference.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int mode : (l2_first_probe ? std::vector<int>{18} : tiled_scale_probe ? std::vector<int>{17} : scale_broadcast_probe ? std::vector<int>{16} : exact_subwarp_probe ? std::vector<int>{14,15} : pair_only_probe ? (pair_layout ? std::vector<int>{13} : std::vector<int>{1}) : std::vector<int>{0,1,2,3,4,5,6,7,8,9,10,11,12})) {
                if (mode == 6 && inner != 640 && inner != 2560) continue;
                const bool inputs = mode != 0;
                if(mode==18)check(mollm_cuda::launch_nvfp4_pipelined_projections(selected,experts,rows,inner,true,nullptr,false,l2_probe_hint));
                else if(mode==17)check(mollm_cuda::launch_nvfp4_tiled_scale_projections(tiled_jobs+set*experts,experts,rows,inner));
                else if(mode==16)check(mollm_cuda::launch_nvfp4_pipelined_projections(selected,experts,rows,inner,true,nullptr,false,0,false,false,false,false,false,false,false,true));
                else if(mode>=14)check(mollm_cuda::launch_nvfp4_exact_subwarp(selected,experts,rows,inner,mode==14?16:8));
                else if(mode==7||mode==8)check(mollm_cuda::launch_nvfp4_row_reuse(selected,experts,rows,inner,mode==7?2:4));
                else check(mollm_cuda::launch_nvfp4_pipelined_projections(selected, experts, rows, inner, inputs, nullptr, mode == 2, mode >= 3 && mode <= 4 ? mode - 2 : 0, mode == 5, mode == 6, mode==9, mode==10, mode==11, mode==12, mode==13));
                check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (std::memcmp(reference.data(), actual.data(), actual.size() * sizeof(float))) {
                    std::fprintf(stderr, "pipeline bit-exact mismatch set=%d inputs=%d\n", set, inputs);
                    std::exit(1);
                }
            }
        }
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto graph_time = [&](int mode) {
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            const int iterations = std::max(sets, 256);
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < iterations; ++i) {
                const auto* selected = jobs.data() + (i % sets) * experts;
                if (mode < 0) mollm_cuda::launch_nvfp4_projections(selected, experts, rows, inner, stream);
                else if(mode==1 && row_scale_prefetch_control)check(mollm_cuda::launch_nvfp4_pipelined_projections(selected,experts,rows,inner,true,stream,false,7));
                else if(mode==18)check(mollm_cuda::launch_nvfp4_pipelined_projections(selected,experts,rows,inner,true,stream,false,l2_probe_hint));
                else if(mode==17)check(mollm_cuda::launch_nvfp4_tiled_scale_projections(tiled_jobs+(i%sets)*experts,experts,rows,inner,stream));
                else if(mode==16)check(mollm_cuda::launch_nvfp4_pipelined_projections(selected,experts,rows,inner,true,stream,false,0,false,false,false,false,false,false,false,true));
                else if(mode>=14)check(mollm_cuda::launch_nvfp4_exact_subwarp(selected,experts,rows,inner,mode==14?16:8,stream));
                else if(mode==7||mode==8)check(mollm_cuda::launch_nvfp4_row_reuse(selected,experts,rows,inner,mode==7?2:4,stream));
                else check(mollm_cuda::launch_nvfp4_pipelined_projections(selected, experts, rows, inner, mode != 0, stream, mode == 2, mode >= 3 && mode <= 4 ? mode - 2 : 0, mode == 5, mode == 6, mode==9, mode==10, mode==11, mode==12, mode==13));
            }
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int i = 0; i < 2; ++i) check(cudaGraphLaunch(executable, stream));
            cudaEvent_t start, stop;
            check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
            check(cudaEventRecord(start, stream));
            for (int i = 0; i < 4; ++i) check(cudaGraphLaunch(executable, stream));
            check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
            float ms = 0;
            check(cudaEventElapsedTime(&ms, start, stop));
            check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
            check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
            check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            mollm_cuda::launch_nvfp4_projections(jobs.data() + ((iterations - 1) % sets) * experts,
                experts, rows, inner, stream);
            check(cudaGetLastError());
            check(cudaMemcpyAsync(reference.data(), output, reference.size() * sizeof(float),
                cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            if (std::memcmp(reference.data(), actual.data(), actual.size() * sizeof(float))) {
                std::fprintf(stderr, "pipeline final graph mismatch mode=%d\n", mode);
                std::exit(1);
            }
            if(mode>=7) {
                auto changed=original;
                cudaGraph_t changed_graph;cudaGraphExec_t changed_exec;
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                if(mode==18)check(mollm_cuda::launch_nvfp4_pipelined_projections(jobs.data()+((iterations-1)%sets)*experts,experts,rows,inner,true,stream,false,l2_probe_hint));
                else if(mode==17)check(mollm_cuda::launch_nvfp4_tiled_scale_projections(tiled_jobs+((iterations-1)%sets)*experts,experts,rows,inner,stream));
                else if(mode==16)check(mollm_cuda::launch_nvfp4_pipelined_projections(jobs.data()+((iterations-1)%sets)*experts,experts,rows,inner,true,stream,false,0,false,false,false,false,false,false,false,true));
                else if(mode>=14)check(mollm_cuda::launch_nvfp4_exact_subwarp(jobs.data()+((iterations-1)%sets)*experts,experts,rows,inner,mode==14?16:8,stream));
                else if(mode>=9)check(mollm_cuda::launch_nvfp4_pipelined_projections(
                    jobs.data()+((iterations-1)%sets)*experts,experts,rows,inner,
                    true,stream,false,0,false,false,mode==9,mode==10,mode==11,mode==12,mode==13));
                else check(mollm_cuda::launch_nvfp4_row_reuse(jobs.data()+((iterations-1)%sets)*experts,
                    experts,rows,inner,mode==7?2:4,stream));
                check(cudaStreamEndCapture(stream,&changed_graph));
                check(cudaGraphInstantiate(&changed_exec,changed_graph,nullptr,nullptr,0));
                for(int replay=0;replay<2;++replay) {
                    for(float& value:changed)value=-value;
                    check(cudaMemcpyAsync(input,changed.data(),inner*sizeof(float),cudaMemcpyHostToDevice,stream));
                    check(cudaGraphLaunch(changed_exec,stream));
                    check(cudaMemcpyAsync(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));
                    mollm_cuda::launch_nvfp4_projections(jobs.data()+((iterations-1)%sets)*experts,
                        experts,rows,inner,stream);check(cudaGetLastError());
                    check(cudaMemcpyAsync(reference.data(),output,reference.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));
                    check(cudaStreamSynchronize(stream));
                    if(std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float))) {
                        std::fprintf(stderr,"row reuse changed-input graph mismatch\n");std::exit(1);
                    }
                }
                check(cudaGraphExecDestroy(changed_exec));check(cudaGraphDestroy(changed_graph));
            }
            return ms * 1000. / (4 * iterations);
        };
        for (int repeat = 0; repeat < 3; ++repeat) {
            if(l2_first_probe){
                double generic,first;
                if(repeat%2){first=graph_time(18);generic=graph_time(1);}
                else{generic=graph_time(1);first=graph_time(18);}
                std::printf("l2-policy hint=%d control_hint=%d rows=%d K=%d pair=%d repeat=%d generic_us=%.3f policy_us=%.3f bit_exact=true\n",l2_probe_hint,row_scale_prefetch_control?7:0,rows,inner,pair_layout,repeat,generic,first);
                continue;
            }
            if(tiled_scale_probe){
                double generic,tiled;
                if(repeat%2){tiled=graph_time(17);generic=graph_time(1);}
                else{generic=graph_time(1);tiled=graph_time(17);}
                std::printf("tiled-scale rows=%d K=%d pair=%d repeat=%d generic_us=%.3f tiled_us=%.3f bit_exact=true\n",rows,inner,pair_layout,repeat,generic,tiled);
                continue;
            }
            if(scale_broadcast_probe){
                double generic,broadcast;
                if(repeat%2){broadcast=graph_time(16);generic=graph_time(1);}
                else{generic=graph_time(1);broadcast=graph_time(16);}
                std::printf("scale-broadcast rows=%d K=%d pair=%d repeat=%d generic_us=%.3f broadcast_us=%.3f bit_exact=true\n",rows,inner,pair_layout,repeat,generic,broadcast);
                continue;
            }
            if(exact_subwarp_probe){
                double generic,sixteen,eight;
                if(repeat%2){eight=graph_time(15);sixteen=graph_time(14);generic=graph_time(1);}
                else{generic=graph_time(1);sixteen=graph_time(14);eight=graph_time(15);}
                std::printf("exact-subwarp rows=%d K=%d pair=%d repeat=%d generic_us=%.3f sixteen_us=%.3f eight_us=%.3f bit_exact=true\n",rows,inner,pair_layout,repeat,generic,sixteen,eight);
                continue;
            }
            if (pair_only_probe) {
                if (!pair_layout) continue;
                double generic,pair_only;
                if(repeat%2){pair_only=graph_time(13);generic=graph_time(1);}
                else{generic=graph_time(1);pair_only=graph_time(13);}
                std::printf("pair-only rows=%d K=%d repeat=%d generic_us=%.3f pair_only_us=%.3f bit_exact=true\n",rows,inner,repeat,generic,pair_only);
                continue;
            }
            const bool supports_fixed = inner == 640 || inner == 2560;
            double fixed = supports_fixed && repeat % 2 ? graph_time(6) : 0;
            double row2=repeat%2?graph_time(7):0,row4=repeat%2?graph_time(8):0;
            double warp_packet=repeat%2?graph_time(9):0;
            double aligned_packets=repeat%2?graph_time(10):0;
            double direct_scale=repeat%2?graph_time(11):0;
            double route_major=repeat%2?graph_time(12):0;
            double control, weights, inputs, cached, cg, cs, direct;
            if (repeat % 2) { direct=graph_time(5); cs=graph_time(4); cg=graph_time(3); cached = graph_time(2); inputs = graph_time(1); weights = graph_time(0); control = graph_time(-1); }
            else { control = graph_time(-1); weights = graph_time(0); inputs = graph_time(1); cached = graph_time(2); cg=graph_time(3); cs=graph_time(4); direct=graph_time(5); }
            if (supports_fixed && !(repeat % 2)) fixed = graph_time(6);
            if(!(repeat%2)){row2=graph_time(7);row4=graph_time(8);}
            if(!(repeat%2))warp_packet=graph_time(9);
            if(!(repeat%2))aligned_packets=graph_time(10);
            if(!(repeat%2))direct_scale=graph_time(11);
            if(!(repeat%2))route_major=graph_time(12);
            std::printf("pipeline_route_major rows=%d K=%d pair=%d repeat=%d original_us=%.3f route_major_us=%.3f\n",
                rows,inner,pair_layout,repeat,inputs,route_major);
            std::printf("pipeline_direct_scale rows=%d K=%d pair=%d repeat=%d original_us=%.3f direct_us=%.3f\n",
                rows,inner,pair_layout,repeat,inputs,direct_scale);
            std::printf("pipeline_aligned_packets rows=%d K=%d pair=%d repeat=%d original_us=%.3f aligned_us=%.3f\n",
                rows,inner,pair_layout,repeat,inputs,aligned_packets);
            std::printf("pipeline_warp_packet rows=%d K=%d pair=%d repeat=%d original_us=%.3f packet_us=%.3f bit_exact_graph=true\n",
                rows,inner,pair_layout,repeat,inputs,warp_packet);
            std::printf("pipeline_row_reuse rows=%d K=%d pair=%d repeat=%d original_us=%.3f row2_us=%.3f row4_us=%.3f bit_exact_graph=true\n",
                rows,inner,pair_layout,repeat,inputs,row2,row4);
            if (supports_fixed) std::printf("pipeline_fixed_inner rows=%d K=%d pair=%d repeat=%d original_us=%.3f fixed_us=%.3f bit_exact=yes\n",
                rows,inner,pair_layout,repeat,inputs,fixed);
            std::printf("pipeline_simt rows=%d K=%d pair=%d sets=%d repeat=%d baseline_us=%.3f weights_us=%.3f inputs_us=%.3f cached_us=%.3f bit_exact=yes\n",
                rows, inner, pair_layout, sets, repeat, control, weights, inputs, cached);
            std::printf("pipeline_load_hint rows=%d K=%d pair=%d repeat=%d cg_us=%.3f cs_us=%.3f bit_exact=yes\n", rows,inner,pair_layout,repeat,cg,cs);
            std::printf("pipeline_direct_fp4 rows=%d K=%d pair=%d repeat=%d original_us=%.3f direct_us=%.3f bit_exact=yes\n", rows,inner,pair_layout,repeat,inputs,direct);
        }
        check(cudaStreamDestroy(stream));
        if(tiled_jobs)check(cudaFree(tiled_jobs));
        if(tiled_scales)check(cudaFree(tiled_scales));
        if (pinned_pool) {
            check(cudaFreeHost(device_weights)); check(cudaFreeHost(device_native_weights));
        } else {
            check(cudaFree(device_weights)); check(cudaFree(device_native_weights));
        }
        check(cudaFree(input)); check(cudaFree(qinput)); check(cudaFree(qscales));
        check(cudaFree(output)); check(cudaFree(partial));
        return;
    }
    const bool width_probe = std::getenv("MOLLM_NVFP4_N32_GROUPED_BENCH") != nullptr;
    const bool compact_probe = width_probe && std::getenv("MOLLM_NVFP4_COMPACT_SCOPE_PROBE");
    const bool async_probe=width_probe && std::getenv("MOLLM_NVFP4_N32_ASYNC_WEIGHTS_PROBE");
    const bool guarded_output=async_probe || prepared_timing;
    if(async_probe && compact_probe){std::fprintf(stderr,"exclusive probe modes\n");std::exit(2);}
    const bool long_width_runs = width_probe && std::getenv("MOLLM_NVFP4_GROUPED_LONG_RUNS");
    if (benchmark && (std::getenv("MOLLM_NVFP4_GROUPED_BENCH") || paired_simt || width_probe)) {
        const bool ragged_pairs = paired_simt && std::getenv("MOLLM_NVFP4_PAIR_RAGGED_BENCH");
        for (int tokens : replay_counts ? std::vector<int>{1} : ragged_pairs ? std::vector<int>{0} : long_width_runs ? std::vector<int>{64,128,256,512,779} : width_probe ? std::vector<int>{1,4,8,16,24,32} : std::vector<int>{1, 2, 4, 8}) {
            std::vector<int> expert_for_job;
            constexpr int ragged_sizes[] = {1,1,1,1,1,1,2,2,3,5};
            for (int expert = 0; expert < experts; ++expert)
                for (int token = 0; token < (replay_counts ? (*replay_counts)[expert] : ragged_pairs ? ragged_sizes[expert % 10] : tokens); ++token)
                    expert_for_job.push_back(expert);
            const int count = static_cast<int>(expert_for_job.size());
            auto* grouped_input = device_allocate<float>(static_cast<size_t>(count) * inner);
            const size_t output_count=static_cast<size_t>(count)*rows;
            auto* grouped_output = device_allocate<float>(output_count+(guarded_output?8:0));
            const uint32_t output_guards[8]={0x12345678,0x76543210,0x55aa55aa,0xaa55aa55,
                0x11223344,0x44332211,0x5a5a5a5a,0xa5a5a5a5};
            if(guarded_output)check(cudaMemcpy(grouped_output+output_count,output_guards,
                sizeof(output_guards),cudaMemcpyHostToDevice));
            auto* grouped_jobs = device_allocate<mollm_cuda::Nvfp4Projection>(static_cast<size_t>(sets) * count);
            auto* batches = device_allocate<mollm_cuda::Nvfp4MmaBatch>(count + 1);
            auto* compact_plan = compact_probe ? device_allocate<mollm_cuda::Nvfp4MmaBatch>(count + 1) : nullptr;
            const bool ranked_pairs=paired_simt && std::getenv("MOLLM_NVFP4_RANK_PAIR_BENCH");
            auto* ranked_jobs=ranked_pairs?device_allocate<mollm_cuda::Nvfp4Projection>(2*count):nullptr;
            const bool prequantize = width_probe || std::getenv("MOLLM_NVFP4_GROUPED_PREQUANT") != nullptr;
            const int minimum = std::getenv("MOLLM_NVFP4_GROUPED_ADAPTIVE") ? (rows == 1280 ? 4 : 8) : 1;
            auto* prepared_input = prequantize ? device_allocate<uint8_t>(static_cast<size_t>(count) * inner) : nullptr;
            auto* prepared_scales = prequantize ? device_allocate<uint8_t>(static_cast<size_t>(count) * inner / 8) : nullptr;
            std::vector<float> values(static_cast<size_t>(count) * inner), scalar(output_count+(guarded_output?8:0)), grouped(scalar.size());
            std::vector<mollm_cuda::Nvfp4Projection> descriptors(static_cast<size_t>(sets) * count);
            for (int job = 0; job < count; ++job)
                for (int k = 0; k < inner; ++k)
                    values[static_cast<size_t>(job) * inner + k] = original[k] * (1 + (job % (ragged_pairs ? 8 : tokens)) * .125f);
            for (int set = 0; set < sets; ++set)
                for (int job = 0; job < count; ++job) {
                    auto& descriptor = descriptors[set * count + job];
                    descriptor = (paired_simt || width_probe ? jobs : native_jobs)[set * experts + expert_for_job[job]];
                    descriptor.input = grouped_input + static_cast<size_t>(job) * inner;
                    descriptor.output = grouped_output + static_cast<size_t>(job) * rows;
                    descriptor.input_global = quantized.global * (1 + expert_for_job[job] * .025f);
                    descriptor.input_global_second = descriptor.input_global * 1.75f;
                    descriptor.output_scale = (job + 1) / static_cast<float>(count);
                }
            check(cudaMemcpy(grouped_input, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(grouped_jobs, descriptors.data(), descriptors.size() * sizeof(descriptors[0]), cudaMemcpyHostToDevice));
            const int split = rows == 1280 ? 640 : 0;
            auto scalar_launch = [&](int i) {
                if (paired_simt) {
                    mollm_cuda::launch_nvfp4_projections(grouped_jobs + (i % sets) * count, count, rows, inner);
                    check(cudaGetLastError());
                    return;
                }
                check(mollm_cuda::launch_nvfp4_mma_projections(grouped_jobs + (i % sets) * count, count, rows, inner, split,
                    nullptr,nullptr,1,width_probe));
            };
            auto grouped_launch = [&](int i) {
                if (paired_simt) {
                    if (ranked_pairs) {
                        const auto* source=grouped_jobs+(i%sets)*count;
                        mollm_cuda::launch_group_nvfp4_jobs(source,source,ranked_jobs,ranked_jobs+count,count,nullptr,batches);
                        check(mollm_cuda::launch_nvfp4_ranked_pairs(ranked_jobs,count,batches,rows,inner));
                    } else if (std::getenv("MOLLM_NVFP4_INLINE_PAIR_BENCH"))
                        check(mollm_cuda::launch_nvfp4_inline_pairs(grouped_jobs+(i%sets)*count,count,rows,inner));
                    else if (std::getenv("MOLLM_NVFP4_PAIR_PLAN_BENCH"))
                        check(mollm_cuda::launch_nvfp4_grouped_pairs(grouped_jobs + (i % sets) * count, count, batches, rows, inner));
                    else
                        check(mollm_cuda::launch_nvfp4_paired_projections(grouped_jobs + (i % sets) * count, count, rows, inner));
                    return;
                }
                check(mollm_cuda::launch_nvfp4_mma_grouped(grouped_jobs + (i % sets) * count, count, batches, rows, inner, split,
                    nullptr, prepared_input, prepared_scales, minimum,width_probe));
            };
            for (int set = 0; set < sets; ++set) {
                scalar_launch(set);
                check(cudaMemcpy(scalar.data(), grouped_output, scalar.size() * sizeof(float), cudaMemcpyDeviceToHost));
                grouped_launch(set);
                check(cudaMemcpy(grouped.data(), grouped_output, grouped.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (std::memcmp(scalar.data(), grouped.data(), scalar.size() * sizeof(float))) {
                    std::fprintf(stderr, "grouped real-shape mismatch tokens=%d\n", tokens); std::exit(1);
                }
            }
            if (width_probe) {
                auto launch_width = [&](int width, int i, bool compact = false, bool async = false, bool prepared = false) {
                    // Match the runtime hint; overflow handling remains the kernel's responsibility.
                    const int hint = compact ? std::min(count, (count + width - 1) / width +
                        (experts * (width - 1) + width - 1) / width) : 0;
                    check(mollm_cuda::launch_nvfp4_mma_grouped(grouped_jobs + (i % sets) * count,count,batches,
                        rows,inner,split,nullptr,prepared_input,prepared_scales,1,true,false,
                        compact ? compact_plan : nullptr,hint,width,false,prepared,async));
                };
                const std::vector<int> widths = async_probe ? std::vector<int>{32} :
                    compact_probe ? std::vector<int>{rows == 1280 ? 32 : 8} : std::vector<int>{8,16,32};
                for (int set=0;set<sets;++set) {
                    scalar_launch(set);
                    check(cudaMemcpy(scalar.data(),grouped_output,scalar.size()*sizeof(float),cudaMemcpyDeviceToHost));
                    for (int width : widths) for (int compact=0;compact <= int(compact_probe);++compact) {
                        launch_width(width,set,compact != 0);
                        check(cudaMemcpy(grouped.data(),grouped_output,grouped.size()*sizeof(float),cudaMemcpyDeviceToHost));
                        if (std::memcmp(scalar.data(),grouped.data(),scalar.size()*sizeof(float))) {
                            std::fprintf(stderr,"N32 real-shape mismatch width=%d set=%d\n",width,set);std::exit(1);
                        }
                        if(prepared_timing) {
                            launch_width(width,set,false,false,true);
                            check(cudaMemcpy(grouped.data(),grouped_output,grouped.size()*sizeof(float),cudaMemcpyDeviceToHost));
                            if(std::memcmp(scalar.data(),grouped.data(),scalar.size()*sizeof(float))) {
                                std::fprintf(stderr,"prepared down mismatch width=%d set=%d\n",width,set);std::exit(1);
                            }
                        }
                    }
                    if(async_probe) {
                        launch_width(32,set,false,true);
                        check(cudaMemcpy(grouped.data(),grouped_output,grouped.size()*sizeof(float),cudaMemcpyDeviceToHost));
                        if(std::memcmp(scalar.data(),grouped.data(),scalar.size()*sizeof(float))) {
                            std::fprintf(stderr,"async N32 full-output mismatch set=%d\n",set);std::exit(1);
                        }
                    }
                }
                if(prepared_timing) {
                    // Timed sets share the input descriptors, but differ in weights.
                    // Reject stale quantization reuse if that contract ever changes.
                    for(int set=1;set<sets;++set)for(int j=0;j<count;++j) {
                        const auto& a=descriptors[j];const auto& b=descriptors[set*count+j];
                        if(a.input!=b.input || a.input_global!=b.input_global ||
                            a.input_global_second!=b.input_global_second) {
                            std::fprintf(stderr,"prepared timing needs identical activation metadata across sets\n");std::exit(2);
                        }
                    }
                    int checked=0;
                    std::vector<float> changed(values.size());
                    for(int set=0;set<sets;++set)for(int width:widths) {
                        cudaGraph_t graph=nullptr;cudaGraphExec_t replay=nullptr;
                        check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
                        check(mollm_cuda::launch_nvfp4_mma_grouped(grouped_jobs+set*count,count,batches,
                            rows,inner,0,cudaStreamPerThread,prepared_input,prepared_scales,1,true,false,
                            nullptr,0,width,false,true));
                        check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
                        check(cudaGraphInstantiateWithFlags(&replay,graph,0));
                        for(int variant=0;variant<3;++variant) {
                            for(size_t i=0;i<changed.size();++i)
                                changed[i]=variant==2?0.f:(int((i*13+variant*11+set)%101)-50)/37.f;
                            if(variant==1)for(size_t i=0;i<changed.size();i+=257)changed[i]=-0.f;
                            check(cudaMemcpy(grouped_input,changed.data(),changed.size()*sizeof(float),cudaMemcpyHostToDevice));
                            scalar_launch(set);
                            check(cudaMemcpy(scalar.data(),grouped_output,scalar.size()*sizeof(float),cudaMemcpyDeviceToHost));
                            // Prepare outside capture/timing, as production activation fusion does.
                            launch_width(8,set);
                            check(cudaGraphLaunch(replay,cudaStreamPerThread));
                            check(cudaStreamSynchronize(cudaStreamPerThread));
                            check(cudaMemcpy(grouped.data(),grouped_output,grouped.size()*sizeof(float),cudaMemcpyDeviceToHost));
                            if(std::memcmp(scalar.data(),grouped.data(),scalar.size()*sizeof(float)) ||
                                std::memcmp(grouped.data()+output_count,output_guards,sizeof(output_guards))) {
                                std::fprintf(stderr,"prepared captured mismatch width=%d set=%d variant=%d\n",width,set,variant);std::exit(1);
                            }
                            ++checked;
                        }
                        check(cudaGraphExecDestroy(replay));check(cudaGraphDestroy(graph));
                    }
                    check(cudaMemcpy(grouped_input,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice));
                    launch_width(8,0);
                    std::printf("prepared_down_capture pair=%d jobs=%d replays=%d widths=8,16,32 minimum=1 full_bytes_guards=true\n",
                        pair_layout,count,checked);
                }
                if(async_probe) {
                    // Actual backend-style captured plan/quantization/kernel
                    // chain, changed inputs, every minimum and rotating set.
                    const auto validate_launch=[&](int set,int minimum,bool async,cudaStream_t stream) {
                        check(mollm_cuda::launch_nvfp4_mma_grouped(grouped_jobs+set*count,count,batches,
                            rows,inner,split,stream,prepared_input,prepared_scales,minimum,true,false,
                            nullptr,0,32,false,false,async));
                    };
                    int checked_replays=0;
                    for(int set=0;set<sets;++set)for(int min=1;min<=8;++min) {
                        cudaGraph_t graph=nullptr;cudaGraphExec_t replay=nullptr;
                        check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
                        validate_launch(set,min,true,cudaStreamPerThread);
                        check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
                        check(cudaGraphInstantiateWithFlags(&replay,graph,0));
                        std::vector<float> changed(values.size());
                        for(int variant=0;variant<3;++variant) {
                            for(size_t i=0;i<changed.size();++i)
                                changed[i]=variant==2?0.f:(int((i*13+variant*11+set)%101)-50)/37.f;
                            if(variant==1)for(size_t i=0;i<changed.size();i+=257)changed[i]=-0.f;
                            check(cudaMemcpy(grouped_input,changed.data(),changed.size()*sizeof(float),cudaMemcpyHostToDevice));
                            validate_launch(set,min,false,cudaStreamPerThread);
                            check(cudaMemcpy(scalar.data(),grouped_output,scalar.size()*sizeof(float),cudaMemcpyDeviceToHost));
                            check(cudaGraphLaunch(replay,cudaStreamPerThread));
                            check(cudaStreamSynchronize(cudaStreamPerThread));
                            check(cudaMemcpy(grouped.data(),grouped_output,grouped.size()*sizeof(float),cudaMemcpyDeviceToHost));
                            if(std::memcmp(scalar.data(),grouped.data(),scalar.size()*sizeof(float)) ||
                               std::memcmp(scalar.data()+output_count,output_guards,sizeof(output_guards)) ||
                               std::memcmp(grouped.data()+output_count,output_guards,sizeof(output_guards))) {
                                std::fprintf(stderr,"async N32 changed-input/guard mismatch set=%d min=%d variant=%d\n",set,min,variant);std::exit(1);
                            }
                            ++checked_replays;
                        }
                        check(cudaGraphExecDestroy(replay));check(cudaGraphDestroy(graph));
                    }
                    check(cudaMemcpy(grouped_input,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice));
                    std::printf("n32_async_capture pair=%d jobs=%d full_bytes_guards=true changed_input_replays=%d minimums=1..8\n",
                        pair_layout,count,checked_replays);
                    for(int repeat=0;repeat<7;++repeat)for(int order=0;order<2;++order) {
                        const bool async=((order+repeat)%2)!=0;
                        const double us=measure([&](int i){launch_width(32,i,false,async);},32);
                        std::printf("n32_async_weights rows=%d K=%d pair=%d jobs=%d sets=%d async=%d repeat=%d us=%.3f bit_exact=true mixed_replay=%d\n",
                            rows,inner,pair_layout,count,sets,int(async),repeat,us,int(replay_counts!=nullptr));
                        std::fflush(stdout);
                    }
                } else if (compact_probe) {
                    const int width = widths.front();
                    for (int repeat=0;repeat<7;++repeat) for (int order=0;order<2;++order) {
                        const bool compact = ((order + repeat) % 2) != 0;
                        const double us=measure([&](int i){launch_width(width,i,compact);},32);
                        std::printf("compact_scope rows=%d K=%d pair=%d jobs=%d sets=%d width=%d compact=%d repeat=%d us=%.3f bit_exact=true mixed_replay=%d\n",
                            rows,inner,pair_layout,count,sets,width,int(compact),repeat,us,int(replay_counts!=nullptr));
                        std::fflush(stdout);
                    }
                } else
                for (int repeat=0;repeat<7;++repeat)
                    for (int width : repeat%2 ? std::vector<int>{32,16,8} : std::vector<int>{8,16,32}) {
                        const double us=measure([&](int i){launch_width(width,i,false,false,prepared_timing);},(long_width_runs || replay_counts)?32:256);
                        std::printf("grouped_width pool=%s rows=%d K=%d pair=%d tokens_per_expert=%d jobs=%d sets=%d width=%d repeat=%d us=%.3f bit_exact=true mixed_replay=%d inputs_prepared=%d\n",
                            device_pool?"cuda":"host",rows,inner,pair_layout,replay_counts?0:tokens,count,sets,width,repeat,us,int(replay_counts!=nullptr),int(prepared_timing));
                        std::fflush(stdout);
                    }
            } else for (int repeat = 0; repeat < 3; ++repeat) {
                double candidate,control;
                if (paired_simt && (ranked_pairs || std::getenv("MOLLM_NVFP4_INLINE_PAIR_BENCH"))) {
                    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
                    auto time = [&](bool paired) {
                        cudaGraph_t graph;cudaGraphExec_t executable;
                        const int iterations=std::max(sets,256);
                        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                        for(int i=0;i<iterations;++i) {
                            const auto* selected=grouped_jobs+(i%sets)*count;
                            if(ranked_pairs) {
                                mollm_cuda::launch_group_nvfp4_jobs(selected,selected,ranked_jobs,ranked_jobs+count,count,stream,paired?batches:nullptr);
                                if(paired)check(mollm_cuda::launch_nvfp4_ranked_pairs(ranked_jobs,count,batches,rows,inner,stream));
                                else check(mollm_cuda::launch_nvfp4_pipelined_projections(ranked_jobs,count,rows,inner,true,stream));
                            } else if(paired)check(mollm_cuda::launch_nvfp4_inline_pairs(selected,count,rows,inner,stream));
                            else check(mollm_cuda::launch_nvfp4_pipelined_projections(selected,count,rows,inner,true,stream));
                        }
                        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
                        for(int warm=0;warm<2;++warm)check(cudaGraphLaunch(executable,stream));
                        cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
                        check(cudaEventRecord(start,stream));for(int replay=0;replay<4;++replay)check(cudaGraphLaunch(executable,stream));
                        check(cudaEventRecord(stop,stream));check(cudaEventSynchronize(stop));float ms=0;check(cudaEventElapsedTime(&ms,start,stop));
                        check(cudaMemcpy(grouped.data(),grouped_output,grouped.size()*sizeof(float),cudaMemcpyDeviceToHost));
                        scalar_launch(iterations-1);check(cudaMemcpy(scalar.data(),grouped_output,scalar.size()*sizeof(float),cudaMemcpyDeviceToHost));
                        if(std::memcmp(grouped.data(),scalar.data(),scalar.size()*sizeof(float))) {std::fprintf(stderr,"inline pairs graph mismatch\n");std::exit(1);}
                        check(cudaEventDestroy(start));check(cudaEventDestroy(stop));check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
                        return ms*1000./(4*iterations);
                    };
                    if(repeat%2){candidate=time(true);control=time(false);}else{control=time(false);candidate=time(true);}
                    check(cudaStreamDestroy(stream));
                } else {candidate=measure(grouped_launch,256);control=measure(scalar_launch,256);}
                std::printf("%s pool=%s rows=%d K=%d tokens_per_expert=%d jobs=%d ragged=%s plan=%s repeat=%d us=%.3f control_us=%.3f bit_exact=true\n",
                    paired_simt ? "paired_simt" : "grouped_real_shape",
                    device_pool ? "cuda" : "host", rows, inner, tokens, count,
                    ragged_pairs ? "true" : "false",
                    std::getenv("MOLLM_NVFP4_PAIR_PLAN_BENCH") ? "true" : "false", repeat, candidate, control);
                std::fflush(stdout);
            }
            check(cudaFree(grouped_input)); check(cudaFree(grouped_output)); check(cudaFree(grouped_jobs)); check(cudaFree(batches));
            check(cudaFree(ranked_jobs));
            check(cudaFree(compact_plan));
            check(cudaFree(prepared_input)); check(cudaFree(prepared_scales));
        }
    } else {
        if (benchmark) {
            mollm_cuda::launch_nvfp4_projections(jobs.data(), experts, rows, inner);
            check(cudaGetLastError());
            check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double error = 0, reference_maximum = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) { std::fprintf(stderr, "nonfinite SIMT output\n"); std::exit(1); }
                error = std::max(error, std::abs(actual[i] - original_expected[i]));
                reference_maximum = std::max(reference_maximum, std::abs(original_expected[i]));
            }
            if (error / std::max(1., reference_maximum) > 2e-5) {
                std::fprintf(stderr, "SIMT allocation-path output mismatch\n"); std::exit(1);
            }
        }
        const double simt = benchmark ? measure([&](int i) {
            mollm_cuda::launch_nvfp4_projections(jobs.data() + (i % sets) * experts, experts, rows, inner);
        }) : 0;
        if (benchmark) std::printf("SIMT pool=%s rows=%d K=%d experts=%d sets=%d us=%.3f\n",
            pinned_pool ? "pinned" : managed_pool ? "managed" : device_pool ? "cuda" : "host", rows, inner, experts, sets, simt);
        for (bool packed : {false, true}) for (int parts : {1, 2, 4, 8}) for (int warps : {1, 2, 4, 8}) {
            auto launch = [&](int i) {
                const auto* selected = (packed ? native_jobs : jobs).data() + (i % sets) * experts;
                const dim3 grid((rows + warps * 16 - 1) / (warps * 16), experts, parts);
                if (packed)
                    nvfp4_mma_probe<true><<<grid, warps * 32>>>(
                        selected, qinput, qscales, quantized.global, rows, inner, partial);
                else
                    nvfp4_mma_probe<false><<<grid, warps * 32>>>(
                        selected, qinput, qscales, quantized.global, rows, inner, partial);
                if (parts > 1)
                    combine_mma_parts<<<(rows * experts + 255) / 256, 256>>>(
                        selected, partial, quantized.global, rows, experts, parts);
            };
            launch(0);
            check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) { std::fprintf(stderr, "nonfinite split MMA\n"); std::exit(1); }
                error = std::max(error, std::abs(actual[i] - expected[i]));
            }
            if (error / std::max(1., maximum) > 2e-5) {
                std::fprintf(stderr, "split MMA mismatch parts=%d warps=%d\n", parts, warps); std::exit(1);
            }
            if (!benchmark && packed) {
                const dim3 fused_grid((rows + warps * 16 - 1) / (warps * 16), experts, parts);
                nvfp4_mma_probe<true, 0, true><<<fused_grid, warps * 32>>>(
                    native_jobs.data(), nullptr, nullptr, quantized.global, rows, inner, partial);
                if (parts > 1)
                    combine_mma_parts<<<(rows * experts + 255) / 256, 256>>>(
                        native_jobs.data(), partial, quantized.global, rows, experts, parts);
                check(cudaGetLastError());
                check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < actual.size(); ++i)
                    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i])
                        > 2e-5 * std::max(1., maximum)) {
                        std::fprintf(stderr, "ragged fused MMA mismatch parts=%d warps=%d\n",
                            parts, warps); std::exit(1);
                    }
            }
            if (benchmark) {
                const double us = measure(launch);
                std::printf("native_FP4_MMA pool=%s rows=%d K=%d packed=%d warps=%d parts=%d us=%.3f kernel_speedup=%.3f\n",
                    pinned_pool ? "pinned" : managed_pool ? "managed" : device_pool ? "cuda" : "host", rows, inner, packed, warps, parts, us, simt / us);
                if (packed && warps == 8 && parts == (inner == 2560 ? 8 : 1)) {
                    const double combined = measure([&](int i) { quantize_gpu(i); launch(i); });
                    std::printf("GPU_quantize_plus_native_MMA rows=%d K=%d us=%.3f speedup_vs_SIMT=%.3f\n",
                        rows, inner, combined, simt / combined);
                    auto prefetch_launch = [&](int i) {
                        const auto* selected = native_jobs.data() + (i % sets) * experts;
                        const dim3 grid((rows + warps * 16 - 1) / (warps * 16), experts, parts);
                        nvfp4_mma_probe<true, 1><<<grid, warps * 32>>>(selected,
                            qinput, qscales, quantized.global, rows, inner, partial);
                        if (parts > 1)
                            combine_mma_parts<<<(rows * experts + 255) / 256, 256>>>(
                                selected, partial, quantized.global, rows, experts, parts);
                    };
                    prefetch_launch(0);
                    check(cudaGetLastError());
                    check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < actual.size(); ++i)
                        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i])
                            > 2e-5 * std::max(1., maximum)) {
                            std::fprintf(stderr, "prefetched MMA mismatch\n"); std::exit(1);
                        }
                    const double prefetched = measure([&](int i) { quantize_gpu(i); prefetch_launch(i); });
                    std::printf("GPU_quantize_plus_prefetched_MMA rows=%d K=%d us=%.3f speedup_vs_no_prefetch=%.3f\n",
                        rows, inner, prefetched, combined / prefetched);
                    auto fused_launch = [&](int i) {
                        const auto* selected = native_jobs.data() + (i % sets) * experts;
                        const dim3 grid((rows + warps * 16 - 1) / (warps * 16), experts, parts);
                        nvfp4_mma_probe<true, 0, true><<<grid, warps * 32>>>(selected,
                            nullptr, nullptr, quantized.global, rows, inner, partial);
                        if (parts > 1)
                            combine_mma_parts<<<(rows * experts + 255) / 256, 256>>>(
                                selected, partial, quantized.global, rows, experts, parts);
                    };
                    fused_launch(0);
                    check(cudaGetLastError());
                    check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < actual.size(); ++i)
                        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i])
                            > 2e-5 * std::max(1., maximum)) {
                            std::fprintf(stderr, "fused quantization MMA mismatch\n"); std::exit(1);
                        }
                    const double fused = measure(fused_launch);
                    std::printf("GPU_fused_quantization_MMA rows=%d K=%d us=%.3f speedup_vs_separate=%.3f speedup_vs_SIMT=%.3f\n",
                        rows, inner, fused, combined / fused, simt / fused);
                }
            }
        }
    }
    if (pinned_pool) {
        check(cudaFreeHost(device_weights)); check(cudaFreeHost(device_native_weights));
    } else {
        check(cudaFree(device_weights)); check(cudaFree(device_native_weights));
    }
    check(cudaFree(input)); check(cudaFree(qinput)); check(cudaFree(qscales)); check(cudaFree(output)); check(cudaFree(partial));
}

// A single full-size synthetic expert payload, never two 64-GiB copies.
// This isolates allocation/address-working-set effects, not model quality.
constexpr size_t large_gate_data = 1280ull * 2560 / 2;
constexpr size_t large_gate_scales = 1280ull * 2560 / 16;
constexpr size_t large_gate_stride = large_gate_data + large_gate_scales + 1280 * sizeof(float);
constexpr size_t large_down_data = 2560ull * 640 / 2;
constexpr size_t large_down_scales = 2560ull * 640 / 16;
constexpr size_t large_down_stride = large_down_data + large_down_scales + 2560 * sizeof(float);
constexpr size_t large_pair_stride = large_gate_stride + large_down_stride;

__global__ void initialize_large_pool(uint32_t* data, size_t words, bool varied = false) {
    for (size_t word = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         word < words; word += static_cast<size_t>(gridDim.x) * blockDim.x) {
        size_t relative = word * 4 % large_pair_stride;
        size_t weights = large_gate_data, scales = large_gate_scales;
        if (relative >= large_gate_stride) {
            relative -= large_gate_stride; weights = large_down_data; scales = large_down_scales;
        }
        if (!varied) {
            data[word] = relative < weights ? 0x63636363u :
                relative < weights + scales ? 0x38383838u : 0x3c800000u;
        } else if (relative < weights) {
            uint32_t seed = static_cast<uint32_t>(word) ^ static_cast<uint32_t>(word >> 32);
            seed ^= seed >> 16; seed *= 0x7feb352du; seed ^= seed >> 15;
            seed *= 0x846ca68bu; seed ^= seed >> 16;
            data[word] = seed;
        } else if (relative < weights + scales) {
            constexpr uint8_t codes[] = {1, 7, 8, 48, 56, 64, 120, 126};
            uint32_t packed = 0;
            for (int byte = 0; byte < 4; ++byte)
                packed |= uint32_t(codes[(word * 3 + byte) % 8]) << (byte * 8);
            data[word] = packed;
        } else {
            data[word] = 0x3b800000u + (static_cast<uint32_t>(word % 8) << 20);
        }
    }
}

void run_large_pool() {
    constexpr int layers = 48, experts = 512, selected = 10;
    constexpr size_t entries = layers * experts;
    constexpr size_t bytes = entries * large_pair_stride;
    const char* option = std::getenv("MOLLM_NVFP4_PROBE_DEVICE_POOL");
    const bool device = option && std::strcmp(option, "1") == 0;
    const bool pipeline_probe = std::getenv("MOLLM_NVFP4_PIPELINE_SIMT_BENCH") != nullptr;
    if (option && std::strcmp(option, "0") != 0 && !device) {
        std::fprintf(stderr, "large pool accepts only host (0) or device (1)\n"); std::exit(1);
    }
    size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (free_bytes < bytes + (1ull << 30)) {
        std::fprintf(stderr, "insufficient free memory for 63.63-GiB diagnostic\n"); std::exit(1);
    }
    std::printf("large_pool=%s bytes=%zu entries=%zu; single expert payload, not model inference\n",
        device ? "cuda" : "host", bytes, entries);
    std::fflush(stdout);
    std::unique_ptr<Arena> host;
    uint8_t* pool = nullptr;
    if (device) pool = device_allocate<uint8_t>(bytes);
    else { host = std::make_unique<Arena>(bytes); pool = host->data; }
    initialize_large_pool<<<65535, 256>>>(reinterpret_cast<uint32_t*>(pool), bytes / 4, pipeline_probe);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::vector<float> x(2560, .125f);
    if (pipeline_probe) for (size_t i = 0; i < x.size(); ++i) x[i] = (int(i % 19) - 9) / 16.f;
    QuantizedActivation quantized(x);
    auto* input = device_allocate<float>(2560);
    auto* packed_input = device_allocate<uint8_t>(1280);
    auto* input_scales = device_allocate<uint8_t>(160);
    auto* output = device_allocate<float>(selected * 2560);
    auto* partial = device_allocate<float>(selected * 2560 * 8);
    check(cudaMemcpy(input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(packed_input, quantized.data.data(), 1280, cudaMemcpyHostToDevice));
    check(cudaMemcpy(input_scales, quantized.scales.data(), 160, cudaMemcpyHostToDevice));
    std::vector<mollm_cuda::Nvfp4Projection> tables(entries * 2);
    for (size_t entry = 0; entry < entries; ++entry) {
        auto* gate = pool + entry * large_pair_stride;
        auto* down = gate + large_gate_stride;
        tables[entry] = {gate, gate + large_gate_data,
            reinterpret_cast<float*>(gate + large_gate_data + large_gate_scales),
            input, output + (entry % selected) * 1280, true};
        tables[entries + entry] = {down, down + large_down_data,
            reinterpret_cast<float*>(down + large_down_data + large_down_scales),
            input, output + (entry % selected) * 2560, true};
    }
    auto* jobs = device_allocate<mollm_cuda::Nvfp4Projection>(tables.size());
    check(cudaMemcpy(jobs, tables.data(), tables.size() * sizeof(tables[0]), cudaMemcpyHostToDevice));
    for (bool down : {false, true}) {
        const int rows = down ? 2560 : 1280, inner = down ? 640 : 2560;
        const int parts = down ? 1 : 8, warps = 8;
        auto choose = [&](int i) {
            const int layer = i % layers;
            const int group = (static_cast<unsigned>(i / layers) * 15427u + layer * 37u) % 51;
            return jobs + (down ? entries : 0) + layer * experts + group * selected;
        };
        auto simt = [&](int i) {
            mollm_cuda::launch_nvfp4_projections(choose(i), selected, rows, inner);
        };
        if (pipeline_probe) {
            std::vector<float> reference(rows * selected), actual(reference.size());
            for (int i : {0, 47, 48, 123, 2447, 5119}) {
                simt(i);
                check(cudaMemcpy(reference.data(), output, reference.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (int variant : {1, 2, 3, 4}) {
                    if (variant <= 2) check(mollm_cuda::launch_nvfp4_pipelined_projections(
                        choose(i), selected, rows, inner, variant == 2));
                    else check(mollm_cuda::launch_nvfp4_deep_pipelined_projections(
                        choose(i), selected, rows, inner, variant == 3 ? 2 : 4));
                    check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    if (std::memcmp(reference.data(), actual.data(), actual.size() * sizeof(float))) {
                        std::fprintf(stderr, "large pipeline bit-exact mismatch i=%d variant=%d\n", i, variant);
                        std::exit(1);
                    }
                }
            }
            for (int repeat = 0; repeat < 3; ++repeat) {
                double times[5]{};
                for (int variant : repeat % 2 ? std::vector<int>{4, 3, 2, 1, 0} : std::vector<int>{0, 1, 2, 3, 4}) {
                    times[variant] = measure_graph([&](int i, cudaStream_t stream) {
                        if (!variant) mollm_cuda::launch_nvfp4_projections(choose(i), selected, rows, inner, stream);
                        else if (variant <= 2) check(mollm_cuda::launch_nvfp4_pipelined_projections(
                            choose(i), selected, rows, inner, variant == 2, stream));
                        else check(mollm_cuda::launch_nvfp4_deep_pipelined_projections(
                            choose(i), selected, rows, inner, variant == 3 ? 2 : 4, stream));
                    }, 5120);
                }
                std::printf("large_pipeline=%s rows=%d K=%d repeat=%d original_us=%.3f weights_us=%.3f inputs_us=%.3f depth2_us=%.3f depth4_us=%.3f bit_exact=true varied_data=true\n",
                    device ? "cuda" : "host", rows, inner, repeat, times[0], times[1], times[2], times[3], times[4]);
                std::fflush(stdout);
            }
            continue;
        }
        auto mma = [&](int i) {
            nvfp4_mma_probe<false><<<dim3((rows + warps * 16 - 1) / (warps * 16), selected, parts), warps * 32>>>(
                choose(i), packed_input, input_scales, quantized.global, rows, inner, partial);
            if (parts > 1) combine_mma_parts<<<(rows * selected + 255) / 256, 256>>>(
                choose(i), partial, quantized.global, rows, selected, parts);
        };
        auto fused = [&](int i) {
            // Constant synthetic weights/scales represent the same matrix in
            // both layouts. This permits native-packed addressing without a
            // second payload; it does not validate checkpoint repacking.
            nvfp4_mma_probe<true, 0, true><<<dim3((rows + warps * 16 - 1) /
                (warps * 16), selected, parts), warps * 32>>>(
                choose(i), nullptr, nullptr, quantized.global, rows, inner, partial);
            if (parts > 1) combine_mma_parts<<<(rows * selected + 255) / 256, 256>>>(
                choose(i), partial, quantized.global, rows, selected, parts);
        };
        auto packed_mma = [&](int i) {
            nvfp4_mma_probe<true><<<dim3((rows + warps * 16 - 1) /
                (warps * 16), selected, parts), warps * 32>>>(
                choose(i), packed_input, input_scales, quantized.global, rows, inner, partial);
            if (parts > 1) combine_mma_parts<<<(rows * selected + 255) / 256, 256>>>(
                choose(i), partial, quantized.global, rows, selected, parts);
        };
        for (int implementation : {0, 1, 2, 3}) {
            if (implementation == 3) packed_mma(123);
            else if (implementation == 2) fused(123);
            else if (implementation == 1) mma(123); else simt(123);
            check(cudaGetLastError());
            std::vector<float> actual(rows * selected);
            check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            const float expected = inner * .5f * (1.5f + 4.f) * .125f / 64.f;
            for (float value : actual) if (!std::isfinite(value) ||
                std::abs(value - expected) > 2e-5f * std::max(1.f, expected)) {
                std::fprintf(stderr, "large-pool output mismatch %g != %g\n", value, expected); std::exit(1);
            }
        }
        for (int repeat = 0; repeat < 3; ++repeat) {
            const double scalar = measure(simt, 5120), native = measure(mma, 5120),
                fused_time = measure(fused, 5120), packed_time = measure(packed_mma, 5120);
            std::printf("large_pool=%s rows=%d K=%d repeat=%d SIMT_us=%.3f native_us=%.3f fused_us=%.3f fused_speedup=%.3f packed_native_us=%.3f\n",
                device ? "cuda" : "host", rows, inner, repeat, scalar, native,
                fused_time, scalar / fused_time, packed_time);
            std::fflush(stdout);
        }
        const char* prefetch_option = std::getenv("MOLLM_NVFP4_PROBE_PREFETCH");
        if (prefetch_option && std::strcmp(prefetch_option, "1") == 0) {
            simt(123);
            std::vector<float> reference(rows * selected), actual(rows * selected);
            check(cudaMemcpy(reference.data(), output, reference.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int ahead : {1, 2, 4}) {
                auto prefetched = [&](int i) {
                    mollm_cuda::launch_nvfp4_prefetch_projections(choose(i), selected, rows, inner, ahead);
                };
                prefetched(123);
                check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (std::memcmp(reference.data(), actual.data(), actual.size() * sizeof(float))) {
                    std::fprintf(stderr, "large-pool prefetch output mismatch\n"); std::exit(1);
                }
                for (int repeat = 0; repeat < 3; ++repeat) {
                    const double candidate = measure(prefetched, 5120), control = measure(simt, 5120);
                    std::printf("large_prefetch=%s rows=%d K=%d ahead=%d repeat=%d us=%.3f control_us=%.3f bit_exact=true\n",
                        device ? "cuda" : "host", rows, inner, ahead, repeat, candidate, control);
                    std::fflush(stdout);
                }
            }
        }
    }
    check(cudaFree(jobs)); check(cudaFree(input)); check(cudaFree(packed_input));
    check(cudaFree(input_scales)); check(cudaFree(output)); check(cudaFree(partial));
    if (device) check(cudaFree(pool));
}
} // namespace

int main() {
    const char* prepared_option=std::getenv("MOLLM_NVFP4_GROUPED_PREPARED_TIMING");
    if(prepared_option && (std::strcmp(prepared_option,"1") ||
        !std::getenv("MOLLM_NVFP4_N32_GROUPED_BENCH") ||
        std::getenv("MOLLM_NVFP4_COMPACT_SCOPE_PROBE") ||
        std::getenv("MOLLM_NVFP4_N32_ASYNC_WEIGHTS_PROBE"))) {
        std::fprintf(stderr,"prepared timing requires exact1 and exclusive grouped-width mode\n");return 2;
    }
    for (int hint : {-1, 1, 2, 4, 7}) {
        if (mollm_cuda::launch_nvfp4_gate_activation(nullptr, 0, 4, 16,
                nullptr, nullptr, nullptr, 4, hint) != cudaErrorInvalidValue) {
            std::fprintf(stderr, "invalid fusion hint accepted: %d\n", hint);
            return 1;
        }
    }
    for (int hint : {0, 3})
        check(mollm_cuda::launch_nvfp4_gate_activation(nullptr, 0, 4, 16,
            nullptr, nullptr, nullptr, 4, hint));
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 12) {
        std::printf("SKIP: this experimental MMA probe targets SM120a/SM121a\n"); return 0;
    }
    int coherent = 0, tables = 0;
    check(cudaDeviceGetAttribute(&coherent, cudaDevAttrPageableMemoryAccess, 0));
    check(cudaDeviceGetAttribute(&tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
    if (!coherent || !tables) {
        std::printf("SKIP: coherent host weights required by this resident probe\n"); return 0;
    }
    std::printf("Device: %s; independent native NVFP4 MMA probe, not model inference\n", properties.name);
    if (std::getenv("MOLLM_NVFP4_N32_GROUPED_BENCH")) {
        std::vector<int> replay_counts;
        const char* replay_node_option=std::getenv("MOLLM_NVFP4_REPLAY_COUNTS_NODE");
        std::string replay_node;
        if(replay_node_option) {
            if(!*replay_node_option || !std::getenv("MOLLM_NVFP4_REPLAY_COUNTS_FILE"))return 2;
            uint64_t value=0;
            for(const char* p=replay_node_option;*p;++p) {
                if(*p<'0'||*p>'9')return 2;
                value=value*10+(*p-'0');
                if(value>UINT32_MAX)return 2;
            }
            if(!value)return 2;
            replay_node=std::to_string(value);
        }
        const char* replay_rows = std::getenv("MOLLM_NVFP4_REPLAY_COUNTS_ROWS");
        if (replay_rows) {
            if (!*replay_rows || !std::getenv("MOLLM_NVFP4_REPLAY_COUNTS_FILE")) return 2;
            for (const char* p=replay_rows;*p;++p) if(*p<'0'||*p>'9') return 2;
            char* end=nullptr;const long value=std::strtol(replay_rows,&end,10);
            if(!end || *end || value<=0 || value>65535) return 2;
        }
        if (const char* path = std::getenv("MOLLM_NVFP4_REPLAY_COUNTS_FILE")) {
            std::ifstream input(path);
            std::string line;
            bool found = false;
            while (std::getline(input,line)) {
                if (line.rfind("nvfp4_expert_counts ",0)!=0) continue;
                if(replay_node_option && line.find(" node="+replay_node+" ")==std::string::npos)continue;
                if (replay_rows && line.find(std::string(" rows=")+replay_rows+" ")==std::string::npos) continue;
                const auto position=line.find(" counts=");
                if(position==std::string::npos || line.find(" unmatched=0 ")==std::string::npos) break;
                const char* cursor=line.c_str()+position+8;
                int total=0;
                while(*cursor) {
                    char* end=nullptr;
                    const long count=std::strtol(cursor,&end,10);
                    if(end==cursor || count<0 || count>65535 || (*end && *end!=',')) break;
                    replay_counts.push_back(static_cast<int>(count));
                    total+=static_cast<int>(count);
                    if(total>65535 || replay_counts.size()>512) break;
                    if(!*end) {found=replay_counts.size()==512 && total>0;break;}
                    cursor=end+1;
                }
                break;
            }
            if(!found) {std::fprintf(stderr,"missing/invalid matching expert-count record: %s\n",path);return 2;}
            std::printf("mixed routing replay: first matching record,node=%s,rows=%s,512 identities; synthetic weights/activations, not model inference\n",replay_node_option?replay_node.c_str():"any",replay_rows?replay_rows:"any");
        }
        for (bool pair : {true,false}) {
            run(1280,2560,pair,true,0,replay_counts.empty()?nullptr:&replay_counts);
            run(2560,640,pair,true,0,replay_counts.empty()?nullptr:&replay_counts);
        }
        return 0;
    }
    if (std::getenv("MOLLM_NVFP4_BF16_MMA_PROBE") || std::getenv("MOLLM_NVFP4_SIMT_SUBWARP") || std::getenv("MOLLM_NVFP4_INTEGER_SCALE") || std::getenv("MOLLM_NVFP4_CTA_PROBE") || std::getenv("MOLLM_NVFP4_READ_PROBE") || std::getenv("MOLLM_NVFP4_INTERLEAVED")) {
        for (int code = 0; code < 256; ++code) if ((code & 127) != 127)
            for (int value = 0; value < 16; ++value) {
                const float product = fp4_value(value) * fp8_value(code);
                if (__bfloat162float(__float2bfloat16_rn(product)) != product) {
                    std::fprintf(stderr, "FP4 times finite E4M3 not exactly BF16\n"); return 1;
                }
            }
        std::puts("All 4064 finite FP4/E4M3 products are exactly representable in BF16; row globals remain FP32.");
        for (bool pair : {false, true}) {
            for (int rows : {3, 7, 19}) for (int inner : {64, 128, 640, 2560})
                run(rows, inner, pair, false);
            run(1280, 2560, pair, true);
            run(2560, 640, pair, true);
        }
        return 0;
    }
    if (std::getenv("MOLLM_NVFP4_NATIVE_PIPELINE_PROBE")) {
        for (bool pair : {false, true}) {
            for (int rows : {3, 7, 19}) for (int inner : {64, 128, 640, 2560}) run(rows, inner, pair, false);
            run(1280, 2560, pair, true);
            run(2560, 640, pair, true);
        }
        return 0;
    }
    const char* large = std::getenv("MOLLM_NVFP4_PROBE_LARGE_POOL");
    if (large && std::strcmp(large, "1") == 0) { run_large_pool(); return 0; }
    if (std::getenv("MOLLM_NVFP4_PIPELINE_SIMT_BENCH") || std::getenv("MOLLM_NVFP4_GATE_ACTIVATION_BENCH")) {
        for (bool pair : {false, true}) {
            run(1280, 2560, pair, true);
            run(2560, 640, pair, true);
        }
        return 0;
    }
    if (std::getenv("MOLLM_NVFP4_GROUPED_BENCH") || std::getenv("MOLLM_NVFP4_PAIRED_SIMT_BENCH")) {
        run(1280, 2560, true, true);
        run(2560, 640, true, true);
        return 0;
    }
    for (bool pair : {false, true}) for (int rows : {3, 7, 19}) for (int inner : {64, 128, 640, 2560})
        run(rows, inner, pair, false);
    for (bool pair : {false, true}) for (int rows : {16, 48, 128})
        for (int inner : {64, 640, 2560}) run(rows, inner, pair, false, .0013892765f);
    // Exercise calibrated-global scaling, including all-zero activations,
    // ragged output tiles and block scales that saturate. These are synthetic
    // layout checks, not validation of a particular checkpoint's quantizer.
    for (float scale : {0.0013892765f, 0.0031854538f, 0.00001f})
        for (bool pair : {false, true}) for (int rows : {3, 19})
            run(rows, 2560, pair, false, scale);
    if (std::getenv("MOLLM_NVFP4_FRAGMENT_CHECK_ONLY")) {
        std::puts("All requested native/prepared fragment checks passed; timings skipped.");
        return 0;
    }
    for (bool pair : {true, false}) {
        run(1280, 2560, pair, true);
        run(2560, 640, pair, true);
    }
    std::puts("All native FP4 fragment/scale checks passed; model accuracy remains unvalidated.");
}
