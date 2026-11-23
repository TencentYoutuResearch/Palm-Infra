// Independent W4/A8 experiment. No inference selector or weight-format change.
#include "kernels/cuda/matmul.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1); }
}
template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

__device__ uint32_t weight_fragment(const Q4B8G32Block* weights,
    int rows, int groups, int row, int group, int column) {
    if (row >= rows) return 0;
    const auto& block = weights[static_cast<size_t>(row / 8) * groups + group];
    const unsigned packed = *reinterpret_cast<const uint16_t*>(block.q[row % 8] + column / 2);
    uint32_t expanded = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        expanded |= static_cast<uint32_t>((int((packed >> (i * 4)) & 15) - 8) & 255) << (i * 8);
    return expanded;
}

__device__ uint32_t weight_int4_fragment(const Q4B8G32Block* weights,
    int rows, int groups, int row, int group, int column) {
    if (row >= rows) return 0;
    const auto& block = weights[static_cast<size_t>(row / 8) * groups + group];
    return *reinterpret_cast<const uint32_t*>(block.q[row % 8] + column / 2) ^ 0x88888888u;
}

template<int Warps = 4, bool Dual4 = false, bool Prequantized = false, bool QuantizeOnly = false>
__global__ void q4_a8_mma(const float* input, const Q4B8G32Block* weights,
    float* output, int rows, int inner, uint32_t* prepared = nullptr, float* prepared_scales = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ uint32_t quantized_storage[Prequantized ? 1 : (Dual4 ? 320 : 640)];
    __shared__ uint32_t quantized_high_storage[!Prequantized && Dual4 ? 320 : 1];
    __shared__ float scales_storage[Prequantized ? 1 : 80];
    uint32_t* quantized = Prequantized ? prepared : quantized_storage;
    uint32_t* quantized_high = Prequantized ? prepared + inner / 8 : quantized_high_storage;
    float* scales = Prequantized ? prepared_scales : scales_storage;
    if constexpr (!Prequantized) {
    for (int base = 0; base < inner / 4; base += blockDim.x) {
        const int vector = base + threadIdx.x;
        const bool valid = vector < inner / 4;
        const float4 value = valid ? reinterpret_cast<const float4*>(input)[vector] : make_float4(0, 0, 0, 0);
        float maximum = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w)));
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 1, 8));
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 2, 8));
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 4, 8));
        constexpr int maximum_code = Dual4 ? 119 : 127;
        const float scale = __fdiv_rn(maximum, float(maximum_code));
        const float inverse = scale > 0 ? __fdiv_rn(1.f, scale) : 0;
        const auto encode = [&](float x) {
            return max(-maximum_code, min(maximum_code, __float2int_rn(x * inverse)));
        };
        if (valid) {
            if constexpr (Dual4) {
                const int codes[4] = {encode(value.x), encode(value.y), encode(value.z), encode(value.w)};
                uint16_t low = 0, high = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    low |= (codes[i] & 15) << (4 * i);
                    high |= (((codes[i] + 8) >> 4) & 15) << (4 * i);
                }
                reinterpret_cast<uint16_t*>(quantized)[vector] = low;
                reinterpret_cast<uint16_t*>(quantized_high)[vector] = high;
            } else {
                quantized[vector] = (encode(value.x) & 255u) | ((encode(value.y) & 255u) << 8) |
                    ((encode(value.z) & 255u) << 16) | ((encode(value.w) & 255u) << 24);
            }
            if (vector % 8 == 0) scales[vector / 8] = scale;
        }
    }
    __syncthreads();
    }
    if constexpr (QuantizeOnly) {
        for (int i = threadIdx.x; i < inner / 4; i += blockDim.x)
            prepared[i] = Dual4 && i >= inner / 8 ? quantized_high[i - inner / 8] : quantized[i];
        for (int i = threadIdx.x; i < inner / 32; i += blockDim.x) prepared_scales[i] = scales[i];
        return;
    }
    const int lane = threadIdx.x % 32, thread = lane % 4;
    const int base_row = (blockIdx.x * Warps + threadIdx.x / 32) * 16;
    if (base_row >= rows) return;
    const int row = base_row + lane / 4, groups = inner / 32;
    float first = 0, second = 0;
    for (int group = 0; group < groups; ++group) {
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        if constexpr (Dual4) {
            const uint32_t a0 = weight_int4_fragment(weights, rows, groups, row, group, thread * 8);
            const uint32_t a1 = weight_int4_fragment(weights, rows, groups, row + 8, group, thread * 8);
            const uint32_t low = quantized[group * 4 + thread], high = quantized_high[group * 4 + thread];
            int h0 = 0, h1 = 0, h2 = 0, h3 = 0;
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                : "r"(a0), "r"(a1), "r"(low));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                : "+r"(h0), "+r"(h1), "+r"(h2), "+r"(h3)
                : "r"(a0), "r"(a1), "r"(high));
            d0 += h0 * 16;
            d2 += h2 * 16;
        } else {
        const uint32_t a0 = weight_fragment(weights, rows, groups, row, group, thread * 4);
        const uint32_t a1 = weight_fragment(weights, rows, groups, row + 8, group, thread * 4);
        const uint32_t a2 = weight_fragment(weights, rows, groups, row, group, thread * 4 + 16);
        const uint32_t a3 = weight_fragment(weights, rows, groups, row + 8, group, thread * 4 + 16);
        const uint32_t b0 = quantized[group * 8 + thread], b1 = quantized[group * 8 + thread + 4];
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        if (thread == 0) {
            if (row < rows) first = fmaf(float(d0), weights[static_cast<size_t>(row / 8) * groups + group].scales[row % 8] * scales[group], first);
            if (row + 8 < rows) second = fmaf(float(d2), weights[static_cast<size_t>((row + 8) / 8) * groups + group].scales[(row + 8) % 8] * scales[group], second);
        }
    }
    if (thread == 0) {
        if (row < rows) output[row] = first;
        if (row + 8 < rows) output[row + 8] = second;
    }
#else
    asm volatile("trap;");
#endif
}

template<bool Dual4, bool Prequantized = false>
void launch_variant(const float* input, const Q4B8G32Block* weights, float* output,
    int rows, int inner, int warps, uint32_t* prepared = nullptr, float* scales = nullptr) {
    if constexpr (Prequantized)
        q4_a8_mma<4, Dual4, false, true><<<1, 128>>>(input, nullptr, nullptr, 1, inner, prepared, scales);
    if (warps == 2) q4_a8_mma<2, Dual4, Prequantized><<<(rows + 31) / 32, 64>>>(input, weights, output, rows, inner, prepared, scales);
    else if (warps == 8) q4_a8_mma<8, Dual4, Prequantized><<<(rows + 127) / 128, 256>>>(input, weights, output, rows, inner, prepared, scales);
    else q4_a8_mma<4, Dual4, Prequantized><<<(rows + 63) / 64, 128>>>(input, weights, output, rows, inner, prepared, scales);
    check(cudaGetLastError());
}

void launch(const float* input, const Q4B8G32Block* weights, float* output,
    int rows, int inner, int warps, bool dual4, uint32_t* prepared, float* scales) {
    if (prepared && dual4) launch_variant<true, true>(input, weights, output, rows, inner, warps, prepared, scales);
    else if (prepared) launch_variant<false, true>(input, weights, output, rows, inner, warps, prepared, scales);
    else if (dual4) launch_variant<true>(input, weights, output, rows, inner, warps);
    else launch_variant<false>(input, weights, output, rows, inner, warps);
}

template<class F> double measure(F fn) {
    for (int i = 0; i < 64; ++i) fn(i);
    check(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    for (int i = 0; i < 256; ++i) fn(i);
    check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
    float ms;
    check(cudaEventElapsedTime(&ms, start, stop));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    return ms * 1000 / 256;
}

void run(int rows, int inner, bool benchmark, bool zero = false, bool dual4 = false) {
    const size_t blocks = static_cast<size_t>((rows + 7) / 8) * (inner / 32);
    const int sets = benchmark ? std::max<size_t>(8, (128ull << 20) / (blocks * sizeof(Q4B8G32Block))) : 1;
    std::vector<Q4B8G32Block> host(blocks * sets);
    for (size_t i = 0; i < host.size(); ++i) for (int r = 0; r < 8; ++r) {
        const bool zero_scale = benchmark ? i % 31 == 0 : (i + r) % 31 == 30;
        host[i].scales[r] = zero_scale ? 0.f : (1 + (i + r) % 17) / 37.f;
        for (int p = 0; p < 16; ++p) host[i].q[r][p] = (p * 13 + r * 7 + i * 19) & 255;
    }
    std::vector<float> input(inner), qscales(inner / 32), actual(rows + 8), fp32(rows), initial(rows + 8, 42.f);
    std::vector<int> quantized(inner);
    for (int i = 0; i < inner; ++i) input[i] = zero ? 0.f : (int((i * 13 + 7) % 101) - 50) / 37.f;
    for (int g = 0; g < inner / 32; ++g) {
        float maximum = 0;
        for (int i = 0; i < 32; ++i) maximum = std::max(maximum, std::abs(input[g * 32 + i]));
        const int maximum_code = dual4 ? 119 : 127;
        qscales[g] = maximum / float(maximum_code);
        const float inverse = qscales[g] > 0 ? 1.f / qscales[g] : 0;
        for (int i = 0; i < 32; ++i) quantized[g * 32 + i] = std::clamp(int(std::nearbyint(input[g * 32 + i] * inverse)), -maximum_code, maximum_code);
    }
    Buffer<Q4B8G32Block> weights(host.size()); Buffer<float> x(inner), y(rows + 8);
    Buffer<uint32_t> prepared(inner / 4); Buffer<float> prepared_scales(inner / 32);
    const bool prequantized = std::getenv("MOLLM_Q4_PREQUANT") != nullptr;
    auto execute = [&](const Q4B8G32Block* w, int warps) {
        launch(x.data, w, y.data, rows, inner, warps, dual4,
            prequantized ? prepared.data : nullptr, prequantized ? prepared_scales.data : nullptr);
    };
    check(cudaMemcpy(weights.data, host.data(), host.size() * sizeof(host[0]), cudaMemcpyHostToDevice));
    check(cudaMemcpy(x.data, input.data(), inner * sizeof(float), cudaMemcpyHostToDevice));
    for (int warps : {2, 4, 8}) {
        double error = 0, magnitude = 0;
        for (int set = 0; set < sets; ++set) {
            check(cudaMemcpy(y.data, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice));
            execute(weights.data + set * blocks, warps);
            check(cudaMemcpy(actual.data(), y.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int r = 0; r < rows; ++r) {
                if (benchmark && r >= 16 && std::abs(r - rows / 2) > 2 && r < rows - 16) continue;
                double expected = 0;
                for (int g = 0; g < inner / 32; ++g) {
                    const auto& block = host[set * blocks + (r / 8) * (inner / 32) + g];
                    int dot = 0;
                    for (int k = 0; k < 32; ++k) {
                        const unsigned byte = block.q[r % 8][k / 2];
                        const int weight = int((byte >> ((k % 2) * 4)) & 15) - 8;
                        dot += weight * quantized[g * 32 + k];
                    }
                    expected += dot * double(block.scales[r % 8] * qscales[g]);
                }
                if (!std::isfinite(actual[r])) { std::fprintf(stderr, "nonfinite W4/A8 MMA\n"); std::exit(1); }
                error = std::max(error, std::abs(actual[r] - expected)); magnitude = std::max(magnitude, std::abs(expected));
            }
            if (!std::all_of(actual.begin() + rows, actual.end(), [](float v) { return v == 42.f; })) {
                std::fprintf(stderr, "W4/A8 output overrun\n"); std::exit(1);
            }
        }
        if (error / std::max(1., magnitude) > 2e-5) {
            std::fprintf(stderr, "W4/A8 mismatch rows=%d K=%d warps=%d error=%g\n", rows, inner, warps, error / std::max(1., magnitude)); std::exit(1);
        }
        std::printf("w4a8 rows=%d K=%d warps=%d zero=%d dual4=%d cpu_quantized_error=%g\n", rows, inner, warps, zero, dual4, error / std::max(1., magnitude));
        if (!benchmark) continue;
        execute(weights.data, warps);
        check(cudaMemcpy(actual.data(), y.data, rows * sizeof(float), cudaMemcpyDeviceToHost));
        mollm_cuda::launch_q4_g32_dense_gemv(x.data, weights.data, y.data, rows, inner);
        check(cudaMemcpy(fp32.data(), y.data, rows * sizeof(float), cudaMemcpyDeviceToHost));
        double diff = 0, energy = 0;
        for (int r = 0; r < rows; ++r) { diff += std::pow(double(actual[r]) - fp32[r], 2); energy += double(fp32[r]) * fp32[r]; }
        for (int repeat = 0; repeat < 3; ++repeat) {
            const double candidate = measure([&](int i) { execute(weights.data + (i % sets) * blocks, warps); });
            const double control = measure([&](int i) { mollm_cuda::launch_q4_g32_dense_gemv(x.data, weights.data + (i % sets) * blocks, y.data, rows, inner); });
            std::printf("w4a8-bench rows=%d K=%d sets=%d warps=%d dual4=%d prequantized=%d repeat=%d us=%.3f control_us=%.3f speedup=%.3f fp32_relative_l2=%g cpu_checked_rows=%d\n",
                rows, inner, sets, warps, dual4, prequantized, repeat, candidate, control, control / candidate, std::sqrt(diff / std::max(1., energy)), rows <= 65 ? rows : 37);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    for (int code = -119; code <= 119; ++code) {
        int low = code & 15;
        if (low >= 8) low -= 16;
        const int high = (code + 8) >> 4;
        if (high < -8 || high > 7 || high * 16 + low != code) return 1;
    }
    cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props, 0));
    if (props.major < 8) { std::puts("SKIP: W4/A8 MMA requires SM80+"); return 0; }
    if (argc == 3) {
        char *end_rows, *end_inner;
        const long rows = std::strtol(argv[1], &end_rows, 10), inner = std::strtol(argv[2], &end_inner, 10);
        if (end_rows == argv[1] || *end_rows || end_inner == argv[2] || *end_inner ||
            rows < 1 || rows > 32768 || inner < 32 || inner > 2560 || inner % 32) {
            std::fprintf(stderr, "Expected rows1..32768 and inner32..2560 divisible by32\n"); return 2;
        }
        run(int(rows), int(inner), true, false, std::getenv("MOLLM_Q4_DUAL_INT4") != nullptr); return 0;
    }
    if (argc != 1) { std::fprintf(stderr, "Usage: %s [rows inner]\n", argv[0]); return 2; }
    for (bool dual4 : {false, true}) {
        for (int rows : {1, 3, 16, 19, 64, 65}) for (int k : {32, 64, 96, 128, 640, 2560}) run(rows, k, false, false, dual4);
        run(19, 2560, false, true, dual4);
    }
    std::puts("W4/A8 quantized arithmetic checks passed; model accuracy and inference speed remain unvalidated.");
}
