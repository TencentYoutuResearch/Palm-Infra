#include "kernels/cpu/matmul/matmul_internal.h"
#include "core/bf16.h"
#include "runtime/threading.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int kMxBlock = 32;

int balanced_parallel_grain(int total, int threads, int minimum,
                            int alignment = 1) {
    threads = std::max(threads, 1);
    alignment = std::max(alignment, 1);
    int grain = std::max(minimum, (total + threads - 1) / threads);
    return ((grain + alignment - 1) / alignment) * alignment;
}

struct Mxfp4ActivationScratch {
    Q4GemvScratch primary;
    std::vector<int8_t> residual_even;
    std::vector<int8_t> residual_odd;
    std::vector<float> residual_scales;
    std::vector<float> fp8_dequant;
    std::vector<float> residual;
};

inline int8_t mxfp4_integer_coefficient(uint8_t nibble) {
    // E2M1 magnitudes multiplied by two:
    //   0, .5, 1, 1.5, 2, 3, 4, 6 -> 0, 1, 2, 3, 4, 6, 8, 12.
    static constexpr int8_t table[16] = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    return table[nibble & 0x0f];
}

inline uint8_t packed_nibble(const uint8_t* row, int k) {
    const uint8_t packed = row[k / 2];
    return (k & 1) ? packed >> 4 : packed & 0x0f;
}

struct Fp8Q8Tables {
    std::array<std::array<int8_t, 256>, 127> values{};
    std::array<float, 127> scales{};
};

const Fp8Q8Tables& fp8_q8_tables() {
    // ARM CPUs do not have native FP8 arithmetic. Map the finite E4M3
    // coefficients to a signed byte and combine the fixed coefficient scale
    // with the checkpoint's E8M0 tile scale after SDOT. This is an
    // on-the-fly register/L1 conversion: the mmap weight remains native FP8
    // and no second resident copy is created.
    static const Fp8Q8Tables tables = [] {
        Fp8Q8Tables result;
        for (int maximum = 0; maximum < 127; ++maximum) {
            const float max_value =
                decode_fp8_e4m3fn(static_cast<uint8_t>(maximum));
            const float scale =
                max_value > 0.0f ? max_value / 127.0f : 1.0f;
            result.scales[maximum] = scale;
            for (int byte = 0; byte < 256; ++byte) {
                const float decoded =
                    decode_fp8_e4m3fn(static_cast<uint8_t>(byte));
                if (!std::isfinite(decoded)) {
                    result.values[maximum][byte] = 0;
                    continue;
                }
                const int quantized = static_cast<int>(
                    std::nearbyint(decoded / scale));
                result.values[maximum][byte] = static_cast<int8_t>(
                    std::max(-127, std::min(127, quantized)));
            }
        }
        return result;
    }();
    return tables;
}

inline int32_t q8_dot32(const int8_t* activation, const int8_t* weight) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t dot = vdupq_n_s32(0);
#if defined(__aarch64__)
    __asm__("sdot %0.4s, %1.16b, %2.16b"
            : "+w"(dot)
            : "w"(vld1q_s8(activation)), "w"(vld1q_s8(weight)));
    __asm__("sdot %0.4s, %1.16b, %2.16b"
            : "+w"(dot)
            : "w"(vld1q_s8(activation + 16)),
              "w"(vld1q_s8(weight + 16)));
#else
    dot = vdotq_s32(
        dot, vld1q_s8(activation), vld1q_s8(weight));
    dot = vdotq_s32(
        dot, vld1q_s8(activation + 16), vld1q_s8(weight + 16));
#endif
    return vaddvq_s32(dot);
#else
    int32_t dot = 0;
    for (int i = 0; i < 32; ++i)
        dot += static_cast<int32_t>(activation[i]) * weight[i];
    return dot;
#endif
}

inline uint8_t fp8_block_maximum(const uint8_t* source, int count) {
    uint8_t maximum = 0;
#if HAS_NEON && defined(__aarch64__)
    uint8x16_t vector_max = vdupq_n_u8(0);
    int i = 0;
    for (; i + 15 < count; i += 16) {
        const uint8x16_t magnitude =
            vandq_u8(vld1q_u8(source + i), vdupq_n_u8(0x7f));
        vector_max = vmaxq_u8(vector_max, magnitude);
    }
    maximum = vmaxvq_u8(vector_max);
    for (; i < count; ++i)
        maximum = std::max(maximum,
                           static_cast<uint8_t>(source[i] & 0x7f));
#else
    for (int i = 0; i < count; ++i)
        maximum = std::max(maximum,
                           static_cast<uint8_t>(source[i] & 0x7f));
#endif
    // 0x7f is E4M3FN NaN rather than a finite maximum.
    return std::min<uint8_t>(maximum, 126);
}

inline void map_fp8_q8_block32(const uint8_t* source, int8_t* output,
                               const std::array<int8_t, 256>& table) {
    for (int i = 0; i < 32; ++i)
        output[i] = table[source[i]];
}

void matmul_mxfp4_reference_range(
    const float* a, const uint8_t* weights, const uint8_t* scales, float* c,
    int M, int N, int K, int lda, int ldc, int m_begin, int m_end,
    int n_begin, int n_end) {
    (void)M;
    (void)N;
    const int bytes_per_row = K / 2;
    const int groups_per_row = K / kMxBlock;
    for (int m = m_begin; m < m_end; ++m) {
        const float* a_row = a + static_cast<size_t>(m) * lda;
        float* c_row = c + static_cast<size_t>(m) * ldc;
        for (int n = n_begin; n < n_end; ++n) {
            const uint8_t* w_row =
                weights + static_cast<size_t>(n) * bytes_per_row;
            const uint8_t* s_row =
                scales + static_cast<size_t>(n) * groups_per_row;
            float sum = 0.0f;
            for (int group = 0; group < groups_per_row; ++group) {
                const int k_begin = group * kMxBlock;
#if HAS_NEON && defined(__aarch64__)
                const int8x16_t coefficient_table = {
                    0, 1, 2, 3, 4, 6, 8, 12,
                    0, -1, -2, -3, -4, -6, -8, -12,
                };
                const uint8x16_t packed =
                    vld1q_u8(w_row + static_cast<size_t>(group) * 16);
                const int8x16_t even = vqtbl1q_s8(
                    coefficient_table,
                    vandq_u8(packed, vdupq_n_u8(0x0f)));
                const int8x16_t odd = vqtbl1q_s8(
                    coefficient_table, vshrq_n_u8(packed, 4));
                const int8x16x2_t ordered = vzipq_s8(even, odd);
                float32x4_t dot0 = vdupq_n_f32(0.0f);
                float32x4_t dot1 = vdupq_n_f32(0.0f);
                for (int half = 0; half < 2; ++half) {
                    const int8x16_t coefficients = ordered.val[half];
                    const int16x8_t low =
                        vmovl_s8(vget_low_s8(coefficients));
                    const int16x8_t high =
                        vmovl_s8(vget_high_s8(coefficients));
                    const int activation_offset = k_begin + half * 16;
                    dot0 = vfmaq_f32(
                        dot0,
                        vld1q_f32(a_row + activation_offset),
                        vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))));
                    dot1 = vfmaq_f32(
                        dot1,
                        vld1q_f32(a_row + activation_offset + 4),
                        vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))));
                    dot0 = vfmaq_f32(
                        dot0,
                        vld1q_f32(a_row + activation_offset + 8),
                        vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))));
                    dot1 = vfmaq_f32(
                        dot1,
                        vld1q_f32(a_row + activation_offset + 12),
                        vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))));
                }
                // Integer coefficients are exactly 2 * E2M1.
                const float block =
                    0.5f * vaddvq_f32(vaddq_f32(dot0, dot1));
#else
                float block = 0.0f;
                for (int k = 0; k < kMxBlock; ++k) {
                    block +=
                        a_row[k_begin + k] *
                        decode_mxfp4_e2m1(
                            packed_nibble(w_row, k_begin + k));
                }
#endif
                sum += block * decode_e8m0(s_row[group]);
            }
            c_row[n] = sum;
        }
    }
}

void matmul_fp8_reference_range(
    const float* a, const uint8_t* weights, const uint8_t* scales, float* c,
    int M, int N, int K, int lda, int ldc, int m_begin, int m_end,
    int n_begin, int n_end) {
    (void)M;
    const int k_blocks = (K + 127) / 128;
    for (int m = m_begin; m < m_end; ++m) {
        const float* a_row = a + static_cast<size_t>(m) * lda;
        float* c_row = c + static_cast<size_t>(m) * ldc;
        for (int n = n_begin; n < n_end; ++n) {
            const uint8_t* w_row =
                weights + static_cast<size_t>(n) * K;
            const uint8_t* s_row =
                scales + static_cast<size_t>(n / 128) * k_blocks;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * decode_fp8_e4m3fn(w_row[k]) *
                       decode_e8m0(s_row[k / 128]);
            }
            c_row[n] = sum;
        }
    }
}

#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
inline int32_t mxfp4_q8_dot32(const uint8_t* packed,
                              const int8_t* q_even,
                              const int8_t* q_odd) {
    const int8x16_t coefficient_table = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    const uint8x16_t bytes = vld1q_u8(packed);
    const uint8x16_t low =
        vandq_u8(bytes, vdupq_n_u8(0x0f));
    const uint8x16_t high = vshrq_n_u8(bytes, 4);
    const int8x16_t w_even =
        vqtbl1q_s8(coefficient_table, low);
    const int8x16_t w_odd =
        vqtbl1q_s8(coefficient_table, high);
    int32x4_t dot = vdupq_n_s32(0);
#if defined(__aarch64__)
    __asm__("sdot %0.4s, %1.16b, %2.16b"
            : "+w"(dot)
            : "w"(w_even), "w"(vld1q_s8(q_even)));
    __asm__("sdot %0.4s, %1.16b, %2.16b"
            : "+w"(dot)
            : "w"(w_odd), "w"(vld1q_s8(q_odd)));
#else
    dot = vdotq_s32(dot, w_even, vld1q_s8(q_even));
    dot = vdotq_s32(dot, w_odd, vld1q_s8(q_odd));
#endif
    return vaddvq_s32(dot);
}
#endif

#if HAS_NEON && defined(__aarch64__)
inline float32x4_t decode_fp8_e4m3fn_u32(uint32x4_t bytes) {
    const uint32x4_t exponent =
        vandq_u32(vshrq_n_u32(bytes, 3), vdupq_n_u32(0x0f));
    const uint32x4_t mantissa =
        vandq_u32(bytes, vdupq_n_u32(0x07));

    // For normal E4M3 values:
    //   (8 + mantissa) * 2^(exponent - 10).
    const float32x4_t coefficient =
        vcvtq_f32_u32(vaddq_u32(mantissa, vdupq_n_u32(8)));
    const uint32x4_t power_bits =
        vshlq_n_u32(vaddq_u32(exponent, vdupq_n_u32(117)), 23);
    float32x4_t value =
        vmulq_f32(coefficient, vreinterpretq_f32_u32(power_bits));

    // Exponent zero is subnormal: mantissa * 2^-9.
    const float32x4_t subnormal =
        vmulq_n_f32(vcvtq_f32_u32(mantissa), 1.0f / 512.0f);
    value = vbslq_f32(vceqq_u32(exponent, vdupq_n_u32(0)),
                      subnormal, value);

    // Move the FP8 sign bit to the IEEE float sign position.
    const uint32x4_t sign =
        vshlq_n_u32(vandq_u32(bytes, vdupq_n_u32(0x80)), 24);
    return vreinterpretq_f32_u32(
        veorq_u32(vreinterpretq_u32_f32(value), sign));
}

inline void accumulate_fp8_16(const float* activation,
                              const uint8_t* weights,
                              float32x4_t& sum0,
                              float32x4_t& sum1,
                              float32x4_t& sum2,
                              float32x4_t& sum3) {
    const uint8x16_t bytes = vld1q_u8(weights);
    const uint16x8_t low16 = vmovl_u8(vget_low_u8(bytes));
    const uint16x8_t high16 = vmovl_u8(vget_high_u8(bytes));
    const float32x4_t w0 =
        decode_fp8_e4m3fn_u32(vmovl_u16(vget_low_u16(low16)));
    const float32x4_t w1 =
        decode_fp8_e4m3fn_u32(vmovl_u16(vget_high_u16(low16)));
    const float32x4_t w2 =
        decode_fp8_e4m3fn_u32(vmovl_u16(vget_low_u16(high16)));
    const float32x4_t w3 =
        decode_fp8_e4m3fn_u32(vmovl_u16(vget_high_u16(high16)));
    sum0 = vfmaq_f32(sum0, vld1q_f32(activation), w0);
    sum1 = vfmaq_f32(sum1, vld1q_f32(activation + 4), w1);
    sum2 = vfmaq_f32(sum2, vld1q_f32(activation + 8), w2);
    sum3 = vfmaq_f32(sum3, vld1q_f32(activation + 12), w3);
}

inline void mxfp4_q8_dot32_dual(
    const uint8_t* packed, const int8_t* q_even, const int8_t* q_odd,
    const int8_t* residual_even, const int8_t* residual_odd,
    int32_t& primary, int32_t& residual) {
    const int8x16_t coefficient_table = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    const uint8x16_t bytes = vld1q_u8(packed);
    const int8x16_t w_even = vqtbl1q_s8(
        coefficient_table, vandq_u8(bytes, vdupq_n_u8(0x0f)));
    const int8x16_t w_odd = vqtbl1q_s8(
        coefficient_table, vshrq_n_u8(bytes, 4));
    int32x4_t primary_dot = vdupq_n_s32(0);
    int32x4_t residual_dot = vdupq_n_s32(0);
    primary_dot = vdotq_s32(
        primary_dot, w_even, vld1q_s8(q_even));
    primary_dot = vdotq_s32(
        primary_dot, w_odd, vld1q_s8(q_odd));
    residual_dot = vdotq_s32(
        residual_dot, w_even, vld1q_s8(residual_even));
    residual_dot = vdotq_s32(
        residual_dot, w_odd, vld1q_s8(residual_odd));
    primary = vaddvq_s32(primary_dot);
    residual = vaddvq_s32(residual_dot);
}

inline void mxfp4_q8_dot32_dual_pair(
    const uint8_t* packed0, const uint8_t* packed1,
    const int8_t* q_even, const int8_t* q_odd,
    const int8_t* residual_even, const int8_t* residual_odd,
    int32_t& primary0, int32_t& residual0,
    int32_t& primary1, int32_t& residual1) {
    const int8x16_t coefficient_table = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
#if defined(__ARM_FEATURE_MATMUL_INT8)
    const uint8x16_t bytes0 = vld1q_u8(packed0);
    const uint8x16_t bytes1 = vld1q_u8(packed1);
    const int8x16_t weight_even0 = vqtbl1q_s8(
        coefficient_table,
        vandq_u8(bytes0, vdupq_n_u8(0x0f)));
    const int8x16_t weight_even1 = vqtbl1q_s8(
        coefficient_table,
        vandq_u8(bytes1, vdupq_n_u8(0x0f)));
    const int8x16_t weight_odd0 = vqtbl1q_s8(
        coefficient_table, vshrq_n_u8(bytes0, 4));
    const int8x16_t weight_odd1 = vqtbl1q_s8(
        coefficient_table, vshrq_n_u8(bytes1, 4));
    int32x4_t dots = vdupq_n_s32(0);
    dots = vmmlaq_s32(
        dots,
        vcombine_s8(vld1_s8(q_even), vld1_s8(residual_even)),
        vcombine_s8(vget_low_s8(weight_even0),
                    vget_low_s8(weight_even1)));
    dots = vmmlaq_s32(
        dots,
        vcombine_s8(vld1_s8(q_even + 8),
                    vld1_s8(residual_even + 8)),
        vcombine_s8(vget_high_s8(weight_even0),
                    vget_high_s8(weight_even1)));
    dots = vmmlaq_s32(
        dots,
        vcombine_s8(vld1_s8(q_odd), vld1_s8(residual_odd)),
        vcombine_s8(vget_low_s8(weight_odd0),
                    vget_low_s8(weight_odd1)));
    dots = vmmlaq_s32(
        dots,
        vcombine_s8(vld1_s8(q_odd + 8),
                    vld1_s8(residual_odd + 8)),
        vcombine_s8(vget_high_s8(weight_odd0),
                    vget_high_s8(weight_odd1)));
    primary0 = vgetq_lane_s32(dots, 0);
    primary1 = vgetq_lane_s32(dots, 1);
    residual0 = vgetq_lane_s32(dots, 2);
    residual1 = vgetq_lane_s32(dots, 3);
    return;
#endif
    const int8x16_t activation_even = vld1q_s8(q_even);
    const int8x16_t activation_odd = vld1q_s8(q_odd);
    auto dot = [&](const uint8_t* packed,
                   int32x4_t& primary_dot,
                   int32x4_t& residual_dot) {
        const uint8x16_t bytes = vld1q_u8(packed);
        const int8x16_t weight_even = vqtbl1q_s8(
            coefficient_table,
            vandq_u8(bytes, vdupq_n_u8(0x0f)));
        const int8x16_t weight_odd = vqtbl1q_s8(
            coefficient_table, vshrq_n_u8(bytes, 4));
        primary_dot = vdotq_s32(
            vdupq_n_s32(0), weight_even, activation_even);
        primary_dot = vdotq_s32(
            primary_dot, weight_odd, activation_odd);
        residual_dot = vdotq_s32(
            vdupq_n_s32(0), weight_even, vld1q_s8(residual_even));
        residual_dot = vdotq_s32(
            residual_dot, weight_odd, vld1q_s8(residual_odd));
    };
    int32x4_t primary_dot0;
    int32x4_t residual_dot0;
    int32x4_t primary_dot1;
    int32x4_t residual_dot1;
    dot(packed0, primary_dot0, residual_dot0);
    dot(packed1, primary_dot1, residual_dot1);
    const int32x4_t primary_pairs =
        vpaddq_s32(primary_dot0, primary_dot1);
    const int32x4_t residual_pairs =
        vpaddq_s32(residual_dot0, residual_dot1);
    const int32x4_t primary_sums =
        vpaddq_s32(primary_pairs, primary_pairs);
    const int32x4_t residual_sums =
        vpaddq_s32(residual_pairs, residual_pairs);
    primary0 = vgetq_lane_s32(primary_sums, 0);
    primary1 = vgetq_lane_s32(primary_sums, 1);
    residual0 = vgetq_lane_s32(residual_sums, 0);
    residual1 = vgetq_lane_s32(residual_sums, 1);
}

inline void mxfp4_q8_dot32_dual_four(
    const uint8_t* packed0, const uint8_t* packed1,
    const uint8_t* packed2, const uint8_t* packed3,
    const int8_t* q_even, const int8_t* q_odd,
    const int8_t* residual_even, const int8_t* residual_odd,
    int32x4_t& primary, int32x4_t& residual) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    int32_t primary0;
    int32_t primary1;
    int32_t primary2;
    int32_t primary3;
    int32_t residual0;
    int32_t residual1;
    int32_t residual2;
    int32_t residual3;
    mxfp4_q8_dot32_dual_pair(
        packed0, packed1, q_even, q_odd,
        residual_even, residual_odd,
        primary0, residual0, primary1, residual1);
    mxfp4_q8_dot32_dual_pair(
        packed2, packed3, q_even, q_odd,
        residual_even, residual_odd,
        primary2, residual2, primary3, residual3);
    const int32_t primary_values[4] = {
        primary0, primary1, primary2, primary3};
    const int32_t residual_values[4] = {
        residual0, residual1, residual2, residual3};
    primary = vld1q_s32(primary_values);
    residual = vld1q_s32(residual_values);
    return;
#endif
    const int8x16_t coefficient_table = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    const int8x16_t activation_even = vld1q_s8(q_even);
    const int8x16_t activation_odd = vld1q_s8(q_odd);
    auto dot = [&](const uint8_t* packed,
                   int32x4_t& primary_dot,
                   int32x4_t& residual_dot) {
        const uint8x16_t bytes = vld1q_u8(packed);
        const int8x16_t weight_even = vqtbl1q_s8(
            coefficient_table,
            vandq_u8(bytes, vdupq_n_u8(0x0f)));
        const int8x16_t weight_odd = vqtbl1q_s8(
            coefficient_table, vshrq_n_u8(bytes, 4));
        primary_dot = vdotq_s32(
            vdupq_n_s32(0), weight_even, activation_even);
        primary_dot = vdotq_s32(
            primary_dot, weight_odd, activation_odd);
        residual_dot = vdotq_s32(
            vdupq_n_s32(0), weight_even, vld1q_s8(residual_even));
        residual_dot = vdotq_s32(
            residual_dot, weight_odd, vld1q_s8(residual_odd));
    };
    int32x4_t primary_dot0;
    int32x4_t residual_dot0;
    int32x4_t primary_dot1;
    int32x4_t residual_dot1;
    int32x4_t primary_dot2;
    int32x4_t residual_dot2;
    int32x4_t primary_dot3;
    int32x4_t residual_dot3;
    dot(packed0, primary_dot0, residual_dot0);
    dot(packed1, primary_dot1, residual_dot1);
    dot(packed2, primary_dot2, residual_dot2);
    dot(packed3, primary_dot3, residual_dot3);
    auto reduce_pairs = [](int32x4_t first, int32x4_t second) {
        const int32x4_t pairs = vpaddq_s32(first, second);
        const int32x4_t sums = vpaddq_s32(pairs, pairs);
        return vget_low_s32(sums);
    };
    primary = vcombine_s32(
        reduce_pairs(primary_dot0, primary_dot1),
        reduce_pairs(primary_dot2, primary_dot3));
    residual = vcombine_s32(
        reduce_pairs(residual_dot0, residual_dot1),
        reduce_pairs(residual_dot2, residual_dot3));
}

#endif

void matmul_fp8_gemv_range(
    const float* activation, const uint8_t* weights, const uint8_t* scales,
    float* output, int N, int K, int n_begin, int n_end) {
    (void)N;
    const int k_blocks = (K + 127) / 128;
    for (int n = n_begin; n < n_end; ++n) {
        const uint8_t* w_row =
            weights + static_cast<size_t>(n) * K;
        const uint8_t* s_row =
            scales + static_cast<size_t>(n / 128) * k_blocks;
        float sum = 0.0f;
        for (int block = 0; block < k_blocks; ++block) {
            const int k_begin = block * 128;
            const int k_end = std::min(k_begin + 128, K);
#if HAS_NEON && defined(__aarch64__)
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            int k = k_begin;
            for (; k + 15 < k_end; k += 16) {
                accumulate_fp8_16(activation + k, w_row + k,
                                  sum0, sum1, sum2, sum3);
            }
            float block_sum =
                vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1),
                                     vaddq_f32(sum2, sum3)));
            for (; k < k_end; ++k)
                block_sum += activation[k] *
                             decode_fp8_e4m3fn(w_row[k]);
#else
            float block_sum = 0.0f;
            for (int k = k_begin; k < k_end; ++k)
                block_sum += activation[k] *
                             decode_fp8_e4m3fn(w_row[k]);
#endif
            sum += block_sum * decode_e8m0(s_row[block]);
        }
        output[n] = sum;
    }
}

void matmul_mxfp4_q8_gemv_range(
    const int8_t* q_even, const int8_t* q_odd, const float* a_scales,
    const int8_t* residual_even, const int8_t* residual_odd,
    const float* residual_scales,
    const uint8_t* weights, const uint8_t* scales, float* output,
    int N, int K, int n_begin, int n_end) {
    (void)N;
    const int groups = K / kMxBlock;
    const int bytes_per_row = K / 2;
    int n = n_begin;
#if HAS_NEON && defined(__aarch64__)
    for (; n + 3 < n_end; n += 4) {
        const uint8_t* w_row0 =
            weights + static_cast<size_t>(n) * bytes_per_row;
        const uint8_t* w_row1 = w_row0 + bytes_per_row;
        const uint8_t* w_row2 = w_row1 + bytes_per_row;
        const uint8_t* w_row3 = w_row2 + bytes_per_row;
        const uint8_t* s_row0 =
            scales + static_cast<size_t>(n) * groups;
        const uint8_t* s_row1 = s_row0 + groups;
        const uint8_t* s_row2 = s_row1 + groups;
        const uint8_t* s_row3 = s_row2 + groups;
        float32x4_t sums = vdupq_n_f32(0.0f);
        for (int group = 0; group < groups; ++group) {
            const size_t packed_offset =
                static_cast<size_t>(group) * 16;
            int32x4_t primary_dot;
            int32x4_t residual_dot;
            mxfp4_q8_dot32_dual_four(
                w_row0 + packed_offset, w_row1 + packed_offset,
                w_row2 + packed_offset, w_row3 + packed_offset,
                q_even + packed_offset, q_odd + packed_offset,
                residual_even + packed_offset,
                residual_odd + packed_offset,
                primary_dot, residual_dot);
            const uint32x4_t scale_codes = {
                s_row0[group], s_row1[group],
                s_row2[group], s_row3[group],
            };
            uint32x4_t scale_bits = vshlq_n_u32(scale_codes, 23);
            scale_bits = vbslq_u32(
                vceqq_u32(scale_codes, vdupq_n_u32(0)),
                vdupq_n_u32(uint32_t{1} << 22), scale_bits);
            scale_bits = vbslq_u32(
                vceqq_u32(scale_codes, vdupq_n_u32(0xff)),
                vdupq_n_u32(0x7fc00000u), scale_bits);
            const float32x4_t weight_scales =
                vreinterpretq_f32_u32(scale_bits);
            const float32x4_t combined = vmlaq_n_f32(
                vmulq_n_f32(vcvtq_f32_s32(primary_dot), a_scales[group]),
                vcvtq_f32_s32(residual_dot), residual_scales[group]);
            sums = vfmaq_f32(
                sums, combined, vmulq_n_f32(weight_scales, 0.5f));
        }
        vst1q_f32(output + n, sums);
    }
    for (; n + 1 < n_end; n += 2) {
        const uint8_t* w_row0 =
            weights + static_cast<size_t>(n) * bytes_per_row;
        const uint8_t* w_row1 = w_row0 + bytes_per_row;
        const uint8_t* s_row0 =
            scales + static_cast<size_t>(n) * groups;
        const uint8_t* s_row1 = s_row0 + groups;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int group = 0; group < groups; ++group) {
            const size_t packed_offset =
                static_cast<size_t>(group) * 16;
            int32_t dot0 = 0;
            int32_t residual_dot0 = 0;
            int32_t dot1 = 0;
            int32_t residual_dot1 = 0;
            mxfp4_q8_dot32_dual_pair(
                w_row0 + packed_offset, w_row1 + packed_offset,
                q_even + packed_offset, q_odd + packed_offset,
                residual_even + packed_offset,
                residual_odd + packed_offset,
                dot0, residual_dot0, dot1, residual_dot1);
            const float primary_scale = a_scales[group];
            const float correction_scale = residual_scales[group];
            sum0 += 0.5f *
                    (static_cast<float>(dot0) * primary_scale +
                     static_cast<float>(residual_dot0) *
                         correction_scale) *
                    decode_e8m0(s_row0[group]);
            sum1 += 0.5f *
                    (static_cast<float>(dot1) * primary_scale +
                     static_cast<float>(residual_dot1) *
                         correction_scale) *
                    decode_e8m0(s_row1[group]);
        }
        output[n] = sum0;
        output[n + 1] = sum1;
    }
#endif
    for (; n < n_end; ++n) {
        const uint8_t* w_row =
            weights + static_cast<size_t>(n) * bytes_per_row;
        const uint8_t* s_row =
            scales + static_cast<size_t>(n) * groups;
        float sum = 0.0f;
        for (int group = 0; group < groups; ++group) {
            int32_t dot = 0;
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
            int32_t residual_dot = 0;
            mxfp4_q8_dot32_dual(
                w_row + static_cast<size_t>(group) * 16,
                q_even + static_cast<size_t>(group) * 16,
                q_odd + static_cast<size_t>(group) * 16,
                residual_even + static_cast<size_t>(group) * 16,
                residual_odd + static_cast<size_t>(group) * 16,
                dot, residual_dot);
#else
            int32_t residual_dot = 0;
            const uint8_t* packed =
                w_row + static_cast<size_t>(group) * 16;
            for (int i = 0; i < 16; ++i) {
                dot += static_cast<int32_t>(
                           mxfp4_integer_coefficient(packed[i] & 0x0f)) *
                       q_even[static_cast<size_t>(group) * 16 + i];
                dot += static_cast<int32_t>(
                           mxfp4_integer_coefficient(packed[i] >> 4)) *
                       q_odd[static_cast<size_t>(group) * 16 + i];
                residual_dot += static_cast<int32_t>(
                                    mxfp4_integer_coefficient(
                                        packed[i] & 0x0f)) *
                                residual_even[
                                    static_cast<size_t>(group) * 16 + i];
                residual_dot += static_cast<int32_t>(
                                    mxfp4_integer_coefficient(
                                        packed[i] >> 4)) *
                                residual_odd[
                                    static_cast<size_t>(group) * 16 + i];
            }
#endif
            // Integer coefficients represent 2 * E2M1.
            sum += 0.5f *
                   (static_cast<float>(dot) * a_scales[group] +
                    static_cast<float>(residual_dot) *
                        residual_scales[group]) *
                   decode_e8m0(s_row[group]);
        }
        output[n] = sum;
    }
}

void matmul_mxfp4_q8_gemm_range(
    const std::vector<Mxfp4ActivationScratch>& quantized,
    const uint8_t* weights, const uint8_t* scales, float* output,
    int M, int N, int K, int ldc, int n_begin, int n_end) {
    (void)N;
    const int groups = K / kMxBlock;
    const int bytes_per_row = K / 2;
    std::vector<float> sums(static_cast<size_t>(M));
    for (int n = n_begin; n < n_end; ++n) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        const uint8_t* w_row =
            weights + static_cast<size_t>(n) * bytes_per_row;
        const uint8_t* s_row =
            scales + static_cast<size_t>(n) * groups;
        for (int group = 0; group < groups; ++group) {
            const uint8_t* packed =
                w_row + static_cast<size_t>(group) * 16;
            const float weight_scale =
                0.5f * decode_e8m0(s_row[group]);
            for (int m = 0; m < M; ++m) {
                const size_t q_offset = static_cast<size_t>(group) * 16;
                const Mxfp4ActivationScratch& row = quantized[m];
                int32_t dot = 0;
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
                int32_t residual_dot = 0;
                mxfp4_q8_dot32_dual(
                    packed, row.primary.qA_even.data() + q_offset,
                    row.primary.qA_odd.data() + q_offset,
                    row.residual_even.data() + q_offset,
                    row.residual_odd.data() + q_offset,
                    dot, residual_dot);
#else
                int32_t residual_dot = 0;
                for (int i = 0; i < 16; ++i) {
                    dot += static_cast<int32_t>(
                               mxfp4_integer_coefficient(
                                    packed[i] & 0x0f)) *
                           row.primary.qA_even[q_offset + i];
                    dot += static_cast<int32_t>(
                               mxfp4_integer_coefficient(
                                    packed[i] >> 4)) *
                           row.primary.qA_odd[q_offset + i];
                    residual_dot += static_cast<int32_t>(
                                        mxfp4_integer_coefficient(
                                            packed[i] & 0x0f)) *
                                    row.residual_even[q_offset + i];
                    residual_dot += static_cast<int32_t>(
                                        mxfp4_integer_coefficient(
                                            packed[i] >> 4)) *
                                    row.residual_odd[q_offset + i];
                }
#endif
                sums[m] +=
                    (static_cast<float>(dot) *
                         row.primary.a_scales[group] +
                     static_cast<float>(residual_dot) *
                         row.residual_scales[group]) *
                    weight_scale;
            }
        }
        for (int m = 0; m < M; ++m)
            output[static_cast<size_t>(m) * ldc + n] = sums[m];
    }
}

void matmul_fp8_gemm_range(
    const float* activation, int lda, const uint8_t* weights,
    const uint8_t* scales, float* output, int M, int N, int K, int ldc,
    int n_begin, int n_end) {
    (void)N;
    const int k_blocks = (K + 127) / 128;
    std::vector<float> sums(static_cast<size_t>(M));
    for (int n = n_begin; n < n_end; ++n) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        const uint8_t* w_row =
            weights + static_cast<size_t>(n) * K;
        const uint8_t* s_row =
            scales + static_cast<size_t>(n / 128) * k_blocks;
        for (int block = 0; block < k_blocks; ++block) {
            const int k_begin = block * 128;
            const int k_end = std::min(k_begin + 128, K);
            const float weight_scale = decode_e8m0(s_row[block]);
            int k = k_begin;
#if HAS_NEON && defined(__aarch64__)
            for (; k + 15 < k_end; k += 16) {
                const uint8x16_t bytes = vld1q_u8(w_row + k);
                const uint16x8_t low16 =
                    vmovl_u8(vget_low_u8(bytes));
                const uint16x8_t high16 =
                    vmovl_u8(vget_high_u8(bytes));
                float32x4_t w0 = decode_fp8_e4m3fn_u32(
                    vmovl_u16(vget_low_u16(low16)));
                float32x4_t w1 = decode_fp8_e4m3fn_u32(
                    vmovl_u16(vget_high_u16(low16)));
                float32x4_t w2 = decode_fp8_e4m3fn_u32(
                    vmovl_u16(vget_low_u16(high16)));
                float32x4_t w3 = decode_fp8_e4m3fn_u32(
                    vmovl_u16(vget_high_u16(high16)));
                w0 = vmulq_n_f32(w0, weight_scale);
                w1 = vmulq_n_f32(w1, weight_scale);
                w2 = vmulq_n_f32(w2, weight_scale);
                w3 = vmulq_n_f32(w3, weight_scale);
                for (int m = 0; m < M; ++m) {
                    const float* a =
                        activation + static_cast<size_t>(m) * lda + k;
                    const float32x4_t dot0 =
                        vmulq_f32(vld1q_f32(a), w0);
                    const float32x4_t dot1 =
                        vmulq_f32(vld1q_f32(a + 4), w1);
                    const float32x4_t dot2 =
                        vmulq_f32(vld1q_f32(a + 8), w2);
                    const float32x4_t dot3 =
                        vmulq_f32(vld1q_f32(a + 12), w3);
                    sums[m] += vaddvq_f32(
                        vaddq_f32(vaddq_f32(dot0, dot1),
                                  vaddq_f32(dot2, dot3)));
                }
            }
#endif
            for (; k < k_end; ++k) {
                const float weight =
                    decode_fp8_e4m3fn(w_row[k]) * weight_scale;
                for (int m = 0; m < M; ++m) {
                    sums[m] +=
                        activation[static_cast<size_t>(m) * lda + k] *
                        weight;
                }
            }
        }
        for (int m = 0; m < M; ++m)
            output[static_cast<size_t>(m) * ldc + n] = sums[m];
    }
}

void matmul_fp8_q8_gemv_range(
    const int8_t* quantized_activation, const float* activation_scales,
    const uint8_t* weights, const uint8_t* scales, float* output,
    int N, int K, int n_begin, int n_end) {
    (void)N;
    const int activation_groups = K / MATMUL_Q8_BLOCK;
    const int weight_k_blocks = (K + 127) / 128;
    const auto& tables = fp8_q8_tables();
    alignas(16) int8_t quantized_weight[MATMUL_Q8_BLOCK];
    for (int n = n_begin; n < n_end; ++n) {
        const uint8_t* weight_row =
            weights + static_cast<size_t>(n) * K;
        const uint8_t* scale_row =
            scales + static_cast<size_t>(n / 128) * weight_k_blocks;
        float sum = 0.0f;
        for (int block = 0; block < weight_k_blocks; ++block) {
            const int block_k = block * 128;
            const int block_count = std::min(128, K - block_k);
            const uint8_t maximum =
                fp8_block_maximum(weight_row + block_k, block_count);
            const auto& table = tables.values[maximum];
            const float weight_scale =
                tables.scales[maximum] * decode_e8m0(scale_row[block]);
            const int group_end =
                std::min(activation_groups, block * 4 + 4);
            for (int group = block * 4; group < group_end; ++group) {
                const int k = group * MATMUL_Q8_BLOCK;
                map_fp8_q8_block32(
                    weight_row + k, quantized_weight, table);
                const int32_t dot = q8_dot32(
                    quantized_activation + k, quantized_weight);
                sum += static_cast<float>(dot) *
                       activation_scales[group] * weight_scale;
            }
        }
        output[n] = sum;
    }
}

void matmul_fp8_q8_gemm_range(
    const int8_t* quantized_activation, const float* activation_scales,
    int activation_stride, int activation_groups,
    const uint8_t* weights, const uint8_t* scales, float* output,
    int M, int N, int K, int ldc, int n_begin, int n_end) {
    (void)N;
    const int weight_k_blocks = (K + 127) / 128;
    const auto& tables = fp8_q8_tables();
    alignas(16) int8_t quantized_weight[MATMUL_Q8_BLOCK];
    std::vector<float> sums(static_cast<size_t>(M));
    for (int n = n_begin; n < n_end; ++n) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        const uint8_t* weight_row =
            weights + static_cast<size_t>(n) * K;
        const uint8_t* scale_row =
            scales + static_cast<size_t>(n / 128) * weight_k_blocks;
        for (int block = 0; block < weight_k_blocks; ++block) {
            const int block_k = block * 128;
            const int block_count = std::min(128, K - block_k);
            const uint8_t maximum =
                fp8_block_maximum(weight_row + block_k, block_count);
            const auto& table = tables.values[maximum];
            const float weight_scale =
                tables.scales[maximum] * decode_e8m0(scale_row[block]);
            const int group_end =
                std::min(activation_groups, block * 4 + 4);
            for (int group = block * 4; group < group_end; ++group) {
                const int k = group * MATMUL_Q8_BLOCK;
                map_fp8_q8_block32(
                    weight_row + k, quantized_weight, table);
                for (int m = 0; m < M; ++m) {
                    const int32_t dot = q8_dot32(
                        quantized_activation +
                            static_cast<size_t>(m) * activation_stride + k,
                        quantized_weight);
                    sums[m] +=
                        static_cast<float>(dot) * weight_scale *
                        activation_scales[
                            static_cast<size_t>(m) * activation_groups +
                            group];
                }
            }
        }
        for (int m = 0; m < M; ++m)
            output[static_cast<size_t>(m) * ldc + n] = sums[m];
    }
}

} // namespace

size_t pack_fp8_e4m3_q8dot_bytes(int N, int K) {
    if (N <= 0 || K <= 0 || K % MATMUL_Q8_BLOCK != 0)
        return 0;
    const int padded_n = ((N + 7) / 8) * 8;
    return static_cast<size_t>(padded_n) * K;
}

bool pack_fp8_e4m3_q8dot(const uint8_t* source,
                         const uint8_t* e8m0_scales,
                         int N, int K,
                         int8_t* packed,
                         float* q8_scales) {
    const size_t packed_bytes = pack_fp8_e4m3_q8dot_bytes(N, K);
    if (!source || !e8m0_scales || !packed || !q8_scales ||
        packed_bytes == 0) {
        return false;
    }
    std::memset(packed, 0, packed_bytes);
    const int groups = K / MATMUL_Q8_BLOCK;
    const int k_blocks = (K + 127) / 128;
    const auto& tables = fp8_q8_tables();
    for (int n = 0; n < N; ++n) {
        const uint8_t* row = source + static_cast<size_t>(n) * K;
        for (int block = 0; block < k_blocks; ++block) {
            const int block_k = block * 128;
            const int block_count = std::min(128, K - block_k);
            const uint8_t maximum =
                fp8_block_maximum(row + block_k, block_count);
            const auto& table = tables.values[maximum];
            const float scale =
                tables.scales[maximum] *
                decode_e8m0(
                    e8m0_scales[
                        static_cast<size_t>(n / 128) * k_blocks + block]);
            const int group_end = std::min(groups, block * 4 + 4);
            for (int group = block * 4; group < group_end; ++group) {
                q8_scales[static_cast<size_t>(n) * groups + group] = scale;
                int8_t* destination =
                    packed +
                    (static_cast<size_t>(n / 8) * groups + group) *
                        8 * MATMUL_Q8_BLOCK +
                    static_cast<size_t>(n % 8) * MATMUL_Q8_BLOCK;
                const uint8_t* group_source =
                    row + group * MATMUL_Q8_BLOCK;
                for (int i = 0; i < MATMUL_Q8_BLOCK; ++i)
                    destination[i] = table[group_source[i]];
            }
        }
    }
    return true;
}

float decode_e8m0(uint8_t value) {
    if (value == 0xff)
        return std::numeric_limits<float>::quiet_NaN();
    // E8M0 stores an unbiased power of two. For 1..254 its byte is exactly
    // the IEEE-754 FP32 exponent field, so constructing the bits avoids an
    // ldexp call in the innermost MXFP4 loop. Code zero denotes 2^-127, which
    // is the FP32 subnormal with mantissa bit 22 set.
    const uint32_t bits =
        value == 0 ? uint32_t{1} << 22
                   : static_cast<uint32_t>(value) << 23;
    float decoded = 0.0f;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

float decode_fp8_e4m3fn(uint8_t value) {
    const int sign = (value & 0x80) ? -1 : 1;
    const int exponent = (value >> 3) & 0x0f;
    const int mantissa = value & 0x07;
    if (exponent == 0x0f && mantissa == 0x07)
        return std::numeric_limits<float>::quiet_NaN();
    if (exponent == 0)
        return sign * std::ldexp(static_cast<float>(mantissa), -9);
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                             exponent - 7);
}

uint8_t encode_fp8_e4m3fn(float value) {
    if (std::isnan(value))
        return 0x7f;
    const uint8_t sign = std::signbit(value) ? 0x80 : 0;
    const float magnitude = std::min(std::fabs(value), 448.0f);
    if (magnitude < (1.0f / 64.0f)) {
        // E4M3 subnormals have a fixed 2^-9 quantum. nearbyint implements
        // the required round-to-nearest-even rule.
        const int code = std::clamp(
            static_cast<int>(std::nearbyint(magnitude * 512.0f)), 0, 8);
        return static_cast<uint8_t>(sign | code);
    }

    uint32_t bits = 0;
    std::memcpy(&bits, &magnitude, sizeof(bits));
    int exponent = static_cast<int>((bits >> 23) & 0xff) - 127;
    // Round the FP32 significand from 23 to 3 fraction bits directly. This is
    // exactly round-to-nearest-even, but avoids ldexp + nearbyint for every
    // activation element entering an FP8/MXFP4 matmul.
    const uint32_t mantissa = bits & 0x7fffffu;
    int significand = 8 + static_cast<int>(mantissa >> 20);
    const uint32_t remainder = mantissa & 0xfffffu;
    constexpr uint32_t halfway = 0x80000u;
    if (remainder > halfway ||
        (remainder == halfway && (significand & 1) != 0)) {
        ++significand;
    }
    if (significand == 16) {
        ++exponent;
        significand = 8;
    }
    const int code = std::clamp((exponent + 6) * 8 + significand, 0, 126);
    return static_cast<uint8_t>(sign | code);
}

float decode_mxfp4_e2m1(uint8_t nibble) {
    return static_cast<float>(
               mxfp4_integer_coefficient(nibble)) *
           0.5f;
}

void kernel_matmul_mxfp4_reference(const Tensor& A, const Tensor& B, Tensor& C,
                                   ThreadPool* thread_pool) {
    const int M = static_cast<int>(A.shape[1]);
    const int K = static_cast<int>(A.shape[0]);
    const int N = static_cast<int>(B.shape[0]);
    if (A.prec != Precision::FP32 || B.prec != Precision::MXFP4 ||
        C.prec != Precision::FP32 || B.shape[1] != K ||
        K <= 0 || K % kMxBlock != 0 || !A.data || !B.data || !C.data ||
        !B.e8m0_scales) {
        return;
    }
    const int lda =
        static_cast<int>(A.stride[1] / sizeof(float));
    const int ldc =
        static_cast<int>(C.stride[1] / sizeof(float));
    auto run = [&](int m_begin, int m_end, int n_begin, int n_end) {
        matmul_mxfp4_reference_range(
            A.ptr<float>(), static_cast<const uint8_t*>(B.data),
            B.e8m0_scales, C.ptr<float>(), M, N, K, lda, ldc,
            m_begin, m_end, n_begin, n_end);
    };
    if (thread_pool && thread_pool->num_threads() > 1 && N > M * 8) {
        thread_pool->parallel_for(
            0, N,
            balanced_parallel_grain(
                N, thread_pool->num_threads(), 1),
            [&](int, int begin, int end) { run(0, M, begin, end); });
    } else if (thread_pool && thread_pool->num_threads() > 1 && M > 1) {
        thread_pool->parallel_for(
            0, M, 1,
            [&](int, int begin, int end) { run(begin, end, 0, N); });
    } else {
        run(0, M, 0, N);
    }
}

bool kernel_matmul_fp8_weight_f32_activation(
    const Tensor& A, const Tensor& B, Tensor& C, ThreadPool* thread_pool) {
    const int M = static_cast<int>(A.shape[1]);
    const int K = static_cast<int>(A.shape[0]);
    const int N = static_cast<int>(B.shape[0]);
    if (A.prec != Precision::FP32 || B.prec != Precision::FP8_E4M3 ||
        C.prec != Precision::FP32 || !A.data || !B.data || !C.data ||
        !B.e8m0_scales || !B.is_fp8_block128 || B.shape[1] != K ||
        C.shape[0] != N || C.shape[1] != M) {
        return false;
    }
    const int lda =
        static_cast<int>(A.stride[1] / sizeof(float));
    const int ldc =
        static_cast<int>(C.stride[1] / sizeof(float));
    auto run = [&](int begin, int end) {
        const auto* weights = static_cast<const uint8_t*>(B.data);
        const int k_blocks = (K + 127) / 128;
        for (int n = begin; n < end; ++n) {
            const uint8_t* weight_row =
                weights + static_cast<size_t>(n) * K;
            const uint8_t* scale_row =
                B.e8m0_scales +
                static_cast<size_t>(n / 128) * k_blocks;
            for (int m = 0; m < M; ++m) {
                const float* activation =
                    A.ptr<float>() + static_cast<size_t>(m) * lda;
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    // wo_a is stored as FP8 in the checkpoint but the
                    // official loader materializes this particular weight
                    // as BF16 before the ordinary grouped einsum.
                    const float weight_value = mollm_round_to_bf16(
                        decode_fp8_e4m3fn(weight_row[k]) *
                        decode_e8m0(scale_row[k / 128]));
                    sum += activation[k] * weight_value;
                }
                C.ptr<float>()[static_cast<size_t>(m) * ldc + n] = sum;
            }
        }
    };
    const int threads = thread_pool ? thread_pool->num_threads() : 1;
    if (thread_pool && threads > 1 && N > 64) {
        thread_pool->parallel_for(
            0, N, balanced_parallel_grain(N, threads, 64),
            [&](int, int begin, int end) { run(begin, end); });
    } else {
        run(0, N);
    }
    return true;
}

void quantize_mxfp4_activation_dual(
    const float* activation, int K, Mxfp4ActivationScratch& scratch) {
    quantize_a_fp8_q8_even_odd(
        activation, K, scratch.primary.qA_even,
        scratch.primary.qA_odd, scratch.primary.a_scales,
        &scratch.fp8_dequant);
    scratch.residual.resize(static_cast<size_t>(K));
    const int groups = K / kMxBlock;
    for (int group = 0; group < groups; ++group) {
        const size_t packed_offset = static_cast<size_t>(group) * 16;
        const int value_offset = group * kMxBlock;
        for (int i = 0; i < 16; ++i) {
            scratch.residual[value_offset + i * 2] =
                scratch.fp8_dequant[value_offset + i * 2] -
                static_cast<float>(
                    scratch.primary.qA_even[packed_offset + i]) *
                    scratch.primary.a_scales[group];
            scratch.residual[value_offset + i * 2 + 1] =
                scratch.fp8_dequant[value_offset + i * 2 + 1] -
                static_cast<float>(
                    scratch.primary.qA_odd[packed_offset + i]) *
                    scratch.primary.a_scales[group];
        }
    }
    quantize_a_q8_blocks_even_odd(
        scratch.residual.data(), K, scratch.residual_even,
        scratch.residual_odd, scratch.residual_scales);
}

bool kernel_matmul_mxfp4_gemv_batch(
    const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& weights,
    std::vector<Tensor>& outputs,
    ThreadPool* thread_pool) {
    const size_t batch = inputs.size();
    if (batch == 0 || weights.size() != batch ||
        outputs.size() != batch || !thread_pool ||
        g_mollm_force_fp32_acc) {
        return false;
    }
    const int K = static_cast<int>(inputs[0].shape[0]);
    const int N = static_cast<int>(weights[0].shape[0]);
    if (K <= 0 || N <= 0 || K % kMxBlock != 0)
        return false;

    struct BatchScratch {
        std::vector<Mxfp4ActivationScratch> activations;
        std::vector<size_t> input_indices;
        std::vector<const uint8_t*> weights;
        std::vector<const uint8_t*> scales;
        std::vector<float*> outputs;
    };
    static thread_local BatchScratch scratch;
    scratch.activations.resize(batch);
    scratch.input_indices.resize(batch);
    scratch.weights.resize(batch);
    scratch.scales.resize(batch);
    scratch.outputs.resize(batch);

    for (size_t i = 0; i < batch; ++i) {
        const Tensor& input = inputs[i];
        const Tensor& weight = weights[i];
        Tensor& output = outputs[i];
        if (input.prec != Precision::FP32 || !input.data ||
            input.shape[0] != K || input.shape[1] != 1 ||
            weight.prec != Precision::MXFP4 || !weight.data ||
            !weight.e8m0_scales || weight.group_size != kMxBlock ||
            weight.shape[0] != N || weight.shape[1] != K ||
            output.prec != Precision::FP32 || !output.data ||
            output.shape[0] != N || output.shape[1] != 1) {
            return false;
        }
        size_t input_index = i;
        for (size_t previous = 0; previous < i; ++previous) {
            if (inputs[previous].data == input.data &&
                inputs[previous].stride[1] == input.stride[1]) {
                input_index = scratch.input_indices[previous];
                break;
            }
        }
        scratch.input_indices[i] = input_index;
        if (input_index == i) {
            quantize_mxfp4_activation_dual(
                input.ptr<float>(), K, scratch.activations[i]);
        }
        scratch.weights[i] =
            static_cast<const uint8_t*>(weight.data);
        scratch.scales[i] = weight.e8m0_scales;
        scratch.outputs[i] = output.ptr<float>();
    }

    MatmulTimer timer;
    timer.set_shape(
        "mxfp4_q8_gemv_batch", static_cast<int>(batch), N, K,
        kMxBlock, K / kMxBlock, false, false,
        thread_pool->num_threads());
    BatchScratch* batch_data = &scratch;
    thread_pool->parallel_for_2d(
        static_cast<int>(batch), 1, N, 64,
        [&](int, int batch_begin, int batch_end,
            int begin, int end) {
            for (int index = batch_begin; index < batch_end; ++index) {
                const size_t i = static_cast<size_t>(index);
                const Mxfp4ActivationScratch& activation =
                    batch_data->activations[
                        batch_data->input_indices[i]];
                matmul_mxfp4_q8_gemv_range(
                    activation.primary.qA_even.data(),
                    activation.primary.qA_odd.data(),
                    activation.primary.a_scales.data(),
                    activation.residual_even.data(),
                    activation.residual_odd.data(),
                    activation.residual_scales.data(),
                    batch_data->weights[i], batch_data->scales[i],
                    batch_data->outputs[i], N, K, begin, end);
            }
        });
    return true;
}

void matmul_dispatch_mxfp4(const Tensor& A, const Tensor& B, Tensor& C,
                           ThreadPool* thread_pool, Activation act,
                           int act_n_begin, int act_n_len,
                           MatmulTimer& timer) {
    const int M = static_cast<int>(A.shape[1]);
    const int K = static_cast<int>(A.shape[0]);
    const int N = static_cast<int>(B.shape[0]);
    const int threads = thread_pool ? thread_pool->num_threads() : 1;
    if (A.prec != Precision::FP32 || C.prec != Precision::FP32 ||
        B.shape[1] != K || K <= 0 || K % kMxBlock != 0 ||
        !B.data || !B.e8m0_scales) {
        timer.set_shape("mxfp4_invalid", M, N, K, 32, K / 32,
                        false, false, threads);
        return;
    }

    if (g_mollm_force_fp32_acc) {
        timer.set_shape("mxfp4_fp8_exact", M, N, K, 32, K / 32,
                        false, false, threads);
        std::vector<float> quantized;
        quantize_a_fp8_dequant(
            A.ptr<float>(), M, K,
            static_cast<int>(A.stride[1] / sizeof(float)), quantized);
        Tensor exact_input = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1,
            quantized.data());
        kernel_matmul_mxfp4_reference(
            exact_input, B, C, thread_pool);
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation(
                C.ptr<float>(), M, N,
                static_cast<int>(C.stride[1] / sizeof(float)),
                0, M, act, act_n_begin, act_n_len);
        }
        return;
    }

    // The row-oriented GEMM below decodes each MXFP4 block once per token.
    // For the small routed batches produced by MoE prefill, reusing the
    // two-output GEMV kernel is substantially faster even after accounting
    // for one thread-pool dispatch per row.
    if (M > 1 && M <= 128) {
        timer.set_shape("mxfp4_q8_small_gemm", M, N, K, 32, K / 32,
                        false, false, threads);
        static thread_local std::vector<Mxfp4ActivationScratch> batch_scratch;
        batch_scratch.resize(static_cast<size_t>(M));
        const int lda =
            static_cast<int>(A.stride[1] / sizeof(float));
        const int ldc =
            static_cast<int>(C.stride[1] / sizeof(float));
        for (int m = 0; m < M; ++m) {
            Mxfp4ActivationScratch& row = batch_scratch[m];
            quantize_mxfp4_activation_dual(
                A.ptr<float>() + static_cast<size_t>(m) * lda, K, row);
        }
        const int chunk =
            balanced_parallel_grain(N, threads, 64, 2);
        for (int m = 0; m < M; ++m) {
            const Mxfp4ActivationScratch& row = batch_scratch[m];
            auto run = [&](int begin, int end) {
                matmul_mxfp4_q8_gemv_range(
                    row.primary.qA_even.data(),
                    row.primary.qA_odd.data(),
                    row.primary.a_scales.data(),
                    row.residual_even.data(),
                    row.residual_odd.data(),
                    row.residual_scales.data(),
                    static_cast<const uint8_t*>(B.data), B.e8m0_scales,
                    C.ptr<float>() + static_cast<size_t>(m) * ldc,
                    N, K, begin, end);
            };
            if (thread_pool && threads > 1 && N > 64) {
                thread_pool->parallel_for(
                    0, N, chunk,
                    [&](int, int begin, int end) { run(begin, end); });
            } else {
                run(0, N);
            }
        }
    } else if (M != 1) {
        timer.set_shape("mxfp4_q8_gemm", M, N, K, 32, K / 32,
                        false, false, threads);
        static thread_local std::vector<Mxfp4ActivationScratch> batch_scratch;
        batch_scratch.resize(static_cast<size_t>(M));
        const int lda =
            static_cast<int>(A.stride[1] / sizeof(float));
        for (int m = 0; m < M; ++m) {
            Mxfp4ActivationScratch& row = batch_scratch[m];
            quantize_mxfp4_activation_dual(
                A.ptr<float>() + static_cast<size_t>(m) * lda, K, row);
        }
        const std::vector<Mxfp4ActivationScratch>* quantized =
            &batch_scratch;
        const int ldc =
            static_cast<int>(C.stride[1] / sizeof(float));
        auto run = [&](int begin, int end) {
            matmul_mxfp4_q8_gemm_range(
                *quantized, static_cast<const uint8_t*>(B.data),
                B.e8m0_scales, C.ptr<float>(), M, N, K, ldc,
                begin, end);
        };
        if (thread_pool && threads > 1 && N > 64) {
            thread_pool->parallel_for(
                0, N, balanced_parallel_grain(N, threads, 64, 2),
                [&](int, int begin, int end) { run(begin, end); });
        } else {
            run(0, N);
        }
    } else {
        timer.set_shape("mxfp4_q8_gemv", M, N, K, 32, K / 32,
                        false, false, threads);
        static thread_local Mxfp4ActivationScratch scratch;
        quantize_mxfp4_activation_dual(A.ptr<float>(), K, scratch);
        // Resolve thread-local scratch on the caller. Referencing `scratch`
        // inside the worker lambda would select each worker's empty TLS copy.
        const int8_t* q_even = scratch.primary.qA_even.data();
        const int8_t* q_odd = scratch.primary.qA_odd.data();
        const float* a_scales = scratch.primary.a_scales.data();
        const int8_t* residual_even = scratch.residual_even.data();
        const int8_t* residual_odd = scratch.residual_odd.data();
        const float* residual_scales = scratch.residual_scales.data();
        auto run = [&](int begin, int end) {
            matmul_mxfp4_q8_gemv_range(
                q_even, q_odd, a_scales,
                residual_even, residual_odd, residual_scales,
                static_cast<const uint8_t*>(B.data), B.e8m0_scales,
                C.ptr<float>(), N, K, begin, end);
        };
        if (thread_pool && threads > 1 && N > 64) {
            thread_pool->parallel_for_2d(
                1, 1, N, 64,
                [&](int, int, int, int begin, int end) {
                    run(begin, end);
                });
        } else {
            run(0, N);
        }
    }
    if (act != Activation::NONE && act_n_len != 0) {
        matmul_apply_activation(
            C.ptr<float>(), M, N,
            static_cast<int>(C.stride[1] / sizeof(float)),
            0, M, act, act_n_begin, act_n_len);
    }
}

void matmul_dispatch_fp8_e4m3(const Tensor& A, const Tensor& B, Tensor& C,
                              ThreadPool* thread_pool, Activation act,
                              int act_n_begin, int act_n_len,
                              MatmulTimer& timer) {
    const int M = static_cast<int>(A.shape[1]);
    const int K = static_cast<int>(A.shape[0]);
    const int N = static_cast<int>(B.shape[0]);
    const int threads = thread_pool ? thread_pool->num_threads() : 1;
    if (A.prec != Precision::FP32 || C.prec != Precision::FP32 ||
        B.shape[1] != K || !B.data || !B.e8m0_scales ||
        !B.is_fp8_block128) {
        timer.set_shape("fp8_e4m3_invalid", M, N, K, 128,
                        (K + 127) / 128, false, false, threads);
        return;
    }
    if (g_mollm_force_fp32_acc) {
        timer.set_shape("fp8_fp8_exact", M, N, K, 128,
                        (K + 127) / 128, false, false, threads);
        std::vector<float> quantized;
        quantize_a_fp8_dequant(
            A.ptr<float>(), M, K,
            static_cast<int>(A.stride[1] / sizeof(float)), quantized);
        auto run = [&](int begin, int end) {
            matmul_fp8_reference_range(
                quantized.data(),
                static_cast<const uint8_t*>(B.data), B.e8m0_scales,
                C.ptr<float>(), M, N, K, K,
                static_cast<int>(C.stride[1] / sizeof(float)),
                0, M, begin, end);
        };
        if (thread_pool && threads > 1 && N > 64) {
            thread_pool->parallel_for(
                0, N, balanced_parallel_grain(N, threads, 64),
                [&](int, int begin, int end) { run(begin, end); });
        } else {
            run(0, N);
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation(
                C.ptr<float>(), M, N,
                static_cast<int>(C.stride[1] / sizeof(float)),
                0, M, act, act_n_begin, act_n_len);
        }
        return;
    }
    if (B.q8_repack_data && B.fp8_q8_scales &&
        K % MATMUL_Q8_BLOCK == 0) {
        Tensor q8_weight = B;
        q8_weight.prec = Precision::INT8;
        q8_weight.scales = B.fp8_q8_scales;
        q8_weight.group_size = MATMUL_Q8_BLOCK;
        q8_weight.groups_per_row =
            static_cast<uint32_t>(K / MATMUL_Q8_BLOCK);
        q8_weight.num_groups =
            static_cast<uint32_t>(
                static_cast<size_t>(N) * q8_weight.groups_per_row);
        matmul_dispatch_int8(
            A, q8_weight, C, thread_pool, act, act_n_begin, act_n_len,
            timer, true);
        return;
    }
    const bool use_q8_dot = K >= MATMUL_Q8_BLOCK &&
                            K % MATMUL_Q8_BLOCK == 0;
    if (use_q8_dot) {
        static thread_local std::vector<int8_t> quantized_activation;
        static thread_local std::vector<float> activation_scales;
        const int lda =
            static_cast<int>(A.stride[1] / sizeof(float));
        quantize_a_fp8_q8_blocks(
            A.ptr<float>(), M, K, lda, K,
            quantized_activation, activation_scales);
        const int activation_groups = K / MATMUL_Q8_BLOCK;
        const int ldc =
            static_cast<int>(C.stride[1] / sizeof(float));
        const int8_t* quantized = quantized_activation.data();
        const float* quant_scales = activation_scales.data();
        auto run = [&](int begin, int end) {
            if (M == 1) {
                matmul_fp8_q8_gemv_range(
                    quantized, quant_scales,
                    static_cast<const uint8_t*>(B.data), B.e8m0_scales,
                    C.ptr<float>(), N, K, begin, end);
            } else {
                matmul_fp8_q8_gemm_range(
                    quantized, quant_scales, K, activation_groups,
                    static_cast<const uint8_t*>(B.data), B.e8m0_scales,
                    C.ptr<float>(), M, N, K, ldc, begin, end);
            }
        };
        timer.set_shape(
            M == 1 ? "fp8_q8_gemv" : "fp8_q8_gemm",
            M, N, K, 128, (K + 127) / 128, false, false, threads);
        if (thread_pool && threads > 1 && N > 64) {
            thread_pool->parallel_for(
                0, N, balanced_parallel_grain(N, threads, 64),
                [&](int, int begin, int end) { run(begin, end); });
        } else {
            run(0, N);
        }
    } else if (M == 1) {
        timer.set_shape("fp8_e4m3_gemv", M, N, K, 128,
                        (K + 127) / 128, false, false, threads);
        auto run = [&](int begin, int end) {
            matmul_fp8_gemv_range(
                A.ptr<float>(), static_cast<const uint8_t*>(B.data),
                B.e8m0_scales, C.ptr<float>(), N, K, begin, end);
        };
        if (thread_pool && threads > 1 && N > 64) {
            thread_pool->parallel_for(
                0, N, balanced_parallel_grain(N, threads, 64),
                [&](int, int begin, int end) { run(begin, end); });
        } else {
            run(0, N);
        }
    } else {
        timer.set_shape("fp8_e4m3_gemm", M, N, K, 128,
                        (K + 127) / 128, false, false, threads);
        const int lda =
            static_cast<int>(A.stride[1] / sizeof(float));
        const int ldc =
            static_cast<int>(C.stride[1] / sizeof(float));
        auto run = [&](int begin, int end) {
            matmul_fp8_gemm_range(
                A.ptr<float>(), lda,
                static_cast<const uint8_t*>(B.data), B.e8m0_scales,
                C.ptr<float>(), M, N, K, ldc, begin, end);
        };
        if (thread_pool && threads > 1 && N > 64) {
            thread_pool->parallel_for(
                0, N, balanced_parallel_grain(N, threads, 64),
                [&](int, int begin, int end) { run(begin, end); });
        } else {
            run(0, N);
        }
    }
    if (act != Activation::NONE && act_n_len != 0) {
        matmul_apply_activation(
            C.ptr<float>(), M, N,
            static_cast<int>(C.stride[1] / sizeof(float)),
            0, M, act, act_n_begin, act_n_len);
    }
}
