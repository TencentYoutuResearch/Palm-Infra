#include "kernels/matmul_internal.h"

#include "runtime/threading.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#if HAS_NEON
#include <arm_neon.h>
#endif

namespace {

const std::array<float, 256> kFp8E4m3fnDecodeTable = [] {
    std::array<float, 256> values{};
    for (int code = 0; code < 256; ++code)
        values[code] = decode_fp8_e4m3fn(static_cast<uint8_t>(code));
    return values;
}();

inline const uint8_t* nvfp4_block_ptr(
    const Tensor& weight, int row, int group,
    int groups_per_row, int bytes_per_row) {
    if (weight.nvfp4_q8_pair_data) {
        return weight.nvfp4_q8_pair_data +
            (static_cast<size_t>(row / 4) * groups_per_row + group) * 32 +
            static_cast<size_t>(row % 4) * 8;
    }
    return static_cast<const uint8_t*>(weight.data) +
        static_cast<size_t>(row) * bytes_per_row + group * 8;
}

#if HAS_NEON && defined(__aarch64__)
struct Nvfp4CoefficientsF16 {
    float16x8_t low;
    float16x8_t high;
};

struct Nvfp4Coefficients16 {
    float32x4_t c0;
    float32x4_t c1;
    float32x4_t c2;
    float32x4_t c3;
};

inline Nvfp4CoefficientsF16 decode_nvfp4_coefficients_f16(
    const uint8_t* packed) {
    // E2M1 coefficients multiplied by two are exact FP16 integers. Their
    // little-endian FP16 encodings all have a zero low byte, so one byte-table
    // lookup supplies every high byte and avoids int8 -> int16 -> int32 ->
    // FP32 conversion chains.
    const uint8x16_t fp16_high_byte_table = {
        0x00, 0x3c, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a,
        0x00, 0xbc, 0xc0, 0xc2, 0xc4, 0xc6, 0xc8, 0xca,
    };
    const uint8x8_t bytes = vld1_u8(packed);
    const uint8x16_t bytes16 = vcombine_u8(bytes, bytes);
    const uint8x16_t even =
        vandq_u8(bytes16, vdupq_n_u8(0x0f));
    const uint8x16_t odd = vshrq_n_u8(bytes16, 4);
    // Only the low halves are real input. Interleave low/high nibbles back to
    // checkpoint order: w0,w1,...,w14,w15.
    const uint8x16_t ordered = vzip1q_u8(even, odd);
    const uint8x16_t high_bytes =
        vqtbl1q_u8(fp16_high_byte_table, ordered);
    const uint8x16_t zeros = vdupq_n_u8(0);
    return {
        vreinterpretq_f16_u8(vzip1q_u8(zeros, high_bytes)),
        vreinterpretq_f16_u8(vzip2q_u8(zeros, high_bytes)),
    };
}

inline int8x16_t decode_nvfp4_coefficients_q8(const uint8_t* packed) {
    // E2M1 coefficients multiplied by two are small exact integers. Keeping
    // that factor in the coefficient lets SDOT consume all 16 values at once;
    // the common 0.5 factor is restored together with the block scale.
    const int8x16_t coefficient_table = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    const uint8x8_t bytes = vld1_u8(packed);
    const uint8x16_t bytes16 = vcombine_u8(bytes, bytes);
    const uint8x16_t low =
        vandq_u8(bytes16, vdupq_n_u8(0x0f));
    const uint8x16_t high = vshrq_n_u8(bytes16, 4);
    return vqtbl1q_s8(coefficient_table, vzip1q_u8(low, high));
}

struct Nvfp4CoefficientPairQ8 {
    int8x16_t first;
    int8x16_t second;
};

inline Nvfp4CoefficientPairQ8 decode_nvfp4_coefficient_pair_q8(
    const uint8_t* packed) {
    const int8x16_t coefficient_table = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    const uint8x16_t bytes = vld1q_u8(packed);
    const uint8x16_t low =
        vandq_u8(bytes, vdupq_n_u8(0x0f));
    const uint8x16_t high = vshrq_n_u8(bytes, 4);
    return {
        vqtbl1q_s8(coefficient_table, vzip1q_u8(low, high)),
        vqtbl1q_s8(coefficient_table, vzip2q_u8(low, high)),
    };
}

inline float nvfp4_dot16_q8(const int8_t* activation,
                            const uint8_t* packed) {
    const int32x4_t partial = vdotq_s32(
        vdupq_n_s32(0), decode_nvfp4_coefficients_q8(packed),
        vld1q_s8(activation));
    return static_cast<float>(vaddvq_s32(partial));
}

inline Nvfp4Coefficients16 decode_nvfp4_coefficients16(
    const uint8_t* packed) {
    const Nvfp4CoefficientsF16 coefficients =
        decode_nvfp4_coefficients_f16(packed);
    return {
        vcvt_f32_f16(vget_low_f16(coefficients.low)),
        vcvt_high_f32_f16(coefficients.low),
        vcvt_f32_f16(vget_low_f16(coefficients.high)),
        vcvt_high_f32_f16(coefficients.high),
    };
}

inline float32x4_t nvfp4_partial16_f16(
    float16x8_t activation_low, float16x8_t activation_high,
    const Nvfp4CoefficientsF16& coefficients) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    sum = vfmlalq_low_f16(sum, activation_low, coefficients.low);
    sum = vfmlalq_high_f16(sum, activation_low, coefficients.low);
    sum = vfmlalq_low_f16(sum, activation_high, coefficients.high);
    return vfmlalq_high_f16(sum, activation_high, coefficients.high);
}

inline float32x4_t accumulate_nvfp4_four_rows_f16(
    float32x4_t sums,
    float16x8_t activation_low, float16x8_t activation_high,
    const uint8_t* row0, int bytes_per_row,
    const uint8_t* scale0, int groups, int group) {
    const size_t offset = static_cast<size_t>(group) * 8;
    const float32x4_t dots = {
        vaddvq_f32(nvfp4_partial16_f16(
            activation_low, activation_high,
            decode_nvfp4_coefficients_f16(row0 + offset))),
        vaddvq_f32(nvfp4_partial16_f16(
            activation_low, activation_high,
            decode_nvfp4_coefficients_f16(
                row0 + bytes_per_row + offset))),
        vaddvq_f32(nvfp4_partial16_f16(
            activation_low, activation_high,
            decode_nvfp4_coefficients_f16(
                row0 + 2 * bytes_per_row + offset))),
        vaddvq_f32(nvfp4_partial16_f16(
            activation_low, activation_high,
            decode_nvfp4_coefficients_f16(
                row0 + 3 * bytes_per_row + offset))),
    };
    const float32x4_t decoded_scales = {
        kFp8E4m3fnDecodeTable[scale0[group]],
        kFp8E4m3fnDecodeTable[scale0[groups + group]],
        kFp8E4m3fnDecodeTable[scale0[2 * groups + group]],
        kFp8E4m3fnDecodeTable[scale0[3 * groups + group]],
    };
    return vfmaq_f32(
        sums, vmulq_n_f32(dots, 0.5f), decoded_scales);
}

void convert_fp32_to_fp16(const float* source, __fp16* destination,
                          int count) {
    int index = 0;
    for (; index + 7 < count; index += 8) {
        const float16x8_t converted = vcombine_f16(
            vcvt_f16_f32(vld1q_f32(source + index)),
            vcvt_f16_f32(vld1q_f32(source + index + 4)));
        vst1q_f16(destination + index, converted);
    }
    for (; index < count; ++index)
        destination[index] = static_cast<__fp16>(source[index]);
}

bool nvfp4_fp16_activation_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("MOLLM_NVFP4_FP16_ACT");
        return !value || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool nvfp4_q8_activation_enabled() {
    return true;
}

inline float32x4_t nvfp4_partial16_decoded(
    const float* activation, const Nvfp4Coefficients16& coefficients) {
    float32x4_t sum = vmulq_f32(
        vld1q_f32(activation), coefficients.c0);
    sum = vfmaq_f32(
        sum, vld1q_f32(activation + 4), coefficients.c1);
    sum = vfmaq_f32(
        sum, vld1q_f32(activation + 8), coefficients.c2);
    return vfmaq_f32(
        sum, vld1q_f32(activation + 12), coefficients.c3);
}

inline float32x4_t nvfp4_partial16(
    float32x4_t activation0, float32x4_t activation1,
    float32x4_t activation2, float32x4_t activation3,
    const uint8_t* packed) {
    const Nvfp4Coefficients16 coefficients =
        decode_nvfp4_coefficients16(packed);
    float32x4_t sum = vmulq_f32(activation0, coefficients.c0);
    sum = vfmaq_f32(sum, activation1, coefficients.c1);
    sum = vfmaq_f32(sum, activation2, coefficients.c2);
    return vfmaq_f32(sum, activation3, coefficients.c3);
}

inline float nvfp4_dot16(const float* activation,
                         const uint8_t* packed) {
    return 0.5f * vaddvq_f32(nvfp4_partial16(
        vld1q_f32(activation), vld1q_f32(activation + 4),
        vld1q_f32(activation + 8), vld1q_f32(activation + 12), packed));
}

inline float32x4_t reduce_four_dots(
    float32x4_t row0, float32x4_t row1,
    float32x4_t row2, float32x4_t row3) {
    return vpaddq_f32(
        vpaddq_f32(row0, row1), vpaddq_f32(row2, row3));
}

inline float32x2_t reduce_two_dots(
    float32x4_t row0, float32x4_t row1) {
    const float32x4_t pairs = vpaddq_f32(row0, row1);
    return vpadd_f32(vget_low_f32(pairs), vget_high_f32(pairs));
}

inline int32x4_t reduce_four_dots_s32(
    int32x4_t row0, int32x4_t row1,
    int32x4_t row2, int32x4_t row3) {
    // SDOT leaves four independent partial sums in each vector. Pairwise
    // reduction keeps all four output rows in SIMD lanes instead of issuing
    // four scalar ADDV reductions and rebuilding a vector afterwards.
    return vpaddq_s32(
        vpaddq_s32(row0, row1), vpaddq_s32(row2, row3));
}

inline float32x4_t accumulate_nvfp4_q8_pair_group(
    float32x4_t sums, int8x16_t q8, const uint8_t* block,
    const uint8_t* scale0, int groups, int group,
    float activation_scale) {
    const Nvfp4CoefficientPairQ8 pair01 =
        decode_nvfp4_coefficient_pair_q8(block);
    const Nvfp4CoefficientPairQ8 pair23 =
        decode_nvfp4_coefficient_pair_q8(block + 16);
    const int32x4_t zero = vdupq_n_s32(0);
    const float32x4_t dots = vcvtq_f32_s32(reduce_four_dots_s32(
        vdotq_s32(zero, pair01.first, q8),
        vdotq_s32(zero, pair01.second, q8),
        vdotq_s32(zero, pair23.first, q8),
        vdotq_s32(zero, pair23.second, q8)));
    const float32x4_t weight_scales = {
        kFp8E4m3fnDecodeTable[scale0[group]],
        kFp8E4m3fnDecodeTable[scale0[groups + group]],
        kFp8E4m3fnDecodeTable[scale0[2 * groups + group]],
        kFp8E4m3fnDecodeTable[scale0[3 * groups + group]],
    };
    return vfmaq_n_f32(
        sums, vmulq_f32(dots, weight_scales), activation_scale);
}

inline float32x4_t accumulate_nvfp4_four_rows(
    float32x4_t sums,
    float32x4_t activation0, float32x4_t activation1,
    float32x4_t activation2, float32x4_t activation3,
    const uint8_t* row0, int bytes_per_row,
    const uint8_t* scale0, int groups, int group) {
    const size_t offset = static_cast<size_t>(group) * 8;
    const float32x4_t dots = vmulq_n_f32(
        reduce_four_dots(
            nvfp4_partial16(
                activation0, activation1, activation2, activation3,
                row0 + offset),
            nvfp4_partial16(
                activation0, activation1, activation2, activation3,
                row0 + bytes_per_row + offset),
            nvfp4_partial16(
                activation0, activation1, activation2, activation3,
                row0 + 2 * bytes_per_row + offset),
            nvfp4_partial16(
                activation0, activation1, activation2, activation3,
                row0 + 3 * bytes_per_row + offset)),
        0.5f);
    const float32x4_t decoded_scales = {
        kFp8E4m3fnDecodeTable[scale0[group]],
        kFp8E4m3fnDecodeTable[scale0[groups + group]],
        kFp8E4m3fnDecodeTable[scale0[2 * groups + group]],
        kFp8E4m3fnDecodeTable[scale0[3 * groups + group]]};
    return vfmaq_f32(sums, dots, decoded_scales);
}

#endif

void matmul_nvfp4_range(const Tensor& A, const Tensor& B, Tensor& C,
                        Activation act, int act_n_begin, int act_n_len,
                        int n_begin, int n_end) {
    const int M = static_cast<int>(A.shape[1]);
    const int K = static_cast<int>(A.shape[0]);
    const int groups_per_row = K / 16;
    const int lda = static_cast<int>(A.stride[1] / sizeof(float));
    const int ldc = static_cast<int>(C.stride[1] / sizeof(float));
    const float* a = A.ptr<float>();
    const auto* packed = static_cast<const uint8_t*>(B.data);
    float* c = C.ptr<float>();

    for (int n = n_begin; n < n_end; ++n) {
        const uint8_t* scales =
            B.nvfp4_scales + static_cast<size_t>(n) * groups_per_row;
        for (int m = 0; m < M; ++m) {
            const float* x = a + static_cast<size_t>(m) * lda;
            float sum = 0.0f;
            for (int group = 0; group < groups_per_row; ++group) {
#if HAS_NEON && defined(__aarch64__)
                const float block = nvfp4_dot16(
                    x + group * 16,
                    nvfp4_block_ptr(
                        B, n, group, groups_per_row, K / 2));
                sum += block * kFp8E4m3fnDecodeTable[scales[group]];
#else
                float block = 0.0f;
                const int k_begin = group * 16;
                for (int k = k_begin; k < k_begin + 16; ++k) {
                const uint8_t* block_data = nvfp4_block_ptr(
                    B, n, group, groups_per_row, K / 2);
                const uint8_t byte = block_data[(k - k_begin) >> 1];
                const uint8_t nibble =
                    (k & 1) ? static_cast<uint8_t>(byte >> 4)
                            : static_cast<uint8_t>(byte & 0x0f);
                    block += x[k] * decode_mxfp4_e2m1(nibble);
                }
                sum += block * kFp8E4m3fnDecodeTable[scales[group]];
#endif
            }
            sum *= B.nvfp4_row_scales[n];
            if (act != Activation::NONE && n >= act_n_begin &&
                n < act_n_begin + act_n_len) {
                sum = apply_activation_scalar(sum, act);
            }
            c[static_cast<size_t>(m) * ldc + n] = sum;
        }
    }
}

#if HAS_NEON && defined(__aarch64__)
void matmul_nvfp4_small_m_range(
    const Tensor& A, const Tensor& B, Tensor& C,
    Activation act, int act_n_begin, int act_n_len,
    int n_begin, int n_end) {
    constexpr int kMaxM = 32;
    const int M = static_cast<int>(A.shape[1]);
    const int K = static_cast<int>(A.shape[0]);
    const int groups = K / 16;
    const int bytes_per_row = K / 2;
    const int lda = static_cast<int>(A.stride[1] / sizeof(float));
    const int ldc = static_cast<int>(C.stride[1] / sizeof(float));
    const int blocks4 = M / 4;
    const int remainder = M - blocks4 * 4;
    const bool has_pair = remainder >= 2;
    const bool has_single = (remainder & 1) != 0;
    const int pair_m = blocks4 * 4;
    const int single_m = pair_m + (has_pair ? 2 : 0);
    const float* activation = A.ptr<float>();
    const auto* packed = static_cast<const uint8_t*>(B.data);
    float* output = C.ptr<float>();

    for (int n = n_begin; n < n_end; ++n) {
        std::array<float32x4_t, kMaxM / 4> sums4;
        for (int block = 0; block < blocks4; ++block)
            sums4[block] = vdupq_n_f32(0.0f);
        float32x2_t sum2 = vdup_n_f32(0.0f);
        float sum1 = 0.0f;
        const uint8_t* scales =
            B.nvfp4_scales + static_cast<size_t>(n) * groups;

        for (int group = 0; group < groups; ++group) {
            const Nvfp4Coefficients16 coefficients =
                decode_nvfp4_coefficients16(
                    nvfp4_block_ptr(
                        B, n, group, groups, bytes_per_row));
            const float scale =
                0.5f * kFp8E4m3fnDecodeTable[scales[group]];
            const int k = group * 16;
            for (int block = 0; block < blocks4; ++block) {
                const int m = block * 4;
                const float32x4_t dots = reduce_four_dots(
                    nvfp4_partial16_decoded(
                        activation + static_cast<size_t>(m) * lda + k,
                        coefficients),
                    nvfp4_partial16_decoded(
                        activation + static_cast<size_t>(m + 1) * lda + k,
                        coefficients),
                    nvfp4_partial16_decoded(
                        activation + static_cast<size_t>(m + 2) * lda + k,
                        coefficients),
                    nvfp4_partial16_decoded(
                        activation + static_cast<size_t>(m + 3) * lda + k,
                        coefficients));
                sums4[block] = vfmaq_n_f32(sums4[block], dots, scale);
            }
            if (has_pair) {
                const float32x2_t dots = reduce_two_dots(
                    nvfp4_partial16_decoded(
                        activation + static_cast<size_t>(pair_m) * lda + k,
                        coefficients),
                    nvfp4_partial16_decoded(
                        activation + static_cast<size_t>(pair_m + 1) * lda + k,
                        coefficients));
                sum2 = vfma_n_f32(sum2, dots, scale);
            }
            if (has_single) {
                sum1 += scale * vaddvq_f32(nvfp4_partial16_decoded(
                    activation + static_cast<size_t>(single_m) * lda + k,
                    coefficients));
            }
        }

        std::array<float, kMaxM> sums;
        for (int block = 0; block < blocks4; ++block)
            vst1q_f32(sums.data() + block * 4, sums4[block]);
        if (has_pair)
            vst1_f32(sums.data() + pair_m, sum2);
        if (has_single)
            sums[single_m] = sum1;
        const float row_scale = B.nvfp4_row_scales[n];
        for (int m = 0; m < M; ++m) {
            float value = sums[m] * row_scale;
            if (act != Activation::NONE && n >= act_n_begin &&
                n < act_n_begin + act_n_len) {
                value = apply_activation_scalar(value, act);
            }
            output[static_cast<size_t>(m) * ldc + n] = value;
        }
    }
}

void matmul_nvfp4_gemv_four_range(
    const Tensor& A, const Tensor& B, Tensor& C,
    Activation act, int act_n_begin, int act_n_len,
    int n_begin, int n_end) {
    const int K = static_cast<int>(A.shape[0]);
    const int groups = K / 16;
    const int bytes_per_row = K / 2;
    const float* activation = A.ptr<float>();
    const auto* packed = static_cast<const uint8_t*>(B.data);
    float* output = C.ptr<float>();
    int n = n_begin;
    for (; n + 3 < n_end; n += 4) {
        const uint8_t* row0 = packed + static_cast<size_t>(n) * bytes_per_row;
        const uint8_t* scale0 =
            B.nvfp4_scales + static_cast<size_t>(n) * groups;
        float32x4_t sums = vdupq_n_f32(0.0f);
        for (int group = 0; group < groups; ++group) {
            const float* x = activation + group * 16;
            const float32x4_t a0 = vld1q_f32(x);
            const float32x4_t a1 = vld1q_f32(x + 4);
            const float32x4_t a2 = vld1q_f32(x + 8);
            const float32x4_t a3 = vld1q_f32(x + 12);
            sums = accumulate_nvfp4_four_rows(
                sums, a0, a1, a2, a3, row0, bytes_per_row,
                scale0, groups, group);
        }
        sums = vmulq_f32(sums, vld1q_f32(B.nvfp4_row_scales + n));
        if (act == Activation::NONE || act_n_len <= 0 ||
            n + 3 < act_n_begin || n >= act_n_begin + act_n_len) {
            vst1q_f32(output + n, sums);
        } else {
            float values[4];
            vst1q_f32(values, sums);
            for (int lane = 0; lane < 4; ++lane) {
                const int column = n + lane;
                if (column >= act_n_begin &&
                    column < act_n_begin + act_n_len)
                    values[lane] = apply_activation_scalar(values[lane], act);
            }
            vst1q_f32(output + n, vld1q_f32(values));
        }
    }
    if (n < n_end) {
        matmul_nvfp4_range(
            A, B, C, act, act_n_begin, act_n_len, n, n_end);
    }
}

void matmul_nvfp4_gemv_four_f16_range(
    const __fp16* activation, const Tensor& B, Tensor& C,
    Activation act, int act_n_begin, int act_n_len,
    int n_begin, int n_end) {
    const int K = static_cast<int>(B.shape[1]);
    const int groups = K / 16;
    const int bytes_per_row = K / 2;
    const auto* packed = static_cast<const uint8_t*>(B.data);
    float* output = C.ptr<float>();
    int n = n_begin;
    for (; n + 3 < n_end; n += 4) {
        const uint8_t* row0 = packed + static_cast<size_t>(n) * bytes_per_row;
        const uint8_t* scale0 =
            B.nvfp4_scales + static_cast<size_t>(n) * groups;
        float32x4_t sums = vdupq_n_f32(0.0f);
        for (int group = 0; group < groups; ++group) {
            const __fp16* x = activation + group * 16;
            const float16x8_t activation_low = vld1q_f16(x);
            const float16x8_t activation_high = vld1q_f16(x + 8);
            sums = accumulate_nvfp4_four_rows_f16(
                sums, activation_low, activation_high, row0,
                bytes_per_row, scale0, groups, group);
        }
        sums = vmulq_f32(sums, vld1q_f32(B.nvfp4_row_scales + n));
        if (act == Activation::NONE || act_n_len <= 0 ||
            n + 3 < act_n_begin || n >= act_n_begin + act_n_len) {
            vst1q_f32(output + n, sums);
        } else {
            float values[4];
            vst1q_f32(values, sums);
            for (int lane = 0; lane < 4; ++lane) {
                const int column = n + lane;
                if (column >= act_n_begin &&
                    column < act_n_begin + act_n_len)
                    values[lane] = apply_activation_scalar(values[lane], act);
            }
            vst1q_f32(output + n, vld1q_f32(values));
        }
    }
    if (n < n_end) {
        // The scalar tail is rare for model shapes and retains the exact FP32
        // path. Model NVFP4 output dimensions are multiples of four.
        Tensor activation_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, K, 1, 1, 1, nullptr);
        std::vector<float> activation_fp32(static_cast<size_t>(K));
        for (int k = 0; k < K; ++k)
            activation_fp32[k] = static_cast<float>(activation[k]);
        activation_tensor.data = activation_fp32.data();
        matmul_nvfp4_range(
            activation_tensor, B, C, act, act_n_begin, act_n_len, n, n_end);
    }
}

void matmul_nvfp4_gemv_four_q8_range(
    const int8_t* activation, const float* activation_scales,
    const Tensor& B, Tensor& C, int n_begin, int n_end) {
    const int K = static_cast<int>(B.shape[1]);
    const int groups = K / 16;
    const int bytes_per_row = K / 2;
    const auto* packed = static_cast<const uint8_t*>(B.data);
    float* output = C.ptr<float>();
    int n = n_begin;
    for (; n + 3 < n_end; n += 4) {
        const uint8_t* row0 = packed + static_cast<size_t>(n) * bytes_per_row;
        const uint8_t* scale0 =
            B.nvfp4_scales + static_cast<size_t>(n) * groups;
        float32x4_t sums = vdupq_n_f32(0.0f);
        for (int group = 0; group < groups; ++group) {
            const int8_t* q8 = activation + group * 16;
            const float32x4_t dots = {
                nvfp4_dot16_q8(q8, row0 + group * 8),
                nvfp4_dot16_q8(
                    q8, row0 + bytes_per_row + group * 8),
                nvfp4_dot16_q8(
                    q8, row0 + 2 * bytes_per_row + group * 8),
                nvfp4_dot16_q8(
                    q8, row0 + 3 * bytes_per_row + group * 8),
            };
            const float32x4_t weight_scales = {
                kFp8E4m3fnDecodeTable[scale0[group]],
                kFp8E4m3fnDecodeTable[scale0[groups + group]],
                kFp8E4m3fnDecodeTable[scale0[2 * groups + group]],
                kFp8E4m3fnDecodeTable[scale0[3 * groups + group]],
            };
            const float scale =
                0.5f * activation_scales[group / 2];
            sums = vfmaq_n_f32(
                sums, vmulq_f32(dots, weight_scales), scale);
        }
        sums = vmulq_f32(
            sums, vld1q_f32(B.nvfp4_row_scales + n));
        vst1q_f32(output + n, sums);
    }
    for (; n < n_end; ++n) {
        const uint8_t* row =
            packed + static_cast<size_t>(n) * bytes_per_row;
        const uint8_t* scales =
            B.nvfp4_scales + static_cast<size_t>(n) * groups;
        float sum = 0.0f;
        for (int group = 0; group < groups; ++group) {
            sum += nvfp4_dot16_q8(
                       activation + group * 16, row + group * 8) *
                   (0.5f * activation_scales[group / 2]) *
                   kFp8E4m3fnDecodeTable[scales[group]];
        }
        output[n] = sum * B.nvfp4_row_scales[n];
    }
}

void matmul_nvfp4_gemv_four_q8_pair_range(
    const int8_t* activation, const float* activation_scales,
    const Tensor& B, Tensor& C, int n_begin, int n_end) {
    const int K = static_cast<int>(B.shape[1]);
    const int groups = K / 16;
    const uint8_t* pair_packed = B.nvfp4_q8_pair_data;
    float* output = C.ptr<float>();
    int n = n_begin;
    for (; n + 3 < n_end; n += 4) {
        const uint8_t* tile = pair_packed +
            static_cast<size_t>(n / 4) * groups * 32;
        const uint8_t* scale0 =
            B.nvfp4_scales + static_cast<size_t>(n) * groups;
        float32x4_t sums = vdupq_n_f32(0.0f);
        int group = 0;
        for (; group + 1 < groups; group += 2) {
            const float scale = 0.5f * activation_scales[group / 2];
            sums = accumulate_nvfp4_q8_pair_group(
                sums, vld1q_s8(activation + group * 16),
                tile + static_cast<size_t>(group) * 32,
                scale0, groups, group, scale);
            sums = accumulate_nvfp4_q8_pair_group(
                sums, vld1q_s8(activation + (group + 1) * 16),
                tile + static_cast<size_t>(group + 1) * 32,
                scale0, groups, group + 1, scale);
        }
        if (group < groups) {
            sums = accumulate_nvfp4_q8_pair_group(
                sums, vld1q_s8(activation + group * 16),
                tile + static_cast<size_t>(group) * 32,
                scale0, groups, group,
                0.5f * activation_scales[group / 2]);
        }
        sums = vmulq_f32(
            sums, vld1q_f32(B.nvfp4_row_scales + n));
        vst1q_f32(output + n, sums);
    }
    if (n < n_end) {
        const int bytes_per_row = K / 2;
        for (; n < n_end; ++n) {
            const uint8_t* scales =
                B.nvfp4_scales + static_cast<size_t>(n) * groups;
            float sum = 0.0f;
            for (int group = 0; group < groups; ++group) {
                sum += nvfp4_dot16_q8(
                           activation + group * 16,
                           nvfp4_block_ptr(
                               B, n, group, groups, bytes_per_row)) *
                       (0.5f * activation_scales[group / 2]) *
                       kFp8E4m3fnDecodeTable[scales[group]];
            }
            output[n] = sum * B.nvfp4_row_scales[n];
        }
    }
}

#endif

}  // namespace

bool pack_nvfp4_q8_pairs(const uint8_t* source, uint8_t* destination,
                         int N, int K) {
    if (!source || !destination || N <= 0 || K <= 0 ||
        (N % 4) != 0 || (K % 16) != 0)
        return false;
    const int groups = K / 16;
    const int bytes_per_row = K / 2;
    for (int n = 0; n < N; n += 4) {
        uint8_t* tile = destination +
            static_cast<size_t>(n / 4) * groups * 32;
        int group = 0;
#if HAS_NEON && defined(__aarch64__)
        const uint64_t* row0 = reinterpret_cast<const uint64_t*>(
            source + static_cast<size_t>(n) * bytes_per_row);
        const uint64_t* row1 = reinterpret_cast<const uint64_t*>(
            source + static_cast<size_t>(n + 1) * bytes_per_row);
        const uint64_t* row2 = reinterpret_cast<const uint64_t*>(
            source + static_cast<size_t>(n + 2) * bytes_per_row);
        const uint64_t* row3 = reinterpret_cast<const uint64_t*>(
            source + static_cast<size_t>(n + 3) * bytes_per_row);
        for (; group + 1 < groups; group += 2) {
            const uint64x2_t values0 = vld1q_u64(row0 + group);
            const uint64x2_t values1 = vld1q_u64(row1 + group);
            const uint64x2_t values2 = vld1q_u64(row2 + group);
            const uint64x2_t values3 = vld1q_u64(row3 + group);
            uint64_t* block = reinterpret_cast<uint64_t*>(
                tile + static_cast<size_t>(group) * 32);
            vst1q_u64(block, vzip1q_u64(values0, values1));
            vst1q_u64(block + 2, vzip1q_u64(values2, values3));
            vst1q_u64(block + 4, vzip2q_u64(values0, values1));
            vst1q_u64(block + 6, vzip2q_u64(values2, values3));
        }
#endif
        for (; group < groups; ++group) {
            uint8_t* block = tile + static_cast<size_t>(group) * 32;
            for (int row = 0; row < 4; ++row)
                std::memcpy(
                    block + row * 8,
                    source + static_cast<size_t>(n + row) * bytes_per_row +
                        group * 8, 8);
        }
    }
    return true;
}

bool kernel_matmul_nvfp4_gemv_batch(
    const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& weights,
    std::vector<Tensor>& outputs,
    ThreadPool* thread_pool) {
#if HAS_NEON && defined(__aarch64__)
    const size_t batch = inputs.size();
    if (batch < 2 || weights.size() != batch || outputs.size() != batch ||
        !thread_pool || thread_pool->num_threads() < 2) {
        return false;
    }
    const int K = static_cast<int>(inputs[0].shape[0]);
    const int N = static_cast<int>(weights[0].shape[0]);
    if (K <= 0 || N <= 0 || K % 16 != 0)
        return false;
    for (size_t i = 0; i < batch; ++i) {
        const Tensor& input = inputs[i];
        const Tensor& weight = weights[i];
        const Tensor& output = outputs[i];
        if (input.prec != Precision::FP32 || !input.data ||
            input.shape[0] != K || input.shape[1] != 1 ||
            weight.prec != Precision::NVFP4 || !weight.data ||
            !weight.nvfp4_scales || !weight.nvfp4_row_scales ||
            weight.shape[0] != N || weight.shape[1] != K ||
            output.prec != Precision::FP32 || !output.data ||
            output.shape[0] != N || output.shape[1] != 1) {
            return false;
        }
    }

    MatmulTimer timer;
    const bool use_q8_activation = nvfp4_q8_activation_enabled();
    const bool use_fp16_activation =
        !use_q8_activation && nvfp4_fp16_activation_enabled();
    timer.set_shape(
        use_q8_activation ? "nvfp4_gemv_batch_q8act"
                          : use_fp16_activation
                                ? "nvfp4_gemv_batch_fp16act"
                                : "nvfp4_gemv_batch",
        static_cast<int>(batch), N, K,
        16, K / 16, false, false, thread_pool->num_threads());
    static thread_local std::vector<__fp16> fp16_activation_storage;
    static thread_local std::vector<const __fp16*> fp16_activations;
    static thread_local std::vector<std::vector<int8_t>> q8_activations;
    static thread_local std::vector<std::vector<float>> q8_activation_scales;
    static thread_local std::vector<const int8_t*> q8_activation_ptrs;
    static thread_local std::vector<const float*> q8_scale_ptrs;
    if (use_fp16_activation) {
        fp16_activation_storage.resize(
            batch * static_cast<size_t>(K));
        fp16_activations.resize(batch);
        for (size_t i = 0; i < batch; ++i) {
            size_t shared = 0;
            for (; shared < i; ++shared) {
                if (inputs[shared].data == inputs[i].data) {
                    fp16_activations[i] = fp16_activations[shared];
                    break;
                }
            }
            if (shared != i)
                continue;
            __fp16* destination = fp16_activation_storage.data() +
                i * static_cast<size_t>(K);
            convert_fp32_to_fp16(
                inputs[i].ptr<float>(), destination, K);
            fp16_activations[i] = destination;
        }
    }
    if (use_q8_activation) {
        q8_activations.resize(batch);
        q8_activation_scales.resize(batch);
        q8_activation_ptrs.resize(batch);
        q8_scale_ptrs.resize(batch);
        for (size_t i = 0; i < batch; ++i) {
            size_t shared = 0;
            for (; shared < i; ++shared) {
                if (inputs[shared].data == inputs[i].data) {
                    q8_activation_ptrs[i] = q8_activation_ptrs[shared];
                    q8_scale_ptrs[i] = q8_scale_ptrs[shared];
                    break;
                }
            }
            if (shared != i) continue;
            quantize_a_q8_blocks(
                inputs[i].ptr<float>(), 1, K, K, K,
                q8_activations[i], q8_activation_scales[i]);
            q8_activation_ptrs[i] = q8_activations[i].data();
            q8_scale_ptrs[i] = q8_activation_scales[i].data();
        }
    }
    const __fp16* const* fp16_activation_ptrs =
        use_fp16_activation ? fp16_activations.data() : nullptr;
    const int8_t* const* q8_activation_data =
        use_q8_activation ? q8_activation_ptrs.data() : nullptr;
    const float* const* q8_scale_data =
        use_q8_activation ? q8_scale_ptrs.data() : nullptr;
    int n_chunk = std::max(
        N / thread_pool->num_threads(), 4);
    n_chunk = ((n_chunk + 3) / 4) * 4;
    thread_pool->parallel_for(
        0, N, n_chunk, [&](int, int n_begin, int n_end) {
            for (size_t i = 0; i < batch; ++i) {
                if (use_q8_activation) {
                    if (weights[i].nvfp4_q8_pair_data) {
                        matmul_nvfp4_gemv_four_q8_pair_range(
                            q8_activation_data[i], q8_scale_data[i],
                            weights[i], outputs[i], n_begin, n_end);
                    } else {
                        matmul_nvfp4_gemv_four_q8_range(
                            q8_activation_data[i], q8_scale_data[i],
                            weights[i], outputs[i], n_begin, n_end);
                    }
                } else if (weights[i].nvfp4_q8_pair_data) {
                    // The pair layout is primarily for Q8 decode. Preserve
                    // correctness when that optional path is disabled by
                    // using the layout-aware exact kernel.
                    matmul_nvfp4_range(
                        inputs[i], weights[i], outputs[i],
                        Activation::NONE, 0, -1, n_begin, n_end);
                } else if (use_fp16_activation) {
                    matmul_nvfp4_gemv_four_f16_range(
                        fp16_activation_ptrs[i], weights[i], outputs[i],
                        Activation::NONE, 0, -1, n_begin, n_end);
                } else {
                    matmul_nvfp4_gemv_four_range(
                        inputs[i], weights[i], outputs[i], Activation::NONE,
                        0, -1, n_begin, n_end);
                }
            }
        });
    return true;
#else
    (void)inputs;
    (void)weights;
    (void)outputs;
    (void)thread_pool;
    return false;
#endif
}

void matmul_dispatch_nvfp4(const Tensor& A, const Tensor& B, Tensor& C,
                           ThreadPool* thread_pool, Activation act,
                           int act_n_begin, int act_n_len,
                           MatmulTimer& timer) {
    const int M = static_cast<int>(A.shape[1]);
    const int K = static_cast<int>(A.shape[0]);
    const int N = static_cast<int>(B.shape[0]);
    const int threads = thread_pool ? thread_pool->num_threads() : 1;
    const bool use_q8_activation =
#if HAS_NEON && defined(__aarch64__)
        M == 1 && act == Activation::NONE &&
        nvfp4_q8_activation_enabled();
#else
        false;
#endif
    const bool use_fp16_activation =
#if HAS_NEON && defined(__aarch64__)
        M == 1 && !use_q8_activation && nvfp4_fp16_activation_enabled();
#else
        false;
#endif
    timer.set_shape(use_q8_activation ? "nvfp4_gemv_q8act"
                                      : use_fp16_activation
                                            ? "nvfp4_gemv_fp16act"
                                            : M == 1 ? "nvfp4_gemv"
                                                     : "nvfp4_gemm",
                    M, N, K, 16, K > 0 ? K / 16 : 0,
                    false, false, threads);
    if (A.prec != Precision::FP32 || B.prec != Precision::NVFP4 ||
        C.prec != Precision::FP32 || B.shape[1] != K || K <= 0 ||
        K % 16 != 0 || !A.data || !B.data || !B.nvfp4_scales ||
        !B.nvfp4_row_scales || !C.data) {
        return;
    }
#if HAS_NEON && defined(__aarch64__)
    static thread_local std::vector<__fp16> fp16_activation;
    static thread_local std::vector<int8_t> q8_activation;
    static thread_local std::vector<float> q8_activation_scales;
    if (use_q8_activation) {
        quantize_a_q8_blocks(
            A.ptr<float>(), 1, K, K, K,
            q8_activation, q8_activation_scales);
    }
    if (use_fp16_activation) {
        fp16_activation.resize(static_cast<size_t>(K));
        convert_fp32_to_fp16(
            A.ptr<float>(), fp16_activation.data(), K);
    }
    const __fp16* fp16_activation_data =
        use_fp16_activation ? fp16_activation.data() : nullptr;
    const int8_t* q8_activation_data =
        use_q8_activation ? q8_activation.data() : nullptr;
    const float* q8_scale_data =
        use_q8_activation ? q8_activation_scales.data() : nullptr;
#endif
    auto run = [&](int begin, int end) {
#if HAS_NEON && defined(__aarch64__)
        if (M == 1) {
            if (use_q8_activation) {
                if (B.nvfp4_q8_pair_data) {
                    matmul_nvfp4_gemv_four_q8_pair_range(
                        q8_activation_data, q8_scale_data,
                        B, C, begin, end);
                } else {
                    matmul_nvfp4_gemv_four_q8_range(
                        q8_activation_data, q8_scale_data,
                        B, C, begin, end);
                }
            } else if (B.nvfp4_q8_pair_data) {
                matmul_nvfp4_range(
                    A, B, C, act, act_n_begin, act_n_len, begin, end);
            } else if (use_fp16_activation) {
                matmul_nvfp4_gemv_four_f16_range(
                    fp16_activation_data, B, C, act,
                    act_n_begin, act_n_len, begin, end);
            } else {
                matmul_nvfp4_gemv_four_range(
                    A, B, C, act, act_n_begin, act_n_len, begin, end);
            }
            return;
        }
        if (M <= 32) {
            matmul_nvfp4_small_m_range(
                A, B, C, act, act_n_begin, act_n_len, begin, end);
            return;
        }
#endif
        matmul_nvfp4_range(A, B, C, act, act_n_begin, act_n_len,
                           begin, end);
    };
    if (thread_pool && threads > 1 && N > 32) {
        int n_chunk = std::max(4, N / threads);
        n_chunk = ((n_chunk + 3) / 4) * 4;
        thread_pool->parallel_for(
            0, N, n_chunk,
            [&](int, int begin, int end) { run(begin, end); });
    } else {
        run(0, N);
    }
}
