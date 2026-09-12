#include "kernels/cpu/matmul/matmul_internal.h"
#include "kernels/cpu/matmul/matmul_profile.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#if HAS_NEON && defined(__aarch64__)
static inline void quantize_q8_block32_neon(const float* src, float& scale,
                                            int8x16_t& q_lo, int8x16_t& q_hi) {
    float32x4_t v0 = vld1q_f32(src + 0);
    float32x4_t v1 = vld1q_f32(src + 4);
    float32x4_t v2 = vld1q_f32(src + 8);
    float32x4_t v3 = vld1q_f32(src + 12);
    float32x4_t v4 = vld1q_f32(src + 16);
    float32x4_t v5 = vld1q_f32(src + 20);
    float32x4_t v6 = vld1q_f32(src + 24);
    float32x4_t v7 = vld1q_f32(src + 28);
    float32x4_t m0 = vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1));
    float32x4_t m1 = vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3));
    float32x4_t m2 = vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5));
    float32x4_t m3 = vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7));
    float amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));

    scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
    float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;

    int32x4_t q0 = vcvtnq_s32_f32(vmulq_n_f32(v0, inv_scale));
    int32x4_t q1 = vcvtnq_s32_f32(vmulq_n_f32(v1, inv_scale));
    int32x4_t q2 = vcvtnq_s32_f32(vmulq_n_f32(v2, inv_scale));
    int32x4_t q3 = vcvtnq_s32_f32(vmulq_n_f32(v3, inv_scale));
    int32x4_t q4 = vcvtnq_s32_f32(vmulq_n_f32(v4, inv_scale));
    int32x4_t q5 = vcvtnq_s32_f32(vmulq_n_f32(v5, inv_scale));
    int32x4_t q6 = vcvtnq_s32_f32(vmulq_n_f32(v6, inv_scale));
    int32x4_t q7 = vcvtnq_s32_f32(vmulq_n_f32(v7, inv_scale));

    int16x8_t q01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
    int16x8_t q23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
    int16x8_t q45 = vcombine_s16(vqmovn_s32(q4), vqmovn_s32(q5));
    int16x8_t q67 = vcombine_s16(vqmovn_s32(q6), vqmovn_s32(q7));
    q_lo = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));
    q_hi = vcombine_s8(vqmovn_s16(q45), vqmovn_s16(q67));
}

static inline uint32x4_t encode_fp8_e4m3fn_neon(float32x4_t values) {
    const uint32x4_t original_bits = vreinterpretq_u32_f32(values);
    const uint32x4_t sign = vandq_u32(
        vshrq_n_u32(original_bits, 24), vdupq_n_u32(0x80));
    const uint32x4_t magnitude_bits = vandq_u32(
        original_bits, vdupq_n_u32(0x7fffffffu));
    const uint32x4_t nan_mask = vandq_u32(
        vceqq_u32(
            vandq_u32(vshrq_n_u32(magnitude_bits, 23),
                      vdupq_n_u32(0xff)),
            vdupq_n_u32(0xff)),
        vcgtq_u32(vandq_u32(magnitude_bits, vdupq_n_u32(0x7fffff)),
                  vdupq_n_u32(0)));
    const float32x4_t magnitude = vminq_f32(
        vabsq_f32(values), vdupq_n_f32(448.0f));
    const uint32x4_t bits = vreinterpretq_u32_f32(magnitude);

    const uint32x4_t subnormal = vminq_u32(
        vcvtnq_u32_f32(vmulq_n_f32(magnitude, 512.0f)),
        vdupq_n_u32(8));

    int32x4_t exponent = vsubq_s32(
        vreinterpretq_s32_u32(
            vandq_u32(vshrq_n_u32(bits, 23), vdupq_n_u32(0xff))),
        vdupq_n_s32(127));
    const uint32x4_t mantissa =
        vandq_u32(bits, vdupq_n_u32(0x7fffff));
    uint32x4_t significand = vaddq_u32(
        vshrq_n_u32(mantissa, 20), vdupq_n_u32(8));
    const uint32x4_t remainder =
        vandq_u32(mantissa, vdupq_n_u32(0xfffff));
    const uint32x4_t round_up = vorrq_u32(
        vcgtq_u32(remainder, vdupq_n_u32(0x80000)),
        vandq_u32(
            vceqq_u32(remainder, vdupq_n_u32(0x80000)),
            vcgtq_u32(vandq_u32(significand, vdupq_n_u32(1)),
                      vdupq_n_u32(0))));
    significand = vaddq_u32(
        significand, vshrq_n_u32(round_up, 31));
    const uint32x4_t carry =
        vceqq_u32(significand, vdupq_n_u32(16));
    exponent = vaddq_s32(
        exponent,
        vreinterpretq_s32_u32(vshrq_n_u32(carry, 31)));
    significand = vbslq_u32(
        carry, vdupq_n_u32(8), significand);
    int32x4_t normal = vaddq_s32(
        vshlq_n_s32(vaddq_s32(exponent, vdupq_n_s32(6)), 3),
        vreinterpretq_s32_u32(significand));
    normal = vmaxq_s32(vdupq_n_s32(0),
                       vminq_s32(normal, vdupq_n_s32(126)));
    uint32x4_t encoded = vbslq_u32(
        vcltq_f32(magnitude, vdupq_n_f32(1.0f / 64.0f)),
        subnormal, vreinterpretq_u32_s32(normal));
    encoded = vorrq_u32(encoded, sign);
    return vbslq_u32(nan_mask, vdupq_n_u32(0x7f), encoded);
}
#endif

namespace {

constexpr int kFp8ActivationBlock = 128;

struct Fp8ActivationQ8Tables {
    std::array<float, 127> decoded{};
    std::array<std::array<int8_t, 256>, 127> quantized{};
};

const Fp8ActivationQ8Tables& fp8_activation_q8_tables() {
    static const Fp8ActivationQ8Tables tables = [] {
        Fp8ActivationQ8Tables result;
        for (size_t code = 0; code < result.decoded.size(); ++code)
            result.decoded[code] =
                decode_fp8_e4m3fn(static_cast<uint8_t>(code));
        for (size_t maximum = 0; maximum < result.quantized.size();
             ++maximum) {
            const float maximum_value = result.decoded[maximum];
            const float inverse =
                maximum_value > 0.0f ? 127.0f / maximum_value : 0.0f;
            for (size_t byte = 0; byte < 256; ++byte) {
                const uint8_t magnitude = std::min<uint8_t>(
                    static_cast<uint8_t>(byte) & 0x7f, 126);
                const float coefficient =
                    (byte & 0x80) != 0
                        ? -result.decoded[magnitude]
                        : result.decoded[magnitude];
                const int quantized = std::clamp(
                    static_cast<int>(std::nearbyint(coefficient * inverse)),
                    -127, 127);
                result.quantized[maximum][byte] =
                    static_cast<int8_t>(quantized);
            }
        }
        return result;
    }();
    return tables;
}

float fp8_ue8m0_scale(const float* values, int count) {
    float maximum = 0.0f;
#if HAS_NEON && defined(__aarch64__)
    float32x4_t vector_maximum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < count; i += 4) {
        vector_maximum = vmaxq_f32(
            vector_maximum, vabsq_f32(vld1q_f32(values + i)));
    }
    maximum = vmaxvq_f32(vector_maximum);
    for (; i < count; ++i)
        maximum = std::max(maximum, std::fabs(values[i]));
#else
    for (int i = 0; i < count; ++i)
        maximum = std::max(maximum, std::fabs(values[i]));
#endif
    // This is fast_round_scale(max(amax, 1e-4) / 448) from the reference
    // implementation. Its E8M0 output is a power of two.
    maximum = std::max(maximum, 1.0e-4f);

    // Let maximum = significand * 2^E, with significand in [1, 2).
    // Since 448 = 1.75 * 2^8, ceil(log2(maximum / 448)) is
    // E - 8, plus one exactly when significand is greater than 1.75.
    // Reading the IEEE fields computes the same power of two without a
    // scalar log2/ceil/ldexp sequence for every 128-value activation block.
    uint32_t bits = 0;
    std::memcpy(&bits, &maximum, sizeof(bits));
    const int unbiased = static_cast<int>((bits >> 23) & 0xffu) - 127;
    const uint32_t mantissa = bits & 0x7fffffu;
    const int exponent = std::clamp(
        unbiased - 8 + (mantissa > 0x600000u ? 1 : 0), -127, 127);
    const uint32_t scale_bits =
        exponent == -127
            ? uint32_t{1} << 22
            : static_cast<uint32_t>(exponent + 127) << 23;
    float scale = 0.0f;
    std::memcpy(&scale, &scale_bits, sizeof(scale));
    return scale;
}

void quantize_fp8_q8_row(const float* input, int K, int8_t* output,
                         float* output_scales, float* fp8_dequant = nullptr) {
    const int q8_blocks =
        (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    const int fp8_blocks =
        (K + kFp8ActivationBlock - 1) / kFp8ActivationBlock;
    const auto& tables = fp8_activation_q8_tables();
    alignas(16) uint8_t encoded_values[kFp8ActivationBlock];
    for (int block = 0; block < fp8_blocks; ++block) {
        const int begin = block * kFp8ActivationBlock;
        const int end = std::min(begin + kFp8ActivationBlock, K);
        const int count = end - begin;
        const float activation_scale =
            fp8_ue8m0_scale(input + begin, count);
        const float inverse_activation_scale = 1.0f / activation_scale;
        int i = 0;
#if HAS_NEON && defined(__aarch64__)
        for (; i + 3 < count; i += 4) {
            alignas(16) uint32_t encoded[4];
            vst1q_u32(
                encoded,
                encode_fp8_e4m3fn_neon(
                    vmulq_n_f32(
                        vld1q_f32(input + begin + i),
                        inverse_activation_scale)));
            for (int lane = 0; lane < 4; ++lane) {
                const uint8_t code =
                    static_cast<uint8_t>(encoded[lane]);
                encoded_values[i + lane] = code;
                const uint8_t magnitude =
                    std::min<uint8_t>(code & 0x7f, 126);
                const float coefficient =
                    (code & 0x80) != 0
                        ? -tables.decoded[magnitude]
                        : tables.decoded[magnitude];
                if (fp8_dequant) {
                    fp8_dequant[begin + i + lane] =
                        coefficient * activation_scale;
                }
            }
        }
#endif
        for (; i < count; ++i) {
            const float normalized = std::clamp(
                input[begin + i] * inverse_activation_scale,
                -448.0f, 448.0f);
            const uint8_t encoded = encode_fp8_e4m3fn(normalized);
            encoded_values[i] = encoded;
            const uint8_t magnitude =
                std::min<uint8_t>(encoded & 0x7f, 126);
            const float coefficient =
                std::signbit(normalized)
                    ? -tables.decoded[magnitude]
                    : tables.decoded[magnitude];
            if (fp8_dequant) {
                fp8_dequant[begin + i] =
                    coefficient * activation_scale;
            }
        }
        // The checkpoint activation scale is fixed for the 128-element FP8
        // block. The internal Q8 representation may use a finer 32-element
        // scale without changing those FP8 values; this reduces the extra
        // approximation introduced solely for SDOT.
        for (int group_begin = 0; group_begin < count;
             group_begin += MATMUL_Q8_BLOCK) {
            const int group_end =
                std::min(group_begin + MATMUL_Q8_BLOCK, count);
            uint8_t maximum_code = 0;
            for (int i = group_begin; i < group_end; ++i) {
                maximum_code = std::max(
                    maximum_code,
                    std::min<uint8_t>(encoded_values[i] & 0x7f, 126));
            }
            const float coefficient_maximum =
                tables.decoded[maximum_code];
            const float coefficient_scale =
                coefficient_maximum > 0.0f
                    ? coefficient_maximum / 127.0f
                    : 1.0f;
            const auto& quantized = tables.quantized[maximum_code];
            for (int i = group_begin; i < group_end; ++i)
                output[begin + i] = quantized[encoded_values[i]];
            const int q8_block =
                (begin + group_begin) / MATMUL_Q8_BLOCK;
            if (q8_block < q8_blocks) {
                output_scales[q8_block] =
                    activation_scale * coefficient_scale;
            }
        }
    }
}

} // namespace

void quantize_a_q8_blocks(const float* A, int M, int K, int lda, int K_storage,
                          std::vector<int8_t>& qA,
                          std::vector<float>& a_scales) {
    auto t0 = std::chrono::steady_clock::now();
    if (K_storage < K)
        K_storage = K;
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    if (K_storage == K) {
        qA.resize((size_t)M * K_storage);
    } else {
        qA.assign((size_t)M * K_storage, 0);
    }
    a_scales.resize((size_t)M * blocks_per_row);
    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * lda;
        int8_t* qa_row = qA.data() + (size_t)m * K_storage;
        float* s_row = a_scales.data() + (size_t)m * blocks_per_row;
        for (int qb = 0; qb < blocks_per_row; qb++) {
            int k_begin = qb * MATMUL_Q8_BLOCK;
            int k_end = std::min(k_begin + MATMUL_Q8_BLOCK, K);
            float amax = 0.f;
#if HAS_NEON && defined(__aarch64__)
            if (k_end - k_begin == MATMUL_Q8_BLOCK) {
                float scale = 1.f;
                int8x16_t q_lo;
                int8x16_t q_hi;
                quantize_q8_block32_neon(a_row + k_begin, scale, q_lo, q_hi);
                s_row[qb] = scale;
                vst1q_s8(qa_row + k_begin, q_lo);
                vst1q_s8(qa_row + k_begin + 16, q_hi);
                continue;
            }
#endif
            for (int k = k_begin; k < k_end; k++) {
                amax = std::max(amax, std::fabs(a_row[k]));
            }
            float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
            float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;
            s_row[qb] = scale;
            for (int k = k_begin; k < k_end; k++) {
                int q = (int)std::nearbyint(a_row[k] * inv_scale);
                q = std::max(-127, std::min(127, q));
                qa_row[k] = (int8_t)q;
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    matmul_record_q8_quant_a(ms);
}

void quantize_a_fp8_q8_blocks(const float* A, int M, int K, int lda,
                              int K_storage, std::vector<int8_t>& qA,
                              std::vector<float>& a_scales) {
    const auto begin = std::chrono::steady_clock::now();
    K_storage = std::max(K_storage, K);
    const int blocks_per_row =
        (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    if (K_storage == K) {
        qA.resize(static_cast<size_t>(M) * K_storage);
    } else {
        qA.assign(static_cast<size_t>(M) * K_storage, 0);
    }
    a_scales.resize(static_cast<size_t>(M) * blocks_per_row);
    for (int m = 0; m < M; ++m) {
        quantize_fp8_q8_row(
            A + static_cast<size_t>(m) * lda, K,
            qA.data() + static_cast<size_t>(m) * K_storage,
            a_scales.data() + static_cast<size_t>(m) * blocks_per_row);
    }
    matmul_record_q8_quant_a(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin)
            .count());
}

void quantize_a_fp8_dequant(const float* A, int M, int K, int lda,
                            std::vector<float>& output) {
    output.resize(static_cast<size_t>(M) * K);
    for (int m = 0; m < M; ++m) {
        const float* input = A + static_cast<size_t>(m) * lda;
        float* destination = output.data() + static_cast<size_t>(m) * K;
        for (int begin = 0; begin < K; begin += kFp8ActivationBlock) {
            const int end = std::min(begin + kFp8ActivationBlock, K);
            const float scale =
                fp8_ue8m0_scale(input + begin, end - begin);
            for (int k = begin; k < end; ++k) {
                destination[k] =
                    decode_fp8_e4m3fn(encode_fp8_e4m3fn(
                        std::clamp(
                            input[k] / scale, -448.0f, 448.0f))) *
                    scale;
            }
        }
    }
}

#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
void quantize_a_q8_blocks_a4(const float* A, int M, int K, int lda,
                             std::vector<Q8A4Block>& qA4) {
    auto t0 = std::chrono::steady_clock::now();
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    int m_tiles = (M + 3) / 4;
    qA4.resize((size_t)m_tiles * blocks_per_row);
    for (int mt = 0; mt < m_tiles; mt++) {
        for (int qb = 0; qb < blocks_per_row; qb++) {
            int k_begin = qb * MATMUL_Q8_BLOCK;
            int k_end = std::min(k_begin + MATMUL_Q8_BLOCK, K);
            Q8A4Block& block = qA4[(size_t)mt * blocks_per_row + qb];
            for (int ar = 0; ar < 4; ar++) {
                int m = mt * 4 + ar;
                if (m >= M)
                    continue;

                const float* a_row = A + (size_t)m * lda;
#if HAS_NEON && defined(__aarch64__)
                if (k_end - k_begin == MATMUL_Q8_BLOCK) {
                    float scale = 1.f;
                    int8x16_t q_lo;
                    int8x16_t q_hi;
                    quantize_q8_block32_neon(a_row + k_begin, scale, q_lo,
                                             q_hi);
                    block.scales[ar] = scale;
                    vst1q_s8(block.even[ar], vuzp1q_s8(q_lo, q_hi));
                    vst1q_s8(block.odd[ar], vuzp2q_s8(q_lo, q_hi));
                    continue;
                }
#endif
                float amax = 0.f;
                for (int k = k_begin; k < k_end; k++) {
                    amax = std::max(amax, std::fabs(a_row[k]));
                }
                float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
                float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;
                block.scales[ar] = scale;
                for (int i = 0; i < 16; i++) {
                    int k0 = k_begin + i * 2;
                    int k1 = k0 + 1;
                    int q0 = (k0 < k_end)
                                 ? (int)std::nearbyint(a_row[k0] * inv_scale)
                                 : 0;
                    int q1 = (k1 < k_end)
                                 ? (int)std::nearbyint(a_row[k1] * inv_scale)
                                 : 0;
                    q0 = std::max(-127, std::min(127, q0));
                    q1 = std::max(-127, std::min(127, q1));
                    block.even[ar][i] = (int8_t)q0;
                    block.odd[ar][i] = (int8_t)q1;
                }
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    matmul_record_q8_quant_a(ms);
}

void quantize_a_q8_blocks_i8mm_a8(const float* A, int M, int K, int lda,
                                  std::vector<Q8A8I8MMBlock>& qA8) {
    auto t0 = std::chrono::steady_clock::now();
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    int m_tiles = (M + 7) / 8;
    qA8.assign((size_t)m_tiles * blocks_per_row, {});

    for (int mt = 0; mt < m_tiles; mt++) {
        for (int qb = 0; qb < blocks_per_row; qb++) {
            Q8A8I8MMBlock& block =
                qA8[(size_t)mt * blocks_per_row + qb];
            for (int ar = 0; ar < 8; ar++) {
                int m = mt * 8 + ar;
                if (m >= M)
                    continue;
                float scale = 1.f;
                int8x16_t q_lo;
                int8x16_t q_hi;
                quantize_q8_block32_neon(
                    A + (size_t)m * lda + qb * MATMUL_Q8_BLOCK, scale, q_lo,
                    q_hi);
                int pair = ar / 2;
                int lane = (ar & 1) * 2;
                block.scale_pairs[pair][lane] = scale;
                block.scale_pairs[pair][lane + 1] = scale;
                int8x16_t q_even = vuzp1q_s8(q_lo, q_hi);
                int8x16_t q_odd = vuzp2q_s8(q_lo, q_hi);
                vst1_s8(block.q[0][ar], vget_low_s8(q_even));
                vst1_s8(block.q[1][ar], vget_high_s8(q_even));
                vst1_s8(block.q[2][ar], vget_low_s8(q_odd));
                vst1_s8(block.q[3][ar], vget_high_s8(q_odd));
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    matmul_record_q8_quant_a(ms);
}
#endif

void quantize_a_q8_blocks_even_odd(const float* A, int K,
                                   std::vector<int8_t>& qA_even,
                                   std::vector<int8_t>& qA_odd,
                                   std::vector<float>& a_scales) {
    auto t0 = std::chrono::steady_clock::now();
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    qA_even.resize((size_t)blocks_per_row * 16);
    qA_odd.resize((size_t)blocks_per_row * 16);
    a_scales.resize((size_t)blocks_per_row);
    for (int qb = 0; qb < blocks_per_row; qb++) {
        int k_begin = qb * MATMUL_Q8_BLOCK;
        int k_end = std::min(k_begin + MATMUL_Q8_BLOCK, K);
        float amax = 0.f;
#if HAS_NEON && defined(__aarch64__)
        if (k_end - k_begin == MATMUL_Q8_BLOCK) {
            float scale = 1.f;
            int8x16_t q_lo;
            int8x16_t q_hi;
            quantize_q8_block32_neon(A + k_begin, scale, q_lo, q_hi);
            a_scales[qb] = scale;
            vst1q_s8(qA_even.data() + (size_t)qb * 16, vuzp1q_s8(q_lo, q_hi));
            vst1q_s8(qA_odd.data() + (size_t)qb * 16, vuzp2q_s8(q_lo, q_hi));
            continue;
        }
#endif
        for (int k = k_begin; k < k_end; k++) {
            amax = std::max(amax, std::fabs(A[k]));
        }
        float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
        float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;
        a_scales[qb] = scale;

        int8_t* even = qA_even.data() + (size_t)qb * 16;
        int8_t* odd = qA_odd.data() + (size_t)qb * 16;
        for (int i = 0; i < 16; i++) {
            int k0 = k_begin + i * 2;
            int k1 = k0 + 1;
            int q0 = (k0 < k_end) ? (int)std::nearbyint(A[k0] * inv_scale) : 0;
            int q1 = (k1 < k_end) ? (int)std::nearbyint(A[k1] * inv_scale) : 0;
            q0 = std::max(-127, std::min(127, q0));
            q1 = std::max(-127, std::min(127, q1));
            even[i] = (int8_t)q0;
            odd[i] = (int8_t)q1;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    matmul_record_q8_quant_a(ms);
}

void quantize_a_fp8_q8_even_odd(const float* A, int K,
                               std::vector<int8_t>& qA_even,
                               std::vector<int8_t>& qA_odd,
                               std::vector<float>& a_scales,
                               std::vector<float>* fp8_dequant) {
    const auto begin = std::chrono::steady_clock::now();
    const int blocks =
        (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    static thread_local std::vector<int8_t> contiguous;
    contiguous.resize(static_cast<size_t>(K));
    a_scales.resize(static_cast<size_t>(blocks));
    if (fp8_dequant)
        fp8_dequant->resize(static_cast<size_t>(K));
    quantize_fp8_q8_row(
        A, K, contiguous.data(), a_scales.data(),
        fp8_dequant ? fp8_dequant->data() : nullptr);
    qA_even.resize(static_cast<size_t>(blocks) * 16);
    qA_odd.resize(static_cast<size_t>(blocks) * 16);
    for (int block = 0; block < blocks; ++block) {
        int8_t* even =
            qA_even.data() + static_cast<size_t>(block) * 16;
        int8_t* odd =
            qA_odd.data() + static_cast<size_t>(block) * 16;
        const int8_t* source =
            contiguous.data() +
            static_cast<size_t>(block) * MATMUL_Q8_BLOCK;
        const int count = std::min(
            MATMUL_Q8_BLOCK, K - block * MATMUL_Q8_BLOCK);
#if HAS_NEON && defined(__aarch64__)
        if (count == MATMUL_Q8_BLOCK) {
            const int8x16_t low = vld1q_s8(source);
            const int8x16_t high = vld1q_s8(source + 16);
            vst1q_s8(even, vuzp1q_s8(low, high));
            vst1q_s8(odd, vuzp2q_s8(low, high));
            continue;
        }
#endif
        for (int i = 0; i < 16; ++i) {
            even[i] = i * 2 < count ? source[i * 2] : 0;
            odd[i] = i * 2 + 1 < count ? source[i * 2 + 1] : 0;
        }
    }
    matmul_record_q8_quant_a(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin)
            .count());
}

void quantize_a_q8_g128_even_odd(const float* A, int K,
                                 std::vector<int8_t>& qA_even,
                                 std::vector<int8_t>& qA_odd,
                                 std::vector<float>& a_scales) {
    auto t0 = std::chrono::steady_clock::now();
    // Callers only select this path for the physical BG128 weight layout.
    int groups = K / 128;
    int blocks = groups * 4;
    qA_even.resize((size_t)blocks * 16);
    qA_odd.resize((size_t)blocks * 16);
    a_scales.resize((size_t)groups);

    for (int g = 0; g < groups; g++) {
        const float* src = A + (size_t)g * 128;
        float amax = 0.f;
#if HAS_NEON && defined(__aarch64__)
        float32x4_t vmax = vdupq_n_f32(0.f);
        for (int k = 0; k < 128; k += 4)
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(src + k)));
        amax = vmaxvq_f32(vmax);
#else
        for (int k = 0; k < 128; k++)
            amax = std::max(amax, std::fabs(src[k]));
#endif
        float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
        float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;
        a_scales[g] = scale;

        for (int qgi = 0; qgi < 4; qgi++) {
            int qb = g * 4 + qgi;
            int8_t* even = qA_even.data() + (size_t)qb * 16;
            int8_t* odd = qA_odd.data() + (size_t)qb * 16;
            const float* block = src + qgi * 32;
#if HAS_NEON && defined(__aarch64__)
            int32x4_t q[8];
            for (int i = 0; i < 8; i++) {
                q[i] = vcvtnq_s32_f32(
                    vmulq_n_f32(vld1q_f32(block + i * 4), inv_scale));
            }
            int16x8_t q01 = vcombine_s16(vqmovn_s32(q[0]), vqmovn_s32(q[1]));
            int16x8_t q23 = vcombine_s16(vqmovn_s32(q[2]), vqmovn_s32(q[3]));
            int16x8_t q45 = vcombine_s16(vqmovn_s32(q[4]), vqmovn_s32(q[5]));
            int16x8_t q67 = vcombine_s16(vqmovn_s32(q[6]), vqmovn_s32(q[7]));
            int8x16_t q_lo =
                vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));
            int8x16_t q_hi =
                vcombine_s8(vqmovn_s16(q45), vqmovn_s16(q67));
            vst1q_s8(even, vuzp1q_s8(q_lo, q_hi));
            vst1q_s8(odd, vuzp2q_s8(q_lo, q_hi));
#else
            for (int i = 0; i < 16; i++) {
                int q0 = (int)std::nearbyint(block[i * 2] * inv_scale);
                int q1 = (int)std::nearbyint(block[i * 2 + 1] * inv_scale);
                even[i] = (int8_t)std::max(-127, std::min(127, q0));
                odd[i] = (int8_t)std::max(-127, std::min(127, q1));
            }
#endif
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    matmul_record_q8_quant_a(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
}
