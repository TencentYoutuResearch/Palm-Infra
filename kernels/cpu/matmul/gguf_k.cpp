#include "kernels/cpu/matmul/gguf_k.h"
#include "kernels/cpu/matmul/gguf_iq_tables.h"
#include "kernels/cpu/matmul/stq1_0.h"
#include "kernels/cpu/matmul/matmul_internal.h"
#include "core/fp16.h"
#include "runtime/threading.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

float half_value(const uint8_t* bytes) {
#if defined(__aarch64__)
    mollm::fp16_t value;
    std::memcpy(&value, bytes, sizeof(value));
    return static_cast<float>(value);
#else
    const uint16_t bits = uint16_t(bytes[0]) | (uint16_t(bytes[1]) << 8);
    const uint32_t sign = uint32_t(bits & 0x8000) << 16;
    int exponent = (bits >> 10) & 31;
    uint32_t mantissa = bits & 1023;
    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        exponent = 1;
        while ((mantissa & 1024) == 0) { mantissa <<= 1; --exponent; }
        result = sign | (uint32_t(exponent + 112) << 23) |
                 ((mantissa & 1023) << 13);
    } else if (exponent == 31) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | (uint32_t(exponent + 112) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &result, sizeof(value));
    return value;
#endif
}

void scale_min(int group, const uint8_t* packed,
               uint8_t& scale, uint8_t& minimum) {
    if (group < 4) {
        scale = packed[group] & 63;
        minimum = packed[group + 4] & 63;
    } else {
        scale = (packed[group + 4] & 15) |
                ((packed[group - 4] >> 6) << 4);
        minimum = (packed[group + 4] >> 4) |
                  ((packed[group] >> 6) << 4);
    }
}

#if defined(__aarch64__)
void store_q5_values(uint8x16_t quantized, float scale, float minimum,
                     float* output) {
    const uint16x8_t low = vmovl_u8(vget_low_u8(quantized));
    const uint16x8_t high = vmovl_u8(vget_high_u8(quantized));
    const float32x4_t min_vec = vdupq_n_f32(minimum);
    vst1q_f32(output, vsubq_f32(vmulq_n_f32(
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(low))), scale), min_vec));
    vst1q_f32(output + 4, vsubq_f32(vmulq_n_f32(
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(low))), scale), min_vec));
    vst1q_f32(output + 8, vsubq_f32(vmulq_n_f32(
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(high))), scale), min_vec));
    vst1q_f32(output + 12, vsubq_f32(vmulq_n_f32(
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(high))), scale), min_vec));
}
#endif

void decode_q4_or_q5(const uint8_t* block, float* out, bool five_bit) {
    const float d = half_value(block);
    const float dmin = half_value(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* high = five_bit ? block + 16 : nullptr;
    const uint8_t* low = block + (five_bit ? 48 : 16);
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8_t s0, m0, s1, m1;
        scale_min(quarter * 2, scales, s0, m0);
        scale_min(quarter * 2 + 1, scales, s1, m1);
        const int offset = quarter * 64;
#if defined(__aarch64__)
        if (five_bit) {
            const float ds0 = d * s0;
            const float ds1 = d * s1;
            const float dm0 = dmin * m0;
            const float dm1 = dmin * m1;
            for (int lane = 0; lane < 32; lane += 16) {
                const uint8x16_t q = vld1q_u8(low + quarter * 32 + lane);
                const uint8x16_t h = vld1q_u8(high + lane);
                const uint8x16_t q0 = vaddq_u8(
                    vandq_u8(q, vdupq_n_u8(15)),
                    vandq_u8(vtstq_u8(h, vdupq_n_u8(1 << (quarter * 2))),
                                 vdupq_n_u8(16)));
                const uint8x16_t q1 = vaddq_u8(
                    vshrq_n_u8(q, 4),
                    vandq_u8(vtstq_u8(h, vdupq_n_u8(2 << (quarter * 2))),
                                 vdupq_n_u8(16)));
                store_q5_values(q0, ds0, dm0, out + offset + lane);
                store_q5_values(q1, ds1, dm1, out + offset + 32 + lane);
            }
            continue;
        }
#endif
        for (int lane = 0; lane < 32; ++lane) {
            const uint8_t q = low[quarter * 32 + lane];
            const uint8_t h = five_bit ? high[lane] : 0;
            const int v0 = (q & 15) +
                (five_bit && (h & (1 << (quarter * 2))) ? 16 : 0);
            const int v1 = (q >> 4) +
                (five_bit && (h & (2 << (quarter * 2))) ? 16 : 0);
            out[offset + lane] = (d * s0) * v0 - dmin * m0;
            out[offset + 32 + lane] = (d * s1) * v1 - dmin * m1;
        }
    }
}

#if defined(__aarch64__)
void store_q6_values(uint8x16_t quantized, float scale, float* output) {
    const int16x8_t low = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(quantized))),
        vdupq_n_s16(32));
    const int16x8_t high = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(quantized))),
        vdupq_n_s16(32));
    vst1q_f32(output, vmulq_n_f32(vcvtq_f32_s32(
        vmovl_s16(vget_low_s16(low))), scale));
    vst1q_f32(output + 4, vmulq_n_f32(vcvtq_f32_s32(
        vmovl_s16(vget_high_s16(low))), scale));
    vst1q_f32(output + 8, vmulq_n_f32(vcvtq_f32_s32(
        vmovl_s16(vget_low_s16(high))), scale));
    vst1q_f32(output + 12, vmulq_n_f32(vcvtq_f32_s32(
        vmovl_s16(vget_high_s16(high))), scale));
}
#endif

void decode_q6(const uint8_t* block, float* out) {
    const uint8_t* low = block;
    const uint8_t* high = block + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
    const float d = half_value(block + 208);
    for (int half = 0; half < 2; ++half) {
        const uint8_t* l = low + half * 64;
        const uint8_t* h = high + half * 32;
        const int8_t* s = scales + half * 8;
        float* y = out + half * 128;
#if defined(__aarch64__)
        for (int lane = 0; lane < 32; lane += 16) {
            const uint8x16_t l0 = vld1q_u8(l + lane);
            const uint8x16_t l1 = vld1q_u8(l + 32 + lane);
            const uint8x16_t h0 = vld1q_u8(h + lane);
            const uint8x16_t q0 = vorrq_u8(
                vandq_u8(l0, vdupq_n_u8(15)),
                vshlq_n_u8(vandq_u8(h0, vdupq_n_u8(3)), 4));
            const uint8x16_t q1 = vorrq_u8(
                vandq_u8(l1, vdupq_n_u8(15)),
                vshlq_n_u8(vandq_u8(vshrq_n_u8(h0, 2),
                                       vdupq_n_u8(3)), 4));
            const uint8x16_t q2 = vorrq_u8(
                vshrq_n_u8(l0, 4),
                vandq_u8(h0, vdupq_n_u8(48)));
            const uint8x16_t q3 = vorrq_u8(
                vshrq_n_u8(l1, 4),
                vshlq_n_u8(vshrq_n_u8(h0, 6), 4));
            const int si = lane / 16;
            store_q6_values(q0, d * s[si], y + lane);
            store_q6_values(q1, d * s[si + 2], y + 32 + lane);
            store_q6_values(q2, d * s[si + 4], y + 64 + lane);
            store_q6_values(q3, d * s[si + 6], y + 96 + lane);
        }
#else
        for (int lane = 0; lane < 32; ++lane) {
            const int si = lane / 16;
            y[lane] = d * s[si] *
                (int((l[lane] & 15) | ((h[lane] & 3) << 4)) - 32);
            y[32 + lane] = d * s[si + 2] *
                (int((l[32 + lane] & 15) | (((h[lane] >> 2) & 3) << 4)) - 32);
            y[64 + lane] = d * s[si + 4] *
                (int((l[lane] >> 4) | (((h[lane] >> 4) & 3) << 4)) - 32);
            y[96 + lane] = d * s[si + 6] *
                (int((l[32 + lane] >> 4) | (((h[lane] >> 6) & 3) << 4)) - 32);
        }
#endif
    }
}

uint32_t load_u32(const uint8_t* bytes) {
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
           (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

#if defined(__aarch64__)
void store_signed_iq_grid8(int8x8_t signed_grid, float scale, float* out) {
    const int16x8_t signed_values = vmovl_s8(signed_grid);
    const float32x4_t low = vcvtq_f32_s32(
        vmovl_s16(vget_low_s16(signed_values)));
    const float32x4_t high = vcvtq_f32_s32(
        vmovl_s16(vget_high_s16(signed_values)));
    vst1q_f32(out, vmulq_n_f32(low, scale));
    vst1q_f32(out + 4, vmulq_n_f32(high, scale));
}

const std::array<int8_t, 256 * 128 * 8>& signed_iq2_grid_table() {
    static const auto table = [] {
        std::array<int8_t, 256 * 128 * 8> result{};
        for (int grid = 0; grid < 256; ++grid) {
            for (int sign_index = 0; sign_index < 128; ++sign_index) {
                const uint8_t signs = iq_signs[sign_index];
                for (int lane = 0; lane < 8; ++lane) {
                    const int value = iq2_grid[grid * 8 + lane];
                    result[(grid * 128 + sign_index) * 8 + lane] =
                        static_cast<int8_t>((signs & (1 << lane)) ? -value : value);
                }
            }
        }
        return result;
    }();
    return table;
}

const std::array<int8_t, 256 * 16 * 4>& signed_iq3_grid_table() {
    static const auto table = [] {
        std::array<int8_t, 256 * 16 * 4> result{};
        for (int grid = 0; grid < 256; ++grid) {
            for (int signs = 0; signs < 16; ++signs) {
                for (int lane = 0; lane < 4; ++lane) {
                    const int value = iq3_grid[grid * 4 + lane];
                    result[(grid * 16 + signs) * 4 + lane] =
                        static_cast<int8_t>((signs & (1 << lane)) ? -value : value);
                }
            }
        }
        return result;
    }();
    return table;
}
#endif

void decode_iq2_xxs(const uint8_t* block, float* out) {
    const float d = half_value(block);
    const uint8_t* qs = block + 2;
#if defined(__aarch64__)
    const auto& signed_grids = signed_iq2_grid_table();
#endif
    for (int group = 0; group < 8; ++group) {
        const uint8_t* packed = qs + group * 8;
        const uint32_t signs_scale = load_u32(packed + 4);
        const float scale = d * (0.5f + (signs_scale >> 28)) * 0.25f;
        for (int part = 0; part < 4; ++part) {
            const int sign_index = (signs_scale >> (7 * part)) & 127;
            float* y = out + group * 32 + part * 8;
#if defined(__aarch64__)
            const int8_t* signed_grid = signed_grids.data() +
                (packed[part] * 128 + sign_index) * 8;
            store_signed_iq_grid8(vld1_s8(signed_grid), scale, y);
#else
            const uint8_t* grid = iq2_grid + packed[part] * 8;
            const uint8_t signs = iq_signs[sign_index];
            for (int lane = 0; lane < 8; ++lane) {
                const int sign = 1 - 2 * int((signs >> lane) & 1);
                y[lane] = scale * (int(grid[lane]) * sign);
            }
#endif
        }
    }
}

void decode_iq3_xxs(const uint8_t* block, float* out) {
    const float d = half_value(block);
    const uint8_t* qs = block + 2;
    const uint8_t* scales_signs = qs + 64;
#if defined(__aarch64__)
    const auto& signed_grids = signed_iq3_grid_table();
#endif
    for (int group = 0; group < 8; ++group) {
        const uint32_t packed = load_u32(scales_signs + group * 4);
        const float scale = d * (0.5f + (packed >> 28)) * 0.5f;
        for (int part = 0; part < 4; ++part) {
            const uint8_t signs = iq_signs[(packed >> (7 * part)) & 127];
            float* y = out + group * 32 + part * 8;
#if defined(__aarch64__)
            const int8_t* grid0 = signed_grids.data() +
                (qs[group * 8 + part * 2] * 16 + (signs & 15)) * 4;
            const int8_t* grid1 = signed_grids.data() +
                (qs[group * 8 + part * 2 + 1] * 16 + (signs >> 4)) * 4;
            uint32_t low, high;
            std::memcpy(&low, grid0, sizeof(low));
            std::memcpy(&high, grid1, sizeof(high));
            store_signed_iq_grid8(vreinterpret_s8_u8(
                vcreate_u8(uint64_t(low) | (uint64_t(high) << 32))),
                scale, y);
#else
            const uint8_t* grid0 = iq3_grid + qs[group * 8 + part * 2] * 4;
            const uint8_t* grid1 = iq3_grid + qs[group * 8 + part * 2 + 1] * 4;
            for (int lane = 0; lane < 4; ++lane) {
                const int sign0 = 1 - 2 * int((signs >> lane) & 1);
                const int sign1 = 1 - 2 * int((signs >> (lane + 4)) & 1);
                y[lane] = scale * (int(grid0[lane]) * sign0);
                y[lane + 4] = scale * (int(grid1[lane]) * sign1);
            }
#endif
        }
    }
}

void decode_iq4_xs(const uint8_t* block, float* out) {
    static constexpr int8_t values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113};
    const float d = half_value(block);
    const uint16_t high = uint16_t(block[2]) | (uint16_t(block[3]) << 8);
    const uint8_t* low = block + 4;
    const uint8_t* qs = block + 8;
    for (int group = 0; group < 8; ++group) {
        const int scale_bits = ((low[group / 2] >> (4 * (group % 2))) & 15) |
                               (((high >> (2 * group)) & 3) << 4);
        const float scale = d * (scale_bits - 32);
        float* y = out + group * 32;
#if defined(__aarch64__)
        const uint8x16_t q = vld1q_u8(qs + group * 16);
        const int8x16_t table = vld1q_s8(values);
        const int8x16_t q0 = vqtbl1q_s8(table, vandq_u8(q, vdupq_n_u8(15)));
        const int8x16_t q1 = vqtbl1q_s8(table, vshrq_n_u8(q, 4));
        store_signed_iq_grid8(vget_low_s8(q0), scale, y);
        store_signed_iq_grid8(vget_high_s8(q0), scale, y + 8);
        store_signed_iq_grid8(vget_low_s8(q1), scale, y + 16);
        store_signed_iq_grid8(vget_high_s8(q1), scale, y + 24);
#else
        for (int lane = 0; lane < 16; ++lane) {
            const uint8_t q = qs[group * 16 + lane];
            y[lane] = scale * values[q & 15];
            y[lane + 16] = scale * values[q >> 4];
        }
#endif
    }
}

template <Precision P>
void decode_native_block(const uint8_t* block, float* out) {
    if constexpr (P == Precision::GGUF_IQ2_XXS) {
        decode_iq2_xxs(block, out);
    } else if constexpr (P == Precision::GGUF_IQ3_XXS) {
        decode_iq3_xxs(block, out);
    } else if constexpr (P == Precision::GGUF_STQ1_0) {
        stq1_0_dequantize_block(
            *reinterpret_cast<const block_stq1_0*>(block), out);
    } else if constexpr (P == Precision::GGUF_IQ4_XS) {
        decode_iq4_xs(block, out);
    } else if constexpr (P == Precision::GGUF_Q8_0) {
        const float scale = half_value(block);
        const auto* values = reinterpret_cast<const int8_t*>(block + 2);
        for (int i = 0; i < 32; ++i) out[i] = scale * values[i];
    } else if constexpr (P == Precision::GGUF_Q4_K) {
        decode_q4_or_q5(block, out, false);
    } else if constexpr (P == Precision::GGUF_Q5_K) {
        decode_q4_or_q5(block, out, true);
    } else if constexpr (P == Precision::GGUF_Q6_K) {
        decode_q6(block, out);
    }
}

#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
// IQ2 grid values are positive integers exactly representable in FP16.
// Keep magnitudes (4 KiB) separate from sign masks (2 KiB), replacing the
// 512 KiB signed-half table. XOR preserves the exact signed value before
// FP32 scaling and accumulation, including the original rounding order.
const std::array<__fp16, 256 * 8>& half_iq2_grid_table() {
    static const auto table = [] {
        std::array<__fp16, 256 * 8> result{};
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<__fp16>(iq2_grid[i]);
        return result;
    }();
    return table;
}
const std::array<uint16_t, 128 * 8>& half_iq_sign_table() {
    static const auto table = [] {
        std::array<uint16_t, 128 * 8> result{};
        for (int sign = 0; sign < 128; ++sign)
            for (int lane = 0; lane < 8; ++lane)
                result[sign * 8 + lane] = (iq_signs[sign] & (1 << lane)) ? 0x8000 : 0;
        return result;
    }();
    return table;
}

// IQ3 magnitudes need only 2 KiB; both formats share the sign-mask table.
const std::array<__fp16, 256 * 4>& half_iq3_grid_table() {
    static const auto table = [] {
        std::array<__fp16, 256 * 4> result{};
        const auto& integers = iq3_grid;
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<__fp16>(integers[i]);
        return result;
    }();
    return table;
}

// Decode IQ2/IQ3 directly into the eight-lane GEMV accumulators, avoiding a
// materialized FP32 block. Keep scale multiplication before the FMA so the
// rounding matches the generic parallel reduction below.
template <Precision P, int Rows>
void matmul_iq_gemv_rows(const uint8_t* encoded, const float* input,
                         float* output, int blocks, size_t bytes,
                         int begin, int end) {
    const uint16_t* half_signs = half_iq_sign_table().data();
    const __fp16* half_grids;
    if constexpr (P == Precision::GGUF_IQ2_XXS)
        half_grids = half_iq2_grid_table().data();
    else
        half_grids = half_iq3_grid_table().data();
    for (; begin < end; begin += Rows) {
        float32x4_t sum0[Rows], sum1[Rows];
        for (int r = 0; r < Rows; ++r)
            sum0[r] = sum1[r] = vdupq_n_f32(0.f);
        for (int block = 0; block < blocks; ++block) {
            const uint8_t* data[Rows];
            float d[Rows];
            for (int r = 0; r < Rows; ++r) {
                data[r] = encoded + (size_t(begin + r) * blocks + block) * bytes;
                d[r] = half_value(data[r]);
            }
            for (int group = 0; group < 8; ++group) {
                uint32_t packed[Rows];
                float scale[Rows];
                for (int r = 0; r < Rows; ++r) {
                    if constexpr (P == Precision::GGUF_IQ2_XXS) {
                        packed[r] = load_u32(data[r] + 2 + group * 8 + 4);
                        scale[r] = d[r] * (0.5f + (packed[r] >> 28)) * 0.25f;
                    } else {
                        packed[r] = load_u32(data[r] + 66 + group * 4);
                        scale[r] = d[r] * (0.5f + (packed[r] >> 28)) * 0.5f;
                    }
                }
                for (int part = 0; part < 4; ++part) {
                    const float* a = input + block * 256 + group * 32 + part * 8;
                    const float32x4_t x0 = vld1q_f32(a), x1 = vld1q_f32(a + 4);
#pragma unroll
                    for (int r = 0; r < Rows; ++r) {
                        const int sign = (packed[r] >> (7 * part)) & 127;
                        float32x4_t v0, v1;
                        if constexpr (P == Precision::GGUF_IQ2_XXS) {
                            const int code = data[r][2 + group * 8 + part];
                            const float16x8_t halves = vreinterpretq_f16_u16(veorq_u16(
                                vreinterpretq_u16_f16(vld1q_f16(half_grids + code * 8)),
                                vld1q_u16(half_signs + sign * 8)));
                            v0 = vmulq_n_f32(vcvt_f32_f16(
                                vget_low_f16(halves)), scale[r]);
                            v1 = vmulq_n_f32(vcvt_f32_f16(
                                vget_high_f16(halves)), scale[r]);
                        } else {
                            const uint8_t* qs = data[r] + 2 + group * 8 + part * 2;
                            const float16x8_t halves = vreinterpretq_f16_u16(veorq_u16(
                                vreinterpretq_u16_f16(vcombine_f16(
                                    vld1_f16(half_grids + qs[0] * 4),
                                    vld1_f16(half_grids + qs[1] * 4))),
                                vld1q_u16(half_signs + sign * 8)));
                            v0 = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(halves)), scale[r]);
                            v1 = vmulq_n_f32(vcvt_f32_f16(vget_high_f16(halves)), scale[r]);
                        }
                        sum0[r] = vfmaq_f32(sum0[r], x0, v0);
                        sum1[r] = vfmaq_f32(sum1[r], x1, v1);
                    }
                }
            }
        }
        for (int r = 0; r < Rows; ++r)
            output[begin + r] = vaddvq_f32(vaddq_f32(sum0[r], sum1[r]));
    }
}

// STQ groups are interleaved across four 16-element lanes. For ordinary
// decode inputs, accumulate each 256-value block before applying its shared
// scale. The original per-value scale order remains available for exceptions.
template <int Rows, bool BlockScale>
void matmul_stq_gemv_rows(const uint8_t* encoded, const float* input,
                          float* output, int blocks, size_t bytes,
                          int begin, int end) {
    const uint8x16x2_t book = {
        vld1q_u8(stq1_0_codebook), vld1q_u8(stq1_0_codebook + 16)};
    static const int8_t shifts[16] = {
        0, -1, -2, -3, -4, -5, -6, -7,
        0, -1, -2, -3, -4, -5, -6, -7};
    for (; begin < end; begin += Rows) {
        float32x4_t sum0[Rows], sum1[Rows];
        for (int r = 0; r < Rows; ++r)
            sum0[r] = sum1[r] = vdupq_n_f32(0.f);
        bool fallback = false;
        for (int block = 0; block < blocks; ++block) {
            float32x4_t block0[Rows], block1[Rows];
            if constexpr (BlockScale) {
                for (int r = 0; r < Rows; ++r)
                    block0[r] = block1[r] = vdupq_n_f32(0.f);
            }
            const uint8_t* data[Rows];
            float scale[Rows];
            for (int r = 0; r < Rows; ++r) {
                data[r] = encoded + (size_t(begin + r) * blocks + block) * bytes;
                scale[r] = half_value(data[r] + 40);
                // Match the STQ decoder's normalization of negative zero.
                if (scale[r] == 0.f) scale[r] = 0.f;
                if constexpr (BlockScale)
                    fallback |= !std::isfinite(scale[r]);
            }
            if constexpr (BlockScale) {
                if (fallback) {
                    matmul_stq_gemv_rows<Rows, false>(encoded, input, output,
                        blocks, bytes, begin, begin + Rows);
                    break;
                }
            }
            for (int chunk = 0; chunk < 4; ++chunk) {
                uint8x16_t packed[Rows];
                for (int r = 0; r < Rows; ++r) {
                    const uint8x8_t q = vld1_u8(data[r] + chunk * 8);
                    const uint8x8x2_t codes = vzip_u8(
                        vand_u8(q, vdup_n_u8(15)), vshr_n_u8(q, 4));
                    const uint8x16_t signs = vcombine_u8(
                        vdup_n_u8(data[r][32 + chunk * 2]),
                        vdup_n_u8(data[r][33 + chunk * 2]));
                    const uint8x16_t bits = vandq_u8(
                        vshlq_u8(signs, vld1q_s8(shifts)), vdupq_n_u8(1));
                    packed[r] = vqtbl2q_u8(book, vorrq_u8(
                        vcombine_u8(codes.val[0], codes.val[1]),
                        vshlq_n_u8(bits, 4)));
                }
                for (int lane = 0; lane < 4; ++lane) {
                    const float* a = input + block * 256 + chunk * 64 + lane * 16;
                    const float32x4_t x0 = vld1q_f32(a), x1 = vld1q_f32(a + 4);
                    const float32x4_t x2 = vld1q_f32(a + 8), x3 = vld1q_f32(a + 12);
#pragma unroll
                    for (int r = 0; r < Rows; ++r) {
                        const uint8x16_t codes = vandq_u8(
                            vshlq_u8(packed[r], vdupq_n_s8(-2 * lane)),
                            vdupq_n_u8(3));
                        // FP16 encodings of -1, 0, 1, 2 have zero low bytes.
                        // Look up their high bytes, then widen to exact FP32
                        // values before the original scale multiplication.
                        const uint8x16_t hi = vqtbl1q_u8(
                            vreinterpretq_u8_u32(vdupq_n_u32(0x403c00bc)), codes);
                        const float16x8_t low = vreinterpretq_f16_u16(
                            vshll_n_u8(vget_low_u8(hi), 8));
                        const float16x8_t high = vreinterpretq_f16_u16(
                            vshll_n_u8(vget_high_u8(hi), 8));
                        const auto v0 = vcvt_f32_f16(vget_low_f16(low));
                        const auto v1 = vcvt_f32_f16(vget_high_f16(low));
                        const auto v2 = vcvt_f32_f16(vget_low_f16(high));
                        const auto v3 = vcvt_f32_f16(vget_high_f16(high));
                        if constexpr (BlockScale) {
                            block0[r] = vfmaq_f32(block0[r], x0, v0);
                            block1[r] = vfmaq_f32(block1[r], x1, v1);
                            block0[r] = vfmaq_f32(block0[r], x2, v2);
                            block1[r] = vfmaq_f32(block1[r], x3, v3);
                        } else {
                            sum0[r] = vfmaq_f32(sum0[r], x0,
                                                 vmulq_n_f32(v0, scale[r]));
                            sum1[r] = vfmaq_f32(sum1[r], x1,
                                                 vmulq_n_f32(v1, scale[r]));
                            sum0[r] = vfmaq_f32(sum0[r], x2,
                                                 vmulq_n_f32(v2, scale[r]));
                            sum1[r] = vfmaq_f32(sum1[r], x3,
                                                 vmulq_n_f32(v3, scale[r]));
                        }
                    }
                }
            }
            if constexpr (BlockScale) {
                for (int r = 0; r < Rows; ++r) {
                    sum0[r] = vfmaq_n_f32(sum0[r], block0[r], scale[r]);
                    sum1[r] = vfmaq_n_f32(sum1[r], block1[r], scale[r]);
                }
            }
        }
        if (fallback) continue;
        for (int r = 0; r < Rows; ++r)
            output[begin + r] = vaddvq_f32(vaddq_f32(sum0[r], sum1[r]));
    }
}

// Finite FP16 d * 6-bit scale * 5-bit q needs at most 22 significant bits,
// within FP32's 24 bits, and cannot overflow or underflow FP32. Its product
// is therefore exact: fusing only the final subtraction preserves rounding.
// Nonfinite block scales retain separate operations and their NaN behavior.
template <bool Fused>
inline float32x4_t q5_reconstruct(float32x4_t q, float scale, float32x4_t minimum) {
    if constexpr (Fused)
        return vfmaq_n_f32(vnegq_f32(minimum), q, scale);
    else
        return vsubq_f32(vmulq_n_f32(q, scale), minimum);
}

// Q5 uses the same eight-lane reduction as the parallel GEMV path, but
// decodes each 16-value slice directly into the FMA instead of an FP32 block.
template <int Rows>
void matmul_q5_gemv_rows(const uint8_t* encoded, const float* input, float* output, int blocks,
                         size_t bytes, int begin, int end) {
    for (; begin < end; begin += Rows) {
        float32x4_t sum0[Rows], sum1[Rows];
        for (int r = 0; r < Rows; ++r)
            sum0[r] = sum1[r] = vdupq_n_f32(0.f);
        for (int block = 0; block < blocks; ++block) {
            const uint8_t* data[Rows];
            float d[Rows], dmin[Rows];
            bool finite_scales = true;
            for (int r = 0; r < Rows; ++r) {
                data[r] = encoded + (size_t(begin + r) * blocks + block) * bytes;
                d[r] = half_value(data[r]);
                dmin[r] = half_value(data[r] + 2);
                finite_scales &= (data[r][1] & 0x7c) != 0x7c && (data[r][3] & 0x7c) != 0x7c;
            }
            auto accumulate = [&](auto tag) {
                constexpr bool fuse = decltype(tag)::value;
                // Fixed groups let shifts and packed scale extraction specialize.
#pragma unroll
                for (int group = 0; group < 8; ++group) {
                    float scale[Rows], minimum[Rows];
                    for (int r = 0; r < Rows; ++r) {
                        uint8_t sc, mn;
                        scale_min(group, data[r] + 4, sc, mn);
                        scale[r] = d[r] * sc;
                        minimum[r] = dmin[r] * mn;
                    }
                    for (int lane = 0; lane < 32; lane += 16) {
                        const float* a = input + block * 256 + group * 32 + lane;
                        const float32x4_t x0 = vld1q_f32(a), x1 = vld1q_f32(a + 4);
                        const float32x4_t x2 = vld1q_f32(a + 8), x3 = vld1q_f32(a + 12);
#pragma unroll
                        for (int r = 0; r < Rows; ++r) {
                            const uint8x16_t q = vld1q_u8(data[r] + 48 + (group / 2) * 32 + lane);
                            const uint8x16_t hi = vld1q_u8(data[r] + 16 + lane);
                            const uint8x16_t low = vandq_u8(
                                vshlq_u8(q, vdupq_n_s8((group & 1) ? -4 : 0)), vdupq_n_u8(15));
                            const uint8x16_t values =
                                vaddq_u8(low, vandq_u8(vtstq_u8(hi, vdupq_n_u8(1 << group)),
                                                       vdupq_n_u8(16)));
                            const uint16x8_t v0 = vmovl_u8(vget_low_u8(values));
                            const uint16x8_t v1 = vmovl_u8(vget_high_u8(values));
                            const float32x4_t mn = vdupq_n_f32(minimum[r]);
                            sum0[r] = vfmaq_f32(
                                sum0[r], x0,
                                q5_reconstruct<fuse>(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v0))),
                                                     scale[r], mn));
                            sum1[r] = vfmaq_f32(
                                sum1[r], x1,
                                q5_reconstruct<fuse>(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v0))),
                                                     scale[r], mn));
                            sum0[r] = vfmaq_f32(
                                sum0[r], x2,
                                q5_reconstruct<fuse>(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v1))),
                                                     scale[r], mn));
                            sum1[r] = vfmaq_f32(
                                sum1[r], x3,
                                q5_reconstruct<fuse>(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v1))),
                                                     scale[r], mn));
                        }
                    }
                    // Keep expanded groups from overlapping register lifetimes.
                    // Clang otherwise spills decoded vectors across groups. This
                    // compiler-only boundary emits no CPU fence; the accumulators
                    // preserve the existing FMA order. The surrounding helper is
                    // restricted to Apple Clang on ARM64.
                    if constexpr (Rows == 2) {
                        asm volatile(""
                                     : "+w"(sum0[0]), "+w"(sum1[0]), "+w"(sum0[1]), "+w"(sum1[1])
                                     :
                                     : "memory");
                    } else {
                        asm volatile("" : "+w"(sum0[0]), "+w"(sum1[0]) : : "memory");
                    }
                }
            };
            if (finite_scales)
                accumulate(std::true_type{});
            else
                accumulate(std::false_type{});
        }
        for (int r = 0; r < Rows; ++r)
            output[begin + r] = vaddvq_f32(vaddq_f32(sum0[r], sum1[r]));
    }
}

template <Precision P, int Groups>
void matmul_token_rows(const uint8_t* encoded, const float* input, float* output,
                       int M, int N, int blocks, int block_values, size_t bytes,
                       int begin, int end) {
    // Keep independent output sums in registers. Small token batches need
    // more output rows in flight to hide the floating-point add latency.
    // SIMD lanes hold independent tokens: preserve each output's k order
    // and round products separately from additions, as in the scalar path.
    constexpr int Rows = Groups <= 2 ? 4 : 2;
    constexpr int Stride = Groups * 4;
    std::array<std::array<float, 256>, Rows> values;
    for (; begin + Rows <= end; begin += Rows) {
        float32x4_t sums[Rows][Groups];
        for (int r = 0; r < Rows; ++r)
            for (int g = 0; g < Groups; ++g)
                sums[r][g] = vdupq_n_f32(0.0f);
        for (int block = 0; block < blocks; ++block) {
            for (int r = 0; r < Rows; ++r) {
                const uint8_t* row = encoded +
                    (size_t(begin + r) * blocks + block) * bytes;
                decode_native_block<P>(row, values[r].data());
            }
            const float* a = input + size_t(block) * block_values * Stride;
            {
#pragma clang fp contract(off)
                for (int k = 0; k < block_values; ++k) {
#pragma unroll
                    for (int g = 0; g < Groups; ++g) {
                        const float32x4_t x = vld1q_f32(a + k * Stride + g * 4);
#pragma unroll
                        for (int r = 0; r < Rows; ++r)
                            sums[r][g] = vaddq_f32(sums[r][g],
                                vmulq_n_f32(x, values[r][k]));
                    }
                }
            }
        }
        for (int r = 0; r < Rows; ++r) {
            float row_sums[Stride];
            for (int g = 0; g < Groups; ++g)
                vst1q_f32(row_sums + g * 4, sums[r][g]);
            for (int m = 0; m < M; ++m)
                output[m * N + begin + r] = row_sums[m];
        }
    }
    for (; begin < end; ++begin) {
        float32x4_t sums[Groups];
        for (int g = 0; g < Groups; ++g)
            sums[g] = vdupq_n_f32(0.0f);
        const uint8_t* row = encoded + size_t(begin) * blocks * bytes;
        for (int block = 0; block < blocks; ++block) {
            decode_native_block<P>(row + block * bytes, values[0].data());
            const float* a = input + size_t(block) * block_values * Stride;
            {
#pragma clang fp contract(off)
                for (int k = 0; k < block_values; ++k) {
#pragma unroll
                    for (int g = 0; g < Groups; ++g)
                        sums[g] = vaddq_f32(sums[g], vmulq_n_f32(
                            vld1q_f32(a + k * Stride + g * 4), values[0][k]));
                }
            }
        }
        float row_sums[Stride];
        for (int g = 0; g < Groups; ++g)
            vst1q_f32(row_sums + g * 4, sums[g]);
        for (int m = 0; m < M; ++m)
            output[m * N + begin] = row_sums[m];
    }
}
#endif

#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
// Sum each Q8 block before applying its shared scale. This changes reduction
// rounding, but avoids materializing dequantized weights. The caller excludes
// extreme/non-finite activations; exceptional scales retain scalar IEEE order.
template <int Rows>
void q8_block_scale(const uint8_t *w, const float *x, float *y, int blocks, int begin, int end) {
#pragma clang fp contract(off)
    for (; begin < end; begin += Rows) {
        float totals[Rows]{};
        bool exceptional_scale[Rows]{};
        for (int b = 0; b < blocks; ++b) {
            float32x4_t s0[Rows], s1[Rows];
            for (int r = 0; r < Rows; ++r)
                s0[r] = s1[r] = vdupq_n_f32(0.f);
            for (int k = 0; k < 32; k += 16) {
                auto x0 = vld1q_f32(x + b * 32 + k), x1 = vld1q_f32(x + b * 32 + k + 4);
                auto x2 = vld1q_f32(x + b * 32 + k + 8), x3 = vld1q_f32(x + b * 32 + k + 12);
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    const auto *p = w + (size_t(begin + r) * blocks + b) * 34 + 2 + k;
                    auto q = vld1q_s8(reinterpret_cast<const int8_t *>(p));
                    auto lo = vmovl_s8(vget_low_s8(q)), hi = vmovl_s8(vget_high_s8(q));
                    s0[r] = vfmaq_f32(s0[r], x0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
                    s1[r] = vfmaq_f32(s1[r], x1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
                    s0[r] = vfmaq_f32(s0[r], x2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
                    s1[r] = vfmaq_f32(s1[r], x3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
                }
            }
            for (int r = 0; r < Rows; ++r) {
                const auto *p = w + (size_t(begin + r) * blocks + b) * 34;
                uint16_t bits;
                std::memcpy(&bits, p, sizeof(bits));
                exceptional_scale[r] |= (bits & 0x7c00) == 0x7c00;
                totals[r] += half_value(p) * vaddvq_f32(vaddq_f32(s0[r], s1[r]));
            }
        }
        for (int r = 0; r < Rows; ++r) {
            if (exceptional_scale[r]) {
                // Preserve scalar-k IEEE behavior for non-finite scales.
                float sum = 0.f;
                for (int b = 0; b < blocks; ++b) {
                    const auto *p = w + (size_t(begin + r) * blocks + b) * 34;
                    const float d = half_value(p);
                    for (int k = 0; k < 32; ++k) {
                        int8_t q;
                        std::memcpy(&q, p + 2 + k, 1);
                        const float weight = d * float(q);
                        sum += weight * x[b * 32 + k];
                    }
                }
                totals[r] = sum;
            }
            y[begin + r] = totals[r];
        }
    }
}

#endif

template <Precision P>
void matmul_native_blocks(const Tensor &A, const Tensor &B, Tensor &C, ThreadPool *pool,
                          Activation act, int act_n_begin, int act_n_len, int block_values,
                          size_t bytes) {
    const int K = static_cast<int>(B.shape[1]);
    const int N = static_cast<int>(B.shape[0]);
    const int M = static_cast<int>(A.shape[1]);
    const int blocks = K / block_values;
#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
    if (M > 32) {
        // Reuse the SIMD token tiles for larger prefill batches. A final
        // singleton tile uses the format-specific GEMV reduction.
        for (int start = 0; start < M; start += 32) {
            Tensor input_tile = A;
            Tensor output_tile = C;
            input_tile.shape[1] = output_tile.shape[1] = std::min(32, M - start);
            input_tile.compute_strides();
            output_tile.compute_strides();
            input_tile.data = static_cast<float*>(A.data) + size_t(start) * K;
            output_tile.data = C.ptr<float>() + size_t(start) * N;
            matmul_native_blocks<P>(input_tile, B, output_tile, pool, act,
                                    act_n_begin, act_n_len, block_values, bytes);
        }
        return;
    }
#endif
    const auto* encoded = static_cast<const uint8_t*>(B.data);
    const float* input = A.ptr<float>();
    float* output = C.ptr<float>();
#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
    const int token_stride = (M + 3) & ~3;
    std::vector<float> token_major;
    if (M >= 2 && M <= 32) {
        // Make adjacent tokens contiguous so NEON can update four independent
        // sums while each token still visits k in the original order.
        token_major.resize(size_t(K) * token_stride, 0.0f);
        for (int k = 0; k < K; ++k)
            for (int m = 0; m < M; ++m)
                token_major[size_t(k) * token_stride + m] =
                    input[size_t(m) * K + k];
    }
    // Keep unscaled block products away from underflow/overflow. Values outside
    // this conservative interval use the existing ordered implementation.
    bool q8_scale_safe = false;
    if constexpr (P == Precision::GGUF_Q8_0) {
        if (M == 1) {
            q8_scale_safe = true;
            for (int k = 0; k < K; ++k) {
                const float value = input[k];
                const float magnitude = value < 0.f ? -value : value;
                if (!(value == 0.f || (magnitude >= 0x1p-60f && magnitude <= 0x1p60f))) {
                    q8_scale_safe = false;
                    break;
                }
            }
        }
    }
    bool stq_block_safe = false;
    if constexpr (P == Precision::GGUF_STQ1_0) {
        if (M == 1) {
            stq_block_safe = true;
            for (int k = 0; k < K; ++k) {
                const float value = input[k];
                const float magnitude = value < 0.f ? -value : value;
                if (!(value == 0.f ||
                      (magnitude >= 0x1p-60f && magnitude <= 0x1p60f))) {
                    stq_block_safe = false;
                    break;
                }
            }
        }
    }
#endif
    auto process = [&](int, int begin, int end) {
        std::array<float, 256> values;
#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
        if (!token_major.empty()) {
            switch (token_stride / 4) {
            case 1:
                matmul_token_rows<P, 1>(encoded, token_major.data(), output, M, N, blocks,
                                        block_values, bytes, begin, end);
                break;
            case 2:
                matmul_token_rows<P, 2>(encoded, token_major.data(), output, M, N, blocks,
                                        block_values, bytes, begin, end);
                break;
            case 3:
                matmul_token_rows<P, 3>(encoded, token_major.data(), output, M, N, blocks,
                                        block_values, bytes, begin, end);
                break;
            case 4:
                matmul_token_rows<P, 4>(encoded, token_major.data(), output, M, N, blocks,
                                        block_values, bytes, begin, end);
                break;
            case 5:
                matmul_token_rows<P, 5>(encoded, token_major.data(), output, M, N, blocks,
                                        block_values, bytes, begin, end);
                break;
            case 6:
                matmul_token_rows<P, 6>(encoded, token_major.data(), output, M, N, blocks,
                                        block_values, bytes, begin, end);
                break;
            case 7:
                matmul_token_rows<P, 7>(encoded, token_major.data(), output, M, N, blocks,
                                        block_values, bytes, begin, end);
                break;
            case 8:
                matmul_token_rows<P, 8>(encoded, token_major.data(), output, M, N, blocks,
                                        block_values, bytes, begin, end);
                break;
            }
            return;
        }
        if (M == 1 && (P == Precision::GGUF_IQ2_XXS ||
                       P == Precision::GGUF_IQ3_XXS)) {
            const int aligned_end = begin + (end - begin) / 4 * 4;
            matmul_iq_gemv_rows<P, 4>(encoded, input, output, blocks,
                bytes, begin, aligned_end);
            matmul_iq_gemv_rows<P, 1>(encoded, input, output, blocks,
                bytes, aligned_end, end);
            return;
        }
        if (M == 1 && P == Precision::GGUF_STQ1_0) {
            const int aligned_end = begin + (end - begin) / 2 * 2;
            if (stq_block_safe) {
                matmul_stq_gemv_rows<2, true>(encoded, input, output, blocks,
                    bytes, begin, aligned_end);
                matmul_stq_gemv_rows<1, true>(encoded, input, output, blocks,
                    bytes, aligned_end, end);
            } else {
                matmul_stq_gemv_rows<2, false>(encoded, input, output, blocks,
                    bytes, begin, aligned_end);
                matmul_stq_gemv_rows<1, false>(encoded, input, output, blocks,
                    bytes, aligned_end, end);
            }
            return;
        }
        if (M == 1 && P == Precision::GGUF_Q5_K) {
            const int aligned_end = begin + (end - begin) / 2 * 2;
            matmul_q5_gemv_rows<2>(encoded, input, output, blocks,
                bytes, begin, aligned_end);
            matmul_q5_gemv_rows<1>(encoded, input, output, blocks,
                bytes, aligned_end, end);
            return;
        }
        // Eight independent K lanes shorten the Q6 decode reduction chain.
        // Dequantization is unchanged; FP32 FMA and the final horizontal sum
        // intentionally differ from the scalar K-order accumulation.
        if (M == 1 && P == Precision::GGUF_Q6_K) {
            for (int n = begin; n < end; ++n) {
                float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
                for (int block = 0; block < blocks; ++block) {
                    decode_native_block<P>(encoded + (size_t(n) * blocks + block) * bytes, values.data());
                    const float* x = input + block * block_values;
                    for (int k = 0; k < block_values; k += 8) {
                        a0 = vfmaq_f32(a0, vld1q_f32(x + k), vld1q_f32(values.data() + k));
                        a1 = vfmaq_f32(a1, vld1q_f32(x + k + 4), vld1q_f32(values.data() + k + 4));
                    }
                }
                output[n] = vaddvq_f32(vaddq_f32(a0, a1));
            }
            return;
        }
        if (M == 1 && P == Precision::GGUF_Q8_0 && q8_scale_safe) {
            const int mid = begin + (end - begin) / 4 * 4;
            q8_block_scale<4>(encoded, input, output, blocks, begin, mid);
            q8_block_scale<1>(encoded, input, output, blocks, mid, end);
            return;
        }
        if (M == 1) {
            // Transpose four values from each row in registers so SIMD lanes
            // hold independent output rows. Each lane keeps the original k
            // order and rounds each product separately from the addition.
            std::array<std::array<float, 256>, 8> row_values;
            for (; begin + 8 <= end; begin += 8) {
                float32x4_t sums[2] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
                for (int block = 0; block < blocks; ++block) {
                    for (int lane = 0; lane < 8; ++lane) {
                        const uint8_t* row = encoded +
                            (size_t(begin + lane) * blocks + block) * bytes;
                        decode_native_block<P>(row, row_values[lane].data());
                    }
                    const float* a = input + block * block_values;
                    {
#pragma clang fp contract(off)
                        for (int k = 0; k < block_values; k += 4) {
                            const float32x4_t x = vld1q_f32(a + k);
#pragma unroll
                            for (int g = 0; g < 2; ++g) {
                                const float32x4_t r0 = vld1q_f32(row_values[g * 4].data() + k);
                                const float32x4_t r1 = vld1q_f32(row_values[g * 4 + 1].data() + k);
                                const float32x4_t r2 = vld1q_f32(row_values[g * 4 + 2].data() + k);
                                const float32x4_t r3 = vld1q_f32(row_values[g * 4 + 3].data() + k);
                                const float32x4_t t0 = vtrn1q_f32(r0, r1);
                                const float32x4_t t1 = vtrn2q_f32(r0, r1);
                                const float32x4_t t2 = vtrn1q_f32(r2, r3);
                                const float32x4_t t3 = vtrn2q_f32(r2, r3);
                                const float32x4_t v0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
                                const float32x4_t v1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
                                const float32x4_t v2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
                                const float32x4_t v3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));
                                sums[g] = vaddq_f32(sums[g], vmulq_laneq_f32(v0, x, 0));
                                sums[g] = vaddq_f32(sums[g], vmulq_laneq_f32(v1, x, 1));
                                sums[g] = vaddq_f32(sums[g], vmulq_laneq_f32(v2, x, 2));
                                sums[g] = vaddq_f32(sums[g], vmulq_laneq_f32(v3, x, 3));
                            }
                        }
                    }
                }
                vst1q_f32(output + begin, sums[0]);
                vst1q_f32(output + begin + 4, sums[1]);
            }
        }
#endif
        for (int n = begin; n < end; ++n) {
            for (int m = 0; m < M; ++m)
                output[m * N + n] = 0.0f;
            const uint8_t* row = encoded + size_t(n) * blocks * bytes;
            for (int block = 0; block < blocks; ++block) {
                decode_native_block<P>(row + block * bytes, values.data());
                for (int m = 0; m < M; ++m) {
                    const float* a = input + size_t(m) * K + block * block_values;
                    float sum = output[m * N + n];
                    for (int k = 0; k < block_values; ++k)
                        sum += a[k] * values[k];
                    output[m * N + n] = sum;
                }
            }
        }
    };
    if (pool) {
        // Rows are independent. Let faster cores take more work on Macs with
        // mixed core speeds. Each row uses a scheduling-independent reduction.
        pool->parallel_for_2d(1, 1, N, 64,
                              [&](int thread_id, int, int, int begin, int end) {
                                  process(thread_id, begin, end);
                              });
    }
    else process(0, 0, N);
    matmul_apply_activation(output, M, N, N, 0, M, act,
                            act_n_begin, act_n_len);
}

}  // namespace

size_t gguf_k_block_bytes(Precision precision) {
    switch (precision) {
    case Precision::GGUF_Q8_0: return 34;
    case Precision::GGUF_IQ2_XXS: return 66;
    case Precision::GGUF_IQ3_XXS: return 98;
    case Precision::GGUF_STQ1_0: return STQ_BLOCK_BYTES;
    case Precision::GGUF_IQ4_XS: return 136;
    case Precision::GGUF_Q4_K: return 144;
    case Precision::GGUF_Q5_K: return 176;
    case Precision::GGUF_Q6_K: return 210;
    default: return 0;
    }
}

size_t gguf_k_block_values(Precision precision) {
    return precision == Precision::GGUF_Q8_0 ? 32 :
           gguf_k_block_bytes(precision) ? 256 : 0;
}

bool gguf_k_dequantize_block(Precision precision, const uint8_t* block,
                             float* output) {
    if (!block || !output) return false;
    switch (precision) {
    case Precision::GGUF_IQ2_XXS:
        decode_native_block<Precision::GGUF_IQ2_XXS>(block, output); return true;
    case Precision::GGUF_IQ3_XXS:
        decode_native_block<Precision::GGUF_IQ3_XXS>(block, output); return true;
    case Precision::GGUF_STQ1_0:
        decode_native_block<Precision::GGUF_STQ1_0>(block, output); return true;
    case Precision::GGUF_IQ4_XS:
        decode_native_block<Precision::GGUF_IQ4_XS>(block, output); return true;
    case Precision::GGUF_Q8_0:
        decode_native_block<Precision::GGUF_Q8_0>(block, output); return true;
    case Precision::GGUF_Q4_K:
        decode_native_block<Precision::GGUF_Q4_K>(block, output); return true;
    case Precision::GGUF_Q5_K:
        decode_native_block<Precision::GGUF_Q5_K>(block, output); return true;
    case Precision::GGUF_Q6_K:
        decode_native_block<Precision::GGUF_Q6_K>(block, output); return true;
    default: return false;
    }
}

void kernel_matmul_gguf_k(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* pool, Activation act,
                          int act_n_begin, int act_n_len) {
    const int K = static_cast<int>(B.shape[1]);
    const int N = static_cast<int>(B.shape[0]);
    const int M = static_cast<int>(A.shape[1]);
    const size_t bytes = gguf_k_block_bytes(B.prec);
    const int block_values = static_cast<int>(gguf_k_block_values(B.prec));
    if (!bytes || !block_values || K % block_values || A.prec != Precision::FP32 ||
        C.prec != Precision::FP32 || A.shape[0] != K || C.shape[0] != N ||
        C.shape[1] != M || !A.data || !B.data || !C.data)
        return;
    switch (B.prec) {
    case Precision::GGUF_IQ2_XXS:
        matmul_native_blocks<Precision::GGUF_IQ2_XXS>(
            A, B, C, pool, act, act_n_begin, act_n_len, block_values, bytes); break;
    case Precision::GGUF_IQ3_XXS:
        matmul_native_blocks<Precision::GGUF_IQ3_XXS>(
            A, B, C, pool, act, act_n_begin, act_n_len, block_values, bytes); break;
    case Precision::GGUF_STQ1_0:
        matmul_native_blocks<Precision::GGUF_STQ1_0>(
            A, B, C, pool, act, act_n_begin, act_n_len, block_values, bytes); break;
    case Precision::GGUF_IQ4_XS:
        matmul_native_blocks<Precision::GGUF_IQ4_XS>(
            A, B, C, pool, act, act_n_begin, act_n_len, block_values, bytes); break;
    case Precision::GGUF_Q8_0:
        matmul_native_blocks<Precision::GGUF_Q8_0>(
            A, B, C, pool, act, act_n_begin, act_n_len, block_values, bytes); break;
    case Precision::GGUF_Q4_K:
        matmul_native_blocks<Precision::GGUF_Q4_K>(
            A, B, C, pool, act, act_n_begin, act_n_len, block_values, bytes); break;
    case Precision::GGUF_Q5_K:
        matmul_native_blocks<Precision::GGUF_Q5_K>(
            A, B, C, pool, act, act_n_begin, act_n_len, block_values, bytes); break;
    case Precision::GGUF_Q6_K:
        matmul_native_blocks<Precision::GGUF_Q6_K>(
            A, B, C, pool, act, act_n_begin, act_n_len, block_values, bytes); break;
    default: break;
    }
}

bool kernel_matmul_gguf_k_gemv_batch(const std::vector<Tensor>& inputs,
                                     const std::vector<Tensor>& weights,
                                     std::vector<Tensor>& outputs,
                                     ThreadPool* pool) {
    if (!pool || inputs.size() < 2 || inputs.size() != weights.size() ||
        inputs.size() != outputs.size())
        return false;
    const Precision precision = weights[0].prec;
    const int K = static_cast<int>(weights[0].shape[1]);
    const int N = static_cast<int>(weights[0].shape[0]);
    const size_t bytes = gguf_k_block_bytes(precision);
    const size_t block_values = gguf_k_block_values(precision);
    if (!bytes || !block_values || K <= 0 || N <= 0 ||
        K % block_values != 0)
        return false;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (weights[i].prec != precision || weights[i].shape[0] != N ||
            weights[i].shape[1] != K || inputs[i].prec != Precision::FP32 ||
            inputs[i].shape[0] != K || inputs[i].shape[1] != 1 ||
            outputs[i].prec != Precision::FP32 || outputs[i].shape[0] != N ||
            outputs[i].shape[1] != 1 || !inputs[i].data ||
            !weights[i].data || !outputs[i].data)
            return false;
    }

    const size_t row_bytes = static_cast<size_t>(K) / block_values * bytes;
    pool->parallel_for_2d(static_cast<int>(inputs.size()), 1, N, 64,
                          [&](int, int expert, int expert_end,
                              int begin, int end) {
        for (; expert < expert_end; ++expert) {
            Tensor weight = weights[expert];
            Tensor output = outputs[expert];
            weight.shape[0] = end - begin;
            output.shape[0] = end - begin;
            weight.data = static_cast<uint8_t*>(weight.data) +
                          static_cast<size_t>(begin) * row_bytes;
            output.data = static_cast<float*>(output.data) + begin;
            kernel_matmul_gguf_k(inputs[expert], weight, output, nullptr,
                                 Activation::NONE, 0, 0);
        }
    });
    return true;
}
