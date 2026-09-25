#include "kernels/cpu/matmul/gguf_k.h"
#include "kernels/cpu/matmul/gguf_iq_tables.h"
#include "kernels/cpu/matmul/stq1_0.h"
#include "kernels/cpu/matmul/matmul_internal.h"
#include "core/fp16.h"
#include "runtime/threading.h"

#include <algorithm>
#include <array>
#include <cstring>

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
        for (int lane = 0; lane < 16; ++lane) {
            const uint8_t q = qs[group * 16 + lane];
            y[lane] = scale * values[q & 15];
            y[lane + 16] = scale * values[q >> 4];
        }
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

template <Precision P>
void matmul_native_blocks(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* pool, Activation act,
                          int act_n_begin, int act_n_len,
                          int block_values, size_t bytes) {
    const int K = static_cast<int>(B.shape[1]);
    const int N = static_cast<int>(B.shape[0]);
    const int M = static_cast<int>(A.shape[1]);
    const int blocks = K / block_values;
    const auto* encoded = static_cast<const uint8_t*>(B.data);
    const float* input = A.ptr<float>();
    float* output = C.ptr<float>();
    auto process = [&](int, int begin, int end) {
        std::array<float, 256> values;
#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
        if (M == 1) {
            // Four independent rows hide the scalar reduction dependency.
            // Clang otherwise fuses the interleaved multiply/add operations,
            // changing Hy4 logits; the original kernel rounds each product.
            std::array<std::array<float, 256>, 4> row_values;
            for (; begin + 4 <= end; begin += 4) {
                float sum0 = 0.0f, sum1 = 0.0f;
                float sum2 = 0.0f, sum3 = 0.0f;
                for (int block = 0; block < blocks; ++block) {
                    for (int lane = 0; lane < 4; ++lane) {
                        const uint8_t* row = encoded +
                            (size_t(begin + lane) * blocks + block) * bytes;
                        decode_native_block<P>(row, row_values[lane].data());
                    }
                    const float* a = input + block * block_values;
                    {
#pragma clang fp contract(off)
                        for (int k = 0; k < block_values; ++k) {
                            const float x = a[k];
                            sum0 += x * row_values[0][k];
                            sum1 += x * row_values[1][k];
                            sum2 += x * row_values[2][k];
                            sum3 += x * row_values[3][k];
                        }
                    }
                }
                output[begin] = sum0;
                output[begin + 1] = sum1;
                output[begin + 2] = sum2;
                output[begin + 3] = sum3;
            }
        }
        if (M >= 2 && M <= 32) {
            // Reuse each decoded block across prefill tokens. Independent
            // accumulators retain the original per-token rounding order.
            std::array<float, 32> sums;
            for (int n = begin; n < end; ++n) {
                sums.fill(0.0f);
                const uint8_t* row = encoded + size_t(n) * blocks * bytes;
                for (int block = 0; block < blocks; ++block) {
                    decode_native_block<P>(row + block * bytes, values.data());
                    for (int m = 0; m + 4 <= M; m += 4) {
                        float sum0 = sums[m], sum1 = sums[m + 1];
                        float sum2 = sums[m + 2], sum3 = sums[m + 3];
                        const float* a0 = input + size_t(m) * K + block * block_values;
                        const float* a1 = a0 + K;
                        const float* a2 = a1 + K;
                        const float* a3 = a2 + K;
                        {
#pragma clang fp contract(off)
                            for (int k = 0; k < block_values; ++k) {
                                const float weight = values[k];
                                sum0 += a0[k] * weight;
                                sum1 += a1[k] * weight;
                                sum2 += a2[k] * weight;
                                sum3 += a3[k] * weight;
                            }
                        }
                        sums[m] = sum0; sums[m + 1] = sum1;
                        sums[m + 2] = sum2; sums[m + 3] = sum3;
                    }
                    for (int m = (M / 4) * 4; m + 2 <= M; m += 2) {
                        float sum0 = sums[m], sum1 = sums[m + 1];
                        const float* a0 = input + size_t(m) * K + block * block_values;
                        const float* a1 = a0 + K;
                        {
#pragma clang fp contract(off)
                            for (int k = 0; k < block_values; ++k) {
                                const float weight = values[k];
                                sum0 += a0[k] * weight;
                                sum1 += a1[k] * weight;
                            }
                        }
                        sums[m] = sum0; sums[m + 1] = sum1;
                    }
                    for (int m = (M / 2) * 2; m < M; ++m) {
                        const float* a = input + size_t(m) * K + block * block_values;
                        float sum = sums[m];
                        for (int k = 0; k < block_values; ++k)
                            sum += a[k] * values[k];
                        sums[m] = sum;
                    }
                }
                for (int m = 0; m < M; ++m) output[m * N + n] = sums[m];
            }
            return;
        }
#endif
        for (int n = begin; n < end; ++n) {
            for (int m = 0; m < M; ++m) output[m * N + n] = 0.0f;
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
        // mixed core speeds, without changing the reduction order per row.
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
