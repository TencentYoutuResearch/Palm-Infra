#include "kernels/cpu/matmul/gguf_k.h"
#include "kernels/cpu/matmul/gguf_iq_tables.h"
#include "kernels/cpu/matmul/stq1_0.h"
#include "kernels/cpu/matmul/matmul_internal.h"
#include "core/fp16.h"
#include "runtime/threading.h"

#include <algorithm>
#include <array>
#include <cstring>

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
    }
}

uint32_t load_u32(const uint8_t* bytes) {
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
           (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

void decode_iq2_xxs(const uint8_t* block, float* out) {
    const float d = half_value(block);
    const uint8_t* qs = block + 2;
    for (int group = 0; group < 8; ++group) {
        const uint8_t* packed = qs + group * 8;
        const uint32_t signs_scale = load_u32(packed + 4);
        const float scale = d * (0.5f + (signs_scale >> 28)) * 0.25f;
        for (int part = 0; part < 4; ++part) {
            const uint8_t* grid = iq2_grid + packed[part] * 8;
            const uint8_t signs = iq_signs[(signs_scale >> (7 * part)) & 127];
            float* y = out + group * 32 + part * 8;
            for (int lane = 0; lane < 8; ++lane) {
                const int sign = 1 - 2 * int((signs >> lane) & 1);
                y[lane] = scale * (int(grid[lane]) * sign);
            }
        }
    }
}

void decode_iq3_xxs(const uint8_t* block, float* out) {
    const float d = half_value(block);
    const uint8_t* qs = block + 2;
    const uint8_t* scales_signs = qs + 64;
    for (int group = 0; group < 8; ++group) {
        const uint32_t packed = load_u32(scales_signs + group * 4);
        const float scale = d * (0.5f + (packed >> 28)) * 0.5f;
        for (int part = 0; part < 4; ++part) {
            const uint8_t signs = iq_signs[(packed >> (7 * part)) & 127];
            const uint8_t* grid0 = iq3_grid + qs[group * 8 + part * 2] * 4;
            const uint8_t* grid1 = iq3_grid + qs[group * 8 + part * 2 + 1] * 4;
            float* y = out + group * 32 + part * 8;
            for (int lane = 0; lane < 4; ++lane) {
                const int sign0 = 1 - 2 * int((signs >> lane) & 1);
                const int sign1 = 1 - 2 * int((signs >> (lane + 4)) & 1);
                y[lane] = scale * (int(grid0[lane]) * sign0);
                y[lane + 4] = scale * (int(grid1[lane]) * sign1);
            }
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
    if (pool) pool->parallel_for(0, N, 16, process);
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
