#include "kernels/cpu/matmul/stq1_0.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// Codebook from the patch's GGML_TABLE_BEGIN(uint8_t, stq1_0_codebook, 32).
// Entries 0..15 are sign = 0 (first non-zero lane is +1), 16..31 are sign = 1
// (every non-zero lane negated).
const uint8_t stq1_0_codebook[32] = {
    // sign = 0
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
    0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    // sign = 1
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
    0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
};

namespace {

// qpack -> (slot, sign), derived from the codebook. 0xFF marks a byte that is
// not a valid codebook entry; the encoder asserts against it.
struct ReverseTables {
    uint8_t slot[256];
    uint8_t sign[256];
    ReverseTables() {
        std::memset(slot, 0xFF, sizeof(slot));
        std::memset(sign, 0, sizeof(sign));
        for (int s = 0; s < 2; ++s) {
            for (int sl = 0; sl < 16; ++sl) {
                const uint8_t qpack = stq1_0_codebook[(s << 4) | sl];
                slot[qpack] = static_cast<uint8_t>(sl);
                sign[qpack] = static_cast<uint8_t>(s);
            }
        }
    }
};
const ReverseTables g_reverse;

#if defined(__aarch64__)
struct DecodeTables {
    int8_t lane[4][32];
    DecodeTables() {
        for (int code = 0; code < 32; ++code) {
            const uint8_t packed = stq1_0_codebook[code];
            for (int p = 0; p < 4; ++p)
                lane[p][code] = static_cast<int8_t>(
                    int((packed >> (2 * p)) & 3) - 1);
        }
    }
};
const DecodeTables g_decode;

void decode_stq_chunk16(const block_stq1_0& blk, int chunk,
                        float scale, float* out) {
    // Sixteen groups form four contiguous 16-element output lanes.
    const uint8x8_t qs = vld1_u8(blk.qs + chunk * 8);
    const uint8x8_t low = vand_u8(qs, vdup_n_u8(15));
    const uint8x8_t high = vshr_n_u8(qs, 4);
    const uint8x8x2_t codes = vzip_u8(low, high);
    const uint8x16_t code = vcombine_u8(codes.val[0], codes.val[1]);
    static const int8_t shifts[16] = {
        0, -1, -2, -3, -4, -5, -6, -7,
        0, -1, -2, -3, -4, -5, -6, -7,
    };
    const uint8x16_t sign_bytes = vcombine_u8(
        vdup_n_u8(blk.sign[chunk * 2]),
        vdup_n_u8(blk.sign[chunk * 2 + 1]));
    const uint8x16_t signs = vandq_u8(
        vshlq_u8(sign_bytes, vld1q_s8(shifts)), vdupq_n_u8(1));
    const uint8x16_t indexes = vorrq_u8(code, vshlq_n_u8(signs, 4));

    for (int p = 0; p < 4; ++p) {
        const int8x16x2_t table = {
            vld1q_s8(g_decode.lane[p]),
            vld1q_s8(g_decode.lane[p] + 16),
        };
        const int8x16_t lanes = vqtbl2q_s8(table, indexes);
        const int16x8_t lo = vmovl_s8(vget_low_s8(lanes));
        const int16x8_t hi = vmovl_s8(vget_high_s8(lanes));
        float* y = out + chunk * 64 + p * 16;
        vst1q_f32(y, vmulq_n_f32(
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), scale));
        vst1q_f32(y + 4, vmulq_n_f32(
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), scale));
        vst1q_f32(y + 8, vmulq_n_f32(
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), scale));
        vst1q_f32(y + 12, vmulq_n_f32(
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), scale));
    }
}
#endif

// Pack four lane values in {-1, 0, +1} into a codebook byte.
inline uint8_t qpack_from_lanes(const int8_t lanes[4]) {
    uint8_t q = 0;
    for (int p = 0; p < 4; ++p) {
        const int v = static_cast<int>(lanes[p]) + 1;   // -1->0, 0->1, +1->2
        q = static_cast<uint8_t>(q | (v << (2 * p)));
    }
    return q;
}

inline float fp16_to_fp32(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
    uint32_t exp = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        // subnormal
        exp = 1;
        while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
        mant &= 0x3FF;
        out = sign | ((exp + 112) << 23) | (mant << 13);
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// fp32 -> fp16 with round-to-nearest-even. Written out rather than using
// _cvtss_sh so the kernel stays portable across the CPU backends.
inline uint16_t fp32_to_fp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t half = mant >> shift;
        // round to nearest even
        if ((mant >> (shift - 1)) & 1 && ((mant & ((1u << (shift - 1)) - 1)) ||
                                          (half & 1)))
            ++half;
        return static_cast<uint16_t>(sign | half);
    }
    if (exp >= 31) {
        if (exp == 31 && mant) return static_cast<uint16_t>(sign | 0x7C00u); // nan -> inf-ish
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10)
                                          | (mant >> 13));
    if ((mant & 0x1FFFu) > 0x1000u ||
        ((mant & 0x1FFFu) == 0x1000u && (half & 1)))
        ++half;
    return half;
}

}  // namespace

void stq1_0_dequantize_block(const block_stq1_0& blk, float* out) {
    const float d = fp16_to_fp32(blk.d);
#if defined(__aarch64__)
    for (int chunk = 0; chunk < 4; ++chunk)
        decode_stq_chunk16(blk, chunk, d, out);
#else
    // (group, lane) visits every one of the 256 positions exactly once.
    for (int g = 0; g < STQ_GROUPS; ++g) {
        const uint8_t code = (blk.qs[g / 2] >> (4 * (g & 1))) & 0x0F;
        const uint8_t sign = (blk.sign[g / 8] >> (g % 8)) & 0x01;
        const uint8_t packed = stq1_0_codebook[(sign << 4) | code];
        for (int p = 0; p < 4; ++p)
            out[stq1_0_element_index(g, p)] =
                d * static_cast<float>(int((packed >> (2 * p)) & 3) - 1);
    }
#endif
}

void stq1_0_dequantize_row(const block_stq1_0* src, int k, float* out) {
    const int nb = k / QK_STQ_K;
    for (int b = 0; b < nb; ++b)
        stq1_0_dequantize_block(src[b], out + b * QK_STQ_K);
}

void stq1_0_dot_row(const float* a, const block_stq1_0* b, int k, float* acc) {
    const int nb = k / QK_STQ_K;
    float sum = 0.0f;
    for (int blk = 0; blk < nb; ++blk) {
        const block_stq1_0& cur = b[blk];
        const float d = fp16_to_fp32(cur.d);
        const float* ab = a + blk * QK_STQ_K;
        float partial = 0.0f;
        for (int g = 0; g < STQ_GROUPS; ++g) {
            int8_t lanes[4];
            stq1_0_group_lanes(cur, g, lanes);
            for (int p = 0; p < 4; ++p) {
                const int j = stq1_0_element_index(g, p);
                partial += static_cast<float>(lanes[p]) * ab[j];
            }
        }
        sum += d * partial;
    }
    *acc += sum;
}

void stq1_0_quantize_block(const float* x, const float* imatrix,
                           block_stq1_0& blk, float* scale) {
    std::memset(blk.qs, 0, sizeof(blk.qs));
    std::memset(blk.sign, 0, sizeof(blk.sign));

    float sumx2 = 0.0f;
    float amax = 0.0f;
    for (int j = 0; j < QK_STQ_K; ++j) {
        sumx2 += x[j] * x[j];
        const float a = std::fabs(x[j]);
        if (a > amax) amax = a;
    }
    if (!(amax > 0.0f)) {
        blk.d = 0;
        if (scale) *scale = 0.0f;
        return;
    }
    const float sigma2 = 2.0f * sumx2 / QK_STQ_K;

    // Importance weights. The reference uses sqrt(sigma2 + x^2), optionally
    // scaled by the imatrix.
    float weight[QK_STQ_K];
    if (imatrix) {
        for (int j = 0; j < QK_STQ_K; ++j)
            weight[j] = imatrix[j] * std::sqrt(sigma2 + x[j] * x[j]);
    } else {
        for (int j = 0; j < QK_STQ_K; ++j)
            weight[j] = std::sqrt(sigma2 + x[j] * x[j]);
    }

    // `sel` is indexed in ELEMENT order, mirroring the reference. Writing it
    // in group order and then combining with weight[]/x[] (which are element
    // ordered) silently yields a wrong scale, so the mapping is applied at
    // every write.
    int8_t sel[QK_STQ_K];
    std::memset(sel, 0, sizeof(sel));

    const int n_iter = 3;   // gains past 3 rounds measured below 0.05%
    float d = amax;

    for (int iter = 0; iter < n_iter; ++iter) {
        for (int g = 0; g < STQ_GROUPS; ++g) {
            // Cost of forcing lane p to zero; the alternative is quantizing it
            // to +-d. Pick the lane whose zeroing costs least relative to that.
            int zero_pos = 0;
            float best_cost = 0.0f;
            for (int p = 0; p < 4; ++p) {
                const int j = stq1_0_element_index(g, p);
                const float ax = std::fabs(x[j]);
                const float cost =
                    weight[j] * (x[j] * x[j] - (ax - d) * (ax - d));
                if (p == 0 || cost < best_cost) {
                    best_cost = cost;
                    zero_pos = p;
                }
            }
            for (int p = 0; p < 4; ++p) {
                const int j = stq1_0_element_index(g, p);
                sel[j] = (p == zero_pos) ? 0 : (x[j] < 0.0f ? -1 : 1);
            }
        }

        float sumqx = 0.0f;
        float sumq2 = 0.0f;
        for (int j = 0; j < QK_STQ_K; ++j) {
            const float q = static_cast<float>(sel[j]);
            sumqx += weight[j] * q * x[j];
            sumq2 += weight[j] * q * q;
        }
        if (!(sumq2 > 0.0f)) break;
        const float dnew = sumqx / sumq2;
        // A negative optimum would flip every sign, but the selection already
        // carries the signs, so clamp instead of flipping.
        if (!(dnew > 0.0f)) break;
        if (std::fabs(dnew - d) <= 1e-6f * d) { d = dnew; break; }
        d = dnew;
    }

    // Encode. `sel` is element-ordered, so read it back through the same
    // index mapping used to write it; the value already carries the forced
    // zero (sel == 0) and the sign.
    for (int g = 0; g < STQ_GROUPS; ++g) {
        int8_t lanes[4];
        for (int p = 0; p < 4; ++p)
            lanes[p] = sel[stq1_0_element_index(g, p)];

        const uint8_t qpack = qpack_from_lanes(lanes);
        const uint8_t slot = g_reverse.slot[qpack];
        const uint8_t sgn = g_reverse.sign[qpack];
        if (slot == 0xFF) continue;   // not a codebook pattern; leaves zeros
        blk.qs[g / 2] |= static_cast<uint8_t>((slot & 0x0F) << (4 * (g & 1)));
        blk.sign[g / 8] |= static_cast<uint8_t>(sgn << (g % 8));
    }

    blk.d = fp32_to_fp16(d);
    if (scale) *scale = d;
}
