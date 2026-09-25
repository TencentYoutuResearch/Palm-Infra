#pragma once

// STQ1_0: AngelSlim's 1.3125 bpw ternary quantization (ggml type id 43).
//
// Layout and codebook follow the author's llama.cpp patch
// (hy4-preview-patch/0002-stq1_0-quant-and-cuda.patch, ggml-common.h).
//
// A super-block of QK_STQ_K weights is 42 bytes:
//
//     uint8_t  qs[QK_STQ_K / 8];    // two 4-bit slot codes per byte
//     uint8_t  sign[QK_STQ_K / 32]; // eight 1-bit table selects per byte
//     uint16_t d;                   // fp16 scale
//
// Every group of 4 weights holds exactly one zero and three equal-magnitude
// non-zero values (+/-1), so the 3^4 = 81 raw combinations collapse to
// 4 zero-positions x 2^3 signs = 32, which fits in 5 bits per group:
//
//     4 bit slot = (zero_pos << 2) | tail_bits
//         tail_bits.0 -- sign of the 2nd non-zero lane (0 => +1)
//         tail_bits.1 -- sign of the 3rd non-zero lane
//     1 bit sign = global flip of every non-zero lane
//
// The codebook turns those 5 bits into one byte holding four 2-bit lanes,
// so decoding is a single table load. Lane encoding is value+1:
// -1 -> 0b00, 0 -> 0b01, +1 -> 0b10, hence `((qpack >> 2p) & 3) - 1`.
//
// Group-to-element mapping is stride-16 within each 64-element chunk:
// group g (0..63), lane p (0..3) lives at element
//
//     (g / 16) * 64 + (g % 16) + p * 16
//
// This is NOT the natural order, and mixing the two silently produces wrong
// results (see the group-vs-element order note in the encoder).

#include <cstdint>

constexpr int QK_STQ_K = 256;
constexpr int STQ_BLOCK_BYTES = 42;
constexpr int STQ_QS_BYTES = QK_STQ_K / 8;      // 32
constexpr int STQ_SIGN_BYTES = QK_STQ_K / 32;   // 8
constexpr int STQ_GROUPS = QK_STQ_K / 4;        // 64

struct block_stq1_0 {
    uint8_t qs[STQ_QS_BYTES];
    uint8_t sign[STQ_SIGN_BYTES];
    uint16_t d;      // fp16 bits
};
static_assert(sizeof(block_stq1_0) == STQ_BLOCK_BYTES,
              "block_stq1_0 must be 42 bytes with no padding");

// index = (sign << 4) | slot  ->  four packed 2-bit lanes
extern const uint8_t stq1_0_codebook[32];

// Element index for (group g, lane p). Exposed because both the encoder and
// the decoder must agree and a mismatch is silent.
inline int stq1_0_element_index(int g, int p) {
    return (g / 16) * 64 + (g % 16) + p * 16;
}

// Decode one group's codebook byte into four lane values in {-1, 0, +1}.
inline void stq1_0_group_lanes(const block_stq1_0& blk, int g, int8_t out[4]) {
    const uint8_t code = (blk.qs[g / 2] >> (4 * (g & 1))) & 0x0F;
    const uint8_t sgn = (blk.sign[g / 8] >> (g % 8)) & 0x01;
    const uint8_t qpack = stq1_0_codebook[(sgn << 4) | code];
    for (int p = 0; p < 4; ++p)
        out[p] = static_cast<int8_t>(((qpack >> (2 * p)) & 0x3) - 1);
}

// Reconstruct a whole block into `out` in element order.
void stq1_0_dequantize_block(const block_stq1_0& blk, float* out);

// Encode 256 floats into one block. `scale` receives the solved scale.
// `imatrix` may be null; when present it weights the scale solve and the
// forced-zero placement, as the reference encoder does.
void stq1_0_quantize_block(const float* x, const float* imatrix,
                           block_stq1_0& blk, float* scale);

// y += A_row . B_row over `k` elements (k multiple of 256), where B is STQ1_0.
void stq1_0_dot_row(const float* a, const block_stq1_0* b, int k, float* acc);

// Row-scale an STQ1_0 row into float (dequantized), for tests and for the
// generic matmul fallback.
void stq1_0_dequantize_row(const block_stq1_0* src, int k, float* out);
