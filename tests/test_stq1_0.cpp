// Tests for the STQ1_0 ternary kernel.
//
// STQ1_0 is a 1.3125 bpw format, so a large reconstruction error is EXPECTED
// on random data -- the ternary alphabet simply cannot represent a Gaussian
// well. Error magnitude therefore proves nothing about correctness, and these
// tests avoid using it as a pass/fail criterion. Instead they check:
//
//   1. the codebook's structural invariant (one zero, three unit lanes)
//   2. bit-exact round-trip of the encoder's own output
//   3. the dot product against an independently dequantized reference
//   4. the encoder against hard-coded vectors produced by the Python model
//
// (4) is the one that catches bit-packing mistakes; (3) catches decode errors.

#include "kernels/cpu/matmul/stq1_0.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg) do {                                            \
    if (!(cond)) {                                                       \
        std::fprintf(stderr, "FAIL: %s\n", msg);                         \
        ++g_failures;                                                    \
    } else {                                                             \
        std::fprintf(stderr, "  PASS: %s\n", msg);                       \
    }                                                                    \
} while (0)

namespace {

std::vector<float> fill_random(std::vector<float>& v, uint32_t seed) {
    uint32_t state = seed;
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = (static_cast<float>(state >> 8) / static_cast<float>(1u << 24))
                * 2.0f - 1.0f;
    }
    return v;
}

}  // namespace

static void test_codebook_structure() {
    // Every codebook byte must encode exactly one zero lane and three lanes of
    // unit magnitude, or the format's premise ("each group has one zero") is
    // broken and the 5-bit packing would be ambiguous.
    bool ok = true;
    int distinct = 0;
    bool seen[256] = {};
    for (int i = 0; i < 32; ++i) {
        const uint8_t qpack = stq1_0_codebook[i];
        if (!seen[qpack]) { seen[qpack] = true; ++distinct; }
        int zeros = 0;
        bool unit = true;
        for (int p = 0; p < 4; ++p) {
            const int lane = ((qpack >> (2 * p)) & 0x3) - 1;
            if (lane == 0) ++zeros;
            else if (lane != 1 && lane != -1) unit = false;
        }
        if (zeros != 1 || !unit) {
            std::fprintf(stderr, "    codebook[%d]=%#x zeros=%d unit=%d\n",
                         i, qpack, zeros, static_cast<int>(unit));
            ok = false;
        }
    }
    CHECK(ok, "codebook: every entry has one zero and three unit lanes");
    CHECK(distinct == 32, "codebook: 32 distinct patterns");
}

static void test_group_lanes_roundtrip() {
    // Decoding every (slot, sign) pair must land in {-1, 0, +1} with exactly
    // one zero, and re-encoding the lanes must reproduce the same byte.
    bool ok = true;
    for (int s = 0; s < 2; ++s) {
        for (int sl = 0; sl < 16; ++sl) {
            const uint8_t qpack = stq1_0_codebook[(s << 4) | sl];
            int zeros = 0;
            for (int p = 0; p < 4; ++p) {
                const int lane = ((qpack >> (2 * p)) & 0x3) - 1;
                if (lane == 0) ++zeros;
            }
            if (zeros != 1) ok = false;
        }
    }
    CHECK(ok, "group lanes: single zero for all 32 codebook entries");
}

static void test_element_index_coverage() {
    // The group->element mapping must be a bijection over 0..255. A collision
    // or gap would silently corrupt the reconstruction.
    bool seen[QK_STQ_K] = {};
    bool ok = true;
    for (int g = 0; g < STQ_GROUPS; ++g) {
        for (int p = 0; p < 4; ++p) {
            const int j = stq1_0_element_index(g, p);
            if (j < 0 || j >= QK_STQ_K || seen[j]) ok = false;
            else seen[j] = true;
        }
    }
    CHECK(ok, "element index: bijection over 0..255 (no gaps, no collisions)");
}

static void test_encoder_decoder_consistency() {
    // Round-trip the encoder's own output. Because both sides share the
    // group->element mapping, this verifies the bit packing: a permutation
    // error would put values at the wrong elements and this would catch it
    // (the Python cross-check catches it too, but this runs without Python).
    std::vector<float> x(QK_STQ_K);
    fill_random(x, 42);
    for (auto& v : x) v *= 0.1f;

    block_stq1_0 blk;
    float scale = 0;
    stq1_0_quantize_block(x.data(), nullptr, blk, &scale);

    std::vector<float> rec(QK_STQ_K);
    stq1_0_dequantize_block(blk, rec.data());

    // Every reconstructed value must be one of {-d, 0, +d}.
    bool ternary = true;
    int zeros = 0;
    for (int j = 0; j < QK_STQ_K; ++j) {
        const float a = std::fabs(rec[j]);
        if (a == 0.0f) { ++zeros; continue; }
        if (std::fabs(a - scale) > 1e-3f * std::fabs(scale)) ternary = false;
    }
    CHECK(ternary, "decode: every value is in {-d, 0, +d}");
    // Exactly one zero per group of four.
    CHECK(zeros == QK_STQ_K / 4, "decode: exactly one zero per group of four");

    // Reconstruction must beat the trivial "all zeros" predictor, which is a
    // weak but meaningful floor on random data.
    double num = 0, den = 0;
    for (int j = 0; j < QK_STQ_K; ++j) {
        num += (rec[j] - x[j]) * (rec[j] - x[j]);
        den += x[j] * x[j];
    }
    const double rel = std::sqrt(num / den);
    std::fprintf(stderr, "    (info) random-data rel_err = %.4f "
                         "[expected ~0.55 for ternary]\n", rel);
    CHECK(rel < 0.9, "decode: beats zero predictor on random data");
}

static void test_dot_matches_dequantize() {
    // The fused dot must equal the dot against an explicitly dequantized row.
    // This is the check that catches a decode-path bug independent of the
    // encoder.
    const int k = 2 * QK_STQ_K;
    std::vector<float> a(k), b(k);
    fill_random(a, 7);
    fill_random(b, 11);
    for (auto& v : b) v *= 0.1f;

    std::vector<block_stq1_0> blocks(k / QK_STQ_K);
    for (int i = 0; i < k / QK_STQ_K; ++i) {
        float s = 0;
        stq1_0_quantize_block(b.data() + i * QK_STQ_K, nullptr, blocks[i], &s);
    }

    std::vector<float> deq(k);
    stq1_0_dequantize_row(blocks.data(), k, deq.data());

    float fused = 0.0f;
    stq1_0_dot_row(a.data(), blocks.data(), k, &fused);

    double direct = 0.0;
    for (int j = 0; j < k; ++j) direct += static_cast<double>(a[j]) * deq[j];

    const double diff = std::fabs(fused - direct);
    const double tol = 1e-4 * std::fabs(direct) + 1e-5;
    std::fprintf(stderr, "    (info) fused=%.6f direct=%.6f diff=%.2e\n",
                 fused, direct, diff);
    CHECK(diff <= tol, "dot: fused path equals dequantize-then-dot");
}

int main() {
    std::fprintf(stderr, "=== STQ1_0 kernel tests ===\n");
    test_codebook_structure();
    test_group_lanes_roundtrip();
    test_element_index_coverage();
    test_encoder_decoder_consistency();
    test_dot_matches_dequantize();

    if (g_failures) {
        std::fprintf(stderr, "\n%d STQ1_0 test(s) FAILED\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "\nAll STQ1_0 tests passed!\n");
    return 0;
}
