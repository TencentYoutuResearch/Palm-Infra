#include "backends/metal/weight_layout.h"
#include "core/quant_layouts.h"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL: %s\n", message);                   \
            ++failures;                                                     \
        }                                                                   \
    } while (0)

void test_bg32() {
    constexpr int rows = 10;
    constexpr int columns = 64;
    constexpr int groups = 2;
    std::vector<Q4B8G32Block> packed(((rows + 7) / 8) * groups);
    for (size_t block_index = 0; block_index < packed.size(); ++block_index) {
        auto& block = packed[block_index];
        for (int channel = 0; channel < 8; ++channel) {
            block.scales[channel] =
                static_cast<float>(block_index * 10 + channel);
            for (int byte = 0; byte < 16; ++byte)
                block.q[channel][byte] =
                    static_cast<uint8_t>(block_index * 31 + channel + byte);
        }
    }

    std::vector<uint8_t> nibbles(rows * columns / 2);
    std::vector<float> scales(rows * groups);
    CHECK(mollm::metal::decode_q4_weight(
              packed.data(), mollm::metal::PackedQ4Layout::BG32,
              rows, columns, groups, nibbles.data(), scales.data()),
          "decode BG32 layout");
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups; ++group) {
            const size_t block_index = static_cast<size_t>(row / 8) * groups + group;
            const auto& block = packed[block_index];
            CHECK(scales[static_cast<size_t>(row) * groups + group] ==
                      block.scales[row % 8],
                  "BG32 scale order");
            for (int byte = 0; byte < 16; ++byte) {
                const size_t output = static_cast<size_t>(row) * (columns / 2) +
                    group * 16 + byte;
                CHECK(nibbles[output] ==
                          static_cast<uint8_t>(block.q[row % 8][byte] ^ 0x88),
                      "BG32 nibble order and sign conversion");
            }
        }
    }
}

void test_bg128() {
    constexpr int rows = 9;
    constexpr int columns = 256;
    constexpr int groups = 2;
    std::vector<Q4B8G128Block> packed(((rows + 7) / 8) * groups);
    for (size_t block_index = 0; block_index < packed.size(); ++block_index) {
        auto& block = packed[block_index];
        for (int channel = 0; channel < 8; ++channel) {
            block.scales[channel] =
                static_cast<float>(block_index * 10 + channel);
            for (int subgroup = 0; subgroup < 4; ++subgroup)
                for (int byte = 0; byte < 16; ++byte)
                    block.q[subgroup][channel][byte] = static_cast<uint8_t>(
                        block_index * 17 + subgroup * 7 + channel + byte);
        }
    }

    std::vector<uint8_t> nibbles(rows * columns / 2);
    std::vector<float> scales(rows * groups);
    CHECK(mollm::metal::decode_q4_weight(
              packed.data(), mollm::metal::PackedQ4Layout::BG128,
              rows, columns, groups, nibbles.data(), scales.data()),
          "decode BG128 layout");
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups; ++group) {
            const size_t block_index = static_cast<size_t>(row / 8) * groups + group;
            const auto& block = packed[block_index];
            CHECK(scales[static_cast<size_t>(row) * groups + group] ==
                      block.scales[row % 8],
                  "BG128 scale order");
            for (int subgroup = 0; subgroup < 4; ++subgroup)
                for (int byte = 0; byte < 16; ++byte) {
                    const size_t output =
                        static_cast<size_t>(row) * (columns / 2) +
                        group * 64 + subgroup * 16 + byte;
                    CHECK(nibbles[output] == static_cast<uint8_t>(
                              block.q[subgroup][row % 8][byte] ^ 0x88),
                          "BG128 nibble order and sign conversion");
                }
        }
    }
}

}  // namespace

int main() {
    test_bg32();
    test_bg128();
    uint8_t byte = 0;
    float scale = 0.0f;
    CHECK(!mollm::metal::decode_q4_weight(
              &byte, mollm::metal::PackedQ4Layout::BG32,
              1, 31, 1, &byte, &scale),
          "reject invalid Q4 dimensions");
    if (failures != 0)
        std::fprintf(stderr, "%d Metal weight-layout checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
