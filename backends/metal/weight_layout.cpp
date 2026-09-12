#include "backends/metal/weight_layout.h"

#include "core/quant_layouts.h"

#include <cstddef>
#include <cstring>

namespace mollm::metal {
namespace {

void copy_offset_binary(uint8_t* destination, const uint8_t* source) {
    constexpr uint64_t sign_bits = 0x8888888888888888ull;
    uint64_t words[2];
    std::memcpy(words, source, sizeof(words));
    words[0] ^= sign_bits;
    words[1] ^= sign_bits;
    std::memcpy(destination, words, sizeof(words));
}

}  // namespace

bool decode_q4_weight(const void* packed, PackedQ4Layout layout,
                      int rows, int columns, int groups_per_row,
                      uint8_t* nibbles, float* scales) {
    const int group_size = layout == PackedQ4Layout::BG32 ? 32 : 128;
    if (!packed || !nibbles || !scales || rows <= 0 || columns <= 0 ||
        groups_per_row <= 0 || columns % group_size != 0 ||
        groups_per_row != columns / group_size)
        return false;

    for (int row = 0; row < rows; ++row) {
        const int tile = row / 8;
        const int channel = row % 8;
        uint8_t* row_nibbles =
            nibbles + static_cast<size_t>(row) * (columns / 2);
        for (int group = 0; group < groups_per_row; ++group) {
            if (layout == PackedQ4Layout::BG32) {
                const auto* blocks =
                    static_cast<const Q4B8G32Block*>(packed);
                const Q4B8G32Block& block =
                    blocks[static_cast<size_t>(tile) * groups_per_row + group];
                scales[static_cast<size_t>(row) * groups_per_row + group] =
                    block.scales[channel];
                copy_offset_binary(
                    row_nibbles + static_cast<size_t>(group * 32) / 2,
                    block.q[channel]);
                continue;
            }

            const auto* blocks = static_cast<const Q4B8G128Block*>(packed);
            const Q4B8G128Block& block =
                blocks[static_cast<size_t>(tile) * groups_per_row + group];
            scales[static_cast<size_t>(row) * groups_per_row + group] =
                block.scales[channel];
            for (int subgroup = 0; subgroup < 4; ++subgroup) {
                copy_offset_binary(
                    row_nibbles +
                        static_cast<size_t>(group * 128 + subgroup * 32) / 2,
                    block.q[subgroup][channel]);
            }
        }
    }
    return true;
}

}  // namespace mollm::metal
