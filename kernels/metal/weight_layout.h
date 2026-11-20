#pragma once

#include <cstdint>

namespace mollm::metal {

enum class PackedQ4Layout {
    BG32,
    BG128,
};

// Convert package-native, eight-row-interleaved Q4 blocks into the
// offset-binary row-major representation consumed by Metal matmul kernels.
bool decode_q4_weight(const void* packed, PackedQ4Layout layout,
                      int rows, int columns, int groups_per_row,
                      uint8_t* nibbles, float* scales);

}  // namespace mollm::metal
