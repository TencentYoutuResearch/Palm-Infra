#pragma once

#include <cstdint>

// Serialized activation IDs shared by all backends. Keep these values in sync
// with Activation(IntEnum) in models/transpile.py.
enum class Activation : int32_t {
    NONE = 0,
    SILU = 1,
    GELU = 2,
    RELU = 3,
    RELU_SQUARED = 4,
};
