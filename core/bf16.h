#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

inline float mollm_round_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline void mollm_round_to_bf16(float* values, size_t count) {
    for (size_t i = 0; i < count; ++i)
        values[i] = mollm_round_to_bf16(values[i]);
}
