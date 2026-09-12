#pragma once

#include <cstdint>

// Stable packed-weight block descriptions shared by CPU packers and device
// backends. They intentionally do not include matmul.h or CPU SIMD headers.
// Package weight payloads guarantee scalar alignment, but not 16-byte
// alignment, so package-native blocks must not require stronger alignment.
struct Q4B8G128Block {
    float scales[8];
    uint8_t q[4][8][16];
};
static_assert(sizeof(Q4B8G128Block) == 544,
              "unexpected Q4B8G128Block size");

struct Q4B8G32Block {
    float scales[8];
    uint8_t q[8][16];
};
static_assert(sizeof(Q4B8G32Block) == 160,
              "unexpected Q4B8G32Block size");

// The x86 VNNI sidecar is allocated by the runtime and retains its stronger
// alignment requirement.
struct alignas(16) Q4B8G32VnniBlock {
    uint8_t q[8][16];
};
static_assert(sizeof(Q4B8G32VnniBlock) == 128,
              "unexpected Q4B8G32VnniBlock size");
