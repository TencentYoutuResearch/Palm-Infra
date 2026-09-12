#pragma once

#include <cstdint>

namespace mollm {

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
using fp16_t = __fp16;
#elif defined(__clang__)
// Clang exposes __fp16 as a storage-only type on x86.
using fp16_t = __fp16;
#else
// GCC supports IEEE binary16 storage on x86 Linux.
using fp16_t = _Float16;
#endif

static_assert(sizeof(fp16_t) == 2,
              "mollm FP16 storage must be binary16");

}  // namespace mollm

#if !defined(__aarch64__) && !defined(__arm64__) && !defined(_M_ARM64) && \
    !defined(__clang__) && !defined(__CUDACC__)
// Legacy kernels still use the ARM spelling for binary16 storage.
using __fp16 = mollm::fp16_t;
#endif
