#pragma once

#include <cstddef>
#include <cstdint>

namespace mollm_cuda {

// A caller-provided device extent, validated on the host before launch.
struct DeviceSpan {
    const void* pointer;
    size_t bytes;
};

// True when every span is non-null, aligned to `alignment`, does not wrap the
// address space, and no two spans overlap.
inline bool device_spans_disjoint(const DeviceSpan* spans, size_t count, size_t alignment = 1) {
    for (size_t i = 0; i < count; ++i) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(spans[i].pointer);
        if (!begin || begin % alignment || spans[i].bytes > UINTPTR_MAX - begin) return false;
        for (size_t j = 0; j < i; ++j) {
            const uintptr_t other = reinterpret_cast<uintptr_t>(spans[j].pointer);
            if (begin < other + spans[j].bytes && other < begin + spans[i].bytes) return false;
        }
    }
    return true;
}

}  // namespace mollm_cuda
