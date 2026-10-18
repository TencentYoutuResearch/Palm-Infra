#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

namespace mollm_memory {

inline bool fits_headroom(size_t available, size_t requested, size_t headroom) {
    return available >= headroom && requested <= available - headroom;
}

// Accept decimal MiB only. In particular, strtoull's optional whitespace and
// sign must not turn malformed safety configuration into a disabled guard.
inline bool parse_cuda_host_headroom(const char* option, size_t& bytes) {
    bytes = 0;
    if (!option || !*option) return false;
    constexpr size_t unit = 1024 * 1024;
    constexpr size_t maximum = std::numeric_limits<size_t>::max() / unit;
    size_t mb = 0;
    for (const char* p = option; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        const size_t digit = static_cast<size_t>(*p - '0');
        if (mb > maximum / 10 ||
            (mb == maximum / 10 && digit > maximum % 10)) return false;
        mb = mb * 10 + digit;
    }
    bytes = mb * unit;
    return true;
}

// Opt-in host-wide guard for unified-memory CUDA systems. Not an allocation
// reservation: concurrent allocations and driver overhead remain unbounded.
// Fail closed when configured but host availability cannot be read.
inline bool read_host_available_memory(size_t& bytes) {
    std::ifstream source("/proc/meminfo");
    std::string label;
    size_t available_kb = 0;
    while (source >> label) {
        if (label == "MemAvailable:") {
            std::string unit;
            if (!(source >> available_kb >> unit) || unit != "kB" ||
                available_kb > std::numeric_limits<size_t>::max() / 1024) return false;
            bytes = available_kb * 1024;
            return true;
        }
        std::string remainder;
        std::getline(source, remainder);
    }
    return false;
}

inline bool cuda_host_allocation_allowed(size_t requested, const char* operation) {
    const char* option = std::getenv("MOLLM_CUDA_HOST_HEADROOM_MB");
    if (!option) return true;
    size_t headroom = 0;
    if (!parse_cuda_host_headroom(option, headroom)) {
        std::fprintf(stderr, "CUDA host budget: invalid MOLLM_CUDA_HOST_HEADROOM_MB\n");
        return false;
    }
    if (headroom == 0) return true;
    size_t available = 0;
    if (!read_host_available_memory(available)) {
        std::fprintf(stderr, "CUDA host budget: cannot read MemAvailable; rejecting %s\n", operation);
        return false;
    }
    if (fits_headroom(available, requested, headroom)) return true;
    std::fprintf(stderr,
        "CUDA host budget: rejecting %s requested=%zu available=%zu headroom=%zu bytes\n",
        operation, requested, available, headroom);
    return false;
}

} // namespace mollm_memory
