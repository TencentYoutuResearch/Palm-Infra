#pragma once

#include <algorithm>
#include <cstdlib>

namespace mollm::metal {

inline int gemv_nsg_cap() {
    static const int cap = [] {
        const char* value = std::getenv("MOLLM_METAL_GEMV_NSG");
        if (!value)
            return 8;
        const int parsed = std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8)
                   ? parsed
                   : 4;
    }();
    return cap;
}

inline int gemv_w4_nr0(int n, int k) {
    const char* value = std::getenv("MOLLM_METAL_GEMV_W4_NR");
    if (value) {
        const int parsed = std::atoi(value);
        if (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8)
            return parsed;
    }
    (void)n;
    (void)k;
    return 1;
}

inline int gemv_w4_nsg_cap() {
    static const int cap = [] {
        const char* value = std::getenv("MOLLM_METAL_GEMV_W4_NSG");
        if (!value)
            return 4;
        const int parsed = std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8)
                   ? parsed
                   : 4;
    }();
    return cap;
}

} // namespace mollm::metal
