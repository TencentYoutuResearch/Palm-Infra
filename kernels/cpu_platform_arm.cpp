#include "kernels/cpu_platform.h"

#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace mollm::cpu {

namespace {

bool detect_i8mm() {
#if MOLLM_ARM_I8MM_KERNELS
    if (std::getenv("MOLLM_ARM_DISABLE_I8MM"))
        return false;
    const char* requested = std::getenv("MOLLM_ARM_ISA");
    if (requested && std::strcmp(requested, "neon") == 0)
        return false;
#if defined(__linux__) && defined(HWCAP2_I8MM)
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#elif defined(__APPLE__)
    int supported = 0;
    size_t size = sizeof(supported);
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &supported, &size, nullptr,
                     0) == 0) {
        return supported != 0;
    }
#endif
#endif
    return false;
}

}  // namespace

const Capabilities& capabilities() {
    static const Capabilities value = [] {
        Capabilities caps;
        caps.arm_neon = true;
        caps.arm_i8mm = detect_i8mm();
        caps.fp16_vector_math = true;
        caps.fp16_kv_cache = true;
        caps.fp16_interleaved_weights = true;
        return caps;
    }();
    return value;
}

const char* isa_name() {
    return capabilities().arm_i8mm ? "arm-neon-i8mm" : "arm-neon";
}

void relax() {
    __asm__ __volatile__("yield" ::: "memory");
}

bool matmul_int4_packed(const Tensor&, const Tensor&, Tensor&, int, int,
                        ThreadPool*) {
    // ARM retains the upstream DOTPROD/i8mm dispatch in matmul_w4.cpp.
    return false;
}

bool matmul_dense_fp32_range(const float*, const float*, float*, int, int, int,
                             int, int, int, int) {
    return false;
}

bool matmul_dense_fp16_range(const float*, const fp16_t*, float*, int, int, int,
                             int, int, int, int, bool) {
    return false;
}

bool matmul_dense_fp16_m2_range_n(
    const float*, const fp16_t*, float*, int, int, int, int, int, int, int) {
    return false;
}

bool matmul_int8_range(const float*, const int8_t*, const float*, float*, int,
                       int, int, int, int, int, int, int, int, int, int, bool) {
    return false;
}

}  // namespace mollm::cpu
