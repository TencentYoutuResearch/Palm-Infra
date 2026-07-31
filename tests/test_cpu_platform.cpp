#include "kernels/cpu_platform.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    const auto& caps = mollm::cpu::capabilities();
    const char* name = mollm::cpu::isa_name();
    if (!name || name[0] == '\0') {
        std::fprintf(stderr, "CPU provider name is empty\n");
        return 1;
    }

    if (caps.x86_avx512 &&
        (caps.x86_isa != mollm::cpu::X86Isa::AVX512 ||
         std::strcmp(name, "x86-avx512") != 0)) {
        std::fprintf(stderr, "inconsistent AVX-512 dispatch state\n");
        return 1;
    }
    if (caps.arm_i8mm &&
        (!caps.arm_neon || std::strcmp(name, "arm-neon-i8mm") != 0)) {
        std::fprintf(stderr, "inconsistent ARM i8mm dispatch state\n");
        return 1;
    }
    if (caps.x86_isa == mollm::cpu::X86Isa::AVX2 &&
        (!caps.x86_avx2 || std::strcmp(name, "x86-avx2") != 0)) {
        std::fprintf(stderr, "inconsistent AVX2 dispatch state\n");
        return 1;
    }
    if (caps.x86_f16c && !caps.x86_avx2 && !caps.x86_avx512) {
        std::fprintf(stderr, "F16C selected without an x86 SIMD tier\n");
        return 1;
    }
    if (caps.x86_avx512_vnni && !caps.x86_avx512) {
        std::fprintf(stderr, "AVX-512 VNNI selected without AVX-512\n");
        return 1;
    }
    if (caps.x86_w4_q8_activations && !caps.x86_avx512_vnni) {
        std::fprintf(stderr,
                     "W4 Q8 activations selected without AVX-512 VNNI\n");
        return 1;
    }

    const char* requested = std::getenv("MOLLM_X86_ISA");
    if ((std::getenv("MOLLM_X86_DISABLE_AVX2") ||
         (requested && std::strcmp(requested, "scalar") == 0)) &&
        std::strcmp(name, "x86-scalar") != 0) {
        std::fprintf(stderr, "scalar override selected %s\n", name);
        return 1;
    }
    if (requested && std::strcmp(requested, "avx2") == 0 &&
        caps.x86_avx512) {
        std::fprintf(stderr, "AVX2 cap incorrectly selected AVX-512\n");
        return 1;
    }
    const char* w4_activation =
        std::getenv("MOLLM_X86_W4_ACTIVATION");
    const bool requested_w4_q8 =
        w4_activation && std::strcmp(w4_activation, "q8") == 0;
    if (!requested_w4_q8 && caps.x86_w4_q8_activations) {
        std::fprintf(stderr,
                     "W4 Q8 activations enabled without explicit opt-in\n");
        return 1;
    }
    if (requested_w4_q8 && caps.x86_avx512_vnni &&
        !caps.x86_w4_q8_activations) {
        std::fprintf(stderr,
                     "W4 Q8 opt-in ignored on selected AVX-512 VNNI\n");
        return 1;
    }
    const char* arm_requested = std::getenv("MOLLM_ARM_ISA");
    if ((std::getenv("MOLLM_ARM_DISABLE_I8MM") ||
         (arm_requested && std::strcmp(arm_requested, "neon") == 0)) &&
        caps.arm_i8mm) {
        std::fprintf(stderr, "ARM NEON cap incorrectly selected i8mm\n");
        return 1;
    }

    std::printf("CPU provider: %s\n", name);
    return 0;
}
