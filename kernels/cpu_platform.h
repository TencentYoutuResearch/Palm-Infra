#pragma once

// CPU architecture boundary.
//
// Graph and engine code use this header for storage-level FP16 and for the
// few execution-policy decisions that genuinely vary by CPU family.  ARM
// intrinsics stay behind this boundary; generic kernels must not infer their
// target from compiler predefined macros.

#include <cstdint>

struct Tensor;
class ThreadPool;

#ifndef MOLLM_CPU_ARM_NEON
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#define MOLLM_CPU_ARM_NEON 1
#else
#define MOLLM_CPU_ARM_NEON 0
#endif
#endif

#ifndef MOLLM_ARM_I8MM_KERNELS
#define MOLLM_ARM_I8MM_KERNELS 0
#endif

#if MOLLM_CPU_ARM_NEON
#include <arm_neon.h>
#endif

namespace mollm::cpu {

enum class X86Isa : uint8_t {
    SCALAR = 0,
    AVX2 = 1,
    AVX512 = 2,
};

#if MOLLM_CPU_ARM_NEON
using fp16_t = __fp16;
#elif defined(__clang__)
// Clang exposes __fp16 as a storage-only type on x86. Keep it as the
// canonical storage spelling so legacy kernel signatures remain compatible.
using fp16_t = __fp16;
#else
// GCC supports IEEE binary16 storage on x86 Linux. It is used only for model
// bytes and scalar conversion; it does not imply native FP16 SIMD.
using fp16_t = _Float16;
#endif

static_assert(sizeof(fp16_t) == 2, "mollm FP16 storage must be binary16");

struct Capabilities {
    bool arm_neon = false;
    bool arm_i8mm = false;
    bool fp16_vector_math = false;
    bool fp16_kv_cache = false;
    bool fp16_interleaved_weights = false;
    bool fp16_m2_shared_weight = false;
    bool x86_avx2 = false;
    bool x86_fma = false;
    bool x86_f16c = false;
    bool x86_avx512 = false;
    bool x86_avx512_vnni = false;
    X86Isa x86_isa = X86Isa::SCALAR;
};

const Capabilities& capabilities();
const char* isa_name();

// Hint while polling a CPU worker.  The ARM and scalar implementations live
// in separately selected translation units so no foreign assembly reaches a
// target compiler.
void relax();

// Handle a package-native packed INT4 matrix when the selected CPU provider
// has a portable decoder.  Returning false leaves the normal matmul dispatch
// to select its architecture-specific kernel.
bool matmul_int4_packed(const Tensor& A, const Tensor& B, Tensor& C, int lda,
                        int ldc, ThreadPool* thread_pool);

// Architecture-provider dense kernels. Returning false asks the caller to use
// the portable scalar implementation. The x86 provider binds these to
// separately compiled AVX2 or AVX-512 translation units after one runtime
// probe.
bool matmul_dense_fp32_range(const float* A, const float* B, float* C, int N,
                             int K, int lda, int K_weight, int ldc,
                             int m_begin, int m_end);
bool matmul_dense_fp16_range(const float* A, const fp16_t* B, float* C, int N,
                             int K, int lda, int K_weight, int ldc,
                             int m_begin, int m_end, bool interleaved);
bool matmul_dense_fp16_m2_range_n(
    const float* A, const fp16_t* B, float* C,
    int N, int K, int lda, int K_weight, int ldc,
    int n_begin, int n_end);
bool matmul_int8_range(const float* A, const int8_t* B, const float* scales,
                       float* C, int N, int K, int group_size,
                       int groups_per_row, int lda, int K_weight, int ldc,
                       int m_begin, int m_end, int n_begin, int n_end,
                       bool interleaved);

}  // namespace mollm::cpu

#if !MOLLM_CPU_ARM_NEON && !defined(__clang__) && !defined(__CUDACC__)
// Legacy CPU kernels still spell their storage element as `__fp16`.  Keep the
// compatibility name at this one architecture boundary while those kernels
// are moved behind providers; no generic caller needs a compiler extension.
using __fp16 = mollm::cpu::fp16_t;
#endif

// Transitional compatibility for existing NEON kernels.  New generic code
// should use mollm::cpu::Capabilities instead of testing this macro.
#ifndef HAS_NEON
#define HAS_NEON MOLLM_CPU_ARM_NEON
#endif
