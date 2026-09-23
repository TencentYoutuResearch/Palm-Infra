#pragma once

// CPU architecture-provider boundary. Runtime code and CPU kernels use this
// contract for capability queries and provider dispatch. ARM intrinsics stay
// behind this boundary; generic kernels must not infer their target from
// compiler predefined macros.

#include <cstdint>

#include "core/fp16.h"

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

using fp16_t = mollm::fp16_t;

enum class X86Isa : uint8_t {
    SCALAR = 0,
    AVX2 = 1,
    AVX512 = 2,
};

struct Capabilities {
    bool arm_neon = false;
    bool arm_i8mm = false;
    bool fp16_vector_math = false;
    bool fp16_kv_cache = false;
    bool fp16_interleaved_weights = false;
    bool x86_avx2 = false;
    bool x86_fma = false;
    bool x86_f16c = false;
    bool x86_avx512 = false;
    bool x86_avx512_vnni = false;
    X86Isa x86_isa = X86Isa::SCALAR;
};

const Capabilities& capabilities();
const char* isa_name();

// worker_relax() is declared by runtime/threading.h: the CPU providers
// implement it, but its contract belongs to the worker pool that calls it.

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
bool matmul_int8_range(const float* A, const int8_t* B, const float* scales,
                       float* C, int N, int K, int group_size,
                       int groups_per_row, int lda, int K_weight, int ldc,
                       int m_begin, int m_end, int n_begin, int n_end,
                       bool interleaved);

}  // namespace mollm::cpu

// Transitional compatibility for existing NEON kernels.  New generic code
// should use mollm::cpu::Capabilities instead of testing this macro.
#ifndef HAS_NEON
#define HAS_NEON MOLLM_CPU_ARM_NEON
#endif
