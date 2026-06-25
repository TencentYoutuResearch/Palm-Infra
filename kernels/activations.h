#pragma once

#include "kernels/tensor.h"  // for HAS_NEON

#include <cmath>
#include <cstdint>

#if HAS_NEON
#include <arm_neon.h>
#endif

// ---------------------------------------------------------------------------
// Activation functions for fused matmul→activation.
//
// Applied at matmul kernel writeback phase (FP32 domain, after FP16 acc → FP32
// conversion). Two entry points:
//   - apply_activation_scalar(float, Activation)  — for scalar tails
//   - apply_activation_f32_neon(float32x4_t, Activation)  — for vectorized paths
//
// Activation enum values must match Python's `Activation(IntEnum)` in
// python/transpile.py.
//
// To add a new activation:
//   1. Add enum value here.
//   2. Add scalar + NEON branches in the apply_* functions.
//   3. Add Python enum value in transpile.py.
// ---------------------------------------------------------------------------

enum class Activation : int32_t {
    NONE = 0,   // identity — fast path, no per-column branch
    SILU = 1,   // x * sigmoid(x) — SwiGLU gate
    GELU = 2,   // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) — tanh approx
    RELU = 3,   // max(0, x)
    // Future: GELU_ERF, SIGMOID, TANH, ...
};

// ---------------------------------------------------------------------------
// sigmoid (FP32, NEON). Also used by SILU.
//
// Polynomial exp approximation good to ~7 bits in [-88, 88]. Same impl as the
// one previously defined static-inline in graph/execute.cpp — moved here to
// share between the eltwise SILU executor and the fused-activation matmul path.
// ---------------------------------------------------------------------------
#if HAS_NEON
static inline float32x4_t sigmoid_f32_neon(float32x4_t x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    float32x4_t neg_x = vnegq_f32(x);
    neg_x = vmaxq_f32(neg_x, vdupq_n_f32(-88.f));
    neg_x = vminq_f32(neg_x, vdupq_n_f32(88.f));
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    float32x4_t t = vmulq_f32(neg_x, log2e);
    float32x4_t n = vrndmq_f32(t);
    float32x4_t f = vsubq_f32(t, n);
    int32x4_t ni = vcvtq_s32_f32(n);
    float32x4_t pow2n = vreinterpretq_f32_s32(
        vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));
    const float32x4_t c0 = vdupq_n_f32(1.0f);
    const float32x4_t c1 = vdupq_n_f32(0.6931472f);
    const float32x4_t c2 = vdupq_n_f32(0.2402265f);
    const float32x4_t c3 = vdupq_n_f32(0.0555049f);
    const float32x4_t c4 = vdupq_n_f32(0.0096813f);
    float32x4_t pow2f = vfmaq_f32(c3, c4, f);
    pow2f = vfmaq_f32(c2, pow2f, f);
    pow2f = vfmaq_f32(c1, pow2f, f);
    pow2f = vfmaq_f32(c0, pow2f, f);
    float32x4_t ex = vmulq_f32(pow2n, pow2f);
    float32x4_t one = vdupq_n_f32(1.0f);
    return vrecpeq_f32(vaddq_f32(one, ex));
}
#endif  // HAS_NEON

// ---------------------------------------------------------------------------
// Scalar activation (used in writeback tail loops)
// ---------------------------------------------------------------------------
inline float apply_activation_scalar(float x, Activation act) {
    switch (act) {
    case Activation::SILU: {
        float s = 1.f / (1.f + std::exp(-x));
        return x * s;
    }
    case Activation::GELU: {
        // tanh approximation
        const float c = 0.7978845608f;  // sqrt(2/pi)
        float x3 = x * x * x;
        float inner = c * (x + 0.044715f * x3);
        return 0.5f * x * (1.f + std::tanh(inner));
    }
    case Activation::RELU:
        return x > 0.f ? x : 0.f;
    case Activation::NONE:
    default:
        return x;
    }
}

// ---------------------------------------------------------------------------
// NEON 4-wide FP32 activation (used in vectorized writeback paths)
// ---------------------------------------------------------------------------
#if HAS_NEON
inline float32x4_t apply_activation_f32_neon(float32x4_t x, Activation act) {
    switch (act) {
    case Activation::SILU:
        return vmulq_f32(x, sigmoid_f32_neon(x));
    case Activation::GELU: {
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // tanh(y) = 2 * sigmoid(2y) - 1
        float32x4_t x3 = vmulq_f32(vmulq_f32(x, x), x);
        float32x4_t inner = vmulq_n_f32(vfmaq_n_f32(x, x3, 0.044715f), 0.7978845608f);
        float32x4_t two_inner = vmulq_n_f32(inner, 2.f);
        float32x4_t sig = sigmoid_f32_neon(two_inner);
        float32x4_t tanh_v = vsubq_f32(vmulq_n_f32(sig, 2.f), vdupq_n_f32(1.f));
        float32x4_t one_plus_tanh = vaddq_f32(vdupq_n_f32(1.f), tanh_v);
        return vmulq_n_f32(vmulq_f32(one_plus_tanh, x), 0.5f);
    }
    case Activation::RELU:
        return vmaxq_f32(x, vdupq_n_f32(0.f));
    case Activation::NONE:
    default:
        return x;
    }
}
#endif  // HAS_NEON

// ---------------------------------------------------------------------------
// Helper: should activation be applied at column index `n`?
//
// `act_n_len < 0`  → apply to whole output (fast path, no per-column check)
// `act_n_len == 0` → apply to no columns (equivalent to NONE, but explicit)
// `act_n_len > 0`  → apply only to columns in [act_n_begin, act_n_begin + act_n_len)
// ---------------------------------------------------------------------------
inline bool activation_applies_at(int n, int act_n_begin, int act_n_len) {
    if (act_n_len < 0) return true;  // whole output
    if (act_n_len == 0) return false;
    return n >= act_n_begin && n < act_n_begin + act_n_len;
}
