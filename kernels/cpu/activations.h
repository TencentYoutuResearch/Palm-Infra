#pragma once

#include "core/activation.h"
#include "backends/cpu/platform.h"

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
// models/transpile.py.
//
// To add a new activation:
//   1. Add enum value in core/activation.h.
//   2. Add scalar + NEON branches in the apply_* functions.
//   3. Add Python enum value in transpile.py.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// sigmoid (FP32, NEON). Also used by SILU.
//
// Evaluate exp(-abs(x)) with range reduction and a degree-six polynomial.
// Keeping the exponent nonpositive avoids overflow at either sigmoid tail.
// The final division retains FP32 accuracy instead of a raw reciprocal estimate.
// ---------------------------------------------------------------------------
#if HAS_NEON
static inline float32x4_t sigmoid_f32_neon(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.f);
    const float32x4_t magnitude = vabsq_f32(x);
    const float32x4_t z = vnegq_f32(vminq_f32(magnitude, vdupq_n_f32(87.f)));
    const float32x4_t n = vrndnq_f32(vmulq_n_f32(z, 1.4426950408889634f));
    // Split ln(2) keeps the reduced argument accurate near the negative tail.
    float32x4_t r = vfmsq_n_f32(z, n, 0.693145751953125f);
    r = vfmsq_n_f32(r, n, 1.428606765330187e-6f);
    float32x4_t p = vdupq_n_f32(1.f / 720.f);
    p = vfmaq_f32(vdupq_n_f32(1.f / 120.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f / 24.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f / 6.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(one, p, r);
    p = vfmaq_f32(one, p, r);
    const int32x4_t exponent = vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127));
    float32x4_t e = vmulq_f32(p, vreinterpretq_f32_s32(vshlq_n_s32(exponent, 23)));
    e = vbslq_f32(vcgtq_f32(magnitude, vdupq_n_f32(87.f)), vdupq_n_f32(0.f), e);
    const float32x4_t numerator = vbslq_f32(vcltq_f32(x, vdupq_n_f32(0.f)), e, one);
    return vdivq_f32(numerator, vaddq_f32(one, e));
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
    case Activation::RELU_SQUARED: {
        float y = x > 0.f ? x : 0.f;
        return y * y;
    }
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
    case Activation::RELU_SQUARED: {
        float32x4_t y = vmaxq_f32(x, vdupq_n_f32(0.f));
        return vmulq_f32(y, y);
    }
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
