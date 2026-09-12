// Shared NEON helpers for GDN prefill and decode kernels.
#pragma once

#if HAS_NEON
#include <arm_neon.h>
#include <cmath>

static inline float gdn_sigmoidf(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static inline float gdn_softplusf(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return std::exp(x);
    return std::log1pf(std::exp(x));
}

// L2-normalise a D-dim vector in-place.
static inline void gdn_l2norm_neon(float* x, int D, float eps) {
    float32x4_t s = vdupq_n_f32(0.f);
    for (int i = 0; i < D; i += 4)
        s = vmlaq_f32(s, vld1q_f32(x + i), vld1q_f32(x + i));
    float inv = 1.f / std::sqrt(vaddvq_f32(s) + eps);
    float32x4_t inv4 = vdupq_n_f32(inv);
    for (int i = 0; i < D; i += 4)
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), inv4));
}

// NEON matvec: y[:V] = M[K*V] @ x[K]   (M row-major, row dk has V elements)
static inline void gdn_matvec_neon(const float* M, const float* x,
                                    float* y, int K, int V) {
    for (int dv = 0; dv < V; dv += 4)
        vst1q_f32(y + dv, vdupq_n_f32(0.f));
    for (int dk = 0; dk < K; dk++) {
        float32x4_t xk4 = vdupq_n_f32(x[dk]);
        const float* row = M + dk * V;
        for (int dv = 0; dv < V; dv += 4) {
            float32x4_t mv = vld1q_f32(row + dv);
            float32x4_t yv = vld1q_f32(y + dv);
            vst1q_f32(y + dv, vmlaq_f32(yv, mv, xk4));
        }
    }
}

// NEON rank-1 update: M[dk*V + dv] += k[dk] * delta[dv]
static inline void gdn_rank1_update_neon(float* M, const float* k,
                                          const float* delta, int K, int V) {
    for (int dk = 0; dk < K; dk++) {
        float32x4_t kk4 = vdupq_n_f32(k[dk]);
        float* row = M + dk * V;
        for (int dv = 0; dv < V; dv += 4) {
            float32x4_t mv = vld1q_f32(row + dv);
            float32x4_t dv4 = vld1q_f32(delta + dv);
            vst1q_f32(row + dv, vmlaq_f32(mv, dv4, kk4));
        }
    }
}

// Core GDN recurrence + RMSNormGated for one (head, token).
// q, k must already be L2-normalised.
//
// Fused passes reduce state memory traffic from 5 passes (3 read + 2 write)
// to 2 passes (1 read+write each). State is 128×128=64KB, so this cuts
// memory traffic from ~320KB to ~128KB per (head, token).
static inline void gdn_recurrence_neon(
    const float* q, const float* k, const float* v,
    float g_t_exp, float beta_t, float* state_h,
    const float* norm_w, const float* z_row,
    float* out_head, int k_dim, int v_dim,
    float scale, float rms_eps, bool sigmoid_output_gate)
{
    alignas(16) float kv_mem[128] = {0};
    alignas(16) float attn_out[128] = {0};

    // ---- Pass 1: fused decay + matvec1 (kv_mem = state @ k) ----
    // Block value columns so the matvec accumulators stay in registers across
    // the complete K loop instead of being loaded/stored for every state row.
    {
        float32x4_t g4 = vdupq_n_f32(g_t_exp);
        int dv = 0;
        for (; dv + 64 <= v_dim; dv += 64) {
            float32x4_t acc[16];
            for (int j = 0; j < 16; j++)
                acc[j] = vdupq_n_f32(0.f);
            for (int dk = 0; dk < k_dim; dk++) {
                float32x4_t k4 = vdupq_n_f32(k[dk]);
                float* row = state_h + dk * v_dim + dv;
                for (int j = 0; j < 16; j++) {
                    float32x4_t rv = vld1q_f32(row + j * 4);
                    rv = vmulq_f32(rv, g4);
                    vst1q_f32(row + j * 4, rv);
                    acc[j] = vmlaq_f32(acc[j], rv, k4);
                }
            }
            for (int j = 0; j < 16; j++)
                vst1q_f32(kv_mem + dv + j * 4, acc[j]);
        }
        for (; dv < v_dim; dv += 4) {
            float32x4_t acc = vdupq_n_f32(0.f);
            for (int dk = 0; dk < k_dim; dk++) {
                float32x4_t k4 = vdupq_n_f32(k[dk]);
                float* row = state_h + dk * v_dim + dv;
                float32x4_t rv = vmulq_f32(vld1q_f32(row), g4);
                vst1q_f32(row, rv);
                acc = vmlaq_f32(acc, rv, k4);
            }
            vst1q_f32(kv_mem + dv, acc);
        }
    }

    // 3. delta = (v - kv_mem) * beta_t
    alignas(16) float delta[128];
    {
        float32x4_t b4 = vdupq_n_f32(beta_t);
        for (int dv = 0; dv < v_dim; dv += 4) {
            float32x4_t vv = vld1q_f32(v + dv);
            float32x4_t kv = vld1q_f32(kv_mem + dv);
            vst1q_f32(delta + dv, vmulq_f32(vsubq_f32(vv, kv), b4));
        }
    }

    // ---- Pass 2: fused rank1_update + matvec2 (attn_out = state @ q) ----
    // Keep both delta and output accumulators resident for a 32-value block.
    {
        int dv = 0;
        for (; dv + 32 <= v_dim; dv += 32) {
            float32x4_t delta_v[8];
            float32x4_t acc[8];
            for (int j = 0; j < 8; j++) {
                delta_v[j] = vld1q_f32(delta + dv + j * 4);
                acc[j] = vdupq_n_f32(0.f);
            }
            for (int dk = 0; dk < k_dim; dk++) {
                float32x4_t k4 = vdupq_n_f32(k[dk]);
                float32x4_t q4 = vdupq_n_f32(q[dk]);
                float* row = state_h + dk * v_dim + dv;
                for (int j = 0; j < 8; j++) {
                    float32x4_t rv = vld1q_f32(row + j * 4);
                    rv = vmlaq_f32(rv, delta_v[j], k4);
                    vst1q_f32(row + j * 4, rv);
                    acc[j] = vmlaq_f32(acc[j], rv, q4);
                }
            }
            for (int j = 0; j < 8; j++)
                vst1q_f32(attn_out + dv + j * 4, acc[j]);
        }
        for (; dv < v_dim; dv += 4) {
            float32x4_t delta_v = vld1q_f32(delta + dv);
            float32x4_t acc = vdupq_n_f32(0.f);
            for (int dk = 0; dk < k_dim; dk++) {
                float32x4_t k4 = vdupq_n_f32(k[dk]);
                float32x4_t q4 = vdupq_n_f32(q[dk]);
                float* row = state_h + dk * v_dim + dv;
                float32x4_t rv = vld1q_f32(row);
                rv = vmlaq_f32(rv, delta_v, k4);
                vst1q_f32(row, rv);
                acc = vmlaq_f32(acc, rv, q4);
            }
            vst1q_f32(attn_out + dv, acc);
        }
    }

    // 5b. attn_out *= scale
    {
        float32x4_t sc4 = vdupq_n_f32(scale);
        for (int dv = 0; dv < v_dim; dv += 4)
            vst1q_f32(attn_out + dv, vmulq_f32(vld1q_f32(attn_out + dv), sc4));
    }

    // 6. RMSNormGated
    {
        float32x4_t ss = vdupq_n_f32(0.f);
        for (int dv = 0; dv < v_dim; dv += 4) {
            float32x4_t a = vld1q_f32(attn_out + dv);
            ss = vmlaq_f32(ss, a, a);
        }
        float rms = 1.f / std::sqrt(vaddvq_f32(ss) / (float)v_dim + rms_eps);
        float32x4_t rms4 = vdupq_n_f32(rms);
        for (int dv = 0; dv < v_dim; dv += 4) {
            float32x4_t ao = vld1q_f32(attn_out + dv);
            float32x4_t nw = vld1q_f32(norm_w + dv);
            float32x4_t normed = vmulq_f32(vmulq_f32(ao, rms4), nw);
            float normed_lane[4];
            vst1q_f32(normed_lane, normed);
            for (int j = 0; j < 4; j++) {
                float z = z_row[dv + j];
                float sigmoid_z = 1.f / (1.f + std::exp(-z));
                float gate = sigmoid_output_gate ? sigmoid_z : z * sigmoid_z;
                out_head[dv + j] = normed_lane[j] * gate;
            }
        }
    }
}

#endif // HAS_NEON
