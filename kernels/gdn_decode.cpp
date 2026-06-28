// NEON-optimised GDN decode kernel for Qwen3.5 linear attention.
// Only for seq_len=1 (decode). Prefill stays on the scalar path.
//
// Key dims (hardcoded for Qwen3.5-0.8B): k_dim=128, v_dim=128, num_heads=16.
// The kernel still reads dims from params for safety but vectorisation
// relies on these being multiples of 4 (which they always are).

#include "kernels/gdn.h"

#if HAS_NEON
#include <arm_neon.h>
#include <cmath>

static inline float sigmoidf(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static inline float softplusf(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return std::exp(x);
    return std::log1pf(std::exp(x));
}

// L2-normalise a D-dim vector in-place (D must be multiple of 4).
// NEON reduce + broadcast.
static inline void l2norm_neon(float* x, int D, float eps) {
    float32x4_t s = vdupq_n_f32(0.f);
    for (int i = 0; i < D; i += 4)
        s = vmlaq_f32(s, vld1q_f32(x + i), vld1q_f32(x + i));
    float sum = vaddvq_f32(s);
    float inv = 1.f / std::sqrt(sum + eps);
    float32x4_t inv4 = vdupq_n_f32(inv);
    for (int i = 0; i < D; i += 4)
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), inv4));
}

// NEON matvec: y[0..V-1] = M[V*K] @ x[K]
// M is [K, V] row-major: row dk has V contiguous elements.
static inline void matvec_128x128_neon(const float* M, const float* x,
                                       float* y, int K, int V) {
    // y[:] = 0
    for (int dv = 0; dv < V; dv += 4)
        vst1q_f32(y + dv, vdupq_n_f32(0.f));

    for (int dk = 0; dk < K; dk++) {
        float xk = x[dk];
        float32x4_t xk4 = vdupq_n_f32(xk);
        const float* row = M + dk * V;
        for (int dv = 0; dv < V; dv += 4) {
            float32x4_t mv = vld1q_f32(row + dv);
            float32x4_t yv = vld1q_f32(y + dv);
            vst1q_f32(y + dv, vmlaq_f32(yv, mv, xk4));
        }
    }
}

// NEON rank-1 update: M[dv + dk*V] += k[dk] * delta[dv]
static inline void rank1_update_128x128_neon(float* M, const float* k,
                                              const float* delta, int K, int V) {
    for (int dk = 0; dk < K; dk++) {
        float kk = k[dk];
        float32x4_t kk4 = vdupq_n_f32(kk);
        float* row = M + dk * V;
        for (int dv = 0; dv < V; dv += 4) {
            float32x4_t mv = vld1q_f32(row + dv);
            float32x4_t dv4 = vld1q_f32(delta + dv);
            vst1q_f32(row + dv, vmlaq_f32(mv, dv4, kk4));
        }
    }
}

// NEON decode for a single head.
// q, k, v are already extracted from qkv_conv and L2-normalised.
static void gdn_decode_head_neon(
    const float* q, const float* k, const float* v,
    float g_t_exp, float beta_t, float* state_h,
    const float* norm_w, const float* z_row,
    float* out_head, int k_dim, int v_dim,
    float scale, float rms_eps)
{
    int state_size = k_dim * v_dim;

    // 1. Decay: state *= g_t_exp
    {
        float32x4_t g4 = vdupq_n_f32(g_t_exp);
        for (int i = 0; i < state_size; i += 4)
            vst1q_f32(state_h + i,
                      vmulq_f32(vld1q_f32(state_h + i), g4));
    }

    // 2. kv_mem = state @ k
    alignas(16) float kv_mem[128] = {0};
    matvec_128x128_neon(state_h, k, kv_mem, k_dim, v_dim);

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

    // 4. state += outer(k, delta)
    rank1_update_128x128_neon(state_h, k, delta, k_dim, v_dim);

    // 5. attn_out = (state @ q) * scale
    alignas(16) float attn_out[128];
    matvec_128x128_neon(state_h, q, attn_out, k_dim, v_dim);
    {
        float32x4_t sc4 = vdupq_n_f32(scale);
        for (int dv = 0; dv < v_dim; dv += 4)
            vst1q_f32(attn_out + dv,
                      vmulq_f32(vld1q_f32(attn_out + dv), sc4));
    }

    // 6. RMSNormGated: out = rms_norm(attn_out, norm_w) * silu(z)
    {
        float32x4_t ss = vdupq_n_f32(0.f);
        for (int dv = 0; dv < v_dim; dv += 4) {
            float32x4_t a = vld1q_f32(attn_out + dv);
            ss = vmlaq_f32(ss, a, a);
        }
        float sum_sq = vaddvq_f32(ss);
        float rms = 1.f / std::sqrt(sum_sq / (float)v_dim + rms_eps);

        float32x4_t rms4 = vdupq_n_f32(rms);
        for (int dv = 0; dv < v_dim; dv += 4) {
            float32x4_t ao = vld1q_f32(attn_out + dv);
            float32x4_t nw = vld1q_f32(norm_w + dv);

            float32x4_t normed = vmulq_f32(vmulq_f32(ao, rms4), nw);

            // silu(z) = z * sigmoid(z) — scalar per lane
            float result[4];
            vst1q_f32(result, normed);
            for (int i = 0; i < 4; i++) {
                float z_val = z_row[dv + i];
                result[i] *= z_val * sigmoidf(z_val);
            }
            vst1q_f32(out_head + dv, vld1q_f32(result));
        }
    }
}

void kernel_gdn_decode_neon(const OpParams& params,
                             const std::vector<const Tensor*>& inputs,
                             std::vector<Tensor*>& outputs) {
    int num_heads  = graph_params::get_i32(params, 0, 16);
    int k_head_dim = graph_params::get_i32(params, 1, 128);
    int v_head_dim = graph_params::get_i32(params, 2, 128);
    float rms_eps  = graph_params::get_f32(params, 0, 1e-6f);
    float l2_eps   = graph_params::get_f32(params, 1, 1e-6f);
    float scale    = graph_params::get_f32(params, 2, 0.f);
    if (scale == 0.f) scale = 1.f / std::sqrt((float)k_head_dim);

    if (inputs.size() < 8 || outputs.empty()) return;

    const float* qkv_data   = inputs[0]->ptr<float>();
    const float* a_data     = inputs[1]->ptr<float>();
    const float* b_data     = inputs[2]->ptr<float>();
    const float* z_data     = inputs[3]->ptr<float>();
    const float* A_log_data = inputs[4]->ptr<float>();
    const float* dtb_data   = inputs[5]->ptr<float>();
    const float* norm_data  = inputs[6]->ptr<float>();
    float* state_data       = reinterpret_cast<float*>(inputs[7]->data);
    float* out_data         = outputs[0]->ptr<float>();

    int qkv_dim  = num_heads * k_head_dim;   // 2048
    int z_dim    = num_heads * v_head_dim;   // 2048
    int state_sz = k_head_dim * v_head_dim;  // 16384

    // Precompute neg_exp_A
    alignas(16) float neg_exp_A[16];
    for (int h = 0; h < num_heads; h++)
        neg_exp_A[h] = -std::exp(A_log_data[h]);

    // decode: seq_len=1, data at t=0
    for (int h = 0; h < num_heads; h++) {
        float* state_h = state_data + h * state_sz;

        // Extract q, k, v from qkv_conv
        alignas(16) float q[128], k_buf[128], v_buf[128];
        int q_base = h * k_head_dim;
        int k_base = qkv_dim + h * k_head_dim;
        int v_base = 2 * qkv_dim + h * v_head_dim;
        for (int d = 0; d < k_head_dim; d++) {
            q[d]     = qkv_data[(q_base + d) * 1 + 0];
            k_buf[d] = qkv_data[(k_base + d) * 1 + 0];
        }
        for (int d = 0; d < v_head_dim; d++)
            v_buf[d] = qkv_data[(v_base + d) * 1 + 0];

        // L2 norm q, k
        l2norm_neon(q, k_head_dim, l2_eps);
        l2norm_neon(k_buf, k_head_dim, l2_eps);

        // g, beta
        float a_h = a_data[h];  // a[t=0, h]
        float b_h = b_data[h];
        float sp = softplusf(a_h + dtb_data[h]);
        float g_t = neg_exp_A[h] * sp;
        float g_t_exp = std::exp(g_t);
        float beta_t = sigmoidf(b_h);

        // Core recurrence + RMSNormGated
        const float* z_row = z_data + h * v_head_dim;  // t=0
        float* out_head = out_data + h * v_head_dim;   // t=0

        gdn_decode_head_neon(q, k_buf, v_buf,
                             g_t_exp, beta_t, state_h,
                             norm_data, z_row, out_head,
                             k_head_dim, v_head_dim, scale, rms_eps);
    }
}

#endif // HAS_NEON
