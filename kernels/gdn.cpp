#include "kernels/gdn.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Scalar GDN implementation (ported from ncnn_llm/src/utils/gdr.cpp).
//
// Layout conventions (matching mlllm's tensor convention):
//   q, k: [k_head_dim, seq_len, num_heads] — shape[0]=k_head_dim (innermost),
//         shape[1]=seq_len, shape[2]=num_heads
//   v:    [v_head_dim, seq_len, num_heads]
//   g, beta: [seq_len, num_heads] — shape[0]=seq_len, shape[1]=num_heads
//   state: [v_head_dim, k_head_dim, num_heads]
//   out:  [v_head_dim, seq_len, num_heads]
//
// state[dk * v_head_dim + dv] for head h is at:
//   state_ptr + h * k_head_dim * v_head_dim + dk * v_head_dim + dv
// (state is row-major [k_head_dim, v_head_dim] per head)
// ---------------------------------------------------------------------------

static inline float sigmoidf(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static inline float softplusf(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return std::exp(x);
    return std::log(1.f + std::exp(x));
}

// L2 normalize a row of dim elements.
static inline void l2norm_row(const float* x, float* out, int dim, float eps = 1e-6f) {
    float sum = 0.f;
    for (int i = 0; i < dim; i++) sum += x[i] * x[i];
    float inv_norm = 1.f / std::sqrt(sum + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_norm;
}

// Core recurrence for a single head, processing seq_len tokens.
// state: [k_head_dim * v_head_dim] (in/out, FP32)
// q/k: [k_head_dim, seq_len] (contiguous per head)
// v:   [v_head_dim, seq_len]
// g/beta: [seq_len]
// out: [v_head_dim, seq_len]
static void gdn_recurrent_scalar(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta,
    float* state, float* out,
    int seq_len, int k_head_dim, int v_head_dim)
{
    float scale = 1.f / std::sqrt((float)k_head_dim);

    for (int t = 0; t < seq_len; t++) {
        const float* q_t = q + t * k_head_dim;
        const float* k_t = k + t * k_head_dim;
        const float* v_t = v + t * v_head_dim;
        float* out_t = out + t * v_head_dim;

        float g_t_exp = std::exp(g[t]);

        // 1. Decay state: state *= exp(g_t)
        int state_size = k_head_dim * v_head_dim;
        for (int i = 0; i < state_size; i++) {
            state[i] *= g_t_exp;
        }

        // 2. kv_mem = state @ k_t  (matvec: [k_dim, v_dim] @ [k_dim] → [v_dim])
        //    state is row-major [k_dim, v_dim]: state[dk * v_dim + dv]
        float kv_mem[256];  // max v_head_dim
        for (int dv = 0; dv < v_head_dim; dv++) kv_mem[dv] = 0.f;
        for (int dk = 0; dk < k_head_dim; dk++) {
            float k_val = k_t[dk];
            const float* state_row = state + dk * v_head_dim;
            for (int dv = 0; dv < v_head_dim; dv++) {
                kv_mem[dv] += state_row[dv] * k_val;
            }
        }

        // 3. delta = (v_t - kv_mem) * beta_t
        float delta[256];
        float beta_t = beta[t];
        for (int dv = 0; dv < v_head_dim; dv++) {
            delta[dv] = (v_t[dv] - kv_mem[dv]) * beta_t;
        }

        // 4. state += outer(k_t, delta)  (rank-1 update)
        for (int dk = 0; dk < k_head_dim; dk++) {
            float k_val = k_t[dk];
            float* state_row = state + dk * v_head_dim;
            for (int dv = 0; dv < v_head_dim; dv++) {
                state_row[dv] += k_val * delta[dv];
            }
        }

        // 5. out_t = (state @ q_t) * scale  (matvec: [k_dim, v_dim]^T @ [k_dim] → [v_dim])
        for (int dv = 0; dv < v_head_dim; dv++) {
            float sum = 0.f;
            for (int dk = 0; dk < k_head_dim; dk++) {
                sum += state[dk * v_head_dim + dv] * q_t[dk];
            }
            out_t[dv] = sum * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill kernel: processes seq_len > 1 tokens.
// Supports per-head parallelism via thread_pool (future).
// ---------------------------------------------------------------------------
void kernel_gdn_prefill(const OpParams& params,
                        const std::vector<const Tensor*>& inputs,
                        std::vector<Tensor*>& outputs,
                        ThreadPool* thread_pool) {
    int num_heads   = graph_params::get_i32(params, 0, 16);
    int k_head_dim  = graph_params::get_i32(params, 1, 128);
    int v_head_dim  = graph_params::get_i32(params, 2, 128);
    bool use_l2norm = graph_params::get_i32(params, 3, 1) != 0;

    if (inputs.size() < 6 || outputs.size() < 2) {
        return;
    }

    const Tensor& q_in = *inputs[0];       // [k_head_dim, seq, num_heads]
    const Tensor& k_in = *inputs[1];       // [k_head_dim, seq, num_heads]
    const Tensor& v_in = *inputs[2];       // [v_head_dim, seq, num_heads]
    const Tensor& g_in = *inputs[3];       // [seq, num_heads]
    const Tensor& beta_in = *inputs[4];    // [seq, num_heads]
    Tensor& state = *outputs[0];           // [v_head_dim, k_head_dim, num_heads]
    Tensor& out = *outputs[1];             // [v_head_dim, seq, num_heads]

    int seq_len = (int)q_in.shape[1];

    const float* q_data = q_in.ptr<float>();
    const float* k_data = k_in.ptr<float>();
    const float* v_data = v_in.ptr<float>();
    const float* g_data = g_in.ptr<float>();
    const float* beta_data = beta_in.ptr<float>();
    float* state_data = state.ptr<float>();
    float* out_data = out.ptr<float>();

    // Per-head stride for q/k: k_head_dim * seq_len
    int qk_head_stride = k_head_dim * seq_len;
    int v_head_stride = v_head_dim * seq_len;
    int state_head_stride = k_head_dim * v_head_dim;

    for (int h = 0; h < num_heads; h++) {
        const float* q_h = q_data + h * qk_head_stride;
        const float* k_h = k_data + h * qk_head_stride;
        const float* v_h = v_data + h * v_head_stride;
        const float* g_h = g_data + h * seq_len;
        const float* beta_h = beta_data + h * seq_len;
        float* state_h = state_data + h * state_head_stride;
        float* out_h = out_data + h * v_head_stride;

        // L2 normalize Q/K per-token if needed.
        if (use_l2norm) {
            int qk_size = seq_len * k_head_dim;
            std::vector<float> q_norm(qk_size);
            std::vector<float> k_norm(qk_size);
            // Normalize each token's Q/K vector.
            for (int t = 0; t < seq_len; t++) {
                l2norm_row(q_h + t * k_head_dim, q_norm.data() + t * k_head_dim, k_head_dim);
                l2norm_row(k_h + t * k_head_dim, k_norm.data() + t * k_head_dim, k_head_dim);
            }
            gdn_recurrent_scalar(q_norm.data(), k_norm.data(), v_h, g_h, beta_h,
                                 state_h, out_h, seq_len, k_head_dim, v_head_dim);
        } else {
            gdn_recurrent_scalar(q_h, k_h, v_h, g_h, beta_h,
                                 state_h, out_h, seq_len, k_head_dim, v_head_dim);
        }
    }
}

// ---------------------------------------------------------------------------
// Decode kernel: single token (seq_len=1).
// Same recurrence but no loop — just one step.
// ---------------------------------------------------------------------------
void kernel_gdn_decode(const OpParams& params,
                       const std::vector<const Tensor*>& inputs,
                       std::vector<Tensor*>& outputs,
                       ThreadPool* thread_pool) {
    int num_heads   = graph_params::get_i32(params, 0, 16);
    int k_head_dim  = graph_params::get_i32(params, 1, 128);
    int v_head_dim  = graph_params::get_i32(params, 2, 128);
    bool use_l2norm = graph_params::get_i32(params, 3, 1) != 0;

    if (inputs.size() < 6 || outputs.size() < 2) return;

    const Tensor& q_in = *inputs[0];       // [k_head_dim, 1, num_heads]
    const Tensor& k_in = *inputs[1];
    const Tensor& v_in = *inputs[2];
    const Tensor& g_in = *inputs[3];       // [1, num_heads]
    const Tensor& beta_in = *inputs[4];
    Tensor& state = *outputs[0];           // [v_head_dim, k_head_dim, num_heads]
    Tensor& out = *outputs[1];            // [v_head_dim, 1, num_heads]

    const float* q_data = q_in.ptr<float>();
    const float* k_data = k_in.ptr<float>();
    const float* v_data = v_in.ptr<float>();
    const float* g_data = g_in.ptr<float>();
    const float* beta_data = beta_in.ptr<float>();
    float* state_data = state.ptr<float>();
    float* out_data = out.ptr<float>();

    int state_head_stride = k_head_dim * v_head_dim;

    for (int h = 0; h < num_heads; h++) {
        const float* q_h = q_data + h * k_head_dim;
        const float* k_h = k_data + h * k_head_dim;
        const float* v_h = v_data + h * v_head_dim;
        float g_h = g_data[h];
        float beta_h = beta_data[h];
        float* state_h = state_data + h * state_head_stride;
        float* out_h = out_data + h * v_head_dim;

        // L2 normalize single token Q/K
        float q_norm[256], k_norm[256];
        const float* q_ptr = q_h;
        const float* k_ptr = k_h;
        if (use_l2norm) {
            l2norm_row(q_h, q_norm, k_head_dim);
            l2norm_row(k_h, k_norm, k_head_dim);
            q_ptr = q_norm;
            k_ptr = k_norm;
        }

        // Single-step recurrence
        gdn_recurrent_scalar(q_ptr, k_ptr, v_h, &g_h, &beta_h,
                             state_h, out_h, 1, k_head_dim, v_head_dim);
    }
}
