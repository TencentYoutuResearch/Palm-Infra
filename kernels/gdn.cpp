#include "kernels/gdn.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Fused GDN core: recurrence + RMSNormGated, scalar implementation.
//
// Consumes inputs in their NATIVE [seq, dim] row-major data layout.
// See gdn.h for the full input/output contract.
//
//   inputs[0] qkv_conv     FP32  data [seq, qkv_total=3*num_heads*k_dim]
//   inputs[1] a_out        FP32  data [seq, num_heads]
//   inputs[2] b_out        FP32  data [seq, num_heads]
//   inputs[3] z_out        FP32  data [seq, num_heads*v_dim]
//   inputs[4] A_log        FP32  [num_heads]               (CONSTANT)
//   inputs[5] dt_bias      FP32  [num_heads]               (CONSTANT)
//   inputs[6] norm_weight  FP32  [v_dim]                    (CONSTANT)
//   inputs[7] gdn_state    FP32  [num_heads, k_dim, v_dim]  (in-place)
//
// Params:
//   i32[0] = num_heads
//   i32[1] = k_head_dim
//   i32[2] = v_head_dim
//   i32[3] = seq_len           (prefill: N, decode: 1)
//   i32[4] = use_qk_l2norm     (1 for Qwen3.5)
//   i32[5] = conv_kernel       (informational, unused)
//
//   f32[0] = rms_eps           (1e-6)
//   f32[1] = l2norm_eps        (1e-6)
//   f32[2] = scale             (1/sqrt(k_dim))
// ---------------------------------------------------------------------------

static inline float sigmoidf(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static inline float softplusf(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return std::exp(x);
    return std::log1pf(std::exp(x));
}

static inline void l2norm_row(const float* x, float* out, int dim, float eps = 1e-6f) {
    float sum = 0.f;
    for (int i = 0; i < dim; i++) sum += x[i] * x[i];
    float inv_norm = 1.f / std::sqrt(sum + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_norm;
}

// Fused recurrence for a single head, processing seq_len tokens.
// qkv layout: [qkv_total, seq] row-major → qkv[dim_idx * seq_len + t]
// a, b layout: [num_heads, seq] → a[t * num_heads + h]
// z layout: [z_dim, seq] → z[t * z_dim + h * v_dim]
// out layout: [z_dim, seq] → out[t * z_dim + h * v_dim]
// state layout: [num_heads, k_dim, v_dim] → state[h * k_dim * v_dim + dk * v_dim + dv]
static void fused_gdn_head(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* neg_exp_A, const float* dt_bias, const float* norm_w,
    float* state, float* out,
    int num_heads, int k_dim, int v_dim, int num_v_heads, int seq_len,
    int data_seq_len,
    bool use_l2norm, float rms_eps, float l2norm_eps, float scale)
{
    int qkv_dim   = num_heads * k_dim;       // key_dim (q/k are key_heads * k_dim)
    int qkv_total = qkv_dim * 2 + num_v_heads * v_dim;  // key_dim*2 + value_dim
    int z_dim     = num_v_heads * v_dim;
    int state_size = k_dim * v_dim;
    int repeat    = num_v_heads / num_heads;  // 1 for 0.8B, 2 for 4B

    std::vector<float> q_pre(k_dim), k_pre(k_dim), v_t(v_dim);
    std::vector<float> q_n(k_dim), k_n(k_dim);
    std::vector<float> kv_mem(v_dim), delta(v_dim), attn_out(v_dim);

    for (int vh = 0; vh < num_v_heads; vh++) {
        int kh = vh / repeat;  // key head index (0.8B: vh==kh, 4B: vh/2)

        float* state_h = state + vh * state_size;
        float nea = neg_exp_A[vh];
        float dtb = dt_bias[vh];

        for (int t = 0; t < seq_len; t++) {
            // Extract q, k from key_heads section of qkv
            // qkv layout: [qkv_total, data_seq_len] row-major
            //   q: dim_idx = kh * k_dim + d
            //   k: dim_idx = qkv_dim + kh * k_dim + d
            //   v: dim_idx = 2 * qkv_dim + vh * v_dim + d
            for (int d = 0; d < k_dim; d++) {
                q_pre[d] = qkv[(kh * k_dim + d) * data_seq_len + t];
                k_pre[d] = qkv[(qkv_dim + kh * k_dim + d) * data_seq_len + t];
            }
            for (int d = 0; d < v_dim; d++)
                v_t[d] = qkv[(2 * qkv_dim + vh * v_dim + d) * data_seq_len + t];

            // g = -exp(A_log[kh]) * softplus(a[t, vh] + dt_bias[kh])
            float a_ht = a[t * num_v_heads + vh];
            float sp = softplusf(a_ht + dtb);
            float g_t = nea * sp;
            float g_t_exp = std::exp(g_t);

            // beta = sigmoid(b[t, vh])
            float b_ht = b[t * num_v_heads + vh];
            float beta_t = sigmoidf(b_ht);

            // L2 normalize q, k (if enabled)
            const float* q_ptr = q_pre.data();
            const float* k_ptr = k_pre.data();
            if (use_l2norm) {
                l2norm_row(q_pre.data(), q_n.data(), k_dim, l2norm_eps);
                l2norm_row(k_pre.data(), k_n.data(), k_dim, l2norm_eps);
                q_ptr = q_n.data();
                k_ptr = k_n.data();
            }

            // 1. Decay: state *= exp(g_t)
            for (int i = 0; i < state_size; i++) state_h[i] *= g_t_exp;

            // 2. kv_mem = state @ k  (matvec: [k_dim, v_dim]^T @ [k_dim] → [v_dim])
            for (int dv = 0; dv < v_dim; dv++) kv_mem[dv] = 0.f;
            for (int dk = 0; dk < k_dim; dk++) {
                float kv = k_ptr[dk];
                const float* row = state_h + dk * v_dim;
                for (int dv = 0; dv < v_dim; dv++)
                    kv_mem[dv] += row[dv] * kv;
            }

            // 3. delta = (v_t - kv_mem) * beta_t
            for (int dv = 0; dv < v_dim; dv++)
                delta[dv] = (v_t[dv] - kv_mem[dv]) * beta_t;

            // 4. state += outer(k, delta)  (rank-1 update)
            for (int dk = 0; dk < k_dim; dk++) {
                float kv = k_ptr[dk];
                float* row = state_h + dk * v_dim;
                for (int dv = 0; dv < v_dim; dv++)
                    row[dv] += kv * delta[dv];
            }

            // 5. attn_out = (state @ q) * scale  (matvec: [k_dim, v_dim]^T @ [k_dim] → [v_dim])
            for (int dv = 0; dv < v_dim; dv++) {
                float s = 0.f;
                for (int dk = 0; dk < k_dim; dk++)
                    s += state_h[dk * v_dim + dv] * q_ptr[dk];
                attn_out[dv] = s * scale;
            }

            // 6. RMSNormGated: out = rms_norm(attn_out, norm_w) * silu(z)
            // Output layout: [z_dim, seq] row-major. Matmul reads lda = stride[1]/es = z_dim.
            // out[global_dim + t * z_dim] matches matmul's A[m*z_dim + k] for out_proj.
            const float* z_row = z + t * z_dim + vh * v_dim;
            float sum_sq = 0.f;
            for (int d = 0; d < v_dim; d++) sum_sq += attn_out[d] * attn_out[d];
            float rms = 1.f / std::sqrt(sum_sq / (float)v_dim + rms_eps);
            for (int d = 0; d < v_dim; d++) {
                float normed = attn_out[d] * rms * norm_w[d];
                float silu_z = z_row[d] * sigmoidf(z_row[d]);
                int global_dim = vh * v_dim + d;
                out[global_dim + t * z_dim] = normed * silu_z;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill: seq_len > 1.
// ---------------------------------------------------------------------------
void kernel_gdn_prefill(const OpParams& params,
                        const std::vector<const Tensor*>& inputs,
                        std::vector<Tensor*>& outputs,
                        ThreadPool* thread_pool) {
    int num_heads   = graph_params::get_i32(params, 0, 16);
    int k_head_dim  = graph_params::get_i32(params, 1, 128);
    int v_head_dim  = graph_params::get_i32(params, 2, 128);
    int seq_len     = graph_params::get_i32(params, 3, 4);
    bool use_l2norm = graph_params::get_i32(params, 4, 1) != 0;
    // params.i32[5] = conv_kernel (unused)
    int n_real      = graph_params::get_i32(params, 6, seq_len);
    int num_v_heads = graph_params::get_i32(params, 7, num_heads);
    float rms_eps   = graph_params::get_f32(params, 0, 1e-6f);
    float l2norm_eps= graph_params::get_f32(params, 1, 1e-6f);
    float scale     = graph_params::get_f32(params, 2, 0.f);
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

    // Precompute neg_exp_A = -exp(A_log[vh]) — per value head
    std::vector<float> neg_exp_A(num_v_heads);
    for (int h = 0; h < num_v_heads; h++)
        neg_exp_A[h] = -std::exp(A_log_data[h]);

    // Zero out output for padding positions.
    int z_dim = num_v_heads * v_head_dim;
    int process_len = (n_real > 0 && n_real < seq_len) ? n_real : seq_len;
    if (process_len < seq_len) {
        for (int t = process_len; t < seq_len; t++) {
            std::memset(out_data + t * z_dim, 0, z_dim * sizeof(float));
        }
    }

    fused_gdn_head(qkv_data, a_data, b_data, z_data,
                   neg_exp_A.data(), dtb_data, norm_data,
                   state_data, out_data,
                   num_heads, k_head_dim, v_head_dim, num_v_heads,
                   process_len, seq_len,
                   use_l2norm, rms_eps, l2norm_eps, scale);
}

// ---------------------------------------------------------------------------
// Decode: single token (seq_len=1).
// ---------------------------------------------------------------------------

#if HAS_NEON
// NEON-optimised decode path (declared in gdn_decode.cpp)
void kernel_gdn_decode_neon(const OpParams& params,
                             const std::vector<const Tensor*>& inputs,
                             std::vector<Tensor*>& outputs);
#endif

void kernel_gdn_decode(const OpParams& params,
                       const std::vector<const Tensor*>& inputs,
                       std::vector<Tensor*>& outputs,
                       ThreadPool* thread_pool) {
#if HAS_NEON
    kernel_gdn_decode_neon(params, inputs, outputs);
#else
    // Fallback: reuse prefill scalar path
    kernel_gdn_prefill(params, inputs, outputs, thread_pool);
#endif
}
