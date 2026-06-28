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
    int num_heads, int k_dim, int v_dim, int seq_len,
    int data_seq_len,  // actual seq_len of the data buffer (may be > seq_len for padded prefill)
    bool use_l2norm, float rms_eps, float l2norm_eps, float scale)
{
    int qkv_dim   = num_heads * k_dim;  // 2048
    int z_dim     = num_heads * v_dim;  // 2048
    int state_size = k_dim * v_dim;     // 16384

    std::vector<float> q_pre(k_dim), k_pre(k_dim), v_t(v_dim);
    std::vector<float> q_n(k_dim), k_n(k_dim);
    std::vector<float> kv_mem(v_dim), delta(v_dim), attn_out(v_dim);

    for (int h = 0; h < num_heads; h++) {
        float* state_h = state + h * state_size;
        float nea = neg_exp_A[h];
        float dtb = dt_bias[h];

        for (int t = 0; t < seq_len; t++) {
            // Extract q, k, v from qkv_conv [qkv_total, data_seq_len] row-major.
            // qkv index: dim_idx = h * k_dim + d  for q,
            //            dim_idx = qkv_dim + h * k_dim + d  for k,
            //            dim_idx = 2 * qkv_dim + h * v_dim + d  for v.
            // Data: qkv[dim_idx * data_seq_len + t]
            int qkv_total = qkv_dim * 3;
            for (int d = 0; d < k_dim; d++) {
                q_pre[d] = qkv[(h * k_dim + d) * data_seq_len + t];
                k_pre[d] = qkv[(qkv_dim + h * k_dim + d) * data_seq_len + t];
            }
            for (int d = 0; d < v_dim; d++)
                v_t[d] = qkv[(2 * qkv_dim + h * v_dim + d) * data_seq_len + t];

            // g = -exp(A_log[h]) * softplus(a[t,h] + dt_bias[h])
            float a_ht = a[t * num_heads + h];
            float sp = softplusf(a_ht + dtb);
            float g_t = nea * sp;
            float g_t_exp = std::exp(g_t);

            // beta = sigmoid(b[t,h])
            float b_ht = b[t * num_heads + h];
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
            // Output layout: [z_dim, seq_len] row-major.
            // out[global_dim * seq_len + t] = value at (dim=global_dim, pos=t)
            //   where global_dim = h * v_dim + d
            // This matches matmul's lda = seq_len (which is shape[1] / es? No, lda = stride[1]/es = z_dim)
            // Actually lda = z_dim. So out[global_dim + t * z_dim] is correct for matmul.
            // Wait - matmul reads A[k + m*lda] where lda = stride[1]/es = z_dim.
            // For row m (output dim), reads A[m*z_dim + k] for k=0..K-1.
            // In [z_dim, seq] row-major: flat[m*z_dim + k]... no.
            // [z_dim=2048, seq=4] row-major: flat[0..3]=pos0..3,dim0; flat[4..7]=pos0..3,dim1.
            // dim m=1 starts at flat[m*seq=4] not flat[m*z_dim=2048]!
            // So matmul expects A[k + m*lda] where lda = z_dim (stride[1]/es).
            // A[0 + 1*2048] = A[2048] in [z_dim,seq] = dim0,pos1?? No.
            // [2048,4] row-major: flat[2048] = dim=0,pos=1 (since 2048=0*2048+1*4? no)
            // Actually: flat index f = dim * 4 + pos. f=2048 => dim=512, pos=0.
            // A[2048] = dim 512, pos 0. But lda=2048 means matmul expects row 1 at offset 2048.
            // Row 1 = dim 1. dim 1 starts at flat[1*4]=flat[4].
            // matmul reads A[4] as the first element of row 1. But it reads A[2048]!
            // MISMATCH!
            //
            // The issue: matmul uses stride[1]=z_dim*es as lda. lda = 2048.
            // But [z_dim,seq] row-major has row stride = seq, not z_dim!
            // [z_dim,seq] row-major: d0=z_dim innermost, d1=seq outer.
            // Row d (dim d) starts at flat[d * seq] = flat[d * 4].
            // lda should be seq=4, but matmul uses stride[1]/es=z_dim=2048.
            //
            // ROOT CAUSE: mlllm's d0-innermost convention!
            // In mlllm, [z_dim, seq] means d0=z_dim innermost, d1=seq outer.
            // compute_strides: stride[0]=es, stride[1]=z_dim*es.
            // matmul lda = stride[1]/es = z_dim.
            // In flat data [z_dim, seq] row-major: flat[d + t*z_dim].
            // Row d starts at flat[d] (since d0=z_dim innermost).
            // So matmul reads A[m*z_dim + k] and GDN must write out[d + t*z_dim].
            // 
            // out[global_dim + t * z_dim] is CORRECT!
            const float* z_row = z + t * z_dim + h * v_dim;
            float sum_sq = 0.f;
            for (int d = 0; d < v_dim; d++) sum_sq += attn_out[d] * attn_out[d];
            float rms = 1.f / std::sqrt(sum_sq / (float)v_dim + rms_eps);
            for (int d = 0; d < v_dim; d++) {
                float normed = attn_out[d] * rms * norm_w[d];
                float silu_z = z_row[d] * sigmoidf(z_row[d]);
                int global_dim = h * v_dim + d;
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
    int n_real      = graph_params::get_i32(params, 6, seq_len); // 0 or missing → all positions
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

    // Precompute neg_exp_A = -exp(A_log[h])
    std::vector<float> neg_exp_A(num_heads);
    for (int h = 0; h < num_heads; h++)
        neg_exp_A[h] = -std::exp(A_log_data[h]);

    // Zero out output for padding positions.
    // Output layout is [z_dim, seq_len] row-major → zero columns process_len..seq_len-1
    int z_dim = num_heads * v_head_dim;
    int process_len = (n_real > 0 && n_real < seq_len) ? n_real : seq_len;
    if (process_len < seq_len) {
        for (int t = process_len; t < seq_len; t++) {
            std::memset(out_data + t * z_dim, 0, z_dim * sizeof(float));
        }
    }

    fused_gdn_head(qkv_data, a_data, b_data, z_data,
                   neg_exp_A.data(), dtb_data, norm_data,
                   state_data, out_data,
                   num_heads, k_head_dim, v_head_dim, process_len, seq_len,
                   use_l2norm, rms_eps, l2norm_eps, scale);
}

// ---------------------------------------------------------------------------
// Decode: single token (seq_len=1).
// ---------------------------------------------------------------------------
void kernel_gdn_decode(const OpParams& params,
                       const std::vector<const Tensor*>& inputs,
                       std::vector<Tensor*>& outputs,
                       ThreadPool* thread_pool) {
    // Decode uses the same params and inputs as prefill.
    // seq_len is 1 for decode.
    kernel_gdn_prefill(params, inputs, outputs, thread_pool);
}
