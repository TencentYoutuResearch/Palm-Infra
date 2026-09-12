// NEON-optimised GDN prefill kernel for Qwen3.5 linear attention.
// Head-outer loop preserves sequential token dependency within each head.
// Heads are parallelised via thread_pool when available.

#include "kernels/cpu/recurrent/gdn.h"
#include "kernels/cpu/arm/gdn_neon.h"
#include "backends/cpu/platform.h"
#include "runtime/threading.h"

#if HAS_NEON
#include <algorithm>
#include <cstring>

void kernel_gdn_prefill_neon(const GdnParams& params,
                              const std::vector<const Tensor*>& inputs,
                              std::vector<Tensor*>& outputs,
                              ThreadPool* thread_pool) {
    int num_heads   = params.num_heads;
    int k_head_dim  = params.k_head_dim;
    int v_head_dim  = params.v_head_dim;
    int seq_len     = params.seq_len;
    int num_v_heads = params.num_v_heads;
    int n_real      = params.real_tokens;
    int flags       = params.flags;
    bool sigmoid_output_gate = (flags & 2) != 0;
    float rms_eps   = params.rms_eps;
    float l2_eps    = params.l2norm_eps;
    float scale     = params.scale;
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

    int qkv_dim   = num_heads * k_head_dim;
    int z_dim     = num_v_heads * v_head_dim;
    int a_row_stride = (int)(inputs[1]->stride[1] / sizeof(float));
    int b_row_stride = (int)(inputs[2]->stride[1] / sizeof(float));
    int z_row_stride = (int)(inputs[3]->stride[1] / sizeof(float));
    int state_sz  = k_head_dim * v_head_dim;
    int repeat    = num_v_heads / num_heads;
    int process_len = (n_real > 0 && n_real < seq_len) ? n_real : seq_len;

    // Zero output for padding positions
    if (process_len < seq_len) {
        for (int t = process_len; t < seq_len; t++)
            std::memset(out_data + t * z_dim, 0, z_dim * sizeof(float));
    }

    // Precompute neg_exp_A per value head
    std::vector<float> neg_exp_A_vec(num_v_heads);
    for (int h = 0; h < num_v_heads; h++)
        neg_exp_A_vec[h] = -std::exp(A_log_data[h]);
    const float* neg_exp_A = neg_exp_A_vec.data();

    // Process heads in parallel. Each value head's tokens are sequential
    // because state is recurrent. When repeat>1, grouping by key head lets
    // repeated value heads reuse the same normalized q/k vectors.
    // parallel_for signature: fn(int thread_id, int begin, int end)
    auto process_head = [&](int /*tid*/, int vh_start, int vh_end) {
        for (int vh = vh_start; vh < vh_end; vh++) {
            int kh = vh / repeat;
            float* state_h = state_data + vh * state_sz;

            for (int t = 0; t < process_len; t++) {
                alignas(16) float q[128], k_buf[128], v_buf[128];
                int q_base = kh * k_head_dim;
                int k_base = qkv_dim + kh * k_head_dim;
                int v_base = 2 * qkv_dim + vh * v_head_dim;
                for (int d = 0; d < k_head_dim; d++) {
                    q[d]     = qkv_data[(q_base + d) * seq_len + t];
                    k_buf[d] = qkv_data[(k_base + d) * seq_len + t];
                }
                for (int d = 0; d < v_head_dim; d++)
                    v_buf[d] = qkv_data[(v_base + d) * seq_len + t];

                gdn_l2norm_neon(q, k_head_dim, l2_eps);
                gdn_l2norm_neon(k_buf, k_head_dim, l2_eps);

                float a_h = a_data[t * a_row_stride + vh];
                float b_h = b_data[t * b_row_stride + vh];
                float sp = gdn_softplusf(a_h + dtb_data[vh]);
                float g_t = neg_exp_A[vh] * sp;
                float g_t_exp = std::exp(g_t);
                float beta_t = gdn_sigmoidf(b_h);

                const float* z_row =
                    z_data + t * z_row_stride + vh * v_head_dim;
                float* out_head = out_data + t * z_dim + vh * v_head_dim;

                gdn_recurrence_neon(q, k_buf, v_buf,
                                    g_t_exp, beta_t, state_h,
                                    norm_data, z_row, out_head,
                                    k_head_dim, v_head_dim, scale, rms_eps,
                                    sigmoid_output_gate);
            }
        }
    };

    auto process_key_head = [&](int /*tid*/, int kh_start, int kh_end) {
        for (int kh = kh_start; kh < kh_end; kh++) {
            int q_base = kh * k_head_dim;
            int k_base = qkv_dim + kh * k_head_dim;

            for (int t = 0; t < process_len; t++) {
                alignas(16) float q[128], k_buf[128], v_buf[128];
                for (int d = 0; d < k_head_dim; d++) {
                    q[d]     = qkv_data[(q_base + d) * seq_len + t];
                    k_buf[d] = qkv_data[(k_base + d) * seq_len + t];
                }

                gdn_l2norm_neon(q, k_head_dim, l2_eps);
                gdn_l2norm_neon(k_buf, k_head_dim, l2_eps);

                int vh_begin = kh * repeat;
                int vh_end = std::min(vh_begin + repeat, num_v_heads);
                for (int vh = vh_begin; vh < vh_end; vh++) {
                    int v_base = 2 * qkv_dim + vh * v_head_dim;
                    for (int d = 0; d < v_head_dim; d++)
                        v_buf[d] = qkv_data[(v_base + d) * seq_len + t];

                    float a_h = a_data[t * a_row_stride + vh];
                    float b_h = b_data[t * b_row_stride + vh];
                    float sp = gdn_softplusf(a_h + dtb_data[vh]);
                    float g_t = neg_exp_A[vh] * sp;
                    float g_t_exp = std::exp(g_t);
                    float beta_t = gdn_sigmoidf(b_h);

                    float* state_h = state_data + vh * state_sz;
                    const float* z_row =
                        z_data + t * z_row_stride + vh * v_head_dim;
                    float* out_head = out_data + t * z_dim + vh * v_head_dim;

                    gdn_recurrence_neon(q, k_buf, v_buf,
                                        g_t_exp, beta_t, state_h,
                                        norm_data, z_row, out_head,
                                        k_head_dim, v_head_dim, scale, rms_eps,
                                        sigmoid_output_gate);
                }
            }
        }
    };

    // Parallelize over heads when safe (repeat=1: independent state per head)
    if (thread_pool && repeat == 1 && num_v_heads >= 4) {
        int chunk = (num_v_heads + thread_pool->num_threads() - 1) / thread_pool->num_threads();
        if (chunk < 1) chunk = 1;
        thread_pool->parallel_for(0, num_v_heads, chunk, process_head);
    } else if (thread_pool && repeat > 1 && num_v_heads >= 8) {
        // Group by key_head so repeated value heads reuse q/k work.
        int num_kh = num_v_heads / repeat;
        int chunk = (num_kh + thread_pool->num_threads() - 1) / thread_pool->num_threads();
        if (chunk < 1) chunk = 1;
        thread_pool->parallel_for(0, num_kh, chunk, process_key_head);
    } else if (repeat > 1) {
        int num_kh = num_v_heads / repeat;
        process_key_head(0, 0, num_kh);
    } else {
        process_head(0, 0, num_v_heads);
    }
}

#endif // HAS_NEON
