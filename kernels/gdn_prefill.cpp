// NEON-optimised GDN prefill kernel for Qwen3.5 linear attention.
// Head-outer loop preserves sequential token dependency within each head.
// Heads are parallelised via thread_pool when available.

#include "kernels/gdn.h"
#include "kernels/gdn_neon.h"
#include "kernels/threading.h"

#if HAS_NEON
#include <cstring>

void kernel_gdn_prefill_neon(const OpParams& params,
                              const std::vector<const Tensor*>& inputs,
                              std::vector<Tensor*>& outputs,
                              ThreadPool* thread_pool) {
    int num_heads   = graph_params::get_i32(params, 0, 16);
    int k_head_dim  = graph_params::get_i32(params, 1, 128);
    int v_head_dim  = graph_params::get_i32(params, 2, 128);
    int seq_len     = graph_params::get_i32(params, 3, 4);
    int num_v_heads = graph_params::get_i32(params, 7, num_heads);
    int n_real      = graph_params::get_i32(params, 6, seq_len);
    float rms_eps   = graph_params::get_f32(params, 0, 1e-6f);
    float l2_eps    = graph_params::get_f32(params, 1, 1e-6f);
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

    int qkv_dim   = num_heads * k_head_dim;
    int z_dim     = num_v_heads * v_head_dim;
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

    // Process heads in parallel. Each head's tokens are sequential (stateful).
    // Heads are independent when repeat=1; when repeat>1, heads sharing a key_head
    // must be serialised. We group by key_head for correctness.
    auto process_head = [&](int vh_start, int vh_end, int /*tid*/) {
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

                float a_h = a_data[t * num_v_heads + vh];
                float b_h = b_data[t * num_v_heads + vh];
                float sp = gdn_softplusf(a_h + dtb_data[vh]);
                float g_t = neg_exp_A[vh] * sp;
                float g_t_exp = std::exp(g_t);
                float beta_t = gdn_sigmoidf(b_h);

                const float* z_row = z_data + t * z_dim + vh * v_head_dim;
                float* out_head = out_data + t * z_dim + vh * v_head_dim;

                gdn_recurrence_neon(q, k_buf, v_buf,
                                    g_t_exp, beta_t, state_h,
                                    norm_data, z_row, out_head,
                                    k_head_dim, v_head_dim, scale, rms_eps);
            }
        }
    };

    // Serial processing. Multi-threading deferred:
    // parallel_for causes intermittent incorrect results in ctest
    // (possible worker thread init race). Serial-vs-parallel unit test
    // passes when run standalone. Investigation ongoing.
    (void)thread_pool;
    process_head(0, num_v_heads, 0);
}

#endif // HAS_NEON
