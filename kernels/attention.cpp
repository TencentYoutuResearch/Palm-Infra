#include "kernels/attention.h"
#include "kernels/matmul.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static inline void softmax_row(float* row, int len) {
    float max_val = -INFINITY;
    for (int i = 0; i < len; i++) {
        if (row[i] > max_val) max_val = row[i];
    }
    float sum = 0.f;
    for (int i = 0; i < len; i++) {
        row[i] = std::exp(row[i] - max_val);
        sum += row[i];
    }
    float inv_sum = 1.f / sum;
    for (int i = 0; i < len; i++) {
        row[i] *= inv_sum;
    }
}

// apply causal mask: mask[j] = -inf for j > past + i
static inline void apply_causal_mask(float* row, int dst_seqlen, int past_seqlen, int src_idx) {
    for (int j = past_seqlen + src_idx + 1; j < dst_seqlen; j++) {
        row[j] = -INFINITY;
    }
}

// ---------------------------------------------------------------------------
// kernel_sdpa
// ---------------------------------------------------------------------------

void kernel_sdpa(const OpParams& params,
                 const std::vector<const Tensor*>& inputs,
                 std::vector<Tensor*>& outputs) {
    // parse params
    int kv_cache    = graph_params::get_i32(params, 0, 2);
    int causal      = graph_params::get_i32(params, 1, 1);
    int num_heads   = graph_params::get_i32(params, 2, 16);
    int num_kv_heads= graph_params::get_i32(params, 3, 16);
    int head_dim    = graph_params::get_i32(params, 4, 192);
    int v_head_dim  = graph_params::get_i32(params, 5, 128);
    float scale     = graph_params::get_f32(params, 0, 0.f);
    if (scale == 0.f) scale = 1.f / std::sqrt((float)head_dim);

    int heads_per_group = num_heads / num_kv_heads;

    const Tensor& Q      = *inputs[0];
    const Tensor& K_cur  = *inputs[1];
    const Tensor& V_cur  = *inputs[2];
    const Tensor* mask   = (inputs.size() > 3 && inputs[3] && inputs[3]->data) ? inputs[3] : nullptr;
    const Tensor* K_cache= (inputs.size() > 4 && inputs[4] && inputs[4]->data) ? inputs[4] : nullptr;
    const Tensor* V_cache= (inputs.size() > 5 && inputs[5] && inputs[5]->data) ? inputs[5] : nullptr;

    Tensor& out       = *outputs[0];
    Tensor* K_cache_out = outputs.size() > 1 ? outputs[1] : nullptr;
    Tensor* V_cache_out = outputs.size() > 2 ? outputs[2] : nullptr;

    int src_seqlen = (int)Q.shape[1];
    int cur_seqlen = (int)K_cur.shape[1];
    int past_seqlen = 0;
    int key_capacity = 0, value_capacity = 0;

    // ---- KV cache append ----
    if (kv_cache == 2 && K_cache && K_cache->data && K_cache->shape[1] > 0) {
        past_seqlen   = (int)K_cache->shape[1];  // view height = valid length
        key_capacity  = (int)K_cache->shape[2];  // full capacity dim
        value_capacity = V_cache ? (int)V_cache->shape[2] : 0;

        size_t es = Q.element_size();
        for (int g = 0; g < num_kv_heads; g++) {
            // append K_cur
            unsigned char* kd = (unsigned char*)K_cache->channel<float>(g);
            memcpy(kd + (size_t)past_seqlen * head_dim * es,
                   K_cur.channel<unsigned char>(g), head_dim * cur_seqlen * es);
            // append V_cur
            unsigned char* vd = (unsigned char*)V_cache->channel<float>(g);
            memcpy(vd + (size_t)past_seqlen * v_head_dim * es,
                   V_cur.channel<unsigned char>(g), v_head_dim * cur_seqlen * es);
        }
    } else if (kv_cache == 1 && K_cache && K_cache->data) {
        past_seqlen = (int)K_cache->shape[1];
    }

    int dst_seqlen = past_seqlen + cur_seqlen;

    // ---- allocate scratch for QK and K/V concatenated ----
    // QK: [dst_seqlen, src_seqlen] per query head (thread)
    // We process one query head at a time, single-threaded for now

    float* qk_row = new float[dst_seqlen];

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_group;

        for (int s = 0; s < src_seqlen; s++) {
            const float* q_ptr = Q.channel<float>(h) + s * Q.stride[1] / sizeof(float);

            // ---- Q * K^T for this query row ----
            for (int j = 0; j < dst_seqlen; j++) {
                float dot = 0.f;

                if (j < past_seqlen && K_cache && K_cache->data) {
                    // from cache
                    const float* k_ptr = K_cache->channel<float>(kv_h) + j * K_cache->stride[1] / sizeof(float);
                    for (int d = 0; d < head_dim; d++) dot += q_ptr[d] * k_ptr[d];
                } else {
                    // from cur
                    int cur_idx = j - past_seqlen;
                    if (cur_idx < cur_seqlen) {
                        const float* k_ptr = K_cur.channel<float>(kv_h) + cur_idx * K_cur.stride[1] / sizeof(float);
                        for (int d = 0; d < head_dim; d++) dot += q_ptr[d] * k_ptr[d];
                    }
                }

                qk_row[j] = dot * scale;
            }

            // ---- apply mask ----
            if (mask) {
                const float* m_ptr = mask->channel<float>(0) + s * mask->stride[1] / sizeof(float);
                for (int j = 0; j < dst_seqlen; j++) qk_row[j] += m_ptr[j];
            } else if (causal) {
                apply_causal_mask(qk_row, dst_seqlen, past_seqlen, s);
            }

            // ---- softmax ----
            softmax_row(qk_row, dst_seqlen);

            // ---- attn * V ----
            float* o_ptr = out.channel<float>(h) + s * out.stride[1] / sizeof(float);
            std::memset(o_ptr, 0, v_head_dim * sizeof(float));

            for (int j = 0; j < dst_seqlen; j++) {
                float a = qk_row[j];
                if (a == 0.f) continue;

                const float* v_ptr;
                if (j < past_seqlen && V_cache && V_cache->data) {
                    v_ptr = V_cache->channel<float>(kv_h) + j * V_cache->stride[1] / sizeof(float);
                } else {
                    int cur_idx = j - past_seqlen;
                    if (cur_idx >= cur_seqlen) continue;
                    v_ptr = V_cur.channel<float>(kv_h) + cur_idx * V_cur.stride[1] / sizeof(float);
                }

                for (int d = 0; d < v_head_dim; d++) {
                    o_ptr[d] += a * v_ptr[d];
                }
            }
        }
    }

    delete[] qk_row;

    // ---- return cache views ----
    if (kv_cache == 2 && K_cache_out && K_cache) {
        *K_cache_out = *K_cache;
        K_cache_out->shape[1] = dst_seqlen;
        K_cache_out->stride[2] = K_cache_out->stride[1] * dst_seqlen;
        K_cache_out->stride[3] = K_cache_out->stride[2];
    }
    if (kv_cache == 2 && V_cache_out && V_cache) {
        *V_cache_out = *V_cache;
        V_cache_out->shape[1] = dst_seqlen;
        V_cache_out->stride[2] = V_cache_out->stride[1] * dst_seqlen;
        V_cache_out->stride[3] = V_cache_out->stride[2];
    }
}
