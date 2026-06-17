#include "kernels/attention.h"
#include "kernels/matmul.h"
#include "engine/engine.h"  // for CacheMetadata, cache_meta, cache_data

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

    // ---- KV cache append ----
    // KV cache buffer layout: [CacheMetadata(64B)] [data]
    // Read past_seqlen from cache metadata header
    if (kv_cache == 2 && K_cache && K_cache->data) {
        const CacheMetadata* meta = cache_meta(K_cache->data);
        past_seqlen = (int)meta->current_seq_len;

        size_t es = Q.element_size();
        // K data starts after metadata header
        const void* k_cache_data = cache_data(K_cache->data);
        const void* v_cache_data = V_cache ? cache_data(V_cache->data) : nullptr;

        for (int g = 0; g < num_kv_heads; g++) {
            // K_cur: shape=[head_dim, seq_len, num_kv_heads]
            const unsigned char* ks = (const unsigned char*)K_cur.channel<unsigned char>(g);
            size_t k_cur_row_stride = K_cur.stride[1];
            // K_cache data: shape=[head_dim, n_ctx, num_kv_heads]
            unsigned char* kd = (unsigned char*)k_cache_data + g * (head_dim * meta->max_seq_len * es);
            for (int s = 0; s < cur_seqlen; s++) {
                std::memcpy(kd + (past_seqlen + s) * head_dim * es,
                            ks + s * k_cur_row_stride,
                            head_dim * es);
            }

            // V_cur
            const unsigned char* vs = (const unsigned char*)V_cur.channel<unsigned char>(g);
            size_t v_cur_row_stride = V_cur.stride[1];
            unsigned char* vd = (unsigned char*)v_cache_data + g * (v_head_dim * meta->max_seq_len * es);
            for (int s = 0; s < cur_seqlen; s++) {
                std::memcpy(vd + (past_seqlen + s) * v_head_dim * es,
                            vs + s * v_cur_row_stride,
                            v_head_dim * es);
            }
        }
    } else if (kv_cache == 1 && K_cache && K_cache->data) {
        const CacheMetadata* meta = cache_meta(K_cache->data);
        past_seqlen = (int)meta->current_seq_len;
    }

    int dst_seqlen = past_seqlen + cur_seqlen;

    // ---- allocate scratch for QK and K/V concatenated ----
    float* qk_row = new float[dst_seqlen];

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_group;

        for (int s = 0; s < src_seqlen; s++) {
            const float* q_ptr = Q.channel<float>(h) + s * Q.stride[1] / sizeof(float);

            // ---- Q * K^T for this query row ----
            for (int j = 0; j < dst_seqlen; j++) {
                float dot = 0.f;

                if (j < past_seqlen && K_cache && K_cache->data) {
                    // from cache (data starts after metadata header)
                    const void* kd = cache_data(K_cache->data);
                    const float* k_ptr = (const float*)((const char*)kd + kv_h * (head_dim * cache_meta(K_cache->data)->max_seq_len * sizeof(float)));
                    k_ptr += j * head_dim;
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
                    const void* vd = cache_data(V_cache->data);
                    v_ptr = (const float*)((const char*)vd + kv_h * (v_head_dim * cache_meta(V_cache->data)->max_seq_len * sizeof(float)));
                    v_ptr += j * v_head_dim;
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
