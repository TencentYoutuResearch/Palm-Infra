#include "kernels/attention.h"
#include "kernels/matmul.h"
#include "engine/engine.h"  // for CacheMetadata, cache_meta, cache_data

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cfloat>

// ---------------------------------------------------------------------------
// FlashAttention-2 FP32 kernels for ARM NEON
// Ported from ncnn-upstream/src/layer/arm/sdpa_arm_flash.h
// 2D tiled (BrxBc) with online softmax - avoids O(MxN) attention matrix
// ---------------------------------------------------------------------------

#if HAS_NEON

// Fast vectorized exp approximation for NEON
// exp(x) = 2^(x * log2e) = 2^n * 2^f, polynomial approx for 2^f
static inline float32x4_t fast_exp_f32x4(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-88.f));
    x = vminq_f32(x, vdupq_n_f32(88.f));
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    float32x4_t t = vmulq_f32(x, log2e);
    float32x4_t n = vrndmq_f32(t);
    float32x4_t f = vsubq_f32(t, n);
    int32x4_t ni = vcvtq_s32_f32(n);
    float32x4_t pow2n = vreinterpretq_f32_s32(
        vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));
    const float32x4_t c0 = vdupq_n_f32(1.0f);
    const float32x4_t c1 = vdupq_n_f32(0.6931472f);
    const float32x4_t c2 = vdupq_n_f32(0.2402265f);
    const float32x4_t c3 = vdupq_n_f32(0.0555049f);
    const float32x4_t c4 = vdupq_n_f32(0.0096813f);
    float32x4_t pow2f = vfmaq_f32(c3, c4, f);
    pow2f = vfmaq_f32(c2, pow2f, f);
    pow2f = vfmaq_f32(c1, pow2f, f);
    pow2f = vfmaq_f32(c0, pow2f, f);
    return vmulq_f32(pow2n, pow2f);
}

static inline float fast_exp_f32(float x) {
    if (x < -88.f) return 0.f;
    if (x > 88.f) return INFINITY;
    float t = x * 1.4426950408889634f;
    float n = floorf(t);
    float f = t - n;
    union { float f; int32_t i; } pow2n;
    pow2n.i = ((int32_t)n + 127) << 23;
    float pow2f = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0555049f + f * 0.0096813f)));
    return pow2n.f * pow2f;
}

// Vectorized dot product for fp32, general d
static inline float dot_fp32_neon(const float* a, const float* b, int d) {
    float32x4_t s0 = vdupq_n_f32(0.f);
    float32x4_t s1 = vdupq_n_f32(0.f);
    int k = 0;
    for (; k + 7 < d; k += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + k), vld1q_f32(b + k));
        s1 = vfmaq_f32(s1, vld1q_f32(a + k + 4), vld1q_f32(b + k + 4));
    }
    for (; k + 3 < d; k += 4)
        s0 = vfmaq_f32(s0, vld1q_f32(a + k), vld1q_f32(b + k));
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; k < d; k++) s += a[k] * b[k];
    return s;
}

// Decode kernel: M=1, Bc=32, online softmax
static void flash_attn_fp32_decode(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d_k, int d_v, float scale, const float* mask)
{
    const int Bc = 32;
    float m = -FLT_MAX;
    float l = 0.f;
    memset(O, 0, d_v * sizeof(float));

    for (int jb = 0; jb < N; jb += Bc) {
        const int bc = (jb + Bc <= N) ? Bc : (N - jb);
        float scores[32];
        for (int j = 0; j < bc; j++) {
            scores[j] = dot_fp32_neon(Q, K + (jb + j) * d_k, d_k) * scale;
            if (mask) scores[j] += mask[jb + j];
        }
        float m_block = scores[0];
        for (int j = 1; j < bc; j++) m_block = fmaxf(m_block, scores[j]);
        float m_new = fmaxf(m, m_block);
        float alpha = (m_new == m) ? 1.f : fast_exp_f32(m - m_new);
        if (l != 0.f && alpha != 1.f) {
            float32x4_t valpha = vdupq_n_f32(alpha);
            int d = 0;
            for (; d + 7 < d_v; d += 8) {
                vst1q_f32(O + d, vmulq_f32(vld1q_f32(O + d), valpha));
                vst1q_f32(O + d + 4, vmulq_f32(vld1q_f32(O + d + 4), valpha));
            }
            for (; d + 3 < d_v; d += 4)
                vst1q_f32(O + d, vmulq_f32(vld1q_f32(O + d), valpha));
            for (; d < d_v; d++) O[d] *= alpha;
        }
        l *= alpha;
        for (int j = 0; j < bc; j++) {
            float p = fast_exp_f32(scores[j] - m_new);
            l += p;
            const float* vj = V + (jb + j) * d_v;
            float32x4_t vp = vdupq_n_f32(p);
            int d = 0;
            for (; d + 7 < d_v; d += 8) {
                vst1q_f32(O + d, vfmaq_f32(vld1q_f32(O + d), vp, vld1q_f32(vj + d)));
                vst1q_f32(O + d + 4, vfmaq_f32(vld1q_f32(O + d + 4), vp, vld1q_f32(vj + d + 4)));
            }
            for (; d + 3 < d_v; d += 4)
                vst1q_f32(O + d, vfmaq_f32(vld1q_f32(O + d), vp, vld1q_f32(vj + d)));
            for (; d < d_v; d++) O[d] += p * vj[d];
        }
        m = m_new;
    }
    if (l > 0.f) {
        float inv_l = 1.f / l;
        float32x4_t vinv = vdupq_n_f32(inv_l);
        int d = 0;
        for (; d + 3 < d_v; d += 4)
            vst1q_f32(O + d, vmulq_f32(vld1q_f32(O + d), vinv));
        for (; d < d_v; d++) O[d] *= inv_l;
    }
}

// Prefill kernel: Br=4, Bc=32, 2D tiled with online softmax
static void flash_attn_fp32_prefill(
    const float* Q, const float* K, const float* V, float* O,
    int M, int N, int d_k, int d_v, float scale, const float* mask)
{
    const int Br = 4;
    const int Bc = 32;
    float* row_m = (float*)malloc(M * sizeof(float));
    float* row_l = (float*)malloc(M * sizeof(float));
    memset(O, 0, M * d_v * sizeof(float));
    for (int i = 0; i < M; i++) { row_m[i] = -FLT_MAX; row_l[i] = 0.f; }

    for (int jb = 0; jb < N; jb += Bc) {
        const int bc = (jb + Bc <= N) ? Bc : (N - jb);
        for (int ib = 0; ib < M; ib += Br) {
            const int br = (ib + Br <= M) ? Br : (M - ib);
            float S[4][32];
            for (int i = 0; i < br; i++) {
                const float* qi = Q + (ib + i) * d_k;
                for (int j = 0; j < bc; j++) {
                    S[i][j] = dot_fp32_neon(qi, K + (jb + j) * d_k, d_k) * scale;
                    if (mask) S[i][j] += mask[(ib + i) * N + jb + j];
                }
            }
            for (int i = 0; i < br; i++) {
                const int row = ib + i;
                float m_old = row_m[row];
                float l_old = row_l[row];
                float* oi = O + row * d_v;
                float m_block = S[i][0];
                for (int j = 1; j < bc; j++) m_block = fmaxf(m_block, S[i][j]);
                float m_new = fmaxf(m_old, m_block);
                float alpha = fast_exp_f32(m_old - m_new);
                {
                    float32x4_t valpha = vdupq_n_f32(alpha);
                    int d = 0;
                    for (; d + 7 < d_v; d += 8) {
                        vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), valpha));
                        vst1q_f32(oi + d + 4, vmulq_f32(vld1q_f32(oi + d + 4), valpha));
                    }
                    for (; d + 3 < d_v; d += 4)
                        vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), valpha));
                    for (; d < d_v; d++) oi[d] *= alpha;
                }
                float l_block = 0.f;
                for (int j = 0; j < bc; j++) {
                    float p = fast_exp_f32(S[i][j] - m_new);
                    l_block += p;
                    const float* vj = V + (jb + j) * d_v;
                    float32x4_t vp = vdupq_n_f32(p);
                    int d = 0;
                    for (; d + 7 < d_v; d += 8) {
                        vst1q_f32(oi + d, vfmaq_f32(vld1q_f32(oi + d), vp, vld1q_f32(vj + d)));
                        vst1q_f32(oi + d + 4, vfmaq_f32(vld1q_f32(oi + d + 4), vp, vld1q_f32(vj + d + 4)));
                    }
                    for (; d + 3 < d_v; d += 4)
                        vst1q_f32(oi + d, vfmaq_f32(vld1q_f32(oi + d), vp, vld1q_f32(vj + d)));
                    for (; d < d_v; d++) oi[d] += p * vj[d];
                }
                row_l[row] = alpha * l_old + l_block;
                row_m[row] = m_new;
            }
        }
    }
    for (int i = 0; i < M; i++) {
        float* oi = O + i * d_v;
        float inv_l = (row_l[i] > 0.f) ? 1.f / row_l[i] : 0.f;
        float32x4_t vinv = vdupq_n_f32(inv_l);
        int d = 0;
        for (; d + 3 < d_v; d += 4)
            vst1q_f32(oi + d, vmulq_f32(vld1q_f32(oi + d), vinv));
        for (; d < d_v; d++) oi[d] *= inv_l;
    }
    free(row_m);
    free(row_l);
}

#endif // HAS_NEON

// ---------------------------------------------------------------------------
// Naive scalar fallback (for non-NEON platforms)
// ---------------------------------------------------------------------------

#if !HAS_NEON

static inline void softmax_row(float* row, int len) {
    float max_val = -INFINITY;
    for (int i = 0; i < len; i++) max_val = fmaxf(max_val, row[i]);
    float sum = 0.f;
    for (int i = 0; i < len; i++) { row[i] = expf(row[i] - max_val); sum += row[i]; }
    float inv_sum = 1.f / sum;
    for (int i = 0; i < len; i++) row[i] *= inv_sum;
}

static void naive_sdpa_head(
    const float* Q, const float* K, const float* V, float* O,
    int M, int N, int d_k, int d_v, float scale, const float* mask)
{
    float* qk_row = new float[N];
    for (int s = 0; s < M; s++) {
        const float* q = Q + s * d_k;
        for (int j = 0; j < N; j++) {
            float dot = 0.f;
            const float* k = K + j * d_k;
            for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
            qk_row[j] = dot * scale;
            if (mask) qk_row[j] += mask[s * N + j];
        }
        softmax_row(qk_row, N);
        float* o = O + s * d_v;
        memset(o, 0, d_v * sizeof(float));
        for (int j = 0; j < N; j++) {
            float a = qk_row[j];
            if (a == 0.f) continue;
            const float* v = V + j * d_v;
            for (int d = 0; d < d_v; d++) o[d] += a * v[d];
        }
    }
    delete[] qk_row;
}

#endif // !HAS_NEON

// ---------------------------------------------------------------------------
// kernel_sdpa
// ---------------------------------------------------------------------------

void kernel_sdpa(const OpParams& params,
                 const std::vector<const Tensor*>& inputs,
                 std::vector<Tensor*>& outputs) {
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
    if (kv_cache == 2 && K_cache && K_cache->data) {
        const CacheMetadata* meta = cache_meta(K_cache->data);
        past_seqlen = (int)meta->current_seq_len;
        size_t es = Q.element_size();
        const void* k_cache_data = cache_data(K_cache->data);
        const void* v_cache_data = V_cache ? cache_data(V_cache->data) : nullptr;

        for (int g = 0; g < num_kv_heads; g++) {
            const unsigned char* ks = (const unsigned char*)K_cur.channel<unsigned char>(g);
            size_t k_cur_row_stride = K_cur.stride[1];
            unsigned char* kd = (unsigned char*)k_cache_data + g * (head_dim * meta->max_seq_len * es);
            for (int s = 0; s < cur_seqlen; s++) {
                std::memcpy(kd + (past_seqlen + s) * head_dim * es,
                            ks + s * k_cur_row_stride, head_dim * es);
            }
            const unsigned char* vs = (const unsigned char*)V_cur.channel<unsigned char>(g);
            size_t v_cur_row_stride = V_cur.stride[1];
            unsigned char* vd = (unsigned char*)v_cache_data + g * (v_head_dim * meta->max_seq_len * es);
            for (int s = 0; s < cur_seqlen; s++) {
                std::memcpy(vd + (past_seqlen + s) * v_head_dim * es,
                            vs + s * v_cur_row_stride, v_head_dim * es);
            }
        }
    } else if (kv_cache == 1 && K_cache && K_cache->data) {
        const CacheMetadata* meta = cache_meta(K_cache->data);
        past_seqlen = (int)meta->current_seq_len;
    }

    int dst_seqlen = past_seqlen + cur_seqlen;

    // ---- Get contiguous K/V pointers per head ----
    // After cache append, K/V are [max_seq_len, head_dim/v_head_dim] per head.
    // For kv_cache=0 (no cache), K/V come from K_cur/V_cur directly.
    auto get_k_ptr = [&](int kv_h) -> const float* {
        if (kv_cache == 2 && K_cache && K_cache->data) {
            return (const float*)cache_data(K_cache->data) +
                   kv_h * (head_dim * cache_meta(K_cache->data)->max_seq_len);
        }
        return (const float*)K_cur.channel<unsigned char>(kv_h);
    };
    auto get_v_ptr = [&](int kv_h) -> const float* {
        if (kv_cache == 2 && V_cache && V_cache->data) {
            return (const float*)cache_data(V_cache->data) +
                   kv_h * (v_head_dim * cache_meta(V_cache->data)->max_seq_len);
        }
        return (const float*)V_cur.channel<unsigned char>(kv_h);
    };

    const float* mask_ptr = mask ? (const float*)mask->channel<unsigned char>(0) : nullptr;

    // Build causal mask if needed (when no explicit mask but causal=1)
    float* causal_mask = nullptr;
    if (!mask_ptr && causal && src_seqlen > 1) {
        causal_mask = new float[src_seqlen * dst_seqlen];
        for (int i = 0; i < src_seqlen; i++) {
            for (int j = 0; j < dst_seqlen; j++) {
                causal_mask[i * dst_seqlen + j] =
                    (j <= past_seqlen + i) ? 0.f : -INFINITY;
            }
        }
        mask_ptr = causal_mask;
    }

#if HAS_NEON
    // ---- Flash attention path ----
    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_group;
        const float* Q_head = (const float*)Q.channel<unsigned char>(h);
        const float* K_head = get_k_ptr(kv_h);
        const float* V_head = get_v_ptr(kv_h);
        float* O_head = (float*)out.channel<unsigned char>(h);

        if (src_seqlen == 1) {
            flash_attn_fp32_decode(Q_head, K_head, V_head, O_head,
                                   dst_seqlen, head_dim, v_head_dim, scale, mask_ptr);
        } else {
            flash_attn_fp32_prefill(Q_head, K_head, V_head, O_head,
                                    src_seqlen, dst_seqlen, head_dim, v_head_dim,
                                    scale, mask_ptr);
        }
    }
#else
    // ---- Naive scalar fallback ----
    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / heads_per_group;
        const float* Q_head = (const float*)Q.channel<unsigned char>(h);
        const float* K_head = get_k_ptr(kv_h);
        const float* V_head = get_v_ptr(kv_h);
        float* O_head = (float*)out.channel<unsigned char>(h);
        naive_sdpa_head(Q_head, K_head, V_head, O_head,
                        src_seqlen, dst_seqlen, head_dim, v_head_dim, scale, mask_ptr);
    }
#endif

    delete[] causal_mask;

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
