#include "kernels/rwkv.h"
#include "kernels/threading.h"

#include <cmath>
#include <cstring>
#include <vector>

#if HAS_NEON
#include <arm_neon.h>

static inline float rwkv_dot_neon(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.f);
    for (int i = 0; i < n; i += 4)
        sum = vfmaq_f32(sum, vld1q_f32(a + i), vld1q_f32(b + i));
    return vaddvq_f32(sum);
}

static inline void rwkv_load_state_fp16(const __fp16* src, float* dst, int n) {
    for (int i = 0; i < n; i += 4)
        vst1q_f32(dst + i, vcvt_f32_f16(vld1_f16(src + i)));
}

static inline void rwkv_store_state_fp16(const float* src, __fp16* dst, int n) {
    for (int i = 0; i < n; i += 4)
        vst1_f16(dst + i, vcvt_f16_f32(vld1q_f32(src + i)));
}
#endif

void kernel_rwkv_token_shift(const OpParams& p,
                             const std::vector<const Tensor*>& in, Tensor& out) {
    if (in.size() < 2) return;
    const int hidden = graph_params::get_i32(p, 0, 0);
    const int seq = graph_params::get_i32(p, 1, 1);
    int real = graph_params::get_i32(p, 2, seq);
    if (real <= 0 || real > seq) real = seq;
    const float* x = in[0]->ptr<float>();
    const bool state_fp16 = in[1]->prec == Precision::FP16;
    __fp16* state16 = state_fp16 ? reinterpret_cast<__fp16*>(in[1]->data) : nullptr;
    float* state32 = state_fp16 ? nullptr : reinterpret_cast<float*>(in[1]->data);
    float* y = out.ptr<float>();
    for (int t = 0; t < real; ++t) {
        const float* row = x + (size_t)t * hidden;
        for (int d = 0; d < hidden; ++d)
            y[(size_t)t * hidden + d] = (state_fp16 ? (float)state16[d] : state32[d]) - row[d];
        if (state_fp16) {
            for (int d = 0; d < hidden; ++d) state16[d] = (__fp16)row[d];
        } else {
            std::memcpy(state32, row, (size_t)hidden * sizeof(float));
        }
    }
    if (real < seq)
        std::memset(y + (size_t)real * hidden, 0,
                    (size_t)(seq - real) * hidden * sizeof(float));
}

void kernel_rwkv_mix(const OpParams&, const std::vector<const Tensor*>& in, Tensor& out) {
    if (in.size() < 3) return;
    const float* x = in[0]->ptr<float>();
    const float* shift = in[1]->ptr<float>();
    const float* mix = in[2]->ptr<float>();
    float* dst = out.ptr<float>();
    const int hidden = (int)in[2]->nelements();
    const int64_t total = in[0]->nelements();
    if (hidden <= 0 || total % hidden != 0) return;
    const int rows = (int)(total / hidden);
    for (int row = 0; row < rows; ++row) {
        const size_t base = (size_t)row * hidden;
        int d = 0;
#if HAS_NEON
        for (; d + 7 < hidden; d += 8) {
            vst1q_f32(dst + base + d, vfmaq_f32(vld1q_f32(x + base + d),
                                                 vld1q_f32(shift + base + d),
                                                 vld1q_f32(mix + d)));
            vst1q_f32(dst + base + d + 4, vfmaq_f32(vld1q_f32(x + base + d + 4),
                                                     vld1q_f32(shift + base + d + 4),
                                                     vld1q_f32(mix + d + 4)));
        }
#endif
        for (; d < hidden; ++d) dst[base + d] = x[base + d] + shift[base + d] * mix[d];
    }
}

void kernel_rwkv_l2_norm(const OpParams& p,
                         const std::vector<const Tensor*>& in, Tensor& out) {
    if (in.empty()) return;
    const int heads = graph_params::get_i32(p, 0, 0);
    const int dhead = graph_params::get_i32(p, 1, 0);
    const float eps = graph_params::get_f32(p, 0, 1e-12f);
    const int hidden = heads * dhead;
    const int tokens = (int)(in[0]->nelements() / hidden);
    const float* src = in[0]->ptr<float>();
    float* dst = out.ptr<float>();
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const size_t base = (size_t)t * hidden + (size_t)h * dhead;
            float sum = 0.f;
#if HAS_NEON
            if ((dhead & 3) == 0) {
                float32x4_t v_sum = vdupq_n_f32(0.f);
                for (int j = 0; j < dhead; j += 4) {
                    const float32x4_t x = vld1q_f32(src + base + j);
                    v_sum = vfmaq_f32(v_sum, x, x);
                }
                sum = vaddvq_f32(v_sum);
            } else
#endif
            {
                for (int j = 0; j < dhead; ++j) sum += src[base + j] * src[base + j];
            }
            const float inv = 1.f / (std::sqrt(sum) + eps);
            int j = 0;
#if HAS_NEON
            for (; j + 3 < dhead; j += 4)
                vst1q_f32(dst + base + j, vmulq_n_f32(vld1q_f32(src + base + j), inv));
#endif
            for (; j < dhead; ++j) dst[base + j] = src[base + j] * inv;
        }
    }
}

void kernel_rwkv_post(const OpParams& p,
                      const std::vector<const Tensor*>& in, Tensor& out,
                      ThreadPool* thread_pool) {
    if (in.size() < 8) return;
    const int heads = graph_params::get_i32(p, 0, 0);
    const int dhead = graph_params::get_i32(p, 1, 0);
    const float eps = graph_params::get_f32(p, 0, 64e-5f);
    const int hidden = heads * dhead;
    if (hidden <= 0) return;
    const int tokens = (int)(in[0]->nelements() / hidden);
    const float* raw = in[0]->ptr<float>();
    const float* r = in[1]->ptr<float>();
    const float* k = in[2]->ptr<float>();
    const float* v = in[3]->ptr<float>();
    const float* rk = in[4]->ptr<float>();
    const float* weight = in[5]->ptr<float>();
    const float* bias = in[6]->ptr<float>();
    const float* gate = in[7]->ptr<float>();
    float* dst = out.ptr<float>();
    const auto process = [&](int, int begin, int end) {
        for (int group = begin; group < end; ++group) {
            const int h = group % heads;
            const size_t base = (size_t)group * dhead;
            const size_t wb = (size_t)h * dhead;
            float mean = 0.f, var = 0.f, bonus = 0.f;
#if HAS_NEON
            if ((dhead & 3) == 0) {
                float32x4_t sum = vdupq_n_f32(0.f), bonus_sum = sum;
                for (int j = 0; j < dhead; j += 4) {
                    sum = vaddq_f32(sum, vld1q_f32(raw + base + j));
                    const float32x4_t q = vmulq_f32(vld1q_f32(r + base + j),
                                                    vld1q_f32(k + base + j));
                    bonus_sum = vfmaq_f32(bonus_sum, q, vld1q_f32(rk + wb + j));
                }
                mean = vaddvq_f32(sum) / dhead;
                bonus = vaddvq_f32(bonus_sum);
                sum = vdupq_n_f32(0.f);
                const float32x4_t mean_v = vdupq_n_f32(mean);
                for (int j = 0; j < dhead; j += 4) {
                    const float32x4_t q = vsubq_f32(vld1q_f32(raw + base + j), mean_v);
                    sum = vfmaq_f32(sum, q, q);
                }
                var = vaddvq_f32(sum) / dhead;
                const float32x4_t inv = vdupq_n_f32(1.f / std::sqrt(var + eps));
                const float32x4_t bonus_v = vdupq_n_f32(bonus);
                for (int j = 0; j < dhead; j += 4) {
                    float32x4_t q = vmulq_f32(vsubq_f32(vld1q_f32(raw + base + j), mean_v), inv);
                    q = vfmaq_f32(vld1q_f32(bias + wb + j), q, vld1q_f32(weight + wb + j));
                    q = vaddq_f32(q, vmulq_f32(bonus_v, vld1q_f32(v + base + j)));
                    q = vmulq_f32(q, vld1q_f32(gate + base + j));
                    vst1q_f32(dst + base + j, q);
                }
                continue;
            }
#endif
            for (int j = 0; j < dhead; ++j) {
                mean += raw[base + j];
                bonus += r[base + j] * k[base + j] * rk[wb + j];
            }
            mean /= dhead;
            for (int j = 0; j < dhead; ++j) {
                const float q = raw[base + j] - mean;
                var += q * q;
            }
            const float inv = 1.f / std::sqrt(var / dhead + eps);
            for (int j = 0; j < dhead; ++j)
                dst[base + j] = ((raw[base + j] - mean) * inv * weight[wb + j] + bias[wb + j] +
                                 bonus * v[base + j]) * gate[base + j];
        }
    };
    const int groups = tokens * heads;
    if (thread_pool && tokens > 1 && groups >= 4)
        thread_pool->parallel_for(0, groups, 1, process);
    else
        process(0, 0, groups);
}

void kernel_rwkv7(const OpParams& p, const std::vector<const Tensor*>& in,
                  Tensor& out, ThreadPool* thread_pool) {
    if (in.size() != 7) return;
    const int heads = graph_params::get_i32(p, 0, 0);
    const int dhead = graph_params::get_i32(p, 1, 0);
    const int seq = graph_params::get_i32(p, 2, 1);
    int real = graph_params::get_i32(p, 3, seq);
    if (real <= 0 || real > seq) real = seq;
    const int hidden = heads * dhead;
    const float* r = in[0]->ptr<float>();
    const float* decay = in[1]->ptr<float>();
    const float* k = in[2]->ptr<float>();
    const float* v = in[3]->ptr<float>();
    const float* a = in[4]->ptr<float>();
    const float* b = in[5]->ptr<float>();
    const bool state_fp16 = in[6]->prec == Precision::FP16;
    __fp16* state16 = state_fp16 ? reinterpret_cast<__fp16*>(in[6]->data) : nullptr;
    float* state32 = state_fp16 ? nullptr : reinterpret_cast<float*>(in[6]->data);
    float* dst = out.ptr<float>();
    const auto process_heads = [&](int, int h_begin, int h_end) {
        std::vector<float> statef;
        if (state_fp16 || real != 1) statef.resize((size_t)dhead * dhead);
        for (int h = h_begin; h < h_end; ++h) {
            const size_t state_base = (size_t)h * dhead * dhead;
            const bool direct_state = !state_fp16 && real == 1;
            float* state = direct_state ? state32 + state_base : statef.data();
            if (!state_fp16 && !direct_state)
                std::memcpy(state, state32 + state_base, (size_t)dhead * dhead * sizeof(float));
            for (int t = 0; t < real; ++t) {
                const size_t base = (size_t)t * hidden + (size_t)h * dhead;
                if (state_fp16) {
#if HAS_NEON
                    if ((dhead & 3) == 0)
                        rwkv_load_state_fp16(state16 + state_base, state, dhead * dhead);
                    else
#endif
                        for (int q = 0; q < dhead * dhead; ++q) state[q] = (float)state16[state_base + q];
                }
                int row = 0;
#if HAS_NEON
                if ((dhead & 3) == 0) {
                    for (; row + 3 < dhead; row += 4) {
                        float32x4_t state_a0 = vdupq_n_f32(0.f), state_a1 = state_a0;
                        float32x4_t state_a2 = state_a0, state_a3 = state_a0;
                        for (int j = 0; j < dhead; j += 4) {
                            const float32x4_t a_v = vld1q_f32(a + base + j);
                            state_a0 = vfmaq_f32(state_a0, vld1q_f32(state + (size_t)(row + 0) * dhead + j), a_v);
                            state_a1 = vfmaq_f32(state_a1, vld1q_f32(state + (size_t)(row + 1) * dhead + j), a_v);
                            state_a2 = vfmaq_f32(state_a2, vld1q_f32(state + (size_t)(row + 2) * dhead + j), a_v);
                            state_a3 = vfmaq_f32(state_a3, vld1q_f32(state + (size_t)(row + 3) * dhead + j), a_v);
                        }
                        const float state_a[4] = {vaddvq_f32(state_a0), vaddvq_f32(state_a1), vaddvq_f32(state_a2), vaddvq_f32(state_a3)};
                        float32x4_t result0 = vdupq_n_f32(0.f), result1 = result0, result2 = result0, result3 = result0;
                        const float32x4_t value0 = vdupq_n_f32(v[base + row + 0]);
                        const float32x4_t value1 = vdupq_n_f32(v[base + row + 1]);
                        const float32x4_t value2 = vdupq_n_f32(v[base + row + 2]);
                        const float32x4_t value3 = vdupq_n_f32(v[base + row + 3]);
                        const float32x4_t state_a_v0 = vdupq_n_f32(state_a[0]);
                        const float32x4_t state_a_v1 = vdupq_n_f32(state_a[1]);
                        const float32x4_t state_a_v2 = vdupq_n_f32(state_a[2]);
                        const float32x4_t state_a_v3 = vdupq_n_f32(state_a[3]);
                        for (int j = 0; j < dhead; j += 4) {
                            const float32x4_t decay_v = vld1q_f32(decay + base + j);
                            const float32x4_t key_v = vld1q_f32(k + base + j);
                            const float32x4_t b_v = vld1q_f32(b + base + j);
                            const float32x4_t r_v = vld1q_f32(r + base + j);
#define RWKV_ROW_STEP(N) \
                            float32x4_t x##N = vmulq_f32(vld1q_f32(state + (size_t)(row + N) * dhead + j), decay_v); \
                            x##N = vfmaq_f32(x##N, value##N, key_v); \
                            x##N = vfmaq_f32(x##N, state_a_v##N, b_v); \
                            vst1q_f32(state + (size_t)(row + N) * dhead + j, x##N); \
                            result##N = vfmaq_f32(result##N, x##N, r_v)
                            RWKV_ROW_STEP(0); RWKV_ROW_STEP(1);
                            RWKV_ROW_STEP(2); RWKV_ROW_STEP(3);
#undef RWKV_ROW_STEP
                        }
                        dst[base + row + 0] = vaddvq_f32(result0);
                        dst[base + row + 1] = vaddvq_f32(result1);
                        dst[base + row + 2] = vaddvq_f32(result2);
                        dst[base + row + 3] = vaddvq_f32(result3);
                    }
                }
#endif
                for (int i = row; i < dhead; ++i) {
                    float state_a = 0.f, result = 0.f;
#if HAS_NEON
                    if ((dhead & 3) == 0)
                        state_a = rwkv_dot_neon(state + (size_t)i * dhead, a + base, dhead);
                    else
#endif
                        for (int j = 0; j < dhead; ++j) state_a += state[(size_t)i * dhead + j] * a[base + j];
                    int j = 0;
#if HAS_NEON
                    float32x4_t result_v = vdupq_n_f32(0.f);
                    const float32x4_t value_v = vdupq_n_f32(v[base + i]);
                    const float32x4_t state_a_v = vdupq_n_f32(state_a);
                    for (; j + 3 < dhead; j += 4) {
                        float32x4_t x = vmulq_f32(vld1q_f32(state + (size_t)i * dhead + j), vld1q_f32(decay + base + j));
                        x = vfmaq_f32(x, value_v, vld1q_f32(k + base + j));
                        x = vfmaq_f32(x, state_a_v, vld1q_f32(b + base + j));
                        vst1q_f32(state + (size_t)i * dhead + j, x);
                        result_v = vfmaq_f32(result_v, x, vld1q_f32(r + base + j));
                    }
                    result = vaddvq_f32(result_v);
#endif
                    for (; j < dhead; ++j) {
                        float& x = state[(size_t)i * dhead + j];
                        x = x * decay[base + j] + v[base + i] * k[base + j] + state_a * b[base + j];
                        result += x * r[base + j];
                    }
                    dst[base + i] = result;
                }
                if (state_fp16) {
#if HAS_NEON
                    if ((dhead & 3) == 0)
                        rwkv_store_state_fp16(state, state16 + state_base, dhead * dhead);
                    else
#endif
                        for (int q = 0; q < dhead * dhead; ++q) state16[state_base + q] = (__fp16)state[q];
                }
            }
            if (!state_fp16 && !direct_state)
                std::memcpy(state32 + state_base, state, (size_t)dhead * dhead * sizeof(float));
        }
    };
    if (thread_pool && heads >= 4)
        thread_pool->parallel_for(0, heads, 1, process_heads);
    else
        process_heads(0, 0, heads);
    if (real < seq)
        std::memset(dst + (size_t)real * hidden, 0,
                    (size_t)(seq - real) * hidden * sizeof(float));
}
