#include "kernels/cpu/x86/gdn_x86.h"

#include "runtime/threading.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace {

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float softplus(float x) {
    if (x > 20.0f)
        return x;
    if (x < -20.0f)
        return std::exp(x);
    return std::log1p(std::exp(x));
}

inline float sum_squares(const float* values, int count) {
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < count; i += 16) {
        const __m512 value = _mm512_loadu_ps(values + i);
        sum = _mm512_fmadd_ps(value, value, sum);
    }
    float result = _mm512_reduce_add_ps(sum);
    for (; i < count; ++i)
        result += values[i] * values[i];
    return result;
}

inline void normalize_l2(const float* input, float* output, int count,
                         float epsilon) {
    const float inverse =
        1.0f / std::sqrt(sum_squares(input, count) + epsilon);
    const __m512 factor = _mm512_set1_ps(inverse);
    int i = 0;
    for (; i + 15 < count; i += 16) {
        _mm512_storeu_ps(
            output + i,
            _mm512_mul_ps(_mm512_loadu_ps(input + i), factor));
    }
    for (; i < count; ++i)
        output[i] = input[i] * inverse;
}

void process_heads(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* neg_exp_a, const float* dt_bias, const float* norm_weight,
    float* state, float* output, int num_heads, int k_dim, int v_dim,
    int num_v_heads, int seq_len, int data_seq_len, int a_stride,
    int b_stride, int z_stride, bool use_l2norm, float rms_epsilon,
    float l2_epsilon, float output_scale, bool sigmoid_output_gate,
    int vh_begin, int vh_end) {
    const int qkv_dim = num_heads * k_dim;
    const int z_dim = num_v_heads * v_dim;
    const int state_size = k_dim * v_dim;
    const int repeat = num_v_heads / num_heads;

    std::vector<float> q_pre(k_dim), k_pre(k_dim), value(v_dim);
    std::vector<float> q_normalized(k_dim), k_normalized(k_dim);
    std::vector<float> memory(v_dim), delta(v_dim), attention(v_dim);

    for (int vh = vh_begin; vh < vh_end; ++vh) {
        const int kh = vh / repeat;
        float* head_state =
            state + static_cast<size_t>(vh) * state_size;
        const float negative_a = neg_exp_a[vh];
        const float bias = dt_bias[vh];

        for (int token = 0; token < seq_len; ++token) {
            for (int d = 0; d < k_dim; ++d) {
                q_pre[d] =
                    qkv[(kh * k_dim + d) * data_seq_len + token];
                k_pre[d] =
                    qkv[(qkv_dim + kh * k_dim + d) * data_seq_len + token];
            }
            for (int d = 0; d < v_dim; ++d) {
                value[d] =
                    qkv[(2 * qkv_dim + vh * v_dim + d) * data_seq_len +
                        token];
            }

            const float decay = std::exp(
                negative_a *
                softplus(a[token * a_stride + vh] + bias));
            const float beta = sigmoid(b[token * b_stride + vh]);

            const float* query = q_pre.data();
            const float* key = k_pre.data();
            if (use_l2norm) {
                normalize_l2(q_pre.data(), q_normalized.data(), k_dim,
                             l2_epsilon);
                normalize_l2(k_pre.data(), k_normalized.data(), k_dim,
                             l2_epsilon);
                query = q_normalized.data();
                key = k_normalized.data();
            }

            const __m512 decay_vector = _mm512_set1_ps(decay);
            int i = 0;
            for (; i + 15 < state_size; i += 16) {
                _mm512_storeu_ps(
                    head_state + i,
                    _mm512_mul_ps(_mm512_loadu_ps(head_state + i),
                                  decay_vector));
            }
            for (; i < state_size; ++i)
                head_state[i] *= decay;

            std::fill(memory.begin(), memory.end(), 0.0f);
            for (int dk = 0; dk < k_dim; ++dk) {
                const __m512 key_vector = _mm512_set1_ps(key[dk]);
                const float* state_row =
                    head_state + static_cast<size_t>(dk) * v_dim;
                int dv = 0;
                for (; dv + 15 < v_dim; dv += 16) {
                    _mm512_storeu_ps(
                        memory.data() + dv,
                        _mm512_fmadd_ps(
                            _mm512_loadu_ps(state_row + dv), key_vector,
                            _mm512_loadu_ps(memory.data() + dv)));
                }
                for (; dv < v_dim; ++dv)
                    memory[dv] += state_row[dv] * key[dk];
            }

            const __m512 beta_vector = _mm512_set1_ps(beta);
            int dv = 0;
            for (; dv + 15 < v_dim; dv += 16) {
                _mm512_storeu_ps(
                    delta.data() + dv,
                    _mm512_mul_ps(
                        _mm512_sub_ps(_mm512_loadu_ps(value.data() + dv),
                                      _mm512_loadu_ps(memory.data() + dv)),
                        beta_vector));
            }
            for (; dv < v_dim; ++dv)
                delta[dv] = (value[dv] - memory[dv]) * beta;

            for (int dk = 0; dk < k_dim; ++dk) {
                const __m512 key_vector = _mm512_set1_ps(key[dk]);
                float* state_row =
                    head_state + static_cast<size_t>(dk) * v_dim;
                int d = 0;
                for (; d + 15 < v_dim; d += 16) {
                    _mm512_storeu_ps(
                        state_row + d,
                        _mm512_fmadd_ps(
                            key_vector, _mm512_loadu_ps(delta.data() + d),
                            _mm512_loadu_ps(state_row + d)));
                }
                for (; d < v_dim; ++d)
                    state_row[d] += key[dk] * delta[d];
            }

            std::fill(attention.begin(), attention.end(), 0.0f);
            for (int dk = 0; dk < k_dim; ++dk) {
                const __m512 query_vector = _mm512_set1_ps(query[dk]);
                const float* state_row =
                    head_state + static_cast<size_t>(dk) * v_dim;
                int d = 0;
                for (; d + 15 < v_dim; d += 16) {
                    _mm512_storeu_ps(
                        attention.data() + d,
                        _mm512_fmadd_ps(
                            _mm512_loadu_ps(state_row + d), query_vector,
                            _mm512_loadu_ps(attention.data() + d)));
                }
                for (; d < v_dim; ++d)
                    attention[d] += state_row[d] * query[dk];
            }

            const float rms = 1.0f /
                std::sqrt(sum_squares(attention.data(), v_dim) *
                              (output_scale * output_scale /
                               static_cast<float>(v_dim)) +
                          rms_epsilon);
            const __m512 norm_factor =
                _mm512_set1_ps(rms * output_scale);
            const float* z_row =
                z + static_cast<size_t>(token) * z_stride + vh * v_dim;
            int d = 0;
            for (; d + 15 < v_dim; d += 16) {
                _mm512_storeu_ps(
                    attention.data() + d,
                    _mm512_mul_ps(
                        _mm512_mul_ps(_mm512_loadu_ps(attention.data() + d),
                                      norm_factor),
                        _mm512_loadu_ps(norm_weight + d)));
            }
            for (; d < v_dim; ++d)
                attention[d] *= rms * output_scale * norm_weight[d];
            for (d = 0; d < v_dim; ++d) {
                const float sigmoid_z = sigmoid(z_row[d]);
                const float gate = sigmoid_output_gate
                    ? sigmoid_z : z_row[d] * sigmoid_z;
                output[vh * v_dim + d + token * z_dim] =
                    attention[d] * gate;
            }
        }
    }
}

}  // namespace

void kernel_gdn_x86_avx512(const OpParams& params,
                           const std::vector<const Tensor*>& inputs,
                           std::vector<Tensor*>& outputs,
                           ThreadPool* thread_pool) {
    if (inputs.size() < 8 || outputs.empty())
        return;
    const int num_heads = graph_params::get_i32(params, 0, 16);
    const int k_dim = graph_params::get_i32(params, 1, 128);
    const int v_dim = graph_params::get_i32(params, 2, 128);
    const int seq_len = graph_params::get_i32(params, 3, 4);
    const int flags = graph_params::get_i32(params, 4, 1);
    const bool use_l2norm = (flags & 1) != 0;
    const int n_real = graph_params::get_i32(params, 6, seq_len);
    const int num_v_heads =
        graph_params::get_i32(params, 7, num_heads);
    const bool sigmoid_output_gate = (flags & 2) != 0;
    const float rms_epsilon =
        graph_params::get_f32(params, 0, 1e-6f);
    const float l2_epsilon =
        graph_params::get_f32(params, 1, 1e-6f);
    float scale = graph_params::get_f32(params, 2, 0.0f);
    if (scale == 0.0f)
        scale = 1.0f / std::sqrt(static_cast<float>(k_dim));

    std::vector<float> neg_exp_a(num_v_heads);
    for (int head = 0; head < num_v_heads; ++head)
        neg_exp_a[head] = -std::exp(inputs[4]->ptr<float>()[head]);

    const int z_dim = num_v_heads * v_dim;
    const int process_len =
        n_real > 0 && n_real < seq_len ? n_real : seq_len;
    float* output = outputs[0]->ptr<float>();
    if (process_len < seq_len) {
        std::memset(output + static_cast<size_t>(process_len) * z_dim, 0,
                    static_cast<size_t>(seq_len - process_len) * z_dim *
                        sizeof(float));
    }

    auto run = [&](int, int begin, int end) {
        process_heads(
            inputs[0]->ptr<float>(), inputs[1]->ptr<float>(),
            inputs[2]->ptr<float>(), inputs[3]->ptr<float>(),
            neg_exp_a.data(), inputs[5]->ptr<float>(),
            inputs[6]->ptr<float>(),
            reinterpret_cast<float*>(inputs[7]->data), output, num_heads,
            k_dim, v_dim, num_v_heads, process_len, seq_len,
            static_cast<int>(inputs[1]->stride[1] / sizeof(float)),
            static_cast<int>(inputs[2]->stride[1] / sizeof(float)),
            static_cast<int>(inputs[3]->stride[1] / sizeof(float)),
            use_l2norm, rms_epsilon, l2_epsilon, scale,
            sigmoid_output_gate, begin, end);
    };
    if (thread_pool && num_v_heads >= 2)
        thread_pool->parallel_for(0, num_v_heads, 1, run);
    else
        run(0, 0, num_v_heads);
}
