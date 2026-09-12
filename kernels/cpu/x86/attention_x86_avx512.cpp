#include "kernels/cpu/x86/attention_x86.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <vector>

namespace {

inline float dot_avx512(const float* lhs, const float* rhs, int count) {
    __m512 first = _mm512_setzero_ps();
    __m512 second = _mm512_setzero_ps();
    int i = 0;
    for (; i + 31 < count; i += 32) {
        first = _mm512_fmadd_ps(_mm512_loadu_ps(lhs + i),
                                _mm512_loadu_ps(rhs + i), first);
        second = _mm512_fmadd_ps(_mm512_loadu_ps(lhs + i + 16),
                                 _mm512_loadu_ps(rhs + i + 16), second);
    }
    for (; i + 15 < count; i += 16) {
        first = _mm512_fmadd_ps(_mm512_loadu_ps(lhs + i),
                                _mm512_loadu_ps(rhs + i), first);
    }
    float result = _mm512_reduce_add_ps(_mm512_add_ps(first, second));
    for (; i < count; ++i)
        result += lhs[i] * rhs[i];
    return result;
}

inline void softmax(float* values, int count) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < count; ++i)
        maximum = std::max(maximum, values[i]);
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        values[i] = std::exp(values[i] - maximum);
        sum += values[i];
    }
    const float inverse = 1.0f / sum;
    for (int i = 0; i < count; ++i)
        values[i] *= inverse;
}

}  // namespace

void sdpa_x86_avx512_head(const float* query, const float* key,
                          const float* value, float* output, int query_tokens,
                          int key_tokens, int key_dim, int value_dim,
                          float scale, const float* mask) {
    std::vector<float> scores(key_tokens);
    for (int token = 0; token < query_tokens; ++token) {
        const float* query_row =
            query + static_cast<size_t>(token) * key_dim;
        const float* mask_row =
            mask ? mask + static_cast<size_t>(token) * key_tokens : nullptr;
        int visible_keys = key_tokens;
        // Causal and padding masks used by the graph have a trailing -inf
        // region. Avoid both QK and PV work for those guaranteed-zero terms.
        if (mask_row) {
            while (visible_keys > 0 &&
                   mask_row[visible_keys - 1] ==
                       -std::numeric_limits<float>::infinity()) {
                --visible_keys;
            }
        }
        float* output_row =
            output + static_cast<size_t>(token) * value_dim;
        if (visible_keys == 0) {
            std::memset(output_row, 0,
                        static_cast<size_t>(value_dim) * sizeof(float));
            continue;
        }
        for (int key_token = 0; key_token < visible_keys; ++key_token) {
            scores[key_token] =
                dot_avx512(
                    query_row,
                    key + static_cast<size_t>(key_token) * key_dim,
                    key_dim) *
                scale;
            if (mask_row)
                scores[key_token] += mask_row[key_token];
        }
        softmax(scores.data(), visible_keys);

        std::memset(output_row, 0,
                    static_cast<size_t>(value_dim) * sizeof(float));
        for (int key_token = 0; key_token < visible_keys; ++key_token) {
            const float probability = scores[key_token];
            if (probability == 0.0f)
                continue;
            const __m512 multiplier =
                _mm512_set1_ps(probability);
            const float* value_row =
                value + static_cast<size_t>(key_token) * value_dim;
            int d = 0;
            for (; d + 15 < value_dim; d += 16) {
                _mm512_storeu_ps(
                    output_row + d,
                    _mm512_fmadd_ps(_mm512_loadu_ps(value_row + d),
                                    multiplier,
                                    _mm512_loadu_ps(output_row + d)));
            }
            for (; d < value_dim; ++d)
                output_row[d] += probability * value_row[d];
        }
    }
}
