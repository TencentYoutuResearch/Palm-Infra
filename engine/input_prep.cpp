#include "engine/input_prep.h"

#include <cmath>

namespace mollm {
namespace detail {

void fill_causal_mask(float* mask, int seq_len, int past_len) {
    const int total = past_len + seq_len;
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < total; ++j) {
            mask[i * total + j] = (j > past_len + i) ? -1e38f : 0.f;
        }
    }
}

void fill_rope_cache(float* cos_cache, float* sin_cache, int seq_len,
                     int start_pos, int rope_dim, float rope_theta) {
    const int half = rope_dim / 2;
    for (int n = 0; n < seq_len; ++n) {
        const int pos = start_pos + n;
        for (int i = 0; i < half; ++i) {
            const float theta =
                1.0f / std::pow(rope_theta, 2.0f * i / rope_dim);
            const float angle = pos * theta;
            cos_cache[n * half + i] = std::cos(angle);
            sin_cache[n * half + i] = std::sin(angle);
        }
    }
}

}  // namespace detail
}  // namespace mollm
