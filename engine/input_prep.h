#pragma once

namespace mollm {
namespace detail {

void fill_causal_mask(float* mask, int seq_len, int past_len);

void fill_rope_cache(float* cos_cache, float* sin_cache, int seq_len,
                     int start_pos, int rope_dim, float rope_theta);

}  // namespace detail
}  // namespace mollm
