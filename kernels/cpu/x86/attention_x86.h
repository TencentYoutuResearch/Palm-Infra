#pragma once

void sdpa_x86_avx512_head(const float* query, const float* key,
                          const float* value, float* output, int query_tokens,
                          int key_tokens, int key_dim, int value_dim,
                          float scale, const float* mask);
