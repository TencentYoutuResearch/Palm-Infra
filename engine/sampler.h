#pragma once

// Samples a token ID from model logits. A non-positive temperature or top_k ==
// 1 selects greedy decoding; otherwise top-k/top-p sampling is applied.
int sample_token(float* logits, int vocab_size, float temperature, int top_k,
                 float top_p, unsigned int* seed);
