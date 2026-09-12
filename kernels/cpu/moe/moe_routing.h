#pragma once

#include <cstdint>
#include <vector>

namespace mollm::detail {

struct MoeRoutingParams {
    int num_experts = 0;
    int top_k = 0;
    int score_func = 0; // 0=softmax, 1=sigmoid, 2=sqrt(softplus)
    bool normalize_topk = true;
    int num_groups = 1;
    int topk_groups = 1;
    float scaling_factor = 1.0f;
};

// Select routes for a row-major [num_tokens, num_experts] logits matrix.
// Bias affects sigmoid-router selection only; returned weights always come
// from the unbiased router scores.
bool select_moe_routes(const float* logits,
                       int num_tokens,
                       const float* bias,
                       const MoeRoutingParams& params,
                       std::vector<int>& expert_indices,
                       std::vector<float>& expert_weights);

// DeepSeek-V4 early layers use a token-id lookup for expert selection while
// retaining score-derived, normalized route weights.
bool select_moe_hash_routes(const float* logits,
                            int num_tokens,
                            const int32_t* token_ids,
                            const int32_t* token_to_experts,
                            int vocab_size,
                            const MoeRoutingParams& params,
                            std::vector<int>& expert_indices,
                            std::vector<float>& expert_weights);

}  // namespace mollm::detail
