#include "kernels/moe_routing.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mollm::detail {
namespace {

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

}  // namespace

bool select_moe_routes(const float* logits,
                       int num_tokens,
                       const float* bias,
                       const MoeRoutingParams& params,
                       std::vector<int>& expert_indices,
                       std::vector<float>& expert_weights) {
    const int num_experts = params.num_experts;
    const int top_k = params.top_k;
    if (!logits || num_tokens <= 0 || num_experts <= 0 ||
        top_k <= 0 || top_k > num_experts ||
        (params.score_func != 0 && params.score_func != 1)) {
        return false;
    }

    int num_groups = std::max(1, params.num_groups);
    int topk_groups = params.topk_groups;
    if (topk_groups <= 0 || topk_groups > num_groups)
        topk_groups = num_groups;
    const int experts_per_group = num_experts / num_groups;

    expert_indices.resize(static_cast<size_t>(num_tokens) * top_k);
    expert_weights.resize(static_cast<size_t>(num_tokens) * top_k);
    std::vector<float> scores(static_cast<size_t>(num_experts));
    std::vector<float> choice_scores(static_cast<size_t>(num_experts));
    std::vector<float> group_scores(static_cast<size_t>(num_groups));
    std::vector<int> group_indices(static_cast<size_t>(topk_groups));
    std::vector<float> group_top(static_cast<size_t>(topk_groups));
    std::vector<unsigned char> keep_group(static_cast<size_t>(num_groups));

    for (int token = 0; token < num_tokens; ++token) {
        const float* token_logits =
            logits + static_cast<size_t>(token) * num_experts;
        int* indices =
            expert_indices.data() + static_cast<size_t>(token) * top_k;
        float* weights =
            expert_weights.data() + static_cast<size_t>(token) * top_k;
        std::fill(indices, indices + top_k, 0);
        std::fill(weights, weights + top_k,
                  -std::numeric_limits<float>::infinity());

        const float* weight_source = token_logits;
        const float* selection_source = token_logits;
        if (params.score_func == 1) {
            for (int expert = 0; expert < num_experts; ++expert) {
                scores[expert] = sigmoid(token_logits[expert]);
                choice_scores[expert] =
                    scores[expert] + (bias ? bias[expert] : 0.0f);
            }

            if (num_groups > 1 && experts_per_group > 0 &&
                topk_groups < num_groups) {
                std::fill(group_scores.begin(), group_scores.end(),
                          -std::numeric_limits<float>::infinity());
                std::fill(group_indices.begin(), group_indices.end(), 0);
                std::fill(group_top.begin(), group_top.end(),
                          -std::numeric_limits<float>::infinity());
                for (int group = 0; group < num_groups; ++group) {
                    float best0 = -std::numeric_limits<float>::infinity();
                    float best1 = best0;
                    const int begin = group * experts_per_group;
                    const int end = group == num_groups - 1
                        ? num_experts : begin + experts_per_group;
                    for (int expert = begin; expert < end; ++expert) {
                        const float value = choice_scores[expert];
                        if (value > best0) {
                            best1 = best0;
                            best0 = value;
                        } else if (value > best1) {
                            best1 = value;
                        }
                    }
                    group_scores[group] =
                        std::isfinite(best1) ? best0 + best1 : best0;
                    for (int k = 0; k < topk_groups; ++k) {
                        if (group_scores[group] > group_top[k]) {
                            for (int shift = topk_groups - 1; shift > k; --shift) {
                                group_top[shift] = group_top[shift - 1];
                                group_indices[shift] = group_indices[shift - 1];
                            }
                            group_top[k] = group_scores[group];
                            group_indices[k] = group;
                            break;
                        }
                    }
                }
                std::fill(keep_group.begin(), keep_group.end(), 0);
                for (int k = 0; k < topk_groups; ++k)
                    keep_group[group_indices[k]] = 1;
                for (int expert = 0; expert < num_experts; ++expert) {
                    const int group =
                        std::min(expert / experts_per_group, num_groups - 1);
                    if (!keep_group[group]) {
                        choice_scores[expert] =
                            -std::numeric_limits<float>::infinity();
                    }
                }
            }
            weight_source = scores.data();
            selection_source = choice_scores.data();
        }

        for (int expert = 0; expert < num_experts; ++expert) {
            const float value = selection_source[expert];
            for (int k = 0; k < top_k; ++k) {
                if (value > weights[k]) {
                    for (int shift = top_k - 1; shift > k; --shift) {
                        weights[shift] = weights[shift - 1];
                        indices[shift] = indices[shift - 1];
                    }
                    weights[k] = value;
                    indices[k] = expert;
                    break;
                }
            }
        }

        float sum = 0.0f;
        if (params.score_func == 1) {
            for (int k = 0; k < top_k; ++k) {
                weights[k] = weight_source[indices[k]];
                sum += weights[k];
            }
            if (params.normalize_topk) {
                const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int k = 0; k < top_k; ++k)
                    weights[k] *= inverse;
            }
            for (int k = 0; k < top_k; ++k)
                weights[k] *= params.scaling_factor;
        } else {
            const float maximum = weights[0];
            for (int k = 0; k < top_k; ++k) {
                weights[k] = std::exp(weights[k] - maximum);
                sum += weights[k];
            }
            const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int k = 0; k < top_k; ++k)
                weights[k] *= inverse;
        }
    }
    return true;
}

}  // namespace mollm::detail
