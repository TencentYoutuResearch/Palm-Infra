#include "kernels/cpu/moe/moe_routing.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

bool close(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

}  // namespace

int main() {
    {
        const float logits[] = {1.0f, 3.0f, 2.0f, 0.0f};
        mollm::detail::MoeRoutingParams params;
        params.num_experts = 4;
        params.top_k = 2;
        std::vector<int> indices;
        std::vector<float> weights;
        check(mollm::detail::select_moe_routes(
                  logits, 1, nullptr, params, indices, weights),
              "select softmax routes");
        check(indices == std::vector<int>({1, 2}),
              "softmax route order");
        const float denom = 1.0f + std::exp(-1.0f);
        check(close(weights[0], 1.0f / denom) &&
                  close(weights[1], std::exp(-1.0f) / denom),
              "softmax weights normalize over selected experts");
    }

    {
        // Selection uses biased sigmoid scores, while returned weights use
        // the original unbiased sigmoid values.
        const float logits[] = {
            2.0f, 1.0f, 0.0f, -1.0f,
            -2.0f, -3.0f, -4.0f, -5.0f,
        };
        const float bias[] = {
            0.0f, 0.0f, 0.0f, 0.0f,
            10.0f, 9.0f, 0.0f, 0.0f,
        };
        mollm::detail::MoeRoutingParams params;
        params.num_experts = 8;
        params.top_k = 2;
        params.score_func = 1;
        params.normalize_topk = false;
        params.num_groups = 2;
        params.topk_groups = 1;
        params.scaling_factor = 2.0f;
        std::vector<int> indices;
        std::vector<float> weights;
        check(mollm::detail::select_moe_routes(
                  logits, 1, bias, params, indices, weights),
              "select grouped sigmoid routes");
        check(indices == std::vector<int>({4, 5}),
              "group pruning follows biased selection scores");
        check(close(weights[0], 2.0f * sigmoid(-2.0f)) &&
                  close(weights[1], 2.0f * sigmoid(-3.0f)),
              "sigmoid route weights remain unbiased");
    }

    {
        const float logits[] = {
            -2.0f, 0.0f, 2.0f, 1.0f,
            3.0f, -1.0f, 0.5f, 2.0f,
        };
        const int32_t tokens[] = {1, 0};
        const int32_t table[] = {
            3, 0,
            2, 1,
        };
        mollm::detail::MoeRoutingParams params;
        params.num_experts = 4;
        params.top_k = 2;
        params.score_func = 2;
        params.scaling_factor = 1.5f;
        std::vector<int> indices;
        std::vector<float> weights;
        check(mollm::detail::select_moe_hash_routes(
                  logits, 2, tokens, table, 2, params,
                  indices, weights),
              "select hash routes");
        check(indices == std::vector<int>({2, 1, 3, 0}),
              "hash routes follow token lookup table");
        check(close(weights[0] + weights[1], 1.5f) &&
                  close(weights[2] + weights[3], 1.5f),
              "hash route weights are normalized and scaled");
    }

    {
        const float logits[] = {-2.0f, 0.0f, 2.0f, 1.0f};
        const float bias[] = {10.0f, 0.0f, 0.0f, 0.0f};
        mollm::detail::MoeRoutingParams params;
        params.num_experts = 4;
        params.top_k = 2;
        params.score_func = 2;
        params.normalize_topk = true;
        params.scaling_factor = 1.5f;
        std::vector<int> indices;
        std::vector<float> weights;
        check(mollm::detail::select_moe_routes(
                  logits, 1, bias, params, indices, weights),
              "select sqrt-softplus routes");
        check(indices == std::vector<int>({0, 2}),
              "sqrt-softplus bias affects selection");
        const float score0 = std::sqrt(std::log1p(std::exp(-2.0f)));
        const float score2 = std::sqrt(std::log1p(std::exp(2.0f)));
        check(close(weights[0], 1.5f * score0 / (score0 + score2)) &&
                  close(weights[1], 1.5f * score2 / (score0 + score2)),
              "sqrt-softplus weights use unbiased normalized scores");
    }

    {
        const float logits[] = {
            1.0f, 1.0f, 0.0f,
            0.0f, 2.0f, 3.0f,
        };
        mollm::detail::MoeRoutingParams params;
        params.num_experts = 3;
        params.top_k = 2;
        std::vector<int> indices;
        std::vector<float> weights;
        check(mollm::detail::select_moe_routes(
                  logits, 2, nullptr, params, indices, weights),
              "select batched routes");
        check(indices == std::vector<int>({0, 1, 2, 1}),
              "batched routing is deterministic on ties");
    }

    if (failures == 0)
        std::printf("All MoE routing tests passed!\n");
    return failures == 0 ? 0 : 1;
}
