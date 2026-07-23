#include "kernels/moe_ssd.h"

#include "kernels/matmul.h"
#include "kernels/moe_routing.h"
#include "kernels/trace.h"

#include <algorithm>
#include <utility>
#include <vector>

bool schedule_moe_cross_layer_prefetch(
    const Tensor& gate_input,
    const Tensor& next_router,
    const Tensor* next_router_bias,
    const MoeSsdTensorSource* next_gate_up,
    const MoeSsdTensorSource* next_down,
    const MoeSsdPredictConfig& config) {
    if (!next_gate_up || !next_down || !next_gate_up->cache ||
        next_gate_up->cache != next_down->cache ||
        gate_input.prec != Precision::FP32 || gate_input.shape[1] != 1 ||
        config.hidden_size <= 0 || config.num_experts <= 0 ||
        config.top_k <= 0 || next_router.shape[0] != config.num_experts ||
        next_router.shape[1] != config.hidden_size) {
        return false;
    }

    // Graph workspaces are reused on the next token, so clone the decode
    // vector before handing it to the asynchronous predictor worker.
    const float* source = gate_input.ptr<float>();
    std::vector<float> input(source, source + config.hidden_size);
    MoeSsdCache* cache = next_gate_up->cache;
    // Leave room for both the speculative and exact routes. Otherwise a
    // predictor can churn a small cache before the real router reaches it.
    if (!cache->can_prefetch_pairs(next_gate_up, next_down,
                                   static_cast<size_t>(config.top_k) * 2)) {
        return false;
    }

    return cache->submit_cross_layer_task(
        [input = std::move(input), &next_router, next_router_bias,
         next_gate_up, next_down, config, cache]() mutable {
            std::vector<float> logits(static_cast<size_t>(config.num_experts));
            Tensor router_input = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL,
                config.hidden_size, 1, 1, 1, input.data());
            Tensor router_output = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL,
                config.num_experts, 1, 1, 1, logits.data());
            {
                mollm_trace::ScopedEvent trace_event(
                    "ssd.predict", "next_layer_router",
                    "{\"layer\":" +
                        std::to_string(next_gate_up->spec.layer) + "}");
                // This uses an idle SSD worker instead of the inference pool,
                // overlapping the current layer's compute.
                kernel_matmul_fp32(
                    router_input, next_router, router_output, nullptr);
            }

            mollm::detail::MoeRoutingParams routing;
            routing.num_experts = config.num_experts;
            routing.top_k = config.top_k;
            routing.score_func = config.router_score_func;
            routing.num_groups = config.n_group;
            routing.topk_groups = config.topk_group;
            std::vector<int> experts;
            std::vector<float> weights;
            const float* bias =
                next_router_bias && next_router_bias->data
                    ? next_router_bias->ptr<float>()
                    : nullptr;
            if (mollm::detail::select_moe_routes(
                    logits.data(), 1, bias, routing, experts, weights)) {
                if (!weights.empty()) {
                    const float best = std::max(weights.front(), 1e-12f);
                    for (float& weight : weights) weight /= best;
                }
                cache->prefetch_many(next_gate_up, next_down, experts, weights);
            }
        });
}
