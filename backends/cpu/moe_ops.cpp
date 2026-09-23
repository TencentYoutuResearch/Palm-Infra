#include "backends/cpu/moe_ops.h"

#include "graph/graph.h"

#include "kernels/cpu/moe/moe.h"

bool dispatch_cpu_moe(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* thread_pool) {
    const OpParams& params = node.params;
    bool success = true;
    auto reject = [&] { success = false; };
    int hidden_size = graph_params::get_i32(params, 0, output ? (int)output->shape[0] : 0);
    int num_experts = graph_params::get_i32(params, 1, 0);
    int top_k = graph_params::get_i32(params, 2, 0);
    int intermediate_size = graph_params::get_i32(params, 3, 0);
    int shared_intermediate_size = graph_params::get_i32(params, 4, intermediate_size);
    int router_score_func = graph_params::get_i32(params, 5, 0);
    bool norm_topk_prob = graph_params::get_i32(params, 6, 1) != 0;
    bool has_shared_expert = graph_params::get_i32(params, 7, 1) != 0;
    int n_group = graph_params::get_i32(params, 8, 1);
    int topk_group = graph_params::get_i32(params, 9, 1);
    bool shared_expert_has_gate =
        graph_params::get_i32(params, 10, 1) != 0;
    int router_bias_input =
        graph_params::get_i32(
            params, 11, has_shared_expert ? 8 : -1);
    int token_ids_input = graph_params::get_i32(params, 12, -1);
    int hash_table_input = graph_params::get_i32(params, 13, -1);
    float routed_scaling_factor = graph_params::get_f32(params, 0, 1.0f);
    float swiglu_limit = graph_params::get_f32(params, 1, 0.0f);
    if (output) {
        if (!kernel_qwen3_moe(
                inputs, *output, thread_pool,
                hidden_size, num_experts, top_k,
                intermediate_size, shared_intermediate_size,
                router_score_func, norm_topk_prob,
                has_shared_expert, n_group, topk_group,
                routed_scaling_factor,
                shared_expert_has_gate, router_bias_input,
                token_ids_input, hash_table_input,
                swiglu_limit)) {
            reject();
        }
    } else {
        reject();
    }
    return success;
}
