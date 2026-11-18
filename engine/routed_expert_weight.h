#pragma once
#include "graph/graph.h"
#include <cstdint>
#include <unordered_map>

namespace mollm {
// Loading hints only. Every consumer must use this constant as a routed
// gate/up or down matrix with the same expert boundaries. Exposed weights,
// aliases through other operators and malformed graphs retain ordinary load.
inline std::unordered_map<uint32_t, int> routed_fp16_expert_rows(const Graph& graph) {
    struct Use { int rows = 0; bool rejected = false; };
    std::unordered_map<uint32_t, const GraphNode*> nodes;
    std::unordered_map<uint32_t, Use> uses;
    for (const auto& node : graph.nodes) nodes[node.id] = &node;
    for (const auto& node : graph.nodes) {
        for (size_t index = 0; index < node.inputs.size(); ++index) {
            const auto id = node.inputs[index];
            auto found = nodes.find(id);
            if (found == nodes.end() || found->second->op_type != OpType::CONSTANT ||
                found->second->out_prec != Precision::FP16) continue;
            auto& use = uses[id];
            if (use.rejected) continue;
            const auto& weight = *found->second;
            if (node.op_type != OpType::MOE || node.inputs.size() != 8 ||
                (index != 2 && index != 3) || node.params.i32.size() < 4) {
                use.rejected = true;
                continue;
            }
            const int64_t hidden = node.params.i32[0], experts = node.params.i32[1];
            const int64_t inter = node.params.i32[3];
            const int64_t rows = index == 2 ? 2 * inter : hidden;
            const int64_t inner = index == 2 ? hidden : inter;
            if (hidden <= 0 || inter <= 0 || experts <= 0 || experts > 1024 ||
                rows > INT32_MAX || weight.out_shape[0] != rows * experts ||
                weight.out_shape[1] != inner || weight.out_shape[2] != 1 ||
                weight.out_shape[3] != 1 || (use.rows && use.rows != rows)) {
                use.rejected = true;
                continue;
            }
            use.rows = static_cast<int>(rows);
        }
    }
    for (auto id : graph.graph_outputs) uses[id].rejected = true;
    for (auto id : graph.graph_inputs) uses[id].rejected = true;
    std::unordered_map<uint32_t, int> result;
    for (const auto& entry : uses)
        if (!entry.second.rejected && entry.second.rows)
            result.emplace(entry.first, entry.second.rows);
    return result;
}
} // namespace mollm
