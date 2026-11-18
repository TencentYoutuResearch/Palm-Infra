#pragma once

#include "graph/graph.h"
#include <algorithm>
#include <climits>
#include <limits>
#include <unordered_map>

namespace mollm::cuda_backend_detail {

// Qualified verifier widths. Other widths retain the ordinary checkpoint path.
inline constexpr size_t verification_workspace_limit_bytes = size_t(1536) << 20;
// Prefix storage grows with the sum of the retained verification widths.
// Keep its bound separate from computed output leases; both are live payload
// bounds, while the existing pools also retain reusable free allocations.
inline constexpr size_t verification_prefix_limit_bytes = size_t(2048) << 20;
// The larger aggregate budgets apply only after a cache admits width6..8.
// A process using widths2..5 keeps its existing live-payload limits.
inline constexpr size_t verification_workspace_budget_bytes(int max_width) {
    return max_width > 5 ? size_t(3072) << 20 : verification_workspace_limit_bytes;
}
inline constexpr size_t verification_prefix_budget_bytes(int max_width) {
    return max_width > 5 ? size_t(4096) << 20 : verification_prefix_limit_bytes;
}
inline constexpr size_t retained_prefix_bank_limit_bytes(int tokens, bool width_cache = false) {
    if (tokens == 2 && width_cache) return size_t(512) << 20;
    return (tokens == 3 || tokens == 4) ? size_t(512) << 20 : (tokens >= 5 && tokens <= 8) ? size_t(1024) << 20 : 0;
}

// Host-only proof for an already prepared, same-width dynamic graph. This does
// not prepare shapes or grant support to another backend. Unknown consumers,
// aliases and changed state geometry require the ordinary checkpoint path.
inline uintptr_t prefix_address(const Tensor& tensor) {
    const auto base = reinterpret_cast<uintptr_t>(tensor.device.buffer);
    return base && tensor.device.offset <= std::numeric_limits<uintptr_t>::max() - base
        ? base + tensor.device.offset : 0;
}

inline bool prefix_state_size(const Tensor& state, uint64_t elements, Precision precision) {
    return elements > 0 && elements <= std::numeric_limits<size_t>::max() / 4 &&
        state.prec == precision && state.is_contiguous() &&
        state.view_span_bytes() == elements * 4;
}

inline bool prefix_writer_covers(const Graph& graph, const GraphNode& node,
                                uint32_t state_id, int tokens) {
    const auto& tensors = graph.runtime.tensors;
    if (tensors.size() != graph.nodes.size() || node.id >= tensors.size() || state_id >= tensors.size() ||
        std::any_of(node.inputs.begin(), node.inputs.end(),
                    [&](uint32_t id) { return id >= tensors.size(); })) return false;
    const Tensor& state = tensors[state_id];
    const Tensor& out = tensors[node.id];
    const auto arg = [&](size_t i) -> const Tensor& { return tensors[node.inputs[i]]; };
    const auto fp32 = [&](const Tensor& t) {
        if (t.prec != Precision::FP32) return false;
        if (prefix_address(t)) return true;
        // finish_graph_temporaries clears borrowed view handles. The executor
        // rebuilds these views from the same-width graph before dispatch.
        // Persistent INPUT/CONSTANT storage and owned outputs cannot use this.
        const auto op = graph.nodes[size_t(&t-tensors.data())].op_type;
        return !t.device.buffer && !t.data &&
            (op == OpType::SLICE || op == OpType::PERMUTE ||
             op == OpType::RESHAPE || op == OpType::CONTIGUOUS);
    };
    const auto same_shape = [](const Tensor& a, const Tensor& b) {
        return std::equal(a.shape, a.shape + 4, b.shape);
    };
    const auto p = [&](size_t i, int fallback) { return graph_params::get_i32(node.params, i, fallback); };
    if (node.op_type == OpType::PLE_LOOKUP) {
        return node.inputs.size() >= 6 && node.inputs[1] == state_id &&
            arg(0).prec == Precision::INT32 && arg(0).nelements() == tokens &&
            arg(4).prec == Precision::INT32 && arg(5).prec == Precision::INT32 &&
            p(0,3) >= 2 && p(1,8) > 0 && p(7,tokens) == tokens &&
            prefix_state_size(state, uint64_t(p(0,3)), Precision::INT32);
    }
    if (!fp32(state) || !fp32(out) || !out.is_contiguous() ||
        prefix_address(out) == prefix_address(state)) return false;
    if (node.op_type == OpType::GATED_DELTANET_PREFILL) {
        if (node.inputs.size() < 8 || node.inputs[7] != state_id) return false;
        for (size_t i = 0; i < 8; ++i) if (!fp32(arg(i))) return false;
        const int heads = p(0,16), value_heads = p(7,heads);
        const int64_t width = (2LL * heads + value_heads) * 128;
        const int64_t output_width = int64_t(value_heads) * 128;
        return heads > 0 && value_heads > 0 && value_heads % heads == 0 &&
            width > 0 && width <= INT_MAX && output_width <= INT_MAX &&
            p(1,128) == 128 && p(2,128) == 128 &&
            p(3,1) == tokens && p(6,p(3,1)) == tokens &&
            arg(0).nelements() >= width * tokens &&
            arg(1).shape[0] >= value_heads && arg(2).shape[0] >= value_heads &&
            arg(3).shape[0] >= output_width && arg(4).nelements() >= value_heads &&
            arg(5).nelements() >= value_heads && arg(6).nelements() >= 128 &&
            arg(1).stride[0] == 4 && arg(2).stride[0] == 4 && arg(3).stride[0] == 4 &&
            out.nelements() >= output_width * tokens &&
            prefix_state_size(state, uint64_t(value_heads) * 128 * 128, Precision::FP32);
    }
    if ((node.op_type != OpType::SHORTCONV && node.op_type != OpType::PLE_DILATED_CONV) ||
        node.inputs.size() != 3 || node.inputs[2] != state_id ||
        !fp32(arg(0)) || !fp32(arg(1)) || !same_shape(arg(0),out) ||
        arg(0).shape[0] <= 0 || arg(0).shape[0] > INT_MAX ||
        arg(0).shape[1] != tokens || arg(0).shape[2] != 1 || arg(0).shape[3] != 1)
        return false;
    const int64_t groups = arg(0).shape[0];
    const int kernel = p(0,4);
    if (kernel <= 1 || arg(0).stride[0] % 4 || arg(0).stride[1] % 4 ||
        prefix_address(arg(0)) == prefix_address(state)) return false;
    if (node.op_type == OpType::SHORTCONV)
        return p(1,tokens) == tokens && arg(1).nelements() >= groups * kernel &&
            prefix_state_size(state, uint64_t(groups) * (kernel-1), Precision::FP32);
    const int dilation = p(1,3);
    const int64_t history = int64_t(kernel-1) * dilation;
    return dilation > 0 && history <= INT_MAX && p(2,tokens) == tokens &&
        arg(0).is_contiguous() && arg(1).is_contiguous() &&
        arg(1).shape[0] == groups && arg(1).shape[1] == kernel &&
        arg(1).shape[2] == 1 && arg(1).shape[3] == 1 &&
        state.shape[0] == groups && state.shape[1] == history &&
        prefix_address(arg(0)) != prefix_address(out) &&
        prefix_state_size(state, uint64_t(groups) * history, Precision::FP32);
}

struct PrefixTensorSpan {
    size_t operator()(const Tensor& tensor) const { return tensor.view_span_bytes(); }
};

template<class StorageSpan = PrefixTensorSpan>
inline bool complete_state_prefix_plan(const Graph& graph,
                                      const std::vector<Tensor*>& states, int tokens,
                                      StorageSpan storage_span = {}) {
    const auto& tensors = graph.runtime.tensors;
    if (tokens < 2 || tokens > 8 || states.empty() || tensors.size() != graph.nodes.size())
        return false;
    std::unordered_map<const Tensor*, uint32_t> pending;
    for (Tensor* state : states) {
        if (!state || !state->view_span_bytes()) continue;
        if (!prefix_address(*state) || !state->is_contiguous()) return false;
        pending.emplace(state, UINT32_MAX);
    }
    if (pending.empty()) return false;
    std::unordered_map<uint32_t, size_t> uses;
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        if (node.id != i) return false;
        auto found = pending.find(&tensors[i]);
        if (found == pending.end()) continue;
        if (node.op_type != OpType::INPUT ||
            std::any_of(node.dim_expr,node.dim_expr+4,[](const auto& e){ return e.is_dynamic(); }))
            return false;
        found->second = node.id;
        uses.emplace(node.id,0);
    }
    struct Range { uintptr_t begin,end; };
    std::vector<Range> ranges;
    for (const auto& [state,id] : pending) {
        if (id == UINT32_MAX) return false;
        const uintptr_t begin = prefix_address(*state);
        const size_t bytes = state->view_span_bytes();
        if (bytes > std::numeric_limits<uintptr_t>::max() - begin) return false;
        ranges.push_back({begin,begin+bytes});
    }
    std::sort(ranges.begin(),ranges.end(),[](const auto& a,const auto& b){return a.begin<b.begin;});
    for(size_t i=1;i<ranges.size();++i)if(ranges[i].begin<ranges[i-1].end)return false;
    // Inspect each graph buffer once; a full per-state graph scan would add
    // significant host work to the verification path this plan accelerates.
    for (const auto& node : graph.nodes) {
        if (uses.count(node.id)) continue;
        const Tensor& other = tensors[node.id];
        uintptr_t address = prefix_address(other);
        if (!address && other.device.buffer) return false;
        if (!address) address = reinterpret_cast<uintptr_t>(other.data);
        const size_t span = storage_span(other);
        if (!address || !span) continue;
        if (span > std::numeric_limits<uintptr_t>::max()-address) return false;
        const auto next=std::lower_bound(ranges.begin(),ranges.end(),address,
            [](const auto& range,uintptr_t point){return range.end<=point;});
        if(next!=ranges.end() && next->begin<address+span)return false;
    }
    for (const auto& node : graph.nodes) for (uint32_t input : node.inputs) {
        if (input >= node.id) return false;
        auto found = uses.find(input);
        if (found == uses.end()) continue;
        if (++found->second != 1 || !prefix_writer_covers(graph,node,input,tokens)) return false;
    }
    return std::all_of(uses.begin(),uses.end(),[](const auto& entry){return entry.second==1;});
}

} // namespace mollm::cuda_backend_detail
