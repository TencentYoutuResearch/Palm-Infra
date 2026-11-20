#pragma once

#include "graph/graph.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace mollm::cuda_backend_detail {

// Live-payload limits for cached verifier workspaces and their recurrent state
// prefixes. Both pools may additionally retain reusable free allocations.
inline constexpr size_t verification_workspace_limit_bytes = size_t(1536) << 20;
inline constexpr size_t verification_prefix_limit_bytes = size_t(2048) << 20;
// Widths 6..8 are admitted only with the wider SM121 verifier and get larger
// aggregate budgets; widths 2..5 keep the limits above.
inline constexpr int verification_wide_min_width = 6;
inline constexpr size_t verification_wide_workspace_limit_bytes = size_t(3072) << 20;
inline constexpr size_t verification_wide_prefix_limit_bytes = size_t(4096) << 20;

inline constexpr size_t verification_workspace_budget_bytes(int max_width) {
    return max_width >= verification_wide_min_width
        ? verification_wide_workspace_limit_bytes : verification_workspace_limit_bytes;
}
inline constexpr size_t verification_prefix_budget_bytes(int max_width) {
    return max_width >= verification_wide_min_width
        ? verification_wide_prefix_limit_bytes : verification_prefix_limit_bytes;
}
// Per-width payload bound for the retained prefix bank; zero rejects the width.
// Width 2 is admitted only together with the width cache.
inline constexpr size_t retained_prefix_bank_limit_bytes(int tokens, bool width_cache = false) {
    if (tokens == 2 && width_cache) return size_t(512) << 20;
    if (tokens == 3 || tokens == 4) return size_t(512) << 20;
    if (tokens >= 5 && tokens <= 8) return size_t(1024) << 20;
    return 0;
}

// Byte extent of a graph tensor's storage; defaults to Tensor::view_span_bytes.
// Backends override it for packed device payloads whose geometry differs from
// the logical strides.
using StorageSpan = std::function<size_t(const Tensor&)>;

// Host-only proof for an already prepared, same-width dynamic graph: every
// recurrent state in `states` must have exactly one known writer that can
// emit per-token prefixes, and no other graph buffer may alias them. This does
// not prepare shapes or grant support to another backend. Unknown consumers,
// aliases and changed state geometry require the ordinary checkpoint path.
bool complete_state_prefix_plan(const Graph& graph, const std::vector<Tensor*>& states,
                                int tokens, const StorageSpan& storage_span = {});

}  // namespace mollm::cuda_backend_detail
