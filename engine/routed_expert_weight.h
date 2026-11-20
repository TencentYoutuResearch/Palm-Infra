#pragma once

#include "graph/graph.h"

#include <cstdint>
#include <unordered_map>

namespace mollm {

// Returns, for each FP16 constant used only as a routed MoE gate/up or down
// matrix, the per-expert row count. Loading hints only: exposed weights,
// aliases through other operators and malformed graphs are omitted and keep
// the ordinary load path.
std::unordered_map<uint32_t, int> routed_fp16_expert_rows(const Graph& graph);

}  // namespace mollm
