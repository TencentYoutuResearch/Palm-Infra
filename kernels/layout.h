#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"

#include <vector>

// Execute a CPU tensor-layout operation.
//
// RESHAPE, SLICE, and PERMUTE may replace output with a borrowed view.
// CONCAT, TILE, CONTIGUOUS, and non-contiguous RESHAPE materialize into the
// storage already assigned to output by the graph executor.
void kernel_layout(const GraphNode& node,
                   const std::vector<const Tensor*>& inputs, Tensor* output);
