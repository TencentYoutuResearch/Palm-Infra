#pragma once

#include "graph/graph.h"

void dispatch_cpu_layout(
    const GraphNode& node, const std::vector<const Tensor*>& inputs,
    Tensor* output);
