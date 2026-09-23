#pragma once

#include "core/tensor.h"

#include <vector>

struct GraphNode;

void dispatch_cpu_layout(
    const GraphNode& node, const std::vector<const Tensor*>& inputs,
    Tensor* output);
