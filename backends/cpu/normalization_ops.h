#pragma once

#include "core/tensor.h"

#include <vector>

struct GraphNode;
class ThreadPool;

bool dispatch_cpu_normalization(
    const GraphNode& node, const std::vector<const Tensor*>& inputs,
    Tensor* output, ThreadPool* thread_pool);
