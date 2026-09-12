#pragma once

#include "graph/graph.h"

class ThreadPool;

bool dispatch_cpu_attention(
    const GraphNode& node, const std::vector<const Tensor*>& inputs,
    Tensor* output, ThreadPool* thread_pool);
