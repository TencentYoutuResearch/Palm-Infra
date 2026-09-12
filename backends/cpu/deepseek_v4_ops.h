#pragma once

#include "graph/graph.h"

class ThreadPool;

bool dispatch_cpu_deepseek_v4(
    const GraphNode& node, const std::vector<const Tensor*>& inputs,
    Tensor* output, ThreadPool* thread_pool);
