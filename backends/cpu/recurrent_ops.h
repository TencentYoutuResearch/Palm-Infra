#pragma once

#include "graph/graph.h"

class ThreadPool;

void dispatch_cpu_recurrent(
    const GraphNode& node, const std::vector<const Tensor*>& inputs,
    Tensor* output, ThreadPool* thread_pool);
