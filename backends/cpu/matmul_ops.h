#pragma once

#include "core/tensor.h"

#include <vector>

struct GraphNode;
class ThreadPool;

void dispatch_cpu_matmul(
    const GraphNode& node, const std::vector<const Tensor*>& inputs,
    Tensor* output, ThreadPool* thread_pool);
