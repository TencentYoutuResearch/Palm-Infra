#pragma once

#include "core/tensor.h"

#include <vector>

struct GraphNode;
class ThreadPool;
class HostMoeFfn;

bool dispatch_cpu_moe(
    const GraphNode& node, const std::vector<const Tensor*>& inputs,
    Tensor* output, ThreadPool* thread_pool,
    HostMoeFfn* host_moe_ffn = nullptr);
