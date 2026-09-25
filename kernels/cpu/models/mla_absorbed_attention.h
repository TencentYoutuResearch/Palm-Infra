#pragma once

#include "core/tensor.h"

#include <vector>

struct GraphNode;
class ThreadPool;

// Dense Hy4 MLA with one head-less [kv_rank + rope_dim, n_ctx] FP16 cache.
// The kernel folds K into Q, streams softmax over the latent cache, then
// applies V once to the weighted latent sum.
bool kernel_mla_absorbed_attention(
    const GraphNode& node, const std::vector<const Tensor*>& inputs,
    Tensor* output, ThreadPool* thread_pool);
