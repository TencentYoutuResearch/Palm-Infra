#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"

#include <vector>

class ThreadPool;

void kernel_gdn_x86_avx512(const OpParams& params,
                           const std::vector<const Tensor*>& inputs,
                           std::vector<Tensor*>& outputs,
                           ThreadPool* thread_pool);
