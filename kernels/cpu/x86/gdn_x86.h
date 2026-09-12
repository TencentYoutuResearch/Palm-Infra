#pragma once

#include "core/gdn_params.h"
#include "core/tensor.h"

#include <vector>

class ThreadPool;

void kernel_gdn_x86_avx512(const GdnParams& params,
                           const std::vector<const Tensor*>& inputs,
                           std::vector<Tensor*>& outputs,
                           ThreadPool* thread_pool);
