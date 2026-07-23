#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"
#include "kernels/threading.h"

#include <vector>

// Depth-wise causal convolution followed by SiLU.
//
// Inputs:
//   inputs[0] x          FP32, logical shape [groups, seq], row-major [seq,
//   groups] inputs[1] weight     FP32, [kernel_size, groups] inputs[2] state
//   FP32, [kernel_size - 1, groups], updated in place
//
// Params:
//   i32[0] = kernel_size
//   i32[1] = number of real tokens when a prefill input is padded
void kernel_shortconv(const OpParams& params,
                      const std::vector<const Tensor*>& inputs, Tensor& output,
                      ThreadPool* thread_pool = nullptr);
