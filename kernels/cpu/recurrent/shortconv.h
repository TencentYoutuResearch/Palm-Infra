#pragma once

#include "core/shortconv_params.h"
#include "core/tensor.h"
#include "runtime/threading.h"

#include <vector>

// Depth-wise causal convolution followed by SiLU.
//
// Inputs:
//   inputs[0] x          FP32, logical shape [groups, seq], row-major [seq,
//   groups] inputs[1] weight     FP32, [kernel_size, groups] inputs[2] state
//   FP32, [kernel_size - 1, groups], updated in place
//
void kernel_shortconv(const ShortConvParams& params,
                      const std::vector<const Tensor*>& inputs, Tensor& output,
                      ThreadPool* thread_pool = nullptr);
