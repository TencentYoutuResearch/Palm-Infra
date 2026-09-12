#pragma once

#include "kernels/tensor.h"

class ThreadPool;

// Qwen4Exp Gated Residual helpers. Large low-rank projections remain ordinary
// graph MATMULs; these kernels only implement the grouped normalization and
// stream-wise reductions surrounding them.
bool kernel_group_rms_norm(const Tensor& input, const Tensor& weight,
                           Tensor& output, int group_size, float eps,
                           ThreadPool* thread_pool = nullptr);

bool kernel_gr_reduce(const Tensor& normalized, const Tensor& gates,
                      Tensor& output, int hidden_size, int hc_count,
                      ThreadPool* thread_pool = nullptr);

bool kernel_gr_inject(const Tensor& branch, const Tensor& residual,
                      const Tensor& gates, Tensor& output,
                      int hidden_size, int hc_count,
                      ThreadPool* thread_pool = nullptr);
