#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"
#include "kernels/threading.h"

#include <vector>

// Execute one CPU elementwise graph op. Inputs may be strided views; output is
// materialized densely. ADD and MUL support NumPy-style singleton broadcasting
// from the second operand.
void kernel_elementwise(OpType op, const std::vector<const Tensor*>& inputs,
                        Tensor* output, ThreadPool* thread_pool = nullptr);
