#pragma once

#include "core/tensor.h"
#include "runtime/threading.h"

#include <vector>

enum class ElementwiseOp {
    ADD, MUL, SIGMOID_MUL, SILU, GELU, TANH, SWIGLU,
    SIGMOID, SIGMOID_EXACT, EXP, EXP_EXACT, SOFTPLUS
};

// Execute one CPU elementwise operation. Inputs may be strided views; output is
// materialized densely. ADD and MUL support NumPy-style singleton broadcasting
// from the second operand.
void kernel_elementwise(ElementwiseOp op, const std::vector<const Tensor*>& inputs,
                        Tensor* output, ThreadPool* thread_pool = nullptr);
