#pragma once

#include "core/tensor.h"

#include <array>
#include <vector>

enum class LayoutOp { RESHAPE, CONCAT, SLICE, TILE, PERMUTE, CONTIGUOUS };

// The backend resolves graph defaults and dynamic dimensions before dispatch.
struct LayoutParams {
    LayoutOp op = LayoutOp::CONTIGUOUS;
    std::array<int64_t, 4> shape = {1, 1, 1, 1};
    std::array<int, 4> repeats = {1, 1, 1, 1};
    std::array<int, 4> order = {0, 1, 2, 3};
    int axis = 0;
    int offset = 0;
    int size = 0;
};

// Execute a CPU tensor-layout operation.
//
// RESHAPE, SLICE, and PERMUTE may replace output with a borrowed view.
// CONCAT, TILE, CONTIGUOUS, and non-contiguous RESHAPE materialize into the
// storage already assigned to output by the graph executor.
void kernel_layout(const LayoutParams& params,
                   const std::vector<const Tensor*>& inputs, Tensor* output);
