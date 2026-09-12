#pragma once

#include "graph/graph.h"

#import <Metal/Metal.h>

#include <vector>

class MetalPipelineCache;

// Encodes binary, unary, and fused pointwise operators.
class MetalElementwiseOps {
public:
    explicit MetalElementwiseOps(MetalPipelineCache* pipelines);

    bool dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs, Tensor* output,
                  id<MTLComputeCommandEncoder> encoder) const;

private:
    MetalPipelineCache* pipelines_ = nullptr;
};
