#pragma once

#include "graph/graph.h"

#import <Metal/Metal.h>

#include <vector>

class MetalPipelineCache;

// Encodes normalization operators, including fused normalization and RoPE.
class MetalNormalizationOps {
public:
    explicit MetalNormalizationOps(MetalPipelineCache* pipelines);

    bool dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs, Tensor* output,
                  id<MTLComputeCommandEncoder> encoder) const;

private:
    MetalPipelineCache* pipelines_ = nullptr;
};
