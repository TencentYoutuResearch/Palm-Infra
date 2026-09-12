#pragma once

#include "graph/graph.h"

#import <Metal/Metal.h>

#include <vector>

class MetalPipelineCache;

// Encodes rotary position embedding and any required input materialization.
class MetalRotaryOps {
public:
    explicit MetalRotaryOps(MetalPipelineCache* pipelines);

    bool dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs, Tensor* output,
                  id<MTLComputeCommandEncoder> encoder) const;

private:
    MetalPipelineCache* pipelines_ = nullptr;
};
