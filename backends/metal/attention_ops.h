#pragma once

#include "graph/graph.h"

#import <Metal/Metal.h>

#include <string>
#include <vector>

class MetalBufferPool;
class MetalCommandContext;
class MetalPipelineCache;

// Encodes SDPA/MLA cache updates and attention compute paths.
class MetalAttentionOps {
public:
    MetalAttentionOps(MetalPipelineCache* pipelines, MetalBufferPool* pool,
                      MetalCommandContext* commands);

    bool dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs, Tensor* output,
                  id<MTLComputeCommandEncoder> encoder,
                  std::string& profile_label) const;

private:
    MetalPipelineCache* pipelines_ = nullptr;
    MetalBufferPool* pool_ = nullptr;
    MetalCommandContext* commands_ = nullptr;
};
