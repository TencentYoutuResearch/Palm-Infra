#pragma once

#include "graph/graph.h"

#import <Metal/Metal.h>

#include <string>
#include <vector>

class MetalBufferPool;
class MetalCommandContext;
class MetalPipelineCache;
class MetalResourceStore;

// Encodes dense and quantized matrix multiplication paths.
class MetalMatmulOps {
public:
  MetalMatmulOps(MetalPipelineCache *pipelines, MetalBufferPool *pool,
                 MetalCommandContext *commands, MetalResourceStore *resources);

  bool dispatch(const GraphNode &node,
                const std::vector<const Tensor *> &inputs, Tensor *output,
                id<MTLComputeCommandEncoder> encoder, bool has_tensor,
                std::string &profile_label) const;

private:
  MetalPipelineCache *pipelines_ = nullptr;
  MetalBufferPool *pool_ = nullptr;
  MetalCommandContext *commands_ = nullptr;
  MetalResourceStore *resources_ = nullptr;
};
