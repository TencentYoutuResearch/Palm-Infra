#pragma once

#include "graph/graph.h"

#import <Metal/Metal.h>

#include <vector>

class MetalPipelineCache;

// Encodes metadata-only views and strided-to-contiguous materialization.
class MetalLayoutOps {
public:
    explicit MetalLayoutOps(MetalPipelineCache* pipelines);

    bool dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs, Tensor* output,
                  id<MTLComputeCommandEncoder> encoder,
                  bool& encoded_gpu_work) const;

private:
    MetalPipelineCache* pipelines_ = nullptr;
};
