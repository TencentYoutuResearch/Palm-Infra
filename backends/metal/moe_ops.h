#pragma once

#include "graph/graph.h"

#import <Metal/Metal.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class MetalBufferPool;
class MetalCommandContext;
class MetalPipelineCache;
class MetalSsdExpertCache;
class MetalSsdSharedExpert;
struct MoeSsdTensorSource;
namespace mollm::detail {
struct MoeRoutingParams;
}

// Owns Metal MoE dispatch, SSD expert streams, and cross-layer prefetch state.
class MetalMoeOps {
public:
    MetalMoeOps(void* device, MetalPipelineCache* pipelines,
                MetalBufferPool* pool, MetalCommandContext* commands,
                bool& dispatch_failed);
    ~MetalMoeOps();

    bool configure_ssd_io(const std::string& package_path,
                          size_t capacity_bytes,
                          int max_commands_in_flight,
                          bool cross_layer_prefetch);

    bool dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs, Tensor* output,
                  id<MTLComputeCommandEncoder> encoder, ThreadPool* thread_pool,
                  bool has_tensor, std::string& profile_label,
                  bool& abort_dispatch);

    bool dispatch_host(const GraphNode& node,
                       const std::vector<const Tensor*>& inputs,
                       Tensor* output, ThreadPool* thread_pool, bool& success);

private:
    struct SsdMoeLayerInfo {
        const Tensor* router = nullptr;
        const Tensor* bias = nullptr;
        const MoeSsdTensorSource* gate_up = nullptr;
        const MoeSsdTensorSource* down = nullptr;
        int hidden = 0;
        int experts = 0;
        int top_k = 0;
        int intermediate = 0;
        int score_func = 0;
        int n_group = 1;
        int topk_group = 1;
        bool norm_topk = true;
        float routed_scale = 1.0f;
        const Tensor* token_to_experts = nullptr;
    };

    bool finish_ssd_prefix(int layer, const char* error_context);
    bool route_moe_on_cpu(const Tensor& input, const Tensor& router,
                          const Tensor* bias,
                          const mollm::detail::MoeRoutingParams& routing,
                          ThreadPool* thread_pool, void* indices_handle,
                          void* weights_handle);

    MetalPipelineCache* pipelines_ = nullptr;
    MetalBufferPool* pool_ = nullptr;
    MetalCommandContext* commands_ = nullptr;
    bool& dispatch_failed_;
    std::unique_ptr<MetalSsdExpertCache> ssd_cache_;
    std::unique_ptr<MetalSsdSharedExpert> ssd_shared_expert_;
    bool ssd_cross_layer_prefetch_ = true;
    std::unordered_map<int, SsdMoeLayerInfo> ssd_moe_layers_;
};
