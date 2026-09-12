#pragma once

#import <Metal/Metal.h>

#include <memory>

// Owns ordinary and function-constant-specialized Metal compute pipelines.
// This is an Objective-C++-only internal component.
class MetalPipelineCache {
public:
    MetalPipelineCache(void* device, void* library);
    ~MetalPipelineCache();

    MetalPipelineCache(const MetalPipelineCache&) = delete;
    MetalPipelineCache& operator=(const MetalPipelineCache&) = delete;

    id<MTLComputePipelineState> pipeline(const char* name);
    id<MTLComputePipelineState> fa2(int dk, int dv, int nsg, int query_tile);
    id<MTLComputePipelineState> gemv2(int rows_per_group);
    id<MTLComputePipelineState> gemv_w4(int rows_per_group);
    id<MTLComputePipelineState> small_m(const char* function_name, int m);
    id<MTLComputePipelineState> w4a16(bool use_m128, bool specialize_g128);
    id<MTLComputePipelineState> moe_select_parallel(bool sigmoid,
                                                    bool grouped);
    id<MTLComputePipelineState> grouped_moe(int group_size,
                                            bool paired_gate_up,
                                            bool large_route_tile);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
