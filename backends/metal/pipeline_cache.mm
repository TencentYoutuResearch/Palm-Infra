#include "backends/metal/pipeline_cache.h"

#import <Foundation/Foundation.h>

#include <cstdio>
#include <string>
#include <unordered_map>

struct MetalPipelineCache::Impl {
    Impl(void* device_handle, void* library_handle)
        : device((__bridge id<MTLDevice>)device_handle),
          library((__bridge id<MTLLibrary>)library_handle) {}

    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> ordinary;
    std::unordered_map<std::string, id<MTLComputePipelineState>> specialized;
};

MetalPipelineCache::MetalPipelineCache(void* device, void* library)
    : impl_(new Impl(device, library)) {}

MetalPipelineCache::~MetalPipelineCache() = default;

id<MTLComputePipelineState> MetalPipelineCache::pipeline(const char* name) {
    const std::string key(name);
    auto found = impl_->ordinary.find(key);
    if (found != impl_->ordinary.end()) return found->second;
    id<MTLFunction> function = [impl_->library newFunctionWithName:@(name)];
    if (!function) {
        std::fprintf(stderr,
                     "MetalBackend: kernel function '%s' not found\n", name);
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> state =
        [impl_->device newComputePipelineStateWithFunction:function
                                                     error:&error];
    if (!state) {
        std::fprintf(stderr, "MetalBackend: pipeline '%s' failed: %s\n",
                     name,
                     error ? error.localizedDescription.UTF8String : "?");
        return nil;
    }
    impl_->ordinary[key] = state;
    return state;
}

id<MTLComputePipelineState> MetalPipelineCache::fa2(
        int dk, int dv, int nsg, int query_tile) {
    char key_buffer[64];
    std::snprintf(key_buffer, sizeof(key_buffer),
                  "fa2:dk%d:dv%d:nsg%d:qt%d", dk, dv, nsg, query_tile);
    const std::string key(key_buffer);
    auto found = impl_->specialized.find(key);
    if (found != impl_->specialized.end()) return found->second;

    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&dk type:MTLDataTypeInt atIndex:0];
    [constants setConstantValue:&dv type:MTLDataTypeInt atIndex:1];
    [constants setConstantValue:&nsg type:MTLDataTypeInt atIndex:9];
    [constants setConstantValue:&query_tile type:MTLDataTypeInt atIndex:11];
    NSError* error = nil;
    id<MTLFunction> function =
        [impl_->library newFunctionWithName:@"sdpa_prefill_fa2_f32"
                            constantValues:constants
                                     error:&error];
    if (!function) {
        std::fprintf(stderr,
                     "MetalBackend: fa2 specialized function failed: %s\n",
                     error ? error.localizedDescription.UTF8String : "?");
        impl_->specialized[key] = nil;
        return nil;
    }
    id<MTLComputePipelineState> state =
        [impl_->device newComputePipelineStateWithFunction:function
                                                     error:&error];
    if (!state)
        std::fprintf(stderr,
                     "MetalBackend: fa2 specialized pipeline failed: %s\n",
                     error ? error.localizedDescription.UTF8String : "?");
    impl_->specialized[key] = state;
    return state;
}

id<MTLComputePipelineState> MetalPipelineCache::gemv2(int rows_per_group) {
    char key_buffer[48];
    std::snprintf(key_buffer, sizeof(key_buffer), "gemv2:nr0%d",
                  rows_per_group);
    const std::string key(key_buffer);
    auto found = impl_->specialized.find(key);
    if (found != impl_->specialized.end()) return found->second;

    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&rows_per_group
                           type:MTLDataTypeInt atIndex:5];
    NSError* error = nil;
    id<MTLFunction> function =
        [impl_->library newFunctionWithName:@"gemv2_f32a_f16b_f32c"
                            constantValues:constants error:&error];
    id<MTLComputePipelineState> state = function
        ? [impl_->device newComputePipelineStateWithFunction:function
                                                       error:&error]
        : nil;
    if (!state)
        std::fprintf(stderr,
                     "MetalBackend: gemv2 nr0=%d pipeline failed: %s\n",
                     rows_per_group,
                     error ? error.localizedDescription.UTF8String : "?");
    impl_->specialized[key] = state;
    return state;
}

id<MTLComputePipelineState> MetalPipelineCache::gemv_w4(
        int rows_per_group) {
    char key_buffer[48];
    std::snprintf(key_buffer, sizeof(key_buffer), "gemv_w4:nr0%d",
                  rows_per_group);
    const std::string key(key_buffer);
    auto found = impl_->specialized.find(key);
    if (found != impl_->specialized.end()) return found->second;

    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&rows_per_group
                           type:MTLDataTypeInt atIndex:6];
    NSError* error = nil;
    id<MTLFunction> function =
        [impl_->library newFunctionWithName:@"gemv_w4_f32a_i4b_f32c"
                            constantValues:constants error:&error];
    id<MTLComputePipelineState> state = function
        ? [impl_->device newComputePipelineStateWithFunction:function
                                                       error:&error]
        : nil;
    if (!state)
        std::fprintf(stderr,
                     "MetalBackend: W4 GEMV nr0=%d pipeline failed: %s\n",
                     rows_per_group,
                     error ? error.localizedDescription.UTF8String : "?");
    impl_->specialized[key] = state;
    return state;
}

id<MTLComputePipelineState> MetalPipelineCache::small_m(
        const char* function_name, int m) {
    const std::string key = std::string("small_m:") + function_name +
        ":m" + std::to_string(m);
    auto found = impl_->specialized.find(key);
    if (found != impl_->specialized.end()) return found->second;

    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&m type:MTLDataTypeInt atIndex:12];
    NSError* error = nil;
    id<MTLFunction> function =
        [impl_->library
            newFunctionWithName:[NSString stringWithUTF8String:function_name]
                  constantValues:constants
                           error:&error];
    id<MTLComputePipelineState> state = function
        ? [impl_->device newComputePipelineStateWithFunction:function
                                                       error:&error]
        : nil;
    if (!state)
        std::fprintf(stderr, "MetalBackend: %s M=%d pipeline failed: %s\n",
                     function_name, m,
                     error ? error.localizedDescription.UTF8String : "?");
    impl_->specialized[key] = state;
    return state;
}

id<MTLComputePipelineState> MetalPipelineCache::w4a16(
        bool use_m128, bool specialize_g128) {
    const char* function_name = use_m128
        ? "gemm_tensor_w4_f32a_i4b_f32c"
        : "gemm_tensor_w4_f32a_i4b_f32c_m64";
    const std::string key = std::string("w4a16:") +
        (specialize_g128 ? "g128:" : "generic:") +
        (use_m128 ? "m128" : "m64");
    auto found = impl_->specialized.find(key);
    if (found != impl_->specialized.end()) return found->second;

    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    bool enabled = specialize_g128;
    [constants setConstantValue:&enabled type:MTLDataTypeBool atIndex:10];
    NSError* error = nil;
    id<MTLFunction> function =
        [impl_->library
            newFunctionWithName:[NSString stringWithUTF8String:function_name]
                  constantValues:constants
                           error:&error];
    id<MTLComputePipelineState> state = function
        ? [impl_->device newComputePipelineStateWithFunction:function
                                                       error:&error]
        : nil;
    if (!state)
        std::fprintf(stderr,
                     "MetalBackend: W4A16 G128 pipeline failed: %s\n",
                     error ? error.localizedDescription.UTF8String : "?");
    impl_->specialized[key] = state;
    return state;
}

id<MTLComputePipelineState> MetalPipelineCache::moe_select_parallel(
        bool sigmoid, bool grouped) {
    const std::string key = std::string("moe_select_parallel:") +
        (sigmoid ? "sigmoid" : "softmax") +
        (grouped ? ":grouped" : ":ungrouped");
    auto found = impl_->specialized.find(key);
    if (found != impl_->specialized.end()) return found->second;

    MTLFunctionConstantValues* constants =
        [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&sigmoid type:MTLDataTypeBool atIndex:7];
    [constants setConstantValue:&grouped type:MTLDataTypeBool atIndex:8];
    NSError* error = nil;
    id<MTLFunction> function =
        [impl_->library newFunctionWithName:@"moe_select_parallel"
                            constantValues:constants error:&error];
    id<MTLComputePipelineState> state = function
        ? [impl_->device newComputePipelineStateWithFunction:function
                                                       error:&error]
        : nil;
    if (!state)
        std::fprintf(stderr,
                     "MetalBackend: parallel %s MoE selector pipeline "
                     "failed: %s\n",
                     grouped ? "grouped sigmoid"
                             : sigmoid ? "sigmoid" : "softmax",
                     error ? error.localizedDescription.UTF8String : "?");
    impl_->specialized[key] = state;
    return state;
}

id<MTLComputePipelineState> MetalPipelineCache::grouped_moe(
        int group_size, bool paired_gate_up, bool large_route_tile) {
    const char* layout = group_size == 32 ? "bg32" : "bg128";
    std::string name = std::string("gemm_grouped_experts_") + layout;
    if (paired_gate_up)
        name += large_route_tile ? "_gate_up_r32" : "_gate_up_r16";
    else
        name += large_route_tile ? "_down_r32" : "_down_r16";
    return pipeline(name.c_str());
}
