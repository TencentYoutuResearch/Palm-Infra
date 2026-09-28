#include "backends/metal/host_matmul.h"
#include "graph/graph.h"
#include "kernels/cpu/activations.h"
#include "runtime/threading.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <climits>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unistd.h>

#ifndef MOLLM_METALLIB_PATH
#define MOLLM_METALLIB_PATH ""
#endif

struct MetalHostMatmulBackend::Impl {
    struct Weight {
        id<MTLBuffer> buffer;
        size_t offset;
        int64_t k, n;
    };
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> input = nil, output = nil;
    std::unordered_map<const void*, Weight> weights;
    std::string error;
    uint64_t completed = 0;
};

MetalHostMatmulBackend::MetalHostMatmulBackend() : impl_(new Impl) {
    @autoreleasepool {
        auto& p = *impl_;
        p.device = MTLCreateSystemDefaultDevice();
        if (!p.device) { p.error = "no Metal device"; return; }
        NSError* error = nil;
        id<MTLLibrary> library = [p.device
            newLibraryWithURL:[NSURL fileURLWithPath:@MOLLM_METALLIB_PATH]
                         error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"q5_gemv"];
        if (function)
            p.pipeline = [p.device newComputePipelineStateWithFunction:function
                                                                     error:&error];
        if (!p.pipeline || p.pipeline.threadExecutionWidth != 32) {
            p.error = error ? error.description.UTF8String
                            : "Q5 Metal pipeline requires SIMD width 32";
            return;
        }
        p.queue = [p.device newCommandQueue];
        if (!p.queue) p.error = "cannot create Metal command queue";
    }
}
MetalHostMatmulBackend::~MetalHostMatmulBackend() = default;

uint64_t MetalHostMatmulBackend::completed_dispatches() const {
    return impl_->completed;
}

bool MetalHostMatmulBackend::available(std::string& reason) const {
    reason = impl_->error;
    return reason.empty() && impl_->pipeline && impl_->queue;
}

bool MetalHostMatmulBackend::register_weight(
        const Tensor& w, const void* mapping, size_t mapping_bytes) {
    auto& p = *impl_;
    if (!p.error.empty() || !p.queue || !mapping || !w.data ||
        w.prec != Precision::GGUF_Q5_K || w.prepared.weight || w.device.buffer ||
        w.shape[0] <= 0 || w.shape[0] > INT_MAX / 8 ||
        w.shape[1] <= 0 || w.shape[1] > INT_MAX || w.shape[1] % 256 ||
        w.shape[2] != 1 || w.shape[3] != 1 || !w.is_contiguous())
        return false;
    const size_t blocks = size_t(w.shape[1]) / 256;
    if (size_t(w.shape[0]) > SIZE_MAX / 176 / blocks) return false;
    const size_t bytes = size_t(w.shape[0]) * blocks * 176;
    const uintptr_t base = reinterpret_cast<uintptr_t>(mapping);
    const uintptr_t address = reinterpret_cast<uintptr_t>(w.data);
    if (address < base || address - base > mapping_bytes ||
        bytes > mapping_bytes - (address - base)) return false;
    const size_t page = size_t(getpagesize()), delta = address % page;
    if (address - base < delta || bytes > SIZE_MAX - delta - (page - 1))
        return false;
    const size_t length = (bytes + delta + page - 1) / page * page;
    if (length > mapping_bytes - (address - base - delta) ||
        length > p.device.maxBufferLength) return false;
    auto old = p.weights.find(w.data);
    if (old != p.weights.end())
        return old->second.k == w.shape[1] && old->second.n == w.shape[0];
    @autoreleasepool {
        id<MTLBuffer> buffer = [p.device
            newBufferWithBytesNoCopy:reinterpret_cast<void*>(address - delta)
            length:length options:MTLResourceStorageModeShared deallocator:nil];
        if (!buffer) return false;
        p.weights.emplace(w.data, Impl::Weight{buffer, delta, w.shape[1], w.shape[0]});
    }
    return true;
}

HostMatmulResult MetalHostMatmulBackend::dispatch(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* pool) {
    if (node.op_type != OpType::MATMUL || inputs.size() != 2 ||
        !inputs[0] || !inputs[1] || !output ||
        graph_params::get_i32(node.params, 0, 0) != int(Activation::NONE))
        return HostMatmulResult::Unsupported;
    const auto& a = *inputs[0]; const auto& w = *inputs[1]; auto& c = *output;
    auto& p = *impl_;
    auto it = p.weights.find(w.data);
    if (it == p.weights.end() || w.prec != Precision::GGUF_Q5_K ||
        w.prepared.weight || w.device.buffer || a.device.buffer || c.device.buffer ||
        !w.is_contiguous() ||
        a.prec != Precision::FP32 || c.prec != Precision::FP32 ||
        !a.data || !c.data || !a.is_contiguous() || !c.is_contiguous() ||
        a.shape[1] != 1 || c.shape[1] != 1 ||
        a.shape[2] != 1 || a.shape[3] != 1 || c.shape[2] != 1 || c.shape[3] != 1 ||
        w.shape[2] != 1 || w.shape[3] != 1 ||
        w.shape[0] != it->second.n || w.shape[1] != it->second.k ||
        a.shape[0] != w.shape[1] || c.shape[0] != w.shape[0])
        return HostMatmulResult::Unsupported;
    @autoreleasepool {
        const size_t input_bytes = size_t(a.shape[0]) * sizeof(float);
        const size_t output_bytes = size_t(c.shape[0]) * sizeof(float);
        if (!p.input || p.input.length < input_bytes)
            p.input = [p.device newBufferWithLength:input_bytes options:MTLResourceStorageModeShared];
        if (!p.output || p.output.length < output_bytes)
            p.output = [p.device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        if (!p.input || !p.output) return HostMatmulResult::Failed;
        std::memcpy(p.input.contents, a.data, input_bytes);
        id<MTLCommandBuffer> command = [p.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return HostMatmulResult::Failed;
        [encoder setComputePipelineState:p.pipeline];
        [encoder setBuffer:it->second.buffer offset:it->second.offset atIndex:0];
        [encoder setBuffer:p.input offset:0 atIndex:1];
        [encoder setBuffer:p.output offset:0 atIndex:2];
        uint32_t shape[] = {uint32_t(a.shape[0]), uint32_t(c.shape[0])};
        [encoder setBytes:shape length:sizeof(shape) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((size_t(c.shape[0])*8+127)/128,1,1)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [encoder endEncoding];
        if (pool) pool->park();
        [command commit];
        [command waitUntilCompleted];
        if (pool) pool->resume();
        if (command.status != MTLCommandBufferStatusCompleted)
            return HostMatmulResult::Failed;
        std::memcpy(c.data, p.output.contents, output_bytes);
        ++p.completed;
    }
    return HostMatmulResult::Completed;
}
