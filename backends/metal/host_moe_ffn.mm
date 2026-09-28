#include "backends/metal/host_moe_ffn.h"

#include "backends/cpu/moe_ops.h"
#include "kernels/cpu/matmul/gguf_iq_tables.h"
#include "storage/ssd_expert_cache/cache.h"
#include "runtime/threading.h"
#include "runtime/trace.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <limits>

#ifndef MOLLM_METALLIB_PATH
#define MOLLM_METALLIB_PATH ""
#endif

namespace {
struct MetalRegistration final : MoeSsdBufferRegistration {
    id<MTLDevice> device = nil;
    id<MTLBuffer> buffer = nil;
    MetalRegistration(id<MTLDevice> owner, id<MTLBuffer> value)
        : device(owner), buffer(value) {}
};
}  // namespace

struct MetalHostMoeFfn::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> gemv = nil, swiglu = nil;
    id<MTLArgumentEncoder> arguments = nil;
    id<MTLBuffer> input = nil, gate = nil, intermediate = nil, output = nil;
    id<MTLBuffer> iq2_grid = nil, iq3_grid = nil, signs = nil;
    std::string error;
    uint64_t completed = 0;
};

MetalHostMoeFfn::MetalHostMoeFfn() : impl_(new Impl) {
    @autoreleasepool {
        auto& p = *impl_;
        p.device = MTLCreateSystemDefaultDevice();
        if (!p.device) { p.error = "no Metal device"; return; }
        NSError* error = nil;
        id<MTLLibrary> library = [p.device
            newLibraryWithURL:[NSURL fileURLWithPath:@MOLLM_METALLIB_PATH]
                         error:&error];
        id<MTLFunction> gemv = [library newFunctionWithName:@"iq_gemv"];
        id<MTLFunction> swiglu = [library newFunctionWithName:@"swiglu"];
        if (gemv) {
            p.gemv = [p.device newComputePipelineStateWithFunction:gemv error:&error];
            p.arguments = [gemv newArgumentEncoderWithBufferIndex:0];
        }
        if (swiglu)
            p.swiglu = [p.device newComputePipelineStateWithFunction:swiglu error:&error];
        if (!p.gemv || !p.swiglu || !p.arguments ||
            p.gemv.threadExecutionWidth != 32) {
            p.error = error ? error.description.UTF8String
                            : "IQ Metal FFN pipelines unavailable";
            return;
        }
        p.queue = [p.device newCommandQueue];
        p.iq2_grid = [p.device newBufferWithBytes:iq2_grid length:sizeof(iq2_grid)
                                         options:MTLResourceStorageModeShared];
        p.iq3_grid = [p.device newBufferWithBytes:iq3_grid length:sizeof(iq3_grid)
                                         options:MTLResourceStorageModeShared];
        p.signs = [p.device newBufferWithBytes:iq_signs length:sizeof(iq_signs)
                                      options:MTLResourceStorageModeShared];
        if (!p.queue || !p.iq2_grid || !p.iq3_grid || !p.signs)
            p.error = "cannot allocate IQ Metal FFN resources";
    }
}

MetalHostMoeFfn::~MetalHostMoeFfn() = default;

bool MetalHostMoeFfn::available(std::string& reason) const {
    reason = impl_->error;
    return reason.empty();
}

uint64_t MetalHostMoeFfn::completed_host_moe_batches() const {
    return impl_->completed;
}

void MetalHostMoeFfn::dispatch(const GraphNode& node,
                              const std::vector<const Tensor*>& inputs,
                              Tensor* output, ThreadPool* pool) {
    dispatch_cpu_moe(node, inputs, output, pool, this);
}

bool MetalHostMoeFfn::dispatch_host_moe(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* pool, bool& success) {
    success = dispatch_cpu_moe(node, inputs, output, pool, this);
    return true;
}

bool MetalHostMoeFfn::run_ready_batch(
        ExpertProvider& provider, const std::vector<ExpertLease>& leases,
        const float* x, int hidden, int intermediate, float limit,
        float* result, ThreadPool* pool) {
    constexpr int H = 6144, I = 2048, B = 8;
    if (leases.size() != B || !x || !result || hidden != H ||
        intermediate != I || limit != 10.0f || !impl_->error.empty())
        return false;
    auto* cache = dynamic_cast<MoeSsdCache*>(&provider);
    if (!cache) return false;
    mollm_trace::ScopedEvent batch_trace("compute", "moe.metal_iq_ffn_batch");
    auto& p = *impl_;
    for (const auto& lease : leases) {
        const auto& g = lease.gate_up;
        const auto& d = lease.down;
        if (!lease || g.prec != Precision::GGUF_IQ2_XXS ||
            d.prec != Precision::GGUF_IQ3_XXS ||
            g.shape[0] != 2 * I || g.shape[1] != H ||
            d.shape[0] != H || d.shape[1] != I ||
            !g.data || !d.data || !g.is_contiguous() || !d.is_contiguous())
            return false;
    }
    auto register_allocation = [&](void* address, size_t bytes)
            -> std::unique_ptr<MoeSsdBufferRegistration> {
        if (!address || !bytes || bytes > p.device.maxBufferLength)
            return nullptr;
        id<MTLBuffer> buffer = [p.device newBufferWithBytesNoCopy:address
            length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        if (!buffer || buffer.contents != address) return nullptr;
        return std::make_unique<MetalRegistration>(p.device, buffer);
    };
    MoeSsdRegisteredPair views[B];
    id<MTLBuffer> gate_weights[B], down_weights[B];
    constexpr size_t gate_bytes = size_t(2 * I) * (H / 256) * 66;
    constexpr size_t down_bytes = size_t(H) * (I / 256) * 98;
    {
        mollm_trace::ScopedEvent registration_trace(
            "compute", "moe.metal_iq_ffn_registration");
        for (int e = 0; e < B; ++e) {
            if (!cache->get_or_register_borrowed_buffers(
                    leases[e], register_allocation, views[e]))
                return false;
            const auto& g = views[e].gate_up;
            const auto& d = views[e].down;
            if (g.payload_offset > g.active_bytes ||
                gate_bytes > g.active_bytes - g.payload_offset ||
                d.payload_offset > d.active_bytes ||
                down_bytes > d.active_bytes - d.payload_offset)
                return false;
            auto* gate_registration =
                dynamic_cast<MetalRegistration*>(g.registration);
            auto* down_registration =
                dynamic_cast<MetalRegistration*>(d.registration);
            if (!gate_registration || !down_registration ||
                gate_registration->device != p.device ||
                down_registration->device != p.device)
                return false;
            gate_weights[e] = gate_registration->buffer;
            down_weights[e] = down_registration->buffer;
            if (static_cast<const uint8_t*>(gate_weights[e].contents) +
                    g.payload_offset != leases[e].gate_up.data ||
                static_cast<const uint8_t*>(down_weights[e].contents) +
                    d.payload_offset != leases[e].down.data)
                return false;
        }
    }
    @autoreleasepool {
        auto ensure = [&](id<MTLBuffer> __strong& buffer, size_t bytes) {
            if (!buffer || buffer.length < bytes)
                buffer = [p.device newBufferWithLength:bytes
                    options:MTLResourceStorageModeShared];
            return buffer != nil;
        };
        if (!ensure(p.input, size_t(B * H) * sizeof(float)) ||
            !ensure(p.gate, size_t(B * (2 * I + 16)) * sizeof(float)) ||
            !ensure(p.intermediate, size_t(B * I) * sizeof(float)) ||
            !ensure(p.output, size_t(B * (H + 16)) * sizeof(float)))
            return false;
        auto* input = static_cast<float*>(p.input.contents);
        for (int e = 0; e < B; ++e)
            std::memcpy(input + size_t(e) * H, x, size_t(H) * sizeof(float));
        id<MTLBuffer> gate_args = [p.device newBufferWithLength:p.arguments.encodedLength
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> down_args = [p.device newBufferWithLength:p.arguments.encodedLength
            options:MTLResourceStorageModeShared];
        if (!gate_args || !down_args) return false;
        [p.arguments setArgumentBuffer:gate_args offset:0];
        for (int e = 0; e < B; ++e)
            [p.arguments setBuffer:gate_weights[e]
                offset:views[e].gate_up.payload_offset atIndex:e];
        [p.arguments setArgumentBuffer:down_args offset:0];
        for (int e = 0; e < B; ++e)
            [p.arguments setBuffer:down_weights[e]
                offset:views[e].down.payload_offset atIndex:e];
        id<MTLCommandBuffer> command = [p.queue commandBuffer];
        if (!command) return false;
        auto encode_gemv = [&](id<MTLBuffer> args, id<MTLBuffer> activation,
                               id<MTLBuffer> output, id<MTLBuffer> grid,
                               id<MTLBuffer> __strong* weights, uint32_t k,
                               uint32_t n, uint32_t prec) {
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) return false;
            [encoder setComputePipelineState:p.gemv];
            [encoder setBuffer:args offset:0 atIndex:0];
            for (int e = 0; e < B; ++e)
                [encoder useResource:weights[e] usage:MTLResourceUsageRead];
            [encoder setBuffer:activation offset:0 atIndex:1];
            [encoder setBuffer:output offset:0 atIndex:2];
            uint32_t shape[] = {k, n, n, prec};
            [encoder setBytes:shape length:sizeof(shape) atIndex:3];
            [encoder setBuffer:grid offset:0 atIndex:4];
            [encoder setBuffer:p.signs offset:0 atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake((n + 127) / 128, B, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
            return true;
        };
        if (!encode_gemv(gate_args, p.input, p.gate, p.iq2_grid,
                         gate_weights, H, 2 * I, 13))
            return false;
        id<MTLComputeCommandEncoder> activation = [command computeCommandEncoder];
        if (!activation) return false;
        [activation setComputePipelineState:p.swiglu];
        [activation setBuffer:p.gate offset:0 atIndex:0];
        [activation setBuffer:p.intermediate offset:0 atIndex:1];
        uint32_t width = I;
        [activation setBytes:&width length:sizeof(width) atIndex:2];
        [activation dispatchThreadgroups:MTLSizeMake((I + 127) / 128, B, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [activation endEncoding];
        if (!encode_gemv(down_args, p.intermediate, p.output, p.iq3_grid,
                         down_weights, I, H, 14))
            return false;
        const uint64_t wait_start = mollm_trace::now_ns();
        if (pool) pool->park();
        [command commit];
        [command waitUntilCompleted];
        if (pool) pool->resume();
        if (mollm_trace::enabled()) {
            const double gpu_ms =
                (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            mollm_trace::record_duration(
                "compute", "moe.metal_iq_ffn_wait", wait_start,
                mollm_trace::now_ns(),
                "{\"gpu_ms\":" + std::to_string(gpu_ms) + "}");
        }
        if (command.status != MTLCommandBufferStatusCompleted)
            return false;
        const auto* output = static_cast<const float*>(p.output.contents);
        for (int e = 0; e < B; ++e)
            std::memcpy(result + size_t(e) * H,
                        output + size_t(e) * (H + 16), size_t(H) * sizeof(float));
        ++p.completed;
    }
    return true;
}
