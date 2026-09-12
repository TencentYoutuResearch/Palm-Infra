#include "backends/metal/ssd_shared_expert.h"

#include "backends/metal/buffer_pool.h"
#include "backends/metal/pipeline_cache.h"
#include "kernels/metal/metal_common.h"
#include "runtime/trace.h"

#import <Metal/Metal.h>

#include <string>

struct MetalSsdSharedExpert::Impl {
    Impl(id<MTLDevice> device_value, MetalBufferPool* pool,
         MetalPipelineCache* pipelines)
        : device(device_value), pool_(pool), pipelines_(pipelines) {}

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> ssd_shared_compute_queue = nil;
    id<MTLSharedEvent> ssd_shared_compute_event = nil;
    uint64_t ssd_shared_compute_event_value = 0;
    MetalBufferPool* pool_ = nullptr;
    MetalPipelineCache* pipelines_ = nullptr;

    bool submit(
            const Tensor& x, const Tensor& gate, const Tensor& up,
            const Tensor& down, const Tensor& scale_weight, int hidden,
            int intermediate, int seq, int layer,
            MetalSsdSharedExpert::Work& work) {
        if (gate.prec != Precision::INT4 ||
            up.prec != Precision::INT4 ||
            down.prec != Precision::INT4 ||
            scale_weight.prec != Precision::FP16 ||
            !x.device.buffer || !gate.device.buffer || !up.device.buffer ||
            !down.device.buffer || !scale_weight.device.buffer) {
            return false;
        }

        work.qx_bytes = static_cast<size_t>(seq) * hidden;
        work.sx_bytes = static_cast<size_t>(seq) * sizeof(float);
        work.intermediate_bytes =
            static_cast<size_t>(intermediate) * sizeof(float);
        work.qintermediate_bytes =
            static_cast<size_t>(intermediate) * sizeof(int8_t);
        work.output_bytes = static_cast<size_t>(hidden) * sizeof(float);
        work.qx = pool_->acquire(work.qx_bytes);
        work.sx = pool_->acquire(work.sx_bytes);
        work.intermediate = pool_->acquire(work.intermediate_bytes);
        work.qintermediate = pool_->acquire(work.qintermediate_bytes);
        work.qintermediate_scale = pool_->acquire(sizeof(float));
        work.scale = pool_->acquire(sizeof(float));
        work.output = pool_->acquire(work.output_bytes);

        id<MTLBuffer> x_buffer = (__bridge id<MTLBuffer>)x.device.buffer;
        id<MTLBuffer> gate_buffer = (__bridge id<MTLBuffer>)gate.device.buffer;
        id<MTLBuffer> up_buffer = (__bridge id<MTLBuffer>)up.device.buffer;
        id<MTLBuffer> down_buffer = (__bridge id<MTLBuffer>)down.device.buffer;
        id<MTLBuffer> scale_weight_buffer =
            (__bridge id<MTLBuffer>)scale_weight.device.buffer;
        id<MTLBuffer> qx = (__bridge id<MTLBuffer>)work.qx;
        id<MTLBuffer> sx = (__bridge id<MTLBuffer>)work.sx;
        id<MTLBuffer> hidden_values =
            (__bridge id<MTLBuffer>)work.intermediate;
        id<MTLBuffer> qhidden =
            (__bridge id<MTLBuffer>)work.qintermediate;
        id<MTLBuffer> qhidden_scale =
            (__bridge id<MTLBuffer>)work.qintermediate_scale;
        id<MTLBuffer> scale = (__bridge id<MTLBuffer>)work.scale;
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)work.output;

        MoeSharedW4Params params{};
        params.hidden = hidden;
        params.intermediate = intermediate;
        params.gate_groups_per_row = static_cast<int>(gate.groups_per_row);
        params.up_groups_per_row = static_cast<int>(up.groups_per_row);
        params.down_groups_per_row = static_cast<int>(down.groups_per_row);
        params.hidden_offset =
            static_cast<uint>(x.device.offset / sizeof(float));

        id<MTLCommandBuffer> shared_cmd =
            [ssd_shared_compute_queue commandBuffer];
        shared_cmd.label = @"mollm shared SSD expert";
        id<MTLComputeCommandEncoder> shared_enc =
            [shared_cmd computeCommandEncoder];
        shared_enc.label = @"mollm shared expert";

        QuantActParams xq{};
        xq.M = seq;
        xq.K = hidden;
        xq.a_offset = params.hidden_offset;
        xq.a_row_stride =
            static_cast<int>(x.stride[1] / sizeof(float));
        [shared_enc setComputePipelineState:
                        pipelines_->pipeline("quantize_act_i8")];
        [shared_enc setBuffer:x_buffer offset:0 atIndex:0];
        [shared_enc setBuffer:qx offset:0 atIndex:2];
        [shared_enc setBytes:&xq length:sizeof(xq) atIndex:3];
        [shared_enc setBuffer:sx offset:0 atIndex:4];
        [shared_enc setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
        [shared_enc dispatchThreadgroups:MTLSizeMake(seq, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];

        [shared_enc setComputePipelineState:
                        pipelines_->pipeline("moe_shared_gate_up_w4_i8")];
        [shared_enc setBuffer:qx offset:0 atIndex:0];
        [shared_enc setBuffer:gate_buffer offset:gate.device.offset atIndex:1];
        [shared_enc setBuffer:hidden_values offset:0 atIndex:2];
        [shared_enc setBytes:&params length:sizeof(params) atIndex:3];
        [shared_enc
            setBuffer:gate_buffer
               offset:gate.device.offset +
                      static_cast<size_t>(intermediate) * hidden / 2
              atIndex:4];
        [shared_enc setBuffer:up_buffer offset:up.device.offset atIndex:5];
        [shared_enc
            setBuffer:up_buffer
               offset:up.device.offset +
                      static_cast<size_t>(intermediate) * hidden / 2
              atIndex:6];
        [shared_enc setBuffer:sx offset:0 atIndex:7];
        [shared_enc dispatchThreadgroups:MTLSizeMake(intermediate, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [shared_enc setComputePipelineState:
                        pipelines_->pipeline("moe_shared_scale_f16")];
        [shared_enc setBuffer:x_buffer offset:0 atIndex:0];
        [shared_enc setBuffer:scale_weight_buffer
                       offset:scale_weight.device.offset
                      atIndex:1];
        [shared_enc setBuffer:scale offset:0 atIndex:2];
        [shared_enc setBytes:&params length:sizeof(params) atIndex:3];
        [shared_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        QuantActParams iq{};
        iq.M = 1;
        iq.K = intermediate;
        iq.a_row_stride = intermediate;
        [shared_enc setComputePipelineState:
                        pipelines_->pipeline("quantize_act_i8")];
        [shared_enc setBuffer:hidden_values offset:0 atIndex:0];
        [shared_enc setBuffer:qhidden offset:0 atIndex:2];
        [shared_enc setBytes:&iq length:sizeof(iq) atIndex:3];
        [shared_enc setBuffer:qhidden_scale offset:0 atIndex:4];
        [shared_enc setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
        [shared_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];

        [shared_enc setComputePipelineState:
                        pipelines_->pipeline("moe_shared_down_w4_i8")];
        [shared_enc setBuffer:qhidden offset:0 atIndex:0];
        [shared_enc setBuffer:down_buffer offset:down.device.offset atIndex:1];
        [shared_enc setBuffer:output offset:0 atIndex:2];
        [shared_enc setBytes:&params length:sizeof(params) atIndex:3];
        [shared_enc
            setBuffer:down_buffer
               offset:down.device.offset +
                      static_cast<size_t>(hidden) * intermediate / 2
              atIndex:4];
        [shared_enc setBuffer:scale offset:0 atIndex:5];
        [shared_enc setBuffer:qhidden_scale offset:0 atIndex:6];
        [shared_enc dispatchThreadgroups:MTLSizeMake(hidden, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [shared_enc endEncoding];

        work.ready_value = ++ssd_shared_compute_event_value;
        [shared_cmd encodeSignalEvent:ssd_shared_compute_event
                                value:work.ready_value];
        const uint64_t start = mollm_trace::now_ns();
        [shared_cmd addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            const uint64_t end = mollm_trace::now_ns();
            const std::string args =
                "{\"layer\":" + std::to_string(layer) + "}";
            mollm_trace::record_duration(
                "metal.ssd", "shared_expert", start, end, args,
                "thread_state_running");
            const double gpu_seconds =
                completed.GPUEndTime - completed.GPUStartTime;
            if (gpu_seconds > 0.0 && end != 0) {
                const uint64_t gpu_ns =
                    static_cast<uint64_t>(gpu_seconds * 1e9);
                mollm_trace::record_duration(
                    "metal.ssd", "shared_expert_gpu",
                    end > gpu_ns ? end - gpu_ns : 0, end, args,
                    "thread_state_running");
            }
        }];
        [shared_cmd commit];
        return true;
    }
};

MetalSsdSharedExpert::MetalSsdSharedExpert(
        void* device, MetalBufferPool* pool, MetalPipelineCache* pipelines)
    : impl_(new Impl((__bridge id<MTLDevice>)device, pool, pipelines)) {}

MetalSsdSharedExpert::~MetalSsdSharedExpert() = default;

bool MetalSsdSharedExpert::configure() {
    if (!impl_) return false;
    impl_->ssd_shared_compute_queue = [impl_->device newCommandQueue];
    impl_->ssd_shared_compute_event = [impl_->device newSharedEvent];
    if (!impl_->ssd_shared_compute_queue ||
        !impl_->ssd_shared_compute_event) {
        impl_->ssd_shared_compute_queue = nil;
        impl_->ssd_shared_compute_event = nil;
        return false;
    }
    return true;
}

bool MetalSsdSharedExpert::submit(
        const Tensor& x, const Tensor& gate, const Tensor& up,
        const Tensor& down, const Tensor& scale_weight, int hidden,
        int intermediate, int seq, int layer, Work& work) {
    if (!impl_ || !impl_->ssd_shared_compute_queue ||
        !impl_->ssd_shared_compute_event) {
        return false;
    }
    return impl_->submit(x, gate, up, down, scale_weight, hidden,
                         intermediate, seq, layer, work);
}

id<MTLSharedEvent> MetalSsdSharedExpert::event() const {
    return impl_ ? impl_->ssd_shared_compute_event : nil;
}
