#include "backends/metal/normalization_ops.h"

#include "backends/metal/pipeline_cache.h"
#include "kernels/metal/metal_common.h"

#include <algorithm>

namespace {

id<MTLBuffer> buffer_of(const Tensor* tensor) {
    return tensor && tensor->device.buffer
        ? (__bridge id<MTLBuffer>)tensor->device.buffer
        : nil;
}

int element_stride(const Tensor& tensor, int dimension) {
    return static_cast<int>(tensor.stride[dimension] / tensor.element_size());
}

uint element_offset(const Tensor& tensor) {
    return static_cast<uint>(tensor.device.offset / tensor.element_size());
}

NSUInteger reduction_threads(id<MTLComputePipelineState> pipeline) {
    return std::min<NSUInteger>(
        256, pipeline.maxTotalThreadsPerThreadgroup);
}

}  // namespace

MetalNormalizationOps::MetalNormalizationOps(MetalPipelineCache* pipelines)
    : pipelines_(pipelines) {}

bool MetalNormalizationOps::dispatch(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, id<MTLComputeCommandEncoder> encoder) const {
    const OpParams& op_params = node.params;
    switch (node.op_type) {
    case OpType::RMS_NORM: {
        const Tensor& input = *inputs[0];
        const Tensor& weight = *inputs[1];
        RmsNormParams params{};
        params.dim0 = static_cast<int>(input.shape[0]);
        params.rows = static_cast<int>(
            input.shape[1] * input.shape[2] * input.shape[3]);
        params.x_offset = element_offset(input);
        // Bind package weights using the full 64-bit byte offset. Encoding it
        // as the shader's uint element offset wraps beyond 16 GiB.
        params.w_offset = 0;
        params.out_offset = element_offset(*output);
        params.x_row_stride = element_stride(input, 1);
        params.out_row_stride = element_stride(*output, 1);
        params.eps = op_params.f32.empty() ? 1e-6f : op_params.f32[0];
        id<MTLComputePipelineState> pipeline =
            pipelines_->pipeline("rms_norm_f32");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_of(&input) offset:0 atIndex:0];
        [encoder setBuffer:buffer_of(&weight)
                offset:weight.device.offset atIndex:1];
        [encoder setBuffer:buffer_of(output) offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        [encoder dispatchThreadgroups:
                    MTLSizeMake(static_cast<NSUInteger>(params.rows), 1, 1)
                threadsPerThreadgroup:
                    MTLSizeMake(reduction_threads(pipeline), 1, 1)];
        return true;
    }

    case OpType::RMS_NORM_ROPE: {
        const Tensor& input = *inputs[0];
        const Tensor& weight = *inputs[1];
        const Tensor& cos = *inputs[2];
        const Tensor& sin = *inputs[3];
        RmsNormRopeParams params{};
        params.dim0 = static_cast<int>(output->shape[0]);
        params.seq_len = static_cast<int>(output->shape[1]);
        params.heads = static_cast<int>(output->shape[2]);
        params.rows = params.seq_len * params.heads;
        params.rope_dim = op_params.i32.empty()
            ? params.dim0
            : op_params.i32[0];
        params.interleave = op_params.i32.size() > 1
            ? op_params.i32[1]
            : 1;
        params.x_offset = element_offset(input);
        params.w_offset = 0;
        params.cos_offset = element_offset(cos);
        params.sin_offset = element_offset(sin);
        params.out_offset = element_offset(*output);
        params.x_row_stride = element_stride(input, 1);
        params.out_row_stride = element_stride(*output, 1);
        params.eps = op_params.f32.empty() ? 1e-6f : op_params.f32[0];
        id<MTLComputePipelineState> pipeline =
            pipelines_->pipeline("rms_norm_rope_f32");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_of(&input) offset:0 atIndex:0];
        [encoder setBuffer:buffer_of(&weight)
                offset:weight.device.offset atIndex:1];
        [encoder setBuffer:buffer_of(output) offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        [encoder setBuffer:buffer_of(&cos) offset:0 atIndex:4];
        [encoder setBuffer:buffer_of(&sin) offset:0 atIndex:5];
        const NSUInteger rope_threads =
            params.seq_len > 1 && params.dim0 == 128 &&
                    params.rope_dim == 128
                ? 32
                : static_cast<NSUInteger>(
                      std::max(32, (params.rope_dim + 1) / 2));
        NSUInteger threads = std::min<NSUInteger>(
            256, ((rope_threads + 31) / 32) * 32);
        threads = std::min(
            threads, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreadgroups:
                    MTLSizeMake(static_cast<NSUInteger>(params.rows), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        return true;
    }

    case OpType::QK_RMS_NORM_ROPE: {
        const Tensor& query = *inputs[0];
        const Tensor& key = *inputs[1];
        const Tensor& query_weight = *inputs[2];
        const Tensor& key_weight = *inputs[3];
        const Tensor& cos = *inputs[4];
        const Tensor& sin = *inputs[5];
        QkRmsNormRopeParams params{};
        params.dim0 = static_cast<int>(output->shape[0]);
        params.seq_len = static_cast<int>(output->shape[1]);
        params.query_heads = op_params.i32.size() > 2
            ? op_params.i32[2]
            : static_cast<int>(output->shape[2]);
        params.rows = params.seq_len * static_cast<int>(output->shape[2]);
        params.rope_dim = op_params.i32.empty()
            ? params.dim0
            : op_params.i32[0];
        params.interleave = op_params.i32.size() > 1
            ? op_params.i32[1]
            : 1;
        params.query_x_offset = element_offset(query);
        params.key_x_offset = element_offset(key);
        params.query_w_offset = 0;
        params.key_w_offset = 0;
        params.cos_offset = element_offset(cos);
        params.sin_offset = element_offset(sin);
        params.out_offset = element_offset(*output);
        params.query_x_row_stride = element_stride(query, 1);
        params.key_x_row_stride = element_stride(key, 1);
        params.out_row_stride = element_stride(*output, 1);
        params.eps = op_params.f32.empty() ? 1e-6f : op_params.f32[0];
        id<MTLComputePipelineState> pipeline =
            pipelines_->pipeline("qk_rms_norm_rope_f32");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_of(&query) offset:0 atIndex:0];
        [encoder setBuffer:buffer_of(&key) offset:0 atIndex:1];
        [encoder setBuffer:buffer_of(&query_weight)
                offset:query_weight.device.offset atIndex:2];
        [encoder setBuffer:buffer_of(&key_weight)
                offset:key_weight.device.offset atIndex:3];
        [encoder setBuffer:buffer_of(output) offset:0 atIndex:4];
        [encoder setBytes:&params length:sizeof(params) atIndex:5];
        [encoder setBuffer:buffer_of(&cos) offset:0 atIndex:6];
        [encoder setBuffer:buffer_of(&sin) offset:0 atIndex:7];
        const NSUInteger rope_threads = static_cast<NSUInteger>(
            std::max(32, (params.rope_dim + 1) / 2));
        NSUInteger threads = std::min<NSUInteger>(
            256, ((rope_threads + 31) / 32) * 32);
        threads = std::min(
            threads, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreadgroups:
                    MTLSizeMake(static_cast<NSUInteger>(params.rows), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        return true;
    }

    case OpType::ADD_RMS_NORM: {
        Tensor& residual = *const_cast<Tensor*>(inputs[0]);
        const Tensor& update = *inputs[1];
        const Tensor& weight = *inputs[2];
        AddRmsNormParams params{};
        params.dim0 = static_cast<int>(residual.shape[0]);
        params.rows = static_cast<int>(
            residual.shape[1] * residual.shape[2] * residual.shape[3]);
        params.residual_offset = element_offset(residual);
        params.update_offset = element_offset(update);
        params.out_offset = element_offset(*output);
        params.residual_row_stride = element_stride(residual, 1);
        params.update_row_stride = element_stride(update, 1);
        params.out_row_stride = element_stride(*output, 1);
        params.eps = op_params.f32.empty() ? 1e-6f : op_params.f32[0];
        id<MTLComputePipelineState> pipeline =
            pipelines_->pipeline("add_rms_norm_f32");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_of(&residual) offset:0 atIndex:0];
        [encoder setBuffer:buffer_of(&update) offset:0 atIndex:1];
        [encoder setBuffer:buffer_of(output) offset:0 atIndex:2];
        [encoder setBuffer:buffer_of(&weight)
                offset:weight.device.offset atIndex:4];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        [encoder dispatchThreadgroups:
                    MTLSizeMake(static_cast<NSUInteger>(params.rows), 1, 1)
                threadsPerThreadgroup:
                    MTLSizeMake(reduction_threads(pipeline), 1, 1)];
        return true;
    }

    case OpType::LAYER_NORM: {
        const Tensor& input = *inputs[0];
        const Tensor& weight = *inputs[1];
        const Tensor& bias = *inputs[2];
        LayerNormParams params{};
        params.dim0 = static_cast<int>(input.shape[0]);
        params.rows = static_cast<int>(
            input.shape[1] * input.shape[2] * input.shape[3]);
        params.x_offset = element_offset(input);
        params.out_offset = element_offset(*output);
        params.x_row_stride = element_stride(input, 1);
        params.out_row_stride = element_stride(*output, 1);
        params.eps = op_params.f32.empty() ? 1e-5f : op_params.f32[0];
        id<MTLComputePipelineState> pipeline =
            pipelines_->pipeline("layer_norm_f32");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_of(&input) offset:0 atIndex:0];
        [encoder setBuffer:buffer_of(&weight)
                offset:weight.device.offset atIndex:1];
        [encoder setBuffer:buffer_of(output) offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        [encoder setBuffer:buffer_of(&bias)
                offset:bias.device.offset atIndex:4];
        [encoder dispatchThreadgroups:
                    MTLSizeMake(static_cast<NSUInteger>(params.rows), 1, 1)
                threadsPerThreadgroup:
                    MTLSizeMake(reduction_threads(pipeline), 1, 1)];
        return true;
    }

    default:
        return false;
    }
}
