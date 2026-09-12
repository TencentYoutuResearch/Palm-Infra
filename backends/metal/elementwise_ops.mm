#include "backends/metal/elementwise_ops.h"

#include "backends/metal/pipeline_cache.h"
#include "kernels/metal/metal_common.h"

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

void dispatch_1d(id<MTLComputeCommandEncoder> encoder,
                 id<MTLComputePipelineState> pipeline, int elements) {
    NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256)
        threads = 256;
    const MTLSize group_size = MTLSizeMake(threads, 1, 1);
    const MTLSize group_count = MTLSizeMake(
        (static_cast<NSUInteger>(elements) + threads - 1) / threads, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
}

}  // namespace

MetalElementwiseOps::MetalElementwiseOps(MetalPipelineCache* pipelines)
    : pipelines_(pipelines) {}

bool MetalElementwiseOps::dispatch(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, id<MTLComputeCommandEncoder> encoder) const {
    switch (node.op_type) {
    case OpType::ADD:
    case OpType::MUL:
    case OpType::SIGMOID_MUL: {
        const Tensor& left = *inputs[0];
        const Tensor& right = *inputs[1];
        EwiseParams params{};
        params.n = static_cast<int>(output->nelements());
        params.broadcast_b = right.nelements() == 1 ? 1 : 0;
        params.shape0 = static_cast<int>(output->shape[0]);
        params.a_row_stride = element_stride(left, 1);
        params.b_row_stride = element_stride(right, 1);
        params.out_row_stride = element_stride(*output, 1);
        params.a_offset = element_offset(left);
        params.b_offset = element_offset(right);
        params.out_offset = element_offset(*output);
        for (int dimension = 0; dimension < 4; ++dimension) {
            params.shape[dimension] =
                static_cast<int>(output->shape[dimension]);
            params.a_stride[dimension] =
                left.shape[dimension] == 1 && output->shape[dimension] != 1
                    ? 0
                    : element_stride(left, dimension);
            params.b_stride[dimension] =
                right.shape[dimension] == 1 && output->shape[dimension] != 1
                    ? 0
                    : element_stride(right, dimension);
            params.out_stride[dimension] =
                element_stride(*output, dimension);
        }
        const char* kernel = node.op_type == OpType::ADD
            ? "add_f32"
            : node.op_type == OpType::MUL
                ? "mul_f32"
                : "sigmoid_mul_f32";
        id<MTLComputePipelineState> pipeline = pipelines_->pipeline(kernel);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_of(&left) offset:0 atIndex:0];
        [encoder setBuffer:buffer_of(&right) offset:0 atIndex:1];
        [encoder setBuffer:buffer_of(output) offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(encoder, pipeline, params.n);
        return true;
    }

    case OpType::SILU: {
        const Tensor& input = *inputs[0];
        EwiseParams params{};
        params.n = static_cast<int>(output->nelements());
        params.a_offset = element_offset(input);
        params.out_offset = element_offset(*output);
        id<MTLComputePipelineState> pipeline =
            pipelines_->pipeline("silu_f32");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_of(&input) offset:0 atIndex:0];
        [encoder setBuffer:buffer_of(output) offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(encoder, pipeline, params.n);
        return true;
    }

    case OpType::SIGMOID:
    case OpType::SIGMOID_EXACT:
    case OpType::GELU:
    case OpType::TANH:
    case OpType::EXP:
    case OpType::EXP_EXACT:
    case OpType::SOFTPLUS: {
        const Tensor& input = *inputs[0];
        EwiseParams params{};
        params.n = static_cast<int>(output->nelements());
        params.shape0 = static_cast<int>(output->shape[0]);
        params.a_row_stride = element_stride(input, 1);
        params.out_row_stride = element_stride(*output, 1);
        params.a_offset = element_offset(input);
        params.out_offset = element_offset(*output);
        const char* kernel = node.op_type == OpType::GELU
            ? "gelu_f32"
            : node.op_type == OpType::TANH
                ? "tanh_f32"
                : node.op_type == OpType::EXP ||
                        node.op_type == OpType::EXP_EXACT
                    ? "exp_f32"
                    : node.op_type == OpType::SOFTPLUS
                        ? "softplus_f32"
                        : "sigmoid_f32";
        id<MTLComputePipelineState> pipeline = pipelines_->pipeline(kernel);
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_of(&input) offset:0 atIndex:0];
        [encoder setBuffer:buffer_of(output) offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(encoder, pipeline, params.n);
        return true;
    }

    case OpType::SWIGLU: {
        const Tensor& merged = *inputs[0];
        SwigluParams params{};
        params.I = static_cast<int>(merged.shape[0]) / 2;
        params.n = static_cast<int>(output->nelements());
        params.merged_offset = element_offset(merged);
        params.out_offset = element_offset(*output);
        params.merged_row_stride = element_stride(merged, 1);
        id<MTLComputePipelineState> pipeline =
            pipelines_->pipeline("swiglu_f32");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_of(&merged) offset:0 atIndex:0];
        [encoder setBuffer:buffer_of(output) offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(encoder, pipeline, params.n);
        return true;
    }

    default:
        return false;
    }
}
