#include "backends/metal/layout_ops.h"

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

void dispatch_1d(id<MTLComputeCommandEncoder> encoder, int elements) {
    constexpr NSUInteger threads = 256;
    const MTLSize group_size = MTLSizeMake(threads, 1, 1);
    const MTLSize group_count = MTLSizeMake(
        (static_cast<NSUInteger>(elements) + threads - 1) / threads, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
}

}  // namespace

MetalLayoutOps::MetalLayoutOps(MetalPipelineCache* pipelines)
    : pipelines_(pipelines) {}

bool MetalLayoutOps::dispatch(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, id<MTLComputeCommandEncoder> encoder,
        bool& encoded_gpu_work) const {
    const OpParams& params = node.params;
    switch (node.op_type) {
    case OpType::INPUT:
    case OpType::CONSTANT:
        encoded_gpu_work = false;
        return true;

    case OpType::RESHAPE: {
        const Tensor& source = *inputs[0];
        if (source.is_contiguous()) {
            encoded_gpu_work = false;
            const int64_t shape[4] = {
                output->shape[0], output->shape[1],
                output->shape[2], output->shape[3]};
            *output = source;
            for (int dimension = 0; dimension < 4; ++dimension)
                output->shape[dimension] = shape[dimension];
            output->compute_strides();
        } else {
            TensorDesc descriptor{};
            for (int dimension = 0; dimension < 4; ++dimension) {
                descriptor.shape[dimension] =
                    static_cast<int>(source.shape[dimension]);
                descriptor.stride[dimension] =
                    element_stride(source, dimension);
            }
            descriptor.offset = element_offset(source);
            id<MTLComputePipelineState> pipeline =
                pipelines_->pipeline("contiguous_f32");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buffer_of(&source) offset:0 atIndex:0];
            [encoder setBuffer:buffer_of(output) offset:0 atIndex:2];
            [encoder setBytes:&descriptor length:sizeof(descriptor) atIndex:3];
            dispatch_1d(encoder, static_cast<int>(output->nelements()));
        }
        return true;
    }

    case OpType::PERMUTE: {
        encoded_gpu_work = false;
        const Tensor& source = *inputs[0];
        const int axes[4] = {
            params.i32.size() > 0 ? params.i32[0] : 0,
            params.i32.size() > 1 ? params.i32[1] : 1,
            params.i32.size() > 2 ? params.i32[2] : 2,
            params.i32.size() > 3 ? params.i32[3] : 3};
        Tensor view = source;
        int64_t shape[4];
        size_t stride[4];
        for (int source_axis = 0; source_axis < 4; ++source_axis) {
            shape[axes[source_axis]] = source.shape[source_axis];
            stride[axes[source_axis]] = source.stride[source_axis];
        }
        for (int dimension = 0; dimension < 4; ++dimension) {
            view.shape[dimension] = shape[dimension];
            view.stride[dimension] = stride[dimension];
        }
        *output = view;
        return true;
    }

    case OpType::SLICE: {
        encoded_gpu_work = false;
        const Tensor& source = *inputs[0];
        const int dimension = params.i32.size() > 0 ? params.i32[0] : 0;
        const int offset = params.i32.size() > 1 ? params.i32[1] : 0;
        const int size = params.i32.size() > 2
            ? params.i32[2]
            : static_cast<int>(source.shape[dimension]);
        *output = source;
        output->device.offset = source.device.offset +
            static_cast<size_t>(offset) * source.stride[dimension];
        output->shape[dimension] = size;
        return true;
    }

    case OpType::CONTIGUOUS: {
        const Tensor& source = *inputs[0];
        TensorDesc descriptor{};
        for (int dimension = 0; dimension < 4; ++dimension) {
            descriptor.shape[dimension] =
                static_cast<int>(source.shape[dimension]);
            descriptor.stride[dimension] =
                element_stride(source, dimension);
        }
        descriptor.offset = element_offset(source);
        [encoder setBuffer:buffer_of(&source) offset:0 atIndex:0];
        [encoder setBuffer:buffer_of(output) offset:0 atIndex:2];
        [encoder setBytes:&descriptor length:sizeof(descriptor) atIndex:3];
        if (descriptor.shape[3] == 1) {
            id<MTLComputePipelineState> pipeline =
                pipelines_->pipeline("contiguous3d_f32");
            [encoder setComputePipelineState:pipeline];
            constexpr NSUInteger threads_x = 64;
            constexpr NSUInteger threads_y = 4;
            const MTLSize group_size =
                MTLSizeMake(threads_x, threads_y, 1);
            const MTLSize group_count = MTLSizeMake(
                (static_cast<NSUInteger>(descriptor.shape[0]) +
                 threads_x - 1) / threads_x,
                (static_cast<NSUInteger>(descriptor.shape[1]) +
                 threads_y - 1) / threads_y,
                static_cast<NSUInteger>(descriptor.shape[2]));
            [encoder dispatchThreadgroups:group_count
                    threadsPerThreadgroup:group_size];
        } else {
            id<MTLComputePipelineState> pipeline =
                pipelines_->pipeline("contiguous_f32");
            [encoder setComputePipelineState:pipeline];
            dispatch_1d(encoder, static_cast<int>(output->nelements()));
        }
        return true;
    }

    default:
        return false;
    }
}
