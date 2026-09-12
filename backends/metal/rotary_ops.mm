#include "backends/metal/rotary_ops.h"

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

void dispatch_grid_1d(id<MTLComputeCommandEncoder> encoder, int elements) {
    constexpr NSUInteger threads = 256;
    const MTLSize group_size = MTLSizeMake(threads, 1, 1);
    const MTLSize group_count = MTLSizeMake(
        (static_cast<NSUInteger>(elements) + threads - 1) / threads, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
}

}  // namespace

MetalRotaryOps::MetalRotaryOps(MetalPipelineCache* pipelines)
    : pipelines_(pipelines) {}

bool MetalRotaryOps::dispatch(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, id<MTLComputeCommandEncoder> encoder) const {
    const OpParams& params = node.params;
    id<MTLComputeCommandEncoder> enc = encoder;
    auto grid1d = [&](int elements) {
        dispatch_grid_1d(encoder, elements);
    };

    switch (node.op_type) {
    case OpType::ROTARY_EMBED: {
        Tensor& X = *output;                 // rope is in-place on the copied input
        const Tensor& in = *inputs[0];
        // Ensure output holds the input data (rope mutates in place). If output
        // is a fresh buffer we must copy input first via contiguous.
        // For phase-1 the graph feeds a CONTIGUOUS output into ROPE; treat rope
        // as reading inputs[0] and writing output, same layout.
        const Tensor& COS = *inputs[1];
        const Tensor& SIN = *inputs[2];
        RopeParams p{};
        p.head_dim = (int)in.shape[0];
        int rope_dim = params.i32.size()>0 ? params.i32[0] : p.head_dim;
        p.rope_dim = rope_dim;
        p.seq_len = (int)in.shape[1];
        p.heads   = (int)in.shape[2];
        p.interleave = params.i32.size()>1 ? params.i32[1] : 1;
        p.x_offset = element_offset(X);
        p.cos_offset = element_offset(COS);
        p.sin_offset = element_offset(SIN);
        // RoPE operates on X. When `in` is a strided view, the copy below
        // materializes it into dense X, so carrying the input strides into the
        // in-place kernel would skip rows and eventually access past X.
        p.x_stride_pos = element_stride(X, 1);
        p.x_stride_head = element_stride(X, 2);
        // Copy input -> output buffer (rope in place), if different buffers.
        if (buffer_of(&in) != buffer_of(&X) || in.device.offset != X.device.offset) {
            // use blit copy via contiguous kernel (contiguous input assumed)
            TensorDesc d{};
            for(int i=0;i<4;i++){d.shape[i]=(int)in.shape[i]; d.stride[i]=element_stride(in,i);}
            d.offset=element_offset(in);
            id<MTLComputePipelineState> cps = pipelines_->pipeline("contiguous_f32");
            [enc setComputePipelineState:cps];
            [enc setBuffer:buffer_of(&in) offset:0 atIndex:0];
            [enc setBuffer:buffer_of(&X) offset:0 atIndex:2];
            [enc setBytes:&d length:sizeof(d) atIndex:3];
            grid1d((int)X.nelements());
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        id<MTLComputePipelineState> ps = pipelines_->pipeline("rope_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&COS) offset:0 atIndex:1];
        [enc setBuffer:buffer_of(&SIN) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        // 3-D grid over (pair, position, head) via bounds-checked threadgroups.
        NSUInteger tx=8, ty=8, tz=4;
        MTLSize tgs = MTLSizeMake(tx,ty,tz);
        MTLSize tgc = MTLSizeMake(((NSUInteger)(rope_dim/2)+tx-1)/tx,
                                  ((NSUInteger)p.seq_len+ty-1)/ty,
                                  ((NSUInteger)p.heads+tz-1)/tz);
        [enc dispatchThreadgroups:tgc threadsPerThreadgroup:tgs];
        return true;
    }

    default:
        return false;
    }
}
