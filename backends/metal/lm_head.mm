#include "backends/metal/lm_head.h"

#include "backends/metal/buffer_pool.h"
#include "backends/metal/command_context.h"
#include "backends/metal/dispatch_tuning.h"
#include "backends/metal/pipeline_cache.h"
#include "kernels/metal/metal_common.h"
#include "core/tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

id<MTLBuffer> buffer_of(const Tensor& tensor) {
    return tensor.device.buffer ? (__bridge id<MTLBuffer>)tensor.device.buffer
                              : nil;
}

id<MTLBuffer> scales_buffer_of(const Tensor& tensor) {
    return tensor.device.scales_buffer
               ? (__bridge id<MTLBuffer>)tensor.device.scales_buffer
               : nil;
}

} // namespace

MetalLmHead::MetalLmHead(MetalBufferPool* pool, MetalCommandContext* commands,
                         MetalPipelineCache* pipelines, bool& dispatch_failed)
    : pool_(pool), commands_(commands), pipelines_(pipelines),
      dispatch_failed_(dispatch_failed) {}

void MetalLmHead::gemv(const float* a_host, const Tensor& weight,
                       float* out_host, int N, int K, int activation) {
    @autoreleasepool {
        // Standalone path used by prefill/raw-logit callers.
        dispatch_failed_ = false;
        void* abuf = pool_->acquire((size_t)K * 4);
        std::memcpy(MetalBufferPool::contents(abuf), a_host, (size_t)K * 4);
        gemv_impl(abuf, 0, weight, out_host, N, K, activation, false);
        pool_->release(abuf, (size_t)K * 4);
    }
}

bool MetalLmHead::small_batch(const float* a_host, const Tensor& weight,
                              float* out_host, int M, int N, int K,
                              int activation) {
    if (!a_host)
        return false;
    @autoreleasepool {
        const size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
        void* abuf_handle = pool_->acquire(a_bytes);
        if (!abuf_handle)
            return false;
        std::memcpy(MetalBufferPool::contents(abuf_handle), a_host, a_bytes);
        const bool ok = small_batch_impl(abuf_handle, 0, weight, out_host, M, N,
                                         K, activation, false);
        pool_->release(abuf_handle, a_bytes);
        return ok;
    }
}

bool MetalLmHead::small_batch_device_and_end_graph(const Tensor& a,
                                                   const Tensor& weight,
                                                   float* out_host, int M,
                                                   int N, int K,
                                                   int activation) {
    return small_batch_impl(a.device.buffer, a.device.offset, weight, out_host, M,
                            N, K, activation, true);
}

bool MetalLmHead::small_batch_argmax_device_and_end_graph(const Tensor& a,
                                                          const Tensor& weight,
                                                          int* top1_out, int M,
                                                          int N, int K,
                                                          int activation) {
    return small_batch_impl(a.device.buffer, a.device.offset, weight, nullptr, M,
                            N, K, activation, true, top1_out);
}

bool MetalLmHead::small_batch_impl(void* a_device, size_t a_byte_offset,
                                   const Tensor& weight, float* out_host, int M,
                                   int N, int K, int activation,
                                   bool finish_open_graph, int* top1_out) {
    if (!a_device || (!out_host && !top1_out) || M < 2 || M > 4 || N <= 0 ||
        K <= 0 ||
        (weight.prec != Precision::INT4 && weight.prec != Precision::INT8) ||
        !weight.device.buffer || weight.group_size == 0 ||
        weight.groups_per_row == 0 ||
        (weight.prec == Precision::INT4 && (K & 1) != 0) ||
        (weight.prec == Precision::INT8 && !weight.device.scales_buffer)) {
        if (finish_open_graph && commands_->cmd)
            commands_->end_graph(dispatch_failed_);
        return false;
    }

    @autoreleasepool {
        dispatch_failed_ = false;
        const size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);
        void* cbuf_handle = pool_->acquire(c_bytes);
        if (!cbuf_handle) {
            if (finish_open_graph && commands_->cmd)
                commands_->end_graph(dispatch_failed_);
            return false;
        }
        constexpr uint kArgMaxThreads = 256;
        const uint argmax_groups =
            top1_out
                ? std::min<uint>(kArgMaxThreads,
                                 (static_cast<uint>(N) + kArgMaxThreads - 1) /
                                     kArgMaxThreads)
                : 0;
        const size_t partial_bytes =
            static_cast<size_t>(argmax_groups) * sizeof(ArgMaxPair);
        const size_t result_bytes = static_cast<size_t>(M) * sizeof(uint);
        void* partial_handle =
            top1_out ? pool_->acquire(partial_bytes) : nullptr;
        void* result_handle = top1_out ? pool_->acquire(result_bytes) : nullptr;
        if (top1_out && (!partial_handle || !result_handle)) {
            if (result_handle)
                pool_->release(result_handle, result_bytes);
            if (partial_handle)
                pool_->release(partial_handle, partial_bytes);
            pool_->release(cbuf_handle, c_bytes);
            if (finish_open_graph && commands_->cmd)
                commands_->end_graph(dispatch_failed_);
            return false;
        }
        if (top1_out)
            std::fill(top1_out, top1_out + M, -1);
        id<MTLBuffer> abuf = (__bridge id<MTLBuffer>)a_device;
        id<MTLBuffer> cbuf = (__bridge id<MTLBuffer>)cbuf_handle;

        MatmulW8Params p{};
        p.M = M;
        p.N = N;
        p.K = K;
        p.a_offset = 0;
        p.c_offset = 0;
        p.a_row_stride = K;
        p.c_row_stride = N;
        p.activation = activation;
        p.act_n_begin = 0;
        p.act_n_len = -1;
        p.group_size = static_cast<int>(weight.group_size);
        p.groups_per_row = static_cast<int>(weight.groups_per_row);

        id<MTLCommandBuffer> command = finish_open_graph
                                           ? commands_->cmd
                                           : [commands_->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            finish_open_graph ? commands_->enc
                              : [command computeCommandEncoder];
        if (!command || !encoder) {
            pool_->release(cbuf_handle, c_bytes);
            if (finish_open_graph && commands_->cmd)
                commands_->end_graph(dispatch_failed_);
            return false;
        }
        if (!finish_open_graph)
            command.label = @"mollm small-M lm_head";
        const bool is_w8 = weight.prec == Precision::INT8;
        const int nsg =
            is_w8
                ? std::min(mollm::metal::gemv_nsg_cap(), (K + 127) / 128)
                : std::min(mollm::metal::gemv_w4_nsg_cap(), (K / 2 + 63) / 64);
        [encoder
            setComputePipelineState:pipelines_->small_m(
                                        is_w8 ? "gemv_w8_small_m_f32a_i8b_f32c"
                                              : "gemv_w4_small_m_f32a_i4b_f32c",
                                        M)];
        [encoder setBuffer:abuf offset:a_byte_offset atIndex:0];
        [encoder setBuffer:buffer_of(weight)
                    offset:weight.device.offset
                   atIndex:1];
        [encoder setBuffer:cbuf offset:0 atIndex:2];
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        if (is_w8) {
            [encoder setBuffer:scales_buffer_of(weight)
                        offset:weight.device.scales_offset
                       atIndex:4];
        } else {
            [encoder setBuffer:buffer_of(weight)
                        offset:static_cast<size_t>(N) * (K / 2)
                       atIndex:4];
        }
        const NSUInteger groups = (static_cast<NSUInteger>(N) + nsg - 1) / nsg;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
        if (top1_out) {
            ArgMaxParams rp{};
            rp.count = static_cast<uint>(N);
            rp.group_count = argmax_groups;
            id<MTLBuffer> partial = (__bridge id<MTLBuffer>)partial_handle;
            id<MTLBuffer> result = (__bridge id<MTLBuffer>)result_handle;
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            for (int m = 0; m < M; ++m) {
                [encoder setComputePipelineState:pipelines_->pipeline(
                                                     "argmax_f32_stage1")];
                [encoder setBuffer:cbuf
                            offset:static_cast<size_t>(m) * N * sizeof(float)
                           atIndex:0];
                [encoder setBuffer:partial offset:0 atIndex:1];
                [encoder setBytes:&rp length:sizeof(rp) atIndex:2];
                [encoder
                     dispatchThreadgroups:MTLSizeMake(argmax_groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kArgMaxThreads, 1, 1)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [encoder setComputePipelineState:pipelines_->pipeline(
                                                     "argmax_f32_stage2")];
                [encoder setBuffer:partial offset:0 atIndex:0];
                [encoder setBuffer:result
                            offset:static_cast<size_t>(m) * sizeof(uint)
                           atIndex:1];
                [encoder setBytes:&rp length:sizeof(rp) atIndex:2];
                [encoder
                     dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kArgMaxThreads, 1, 1)];
                if (m + 1 < M)
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        }
        if (finish_open_graph) {
            commands_->end_graph(dispatch_failed_);
        } else {
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
        }
        if (command.status == MTLCommandBufferStatusError) {
            NSError* error = command.error;
            fprintf(stderr,
                    "MetalBackend: small-M lm_head command failed: %s\n",
                    error ? error.localizedDescription.UTF8String : "?");
            dispatch_failed_ = true;
        } else {
            if (commands_->profile) {
                const double gpu_ms =
                    (command.GPUEndTime - command.GPUStartTime) * 1000.0;
                const char* quant = is_w8 ? "W8" : "W4";
                auto& stat =
                    commands_->op_stats[std::string(
                                            top1_out ? "LM_HEAD_SMALL_M_ARGMAX_"
                                                     : "LM_HEAD_SMALL_M_") +
                                        quant + "[M=" + std::to_string(M) +
                                        ",N=" + std::to_string(N) +
                                        ",K=" + std::to_string(K) + "]"];
                stat.gpu_ms += gpu_ms;
                ++stat.calls;
            }
            if (top1_out) {
                const auto* result = static_cast<const uint*>(
                    MetalBufferPool::contents(result_handle));
                for (int m = 0; m < M; ++m) {
                    if (result[m] < static_cast<uint>(N))
                        top1_out[m] = static_cast<int>(result[m]);
                }
            } else {
                std::memcpy(out_host, MetalBufferPool::contents(cbuf_handle),
                            c_bytes);
            }
        }
        if (result_handle)
            pool_->release(result_handle, result_bytes);
        if (partial_handle)
            pool_->release(partial_handle, partial_bytes);
        pool_->release(cbuf_handle, c_bytes);
        return !dispatch_failed_;
    }
}

void MetalLmHead::gemv_device_and_end_graph(const Tensor& a,
                                            size_t a_element_offset,
                                            const Tensor& weight,
                                            float* out_host, int N, int K,
                                            int activation) {
    gemv_impl(a.device.buffer, a.device.offset + a_element_offset * sizeof(float),
              weight, out_host, N, K, activation, true);
}

int MetalLmHead::argmax_device_and_end_graph(const Tensor& a,
                                             size_t a_element_offset,
                                             const Tensor& weight, int N, int K,
                                             int activation,
                                             Tensor* hidden_copy) {
    int token = -1;
    gemv_impl(a.device.buffer, a.device.offset + a_element_offset * sizeof(float),
              weight, nullptr, N, K, activation, true, &token, hidden_copy);
    return token;
}

void MetalLmHead::gemv_impl(void* a_device, size_t a_byte_offset,
                            const Tensor& weight, float* out_host, int N, int K,
                            int activation, bool finish_open_graph,
                            int* top1_out, Tensor* hidden_copy) {
    @autoreleasepool {
        if (top1_out)
            *top1_out = -1;
        void* cbuf = pool_->acquire((size_t)N * 4);
        constexpr uint kArgMaxThreads = 256;
        const uint argmax_groups =
            top1_out
                ? std::min<uint>(kArgMaxThreads,
                                 (static_cast<uint>(N) + kArgMaxThreads - 1) /
                                     kArgMaxThreads)
                : 0;
        const size_t partial_bytes =
            static_cast<size_t>(argmax_groups) * sizeof(ArgMaxPair);
        void* partial_handle =
            top1_out ? pool_->acquire(partial_bytes) : nullptr;
        void* result_handle = top1_out ? pool_->acquire(sizeof(uint)) : nullptr;
        const bool reduce_top1 =
            top1_out && partial_handle && result_handle && argmax_groups > 0;
        MatmulParams p{};
        p.M = 1;
        p.N = N;
        p.K = K;
        p.a_offset = 0;
        p.b_offset = 0; // bind B at its byte offset below (64-bit, no overflow)
        p.c_offset = 0;
        p.a_row_stride = K;
        p.b_row_stride = (int)weight.shape[1]; // K
        p.c_row_stride = N;
        p.activation = activation;

        id<MTLBuffer> A = (__bridge id<MTLBuffer>)a_device;
        id<MTLBuffer> B = buffer_of(weight);
        id<MTLBuffer> C = (__bridge id<MTLBuffer>)cbuf;

        id<MTLCommandBuffer> cmd = finish_open_graph
                                       ? commands_->cmd
                                       : [commands_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc =
            finish_open_graph ? commands_->enc : [cmd computeCommandEncoder];
        assert(cmd && enc);
        [enc setBuffer:A offset:a_byte_offset atIndex:0];
        [enc setBuffer:B offset:weight.device.offset atIndex:1];
        [enc setBuffer:C offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        id<MTLComputePipelineState> ps = nil;
        if (weight.prec == Precision::INT8) {
            MatmulW8Params w{};
            w.M = 1;
            w.N = N;
            w.K = K;
            w.a_offset = 0;
            w.c_offset = 0;
            w.a_row_stride = K;
            w.c_row_stride = N;
            w.activation = activation;
            w.group_size = (int)weight.group_size;
            w.groups_per_row = (int)weight.groups_per_row;
            const size_t scales_boff = weight.device.scales_offset;
            constexpr int NR0 = 2;
            const int NSG =
                std::min(mollm::metal::gemv_nsg_cap(), (K + 127) / 128);
            ps = pipelines_->pipeline("gemv_w8_f32a_i8b_f32c");
            [enc setComputePipelineState:ps];
            [enc setBuffer:scales_buffer_of(weight)
                    offset:scales_boff
                   atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            const NSUInteger tgcount = ((NSUInteger)N + NR0 - 1) / NR0;
            [enc setThreadgroupMemoryLength:NR0 * 32 * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(tgcount, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)NSG, 1)];
        } else if (weight.prec == Precision::INT4) {
            MatmulW8Params w{};
            w.M = 1;
            w.N = N;
            w.K = K;
            w.a_offset = 0;
            w.c_offset = 0;
            w.a_row_stride = K;
            w.c_row_stride = N;
            w.activation = activation;
            w.group_size = (int)weight.group_size;
            w.groups_per_row = (int)weight.groups_per_row;
            const size_t scales_boff = (size_t)N * (K / 2);
            const int NR0 = mollm::metal::gemv_w4_nr0(N, K);
            const int NSG =
                std::min(mollm::metal::gemv_w4_nsg_cap(), (K / 2 + 63) / 64);
            ps = pipelines_->gemv_w4(NR0);
            [enc setComputePipelineState:ps];
            [enc setBuffer:B offset:scales_boff atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            [enc setThreadgroupMemoryLength:(NSUInteger)(NR0 * 32 *
                                                         sizeof(float))
                                    atIndex:0];
            const NSUInteger rows_per_tg =
                (NSUInteger)NR0 * (NSUInteger)std::max(1, NSG);
            const NSUInteger tgcount =
                ((NSUInteger)N + rows_per_tg - 1) / rows_per_tg;
            [enc dispatchThreadgroups:MTLSizeMake(tgcount, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(
                                          32 * (NSUInteger)std::max(1, NSG), 1,
                                          1)];
        } else {
            // FP16 uses tuned gemv2 (NR0=2 + NSG-split K).
            constexpr int NR0 = 2;
            const int NSG =
                std::min(mollm::metal::gemv_nsg_cap(), (K + 127) / 128);
            ps = pipelines_->gemv2(NR0);
            if (ps) {
                [enc setComputePipelineState:ps];
                [enc setThreadgroupMemoryLength:NR0 * 32 * sizeof(float)
                                        atIndex:0];
                const NSUInteger tgcount = ((NSUInteger)N + NR0 - 1) / NR0;
                [enc dispatchThreadgroups:MTLSizeMake(tgcount, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)NSG, 1)];
            } else {
                ps = pipelines_->pipeline("gemv_f32a_f16b_f32c");
                const NSUInteger rows_per_tg = 8;
                [enc setComputePipelineState:ps];
                NSUInteger tgcount =
                    ((NSUInteger)N + rows_per_tg - 1) / rows_per_tg;
                [enc dispatchThreadgroups:MTLSizeMake(tgcount, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(rows_per_tg * 32, 1, 1)];
            }
        }
        if (reduce_top1) {
            ArgMaxParams rp{};
            rp.count = static_cast<uint>(N);
            rp.group_count = argmax_groups;
            id<MTLBuffer> partial = (__bridge id<MTLBuffer>)partial_handle;
            id<MTLBuffer> result = (__bridge id<MTLBuffer>)result_handle;
            [enc setComputePipelineState:pipelines_->pipeline(
                                             "argmax_f32_stage1")];
            [enc setBuffer:C offset:0 atIndex:0];
            [enc setBuffer:partial offset:0 atIndex:1];
            [enc setBytes:&rp length:sizeof(rp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(argmax_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kArgMaxThreads, 1, 1)];
            [enc setComputePipelineState:pipelines_->pipeline(
                                             "argmax_f32_stage2")];
            [enc setBuffer:partial offset:0 atIndex:0];
            [enc setBuffer:result offset:0 atIndex:1];
            [enc setBytes:&rp length:sizeof(rp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kArgMaxThreads, 1, 1)];
        }
        if (finish_open_graph && hidden_copy && hidden_copy->device.buffer &&
            hidden_copy->nbytes() >= static_cast<size_t>(K) * sizeof(float)) {
            // Preserve the recursively predicted hidden state on GPU for the
            // next MTP depth.  The next graph consumes this buffer before its
            // own tail overwrites it, so one persistent ping buffer is enough.
            [enc endEncoding];
            commands_->enc = nil;
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:A
                     sourceOffset:a_byte_offset
                         toBuffer:(__bridge id<MTLBuffer>)
                                      hidden_copy->device.buffer
                destinationOffset:hidden_copy->device.offset
                             size:static_cast<size_t>(K) * sizeof(float)];
            [blit endEncoding];
        }
        if (finish_open_graph) {
            // end_graph() commits this tail after previously submitted chunks;
            // one queue preserves ordering, and its single wait covers both the
            // graph and lm_head.
            commands_->end_graph(dispatch_failed_);
        } else {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                NSError* e = cmd.error;
                fprintf(stderr,
                        "MetalBackend: lm_head command buffer error: %s\n",
                        e ? e.localizedDescription.UTF8String : "?");
                dispatch_failed_ = true;
            }
        }

        if (!dispatch_failed_) {
            if (commands_->profile) {
                const double gpu_ms =
                    (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0;
                const char* quant = weight.prec == Precision::INT8   ? "W8"
                                    : weight.prec == Precision::INT4 ? "W4"
                                                                     : "FP16";
                auto& stat =
                    commands_
                        ->op_stats[std::string(top1_out ? "MTP_LM_HEAD_ARGMAX_"
                                                        : "LM_HEAD_GEMV_") +
                                   quant + "[N=" + std::to_string(N) +
                                   ",K=" + std::to_string(K) + "]"];
                stat.gpu_ms += gpu_ms;
                ++stat.calls;
            }
            if (reduce_top1) {
                const uint token = *static_cast<const uint*>(
                    MetalBufferPool::contents(result_handle));
                if (token < static_cast<uint>(N))
                    *top1_out = static_cast<int>(token);
            } else if (out_host) {
                std::memcpy(out_host, MetalBufferPool::contents(cbuf),
                            (size_t)N * 4);
            }
        }
        if (result_handle)
            pool_->release(result_handle, sizeof(uint));
        if (partial_handle)
            pool_->release(partial_handle, partial_bytes);
        pool_->release(cbuf, (size_t)N * 4);
    }
}
