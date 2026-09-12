#include "backends/metal/moe_ops.h"

#include "backends/metal/buffer_pool.h"
#include "backends/metal/command_context.h"
#include "backends/metal/dispatch_tuning.h"
#include "backends/metal/pipeline_cache.h"
#include "backends/metal/ssd_expert_cache.h"
#include "backends/metal/ssd_shared_expert.h"
#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/moe/moe.h"
#include "kernels/cpu/moe/moe_routing.h"
#include "kernels/metal/metal_common.h"
#include "runtime/trace.h"
#include "storage/ssd_expert_cache/cache.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace {

id<MTLBuffer> buffer_of(const Tensor* tensor) {
    return tensor && tensor->device.buffer
        ? (__bridge id<MTLBuffer>)tensor->device.buffer
        : nil;
}

id<MTLBuffer> scales_buffer_of(const Tensor* tensor) {
    return tensor && tensor->device.scales_buffer
        ? (__bridge id<MTLBuffer>)tensor->device.scales_buffer
        : nil;
}

int element_stride(const Tensor& tensor, int dimension) {
    return static_cast<int>(tensor.stride[dimension] / tensor.element_size());
}

uint element_offset(const Tensor& tensor) {
    return static_cast<uint>(tensor.device.offset / tensor.element_size());
}

}  // namespace

MetalMoeOps::MetalMoeOps(
        void* device, MetalPipelineCache* pipelines, MetalBufferPool* pool,
        MetalCommandContext* commands, bool& dispatch_failed)
    : pipelines_(pipelines), pool_(pool), commands_(commands),
      dispatch_failed_(dispatch_failed),
      ssd_cache_(new MetalSsdExpertCache(device)),
      ssd_shared_expert_(new MetalSsdSharedExpert(device, pool, pipelines)) {}

MetalMoeOps::~MetalMoeOps() = default;

bool MetalMoeOps::configure_ssd_io(
        const std::string& package_path, size_t capacity_bytes,
        int max_commands_in_flight, bool cross_layer_prefetch) {
    if (!ssd_cache_ || package_path.empty() || capacity_bytes == 0)
        return false;
    if (!ssd_shared_expert_ || !ssd_shared_expert_->configure()) {
        fprintf(stderr, "MetalBackend: shared-expert command setup failed\n");
        return false;
    }
    if (!ssd_cache_->configure(
            package_path, capacity_bytes, max_commands_in_flight))
        return false;
    ssd_cross_layer_prefetch_ = cross_layer_prefetch;
    return true;
}

bool MetalMoeOps::finish_ssd_prefix(int layer, const char* error_context) {
    [commands_->enc endEncoding];
    commands_->enc = nil;
    const uint64_t wait_start = mollm_trace::now_ns();
    [commands_->cmd commit];
    [commands_->cmd waitUntilCompleted];
    const uint64_t wait_end = mollm_trace::now_ns();
    const std::string args =
        "{\"layer\":" + std::to_string(layer) + "}";
    mollm_trace::record_duration(
        "metal.ssd", "prefix_wait", wait_start, wait_end, args,
        "thread_state_iowait");
    const double gpu_seconds =
        commands_->cmd.GPUEndTime - commands_->cmd.GPUStartTime;
    if (gpu_seconds > 0.0 && wait_end != 0) {
        const uint64_t gpu_ns = static_cast<uint64_t>(gpu_seconds * 1e9);
        mollm_trace::record_duration(
            "metal.ssd", "prefix_gpu",
            wait_end > gpu_ns ? wait_end - gpu_ns : 0, wait_end, args,
            "thread_state_running");
    }
    if (commands_->cmd.status == MTLCommandBufferStatusError) {
        NSError* error = commands_->cmd.error;
        fprintf(stderr, "MetalBackend: %s failed: %s\n", error_context,
                error ? error.localizedDescription.UTF8String : "?");
        commands_->cmd = nil;
        return false;
    }
    commands_->cmd = nil;
    return true;
}

bool MetalMoeOps::route_moe_on_cpu(
        const Tensor& input, const Tensor& router, const Tensor* bias,
        const mollm::detail::MoeRoutingParams& routing, ThreadPool* thread_pool,
        void* indices_handle, void* weights_handle) {
    if (!router.data)
        return false;
    const int seq = static_cast<int>(input.shape[1]);
    std::vector<float> logits(static_cast<size_t>(seq) * routing.num_experts);
    Tensor output = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, routing.num_experts, seq, 1, 1,
        logits.data());
    kernel_matmul_fp32(input, router, output, thread_pool, Activation::NONE,
                      0, -1, true);
    std::vector<int> indices;
    std::vector<float> weights;
    const float* bias_data = bias && bias->data ? bias->ptr<float>() : nullptr;
    if (!mollm::detail::select_moe_routes(
            logits.data(), seq, bias_data, routing, indices, weights))
        return false;
    std::memcpy(MetalBufferPool::contents(indices_handle), indices.data(),
                indices.size() * sizeof(int));
    std::memcpy(MetalBufferPool::contents(weights_handle), weights.data(),
                weights.size() * sizeof(float));
    return true;
}

bool MetalMoeOps::dispatch(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, id<MTLComputeCommandEncoder> encoder,
        ThreadPool* thread_pool, bool has_tensor, std::string& profile_label,
        bool& abort_dispatch) {
    abort_dispatch = false;
    const OpParams& params = node.params;
    id<MTLComputeCommandEncoder> enc = encoder;
    auto grid1d = [&](int n) {
        constexpr NSUInteger threads = 256;
        [enc dispatchThreadgroups:
                 MTLSizeMake((static_cast<NSUInteger>(n) + threads - 1) /
                                 threads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    };

    switch (node.op_type) {
    case OpType::MOE: {
        int hidden_size = params.i32.size()>0 ? params.i32[0] : (int)output->shape[0];
        int num_experts = params.i32.size()>1 ? params.i32[1] : 0;
        int top_k = params.i32.size()>2 ? params.i32[2] : 0;
        int intermediate = params.i32.size()>3 ? params.i32[3] : 0;
        int shared_intermediate = params.i32.size()>4 ? params.i32[4] : intermediate;
        int router_score_func = params.i32.size()>5 ? params.i32[5] : 0;
        bool norm_topk = params.i32.size()>6 ? params.i32[6] != 0 : true;
        bool has_shared = params.i32.size()>7 ? params.i32[7] != 0 : true;
        int n_group = params.i32.size()>8 ? params.i32[8] : 1;
        int topk_group = params.i32.size()>9 ? params.i32[9] : 1;
        int router_bias_input =
            params.i32.size()>11 ? params.i32[11]
                                 : (has_shared ? 8 : -1);
        const Tensor* router_bias =
            router_bias_input >= 0 &&
                    static_cast<size_t>(router_bias_input) < inputs.size()
                ? inputs[router_bias_input]
                : nullptr;
        float routed_scale = params.f32.size()>0 ? params.f32[0] : 1.0f;
        const auto* ssd_gate = inputs.size() > 2
            ? dynamic_cast<const MoeSsdTensorSource*>(inputs[2]->moe_ssd_source)
            : nullptr;
        const auto* ssd_down = inputs.size() > 3
            ? dynamic_cast<const MoeSsdTensorSource*>(inputs[3]->moe_ssd_source)
            : nullptr;
        // Qwen-style W4 routed experts stay on the GPU. Resident package
        // weights use native BG128 blocks: short prefill uses independent
        // selected-route kernels, while long prefill groups routes by expert
        // to reuse each weight tile. SSD experts use the same block layout in
        // cache slots for decode.
        const bool ssd_w4 =
            ssd_gate && ssd_down && ssd_gate->cache == ssd_down->cache &&
            ssd_gate->spec.precision == Precision::INT4 &&
            ssd_down->spec.precision == Precision::INT4;
        if (ssd_w4 && inputs[0]->shape[1] == 1) {
            ssd_moe_layers_[ssd_gate->spec.layer] = {
                inputs[1],
                router_bias,
                ssd_gate,
                ssd_down,
                hidden_size,
                num_experts,
                top_k,
                intermediate,
                router_score_func,
                std::max(1, n_group),
                std::max(1, topk_group),
                norm_topk,
                routed_scale,
            };
        }
        const bool supported_router =
            router_score_func == 0 ||
            (router_score_func == 1 && router_bias);
        const int moe_seq = (int)inputs[0]->shape[1];
        const bool resident_w8 =
            !ssd_gate && !ssd_down && !has_shared &&
            inputs[2]->prec == Precision::INT8 &&
            inputs[3]->prec == Precision::INT8 &&
            inputs[2]->device.scales_buffer &&
            inputs[3]->device.scales_buffer;
        const bool resident_quant_prefill =
            moe_seq > 1 && !ssd_w4 && !has_shared;
        const bool gpu_quant_moe =
            has_tensor && supported_router &&
            inputs[1]->prec == Precision::FP16 &&
            ((inputs[2]->prec == Precision::INT4 &&
              inputs[3]->prec == Precision::INT4) || resident_w8) &&
            top_k <= 16 && n_group <= 16 &&
            (moe_seq == 1 || resident_quant_prefill) &&
            (ssd_w4 || !has_shared);
        if (gpu_quant_moe) {
            const Tensor& x = *inputs[0]; const Tensor& router = *inputs[1];
            const Tensor& gu = *inputs[2]; const Tensor& down = *inputs[3];
            const Tensor* bias = router_bias;
            int seq = (int)x.shape[1];
            if (commands_->profile)
                profile_label += "[S=" + std::to_string(seq) + "]";
            size_t idx_bytes=(size_t)seq*top_k*sizeof(int);
            size_t tw_bytes=(size_t)seq*top_k*sizeof(float);
            size_t logits_bytes=(size_t)seq*num_experts*sizeof(float);
            size_t merged_bytes=(size_t)seq*top_k*2*intermediate*sizeof(float);
            void* idx_h=pool_->acquire(idx_bytes), *tw_h=pool_->acquire(tw_bytes);
            void* logits_h=pool_->acquire(logits_bytes);
            void* merged_h=pool_->acquire(merged_bytes);
            id<MTLBuffer> idx=(__bridge id<MTLBuffer>)idx_h;
            id<MTLBuffer> tw=(__bridge id<MTLBuffer>)tw_h;
            id<MTLBuffer> logits=(__bridge id<MTLBuffer>)logits_h;
            id<MTLBuffer> merged=(__bridge id<MTLBuffer>)merged_h;
            auto profile_resident_moe_stage = [&](const char* label) {
                if (!commands_->profile || ssd_w4) return;
                if (commands_->enc) {
                    [commands_->enc endEncoding];
                    commands_->enc = nil;
                }
                if (commands_->cmd) {
                    [commands_->cmd commit];
                    [commands_->cmd waitUntilCompleted];
                    const double gpu_ms =
                        (commands_->cmd.GPUEndTime -
                         commands_->cmd.GPUStartTime) * 1000.0;
                    const std::string stage_label =
                        std::string(label) +
                        "[S=" + std::to_string(seq) + "]";
                    auto& stat = commands_->op_stats[stage_label];
                    stat.gpu_ms += gpu_ms;
                    stat.calls += 1;
                }
                commands_->cmd = [commands_->queue commandBuffer];
                commands_->enc = [commands_->cmd computeCommandEncoder];
                enc = commands_->enc;
            };
            const SsdMoeLayerInfo* predicted_layer = nullptr;
            void* predicted_idx_h = nullptr;
            void* predicted_tw_h = nullptr;
            size_t predicted_idx_bytes = 0;
            size_t predicted_tw_bytes = 0;
            MetalSsdSharedExpert::Work shared_work;
            if (ssd_w4 && ssd_cross_layer_prefetch_) {
                auto next = ssd_moe_layers_.find(
                    ssd_gate->spec.layer + 1);
                if (next != ssd_moe_layers_.end() &&
                    next->second.hidden == hidden_size &&
                    next->second.router &&
                    next->second.router->device.buffer &&
                    next->second.router->prec == Precision::FP16 &&
                    next->second.top_k > 0 &&
                    next->second.top_k <= 16 &&
                    (next->second.score_func == 0 ||
                     (next->second.score_func == 1 &&
                      next->second.bias &&
                      next->second.bias->device.buffer))) {
                    predicted_layer = &next->second;
                    predicted_idx_bytes =
                        (size_t)seq * predicted_layer->top_k * sizeof(int);
                    predicted_tw_bytes =
                        (size_t)seq * predicted_layer->top_k * sizeof(float);
                    predicted_idx_h =
                        pool_->acquire(predicted_idx_bytes);
                    predicted_tw_h =
                        pool_->acquire(predicted_tw_bytes);
                }
            }
            MoeW4Params mp{};
            mp.hidden=hidden_size;mp.experts=num_experts;mp.top_k=top_k;
            mp.intermediate=intermediate;mp.seq_len=seq;mp.n_group=std::max(1,n_group);
            mp.topk_group=std::max(1,topk_group);mp.norm_topk=norm_topk;
            mp.routed_scale=routed_scale;mp.hidden_offset=element_offset(x);mp.output_offset=element_offset(*output);
            mp.hidden_row_stride=element_stride(x,1);mp.output_row_stride=element_stride(*output,1);
            mp.gu_groups_per_row=(int)gu.groups_per_row;
            mp.down_groups_per_row=(int)down.groups_per_row;
            mp.gu_group_size=(int)gu.group_size;
            mp.down_group_size=(int)down.group_size;
            size_t gu_rows=(size_t)num_experts*2*intermediate;
            size_t down_rows=(size_t)num_experts*hidden_size;
            auto native_bg128 = [](const Tensor& w,
                                   int rows_per_expert) {
                return w.is_q4_g128_packed && w.q4_g128_data &&
                       w.group_size == 128 &&
                       rows_per_expert % 8 == 0;
            };
            auto native_bg32 = [](const Tensor& w,
                                  int rows_per_expert) {
                return w.is_q4_g32_packed && w.q4_g32_data &&
                       w.group_size == 32 &&
                       rows_per_expert % 8 == 0;
            };
            const bool native_gu =
                native_bg128(gu, 2 * intermediate);
            const int native_gu_group =
                native_gu
                    ? 128
                    : (native_bg32(gu, 2 * intermediate) ? 32 : 0);
            void* resident_qx_h = nullptr;
            void* resident_sx_h = nullptr;

            if (ssd_w4) {
                // Direct MTLIO submission needs host-visible route IDs. On
                // UMA, routing this tiny GEMV on the CPU avoids a separate GPU
                // router command and writes directly into the Shared buffers
                // consumed by the expert kernels.
                if (!finish_ssd_prefix(
                        ssd_gate->spec.layer, "pre-router command")) {
                    break;
                }
                enc = nil;

                const auto* input_bytes =
                    static_cast<const uint8_t*>([buffer_of(&x) contents]) +
                    x.device.offset;
                Tensor cpu_input = Tensor::create(
                    Precision::FP32, MemoryType::EXTERNAL,
                    hidden_size, seq, 1, 1,
                    const_cast<uint8_t*>(input_bytes));
                if (has_shared &&
                    !ssd_shared_expert_->submit(
                        x, *inputs[4], *inputs[5], *inputs[6], *inputs[7],
                        hidden_size, shared_intermediate, seq,
                        ssd_gate->spec.layer, shared_work)) {
                    fprintf(stderr,
                            "MetalBackend: unsupported shared expert "
                            "weight format in layer %d\n",
                            ssd_gate->spec.layer);
                    break;
                }
                const uint64_t cpu_router_start =
                    mollm_trace::now_ns();
                mollm::detail::MoeRoutingParams routing;
                routing.num_experts = num_experts;
                routing.top_k = top_k;
                routing.score_func = router_score_func;
                routing.normalize_topk = norm_topk;
                routing.num_groups = std::max(1, n_group);
                routing.topk_groups = std::max(1, topk_group);
                routing.scaling_factor = routed_scale;
                bool routed = route_moe_on_cpu(
                    cpu_input, router, bias, routing, thread_pool,
                    idx_h, tw_h);
                if (routed && predicted_layer) {
                    routing.num_experts = predicted_layer->experts;
                    routing.top_k = predicted_layer->top_k;
                    routing.score_func = predicted_layer->score_func;
                    routing.normalize_topk = predicted_layer->norm_topk;
                    routing.num_groups = predicted_layer->n_group;
                    routing.topk_groups = predicted_layer->topk_group;
                    routing.scaling_factor =
                        predicted_layer->routed_scale;
                    routed = route_moe_on_cpu(
                        cpu_input, *predicted_layer->router,
                        predicted_layer->bias, routing, thread_pool,
                        predicted_idx_h, predicted_tw_h);
                }
                if (!routed) {
                    fprintf(stderr,
                            "MetalBackend: CPU SSD router failed in layer %d\n",
                            ssd_gate->spec.layer);
                    break;
                }
                mollm_trace::record_duration(
                    "metal.ssd", "cpu_router",
                    cpu_router_start, mollm_trace::now_ns(),
                    "{\"layer\":" +
                        std::to_string(ssd_gate->spec.layer) + "}",
                    "thread_state_running");
            }

            if (!ssd_w4) {
                const int router_nsg =
                    std::min(
                        mollm::metal::gemv_nsg_cap(),
                        (hidden_size + 127) / 128);
                const bool fuse_router_quant =
                    native_gu && seq == 1 && router_nsg == 8;
                const bool fuse_router_quant_small =
                    native_gu && seq >= 2 && seq <= 4;
                if (fuse_router_quant || fuse_router_quant_small) {
                    const size_t qx_bytes =
                        (size_t)seq * hidden_size;
                    const size_t sx_bytes =
                        (size_t)seq * gu.groups_per_row *
                        sizeof(float);
                    resident_qx_h = pool_->acquire(qx_bytes);
                    resident_sx_h = pool_->acquire(sx_bytes);
                    [enc setComputePipelineState:
                             (fuse_router_quant_small
                                  ? pipelines_->small_m(
                                        "moe_router_quantize_bg128_small_m",
                                        seq)
                                  : pipelines_->pipeline(
                                        "moe_router_quantize_bg128"))];
                    [enc setBuffer:buffer_of(&x) offset:0 atIndex:0];
                    [enc setBuffer:buffer_of(&router)
                            offset:router.device.offset atIndex:1];
                    [enc setBuffer:logits offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc setBuffer:
                             (__bridge id<MTLBuffer>)resident_qx_h
                            offset:0 atIndex:4];
                    [enc setBuffer:
                             (__bridge id<MTLBuffer>)resident_sx_h
                            offset:0 atIndex:5];
                    [enc setThreadgroupMemoryLength:
                             (fuse_router_quant_small ? 16 : 2 * 32) *
                                 sizeof(float)
                            atIndex:0];
                    const NSUInteger router_groups =
                        fuse_router_quant_small
                            ? ((NSUInteger)num_experts + 15) / 16
                            : ((NSUInteger)num_experts + 1) / 2;
                    const NSUInteger quant_groups =
                        fuse_router_quant_small
                            ? ((NSUInteger)seq * gu.groups_per_row + 3) / 4
                            : ((NSUInteger)gu.groups_per_row + 1) / 2;
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 router_groups + quant_groups, 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(
                                 32,
                                 fuse_router_quant_small ? 16 : 8,
                                 1)];
                } else if (seq == 1) {
                    MatmulParams router_mp{};
                    router_mp.M = seq;
                    router_mp.N = num_experts;
                    router_mp.K = hidden_size;
                    router_mp.a_offset = element_offset(x);
                    router_mp.b_offset = 0;
                    router_mp.c_offset = 0;
                    router_mp.a_row_stride = element_stride(x, 1);
                    router_mp.b_row_stride = hidden_size;
                    router_mp.c_row_stride = num_experts;
                    router_mp.activation = 0;
                    router_mp.act_n_begin = 0;
                    router_mp.act_n_len = -1;
                    [enc setBuffer:buffer_of(&x) offset:0 atIndex:0];
                    [enc setBuffer:buffer_of(&router)
                            offset:router.device.offset atIndex:1];
                    [enc setBuffer:logits offset:0 atIndex:2];
                    [enc setBytes:&router_mp
                           length:sizeof(router_mp) atIndex:3];
                    constexpr int router_rows_per_tg = 2;
                    id<MTLComputePipelineState> router_ps =
                        pipelines_->gemv2(router_rows_per_tg);
                    [enc setComputePipelineState:router_ps];
                    [enc setThreadgroupMemoryLength:
                             router_rows_per_tg * 32 * sizeof(float)
                                            atIndex:0];
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (num_experts +
                                  router_rows_per_tg - 1) /
                                     router_rows_per_tg,
                                 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(
                                 32, std::max(1, router_nsg), 1)];
                } else if (seq <= 4) {
                    // Tiny speculative-verification batches are far below
                    // the tensor router's 64-row tile. Scan each FP16 router
                    // row once and reuse it across all token activations.
                    MatmulParams router_mp{};
                    router_mp.M = seq;
                    router_mp.N = num_experts;
                    router_mp.K = hidden_size;
                    router_mp.a_offset = element_offset(x);
                    router_mp.b_offset = 0;
                    router_mp.c_offset = 0;
                    router_mp.a_row_stride = element_stride(x, 1);
                    router_mp.b_row_stride = hidden_size;
                    router_mp.c_row_stride = num_experts;
                    router_mp.activation = 0;
                    router_mp.act_n_begin = 0;
                    router_mp.act_n_len = -1;
                    const int router_groups = std::min(
                        mollm::metal::gemv_nsg_cap(), (hidden_size + 127) / 128);
                    [enc setComputePipelineState:
                             pipelines_->small_m(
                                 "gemv_small_m_f32a_f16b_f32c", seq)];
                    [enc setBuffer:buffer_of(&x) offset:0 atIndex:0];
                    [enc setBuffer:buffer_of(&router)
                            offset:router.device.offset atIndex:1];
                    [enc setBuffer:logits offset:0 atIndex:2];
                    [enc setBytes:&router_mp
                           length:sizeof(router_mp) atIndex:3];
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 ((NSUInteger)num_experts +
                                  router_groups - 1) /
                                     router_groups,
                                 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(32 * router_groups, 1, 1)];
                } else {
                    // Multi-token resident MoE stays entirely on the GPU.
                    // The decode-specialized router GEMV only consumes the
                    // first activation row, so use the same tensor GEMM path
                    // as an ordinary FP16 projection for prefill.
                    MatmulParams router_mp{};
                    router_mp.M = seq;
                    router_mp.N = num_experts;
                    router_mp.K = hidden_size;
                    router_mp.a_offset = element_offset(x);
                    router_mp.b_offset = 0;
                    router_mp.c_offset = 0;
                    router_mp.a_row_stride = element_stride(x, 1);
                    router_mp.b_row_stride = hidden_size;
                    router_mp.c_row_stride = num_experts;
                    router_mp.activation = 0;
                    router_mp.act_n_begin = 0;
                    router_mp.act_n_len = -1;

                    const bool small_router =
                        num_experts <= 128;
                    MatmulParams tensor_mp = router_mp;
                    tensor_mp.a_offset = 0;
                    id<MTLBuffer> activation = nil;
                    NSUInteger activation_offset = 0;
                    if (small_router) {
                        // Router top-k is unusually sensitive to activation
                        // rounding. Consume the FP32 residual stream directly
                        // so near-tied experts do not diverge solely because
                        // Metal rounded the router input to FP16.
                        activation = buffer_of(&x);
                        activation_offset = x.device.offset;
                    } else {
                        const size_t activation_bytes =
                            (size_t)seq * (size_t)hidden_size *
                            sizeof(uint16_t);
                        void* activation_h =
                            pool_->acquire(activation_bytes);
                        activation =
                            (__bridge id<MTLBuffer>)activation_h;
                        commands_->pending_free.push_back(
                            {activation_h, activation_bytes});

                        [enc setComputePipelineState:
                                 pipelines_->pipeline(
                                     "matmul_cast_f32_to_f16")];
                        [enc setBuffer:buffer_of(&x) offset:0 atIndex:0];
                        [enc setBuffer:activation offset:0 atIndex:2];
                        [enc setBytes:&router_mp
                               length:sizeof(router_mp) atIndex:3];
                        grid1d(
                            (seq * hidden_size + 3) / 4);
                        [enc memoryBarrierWithScope:
                                 MTLBarrierScopeBuffers];
                        tensor_mp.a_row_stride = hidden_size;
                    }
                    const NSUInteger router_tile_m =
                        small_router ? 64 : 128;
                    const NSUInteger router_tile_n =
                        small_router ? 32 : 64;
                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 small_router
                                     ? "gemm_tensor_router_f32a_f16b_f32c"
                                     : "gemm_tensor_direct_f16a_f16b_f32c")];
                    [enc setBuffer:activation
                            offset:activation_offset atIndex:0];
                    [enc setBuffer:buffer_of(&router)
                            offset:router.device.offset atIndex:1];
                    [enc setBuffer:logits offset:0 atIndex:2];
                    [enc setBytes:&tensor_mp
                           length:sizeof(tensor_mp) atIndex:3];
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 ((NSUInteger)seq +
                                  router_tile_m - 1) /
                                     router_tile_m,
                                 ((NSUInteger)num_experts +
                                  router_tile_n - 1) /
                                     router_tile_n,
                                 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(128, 1, 1)];
                }
                profile_resident_moe_stage(
                    (fuse_router_quant || fuse_router_quant_small)
                        ? "MOE.router_quant"
                        : "MOE.router");
                const bool parallel_select =
                    num_experts <= 256 &&
                    top_k <= 16 &&
                    (router_score_func == 0 ||
                     (router_score_func == 1 &&
                      n_group <= 16 &&
                      topk_group <= n_group));
                id<MTLComputePipelineState> select_pipeline =
                    parallel_select
                        ? pipelines_->moe_select_parallel(
                              router_score_func == 1,
                              router_score_func == 1 &&
                                  n_group > 1)
                        : pipelines_->pipeline(
                              router_score_func == 0
                                  ? "moe_select_softmax"
                                  : "moe_select_sigmoid");
                [enc setComputePipelineState:select_pipeline];
                [enc setBuffer:logits offset:0 atIndex:0];
                [enc setBuffer:idx offset:0 atIndex:1];
                [enc setBuffer:tw offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                if (bias) {
                    [enc setBuffer:buffer_of(bias)
                            offset:bias->device.offset atIndex:4];
                } else if (parallel_select) {
                    // The softmax specialization does not read this binding,
                    // but Metal validation still requires every declared
                    // argument to be present.
                    [enc setBuffer:logits offset:0 atIndex:4];
                }
                if (parallel_select) {
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (NSUInteger)seq, 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(
                                 num_experts <= 128 ? 128 : 256,
                                 1, 1)];
                } else {
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 ((NSUInteger)seq + 63) / 64, 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(64, 1, 1)];
                }
                profile_resident_moe_stage("MOE.select");
            }

            // Resident W8 expert tensors stay in their package-native
            // row-major layout. Run only the selected rows on Metal and bind
            // their scale arrays independently; this avoids both the generic
            // CPU fallback and a model-sized CPU packed sidecar.
            if (resident_w8) {
                const bool w8pc =
                    gu.groups_per_row == 1 &&
                    down.groups_per_row == 1;
                constexpr int activation_group = 128;
                if (w8pc && seq >= 64 && has_tensor &&
                    hidden_size % activation_group == 0 &&
                    intermediate % activation_group == 0) {
                    const int selections = seq * top_k;
                    const size_t qx_bytes =
                        (size_t)seq * hidden_size;
                    const size_t sx_bytes =
                        (size_t)seq *
                        ((hidden_size + activation_group - 1) /
                         activation_group) * sizeof(float);
                    const size_t qi_bytes =
                        (size_t)selections * intermediate;
                    const size_t si_bytes =
                        (size_t)selections *
                        ((intermediate + activation_group - 1) /
                         activation_group) * sizeof(float);
                    const size_t selected_bytes =
                        (size_t)selections * hidden_size * sizeof(float);
                    const size_t expert_counts_bytes =
                        (size_t)num_experts * sizeof(uint32_t);
                    const size_t expert_routes_bytes =
                        (size_t)num_experts * (size_t)seq *
                        sizeof(int32_t);
                    const size_t jobs_queue_bytes =
                        (size_t)selections * 2 * sizeof(uint32_t);
                    const size_t jobs_bytes = 2 * jobs_queue_bytes;
                    const size_t job_count_bytes =
                        2 * sizeof(uint32_t);
                    const size_t dispatch_bytes =
                        12 * sizeof(uint32_t);

                    void* qx_h = pool_->acquire(qx_bytes);
                    void* sx_h = pool_->acquire(sx_bytes);
                    void* qi_h = pool_->acquire(qi_bytes);
                    void* si_h = pool_->acquire(si_bytes);
                    void* selected_h =
                        pool_->acquire(selected_bytes);
                    void* counts_h =
                        pool_->acquire(expert_counts_bytes);
                    void* routes_h =
                        pool_->acquire(expert_routes_bytes);
                    void* jobs_h = pool_->acquire(jobs_bytes);
                    void* job_count_h =
                        pool_->acquire(job_count_bytes);
                    void* dispatch_h =
                        pool_->acquire(dispatch_bytes);
                    id<MTLBuffer> qx = (__bridge id<MTLBuffer>)qx_h;
                    id<MTLBuffer> sx = (__bridge id<MTLBuffer>)sx_h;
                    id<MTLBuffer> qi = (__bridge id<MTLBuffer>)qi_h;
                    id<MTLBuffer> si = (__bridge id<MTLBuffer>)si_h;
                    id<MTLBuffer> selected =
                        (__bridge id<MTLBuffer>)selected_h;
                    id<MTLBuffer> counts =
                        (__bridge id<MTLBuffer>)counts_h;
                    id<MTLBuffer> routes =
                        (__bridge id<MTLBuffer>)routes_h;
                    id<MTLBuffer> jobs =
                        (__bridge id<MTLBuffer>)jobs_h;
                    id<MTLBuffer> job_count =
                        (__bridge id<MTLBuffer>)job_count_h;
                    id<MTLBuffer> dispatch =
                        (__bridge id<MTLBuffer>)dispatch_h;

                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 "moe_reset_expert_counts")];
                    [enc setBuffer:counts offset:0 atIndex:0];
                    [enc setBuffer:job_count offset:0 atIndex:1];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    grid1d(num_experts);
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 "moe_build_expert_routes")];
                    [enc setBuffer:idx offset:0 atIndex:0];
                    [enc setBuffer:counts offset:0 atIndex:1];
                    [enc setBuffer:routes offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc dispatchThreadgroups:
                             MTLSizeMake(num_experts, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 "moe_build_grouped_jobs")];
                    [enc setBuffer:counts offset:0 atIndex:0];
                    [enc setBuffer:jobs offset:0 atIndex:1];
                    [enc setBuffer:job_count offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc setBuffer:jobs offset:jobs_queue_bytes atIndex:4];
                    grid1d(num_experts);
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 "moe_finalize_grouped_dispatch")];
                    [enc setBuffer:job_count offset:0 atIndex:0];
                    [enc setBuffer:dispatch offset:0 atIndex:1];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    grid1d(1);
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    profile_resident_moe_stage("MOE.group_routes_w8");

                    auto quantize_rows = [&](id<MTLBuffer> src,
                                             uint src_offset,
                                             int rows, int K,
                                             int row_stride,
                                             id<MTLBuffer> dst,
                                             id<MTLBuffer> dst_scales) {
                        QuantActParams qp{};
                        qp.M = rows;
                        qp.K = K;
                        qp.a_offset = src_offset;
                        qp.a_row_stride = row_stride;
                        qp.block_size = activation_group;
                        [enc setComputePipelineState:
                                 pipelines_->pipeline(
                                     "quantize_act_i8_blocks")];
                        [enc setBuffer:src offset:0 atIndex:0];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&qp length:sizeof(qp) atIndex:3];
                        [enc setBuffer:dst_scales offset:0 atIndex:4];
                        constexpr NSUInteger nsg = 4;
                        const NSUInteger blocks =
                            ((NSUInteger)K + activation_group - 1) /
                            activation_group;
                        [enc setThreadgroupMemoryLength:
                                 nsg * sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:
                                 MTLSizeMake(
                                     (NSUInteger)rows * blocks,
                                     1, 1)
                            threadsPerThreadgroup:
                                 MTLSizeMake(32, nsg, 1)];
                    };
                    auto grouped_w8 = [&](id<MTLBuffer> activation,
                                           id<MTLBuffer> activation_scales,
                                           const Tensor& weight,
                                           int output_rows, int inner,
                                           int rows_per_expert,
                                           bool activation_by_token,
                                           bool paired_gate_up,
                                           bool large_route_tile,
                                           id<MTLBuffer> destination,
                                           int destination_stride) {
                        GroupedW4A8Params gp{};
                        gp.experts = num_experts;
                        gp.max_routes = seq;
                        gp.top_k = top_k;
                        gp.N = output_rows;
                        gp.K = inner;
                        gp.c_row_stride = destination_stride;
                        gp.groups_per_row =
                            (inner + activation_group - 1) /
                            activation_group;
                        gp.rows_per_expert = rows_per_expert;
                        gp.activation_by_token =
                            activation_by_token ? 1 : 0;
                        const char* pipeline_name = paired_gate_up
                            ? (large_route_tile
                                   ? "gemm_grouped_experts_w8_gate_up_r32"
                                   : "gemm_grouped_experts_w8_gate_up_r16")
                            : (large_route_tile
                                   ? "gemm_grouped_experts_w8_down_r32"
                                   : "gemm_grouped_experts_w8_down_r16");
                        [enc setComputePipelineState:
                                 pipelines_->pipeline(pipeline_name)];
                        [enc setBuffer:activation offset:0 atIndex:0];
                        [enc setBuffer:buffer_of(&weight)
                                offset:weight.device.offset atIndex:1];
                        [enc setBuffer:destination offset:0 atIndex:2];
                        [enc setBytes:&gp length:sizeof(gp) atIndex:3];
                        [enc setBuffer:activation_scales
                                offset:0 atIndex:4];
                        [enc setBuffer:scales_buffer_of(&weight)
                                offset:weight.device.scales_offset atIndex:5];
                        [enc setBuffer:counts offset:0 atIndex:6];
                        [enc setBuffer:routes offset:0 atIndex:7];
                        [enc setBuffer:jobs
                                offset:large_route_tile
                                    ? jobs_queue_bytes : 0
                               atIndex:8];
                        const NSUInteger route_tile =
                            large_route_tile ? 32 : 16;
                        const NSUInteger projected_rows =
                            paired_gate_up ? 64 : 64;
                        const NSUInteger staging_bytes =
                            (projected_rows + route_tile) * 32;
                        const NSUInteger result_bytes =
                            projected_rows * route_tile * sizeof(int32_t);
                        [enc setThreadgroupMemoryLength:
                                 std::max(staging_bytes, result_bytes)
                                                       atIndex:0];
                        const NSUInteger record = paired_gate_up
                            ? (large_route_tile ? 1 : 0)
                            : (large_route_tile ? 3 : 2);
                        [enc dispatchThreadgroupsWithIndirectBuffer:dispatch
                            indirectBufferOffset:
                                record * 3 * sizeof(uint32_t)
                            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    };

                    quantize_rows(
                        buffer_of(&x), (uint)element_offset(x), seq,
                        hidden_size, element_stride(x, 1), qx, sx);
                    grouped_w8(
                        qx, sx, gu, intermediate, hidden_size,
                        2 * intermediate, true, true, false,
                        merged, intermediate);
                    grouped_w8(
                        qx, sx, gu, intermediate, hidden_size,
                        2 * intermediate, true, true, true,
                        merged, intermediate);
                    profile_resident_moe_stage("MOE.gate_up_w8a8_grouped");

                    quantize_rows(
                        merged, 0, selections, intermediate,
                        intermediate, qi, si);
                    grouped_w8(
                        qi, si, down, hidden_size, intermediate,
                        hidden_size, false, false, false,
                        selected, hidden_size);
                    grouped_w8(
                        qi, si, down, hidden_size, intermediate,
                        hidden_size, false, false, true,
                        selected, hidden_size);
                    profile_resident_moe_stage("MOE.down_w8a8_grouped");

                    [enc setComputePipelineState:
                             pipelines_->pipeline("moe_combine_selected")];
                    [enc setBuffer:selected offset:0 atIndex:0];
                    [enc setBuffer:buffer_of(output) offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc setBuffer:tw offset:0 atIndex:4];
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (hidden_size + 63) / 64,
                                 (seq + 3) / 4, 1)
                        threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];

                    commands_->pending_free.push_back({qx_h, qx_bytes});
                    commands_->pending_free.push_back({sx_h, sx_bytes});
                    commands_->pending_free.push_back({qi_h, qi_bytes});
                    commands_->pending_free.push_back({si_h, si_bytes});
                    commands_->pending_free.push_back(
                        {selected_h, selected_bytes});
                    commands_->pending_free.push_back(
                        {counts_h, expert_counts_bytes});
                    commands_->pending_free.push_back(
                        {routes_h, expert_routes_bytes});
                    commands_->pending_free.push_back({jobs_h, jobs_bytes});
                    commands_->pending_free.push_back(
                        {job_count_h, job_count_bytes});
                    commands_->pending_free.push_back(
                        {dispatch_h, dispatch_bytes});
                    commands_->pending_free.push_back({idx_h, idx_bytes});
                    commands_->pending_free.push_back({tw_h, tw_bytes});
                    commands_->pending_free.push_back({logits_h, logits_bytes});
                    commands_->pending_free.push_back({merged_h, merged_bytes});
                    break;
                }
                if (w8pc && seq == 1) {
                    const int selections = seq * top_k;
                    const size_t qx_bytes =
                        (size_t)seq * hidden_size;
                    const size_t sx_bytes =
                        (size_t)seq * sizeof(float);
                    const size_t qi_bytes =
                        (size_t)selections * intermediate;
                    const size_t si_bytes =
                        (size_t)selections * sizeof(float);
                    const size_t selected_bytes =
                        (size_t)selections * hidden_size * sizeof(float);
                    void* qx_h = pool_->acquire(qx_bytes);
                    void* sx_h = pool_->acquire(sx_bytes);
                    void* qi_h = pool_->acquire(qi_bytes);
                    void* si_h = pool_->acquire(si_bytes);
                    void* selected_h =
                        pool_->acquire(selected_bytes);
                    id<MTLBuffer> qx = (__bridge id<MTLBuffer>)qx_h;
                    id<MTLBuffer> sx = (__bridge id<MTLBuffer>)sx_h;
                    id<MTLBuffer> qi = (__bridge id<MTLBuffer>)qi_h;
                    id<MTLBuffer> si = (__bridge id<MTLBuffer>)si_h;
                    id<MTLBuffer> selected =
                        (__bridge id<MTLBuffer>)selected_h;

                    auto quantize_rows = [&](id<MTLBuffer> src,
                                             uint src_offset,
                                             int rows, int K,
                                             int row_stride,
                                             id<MTLBuffer> dst,
                                             id<MTLBuffer> dst_scales) {
                        QuantActParams qp{};
                        qp.M = rows;
                        qp.K = K;
                        qp.a_offset = src_offset;
                        qp.a_row_stride = row_stride;
                        [enc setComputePipelineState:
                                 pipelines_->pipeline("quantize_act_i8")];
                        [enc setBuffer:src offset:0 atIndex:0];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&qp length:sizeof(qp) atIndex:3];
                        [enc setBuffer:dst_scales offset:0 atIndex:4];
                        [enc setThreadgroupMemoryLength:
                                 8 * sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:
                                 MTLSizeMake(rows, 1, 1)
                            threadsPerThreadgroup:
                                 MTLSizeMake(32, 8, 1)];
                    };
                    auto selected_w8 = [&](id<MTLBuffer> activation,
                                           id<MTLBuffer> activation_scales,
                                           const Tensor& weight,
                                           int N, int K,
                                           int rows_per_expert,
                                           int activation_rows,
                                           int activation_repeat,
                                           id<MTLBuffer> dst,
                                           int dst_stride) {
                        SelectedW4A8Params sp{};
                        sp.selections = selections;
                        sp.N = N;
                        sp.K = K;
                        sp.c_offset = 0;
                        sp.c_row_stride = dst_stride;
                        sp.group_size = K;
                        sp.groups_per_row = 1;
                        sp.rows_per_expert = rows_per_expert;
                        sp.activation_rows = activation_rows;
                        sp.activation_repeat = activation_repeat;
                        [enc setComputePipelineState:
                                 pipelines_->pipeline(
                                     "gemv_selected_experts_w8_i8a_i8b_f32c")];
                        [enc setBuffer:activation offset:0 atIndex:0];
                        [enc setBuffer:buffer_of(&weight)
                                offset:weight.device.offset atIndex:1];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                        [enc setBuffer:activation_scales
                                offset:0 atIndex:4];
                        [enc setBuffer:scales_buffer_of(&weight)
                                offset:weight.device.scales_offset atIndex:5];
                        [enc setBuffer:idx offset:0 atIndex:6];
                        constexpr NSUInteger w8_nsg = 4;
                        constexpr NSUInteger rows_per_tg = w8_nsg * 8;
                        [enc dispatchThreadgroups:
                                 MTLSizeMake(
                                     (N + rows_per_tg - 1) / rows_per_tg,
                                     1, selections)
                            threadsPerThreadgroup:
                                 MTLSizeMake(32, w8_nsg, 1)];
                    };

                    quantize_rows(
                        buffer_of(&x), (uint)element_offset(x), seq,
                        hidden_size, element_stride(x, 1), qx, sx);
                    selected_w8(
                        qx, sx, gu, 2 * intermediate, hidden_size,
                        2 * intermediate, seq, top_k,
                        merged, 2 * intermediate);
                    profile_resident_moe_stage("MOE.gate_up_w8a8");

                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 "moe_swiglu_quantize_row")];
                    [enc setBuffer:merged offset:0 atIndex:0];
                    [enc setBuffer:qi offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc setBuffer:si offset:0 atIndex:4];
                    constexpr NSUInteger swiglu_nsg = 8;
                    [enc setThreadgroupMemoryLength:
                             swiglu_nsg * sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:
                             MTLSizeMake(selections, 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(32, swiglu_nsg, 1)];

                    selected_w8(
                        qi, si, down, hidden_size, intermediate,
                        hidden_size, selections, 1,
                        selected, hidden_size);
                    profile_resident_moe_stage("MOE.down_w8a8");

                    [enc setComputePipelineState:
                             pipelines_->pipeline("moe_combine_selected")];
                    [enc setBuffer:selected offset:0 atIndex:0];
                    [enc setBuffer:buffer_of(output) offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc setBuffer:tw offset:0 atIndex:4];
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (hidden_size + 255) / 256, 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(256, 1, 1)];
                    commands_->pending_free.push_back({qx_h, qx_bytes});
                    commands_->pending_free.push_back({sx_h, sx_bytes});
                    commands_->pending_free.push_back({qi_h, qi_bytes});
                    commands_->pending_free.push_back({si_h, si_bytes});
                    commands_->pending_free.push_back(
                        {selected_h, selected_bytes});
                    commands_->pending_free.push_back({idx_h, idx_bytes});
                    commands_->pending_free.push_back({tw_h, tw_bytes});
                    commands_->pending_free.push_back({logits_h, logits_bytes});
                    commands_->pending_free.push_back({merged_h, merged_bytes});
                    break;
                }
                const bool exact_decode =
                    seq == 1 && gu.groups_per_row == 1 &&
                    down.groups_per_row == 1;
                const bool fused_small_gate_up =
                    seq >= 2 && seq <= 4;
                const bool compact_down = seq >= 2 && seq <= 4;
                [enc setComputePipelineState:
                         pipelines_->pipeline(
                             exact_decode
                                 ? "moe_gate_up_w8_precise"
                                 : (fused_small_gate_up
                                        ? "moe_gate_up_swiglu_w8_r4"
                                        : "moe_gate_up_w8"))];
                [enc setBuffer:buffer_of(&x) offset:0 atIndex:0];
                [enc setBuffer:buffer_of(&gu)
                        offset:gu.device.offset atIndex:1];
                [enc setBuffer:merged offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc setBuffer:scales_buffer_of(&gu)
                        offset:gu.device.scales_offset atIndex:4];
                [enc setBuffer:idx offset:0 atIndex:5];
                if (exact_decode)
                    [enc setThreadgroupMemoryLength:
                             4 * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:
                         MTLSizeMake(
                             exact_decode
                                 ? (2 * intermediate + 3) / 4
                                 : (fused_small_gate_up
                                        ? (intermediate + 15) / 16
                                        : (2 * intermediate + 31) / 32),
                                     top_k, seq)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                profile_resident_moe_stage("MOE.gate_up_w8");

                if (!fused_small_gate_up) {
                    [enc setComputePipelineState:
                             pipelines_->pipeline("moe_swiglu_selected")];
                    [enc setBuffer:merged offset:0 atIndex:0];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    const NSUInteger sw_n =
                        (NSUInteger)seq * top_k * intermediate;
                    [enc dispatchThreadgroups:
                             MTLSizeMake((sw_n + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }

                [enc setComputePipelineState:
                         pipelines_->pipeline(
                             exact_decode
                                 ? "moe_down_combine_w8_precise"
                                 : (compact_down
                                        ? "moe_down_combine_w8_r4"
                                        : "moe_down_combine_w8"))];
                [enc setBuffer:merged offset:0 atIndex:0];
                [enc setBuffer:buffer_of(&down)
                        offset:down.device.offset atIndex:1];
                [enc setBuffer:buffer_of(output) offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc setBuffer:scales_buffer_of(&down)
                        offset:down.device.scales_offset atIndex:4];
                [enc setBuffer:idx offset:0 atIndex:5];
                [enc setBuffer:tw offset:0 atIndex:6];
                if (exact_decode)
                    [enc setThreadgroupMemoryLength:
                             4 * sizeof(float) atIndex:0];
                [enc dispatchThreadgroups:
                         MTLSizeMake(
                             exact_decode
                                 ? (hidden_size + 3) / 4
                                 : (hidden_size +
                                    (compact_down ? 15 : 31)) /
                                       (compact_down ? 16 : 32),
                             seq, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                profile_resident_moe_stage("MOE.down_w8");

                commands_->pending_free.push_back({idx_h, idx_bytes});
                commands_->pending_free.push_back({tw_h, tw_bytes});
                commands_->pending_free.push_back({logits_h, logits_bytes});
                commands_->pending_free.push_back({merged_h, merged_bytes});
                break;
            }

#ifdef MOLLM_METAL_TENSOR
            if (ssd_w4) {
                const int* exact_routes =
                    static_cast<const int*>(MetalBufferPool::contents(idx_h));
                std::vector<int> experts(exact_routes,
                                         exact_routes + seq * top_k);
                std::vector<MetalSsdExpertCache::ExpertView> expert_views;
                const uint64_t demand_start = mollm_trace::now_ns();
                if (!ssd_cache_->acquire(
                        ssd_gate->spec, ssd_down->spec, experts,
                        expert_views)) {
                    fprintf(stderr,
                            "MetalBackend: failed to load SSD experts for "
                            "layer %d\n",
                            ssd_gate->spec.layer);
                    break;
                }
                mollm_trace::record_duration(
                    "metal.ssd", "demand_schedule", demand_start,
                    mollm_trace::now_ns(),
                    "{\"layer\":" +
                        std::to_string(ssd_gate->spec.layer) + "}",
                    "good");

                if (predicted_layer) {
                    const int* predicted_routes =
                        static_cast<const int*>(
                            MetalBufferPool::contents(predicted_idx_h));
                    std::vector<int> predicted_experts(
                        predicted_routes,
                        predicted_routes +
                            seq * predicted_layer->top_k);
                    std::vector<MetalSsdExpertCache::ExpertView> predicted_views;
                    // Demand I/O was submitted first. This speculative command
                    // may execute concurrently with it and overlaps the current
                    // layer's expert compute. Per-batch readiness events keep
                    // cache slots safe without globally serializing MTLIO.
                    const uint64_t prefetch_start =
                        mollm_trace::now_ns();
                    ssd_cache_->acquire(
                        predicted_layer->gate_up->spec,
                        predicted_layer->down->spec,
                        predicted_experts, predicted_views, true);
                    mollm_trace::record_duration(
                        "metal.ssd", "prefetch_schedule", prefetch_start,
                        mollm_trace::now_ns(),
                        "{\"layer\":" +
                            std::to_string(
                                predicted_layer->gate_up->spec.layer) +
                            "}",
                        "rail_load");
                    pool_->release(
                        predicted_idx_h, predicted_idx_bytes);
                    pool_->release(
                        predicted_tw_h, predicted_tw_bytes);
                    predicted_layer = nullptr;
                }

                commands_->cmd = [commands_->queue commandBuffer];
                commands_->cmd.label = @"mollm Metal SSD expert";
                const int selections = seq * top_k;
                const size_t qx_bytes = (size_t)seq * hidden_size;
                const size_t sx_bytes = (size_t)seq * sizeof(float);
                void* qx_h = pool_->acquire(qx_bytes);
                void* sx_h = pool_->acquire(sx_bytes);
                id<MTLBuffer> qx = (__bridge id<MTLBuffer>)qx_h;
                id<MTLBuffer> sx = (__bridge id<MTLBuffer>)sx_h;
                bool x_quantized = false;
                void* shared_qx_h = shared_work.qx;
                void* shared_sx_h = shared_work.sx;
                void* shared_inter_h = shared_work.intermediate;
                void* shared_qinter_h = shared_work.qintermediate;
                void* shared_qinter_scale_h =
                    shared_work.qintermediate_scale;
                void* shared_scale_h = shared_work.scale;
                void* shared_output_h = shared_work.output;
                size_t shared_inter_bytes =
                    shared_work.intermediate_bytes;
                size_t shared_qinter_bytes =
                    shared_work.qintermediate_bytes;
                size_t shared_output_bytes = shared_work.output_bytes;
                id<MTLBuffer> shared_output =
                    (__bridge id<MTLBuffer>)shared_output_h;
                uint64_t shared_ready_value = shared_work.ready_value;
                std::vector<int> ordered_selections;
                ordered_selections.reserve(selections);
                auto selection_ready = [&](int selection) {
                    const auto& view = expert_views[selection];
                    return !view.gate_ready_event ||
                           view.gate_ready_event.signaledValue >=
                               view.gate_ready_value;
                };
                for (int selection = 0; selection < selections;
                     ++selection) {
                    if (selection_ready(selection)) {
                        ordered_selections.push_back(selection);
                    }
                }
                const int observed_ready =
                    static_cast<int>(ordered_selections.size());
                constexpr int kMinimumReadyToSplit = 4;
                int ready_selections =
                    observed_ready >= kMinimumReadyToSplit
                        ? observed_ready
                        : 0;
                if (ready_selections == 0) {
                    ordered_selections.clear();
                    for (int selection = 0; selection < selections;
                         ++selection) {
                        ordered_selections.push_back(selection);
                    }
                } else {
                    for (int selection = 0; selection < selections;
                         ++selection) {
                        if (!selection_ready(selection)) {
                            ordered_selections.push_back(selection);
                        }
                    }
                }
                auto encode_gate_waits = [&](int ordered_begin) {
                    std::unordered_set<void*> waited_events;
                    for (int ordered = ordered_begin;
                         ordered < selections; ++ordered) {
                        const auto& view =
                            expert_views[ordered_selections[ordered]];
                        if (!view.gate_ready_event ||
                            view.gate_ready_event.signaledValue >=
                                view.gate_ready_value)
                            continue;
                        void* event_key =
                            (__bridge void*)view.gate_ready_event;
                        if (waited_events.insert(event_key).second) {
                            [commands_->cmd
                                encodeWaitForEvent:view.gate_ready_event
                                             value:view.gate_ready_value];
                        }
                    }
                };
                if (ready_selections == 0)
                    encode_gate_waits(0);
                commands_->enc = [commands_->cmd computeCommandEncoder];
                commands_->enc.label = @"mollm Metal SSD expert";
                enc = commands_->enc;

                const size_t qi_bytes = (size_t)selections * intermediate;
                const size_t si_bytes = (size_t)selections * sizeof(float);
                const size_t selected_bytes =
                    (size_t)selections * hidden_size * sizeof(float);
                const size_t slot_offsets_bytes =
                    (size_t)selections * 2 * sizeof(uint64_t);
                const size_t selection_indices_bytes =
                    (size_t)selections * sizeof(uint32_t);
                void* qi_h = pool_->acquire(qi_bytes);
                void* si_h = pool_->acquire(si_bytes);
                void* selected_h = pool_->acquire(selected_bytes);
                void* slot_offsets_h =
                    pool_->acquire(slot_offsets_bytes);
                void* selection_indices_h =
                    pool_->acquire(selection_indices_bytes);
                id<MTLBuffer> qi = (__bridge id<MTLBuffer>)qi_h;
                id<MTLBuffer> si = (__bridge id<MTLBuffer>)si_h;
                id<MTLBuffer> selected =
                    (__bridge id<MTLBuffer>)selected_h;
                id<MTLBuffer> slot_offsets =
                    (__bridge id<MTLBuffer>)slot_offsets_h;
                id<MTLBuffer> selection_indices =
                    (__bridge id<MTLBuffer>)selection_indices_h;
                auto* slot_offsets_data = static_cast<uint64_t*>(
                    MetalBufferPool::contents(slot_offsets_h));
                auto* selection_indices_data = static_cast<uint32_t*>(
                    MetalBufferPool::contents(selection_indices_h));
                for (int ordered = 0; ordered < selections; ++ordered) {
                    const int selection = ordered_selections[ordered];
                    selection_indices_data[ordered] =
                        static_cast<uint32_t>(selection);
                    slot_offsets_data[ordered] =
                        expert_views[selection].gate_up_offset;
                    slot_offsets_data[selections + ordered] =
                        expert_views[selection].down_offset;
                }

                auto quantize = [&](id<MTLBuffer> src, uint src_off, int rows,
                                    int K, int row_stride, id<MTLBuffer> dst,
                                    id<MTLBuffer> scales) {
                    QuantActParams q{};
                    q.M = rows;
                    q.K = K;
                    q.a_offset = src_off;
                    q.a_row_stride = row_stride;
                    [enc setComputePipelineState:
                             pipelines_->pipeline("quantize_act_i8")];
                    [enc setBuffer:src offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    [enc setBytes:&q length:sizeof(q) atIndex:3];
                    [enc setBuffer:scales offset:0 atIndex:4];
                    [enc setThreadgroupMemoryLength:8 * sizeof(float)
                                            atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
                };
                if (!x_quantized) {
                    quantize(buffer_of(&x), (uint)element_offset(x), seq, hidden_size,
                             element_stride(x, 1), qx, sx);
                }

                auto selected_bg128 = [&](id<MTLBuffer> activation,
                                          id<MTLBuffer> activation_scale,
                                          id<MTLBuffer> weight,
                                          size_t slot_offsets_offset,
                                          size_t selection_indices_offset,
                                          int group_selections,
                                          int N, int K,
                                          int groups_per_row,
                                          int activation_rows,
                                          int activation_repeat,
                                          id<MTLBuffer> dst,
                                          int dst_stride) {
                    SelectedW4A8Params sp{};
                    sp.selections = group_selections;
                    sp.N = N;
                    sp.K = K;
                    sp.c_offset = 0;
                    sp.c_row_stride = dst_stride;
                    sp.group_size = 128;
                    sp.groups_per_row = groups_per_row;
                    sp.rows_per_expert = N;
                    sp.activation_rows = activation_rows;
                    sp.activation_repeat = activation_repeat;
                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 "gemv_selected_slots_bg128_i8a_i4b_f32c")];
                    [enc setBuffer:activation offset:0 atIndex:0];
                    [enc setBuffer:weight offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                    [enc setBuffer:activation_scale offset:0 atIndex:4];
                    [enc setBuffer:slot_offsets
                            offset:slot_offsets_offset
                           atIndex:6];
                    [enc setBuffer:selection_indices
                            offset:selection_indices_offset
                           atIndex:7];
                    [enc dispatchThreadgroups:
                             MTLSizeMake((N + 31) / 32, 1,
                                         group_selections)
                        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                };

                if (ready_selections > 0) {
                    selected_bg128(
                        qx, sx, expert_views[0].gate_up, 0, 0,
                        ready_selections,
                        2 * intermediate, hidden_size,
                        ssd_gate->spec.groups_per_row, seq, top_k,
                        merged, 2 * intermediate);
                }
                const int pending_selections =
                    selections - ready_selections;
                if (pending_selections > 0 &&
                    ready_selections > 0) {
                    [enc endEncoding];
                    commands_->enc = nil;
                    encode_gate_waits(ready_selections);
                    commands_->enc =
                        [commands_->cmd computeCommandEncoder];
                    commands_->enc.label =
                        @"mollm pending Metal SSD expert";
                    enc = commands_->enc;
                }
                if (pending_selections > 0) {
                    selected_bg128(
                        qx, sx, expert_views[0].gate_up,
                        (size_t)ready_selections *
                            sizeof(uint64_t),
                        (size_t)ready_selections *
                            sizeof(uint32_t),
                        pending_selections,
                        2 * intermediate, hidden_size,
                        ssd_gate->spec.groups_per_row, seq, top_k,
                        merged, 2 * intermediate);
                }

                [enc setComputePipelineState:
                         pipelines_->pipeline("moe_swiglu_selected")];
                [enc setBuffer:merged offset:0 atIndex:0];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                const NSUInteger sw_n =
                    (NSUInteger)selections * intermediate;
                [enc dispatchThreadgroups:
                         MTLSizeMake((sw_n + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                quantize(merged, 0, selections, intermediate,
                         2 * intermediate, qi, si);

                std::unordered_set<void*> waited_down_events;
                bool has_pending_down = false;
                for (const auto& view : expert_views) {
                    if (view.down_ready_event &&
                        view.down_ready_event.signaledValue <
                            view.down_ready_value) {
                        has_pending_down = true;
                        break;
                    }
                }
                if (has_pending_down) {
                    [enc endEncoding];
                    commands_->enc = nil;
                    for (const auto& view : expert_views) {
                        if (!view.down_ready_event ||
                            view.down_ready_event.signaledValue >=
                                view.down_ready_value)
                            continue;
                        void* event_key =
                            (__bridge void*)view.down_ready_event;
                        if (waited_down_events.insert(event_key).second) {
                            [commands_->cmd
                                encodeWaitForEvent:view.down_ready_event
                                             value:view.down_ready_value];
                        }
                    }
                    commands_->enc =
                        [commands_->cmd computeCommandEncoder];
                    commands_->enc.label =
                        @"mollm Metal SSD down experts";
                    enc = commands_->enc;
                }

                selected_bg128(
                    qi, si, expert_views[0].down,
                    (size_t)selections * sizeof(uint64_t),
                    0, selections,
                    hidden_size, intermediate,
                    ssd_down->spec.groups_per_row, selections, 1,
                    selected, hidden_size);

                [enc setComputePipelineState:
                         pipelines_->pipeline(
                             "moe_combine_selected")];
                [enc setBuffer:selected offset:0 atIndex:0];
                [enc setBuffer:buffer_of(output)
                        offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc setBuffer:tw offset:0 atIndex:4];
                [enc dispatchThreadgroups:
                         MTLSizeMake(
                             (hidden_size + 63) / 64,
                             (seq + 3) / 4, 1)
                    threadsPerThreadgroup:
                         MTLSizeMake(64, 4, 1)];

                if (has_shared) {
                    [enc endEncoding];
                    commands_->enc = nil;
                    id<MTLSharedEvent> shared_event =
                        ssd_shared_expert_->event();
                    [commands_->cmd
                        encodeWaitForEvent:shared_event
                                   value:shared_ready_value];
                    commands_->enc = [commands_->cmd computeCommandEncoder];
                    commands_->enc.label = @"mollm combine SSD experts";
                    enc = commands_->enc;
                    const uint count = (uint)hidden_size;
                    [enc setComputePipelineState:
                             pipelines_->pipeline("add_inplace_f32")];
                    [enc setBuffer:buffer_of(output)
                            offset:output->device.offset
                           atIndex:0];
                    [enc setBuffer:shared_output offset:0 atIndex:1];
                    [enc setBytes:&count length:sizeof(count) atIndex:3];
                    [enc dispatchThreadgroups:
                             MTLSizeMake((hidden_size + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    commands_->pending_free.push_back(
                        {shared_inter_h, shared_inter_bytes});
                    commands_->pending_free.push_back(
                        {shared_qx_h, qx_bytes});
                    commands_->pending_free.push_back(
                        {shared_sx_h, sx_bytes});
                    commands_->pending_free.push_back(
                        {shared_qinter_h, shared_qinter_bytes});
                    commands_->pending_free.push_back(
                        {shared_qinter_scale_h, sizeof(float)});
                    commands_->pending_free.push_back(
                        {shared_scale_h, sizeof(float)});
                    commands_->pending_free.push_back(
                        {shared_output_h, shared_output_bytes});
                }
                commands_->pending_free.push_back({qx_h, qx_bytes});
                commands_->pending_free.push_back({sx_h, sx_bytes});
                commands_->pending_free.push_back({qi_h, qi_bytes});
                commands_->pending_free.push_back({si_h, si_bytes});
                commands_->pending_free.push_back(
                    {selected_h, selected_bytes});
                commands_->pending_free.push_back(
                    {slot_offsets_h, slot_offsets_bytes});
                commands_->pending_free.push_back(
                    {selection_indices_h, selection_indices_bytes});
                commands_->pending_free.push_back({idx_h, idx_bytes});
                commands_->pending_free.push_back({tw_h, tw_bytes});
                commands_->pending_free.push_back({logits_h, logits_bytes});
                commands_->pending_free.push_back({merged_h, merged_bytes});
                break;
            }

            if (has_tensor) {
                int selections=seq*top_k;
                const bool native_down =
                    native_bg128(down, hidden_size);
                const int native_down_group =
                    native_down
                        ? 128
                        : (native_bg32(down, hidden_size) ? 32 : 0);
                const bool grouped_prefill =
                    seq >= 64 && native_gu_group != 0 &&
                    native_gu_group == native_down_group;
                size_t qx_bytes=(size_t)seq*hidden_size;
                size_t sx_bytes=(size_t)seq *
                    (native_gu_group ? gu.groups_per_row : 1) *
                    sizeof(float);
                size_t qi_bytes=(size_t)selections*intermediate;
                size_t si_bytes=(size_t)selections *
                    (native_down_group ? down.groups_per_row : 1) *
                    sizeof(float);
                size_t selected_bytes =
                    (size_t)selections * hidden_size * sizeof(float);
                void* qx_h=resident_qx_h
                    ? resident_qx_h
                    : pool_->acquire(qx_bytes);
                void* sx_h=resident_sx_h
                    ? resident_sx_h
                    : pool_->acquire(sx_bytes);
                void* qi_h=pool_->acquire(qi_bytes),*si_h=pool_->acquire(si_bytes);
                void* selected_h =
                    pool_->acquire(selected_bytes);
                id<MTLBuffer> qx=(__bridge id<MTLBuffer>)qx_h;
                id<MTLBuffer> sx=(__bridge id<MTLBuffer>)sx_h;
                id<MTLBuffer> qi=(__bridge id<MTLBuffer>)qi_h;
                id<MTLBuffer> si=(__bridge id<MTLBuffer>)si_h;
                id<MTLBuffer> selected =
                    (__bridge id<MTLBuffer>)selected_h;
                void* expert_counts_h = nullptr;
                void* expert_routes_h = nullptr;
                void* grouped_jobs_h = nullptr;
                void* grouped_job_count_h = nullptr;
                void* grouped_dispatch_h = nullptr;
                id<MTLBuffer> expert_counts = nil;
                id<MTLBuffer> expert_routes = nil;
                id<MTLBuffer> grouped_jobs = nil;
                id<MTLBuffer> grouped_job_count = nil;
                id<MTLBuffer> grouped_dispatch = nil;
                size_t expert_counts_bytes = 0;
                size_t expert_routes_bytes = 0;
                size_t grouped_jobs_queue_bytes = 0;
                size_t grouped_jobs_bytes = 0;
                size_t grouped_job_count_bytes = 0;
                size_t grouped_dispatch_bytes = 0;
                if (grouped_prefill) {
                    expert_counts_bytes =
                        (size_t)num_experts * sizeof(uint32_t);
                    expert_routes_bytes =
                        (size_t)num_experts * (size_t)seq *
                        sizeof(int32_t);
                    grouped_jobs_queue_bytes =
                        (size_t)selections * 2 * sizeof(uint32_t);
                    grouped_jobs_bytes =
                        2 * grouped_jobs_queue_bytes;
                    grouped_job_count_bytes =
                        2 * sizeof(uint32_t);
                    grouped_dispatch_bytes =
                        12 * sizeof(uint32_t);
                    expert_counts_h =
                        pool_->acquire(expert_counts_bytes);
                    expert_routes_h =
                        pool_->acquire(expert_routes_bytes);
                    grouped_jobs_h =
                        pool_->acquire(grouped_jobs_bytes);
                    grouped_job_count_h =
                        pool_->acquire(grouped_job_count_bytes);
                    grouped_dispatch_h =
                        pool_->acquire(grouped_dispatch_bytes);
                    expert_counts =
                        (__bridge id<MTLBuffer>)expert_counts_h;
                    expert_routes =
                        (__bridge id<MTLBuffer>)expert_routes_h;
                    grouped_jobs =
                        (__bridge id<MTLBuffer>)grouped_jobs_h;
                    grouped_job_count =
                        (__bridge id<MTLBuffer>)grouped_job_count_h;
                    grouped_dispatch =
                        (__bridge id<MTLBuffer>)grouped_dispatch_h;

                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 "moe_reset_expert_counts")];
                    [enc setBuffer:expert_counts
                            offset:0 atIndex:0];
                    [enc setBuffer:grouped_job_count
                            offset:0 atIndex:1];
                    [enc setBytes:&mp
                           length:sizeof(mp) atIndex:3];
                    grid1d(num_experts);
                    [enc memoryBarrierWithScope:
                             MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 "moe_build_expert_routes")];
                    [enc setBuffer:idx offset:0 atIndex:0];
                    [enc setBuffer:expert_counts
                            offset:0 atIndex:1];
                    [enc setBuffer:expert_routes
                            offset:0 atIndex:2];
                    [enc setBytes:&mp
                           length:sizeof(mp) atIndex:3];
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (NSUInteger)num_experts, 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(128, 1, 1)];
                    [enc memoryBarrierWithScope:
                             MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 "moe_build_grouped_jobs")];
                    [enc setBuffer:expert_counts
                            offset:0 atIndex:0];
                    [enc setBuffer:grouped_jobs
                            offset:0 atIndex:1];
                    [enc setBuffer:grouped_job_count
                            offset:0 atIndex:2];
                    [enc setBytes:&mp
                           length:sizeof(mp) atIndex:3];
                    [enc setBuffer:grouped_jobs
                            offset:grouped_jobs_queue_bytes
                           atIndex:4];
                    grid1d(num_experts);
                    [enc memoryBarrierWithScope:
                             MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 "moe_finalize_grouped_dispatch")];
                    [enc setBuffer:grouped_job_count
                            offset:0 atIndex:0];
                    [enc setBuffer:grouped_dispatch
                            offset:0 atIndex:1];
                    [enc setBytes:&mp
                           length:sizeof(mp) atIndex:3];
                    grid1d(1);
                    [enc memoryBarrierWithScope:
                             MTLBarrierScopeBuffers];
                    if (commands_->profile) {
                        profile_resident_moe_stage(
                            "MOE.group_routes");
                        const auto* counts =
                            static_cast<const uint32_t*>(
                                MetalBufferPool::contents(
                                    expert_counts_h));
                        const auto* jobs =
                            static_cast<const uint32_t*>(
                                MetalBufferPool::contents(
                                    grouped_job_count_h));
                        uint64_t routes = 0;
                        uint32_t nonempty = 0;
                        uint32_t max_routes = 0;
                        uint32_t theoretical_jobs16 = 0;
                        uint32_t theoretical_jobs32 = 0;
                        for (int expert = 0;
                             expert < num_experts; ++expert) {
                            routes += counts[expert];
                            nonempty += counts[expert] != 0;
                            theoretical_jobs16 +=
                                (counts[expert] + 15) / 16;
                            theoretical_jobs32 +=
                                (counts[expert] + 31) / 32;
                            max_routes =
                                std::max(max_routes, counts[expert]);
                        }
                        const bool use_large =
                            jobs[0] != 0 &&
                            jobs[1] * 5u <= jobs[0] * 3u;
                        const uint64_t capacity =
                            use_large
                                ? (uint64_t)jobs[1] *
                                      MOLLM_GROUPED_MOE_ROUTE_TILE_LARGE
                                : (uint64_t)jobs[0] *
                                      MOLLM_GROUPED_MOE_ROUTE_TILE_SMALL;
                        fprintf(
                            stderr,
                            "[metal-moe] S=%d routes=%llu "
                            "nonempty=%u jobs16=%u jobs32=%u "
                            "all16=%u all32=%u "
                            "tile=%u "
                            "capacity=%llu util=%.1f%% "
                            "max_routes=%u\n",
                            seq,
                            (unsigned long long)routes,
                            nonempty, jobs[0], jobs[1],
                            theoretical_jobs16,
                            theoretical_jobs32,
                            use_large ? 32u : 16u,
                            (unsigned long long)capacity,
                            capacity
                                ? 100.0 * (double)routes /
                                      (double)capacity
                                : 0.0,
                            max_routes);
                    }
                }
                auto quantize = [&](id<MTLBuffer> src, uint src_off,
                                    int rows, int K, int row_stride,
                                    int block_size, id<MTLBuffer> dst,
                                    id<MTLBuffer> scales) {
                    QuantActParams q{};q.M=rows;q.K=K;q.a_offset=src_off;q.a_row_stride=row_stride;
                    q.block_size = block_size;
                    [enc setComputePipelineState:pipelines_->pipeline(
                        block_size == 32
                            ? "quantize_act_i8_block32"
                            : block_size
                                ? "quantize_act_i8_blocks"
                                : "quantize_act_i8")];
                    [enc setBuffer:src offset:0 atIndex:0];[enc setBuffer:dst offset:0 atIndex:2];
                    [enc setBytes:&q length:sizeof(q) atIndex:3];[enc setBuffer:scales offset:0 atIndex:4];
                    constexpr NSUInteger nsg = 4;
                    if (block_size != 32)
                        [enc setThreadgroupMemoryLength:
                                 nsg*sizeof(float) atIndex:0];
                    const NSUInteger blocks =
                        block_size
                            ? (K + block_size - 1) / block_size
                            : 1;
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (NSUInteger)rows *
                                     (block_size == 32
                                          ? (blocks + nsg - 1) / nsg
                                          : blocks),
                                 1,1)
                        threadsPerThreadgroup:
                             MTLSizeMake(32,nsg,1)];
                };
                if (!resident_qx_h) {
                    quantize(buffer_of(&x),(uint)element_offset(x),seq,hidden_size,
                             element_stride(x,1),native_gu_group,qx,sx);
                }
                auto selected_gemm = [&](id<MTLBuffer> a,id<MTLBuffer> sa,const Tensor& w,
                                         size_t rows_total,int N,int K,int rows_per_expert,
                                         int activation_rows,int repeat,id<MTLBuffer> dst,int dst_stride) {
                    SelectedW4A8Params sp{};sp.selections=selections;sp.N=N;sp.K=K;
                    sp.c_offset=0;sp.c_row_stride=dst_stride;sp.group_size=(int)w.group_size;
                    sp.groups_per_row=(int)w.groups_per_row;sp.rows_per_expert=rows_per_expert;
                    sp.activation_rows=activation_rows;sp.activation_repeat=repeat;
                    if (native_bg128(w, rows_per_expert)) {
                        [enc setComputePipelineState:
                                 pipelines_->pipeline(
                                     "gemv_selected_experts_bg128_"
                                     "i8a_i4b_f32c")];
                        [enc setBuffer:a offset:0 atIndex:0];
                        [enc setBuffer:buffer_of(&w)
                                offset:w.device.offset atIndex:1];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                        [enc setBuffer:sa offset:0 atIndex:4];
                        [enc setBuffer:idx offset:0 atIndex:6];
                        const NSUInteger nsg =
                            seq >= 2 && seq <= 4 ? 2 : 4;
                        const NSUInteger rows_per_tg = nsg * 8;
                        [enc dispatchThreadgroups:
                                 MTLSizeMake(
                                     (N + rows_per_tg - 1) / rows_per_tg,
                                     1, selections)
                            threadsPerThreadgroup:
                                 MTLSizeMake(32,nsg,1)];
                    } else if (native_bg32(
                                   w, rows_per_expert)) {
                        [enc setComputePipelineState:
                                 pipelines_->pipeline(
                                     "gemv_selected_experts_bg32_"
                                     "i8a_i4b_f32c")];
                        [enc setBuffer:a offset:0 atIndex:0];
                        [enc setBuffer:buffer_of(&w)
                                offset:w.device.offset atIndex:1];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                        [enc setBuffer:sa offset:0 atIndex:4];
                        [enc setBuffer:idx offset:0 atIndex:6];
                        const NSUInteger nsg = seq <= 4 ? 2 : 4;
                        const NSUInteger rows_per_tg = nsg * 8;
                        [enc dispatchThreadgroups:
                                 MTLSizeMake(
                                     (N + rows_per_tg - 1) / rows_per_tg,
                                     1, selections)
                            threadsPerThreadgroup:
                                 MTLSizeMake(32,nsg,1)];
                    } else {
                        [enc setComputePipelineState:
                                 pipelines_->pipeline(
                                     "gemm_selected_w4a8_i8a_i4b_f32c")];
                        [enc setBuffer:a offset:0 atIndex:0];
                        [enc setBuffer:buffer_of(&w)
                                offset:w.device.offset atIndex:1];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                        [enc setBuffer:sa offset:0 atIndex:4];
                        [enc setBuffer:buffer_of(&w)
                                offset:w.device.offset+rows_total*(K/2)
                               atIndex:5];
                        [enc setBuffer:idx offset:0 atIndex:6];
                        [enc setThreadgroupMemoryLength:
                                 2*64*64*sizeof(int32_t) atIndex:0];
                        [enc dispatchThreadgroups:
                                 MTLSizeMake(1,(N+63)/64,selections)
                                    threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                    }
                };
                auto grouped_gemm = [&](
                    id<MTLBuffer> activation,
                    id<MTLBuffer> activation_scales,
                    const Tensor& weight,
                    int output_rows,
                    int inner,
                    int rows_per_expert,
                    bool activation_by_token,
                    bool paired_gate_up,
                    bool large_route_tile,
                    id<MTLBuffer> destination,
                    int destination_stride) {
                    GroupedW4A8Params gp{};
                    gp.experts = num_experts;
                    gp.max_routes = seq;
                    gp.top_k = top_k;
                    gp.N = output_rows;
                    gp.K = inner;
                    gp.c_row_stride = destination_stride;
                    gp.groups_per_row =
                        (int)weight.groups_per_row;
                    gp.rows_per_expert = rows_per_expert;
                    gp.activation_by_token =
                        activation_by_token ? 1 : 0;
                    [enc setBuffer:activation
                            offset:0 atIndex:0];
                    [enc setBuffer:buffer_of(&weight)
                            offset:weight.device.offset atIndex:1];
                    [enc setBuffer:destination
                            offset:0 atIndex:2];
                    [enc setBytes:&gp
                           length:sizeof(gp) atIndex:3];
                    [enc setBuffer:activation_scales
                            offset:0 atIndex:4];
                    [enc setBuffer:expert_counts
                            offset:0 atIndex:5];
                    [enc setBuffer:expert_routes
                            offset:0 atIndex:6];
                    const NSUInteger projections =
                        paired_gate_up ? 2 : 1;
                    const NSUInteger output_tile =
                        paired_gate_up
                            ? MOLLM_GROUPED_MOE_GATE_UP_OUTPUT_TILE
                            : MOLLM_GROUPED_MOE_DOWN_OUTPUT_TILE;
                    const NSUInteger packed_weight_bytes =
                        projections *
                        (weight.group_size / 32) *
                        output_tile * 32 / 2;
                    const NSUInteger route_tile =
                        large_route_tile
                            ? MOLLM_GROUPED_MOE_ROUTE_TILE_LARGE
                            : MOLLM_GROUPED_MOE_ROUTE_TILE_SMALL;
                    const NSUInteger staging_bytes =
                        packed_weight_bytes +
                        4 * route_tile * 32 +
                        (projections * output_tile + route_tile) *
                            sizeof(float);
                    const NSUInteger paired_accumulator_bytes =
                        paired_gate_up
                            ? projections * output_tile *
                                  route_tile * sizeof(float)
                            : 0;
                    const NSUInteger total_threadgroup_bytes =
                        std::max(
                            staging_bytes,
                            paired_accumulator_bytes);
                    [enc setComputePipelineState:
                             pipelines_->grouped_moe(
                                 (int)weight.group_size,
                                 paired_gate_up,
                                 large_route_tile)];
                    [enc setBuffer:grouped_jobs
                            offset:large_route_tile
                                ? grouped_jobs_queue_bytes
                                : 0
                           atIndex:7];
                    [enc setThreadgroupMemoryLength:
                             total_threadgroup_bytes atIndex:0];
                    const NSUInteger indirect_record =
                        paired_gate_up
                            ? (large_route_tile ? 1 : 0)
                            : (large_route_tile ? 3 : 2);
                    const NSUInteger indirect_offset =
                        indirect_record * 3 * sizeof(uint32_t);
                    [enc dispatchThreadgroupsWithIndirectBuffer:
                             grouped_dispatch
                        indirectBufferOffset:indirect_offset
                        threadsPerThreadgroup:
                             MTLSizeMake(
                                 32 * MOLLM_GROUPED_MOE_SIMDGROUPS,
                                 1, 1)];
                };
                if (grouped_prefill) {
                    grouped_gemm(
                        qx, sx, gu, intermediate,
                        hidden_size, 2 * intermediate, true, true, false,
                        merged, intermediate);
                    grouped_gemm(
                        qx, sx, gu, intermediate,
                        hidden_size, 2 * intermediate, true, true, true,
                        merged, intermediate);
                } else {
                    selected_gemm(
                        qx,sx,gu,gu_rows,2*intermediate,
                        hidden_size,2*intermediate,seq,top_k,
                        merged,2*intermediate);
                }
                profile_resident_moe_stage("MOE.gate_up");
                if (native_down) {
                    [enc setComputePipelineState:
                             pipelines_->pipeline(
                                 grouped_prefill
                                     ? "moe_quantize_selected_blocks"
                                     : "moe_swiglu_quantize_blocks")];
                    [enc setBuffer:merged offset:0 atIndex:0];
                    [enc setBuffer:qi offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc setBuffer:si offset:0 atIndex:4];
                    [enc setThreadgroupMemoryLength:
                             grouped_prefill
                                 ? 0
                                 : 4 * sizeof(float)
                                            atIndex:0];
                    const NSUInteger blocks =
                        ((NSUInteger)intermediate + 127) / 128;
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (NSUInteger)selections * blocks,
                                 1, 1)
                        threadsPerThreadgroup:
                             grouped_prefill
                                 ? MTLSizeMake(32,1,1)
                                 : MTLSizeMake(32,4,1)];
                } else if (!grouped_prefill) {
                    if (native_down_group == 32) {
                        [enc setComputePipelineState:
                                 pipelines_->pipeline(
                                     "moe_swiglu_quantize_block32")];
                        [enc setBuffer:merged offset:0 atIndex:0];
                        [enc setBuffer:qi offset:0 atIndex:2];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                        [enc setBuffer:si offset:0 atIndex:4];
                        constexpr NSUInteger nsg = 4;
                        const NSUInteger blocks =
                            ((NSUInteger)intermediate + 31) / 32;
                        const NSUInteger block_groups =
                            (blocks + nsg - 1) / nsg;
                        [enc dispatchThreadgroups:
                                 MTLSizeMake(
                                     (NSUInteger)selections *
                                         block_groups,
                                     1,1)
                            threadsPerThreadgroup:
                                 MTLSizeMake(32,nsg,1)];
                    } else {
                        [enc setComputePipelineState:
                                 pipelines_->pipeline(
                                     "moe_swiglu_selected")];
                        [enc setBuffer:merged offset:0 atIndex:0];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                        const NSUInteger sw_n =
                            (NSUInteger)selections * intermediate;
                        [enc dispatchThreadgroups:
                                 MTLSizeMake((sw_n+255)/256,1,1)
                            threadsPerThreadgroup:
                                 MTLSizeMake(256,1,1)];
                        quantize(merged,0,selections,intermediate,
                                 2*intermediate,
                                 native_down_group,qi,si);
                    }
                } else {
                    quantize(merged,0,selections,intermediate,
                             intermediate,native_down_group,qi,si);
                }
                profile_resident_moe_stage("MOE.swiglu_quant");
                if (grouped_prefill) {
                    grouped_gemm(
                        qi, si, down, hidden_size,
                        intermediate, hidden_size, false, false, false,
                        selected, hidden_size);
                    grouped_gemm(
                        qi, si, down, hidden_size,
                        intermediate, hidden_size, false, false, true,
                        selected, hidden_size);
                } else {
                    selected_gemm(
                        qi,si,down,down_rows,hidden_size,intermediate,
                        hidden_size,selections,1,selected,hidden_size);
                }
                profile_resident_moe_stage("MOE.down");
                [enc setComputePipelineState:
                         pipelines_->pipeline(
                             "moe_combine_selected")];
                [enc setBuffer:selected offset:0 atIndex:0];
                [enc setBuffer:buffer_of(output)
                        offset:0 atIndex:2];
                [enc setBytes:&mp
                       length:sizeof(mp) atIndex:3];
                [enc setBuffer:tw offset:0 atIndex:4];
                if (seq == 1) {
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (hidden_size + 255) / 256,
                                 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(256, 1, 1)];
                } else {
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (hidden_size + 63) / 64,
                                 (seq + 3) / 4, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(64, 4, 1)];
                }
                commands_->pending_free.push_back({qx_h,qx_bytes});commands_->pending_free.push_back({sx_h,sx_bytes});
                commands_->pending_free.push_back({qi_h,qi_bytes});commands_->pending_free.push_back({si_h,si_bytes});
                commands_->pending_free.push_back(
                    {selected_h,selected_bytes});
                if (grouped_prefill) {
                    commands_->pending_free.push_back(
                        {expert_counts_h, expert_counts_bytes});
                    commands_->pending_free.push_back(
                        {expert_routes_h, expert_routes_bytes});
                    commands_->pending_free.push_back(
                        {grouped_jobs_h, grouped_jobs_bytes});
                    commands_->pending_free.push_back(
                        {grouped_job_count_h,
                         grouped_job_count_bytes});
                    commands_->pending_free.push_back(
                        {grouped_dispatch_h,
                         grouped_dispatch_bytes});
                }
                commands_->pending_free.push_back({idx_h,idx_bytes});commands_->pending_free.push_back({tw_h,tw_bytes});
                commands_->pending_free.push_back({logits_h,logits_bytes});commands_->pending_free.push_back({merged_h,merged_bytes});
                break;
            }
#endif

            [enc setComputePipelineState:pipelines_->pipeline("moe_gate_up_w4")];
            [enc setBuffer:buffer_of(&x) offset:0 atIndex:0];
            [enc setBuffer:buffer_of(&gu) offset:gu.device.offset atIndex:1];
            [enc setBuffer:merged offset:0 atIndex:2];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
            [enc setBuffer:buffer_of(&gu) offset:gu.device.offset+gu_rows*(hidden_size/2) atIndex:4];
            [enc setBuffer:idx offset:0 atIndex:5];
            [enc setThreadgroupMemoryLength:4*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((2*intermediate+3)/4,top_k,seq)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [enc setComputePipelineState:pipelines_->pipeline("moe_swiglu_selected")];
            [enc setBuffer:merged offset:0 atIndex:0];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
            NSUInteger sw_n=(NSUInteger)seq*top_k*intermediate;
            [enc dispatchThreadgroups:MTLSizeMake((sw_n+255)/256,1,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            [enc setComputePipelineState:pipelines_->pipeline("moe_down_combine_w4")];
            [enc setBuffer:merged offset:0 atIndex:0];
            [enc setBuffer:buffer_of(&down) offset:down.device.offset atIndex:1];
            [enc setBuffer:buffer_of(output) offset:0 atIndex:2];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
            [enc setBuffer:buffer_of(&down) offset:down.device.offset+down_rows*(intermediate/2) atIndex:4];
            [enc setBuffer:idx offset:0 atIndex:5];[enc setBuffer:tw offset:0 atIndex:6];
            [enc setThreadgroupMemoryLength:4*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((hidden_size+3)/4,seq,1)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            commands_->pending_free.push_back({idx_h,idx_bytes});
            commands_->pending_free.push_back({tw_h,tw_bytes});
            commands_->pending_free.push_back({logits_h,logits_bytes});
            commands_->pending_free.push_back({merged_h,merged_bytes});
            break;
        }

        // Generic correctness fallback for FP16/W8/shared-expert variants.
        if (commands_->enc) { [commands_->enc endEncoding]; commands_->enc = nil; }
        if (commands_->cmd) {
            [commands_->cmd commit];
            [commands_->cmd waitUntilCompleted];
            if (commands_->cmd.status == MTLCommandBufferStatusError) {
                NSError* e = commands_->cmd.error;
                fprintf(stderr, "MetalBackend: pre-MOE command buffer error: %s\n",
                        e ? e.localizedDescription.UTF8String : "?");
                dispatch_failed_ = true;
            }
            commands_->cmd = nil;
        }

        if (dispatch_failed_) {
            abort_dispatch = true;
            return true;
        }

        kernel_qwen3_moe(inputs, *output, thread_pool,
                         hidden_size, num_experts, top_k,
                         intermediate, shared_intermediate,
                         router_score_func, norm_topk, has_shared,
                         n_group, topk_group, routed_scale);

        commands_->cmd = [commands_->queue commandBuffer];
        commands_->enc = [commands_->cmd computeCommandEncoder];
        break;
    }

    default:
        return false;
    }
    return true;
}

bool MetalMoeOps::dispatch_host(
    const GraphNode& node, const std::vector<const Tensor*>& inputs,
    Tensor* output, ThreadPool* thread_pool, bool& success) {
    success = false;
    if (node.op_type != OpType::MOE || !output || inputs.size() < 7 ||
        !inputs[0] || !inputs[1] || !inputs[2] || !inputs[3] ||
        !inputs[4] || !inputs[5] || !inputs[6] ||
        inputs[0]->shape[1] != 1) {
        return false;
    }

    const OpParams& params = node.params;
    const int hidden_size = graph_params::get_i32(
        params, 0, static_cast<int>(output->shape[0]));
    const int num_experts = graph_params::get_i32(params, 1, 0);
    const int top_k = graph_params::get_i32(params, 2, 0);
    const int intermediate = graph_params::get_i32(params, 3, 0);
    const int shared_intermediate = graph_params::get_i32(
        params, 4, intermediate);
    const int score_func = graph_params::get_i32(params, 5, 0);
    const bool normalize_topk =
        graph_params::get_i32(params, 6, 1) != 0;
    const bool has_shared = graph_params::get_i32(params, 7, 1) != 0;
    const int num_groups = graph_params::get_i32(params, 8, 1);
    const int topk_groups = graph_params::get_i32(params, 9, 1);
    const bool shared_has_gate =
        graph_params::get_i32(params, 10, 1) != 0;
    const int bias_input = graph_params::get_i32(params, 11, -1);
    const int token_input = graph_params::get_i32(params, 12, -1);
    const int hash_input = graph_params::get_i32(params, 13, -1);
    const float routed_scale = graph_params::get_f32(params, 0, 1.0f);
    const float swiglu_limit = graph_params::get_f32(params, 1, 0.0f);

    const auto* gate_source = dynamic_cast<const MoeSsdTensorSource*>(
        inputs[2]->moe_ssd_source);
    const auto* down_source = dynamic_cast<const MoeSsdTensorSource*>(
        inputs[3]->moe_ssd_source);
    const bool supported =
        ssd_cache_ && ssd_cache_->available() &&
        gate_source && down_source &&
        gate_source->cache == down_source->cache &&
        gate_source->spec.precision == Precision::MXFP4 &&
        down_source->spec.precision == Precision::MXFP4 &&
        gate_source->spec.group_size == 32 &&
        down_source->spec.group_size == 32 && has_shared &&
        !shared_has_gate && hidden_size > 0 && num_experts > 0 &&
        top_k > 0 && top_k <= num_experts && intermediate > 0 &&
        output->shape[0] == hidden_size && output->shape[1] == 1;
    if (!supported)
        return false;

    const Tensor& hidden = *inputs[0];
    const Tensor& router = *inputs[1];
    const Tensor* bias =
        bias_input >= 0 && static_cast<size_t>(bias_input) < inputs.size()
            ? inputs[bias_input] : nullptr;
    const Tensor* token_ids =
        token_input >= 0 && static_cast<size_t>(token_input) < inputs.size()
            ? inputs[token_input] : nullptr;
    const Tensor* token_to_experts =
        hash_input >= 0 && static_cast<size_t>(hash_input) < inputs.size()
            ? inputs[hash_input] : nullptr;

    // The host-MoE path bypasses MetalBackend::dispatch(), so learn its layer
    // metadata here. From the second decode token onward this makes the next
    // layer's router available before its exact gate executes.
    ssd_moe_layers_[gate_source->spec.layer] = {
        &router,
        bias,
        gate_source,
        down_source,
        hidden_size,
        num_experts,
        top_k,
        intermediate,
        score_func,
        std::max(1, num_groups),
        std::max(1, topk_groups),
        normalize_topk,
        routed_scale,
        token_to_experts,
    };

    std::vector<float> logits(static_cast<size_t>(num_experts));
    Tensor logits_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        num_experts, 1, 1, 1, logits.data());
    kernel_matmul_fp32(
        hidden, router, logits_tensor, thread_pool,
        Activation::NONE, 0, -1, true);

    mollm::detail::MoeRoutingParams routing;
    routing.num_experts = num_experts;
    routing.top_k = top_k;
    routing.score_func = score_func;
    routing.normalize_topk = normalize_topk;
    routing.num_groups = std::max(1, num_groups);
    routing.topk_groups = std::max(1, topk_groups);
    routing.scaling_factor = routed_scale;
    std::vector<int> routes;
    std::vector<float> route_weights;
    bool routed = false;
    if (token_ids && token_to_experts &&
        token_ids->prec == Precision::INT32 &&
        token_to_experts->prec == Precision::INT32) {
        routed = mollm::detail::select_moe_hash_routes(
            logits.data(), 1, token_ids->ptr<int32_t>(),
            token_to_experts->ptr<int32_t>(),
            static_cast<int>(token_to_experts->shape[1]), routing,
            routes, route_weights);
    } else {
        routed = mollm::detail::select_moe_routes(
            logits.data(), 1,
            bias && bias->data ? bias->ptr<float>() : nullptr,
            routing, routes, route_weights);
    }
    if (!routed || static_cast<int>(routes.size()) != top_k ||
        static_cast<int>(route_weights.size()) != top_k) {
        success = false;
        return true;
    }

    std::vector<MetalSsdExpertCache::ExpertView> views;
    const uint64_t io_start = mollm_trace::now_ns();
    if (!ssd_cache_->acquire(
            gate_source->spec, down_source->spec, routes, views) ||
        static_cast<int>(views.size()) != top_k) {
        success = false;
        return true;
    }
    mollm_trace::record_duration(
        "metal.ssd", "mxfp4_demand_schedule", io_start,
        mollm_trace::now_ns(),
        "{\"layer\":" + std::to_string(gate_source->spec.layer) + "}",
        "good");

    if (ssd_cross_layer_prefetch_) {
        const auto next = ssd_moe_layers_.find(
            gate_source->spec.layer + 1);
        if (next != ssd_moe_layers_.end()) {
            const SsdMoeLayerInfo& info = next->second;
            if (info.router && info.gate_up && info.down &&
                info.hidden == hidden_size && info.experts > 0 &&
                info.top_k > 0 && info.top_k <= info.experts) {
                const uint64_t prefetch_start = mollm_trace::now_ns();
                std::vector<float> predicted_logits(
                    static_cast<size_t>(info.experts));
                Tensor predicted_logits_tensor = Tensor::create(
                    Precision::FP32, MemoryType::EXTERNAL,
                    info.experts, 1, 1, 1, predicted_logits.data());
                kernel_matmul_fp32(
                    hidden, *info.router, predicted_logits_tensor,
                    thread_pool, Activation::NONE, 0, -1, true);

                mollm::detail::MoeRoutingParams predicted_routing;
                predicted_routing.num_experts = info.experts;
                predicted_routing.top_k = info.top_k;
                predicted_routing.score_func = info.score_func;
                predicted_routing.normalize_topk = info.norm_topk;
                predicted_routing.num_groups = info.n_group;
                predicted_routing.topk_groups = info.topk_group;
                predicted_routing.scaling_factor = info.routed_scale;
                std::vector<int> predicted_routes;
                std::vector<float> predicted_weights;
                bool predicted = false;
                if (info.token_to_experts && token_ids &&
                    info.token_to_experts->prec == Precision::INT32 &&
                    token_ids->prec == Precision::INT32) {
                    predicted = mollm::detail::select_moe_hash_routes(
                        predicted_logits.data(), 1,
                        token_ids->ptr<int32_t>(),
                        info.token_to_experts->ptr<int32_t>(),
                        static_cast<int>(info.token_to_experts->shape[1]),
                        predicted_routing, predicted_routes,
                        predicted_weights);
                } else {
                    predicted = mollm::detail::select_moe_routes(
                        predicted_logits.data(), 1,
                        info.bias && info.bias->data
                            ? info.bias->ptr<float>() : nullptr,
                        predicted_routing, predicted_routes,
                        predicted_weights);
                }
                std::vector<MetalSsdExpertCache::ExpertView> predicted_views;
                if (predicted) {
                    ssd_cache_->acquire(
                        info.gate_up->spec, info.down->spec,
                        predicted_routes, predicted_views, true);
                }
                mollm_trace::record_duration(
                    "metal.ssd", "mxfp4_cross_layer_prefetch",
                    prefetch_start, mollm_trace::now_ns(),
                    "{\"layer\":" +
                        std::to_string(info.gate_up->spec.layer) +
                        ",\"experts\":" +
                        std::to_string(predicted_routes.size()) + "}",
                    "yellow");
            }
        }
    }

    const size_t hidden_bytes =
        static_cast<size_t>(hidden_size) * sizeof(float);
    const size_t merged_bytes =
        static_cast<size_t>(top_k) * 2 * intermediate * sizeof(float);
    const size_t selected_bytes =
        static_cast<size_t>(top_k) * hidden_size * sizeof(float);
    const size_t quantized_hidden_bytes = static_cast<size_t>(hidden_size);
    const size_t hidden_scale_bytes =
        static_cast<size_t>(hidden_size / 32) * sizeof(float);
    const size_t quantized_intermediate_bytes =
        static_cast<size_t>(top_k) * intermediate;
    const size_t intermediate_scale_bytes =
        static_cast<size_t>(top_k) * (intermediate / 32) * sizeof(float);
    const size_t weights_bytes =
        static_cast<size_t>(top_k) * sizeof(float);
    const size_t offsets_bytes =
        static_cast<size_t>(2 * top_k) * sizeof(uint64_t);
    const size_t indices_bytes =
        static_cast<size_t>(top_k) * sizeof(uint32_t);
    void* hidden_handle = pool_->acquire(hidden_bytes);
    void* merged_handle = pool_->acquire(merged_bytes);
    void* selected_handle = pool_->acquire(selected_bytes);
    void* quantized_hidden_handle =
        pool_->acquire(quantized_hidden_bytes);
    void* hidden_scale_handle = pool_->acquire(hidden_scale_bytes);
    void* residual_hidden_handle =
        pool_->acquire(quantized_hidden_bytes);
    void* residual_hidden_scale_handle =
        pool_->acquire(hidden_scale_bytes);
    void* quantized_intermediate_handle =
        pool_->acquire(quantized_intermediate_bytes);
    void* intermediate_scale_handle =
        pool_->acquire(intermediate_scale_bytes);
    void* residual_intermediate_handle =
        pool_->acquire(quantized_intermediate_bytes);
    void* residual_intermediate_scale_handle =
        pool_->acquire(intermediate_scale_bytes);
    void* weights_handle = pool_->acquire(weights_bytes);
    void* offsets_handle = pool_->acquire(offsets_bytes);
    void* indices_handle = pool_->acquire(indices_bytes);
    auto release_buffers = [&] {
        pool_->release(hidden_handle, hidden_bytes);
        pool_->release(merged_handle, merged_bytes);
        pool_->release(selected_handle, selected_bytes);
        pool_->release(
            quantized_hidden_handle, quantized_hidden_bytes);
        pool_->release(hidden_scale_handle, hidden_scale_bytes);
        pool_->release(
            residual_hidden_handle, quantized_hidden_bytes);
        pool_->release(
            residual_hidden_scale_handle, hidden_scale_bytes);
        pool_->release(
            quantized_intermediate_handle, quantized_intermediate_bytes);
        pool_->release(
            intermediate_scale_handle, intermediate_scale_bytes);
        pool_->release(
            residual_intermediate_handle, quantized_intermediate_bytes);
        pool_->release(
            residual_intermediate_scale_handle, intermediate_scale_bytes);
        pool_->release(weights_handle, weights_bytes);
        pool_->release(offsets_handle, offsets_bytes);
        pool_->release(indices_handle, indices_bytes);
    };
    if (!hidden_handle || !merged_handle || !selected_handle ||
        !quantized_hidden_handle || !hidden_scale_handle ||
        !residual_hidden_handle || !residual_hidden_scale_handle ||
        !quantized_intermediate_handle || !intermediate_scale_handle ||
        !residual_intermediate_handle ||
        !residual_intermediate_scale_handle ||
        !weights_handle || !offsets_handle || !indices_handle) {
        release_buffers();
        success = false;
        return true;
    }
    std::memcpy(
        MetalBufferPool::contents(hidden_handle), hidden.data, hidden_bytes);
    std::memcpy(
        MetalBufferPool::contents(weights_handle), route_weights.data(),
        weights_bytes);
    auto* offsets = static_cast<uint64_t*>(
        MetalBufferPool::contents(offsets_handle));
    auto* indices = static_cast<uint32_t*>(
        MetalBufferPool::contents(indices_handle));
    for (int selection = 0; selection < top_k; ++selection) {
        offsets[selection] = views[selection].gate_up_offset;
        offsets[top_k + selection] = views[selection].down_offset;
        indices[selection] = static_cast<uint32_t>(selection);
    }

    id<MTLBuffer> hidden_buffer =
        (__bridge id<MTLBuffer>)hidden_handle;
    id<MTLBuffer> merged_buffer =
        (__bridge id<MTLBuffer>)merged_handle;
    id<MTLBuffer> selected_buffer =
        (__bridge id<MTLBuffer>)selected_handle;
    id<MTLBuffer> quantized_hidden_buffer =
        (__bridge id<MTLBuffer>)quantized_hidden_handle;
    id<MTLBuffer> hidden_scale_buffer =
        (__bridge id<MTLBuffer>)hidden_scale_handle;
    id<MTLBuffer> residual_hidden_buffer =
        (__bridge id<MTLBuffer>)residual_hidden_handle;
    id<MTLBuffer> residual_hidden_scale_buffer =
        (__bridge id<MTLBuffer>)residual_hidden_scale_handle;
    id<MTLBuffer> quantized_intermediate_buffer =
        (__bridge id<MTLBuffer>)quantized_intermediate_handle;
    id<MTLBuffer> intermediate_scale_buffer =
        (__bridge id<MTLBuffer>)intermediate_scale_handle;
    id<MTLBuffer> residual_intermediate_buffer =
        (__bridge id<MTLBuffer>)residual_intermediate_handle;
    id<MTLBuffer> residual_intermediate_scale_buffer =
        (__bridge id<MTLBuffer>)residual_intermediate_scale_handle;
    id<MTLBuffer> route_weight_buffer =
        (__bridge id<MTLBuffer>)weights_handle;
    id<MTLBuffer> offset_buffer =
        (__bridge id<MTLBuffer>)offsets_handle;
    id<MTLBuffer> index_buffer =
        (__bridge id<MTLBuffer>)indices_handle;

    id<MTLCommandBuffer> command = [commands_->queue commandBuffer];
    command.label = @"mollm hybrid MXFP4 SSD MoE";
    std::unordered_set<void*> waited_events;
    for (const auto& view : views) {
        if (view.gate_ready_event &&
            view.gate_ready_event.signaledValue < view.gate_ready_value) {
            void* key = (__bridge void*)view.gate_ready_event;
            if (waited_events.insert(key).second) {
                [command encodeWaitForEvent:view.gate_ready_event
                                      value:view.gate_ready_value];
            }
        }
    }

    MoeW4Params moe_params{};
    moe_params.hidden = hidden_size;
    moe_params.experts = num_experts;
    moe_params.top_k = top_k;
    moe_params.intermediate = intermediate;
    moe_params.seq_len = 1;
    moe_params.routed_scale = routed_scale;
    moe_params.swiglu_limit = swiglu_limit;

    SelectedMxfp4Params gate_params{};
    gate_params.selections = top_k;
    gate_params.N = 2 * intermediate;
    gate_params.K = hidden_size;
    gate_params.c_row_stride = 2 * intermediate;
    gate_params.groups_per_row = hidden_size / 32;
    gate_params.activation_repeat = top_k;
    gate_params.activation_row_stride = hidden_size;

    id<MTLComputeCommandEncoder> encoder =
        [command computeCommandEncoder];
    QuantActParams hidden_quant{};
    hidden_quant.M = 1;
    hidden_quant.K = hidden_size;
    hidden_quant.a_row_stride = hidden_size;
    hidden_quant.block_size = 32;
    [encoder setComputePipelineState:
                 pipelines_->pipeline("quantize_act_fp8_i8_block32")];
    [encoder setBuffer:hidden_buffer offset:0 atIndex:0];
    [encoder setBuffer:quantized_hidden_buffer offset:0 atIndex:2];
    [encoder setBytes:&hidden_quant length:sizeof(hidden_quant) atIndex:3];
    [encoder setBuffer:hidden_scale_buffer offset:0 atIndex:4];
    [encoder setBuffer:residual_hidden_buffer offset:0 atIndex:5];
    [encoder setBuffer:residual_hidden_scale_buffer offset:0 atIndex:6];
    constexpr NSUInteger quant_simdgroups = 4;
    const NSUInteger hidden_blocks =
        (static_cast<NSUInteger>(hidden_size) + 127) / 128;
    [encoder dispatchThreadgroups:
                 MTLSizeMake(hidden_blocks, 1, 1)
             threadsPerThreadgroup:
                 MTLSizeMake(32, quant_simdgroups, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:
                 pipelines_->pipeline(
                     "gemv_selected_slots_mxfp4_i8a_f32c")];
    [encoder setBuffer:quantized_hidden_buffer offset:0 atIndex:0];
    [encoder setBuffer:views.front().gate_up offset:0 atIndex:1];
    [encoder setBuffer:merged_buffer offset:0 atIndex:2];
    [encoder setBytes:&gate_params length:sizeof(gate_params) atIndex:3];
    [encoder setBuffer:hidden_scale_buffer offset:0 atIndex:4];
    [encoder setBuffer:residual_hidden_buffer offset:0 atIndex:5];
    [encoder setBuffer:offset_buffer offset:0 atIndex:6];
    [encoder setBuffer:index_buffer offset:0 atIndex:7];
    [encoder setBuffer:residual_hidden_scale_buffer offset:0 atIndex:8];
    [encoder setThreadgroupMemoryLength:
                 static_cast<NSUInteger>(
                     2 * hidden_size +
                     2 * (hidden_size / 32) * sizeof(float))
                              atIndex:0];
    [encoder dispatchThreadgroups:
                 MTLSizeMake((2 * intermediate + 63) / 64, 1, top_k)
             threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [encoder setComputePipelineState:
                 pipelines_->pipeline("moe_swiglu_route_bf16")];
    [encoder setBuffer:merged_buffer offset:0 atIndex:0];
    [encoder setBytes:&moe_params length:sizeof(moe_params) atIndex:3];
    [encoder setBuffer:route_weight_buffer offset:0 atIndex:4];
    [encoder dispatchThreads:
                 MTLSizeMake(
                     static_cast<NSUInteger>(top_k) * intermediate,
                     1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    QuantActParams intermediate_quant{};
    intermediate_quant.M = top_k;
    intermediate_quant.K = intermediate;
    intermediate_quant.a_row_stride = 2 * intermediate;
    intermediate_quant.block_size = 32;
    [encoder setComputePipelineState:
                 pipelines_->pipeline("quantize_act_fp8_i8_block32")];
    [encoder setBuffer:merged_buffer offset:0 atIndex:0];
    [encoder setBuffer:quantized_intermediate_buffer offset:0 atIndex:2];
    [encoder setBytes:&intermediate_quant
                length:sizeof(intermediate_quant) atIndex:3];
    [encoder setBuffer:intermediate_scale_buffer offset:0 atIndex:4];
    [encoder setBuffer:residual_intermediate_buffer offset:0 atIndex:5];
    [encoder setBuffer:residual_intermediate_scale_buffer
                 offset:0 atIndex:6];
    const NSUInteger intermediate_fp8_blocks =
        (static_cast<NSUInteger>(intermediate) + 127) / 128;
    [encoder dispatchThreadgroups:
                 MTLSizeMake(
                     static_cast<NSUInteger>(top_k) *
                         intermediate_fp8_blocks,
                     1, 1)
             threadsPerThreadgroup:
                 MTLSizeMake(32, quant_simdgroups, 1)];
    [encoder endEncoding];

    waited_events.clear();
    for (const auto& view : views) {
        if (view.down_ready_event &&
            view.down_ready_event.signaledValue < view.down_ready_value) {
            void* key = (__bridge void*)view.down_ready_event;
            if (waited_events.insert(key).second) {
                [command encodeWaitForEvent:view.down_ready_event
                                      value:view.down_ready_value];
            }
        }
    }

    SelectedMxfp4Params down_params{};
    down_params.selections = top_k;
    down_params.N = hidden_size;
    down_params.K = intermediate;
    down_params.c_row_stride = hidden_size;
    down_params.groups_per_row = intermediate / 32;
    down_params.activation_repeat = 1;
    down_params.activation_row_stride = intermediate;
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:
                 pipelines_->pipeline(
                     "gemv_selected_slots_mxfp4_i8a_f32c")];
    [encoder setBuffer:quantized_intermediate_buffer offset:0 atIndex:0];
    [encoder setBuffer:views.front().down offset:0 atIndex:1];
    [encoder setBuffer:selected_buffer offset:0 atIndex:2];
    [encoder setBytes:&down_params length:sizeof(down_params) atIndex:3];
    [encoder setBuffer:intermediate_scale_buffer offset:0 atIndex:4];
    [encoder setBuffer:residual_intermediate_buffer offset:0 atIndex:5];
    [encoder setBuffer:offset_buffer
                 offset:static_cast<NSUInteger>(top_k) * sizeof(uint64_t)
                atIndex:6];
    [encoder setBuffer:index_buffer offset:0 atIndex:7];
    [encoder setBuffer:residual_intermediate_scale_buffer
                 offset:0 atIndex:8];
    [encoder setThreadgroupMemoryLength:
                 static_cast<NSUInteger>(
                     2 * intermediate +
                     2 * (intermediate / 32) * sizeof(float))
                              atIndex:0];
    [encoder dispatchThreadgroups:
                 MTLSizeMake((hidden_size + 63) / 64, 1, top_k)
             threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [encoder endEncoding];
    const uint64_t gpu_start = mollm_trace::now_ns();
    [command commit];

    std::vector<float> shared_values(static_cast<size_t>(hidden_size));
    Tensor shared_output = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        hidden_size, 1, 1, 1, shared_values.data());
    const bool shared_ok = kernel_moe_shared_expert(
        hidden, *inputs[4], *inputs[5], *inputs[6], nullptr,
        shared_output, thread_pool, shared_intermediate, true,
        swiglu_limit);

    [command waitUntilCompleted];
    const uint64_t gpu_end = mollm_trace::now_ns();
    mollm_trace::record_duration(
        "metal.ssd", "mxfp4_routed_experts", gpu_start, gpu_end,
        "{\"layer\":" + std::to_string(gate_source->spec.layer) + "}",
        "thread_state_running");
    const double gpu_seconds = command.GPUEndTime - command.GPUStartTime;
    if (gpu_seconds > 0.0 && gpu_end != 0) {
        const uint64_t gpu_ns =
            static_cast<uint64_t>(gpu_seconds * 1e9);
        mollm_trace::record_duration(
            "metal.ssd", "mxfp4_gpu",
            gpu_end > gpu_ns ? gpu_end - gpu_ns : 0, gpu_end,
            "{\"layer\":" + std::to_string(gate_source->spec.layer) + "}",
            "thread_state_running");
    }
    if (!shared_ok || command.status == MTLCommandBufferStatusError) {
        if (command.status == MTLCommandBufferStatusError) {
            NSError* error = command.error;
            std::fprintf(
                stderr, "MetalBackend: hybrid MXFP4 MoE failed: %s\n",
                error ? error.localizedDescription.UTF8String : "?");
        }
        release_buffers();
        success = false;
        return true;
    }

    const float* selected = static_cast<const float*>(
        MetalBufferPool::contents(selected_handle));
    float* destination = output->ptr<float>();
    std::fill(destination, destination + hidden_size, 0.0f);
    std::vector<int> order(static_cast<size_t>(top_k));
    for (int selection = 0; selection < top_k; ++selection)
        order[selection] = selection;
    std::stable_sort(
        order.begin(), order.end(),
        [&](int left, int right) { return routes[left] < routes[right]; });
    for (int selection : order) {
        const float* source =
            selected + static_cast<size_t>(selection) * hidden_size;
        for (int dim = 0; dim < hidden_size; ++dim)
            destination[dim] += source[dim];
    }
    for (int dim = 0; dim < hidden_size; ++dim)
        destination[dim] += shared_values[dim];
    release_buffers();
    success = true;
    return true;
}
