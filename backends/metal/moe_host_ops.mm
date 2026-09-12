#include "backends/metal/moe_ops.h"

#include "backends/metal/buffer_pool.h"
#include "backends/metal/command_context.h"
#include "backends/metal/pipeline_cache.h"
#include "backends/metal/ssd_expert_cache.h"
#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/moe/moe.h"
#include "kernels/cpu/moe/moe_routing.h"
#include "kernels/metal/metal_common.h"
#include "runtime/trace.h"
#include "storage/ssd_expert_cache/cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

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
