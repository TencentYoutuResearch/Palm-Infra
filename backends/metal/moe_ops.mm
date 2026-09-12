#include "backends/metal/moe_ops.h"

#include "backends/metal/buffer_pool.h"
#include "backends/metal/command_context.h"
#include "backends/metal/pipeline_cache.h"
#include "backends/metal/ssd_expert_cache.h"
#include "backends/metal/ssd_shared_expert.h"
#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/moe/moe_routing.h"
#include "runtime/trace.h"
#include "storage/ssd_expert_cache/cache.h"

#include <cstdio>
#include <cstring>
#include <vector>


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
