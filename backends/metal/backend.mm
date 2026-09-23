#include "backends/metal/backend.h"
#include "backends/metal/attention_ops.h"
#include "backends/metal/buffer_pool.h"
#include "backends/metal/command_context.h"
#include "backends/metal/elementwise_ops.h"
#include "backends/metal/layout_ops.h"
#include "backends/metal/lm_head.h"
#include "backends/metal/matmul_ops.h"
#include "backends/metal/moe_ops.h"
#include "backends/metal/normalization_ops.h"
#include "backends/metal/pipeline_cache.h"
#include "backends/metal/recurrent_ops.h"
#include "backends/metal/resource_store.h"
#include "backends/metal/rotary_ops.h"
#include "graph/graph.h"
#include "kernels/cpu/matmul/matmul.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef MOLLM_METALLIB_PATH
#define MOLLM_METALLIB_PATH ""
#endif

// ===========================================================================
// MetalBackend::Impl
// ===========================================================================

struct MetalBackend::Impl {
    id<MTLDevice>            device = nil;
    id<MTLLibrary>           library = nil;

    std::unique_ptr<MetalBufferPool> pool;
    std::unique_ptr<MetalCommandContext> commands;
    std::unique_ptr<MetalPipelineCache> pipeline_cache;
    std::unique_ptr<MetalAttentionOps> attention_ops;
    std::unique_ptr<MetalElementwiseOps> elementwise_ops;
    std::unique_ptr<MetalLayoutOps> layout_ops;
    std::unique_ptr<MetalLmHead> lm_head;
    std::unique_ptr<MetalMatmulOps> matmul_ops;
    std::unique_ptr<MetalMoeOps> moe_ops;
    std::unique_ptr<MetalNormalizationOps> normalization_ops;
    std::unique_ptr<MetalRecurrentOps> recurrent_ops;
    std::unique_ptr<MetalRotaryOps> rotary_ops;
    std::unique_ptr<MetalResourceStore> resources;

    bool                     dispatch_failed = false;

    // True iff the tensor-API GEMM kernel is compiled AND the GPU supports the
    // Metal 4 tensor family (M5/A19+). Set in the constructor.
    bool has_tensor = false;

    bool ok = false;

    id<MTLComputePipelineState> pipeline(const char* name) {
        return pipeline_cache->pipeline(name);
    }
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

namespace {

int metal_cmd_chunk_ops() {
    static const int chunk = [] {
        const char* value = std::getenv("MOLLM_METAL_CMD_CHUNK");
        // Count only dispatches that actually encode GPU work. Forty keeps
        // enough work queued while allowing CPU graph encoding to overlap the
        // submitted prefix on the small decode graphs. With lm_head appended
        // to the graph tail, 32 keeps the encoder closer to the GPU without
        // fragmenting the queue into overly small command buffers.
        if (!value) return 32;
        return std::max(0, std::atoi(value));
    }();
    return chunk;
}

} // namespace

// ===========================================================================
// construction
// ===========================================================================

MetalBackend::MetalBackend(const std::string& metallib_path) : impl_(new Impl) {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            fprintf(stderr, "MetalBackend: no Metal device\n");
            return;
        }
        id<MTLCommandQueue> queue = [impl_->device newCommandQueue];

        NSError* err = nil;
        std::string path = metallib_path.empty() ? std::string(MOLLM_METALLIB_PATH)
                                                  : metallib_path;
        if (!path.empty()) {
            NSString* p = @(path.c_str());
            impl_->library = [impl_->device newLibraryWithURL:[NSURL fileURLWithPath:p]
                                                        error:&err];
        }
        if (!impl_->library) {
            fprintf(stderr, "MetalBackend: failed to load metallib '%s': %s\n",
                    path.c_str(), err ? err.localizedDescription.UTF8String : "no path");
            return;
        }
        impl_->pipeline_cache.reset(new MetalPipelineCache(
            (__bridge void*)impl_->device,
            (__bridge void*)impl_->library));
        impl_->elementwise_ops.reset(
            new MetalElementwiseOps(impl_->pipeline_cache.get()));
        impl_->layout_ops.reset(
            new MetalLayoutOps(impl_->pipeline_cache.get()));
        impl_->normalization_ops.reset(
            new MetalNormalizationOps(impl_->pipeline_cache.get()));
        impl_->pool.reset(new MetalBufferPool((__bridge void*)impl_->device));
        impl_->commands.reset(new MetalCommandContext(
            (__bridge void*)queue, impl_->pool.get()));
        impl_->attention_ops.reset(new MetalAttentionOps(
            impl_->pipeline_cache.get(), impl_->pool.get(),
            impl_->commands.get()));
        impl_->recurrent_ops.reset(new MetalRecurrentOps(
            impl_->pipeline_cache.get(), impl_->pool.get(),
            impl_->commands.get()));
        impl_->rotary_ops.reset(
            new MetalRotaryOps(impl_->pipeline_cache.get()));
        impl_->lm_head.reset(new MetalLmHead(
            impl_->pool.get(), impl_->commands.get(),
            impl_->pipeline_cache.get(), impl_->dispatch_failed));
        impl_->resources.reset(
            new MetalResourceStore((__bridge void*)impl_->device));
        impl_->matmul_ops.reset(new MetalMatmulOps(
            impl_->pipeline_cache.get(), impl_->pool.get(),
            impl_->commands.get(), impl_->resources.get()));
        impl_->moe_ops.reset(new MetalMoeOps(
            (__bridge void*)impl_->device, impl_->pipeline_cache.get(),
            impl_->pool.get(), impl_->commands.get(),
            impl_->dispatch_failed));

        // Enable the tensor-API GEMM only if the kernel was compiled (metallib
        // built with -DMOLLM_METAL_TENSOR) AND the GPU is M5/A19+ (MTLGPUFamily
        // Metal4), and the pipeline actually loads.
#ifdef MOLLM_METAL_TENSOR
        bool fam = false;
        if (@available(macOS 15.0, *)) {
            fam = [impl_->device supportsFamily:MTLGPUFamilyMetal4];
        }
        // Metal 4 tensor-API GEMM is correct (parity-tested) and ~2.3x faster
        // than the simdgroup path (prefill 940 vs 403 t/s). Enable it whenever
        // the device and compiled pipeline support it.
        if (fam &&
            impl_->pipeline("gemm_tensor_direct_f16a_f16b_f32c") != nil) {
            impl_->has_tensor = true;
        }
        if (getenv("MOLLM_METAL_DEBUG"))
            fprintf(stderr, "MetalBackend: tensor GEMM %s\n",
                    impl_->has_tensor ? "ENABLED" : "disabled");
#endif
        impl_->ok = true;
    }
}

MetalBackend::~MetalBackend() {
    if (impl_) {
        dump_profile();  // report per-op GPU time table if MOLLM_METAL_PROFILE
        impl_->moe_ops.reset();
        impl_->lm_head.reset();
        impl_->matmul_ops.reset();
        impl_->attention_ops.reset();
        impl_->elementwise_ops.reset();
        impl_->layout_ops.reset();
        impl_->normalization_ops.reset();
        impl_->recurrent_ops.reset();
        impl_->rotary_ops.reset();
        impl_->commands.reset();
        impl_->pipeline_cache.reset();
        impl_->resources.reset();
        impl_->pool.reset();
    }
}

bool MetalBackend::available() const { return impl_ && impl_->ok; }

void MetalBackend::begin_execution() {
    // Host routing and host-resident MoE paths share the CPU quantization cache.
    matmul_reset_activation_cache();
}

void MetalBackend::clear_dispatch_error() {
    impl_->dispatch_failed = false;
}

bool MetalBackend::dispatch_failed() const {
    return impl_->dispatch_failed;
}

void MetalBackend::lm_head_gemv(
    const float* activation_host, const Tensor& weight, float* output_host,
    int n, int k, int activation) {
    impl_->lm_head->gemv(
        activation_host, weight, output_host, n, k, activation);
}

bool MetalBackend::lm_head_small_batch(
    const float* activation_host, const Tensor& weight, float* output_host,
    int m, int n, int k, int activation) {
    return impl_->lm_head->small_batch(
        activation_host, weight, output_host, m, n, k, activation);
}

bool MetalBackend::lm_head_small_batch_device_and_end_graph(
    const Tensor& activation, const Tensor& weight, float* output_host,
    int m, int n, int k, int activation_kind) {
    return impl_->lm_head->small_batch_device_and_end_graph(
        activation, weight, output_host, m, n, k, activation_kind);
}

bool MetalBackend::lm_head_small_batch_argmax_device_and_end_graph(
    const Tensor& activation, const Tensor& weight, int* top1_output,
    int m, int n, int k, int activation_kind) {
    return impl_->lm_head->small_batch_argmax_device_and_end_graph(
        activation, weight, top1_output, m, n, k, activation_kind);
}

void MetalBackend::lm_head_gemv_device_and_end_graph(
    const Tensor& activation, size_t activation_element_offset,
    const Tensor& weight, float* output_host, int n, int k,
    int activation_kind) {
    impl_->lm_head->gemv_device_and_end_graph(
        activation, activation_element_offset, weight, output_host,
        n, k, activation_kind);
}

int MetalBackend::lm_head_argmax_device_and_end_graph(
    const Tensor& activation, size_t activation_element_offset,
    const Tensor& weight, int n, int k, int activation_kind,
    Tensor* hidden_copy) {
    return impl_->lm_head->argmax_device_and_end_graph(
        activation, activation_element_offset, weight, n, k,
        activation_kind, hidden_copy);
}

// ===========================================================================
// weight region + persistent buffers
// ===========================================================================

bool MetalBackend::has_tensor_path() const {
    return impl_->has_tensor;
}

bool MetalBackend::register_weight_region(void* base, size_t size) {
    return impl_->ok && impl_->resources->register_weight_region(base, size);
}

void MetalBackend::enable_weight_copy_mode() {
    if (impl_->resources) impl_->resources->enable_weight_copy_mode();
}

bool MetalBackend::has_weight_copies() const {
    return impl_->resources && impl_->resources->has_weight_copies();
}

bool MetalBackend::configure_moe_ssd_io(
        const std::string& package_path, size_t capacity_bytes,
        int max_commands_in_flight, bool cross_layer_prefetch) {
    return impl_->ok && impl_->moe_ops &&
        impl_->moe_ops->configure_ssd_io(
            package_path, capacity_bytes, max_commands_in_flight,
            cross_layer_prefetch);
}

void MetalBackend::wrap_weight(Tensor& t) {
    if (impl_->resources) impl_->resources->wrap_weight(t);
}

void MetalBackend::wrap_weight_int8(Tensor& t) {
    if (impl_->resources) impl_->resources->wrap_weight_int8(t);
}

void MetalBackend::wrap_weight_int4(Tensor& t, bool keep_native_experts) {
    if (impl_->resources)
        impl_->resources->wrap_weight_int4(t, keep_native_experts);
}

void MetalBackend::alloc_persistent(
    Tensor& t, size_t nbytes, PersistentHostAccess host_access,
    size_t host_prefix_bytes) {
    // Metal currently backs every persistent allocation with a
    // MTLResourceStorageModeShared buffer, and MetalResourceStore::alloc_persistent
    // exposes its contents through t.data. The host can therefore read and write
    // any part of the allocation, which satisfies all PersistentHostAccess modes
    // equally. host_access and host_prefix_bytes are deliberately unused rather
    // than forgotten: the transfer methods below still keep the host and device
    // views coherent, so callers must go through them regardless.
    //
    // If a future backend moves KV/state storage to Private or Managed mode,
    // this is the place that must start honouring host_prefix_bytes (to size a
    // host mirror) and must stop publishing t.data for NONE.
    (void)host_access;
    (void)host_prefix_bytes;
    if (impl_->resources) impl_->resources->alloc_persistent(t, nbytes);
}

void MetalBackend::upload_input(Tensor& t, const std::string& key,
                                const void* host_src, size_t nbytes) {
    if (impl_->resources)
        impl_->resources->upload_input(t, key, host_src, nbytes);
}

void MetalBackend::upload_zero_input(Tensor& t, const std::string& key,
                                     size_t nbytes) {
    if (impl_->resources)
        impl_->resources->upload_zero_input(t, key, nbytes);
}

// ===========================================================================
// allocation hooks
// ===========================================================================

void* MetalBackend::alloc_output(Tensor& out, size_t nbytes, BufferPool* /*pool*/) {
    void* buf = impl_->pool->acquire(nbytes);
    if (!buf) return nullptr;
    out.device.buffer = buf;
    out.device.offset = 0;
    out.mem_type = MemoryType::POOLED;
    out.owner_id = 0;   // device pool; executor skips host owner-id checks
    out.storage_id = 0;
    // Provide a real host pointer (Shared storage) so out.data != nullptr and
    // boundary readback / debug diffing work.
    out.data = MetalBufferPool::contents(buf);
    return out.data;
}

void MetalBackend::free_output(Tensor& t, BufferPool* /*pool*/) {
    // The whole graph is encoded into one command buffer and executed lazily at
    // end_graph(). Releasing a buffer to the pool now would let a later node
    // reacquire and overwrite it while earlier (not-yet-executed) kernels still
    // depend on its contents. Defer all frees until after waitUntilCompleted.
    impl_->commands->release_or_defer(t.device.buffer, t.nbytes());
}

// ===========================================================================
// command buffer lifecycle
// ===========================================================================

void MetalBackend::begin_graph() {
    impl_->commands->begin_graph();
}

void MetalBackend::synchronize_for_host_read() {
    if (impl_->commands)
        impl_->commands->synchronize_for_host_read(impl_->dispatch_failed);
}

// Debug: commit + wait after each op so intermediate device buffers are
// host-readable for per-node CPU/Metal diffing. Enabled by MOLLM_METAL_SYNC_EACH.
void MetalBackend::sync_point() {
    if (impl_->commands)
        impl_->commands->sync_point(impl_->dispatch_failed);
}

void MetalBackend::dump_profile() {
    if (impl_->commands) impl_->commands->dump_profile();
}

void MetalBackend::end_graph() {
    if (impl_->commands)
        impl_->commands->end_graph(impl_->dispatch_failed);
}

// ===========================================================================
// dispatch
// ===========================================================================

void MetalBackend::dispatch(const GraphNode& node,
                            const std::vector<const Tensor*>& inputs,
                            Tensor* output, ThreadPool* thread_pool) {
    // Hybrid CPU/Metal operators may synchronize for a host read in the
    // middle of a graph, which intentionally closes the current command
    // buffer. Resume GPU encoding lazily for the next device operation.
    id<MTLComputeCommandEncoder> enc = impl_->commands->ensure_encoder();
    const OpType op = node.op_type;
    std::string profile_label = op_type_name(op);
    bool encoded_gpu_work = true;

    bool handled = impl_->layout_ops && impl_->layout_ops->dispatch(
        node, inputs, output, enc, encoded_gpu_work);
    if (!handled && impl_->elementwise_ops) {
        handled = impl_->elementwise_ops->dispatch(
            node, inputs, output, enc);
    }
    if (!handled && impl_->normalization_ops) {
        handled = impl_->normalization_ops->dispatch(
            node, inputs, output, enc);
    }
    if (!handled && impl_->recurrent_ops) {
        handled = impl_->recurrent_ops->dispatch(
            node, inputs, output, enc, profile_label);
    }
    if (!handled && impl_->rotary_ops) {
        handled = impl_->rotary_ops->dispatch(node, inputs, output, enc);
    }
    if (!handled && impl_->attention_ops) {
        handled = impl_->attention_ops->dispatch(
            node, inputs, output, enc, profile_label);
    }
    if (!handled && impl_->matmul_ops) {
        handled = impl_->matmul_ops->dispatch(
            node, inputs, output, enc, impl_->has_tensor, profile_label);
    }
    bool abort_dispatch = false;
    if (!handled && impl_->moe_ops) {
        handled = impl_->moe_ops->dispatch(
            node, inputs, output, enc, thread_pool, impl_->has_tensor,
            profile_label, abort_dispatch);
    }
    if (abort_dispatch)
        return;
    if (!handled) {
        fprintf(stderr, "MetalBackend: unsupported op %d\n", (int)op);
        assert(false && "unsupported metal op");
    }
    // Per-op flush: debug diffing (MOLLM_METAL_SYNC_EACH) and/or per-op GPU
    // timing (MOLLM_METAL_PROFILE). Both need each op in its own command buffer.
    if (impl_->commands->profile) {
        if (impl_->commands->enc) { [impl_->commands->enc endEncoding]; impl_->commands->enc = nil; }
        if (impl_->commands->cmd) {
            [impl_->commands->cmd commit];
            [impl_->commands->cmd waitUntilCompleted];
            double gpu_ms = (impl_->commands->cmd.GPUEndTime - impl_->commands->cmd.GPUStartTime) * 1000.0;
            auto& st = impl_->commands->op_stats[profile_label];
            st.gpu_ms += gpu_ms;
            st.calls  += 1;
            impl_->commands->cmd = nil;
        }
        impl_->commands->cmd = [impl_->commands->queue commandBuffer];
        impl_->commands->enc = [impl_->commands->cmd computeCommandEncoder];
    } else {
        sync_point();  // no-op unless MOLLM_METAL_SYNC_EACH (per-op debug flush)
        const int chunk_ops = metal_cmd_chunk_ops();
        if (impl_->commands->chunk_graph && chunk_ops > 0 &&
            !getenv("MOLLM_METAL_SYNC_EACH") &&
            !getenv("MOLLM_METAL_GPU_TIME") &&
            encoded_gpu_work && ++impl_->commands->ops_in_cmd >= chunk_ops) {
            // Submit a prefix without waiting. Command buffers from one queue
            // execute in order, so later graph nodes retain their dependencies
            // while CPU encoding overlaps execution of the submitted prefix.
            [impl_->commands->enc endEncoding];
            impl_->commands->enc = nil;
            [impl_->commands->cmd commit];
            impl_->commands->cmd = [impl_->commands->queue commandBuffer];
            impl_->commands->enc = [impl_->commands->cmd computeCommandEncoder];
            impl_->commands->ops_in_cmd = 0;
        }
    }
}

bool MetalBackend::dispatch_host_moe(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* thread_pool, bool& success) {
    if (!impl_->moe_ops) {
        success = false;
        return false;
    }
    return impl_->moe_ops->dispatch_host(
        node, inputs, output, thread_pool, success);
}
