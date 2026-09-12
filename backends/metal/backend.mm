#include "backends/metal/backend.h"
#include "backends/metal/buffer_pool.h"
#include "backends/metal/command_context.h"
#include "backends/metal/dispatch_tuning.h"
#include "backends/metal/layout_ops.h"
#include "backends/metal/lm_head.h"
#include "backends/metal/pipeline_cache.h"
#include "backends/metal/resource_store.h"
#include "backends/metal/ssd_expert_cache.h"
#include "backends/metal/ssd_shared_expert.h"
#include "graph/graph.h"
#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/moe/moe.h"
#include "kernels/cpu/moe/moe_routing.h"
#include "storage/ssd_expert_cache/cache.h"
#include "kernels/metal/metal_common.h"
#include "runtime/trace.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    std::unique_ptr<MetalLayoutOps> layout_ops;
    std::unique_ptr<MetalLmHead> lm_head;
    std::unique_ptr<MetalResourceStore> resources;
    std::unique_ptr<MetalSsdExpertCache> ssd_cache;
    std::unique_ptr<MetalSsdSharedExpert> ssd_shared_expert;

    bool                     dispatch_failed = false;

    bool ssd_cross_layer_prefetch = true;
    struct SsdMoeLayerInfo {
        const Tensor* router = nullptr;
        const Tensor* bias = nullptr;
        const MoeSsdTensorSource* gate_up = nullptr;
        const MoeSsdTensorSource* down = nullptr;
        int hidden = 0;
        int experts = 0;
        int top_k = 0;
        int intermediate = 0;
        int score_func = 0;
        int n_group = 1;
        int topk_group = 1;
        bool norm_topk = true;
        float routed_scale = 1.0f;
        const Tensor* token_to_experts = nullptr;
    };
    std::unordered_map<int, SsdMoeLayerInfo> ssd_moe_layers;

    // True iff the tensor-API GEMM kernel is compiled AND the GPU supports the
    // Metal 4 tensor family (M5/A19+). Set in the constructor.
    bool has_tensor = false;

    bool ok = false;

    bool finish_ssd_prefix(int layer, const char* error_context) {
        [commands->enc endEncoding];
        commands->enc = nil;
        const uint64_t wait_start = mollm_trace::now_ns();
        [commands->cmd commit];
        [commands->cmd waitUntilCompleted];
        const uint64_t wait_end = mollm_trace::now_ns();
        const std::string args =
            "{\"layer\":" + std::to_string(layer) + "}";
        mollm_trace::record_duration(
            "metal.ssd", "prefix_wait", wait_start, wait_end, args,
            "thread_state_iowait");
        const double gpu_seconds =
            commands->cmd.GPUEndTime - commands->cmd.GPUStartTime;
        if (gpu_seconds > 0.0 && wait_end != 0) {
            const uint64_t gpu_ns =
                static_cast<uint64_t>(gpu_seconds * 1e9);
            mollm_trace::record_duration(
                "metal.ssd", "prefix_gpu",
                wait_end > gpu_ns ? wait_end - gpu_ns : 0,
                wait_end, args, "thread_state_running");
        }
        if (commands->cmd.status == MTLCommandBufferStatusError) {
            NSError* error = commands->cmd.error;
            fprintf(stderr, "MetalBackend: %s failed: %s\n", error_context,
                    error ? error.localizedDescription.UTF8String : "?");
            commands->cmd = nil;
            return false;
        }
        commands->cmd = nil;
        return true;
    }

    bool route_moe_on_cpu(const Tensor& input, const Tensor& router,
                          const Tensor* bias,
                          const mollm::detail::MoeRoutingParams& routing,
                          ThreadPool* thread_pool, void* indices_handle,
                          void* weights_handle) {
        if (!router.data)
            return false;
        const int seq = static_cast<int>(input.shape[1]);
        std::vector<float> logits(
            static_cast<size_t>(seq) * routing.num_experts);
        Tensor output = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, routing.num_experts,
            seq, 1, 1, logits.data());
        kernel_matmul_fp32(input, router, output, thread_pool,
                          Activation::NONE, 0, -1, true);
        std::vector<int> indices;
        std::vector<float> weights;
        const float* bias_data =
            bias && bias->data ? bias->ptr<float>() : nullptr;
        if (!mollm::detail::select_moe_routes(
                logits.data(), seq, bias_data, routing, indices, weights)) {
            return false;
        }
        std::memcpy(MetalBufferPool::contents(indices_handle), indices.data(),
                    indices.size() * sizeof(int));
        std::memcpy(MetalBufferPool::contents(weights_handle), weights.data(),
                    weights.size() * sizeof(float));
        return true;
    }

    id<MTLComputePipelineState> pipeline(const char* name) {
        return pipeline_cache->pipeline(name);
    }

    // Specialized-pipeline cache keyed by name + function-constant tuple. The
    // flash-attention prefill kernel bakes DK, DV, its SIMD-group split, and
    // query tile into the generated pipeline. A failed specialization returns
    // nil so the caller can use the generic prefill kernel.
    id<MTLComputePipelineState> pipeline_fa2(
            int dk, int dv, int nsg, int qt) {
        return pipeline_cache->fa2(dk, dv, nsg, qt);
    }

    // GEMV specialized by NR0 (output rows per threadgroup) via function constant 5.
    id<MTLComputePipelineState> pipeline_gemv2(int nr0) {
        return pipeline_cache->gemv2(nr0);
    }

    id<MTLComputePipelineState> pipeline_gemv_w4(int nr0) {
        return pipeline_cache->gemv_w4(nr0);
    }

    id<MTLComputePipelineState> pipeline_small_m(
            const char* function_name, int m) {
        return pipeline_cache->small_m(function_name, m);
    }

    id<MTLComputePipelineState> pipeline_w4a16(
            bool use_m128, bool specialize_g128) {
        return pipeline_cache->w4a16(use_m128, specialize_g128);
    }

    id<MTLComputePipelineState> pipeline_moe_select_parallel(
            bool sigmoid, bool grouped) {
        return pipeline_cache->moe_select_parallel(sigmoid, grouped);
    }

    id<MTLComputePipelineState> pipeline_grouped_moe(
            int group_size, bool paired_gate_up,
            bool large_route_tile) {
        return pipeline_cache->grouped_moe(
            group_size, paired_gate_up, large_route_tile);
    }

};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

namespace {

// Resolve the MTLBuffer backing a tensor. Returns nil if the tensor has no
// device buffer.
id<MTLBuffer> buf_of(const Tensor* t) {
    if (!t || !t->device.buffer) return nil;
    return (__bridge id<MTLBuffer>)t->device.buffer;
}

id<MTLBuffer> scales_buf_of(const Tensor* t) {
    if (!t || !t->device.scales_buffer) return nil;
    return (__bridge id<MTLBuffer>)t->device.scales_buffer;
}

// element size in bytes for a precision, for offset math.
size_t esize(Precision p) {
    switch (p) {
    case Precision::FP32: return 4;
    case Precision::FP16: return 2;
    case Precision::INT8: return 1;
    case Precision::INT4: return 1;
    case Precision::FP8_E4M3: return 1;
    case Precision::MXFP4: return 1;
    case Precision::INT32: return 4;
    case Precision::RAW_U8: return 1;
    case Precision::NVFP4: return 1;
    }
    return 4;
}

// element stride from byte stride
int estride(const Tensor& t, int dim) {
    return (int)(t.stride[dim] / esize(t.prec));
}

// element offset into the bound buffer (device.offset is in bytes)
uint eoffset(const Tensor& t) {
    return (uint)(t.device.offset / esize(t.prec));
}

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
        impl_->layout_ops.reset(
            new MetalLayoutOps(impl_->pipeline_cache.get()));
        impl_->pool.reset(new MetalBufferPool((__bridge void*)impl_->device));
        impl_->commands.reset(new MetalCommandContext(
            (__bridge void*)queue, impl_->pool.get()));
        impl_->lm_head.reset(new MetalLmHead(
            impl_->pool.get(), impl_->commands.get(),
            impl_->pipeline_cache.get(), impl_->dispatch_failed));
        impl_->resources.reset(
            new MetalResourceStore((__bridge void*)impl_->device));
        impl_->ssd_cache.reset(
            new MetalSsdExpertCache((__bridge void*)impl_->device));
        impl_->ssd_shared_expert.reset(new MetalSsdSharedExpert(
            (__bridge void*)impl_->device, impl_->pool.get(),
            impl_->pipeline_cache.get()));

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
        impl_->ssd_cache.reset();
        impl_->ssd_shared_expert.reset();
        impl_->lm_head.reset();
        impl_->layout_ops.reset();
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
    if (!impl_->ok || !impl_->ssd_cache || package_path.empty() ||
        capacity_bytes == 0) {
        return false;
    }
    if (!impl_->ssd_shared_expert ||
        !impl_->ssd_shared_expert->configure()) {
        fprintf(stderr,
                "MetalBackend: shared-expert command setup failed\n");
        return false;
    }
    if (!impl_->ssd_cache->configure(
            package_path, capacity_bytes, max_commands_in_flight)) {
        return false;
    }
    impl_->ssd_cross_layer_prefetch = cross_layer_prefetch;
    return true;
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
    const OpParams& params = node.params;
    const OpType op = node.op_type;
    std::string profile_label = op_type_name(op);
    bool encoded_gpu_work = true;

    auto dispatch_1d = [&](id<MTLComputePipelineState> ps, int n) {
        [enc setComputePipelineState:ps];
        NSUInteger tg = ps.maxTotalThreadsPerThreadgroup;
        if (tg > 256) tg = 256;
        MTLSize tgs  = MTLSizeMake(tg, 1, 1);
        MTLSize tgcount = MTLSizeMake(((NSUInteger)n + tg - 1) / tg, 1, 1);
        [enc dispatchThreadgroups:tgcount threadsPerThreadgroup:tgs];
    };
    // 1-D grid over `n` elements using a bounds-checked threadgroup dispatch.
    auto grid1d = [&](int n) {
        NSUInteger tg = 256;
        MTLSize tgs  = MTLSizeMake(tg, 1, 1);
        MTLSize tgc  = MTLSizeMake(((NSUInteger)n + tg - 1) / tg, 1, 1);
        [enc dispatchThreadgroups:tgc threadsPerThreadgroup:tgs];
    };

    const bool handled_layout = impl_->layout_ops &&
        impl_->layout_ops->dispatch(
            node, inputs, output, enc, encoded_gpu_work);
    if (!handled_layout) {
        switch (op) {
    case OpType::MATMUL:
    case OpType::GEMV_SPARSE_A: {
        const Tensor& A = *inputs[0];
        const Tensor& B = *inputs[1];
        Tensor& C = *output;
        // Mirror kernel_matmul_fp32 exactly: A is [K(inner), M], B is the weight
        // stored logically [N, K] with shape[0]=N, shape[1]=K and K contiguous
        // (row stride = K elements), C is [N(inner), M].
        MatmulParams p{};
        p.M = (int)A.shape[1];
        p.K = (int)A.shape[0];
        p.N = (int)B.shape[0];
        p.a_offset = eoffset(A);
        p.b_offset = eoffset(B);
        p.c_offset = eoffset(C);
        p.a_row_stride = estride(A, 1);       // elements between rows of A (>= K)
        p.b_row_stride = (int)B.shape[1];     // K: elements between weight rows
        p.c_row_stride = estride(C, 1);       // elements between rows of C (>= N)
        // fused activation: params.i32[0]=Activation (0 NONE, 1 SILU).
        int act = params.i32.size()>0 ? params.i32[0] : 0;
        p.activation = (act >= 0 && act <= 4) ? act : 0;
        p.act_n_begin = params.i32.size()>1 ? params.i32[1] : 0;
        p.act_n_len = params.i32.size()>2 ? params.i32[2] : -1;
        auto cast_activation_to_f16 = [&]() -> id<MTLBuffer> {
            const size_t bytes =
                (size_t)p.M * (size_t)p.K * sizeof(uint16_t);
            void* handle = impl_->pool->acquire(bytes);
            id<MTLBuffer> result = (__bridge id<MTLBuffer>)handle;
            impl_->commands->pending_free.push_back({handle, bytes});
            id<MTLComputePipelineState> ps =
                impl_->pipeline("matmul_cast_f32_to_f16");
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:result offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            grid1d((p.M * p.K + 3) / 4);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            return result;
        };
        // Decode graphs use M=1 throughout. Enable prefix submission only
        // after observing that invariant; large-M prefill benefits from one
        // command buffer and does not need CPU/GPU encoding overlap.
        if (p.M == 1) impl_->commands->chunk_graph = true;

        if (p.M >= 2 && p.M <= 4 && p.N <= 512 &&
            B.prec == Precision::FP16) {
            profile_label = "FP16_SMALL_M[M=" + std::to_string(p.M) +
                            ",N=" + std::to_string(p.N) +
                            ",K=" + std::to_string(p.K) + "]";
            MatmulParams small = p;
            small.b_offset = 0;
            const int nsg =
                std::min(mollm::metal::gemv_nsg_cap(), (p.K + 127) / 128);
            [enc setComputePipelineState:
                     impl_->pipeline_small_m(
                         "gemv_small_m_f32a_f16b_f32c", p.M)];
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBytes:&small length:sizeof(small) atIndex:3];
            const NSUInteger groups =
                ((NSUInteger)p.N + nsg - 1) / nsg;
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
            break;
        }

        if (p.M >= 2 && p.M <= 4 && B.prec == Precision::INT8) {
            profile_label = "W8_SMALL_M[M=" + std::to_string(p.M) +
                            ",N=" + std::to_string(p.N) +
                            ",K=" + std::to_string(p.K) + "]";
            MatmulW8Params w{};
            w.M = p.M; w.N = p.N; w.K = p.K;
            w.a_offset = eoffset(A); w.c_offset = eoffset(C);
            w.a_row_stride = p.a_row_stride;
            w.c_row_stride = p.c_row_stride;
            w.activation = p.activation;
            w.act_n_begin = p.act_n_begin;
            w.act_n_len = p.act_n_len;
            w.group_size = (int)B.group_size;
            w.groups_per_row = (int)B.groups_per_row;
            const int nsg =
                std::min(mollm::metal::gemv_nsg_cap(), (p.K + 127) / 128);
            [enc setComputePipelineState:
                     impl_->pipeline_small_m(
                         "gemv_w8_small_m_f32a_i8b_f32c", p.M)];
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBuffer:scales_buf_of(&B)
                     offset:B.device.scales_offset atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            const NSUInteger groups =
                ((NSUInteger)p.N + nsg - 1) / nsg;
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
            break;
        }

        if (p.M == 1 && B.prec == Precision::INT8) {
            profile_label = "W8_GEMV[N=" + std::to_string(p.N) +
                            ",K=" + std::to_string(p.K) + "]";
            // W8 decode: int8 weight x float activation, per-group weight scale.
            // Weight int8 + fp32 scales both live in the weight region; bind each
            // at its byte offset (scales offset relative to weight_base).
            MatmulW8Params w{};
            w.M = p.M; w.N = p.N; w.K = p.K;
            w.a_offset = eoffset(A); w.c_offset = eoffset(C);
            w.a_row_stride = p.a_row_stride; w.c_row_stride = p.c_row_stride;
            w.activation = p.activation;
            w.act_n_begin = p.act_n_begin; w.act_n_len = p.act_n_len;
            w.group_size = (int)B.group_size;
            w.groups_per_row = (int)B.groups_per_row;
            size_t scales_boff = B.device.scales_offset;
            const int NR0 = 2;
            const int NSG =
                std::min(mollm::metal::gemv_nsg_cap(), (p.K + 127) / 128);
            id<MTLComputePipelineState> ps =
                impl_->pipeline("gemv_w8_f32a_i8b_f32c");
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBuffer:scales_buf_of(&B) offset:scales_boff atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            [enc setThreadgroupMemoryLength:
                (NSUInteger)(NR0 * 32 * sizeof(float)) atIndex:0];
            NSUInteger tgc = ((NSUInteger)p.N + NR0 - 1) / NR0;
            [enc dispatchThreadgroups:MTLSizeMake(tgc,1,1)
                threadsPerThreadgroup:MTLSizeMake(32,(NSUInteger)NSG,1)];
            break;
        }
        if (p.M >= 2 && p.M <= 4 && B.prec == Precision::INT4) {
            profile_label = "W4_SMALL_M[M=" + std::to_string(p.M) +
                            ",N=" + std::to_string(p.N) +
                            ",K=" + std::to_string(p.K) + "]";
            MatmulW8Params w{};
            w.M = p.M; w.N = p.N; w.K = p.K;
            w.a_offset = eoffset(A); w.c_offset = eoffset(C);
            w.a_row_stride = p.a_row_stride;
            w.c_row_stride = p.c_row_stride;
            w.activation = p.activation;
            w.act_n_begin = p.act_n_begin;
            w.act_n_len = p.act_n_len;
            w.group_size = (int)B.group_size;
            w.groups_per_row = (int)B.groups_per_row;
            const size_t scales_boff = (size_t)p.N * (p.K / 2);
            const int NSG = std::min(
                mollm::metal::gemv_w4_nsg_cap(), (p.K / 2 + 63) / 64);
            id<MTLComputePipelineState> ps =
                impl_->pipeline_small_m(
                    "gemv_w4_small_m_f32a_i4b_f32c", p.M);
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBuffer:buf_of(&B) offset:scales_boff atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            const NSUInteger rows_per_tg =
                (NSUInteger)std::max(1, NSG);
            const NSUInteger tgc =
                ((NSUInteger)p.N + rows_per_tg - 1) / rows_per_tg;
            [enc dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
                threadsPerThreadgroup:
                    MTLSizeMake(32 * (NSUInteger)std::max(1, NSG), 1, 1)];
            break;
        }
        if (p.M == 1 && B.prec == Precision::INT4) {
            profile_label = "W4_GEMV[N=" + std::to_string(p.N) +
                            ",K=" + std::to_string(p.K) + "]";
            // W4 decode: per-group symmetric int4 weight x float activation.
            MatmulW8Params w{};
            w.M = p.M; w.N = p.N; w.K = p.K;
            w.a_offset = eoffset(A); w.c_offset = eoffset(C);
            w.a_row_stride = p.a_row_stride; w.c_row_stride = p.c_row_stride;
            w.activation = p.activation;
            w.act_n_begin = p.act_n_begin; w.act_n_len = p.act_n_len;
            w.group_size = (int)B.group_size;
            w.groups_per_row = (int)B.groups_per_row;
            // Decoded W4 buffer layout: [ nibbles (N*K/2) | scales (N*gpr f32) ].
            size_t scales_boff = (size_t)p.N * (p.K / 2);
            const int NR0 = mollm::metal::gemv_w4_nr0(p.N, p.K);
            const int NSG =
                std::min(mollm::metal::gemv_w4_nsg_cap(), (p.K / 2 + 63) / 64);
            id<MTLComputePipelineState> ps = impl_->pipeline_gemv_w4(NR0);
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBuffer:buf_of(&B) offset:scales_boff atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            [enc setThreadgroupMemoryLength:(NSUInteger)(NR0*32*sizeof(float)) atIndex:0];
            NSUInteger rows_per_tg = (NSUInteger)NR0 *
                                     (NSUInteger)std::max(1,NSG);
            NSUInteger tgc = ((NSUInteger)p.N + rows_per_tg - 1)/rows_per_tg;
            [enc dispatchThreadgroups:MTLSizeMake(tgc,1,1)
                threadsPerThreadgroup:
                    MTLSizeMake(32*(NSUInteger)std::max(1,NSG),1,1)];
            break;
        }
        if (p.M == 1) {
            profile_label = "MATMUL_FP16_GEMV";
            // GEMV v2: each threadgroup owns NR0=2 output rows; NSG simdgroups
            // split K and reduce via shmem, so large-K matmuls (down_proj K=9728)
            // get up to 4x32 lanes.
            constexpr bool gemv_old = false;
            constexpr int NR0 = 2;
            // Bind weight at BYTE offset (64-bit) to avoid uint32 element-offset
            // overflow for late weights in the 8.8GB region (esp. lm_head).
            MatmulParams pv = p; pv.b_offset = 0;
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBytes:&pv length:sizeof(pv) atIndex:3];
            id<MTLComputePipelineState> gemv2_ps = gemv_old ? nil : impl_->pipeline_gemv2(NR0);
            if (gemv2_ps) {
                [enc setComputePipelineState:gemv2_ps];
                // Eight SIMD groups give the best cross-model decode throughput
                // on M5 Pro. The environment override keeps this tunable for
                // future GPU families.
                int nsg =
                    std::min(mollm::metal::gemv_nsg_cap(), (p.K + 127) / 128);
                if (nsg < 1) nsg = 1;
                [enc setThreadgroupMemoryLength:(NSUInteger)(NR0 * 32 * sizeof(float)) atIndex:0];
                NSUInteger tgcount = ((NSUInteger)p.N + NR0 - 1) / NR0;
                [enc dispatchThreadgroups:MTLSizeMake(tgcount,1,1)
                    threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            } else {
                id<MTLComputePipelineState> ps = impl_->pipeline("gemv_f32a_f16b_f32c");
                const NSUInteger rows_per_tg = 8;         // 8*32 = 256 threads/tg
                [enc setComputePipelineState:ps];
                NSUInteger tgcount = ((NSUInteger)p.N + rows_per_tg - 1) / rows_per_tg;
                [enc dispatchThreadgroups:MTLSizeMake(tgcount,1,1)
                    threadsPerThreadgroup:MTLSizeMake(rows_per_tg * 32, 1, 1)];
            }
        } else if (B.prec == Precision::INT8) {
            // W8 prefill GEMM. Two paths:
            //  - W8A8 (opt-in, per-channel weights only): quantize activations
            //    per-token to int8, run int8xint8->int32 MMA, dequant at store.
            //  - W8A16 (default): cast the activation matrix once, dequant int8
            //    weight->half during staging, and use FP16-input tensor MMA with
            //    FP32 accumulation. Requires the tensor path.
#ifdef MOLLM_METAL_TENSOR
            static const bool w8a8 = (getenv("MOLLM_METAL_W8A8") != nullptr);
            if (impl_->has_tensor && w8a8 && B.groups_per_row == 1) {
                profile_label = "MATMUL_W8A8_GEMM";
                // --- W8A8: quantize A -> int8 scratch, then int8 MMA ----------
                size_t a_i8_bytes = (size_t)p.M * (size_t)p.K;      // [M,K] contiguous
                size_t sa_bytes   = (size_t)p.M * sizeof(float);    // scale_a[M]
                void* a_i8_h = impl_->pool->acquire(a_i8_bytes);
                void* sa_h   = impl_->pool->acquire(sa_bytes);
                id<MTLBuffer> a_i8 = (__bridge id<MTLBuffer>)a_i8_h;
                id<MTLBuffer> sa   = (__bridge id<MTLBuffer>)sa_h;
                impl_->commands->pending_free.push_back({a_i8_h, a_i8_bytes});
                impl_->commands->pending_free.push_back({sa_h,   sa_bytes});

                // 1) per-token activation quantization.
                {
                    QuantActParams q{};
                    q.M = p.M; q.K = p.K;
                    q.a_offset = A.device.offset / sizeof(float);   // A bound at 0
                    q.a_row_stride = p.a_row_stride;
                    id<MTLComputePipelineState> qps = impl_->pipeline("quantize_act_i8");
                    [enc setComputePipelineState:qps];
                    [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
                    [enc setBuffer:a_i8 offset:0 atIndex:2];
                    [enc setBuffer:sa   offset:0 atIndex:4];
                    [enc setBytes:&q length:sizeof(q) atIndex:3];
                    const NSUInteger nsg = 8;   // 8*32 = 256 threads/row
                    [enc setThreadgroupMemoryLength:nsg*sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.M,1,1)
                        threadsPerThreadgroup:MTLSizeMake(32,nsg,1)];
                }
                // 2) int8xint8->int32 GEMM with dequant at store.
                {
                    MatmulW8A8Params w{};
                    w.M = p.M; w.N = p.N; w.K = p.K;
                    w.c_offset = eoffset(C);
                    w.c_row_stride = p.c_row_stride;
                    w.activation = p.activation;
                    w.act_n_begin = p.act_n_begin; w.act_n_len = p.act_n_len;
                    size_t scales_boff = B.device.scales_offset;
                    id<MTLComputePipelineState> ps = impl_->pipeline("gemm_w8a8_i8a_i8b_f32c");
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:a_i8 offset:0 atIndex:0];
                    [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
                    [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                    [enc setBytes:&w length:sizeof(w) atIndex:3];
                    [enc setBuffer:sa offset:0 atIndex:4];
                    [enc setBuffer:scales_buf_of(&B) offset:scales_boff atIndex:5];
                    // NRB=64 (M) x NRA=64 (N) tile / threadgroup, 128 threads;
                    // int32 accumulators staged in 64*64*4 = 16KB threadgroup.
                    [enc setThreadgroupMemoryLength:64*64*sizeof(int32_t) atIndex:0];
                    MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + 63)/64,
                                              ((NSUInteger)p.N + 63)/64, 1);
                    [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                }
                break;
            }
            if (impl_->has_tensor) {
                profile_label = "MATMUL_W8A16_GEMM";
                MatmulW8Params w{};
                w.M = p.M; w.N = p.N; w.K = p.K;
                w.a_offset = 0; w.c_offset = eoffset(C);
                w.a_row_stride = p.a_row_stride; w.c_row_stride = p.c_row_stride;
                w.activation = p.activation;
                w.act_n_begin = p.act_n_begin; w.act_n_len = p.act_n_len;
                w.group_size = (int)B.group_size;
                w.groups_per_row = (int)B.groups_per_row;
                size_t scales_boff = B.device.scales_offset;
                id<MTLBuffer> ah = cast_activation_to_f16();
                w.a_row_stride = p.K;
                // M64 improves occupancy for the smaller projection shapes.
                // Very large gate/up or down projections amortize the larger
                // M128 cooperative accumulator and issue fewer threadgroups.
                const bool use_m128 =
                    p.K >= 2560 && (p.N >= 8192 || p.K >= 8192);
                id<MTLComputePipelineState> ps = impl_->pipeline(
                    use_m128 ? "gemm_tensor_w8_f16a_i8b_f32c"
                             : "gemm_tensor_w8_f16a_i8b_f32c_m64");
                [enc setComputePipelineState:ps];
                [enc setBuffer:ah offset:0 atIndex:0];
                [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
                [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                [enc setBuffer:scales_buf_of(&B) offset:scales_boff atIndex:4];
                [enc setBytes:&w length:sizeof(w) atIndex:3];
                [enc setThreadgroupMemoryLength:64*32*sizeof(uint16_t) atIndex:0];
                const NSUInteger m_tile = use_m128 ? 128 : 64;
                MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + m_tile - 1)/m_tile,
                                          ((NSUInteger)p.N + 63)/64, 1);
                [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                if (w.activation != 0 && w.act_n_len != 0) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    id<MTLComputePipelineState> aps =
                        impl_->pipeline("matmul_w8_activation_range_f32");
                    [enc setComputePipelineState:aps];
                    [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                    [enc setBytes:&w length:sizeof(w) atIndex:3];
                    grid1d(p.M * p.N);
                }
            } else
#endif
            {
                fprintf(stderr, "MetalBackend: W8 GEMM requires tensor path (M5/A19+)\n");
                assert(false && "W8 GEMM needs tensor path");
            }
        } else if (B.prec == Precision::INT4) {
            // W4 prefill GEMM. W4A16 keeps activations in FP32 and unpacks
            // weights to half while staging for better numerical parity.
            // W4A8 quantizes activations and remains the throughput baseline.
#ifdef MOLLM_METAL_TENSOR
            if (impl_->has_tensor) {
                // Default balanced path: unactivated projections use half
                // tensor GEMM; fused gate/up projections use K64 activation
                // quantization for the entire output. K64 is closer to the CPU
                // reference than half staging while combining two K32 tensor
                // operations before each FP32 dequantization.
                const char* w4_mode =
                    std::getenv("MOLLM_METAL_W4_PREFILL_MODE");
                const bool fast =
                    w4_mode && std::strcmp(w4_mode, "fast") == 0;
                const bool accurate =
                    w4_mode && std::strcmp(w4_mode, "accurate") == 0;
                const bool w4a16 =
                    fast || (!accurate && p.activation == 0);
                if (w4a16) {
                    profile_label = "MATMUL_W4A16_GEMM";
                    if (impl_->commands->profile) {
                        profile_label +=
                            "[M=" + std::to_string(p.M) +
                            ",N=" + std::to_string(p.N) +
                            ",K=" + std::to_string(p.K) + "]";
                    }
                    MatmulW8Params w{};
                    w.M = p.M; w.N = p.N; w.K = p.K;
                    w.a_offset = 0; w.c_offset = eoffset(C);
                    w.a_row_stride = p.a_row_stride;
                    w.c_row_stride = p.c_row_stride;
                    w.activation = p.activation;
                    w.act_n_begin = p.act_n_begin;
                    w.act_n_len = p.act_n_len;
                    w.group_size = (int)B.group_size;
                    w.groups_per_row = (int)B.groups_per_row;
                    size_t scales_boff = (size_t)p.N * (p.K / 2);
                    const bool use_m128 =
                        std::min(p.N, p.K) >= 2560 &&
                        std::max(p.N, p.K) >= 4096;
                    const bool specialize_g128 =
                        w.group_size == 128 &&
                        p.K % 128 == 0;
                    id<MTLComputePipelineState> ps =
                        impl_->pipeline_w4a16(
                            use_m128, specialize_g128);
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:buf_of(&A)
                           offset:A.device.offset atIndex:0];
                    [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
                    [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                    [enc setBytes:&w length:sizeof(w) atIndex:3];
                    [enc setBuffer:buf_of(&B) offset:scales_boff atIndex:4];
                    [enc setThreadgroupMemoryLength:
                             64 * 128 * sizeof(uint16_t)
                                            atIndex:0];
                    MTLSize tgc =
                        MTLSizeMake(
                            ((NSUInteger)p.M + (use_m128 ? 127 : 63)) /
                                (use_m128 ? 128 : 64),
                            ((NSUInteger)p.N + 63) / 64, 1);
                    [enc dispatchThreadgroups:tgc
                        threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                    if (w.activation != 0 && w.act_n_len != 0) {
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        id<MTLComputePipelineState> aps =
                            impl_->pipeline("matmul_w8_activation_range_f32");
                        [enc setComputePipelineState:aps];
                        [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                        [enc setBytes:&w length:sizeof(w) atIndex:3];
                        grid1d(p.M * p.N);
                    }
                    break;
                }
                size_t a_i8_bytes = (size_t)p.M * (size_t)p.K;
                const bool block32 = accurate;
                const bool block64 =
                    !accurate && B.is_q4_g128_packed;
                profile_label =
                    block32 ? "MATMUL_W4_BLOCK_GEMM"
                            : (block64 ? "MATMUL_W4_BLOCK64_GEMM"
                                       : "MATMUL_W4_GROUP128_GEMM");
                const int a_blocks =
                    block32 ? (p.K + 31) / 32
                            : (block64 ? (p.K + 63) / 64
                                       : (int)B.groups_per_row);
                size_t sa_bytes =
                    (size_t)p.M * (size_t)a_blocks * sizeof(float);
                void* a_i8_h = impl_->pool->acquire(a_i8_bytes);
                void* sa_h   = impl_->pool->acquire(sa_bytes);
                id<MTLBuffer> a_i8 = (__bridge id<MTLBuffer>)a_i8_h;
                id<MTLBuffer> sa   = (__bridge id<MTLBuffer>)sa_h;
                impl_->commands->pending_free.push_back({a_i8_h, a_i8_bytes});
                impl_->commands->pending_free.push_back({sa_h,   sa_bytes});

                // 1) per-token activation quantization -> int8 [M,K] + scale_a[M].
                {
                    QuantActParams q{};
                    q.M = p.M; q.K = p.K;
                    q.a_offset = A.device.offset / sizeof(float);
                    q.a_row_stride = p.a_row_stride;
                    q.block_size =
                        block32 ? 32
                                : (block64 ? 64 : (int)B.group_size);
                    id<MTLComputePipelineState> qps = impl_->pipeline(
                        block32 ? "quantize_act_i8_block32"
                                : (block64
                                       ? "quantize_act_i8_block64"
                                       : "quantize_act_i8_blocks"));
                    [enc setComputePipelineState:qps];
                    [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
                    [enc setBuffer:a_i8 offset:0 atIndex:2];
                    [enc setBuffer:sa   offset:0 atIndex:4];
                    [enc setBytes:&q length:sizeof(q) atIndex:3];
                    if (block32 || block64) {
                        constexpr NSUInteger nsg = 4;
                        const NSUInteger block_groups =
                            ((NSUInteger)a_blocks + nsg - 1) / nsg;
                        [enc dispatchThreadgroups:
                                MTLSizeMake((NSUInteger)p.M *
                                                block_groups, 1, 1)
                            threadsPerThreadgroup:
                                MTLSizeMake(32,nsg,1)];
                    } else {
                        const NSUInteger nsg = 4;
                        [enc setThreadgroupMemoryLength:
                                nsg*sizeof(float) atIndex:0];
                        [enc dispatchThreadgroups:
                                MTLSizeMake((NSUInteger)p.M *
                                                (NSUInteger)a_blocks, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(32,nsg,1)];
                    }
                }
                // 2) int8 x per-group int4 GEMM with per-group dequant.
                {
                    MatmulW4A8Params w{};
                    w.M = p.M;
                    w.N = p.N;
                    w.K = p.K;
                    w.c_offset = eoffset(C);
                    w.c_row_stride = p.c_row_stride;
                    w.activation = p.activation;
                    w.act_n_begin = p.act_n_begin; w.act_n_len = p.act_n_len;
                    w.group_size = (int)B.group_size;
                    w.groups_per_row = (int)B.groups_per_row;
                    char* native_ptr =
                        (char*)B.q4_g128_data;
                    void* native_buffer_handle = nullptr;
                    size_t native_buffer_offset = 0;
                    const bool native_bg128 =
                        B.is_q4_g128_packed && native_ptr &&
                        impl_->resources->locate_weight(
                            native_ptr, 1, native_buffer_handle,
                            native_buffer_offset);
                    // Decoded W4 buffer: [ nibbles (N*K/2) | scales (N*gpr f32) ].
                    size_t scales_boff = (size_t)p.N * (p.K / 2);
                    id<MTLComputePipelineState> ps = impl_->pipeline(
                        block64 && native_bg128
                            ? (p.K <= 1024
                                   ? "gemm_w4a8_block64_bg128_smallk_i8a_i4b_f32c"
                                   : "gemm_w4a8_block64_bg128_i8a_i4b_f32c")
                            : (block32
                            ? (native_bg128
                                   ? "gemm_w4a8_block32_bg128_i8a_i4b_f32c"
                                   : "gemm_w4a8_block32_i8a_i4b_f32c")
                            : (native_bg128
                                   ? "gemm_w4a8_bg128_i8a_i4b_f32c"
                                   : "gemm_w4a8_i8a_i4b_f32c")));
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:a_i8 offset:0 atIndex:0];
                    if (native_bg128) {
                        [enc setBuffer:(__bridge id<MTLBuffer>)native_buffer_handle
                               offset:native_buffer_offset
                              atIndex:1];
                    } else {
                        [enc setBuffer:buf_of(&B)
                               offset:B.device.offset
                              atIndex:1];
                    }
                    [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                    [enc setBytes:&w length:sizeof(w) atIndex:3];
                    [enc setBuffer:sa offset:0 atIndex:4];
                    [enc setBuffer:buf_of(&B) offset:scales_boff atIndex:5];
                    const NSUInteger tile_m = 64;
                    const NSUInteger tile_n = 16;
                    const NSUInteger tg_mem =
                        block64
                            ? (p.K <= 1024
                                   ? 2*tile_m*tile_n*sizeof(int32_t) +
                                         tile_n*64
                                   : tile_n*64/2 +
                                         (tile_n + tile_m)*sizeof(float))
                            : 2*tile_m*tile_n*sizeof(int32_t) +
                                  (block32 ? tile_n*32 : 0);
                    [enc setThreadgroupMemoryLength:tg_mem atIndex:0];
                    MTLSize tgc = MTLSizeMake(
                        ((NSUInteger)p.M + tile_m - 1)/tile_m,
                        ((NSUInteger)w.N + tile_n - 1)/tile_n, 1);
                    [enc dispatchThreadgroups:tgc
                        threadsPerThreadgroup:
                            MTLSizeMake(128,1,1)];

                }
            } else
#endif
            {
                fprintf(stderr, "MetalBackend: W4 GEMM requires tensor path (M5/A19+)\n");
                assert(false && "W4 GEMM needs tensor path");
            }
        } else {
            // Tiled/tensor GEMM. Apply an optional fused graph activation as a
            // lightweight post-pass so activation-bearing FP16 nodes retain
            // the same high-throughput matrix path.
            // Weight buffer B bound at its 64-bit BYTE offset with
            // in-shader b_offset=0 to avoid uint32 element-offset overflow for
            // the 8.8GB weight region (incl. lm_head).
            // Default: 32x32 half-staged tile (acc[4]/sg). This is the OCCUPANCY
            // sweet spot on M5 Pro: larger tiles / more accumulators per
            // simdgroup (tested 64x32 -> 219 t/s, 32x64 acc[8] -> 226 t/s, TK=32
            // -> 357) all LOWER perf by reducing resident threadgroup count.
            // Apple GPUs favor many small threadgroups over deep register
            // blocking (opposite of NVIDIA).
            // Weight bound at 64-bit byte offset, b_offset=0.
            MatmulParams pt = p; pt.b_offset = 0;
#ifdef MOLLM_METAL_TENSOR
            if (impl_->has_tensor) {
                profile_label = "MATMUL_FP16_TENSOR";
                // Metal 4 tensor-API GEMM (fast path on M5/A19+): large-K
                // weights are staged in a 64-row tile and activations cast once
                // to an FP16 device tensor. Medium K uses direct device tensors.
                // grid: tgpig.y = N/64, tgpig.x = M/128.
                const bool direct_weights =
                    p.K >= 512 && p.K <= 1024;
                id<MTLComputePipelineState> ps = impl_->pipeline(
                    direct_weights
                        ? "gemm_tensor_direct_f32a_f16b_f32c"
                        : "gemm_tensor_direct_f16a_f16b_f32c");
                // Bind A (activations) and B (weights) at their 64-bit byte
                // offsets; zero the in-shader element offsets accordingly.
                MatmulParams ptt = pt; ptt.a_offset = 0; ptt.b_offset = 0;
                id<MTLBuffer> activation_buffer = buf_of(&A);
                NSUInteger activation_offset = A.device.offset;
                if (!direct_weights) {
                    activation_buffer = cast_activation_to_f16();
                    activation_offset = 0;
                    ptt.a_row_stride = p.K;
                }
                [enc setComputePipelineState:ps];
                [enc setBuffer:activation_buffer offset:activation_offset atIndex:0];
                [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
                [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                [enc setBytes:&ptt length:sizeof(ptt) atIndex:3];
                MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + 127)/128,
                                          ((NSUInteger)p.N + 63)/64, 1);
                [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            } else
#endif
            {
                profile_label = "MATMUL_FP16_TILED";
                id<MTLComputePipelineState> ps = impl_->pipeline("gemm_tiled_f32a_f16b_f32c");
                [enc setComputePipelineState:ps];
                [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
                [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:1];
                [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                [enc setBytes:&pt length:sizeof(pt) atIndex:3];
                // Half-staged tiles, TK=8: (32*8 + 32*8) halves = 1KB; the FP32
                // edge store scratch (256 floats = 1KB) reuses the region.
                [enc setThreadgroupMemoryLength:1024 atIndex:0];
                MTLSize tgc = MTLSizeMake(((NSUInteger)p.N + 31)/32,
                                          ((NSUInteger)p.M + 31)/32, 1);
                [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            }
            if (p.activation != 0 && p.act_n_len != 0) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                id<MTLComputePipelineState> aps =
                    impl_->pipeline("matmul_activation_range_f32");
                [enc setComputePipelineState:aps];
                [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                grid1d(p.M * p.N);
            }
        }
        if (impl_->commands->profile) {
            profile_label += "[M=" + std::to_string(p.M) +
                             ",N=" + std::to_string(p.N) +
                             ",K=" + std::to_string(p.K) + "]";
        }
        break;
    }

    case OpType::RMS_NORM: {
        const Tensor& X = *inputs[0];
        const Tensor& W = *inputs[1];
        Tensor& O = *output;
        RmsNormParams p{};
        p.dim0 = (int)X.shape[0];
        p.rows = (int)(X.shape[1]*X.shape[2]*X.shape[3]);
        p.x_offset = eoffset(X);
        // Bind large-package weights at their 64-bit byte offset. A uint32
        // element offset overflows once the shared region exceeds 16GB.
        p.w_offset = 0;
        p.out_offset = eoffset(O);
        p.x_row_stride = estride(X, 1);
        p.out_row_stride = estride(O, 1);
        p.eps = params.f32.size()>0 ? params.f32[0] : 1e-6f;
        id<MTLComputePipelineState> ps = impl_->pipeline("rms_norm_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&W) offset:W.device.offset atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        NSUInteger tg = 256;
        if (tg > ps.maxTotalThreadsPerThreadgroup) tg = ps.maxTotalThreadsPerThreadgroup;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.rows,1,1)
            threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        break;
    }

    case OpType::RMS_NORM_ROPE: {
        const Tensor& X = *inputs[0];
        const Tensor& W = *inputs[1];
        const Tensor& COS = *inputs[2];
        const Tensor& SIN = *inputs[3];
        Tensor& O = *output;
        RmsNormRopeParams p{};
        p.dim0 = (int)O.shape[0];
        p.seq_len = (int)O.shape[1];
        p.heads = (int)O.shape[2];
        p.rows = p.seq_len * p.heads;
        p.rope_dim = params.i32.size()>0 ? params.i32[0] : p.dim0;
        p.interleave = params.i32.size()>1 ? params.i32[1] : 1;
        p.x_offset = eoffset(X);
        // Bind package weights at their full 64-bit byte offset. Encoding the
        // package-relative offset in the uint shader parameter wraps for
        // FP32 constants beyond 16 GiB.
        p.w_offset = 0;
        p.cos_offset = eoffset(COS);
        p.sin_offset = eoffset(SIN);
        p.out_offset = eoffset(O);
        p.x_row_stride = estride(X, 1);
        p.out_row_stride = estride(O, 1);
        p.eps = params.f32.size()>0 ? params.f32[0] : 1e-6f;
        id<MTLComputePipelineState> ps =
            impl_->pipeline("rms_norm_rope_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&W) offset:W.device.offset atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setBuffer:buf_of(&COS) offset:0 atIndex:4];
        [enc setBuffer:buf_of(&SIN) offset:0 atIndex:5];
        const NSUInteger rope_threads =
            p.seq_len > 1 &&
                    p.dim0 == 128 &&
                    p.rope_dim == 128
                ? 32
                : (NSUInteger)std::max(
                      32, (p.rope_dim + 1) / 2);
        NSUInteger tg = std::min<NSUInteger>(
            256, ((rope_threads + 31) / 32) * 32);
        tg = std::min(tg, ps.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.rows,1,1)
            threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        break;
    }

    case OpType::QK_RMS_NORM_ROPE: {
        const Tensor& query = *inputs[0];
        const Tensor& key = *inputs[1];
        const Tensor& query_weight = *inputs[2];
        const Tensor& key_weight = *inputs[3];
        const Tensor& cos = *inputs[4];
        const Tensor& sin = *inputs[5];
        Tensor& out = *output;
        QkRmsNormRopeParams p{};
        p.dim0 = (int)out.shape[0];
        p.seq_len = (int)out.shape[1];
        p.query_heads =
            params.i32.size()>2 ? params.i32[2] : (int)out.shape[2];
        p.rows = p.seq_len * (int)out.shape[2];
        p.rope_dim = params.i32.size()>0 ? params.i32[0] : p.dim0;
        p.interleave = params.i32.size()>1 ? params.i32[1] : 1;
        p.query_x_offset = eoffset(query);
        p.key_x_offset = eoffset(key);
        p.query_w_offset = 0;
        p.key_w_offset = 0;
        p.cos_offset = eoffset(cos);
        p.sin_offset = eoffset(sin);
        p.out_offset = eoffset(out);
        p.query_x_row_stride = estride(query, 1);
        p.key_x_row_stride = estride(key, 1);
        p.out_row_stride = estride(out, 1);
        p.eps = params.f32.size()>0 ? params.f32[0] : 1e-6f;
        id<MTLComputePipelineState> ps =
            impl_->pipeline("qk_rms_norm_rope_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&query) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&key) offset:0 atIndex:1];
        [enc setBuffer:buf_of(&query_weight)
               offset:query_weight.device.offset atIndex:2];
        [enc setBuffer:buf_of(&key_weight)
               offset:key_weight.device.offset atIndex:3];
        [enc setBuffer:buf_of(&out) offset:0 atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        [enc setBuffer:buf_of(&cos) offset:0 atIndex:6];
        [enc setBuffer:buf_of(&sin) offset:0 atIndex:7];
        const NSUInteger rope_threads =
            (NSUInteger)std::max(32, (p.rope_dim + 1) / 2);
        NSUInteger tg = std::min<NSUInteger>(
            256, ((rope_threads + 31) / 32) * 32);
        tg = std::min(tg, ps.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.rows,1,1)
            threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        break;
    }

    case OpType::ADD_RMS_NORM: {
        Tensor& residual = *const_cast<Tensor*>(inputs[0]);
        const Tensor& update = *inputs[1];
        const Tensor& weight = *inputs[2];
        Tensor& out = *output;
        AddRmsNormParams p{};
        p.dim0 = (int)residual.shape[0];
        p.rows = (int)(
            residual.shape[1] * residual.shape[2] * residual.shape[3]);
        p.residual_offset = eoffset(residual);
        p.update_offset = eoffset(update);
        p.out_offset = eoffset(out);
        p.residual_row_stride = estride(residual, 1);
        p.update_row_stride = estride(update, 1);
        p.out_row_stride = estride(out, 1);
        p.eps = params.f32.size() > 0 ? params.f32[0] : 1e-6f;
        id<MTLComputePipelineState> ps =
            impl_->pipeline("add_rms_norm_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&residual) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&update) offset:0 atIndex:1];
        [enc setBuffer:buf_of(&out) offset:0 atIndex:2];
        [enc setBuffer:buf_of(&weight)
               offset:weight.device.offset atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        NSUInteger tg = std::min<NSUInteger>(
            256, ps.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.rows,1,1)
            threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        break;
    }

    case OpType::LAYER_NORM: {
        const Tensor& X = *inputs[0];
        const Tensor& W = *inputs[1];
        const Tensor& B = *inputs[2];
        Tensor& O = *output;
        LayerNormParams p{};
        p.dim0 = (int)X.shape[0];
        p.rows = (int)(X.shape[1] * X.shape[2] * X.shape[3]);
        p.x_offset = eoffset(X);
        p.out_offset = eoffset(O);
        p.x_row_stride = estride(X, 1);
        p.out_row_stride = estride(O, 1);
        p.eps = params.f32.size() > 0 ? params.f32[0] : 1e-5f;
        id<MTLComputePipelineState> ps = impl_->pipeline("layer_norm_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&W) offset:W.device.offset atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setBuffer:buf_of(&B) offset:B.device.offset atIndex:4];
        NSUInteger tg = std::min<NSUInteger>(
            256, ps.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        break;
    }

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
        p.x_offset = eoffset(X);
        p.cos_offset = eoffset(COS);
        p.sin_offset = eoffset(SIN);
        // RoPE operates on X. When `in` is a strided view, the copy below
        // materializes it into dense X, so carrying the input strides into the
        // in-place kernel would skip rows and eventually access past X.
        p.x_stride_pos = estride(X, 1);
        p.x_stride_head = estride(X, 2);
        // Copy input -> output buffer (rope in place), if different buffers.
        if (buf_of(&in) != buf_of(&X) || in.device.offset != X.device.offset) {
            // use blit copy via contiguous kernel (contiguous input assumed)
            TensorDesc d{};
            for(int i=0;i<4;i++){d.shape[i]=(int)in.shape[i]; d.stride[i]=estride(in,i);}            
            d.offset=eoffset(in);
            id<MTLComputePipelineState> cps = impl_->pipeline("contiguous_f32");
            [enc setComputePipelineState:cps];
            [enc setBuffer:buf_of(&in) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&X) offset:0 atIndex:2];
            [enc setBytes:&d length:sizeof(d) atIndex:3];
            grid1d((int)X.nelements());
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        id<MTLComputePipelineState> ps = impl_->pipeline("rope_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&COS) offset:0 atIndex:1];
        [enc setBuffer:buf_of(&SIN) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        // 3-D grid over (pair, position, head) via bounds-checked threadgroups.
        NSUInteger tx=8, ty=8, tz=4;
        MTLSize tgs = MTLSizeMake(tx,ty,tz);
        MTLSize tgc = MTLSizeMake(((NSUInteger)(rope_dim/2)+tx-1)/tx,
                                  ((NSUInteger)p.seq_len+ty-1)/ty,
                                  ((NSUInteger)p.heads+tz-1)/tz);
        [enc dispatchThreadgroups:tgc threadsPerThreadgroup:tgs];
        break;
    }

    case OpType::ADD:
    case OpType::MUL:
    case OpType::SIGMOID_MUL: {
        const Tensor& A = *inputs[0];
        const Tensor& B = *inputs[1];
        Tensor& O = *output;
        EwiseParams p{};
        p.n = (int)O.nelements();
        p.broadcast_b = (B.nelements()==1) ? 1 : 0;
        p.shape0 = (int)O.shape[0];
        p.a_row_stride = estride(A, 1);
        p.b_row_stride = estride(B, 1);
        p.out_row_stride = estride(O, 1);
        p.a_offset = eoffset(A);
        p.b_offset = eoffset(B);
        p.out_offset = eoffset(O);
        for (int d = 0; d < 4; ++d) {
            p.shape[d] = (int)O.shape[d];
            p.a_stride[d] = A.shape[d] == 1 && O.shape[d] != 1
                ? 0 : estride(A, d);
            p.b_stride[d] = B.shape[d] == 1 && O.shape[d] != 1
                ? 0 : estride(B, d);
            p.out_stride[d] = estride(O, d);
        }
        const char* kernel =
            op == OpType::ADD ? "add_f32" :
            op == OpType::MUL ? "mul_f32" : "sigmoid_mul_f32";
        id<MTLComputePipelineState> ps = impl_->pipeline(kernel);
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&B) offset:0 atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(ps, p.n);
        break;
    }

    case OpType::SILU: {
        const Tensor& X = *inputs[0];
        Tensor& O = *output;
        EwiseParams p{};
        p.n = (int)O.nelements();
        p.a_offset = eoffset(X);
        p.out_offset = eoffset(O);
        id<MTLComputePipelineState> ps = impl_->pipeline("silu_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(ps, p.n);
        break;
    }

    case OpType::SIGMOID:
    case OpType::SIGMOID_EXACT:
    case OpType::GELU:
    case OpType::TANH:
    case OpType::EXP:
    case OpType::EXP_EXACT:
    case OpType::SOFTPLUS: {
        const Tensor& X = *inputs[0];
        Tensor& O = *output;
        EwiseParams p{};
        p.n = (int)O.nelements();
        p.shape0 = (int)O.shape[0];
        p.a_row_stride = estride(X, 1);
        p.out_row_stride = estride(O, 1);
        p.a_offset = eoffset(X);
        p.out_offset = eoffset(O);
        const char* kernel =
            (op == OpType::GELU) ? "gelu_f32" :
            (op == OpType::TANH) ? "tanh_f32" :
            (op == OpType::EXP || op == OpType::EXP_EXACT) ? "exp_f32" :
            (op == OpType::SOFTPLUS) ? "softplus_f32" : "sigmoid_f32";
        id<MTLComputePipelineState> ps = impl_->pipeline(kernel);
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(ps, p.n);
        break;
    }

    case OpType::RWKV_TOKEN_SHIFT: {
        const Tensor& X = *inputs[0];
        const Tensor& STATE = *inputs[1];
        Tensor& O = *output;
        RwkvTokenShiftParams p{};
        p.hidden = params.i32.size() > 0 ? params.i32[0] : (int)X.shape[0];
        p.seq = params.i32.size() > 1 ? params.i32[1] : (int)X.shape[1];
        p.real = params.i32.size() > 2 ? params.i32[2] : p.seq;
        if (p.real <= 0 || p.real > p.seq) p.real = p.seq;
        p.state_fp16 = STATE.prec == Precision::FP16;
        p.x_offset = eoffset(X);
        p.state_offset = eoffset(STATE);
        p.out_offset = eoffset(O);
        id<MTLComputePipelineState> ps =
            impl_->pipeline("rwkv_token_shift_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&STATE) offset:0 atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(ps, p.hidden);
        break;
    }

    case OpType::RWKV_MIX: {
        const Tensor& X = *inputs[0];
        const Tensor& SHIFT = *inputs[1];
        const Tensor& MIX = *inputs[2];
        Tensor& O = *output;
        RwkvMixParams p{};
        p.hidden = (int)MIX.nelements();
        p.total = (int)X.nelements();
        p.x_offset = eoffset(X);
        p.shift_offset = eoffset(SHIFT);
        p.mix_offset = eoffset(MIX);
        p.out_offset = eoffset(O);
        id<MTLComputePipelineState> ps = impl_->pipeline("rwkv_mix_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&SHIFT) offset:0 atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setBuffer:buf_of(&MIX) offset:0 atIndex:4];
        dispatch_1d(ps, p.total);
        break;
    }

    case OpType::RWKV_L2_NORM: {
        const Tensor& X = *inputs[0];
        Tensor& O = *output;
        RwkvL2NormParams p{};
        p.heads = params.i32.size() > 0 ? params.i32[0] : 0;
        p.head_size = params.i32.size() > 1 ? params.i32[1] : 0;
        p.groups = p.head_size > 0
            ? (int)(X.nelements() / p.head_size) : 0;
        p.x_offset = eoffset(X);
        p.out_offset = eoffset(O);
        p.eps = params.f32.size() > 0 ? params.f32[0] : 1e-12f;
        id<MTLComputePipelineState> ps =
            impl_->pipeline("rwkv_l2_norm_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        break;
    }

    case OpType::RWKV_POST: {
        const Tensor& RAW = *inputs[0];
        const Tensor& R = *inputs[1];
        const Tensor& K = *inputs[2];
        const Tensor& V = *inputs[3];
        const Tensor& RK = *inputs[4];
        const Tensor& W = *inputs[5];
        const Tensor& BIAS = *inputs[6];
        const Tensor& GATE = *inputs[7];
        Tensor& O = *output;
        RwkvPostParams p{};
        p.heads = params.i32.size() > 0 ? params.i32[0] : 0;
        p.head_size = params.i32.size() > 1 ? params.i32[1] : 0;
        p.groups = p.head_size > 0
            ? (int)(RAW.nelements() / p.head_size) : 0;
        p.raw_offset = eoffset(RAW);
        p.r_offset = eoffset(R);
        p.k_offset = eoffset(K);
        p.v_offset = eoffset(V);
        p.rk_offset = eoffset(RK);
        p.weight_offset = eoffset(W);
        p.bias_offset = eoffset(BIAS);
        p.gate_offset = eoffset(GATE);
        p.out_offset = eoffset(O);
        p.eps = params.f32.size() > 0 ? params.f32[0] : 64e-5f;
        id<MTLComputePipelineState> ps = impl_->pipeline("rwkv_post_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&RAW) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&R) offset:0 atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setBuffer:buf_of(&K) offset:0 atIndex:4];
        [enc setBuffer:buf_of(&V) offset:0 atIndex:5];
        [enc setBuffer:buf_of(&RK) offset:0 atIndex:6];
        [enc setBuffer:buf_of(&W) offset:0 atIndex:7];
        [enc setBuffer:buf_of(&BIAS) offset:0 atIndex:8];
        [enc setBuffer:buf_of(&GATE) offset:0 atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        break;
    }

    case OpType::RWKV7: {
        const Tensor& R = *inputs[0];
        const Tensor& DECAY = *inputs[1];
        const Tensor& K = *inputs[2];
        const Tensor& V = *inputs[3];
        const Tensor& A = *inputs[4];
        const Tensor& B = *inputs[5];
        const Tensor& STATE = *inputs[6];
        Tensor& O = *output;
        Rwkv7Params p{};
        p.heads = params.i32.size() > 0 ? params.i32[0] : 0;
        p.head_size = params.i32.size() > 1 ? params.i32[1] : 0;
        p.seq = params.i32.size() > 2 ? params.i32[2] : (int)R.shape[1];
        p.real = params.i32.size() > 3 ? params.i32[3] : p.seq;
        if (p.real <= 0 || p.real > p.seq) p.real = p.seq;
        p.state_fp16 = STATE.prec == Precision::FP16;
        p.r_offset = eoffset(R);
        p.decay_offset = eoffset(DECAY);
        p.k_offset = eoffset(K);
        p.v_offset = eoffset(V);
        p.a_offset = eoffset(A);
        p.b_offset = eoffset(B);
        p.state_offset = eoffset(STATE);
        p.out_offset = eoffset(O);
        const bool use_h64_tgstate =
            !p.state_fp16 && p.head_size == 64 && p.real > 1;
        id<MTLComputePipelineState> ps = impl_->pipeline(
            use_h64_tgstate ? "rwkv7_h64_tgstate_fp32" : "rwkv7_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&R) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&DECAY) offset:0 atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setBuffer:buf_of(&K) offset:0 atIndex:4];
        [enc setBuffer:buf_of(&V) offset:0 atIndex:5];
        [enc setBuffer:buf_of(&A) offset:0 atIndex:6];
        [enc setBuffer:buf_of(&B) offset:0 atIndex:7];
        [enc setBuffer:buf_of(&STATE) offset:0 atIndex:8];
        if (use_h64_tgstate) {
            profile_label = "RWKV7_H64_TGSTATE";
            [enc dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)p.heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            const NSUInteger rows_per_group = 32;
            const NSUInteger groups_per_head =
                ((NSUInteger)p.head_size + rows_per_group - 1) / rows_per_group;
            [enc dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)p.heads * groups_per_head, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(rows_per_group, 1, 1)];
        }
        break;
    }

    case OpType::GATED_DELTANET_CONV_DECODE: {
        // Decode-only fusion: raw qkv + conv weight/state are appended to the
        // regular GDN inputs. One threadgroup owns one key head and all of its
        // value heads, so shared q/k convolution state is updated exactly once.
        const Tensor& QKV = *inputs[0]; const Tensor& Aa = *inputs[1];
        const Tensor& Bb  = *inputs[2]; const Tensor& Zz = *inputs[3];
        const Tensor& ALG = *inputs[4]; const Tensor& DTB = *inputs[5];
        const Tensor& NRM = *inputs[6]; const Tensor& ST  = *inputs[7];
        const Tensor& CW  = *inputs[8]; const Tensor& CS  = *inputs[9];
        Tensor& O = *output;
        GdnParams p{};
        p.num_heads   = params.i32.size()>0 ? params.i32[0] : 16;
        p.k_dim       = params.i32.size()>1 ? params.i32[1] : 128;
        p.v_dim       = params.i32.size()>2 ? params.i32[2] : 128;
        p.seq_len     = 1;
        const int gdn_flags = params.i32.size()>4 ? params.i32[4] : 1;
        p.use_qk_l2norm = gdn_flags & 1;
        p.output_gate_type = (gdn_flags & 2) != 0;
        p.conv_kernel = params.i32.size()>5 ? params.i32[5] : 4;
        p.n_real      = 1;
        p.num_v_heads = (params.i32.size()>7 && params.i32[7]>0)
                            ? params.i32[7] : p.num_heads;
        p.rms_eps     = params.f32.size()>0 ? params.f32[0] : 1e-6f;
        p.l2_eps      = params.f32.size()>1 ? params.f32[1] : 1e-6f;
        p.scale       = params.f32.size()>2 ? params.f32[2] : 0.f;
        if (p.scale == 0.f) p.scale = 1.f / std::sqrt((float)p.k_dim);
        p.qkv_offset = eoffset(QKV); p.a_offset = eoffset(Aa);
        p.b_offset = eoffset(Bb); p.z_offset = eoffset(Zz);
        p.Alog_offset = eoffset(ALG); p.dtb_offset = eoffset(DTB);
        p.norm_offset = eoffset(NRM); p.state_offset = eoffset(ST);
        p.out_offset = eoffset(O);
        p.a_row_stride = estride(Aa, 1);
        p.b_row_stride = estride(Bb, 1);
        p.z_row_stride = estride(Zz, 1);
        p.conv_weight_offset = eoffset(CW);
        p.conv_state_offset = eoffset(CS);

        const int repeat = p.num_v_heads / p.num_heads;
        const NSUInteger threads = (NSUInteger)(repeat * p.v_dim);
        const NSUInteger qk_nsg = ((NSUInteger)p.k_dim + 31) / 32;
        const NSUInteger v_nsg = ((NSUInteger)p.v_dim + 31) / 32;
        const NSUInteger smem_floats =
            (NSUInteger)(2*p.k_dim) + 2*qk_nsg +
            (NSUInteger)repeat*v_nsg;
        id<MTLComputePipelineState> ps =
            impl_->pipeline("gdn_conv_decode_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&QKV) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&Aa)  offset:0 atIndex:1];
        [enc setBuffer:buf_of(&Bb)  offset:0 atIndex:2];
        [enc setBuffer:buf_of(&O)   offset:0 atIndex:4];
        [enc setBuffer:buf_of(&Zz)  offset:0 atIndex:5];
        [enc setBuffer:buf_of(&ALG) offset:0 atIndex:6];
        [enc setBuffer:buf_of(&DTB) offset:0 atIndex:7];
        [enc setBuffer:buf_of(&NRM) offset:0 atIndex:8];
        [enc setBuffer:buf_of(&ST)  offset:0 atIndex:9];
        [enc setBuffer:buf_of(&CW)  offset:0 atIndex:10];
        [enc setBuffer:buf_of(&CS)  offset:0 atIndex:11];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setThreadgroupMemoryLength:
                smem_floats*sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:
                MTLSizeMake((NSUInteger)p.num_heads,1,1)
            threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        profile_label = "GDN_CONV_DECODE";
        break;
    }

    case OpType::GATED_DELTANET_DECODE:
    case OpType::GATED_DELTANET_PREFILL: {
        // Fused Gated Delta Rule + RMSNormGated. inputs (gdn.h contract):
        // [0]qkv [1]a [2]b [3]z [4]A_log [5]dt_bias [6]norm_w [7]state; out[0].
        // One threadgroup per value head, v_dim threads. decode=seq1; prefill
        // loops seq serially. qkv layout: decode [seq,dim](seq=1); prefill [dim,seq].
        const Tensor& QKV = *inputs[0]; const Tensor& Aa = *inputs[1];
        const Tensor& Bb  = *inputs[2]; const Tensor& Zz = *inputs[3];
        const Tensor& ALG = *inputs[4]; const Tensor& DTB = *inputs[5];
        const Tensor& NRM = *inputs[6]; const Tensor& ST  = *inputs[7];
        Tensor& O = *output;
        GdnParams p{};
        p.num_heads   = params.i32.size()>0 ? params.i32[0] : 16;
        p.k_dim       = params.i32.size()>1 ? params.i32[1] : 128;
        p.v_dim       = params.i32.size()>2 ? params.i32[2] : 128;
        p.seq_len     = params.i32.size()>3 ? params.i32[3] : 1;
        const int gdn_flags = params.i32.size()>4 ? params.i32[4] : 1;
        p.use_qk_l2norm = gdn_flags & 1;
        p.output_gate_type = (gdn_flags & 2) != 0;
        p.n_real      = params.i32.size()>6 ? params.i32[6] : 0;
        p.num_v_heads = (params.i32.size()>7 && params.i32[7]>0) ? params.i32[7] : p.num_heads;
        p.rms_eps     = params.f32.size()>0 ? params.f32[0] : 1e-6f;
        p.l2_eps      = params.f32.size()>1 ? params.f32[1] : 1e-6f;
        p.scale       = params.f32.size()>2 ? params.f32[2] : 0.f;
        if (p.scale == 0.f) p.scale = 1.f / std::sqrt((float)p.k_dim);
        p.qkv_offset = eoffset(QKV); p.a_offset = eoffset(Aa); p.b_offset = eoffset(Bb);
        p.z_offset = eoffset(Zz); p.Alog_offset = eoffset(ALG); p.dtb_offset = eoffset(DTB);
        p.norm_offset = eoffset(NRM); p.state_offset = eoffset(ST); p.out_offset = eoffset(O);
        p.a_row_stride = estride(Aa, 1);
        p.b_row_stride = estride(Bb, 1);
        p.z_row_stride = estride(Zz, 1);
        const bool prefill =
            op == OpType::GATED_DELTANET_PREFILL;
        const bool row_recurrence =
            prefill && p.k_dim == 128 &&
            p.v_dim > 0 && p.v_dim <= 1024;
        if (row_recurrence) {
            const size_t qk_bytes =
                (size_t)p.seq_len * (size_t)p.num_heads *
                (size_t)p.k_dim * sizeof(float);
            const size_t gate_bytes =
                (size_t)p.seq_len * (size_t)p.num_v_heads *
                sizeof(float);
            const size_t raw_bytes =
                (size_t)p.seq_len * (size_t)p.num_v_heads *
                (size_t)p.v_dim * sizeof(float);
            void* qn_h = impl_->pool->acquire(qk_bytes);
            void* kn_h = impl_->pool->acquire(qk_bytes);
            void* ge_h = impl_->pool->acquire(gate_bytes);
            void* be_h = impl_->pool->acquire(gate_bytes);
            void* raw_h = impl_->pool->acquire(raw_bytes);
            id<MTLBuffer> qn = (__bridge id<MTLBuffer>)qn_h;
            id<MTLBuffer> kn = (__bridge id<MTLBuffer>)kn_h;
            id<MTLBuffer> ge = (__bridge id<MTLBuffer>)ge_h;
            id<MTLBuffer> be = (__bridge id<MTLBuffer>)be_h;
            id<MTLBuffer> raw = (__bridge id<MTLBuffer>)raw_h;
            impl_->commands->pending_free.push_back({qn_h, qk_bytes});
            impl_->commands->pending_free.push_back({kn_h, qk_bytes});
            impl_->commands->pending_free.push_back({ge_h, gate_bytes});
            impl_->commands->pending_free.push_back({be_h, gate_bytes});
            impl_->commands->pending_free.push_back({raw_h, raw_bytes});

            id<MTLComputePipelineState> prep =
                impl_->pipeline("gdn_prepare_qk_f32");
            [enc setComputePipelineState:prep];
            [enc setBuffer:buf_of(&QKV) offset:0 atIndex:0];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc setBuffer:qn offset:0 atIndex:10];
            [enc setBuffer:kn offset:0 atIndex:11];
            [enc setThreadgroupMemoryLength:8*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)p.seq_len *
                                    (NSUInteger)p.num_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            id<MTLComputePipelineState> gates =
                impl_->pipeline("gdn_prepare_gates_f32");
            [enc setComputePipelineState:gates];
            [enc setBuffer:buf_of(&Aa) offset:0 atIndex:1];
            [enc setBuffer:buf_of(&Bb) offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc setBuffer:buf_of(&ALG) offset:0 atIndex:6];
            [enc setBuffer:buf_of(&DTB) offset:0 atIndex:7];
            [enc setBuffer:ge offset:0 atIndex:12];
            [enc setBuffer:be offset:0 atIndex:13];
            grid1d(p.seq_len * p.num_v_heads);

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            id<MTLComputePipelineState> recur =
                impl_->pipeline("gdn_recurrence_rows_f32");
            [enc setComputePipelineState:recur];
            [enc setBuffer:buf_of(&QKV) offset:0 atIndex:0];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc setBuffer:buf_of(&ST) offset:0 atIndex:9];
            [enc setBuffer:qn offset:0 atIndex:10];
            [enc setBuffer:kn offset:0 atIndex:11];
            [enc setBuffer:ge offset:0 atIndex:12];
            [enc setBuffer:be offset:0 atIndex:13];
            [enc setBuffer:raw offset:0 atIndex:14];
            [enc dispatchThreadgroups:
                    MTLSizeMake(((NSUInteger)p.v_dim + 3)/4,
                                (NSUInteger)p.num_v_heads, 1)
                threadsPerThreadgroup:MTLSizeMake(32,4,1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            id<MTLComputePipelineState> post =
                impl_->pipeline("gdn_post_f32");
            [enc setComputePipelineState:post];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc setBuffer:buf_of(&O) offset:0 atIndex:4];
            [enc setBuffer:buf_of(&Zz) offset:0 atIndex:5];
            [enc setBuffer:buf_of(&NRM) offset:0 atIndex:8];
            [enc setBuffer:raw offset:0 atIndex:14];
            const NSUInteger post_threads = (NSUInteger)p.v_dim;
            const NSUInteger post_nsg = (post_threads + 31) / 32;
            [enc setThreadgroupMemoryLength:
                    post_nsg*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)p.seq_len *
                                    (NSUInteger)p.num_v_heads, 1, 1)
                threadsPerThreadgroup:
                    MTLSizeMake(post_threads,1,1)];
            break;
        }
        const bool kparallel =
            prefill && p.v_dim > 0 && 4*p.v_dim <= 1024;
        const char* gk =
            kparallel
                ? "gdn_prefill_kparallel_f32"
                : (prefill ? "gdn_prefill_f32" : "gdn_decode_f32");
        id<MTLComputePipelineState> ps = impl_->pipeline(gk);
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&QKV) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&Aa)  offset:0 atIndex:1];
        [enc setBuffer:buf_of(&Bb)  offset:0 atIndex:2];
        [enc setBuffer:buf_of(&O)   offset:0 atIndex:4];
        [enc setBuffer:buf_of(&Zz)  offset:0 atIndex:5];
        [enc setBuffer:buf_of(&ALG) offset:0 atIndex:6];
        [enc setBuffer:buf_of(&DTB) offset:0 atIndex:7];
        [enc setBuffer:buf_of(&NRM) offset:0 atIndex:8];
        [enc setBuffer:buf_of(&ST)  offset:0 atIndex:9];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        const NSUInteger threads =
            kparallel ? (NSUInteger)(4*p.v_dim)
                      : (NSUInteger)p.v_dim;
        // 4-way prefill: q/k + three SIMD reductions + four partial pairs,
        // delta, attn, and two gate scalars. Decode retains q/k + red[V].
        const NSUInteger nsg = (threads + 31) / 32;
        NSUInteger smem =
            (kparallel
                 ? (NSUInteger)(2*p.k_dim + 3*nsg + 10*p.v_dim + 2)
                 : (NSUInteger)(2*p.k_dim + p.v_dim)) *
            sizeof(float);
        [enc setThreadgroupMemoryLength:smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.num_v_heads,1,1)
            threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        break;
    }

    case OpType::SHORTCONV: {
        // Depth-wise causal conv1d + silu. inputs = {x, w, conv_state}; output.
        // conv_state is a persistent device buffer, read+written in-place. One
        // thread per group (groups is large: 6144/8192).
        const Tensor& X = *inputs[0];
        const Tensor& W = *inputs[1];
        const Tensor& STATE = *inputs[2];   // GPU buffer written in-place by kernel
        Tensor& O = *output;
        ShortConvParams p{};
        p.kernel_size = params.i32.size()>0 ? params.i32[0] : 4;
        p.groups = (int)X.shape[0];
        p.seq = (int)X.shape[1];
        p.n_real = params.i32.size()>1 ? params.i32[1] : p.seq;
        p.x_offset = eoffset(X);
        p.x_row_stride = estride(X, 1);
        p.w_offset = eoffset(W);
        p.state_offset = eoffset(STATE);
        p.out_offset = eoffset(O);
        id<MTLComputePipelineState> ps = impl_->pipeline("shortconv_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&W) offset:0 atIndex:1];
        [enc setBuffer:buf_of(&STATE) offset:0 atIndex:2];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        // One thread per group; bounds-checked threadgroups (M5 dispatchThreads bug).
        NSUInteger tg = 64;
        MTLSize tgc = MTLSizeMake(((NSUInteger)p.groups + tg - 1)/tg, 1, 1);
        [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        break;
    }

    case OpType::SWIGLU: {
        // Fused silu(gate)*up over a merged [2I, rows] tensor. Reads both halves
        // from the single merged buffer (merged row stride = 2I), writes dense
        // [I, rows]. Splits internally — does NOT rely on stride-aware slice views.
        const Tensor& M = *inputs[0];
        Tensor& O = *output;
        SwigluParams p{};
        p.I = (int)M.shape[0] / 2;
        p.n = (int)O.nelements();
        p.merged_offset = eoffset(M);
        p.out_offset = eoffset(O);
        p.merged_row_stride = estride(M, 1);   // elements between tokens (= 2I)
        id<MTLComputePipelineState> ps = impl_->pipeline("swiglu_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&M) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(ps, p.n);
        break;
    }

    case OpType::SDPA:
    case OpType::SDPA_MLA: {
        // inputs = {Q, K_cur, V_cur, mask?, K_cache?, V_cache?}
        const Tensor& Q     = *inputs[0];
        const Tensor& K_cur = *inputs[1];
        const Tensor& V_cur = *inputs[2];
        const Tensor* mask    = (inputs.size()>3 && inputs[3] && inputs[3]->data) ? inputs[3] : nullptr;
        const Tensor* K_cache = (inputs.size()>4 && inputs[4] && inputs[4]->data) ? inputs[4] : nullptr;
        const Tensor* V_cache = (inputs.size()>5 && inputs[5] && inputs[5]->data) ? inputs[5] : nullptr;
        Tensor& out = *output;

        int kv_cache   = params.i32.size()>0 ? params.i32[0] : 2;
        int causal     = params.i32.size()>1 ? params.i32[1] : 1;
        int num_heads  = params.i32.size()>2 ? params.i32[2] : (int)Q.shape[2];
        int num_kv     = params.i32.size()>3 ? params.i32[3] : (int)K_cur.shape[2];
        int head_dim   = params.i32.size()>4 ? params.i32[4] : (int)Q.shape[0];
        int v_head_dim = params.i32.size()>5 ? params.i32[5] : (int)V_cur.shape[0];
        float scale    = params.f32.size()>0 ? params.f32[0] : 0.f;
        if (scale == 0.f) scale = 1.f / std::sqrt((float)head_dim);

        int src_seqlen = (int)Q.shape[1];
        int cur_seqlen = (int)K_cur.shape[1];
        // Cache metadata lives in the Shared buffer's host-visible header.
        int past = 0, max_seq = 0;
        if (kv_cache == 2 && K_cache && K_cache->data) {
            const auto* meta = reinterpret_cast<const uint64_t*>(K_cache->data);
            past    = (int)meta[0];  // current_seq_len
            max_seq = (int)meta[1];  // max_seq_len
        }
        std::string sdpa_profile_suffix;
        if (impl_->commands->profile) {
            sdpa_profile_suffix =
                "[S=" + std::to_string(src_seqlen) +
                ",P=" + std::to_string(past) +
                ",H=" + std::to_string(num_heads) +
                ",HKV=" + std::to_string(num_kv) +
                ",DK=" + std::to_string(head_dim) +
                ",DV=" + std::to_string(v_head_dim) + "]";
            profile_label += sdpa_profile_suffix;
        }

        auto profile_sdpa_stage = [&](const char* label) {
            if (!impl_->commands->profile) return;
            if (impl_->commands->enc) {
                [impl_->commands->enc endEncoding];
                impl_->commands->enc = nil;
            }
            if (impl_->commands->cmd) {
                [impl_->commands->cmd commit];
                [impl_->commands->cmd waitUntilCompleted];
                const double gpu_ms =
                    (impl_->commands->cmd.GPUEndTime -
                     impl_->commands->cmd.GPUStartTime) * 1000.0;
                auto& stat =
                    impl_->commands->op_stats[std::string(label) + sdpa_profile_suffix];
                stat.gpu_ms += gpu_ms;
                stat.calls += 1;
            }
            impl_->commands->cmd = [impl_->commands->queue commandBuffer];
            impl_->commands->enc = [impl_->commands->cmd computeCommandEncoder];
            enc = impl_->commands->enc;
        };

        int dst_seqlen = past + cur_seqlen;
        // Cache data begins 64 bytes past the buffer base (CacheMetadata header).
        // FP16 cache: element offset = (device.offset + 64) / 2.
        const size_t CACHE_HDR = 64;
        uint k_cache_eoff = (uint)((K_cache ? K_cache->device.offset : 0) + CACHE_HDR) / 2;
        uint v_cache_eoff = (uint)((V_cache ? V_cache->device.offset : 0) + CACHE_HDR) / 2;

        // 1) Append K_cur/V_cur (FP32) into the FP16 cache at position past+s.
        SdpaAppendKvParams ap{};
        if (kv_cache == 2 && K_cache && V_cache) {
            ap.num_kv_heads = num_kv;
            ap.cur_seqlen = cur_seqlen;
            ap.past_seqlen = past;
            ap.max_seq_len = max_seq;
            ap.k_dim = head_dim;
            ap.v_dim = v_head_dim;
            ap.k_cur_offset = eoffset(K_cur);
            ap.k_stride_head = estride(K_cur, 2);
            ap.k_stride_pos = estride(K_cur, 1);
            ap.k_cache_offset = k_cache_eoff;
            ap.v_cur_offset = eoffset(V_cur);
            ap.v_stride_head = estride(V_cur, 2);
            ap.v_stride_pos = estride(V_cur, 1);
            ap.v_cache_offset = v_cache_eoff;
        }
        if (kv_cache == 2 && K_cache && V_cache) {
            id<MTLComputePipelineState> aps =
                impl_->pipeline("sdpa_append_kv_f32_to_f16");
            [enc setComputePipelineState:aps];
            [enc setBuffer:buf_of(&K_cur) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&V_cur) offset:0 atIndex:1];
            [enc setBuffer:buf_of(K_cache) offset:0 atIndex:2];
            [enc setBytes:&ap length:sizeof(ap) atIndex:3];
            [enc setBuffer:buf_of(V_cache) offset:0 atIndex:4];
            const NSUInteger tx = cur_seqlen == 1 ? 32 : 8;
            const NSUInteger ty = cur_seqlen == 1 ? 1 : 8;
            const NSUInteger tz = 4;
            [enc dispatchThreadgroups:
                     MTLSizeMake(
                         ((NSUInteger)std::max(head_dim, v_head_dim) + tx - 1) / tx,
                         ((NSUInteger)cur_seqlen + ty - 1) / ty,
                         ((NSUInteger)num_kv + tz - 1) / tz)
                threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
            // Attention immediately reads the cache regions written by the two
            // cache writes in this same compute encoder.
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            profile_sdpa_stage("SDPA.append");
        }

        // 2) Attention compute.
        SdpaParams sp{};
        sp.num_heads = num_heads;
        sp.num_kv_heads = num_kv;
        sp.head_dim = head_dim;
        sp.v_head_dim = v_head_dim;
        sp.src_seqlen = src_seqlen;
        sp.dst_seqlen = dst_seqlen;
        sp.past_seqlen = past;
        sp.max_seq_len = max_seq;
        sp.causal = causal;
        sp.scale = scale;
        sp.q_offset = eoffset(Q);
        sp.q_stride_pos = estride(Q, 1);
        sp.q_stride_head = estride(Q, 2);
        sp.k_cache_offset = k_cache_eoff;
        sp.v_cache_offset = v_cache_eoff;
        sp.o_offset = eoffset(out);
        sp.o_stride_pos = estride(out, 1);
        sp.o_stride_head = estride(out, 2);
        sp.has_mask = mask ? 1 : 0;
        sp.mask_offset = mask ? eoffset(*mask) : 0;
        sp.mask_stride_row = mask ? estride(*mask, 1) : 0;

        bool decode_path = (src_seqlen == 1);
        // Prefill SDPA routing:
        //   default -> sdpa_prefill_fa2_f32 (flash attention: query-split,
        //              direct-global K/V MMA, threadgroup-O elementwise rescale).
        // FA2 requires DK/DV % 8 and <= 256. C=64 and PV is padded to
        // 64, so both 8x8 QK and PV stripes divide evenly across NSG=8.
        constexpr bool use_simple = false;
        const int FA2_NSG = 8;
        const int FA2_Q =
            src_seqlen >= 16 &&
                    head_dim <= 192 &&
                    v_head_dim <= 128
                ? 16
                : 8;
        // fa2 declares function constants without defaults, so it can ONLY be built
        // via the specialized (pipeline_fa2) path — never the plain name-keyed
        // pipeline. Probe the specialized pipeline directly as the guard.
        id<MTLComputePipelineState> fa2_ps = nil;
        bool fa2_pre = !decode_path && !use_simple
            && (head_dim % 8 == 0) && (v_head_dim % 8 == 0)
            && head_dim <= 256 && v_head_dim <= 256;
        if (fa2_pre)
            fa2_ps = impl_->pipeline_fa2(
                head_dim, v_head_dim, FA2_NSG, FA2_Q);
        bool fa2_ok = (fa2_ps != nil);

        const bool decode_192_128 = decode_path && head_dim == 192 && v_head_dim == 128
            && std::getenv("MOLLM_METAL_SDPA_DECODE_GENERIC") == nullptr;
        const bool decode_192_128_multi = decode_192_128 && dst_seqlen >= 768;
        const bool decode_128_128 = decode_path
            && head_dim == 128 && v_head_dim == 128
            && std::getenv("MOLLM_METAL_SDPA_DECODE_GENERIC") == nullptr;
        const bool decode_256_256 = decode_path
            && head_dim == 256 && v_head_dim == 256
            && std::getenv("MOLLM_METAL_SDPA_DECODE_GENERIC") == nullptr;
        const char* kname = decode_192_128_multi ? "sdpa_decode_192_128_partial_f32"
                           : decode_192_128 ? "sdpa_decode_192_128_f32"
                           : decode_128_128 ? "sdpa_decode_128_128_f32"
                           : decode_256_256 ? "sdpa_decode_256_256_f32"
                           : decode_path ? "sdpa_decode_f32"
                           : "sdpa_prefill_f32";
        // FA2 uses its DK/DV/NSG-specialized pipeline; all other paths use
        // name-keyed pipelines.
        id<MTLComputePipelineState> ps =
            fa2_ok ? fa2_ps : impl_->pipeline(kname);
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&Q) offset:0 atIndex:0];
        [enc setBuffer:(K_cache?buf_of(K_cache):buf_of(&K_cur)) offset:0 atIndex:1];
        [enc setBuffer:(V_cache?buf_of(V_cache):buf_of(&V_cur)) offset:0 atIndex:2];
        [enc setBuffer:buf_of(&out) offset:0 atIndex:4];
        // Buffer index 5 (mask) must always be bound; use Q as a dummy if no mask.
        [enc setBuffer:(mask?buf_of(mask):buf_of(&Q)) offset:0 atIndex:5];
        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
        if (decode_192_128_multi) {
            constexpr int nparts = 32;
            const size_t partial_bytes =
                (size_t)num_heads * nparts * (128 + 2) * sizeof(float);
            void* partial_h = impl_->pool->acquire(partial_bytes);
            id<MTLBuffer> partial = (__bridge id<MTLBuffer>)partial_h;
            impl_->commands->pending_free.push_back({partial_h, partial_bytes});
            [enc setBuffer:partial offset:0 atIndex:7];
            [enc setBytes:&nparts length:sizeof(nparts) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(nparts,(NSUInteger)num_heads,1)
                threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:
                impl_->pipeline("sdpa_decode_192_128_reduce_f32")];
            [enc setBuffer:partial offset:0 atIndex:7];
            [enc setBuffer:buf_of(&out) offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:3];
            [enc setBytes:&nparts length:sizeof(nparts) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_heads,1,1)
                threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        } else if (decode_path) {
            // The fused 192/128 path uses eight SIMD groups; the generic path
            // uses 256 threads to split the score and output loops.
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_heads,1,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        } else if (fa2_ok) {
            // Flash attention: one threadgroup (NSG simdgroups, 32*NSG threads) per
            // (query-tile=Q, head). Threadgroup memory:
            //   sq[Q*DK] half + so[Q*PV] float + ss[Q*SH] float,
            //   C=64, SH=C+40 (bank-conflict padding), and PV=PAD(DV,64).
            const int PV = ((v_head_dim + 63) / 64) * 64;
            const int SH = 64 + 40;
            size_t tg_bytes = (size_t)FA2_Q * head_dim * 2   // sq (half)
                            + (size_t)FA2_Q * PV * 4         // so (float)
                            + (size_t)FA2_Q * SH * 4;        // ss (float)
            [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
            NSUInteger q_tiles = ((NSUInteger)src_seqlen + FA2_Q - 1) / FA2_Q;
            [enc dispatchThreadgroups:MTLSizeMake(q_tiles,(NSUInteger)num_heads,1)
                threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)FA2_NSG, 1)];
        } else {
            // One SIMD group (32 lanes) per (query pos, head); 4 groups/tg.
            const NSUInteger sg_per_tg = 4;
            NSUInteger nq = (NSUInteger)num_heads * (NSUInteger)src_seqlen;
            NSUInteger tgc = (nq + sg_per_tg - 1) / sg_per_tg;
            [enc dispatchThreadgroups:MTLSizeMake(tgc,1,1)
                threadsPerThreadgroup:MTLSizeMake(sg_per_tg*32,1,1)];
        }
        break;
    }

    case OpType::TILE: {
        // MLA: broadcast k_rope [rope_dim, seq, 1] -> [.., .., num_heads] along
        // dim 2. Only the dim-2 fast path is implemented (all MLA needs).
        const Tensor& src = *inputs[0];
        int reps[4] = {1,1,1,1};
        for (int i = 0; i < 4 && i < (int)params.i32.size(); i++) reps[i] = params.i32[i];
        if (!(reps[0]==1 && reps[1]==1 && reps[3]==1 && reps[2]>=1)) {
            fprintf(stderr, "MetalBackend: TILE only supports dim-2 broadcast "
                            "(reps=%d,%d,%d,%d)\n", reps[0],reps[1],reps[2],reps[3]);
            assert(false && "metal TILE: dim-2 only");
            break;
        }
        TensorDesc d{};
        d.shape[0]=(int)src.shape[0]; d.shape[1]=(int)src.shape[1];
        d.shape[2]=reps[2];           d.shape[3]=1;
        for (int i=0;i<4;i++) d.stride[i]=estride(src,i);
        d.offset = eoffset(src);
        id<MTLComputePipelineState> ps = impl_->pipeline("tile_dim2_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&src) offset:0 atIndex:0];
        [enc setBuffer:buf_of(output) offset:0 atIndex:2];
        [enc setBytes:&d length:sizeof(d) atIndex:3];
        const NSUInteger tx = 64, ty = 4;
        MTLSize tgs = MTLSizeMake(tx, ty, 1);
        MTLSize tgc = MTLSizeMake(((NSUInteger)d.shape[0]+tx-1)/tx,
                                  ((NSUInteger)d.shape[1]+ty-1)/ty,
                                  (NSUInteger)d.shape[2]);
        [enc dispatchThreadgroups:tgc threadsPerThreadgroup:tgs];
        break;
    }

    case OpType::CONCAT: {
        // MLA: q_full=[q_nope|q_rope], k_full=[k_nope|k_rope] along dim 0. Only
        // dim-0 (the MLA case) is implemented. One dispatch per input slab.
        int dim = params.i32.size()>0 ? params.i32[0] : 0;
        if (dim != 0) {
            fprintf(stderr, "MetalBackend: CONCAT only supports dim=0 (got %d)\n", dim);
            assert(false && "metal CONCAT: dim-0 only");
            break;
        }
        id<MTLComputePipelineState> ps = impl_->pipeline("concat_dim0_f32");
        [enc setComputePipelineState:ps];
        int dim_offset = 0;
        for (size_t i = 0; i < inputs.size(); i++) {
            if (!inputs[i] || !inputs[i]->device.buffer) continue;
            const Tensor& src = *inputs[i];
            ConcatParams p{};
            for (int k=0;k<4;k++){ p.shape[k]=(int)src.shape[k]; p.stride[k]=estride(src,k); }
            p.offset = eoffset(src);
            p.dim_offset = dim_offset;
            p.out_shape0 = (int)output->shape[0];
            [enc setBuffer:buf_of(&src) offset:0 atIndex:0];
            [enc setBuffer:buf_of(output) offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            const NSUInteger tx = 64, ty = 4;
            MTLSize tgs = MTLSizeMake(tx, ty, 1);
            MTLSize tgc = MTLSizeMake(((NSUInteger)p.shape[0]+tx-1)/tx,
                                      ((NSUInteger)p.shape[1]+ty-1)/ty,
                                      (NSUInteger)p.shape[2]);
            [enc dispatchThreadgroups:tgc threadsPerThreadgroup:tgs];
            dim_offset += (int)src.shape[0];
        }
        break;
    }

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
            impl_->ssd_moe_layers[ssd_gate->spec.layer] = {
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
            impl_->has_tensor && supported_router &&
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
            if (impl_->commands->profile)
                profile_label += "[S=" + std::to_string(seq) + "]";
            size_t idx_bytes=(size_t)seq*top_k*sizeof(int);
            size_t tw_bytes=(size_t)seq*top_k*sizeof(float);
            size_t logits_bytes=(size_t)seq*num_experts*sizeof(float);
            size_t merged_bytes=(size_t)seq*top_k*2*intermediate*sizeof(float);
            void* idx_h=impl_->pool->acquire(idx_bytes), *tw_h=impl_->pool->acquire(tw_bytes);
            void* logits_h=impl_->pool->acquire(logits_bytes);
            void* merged_h=impl_->pool->acquire(merged_bytes);
            id<MTLBuffer> idx=(__bridge id<MTLBuffer>)idx_h;
            id<MTLBuffer> tw=(__bridge id<MTLBuffer>)tw_h;
            id<MTLBuffer> logits=(__bridge id<MTLBuffer>)logits_h;
            id<MTLBuffer> merged=(__bridge id<MTLBuffer>)merged_h;
            auto profile_resident_moe_stage = [&](const char* label) {
                if (!impl_->commands->profile || ssd_w4) return;
                if (impl_->commands->enc) {
                    [impl_->commands->enc endEncoding];
                    impl_->commands->enc = nil;
                }
                if (impl_->commands->cmd) {
                    [impl_->commands->cmd commit];
                    [impl_->commands->cmd waitUntilCompleted];
                    const double gpu_ms =
                        (impl_->commands->cmd.GPUEndTime -
                         impl_->commands->cmd.GPUStartTime) * 1000.0;
                    const std::string stage_label =
                        std::string(label) +
                        "[S=" + std::to_string(seq) + "]";
                    auto& stat = impl_->commands->op_stats[stage_label];
                    stat.gpu_ms += gpu_ms;
                    stat.calls += 1;
                }
                impl_->commands->cmd = [impl_->commands->queue commandBuffer];
                impl_->commands->enc = [impl_->commands->cmd computeCommandEncoder];
                enc = impl_->commands->enc;
            };
            const Impl::SsdMoeLayerInfo* predicted_layer = nullptr;
            void* predicted_idx_h = nullptr;
            void* predicted_tw_h = nullptr;
            size_t predicted_idx_bytes = 0;
            size_t predicted_tw_bytes = 0;
            MetalSsdSharedExpert::Work shared_work;
            if (ssd_w4 && impl_->ssd_cross_layer_prefetch) {
                auto next = impl_->ssd_moe_layers.find(
                    ssd_gate->spec.layer + 1);
                if (next != impl_->ssd_moe_layers.end() &&
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
                        impl_->pool->acquire(predicted_idx_bytes);
                    predicted_tw_h =
                        impl_->pool->acquire(predicted_tw_bytes);
                }
            }
            MoeW4Params mp{};
            mp.hidden=hidden_size;mp.experts=num_experts;mp.top_k=top_k;
            mp.intermediate=intermediate;mp.seq_len=seq;mp.n_group=std::max(1,n_group);
            mp.topk_group=std::max(1,topk_group);mp.norm_topk=norm_topk;
            mp.routed_scale=routed_scale;mp.hidden_offset=eoffset(x);mp.output_offset=eoffset(*output);
            mp.hidden_row_stride=estride(x,1);mp.output_row_stride=estride(*output,1);
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
                if (!impl_->finish_ssd_prefix(
                        ssd_gate->spec.layer, "pre-router command")) {
                    break;
                }
                enc = nil;

                const auto* input_bytes =
                    static_cast<const uint8_t*>([buf_of(&x) contents]) +
                    x.device.offset;
                Tensor cpu_input = Tensor::create(
                    Precision::FP32, MemoryType::EXTERNAL,
                    hidden_size, seq, 1, 1,
                    const_cast<uint8_t*>(input_bytes));
                if (has_shared &&
                    !impl_->ssd_shared_expert->submit(
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
                bool routed = impl_->route_moe_on_cpu(
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
                    routed = impl_->route_moe_on_cpu(
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
                    resident_qx_h = impl_->pool->acquire(qx_bytes);
                    resident_sx_h = impl_->pool->acquire(sx_bytes);
                    [enc setComputePipelineState:
                             (fuse_router_quant_small
                                  ? impl_->pipeline_small_m(
                                        "moe_router_quantize_bg128_small_m",
                                        seq)
                                  : impl_->pipeline(
                                        "moe_router_quantize_bg128"))];
                    [enc setBuffer:buf_of(&x) offset:0 atIndex:0];
                    [enc setBuffer:buf_of(&router)
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
                    router_mp.a_offset = eoffset(x);
                    router_mp.b_offset = 0;
                    router_mp.c_offset = 0;
                    router_mp.a_row_stride = estride(x, 1);
                    router_mp.b_row_stride = hidden_size;
                    router_mp.c_row_stride = num_experts;
                    router_mp.activation = 0;
                    router_mp.act_n_begin = 0;
                    router_mp.act_n_len = -1;
                    [enc setBuffer:buf_of(&x) offset:0 atIndex:0];
                    [enc setBuffer:buf_of(&router)
                            offset:router.device.offset atIndex:1];
                    [enc setBuffer:logits offset:0 atIndex:2];
                    [enc setBytes:&router_mp
                           length:sizeof(router_mp) atIndex:3];
                    constexpr int router_rows_per_tg = 2;
                    id<MTLComputePipelineState> router_ps =
                        impl_->pipeline_gemv2(router_rows_per_tg);
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
                    router_mp.a_offset = eoffset(x);
                    router_mp.b_offset = 0;
                    router_mp.c_offset = 0;
                    router_mp.a_row_stride = estride(x, 1);
                    router_mp.b_row_stride = hidden_size;
                    router_mp.c_row_stride = num_experts;
                    router_mp.activation = 0;
                    router_mp.act_n_begin = 0;
                    router_mp.act_n_len = -1;
                    const int router_groups = std::min(
                        mollm::metal::gemv_nsg_cap(), (hidden_size + 127) / 128);
                    [enc setComputePipelineState:
                             impl_->pipeline_small_m(
                                 "gemv_small_m_f32a_f16b_f32c", seq)];
                    [enc setBuffer:buf_of(&x) offset:0 atIndex:0];
                    [enc setBuffer:buf_of(&router)
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
                    router_mp.a_offset = eoffset(x);
                    router_mp.b_offset = 0;
                    router_mp.c_offset = 0;
                    router_mp.a_row_stride = estride(x, 1);
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
                        activation = buf_of(&x);
                        activation_offset = x.device.offset;
                    } else {
                        const size_t activation_bytes =
                            (size_t)seq * (size_t)hidden_size *
                            sizeof(uint16_t);
                        void* activation_h =
                            impl_->pool->acquire(activation_bytes);
                        activation =
                            (__bridge id<MTLBuffer>)activation_h;
                        impl_->commands->pending_free.push_back(
                            {activation_h, activation_bytes});

                        [enc setComputePipelineState:
                                 impl_->pipeline(
                                     "matmul_cast_f32_to_f16")];
                        [enc setBuffer:buf_of(&x) offset:0 atIndex:0];
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
                             impl_->pipeline(
                                 small_router
                                     ? "gemm_tensor_router_f32a_f16b_f32c"
                                     : "gemm_tensor_direct_f16a_f16b_f32c")];
                    [enc setBuffer:activation
                            offset:activation_offset atIndex:0];
                    [enc setBuffer:buf_of(&router)
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
                        ? impl_->pipeline_moe_select_parallel(
                              router_score_func == 1,
                              router_score_func == 1 &&
                                  n_group > 1)
                        : impl_->pipeline(
                              router_score_func == 0
                                  ? "moe_select_softmax"
                                  : "moe_select_sigmoid");
                [enc setComputePipelineState:select_pipeline];
                [enc setBuffer:logits offset:0 atIndex:0];
                [enc setBuffer:idx offset:0 atIndex:1];
                [enc setBuffer:tw offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                if (bias) {
                    [enc setBuffer:buf_of(bias)
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
                if (w8pc && seq >= 64 && impl_->has_tensor &&
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

                    void* qx_h = impl_->pool->acquire(qx_bytes);
                    void* sx_h = impl_->pool->acquire(sx_bytes);
                    void* qi_h = impl_->pool->acquire(qi_bytes);
                    void* si_h = impl_->pool->acquire(si_bytes);
                    void* selected_h =
                        impl_->pool->acquire(selected_bytes);
                    void* counts_h =
                        impl_->pool->acquire(expert_counts_bytes);
                    void* routes_h =
                        impl_->pool->acquire(expert_routes_bytes);
                    void* jobs_h = impl_->pool->acquire(jobs_bytes);
                    void* job_count_h =
                        impl_->pool->acquire(job_count_bytes);
                    void* dispatch_h =
                        impl_->pool->acquire(dispatch_bytes);
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
                             impl_->pipeline(
                                 "moe_reset_expert_counts")];
                    [enc setBuffer:counts offset:0 atIndex:0];
                    [enc setBuffer:job_count offset:0 atIndex:1];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    grid1d(num_experts);
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:
                             impl_->pipeline(
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
                             impl_->pipeline(
                                 "moe_build_grouped_jobs")];
                    [enc setBuffer:counts offset:0 atIndex:0];
                    [enc setBuffer:jobs offset:0 atIndex:1];
                    [enc setBuffer:job_count offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc setBuffer:jobs offset:jobs_queue_bytes atIndex:4];
                    grid1d(num_experts);
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    [enc setComputePipelineState:
                             impl_->pipeline(
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
                                 impl_->pipeline(
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
                                 impl_->pipeline(pipeline_name)];
                        [enc setBuffer:activation offset:0 atIndex:0];
                        [enc setBuffer:buf_of(&weight)
                                offset:weight.device.offset atIndex:1];
                        [enc setBuffer:destination offset:0 atIndex:2];
                        [enc setBytes:&gp length:sizeof(gp) atIndex:3];
                        [enc setBuffer:activation_scales
                                offset:0 atIndex:4];
                        [enc setBuffer:scales_buf_of(&weight)
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
                        buf_of(&x), (uint)eoffset(x), seq,
                        hidden_size, estride(x, 1), qx, sx);
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
                             impl_->pipeline("moe_combine_selected")];
                    [enc setBuffer:selected offset:0 atIndex:0];
                    [enc setBuffer:buf_of(output) offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc setBuffer:tw offset:0 atIndex:4];
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (hidden_size + 63) / 64,
                                 (seq + 3) / 4, 1)
                        threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];

                    impl_->commands->pending_free.push_back({qx_h, qx_bytes});
                    impl_->commands->pending_free.push_back({sx_h, sx_bytes});
                    impl_->commands->pending_free.push_back({qi_h, qi_bytes});
                    impl_->commands->pending_free.push_back({si_h, si_bytes});
                    impl_->commands->pending_free.push_back(
                        {selected_h, selected_bytes});
                    impl_->commands->pending_free.push_back(
                        {counts_h, expert_counts_bytes});
                    impl_->commands->pending_free.push_back(
                        {routes_h, expert_routes_bytes});
                    impl_->commands->pending_free.push_back({jobs_h, jobs_bytes});
                    impl_->commands->pending_free.push_back(
                        {job_count_h, job_count_bytes});
                    impl_->commands->pending_free.push_back(
                        {dispatch_h, dispatch_bytes});
                    impl_->commands->pending_free.push_back({idx_h, idx_bytes});
                    impl_->commands->pending_free.push_back({tw_h, tw_bytes});
                    impl_->commands->pending_free.push_back({logits_h, logits_bytes});
                    impl_->commands->pending_free.push_back({merged_h, merged_bytes});
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
                    void* qx_h = impl_->pool->acquire(qx_bytes);
                    void* sx_h = impl_->pool->acquire(sx_bytes);
                    void* qi_h = impl_->pool->acquire(qi_bytes);
                    void* si_h = impl_->pool->acquire(si_bytes);
                    void* selected_h =
                        impl_->pool->acquire(selected_bytes);
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
                                 impl_->pipeline("quantize_act_i8")];
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
                                 impl_->pipeline(
                                     "gemv_selected_experts_w8_i8a_i8b_f32c")];
                        [enc setBuffer:activation offset:0 atIndex:0];
                        [enc setBuffer:buf_of(&weight)
                                offset:weight.device.offset atIndex:1];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                        [enc setBuffer:activation_scales
                                offset:0 atIndex:4];
                        [enc setBuffer:scales_buf_of(&weight)
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
                        buf_of(&x), (uint)eoffset(x), seq,
                        hidden_size, estride(x, 1), qx, sx);
                    selected_w8(
                        qx, sx, gu, 2 * intermediate, hidden_size,
                        2 * intermediate, seq, top_k,
                        merged, 2 * intermediate);
                    profile_resident_moe_stage("MOE.gate_up_w8a8");

                    [enc setComputePipelineState:
                             impl_->pipeline(
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
                             impl_->pipeline("moe_combine_selected")];
                    [enc setBuffer:selected offset:0 atIndex:0];
                    [enc setBuffer:buf_of(output) offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc setBuffer:tw offset:0 atIndex:4];
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 (hidden_size + 255) / 256, 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(256, 1, 1)];
                    impl_->commands->pending_free.push_back({qx_h, qx_bytes});
                    impl_->commands->pending_free.push_back({sx_h, sx_bytes});
                    impl_->commands->pending_free.push_back({qi_h, qi_bytes});
                    impl_->commands->pending_free.push_back({si_h, si_bytes});
                    impl_->commands->pending_free.push_back(
                        {selected_h, selected_bytes});
                    impl_->commands->pending_free.push_back({idx_h, idx_bytes});
                    impl_->commands->pending_free.push_back({tw_h, tw_bytes});
                    impl_->commands->pending_free.push_back({logits_h, logits_bytes});
                    impl_->commands->pending_free.push_back({merged_h, merged_bytes});
                    break;
                }
                const bool exact_decode =
                    seq == 1 && gu.groups_per_row == 1 &&
                    down.groups_per_row == 1;
                const bool fused_small_gate_up =
                    seq >= 2 && seq <= 4;
                const bool compact_down = seq >= 2 && seq <= 4;
                [enc setComputePipelineState:
                         impl_->pipeline(
                             exact_decode
                                 ? "moe_gate_up_w8_precise"
                                 : (fused_small_gate_up
                                        ? "moe_gate_up_swiglu_w8_r4"
                                        : "moe_gate_up_w8"))];
                [enc setBuffer:buf_of(&x) offset:0 atIndex:0];
                [enc setBuffer:buf_of(&gu)
                        offset:gu.device.offset atIndex:1];
                [enc setBuffer:merged offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc setBuffer:scales_buf_of(&gu)
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
                             impl_->pipeline("moe_swiglu_selected")];
                    [enc setBuffer:merged offset:0 atIndex:0];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    const NSUInteger sw_n =
                        (NSUInteger)seq * top_k * intermediate;
                    [enc dispatchThreadgroups:
                             MTLSizeMake((sw_n + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }

                [enc setComputePipelineState:
                         impl_->pipeline(
                             exact_decode
                                 ? "moe_down_combine_w8_precise"
                                 : (compact_down
                                        ? "moe_down_combine_w8_r4"
                                        : "moe_down_combine_w8"))];
                [enc setBuffer:merged offset:0 atIndex:0];
                [enc setBuffer:buf_of(&down)
                        offset:down.device.offset atIndex:1];
                [enc setBuffer:buf_of(output) offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc setBuffer:scales_buf_of(&down)
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

                impl_->commands->pending_free.push_back({idx_h, idx_bytes});
                impl_->commands->pending_free.push_back({tw_h, tw_bytes});
                impl_->commands->pending_free.push_back({logits_h, logits_bytes});
                impl_->commands->pending_free.push_back({merged_h, merged_bytes});
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
                if (!impl_->ssd_cache->acquire(
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
                    impl_->ssd_cache->acquire(
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
                    impl_->pool->release(
                        predicted_idx_h, predicted_idx_bytes);
                    impl_->pool->release(
                        predicted_tw_h, predicted_tw_bytes);
                    predicted_layer = nullptr;
                }

                impl_->commands->cmd = [impl_->commands->queue commandBuffer];
                impl_->commands->cmd.label = @"mollm Metal SSD expert";
                const int selections = seq * top_k;
                const size_t qx_bytes = (size_t)seq * hidden_size;
                const size_t sx_bytes = (size_t)seq * sizeof(float);
                void* qx_h = impl_->pool->acquire(qx_bytes);
                void* sx_h = impl_->pool->acquire(sx_bytes);
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
                            [impl_->commands->cmd
                                encodeWaitForEvent:view.gate_ready_event
                                             value:view.gate_ready_value];
                        }
                    }
                };
                if (ready_selections == 0)
                    encode_gate_waits(0);
                impl_->commands->enc = [impl_->commands->cmd computeCommandEncoder];
                impl_->commands->enc.label = @"mollm Metal SSD expert";
                enc = impl_->commands->enc;

                const size_t qi_bytes = (size_t)selections * intermediate;
                const size_t si_bytes = (size_t)selections * sizeof(float);
                const size_t selected_bytes =
                    (size_t)selections * hidden_size * sizeof(float);
                const size_t slot_offsets_bytes =
                    (size_t)selections * 2 * sizeof(uint64_t);
                const size_t selection_indices_bytes =
                    (size_t)selections * sizeof(uint32_t);
                void* qi_h = impl_->pool->acquire(qi_bytes);
                void* si_h = impl_->pool->acquire(si_bytes);
                void* selected_h = impl_->pool->acquire(selected_bytes);
                void* slot_offsets_h =
                    impl_->pool->acquire(slot_offsets_bytes);
                void* selection_indices_h =
                    impl_->pool->acquire(selection_indices_bytes);
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
                             impl_->pipeline("quantize_act_i8")];
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
                    quantize(buf_of(&x), (uint)eoffset(x), seq, hidden_size,
                             estride(x, 1), qx, sx);
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
                             impl_->pipeline(
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
                    impl_->commands->enc = nil;
                    encode_gate_waits(ready_selections);
                    impl_->commands->enc =
                        [impl_->commands->cmd computeCommandEncoder];
                    impl_->commands->enc.label =
                        @"mollm pending Metal SSD expert";
                    enc = impl_->commands->enc;
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
                         impl_->pipeline("moe_swiglu_selected")];
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
                    impl_->commands->enc = nil;
                    for (const auto& view : expert_views) {
                        if (!view.down_ready_event ||
                            view.down_ready_event.signaledValue >=
                                view.down_ready_value)
                            continue;
                        void* event_key =
                            (__bridge void*)view.down_ready_event;
                        if (waited_down_events.insert(event_key).second) {
                            [impl_->commands->cmd
                                encodeWaitForEvent:view.down_ready_event
                                             value:view.down_ready_value];
                        }
                    }
                    impl_->commands->enc =
                        [impl_->commands->cmd computeCommandEncoder];
                    impl_->commands->enc.label =
                        @"mollm Metal SSD down experts";
                    enc = impl_->commands->enc;
                }

                selected_bg128(
                    qi, si, expert_views[0].down,
                    (size_t)selections * sizeof(uint64_t),
                    0, selections,
                    hidden_size, intermediate,
                    ssd_down->spec.groups_per_row, selections, 1,
                    selected, hidden_size);

                [enc setComputePipelineState:
                         impl_->pipeline(
                             "moe_combine_selected")];
                [enc setBuffer:selected offset:0 atIndex:0];
                [enc setBuffer:buf_of(output)
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
                    impl_->commands->enc = nil;
                    id<MTLSharedEvent> shared_event =
                        impl_->ssd_shared_expert->event();
                    [impl_->commands->cmd
                        encodeWaitForEvent:shared_event
                                   value:shared_ready_value];
                    impl_->commands->enc = [impl_->commands->cmd computeCommandEncoder];
                    impl_->commands->enc.label = @"mollm combine SSD experts";
                    enc = impl_->commands->enc;
                    const uint count = (uint)hidden_size;
                    [enc setComputePipelineState:
                             impl_->pipeline("add_inplace_f32")];
                    [enc setBuffer:buf_of(output)
                            offset:output->device.offset
                           atIndex:0];
                    [enc setBuffer:shared_output offset:0 atIndex:1];
                    [enc setBytes:&count length:sizeof(count) atIndex:3];
                    [enc dispatchThreadgroups:
                             MTLSizeMake((hidden_size + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    impl_->commands->pending_free.push_back(
                        {shared_inter_h, shared_inter_bytes});
                    impl_->commands->pending_free.push_back(
                        {shared_qx_h, qx_bytes});
                    impl_->commands->pending_free.push_back(
                        {shared_sx_h, sx_bytes});
                    impl_->commands->pending_free.push_back(
                        {shared_qinter_h, shared_qinter_bytes});
                    impl_->commands->pending_free.push_back(
                        {shared_qinter_scale_h, sizeof(float)});
                    impl_->commands->pending_free.push_back(
                        {shared_scale_h, sizeof(float)});
                    impl_->commands->pending_free.push_back(
                        {shared_output_h, shared_output_bytes});
                }
                impl_->commands->pending_free.push_back({qx_h, qx_bytes});
                impl_->commands->pending_free.push_back({sx_h, sx_bytes});
                impl_->commands->pending_free.push_back({qi_h, qi_bytes});
                impl_->commands->pending_free.push_back({si_h, si_bytes});
                impl_->commands->pending_free.push_back(
                    {selected_h, selected_bytes});
                impl_->commands->pending_free.push_back(
                    {slot_offsets_h, slot_offsets_bytes});
                impl_->commands->pending_free.push_back(
                    {selection_indices_h, selection_indices_bytes});
                impl_->commands->pending_free.push_back({idx_h, idx_bytes});
                impl_->commands->pending_free.push_back({tw_h, tw_bytes});
                impl_->commands->pending_free.push_back({logits_h, logits_bytes});
                impl_->commands->pending_free.push_back({merged_h, merged_bytes});
                break;
            }

            if (impl_->has_tensor) {
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
                    : impl_->pool->acquire(qx_bytes);
                void* sx_h=resident_sx_h
                    ? resident_sx_h
                    : impl_->pool->acquire(sx_bytes);
                void* qi_h=impl_->pool->acquire(qi_bytes),*si_h=impl_->pool->acquire(si_bytes);
                void* selected_h =
                    impl_->pool->acquire(selected_bytes);
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
                        impl_->pool->acquire(expert_counts_bytes);
                    expert_routes_h =
                        impl_->pool->acquire(expert_routes_bytes);
                    grouped_jobs_h =
                        impl_->pool->acquire(grouped_jobs_bytes);
                    grouped_job_count_h =
                        impl_->pool->acquire(grouped_job_count_bytes);
                    grouped_dispatch_h =
                        impl_->pool->acquire(grouped_dispatch_bytes);
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
                             impl_->pipeline(
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
                             impl_->pipeline(
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
                             impl_->pipeline(
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
                             impl_->pipeline(
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
                    if (impl_->commands->profile) {
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
                    [enc setComputePipelineState:impl_->pipeline(
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
                    quantize(buf_of(&x),(uint)eoffset(x),seq,hidden_size,
                             estride(x,1),native_gu_group,qx,sx);
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
                                 impl_->pipeline(
                                     "gemv_selected_experts_bg128_"
                                     "i8a_i4b_f32c")];
                        [enc setBuffer:a offset:0 atIndex:0];
                        [enc setBuffer:buf_of(&w)
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
                                 impl_->pipeline(
                                     "gemv_selected_experts_bg32_"
                                     "i8a_i4b_f32c")];
                        [enc setBuffer:a offset:0 atIndex:0];
                        [enc setBuffer:buf_of(&w)
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
                                 impl_->pipeline(
                                     "gemm_selected_w4a8_i8a_i4b_f32c")];
                        [enc setBuffer:a offset:0 atIndex:0];
                        [enc setBuffer:buf_of(&w)
                                offset:w.device.offset atIndex:1];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                        [enc setBuffer:sa offset:0 atIndex:4];
                        [enc setBuffer:buf_of(&w)
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
                    [enc setBuffer:buf_of(&weight)
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
                             impl_->pipeline_grouped_moe(
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
                             impl_->pipeline(
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
                                 impl_->pipeline(
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
                                 impl_->pipeline(
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
                         impl_->pipeline(
                             "moe_combine_selected")];
                [enc setBuffer:selected offset:0 atIndex:0];
                [enc setBuffer:buf_of(output)
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
                impl_->commands->pending_free.push_back({qx_h,qx_bytes});impl_->commands->pending_free.push_back({sx_h,sx_bytes});
                impl_->commands->pending_free.push_back({qi_h,qi_bytes});impl_->commands->pending_free.push_back({si_h,si_bytes});
                impl_->commands->pending_free.push_back(
                    {selected_h,selected_bytes});
                if (grouped_prefill) {
                    impl_->commands->pending_free.push_back(
                        {expert_counts_h, expert_counts_bytes});
                    impl_->commands->pending_free.push_back(
                        {expert_routes_h, expert_routes_bytes});
                    impl_->commands->pending_free.push_back(
                        {grouped_jobs_h, grouped_jobs_bytes});
                    impl_->commands->pending_free.push_back(
                        {grouped_job_count_h,
                         grouped_job_count_bytes});
                    impl_->commands->pending_free.push_back(
                        {grouped_dispatch_h,
                         grouped_dispatch_bytes});
                }
                impl_->commands->pending_free.push_back({idx_h,idx_bytes});impl_->commands->pending_free.push_back({tw_h,tw_bytes});
                impl_->commands->pending_free.push_back({logits_h,logits_bytes});impl_->commands->pending_free.push_back({merged_h,merged_bytes});
                break;
            }
#endif

            [enc setComputePipelineState:impl_->pipeline("moe_gate_up_w4")];
            [enc setBuffer:buf_of(&x) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&gu) offset:gu.device.offset atIndex:1];
            [enc setBuffer:merged offset:0 atIndex:2];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
            [enc setBuffer:buf_of(&gu) offset:gu.device.offset+gu_rows*(hidden_size/2) atIndex:4];
            [enc setBuffer:idx offset:0 atIndex:5];
            [enc setThreadgroupMemoryLength:4*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((2*intermediate+3)/4,top_k,seq)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [enc setComputePipelineState:impl_->pipeline("moe_swiglu_selected")];
            [enc setBuffer:merged offset:0 atIndex:0];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
            NSUInteger sw_n=(NSUInteger)seq*top_k*intermediate;
            [enc dispatchThreadgroups:MTLSizeMake((sw_n+255)/256,1,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            [enc setComputePipelineState:impl_->pipeline("moe_down_combine_w4")];
            [enc setBuffer:merged offset:0 atIndex:0];
            [enc setBuffer:buf_of(&down) offset:down.device.offset atIndex:1];
            [enc setBuffer:buf_of(output) offset:0 atIndex:2];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
            [enc setBuffer:buf_of(&down) offset:down.device.offset+down_rows*(intermediate/2) atIndex:4];
            [enc setBuffer:idx offset:0 atIndex:5];[enc setBuffer:tw offset:0 atIndex:6];
            [enc setThreadgroupMemoryLength:4*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((hidden_size+3)/4,seq,1)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            impl_->commands->pending_free.push_back({idx_h,idx_bytes});
            impl_->commands->pending_free.push_back({tw_h,tw_bytes});
            impl_->commands->pending_free.push_back({logits_h,logits_bytes});
            impl_->commands->pending_free.push_back({merged_h,merged_bytes});
            break;
        }

        // Generic correctness fallback for FP16/W8/shared-expert variants.
        if (impl_->commands->enc) { [impl_->commands->enc endEncoding]; impl_->commands->enc = nil; }
        if (impl_->commands->cmd) {
            [impl_->commands->cmd commit];
            [impl_->commands->cmd waitUntilCompleted];
            if (impl_->commands->cmd.status == MTLCommandBufferStatusError) {
                NSError* e = impl_->commands->cmd.error;
                fprintf(stderr, "MetalBackend: pre-MOE command buffer error: %s\n",
                        e ? e.localizedDescription.UTF8String : "?");
                impl_->dispatch_failed = true;
            }
            impl_->commands->cmd = nil;
        }

        if (impl_->dispatch_failed)
            return;

        kernel_qwen3_moe(inputs, *output, thread_pool,
                         hidden_size, num_experts, top_k,
                         intermediate, shared_intermediate,
                         router_score_func, norm_topk, has_shared,
                         n_group, topk_group, routed_scale);

        impl_->commands->cmd = [impl_->commands->queue commandBuffer];
        impl_->commands->enc = [impl_->commands->cmd computeCommandEncoder];
        break;
    }

    default:
        fprintf(stderr, "MetalBackend: unsupported op %d\n", (int)op);
        assert(false && "unsupported metal op");
        break;
        }
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
        impl_->ssd_cache && impl_->ssd_cache->available() &&
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
    impl_->ssd_moe_layers[gate_source->spec.layer] = {
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
    if (!impl_->ssd_cache->acquire(
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

    if (impl_->ssd_cross_layer_prefetch) {
        const auto next = impl_->ssd_moe_layers.find(
            gate_source->spec.layer + 1);
        if (next != impl_->ssd_moe_layers.end()) {
            const Impl::SsdMoeLayerInfo& info = next->second;
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
                    impl_->ssd_cache->acquire(
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
    void* hidden_handle = impl_->pool->acquire(hidden_bytes);
    void* merged_handle = impl_->pool->acquire(merged_bytes);
    void* selected_handle = impl_->pool->acquire(selected_bytes);
    void* quantized_hidden_handle =
        impl_->pool->acquire(quantized_hidden_bytes);
    void* hidden_scale_handle = impl_->pool->acquire(hidden_scale_bytes);
    void* residual_hidden_handle =
        impl_->pool->acquire(quantized_hidden_bytes);
    void* residual_hidden_scale_handle =
        impl_->pool->acquire(hidden_scale_bytes);
    void* quantized_intermediate_handle =
        impl_->pool->acquire(quantized_intermediate_bytes);
    void* intermediate_scale_handle =
        impl_->pool->acquire(intermediate_scale_bytes);
    void* residual_intermediate_handle =
        impl_->pool->acquire(quantized_intermediate_bytes);
    void* residual_intermediate_scale_handle =
        impl_->pool->acquire(intermediate_scale_bytes);
    void* weights_handle = impl_->pool->acquire(weights_bytes);
    void* offsets_handle = impl_->pool->acquire(offsets_bytes);
    void* indices_handle = impl_->pool->acquire(indices_bytes);
    auto release_buffers = [&] {
        impl_->pool->release(hidden_handle, hidden_bytes);
        impl_->pool->release(merged_handle, merged_bytes);
        impl_->pool->release(selected_handle, selected_bytes);
        impl_->pool->release(
            quantized_hidden_handle, quantized_hidden_bytes);
        impl_->pool->release(hidden_scale_handle, hidden_scale_bytes);
        impl_->pool->release(
            residual_hidden_handle, quantized_hidden_bytes);
        impl_->pool->release(
            residual_hidden_scale_handle, hidden_scale_bytes);
        impl_->pool->release(
            quantized_intermediate_handle, quantized_intermediate_bytes);
        impl_->pool->release(
            intermediate_scale_handle, intermediate_scale_bytes);
        impl_->pool->release(
            residual_intermediate_handle, quantized_intermediate_bytes);
        impl_->pool->release(
            residual_intermediate_scale_handle, intermediate_scale_bytes);
        impl_->pool->release(weights_handle, weights_bytes);
        impl_->pool->release(offsets_handle, offsets_bytes);
        impl_->pool->release(indices_handle, indices_bytes);
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

    id<MTLCommandBuffer> command = [impl_->commands->queue commandBuffer];
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
                 impl_->pipeline("quantize_act_fp8_i8_block32")];
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
                 impl_->pipeline(
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
                 impl_->pipeline("moe_swiglu_route_bf16")];
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
                 impl_->pipeline("quantize_act_fp8_i8_block32")];
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
                 impl_->pipeline(
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
