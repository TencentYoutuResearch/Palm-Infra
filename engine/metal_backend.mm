#include "engine/metal_backend.h"
#include "graph/metal_pool.h"
#include "graph/graph.h"
#include "kernels/metal/metal_common.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <os/signpost.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef MOLLM_METALLIB_PATH
#define MOLLM_METALLIB_PATH ""
#endif

// ===========================================================================
// MetalBackend::Impl
// ===========================================================================

struct MetalBackend::Impl {
    id<MTLDevice>            device = nil;
    id<MTLCommandQueue>      queue  = nil;
    id<MTLLibrary>           library = nil;

    std::unique_ptr<MetalBufferPool> pool;

    // pipeline cache by kernel function name
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    // one MTLBuffer wrapping the whole package weight region (zero-copy mmap)
    id<MTLBuffer>            weight_buffer = nil;
    void*                    weight_base   = nullptr;
    size_t                   weight_size   = 0;

    // persistent device buffers owned by the backend (KV cache)
    std::vector<id<MTLBuffer>> persistent;

    // reusable per-key boundary input buffers (hidden/mask/cos/sin), keyed by
    // graph INPUT node name; grown on demand.
    std::unordered_map<std::string, id<MTLBuffer>> input_buffers;
    std::unordered_map<std::string, size_t> input_capacity;

    // Buffers freed during graph encoding, returned to the pool only after the
    // command buffer completes (deferred GPU execution — see free_output).
    std::vector<std::pair<void*, size_t>> pending_free;

    // GPU timing accumulators (MOLLM_METAL_GPU_TIME).
    double   gpu_time_ms = 0.0;
    uint64_t gpu_graphs  = 0;

    // Per-op-type GPU-time profiling (MOLLM_METAL_PROFILE). When on, dispatch()
    // commits+waits each op separately and attributes the command buffer's GPU
    // time to the op type. Reported (and reset) via dump_profile().
    struct OpStat { double gpu_ms = 0.0; uint64_t calls = 0; };
    std::map<int, OpStat> op_stats;  // OpType -> stat
    bool profile = false;

    // True iff the tensor-API GEMM kernel is compiled AND the GPU supports the
    // Metal 4 tensor family (M5/A19+). Set in the constructor.
    bool has_tensor = false;

    // os_signpost log for Instruments "Points of Interest" (CPU-side phase
    // markers, Apple's analogue of NVTX). Lazily created.
    os_log_t signpost_log = nullptr;
    os_log_t sp() {
        if (!signpost_log) signpost_log = os_log_create("com.mollm.metal", "profiling");
        return signpost_log;
    }

    // current command buffer / encoder for one graph run
    id<MTLCommandBuffer>        cmd = nil;
    id<MTLComputeCommandEncoder> enc = nil;

    bool ok = false;

    id<MTLComputePipelineState> pipeline(const char* name) {
        std::string key(name);
        auto it = pipelines.find(key);
        if (it != pipelines.end()) return it->second;
        id<MTLFunction> fn = [library newFunctionWithName:@(name)];
        if (!fn) {
            fprintf(stderr, "MetalBackend: kernel function '%s' not found\n", name);
            return nil;
        }
        NSError* err = nil;
        id<MTLComputePipelineState> ps =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) {
            fprintf(stderr, "MetalBackend: pipeline '%s' failed: %s\n",
                    name, err ? err.localizedDescription.UTF8String : "?");
            return nil;
        }
        pipelines[key] = ps;
        return ps;
    }

    // Specialized-pipeline cache keyed by name + function-constant tuple. Used by
    // the flash-attention prefill kernel to bake dk/dv in at compile time (Phase B).
    // Falls back to nil on failure; caller then uses the plain name-keyed pipeline.
    std::unordered_map<std::string, id<MTLComputePipelineState>> spec_pipelines;
    id<MTLComputePipelineState> pipeline_fa2(int dk, int dv) {
        char keyc[64];
        snprintf(keyc, sizeof(keyc), "fa2:dk%d:dv%d", dk, dv);
        std::string key(keyc);
        auto it = spec_pipelines.find(key);
        if (it != spec_pipelines.end()) return it->second;

        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&dk type:MTLDataTypeInt atIndex:0];  // FC_SDPA_DK
        [cv setConstantValue:&dv type:MTLDataTypeInt atIndex:1];  // FC_SDPA_DV
        NSError* err = nil;
        id<MTLFunction> fn = [library newFunctionWithName:@"sdpa_prefill_fa2_f32"
                                          constantValues:cv error:&err];
        if (!fn) {
            fprintf(stderr, "MetalBackend: fa2 specialized function failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "?");
            spec_pipelines[key] = nil;
            return nil;
        }
        id<MTLComputePipelineState> ps =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) {
            fprintf(stderr, "MetalBackend: fa2 specialized pipeline failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "?");
        }
        spec_pipelines[key] = ps;
        return ps;
    }

    // GEMV specialized by NR0 (output rows per threadgroup) via function constant 5.
    id<MTLComputePipelineState> pipeline_gemv2(int nr0) {
        char keyc[48];
        snprintf(keyc, sizeof(keyc), "gemv2:nr0%d", nr0);
        std::string key(keyc);
        auto it = spec_pipelines.find(key);
        if (it != spec_pipelines.end()) return it->second;

        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&nr0 type:MTLDataTypeInt atIndex:5];  // FC_GEMV_NR0
        NSError* err = nil;
        id<MTLFunction> fn = [library newFunctionWithName:@"gemv2_f32a_f16b_f32c"
                                          constantValues:cv error:&err];
        id<MTLComputePipelineState> ps = fn
            ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!ps) fprintf(stderr, "MetalBackend: gemv2 nr0=%d pipeline failed: %s\n",
                         nr0, err ? err.localizedDescription.UTF8String : "?");
        spec_pipelines[key] = ps;
        return ps;
    }
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

namespace {

// Resolve the MTLBuffer backing a tensor. Returns nil if the tensor has no
// device buffer.
id<MTLBuffer> buf_of(const Tensor* t) {
    if (!t || !t->device_data) return nil;
    return (__bridge id<MTLBuffer>)t->device_data;
}

// element size in bytes for a precision, for offset math.
size_t esize(Precision p) {
    switch (p) {
    case Precision::FP32: return 4;
    case Precision::FP16: return 2;
    case Precision::INT8: return 1;
    case Precision::INT4: return 1;
    }
    return 4;
}

// element stride from byte stride
int estride(const Tensor& t, int dim) {
    return (int)(t.stride[dim] / esize(t.prec));
}

// element offset into the bound buffer (device_offset is in bytes)
uint eoffset(const Tensor& t) {
    return (uint)(t.device_offset / esize(t.prec));
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
        impl_->queue = [impl_->device newCommandQueue];

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
        impl_->pool.reset(new MetalBufferPool((__bridge void*)impl_->device));
        impl_->profile = getenv("MOLLM_METAL_PROFILE") != nullptr;

        // Enable the tensor-API GEMM only if the kernel was compiled (metallib
        // built with -DMOLLM_METAL_TENSOR) AND the GPU is M5/A19+ (MTLGPUFamily
        // Metal4), and the pipeline actually loads. Overridable via env.
#ifdef MOLLM_METAL_TENSOR
        if (!getenv("MOLLM_METAL_NO_TENSOR")) {
            bool fam = false;
            if (@available(macOS 15.0, *)) {
                fam = [impl_->device supportsFamily:MTLGPUFamilyMetal4];
            }
            // Metal 4 tensor-API GEMM is correct (parity-tested) and ~2.3x faster
            // than the simdgroup path (prefill 940 vs 403 t/s). Enabled by default
            // on M5/A19+; disable with MOLLM_METAL_NO_TENSOR_GEMM for A/B testing.
            if (fam && !getenv("MOLLM_METAL_NO_TENSOR_GEMM") &&
                impl_->pipeline("gemm_tensor_f32a_f16b_f32c") != nil) {
                impl_->has_tensor = true;
            }
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
        impl_->pipelines.clear();
        impl_->spec_pipelines.clear();
        impl_->persistent.clear();
        impl_->weight_buffer = nil;
        impl_->pool.reset();
    }
}

bool MetalBackend::available() const { return impl_ && impl_->ok; }
void* MetalBackend::device() const { return impl_ ? (__bridge void*)impl_->device : nullptr; }

void MetalBackend::lm_head_gemv(const float* a_host, const Tensor& weight,
                                float* out_host, int N, int K, int activation) {
    @autoreleasepool {
        // Reusable input buffer for the activation vector; output buffer.
        void* abuf = impl_->pool->acquire((size_t)K * 4);
        void* cbuf = impl_->pool->acquire((size_t)N * 4);
        std::memcpy(MetalBufferPool::contents(abuf), a_host, (size_t)K * 4);

        MatmulParams p{};
        p.M = 1; p.N = N; p.K = K;
        p.a_offset = 0;
        p.b_offset = 0;  // bind B at its byte offset below (64-bit, no overflow)
        p.c_offset = 0;
        p.a_row_stride = K;
        p.b_row_stride = (int)weight.shape[1];  // K
        p.c_row_stride = N;
        p.activation = activation;

        id<MTLBuffer> A = (__bridge id<MTLBuffer>)abuf;
        id<MTLBuffer> B = buf_of(&weight);
        id<MTLBuffer> C = (__bridge id<MTLBuffer>)cbuf;

        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:B offset:weight.device_offset atIndex:1];
        [enc setBuffer:C offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        // Use the tuned gemv2 (NR0=2 + NSG-split K) — same win as graph GEMVs.
        // lm_head N is huge (vocab), K=hidden; the NSG K-split + 2-row reuse help.
        const int NR0 = 2, NSG = std::min(8, (K + 127) / 128);
        id<MTLComputePipelineState> ps = impl_->pipeline_gemv2(NR0);
        if (ps) {
            [enc setComputePipelineState:ps];
            [enc setThreadgroupMemoryLength:(NSUInteger)(NR0 * 32 * sizeof(float)) atIndex:0];
            NSUInteger tgcount = ((NSUInteger)N + NR0 - 1) / NR0;
            [enc dispatchThreadgroups:MTLSizeMake(tgcount,1,1)
                threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)NSG, 1)];
        } else {
            ps = impl_->pipeline("gemv_f32a_f16b_f32c");
            const NSUInteger rows_per_tg = 8;
            [enc setComputePipelineState:ps];
            NSUInteger tgcount = ((NSUInteger)N + rows_per_tg - 1) / rows_per_tg;
            [enc dispatchThreadgroups:MTLSizeMake(tgcount,1,1)
                threadsPerThreadgroup:MTLSizeMake(rows_per_tg * 32, 1, 1)];
        }
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(out_host, MetalBufferPool::contents(cbuf), (size_t)N * 4);
        impl_->pool->release(abuf, (size_t)K * 4);
        impl_->pool->release(cbuf, (size_t)N * 4);
    }
}

// ===========================================================================
// weight region + persistent buffers
// ===========================================================================

bool MetalBackend::register_weight_region(void* base, size_t size) {
    if (!impl_->ok || !base || size == 0) return false;
    @autoreleasepool {
        id<MTLBuffer> b = [impl_->device newBufferWithBytesNoCopy:base
                                                           length:size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!b) {
            fprintf(stderr, "MetalBackend: newBufferWithBytesNoCopy(%zu) failed "
                            "(maxBufferLength=%llu)\n",
                    size, (unsigned long long)impl_->device.maxBufferLength);
            return false;
        }
        impl_->weight_buffer = b;
        impl_->weight_base = base;
        impl_->weight_size = size;
    }
    return true;
}

void MetalBackend::wrap_weight(Tensor& t) {
    if (!impl_->weight_buffer || !t.data) return;
    char* base = (char*)impl_->weight_base;
    char* ptr  = (char*)t.data;
    if (ptr < base || ptr >= base + impl_->weight_size) {
        // Weight lies outside the registered region — allocate a copy instead.
        alloc_persistent(t, t.nbytes());
        std::memcpy(t.data, ptr, t.nbytes());
        return;
    }
    t.device_data = (__bridge void*)impl_->weight_buffer;
    t.device_offset = (size_t)(ptr - base);
}

void MetalBackend::alloc_persistent(Tensor& t, size_t nbytes) {
    @autoreleasepool {
        id<MTLBuffer> b = [impl_->device newBufferWithLength:nbytes
                                                     options:MTLResourceStorageModeShared];
        impl_->persistent.push_back(b);
        t.device_data = (__bridge void*)b;
        t.device_offset = 0;
        t.data = [b contents];
    }
}

void MetalBackend::upload_input(Tensor& t, const std::string& key,
                                const void* host_src, size_t nbytes) {
    id<MTLBuffer> buf = nil;
    auto it = impl_->input_buffers.find(key);
    if (it != impl_->input_buffers.end() && impl_->input_capacity[key] >= nbytes) {
        buf = it->second;
    } else {
        buf = [impl_->device newBufferWithLength:nbytes
                                         options:MTLResourceStorageModeShared];
        impl_->input_buffers[key] = buf;
        impl_->input_capacity[key] = nbytes;
    }
    if (host_src) std::memcpy([buf contents], host_src, nbytes);
    t.device_data = (__bridge void*)buf;
    t.device_offset = 0;
}

// ===========================================================================
// allocation hooks
// ===========================================================================

void* MetalBackend::alloc_output(Tensor& out, size_t nbytes, BufferPool* /*pool*/) {
    void* buf = impl_->pool->acquire(nbytes);
    if (!buf) return nullptr;
    out.device_data = buf;
    out.device_offset = 0;
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
    if (t.device_data) impl_->pending_free.push_back({t.device_data, t.nbytes()});
}

// ===========================================================================
// command buffer lifecycle
// ===========================================================================

void MetalBackend::begin_graph() {
    impl_->cmd = [impl_->queue commandBuffer];
    impl_->cmd.label = @"mollm graph";
    impl_->enc = [impl_->cmd computeCommandEncoder];
    impl_->enc.label = @"mollm compute";
    // os_signpost interval for the whole graph run — visible in Instruments'
    // "Points of Interest" track (Apple's NVTX analogue) alongside the Metal
    // System Trace GPU timeline.
    os_signpost_interval_begin(impl_->sp(), OS_SIGNPOST_ID_EXCLUSIVE, "graph");
}

// Debug: commit + wait after each op so intermediate device buffers are
// host-readable for per-node CPU/Metal diffing. Enabled by MOLLM_METAL_SYNC_EACH.
void MetalBackend::sync_point() {
    if (!getenv("MOLLM_METAL_SYNC_EACH")) return;
    if (impl_->enc) { [impl_->enc endEncoding]; impl_->enc = nil; }
    if (impl_->cmd) {
        [impl_->cmd commit];
        [impl_->cmd waitUntilCompleted];
        impl_->cmd = nil;
    }
    impl_->cmd = [impl_->queue commandBuffer];
    impl_->enc = [impl_->cmd computeCommandEncoder];
}

static const char* op_name(int op) {
    switch (op) {
    case 10:  return "MATMUL";
    case 20:  return "RMS_NORM";
    case 40:  return "ROTARY_EMBED";
    case 50:  return "SDPA";
    case 51:  return "SDPA_MLA";
    case 60:  return "RESHAPE";
    case 61:  return "PERMUTE";
    case 65:  return "CONTIGUOUS";
    case 70:  return "ADD";
    case 71:  return "MUL";
    case 30:  return "SILU";
    default:  return "OP";
    }
}

void MetalBackend::dump_profile() {
    if (!impl_->profile || impl_->op_stats.empty()) return;
    double total = 0.0;
    for (auto& kv : impl_->op_stats) total += kv.second.gpu_ms;
    fprintf(stderr, "\n=== Metal per-op GPU time (MOLLM_METAL_PROFILE) ===\n");
    fprintf(stderr, "%-14s %10s %8s %10s %6s\n", "op", "gpu_ms", "calls", "us/call", "%%");
    // Sort by total gpu_ms descending for readability.
    std::vector<std::pair<int, Impl::OpStat>> rows(impl_->op_stats.begin(), impl_->op_stats.end());
    std::sort(rows.begin(), rows.end(),
              [](auto& a, auto& b){ return a.second.gpu_ms > b.second.gpu_ms; });
    for (auto& r : rows) {
        double per_call_us = r.second.calls ? (r.second.gpu_ms * 1000.0 / r.second.calls) : 0.0;
        fprintf(stderr, "%-14s %10.3f %8llu %10.2f %6.1f\n",
                op_name(r.first), r.second.gpu_ms,
                (unsigned long long)r.second.calls, per_call_us,
                total > 0 ? 100.0 * r.second.gpu_ms / total : 0.0);
    }
    fprintf(stderr, "%-14s %10.3f\n", "TOTAL", total);
    impl_->op_stats.clear();
}

void MetalBackend::end_graph() {
    if (impl_->enc) { [impl_->enc endEncoding]; impl_->enc = nil; }
    if (impl_->cmd) {
        [impl_->cmd commit];
        [impl_->cmd waitUntilCompleted];
        if (impl_->cmd.status == MTLCommandBufferStatusError) {
            NSError* e = impl_->cmd.error;
            fprintf(stderr, "MetalBackend: command buffer error: %s\n",
                    e ? e.localizedDescription.UTF8String : "?");
        }
        if (getenv("MOLLM_METAL_GPU_TIME")) {
            double gpu_ms = (impl_->cmd.GPUEndTime - impl_->cmd.GPUStartTime) * 1000.0;
            impl_->gpu_time_ms += gpu_ms;
            impl_->gpu_graphs += 1;
            fprintf(stderr, "[metal] graph GPU time %.3f ms (cumulative %.1f ms over %llu graphs)\n",
                    gpu_ms, impl_->gpu_time_ms, (unsigned long long)impl_->gpu_graphs);
        }
        impl_->cmd = nil;
    }
    // Now that all GPU work has completed, return deferred-freed buffers to the
    // pool for reuse by the next graph run.
    for (auto& pf : impl_->pending_free) impl_->pool->release(pf.first, pf.second);
    impl_->pending_free.clear();
    os_signpost_interval_end(impl_->sp(), OS_SIGNPOST_ID_EXCLUSIVE, "graph");
}

// ===========================================================================
// dispatch
// ===========================================================================

void MetalBackend::dispatch(const GraphNode& node,
                            const std::vector<const Tensor*>& inputs,
                            Tensor* output, ThreadPool* /*thread_pool*/) {
    id<MTLComputeCommandEncoder> enc = impl_->enc;
    const OpParams& params = node.params;
    const OpType op = node.op_type;

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

    switch (op) {
    // --- view ops: metadata only, alias the input's device buffer ---
    case OpType::INPUT:
    case OpType::CONSTANT:
        break;

    case OpType::RESHAPE: {
        const Tensor& src = *inputs[0];
        if (src.is_contiguous()) {
            // zero-copy: alias device buffer + offset, keep new shape
            void* dd = src.device_data;
            size_t doff = src.device_offset;
            int64_t sh[4] = { output->shape[0], output->shape[1],
                              output->shape[2], output->shape[3] };
            *output = src;
            output->shape[0]=sh[0]; output->shape[1]=sh[1];
            output->shape[2]=sh[2]; output->shape[3]=sh[3];
            output->compute_strides();
            output->device_data = dd;
            output->device_offset = doff;
        } else {
            // materialize via contiguous kernel (output buffer already allocated)
            TensorDesc d{};
            for (int i=0;i<4;i++){ d.shape[i]=(int)src.shape[i]; d.stride[i]=estride(src,i);}            
            d.offset = eoffset(src);
            id<MTLComputePipelineState> ps = impl_->pipeline("contiguous_f32");
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf_of(&src) offset:0 atIndex:0];
            [enc setBuffer:buf_of(output) offset:0 atIndex:2];
            [enc setBytes:&d length:sizeof(d) atIndex:3];
            grid1d((int)output->nelements());
        }
        break;
    }

    case OpType::PERMUTE: {
        // zero-copy: reuse device buffer + offset, shape/stride already set by
        // the CPU permute() metadata path via *output = permuted view.
        const Tensor& src = *inputs[0];
        // Recompute permuted shape/stride from params (axis order) like CPU.
        // The executor left output shape from out_shape; but PERMUTE needs the
        // permuted strides. Mirror kernels: params.i32[0..3] = axis order.
        int a0=params.i32.size()>0?params.i32[0]:0;
        int a1=params.i32.size()>1?params.i32[1]:1;
        int a2=params.i32.size()>2?params.i32[2]:2;
        int a3=params.i32.size()>3?params.i32[3]:3;
        Tensor v = src;
        int64_t ns[4]; size_t nst[4];
        ns[a0]=src.shape[0]; nst[a0]=src.stride[0];
        ns[a1]=src.shape[1]; nst[a1]=src.stride[1];
        ns[a2]=src.shape[2]; nst[a2]=src.stride[2];
        ns[a3]=src.shape[3]; nst[a3]=src.stride[3];
        for(int i=0;i<4;i++){v.shape[i]=ns[i]; v.stride[i]=nst[i];}
        *output = v;
        output->device_data = src.device_data;
        output->device_offset = src.device_offset;
        break;
    }

    case OpType::SLICE: {
        // zero-copy: view of the parent along `dim`, preserving stride layout.
        // Mirrors the CPU SLICE (execute.cpp): device_offset advances by
        // offset*stride[dim] (bytes), shape[dim] shrinks to size.
        const Tensor& src = *inputs[0];
        int dim    = params.i32.size()>0 ? params.i32[0] : 0;
        int offset = params.i32.size()>1 ? params.i32[1] : 0;
        int size   = params.i32.size()>2 ? params.i32[2] : (int)src.shape[dim];
        *output = src;
        output->device_data = src.device_data;
        output->device_offset = src.device_offset + (size_t)offset * src.stride[dim];
        output->shape[dim] = size;
        break;
    }

    case OpType::CONTIGUOUS: {
        const Tensor& src = *inputs[0];
        // Fast path: if src is already dense row-major, CONTIGUOUS is a no-op —
        // alias the device buffer (zero-copy) and skip the kernel dispatch. This
        // is common in DECODE (seq=1) where the attention transposes collapse to
        // identity, saving ~144 dispatches/step.
        if (src.is_contiguous()) {
            *output = src;
            output->device_data = src.device_data;
            output->device_offset = src.device_offset;
            break;
        }
        TensorDesc d{};
        for (int i=0;i<4;i++){ d.shape[i]=(int)src.shape[i]; d.stride[i]=estride(src,i);}        
        d.offset = eoffset(src);
        [enc setBuffer:buf_of(&src) offset:0 atIndex:0];
        [enc setBuffer:buf_of(output) offset:0 atIndex:2];
        [enc setBytes:&d length:sizeof(d) atIndex:3];
        // 3D fast path (no per-element div/mod) when the tensor collapses to
        // <=3 dims (shape[3]==1, the common attention transpose case).
        if (d.shape[3] == 1) {
            // NOTE: this M5 Pro GPU returns WRONG partial results with
            // dispatchThreads: (non-uniform threadgroups); use dispatchThreadgroups:
            // with a rounded-up grid + in-kernel bounds check (see M1 notes).
            id<MTLComputePipelineState> ps = impl_->pipeline("contiguous3d_f32");
            [enc setComputePipelineState:ps];
            const NSUInteger tx = 64, ty = 4;
            MTLSize tgs = MTLSizeMake(tx, ty, 1);
            MTLSize tgc = MTLSizeMake(((NSUInteger)d.shape[0] + tx - 1)/tx,
                                      ((NSUInteger)d.shape[1] + ty - 1)/ty,
                                      (NSUInteger)d.shape[2]);
            [enc dispatchThreadgroups:tgc threadsPerThreadgroup:tgs];
        } else {
            id<MTLComputePipelineState> ps = impl_->pipeline("contiguous_f32");
            [enc setComputePipelineState:ps];
            grid1d((int)output->nelements());
        }
        break;
    }

    case OpType::MATMUL: {
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
        // fused activation: params.i32[0]=activation (0 NONE, 3 SILU per Activation enum)
        int act = params.i32.size()>0 ? params.i32[0] : 0;
        p.activation = (act == 3) ? 1 : 0;   // map Activation::SILU -> shader 1

        if (p.M == 1) {
            // GEMV v2 (llama mul_mv style): each threadgroup owns NR0=2 output
            // rows; NSG simdgroups split K and reduce via shmem. NSG scales with
            // K (like llama: min(4,(K+127)/128)) so large-K matmuls (down_proj
            // K=9728) get up to 4x32 lanes. MOLLM_METAL_GEMV_OLD=1 -> old kernel.
            static const bool gemv_old = (getenv("MOLLM_METAL_GEMV_OLD") != nullptr);
            // NR0 = output rows per threadgroup (activation reuse). Default 2;
            // tune via MOLLM_METAL_GEMV_NR0 (1..8).
            static const int NR0 = []{
                const char* e = getenv("MOLLM_METAL_GEMV_NR0");
                int v = e ? atoi(e) : 2;
                if (v < 1) v = 1; if (v > 8) v = 8;
                return v;
            }();
            // Bind weight at BYTE offset (64-bit) to avoid uint32 element-offset
            // overflow for late weights in the 8.8GB region (esp. lm_head).
            MatmulParams pv = p; pv.b_offset = 0;
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBytes:&pv length:sizeof(pv) atIndex:3];
            id<MTLComputePipelineState> gemv2_ps = gemv_old ? nil : impl_->pipeline_gemv2(NR0);
            if (gemv2_ps) {
                [enc setComputePipelineState:gemv2_ps];
                // NSG cap: number of simdgroups splitting K. 8 is the M5 sweet
                // spot (large-K down_proj K=9728 / o_proj K=4096 get more lanes;
                // plateaus past 8, tapers at 16). Tunable via env.
                static const int NSG_CAP = []{
                    const char* e = getenv("MOLLM_METAL_GEMV_NSG_CAP");
                    int v = e ? atoi(e) : 8; if (v < 1) v = 1; if (v > 32) v = 32; return v;
                }();
                int nsg = std::min(NSG_CAP, (p.K + 127) / 128);
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
        } else if (p.activation == 0) {
            // Tiled GEMM (simdgroup 8x8 MMA; activation=NONE — Qwen3 emits SILU
            // separately). Weight buffer B bound at its 64-bit BYTE offset with
            // in-shader b_offset=0 to avoid uint32 element-offset overflow for
            // the 8.8GB weight region (incl. lm_head).
            // Default: 32x32 half-staged tile (acc[4]/sg). This is the OCCUPANCY
            // sweet spot on M5 Pro: larger tiles / more accumulators per
            // simdgroup (tested 64x32 -> 219 t/s, 32x64 acc[8] -> 226 t/s, TK=32
            // -> 357) all LOWER perf by reducing resident threadgroup count.
            // Apple GPUs favor many small threadgroups over deep register
            // blocking (opposite of NVIDIA). Those variants kept behind
            // MOLLM_METAL_GEMM64 / MOLLM_METAL_GEMMN64 for experimentation.
            // Weight bound at 64-bit byte offset, b_offset=0.
            MatmulParams pt = p; pt.b_offset = 0;
#ifdef MOLLM_METAL_TENSOR
            if (impl_->has_tensor) {
                // Metal 4 tensor-API GEMM (fast path on M5/A19+). Mirrors llama:
                // weights (our N) staged NRA=64 tile; activations (our M) device
                // NRB=128 tile; NK=32. A staged as half → NRA*NK=2048 halves=4KB.
                // grid: tgpig.y = N/64, tgpig.x = M/128.
                id<MTLComputePipelineState> ps = impl_->pipeline("gemm_tensor_f32a_f16b_f32c");
                // Bind A (activations) and B (weights) at their 64-bit byte
                // offsets; zero the in-shader element offsets accordingly.
                MatmulParams ptt = pt; ptt.a_offset = 0; ptt.b_offset = 0;
                [enc setComputePipelineState:ps];
                [enc setBuffer:buf_of(&A) offset:A.device_offset atIndex:0];
                [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
                [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                [enc setBytes:&ptt length:sizeof(ptt) atIndex:3];
                [enc setThreadgroupMemoryLength:64*32*sizeof(uint16_t) atIndex:0];
                MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + 127)/128,
                                          ((NSUInteger)p.N + 63)/64, 1);
                [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            } else
#endif
            if (getenv("MOLLM_METAL_GEMMN64")) {
                id<MTLComputePipelineState> ps = impl_->pipeline("gemm_tiledN64_f32a_f16b_f32c");
                [enc setComputePipelineState:ps];
                [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
                [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
                [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                [enc setBytes:&pt length:sizeof(pt) atIndex:3];
                [enc setThreadgroupMemoryLength:(32*8 + 64*8)*2 atIndex:0];
                MTLSize tgc = MTLSizeMake(((NSUInteger)p.N + 63)/64,
                                          ((NSUInteger)p.M + 31)/32, 1);
                [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            } else if (getenv("MOLLM_METAL_GEMM64")) {
                id<MTLComputePipelineState> ps = impl_->pipeline("gemm_tiled64_f32a_f16b_f32c");
                [enc setComputePipelineState:ps];
                [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
                [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
                [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                [enc setBytes:&pt length:sizeof(pt) atIndex:3];
                [enc setThreadgroupMemoryLength:768*sizeof(float) atIndex:0];  // staging
                [enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:1];  // store scratch
                MTLSize tgc = MTLSizeMake(((NSUInteger)p.N + 31)/32,
                                          ((NSUInteger)p.M + 63)/64, 1);
                [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            } else {
                id<MTLComputePipelineState> ps = impl_->pipeline("gemm_tiled_f32a_f16b_f32c");
                [enc setComputePipelineState:ps];
                [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
                [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
                [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                [enc setBytes:&pt length:sizeof(pt) atIndex:3];
                // Half-staged tiles, TK=8: (32*8 + 32*8) halves = 1KB; the FP32
                // edge store scratch (256 floats = 1KB) reuses the region.
                [enc setThreadgroupMemoryLength:1024 atIndex:0];
                MTLSize tgc = MTLSizeMake(((NSUInteger)p.N + 31)/32,
                                          ((NSUInteger)p.M + 31)/32, 1);
                [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            }
        } else {
            // Fused-activation GEMM (rare in dense graph): vectorized scalar path.
            id<MTLComputePipelineState> ps = impl_->pipeline("gemm_f32a_f16b_f32c");
            MatmulParams pg = p; pg.b_offset = 0;  // bind B at byte offset (no uint32 overflow)
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBytes:&pg length:sizeof(pg) atIndex:3];
            grid1d(p.M * p.N);
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
        p.w_offset = eoffset(W);
        p.out_offset = eoffset(O);
        p.x_row_stride = estride(X, 1);
        p.out_row_stride = estride(O, 1);
        p.eps = params.f32.size()>0 ? params.f32[0] : 1e-6f;
        id<MTLComputePipelineState> ps = impl_->pipeline("rms_norm_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buf_of(&W) offset:0 atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        NSUInteger tg = 256;
        if (tg > ps.maxTotalThreadsPerThreadgroup) tg = ps.maxTotalThreadsPerThreadgroup;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.rows,1,1)
            threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
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
        p.x_offset = eoffset(X);
        p.cos_offset = eoffset(COS);
        p.sin_offset = eoffset(SIN);
        p.x_stride_pos = estride(in, 1);
        p.x_stride_head = estride(in, 2);
        // Copy input -> output buffer (rope in place), if different buffers.
        if (buf_of(&in) != buf_of(&X) || in.device_offset != X.device_offset) {
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
    case OpType::MUL: {
        const Tensor& A = *inputs[0];
        const Tensor& B = *inputs[1];
        Tensor& O = *output;
        EwiseParams p{};
        p.n = (int)O.nelements();
        p.broadcast_b = (B.nelements()==1) ? 1 : 0;
        p.a_offset = eoffset(A);
        p.b_offset = eoffset(B);
        p.out_offset = eoffset(O);
        id<MTLComputePipelineState> ps =
            impl_->pipeline(op==OpType::ADD ? "add_f32" : "mul_f32");
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
        int dst_seqlen = past + cur_seqlen;

        // Cache data begins 64 bytes past the buffer base (CacheMetadata header).
        // FP16 cache: element offset = (device_offset + 64) / 2.
        const size_t CACHE_HDR = 64;
        uint k_cache_eoff = (uint)((K_cache ? K_cache->device_offset : 0) + CACHE_HDR) / 2;
        uint v_cache_eoff = (uint)((V_cache ? V_cache->device_offset : 0) + CACHE_HDR) / 2;

        // 1) Append K_cur/V_cur (FP32) into the FP16 cache at position past+s.
        if (kv_cache == 2 && K_cache && V_cache) {
            id<MTLComputePipelineState> aps = impl_->pipeline("sdpa_append_f32_to_f16");
            auto append = [&](const Tensor& cur, const Tensor& cache, int width, uint cache_eoff) {
                SdpaAppendParams ap{};
                ap.num_kv_heads = num_kv;
                ap.cur_seqlen = cur_seqlen;
                ap.past_seqlen = past;
                ap.head_dim = width;
                ap.max_seq_len = max_seq;
                ap.cur_offset = eoffset(cur);
                ap.cur_stride_head = estride(cur, 2);
                ap.cur_stride_pos = estride(cur, 1);
                ap.cache_offset = cache_eoff;
                [enc setComputePipelineState:aps];
                [enc setBuffer:buf_of(&cur) offset:0 atIndex:0];
                [enc setBuffer:buf_of(&cache) offset:0 atIndex:2];
                [enc setBytes:&ap length:sizeof(ap) atIndex:3];
                NSUInteger tx=8,ty=8,tz=4;
                MTLSize tgs=MTLSizeMake(tx,ty,tz);
                MTLSize tgc=MTLSizeMake(((NSUInteger)width+tx-1)/tx,
                                        ((NSUInteger)cur_seqlen+ty-1)/ty,
                                        ((NSUInteger)num_kv+tz-1)/tz);
                [enc dispatchThreadgroups:tgc threadsPerThreadgroup:tgs];
            };
            append(K_cur, *K_cache, head_dim, k_cache_eoff);
            append(V_cur, *V_cache, v_head_dim, v_cache_eoff);
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
        //   default -> sdpa_prefill_fa2_f32 (llama-style FA: query-split, direct-
        //              global K/V MMA, threadgroup-O elementwise rescale).
        //   MOLLM_METAL_SDPA_SIMPLE=1 -> per-query online-softmax sdpa_prefill_f32.
        // fa2 requires dk/dv %8, <=256 (C=64 & PV8 divisible by NSG=4 always hold).
        static const bool use_simple = (getenv("MOLLM_METAL_SDPA_SIMPLE") != nullptr);
        const int FA2_NSG = 4, FA2_Q = 8;
        // fa2 declares function constants without defaults, so it can ONLY be built
        // via the specialized (pipeline_fa2) path — never the plain name-keyed
        // pipeline. Probe the specialized pipeline directly as the guard.
        id<MTLComputePipelineState> fa2_ps = nil;
        bool fa2_pre = !decode_path && !use_simple
            && (head_dim % 8 == 0) && (v_head_dim % 8 == 0)
            && head_dim <= 256 && v_head_dim <= 256;
        if (fa2_pre) fa2_ps = impl_->pipeline_fa2(head_dim, v_head_dim);
        bool fa2_ok = (fa2_ps != nil);

        const char* kname = decode_path ? "sdpa_decode_f32" : "sdpa_prefill_f32";
        // fa2 uses its dk/dv-specialized pipeline; all other paths use name-keyed.
        id<MTLComputePipelineState> ps = fa2_ok ? fa2_ps : impl_->pipeline(kname);
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_of(&Q) offset:0 atIndex:0];
        [enc setBuffer:(K_cache?buf_of(K_cache):buf_of(&K_cur)) offset:0 atIndex:1];
        [enc setBuffer:(V_cache?buf_of(V_cache):buf_of(&V_cur)) offset:0 atIndex:2];
        [enc setBuffer:buf_of(&out) offset:0 atIndex:4];
        // Buffer index 5 (mask) must always be bound; use Q as a dummy if no mask.
        [enc setBuffer:(mask?buf_of(mask):buf_of(&Q)) offset:0 atIndex:5];
        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
        if (decode_path) {
            // One threadgroup (256 threads) per head; threads split the keys.
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_heads,1,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        } else if (fa2_ok) {
            // llama-style FA: one threadgroup (NSG simdgroups, 32*NSG threads) per
            // (query-tile=Q, head). Threadgroup memory (bytes):
            //   sq[Q*DK] half(2B) + so[Q*PV] float(4B) + ss[Q*SH] float(4B),
            //   SH=2*C, PV=PAD(DV,64).
            const int PV = ((v_head_dim + 63) / 64) * 64;
            const int SH = 2 * 64;
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

    default:
        fprintf(stderr, "MetalBackend: unsupported op %d for phase-1 dense path\n",
                (int)op);
        assert(false && "unsupported metal op");
        break;
    }
    // Per-op flush: debug diffing (MOLLM_METAL_SYNC_EACH) and/or per-op GPU
    // timing (MOLLM_METAL_PROFILE). Both need each op in its own command buffer.
    if (impl_->profile) {
        if (impl_->enc) { [impl_->enc endEncoding]; impl_->enc = nil; }
        if (impl_->cmd) {
            [impl_->cmd commit];
            [impl_->cmd waitUntilCompleted];
            double gpu_ms = (impl_->cmd.GPUEndTime - impl_->cmd.GPUStartTime) * 1000.0;
            auto& st = impl_->op_stats[(int)op];
            st.gpu_ms += gpu_ms;
            st.calls  += 1;
            impl_->cmd = nil;
        }
        impl_->cmd = [impl_->queue commandBuffer];
        impl_->enc = [impl_->cmd computeCommandEncoder];
    } else {
        sync_point();  // no-op unless MOLLM_METAL_SYNC_EACH (per-op debug flush)
    }
}
