#include "engine/metal_backend.h"
#include "graph/metal_pool.h"
#include "graph/graph.h"
#include "kernels/moe.h"
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

// CPU-side INT4 BG128 packed block (mirror of the one in kernels/matmul.cpp),
// used to decode int4 weights into a Metal-friendly raw layout at load time.
struct alignas(16) Q4B8G128Block {
    float   scales[8];
    uint8_t q[4][8][16];
};

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
    bool                     copy_weights = false;

    // persistent device buffers owned by the backend (KV cache)
    std::vector<id<MTLBuffer>> persistent;
    // Dense weight copies used only during Metal prefill in SSD hybrid mode.
    // Kept separate so they can be dropped before CPU expert decode on UMA.
    std::vector<id<MTLBuffer>> weight_copies;
    std::unordered_map<const void*, id<MTLBuffer>> copied_weights;
    std::unordered_map<const void*, id<MTLBuffer>> decoded_q4_weights;

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
    // MATMUL is split by concrete kernel path so decode profiles distinguish
    // quantized/FP16 GEMV from prefill tensor GEMM.
    std::map<std::string, OpStat> op_stats;
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
    // the flash-attention prefill kernel to bake dk/dv in at compile time.
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

    id<MTLComputePipelineState> pipeline_gemv_w4(int nr0) {
        char keyc[48];
        snprintf(keyc, sizeof(keyc), "gemv_w4:nr0%d", nr0);
        std::string key(keyc);
        auto it = spec_pipelines.find(key);
        if (it != spec_pipelines.end()) return it->second;

        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&nr0 type:MTLDataTypeInt atIndex:6];
        NSError* err = nil;
        id<MTLFunction> fn =
            [library newFunctionWithName:@"gemv_w4_f32a_i4b_f32c"
                          constantValues:cv error:&err];
        id<MTLComputePipelineState> ps = fn
            ? [device newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!ps)
            fprintf(stderr,
                    "MetalBackend: W4 GEMV nr0=%d pipeline failed: %s\n",
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

int gemv_nsg_cap() {
    static const int cap = [] {
        const char* value = std::getenv("MOLLM_METAL_GEMV_NSG");
        if (!value) return 4;
        const int parsed = std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8)
                   ? parsed
                   : 4;
    }();
    return cap;
}

int gemv_w4_nr0(int n, int k) {
    const char* value = std::getenv("MOLLM_METAL_GEMV_W4_NR");
    if (value) {
        const int parsed = std::atoi(value);
        if (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8)
            return parsed;
    }
    // Very wide projection matrices have enough row parallelism to amortize
    // eight rows per activation load; narrower projections retain NR4 to keep
    // register pressure and tail work down.
    if (n >= 12000 && k >= 1024) return 8;
    return 4;
}

int gemv_w4_nsg_cap() {
    static const int cap = [] {
        const char* value = std::getenv("MOLLM_METAL_GEMV_W4_NSG");
        if (!value) return 2;
        const int parsed = std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8)
                   ? parsed
                   : 2;
    }();
    return cap;
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
        // Metal4), and the pipeline actually loads.
#ifdef MOLLM_METAL_TENSOR
        bool fam = false;
        if (@available(macOS 15.0, *)) {
            fam = [impl_->device supportsFamily:MTLGPUFamilyMetal4];
        }
        // Metal 4 tensor-API GEMM is correct (parity-tested) and ~2.3x faster
        // than the simdgroup path (prefill 940 vs 403 t/s). Enable it whenever
        // the device and compiled pipeline support it.
        if (fam && impl_->pipeline("gemm_tensor_f32a_f16b_f32c") != nil) {
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
        impl_->pipelines.clear();
        impl_->spec_pipelines.clear();
        impl_->copied_weights.clear();
        impl_->decoded_q4_weights.clear();
        impl_->weight_copies.clear();
        impl_->persistent.clear();
        impl_->weight_buffer = nil;
        impl_->pool.reset();
    }
}

bool MetalBackend::available() const { return impl_ && impl_->ok; }

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
        const int NR0 = 2, NSG = std::min(gemv_nsg_cap(), (K + 127) / 128);
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

bool MetalBackend::has_tensor_path() const {
    return impl_->has_tensor;
}

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

void MetalBackend::enable_weight_copy_mode() {
    impl_->copy_weights = true;
}

void MetalBackend::release_weight_copies() {
    if (!impl_->copy_weights) return;
    synchronize_for_host_read();
    impl_->copied_weights.clear();
    impl_->decoded_q4_weights.clear();
    impl_->weight_copies.clear();
}

bool MetalBackend::has_weight_copies() const {
    return !impl_->weight_copies.empty();
}

void MetalBackend::wrap_weight(Tensor& t) {
    if (!t.data) return;
    if (!impl_->weight_buffer) {
        if (!impl_->copy_weights) return;
        // INT4 g128 is decoded after quant metadata is configured. INT8 needs
        // its scale storage co-located and is not yet supported by the hybrid
        // path. FP16/FP32 constants can be copied immediately.
        if (t.prec == Precision::FP16 || t.prec == Precision::FP32) {
            void* src = t.data;
            size_t bytes = t.nbytes();
            auto found = impl_->copied_weights.find(src);
            if (found != impl_->copied_weights.end()) {
                t.device_data = (__bridge void*)found->second;
                t.device_offset = 0;
            } else {
                @autoreleasepool {
                    id<MTLBuffer> b =
                        [impl_->device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
                    std::memcpy([b contents], src, bytes);
                    impl_->weight_copies.push_back(b);
                    impl_->copied_weights[src] = b;
                    t.device_data = (__bridge void*)b;
                    t.device_offset = 0;
                }
            }
        }
        return;
    }
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

void MetalBackend::wrap_weight_int4_g128(Tensor& t, bool keep_native_bg128) {
    // INT4 weights ship in the CPU-only Q4B8G128Block layout (scales+nibbles
    // interleaved per 128-K block). The Metal W4A8 kernels want a simple raw
    // [N,K/2] nibble array + [N,gpr] fp32 scales, so decode once at load time
    // into a dedicated device buffer: [ nibbles (N*K/2) | scales (N*gpr f32) ].
    // device_offset stays 0 (nibbles at start); scales live at byte N*(K/2),
    // which the W4 dispatch binds directly (co-located, no weight_base math).
    if (!(t.prec == Precision::INT4 && t.is_q4_g128_packed && t.q4_g128_data)) return;
    // Ordinary linear weights are [N,K,1,1]. Fused MoE expert weights retain
    // their logical 3-D shape [E,N_per_expert,K], but the packed storage is the
    // same flat sequence of rows. Flatten every dimension before the final K
    // dimension so both layouts decode identically.
    int last = 3;
    while (last > 1 && t.shape[last] == 1) --last;
    const int K = (int)t.shape[last];
    int64_t rows64 = 1;
    for (int d = 0; d < last; ++d) rows64 *= t.shape[d];
    const int N = (int)rows64;
    // Expert tensors dominate MoE package size. Keep their native BG128 blocks
    // zero-copy; the selected-expert tensor kernel decodes blocks while staging.
    // Materializing a second raw-W4 copy here adds ~9GB and causes UMA paging.
    // Aggregate packages serialize experts flattened as [E*N,K], so the
    // loader passes keep_native_bg128 based on the explicit weight role.
    if (keep_native_bg128 || last >= 2) {
        wrap_weight(t);
        return;
    }
    const int gpr = (int)t.groups_per_row;         // K/128
    const size_t nib_bytes = (size_t)N * (K / 2);
    const size_t sc_bytes  = (size_t)N * gpr * sizeof(float);
    auto cached = impl_->decoded_q4_weights.find(t.q4_g128_data);
    if (cached != impl_->decoded_q4_weights.end()) {
        t.device_data = (__bridge void*)cached->second;
        t.device_offset = 0;
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> b = [impl_->device newBufferWithLength:nib_bytes + sc_bytes
                                                     options:MTLResourceStorageModeShared];
        if (impl_->copy_weights)
            impl_->weight_copies.push_back(b);
        else
            impl_->persistent.push_back(b);
        uint8_t* nib = (uint8_t*)[b contents];
        float*   sc  = (float*)(nib + nib_bytes);
        const auto* blocks = reinterpret_cast<const Q4B8G128Block*>(t.q4_g128_data);
        // Block (n/8, g) holds 8 channels x 4 K-sub-blocks(32K) x 16 bytes.
        // channel c (n=n_tile+c), k-sub qgi, byte b -> raw k = g*128+qgi*32+2b
        // (low nibble) and +1 (high). Copy the 16 raw bytes straight through
        // (same nibble order as raw [N,K/2]); copy per-channel per-group scale.
        for (int n = 0; n < N; n++) {
            int nt = (n / 8), c = n % 8;
            uint8_t* nrow = nib + (size_t)n * (K / 2);
            for (int g = 0; g < gpr; g++) {
                const Q4B8G128Block& blk = blocks[(size_t)nt * gpr + g];
                sc[(size_t)n * gpr + g] = blk.scales[c];
                for (int qgi = 0; qgi < 4; qgi++)
                    std::memcpy(nrow + (size_t)(g * 128 + qgi * 32) / 2,
                                blk.q[qgi][c], 16);
            }
        }
        t.device_data = (__bridge void*)b;
        t.device_offset = 0;
        impl_->decoded_q4_weights[t.q4_g128_data] = b;
        // Keep t.scales pointing at the package's CPU layout. Metal binds the
        // co-located decoded scales by byte offset, while hybrid/CPU kernels
        // still need the original BG128 tensor metadata.
    }
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
    if (!t.device_data) return;
    if (impl_->cmd) impl_->pending_free.push_back({t.device_data, t.nbytes()});
    else impl_->pool->release(t.device_data, t.nbytes());
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

void MetalBackend::synchronize_for_host_read() {
    if (impl_->enc) {
        [impl_->enc endEncoding];
        impl_->enc = nil;
    }
    if (impl_->cmd) {
        [impl_->cmd commit];
        [impl_->cmd waitUntilCompleted];
        if (impl_->cmd.status == MTLCommandBufferStatusError) {
            NSError* e = impl_->cmd.error;
            fprintf(stderr, "MetalBackend: host-read sync failed: %s\n",
                    e ? e.localizedDescription.UTF8String : "?");
        }
        impl_->cmd = nil;
    }
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

void MetalBackend::dump_profile() {
    if (!impl_->profile || impl_->op_stats.empty()) return;
    double total = 0.0;
    for (auto& kv : impl_->op_stats) total += kv.second.gpu_ms;
    fprintf(stderr, "\n=== Metal per-op GPU time (MOLLM_METAL_PROFILE) ===\n");
    fprintf(stderr, "%-32s %10s %8s %10s %6s\n",
            "op", "gpu_ms", "calls", "us/call", "%%");
    // Sort by total gpu_ms descending for readability.
    std::vector<std::pair<std::string, Impl::OpStat>> rows(
        impl_->op_stats.begin(), impl_->op_stats.end());
    std::sort(rows.begin(), rows.end(),
              [](auto& a, auto& b){ return a.second.gpu_ms > b.second.gpu_ms; });
    for (auto& r : rows) {
        double per_call_us = r.second.calls ? (r.second.gpu_ms * 1000.0 / r.second.calls) : 0.0;
        fprintf(stderr, "%-32s %10.3f %8llu %10.2f %6.1f\n",
                r.first.c_str(), r.second.gpu_ms,
                (unsigned long long)r.second.calls, per_call_us,
                total > 0 ? 100.0 * r.second.gpu_ms / total : 0.0);
    }
    fprintf(stderr, "%-32s %10.3f\n", "TOTAL", total);
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
                            Tensor* output, ThreadPool* thread_pool) {
    id<MTLComputeCommandEncoder> enc = impl_->enc;
    const OpParams& params = node.params;
    const OpType op = node.op_type;
    std::string profile_label = op_type_name(op);

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
        // Always materialize via kernel — don't add a zero-copy alias here: the
        // executor doesn't treat CONTIGUOUS as view-like, so an alias dangles once
        // src is deferred-freed, corrupting results under whole-graph execution.
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

        if (p.M == 1 && B.prec == Precision::INT8) {
            profile_label = "MATMUL_W8_GEMV";
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
            size_t scales_boff = (char*)B.scales - (char*)impl_->weight_base;
            const int NR0 = 2;
            const int NSG = std::min(gemv_nsg_cap(), (p.K + 127) / 128);
            id<MTLComputePipelineState> ps = impl_->pipeline("gemv_w8_f32a_i8b_f32c");
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBuffer:impl_->weight_buffer offset:scales_boff atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            [enc setThreadgroupMemoryLength:(NSUInteger)(NR0*32*sizeof(float)) atIndex:0];
            NSUInteger tgc = ((NSUInteger)p.N + NR0 - 1)/NR0;
            [enc dispatchThreadgroups:MTLSizeMake(tgc,1,1)
                threadsPerThreadgroup:MTLSizeMake(32,(NSUInteger)NSG,1)];
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
            const int NR0 = gemv_w4_nr0(p.N, p.K);
            const int NSG =
                std::min(gemv_w4_nsg_cap(), (p.K / 2 + 63) / 64);
            id<MTLComputePipelineState> ps = impl_->pipeline_gemv_w4(NR0);
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBuffer:buf_of(&B) offset:scales_boff atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            [enc setThreadgroupMemoryLength:(NSUInteger)(NR0*32*sizeof(float)) atIndex:0];
            NSUInteger tgc = ((NSUInteger)p.N + NR0 - 1)/NR0;
            [enc dispatchThreadgroups:MTLSizeMake(tgc,1,1)
                threadsPerThreadgroup:MTLSizeMake(32,(NSUInteger)std::max(1,NSG),1)];
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
            [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBytes:&pv length:sizeof(pv) atIndex:3];
            id<MTLComputePipelineState> gemv2_ps = gemv_old ? nil : impl_->pipeline_gemv2(NR0);
            if (gemv2_ps) {
                [enc setComputePipelineState:gemv2_ps];
                // Four SIMD groups give the best cross-model decode throughput
                // on M5 Pro: enough K parallelism without the occupancy and
                // reduction overhead of eight groups. The environment override
                // keeps this tunable for future GPU families.
                int nsg =
                    std::min(gemv_nsg_cap(), (p.K + 127) / 128);
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
            //  - W8A16 (default): dequant int8 weight->half during staging, reuse
            //    the fp16 tensor MMA. Requires the tensor path.
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
                impl_->pending_free.push_back({a_i8_h, a_i8_bytes});
                impl_->pending_free.push_back({sa_h,   sa_bytes});

                // 1) per-token activation quantization.
                {
                    QuantActParams q{};
                    q.M = p.M; q.K = p.K;
                    q.a_offset = A.device_offset / sizeof(float);   // A bound at 0
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
                    size_t scales_boff = (char*)B.scales - (char*)impl_->weight_base;
                    id<MTLComputePipelineState> ps = impl_->pipeline("gemm_w8a8_i8a_i8b_f32c");
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:a_i8 offset:0 atIndex:0];
                    [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
                    [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                    [enc setBytes:&w length:sizeof(w) atIndex:3];
                    [enc setBuffer:sa offset:0 atIndex:4];
                    [enc setBuffer:impl_->weight_buffer offset:scales_boff atIndex:5];
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
                size_t scales_boff = (char*)B.scales - (char*)impl_->weight_base;
                id<MTLComputePipelineState> ps = impl_->pipeline("gemm_tensor_w8_f32a_i8b_f32c");
                [enc setComputePipelineState:ps];
                [enc setBuffer:buf_of(&A) offset:A.device_offset atIndex:0];
                [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
                [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                [enc setBuffer:impl_->weight_buffer offset:scales_boff atIndex:4];
                [enc setBytes:&w length:sizeof(w) atIndex:3];
                [enc setThreadgroupMemoryLength:64*32*sizeof(uint16_t) atIndex:0];
                MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + 127)/128,
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
                // W4A16 is both faster and more accurate on the Metal 4 tensor
                // path: Qwen3.5-4B pp256 improved from ~404 to ~910 t/s and
                // Youtu-2B from ~980 to ~3055 t/s. It also halves the absolute
                // CE delta versus CPU on Qwen3.6-35B-A3B. Keep W4A8 as an
                // explicit diagnostic fallback for older tuning comparisons.
                static const bool w4a16 =
                    std::getenv("MOLLM_METAL_W4A8") == nullptr;
                if (w4a16) {
                    profile_label = "MATMUL_W4A16_GEMM";
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
                    id<MTLComputePipelineState> ps =
                        impl_->pipeline("gemm_tensor_w4_f32a_i4b_f32c");
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:buf_of(&A) offset:A.device_offset atIndex:0];
                    [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
                    [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                    [enc setBytes:&w length:sizeof(w) atIndex:3];
                    [enc setBuffer:buf_of(&B) offset:scales_boff atIndex:4];
                    [enc setThreadgroupMemoryLength:64*32*sizeof(uint16_t)
                                            atIndex:0];
                    MTLSize tgc =
                        MTLSizeMake(((NSUInteger)p.M + 127)/128,
                                    ((NSUInteger)p.N + 63)/64, 1);
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
                profile_label = "MATMUL_W4A8_GEMM";
                size_t sa_bytes   = (size_t)p.M * sizeof(float);
                void* a_i8_h = impl_->pool->acquire(a_i8_bytes);
                void* sa_h   = impl_->pool->acquire(sa_bytes);
                id<MTLBuffer> a_i8 = (__bridge id<MTLBuffer>)a_i8_h;
                id<MTLBuffer> sa   = (__bridge id<MTLBuffer>)sa_h;
                impl_->pending_free.push_back({a_i8_h, a_i8_bytes});
                impl_->pending_free.push_back({sa_h,   sa_bytes});

                // 1) per-token activation quantization -> int8 [M,K] + scale_a[M].
                {
                    QuantActParams q{};
                    q.M = p.M; q.K = p.K;
                    q.a_offset = A.device_offset / sizeof(float);
                    q.a_row_stride = p.a_row_stride;
                    id<MTLComputePipelineState> qps = impl_->pipeline("quantize_act_i8");
                    [enc setComputePipelineState:qps];
                    [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
                    [enc setBuffer:a_i8 offset:0 atIndex:2];
                    [enc setBuffer:sa   offset:0 atIndex:4];
                    [enc setBytes:&q length:sizeof(q) atIndex:3];
                    const NSUInteger nsg = 8;
                    [enc setThreadgroupMemoryLength:nsg*sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.M,1,1)
                        threadsPerThreadgroup:MTLSizeMake(32,nsg,1)];
                }
                // 2) int8 x per-group int4 GEMM with per-group dequant.
                {
                    MatmulW4A8Params w{};
                    w.M = p.M; w.N = p.N; w.K = p.K;
                    w.c_offset = eoffset(C);
                    w.c_row_stride = p.c_row_stride;
                    w.activation = p.activation;
                    w.act_n_begin = p.act_n_begin; w.act_n_len = p.act_n_len;
                    w.group_size = (int)B.group_size;
                    w.groups_per_row = (int)B.groups_per_row;
                    // Decoded W4 buffer: [ nibbles (N*K/2) | scales (N*gpr f32) ].
                    size_t scales_boff = (size_t)p.N * (p.K / 2);
                    id<MTLComputePipelineState> ps = impl_->pipeline("gemm_w4a8_i8a_i4b_f32c");
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:a_i8 offset:0 atIndex:0];
                    [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
                    [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                    [enc setBytes:&w length:sizeof(w) atIndex:3];
                    [enc setBuffer:sa offset:0 atIndex:4];
                    [enc setBuffer:buf_of(&B) offset:scales_boff atIndex:5];
                    // facc (64*64 f32) + scratch (64*64 i32) = 32KB threadgroup.
                    [enc setThreadgroupMemoryLength:2*64*64*sizeof(int32_t) atIndex:0];
                    MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + 63)/64,
                                              ((NSUInteger)p.N + 63)/64, 1);
                    [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                }
            } else
#endif
            {
                fprintf(stderr, "MetalBackend: W4 GEMM requires tensor path (M5/A19+)\n");
                assert(false && "W4 GEMM needs tensor path");
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
            // blocking (opposite of NVIDIA).
            // Weight bound at 64-bit byte offset, b_offset=0.
            MatmulParams pt = p; pt.b_offset = 0;
#ifdef MOLLM_METAL_TENSOR
            if (impl_->has_tensor) {
                profile_label = "MATMUL_FP16_TENSOR";
                // Metal 4 tensor-API GEMM (fast path on M5/A19+): weights staged in
                // a 64-row tile, activations as a device tensor, NK=32.
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
            {
                profile_label = "MATMUL_FP16_TILED";
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
            profile_label = "MATMUL_FP16_FUSED";
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
        [enc setBuffer:buf_of(&W) offset:W.device_offset atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        NSUInteger tg = 256;
        if (tg > ps.maxTotalThreadsPerThreadgroup) tg = ps.maxTotalThreadsPerThreadgroup;
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
        [enc setBuffer:buf_of(&W) offset:W.device_offset atIndex:1];
        [enc setBuffer:buf_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:4];
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
    case OpType::MUL: {
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
        id<MTLComputePipelineState> ps = impl_->pipeline("rwkv7_f32");
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
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)p.head_size, 1, 1)];
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
        p.use_qk_l2norm = params.i32.size()>4 ? params.i32[4] : 1;
        p.n_real      = params.i32.size()>6 ? params.i32[6] : 0;
        p.num_v_heads = (params.i32.size()>7 && params.i32[7]>0) ? params.i32[7] : p.num_heads;
        p.rms_eps     = params.f32.size()>0 ? params.f32[0] : 1e-6f;
        p.l2_eps      = params.f32.size()>1 ? params.f32[1] : 1e-6f;
        p.scale       = params.f32.size()>2 ? params.f32[2] : 0.f;
        if (p.scale == 0.f) p.scale = 1.f / std::sqrt((float)p.k_dim);
        p.qkv_offset = eoffset(QKV); p.a_offset = eoffset(Aa); p.b_offset = eoffset(Bb);
        p.z_offset = eoffset(Zz); p.Alog_offset = eoffset(ALG); p.dtb_offset = eoffset(DTB);
        p.norm_offset = eoffset(NRM); p.state_offset = eoffset(ST); p.out_offset = eoffset(O);
        const char* gk = (op == OpType::GATED_DELTANET_PREFILL) ? "gdn_prefill_f32" : "gdn_decode_f32";
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
        // threadgroup: sq[K] + sk[K] + red[V] floats.
        NSUInteger smem = (NSUInteger)(2*p.k_dim + p.v_dim) * sizeof(float);
        [enc setThreadgroupMemoryLength:smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.num_v_heads,1,1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)p.v_dim,1,1)];
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
            // Attention immediately reads the cache regions written by the two
            // append dispatches in this same compute encoder.
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
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
        // fa2 requires dk/dv %8, <=256 (C=64 & PV8 divisible by NSG=4 always hold).
        constexpr bool use_simple = false;
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
            // Flash attention: one threadgroup (NSG simdgroups, 32*NSG threads) per
            // (query-tile=Q, head). Threadgroup memory:
            //   sq[Q*DK] half + so[Q*PV] float + ss[Q*SH] float, SH=2*C, PV=PAD(DV,64).
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
            if (!inputs[i] || !inputs[i]->device_data) continue;
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
        float routed_scale = params.f32.size()>0 ? params.f32[0] : 1.0f;

        // Youtu-v2/Qwen3 W4 routed experts: keep the whole operation on GPU.
        // The decoded expert buffers are flat row-major despite their logical
        // 3-D Tensor shape (see wrap_weight_int4_g128).
        // Decode uses selected-expert tensor MMA by default on Metal 4 GPUs.
        // Prefill stays on the CPU hybrid path until expert-grouped M tiles are
        // available.
        bool gpu_w4 = impl_->has_tensor &&
            !has_shared && router_score_func == 1 && inputs.size() > 8 &&
            inputs[1]->prec == Precision::FP16 && inputs[2]->prec == Precision::INT4 &&
            inputs[3]->prec == Precision::INT4 && top_k <= 16 && n_group <= 16 &&
            inputs[0]->shape[1] == 1;  // selected M=1 tensor tiles are decode-only
        if (gpu_w4) {
            const Tensor& x = *inputs[0]; const Tensor& router = *inputs[1];
            const Tensor& gu = *inputs[2]; const Tensor& down = *inputs[3];
            const Tensor& bias = *inputs[8];
            int seq = (int)x.shape[1];
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
            MoeW4Params mp{};
            mp.hidden=hidden_size;mp.experts=num_experts;mp.top_k=top_k;
            mp.intermediate=intermediate;mp.seq_len=seq;mp.n_group=std::max(1,n_group);
            mp.topk_group=std::max(1,topk_group);mp.norm_topk=norm_topk;
            mp.routed_scale=routed_scale;mp.hidden_offset=eoffset(x);mp.output_offset=eoffset(*output);
            mp.hidden_row_stride=estride(x,1);mp.output_row_stride=estride(*output,1);
            mp.gu_groups_per_row=(int)gu.groups_per_row;
            mp.down_groups_per_row=(int)down.groups_per_row;
            size_t gu_rows=(size_t)num_experts*2*intermediate;
            size_t down_rows=(size_t)num_experts*hidden_size;

            [enc setComputePipelineState:impl_->pipeline("moe_router_logits_f16")];
            [enc setBuffer:buf_of(&x) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&router) offset:router.device_offset atIndex:1];
            [enc setBuffer:logits offset:0 atIndex:2];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
            [enc setThreadgroupMemoryLength:4*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(num_experts,seq,1)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [enc setComputePipelineState:impl_->pipeline("moe_select_sigmoid")];
            [enc setBuffer:logits offset:0 atIndex:0];[enc setBuffer:idx offset:0 atIndex:1];
            [enc setBuffer:tw offset:0 atIndex:2];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
            [enc setBuffer:buf_of(&bias) offset:bias.device_offset atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)seq+63)/64,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];

#ifdef MOLLM_METAL_TENSOR
            if (impl_->has_tensor) {
                int selections=seq*top_k;
                size_t qx_bytes=(size_t)seq*hidden_size, sx_bytes=(size_t)seq*sizeof(float);
                size_t qi_bytes=(size_t)selections*intermediate;
                size_t si_bytes=(size_t)selections*sizeof(float);
                size_t selected_bytes=(size_t)selections*hidden_size*sizeof(float);
                void* qx_h=impl_->pool->acquire(qx_bytes),*sx_h=impl_->pool->acquire(sx_bytes);
                void* qi_h=impl_->pool->acquire(qi_bytes),*si_h=impl_->pool->acquire(si_bytes);
                void* selected_h=impl_->pool->acquire(selected_bytes);
                id<MTLBuffer> qx=(__bridge id<MTLBuffer>)qx_h;
                id<MTLBuffer> sx=(__bridge id<MTLBuffer>)sx_h;
                id<MTLBuffer> qi=(__bridge id<MTLBuffer>)qi_h;
                id<MTLBuffer> si=(__bridge id<MTLBuffer>)si_h;
                id<MTLBuffer> selected=(__bridge id<MTLBuffer>)selected_h;
                auto quantize = [&](id<MTLBuffer> src, uint src_off, int rows, int K,
                                    int row_stride, id<MTLBuffer> dst, id<MTLBuffer> scales) {
                    QuantActParams q{};q.M=rows;q.K=K;q.a_offset=src_off;q.a_row_stride=row_stride;
                    [enc setComputePipelineState:impl_->pipeline("quantize_act_i8")];
                    [enc setBuffer:src offset:0 atIndex:0];[enc setBuffer:dst offset:0 atIndex:2];
                    [enc setBytes:&q length:sizeof(q) atIndex:3];[enc setBuffer:scales offset:0 atIndex:4];
                    [enc setThreadgroupMemoryLength:8*sizeof(float) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(rows,1,1)
                        threadsPerThreadgroup:MTLSizeMake(32,8,1)];
                };
                quantize(buf_of(&x),(uint)eoffset(x),seq,hidden_size,estride(x,1),qx,sx);
                auto selected_gemm = [&](id<MTLBuffer> a,id<MTLBuffer> sa,const Tensor& w,
                                         size_t rows_total,int N,int K,int rows_per_expert,
                                         int activation_rows,int repeat,id<MTLBuffer> dst,int dst_stride) {
                    SelectedW4A8Params sp{};sp.selections=selections;sp.N=N;sp.K=K;
                    sp.c_row_stride=dst_stride;sp.group_size=(int)w.group_size;
                    sp.groups_per_row=(int)w.groups_per_row;sp.rows_per_expert=rows_per_expert;
                    sp.activation_rows=activation_rows;sp.activation_repeat=repeat;
                    [enc setComputePipelineState:impl_->pipeline("gemm_selected_w4a8_i8a_i4b_f32c")];
                    [enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:buf_of(&w) offset:w.device_offset atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];[enc setBytes:&sp length:sizeof(sp) atIndex:3];
                    [enc setBuffer:sa offset:0 atIndex:4];
                    [enc setBuffer:buf_of(&w) offset:w.device_offset+rows_total*(K/2) atIndex:5];
                    [enc setBuffer:idx offset:0 atIndex:6];
                    [enc setThreadgroupMemoryLength:2*64*64*sizeof(int32_t) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(1,(N+63)/64,selections)
                        threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                };
                selected_gemm(qx,sx,gu,gu_rows,2*intermediate,hidden_size,
                              2*intermediate,seq,top_k,merged,2*intermediate);
                [enc setComputePipelineState:impl_->pipeline("moe_swiglu_selected")];
                [enc setBuffer:merged offset:0 atIndex:0];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
                NSUInteger sw_n=(NSUInteger)selections*intermediate;
                [enc dispatchThreadgroups:MTLSizeMake((sw_n+255)/256,1,1)
                    threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                quantize(merged,0,selections,intermediate,2*intermediate,qi,si);
                selected_gemm(qi,si,down,down_rows,hidden_size,intermediate,
                              hidden_size,selections,1,selected,hidden_size);
                [enc setComputePipelineState:impl_->pipeline("moe_combine_selected")];
                [enc setBuffer:selected offset:0 atIndex:0];[enc setBuffer:buf_of(output) offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];[enc setBuffer:tw offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((hidden_size+63)/64,(seq+3)/4,1)
                    threadsPerThreadgroup:MTLSizeMake(64,4,1)];
                impl_->pending_free.push_back({qx_h,qx_bytes});impl_->pending_free.push_back({sx_h,sx_bytes});
                impl_->pending_free.push_back({qi_h,qi_bytes});impl_->pending_free.push_back({si_h,si_bytes});
                impl_->pending_free.push_back({selected_h,selected_bytes});
                impl_->pending_free.push_back({idx_h,idx_bytes});impl_->pending_free.push_back({tw_h,tw_bytes});
                impl_->pending_free.push_back({logits_h,logits_bytes});impl_->pending_free.push_back({merged_h,merged_bytes});
                break;
            }
#endif

            [enc setComputePipelineState:impl_->pipeline("moe_gate_up_w4")];
            [enc setBuffer:buf_of(&x) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&gu) offset:gu.device_offset atIndex:1];
            [enc setBuffer:merged offset:0 atIndex:2];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
            [enc setBuffer:buf_of(&gu) offset:gu.device_offset+gu_rows*(hidden_size/2) atIndex:4];
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
            [enc setBuffer:buf_of(&down) offset:down.device_offset atIndex:1];
            [enc setBuffer:buf_of(output) offset:0 atIndex:2];[enc setBytes:&mp length:sizeof(mp) atIndex:3];
            [enc setBuffer:buf_of(&down) offset:down.device_offset+down_rows*(intermediate/2) atIndex:4];
            [enc setBuffer:idx offset:0 atIndex:5];[enc setBuffer:tw offset:0 atIndex:6];
            [enc setThreadgroupMemoryLength:4*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((hidden_size+3)/4,seq,1)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            impl_->pending_free.push_back({idx_h,idx_bytes});
            impl_->pending_free.push_back({tw_h,tw_bytes});
            impl_->pending_free.push_back({logits_h,logits_bytes});
            impl_->pending_free.push_back({merged_h,merged_bytes});
            break;
        }

        // Generic correctness fallback for FP16/W8/shared-expert variants.
        if (impl_->enc) { [impl_->enc endEncoding]; impl_->enc = nil; }
        if (impl_->cmd) {
            [impl_->cmd commit];
            [impl_->cmd waitUntilCompleted];
            if (impl_->cmd.status == MTLCommandBufferStatusError) {
                NSError* e = impl_->cmd.error;
                fprintf(stderr, "MetalBackend: pre-MOE command buffer error: %s\n",
                        e ? e.localizedDescription.UTF8String : "?");
            }
            impl_->cmd = nil;
        }

        kernel_qwen3_moe(inputs, *output, thread_pool,
                         hidden_size, num_experts, top_k,
                         intermediate, shared_intermediate,
                         router_score_func, norm_topk, has_shared,
                         n_group, topk_group, routed_scale);

        impl_->cmd = [impl_->queue commandBuffer];
        impl_->enc = [impl_->cmd computeCommandEncoder];
        break;
    }

    default:
        fprintf(stderr, "MetalBackend: unsupported op %d\n", (int)op);
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
            auto& st = impl_->op_stats[profile_label];
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
