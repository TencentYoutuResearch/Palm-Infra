#include "engine/metal_backend.h"
#include "graph/metal_pool.h"
#include "graph/graph.h"
#include "graph/mmap_file.h"
#include "kernels/matmul.h"
#include "kernels/moe.h"
#include "kernels/moe_routing.h"
#include "kernels/moe_ssd.h"
#include "kernels/metal/metal_common.h"
#include "kernels/trace.h"

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
#include <unordered_set>
#include <vector>

#ifndef MOLLM_METALLIB_PATH
#define MOLLM_METALLIB_PATH ""
#endif

// CPU-side packed INT4 blocks, mirrored from kernels/matmul_internal.h. Dense
// weights are decoded into a Metal-friendly raw layout at load time; aggregate
// experts remain native and are read directly by specialized MoE kernels.
struct alignas(16) Q4B8G128Block {
    float   scales[8];
    uint8_t q[4][8][16];
};
struct alignas(16) Q4B8G32Block {
    float   scales[8];
    uint8_t q[8][16];
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

    struct SsdExpertBuffers {
        id<MTLBuffer> gate_up = nil;
        id<MTLBuffer> down = nil;
        id<MTLSharedEvent> gate_ready_event = nil;
        id<MTLSharedEvent> down_ready_event = nil;
        size_t gate_up_offset = 0;
        size_t down_offset = 0;
        size_t slot = 0;
        size_t bytes = 0;
        uint64_t used_at = 0;
        uint64_t gate_ready_value = 0;
        uint64_t down_ready_value = 0;
    };
    struct SsdExpertView {
        id<MTLBuffer> gate_up = nil;
        id<MTLBuffer> down = nil;
        id<MTLSharedEvent> gate_ready_event = nil;
        id<MTLSharedEvent> down_ready_event = nil;
        size_t gate_up_offset = 0;
        size_t down_offset = 0;
        uint64_t gate_ready_value = 0;
        uint64_t down_ready_value = 0;
    };
    id<MTLIOCommandQueue> ssd_io_queue = nil;
    id<MTLIOFileHandle> ssd_file = nil;
    id<MTLCommandQueue> ssd_shared_compute_queue = nil;
    id<MTLSharedEvent> ssd_shared_compute_event = nil;
    uint64_t ssd_shared_compute_event_value = 0;
    bool ssd_cross_layer_prefetch = true;
    std::unordered_map<uint64_t, SsdExpertBuffers> ssd_experts;
    id<MTLBuffer> ssd_arena = nil;
    size_t ssd_slot_bytes = 0;
    std::vector<size_t> ssd_free_slots;
    size_t ssd_capacity_bytes = 0;
    size_t ssd_resident_bytes = 0;
    uint64_t ssd_clock = 0;
    uint64_t ssd_hits = 0;
    uint64_t ssd_misses = 0;
    uint64_t ssd_bytes_read = 0;
    int ssd_active_layer = -1;
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
    };
    std::unordered_map<int, SsdMoeLayerInfo> ssd_moe_layers;

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
    int                         ops_in_cmd = 0;
    bool                        chunk_graph = false;

    bool ok = false;

    static uint64_t ssd_key(int layer, int expert) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
               static_cast<uint32_t>(expert);
    }

    bool acquire_ssd_experts(const MoeSsdTensorSpec& gate,
                             const MoeSsdTensorSpec& down,
                             const std::vector<int>& experts,
                             std::vector<SsdExpertView>& views,
                             bool speculative = false) {
        views.clear();
        if (!ssd_io_queue || !ssd_file ||
            gate.layer != down.layer ||
            gate.precision != Precision::INT4 ||
            down.precision != Precision::INT4 ||
            (gate.flags & MappedFile::FLAG_INT4_BG128) == 0 ||
            (down.flags & MappedFile::FLAG_INT4_BG128) == 0)
            return false;

        const size_t pair_bytes =
            static_cast<size_t>(gate.data_bytes + down.data_bytes);
        if (pair_bytes == 0 || pair_bytes > ssd_capacity_bytes)
            return false;
        if (!speculative)
            ssd_active_layer = gate.layer;
        if (!ssd_arena) {
            constexpr size_t alignment = 4096;
            ssd_slot_bytes =
                (pair_bytes + alignment - 1) & ~(alignment - 1);
            const size_t slot_count = ssd_capacity_bytes / ssd_slot_bytes;
            if (slot_count == 0)
                return false;
            const size_t arena_bytes = slot_count * ssd_slot_bytes;
            ssd_arena =
                [device newBufferWithLength:arena_bytes
                                    options:MTLResourceStorageModeShared];
            if (!ssd_arena)
                return false;
            ssd_free_slots.reserve(slot_count);
            for (size_t slot = slot_count; slot > 0; --slot)
                ssd_free_slots.push_back(slot - 1);
        } else if (pair_bytes > ssd_slot_bytes) {
            return false;
        }

        std::vector<int> missing;
        std::unordered_set<uint64_t> requested;
        for (int expert : experts) {
            const uint64_t key = ssd_key(gate.layer, expert);
            const bool first_request = requested.insert(key).second;
            auto found = ssd_experts.find(key);
            if (found == ssd_experts.end()) {
                if (first_request) {
                    missing.push_back(expert);
                    ++ssd_misses;
                }
            } else {
                found->second.used_at = ++ssd_clock;
                if (first_request)
                    ++ssd_hits;
            }
        }

        // Submit file reads in physical expert order. The gate/up and down
        // aggregates occupy two distant package ranges; alternating between
        // them for every route turns one top-k request into repeated long
        // offset jumps. Sorting and issuing one range at a time gives Metal
        // I/O a monotonically increasing stream within each aggregate.
        std::sort(missing.begin(), missing.end());
        const size_t required = pair_bytes * missing.size();
        while (ssd_resident_bytes + required > ssd_capacity_bytes) {
            auto victim = ssd_experts.end();
            int victim_priority = std::numeric_limits<int>::max();
            for (auto it = ssd_experts.begin(); it != ssd_experts.end(); ++it) {
                if (requested.count(it->first) ||
                    (it->second.gate_ready_event &&
                     it->second.gate_ready_event.signaledValue <
                         it->second.gate_ready_value) ||
                    (it->second.down_ready_event &&
                     it->second.down_ready_event.signaledValue <
                         it->second.down_ready_value))
                    continue;
                const int entry_layer =
                    static_cast<int>(it->first >> 32);
                // Layers already consumed in the current forward pass are the
                // cheapest victims. Preserve future-layer entries so the next
                // demand can hit, rather than letting global LRU turn a small
                // cache into a layer-by-layer streaming window.
                int priority = entry_layer < ssd_active_layer ? 0 : 2;
                if (entry_layer == ssd_active_layer) {
                    // A speculative next-layer load is submitted before the
                    // current layer's routed command executes, so its slots
                    // must not overwrite current expert weights.
                    if (speculative)
                        continue;
                    priority = 1;
                }
                if (victim == ssd_experts.end() ||
                    priority < victim_priority ||
                    (priority == victim_priority &&
                     it->second.used_at < victim->second.used_at)) {
                    victim = it;
                    victim_priority = priority;
                }
            }
            if (victim == ssd_experts.end())
                return false;
            ssd_resident_bytes -= victim->second.bytes;
            ssd_free_slots.push_back(victim->second.slot);
            ssd_experts.erase(victim);
        }

        if (!missing.empty()) {
            id<MTLIOCommandBuffer> gate_io =
                [ssd_io_queue commandBuffer];
            id<MTLIOCommandBuffer> down_io =
                [ssd_io_queue commandBuffer];
            if (!gate_io || !down_io)
                return false;
            // Give every load batch its own completion event. A single global
            // monotonically-valued event would require command N+1 to wait for
            // N: otherwise a fast later signal could incorrectly make an
            // earlier load appear complete. Independent events preserve slot
            // readiness while allowing the concurrent MTLIO queue to let a
            // demand read bypass an older speculative prefetch.
            id<MTLSharedEvent> gate_ready_event = [device newSharedEvent];
            id<MTLSharedEvent> down_ready_event = [device newSharedEvent];
            if (!gate_ready_event || !down_ready_event)
                return false;
            constexpr uint64_t load_ready_value = 1;
            for (int expert : missing) {
                if (ssd_free_slots.empty())
                    return false;
                SsdExpertBuffers entry;
                entry.slot = ssd_free_slots.back();
                ssd_free_slots.pop_back();
                entry.gate_up = ssd_arena;
                entry.down = ssd_arena;
                entry.gate_ready_event = gate_ready_event;
                entry.down_ready_event = down_ready_event;
                entry.gate_up_offset = entry.slot * ssd_slot_bytes;
                entry.down_offset =
                    entry.gate_up_offset + static_cast<size_t>(gate.data_bytes);
                entry.bytes = pair_bytes;
                entry.used_at = ++ssd_clock;
                entry.gate_ready_value = load_ready_value;
                entry.down_ready_value = load_ready_value;
                ssd_experts.emplace(ssd_key(gate.layer, expert),
                                    std::move(entry));
            }
            for (int expert : missing) {
                const auto& entry =
                    ssd_experts.at(ssd_key(gate.layer, expert));
                [gate_io
                    loadBuffer:entry.gate_up
                         offset:entry.gate_up_offset
                           size:gate.data_bytes
                   sourceHandle:ssd_file
             sourceHandleOffset:gate.data_offset +
                                static_cast<uint64_t>(expert) *
                                    gate.data_bytes];
            }
            for (int expert : missing) {
                const auto& entry =
                    ssd_experts.at(ssd_key(gate.layer, expert));
                [down_io
                    loadBuffer:entry.down
                         offset:entry.down_offset
                           size:down.data_bytes
                   sourceHandle:ssd_file
             sourceHandleOffset:down.data_offset +
                                static_cast<uint64_t>(expert) *
                                    down.data_bytes];
            }
            const uint64_t gate_start = mollm_trace::now_ns();
            const uint64_t down_start = gate_start;
            [gate_io signalEvent:gate_ready_event value:load_ready_value];
            [down_io signalEvent:down_ready_event value:load_ready_value];
            const int layer = gate.layer;
            const size_t expert_count = missing.size();
            const size_t gate_bytes =
                static_cast<size_t>(gate.data_bytes) * expert_count;
            const size_t down_bytes =
                static_cast<size_t>(down.data_bytes) * expert_count;
            [gate_io
                addCompletedHandler:^(id<MTLIOCommandBuffer> completed) {
                if (gate_start != 0) {
                    mollm_trace::record_duration(
                        "metal.ssd", "io.load_gate_up", gate_start,
                        mollm_trace::now_ns(),
                        "{\"layer\":" + std::to_string(layer) +
                        ",\"experts\":" + std::to_string(expert_count) +
                        ",\"bytes\":" + std::to_string(gate_bytes) + "}");
                }
                if (completed.status != MTLIOStatusComplete) {
                    fprintf(stderr,
                            "MetalBackend: asynchronous gate/up SSD read "
                            "failed in layer %d\n",
                            layer);
                }
            }];
            [down_io
                addCompletedHandler:^(id<MTLIOCommandBuffer> completed) {
                if (down_start != 0) {
                    mollm_trace::record_duration(
                        "metal.ssd", "io.load_down", down_start,
                        mollm_trace::now_ns(),
                        "{\"layer\":" + std::to_string(layer) +
                        ",\"experts\":" + std::to_string(expert_count) +
                        ",\"bytes\":" + std::to_string(down_bytes) + "}");
                }
                if (completed.status != MTLIOStatusComplete) {
                    fprintf(stderr,
                            "MetalBackend: asynchronous down SSD read "
                            "failed in layer %d\n",
                            layer);
                }
            }];
            [gate_io commit];
            [down_io commit];
            ssd_resident_bytes += required;
            ssd_bytes_read += required;
        }

        views.reserve(experts.size());
        for (int expert : experts) {
            auto found = ssd_experts.find(ssd_key(gate.layer, expert));
            if (found == ssd_experts.end())
                return false;
            views.push_back({
                found->second.gate_up,
                found->second.down,
                found->second.gate_ready_event,
                found->second.down_ready_event,
                found->second.gate_up_offset,
                found->second.down_offset,
                found->second.gate_ready_value,
                found->second.down_ready_value,
            });
        }
        return true;
    }

    bool finish_ssd_prefix(int layer, const char* error_context) {
        [enc endEncoding];
        enc = nil;
        const uint64_t wait_start = mollm_trace::now_ns();
        [cmd commit];
        [cmd waitUntilCompleted];
        const uint64_t wait_end = mollm_trace::now_ns();
        const std::string args =
            "{\"layer\":" + std::to_string(layer) + "}";
        mollm_trace::record_duration(
            "metal.ssd", "prefix_wait", wait_start, wait_end, args,
            "thread_state_iowait");
        const double gpu_seconds = cmd.GPUEndTime - cmd.GPUStartTime;
        if (gpu_seconds > 0.0 && wait_end != 0) {
            const uint64_t gpu_ns =
                static_cast<uint64_t>(gpu_seconds * 1e9);
            mollm_trace::record_duration(
                "metal.ssd", "prefix_gpu",
                wait_end > gpu_ns ? wait_end - gpu_ns : 0,
                wait_end, args, "thread_state_running");
        }
        if (cmd.status == MTLCommandBufferStatusError) {
            NSError* error = cmd.error;
            fprintf(stderr, "MetalBackend: %s failed: %s\n", error_context,
                    error ? error.localizedDescription.UTF8String : "?");
            cmd = nil;
            return false;
        }
        cmd = nil;
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

    struct SsdSharedExpertWork {
        void* qx = nullptr;
        void* sx = nullptr;
        void* intermediate = nullptr;
        void* qintermediate = nullptr;
        void* qintermediate_scale = nullptr;
        void* scale = nullptr;
        void* output = nullptr;
        size_t qx_bytes = 0;
        size_t sx_bytes = 0;
        size_t intermediate_bytes = 0;
        size_t qintermediate_bytes = 0;
        size_t output_bytes = 0;
        uint64_t ready_value = 0;
    };

    bool submit_ssd_shared_expert(
        const Tensor& x, const Tensor& gate, const Tensor& up,
        const Tensor& down, const Tensor& scale_weight, int hidden,
        int intermediate, int seq, int layer, SsdSharedExpertWork& work) {
        if (gate.prec != Precision::INT4 ||
            up.prec != Precision::INT4 ||
            down.prec != Precision::INT4 ||
            scale_weight.prec != Precision::FP16 ||
            !x.device_data || !gate.device_data || !up.device_data ||
            !down.device_data || !scale_weight.device_data) {
            return false;
        }

        work.qx_bytes = static_cast<size_t>(seq) * hidden;
        work.sx_bytes = static_cast<size_t>(seq) * sizeof(float);
        work.intermediate_bytes =
            static_cast<size_t>(intermediate) * sizeof(float);
        work.qintermediate_bytes =
            static_cast<size_t>(intermediate) * sizeof(int8_t);
        work.output_bytes = static_cast<size_t>(hidden) * sizeof(float);
        work.qx = pool->acquire(work.qx_bytes);
        work.sx = pool->acquire(work.sx_bytes);
        work.intermediate = pool->acquire(work.intermediate_bytes);
        work.qintermediate = pool->acquire(work.qintermediate_bytes);
        work.qintermediate_scale = pool->acquire(sizeof(float));
        work.scale = pool->acquire(sizeof(float));
        work.output = pool->acquire(work.output_bytes);

        id<MTLBuffer> x_buffer = (__bridge id<MTLBuffer>)x.device_data;
        id<MTLBuffer> gate_buffer = (__bridge id<MTLBuffer>)gate.device_data;
        id<MTLBuffer> up_buffer = (__bridge id<MTLBuffer>)up.device_data;
        id<MTLBuffer> down_buffer = (__bridge id<MTLBuffer>)down.device_data;
        id<MTLBuffer> scale_weight_buffer =
            (__bridge id<MTLBuffer>)scale_weight.device_data;
        id<MTLBuffer> qx = (__bridge id<MTLBuffer>)work.qx;
        id<MTLBuffer> sx = (__bridge id<MTLBuffer>)work.sx;
        id<MTLBuffer> hidden_values =
            (__bridge id<MTLBuffer>)work.intermediate;
        id<MTLBuffer> qhidden =
            (__bridge id<MTLBuffer>)work.qintermediate;
        id<MTLBuffer> qhidden_scale =
            (__bridge id<MTLBuffer>)work.qintermediate_scale;
        id<MTLBuffer> scale = (__bridge id<MTLBuffer>)work.scale;
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)work.output;

        MoeSharedW4Params params{};
        params.hidden = hidden;
        params.intermediate = intermediate;
        params.gate_groups_per_row = static_cast<int>(gate.groups_per_row);
        params.up_groups_per_row = static_cast<int>(up.groups_per_row);
        params.down_groups_per_row = static_cast<int>(down.groups_per_row);
        params.hidden_offset =
            static_cast<uint>(x.device_offset / sizeof(float));

        id<MTLCommandBuffer> shared_cmd =
            [ssd_shared_compute_queue commandBuffer];
        shared_cmd.label = @"mollm shared SSD expert";
        id<MTLComputeCommandEncoder> shared_enc =
            [shared_cmd computeCommandEncoder];
        shared_enc.label = @"mollm shared expert";

        QuantActParams xq{};
        xq.M = seq;
        xq.K = hidden;
        xq.a_offset = params.hidden_offset;
        xq.a_row_stride =
            static_cast<int>(x.stride[1] / sizeof(float));
        [shared_enc setComputePipelineState:pipeline("quantize_act_i8")];
        [shared_enc setBuffer:x_buffer offset:0 atIndex:0];
        [shared_enc setBuffer:qx offset:0 atIndex:2];
        [shared_enc setBytes:&xq length:sizeof(xq) atIndex:3];
        [shared_enc setBuffer:sx offset:0 atIndex:4];
        [shared_enc setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
        [shared_enc dispatchThreadgroups:MTLSizeMake(seq, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];

        [shared_enc setComputePipelineState:
                        pipeline("moe_shared_gate_up_w4_i8")];
        [shared_enc setBuffer:qx offset:0 atIndex:0];
        [shared_enc setBuffer:gate_buffer offset:gate.device_offset atIndex:1];
        [shared_enc setBuffer:hidden_values offset:0 atIndex:2];
        [shared_enc setBytes:&params length:sizeof(params) atIndex:3];
        [shared_enc
            setBuffer:gate_buffer
               offset:gate.device_offset +
                      static_cast<size_t>(intermediate) * hidden / 2
              atIndex:4];
        [shared_enc setBuffer:up_buffer offset:up.device_offset atIndex:5];
        [shared_enc
            setBuffer:up_buffer
               offset:up.device_offset +
                      static_cast<size_t>(intermediate) * hidden / 2
              atIndex:6];
        [shared_enc setBuffer:sx offset:0 atIndex:7];
        [shared_enc dispatchThreadgroups:MTLSizeMake(intermediate, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [shared_enc setComputePipelineState:
                        pipeline("moe_shared_scale_f16")];
        [shared_enc setBuffer:x_buffer offset:0 atIndex:0];
        [shared_enc setBuffer:scale_weight_buffer
                       offset:scale_weight.device_offset
                      atIndex:1];
        [shared_enc setBuffer:scale offset:0 atIndex:2];
        [shared_enc setBytes:&params length:sizeof(params) atIndex:3];
        [shared_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        QuantActParams iq{};
        iq.M = 1;
        iq.K = intermediate;
        iq.a_row_stride = intermediate;
        [shared_enc setComputePipelineState:pipeline("quantize_act_i8")];
        [shared_enc setBuffer:hidden_values offset:0 atIndex:0];
        [shared_enc setBuffer:qhidden offset:0 atIndex:2];
        [shared_enc setBytes:&iq length:sizeof(iq) atIndex:3];
        [shared_enc setBuffer:qhidden_scale offset:0 atIndex:4];
        [shared_enc setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
        [shared_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];

        [shared_enc setComputePipelineState:
                        pipeline("moe_shared_down_w4_i8")];
        [shared_enc setBuffer:qhidden offset:0 atIndex:0];
        [shared_enc setBuffer:down_buffer offset:down.device_offset atIndex:1];
        [shared_enc setBuffer:output offset:0 atIndex:2];
        [shared_enc setBytes:&params length:sizeof(params) atIndex:3];
        [shared_enc
            setBuffer:down_buffer
               offset:down.device_offset +
                      static_cast<size_t>(hidden) * intermediate / 2
              atIndex:4];
        [shared_enc setBuffer:scale offset:0 atIndex:5];
        [shared_enc setBuffer:qhidden_scale offset:0 atIndex:6];
        [shared_enc dispatchThreadgroups:MTLSizeMake(hidden, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [shared_enc endEncoding];

        work.ready_value = ++ssd_shared_compute_event_value;
        [shared_cmd encodeSignalEvent:ssd_shared_compute_event
                                value:work.ready_value];
        const uint64_t start = mollm_trace::now_ns();
        [shared_cmd addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            const uint64_t end = mollm_trace::now_ns();
            const std::string args =
                "{\"layer\":" + std::to_string(layer) + "}";
            mollm_trace::record_duration(
                "metal.ssd", "shared_expert", start, end, args,
                "thread_state_running");
            const double gpu_seconds =
                completed.GPUEndTime - completed.GPUStartTime;
            if (gpu_seconds > 0.0 && end != 0) {
                const uint64_t gpu_ns =
                    static_cast<uint64_t>(gpu_seconds * 1e9);
                mollm_trace::record_duration(
                    "metal.ssd", "shared_expert_gpu",
                    end > gpu_ns ? end - gpu_ns : 0, end, args,
                    "thread_state_running");
            }
        }];
        [shared_cmd commit];
        return true;
    }

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

    // Specialized-pipeline cache keyed by name + function-constant tuple. The
    // flash-attention prefill kernel bakes DK, DV, its SIMD-group split, and
    // query tile into the generated pipeline. A failed specialization returns
    // nil so the caller can use the generic prefill kernel.
    std::unordered_map<std::string, id<MTLComputePipelineState>> spec_pipelines;
    id<MTLComputePipelineState> pipeline_fa2(
            int dk, int dv, int nsg, int qt) {
        char keyc[64];
        snprintf(
            keyc, sizeof(keyc),
            "fa2:dk%d:dv%d:nsg%d:qt%d",
            dk, dv, nsg, qt);
        std::string key(keyc);
        auto it = spec_pipelines.find(key);
        if (it != spec_pipelines.end()) return it->second;

        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&dk type:MTLDataTypeInt atIndex:0];  // FC_SDPA_DK
        [cv setConstantValue:&dv type:MTLDataTypeInt atIndex:1];  // FC_SDPA_DV
        [cv setConstantValue:&nsg
                       type:MTLDataTypeInt atIndex:9];
        [cv setConstantValue:&qt
                       type:MTLDataTypeInt atIndex:11];
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

    id<MTLComputePipelineState> pipeline_w4a16(
            bool use_m128, bool specialize_g128) {
        const char* function_name =
            use_m128
                ? "gemm_tensor_w4_f32a_i4b_f32c"
                : "gemm_tensor_w4_f32a_i4b_f32c_m64";
        const std::string key =
            std::string("w4a16:") +
            (specialize_g128 ? "g128:" : "generic:") +
            (use_m128 ? "m128" : "m64");
        auto it = spec_pipelines.find(key);
        if (it != spec_pipelines.end())
            return it->second;

        MTLFunctionConstantValues* cv =
            [[MTLFunctionConstantValues alloc] init];
        bool enabled = specialize_g128;
        [cv setConstantValue:&enabled
                       type:MTLDataTypeBool
                    atIndex:10];
        NSError* err = nil;
        id<MTLFunction> fn =
            [library
                newFunctionWithName:
                    [NSString stringWithUTF8String:function_name]
                          constantValues:cv
                                   error:&err];
        id<MTLComputePipelineState> ps =
            fn ? [device
                     newComputePipelineStateWithFunction:fn
                                                   error:&err]
               : nil;
        if (!ps)
            fprintf(
                stderr,
                "MetalBackend: W4A16 G128 pipeline failed: %s\n",
                err ? err.localizedDescription.UTF8String : "?");
        spec_pipelines[key] = ps;
        return ps;
    }

    id<MTLComputePipelineState> pipeline_moe_select_parallel(
            bool sigmoid, bool grouped) {
        const std::string key =
            std::string("moe_select_parallel:") +
            (sigmoid ? "sigmoid" : "softmax") +
            (grouped ? ":grouped" : ":ungrouped");
        auto it = spec_pipelines.find(key);
        if (it != spec_pipelines.end()) return it->second;

        MTLFunctionConstantValues* cv =
            [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&sigmoid
                       type:MTLDataTypeBool
                    atIndex:7];
        [cv setConstantValue:&grouped
                       type:MTLDataTypeBool
                    atIndex:8];
        NSError* err = nil;
        id<MTLFunction> fn =
            [library newFunctionWithName:@"moe_select_parallel"
                          constantValues:cv error:&err];
        id<MTLComputePipelineState> ps = fn
            ? [device newComputePipelineStateWithFunction:fn error:&err]
            : nil;
        if (!ps)
            fprintf(stderr,
                    "MetalBackend: parallel %s MoE selector pipeline "
                    "failed: %s\n",
                    grouped ? "grouped sigmoid"
                            : sigmoid ? "sigmoid" : "softmax",
                    err ? err.localizedDescription.UTF8String : "?");
        spec_pipelines[key] = ps;
        return ps;
    }

    id<MTLComputePipelineState> pipeline_grouped_moe(
            int group_size, bool paired_gate_up,
            bool large_route_tile) {
        const char* layout =
            group_size == 32 ? "bg32" : "bg128";
        if (paired_gate_up) {
            const std::string name =
                std::string("gemm_grouped_experts_") + layout +
                (large_route_tile
                     ? "_gate_up_r32"
                     : "_gate_up_r16");
            return pipeline(name.c_str());
        }
        const std::string name =
            std::string("gemm_grouped_experts_") + layout +
            (large_route_tile
                 ? "_down_r32"
                 : "_down_r16");
        return pipeline(name.c_str());
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
    case Precision::FP8_E4M3: return 1;
    case Precision::MXFP4: return 1;
    case Precision::INT32: return 4;
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
        if (!value) return 8;
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
    (void)n;
    (void)k;
    return 1;
}

int gemv_w4_nsg_cap() {
    static const int cap = [] {
        const char* value = std::getenv("MOLLM_METAL_GEMV_W4_NSG");
        if (!value) return 4;
        const int parsed = std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8)
                   ? parsed
                   : 4;
    }();
    return cap;
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
        if (impl_->ssd_io_queue) {
            fprintf(stderr,
                    "MetalBackend: SSD cache hits=%llu misses=%llu "
                    "read_mb=%.1f resident_mb=%.1f\n",
                    (unsigned long long)impl_->ssd_hits,
                    (unsigned long long)impl_->ssd_misses,
                    impl_->ssd_bytes_read / 1e6,
                    impl_->ssd_resident_bytes / 1e6);
        }
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
        // Standalone path used by prefill/raw-logit callers.
        void* abuf = impl_->pool->acquire((size_t)K * 4);
        std::memcpy(MetalBufferPool::contents(abuf), a_host, (size_t)K * 4);
        lm_head_gemv_impl(abuf, 0, weight, out_host, N, K, activation,
                          false);
        impl_->pool->release(abuf, (size_t)K * 4);
    }
}

void MetalBackend::lm_head_gemv_device_and_end_graph(
    const Tensor& a, size_t a_element_offset, const Tensor& weight,
    float* out_host, int N, int K, int activation) {
    lm_head_gemv_impl(a.device_data,
                      a.device_offset + a_element_offset*sizeof(float),
                      weight, out_host, N, K, activation, true);
}

void MetalBackend::lm_head_gemv_impl(
    void* a_device, size_t a_byte_offset, const Tensor& weight,
    float* out_host, int N, int K, int activation,
    bool finish_open_graph) {
    @autoreleasepool {
        void* cbuf = impl_->pool->acquire((size_t)N * 4);
        MatmulParams p{};
        p.M = 1; p.N = N; p.K = K;
        p.a_offset = 0;
        p.b_offset = 0;  // bind B at its byte offset below (64-bit, no overflow)
        p.c_offset = 0;
        p.a_row_stride = K;
        p.b_row_stride = (int)weight.shape[1];  // K
        p.c_row_stride = N;
        p.activation = activation;

        id<MTLBuffer> A = (__bridge id<MTLBuffer>)a_device;
        id<MTLBuffer> B = buf_of(&weight);
        id<MTLBuffer> C = (__bridge id<MTLBuffer>)cbuf;

        id<MTLCommandBuffer> cmd =
            finish_open_graph ? impl_->cmd : [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc =
            finish_open_graph ? impl_->enc : [cmd computeCommandEncoder];
        assert(cmd && enc);
        [enc setBuffer:A offset:a_byte_offset atIndex:0];
        [enc setBuffer:B offset:weight.device_offset atIndex:1];
        [enc setBuffer:C offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        id<MTLComputePipelineState> ps = nil;
        if (weight.prec == Precision::INT8) {
            MatmulW8Params w{};
            w.M=1; w.N=N; w.K=K;
            w.a_offset=0; w.c_offset=0;
            w.a_row_stride=K; w.c_row_stride=N;
            w.activation=activation;
            w.group_size=(int)weight.group_size;
            w.groups_per_row=(int)weight.groups_per_row;
            const size_t scales_boff =
                (char*)weight.scales - (char*)impl_->weight_base;
            constexpr int NR0=2;
            const int NSG =
                std::min(gemv_nsg_cap(), (K+127)/128);
            ps=impl_->pipeline("gemv_w8_f32a_i8b_f32c");
            [enc setComputePipelineState:ps];
            [enc setBuffer:impl_->weight_buffer
                   offset:scales_boff atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            const NSUInteger tgcount =
                ((NSUInteger)N+NR0-1)/NR0;
            [enc setThreadgroupMemoryLength:
                    NR0*32*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(tgcount,1,1)
                threadsPerThreadgroup:
                    MTLSizeMake(32,(NSUInteger)NSG,1)];
        } else if (weight.prec == Precision::INT4) {
            MatmulW8Params w{};
            w.M=1; w.N=N; w.K=K;
            w.a_offset=0; w.c_offset=0;
            w.a_row_stride=K; w.c_row_stride=N;
            w.activation=activation;
            w.group_size=(int)weight.group_size;
            w.groups_per_row=(int)weight.groups_per_row;
            const size_t scales_boff=(size_t)N*(K/2);
            const int NR0=gemv_w4_nr0(N,K);
            const int NSG=std::min(
                gemv_w4_nsg_cap(), (K/2+63)/64);
            ps=impl_->pipeline_gemv_w4(NR0);
            [enc setComputePipelineState:ps];
            [enc setBuffer:B offset:scales_boff atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            [enc setThreadgroupMemoryLength:
                    (NSUInteger)(NR0*32*sizeof(float)) atIndex:0];
            const NSUInteger rows_per_tg =
                (NSUInteger)NR0*(NSUInteger)std::max(1,NSG);
            const NSUInteger tgcount =
                ((NSUInteger)N+rows_per_tg-1)/rows_per_tg;
            [enc dispatchThreadgroups:MTLSizeMake(tgcount,1,1)
                threadsPerThreadgroup:
                    MTLSizeMake(32*(NSUInteger)std::max(1,NSG),1,1)];
        } else {
            // FP16 uses tuned gemv2 (NR0=2 + NSG-split K).
            constexpr int NR0=2;
            const int NSG=std::min(
                gemv_nsg_cap(), (K+127)/128);
            ps=impl_->pipeline_gemv2(NR0);
            if (ps) {
                [enc setComputePipelineState:ps];
                [enc setThreadgroupMemoryLength:
                        NR0*32*sizeof(float) atIndex:0];
                const NSUInteger tgcount =
                    ((NSUInteger)N+NR0-1)/NR0;
                [enc dispatchThreadgroups:MTLSizeMake(tgcount,1,1)
                    threadsPerThreadgroup:
                        MTLSizeMake(32,(NSUInteger)NSG,1)];
            } else {
            ps = impl_->pipeline("gemv_f32a_f16b_f32c");
            const NSUInteger rows_per_tg = 8;
            [enc setComputePipelineState:ps];
            NSUInteger tgcount = ((NSUInteger)N + rows_per_tg - 1) / rows_per_tg;
            [enc dispatchThreadgroups:MTLSizeMake(tgcount,1,1)
                threadsPerThreadgroup:MTLSizeMake(rows_per_tg * 32, 1, 1)];
            }
        }
        if (finish_open_graph) {
            // end_graph() commits this tail after previously submitted chunks;
            // one queue preserves ordering, and its single wait covers both the
            // graph and lm_head.
            end_graph();
        } else {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        std::memcpy(out_host, MetalBufferPool::contents(cbuf), (size_t)N * 4);
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

bool MetalBackend::has_weight_copies() const {
    return !impl_->weight_copies.empty();
}

bool MetalBackend::configure_moe_ssd_io(const std::string& package_path,
                                        size_t capacity_bytes,
                                        int max_commands_in_flight,
                                        bool cross_layer_prefetch) {
    if (!impl_->ok || package_path.empty() || capacity_bytes == 0)
        return false;
    if (@available(macOS 13.0, *)) {
        NSError* error = nil;
        MTLIOCommandQueueDescriptor* descriptor =
            [[MTLIOCommandQueueDescriptor alloc] init];
        descriptor.type = MTLIOCommandQueueTypeConcurrent;
        descriptor.priority = MTLIOPriorityHigh;
        descriptor.maxCommandsInFlight =
            static_cast<NSUInteger>(std::max(1, max_commands_in_flight));
        descriptor.maxCommandBufferCount =
            static_cast<NSUInteger>(std::max(2, max_commands_in_flight * 2));
        impl_->ssd_io_queue =
            [impl_->device newIOCommandQueueWithDescriptor:descriptor
                                                     error:&error];
        NSURL* url = [NSURL
            fileURLWithPath:[NSString stringWithUTF8String:package_path.c_str()]];
        if (@available(macOS 14.0, *))
            impl_->ssd_file =
                [impl_->device newIOFileHandleWithURL:url error:&error];
        else
            impl_->ssd_file = [impl_->device newIOHandleWithURL:url
                                                           error:&error];
        impl_->ssd_shared_compute_queue = [impl_->device newCommandQueue];
        impl_->ssd_shared_compute_event = [impl_->device newSharedEvent];
        if (!impl_->ssd_io_queue || !impl_->ssd_file ||
            !impl_->ssd_shared_compute_queue || !impl_->ssd_shared_compute_event) {
            fprintf(stderr, "MetalBackend: Metal SSD I/O setup failed: %s\n",
                    error ? error.localizedDescription.UTF8String : "?");
            impl_->ssd_io_queue = nil;
            impl_->ssd_file = nil;
            impl_->ssd_shared_compute_queue = nil;
            impl_->ssd_shared_compute_event = nil;
            return false;
        }
        impl_->ssd_capacity_bytes = capacity_bytes;
        impl_->ssd_cross_layer_prefetch = cross_layer_prefetch;
        impl_->ssd_experts.reserve(
            std::min<size_t>(capacity_bytes / (1024 * 1024), 8192));
        fprintf(stderr,
                "MetalBackend: direct Metal SSD cache enabled "
                "(%.1f MB, queue depth %d)\n",
                capacity_bytes / 1e6, std::max(1, max_commands_in_flight));
        return true;
    }
    fprintf(stderr,
            "MetalBackend: direct Metal SSD cache requires macOS 13 or newer\n");
    return false;
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

void MetalBackend::wrap_weight_int4(Tensor& t, bool keep_native_experts) {
    const bool bg32 =
        t.prec == Precision::INT4 && t.is_q4_g32_packed && t.q4_g32_data;
    const bool bg128 =
        t.prec == Precision::INT4 && t.is_q4_g128_packed && t.q4_g128_data;
    if (!bg32 && !bg128) return;

    // Packed weights interleave scales and nibbles per K block. Ordinary Metal
    // matmul kernels want a simple raw
    // [N,K/2] nibble array + [N,gpr] fp32 scales, so decode once at load time
    // into a dedicated device buffer: [ nibbles (N*K/2) | scales (N*gpr f32) ].
    // The package stores signed int4 in two's-complement nibble form. XOR the
    // sign bit while copying to make the Metal-only buffer offset-binary
    // (q + 8); this lets hot kernels decode with a subtract instead of a
    // per-vector sign-bit XOR. CPU/native packed storage remains unchanged.
    // device_offset stays 0 (nibbles at start); scales live at byte N*(K/2),
    // which the W4 dispatch binds directly (co-located, no weight_base math).
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
    // Expert tensors dominate MoE package size. Keep their native packed blocks
    // zero-copy; the selected-expert tensor kernel decodes blocks while staging.
    // Materializing a second raw-W4 copy here adds ~9GB and causes UMA paging.
    // Aggregate packages serialize experts flattened as [E*N,K], so the
    // loader passes keep_native_experts based on the explicit weight role.
    if (keep_native_experts || last >= 2) {
        wrap_weight(t);
        return;
    }
    const int gpr = (int)t.groups_per_row;
    const size_t nib_bytes = (size_t)N * (K / 2);
    const size_t sc_bytes  = (size_t)N * gpr * sizeof(float);
    const void* packed_data = bg32 ? t.q4_g32_data : t.q4_g128_data;
    auto cached = impl_->decoded_q4_weights.find(packed_data);
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
        constexpr uint64_t sign_bits = 0x8888888888888888ull;
        for (int n = 0; n < N; n++) {
            int nt = (n / 8), c = n % 8;
            uint8_t* nrow = nib + (size_t)n * (K / 2);
            for (int g = 0; g < gpr; g++) {
                if (bg32) {
                    const auto* blocks =
                        reinterpret_cast<const Q4B8G32Block*>(packed_data);
                    const Q4B8G32Block& block =
                        blocks[(size_t)nt * gpr + g];
                    sc[(size_t)n * gpr + g] = block.scales[c];
                    uint8_t* dst =
                        nrow + (size_t)(g * 32) / 2;
                    const uint64_t* src64 =
                        reinterpret_cast<const uint64_t*>(block.q[c]);
                    uint64_t* dst64 = reinterpret_cast<uint64_t*>(dst);
                    dst64[0] = src64[0] ^ sign_bits;
                    dst64[1] = src64[1] ^ sign_bits;
                } else {
                    const auto* blocks =
                        reinterpret_cast<const Q4B8G128Block*>(packed_data);
                    const Q4B8G128Block& block =
                        blocks[(size_t)nt * gpr + g];
                    sc[(size_t)n * gpr + g] = block.scales[c];
                    for (int qgi = 0; qgi < 4; qgi++) {
                        uint8_t* dst =
                            nrow +
                            (size_t)(g * 128 + qgi * 32) / 2;
                        const uint64_t* src64 =
                            reinterpret_cast<const uint64_t*>(
                                block.q[qgi][c]);
                        uint64_t* dst64 =
                            reinterpret_cast<uint64_t*>(dst);
                        dst64[0] = src64[0] ^ sign_bits;
                        dst64[1] = src64[1] ^ sign_bits;
                    }
                }
            }
        }
        t.device_data = (__bridge void*)b;
        t.device_offset = 0;
        impl_->decoded_q4_weights[packed_data] = b;
        // Keep t.scales pointing at the package's CPU layout. Metal binds the
        // co-located decoded scales by byte offset, while hybrid/CPU kernels
        // still need the original packed tensor metadata.
    }
}

void MetalBackend::alloc_persistent(
    Tensor& t, size_t nbytes, PersistentHostAccess host_access,
    size_t host_prefix_bytes) {
    (void)host_access;
    (void)host_prefix_bytes;
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
    impl_->ops_in_cmd = 0;
    impl_->chunk_graph = false;
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
    // Hybrid CPU/Metal operators may synchronize for a host read in the
    // middle of a graph, which intentionally closes the current command
    // buffer. Resume GPU encoding lazily for the next device operation.
    if (!impl_->cmd) {
        impl_->cmd = [impl_->queue commandBuffer];
        impl_->cmd.label = @"mollm graph continuation";
        impl_->enc = [impl_->cmd computeCommandEncoder];
        impl_->enc.label = @"mollm compute continuation";
        impl_->ops_in_cmd = 0;
    } else if (!impl_->enc) {
        impl_->enc = [impl_->cmd computeCommandEncoder];
        impl_->enc.label = @"mollm compute continuation";
    }
    id<MTLComputeCommandEncoder> enc = impl_->enc;
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

    switch (op) {
    // --- view ops: metadata only, alias the input's device buffer ---
    case OpType::INPUT:
    case OpType::CONSTANT:
        encoded_gpu_work = false;
        break;

    case OpType::RESHAPE: {
        const Tensor& src = *inputs[0];
        if (src.is_contiguous()) {
            encoded_gpu_work = false;
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
        encoded_gpu_work = false;
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
        encoded_gpu_work = false;
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
        // Dense inputs are handled as zero-copy aliases by the executor. Only
        // genuinely strided layouts reach this materialization kernel.
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
        auto cast_activation_to_f16 = [&]() -> id<MTLBuffer> {
            const size_t bytes =
                (size_t)p.M * (size_t)p.K * sizeof(uint16_t);
            void* handle = impl_->pool->acquire(bytes);
            id<MTLBuffer> result = (__bridge id<MTLBuffer>)handle;
            impl_->pending_free.push_back({handle, bytes});
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
        if (p.M == 1) impl_->chunk_graph = true;

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
            size_t scales_boff = (char*)B.scales - (char*)impl_->weight_base;
            const int NR0 = 2;
            const int NSG =
                std::min(gemv_nsg_cap(), (p.K + 127) / 128);
            id<MTLComputePipelineState> ps =
                impl_->pipeline("gemv_w8_f32a_i8b_f32c");
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf_of(&A) offset:0 atIndex:0];
            [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBuffer:impl_->weight_buffer offset:scales_boff atIndex:4];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            [enc setThreadgroupMemoryLength:
                (NSUInteger)(NR0 * 32 * sizeof(float)) atIndex:0];
            NSUInteger tgc = ((NSUInteger)p.N + NR0 - 1) / NR0;
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
            [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
            [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
            [enc setBytes:&pv length:sizeof(pv) atIndex:3];
            id<MTLComputePipelineState> gemv2_ps = gemv_old ? nil : impl_->pipeline_gemv2(NR0);
            if (gemv2_ps) {
                [enc setComputePipelineState:gemv2_ps];
                // Eight SIMD groups give the best cross-model decode throughput
                // on M5 Pro. The environment override keeps this tunable for
                // future GPU families.
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
                [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
                [enc setBuffer:buf_of(&C) offset:0 atIndex:2];
                [enc setBuffer:impl_->weight_buffer offset:scales_boff atIndex:4];
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
                    if (impl_->profile) {
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
                           offset:A.device_offset atIndex:0];
                    [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
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
                impl_->pending_free.push_back({a_i8_h, a_i8_bytes});
                impl_->pending_free.push_back({sa_h,   sa_bytes});

                // 1) per-token activation quantization -> int8 [M,K] + scale_a[M].
                {
                    QuantActParams q{};
                    q.M = p.M; q.K = p.K;
                    q.a_offset = A.device_offset / sizeof(float);
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
                        constexpr NSUInteger nsg = 2;
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
                    char* weight_base =
                        (char*)impl_->weight_base;
                    const bool native_bg128 =
                        B.is_q4_g128_packed && native_ptr &&
                        impl_->weight_buffer && weight_base &&
                        native_ptr >= weight_base &&
                        native_ptr < weight_base + impl_->weight_size;
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
                        [enc setBuffer:impl_->weight_buffer
                               offset:(size_t)(native_ptr - weight_base)
                              atIndex:1];
                    } else {
                        [enc setBuffer:buf_of(&B)
                               offset:B.device_offset
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
                NSUInteger activation_offset = A.device_offset;
                if (!direct_weights) {
                    activation_buffer = cast_activation_to_f16();
                    activation_offset = 0;
                    ptt.a_row_stride = p.K;
                }
                [enc setComputePipelineState:ps];
                [enc setBuffer:activation_buffer offset:activation_offset atIndex:0];
                [enc setBuffer:buf_of(&B) offset:B.device_offset atIndex:1];
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
        if (impl_->profile) {
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
        [enc setBuffer:buf_of(&W) offset:W.device_offset atIndex:1];
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
        [enc setBuffer:buf_of(&W) offset:W.device_offset atIndex:1];
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
               offset:query_weight.device_offset atIndex:2];
        [enc setBuffer:buf_of(&key_weight)
               offset:key_weight.device_offset atIndex:3];
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
               offset:weight.device_offset atIndex:4];
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
        p.use_qk_l2norm = params.i32.size()>4 ? params.i32[4] : 1;
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
            impl_->pending_free.push_back({qn_h, qk_bytes});
            impl_->pending_free.push_back({kn_h, qk_bytes});
            impl_->pending_free.push_back({ge_h, gate_bytes});
            impl_->pending_free.push_back({be_h, gate_bytes});
            impl_->pending_free.push_back({raw_h, raw_bytes});

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
        if (impl_->profile) {
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
            if (!impl_->profile) return;
            if (impl_->enc) {
                [impl_->enc endEncoding];
                impl_->enc = nil;
            }
            if (impl_->cmd) {
                [impl_->cmd commit];
                [impl_->cmd waitUntilCompleted];
                const double gpu_ms =
                    (impl_->cmd.GPUEndTime -
                     impl_->cmd.GPUStartTime) * 1000.0;
                auto& stat =
                    impl_->op_stats[std::string(label) + sdpa_profile_suffix];
                stat.gpu_ms += gpu_ms;
                stat.calls += 1;
            }
            impl_->cmd = [impl_->queue commandBuffer];
            impl_->enc = [impl_->cmd computeCommandEncoder];
            enc = impl_->enc;
        };

        int dst_seqlen = past + cur_seqlen;
        // Cache data begins 64 bytes past the buffer base (CacheMetadata header).
        // FP16 cache: element offset = (device_offset + 64) / 2.
        const size_t CACHE_HDR = 64;
        uint k_cache_eoff = (uint)((K_cache ? K_cache->device_offset : 0) + CACHE_HDR) / 2;
        uint v_cache_eoff = (uint)((V_cache ? V_cache->device_offset : 0) + CACHE_HDR) / 2;

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
            impl_->pending_free.push_back({partial_h, partial_bytes});
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
            ? static_cast<const MoeSsdTensorSource*>(inputs[2]->moe_ssd_source)
            : nullptr;
        const auto* ssd_down = inputs.size() > 3
            ? static_cast<const MoeSsdTensorSource*>(inputs[3]->moe_ssd_source)
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
        const bool resident_w4_prefill =
            moe_seq > 1 && !ssd_w4 && !has_shared;
        bool gpu_w4 = impl_->has_tensor && supported_router &&
            inputs[1]->prec == Precision::FP16 && inputs[2]->prec == Precision::INT4 &&
            inputs[3]->prec == Precision::INT4 && top_k <= 16 && n_group <= 16 &&
            (moe_seq == 1 || resident_w4_prefill) &&
            (ssd_w4 || !has_shared);
        if (gpu_w4) {
            const Tensor& x = *inputs[0]; const Tensor& router = *inputs[1];
            const Tensor& gu = *inputs[2]; const Tensor& down = *inputs[3];
            const Tensor* bias = router_bias;
            int seq = (int)x.shape[1];
            if (impl_->profile)
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
                if (!impl_->profile || ssd_w4) return;
                if (impl_->enc) {
                    [impl_->enc endEncoding];
                    impl_->enc = nil;
                }
                if (impl_->cmd) {
                    [impl_->cmd commit];
                    [impl_->cmd waitUntilCompleted];
                    const double gpu_ms =
                        (impl_->cmd.GPUEndTime -
                         impl_->cmd.GPUStartTime) * 1000.0;
                    const std::string stage_label =
                        std::string(label) +
                        "[S=" + std::to_string(seq) + "]";
                    auto& stat = impl_->op_stats[stage_label];
                    stat.gpu_ms += gpu_ms;
                    stat.calls += 1;
                }
                impl_->cmd = [impl_->queue commandBuffer];
                impl_->enc = [impl_->cmd computeCommandEncoder];
                enc = impl_->enc;
            };
            const Impl::SsdMoeLayerInfo* predicted_layer = nullptr;
            void* predicted_idx_h = nullptr;
            void* predicted_tw_h = nullptr;
            size_t predicted_idx_bytes = 0;
            size_t predicted_tw_bytes = 0;
            Impl::SsdSharedExpertWork shared_work;
            if (ssd_w4 && impl_->ssd_cross_layer_prefetch) {
                auto next = impl_->ssd_moe_layers.find(
                    ssd_gate->spec.layer + 1);
                if (next != impl_->ssd_moe_layers.end() &&
                    next->second.hidden == hidden_size &&
                    next->second.router &&
                    next->second.router->device_data &&
                    next->second.router->prec == Precision::FP16 &&
                    next->second.top_k > 0 &&
                    next->second.top_k <= 16 &&
                    (next->second.score_func == 0 ||
                     (next->second.score_func == 1 &&
                      next->second.bias &&
                      next->second.bias->device_data))) {
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
                    x.device_offset;
                Tensor cpu_input = Tensor::create(
                    Precision::FP32, MemoryType::EXTERNAL,
                    hidden_size, seq, 1, 1,
                    const_cast<uint8_t*>(input_bytes));
                if (has_shared &&
                    !impl_->submit_ssd_shared_expert(
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
                        gemv_nsg_cap(),
                        (hidden_size + 127) / 128);
                const bool fuse_router_quant =
                    native_gu && seq == 1 && router_nsg == 8;
                if (fuse_router_quant) {
                    const size_t qx_bytes =
                        (size_t)seq * hidden_size;
                    const size_t sx_bytes =
                        (size_t)seq * gu.groups_per_row *
                        sizeof(float);
                    resident_qx_h = impl_->pool->acquire(qx_bytes);
                    resident_sx_h = impl_->pool->acquire(sx_bytes);
                    [enc setComputePipelineState:
                             impl_->pipeline(
                                 "moe_router_quantize_bg128")];
                    [enc setBuffer:buf_of(&x) offset:0 atIndex:0];
                    [enc setBuffer:buf_of(&router)
                            offset:router.device_offset atIndex:1];
                    [enc setBuffer:logits offset:0 atIndex:2];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                    [enc setBuffer:
                             (__bridge id<MTLBuffer>)resident_qx_h
                            offset:0 atIndex:4];
                    [enc setBuffer:
                             (__bridge id<MTLBuffer>)resident_sx_h
                            offset:0 atIndex:5];
                    [enc setThreadgroupMemoryLength:
                             2 * 32 * sizeof(float) atIndex:0];
                    const NSUInteger router_groups =
                        ((NSUInteger)num_experts + 1) / 2;
                    const NSUInteger quant_groups =
                        ((NSUInteger)gu.groups_per_row + 1) / 2;
                    [enc dispatchThreadgroups:
                             MTLSizeMake(
                                 router_groups + quant_groups, 1, 1)
                        threadsPerThreadgroup:
                             MTLSizeMake(32, 8, 1)];
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
                            offset:router.device_offset atIndex:1];
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

                    const size_t activation_bytes =
                        (size_t)seq * (size_t)hidden_size *
                        sizeof(uint16_t);
                    void* activation_h =
                        impl_->pool->acquire(activation_bytes);
                    id<MTLBuffer> activation =
                        (__bridge id<MTLBuffer>)activation_h;
                    impl_->pending_free.push_back(
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

                    MatmulParams tensor_mp = router_mp;
                    tensor_mp.a_offset = 0;
                    tensor_mp.a_row_stride = hidden_size;
                    const bool small_router =
                        num_experts <= 128;
                    const NSUInteger router_tile_m =
                        small_router ? 64 : 128;
                    const NSUInteger router_tile_n =
                        small_router ? 32 : 64;
                    [enc setComputePipelineState:
                             impl_->pipeline(
                                 small_router
                                     ? "gemm_tensor_router_f16a_f16b_f32c"
                                     : "gemm_tensor_direct_f16a_f16b_f32c")];
                    [enc setBuffer:activation offset:0 atIndex:0];
                    [enc setBuffer:buf_of(&router)
                            offset:router.device_offset atIndex:1];
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
                    fuse_router_quant
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
                            offset:bias->device_offset atIndex:4];
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

#ifdef MOLLM_METAL_TENSOR
            if (ssd_w4) {
                const int* exact_routes =
                    static_cast<const int*>(MetalBufferPool::contents(idx_h));
                std::vector<int> experts(exact_routes,
                                         exact_routes + seq * top_k);
                std::vector<Impl::SsdExpertView> expert_views;
                const uint64_t demand_start = mollm_trace::now_ns();
                if (!impl_->acquire_ssd_experts(
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
                    std::vector<Impl::SsdExpertView> predicted_views;
                    // Demand I/O was submitted first. This speculative command
                    // may execute concurrently with it and overlaps the current
                    // layer's expert compute. Per-batch readiness events keep
                    // cache slots safe without globally serializing MTLIO.
                    const uint64_t prefetch_start =
                        mollm_trace::now_ns();
                    impl_->acquire_ssd_experts(
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

                impl_->cmd = [impl_->queue commandBuffer];
                impl_->cmd.label = @"mollm Metal SSD expert";
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
                            [impl_->cmd
                                encodeWaitForEvent:view.gate_ready_event
                                             value:view.gate_ready_value];
                        }
                    }
                };
                if (ready_selections == 0)
                    encode_gate_waits(0);
                impl_->enc = [impl_->cmd computeCommandEncoder];
                impl_->enc.label = @"mollm Metal SSD expert";
                enc = impl_->enc;

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
                    impl_->enc = nil;
                    encode_gate_waits(ready_selections);
                    impl_->enc =
                        [impl_->cmd computeCommandEncoder];
                    impl_->enc.label =
                        @"mollm pending Metal SSD expert";
                    enc = impl_->enc;
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
                    impl_->enc = nil;
                    for (const auto& view : expert_views) {
                        if (!view.down_ready_event ||
                            view.down_ready_event.signaledValue >=
                                view.down_ready_value)
                            continue;
                        void* event_key =
                            (__bridge void*)view.down_ready_event;
                        if (waited_down_events.insert(event_key).second) {
                            [impl_->cmd
                                encodeWaitForEvent:view.down_ready_event
                                             value:view.down_ready_value];
                        }
                    }
                    impl_->enc =
                        [impl_->cmd computeCommandEncoder];
                    impl_->enc.label =
                        @"mollm Metal SSD down experts";
                    enc = impl_->enc;
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
                    impl_->enc = nil;
                    [impl_->cmd
                        encodeWaitForEvent:impl_->ssd_shared_compute_event
                                   value:shared_ready_value];
                    impl_->enc = [impl_->cmd computeCommandEncoder];
                    impl_->enc.label = @"mollm combine SSD experts";
                    enc = impl_->enc;
                    const uint count = (uint)hidden_size;
                    [enc setComputePipelineState:
                             impl_->pipeline("add_inplace_f32")];
                    [enc setBuffer:buf_of(output)
                            offset:output->device_offset
                           atIndex:0];
                    [enc setBuffer:shared_output offset:0 atIndex:1];
                    [enc setBytes:&count length:sizeof(count) atIndex:3];
                    [enc dispatchThreadgroups:
                             MTLSizeMake((hidden_size + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    impl_->pending_free.push_back(
                        {shared_inter_h, shared_inter_bytes});
                    impl_->pending_free.push_back(
                        {shared_qx_h, qx_bytes});
                    impl_->pending_free.push_back(
                        {shared_sx_h, sx_bytes});
                    impl_->pending_free.push_back(
                        {shared_qinter_h, shared_qinter_bytes});
                    impl_->pending_free.push_back(
                        {shared_qinter_scale_h, sizeof(float)});
                    impl_->pending_free.push_back(
                        {shared_scale_h, sizeof(float)});
                    impl_->pending_free.push_back(
                        {shared_output_h, shared_output_bytes});
                }
                impl_->pending_free.push_back({qx_h, qx_bytes});
                impl_->pending_free.push_back({sx_h, sx_bytes});
                impl_->pending_free.push_back({qi_h, qi_bytes});
                impl_->pending_free.push_back({si_h, si_bytes});
                impl_->pending_free.push_back(
                    {selected_h, selected_bytes});
                impl_->pending_free.push_back(
                    {slot_offsets_h, slot_offsets_bytes});
                impl_->pending_free.push_back(
                    {selection_indices_h, selection_indices_bytes});
                impl_->pending_free.push_back({idx_h, idx_bytes});
                impl_->pending_free.push_back({tw_h, tw_bytes});
                impl_->pending_free.push_back({logits_h, logits_bytes});
                impl_->pending_free.push_back({merged_h, merged_bytes});
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
                    if (impl_->profile) {
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
                                offset:w.device_offset atIndex:1];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                        [enc setBuffer:sa offset:0 atIndex:4];
                        [enc setBuffer:idx offset:0 atIndex:6];
                        constexpr NSUInteger nsg = 4;
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
                                offset:w.device_offset atIndex:1];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                        [enc setBuffer:sa offset:0 atIndex:4];
                        [enc setBuffer:idx offset:0 atIndex:6];
                        constexpr NSUInteger nsg = 2;
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
                                offset:w.device_offset atIndex:1];
                        [enc setBuffer:dst offset:0 atIndex:2];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                        [enc setBuffer:sa offset:0 atIndex:4];
                        [enc setBuffer:buf_of(&w)
                                offset:w.device_offset+rows_total*(K/2)
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
                            offset:weight.device_offset atIndex:1];
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
                impl_->pending_free.push_back({qx_h,qx_bytes});impl_->pending_free.push_back({sx_h,sx_bytes});
                impl_->pending_free.push_back({qi_h,qi_bytes});impl_->pending_free.push_back({si_h,si_bytes});
                impl_->pending_free.push_back(
                    {selected_h,selected_bytes});
                if (grouped_prefill) {
                    impl_->pending_free.push_back(
                        {expert_counts_h, expert_counts_bytes});
                    impl_->pending_free.push_back(
                        {expert_routes_h, expert_routes_bytes});
                    impl_->pending_free.push_back(
                        {grouped_jobs_h, grouped_jobs_bytes});
                    impl_->pending_free.push_back(
                        {grouped_job_count_h,
                         grouped_job_count_bytes});
                    impl_->pending_free.push_back(
                        {grouped_dispatch_h,
                         grouped_dispatch_bytes});
                }
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
        const int chunk_ops = metal_cmd_chunk_ops();
        if (impl_->chunk_graph && chunk_ops > 0 &&
            !getenv("MOLLM_METAL_SYNC_EACH") &&
            !getenv("MOLLM_METAL_GPU_TIME") &&
            encoded_gpu_work && ++impl_->ops_in_cmd >= chunk_ops) {
            // Submit a prefix without waiting. Command buffers from one queue
            // execute in order, so later graph nodes retain their dependencies
            // while CPU encoding overlaps execution of the submitted prefix.
            [impl_->enc endEncoding];
            impl_->enc = nil;
            [impl_->cmd commit];
            impl_->cmd = [impl_->queue commandBuffer];
            impl_->enc = [impl_->cmd computeCommandEncoder];
            impl_->ops_in_cmd = 0;
        }
    }
}
