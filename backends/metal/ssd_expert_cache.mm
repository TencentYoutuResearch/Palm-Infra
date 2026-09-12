#include "backends/metal/ssd_expert_cache.h"

#include "runtime/trace.h"
#include "storage/mapped_file.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct MetalSsdExpertCache::Impl {
    struct ExpertBuffers {
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

    explicit Impl(id<MTLDevice> device_value) : device(device_value) {}

    id<MTLDevice> device = nil;
    id<MTLIOCommandQueue> ssd_io_queue = nil;
    id<MTLIOFileHandle> ssd_file = nil;
    std::unordered_map<uint64_t, ExpertBuffers> ssd_experts;
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

    static uint64_t ssd_key(int layer, int expert) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
               static_cast<uint32_t>(expert);
    }

    bool acquire(
            const MoeSsdTensorSpec& gate, const MoeSsdTensorSpec& down,
            const std::vector<int>& experts,
            std::vector<MetalSsdExpertCache::ExpertView>& views,
            bool speculative = false) {
        views.clear();
        const bool bg128 =
            gate.precision == Precision::INT4 &&
            down.precision == Precision::INT4 &&
            (gate.flags & MappedFile::FLAG_INT4_BG128) != 0 &&
            (down.flags & MappedFile::FLAG_INT4_BG128) != 0;
        const bool mxfp4 =
            gate.precision == Precision::MXFP4 &&
            down.precision == Precision::MXFP4 &&
            gate.group_size == 32 && down.group_size == 32;
        if (!ssd_io_queue || !ssd_file || gate.layer != down.layer ||
            (!bg128 && !mxfp4))
            return false;

        const uint64_t gate_tensor_bytes =
            gate.data_bytes + gate.scales_bytes;
        const uint64_t down_tensor_bytes =
            down.data_bytes + down.scales_bytes;
        if ((gate.scales_bytes != 0 &&
             gate.scales_file_offset(0) !=
                 gate.data_file_offset(0) + gate.data_bytes) ||
            (down.scales_bytes != 0 &&
             down.scales_file_offset(0) !=
                 down.data_file_offset(0) + down.data_bytes)) {
            return false;
        }

        const size_t pair_bytes =
            static_cast<size_t>(gate_tensor_bytes + down_tensor_bytes);
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
                ExpertBuffers entry;
                entry.slot = ssd_free_slots.back();
                ssd_free_slots.pop_back();
                entry.gate_up = ssd_arena;
                entry.down = ssd_arena;
                entry.gate_ready_event = gate_ready_event;
                entry.down_ready_event = down_ready_event;
                entry.gate_up_offset = entry.slot * ssd_slot_bytes;
                entry.down_offset =
                    entry.gate_up_offset +
                    static_cast<size_t>(gate_tensor_bytes);
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
                           size:gate_tensor_bytes
                   sourceHandle:ssd_file
             sourceHandleOffset:gate.data_file_offset(expert)];
            }
            for (int expert : missing) {
                const auto& entry =
                    ssd_experts.at(ssd_key(gate.layer, expert));
                [down_io
                    loadBuffer:entry.down
                         offset:entry.down_offset
                           size:down_tensor_bytes
                   sourceHandle:ssd_file
             sourceHandleOffset:down.data_file_offset(expert)];
            }
            const uint64_t gate_start = mollm_trace::now_ns();
            const uint64_t down_start = gate_start;
            [gate_io signalEvent:gate_ready_event value:load_ready_value];
            [down_io signalEvent:down_ready_event value:load_ready_value];
            const int layer = gate.layer;
            const size_t expert_count = missing.size();
            const size_t gate_bytes =
                static_cast<size_t>(gate_tensor_bytes) * expert_count;
            const size_t down_bytes =
                static_cast<size_t>(down_tensor_bytes) * expert_count;
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
};

MetalSsdExpertCache::MetalSsdExpertCache(void* device)
    : impl_(new Impl((__bridge id<MTLDevice>)device)) {}

MetalSsdExpertCache::~MetalSsdExpertCache() {
    if (impl_ && impl_->ssd_io_queue) {
        std::fprintf(stderr,
                     "MetalBackend: SSD cache hits=%llu misses=%llu "
                     "read_mb=%.1f resident_mb=%.1f\n",
                     static_cast<unsigned long long>(impl_->ssd_hits),
                     static_cast<unsigned long long>(impl_->ssd_misses),
                     impl_->ssd_bytes_read / 1e6,
                     impl_->ssd_resident_bytes / 1e6);
    }
}

bool MetalSsdExpertCache::configure(const std::string& package_path,
                                    size_t capacity_bytes,
                                    int max_commands_in_flight) {
    if (!impl_ || package_path.empty() || capacity_bytes == 0)
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
        if (!impl_->ssd_io_queue || !impl_->ssd_file) {
            std::fprintf(stderr,
                         "MetalBackend: Metal SSD I/O setup failed: %s\n",
                         error ? error.localizedDescription.UTF8String : "?");
            impl_->ssd_io_queue = nil;
            impl_->ssd_file = nil;
            return false;
        }
        impl_->ssd_capacity_bytes = capacity_bytes;
        impl_->ssd_experts.reserve(
            std::min<size_t>(capacity_bytes / (1024 * 1024), 8192));
        std::fprintf(stderr,
                     "MetalBackend: direct Metal SSD cache enabled "
                     "(%.1f MB, queue depth %d)\n",
                     capacity_bytes / 1e6,
                     std::max(1, max_commands_in_flight));
        return true;
    }
    std::fprintf(stderr,
                 "MetalBackend: direct Metal SSD cache requires macOS 13 or "
                 "newer\n");
    return false;
}

bool MetalSsdExpertCache::available() const {
    return impl_ && impl_->ssd_io_queue && impl_->ssd_file;
}

bool MetalSsdExpertCache::acquire(
        const MoeSsdTensorSpec& gate, const MoeSsdTensorSpec& down,
        const std::vector<int>& experts, std::vector<ExpertView>& views,
        bool speculative) {
    if (!impl_) {
        views.clear();
        return false;
    }
    return impl_->acquire(gate, down, experts, views, speculative);
}
