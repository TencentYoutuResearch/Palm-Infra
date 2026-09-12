#pragma once

#include "storage/ssd_expert_cache/cache.h"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Owns the Metal I/O expert cache: package file access, the shared arena,
// readiness events, slot eviction, and cache statistics. Routed-expert
// dispatch consumes ExpertView values without managing storage policy.
class MetalSsdExpertCache {
public:
    struct ExpertView {
        id<MTLBuffer> gate_up = nil;
        id<MTLBuffer> down = nil;
        id<MTLSharedEvent> gate_ready_event = nil;
        id<MTLSharedEvent> down_ready_event = nil;
        size_t gate_up_offset = 0;
        size_t down_offset = 0;
        uint64_t gate_ready_value = 0;
        uint64_t down_ready_value = 0;
    };

    explicit MetalSsdExpertCache(void* device);
    ~MetalSsdExpertCache();

    MetalSsdExpertCache(const MetalSsdExpertCache&) = delete;
    MetalSsdExpertCache& operator=(const MetalSsdExpertCache&) = delete;

    bool configure(const std::string& package_path, size_t capacity_bytes,
                   int max_commands_in_flight);
    bool available() const;

    bool acquire(const MoeSsdTensorSpec& gate,
                 const MoeSsdTensorSpec& down,
                 const std::vector<int>& experts,
                 std::vector<ExpertView>& views,
                 bool speculative = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
