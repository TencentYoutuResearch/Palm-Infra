#pragma once

#import <Metal/Metal.h>
#import <os/log.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

class MetalBufferPool;

// Owns the active graph command stream and its profiling/deferred-release
// state. Dispatch code may access cmd/enc while encoding complex multi-stage
// operators; graph lifecycle and error handling stay centralized here.
class MetalCommandContext {
public:
    struct OpStat {
        double gpu_ms = 0.0;
        uint64_t calls = 0;
    };

    MetalCommandContext(void* queue, MetalBufferPool* pool);
    ~MetalCommandContext();

    MetalCommandContext(const MetalCommandContext&) = delete;
    MetalCommandContext& operator=(const MetalCommandContext&) = delete;

    void begin_graph();
    id<MTLComputeCommandEncoder> ensure_encoder();
    void synchronize_for_host_read(bool& dispatch_failed);
    void sync_point(bool& dispatch_failed);
    void end_graph(bool& dispatch_failed);
    void dump_profile();
    void release_or_defer(void* buffer, size_t bytes);

    id<MTLCommandQueue> queue = nil;
    id<MTLCommandBuffer> cmd = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    int ops_in_cmd = 0;
    bool chunk_graph = false;
    bool profile = false;
    std::map<std::string, OpStat> op_stats;
    std::vector<std::pair<void*, size_t>> pending_free;

private:
    MetalBufferPool* pool_ = nullptr;
    double gpu_time_ms_ = 0.0;
    uint64_t gpu_graphs_ = 0;
    os_log_t signpost_log_ = nullptr;
};
