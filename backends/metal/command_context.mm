#include "backends/metal/command_context.h"

#include "backends/metal/buffer_pool.h"

#import <Foundation/Foundation.h>
#import <os/signpost.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

MetalCommandContext::MetalCommandContext(void* queue_handle,
                                         MetalBufferPool* pool)
    : queue((__bridge id<MTLCommandQueue>)queue_handle),
      profile(std::getenv("MOLLM_METAL_PROFILE") != nullptr),
      pool_(pool),
      signpost_log_(os_log_create("com.mollm.metal", "profiling")) {}

MetalCommandContext::~MetalCommandContext() = default;

void MetalCommandContext::begin_graph() {
    cmd = [queue commandBuffer];
    cmd.label = @"mollm graph";
    enc = [cmd computeCommandEncoder];
    enc.label = @"mollm compute";
    ops_in_cmd = 0;
    chunk_graph = false;
    os_signpost_interval_begin(
        signpost_log_, OS_SIGNPOST_ID_EXCLUSIVE, "graph");
}

id<MTLComputeCommandEncoder> MetalCommandContext::ensure_encoder() {
    if (!cmd) {
        cmd = [queue commandBuffer];
        cmd.label = @"mollm graph continuation";
        enc = [cmd computeCommandEncoder];
        enc.label = @"mollm compute continuation";
        ops_in_cmd = 0;
    } else if (!enc) {
        enc = [cmd computeCommandEncoder];
        enc.label = @"mollm compute continuation";
    }
    return enc;
}

void MetalCommandContext::synchronize_for_host_read(bool& dispatch_failed) {
    if (enc) {
        [enc endEncoding];
        enc = nil;
    }
    if (!cmd) return;
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError) {
        NSError* error = cmd.error;
        std::fprintf(stderr, "MetalBackend: host-read sync failed: %s\n",
                     error ? error.localizedDescription.UTF8String : "?");
        dispatch_failed = true;
    }
    cmd = nil;
}

void MetalCommandContext::sync_point(bool& dispatch_failed) {
    if (!std::getenv("MOLLM_METAL_SYNC_EACH")) return;
    if (enc) {
        [enc endEncoding];
        enc = nil;
    }
    if (cmd) {
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status == MTLCommandBufferStatusError) {
            NSError* error = cmd.error;
            std::fprintf(stderr,
                         "MetalBackend: sync-point command buffer error: %s\n",
                         error ? error.localizedDescription.UTF8String : "?");
            dispatch_failed = true;
        }
        cmd = nil;
    }
    cmd = [queue commandBuffer];
    enc = [cmd computeCommandEncoder];
}

void MetalCommandContext::dump_profile() {
    if (!profile || op_stats.empty()) return;
    double total = 0.0;
    for (const auto& item : op_stats) total += item.second.gpu_ms;
    std::fprintf(stderr,
                 "\n=== Metal per-op GPU time (MOLLM_METAL_PROFILE) ===\n");
    std::fprintf(stderr, "%-32s %10s %8s %10s %6s\n",
                 "op", "gpu_ms", "calls", "us/call", "%%");
    std::vector<std::pair<std::string, OpStat>> rows(
        op_stats.begin(), op_stats.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& left, const auto& right) {
                  return left.second.gpu_ms > right.second.gpu_ms;
              });
    for (const auto& row : rows) {
        const double per_call_us = row.second.calls
            ? row.second.gpu_ms * 1000.0 / row.second.calls
            : 0.0;
        std::fprintf(stderr, "%-32s %10.3f %8llu %10.2f %6.1f\n",
                     row.first.c_str(), row.second.gpu_ms,
                     static_cast<unsigned long long>(row.second.calls),
                     per_call_us,
                     total > 0.0 ? 100.0 * row.second.gpu_ms / total : 0.0);
    }
    std::fprintf(stderr, "%-32s %10.3f\n", "TOTAL", total);
    op_stats.clear();
}

void MetalCommandContext::end_graph(bool& dispatch_failed) {
    if (enc) {
        [enc endEncoding];
        enc = nil;
    }
    if (cmd) {
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status == MTLCommandBufferStatusError) {
            NSError* error = cmd.error;
            std::fprintf(stderr, "MetalBackend: command buffer error: %s\n",
                         error ? error.localizedDescription.UTF8String : "?");
            dispatch_failed = true;
        }
        if (std::getenv("MOLLM_METAL_GPU_TIME")) {
            const double gpu_ms =
                (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0;
            gpu_time_ms_ += gpu_ms;
            ++gpu_graphs_;
            std::fprintf(stderr,
                         "[metal] graph GPU time %.3f ms "
                         "(cumulative %.1f ms over %llu graphs)\n",
                         gpu_ms, gpu_time_ms_,
                         static_cast<unsigned long long>(gpu_graphs_));
        }
        cmd = nil;
    }
    for (const auto& pending : pending_free)
        pool_->release(pending.first, pending.second);
    pending_free.clear();
    os_signpost_interval_end(
        signpost_log_, OS_SIGNPOST_ID_EXCLUSIVE, "graph");
}

void MetalCommandContext::release_or_defer(void* buffer, size_t bytes) {
    if (!buffer) return;
    if (cmd)
        pending_free.push_back({buffer, bytes});
    else
        pool_->release(buffer, bytes);
}
