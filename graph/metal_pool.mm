#include "graph/metal_pool.h"

#import <Metal/Metal.h>

// ---------------------------------------------------------------------------
// MetalBufferPool implementation (Objective-C++).
// Buffers are retained in ARC-off manual style: we hold them in C++ containers
// as __bridge_retained void* and balance with __bridge_transfer on free.
// ---------------------------------------------------------------------------

// This TU is compiled WITH ARC (-fobjc-arc); bridge casts manage ownership.

MetalBufferPool::MetalBufferPool(void* device) {
    // Retain the device so it outlives the pool.
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    device_ = (__bridge_retained void*)dev;
}

MetalBufferPool::~MetalBufferPool() {
    clear();
    if (device_) {
        id<MTLDevice> dev = (__bridge_transfer id<MTLDevice>)device_;
        (void)dev;  // released at end of scope
        device_ = nullptr;
    }
}

size_t MetalBufferPool::round_up(size_t bytes) {
    if (bytes < MIN_BUCKET) bytes = MIN_BUCKET;
    bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    size_t p = MIN_BUCKET;
    while (p < bytes) p <<= 1;
    return p;
}

void* MetalBufferPool::acquire(size_t bytes) {
    size_t bucket = round_up(bytes);

    auto it = free_.find(bucket);
    if (it != free_.end() && !it->second.empty()) {
        void* buf = it->second.back();
        it->second.pop_back();
        active_[buf] = bucket;
        active_bytes_ += bucket;
        if (active_bytes_ > peak_) peak_ = active_bytes_;
        return buf;
    }

    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    id<MTLBuffer> mtl = [dev newBufferWithLength:bucket
                                         options:MTLResourceStorageModeShared];
    if (!mtl) return nullptr;
    void* buf = (__bridge_retained void*)mtl;
    active_[buf] = bucket;
    active_bytes_ += bucket;
    if (active_bytes_ > peak_) peak_ = active_bytes_;
    return buf;
}

void MetalBufferPool::release(void* buffer, size_t /*bytes*/) {
    if (!buffer)
        return;
    auto ai = active_.find(buffer);
    if (ai == active_.end())
        return;

    const size_t bucket = ai->second;
    active_bytes_ -= bucket;
    active_.erase(ai);
    free_[bucket].push_back(buffer);
}

void* MetalBufferPool::contents(void* buffer) {
    if (!buffer) return nullptr;
    id<MTLBuffer> mtl = (__bridge id<MTLBuffer>)buffer;
    return [mtl contents];
}

void MetalBufferPool::clear() {
    auto free_one = [](void* b) {
        if (!b) return;
        id<MTLBuffer> mtl = (__bridge_transfer id<MTLBuffer>)b;
        (void)mtl;  // released
    };
    for (auto& kv : free_) {
        for (void* b : kv.second) free_one(b);
    }
    free_.clear();
    for (auto& kv : active_) free_one(kv.first);
    active_.clear();
    active_bytes_ = 0;
}
