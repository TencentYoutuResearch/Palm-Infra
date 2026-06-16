#include "graph/buffer_pool.h"
#include <algorithm>
#include <cstdlib>

// ---------------------------------------------------------------------------
// aligned allocation helper
// ---------------------------------------------------------------------------

namespace {

void* aligned_alloc_bytes(size_t size, size_t alignment) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return ptr;
#endif
}

void aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// round_up
// ---------------------------------------------------------------------------

size_t BufferPool::round_up(size_t bytes) {
    if (bytes < MIN_BUCKET) bytes = MIN_BUCKET;

    // align to ALIGNMENT
    bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

    // next power of two
    size_t p = MIN_BUCKET;
    while (p < bytes) p <<= 1;
    return p;
}

// ---------------------------------------------------------------------------
// acquire
// ---------------------------------------------------------------------------

void* BufferPool::acquire(size_t bytes) {
    size_t bucket = round_up(bytes);

    auto it = free_.find(bucket);
    if (it != free_.end() && !it->second.empty()) {
        // reuse from freelist
        auto& list = it->second;
        void* ptr = list.back();
        list.pop_back();
        active_ += bucket;
        if (active_ > peak_) peak_ = active_;
        return ptr;
    }

    // allocate fresh — must be aligned
    void* buf = aligned_alloc_bytes(bucket, ALIGNMENT);
    if (!buf) return nullptr;
    active_ += bucket;
    if (active_ > peak_) peak_ = active_;
    return buf;
}

// ---------------------------------------------------------------------------
// release
// ---------------------------------------------------------------------------

void BufferPool::release(void* ptr, size_t bytes) {
    if (!ptr) return;

    size_t bucket = round_up(bytes);
    active_ -= bucket;
    free_[bucket].push_back(ptr);
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

void BufferPool::clear() {
    for (auto& [size, list] : free_) {
        for (void* ptr : list) {
            // Safety: check pointer is not null
            if (ptr) aligned_free(ptr);
        }
    }
    free_.clear();
    active_ = 0;
    // peak_ intentionally preserved — it's a high-water mark for the session
}

// ---------------------------------------------------------------------------
// reset (return all active to freelist without freeing — caller must track
//         its own active allocations and call reset() between prefill/decode)
// ---------------------------------------------------------------------------

void BufferPool::reset() {
    // reset() is designed to be called after the caller has manually
    // released all active buffers.  We just clear the freelist here
    // to match the semantics: after reset, everything is available again.
    //
    // The actual "return to freelist" happens per-buffer via release().
    clear();
}

// ---------------------------------------------------------------------------
// pool_bytes
// ---------------------------------------------------------------------------

size_t BufferPool::pool_bytes() const {
    size_t total = 0;
    for (auto& [size, list] : free_) {
        total += size * list.size();
    }
    return total;
}
