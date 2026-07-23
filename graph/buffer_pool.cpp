#include "graph/buffer_pool.h"
#include <algorithm>
#include <cassert>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <utility>

// ---------------------------------------------------------------------------
// aligned allocation helper
// ---------------------------------------------------------------------------

namespace {

std::atomic<uint32_t> g_next_pool_id{1};
#ifndef MOLLM_DISABLE_STORAGE_DEBUG
std::atomic<uint64_t> g_next_storage_id{1};
#endif

uint32_t next_pool_id() {
    return g_next_pool_id.fetch_add(1, std::memory_order_relaxed);
}

#ifndef MOLLM_DISABLE_STORAGE_DEBUG
uint64_t next_storage_id() {
    return g_next_storage_id.fetch_add(1, std::memory_order_relaxed);
}
#endif

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

BufferPool::BufferPool() : id_(next_pool_id()) {}

BufferPool::BufferPool(BufferPool&& other) noexcept
    : free_(std::move(other.free_))
#ifndef MOLLM_DISABLE_ACTIVE_TRACKING
      , active_allocs_(std::move(other.active_allocs_))
#endif
#ifndef MOLLM_DISABLE_STORAGE_DEBUG
      , storage_ids_(std::move(other.storage_ids_))
#endif
      , id_(other.id_), active_bytes_(other.active_bytes_),
      peak_(other.peak_), acquire_count_(other.acquire_count_),
      release_count_(other.release_count_) {
    other.free_.clear();
#ifndef MOLLM_DISABLE_ACTIVE_TRACKING
    other.active_allocs_.clear();
#endif
#ifndef MOLLM_DISABLE_STORAGE_DEBUG
    other.storage_ids_.clear();
#endif
    other.id_ = next_pool_id();
    other.active_bytes_ = 0;
    other.peak_ = 0;
    other.acquire_count_ = 0;
    other.release_count_ = 0;
}

BufferPool& BufferPool::operator=(BufferPool&& other) noexcept {
    if (this == &other)
        return *this;

    clear();
    free_ = std::move(other.free_);
#ifndef MOLLM_DISABLE_ACTIVE_TRACKING
    active_allocs_ = std::move(other.active_allocs_);
#endif
#ifndef MOLLM_DISABLE_STORAGE_DEBUG
    storage_ids_ = std::move(other.storage_ids_);
#endif
    id_ = other.id_;
    active_bytes_ = other.active_bytes_;
    peak_ = other.peak_;
    acquire_count_ = other.acquire_count_;
    release_count_ = other.release_count_;

    other.free_.clear();
#ifndef MOLLM_DISABLE_ACTIVE_TRACKING
    other.active_allocs_.clear();
#endif
#ifndef MOLLM_DISABLE_STORAGE_DEBUG
    other.storage_ids_.clear();
#endif
    other.id_ = next_pool_id();
    other.active_bytes_ = 0;
    other.peak_ = 0;
    other.acquire_count_ = 0;
    other.release_count_ = 0;
    return *this;
}

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
    acquire_count_++;

    auto it = free_.find(bucket);
    if (it != free_.end() && !it->second.empty()) {
        // reuse from freelist
        auto& list = it->second;
        void* ptr = list.back();
        list.pop_back();
#ifndef MOLLM_DISABLE_ACTIVE_TRACKING
        active_allocs_[ptr] = bucket;
#endif
        active_bytes_ += bucket;
        if (active_bytes_ > peak_) peak_ = active_bytes_;
        return ptr;
    }

    // allocate fresh — must be aligned
    void* buf = aligned_alloc_bytes(bucket, ALIGNMENT);
    if (!buf) return nullptr;
#ifndef MOLLM_DISABLE_ACTIVE_TRACKING
    active_allocs_[buf] = bucket;
#endif
#ifndef MOLLM_DISABLE_STORAGE_DEBUG
    storage_ids_[buf] = next_storage_id();
#endif
    active_bytes_ += bucket;
    if (active_bytes_ > peak_) peak_ = active_bytes_;
    return buf;
}

// ---------------------------------------------------------------------------
// release
// ---------------------------------------------------------------------------

void BufferPool::release(void* ptr, size_t bytes) {
    if (!ptr) return;

    size_t bucket = round_up(bytes);
#ifndef MOLLM_DISABLE_ACTIVE_TRACKING
    auto it = active_allocs_.find(ptr);
    if (it == active_allocs_.end()) {
        std::fprintf(stderr, "BufferPool::release: foreign or double-released pointer %p\n", ptr);
        assert(false && "BufferPool::release foreign/double pointer");
        return;
    }
    if (it->second != bucket) {
        std::fprintf(stderr,
                     "BufferPool::release: bucket mismatch for %p (got %zu, expected %zu)\n",
                     ptr, bucket, it->second);
        assert(false && "BufferPool::release size/bucket mismatch");
        return;
    }

    release_count_++;
    active_allocs_.erase(it);
#else
    release_count_++;
#endif
    active_bytes_ -= bucket;
    free_[bucket].push_back(ptr);
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

void BufferPool::clear() {
#ifndef MOLLM_DISABLE_ACTIVE_TRACKING
    for (auto& [ptr, bucket] : active_allocs_) {
        (void)bucket;
        if (ptr) aligned_free(ptr);
    }
    active_allocs_.clear();
#endif

    for (auto& [size, list] : free_) {
        (void)size;
        for (void* ptr : list) {
            // Safety: check pointer is not null
            if (ptr) aligned_free(ptr);
        }
    }
    free_.clear();
#ifndef MOLLM_DISABLE_STORAGE_DEBUG
    storage_ids_.clear();
#endif
    active_bytes_ = 0;
    // peak_ intentionally preserved — it's a high-water mark for the session
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void BufferPool::reset() {
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

uint64_t BufferPool::storage_id(void* ptr) const {
#ifdef MOLLM_DISABLE_STORAGE_DEBUG
    (void)ptr;
    return 0;
#else
    auto it = storage_ids_.find(ptr);
    return it == storage_ids_.end() ? 0 : it->second;
#endif
}
