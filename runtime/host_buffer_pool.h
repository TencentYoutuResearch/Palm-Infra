#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// mollm — BufferPool
//
// Power-of-2 bucketed freelist allocator.
//   - acquire(size) → reuses a free buffer from the matching bucket, or
//     allocates a fresh one.
//   - release(ptr, size) → returns the buffer to the freelist (no free()).
//   - clear() → actually frees all active and free pooled memory.
//   - reset() → alias for clear(); callers should explicitly release reusable
//     temporaries instead of relying on reset to preserve them.
//
// Thread-compatible (caller must serialize if used from multiple threads).
// ---------------------------------------------------------------------------

class BufferPool {
public:
    static constexpr size_t MIN_BUCKET = 1024;   // 1 KB
    static constexpr size_t ALIGNMENT  = 64;      // cache-line alignment

    BufferPool();
    ~BufferPool() { clear(); }

    // Not copyable. Moving transfers all owned allocations and the pool id so
    // existing Tensor::owner_id values remain valid.
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&& other) noexcept;
    BufferPool& operator=(BufferPool&& other) noexcept;

    /// Acquire a buffer of at least `bytes` bytes.
    void* acquire(size_t bytes);

    /// Return a buffer to the pool (does NOT call free).
    void release(void* ptr, size_t bytes);

    /// Free all buffers held in the pool, including active allocations.
    /// After this call the pool is empty and outstanding pointers are invalid.
    void clear();

    /// Clear all pool-owned memory. Kept for compatibility with older call
    /// sites; this does not preserve active buffers for reuse.
    void reset();

    // --- metrics ---
    size_t active_bytes() const { return active_bytes_; }
    size_t peak_bytes()   const { return peak_; }
    size_t acquire_count() const { return acquire_count_; }
    size_t release_count() const { return release_count_; }
#ifdef MOLLM_DISABLE_STORAGE_DEBUG
    uint32_t id() const { return 0; }
#else
    uint32_t id() const { return id_; }
#endif
    uint64_t storage_id(void* ptr) const;

    /// Total bytes held in the freelist (not currently in use).
    size_t pool_bytes() const;

private:
    /// Round up to the next power-of-two bucket, then align to ALIGNMENT.
    static size_t round_up(size_t bytes);

    // bucket_size → list of free buffers
    std::unordered_map<size_t, std::vector<void*>> free_;

    // active pointer → bucket_size. This lets clear() free unreleased buffers
    // and lets release() catch double-release / foreign-pointer mistakes.
#ifndef MOLLM_DISABLE_ACTIVE_TRACKING
    std::unordered_map<void*, size_t> active_allocs_;
#endif
#ifndef MOLLM_DISABLE_STORAGE_DEBUG
    std::unordered_map<void*, uint64_t> storage_ids_;
#endif

    uint32_t id_ = 0;
    size_t active_bytes_ = 0;   // bytes currently handed out via acquire()
    size_t peak_   = 0;        // high-water mark of active_bytes_
    size_t acquire_count_ = 0;
    size_t release_count_ = 0;
};
