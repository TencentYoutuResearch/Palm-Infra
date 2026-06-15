#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// PROJECT_NAME — BufferPool
//
// Power-of-2 bucketed freelist allocator.
//   - acquire(size) → reuses a free buffer from the matching bucket, or
//     allocates a fresh one.
//   - release(ptr, size) → returns the buffer to the freelist (no free()).
//   - clear() → actually frees all pooled memory.
//   - reset() → returns all active buffers to the freelist without freeing.
//
// Thread-compatible (caller must serialize if used from multiple threads).
// ---------------------------------------------------------------------------

class BufferPool {
public:
    static constexpr size_t MIN_BUCKET = 1024;   // 1 KB
    static constexpr size_t ALIGNMENT  = 64;      // cache-line alignment

    BufferPool()  = default;
    ~BufferPool() { clear(); }

    // not copyable / movable (owns unique_ptrs)
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&&) = default;
    BufferPool& operator=(BufferPool&&) = default;

    /// Acquire a buffer of at least `bytes` bytes.
    void* acquire(size_t bytes);

    /// Return a buffer to the pool (does NOT call free).
    void release(void* ptr, size_t bytes);

    /// Free all buffers held in the pool.  After this call the pool is empty.
    void clear();

    /// Move all currently-active buffers back to the freelist without freeing
    /// the underlying memory.  Useful when transitioning from prefill to decode:
    /// prefill's large temporary buffers are returned to the pool so decode can
    /// reuse them.
    void reset();

    // --- metrics ---
    size_t active_bytes() const { return active_; }
    size_t peak_bytes()   const { return peak_; }

    /// Total bytes held in the freelist (not currently in use).
    size_t pool_bytes() const;

private:
    /// Round up to the next power-of-two bucket, then align to ALIGNMENT.
    static size_t round_up(size_t bytes);

    // bucket_size → list of free buffers
    std::unordered_map<size_t, std::vector<void*>> free_;

    size_t active_ = 0;   // bytes currently handed out via acquire()
    size_t peak_   = 0;   // high-water mark of active_
};
