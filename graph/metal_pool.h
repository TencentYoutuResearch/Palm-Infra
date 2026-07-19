#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// mollm — MetalBufferPool
//
// Device-side analogue of BufferPool for the Metal backend. Manages a
// power-of-2 bucketed freelist of id<MTLBuffer> objects (MTLResourceStorage-
// ModeShared, so they are CPU-visible on Apple UMA for boundary readback and
// debug diffing).
//
//   acquire(bytes) -> opaque id<MTLBuffer> handle (as void*), reused from the
//     freelist or freshly allocated.
//   release(buf, bytes) -> returns the buffer to the freelist (does not free).
//   contents(buf) -> the CPU-visible [buf contents] pointer.
//   clear() -> releases all buffers back to Metal.
//
// The header is plain C++ (opaque void* handles) so it can be included by
// non-ObjC translation units. The implementation is in metal_pool.mm.
// ---------------------------------------------------------------------------

class MetalBufferPool {
public:
    static constexpr size_t MIN_BUCKET = 1024;   // 1 KB
    static constexpr size_t ALIGNMENT  = 256;     // Metal buffer alignment

    // device is an id<MTLDevice> passed as void*.
    explicit MetalBufferPool(void* device);
    ~MetalBufferPool();

    MetalBufferPool(const MetalBufferPool&) = delete;
    MetalBufferPool& operator=(const MetalBufferPool&) = delete;

    /// Acquire an id<MTLBuffer> of at least `bytes`. Returns opaque handle.
    void* acquire(size_t bytes);

    /// Return a buffer to the pool (does NOT free the MTLBuffer).
    void release(void* buffer, size_t bytes);

    /// CPU-visible contents pointer of a buffer (Shared storage).
    static void* contents(void* buffer);

    /// Free all pooled MTLBuffers.
    void clear();

    size_t active_bytes() const { return active_bytes_; }
    size_t peak_bytes()   const { return peak_; }

private:
    static size_t round_up(size_t bytes);

    void* device_ = nullptr;  // id<MTLDevice>
    std::unordered_map<size_t, std::vector<void*>> free_;    // bucket -> buffers
    std::unordered_map<void*, size_t> active_;               // buffer -> bucket

    size_t active_bytes_ = 0;
    size_t peak_ = 0;
};
