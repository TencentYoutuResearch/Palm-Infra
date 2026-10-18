#pragma once

#include "storage/ssd_expert_cache/cache.h"
#include "runtime/cuda_host_memory_budget.h"

#include <cstdint>
#include <memory>
#include <limits>

#include <sys/mman.h>
#if defined(MOLLM_CUDA)
#include <cuda_runtime_api.h>
struct MoeCudaDeviceScope {
    int previous = -1;
    bool changed = false, ok = false;
    explicit MoeCudaDeviceScope(int device) {
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        changed = previous != device;
        ok = !changed || cudaSetDevice(device) == cudaSuccess;
    }
    ~MoeCudaDeviceScope() { if (changed && ok) cudaSetDevice(previous); }
};
#endif

// pread() overwrites every byte in a component. std::vector::resize() clears
// these multi-megabyte buffers first, making request_many CPU-bound as the
// cache grows. Allocate them uninitialized instead.
struct MoeSsdCache::ResidentArena {
    uint8_t* base = nullptr;
    size_t capacity = 0, used = 0;
    bool locked = false;
    bool device_only = false;
    int device_id = -1;
    ~ResidentArena() {
#if defined(MOLLM_CUDA)
        if (device_only) {
            MoeCudaDeviceScope scope(device_id);
            if (base && scope.ok) cudaFree(base);
            return;
        }
#endif
        if (locked) munlock(base, capacity);
        if (base) munmap(base, capacity);
    }
    bool create(size_t bytes, bool use_device = false) {
        constexpr size_t huge = 2 * 1024 * 1024;
        if (bytes > std::numeric_limits<size_t>::max() - 2 * huge) return false;
        capacity = (bytes + huge - 1) & ~(huge - 1);
        if (use_device) {
#if defined(MOLLM_CUDA)
            void* allocation = nullptr;
            if (!mollm_memory::cuda_host_allocation_allowed(capacity, "resident expert arena")) return false;
            if (cudaGetDevice(&device_id) != cudaSuccess ||
                cudaMalloc(&allocation, capacity) != cudaSuccess) return false;
            base = static_cast<uint8_t*>(allocation);
            device_only = true;
            locked = true; // Driver-owned device residency, not a CPU mlock.
            return true;
#else
            return false;
#endif
        }
        void* mapping = mmap(nullptr, capacity + huge, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) { capacity = 0; return false; }
        const uintptr_t raw = reinterpret_cast<uintptr_t>(mapping);
        const uintptr_t aligned = (raw + huge - 1) & ~(uintptr_t(huge) - 1);
        const size_t prefix = aligned - raw;
        if (prefix) munmap(mapping, prefix);
        if (huge > prefix) munmap(reinterpret_cast<void*>(aligned + capacity), huge - prefix);
        base = reinterpret_cast<uint8_t*>(aligned);
#ifdef MADV_HUGEPAGE
        madvise(base, capacity, MADV_HUGEPAGE);
#endif
        // Lock the whole arena before slicing; partial-component mlock calls
        // would split its VMAs and undermine large-page eligibility.
        locked = mlock(base, capacity) == 0;
        return locked;
    }
    uint8_t* allocate(size_t bytes) {
        if (bytes == 0) return nullptr;
        const size_t padded = (bytes + 63) & ~size_t(63);
        if (padded < bytes || padded > capacity - used) return nullptr;
        uint8_t* result = base + used;
        used += padded;
        return result;
    }
};

struct MoeSsdCache::ByteBuffer {
    std::unique_ptr<uint8_t[]> storage;
    std::shared_ptr<ResidentArena> arena;
    uint8_t* slice = nullptr;
    size_t length = 0;
    bool locked = false;

    ~ByteBuffer() {
        unlock();
    }

    void resize(size_t bytes) {
        if (bytes == length) return;
        unlock();
        arena.reset();
        slice = nullptr;
        storage.reset(bytes == 0 ? nullptr : new uint8_t[bytes]);
        length = bytes;
    }
    bool resize_in_arena(size_t bytes, const std::shared_ptr<ResidentArena>& owner) {
        unlock();
        storage.reset();
        arena = owner;
        slice = owner->allocate(bytes);
        length = bytes;
        locked = owner->locked;
        return bytes == 0 || slice != nullptr;
    }
    bool lock() {
        if (locked || length == 0) return true;
        locked = arena ? arena->locked : mlock(storage.get(), length) == 0;
        return locked;
    }
    void unlock() {
        if (locked) {
            if (!arena) munlock(storage.get(), length);
            locked = false;
        }
    }
    bool empty() const { return length == 0; }
    size_t size() const { return length; }
    uint8_t* data() { return arena ? slice : storage.get(); }
    const uint8_t* data() const { return arena ? slice : storage.get(); }
};

struct MoeSsdCache::Entry {
    enum class State {
        Loading,
        LoadingFailed,
        Ready,
        Failed,
    };

    const MoeSsdTensorSource* gate_up = nullptr;
    const MoeSsdTensorSource* down = nullptr;
    int expert = -1;
    uint64_t used_at = 0;
    State state = State::Loading;
    int pending_reads = 0;
    size_t pins = 0;  // Active ExpertLeases; pinned entries cannot be evicted.
    bool fresh_miss = false;  // first acquire after a queued miss is not a hit
    bool speculative = false;
    // Immutable for one residency lifetime. Unlike speculative, this remains
    // true after a demand promotes or consumes an in-flight prefetch.
    bool load_origin_speculative = false;
    bool retained = false;
    uint64_t forward_epoch = 0;
    uint64_t prediction_epoch = 0;
    float prediction_confidence = 0.0f;
    ByteBuffer gate_up_data;
    ByteBuffer gate_up_scales;
    ByteBuffer down_data;
    ByteBuffer down_scales;

    size_t bytes() const {
        return gate_up_data.size() + gate_up_scales.size() +
               down_data.size() + down_scales.size();
    }

    bool is_loading() const {
        return state == State::Loading || state == State::LoadingFailed;
    }
    bool is_failed() const {
        return state == State::LoadingFailed || state == State::Failed;
    }
    bool is_ready() const { return state == State::Ready; }
};
