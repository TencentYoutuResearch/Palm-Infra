#pragma once

#include "storage/ssd_expert_cache/cache.h"
#include "runtime/cuda_host_memory_budget.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <sys/mman.h>

#if defined(__APPLE__)
#include <unistd.h>
#endif

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

// Pool mutation is serialized by the cache mutex, or by exclusive teardown
// after workers stop. Only unpinned, completed entries return buffers here;
// I/O workers access active ByteBuffers.
struct MoeSsdCache::BufferPool {
#if defined(__APPLE__)
    struct Mapping {
        uint8_t* storage = nullptr;
        size_t mapped = 0;
        size_t active = 0;
        size_t locked = 0;
        std::unique_ptr<MoeSsdBufferRegistration> registration;
    };
    size_t capacity = 0;
    size_t bytes = 0;
    uint64_t reuses = 0;
    std::vector<Mapping> mappings;

    explicit BufferPool(size_t limit) : capacity(limit) {
        // put() is called by destructors and must never allocate or throw.
        mappings.reserve(limit / static_cast<size_t>(getpagesize()));
    }
    ~BufferPool() { clear(); }
    void clear() noexcept {
        for (auto& m : mappings) {
            m.registration.reset();
            munmap(m.storage, m.mapped);
        }
        mappings.clear();
        bytes = 0;
    }
    void put(Mapping m) noexcept {
        if (!m.storage) return;
        if (!m.active || m.active > capacity) {
            m.registration.reset();
            munmap(m.storage, m.mapped);
            return;
        }
        while (bytes > capacity - m.active) {
            Mapping old = std::move(mappings.front());
            mappings.erase(mappings.begin());
            bytes -= old.active;
            old.registration.reset();
            munmap(old.storage, old.mapped);
        }
        const size_t active = m.active;
        mappings.push_back(std::move(m));
        bytes += active;
    }
    Mapping take(size_t active) noexcept {
        for (auto it = mappings.begin(); it != mappings.end(); ++it) {
            if (it->active != active) continue;
            Mapping result = std::move(*it);
            bytes -= result.active;
            mappings.erase(it);
            ++reuses;
            return result;
        }
        return {};
    }
#else
    void clear() noexcept {}
#endif
};

// pread() overwrites every byte in a component. std::vector::resize() clears
// these multi-megabyte buffers first, making request_many CPU-bound as the
// cache grows. Avoid clearing the payload in userspace.
#if defined(__APPLE__)
struct MoeSsdCache::ByteBuffer {
    BufferPool* pool = nullptr;
    std::shared_ptr<ResidentArena> arena;
    uint8_t* slice = nullptr;
    uint8_t* storage = nullptr;
    size_t length = 0;
    size_t read_alignment = 0;
    size_t payload_offset = 0;
    // Reserved address space is not committed cache capacity. Only the
    // page-rounded active prefix can hold expert data.
    size_t mapped_bytes = 0;
    size_t active_bytes = 0;
    size_t locked_bytes = 0;
    std::unique_ptr<MoeSsdBufferRegistration> registration;

    ByteBuffer() = default;
    ByteBuffer(const ByteBuffer&) = delete;
    ByteBuffer& operator=(const ByteBuffer&) = delete;
    ~ByteBuffer() { reset(); }

    void reset() {
        if (storage) {
            if (pool) {
                pool->put({storage, mapped_bytes, active_bytes, locked_bytes,
                           std::move(registration)});
            } else {
                registration.reset();
                munmap(storage, mapped_bytes);
            }
        }
        storage = nullptr;
        arena.reset();
        slice = nullptr;
        length = mapped_bytes = active_bytes = locked_bytes = payload_offset = 0;
    }

    void resize(size_t bytes) {
        if (bytes == length && !arena) return;
        if (arena) reset();
        if (bytes == 0) {
            reset();
            return;
        }
        const size_t page = static_cast<size_t>(getpagesize());
        if (bytes > std::numeric_limits<size_t>::max() - read_alignment - (page - 1))
            throw std::bad_alloc();
        const size_t rounded = (bytes + read_alignment + page - 1) / page * page;
        if (pool && rounded != active_bytes) {
            auto reused = pool->take(rounded);
            if (reused.storage) {
                reset();
                registration = std::move(reused.registration);
                storage = reused.storage;
                mapped_bytes = reused.mapped;
                active_bytes = reused.active;
                locked_bytes = reused.locked;
                length = bytes;
                return;
            }
        }
        if (rounded > mapped_bytes) {
            // Leave room for changing quantization formats without moving
            // the prefix. Pages are enabled and faulted only as needed.
            const size_t reserve = rounded <= std::numeric_limits<size_t>::max() / 2
                ? 2 * rounded : rounded;
            void* next = mmap(nullptr, reserve, PROT_NONE,
                              MAP_PRIVATE | MAP_ANON, -1, 0);
            if (next == MAP_FAILED) throw std::bad_alloc();
            if (mprotect(next, rounded, PROT_READ | PROT_WRITE) != 0) {
                munmap(next, reserve);
                throw std::bad_alloc();
            }
            reset();
            storage = static_cast<uint8_t*>(next);
            mapped_bytes = reserve;
        } else if (rounded < active_bytes) {
            registration.reset();
            // Replace only our owned tail mapping, releasing its physical
            // pages while retaining the address range for a later extension.
            if (locked_bytes > rounded) {
                munlock(storage + rounded, locked_bytes - rounded);
                locked_bytes = rounded;
            }
            if (mmap(storage + rounded, active_bytes - rounded, PROT_NONE,
                     MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) == MAP_FAILED)
                throw std::bad_alloc();
        } else if (rounded > active_bytes) {
            registration.reset();
            if (mprotect(storage + active_bytes, rounded - active_bytes,
                         PROT_READ | PROT_WRITE) != 0)
                throw std::bad_alloc();
        }
        length = bytes;
        active_bytes = rounded;
    }

    bool resize_in_arena(size_t bytes, const std::shared_ptr<ResidentArena>& owner) {
        reset();
        arena = owner;
        slice = owner->allocate(bytes);
        length = bytes;
        return bytes == 0 || slice != nullptr;
    }
    bool lock() {
        if (arena) return arena->locked;
        if (locked_bytes == active_bytes) return true;
        if (mlock(storage + locked_bytes, active_bytes - locked_bytes) != 0)
            return false;
        locked_bytes = active_bytes;
        return true;
    }
    void unlock() {
        if (locked_bytes) munlock(storage, locked_bytes);
        locked_bytes = 0;
    }
    bool is_locked() const { return arena ? arena->locked : locked_bytes == active_bytes; }
    bool empty() const { return length == 0; }
    size_t size() const { return length; }
    size_t capacity_bytes() const { return !arena && read_alignment ? active_bytes : length; }
    uint8_t* data() { return arena ? slice : storage ? storage + payload_offset : nullptr; }
    const uint8_t* data() const { return arena ? slice : storage ? storage + payload_offset : nullptr; }
};
#else
struct MoeSsdCache::ByteBuffer {
    struct Free { void operator()(uint8_t* p) const { std::free(p); } };
    std::unique_ptr<uint8_t, Free> storage;
    std::shared_ptr<ResidentArena> arena;
    uint8_t* slice = nullptr;
    size_t length = 0;
    // Nonzero when the transport reads aligned ranges straight from the
    // device. The storage is then aligned and holds the whole blocks that
    // cover the payload, which starts payload_offset bytes in, wherever its
    // first block puts it.
    size_t direct_alignment = 0;
    size_t allocated = 0;
    size_t payload_offset = 0;
    // Bytes that follow the payload in the package and are read with it in
    // one request: a tensor's scales after its data. They belong to a sibling
    // buffer, which is then a view into this storage and owns nothing.
    size_t trailing = 0;
    uint8_t* view = nullptr;
    bool is_view = false;
    bool locked = false;

    ~ByteBuffer() {
        unlock();
    }

    void release() {
        unlock();
        storage.reset();
        arena.reset();
        slice = view = nullptr;
        is_view = false;
        length = allocated = payload_offset = trailing = 0;
    }
    void resize(size_t bytes) { resize(bytes, 0); }
    // Own storage for bytes of payload followed by trailing_bytes of a sibling's.
    void resize(size_t bytes, size_t trailing_bytes) {
        if (bytes == length && trailing_bytes == trailing && !is_view && !arena) return;
        release();
        if (bytes == 0) return;
        if (direct_alignment) {
            if (trailing_bytes > std::numeric_limits<size_t>::max() - bytes ||
                bytes + trailing_bytes > std::numeric_limits<size_t>::max() - 2 * direct_alignment)
                throw std::bad_alloc();
            // The first block may start up to one unit before the payload.
            allocated = (bytes + trailing_bytes + 2 * direct_alignment - 1) /
                direct_alignment * direct_alignment;
            void* aligned = nullptr;
            if (posix_memalign(&aligned, direct_alignment, allocated) != 0) throw std::bad_alloc();
            storage.reset(static_cast<uint8_t*>(aligned));
        } else {
            if (trailing_bytes > std::numeric_limits<size_t>::max() - bytes) throw std::bad_alloc();
            allocated = bytes + trailing_bytes;
            storage.reset(static_cast<uint8_t*>(std::malloc(allocated)));
            if (!storage) throw std::bad_alloc();
        }
        length = bytes;
        trailing = trailing_bytes;
    }
    // Describe bytes that a sibling's storage holds; point() binds the view
    // once the sibling has been read.
    void resize_as_view(size_t bytes) {
        release();
        length = bytes;
        is_view = bytes != 0;
    }
    void point(uint8_t* data) { view = data; }
    bool resize_in_arena(size_t bytes, const std::shared_ptr<ResidentArena>& owner) {
        release();
        arena = owner;
        slice = owner->allocate(bytes);
        length = bytes;
        locked = owner->locked;
        return bytes == 0 || slice != nullptr;
    }
    // True when a read can go straight from the device into this buffer.
    bool direct() const { return direct_alignment && !arena && storage; }
    bool lock() {
        if (locked || length == 0 || is_view) return true;
        locked = arena ? arena->locked : mlock(storage.get(), allocated) == 0;
        return locked;
    }
    void unlock() {
        if (locked) {
            if (!arena && !is_view) munlock(storage.get(), allocated);
            locked = false;
        }
    }
    bool empty() const { return length == 0; }
    size_t size() const { return length; }
    size_t capacity_bytes() const { return length; }
    bool is_locked() const { return locked || is_view; }
    uint8_t* data() {
        return arena ? slice : is_view ? view : storage ? storage.get() + payload_offset : nullptr;
    }
    const uint8_t* data() const {
        return arena ? slice : is_view ? view : storage ? storage.get() + payload_offset : nullptr;
    }
};
#endif

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

    size_t capacity_bytes() const {
        return gate_up_data.capacity_bytes() + gate_up_scales.capacity_bytes() +
               down_data.capacity_bytes() + down_scales.capacity_bytes();
    }

    bool is_loading() const {
        return state == State::Loading || state == State::LoadingFailed;
    }
    bool is_failed() const {
        return state == State::LoadingFailed || state == State::Failed;
    }
    bool is_ready() const { return state == State::Ready; }
};
