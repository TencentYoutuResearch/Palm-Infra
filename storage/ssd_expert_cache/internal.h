#pragma once

#include "storage/ssd_expert_cache/cache.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <new>

#include <sys/mman.h>
#if defined(__APPLE__)
#include <unistd.h>
#endif

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
        length = mapped_bytes = active_bytes = locked_bytes = payload_offset = 0;
    }

    void resize(size_t bytes) {
        if (bytes == length) return;
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

    bool lock() {
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
    bool empty() const { return length == 0; }
    size_t size() const { return length; }
    size_t capacity_bytes() const { return read_alignment ? active_bytes : length; }
    uint8_t* data() { return storage ? storage + payload_offset : nullptr; }
    const uint8_t* data() const { return storage ? storage + payload_offset : nullptr; }
};
#else
struct MoeSsdCache::ByteBuffer {
    std::unique_ptr<uint8_t[]> storage;
    size_t length = 0;
    bool locked = false;

    ~ByteBuffer() {
        unlock();
    }

    void resize(size_t bytes) {
        if (bytes == length) return;
        unlock();
        storage.reset(bytes == 0 ? nullptr : new uint8_t[bytes]);
        length = bytes;
    }
    bool lock() {
        if (locked || length == 0) return true;
        locked = mlock(storage.get(), length) == 0;
        return locked;
    }
    void unlock() {
        if (locked) {
            munlock(storage.get(), length);
            locked = false;
        }
    }
    bool empty() const { return length == 0; }
    size_t size() const { return length; }
    size_t capacity_bytes() const { return length; }
    uint8_t* data() { return storage.get(); }
    const uint8_t* data() const { return storage.get(); }
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
