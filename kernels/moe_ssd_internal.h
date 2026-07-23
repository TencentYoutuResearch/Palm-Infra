#pragma once

#include "kernels/moe_ssd.h"

#include <cstdint>
#include <memory>

// pread() overwrites every byte in a component. std::vector::resize() clears
// these multi-megabyte buffers first, making request_many CPU-bound as the
// cache grows. Allocate them uninitialized instead.
struct MoeSsdCache::ByteBuffer {
    std::unique_ptr<uint8_t[]> storage;
    size_t length = 0;

    void resize(size_t bytes) {
        if (bytes == length) return;
        storage.reset(bytes == 0 ? nullptr : new uint8_t[bytes]);
        length = bytes;
    }
    bool empty() const { return length == 0; }
    size_t size() const { return length; }
    uint8_t* data() { return storage.get(); }
    const uint8_t* data() const { return storage.get(); }
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
    bool fresh_miss = false;  // first acquire after a queued miss is not a hit
    bool speculative = false;
    uint64_t forward_epoch = 0;
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
