#pragma once

#include "runtime/backend.h"
#include <algorithm>
#include <vector>

namespace mollm::detail {

// Correctness-first checkpoint for plain recurrent buffers, not KV caches or
// weights. Optional backend-owned snapshots avoid host staging; unsupported
// collections use explicit transfer hooks (never dereference opaque device
// pointers). Only speculative verification uses this path.
class RecurrentCheckpoint {
public:
    bool capture(Backend& backend, const std::vector<Tensor*>& states) {
        entries_.clear();
        device_snapshot_.reset();
        device_backend_ = nullptr;
        valid_ = false;
        if (backend.supports_state_snapshot(states)) {
            device_snapshot_ = backend.capture_state_snapshot(states);
            if (!device_snapshot_) return false;
            device_backend_ = &backend;
            valid_ = true;
            return true;
        }
        for (Tensor* state : states) {
            if (!state) continue;
            if (std::any_of(entries_.begin(), entries_.end(),
                    [state](const Entry& e) { return e.state == state; }))
                continue;
            const size_t bytes = state->view_span_bytes();
            if (!bytes) continue;
            Entry entry{state, std::vector<uint8_t>(bytes)};
            if (!backend.copy_to_host(*state, entry.bytes.data(), bytes)) {
                entries_.clear();
                return false;
            }
            entries_.push_back(std::move(entry));
        }
        valid_ = true;
        return true;
    }

    bool restore(Backend& backend) const {
        if (!valid_) return false;
        if (device_snapshot_)
            return device_backend_ == &backend && device_snapshot_->restore();
        // Validate all spans before writing any state.
        for (const auto& entry : entries_)
            if (entry.state->view_span_bytes() != entry.bytes.size())
                return false;
        for (const auto& entry : entries_)
            if (!backend.copy_from_host(entry.bytes.data(), *entry.state,
                                        entry.bytes.size()))
                return false;
        return true;
    }

    bool empty() const { return device_snapshot_ ? bytes() == 0 : entries_.empty(); }
    bool device_resident() const { return device_snapshot_ != nullptr; }
    size_t bytes() const {
        if (device_snapshot_) return device_snapshot_->bytes();
        size_t total = 0;
        for (const auto& entry : entries_) total += entry.bytes.size();
        return total;
    }

private:
    struct Entry { Tensor* state; std::vector<uint8_t> bytes; };
    std::vector<Entry> entries_;
    bool valid_ = false;
    std::unique_ptr<BackendStateSnapshot> device_snapshot_;
    Backend* device_backend_ = nullptr;
};

} // namespace mollm::detail
