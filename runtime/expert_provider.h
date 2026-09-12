#pragma once

#include "kernels/tensor.h"

#include <utility>
#include <vector>

class ExpertProvider;

// Stable source identity. The provider owns sources and outlives all leases.
// Storage-specific descriptors belong to the provider, not to compute code.
struct ExpertSource {
    virtual ~ExpertSource() = default;
    ExpertProvider* provider = nullptr;
    int layer = -1;
};

class ExpertLease {
public:
    Tensor gate_up;
    Tensor down;

    ExpertLease() = default;
    ExpertLease(ExpertProvider& owner, void* token, const Tensor& gate,
                const Tensor& down_weight)
        : gate_up(gate), down(down_weight), owner_(&owner), token_(token) {}
    ~ExpertLease() { reset(); }
    ExpertLease(const ExpertLease&) = delete;
    ExpertLease& operator=(const ExpertLease&) = delete;
    ExpertLease(ExpertLease&& other) noexcept { *this = std::move(other); }
    ExpertLease& operator=(ExpertLease&& other) noexcept {
        if (this != &other) {
            reset();
            gate_up = other.gate_up;
            down = other.down;
            owner_ = std::exchange(other.owner_, nullptr);
            token_ = std::exchange(other.token_, nullptr);
            other.gate_up = {};
            other.down = {};
        }
        return *this;
    }
    explicit operator bool() const { return owner_ != nullptr; }
    void reset() noexcept;

private:
    ExpertProvider* owner_ = nullptr;
    void* token_ = nullptr;
};

class ExpertProvider {
public:
    virtual ~ExpertProvider() = default;
    // Replaces any existing lease. A successful borrow pins both weight views.
    // wait=false only borrows an already-ready pair and never queues/waits for I/O.
    virtual bool borrow(const ExpertSource*, const ExpertSource*, int expert,
                        ExpertLease&, bool wait = true) = 0;
    virtual bool request_many(const ExpertSource*, const ExpertSource*,
                              const std::vector<int>&) = 0;
    virtual bool retain_for_next_forward(const ExpertSource*, const ExpertSource*,
                                         const std::vector<int>&, bool) = 0;
    virtual size_t resident_count(const ExpertSource*, const ExpertSource*,
                                  const std::vector<int>&) const = 0;
    virtual bool contains(const ExpertSource*, const ExpertSource*, int) const = 0;
    virtual bool is_ready(const ExpertSource*, const ExpertSource*, int) const = 0;
    // Eviction is distinct from ending a borrow: unborrowed weights may stay cached.
    virtual bool evict(const ExpertSource*, const ExpertSource*, int) = 0;

protected:
    friend class ExpertLease;
    virtual void release_lease(void* token) noexcept = 0;
};

inline void ExpertLease::reset() noexcept {
    if (owner_) owner_->release_lease(token_);
    owner_ = nullptr;
    token_ = nullptr;
    gate_up = {};
    down = {};
}
