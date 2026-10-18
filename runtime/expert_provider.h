#pragma once

#include "core/tensor.h"

#include <utility>
#include <memory>
#include <vector>

class ExpertProvider;

// Stable source identity. The provider owns sources and outlives all leases.
// Storage-specific descriptors belong to the provider, not to compute code.
struct ExpertSource {
    virtual ~ExpertSource() = default;
    ExpertProvider* provider = nullptr;
    int layer = -1;
    // Whole-layer resident borrows may expose CUDA-device addresses. Ordinary
    // borrow() must still return CPU-accessible views for reference fallback.
    bool resident_device_views = false;
    bool resident_nvfp4_mma_layout = false;
    bool resident_nvfp4_gate_codes_kmajor = false;
    bool resident_nvfp4_gate_scales_kmajor = false;
    // Original calibrated activation globals, expert-major/projection-minor.
    // Empty means unavailable, never permission to substitute dynamic amax.
    std::vector<float> nvfp4_input_scales;
    uint32_t nvfp4_input_scales_per_expert = 0;
};

class ExpertLease {
public:
    Tensor gate_up;
    Tensor down;
    // Optional backing for temporary host-accessible copies of device views.
    std::shared_ptr<void> owned_payload;

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
            owned_payload = std::move(other.owned_payload);
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
    // Optional resident capability: pin the complete layer, in expert-ID
    // order, without I/O. Only providers configured for full residency should
    // opt in; a large demand cache must not accidentally become permanently
    // pinned. The provider must outlive these leases, as for borrow().
    virtual bool borrow_resident_layer(const ExpertSource*, const ExpertSource*,
                                       std::vector<ExpertLease>&) { return false; }
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
    owned_payload.reset();
}
