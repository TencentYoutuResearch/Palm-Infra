#include "storage/ssd_expert_cache/cache.h"
#include "storage/ssd_expert_cache/internal.h"

#include <cassert>

namespace {
const MoeSsdTensorSource* ssd_source(const ExpertSource* source) {
    return dynamic_cast<const MoeSsdTensorSource*>(source);
}
}  // namespace

bool MoeSsdCache::borrow(const ExpertSource* gate, const ExpertSource* down,
                         int expert, ExpertLease& lease, bool wait) {
    lease.reset();
    Tensor gate_weight, down_weight;
    Entry* entry = nullptr;
    if (!acquire_impl(ssd_source(gate), ssd_source(down), expert,
                      gate_weight, down_weight, &entry, wait))
        return false;
    lease = ExpertLease(*this, entry, gate_weight, down_weight);
    return true;
}

#if defined(__APPLE__)
bool MoeSsdCache::get_or_register_borrowed_buffers(
    const ExpertLease& lease,
    const std::function<std::unique_ptr<MoeSsdBufferRegistration>(void*, size_t)>& factory,
    MoeSsdRegisteredPair& out) {
    out = {};
    if (lease.owner_ != this || !lease.token_ || !factory) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto* entry = static_cast<Entry*>(lease.token_);
    if (!entry->is_ready() || entry->pins == 0 ||
        lease.gate_up.data != entry->gate_up_data.data() ||
        lease.down.data != entry->down_data.data())
        return false;

    auto register_buffer = [&](ByteBuffer& buffer, MoeSsdRegisteredBuffer& view) {
        if (!buffer.storage || !buffer.active_bytes ||
            buffer.payload_offset >= buffer.active_bytes)
            return false;
        if (!buffer.registration)
            buffer.registration = factory(buffer.storage, buffer.active_bytes);
        if (!buffer.registration) return false;
        view = {buffer.registration.get(), buffer.payload_offset, buffer.active_bytes};
        return true;
    };
    MoeSsdRegisteredPair pair;
    if (!register_buffer(entry->gate_up_data, pair.gate_up) ||
        !register_buffer(entry->down_data, pair.down))
        return false;
    out = pair;
    return true;
}
#endif

void MoeSsdCache::release_lease(void* token) noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto* entry = static_cast<Entry*>(token);
        assert(entry && entry->pins > 0);
        --entry->pins;
    }
    ready_cv_.notify_all();
}

bool MoeSsdCache::request_many(const ExpertSource* gate, const ExpertSource* down,
                              const std::vector<int>& experts) {
    return request_many(ssd_source(gate), ssd_source(down), experts);
}

bool MoeSsdCache::retain_for_next_forward(
    const ExpertSource* gate, const ExpertSource* down,
    const std::vector<int>& experts, bool keep) {
    return retain_for_next_forward(ssd_source(gate), ssd_source(down), experts, keep);
}

size_t MoeSsdCache::resident_count(const ExpertSource* gate, const ExpertSource* down,
                                  const std::vector<int>& experts) const {
    return resident_count(ssd_source(gate), ssd_source(down), experts);
}

bool MoeSsdCache::contains(const ExpertSource* gate, const ExpertSource* down,
                          int expert) const {
    return contains(ssd_source(gate), ssd_source(down), expert);
}

bool MoeSsdCache::is_ready(const ExpertSource* gate, const ExpertSource* down,
                          int expert) const {
    return is_ready(ssd_source(gate), ssd_source(down), expert);
}

bool MoeSsdCache::evict(const ExpertSource* gate, const ExpertSource* down,
                       int expert) {
    return release(ssd_source(gate), ssd_source(down), expert);
}
