#include "storage/ssd_expert_cache/cache.h"
#include "storage/ssd_expert_cache/internal.h"
#include "storage/mapped_file.h"

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
#if defined(MOLLM_CUDA)
    const auto arena = entry->gate_up_data.arena;
    if (arena && arena->device_only) {
        MoeCudaDeviceScope scope(arena->device_id);
        if (!scope.ok) { lease.reset(); return false; }
        struct HostCopy { std::unique_ptr<uint8_t[]> buffers[4]; };
        auto copy = std::make_shared<HostCopy>();
        for (int component = 0; component < 4; ++component) {
            const auto& buffer = component_buffer(*entry, component);
            if (buffer.empty()) continue;
            copy->buffers[component].reset(new uint8_t[buffer.size()]);
            if (cudaMemcpy(copy->buffers[component].get(), buffer.data(), buffer.size(),
                    cudaMemcpyDeviceToHost) != cudaSuccess) {
                lease.reset(); return false;
            }
        }
        for (int projection = 0; projection < 2; ++projection) {
            const auto* source = projection == 0 ? entry->gate_up : entry->down;
            if (source->resident_nvfp4_gate_codes_kmajor) {
                const auto& spec = source->spec;
                auto* weights = copy->buffers[projection * 2].get();
                const bool pair = (spec.flags & MappedFile::FLAG_NVFP4_Q8_PAIR) != 0;
                const size_t blocks = pair ? 320 : 1280, tile_bytes = pair ? 128 : 32;
                std::vector<uint8_t> original_weights(spec.data_bytes);
                for (size_t q = 0; q < 40; ++q)
                    for (size_t row = 0; row < blocks; ++row)
                        std::memcpy(original_weights.data() + (row * 40 + q) * tile_bytes,
                            weights + (q * blocks + row) * tile_bytes, tile_bytes);
                std::memcpy(weights, original_weights.data(), original_weights.size());
                if(source->resident_nvfp4_gate_scales_kmajor) {
                    auto* scales=copy->buffers[projection*2+1].get();
                    std::vector<uint8_t> original_scales(size_t(spec.rows)*spec.groups_per_row);
                    for(int row=0;row<1280;++row)for(int group=0;group<160;++group)
                        original_scales[size_t(row)*160+group]=
                            scales[(size_t(group/4)*1280+row)*4+group%4];
                    std::memcpy(scales,original_scales.data(),original_scales.size());
                }
                continue; // Row-global scales remain canonical.
            }
            if (!source->resident_nvfp4_mma_layout) continue;
            const auto& spec = source->spec;
            auto* weights = copy->buffers[projection * 2].get();
            auto* scales = copy->buffers[projection * 2 + 1].get();
            std::vector<uint8_t> original_weights(spec.data_bytes);
            const size_t block_bytes = static_cast<size_t>(spec.rows) * spec.groups_per_row;
            std::vector<uint8_t> original_scales(block_bytes);
            const bool pair = (spec.flags & MappedFile::FLAG_NVFP4_Q8_PAIR) != 0;
            for (int row = 0; row < spec.rows; ++row) {
                for (int col = 0; col < spec.cols; col += 8) {
                    const size_t tile = static_cast<size_t>(row / 16) * (spec.cols / 64) + col / 64;
                    const size_t native = (tile * 128 + ((col % 64 / 32) * 2 + row % 16 / 8) * 32
                        + (row % 8) * 4 + col % 32 / 8) * 4;
                    const size_t target = pair
                        ? (static_cast<size_t>(row / 4) * spec.groups_per_row + col / 16) * 32
                            + (row % 4) * 8 + (col % 16) / 2
                        : static_cast<size_t>(row) * (spec.cols / 2) + col / 2;
                    std::memcpy(original_weights.data() + target, weights + native, 4);
                }
                for (unsigned group = 0; group < spec.groups_per_row; ++group) {
                    const size_t tile = static_cast<size_t>(row / 16) * (spec.cols / 64) + group / 4;
                    original_scales[static_cast<size_t>(row) * spec.groups_per_row + group] =
                        scales[tile * 64 + (row % 8) * 8 + (row % 16 / 8) * 4 + group % 4];
                }
            }
            std::memcpy(weights, original_weights.data(), original_weights.size());
            std::memcpy(scales, original_scales.data(), original_scales.size());
        }
        lease.gate_up = make_tensor(*entry->gate_up, copy->buffers[0].get(), copy->buffers[1].get());
        lease.down = make_tensor(*entry->down, copy->buffers[2].get(), copy->buffers[3].get());
        lease.gate_up.nvfp4_mma_layout = false;
        lease.down.nvfp4_mma_layout = false;
        lease.gate_up.nvfp4_gate_codes_kmajor = false;
        lease.down.nvfp4_gate_codes_kmajor = false;
        lease.gate_up.nvfp4_gate_scales_kmajor = false;
        lease.down.nvfp4_gate_scales_kmajor = false;
        lease.owned_payload = std::move(copy);
    }
#endif
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

bool MoeSsdCache::borrow_resident_layer(
    const ExpertSource* gate_source, const ExpertSource* down_source,
    std::vector<ExpertLease>& leases) {
    leases.clear();
    const auto* gate = ssd_source(gate_source);
    const auto* down = ssd_source(down_source);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!fully_resident_ || !valid_pair(gate, down, 0)) return false;
    std::vector<Entry*> entries;
    entries.reserve(gate->spec.num_experts);
    for (int expert = 0; expert < gate->spec.num_experts; ++expert) {
        Entry* entry = find_entry_locked(gate, down, expert);
        if (!entry || entry->is_loading() || entry->is_failed()) return false;
        entries.push_back(entry);
    }
    leases.reserve(entries.size());
    for (Entry* entry : entries) {
        ++entry->pins;
        leases.emplace_back(*this, entry,
            make_tensor(*gate, entry->gate_up_data.data(), entry->gate_up_scales.data()),
            make_tensor(*down, entry->down_data.data(), entry->down_scales.data()));
    }
    return true;
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
