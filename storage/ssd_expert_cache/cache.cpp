#include "storage/ssd_expert_cache/cache.h"
#include "storage/ssd_expert_cache/internal.h"

#include "storage/mapped_file.h"
#include "runtime/trace.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

bool component_range_fits(uint64_t offset, uint64_t bytes_per_expert,
                          uint64_t expert_stride, int num_experts,
                          uint64_t file_size) {
    if (bytes_per_expert == 0)
        return true;
    const uint64_t count = static_cast<uint64_t>(num_experts);
    const uint64_t stride = expert_stride != 0
                                ? expert_stride : bytes_per_expert;
    if (count == 0 || (count - 1) >
                          std::numeric_limits<uint64_t>::max() / stride)
        return false;
    const uint64_t last = (count - 1) * stride;
    return offset <= file_size && last <= file_size - offset &&
           bytes_per_expert <= file_size - offset - last;
}

bool source_bytes(const MoeSsdTensorSpec& spec, size_t& bytes) {
    if (spec.data_bytes > std::numeric_limits<size_t>::max() ||
        spec.scales_bytes > std::numeric_limits<size_t>::max()) {
        return false;
    }
    const size_t data = static_cast<size_t>(spec.data_bytes);
    const size_t scales = static_cast<size_t>(spec.scales_bytes);
    if (scales > std::numeric_limits<size_t>::max() - data)
        return false;
    bytes = data + scales;
    return true;
}

bool components_fit_expert_stride(const MoeSsdTensorSpec& spec) {
    if (spec.expert_stride == 0)
        return true;
    if (spec.data_bytes > spec.expert_stride ||
        spec.scales_bytes > spec.expert_stride)
        return false;
    if (spec.scales_bytes == 0)
        return true;
    if (spec.scales_offset >= spec.data_offset) {
        const uint64_t relative = spec.scales_offset - spec.data_offset;
        return relative >= spec.data_bytes &&
               relative <= spec.expert_stride - spec.scales_bytes;
    }
    const uint64_t relative = spec.data_offset - spec.scales_offset;
    return relative >= spec.scales_bytes &&
           relative <= spec.expert_stride - spec.data_bytes;
}

bool expert_pair_bytes(const MoeSsdTensorSource* gate_up,
                       const MoeSsdTensorSource* down, size_t& bytes) {
    size_t gate_up_bytes = 0;
    size_t down_bytes = 0;
    if (!source_bytes(gate_up->spec, gate_up_bytes) ||
        !source_bytes(down->spec, down_bytes) ||
        down_bytes > std::numeric_limits<size_t>::max() - gate_up_bytes) {
        return false;
    }
    bytes = gate_up_bytes + down_bytes;
    return true;
}

size_t saturating_multiply(size_t value, size_t count) {
    if (value != 0 && count > std::numeric_limits<size_t>::max() / value)
        return std::numeric_limits<size_t>::max();
    return value * count;
}

}  // namespace

MoeSsdCache::MoeSsdCache() = default;

MoeSsdCache::~MoeSsdCache() {
    stop_io_workers();
    if (fd_ >= 0) close(fd_);
}

bool MoeSsdCache::clear_resident() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!io_jobs_.empty() || !low_priority_io_jobs_.empty()) return false;
    for (const auto& entry : entries_) {
        if (entry->is_loading() || entry->pins != 0) return false;
    }
    entries_.clear();
    entry_locations_.clear();
    layer_entries_.clear();
    layer_resident_bytes_.clear();
    layer_route_widths_.clear();
    retained_experts_.clear();
    retained_scores_.clear();
    pending_predictions_.clear();
    resident_bytes_ = 0;
    active_layer_ = -1;
    return true;
}

void MoeSsdCache::stop_io_workers() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_io_ = true;
    }
    io_cv_.notify_all();
    cross_layer_cv_.notify_all();
    for (std::thread& worker : io_workers_) {
        if (worker.joinable()) worker.join();
    }
    io_workers_.clear();
    if (cross_layer_worker_.joinable()) cross_layer_worker_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    io_jobs_.clear();
    low_priority_io_jobs_.clear();
    cross_layer_tasks_.clear();
    stop_io_ = false;
}

bool MoeSsdCache::open(const std::string& package_path, size_t capacity_bytes,
                       int io_workers, bool enable_cross_layer_worker,
                       bool lock_expert_pages) {
    // Like initial registration, reopening requires the caller to stop new
    // requests. Do not invalidate any already-issued expert leases.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& entry : entries_)
            if (entry->pins != 0) return false;
    }
    if (capacity_bytes == 0 || io_workers < 1) {
        std::fprintf(stderr, "MoE SSD: cache capacity and I/O worker count must be non-zero\n");
        return false;
    }
    int fd = ::open(package_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "MoE SSD: failed to open %s: %s\n",
                     package_path.c_str(), std::strerror(errno));
        return false;
    }
    struct stat file_stat {};
    if (fstat(fd, &file_stat) != 0 || file_stat.st_size < 0) {
        std::fprintf(stderr, "MoE SSD: failed to stat %s: %s\n",
                     package_path.c_str(), std::strerror(errno));
        close(fd);
        return false;
    }
#if defined(__APPLE__)
    // The application cache below is the residency policy for routed
    // experts.  Letting pread() also populate the kernel file cache would
    // silently retain much more than --ssd-cache-mb and make a warm process
    // look like a faster SSD.  This only applies to the dedicated SSD fd;
    // regular (non-expert) package mmap weights retain their normal policy.
    if (fcntl(fd, F_NOCACHE, 1) != 0) {
        std::fprintf(stderr, "MoE SSD: warning: could not disable macOS file cache: %s\n",
                     std::strerror(errno));
    }
#endif
    stop_io_workers();
    if (fd_ >= 0) close(fd_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fd_ = fd;
        file_size_ = static_cast<uint64_t>(file_stat.st_size);
        io_workers_count_ = io_workers;
        next_trace_id_ = 1;
        capacity_bytes_ = capacity_bytes;
        lock_expert_pages_ = lock_expert_pages;
        resident_bytes_ = 0;
        clock_ = hits_ = misses_ = evictions_ = bytes_read_ = 0;
        demand_load_bytes_ = prefetch_load_bytes_ = 0;
        useful_prefetch_bytes_ = unused_prefetch_bytes_ = 0;
        expert_bytes_acquired_ = 0;
        slot_waits_ = slot_wait_ns_ = 0;
        cross_layer_tasks_count_ = cross_layer_dropped_ = 0;
        cross_layer_experts_ = cross_layer_used_ = cross_layer_rejected_ = 0;
        cross_layer_rank_attempts_.clear();
        cross_layer_rank_hits_.clear();
        cross_layer_rank_confidence_sum_.clear();
        prediction_policy_attempts_.clear();
        prediction_policy_hits_.clear();
        sources_.clear();
        layers_.clear();
        layer_layouts_.clear();
        layer_capacity_bytes_.clear();
        entries_.clear();
        entry_locations_.clear();
        layer_entries_.clear();
        layer_resident_bytes_.clear();
        layer_route_widths_.clear();
        retained_experts_.clear();
        retained_scores_.clear();
        pending_predictions_.clear();
        layer_stats_.clear();
        last_evicted_epoch_.clear();
    }
    io_workers_.reserve((size_t)io_workers);
    for (int i = 0; i < io_workers; i++) {
        io_workers_.emplace_back(&MoeSsdCache::io_worker_main, this, i);
    }
    if (enable_cross_layer_worker) {
        cross_layer_worker_ = std::thread(&MoeSsdCache::cross_layer_worker_main, this);
    }
    return true;
}

bool MoeSsdCache::add_source(const MoeSsdTensorSpec& spec) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fd_ < 0 || spec.weight_ref.empty() || spec.layer < 0 ||
        spec.num_experts <= 0 || spec.rows <= 0 || spec.cols <= 0 ||
        spec.data_bytes == 0) {
        std::fprintf(stderr, "MoE SSD: invalid expert storage metadata for %s\n",
                     spec.weight_ref.c_str());
        return false;
    }
    size_t bytes = 0;
    if (!source_bytes(spec, bytes) || !components_fit_expert_stride(spec) ||
        !component_range_fits(spec.data_offset, spec.data_bytes,
                              spec.expert_stride, spec.num_experts,
                              file_size_) ||
        !component_range_fits(spec.scales_offset, spec.scales_bytes,
                              spec.expert_stride, spec.num_experts,
                              file_size_)) {
        std::fprintf(stderr,
                     "MoE SSD: expert storage range is invalid for %s\n",
                     spec.weight_ref.c_str());
        return false;
    }
    if (spec.precision != Precision::FP16 && spec.precision != Precision::FP32 &&
        spec.precision != Precision::INT8 &&
        spec.precision != Precision::INT4 &&
        spec.precision != Precision::MXFP4 &&
        spec.precision != Precision::NVFP4) {
        std::fprintf(stderr, "MoE SSD: unsupported precision for %s\n",
                     spec.weight_ref.c_str());
        return false;
    }
    bool has_embedded_scales = false;
    if (spec.precision == Precision::NVFP4) {
        const uint64_t rows = static_cast<uint64_t>(spec.rows);
        const uint64_t cols = static_cast<uint64_t>(spec.cols);
        const uint64_t groups = cols / 16;
        const uint64_t expected_data = rows * cols / 2;
        const uint64_t expected_scales = rows * groups + rows * sizeof(float);
        constexpr uint32_t nvfp4_flags =
            MappedFile::FLAG_EXPERT_INTERLEAVED |
            MappedFile::FLAG_NVFP4_Q8_PAIR;
        const bool pair_packed =
            (spec.flags & MappedFile::FLAG_NVFP4_Q8_PAIR) != 0;
        if ((spec.flags & ~nvfp4_flags) != 0 ||
            spec.group_size != 16 || cols % 16 != 0 ||
            (pair_packed && rows % 4 != 0) ||
            spec.groups_per_row != groups ||
            spec.data_bytes != expected_data ||
            spec.scales_bytes != expected_scales) {
            std::fprintf(stderr,
                         "MoE SSD: NVFP4 expert %s has invalid packed "
                         "data or scale metadata\n",
                         spec.weight_ref.c_str());
            return false;
        }
    }
    if (spec.precision == Precision::INT4) {
        constexpr uint32_t layout_flags =
            MappedFile::FLAG_INT4_BG32 | MappedFile::FLAG_INT4_BG128;
        const uint32_t layout = spec.flags & layout_flags;
        const bool bg32 = layout == MappedFile::FLAG_INT4_BG32;
        const bool bg128 = layout == MappedFile::FLAG_INT4_BG128;
        const uint32_t group = bg32 ? 32u : 128u;
        const uint64_t block_bytes = bg32 ? 160u : 544u;
        const uint64_t groups_per_row =
            group != 0 ? static_cast<uint64_t>(spec.cols) / group : 0;
        const uint64_t row_blocks =
            (static_cast<uint64_t>(spec.rows) + 7) / 8;
        const bool size_fits =
            groups_per_row == 0 ||
            row_blocks <= std::numeric_limits<uint64_t>::max() /
                              groups_per_row;
        const uint64_t blocks = size_fits
            ? row_blocks * groups_per_row : 0;
        const bool bytes_fit =
            blocks <= std::numeric_limits<uint64_t>::max() / block_bytes;
        const uint64_t expected_bytes =
            bytes_fit ? blocks * block_bytes : 0;
        if ((!bg32 && !bg128) || (spec.flags & ~layout_flags) != 0 ||
            spec.group_size != group || spec.cols % group != 0 ||
            spec.rows % 8 != 0 || spec.groups_per_row != groups_per_row ||
            !size_fits || !bytes_fit || spec.data_bytes != expected_bytes ||
            spec.scales_bytes != 0) {
            std::fprintf(
                stderr,
                "MoE SSD: INT4 expert %s is not canonical BG32/BG128; "
                "reconvert the model\n",
                spec.weight_ref.c_str());
            return false;
        }
        has_embedded_scales = true;
    }
    if ((spec.precision == Precision::INT8 ||
         spec.precision == Precision::INT4 ||
         spec.precision == Precision::MXFP4 ||
         spec.precision == Precision::NVFP4) &&
        (spec.group_size == 0 || spec.groups_per_row == 0 ||
         (spec.scales_bytes == 0 && !has_embedded_scales))) {
        std::fprintf(stderr, "MoE SSD: quantized expert %s lacks scale metadata\n",
                     spec.weight_ref.c_str());
        return false;
    }
    LayerLayout& layout = layer_layouts_[spec.layer];
    if (layout.num_experts != 0 && layout.num_experts != spec.num_experts) {
        std::fprintf(stderr, "MoE SSD: inconsistent expert count in layer %d\n", spec.layer);
        return false;
    }
    if (bytes > std::numeric_limits<size_t>::max() - layout.pair_bytes) {
        std::fprintf(stderr, "MoE SSD: expert pair size overflows for layer %d\n",
                     spec.layer);
        return false;
    }
    MoeSsdTensorSource source;
    source.spec = spec;
    source.cache = this;
    source.provider = this;
    source.layer = spec.layer;
    auto inserted = sources_.emplace(spec.weight_ref, std::move(source));
    if (!inserted.second) {
        std::fprintf(stderr, "MoE SSD: duplicate expert storage metadata for %s\n",
                     spec.weight_ref.c_str());
        return false;
    }
    layers_.insert(spec.layer);
    layout.num_experts = spec.num_experts;
    layout.pair_bytes += bytes;
    return true;
}

bool MoeSsdCache::configure_shallow_favoring(int shallow_layers) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!entries_.empty()) {
        std::fprintf(stderr, "MoE SSD: cache layout must be configured before expert I/O\n");
        return false;
    }
    if (global_capacity_pool_) {
        // In a shared pool, shallow preference is an eviction tie-break rather
        // than a hard quota. This keeps deep layers able to borrow all idle
        // space while retaining the Fate motivation for protecting shallow
        // layers, whose cross-layer predictions are less reliable.
        shallow_favoring_layers_ = std::max(0, shallow_layers);
        return true;
    }
    layer_capacity_bytes_.clear();
    if (shallow_layers <= 0 || layers_.empty()) return true;

    std::vector<int> ordered_layers(layers_.begin(), layers_.end());
    std::sort(ordered_layers.begin(), ordered_layers.end());
    size_t minimum_bytes = 0;
    for (int layer : ordered_layers) {
        const auto it = layer_layouts_.find(layer);
        if (it == layer_layouts_.end() || it->second.pair_bytes == 0 ||
            it->second.num_experts <= 0 ||
            it->second.pair_bytes > capacity_bytes_ - minimum_bytes) {
            std::fprintf(stderr, "MoE SSD: cache is too small for one expert in every MoE layer\n");
            layer_capacity_bytes_.clear();
            return false;
        }
        layer_capacity_bytes_[layer] = it->second.pair_bytes;
        minimum_bytes += it->second.pair_bytes;
    }

    size_t remaining = capacity_bytes_ - minimum_bytes;
    const size_t shallow_count = std::min<size_t>(
        static_cast<size_t>(shallow_layers), ordered_layers.size());
    // Fate-style shallow priority: fill each early layer completely before
    // allocating the remainder to deeper layers. The one-pair baseline above
    // keeps every deep layer streamable in our per-layer cache design.
    for (size_t i = 0; i < shallow_count && remaining != 0; ++i) {
        const int layer = ordered_layers[i];
        const LayerLayout& layout = layer_layouts_.at(layer);
        const size_t full_layer_bytes = saturating_multiply(
            layout.pair_bytes, static_cast<size_t>(layout.num_experts));
        const size_t already = layer_capacity_bytes_[layer];
        const size_t additional = full_layer_bytes > already ? full_layer_bytes - already : 0;
        const size_t granted = std::min(remaining, additional);
        layer_capacity_bytes_[layer] += granted;
        remaining -= granted;
    }
    const size_t deep_count = ordered_layers.size() - shallow_count;
    if (deep_count != 0 && remaining != 0) {
        const size_t per_layer = remaining / deep_count;
        size_t extra = remaining % deep_count;
        for (size_t i = shallow_count; i < ordered_layers.size(); ++i) {
            layer_capacity_bytes_[ordered_layers[i]] += per_layer;
            if (extra != 0) {
                ++layer_capacity_bytes_[ordered_layers[i]];
                --extra;
            }
        }
    }
    return true;
}

bool MoeSsdCache::set_global_capacity_pool(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!entries_.empty()) {
        std::fprintf(stderr, "MoE SSD: cache policy must be configured before expert I/O\n");
        return false;
    }
    global_capacity_pool_ = enabled;
    if (enabled) {
        layer_capacity_bytes_.clear();
    } else {
        shallow_favoring_layers_ = 0;
    }
    return true;
}

void MoeSsdCache::begin_forward_pass() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (++forward_epoch_ == 0) {
        forward_epoch_ = 1;
        for (const auto& entry : entries_) entry->forward_epoch = 0;
    }
    active_layer_ = -1;
}

const MoeSsdTensorSource* MoeSsdCache::find_source(const std::string& weight_ref) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sources_.find(weight_ref);
    if (it != sources_.end()) return &it->second;
    if (weight_ref.rfind("./", 0) == 0) {
        it = sources_.find(weight_ref.substr(2));
    } else {
        it = sources_.find("./" + weight_ref);
    }
    return it == sources_.end() ? nullptr : &it->second;
}

bool MoeSsdCache::valid_pair(const MoeSsdTensorSource* gate_up,
                             const MoeSsdTensorSource* down,
                             int expert) const {
    return gate_up && down && gate_up->cache == this && down->cache == this &&
           gate_up->spec.layer == down->spec.layer &&
           gate_up->spec.num_experts == down->spec.num_experts &&
           expert >= 0 && expert < gate_up->spec.num_experts;
}

MoeSsdCache::Entry* MoeSsdCache::find_entry_locked(
    const MoeSsdTensorSource* gate_up, const MoeSsdTensorSource* down, int expert) {
    return const_cast<Entry*>(
        static_cast<const MoeSsdCache*>(this)->find_entry_locked(
            gate_up, down, expert));
}

const MoeSsdCache::Entry* MoeSsdCache::find_entry_locked(
    const MoeSsdTensorSource* gate_up,
    const MoeSsdTensorSource* down,
    int expert) const {
    auto layer = layer_entries_.find(gate_up->spec.layer);
    if (layer == layer_entries_.end()) return nullptr;
    for (const Entry* entry : layer->second) {
        if (entry->gate_up == gate_up && entry->down == down &&
            entry->expert == expert) {
            return entry;
        }
    }
    return nullptr;
}

std::unique_ptr<MoeSsdCache::Entry> MoeSsdCache::remove_entry_locked(
    Entry* entry, bool count_eviction) {
    if (!entry || entry->is_loading() || entry->pins != 0)
        return nullptr;

    const auto location = entry_locations_.find(entry);
    if (location == entry_locations_.end())
        return nullptr;

    const int layer = entry->gate_up->spec.layer;
    const auto layer_entries = layer_entries_.find(layer);
    const auto layer_resident = layer_resident_bytes_.find(layer);
    if (layer_entries == layer_entries_.end() ||
        layer_resident == layer_resident_bytes_.end()) {
        return nullptr;
    }
    const auto in_layer =
        std::find(layer_entries->second.begin(), layer_entries->second.end(),
                  entry);
    const size_t bytes = entry->bytes();
    if (in_layer == layer_entries->second.end() ||
        layer_resident->second < bytes || resident_bytes_ < bytes) {
        return nullptr;
    }

    std::unique_ptr<Entry> removed = std::move(*location->second);
    entries_.erase(location->second);
    entry_locations_.erase(location);
    layer_entries->second.erase(in_layer);
    layer_resident->second -= bytes;
    resident_bytes_ -= bytes;
    if (count_eviction) {
        ++evictions_;
        LayerCounters& counters = layer_stats_[layer];
        if (entry->speculative) {
            ++counters.unused_prefetch_evictions;
            unused_prefetch_bytes_ += bytes;
        }
        const uint64_t key =
            (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
            static_cast<uint32_t>(entry->expert);
        last_evicted_epoch_[key] = forward_epoch_;
    }
    return removed;
}

MoeSsdCache::Entry* MoeSsdCache::reserve_entry_locked(
    const MoeSsdTensorSource* gate_up, const MoeSsdTensorSource* down, int expert,
    bool speculative, float prediction_confidence) {
    size_t required = 0;
    if (!expert_pair_bytes(gate_up, down, required))
        return nullptr;
    const size_t layer_capacity = layer_capacity_bytes_locked(gate_up->spec.layer);
    const size_t capacity = global_capacity_pool_ ? capacity_bytes_ : layer_capacity;
    if (required > capacity) {
        std::fprintf(stderr,
                     "MoE SSD: one expert pair needs %.1f MB but cache holds %.1f MB\n",
                     required / 1e6, capacity / 1e6);
        return nullptr;
    }

    const int layer = gate_up->spec.layer;
    const uint64_t eviction_key =
        (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
        static_cast<uint32_t>(expert);
    const auto last_evicted = last_evicted_epoch_.find(eviction_key);
    if (last_evicted != last_evicted_epoch_.end() &&
        forward_epoch_ >= last_evicted->second &&
        forward_epoch_ - last_evicted->second <= 1) {
        ++layer_stats_[layer].short_term_reloads;
        last_evicted_epoch_.erase(last_evicted);
    }
    size_t& layer_resident = layer_resident_bytes_[layer];
    std::vector<Entry*>& layer_entries = layer_entries_[layer];
    std::unique_ptr<Entry> recycled;
    while (true) {
        const size_t used = global_capacity_pool_ ? resident_bytes_ : layer_resident;
        if (used <= capacity && required <= capacity - used)
            break;
        Entry* victim_entry = nullptr;
        if (global_capacity_pool_) {
            for (const auto& entry : entries_) {
                if (entry->is_loading() || entry->pins != 0) continue;
                if (!victim_entry || global_victim_before_locked(entry.get(), victim_entry)) {
                    victim_entry = entry.get();
                }
            }
        } else {
            auto victim = std::min_element(layer_entries.begin(), layer_entries.end(),
                [](const Entry* a, const Entry* b) {
                    const bool a_busy = a->is_loading() || a->pins != 0;
                    const bool b_busy = b->is_loading() || b->pins != 0;
                    if (a_busy != b_busy) return !a_busy;
                    return a->used_at < b->used_at;
                });
            if (victim != layer_entries.end()) victim_entry = *victim;
        }
        if (!victim_entry || victim_entry->is_loading() || victim_entry->pins != 0) {
            // The asynchronous request window for this layer is full. The
            // caller can let workers finish and retry later.
            return nullptr;
        }
        if (speculative) {
            const int victim_layer = victim_entry->gate_up->spec.layer;
            const bool stale = victim_entry->forward_epoch != forward_epoch_;
            const bool left = active_layer_ >= 0 && victim_layer < active_layer_;
            const bool weaker_prediction =
                victim_entry->speculative &&
                victim_entry->prediction_epoch == forward_epoch_ &&
                victim_entry->prediction_confidence < prediction_confidence;
            // Advisory reads must not evict a demand-loaded entry which may
            // still be useful later in this forward pass. They may consume
            // stale/left-layer space or replace a weaker prediction.
            if (!stale && !left && !weaker_prediction)
                return nullptr;
        }
        auto removed = remove_entry_locked(victim_entry, true);
        if (!removed)
            return nullptr;
        if (!recycled)
            recycled = std::move(removed);
    }

    auto entry = recycled ? std::move(recycled) : std::make_unique<Entry>();
    entry->gate_up = gate_up;
    entry->down = down;
    entry->expert = expert;
    entry->used_at = ++clock_;
    entry->state = Entry::State::Loading;
    entry->pending_reads = 0;
    entry->fresh_miss = true;
    entry->speculative = speculative;
    entry->load_origin_speculative = speculative;
    const auto retained = retained_experts_.find(layer);
    entry->retained =
        retained != retained_experts_.end() &&
        std::find(retained->second.begin(), retained->second.end(), expert) !=
            retained->second.end();
    entry->forward_epoch = forward_epoch_;
    entry->prediction_epoch = speculative ? forward_epoch_ : 0;
    entry->prediction_confidence = speculative ? prediction_confidence : 0.0f;
    entry->gate_up_data.resize(static_cast<size_t>(gate_up->spec.data_bytes));
    entry->gate_up_scales.resize(static_cast<size_t>(gate_up->spec.scales_bytes));
    entry->down_data.resize(static_cast<size_t>(down->spec.data_bytes));
    entry->down_scales.resize(static_cast<size_t>(down->spec.scales_bytes));
    resident_bytes_ += entry->bytes();
    layer_resident += entry->bytes();
    Entry* raw = entry.get();
    entries_.push_back(std::move(entry));
    auto location = std::prev(entries_.end());
    entry_locations_.emplace(raw, location);
    layer_entries.push_back(raw);
    return raw;
}

bool MoeSsdCache::prefer_left_layer_eviction_locked() const {
    if (active_layer_ < 0 || layers_.empty())
        return false;
    const auto layout = layer_layouts_.find(active_layer_);
    const auto route = layer_route_widths_.find(active_layer_);
    if (layout == layer_layouts_.end() || layout->second.pair_bytes == 0 ||
        route == layer_route_widths_.end() || route->second == 0) {
        return false;
    }
    const size_t fair_share = capacity_bytes_ / layers_.size();
    const size_t fair_share_pairs = fair_share / layout->second.pair_bytes;
    return fair_share_pairs < route->second;
}

bool MoeSsdCache::global_victim_before_locked(const Entry* candidate,
                                              const Entry* current) const {
    const bool prefer_left = prefer_left_layer_eviction_locked();
    auto rank = [&](const Entry* entry) {
        const bool stale = entry->forward_epoch != forward_epoch_;
        const int layer = entry->gate_up->spec.layer;
        const bool shallow = shallow_favoring_layers_ > 0 && layer < shallow_favoring_layers_;
        const bool left = active_layer_ >= 0 && layer < active_layer_;
        int value = 0;
        // A streaming-sized cache cannot preserve one complete route per
        // layer, so recycle finished layers before displacing entries which
        // may still be used later in this pass. Once every layer's fair share
        // can hold the current route, least-stale ordering instead preserves
        // the stronger cross-token locality.
        if ((prefer_left && left) || (!prefer_left && stale))
            value = shallow ? 1 : 0;
        else if ((prefer_left && stale) || (!prefer_left && left))
            value = shallow ? 3 : 2;
        else value = shallow ? 5 : 4;
        return value + (entry->retained ? 6 : 0);
    };
    const int candidate_rank = rank(candidate);
    const int current_rank = rank(current);
    if (candidate_rank != current_rank) return candidate_rank < current_rank;
    const bool candidate_stale =
        candidate->forward_epoch != forward_epoch_;
    const bool current_stale =
        current->forward_epoch != forward_epoch_;
    const int candidate_layer = candidate->gate_up->spec.layer;
    const int current_layer = current->gate_up->spec.layer;
    const bool candidate_future =
        candidate_stale && active_layer_ >= 0 &&
        candidate_layer >= active_layer_;
    const bool current_future =
        current_stale && active_layer_ >= 0 &&
        current_layer >= active_layer_;
    if (candidate_future && current_future &&
        candidate_layer != current_layer) {
        // If future residency must be sacrificed, evict the entry whose next
        // possible use is farthest away.
        return candidate_layer > current_layer;
    }
    const bool candidate_predicted = candidate->prediction_epoch == forward_epoch_;
    const bool current_predicted = current->prediction_epoch == forward_epoch_;
    if (candidate_predicted != current_predicted) return !candidate_predicted;
    if (candidate_predicted &&
        candidate->prediction_confidence != current->prediction_confidence) {
        return candidate->prediction_confidence < current->prediction_confidence;
    }
    return candidate->used_at < current->used_at;
}

bool MoeSsdCache::retain_for_next_forward(
    const MoeSsdTensorSource* gate_up,
    const MoeSsdTensorSource* down,
    const std::vector<int>& experts,
    bool keep_cross_token) {
    if (!gate_up || !down || !valid_pair(gate_up, down, 0))
        return false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!global_capacity_pool_)
        return true;
    const int layer = gate_up->spec.layer;
    const auto layout = layer_layouts_.find(layer);
    if (layout == layer_layouts_.end() || layout->second.pair_bytes == 0)
        return false;
    const size_t layer_share =
        layers_.empty() ? capacity_bytes_ : capacity_bytes_ / layers_.size();
    size_t retain_limit =
        std::max<size_t>(1, layer_share / layout->second.pair_bytes);
    // Do not let a large per-layer fair share pin a long tail of nearly cold
    // history. Current-route experts remain first, followed by decayed
    // frequency candidates up to this soft multiple. Smaller fair shares are
    // unchanged.
    constexpr size_t route_multiplier = 12;
    if (route_multiplier != 0 && !experts.empty()) {
        const size_t route_cap = experts.size() >
                std::numeric_limits<size_t>::max() / route_multiplier
            ? std::numeric_limits<size_t>::max()
            : experts.size() * route_multiplier;
        retain_limit = std::min(retain_limit, route_cap);
    }
    std::vector<int>& retained = retained_experts_[layer];
    std::vector<float>& scores = retained_scores_[layer];
    scores.resize(static_cast<size_t>(gate_up->spec.num_experts), 0.0f);
    constexpr float kFrequencyDecay = 0.96875f;
    for (float& score : scores)
        score *= kFrequencyDecay;
    for (int expert : experts) {
        if (expert >= 0 && expert < gate_up->spec.num_experts)
            scores[static_cast<size_t>(expert)] += 1.0f;
    }

    // An exactly known next route can be loaded on demand before this layer
    // starts, so keeping its high-entropy previous route would only turn the
    // shared pool into a hard per-layer reservation.
    if (!keep_cross_token) {
        retained.clear();
        const auto entries = layer_entries_.find(layer);
        if (entries != layer_entries_.end()) {
            for (Entry* entry : entries->second)
                entry->retained = false;
        }
        return true;
    }

    std::vector<int> next;
    next.reserve(retain_limit);
    // Never let historical frequency displace an expert needed by the
    // current route.
    for (int expert : experts) {
        if (expert < 0 || expert >= gate_up->spec.num_experts ||
            std::find(next.begin(), next.end(), expert) != next.end()) {
            continue;
        }
        next.push_back(expert);
        if (next.size() == retain_limit)
            break;
    }

    struct Candidate {
        int expert = -1;
        float score = 0.0f;
        uint64_t used_at = 0;
    };
    std::vector<Candidate> candidates;
    const auto entries = layer_entries_.find(layer);
    if (entries != layer_entries_.end()) {
        candidates.reserve(entries->second.size());
        for (const Entry* entry : entries->second) {
            if (entry->gate_up != gate_up || entry->down != down ||
                std::find(next.begin(), next.end(), entry->expert) !=
                    next.end()) {
                continue;
            }
            candidates.push_back({
                entry->expert,
                scores[static_cast<size_t>(entry->expert)],
                entry->used_at,
            });
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) {
                  if (lhs.score != rhs.score)
                      return lhs.score > rhs.score;
                  return lhs.used_at > rhs.used_at;
              });
    for (const Candidate& candidate : candidates) {
        if (next.size() == retain_limit)
            break;
        next.push_back(candidate.expert);
    }
    retained = std::move(next);
    if (entries != layer_entries_.end()) {
        for (Entry* entry : entries->second) {
            entry->retained =
                entry->gate_up == gate_up && entry->down == down &&
                std::find(retained.begin(), retained.end(), entry->expert) !=
                    retained.end();
        }
    }
    return true;
}

Tensor MoeSsdCache::make_tensor(const MoeSsdTensorSource& source,
                                const uint8_t* data,
                                const uint8_t* scales) const {
    const MoeSsdTensorSpec& s = source.spec;
    Tensor t = Tensor::create(s.precision, MemoryType::EXTERNAL,
                              s.rows, s.cols, 1, 1,
                              const_cast<uint8_t*>(data));
    if (s.precision == Precision::INT8 || s.precision == Precision::INT4) {
        t.scales = reinterpret_cast<const float*>(scales);
        t.group_size = s.group_size;
        t.groups_per_row = s.groups_per_row;
        t.num_groups = static_cast<uint32_t>(s.rows) * s.groups_per_row;
        t.is_q4_repacked = false;
        t.is_q4_g32_packed = (s.flags & MappedFile::FLAG_INT4_BG32) != 0;
        t.is_q4_g128_packed = (s.flags & MappedFile::FLAG_INT4_BG128) != 0;
        if (t.is_q4_g32_packed) t.q4_g32_data = t.data;
        if (t.is_q4_g128_packed) t.q4_g128_data = t.data;
    } else if (s.precision == Precision::MXFP4) {
        t.e8m0_scales = scales;
        t.group_size = s.group_size;
        t.groups_per_row = s.groups_per_row;
        t.num_groups = static_cast<uint32_t>(s.rows) * s.groups_per_row;
    } else if (s.precision == Precision::NVFP4) {
        const size_t block_scale_bytes =
            static_cast<size_t>(s.rows) * s.groups_per_row;
        const size_t row_scale_bytes =
            static_cast<size_t>(s.rows) * sizeof(float);
        if (!scales ||
            s.scales_bytes != block_scale_bytes + row_scale_bytes)
            return Tensor{};
        t.nvfp4_scales = scales;
        t.nvfp4_row_scales = reinterpret_cast<const float*>(
            scales + block_scale_bytes);
        if ((s.flags & MappedFile::FLAG_NVFP4_Q8_PAIR) != 0)
            t.nvfp4_q8_pair_data = data;
        t.group_size = s.group_size;
        t.groups_per_row = s.groups_per_row;
        t.num_groups = static_cast<uint32_t>(s.rows) * s.groups_per_row;
    }
    return t;
}

bool MoeSsdCache::request_many(const MoeSsdTensorSource* gate_up,
                               const MoeSsdTensorSource* down,
                               const std::vector<int>& experts) {
    return request_many_impl(gate_up, down, experts, false);
}

bool MoeSsdCache::prefetch_many(const MoeSsdTensorSource* gate_up,
                                const MoeSsdTensorSource* down,
                                const std::vector<int>& experts,
                                const std::vector<float>& confidence,
                                size_t prefetch_count) {
    return request_many_impl(gate_up, down, experts, true, confidence,
                             prefetch_count);
}

bool MoeSsdCache::request_many_impl(const MoeSsdTensorSource* gate_up,
                                    const MoeSsdTensorSource* down,
                                    const std::vector<int>& experts,
                                    bool speculative,
                                    const std::vector<float>& confidence,
                                    size_t request_count) {
    if (!gate_up || !down || !valid_pair(gate_up, down, 0)) {
        std::fprintf(stderr, "MoE SSD: invalid expert pair request\n");
        return false;
    }
    const uint64_t trace_start = mollm_trace::now_ns();
    std::vector<uint8_t> seen((size_t)gate_up->spec.num_experts, 0);
    struct MissingExpert {
        int expert = -1;
        float prediction_confidence = 0.0f;
    };
    std::vector<MissingExpert> missing_experts;
    std::vector<Entry*> queued_entries;
    bool promoted_reads = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!speculative) active_layer_ = gate_up->spec.layer;
        if (speculative) {
            pending_predictions_[gate_up->spec.layer] =
                PredictionRecord{forward_epoch_, experts, confidence};
        } else {
            const auto prediction = pending_predictions_.find(gate_up->spec.layer);
            if (prediction != pending_predictions_.end() &&
                prediction->second.forward_epoch == forward_epoch_) {
                const std::vector<int>& predicted = prediction->second.experts;
                if (cross_layer_rank_attempts_.size() < predicted.size()) {
                    cross_layer_rank_attempts_.resize(predicted.size());
                    cross_layer_rank_hits_.resize(predicted.size());
                    cross_layer_rank_confidence_sum_.resize(predicted.size());
                }
                if (prediction_policy_attempts_.size() < predicted.size()) {
                    prediction_policy_attempts_.resize(predicted.size());
                    prediction_policy_hits_.resize(predicted.size());
                }
                for (size_t rank = 0; rank < predicted.size(); ++rank) {
                    ++cross_layer_rank_attempts_[rank];
                    if (rank < prediction->second.confidence.size()) {
                        cross_layer_rank_confidence_sum_[rank] +=
                            prediction->second.confidence[rank];
                    }
                    ++prediction_policy_attempts_[rank];
                    const bool matched =
                        std::find(experts.begin(), experts.end(),
                                  predicted[rank]) != experts.end();
                    if (matched) {
                        ++cross_layer_rank_hits_[rank];
                        ++prediction_policy_hits_[rank];
                    }
                    LayerCounters& layer_counters =
                        layer_stats_[gate_up->spec.layer];
                    ++layer_counters.prediction_attempts;
                    if (matched) ++layer_counters.prediction_matches;
                }

                pending_predictions_.erase(prediction);
            }
        }
        const size_t count = std::min(experts.size(), request_count);
        if (speculative)
            layer_stats_[gate_up->spec.layer].prefetch_selected += count;
        missing_experts.reserve(count);
        queued_entries.reserve(count);
        std::vector<Entry*> demanded_loading_entries;

        // Protect every resident member of this route before reserving any
        // misses. Reserving while walking the route can otherwise evict a
        // cached expert that appears later in the same top-k list, turning a
        // guaranteed hit into an avoidable read.
        for (size_t index = 0; index < count; ++index) {
            const int expert = experts[index];
            const float prediction_confidence =
                speculative && index < confidence.size()
                    ? std::max(0.0f, confidence[index]) : 0.0f;
            if (expert < 0 || expert >= gate_up->spec.num_experts || seen[(size_t)expert]) continue;
            seen[(size_t)expert] = 1;
            Entry* entry = find_entry_locked(gate_up, down, expert);
            if (entry && entry->state == Entry::State::Failed) {
                if (remove_entry_locked(entry, false))
                    entry = nullptr;
            }
            if (entry) {
                entry->used_at = ++clock_;
                entry->forward_epoch = forward_epoch_;
                if (speculative) {
                    entry->prediction_epoch = forward_epoch_;
                    entry->prediction_confidence =
                        std::max(entry->prediction_confidence, prediction_confidence);
                } else if (entry->is_loading()) {
                    demanded_loading_entries.push_back(entry);
                }
                continue;
            }
            missing_experts.push_back({expert, prediction_confidence});
        }
        if (!speculative && !demanded_loading_entries.empty()) {
            // A correct prediction may still have component jobs waiting in
            // the advisory queue. Once the real router asks for that entry,
            // promote those jobs ahead of unrelated demand misses; otherwise
            // a prediction hit can paradoxically complete after a cold miss.
            auto job = low_priority_io_jobs_.begin();
            while (job != low_priority_io_jobs_.end()) {
                const bool demanded = std::any_of(
                    job->entries.begin(), job->entries.end(),
                    [&](const Entry* entry) {
                        return std::find(
                                   demanded_loading_entries.begin(),
                                   demanded_loading_entries.end(), entry) !=
                               demanded_loading_entries.end();
                    });
                if (!demanded) {
                    ++job;
                    continue;
                }
                job->speculative = false;
                io_jobs_.push_back(std::move(*job));
                job = low_priority_io_jobs_.erase(job);
                promoted_reads = true;
            }
        }
        if (!speculative) {
            layer_route_widths_[gate_up->spec.layer] =
                static_cast<size_t>(
                    std::count(seen.begin(), seen.end(), uint8_t{1}));
        }
        for (const MissingExpert& missing : missing_experts) {
            Entry* entry = reserve_entry_locked(
                gate_up, down, missing.expert, speculative,
                missing.prediction_confidence);
            // Do not reserve beyond the per-layer byte budget. acquire() will
            // submit this expert later after an in-flight slot becomes ready.
            if (!entry) {
                if (speculative) ++cross_layer_rejected_;
                continue;
            }
            if (speculative) {
                ++cross_layer_experts_;
                ++layer_stats_[gate_up->spec.layer].prefetch_admitted;
            }
            else ++misses_;
            queued_entries.push_back(entry);
        }
        enqueue_entry_reads_locked(queued_entries, speculative);
    }
    if (!queued_entries.empty() || promoted_reads) io_cv_.notify_all();
    if (trace_start != 0) {
        std::string trace_args =
            "{\"layer\":" + std::to_string(gate_up->spec.layer) +
            ",\"requested\":" + std::to_string(experts.size()) +
            ",\"queued\":" + std::to_string(queued_entries.size()) +
            ",\"promoted\":" + (promoted_reads ? "true" : "false") +
            ",\"experts\":[";
        for (size_t i = 0; i < experts.size(); ++i) {
            if (i != 0) trace_args += ',';
            trace_args += std::to_string(experts[i]);
        }
        trace_args += ']';
        if (speculative) {
            trace_args += ",\"confidence\":[";
            for (size_t i = 0; i < confidence.size(); ++i) {
                if (i != 0) trace_args += ',';
                trace_args += std::to_string(confidence[i]);
            }
            trace_args += ']';
        }
        trace_args += '}';
        mollm_trace::record_duration(
            speculative ? "ssd.predict" : "ssd",
            speculative ? "prefetch_many" : "request_many", trace_start, mollm_trace::now_ns(),
            trace_args);
    }
    return true;
}

size_t MoeSsdCache::recommended_prefetch_count(size_t predicted_count) const {
    if (predicted_count == 0) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    constexpr uint64_t kMinimumSamples = 128;
    constexpr double kMinimumHitRate = 0.80;
    constexpr size_t configured_minimum = 3;
    const size_t half_minimum =
        std::max<size_t>(1, predicted_count / 2);
    const size_t minimum = std::min(half_minimum, configured_minimum);
    size_t count = predicted_count;
    while (count > minimum) {
        const size_t rank = count - 1;
        if (rank >= prediction_policy_attempts_.size() ||
            prediction_policy_attempts_[rank] < kMinimumSamples) {
            break;
        }
        const double hit_rate =
            static_cast<double>(prediction_policy_hits_[rank]) /
            static_cast<double>(prediction_policy_attempts_[rank]);
        if (hit_rate >= kMinimumHitRate) break;
        --count;
    }
    return count;
}

bool MoeSsdCache::submit_cross_layer_task(std::function<void()> task) {
    if (!task) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    constexpr size_t kMaxQueuedTasks = 2;
    if (stop_io_ || !cross_layer_worker_.joinable() ||
        cross_layer_tasks_.size() >= kMaxQueuedTasks) {
        ++cross_layer_dropped_;
        return false;
    }
    cross_layer_tasks_.push_back(std::move(task));
    ++cross_layer_tasks_count_;
    cross_layer_cv_.notify_one();
    return true;
}

bool MoeSsdCache::can_prefetch_pairs(const MoeSsdTensorSource* gate_up,
                                     const MoeSsdTensorSource* down,
                                     size_t pairs) const {
    if (!valid_pair(gate_up, down, 0) || pairs == 0) return false;
    size_t pair_bytes = 0;
    if (!expert_pair_bytes(gate_up, down, pair_bytes))
        return false;
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t capacity = global_capacity_pool_ ? capacity_bytes_
                                                   : layer_capacity_bytes_locked(gate_up->spec.layer);
    return pair_bytes <= capacity / pairs;
}

size_t MoeSsdCache::layer_capacity_bytes(int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return layer_capacity_bytes_locked(layer);
}

size_t MoeSsdCache::layer_capacity_bytes_locked(int layer) const {
    const auto configured = layer_capacity_bytes_.find(layer);
    if (configured != layer_capacity_bytes_.end()) return configured->second;
    return layers_.empty() ? capacity_bytes_ : capacity_bytes_ / layers_.size();
}

size_t MoeSsdCache::resident_count(const MoeSsdTensorSource* gate_up,
                                   const MoeSsdTensorSource* down,
                                   const std::vector<int>& experts) const {
    if (!valid_pair(gate_up, down, 0)) return 0;
    std::vector<uint8_t> seen((size_t)gate_up->spec.num_experts, 0);
    size_t count = 0;
    std::lock_guard<std::mutex> lock(mutex_);
    for (int expert : experts) {
        if (expert < 0 || expert >= gate_up->spec.num_experts || seen[(size_t)expert]) continue;
        seen[(size_t)expert] = 1;
        const Entry* entry = find_entry_locked(gate_up, down, expert);
        if (entry && !entry->is_failed())
            ++count;
    }
    return count;
}

bool MoeSsdCache::contains(const MoeSsdTensorSource* gate_up,
                           const MoeSsdTensorSource* down,
                           int expert) const {
    if (!valid_pair(gate_up, down, expert)) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    const Entry* entry = find_entry_locked(gate_up, down, expert);
    return entry && !entry->is_failed();
}

bool MoeSsdCache::is_ready(const MoeSsdTensorSource* gate_up,
                           const MoeSsdTensorSource* down,
                           int expert) const {
    if (!valid_pair(gate_up, down, expert)) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    const Entry* entry = find_entry_locked(gate_up, down, expert);
    return entry && entry->is_ready();
}

bool MoeSsdCache::release(const MoeSsdTensorSource* gate_up,
                          const MoeSsdTensorSource* down,
                          int expert) {
    if (!valid_pair(gate_up, down, expert)) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    Entry* entry = find_entry_locked(gate_up, down, expert);
    // release() is called only after routed compute has finished. Do not
    // invalidate an in-flight asynchronous read if a caller violates that
    // lifetime contract.
    if (!entry || entry->is_loading()) return false;
    return remove_entry_locked(entry, true) != nullptr;
}

bool MoeSsdCache::acquire(const MoeSsdTensorSource* gate_up,
                          const MoeSsdTensorSource* down,
                          int expert,
                          Tensor& gate_up_out,
                          Tensor& down_out) {
    return acquire_impl(gate_up, down, expert, gate_up_out, down_out, nullptr, true);
}

bool MoeSsdCache::acquire_impl(const MoeSsdTensorSource* gate_up,
                              const MoeSsdTensorSource* down, int expert,
                              Tensor& gate_up_out, Tensor& down_out,
                              Entry** pinned_entry, bool wait) {
    if (!valid_pair(gate_up, down, expert)) {
        std::fprintf(stderr, "MoE SSD: invalid expert pair request\n");
        return false;
    }
    const uint64_t trace_start = mollm_trace::now_ns();
    using SteadyClock = std::chrono::steady_clock;
    SteadyClock::time_point wait_start;
    bool waited = false;
    auto begin_wait = [&] {
        if (!waited) {
            wait_start = SteadyClock::now();
            waited = true;
        }
    };
    std::unique_lock<std::mutex> lock(mutex_);
    size_t required = 0;
    if (!expert_pair_bytes(gate_up, down, required))
        return false;
    const size_t capacity = global_capacity_pool_ ? capacity_bytes_
                                                   : layer_capacity_bytes_locked(gate_up->spec.layer);
    if (required > capacity) {
        std::fprintf(stderr,
                     "MoE SSD: one expert pair needs %.1f MB but cache holds %.1f MB\n",
                     required / 1e6, capacity / 1e6);
        return false;
    }
    if (global_capacity_pool_) active_layer_ = gate_up->spec.layer;
    Entry* entry = find_entry_locked(gate_up, down, expert);
    if (!wait && (!entry || !entry->is_ready())) return false;
    if (entry && entry->state == Entry::State::Failed) {
        if (remove_entry_locked(entry, false))
            entry = nullptr;
    }
    // Protect the entry before releasing the mutex to wait for I/O. Otherwise
    // another requester could evict a just-completed read before we wake up.
    struct PendingPin {
        Entry* entry = nullptr;
        std::condition_variable& ready;
        void hold(Entry* value) {
            if (value && !entry) {
                entry = value;
                ++entry->pins;
            }
        }
        ~PendingPin() {
            if (entry) {
                --entry->pins;
                ready.notify_all();
            }
        }
    } pin{nullptr, ready_cv_};
    pin.hold(entry);
    while (!entry) {
        entry = reserve_entry_locked(gate_up, down, expert);
        if (entry) {
            pin.hold(entry);
            ++misses_;
            enqueue_entry_reads_locked({entry});
            lock.unlock();
            io_cv_.notify_all();
            lock.lock();
            break;
        }
        // All slots are loading. Let one complete, then evict it if needed
        // and submit this deferred expert.
        begin_wait();
        const auto slot_wait_start = SteadyClock::now();
        ready_cv_.wait(lock);
        slot_wait_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                SteadyClock::now() - slot_wait_start).count());
        ++slot_waits_;
        entry = find_entry_locked(gate_up, down, expert);
        pin.hold(entry);
    }
    if (entry && entry->is_loading()) begin_wait();
    ready_cv_.wait(lock, [&] { return entry && !entry->is_loading(); });
    if (!entry || !entry->is_ready()) {
        std::fprintf(stderr, "MoE SSD: failed to load expert %d\n", expert);
        return false;
    }
    const bool fresh_miss = entry->fresh_miss;
    expert_bytes_acquired_ += entry->bytes();
    entry->used_at = ++clock_;
    entry->forward_epoch = forward_epoch_;
    if (entry->speculative) {
        ++cross_layer_used_;
        useful_prefetch_bytes_ += entry->bytes();
        entry->speculative = false;
    }
    entry->prediction_epoch = 0;
    entry->prediction_confidence = 0.0f;
    if (fresh_miss) {
        entry->fresh_miss = false;
    } else {
        ++hits_;
    }
    gate_up_out = make_tensor(
        *gate_up, entry->gate_up_data.data(), entry->gate_up_scales.data());
    down_out = make_tensor(
        *down, entry->down_data.data(), entry->down_scales.data());
    LayerCounters& layer_counters = layer_stats_[gate_up->spec.layer];
    ++layer_counters.demand_acquires;
    if (fresh_miss) ++layer_counters.demand_misses;
    else ++layer_counters.demand_hits;
    if (waited) {
        layer_counters.acquire_wait_ns +=
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    SteadyClock::now() - wait_start).count());
    }
    if (trace_start != 0) {
        mollm_trace::record_duration(
            "ssd", "acquire", trace_start, mollm_trace::now_ns(),
            "{\"layer\":" + std::to_string(gate_up->spec.layer) +
            ",\"expert\":" + std::to_string(expert) + "}");
    }
    if (pinned_entry) {
        *pinned_entry = entry;
        pin.entry = nullptr;  // Transfer the pin to the returned ExpertLease.
    }
    return true;
}

MoeSsdCache::Stats MoeSsdCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats result;
    result.hits = hits_;
    result.misses = misses_;
    result.evictions = evictions_;
    result.bytes_read = bytes_read_;
    result.demand_load_bytes = demand_load_bytes_;
    result.prefetch_load_bytes = prefetch_load_bytes_;
    result.useful_prefetch_bytes = useful_prefetch_bytes_;
    result.unused_prefetch_bytes = unused_prefetch_bytes_;
    result.expert_bytes_acquired = expert_bytes_acquired_;
    result.slot_waits = slot_waits_;
    result.slot_wait_ns = slot_wait_ns_;
    result.cross_layer_tasks = cross_layer_tasks_count_;
    result.cross_layer_dropped = cross_layer_dropped_;
    result.cross_layer_experts = cross_layer_experts_;
    result.cross_layer_used = cross_layer_used_;
    result.cross_layer_rejected = cross_layer_rejected_;
    result.resident_bytes = resident_bytes_;
    result.cross_layer_rank_attempts = cross_layer_rank_attempts_;
    result.cross_layer_rank_hits = cross_layer_rank_hits_;
    result.cross_layer_rank_confidence_sum =
        cross_layer_rank_confidence_sum_;
    result.layers.reserve(layer_stats_.size());
    for (const auto& [layer, counters] : layer_stats_) {
        result.layers.push_back({
            layer,
            counters.demand_acquires,
            counters.demand_hits,
            counters.demand_misses,
            counters.acquire_wait_ns,
            counters.prediction_attempts,
            counters.prediction_matches,
            counters.prefetch_selected,
            counters.prefetch_admitted,
            counters.unused_prefetch_evictions,
            counters.short_term_reloads,
        });
    }
    std::sort(result.layers.begin(), result.layers.end(),
              [](const LayerStats& lhs, const LayerStats& rhs) {
                  return lhs.layer < rhs.layer;
              });
    return result;
}

void MoeSsdCache::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    hits_ = 0;
    misses_ = 0;
    evictions_ = 0;
    bytes_read_ = 0;
    demand_load_bytes_ = 0;
    prefetch_load_bytes_ = 0;
    useful_prefetch_bytes_ = 0;
    unused_prefetch_bytes_ = 0;
    expert_bytes_acquired_ = 0;
    slot_waits_ = slot_wait_ns_ = 0;
    cross_layer_tasks_count_ = 0;
    cross_layer_dropped_ = 0;
    cross_layer_experts_ = 0;
    cross_layer_used_ = 0;
    cross_layer_rejected_ = 0;
    cross_layer_rank_attempts_.clear();
    cross_layer_rank_hits_.clear();
    cross_layer_rank_confidence_sum_.clear();
    // Do not clear pending_predictions_: benchmarks reset counters at the
    // prefill/decode boundary, and prediction records are live cache-policy
    // state that may be consumed by the following decode layer.
    layer_stats_.clear();
    last_evicted_epoch_.clear();
}
