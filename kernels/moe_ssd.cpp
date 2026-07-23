#include "kernels/moe_ssd.h"
#include "kernels/moe_ssd_internal.h"

#include "graph/mmap_file.h"
#include "kernels/trace.h"

#include <algorithm>
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
                          int num_experts, uint64_t file_size) {
    if (bytes_per_expert == 0)
        return true;
    const uint64_t count = static_cast<uint64_t>(num_experts);
    if (count > std::numeric_limits<uint64_t>::max() / bytes_per_expert)
        return false;
    const uint64_t total = count * bytes_per_expert;
    return offset <= file_size && total <= file_size - offset;
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
                       int io_workers, bool enable_cross_layer_worker) {
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
        resident_bytes_ = 0;
        clock_ = hits_ = misses_ = evictions_ = bytes_read_ = 0;
        cross_layer_tasks_count_ = cross_layer_dropped_ = 0;
        cross_layer_experts_ = cross_layer_used_ = 0;
        sources_.clear();
        layers_.clear();
        layer_layouts_.clear();
        layer_capacity_bytes_.clear();
        entries_.clear();
        entry_locations_.clear();
        layer_entries_.clear();
        layer_resident_bytes_.clear();
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
    if (!source_bytes(spec, bytes) ||
        !component_range_fits(spec.data_offset, spec.data_bytes,
                              spec.num_experts, file_size_) ||
        !component_range_fits(spec.scales_offset, spec.scales_bytes,
                              spec.num_experts, file_size_)) {
        std::fprintf(stderr,
                     "MoE SSD: expert storage range is invalid for %s\n",
                     spec.weight_ref.c_str());
        return false;
    }
    if (spec.precision != Precision::FP16 && spec.precision != Precision::FP32 &&
        spec.precision != Precision::INT8 && spec.precision != Precision::INT4) {
        std::fprintf(stderr, "MoE SSD: unsupported precision for %s\n",
                     spec.weight_ref.c_str());
        return false;
    }
    if ((spec.precision == Precision::INT8 || spec.precision == Precision::INT4) &&
        (spec.group_size == 0 || spec.groups_per_row == 0 || spec.scales_bytes == 0)) {
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
        if (entry->gate_up == gate_up && entry->down == down && entry->expert == expert) {
            return entry;
        }
    }
    return nullptr;
}

std::unique_ptr<MoeSsdCache::Entry> MoeSsdCache::remove_entry_locked(
    Entry* entry, bool count_eviction) {
    if (!entry || entry->is_loading())
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
    if (count_eviction)
        ++evictions_;
    return removed;
}

MoeSsdCache::Entry* MoeSsdCache::reserve_entry_locked(
    const MoeSsdTensorSource* gate_up, const MoeSsdTensorSource* down, int expert,
    bool speculative) {
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
                if (entry->is_loading()) continue;
                if (!victim_entry || global_victim_before_locked(entry.get(), victim_entry)) {
                    victim_entry = entry.get();
                }
            }
        } else {
            auto victim = std::min_element(layer_entries.begin(), layer_entries.end(),
                [](const Entry* a, const Entry* b) {
                    if (a->is_loading() != b->is_loading())
                        return !a->is_loading();
                    return a->used_at < b->used_at;
                });
            if (victim != layer_entries.end()) victim_entry = *victim;
        }
        if (!victim_entry || victim_entry->is_loading()) {
            // The asynchronous request window for this layer is full. The
            // caller can let workers finish and retry later.
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
    entry->forward_epoch = forward_epoch_;
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

bool MoeSsdCache::global_victim_before_locked(const Entry* candidate,
                                              const Entry* current) const {
    auto rank = [&](const Entry* entry) {
        const bool stale = entry->forward_epoch != forward_epoch_;
        const int layer = entry->gate_up->spec.layer;
        const bool shallow = shallow_favoring_layers_ > 0 && layer < shallow_favoring_layers_;
        const bool left = active_layer_ >= 0 && layer < active_layer_;
        // Least-Stale first; within a pass, left layers have no remaining use.
        // A shallow entry is protected as a tie-break within either category.
        if (stale) return shallow ? 1 : 0;
        if (left) return shallow ? 3 : 2;
        return shallow ? 5 : 4;
    };
    const int candidate_rank = rank(candidate);
    const int current_rank = rank(current);
    if (candidate_rank != current_rank) return candidate_rank < current_rank;
    return candidate->used_at < current->used_at;
}

Tensor MoeSsdCache::make_tensor(const MoeSsdTensorSource& source,
                                 const uint8_t* data,
                                 const uint8_t* scales) {
    const MoeSsdTensorSpec& s = source.spec;
    Tensor t = Tensor::create(s.precision, MemoryType::EXTERNAL,
                              s.rows, s.cols, 1, 1,
                              const_cast<uint8_t*>(data));
    if (s.precision == Precision::INT8 || s.precision == Precision::INT4) {
        t.scales = reinterpret_cast<const float*>(scales);
        t.group_size = s.group_size;
        t.groups_per_row = s.groups_per_row;
        t.num_groups = static_cast<uint32_t>(s.rows) * s.groups_per_row;
        t.is_q4_repacked = (s.flags & MappedFile::FLAG_INT4_Q4DOT) != 0;
        t.is_q4_g128_packed = (s.flags & MappedFile::FLAG_INT4_BG128) != 0;
        if (t.is_q4_g128_packed) t.q4_g128_data = t.data;
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
                                const std::vector<int>& experts) {
    return request_many_impl(gate_up, down, experts, true);
}

bool MoeSsdCache::request_many_impl(const MoeSsdTensorSource* gate_up,
                                    const MoeSsdTensorSource* down,
                                    const std::vector<int>& experts,
                                    bool speculative) {
    if (!gate_up || !down || !valid_pair(gate_up, down, 0)) {
        std::fprintf(stderr, "MoE SSD: invalid expert pair request\n");
        return false;
    }
    const uint64_t trace_start = mollm_trace::now_ns();
    std::vector<uint8_t> seen((size_t)gate_up->spec.num_experts, 0);
    std::vector<Entry*> queued_entries;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!speculative) active_layer_ = gate_up->spec.layer;
        for (int expert : experts) {
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
                continue;
            }
            entry = reserve_entry_locked(gate_up, down, expert, speculative);
            // Do not reserve beyond the per-layer byte budget. acquire() will
            // submit this expert later after an in-flight slot becomes ready.
            if (!entry) continue;
            if (speculative) ++cross_layer_experts_;
            else ++misses_;
            queued_entries.push_back(entry);
        }
        enqueue_entry_reads_locked(queued_entries, speculative);
    }
    if (!queued_entries.empty()) io_cv_.notify_all();
    if (trace_start != 0) {
        mollm_trace::record_duration(
            speculative ? "ssd.predict" : "ssd",
            speculative ? "prefetch_many" : "request_many", trace_start, mollm_trace::now_ns(),
            "{\"layer\":" + std::to_string(gate_up->spec.layer) +
            ",\"requested\":" + std::to_string(experts.size()) +
            ",\"queued\":" + std::to_string(queued_entries.size()) + "}");
    }
    return true;
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
    if (!valid_pair(gate_up, down, expert)) {
        std::fprintf(stderr, "MoE SSD: invalid expert pair request\n");
        return false;
    }
    const uint64_t trace_start = mollm_trace::now_ns();
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
    if (entry && entry->state == Entry::State::Failed) {
        if (remove_entry_locked(entry, false))
            entry = nullptr;
    }
    while (!entry) {
        entry = reserve_entry_locked(gate_up, down, expert);
        if (entry) {
            ++misses_;
            enqueue_entry_reads_locked({entry});
            lock.unlock();
            io_cv_.notify_all();
            lock.lock();
            break;
        }
        // All slots are loading. Let one complete, then evict it if needed
        // and submit this deferred expert.
        ready_cv_.wait(lock);
        entry = find_entry_locked(gate_up, down, expert);
    }
    ready_cv_.wait(lock, [&] { return entry && !entry->is_loading(); });
    if (!entry || !entry->is_ready()) {
        std::fprintf(stderr, "MoE SSD: failed to load expert %d\n", expert);
        return false;
    }
    entry->used_at = ++clock_;
    entry->forward_epoch = forward_epoch_;
    if (entry->speculative) {
        ++cross_layer_used_;
        entry->speculative = false;
    }
    if (entry->fresh_miss) {
        entry->fresh_miss = false;
    } else {
        ++hits_;
    }
    gate_up_out = make_tensor(*gate_up, entry->gate_up_data.data(), entry->gate_up_scales.data());
    down_out = make_tensor(*down, entry->down_data.data(), entry->down_scales.data());
    if (trace_start != 0) {
        mollm_trace::record_duration(
            "ssd", "acquire", trace_start, mollm_trace::now_ns(),
            "{\"layer\":" + std::to_string(gate_up->spec.layer) +
            ",\"expert\":" + std::to_string(expert) + "}");
    }
    return true;
}

MoeSsdCache::Stats MoeSsdCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {hits_, misses_, evictions_, bytes_read_, cross_layer_tasks_count_,
            cross_layer_dropped_, cross_layer_experts_, cross_layer_used_, resident_bytes_};
}

void MoeSsdCache::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    hits_ = 0;
    misses_ = 0;
    evictions_ = 0;
    bytes_read_ = 0;
    cross_layer_tasks_count_ = 0;
    cross_layer_dropped_ = 0;
    cross_layer_experts_ = 0;
    cross_layer_used_ = 0;
}
