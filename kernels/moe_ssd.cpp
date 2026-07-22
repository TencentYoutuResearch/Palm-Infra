#include "kernels/moe_ssd.h"

#include "graph/mmap_file.h"
#include "kernels/trace.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

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
    const MoeSsdTensorSource* gate_up = nullptr;
    const MoeSsdTensorSource* down = nullptr;
    int expert = -1;
    uint64_t used_at = 0;
    bool loading = false;
    bool ready = false;
    bool failed = false;
    int pending_reads = 0;
    bool fresh_miss = false;  // first acquire after a queued miss is not a hit
    bool speculative = false;
    ByteBuffer gate_up_data;
    ByteBuffer gate_up_scales;
    ByteBuffer down_data;
    ByteBuffer down_scales;

    size_t bytes() const {
        return gate_up_data.size() + gate_up_scales.size() +
               down_data.size() + down_scales.size();
    }
};

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
        io_workers_count_ = io_workers;
        next_trace_id_ = 1;
        capacity_bytes_ = capacity_bytes;
        resident_bytes_ = 0;
        clock_ = hits_ = misses_ = evictions_ = bytes_read_ = 0;
        cross_layer_tasks_count_ = cross_layer_dropped_ = 0;
        cross_layer_experts_ = cross_layer_used_ = 0;
        sources_.clear();
        layers_.clear();
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
    if (fd_ < 0 || spec.weight_ref.empty() || spec.layer < 0 ||
        spec.num_experts <= 0 || spec.rows <= 0 || spec.cols <= 0 ||
        spec.data_bytes == 0) {
        std::fprintf(stderr, "MoE SSD: invalid expert storage metadata for %s\n",
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
    MoeSsdTensorSource source;
    source.spec = spec;
    source.cache = this;
    std::lock_guard<std::mutex> lock(mutex_);
    auto inserted = sources_.emplace(spec.weight_ref, std::move(source));
    if (!inserted.second) {
        std::fprintf(stderr, "MoE SSD: duplicate expert storage metadata for %s\n",
                     spec.weight_ref.c_str());
        return false;
    }
    layers_.insert(spec.layer);
    return true;
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
    auto layer = layer_entries_.find(gate_up->spec.layer);
    if (layer == layer_entries_.end()) return nullptr;
    for (Entry* entry : layer->second) {
        if (entry->gate_up == gate_up && entry->down == down && entry->expert == expert) {
            return entry;
        }
    }
    return nullptr;
}

MoeSsdCache::Entry* MoeSsdCache::reserve_entry_locked(
    const MoeSsdTensorSource* gate_up, const MoeSsdTensorSource* down, int expert,
    bool speculative) {
    const size_t required = static_cast<size_t>(gate_up->spec.data_bytes) +
                            static_cast<size_t>(gate_up->spec.scales_bytes) +
                            static_cast<size_t>(down->spec.data_bytes) +
                            static_cast<size_t>(down->spec.scales_bytes);
    const size_t layer_capacity = layers_.empty() ? capacity_bytes_
        : capacity_bytes_ / layers_.size();
    if (required > layer_capacity) {
        std::fprintf(stderr,
                     "MoE SSD: one expert pair needs %.1f MB but per-layer cache holds %.1f MB\n",
                     required / 1e6, layer_capacity / 1e6);
        return nullptr;
    }

    const int layer = gate_up->spec.layer;
    size_t& layer_resident = layer_resident_bytes_[layer];
    std::vector<Entry*>& layer_entries = layer_entries_[layer];
    std::unique_ptr<Entry> recycled;
    while (layer_resident + required > layer_capacity) {
        auto victim = std::min_element(layer_entries.begin(), layer_entries.end(),
            [](const Entry* a, const Entry* b) {
                if (a->loading != b->loading) return !a->loading;
                return a->used_at < b->used_at;
            });
        if (victim == layer_entries.end() || (*victim)->loading) {
            // The asynchronous request window for this layer is full. The
            // caller can let workers finish and retry later.
            return nullptr;
        }
        Entry* victim_entry = *victim;
        layer_resident -= victim_entry->bytes();
        resident_bytes_ -= victim_entry->bytes();
        auto location = entry_locations_.find(victim_entry);
        if (location == entry_locations_.end()) return nullptr;
        if (!recycled) recycled = std::move(*location->second);
        entries_.erase(location->second);
        entry_locations_.erase(location);
        layer_entries.erase(victim);
        ++evictions_;
    }

    auto entry = recycled ? std::move(recycled) : std::make_unique<Entry>();
    entry->gate_up = gate_up;
    entry->down = down;
    entry->expert = expert;
    entry->used_at = ++clock_;
    entry->loading = true;
    entry->ready = false;
    entry->failed = false;
    entry->pending_reads = 0;
    entry->fresh_miss = true;
    entry->speculative = speculative;
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

bool MoeSsdCache::read_exact(uint64_t offset, void* dst, size_t bytes) const {
    uint8_t* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd_, out + done, bytes - done,
                          static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            std::fprintf(stderr, "MoE SSD: pread failed at offset %llu: %s\n",
                         (unsigned long long)(offset + done),
                         n == 0 ? "unexpected EOF" : std::strerror(errno));
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

const MoeSsdCache::ByteBuffer& MoeSsdCache::component_buffer(const Entry& entry,
                                                              uint8_t component) {
    switch (component) {
    case 0: return entry.gate_up_data;
    case 1: return entry.gate_up_scales;
    case 2: return entry.down_data;
    default: return entry.down_scales;
    }
}

MoeSsdCache::ByteBuffer& MoeSsdCache::component_buffer(Entry& entry, uint8_t component) {
    return const_cast<MoeSsdCache::ByteBuffer&>(component_buffer(
        static_cast<const Entry&>(entry), component));
}

uint64_t MoeSsdCache::component_offset(const Entry& entry, uint8_t component) {
    const uint64_t expert = static_cast<uint64_t>(entry.expert);
    switch (component) {
    case 0:
        return entry.gate_up->spec.data_offset + expert * entry.gate_up->spec.data_bytes;
    case 1:
        return entry.gate_up->spec.scales_offset + expert * entry.gate_up->spec.scales_bytes;
    case 2:
        return entry.down->spec.data_offset + expert * entry.down->spec.data_bytes;
    default:
        return entry.down->spec.scales_offset + expert * entry.down->spec.scales_bytes;
    }
}

void MoeSsdCache::enqueue_entry_reads_locked(const std::vector<Entry*>& entries,
                                              bool low_priority) {
    if (entries.empty()) return;
    std::vector<Entry*> sorted = entries;
    std::sort(sorted.begin(), sorted.end(),
              [](const Entry* a, const Entry* b) { return a->expert < b->expert; });

    // Four independent file regions make up one expert pair. Keep enough
    // component jobs in flight to fill the I/O workers, but queue only a
    // small expert batch at a time so the first pairs become ready while the
    // main thread is already computing them. A single all-expert batch made
    // every pair wait behind the final down/scales reads and lost that overlap.
    const size_t batch_entries = std::max<size_t>(1, (size_t)io_workers_count_ / 4);
    for (size_t batch_begin = 0; batch_begin < sorted.size(); batch_begin += batch_entries) {
        const size_t batch_end = std::min(sorted.size(), batch_begin + batch_entries);
        // Group only truly adjacent expert ids. Their slices are contiguous in
        // the package, so one larger pread can feed every entry in the run.
        for (uint8_t component = 0; component < 4; component++) {
            size_t begin = batch_begin;
            while (begin < batch_end) {
                if (component_buffer(*sorted[begin], component).empty()) {
                    ++begin;
                    continue;
                }
                size_t end = begin + 1;
                while (end < batch_end &&
                       !component_buffer(*sorted[end], component).empty() &&
                       sorted[end]->expert == sorted[end - 1]->expert + 1) {
                    ++end;
                }
                IoJob job;
                job.component = component;
                job.trace_id = next_trace_id_++;
                job.speculative = low_priority;
                job.entries.assign(sorted.begin() + static_cast<ptrdiff_t>(begin),
                                   sorted.begin() + static_cast<ptrdiff_t>(end));
                for (Entry* entry : job.entries) ++entry->pending_reads;
                if (mollm_trace::enabled()) {
                    const Entry& first = *job.entries.front();
                    mollm_trace::record_flow(
                        "ssd.io", "queued_read", mollm_trace::now_ns(), job.trace_id, true,
                        "{\"layer\":" + std::to_string(first.gate_up->spec.layer) +
                        ",\"first_expert\":" + std::to_string(first.expert) +
                        ",\"experts\":" + std::to_string(job.entries.size()) + "}");
                }
                if (low_priority) low_priority_io_jobs_.push_back(std::move(job));
                else io_jobs_.push_back(std::move(job));
                begin = end;
            }
        }
    }
}

bool MoeSsdCache::read_job(const IoJob& job) {
    if (job.entries.empty()) return false;
    const size_t bytes_per_entry = component_buffer(*job.entries.front(), job.component).size();
    if (bytes_per_entry == 0) return true;
    const uint64_t offset = component_offset(*job.entries.front(), job.component);
    if (job.entries.size() == 1) {
        ByteBuffer& dst = component_buffer(*job.entries.front(), job.component);
        return read_exact(offset, dst.data(), dst.size());
    }

    const size_t merged_bytes = bytes_per_entry * job.entries.size();
    std::unique_ptr<uint8_t[]> merged(new uint8_t[merged_bytes]);
    if (!read_exact(offset, merged.get(), merged_bytes)) return false;
    for (size_t i = 0; i < job.entries.size(); i++) {
        ByteBuffer& dst = component_buffer(*job.entries[i], job.component);
        std::memcpy(dst.data(), merged.get() + i * bytes_per_entry, bytes_per_entry);
    }
    return true;
}

void MoeSsdCache::io_worker_main(int worker_index) {
    if (mollm_trace::enabled()) {
        const std::string name = "ssd-io-" + std::to_string(worker_index);
        mollm_trace::set_thread_name(name.c_str());
    }
    for (;;) {
        IoJob job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            io_cv_.wait(lock, [&] {
                return stop_io_ || !io_jobs_.empty() || !low_priority_io_jobs_.empty();
            });
            if (stop_io_ && io_jobs_.empty() && low_priority_io_jobs_.empty()) return;
            if (!io_jobs_.empty()) {
                job = std::move(io_jobs_.front());
                io_jobs_.pop_front();
            } else {
                job = std::move(low_priority_io_jobs_.front());
                low_priority_io_jobs_.pop_front();
            }
        }
        std::string trace_args;
        if (mollm_trace::enabled() && !job.entries.empty()) {
            const Entry& first = *job.entries.front();
            const char* component = job.component == 0 ? "gate_data"
                                  : job.component == 1 ? "gate_scales"
                                  : job.component == 2 ? "down_data" : "down_scales";
            trace_args = "{\"layer\":" + std::to_string(first.gate_up->spec.layer) +
                         ",\"first_expert\":" + std::to_string(first.expert) +
                         ",\"experts\":" + std::to_string(job.entries.size()) +
                         ",\"component\":\"" + component + "\""
                         ",\"kind\":\"" + (job.speculative ? "prefetch" : "route") + "\"}";
        }
        bool ok = false;
        {
            mollm_trace::ScopedEvent trace_event(
                "ssd.io", job.speculative ? "pread.prefetch" : "pread.route", trace_args,
                job.speculative ? "yellow" : "good");
            ok = read_job(job);
        }
        if (job.trace_id != 0) {
            mollm_trace::record_flow("ssd.io", "queued_read", mollm_trace::now_ns(),
                                     job.trace_id, false, trace_args);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (Entry* entry : job.entries) {
                if (entry->pending_reads > 0) --entry->pending_reads;
                if (!ok) entry->failed = true;
                if (entry->pending_reads == 0) {
                    entry->loading = false;
                    entry->ready = !entry->failed;
                    if (entry->ready) bytes_read_ += entry->bytes();
                }
            }
        }
        ready_cv_.notify_all();
    }
}

void MoeSsdCache::cross_layer_worker_main() {
    if (mollm_trace::enabled()) mollm_trace::set_thread_name("ssd-predict");
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cross_layer_cv_.wait(lock, [&] { return stop_io_ || !cross_layer_tasks_.empty(); });
            if (stop_io_) return;
            task = std::move(cross_layer_tasks_.front());
            cross_layer_tasks_.pop_front();
        }
        mollm_trace::ScopedEvent trace_event("ssd.predict", "cross_layer_gate");
        task();
    }
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
        for (int expert : experts) {
            if (expert < 0 || expert >= gate_up->spec.num_experts || seen[(size_t)expert]) continue;
            seen[(size_t)expert] = 1;
            Entry* entry = find_entry_locked(gate_up, down, expert);
            if (entry) {
                entry->used_at = ++clock_;
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
    const size_t pair_bytes = static_cast<size_t>(gate_up->spec.data_bytes) +
                              static_cast<size_t>(gate_up->spec.scales_bytes) +
                              static_cast<size_t>(down->spec.data_bytes) +
                              static_cast<size_t>(down->spec.scales_bytes);
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t layer_capacity = layers_.empty() ? capacity_bytes_
        : capacity_bytes_ / layers_.size();
    return pair_bytes <= layer_capacity / pairs;
}

size_t MoeSsdCache::resident_count(const MoeSsdTensorSource* gate_up,
                                   const MoeSsdTensorSource* down,
                                   const std::vector<int>& experts) const {
    if (!valid_pair(gate_up, down, 0)) return 0;
    std::vector<uint8_t> seen((size_t)gate_up->spec.num_experts, 0);
    size_t count = 0;
    std::lock_guard<std::mutex> lock(mutex_);
    auto layer = layer_entries_.find(gate_up->spec.layer);
    if (layer == layer_entries_.end()) return 0;
    for (int expert : experts) {
        if (expert < 0 || expert >= gate_up->spec.num_experts || seen[(size_t)expert]) continue;
        seen[(size_t)expert] = 1;
        for (const Entry* entry : layer->second) {
            if (entry->gate_up == gate_up && entry->down == down && entry->expert == expert) {
                ++count;
                break;
            }
        }
    }
    return count;
}

bool MoeSsdCache::contains(const MoeSsdTensorSource* gate_up,
                           const MoeSsdTensorSource* down,
                           int expert) const {
    if (!valid_pair(gate_up, down, expert)) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto layer = layer_entries_.find(gate_up->spec.layer);
    if (layer == layer_entries_.end()) return false;
    for (const Entry* entry : layer->second) {
        if (entry->gate_up == gate_up && entry->down == down && entry->expert == expert) {
            return true;
        }
    }
    return false;
}

bool MoeSsdCache::release(const MoeSsdTensorSource* gate_up,
                          const MoeSsdTensorSource* down,
                          int expert) {
    if (!valid_pair(gate_up, down, expert)) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto layer = layer_entries_.find(gate_up->spec.layer);
    if (layer == layer_entries_.end()) return false;
    for (auto it = layer->second.begin(); it != layer->second.end(); ++it) {
        Entry* entry = *it;
        if (entry->gate_up != gate_up || entry->down != down || entry->expert != expert) continue;
        // release() is called only after routed compute has finished. Do not
        // invalidate an in-flight asynchronous read if a caller violates that
        // lifetime contract.
        if (entry->loading) return false;
        const size_t bytes = entry->bytes();
        resident_bytes_ -= bytes;
        layer_resident_bytes_[gate_up->spec.layer] -= bytes;
        auto location = entry_locations_.find(entry);
        if (location == entry_locations_.end()) return false;
        entries_.erase(location->second);
        entry_locations_.erase(location);
        layer->second.erase(it);
        ++evictions_;
        return true;
    }
    return false;
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
    const size_t required = static_cast<size_t>(gate_up->spec.data_bytes) +
                            static_cast<size_t>(gate_up->spec.scales_bytes) +
                            static_cast<size_t>(down->spec.data_bytes) +
                            static_cast<size_t>(down->spec.scales_bytes);
    const size_t layer_capacity = layers_.empty() ? capacity_bytes_
        : capacity_bytes_ / layers_.size();
    if (required > layer_capacity) {
        std::fprintf(stderr,
                     "MoE SSD: one expert pair needs %.1f MB but per-layer cache holds %.1f MB\n",
                     required / 1e6, layer_capacity / 1e6);
        return false;
    }
    Entry* entry = find_entry_locked(gate_up, down, expert);
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
    ready_cv_.wait(lock, [&] { return entry && !entry->loading; });
    if (!entry || !entry->ready || entry->failed) {
        std::fprintf(stderr, "MoE SSD: failed to load expert %d\n", expert);
        return false;
    }
    entry->used_at = ++clock_;
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
