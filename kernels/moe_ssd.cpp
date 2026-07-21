#include "kernels/moe_ssd.h"

#include "graph/mmap_file.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

struct MoeSsdCache::Entry {
    const MoeSsdTensorSource* gate_up = nullptr;
    const MoeSsdTensorSource* down = nullptr;
    int expert = -1;
    uint64_t used_at = 0;
    bool loading = false;
    bool ready = false;
    bool failed = false;
    bool fresh_miss = false;  // first acquire after a queued miss is not a hit
    std::vector<uint8_t> gate_up_data;
    std::vector<uint8_t> gate_up_scales;
    std::vector<uint8_t> down_data;
    std::vector<uint8_t> down_scales;

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
    for (std::thread& worker : io_workers_) {
        if (worker.joinable()) worker.join();
    }
    io_workers_.clear();
    std::lock_guard<std::mutex> lock(mutex_);
    io_jobs_.clear();
    stop_io_ = false;
}

bool MoeSsdCache::open(const std::string& package_path, size_t capacity_bytes,
                       int io_workers) {
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
        capacity_bytes_ = capacity_bytes;
        resident_bytes_ = 0;
        clock_ = hits_ = misses_ = evictions_ = bytes_read_ = 0;
        sources_.clear();
        layers_.clear();
        entries_.clear();
    }
    io_workers_.reserve((size_t)io_workers);
    for (int i = 0; i < io_workers; i++) {
        io_workers_.emplace_back(&MoeSsdCache::io_worker_main, this);
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
    for (const auto& entry : entries_) {
        if (entry->gate_up == gate_up && entry->down == down && entry->expert == expert) {
            return entry.get();
        }
    }
    return nullptr;
}

MoeSsdCache::Entry* MoeSsdCache::reserve_entry_locked(
    const MoeSsdTensorSource* gate_up, const MoeSsdTensorSource* down, int expert) {
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

    size_t layer_resident = 0;
    for (const auto& entry : entries_) {
        if (entry->gate_up->spec.layer == gate_up->spec.layer) {
            layer_resident += entry->bytes();
        }
    }
    std::unique_ptr<Entry> recycled;
    while (layer_resident + required > layer_capacity) {
        auto victim = std::min_element(entries_.begin(), entries_.end(),
            [&](const std::unique_ptr<Entry>& a, const std::unique_ptr<Entry>& b) {
                const bool a_here = a->gate_up->spec.layer == gate_up->spec.layer;
                const bool b_here = b->gate_up->spec.layer == gate_up->spec.layer;
                if (a_here != b_here) return a_here;
                if (!a_here) return false;
                if (a->loading != b->loading) return !a->loading;
                return a->used_at < b->used_at;
            });
        if (victim == entries_.end() ||
            (*victim)->gate_up->spec.layer != gate_up->spec.layer || (*victim)->loading) {
            // The asynchronous request window for this layer is full. The
            // caller can let workers finish and retry later.
            return nullptr;
        }
        layer_resident -= (*victim)->bytes();
        resident_bytes_ -= (*victim)->bytes();
        if (!recycled) recycled = std::move(*victim);
        entries_.erase(victim);
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
    entry->fresh_miss = true;
    entry->gate_up_data.resize(static_cast<size_t>(gate_up->spec.data_bytes));
    entry->gate_up_scales.resize(static_cast<size_t>(gate_up->spec.scales_bytes));
    entry->down_data.resize(static_cast<size_t>(down->spec.data_bytes));
    entry->down_scales.resize(static_cast<size_t>(down->spec.scales_bytes));
    resident_bytes_ += entry->bytes();
    Entry* raw = entry.get();
    entries_.push_back(std::move(entry));
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

bool MoeSsdCache::read_entry(Entry& entry) {
    const uint64_t e = static_cast<uint64_t>(entry.expert);
    return read_exact(entry.gate_up->spec.data_offset + e * entry.gate_up->spec.data_bytes,
                      entry.gate_up_data.data(), entry.gate_up_data.size()) &&
           (entry.gate_up_scales.empty() ||
            read_exact(entry.gate_up->spec.scales_offset + e * entry.gate_up->spec.scales_bytes,
                       entry.gate_up_scales.data(), entry.gate_up_scales.size())) &&
           read_exact(entry.down->spec.data_offset + e * entry.down->spec.data_bytes,
                      entry.down_data.data(), entry.down_data.size()) &&
           (entry.down_scales.empty() ||
            read_exact(entry.down->spec.scales_offset + e * entry.down->spec.scales_bytes,
                       entry.down_scales.data(), entry.down_scales.size()));
}

void MoeSsdCache::io_worker_main() {
    for (;;) {
        IoJob job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            io_cv_.wait(lock, [&] { return stop_io_ || !io_jobs_.empty(); });
            if (stop_io_ && io_jobs_.empty()) return;
            job = io_jobs_.front();
            io_jobs_.pop_front();
        }
        bool ok = job.entry && read_entry(*job.entry);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job.entry->loading = false;
            job.entry->ready = ok;
            job.entry->failed = !ok;
            if (ok) bytes_read_ += job.entry->bytes();
        }
        ready_cv_.notify_all();
    }
}

Tensor MoeSsdCache::make_tensor(const MoeSsdTensorSource& source,
                                 const std::vector<uint8_t>& data,
                                 const std::vector<uint8_t>& scales) {
    const MoeSsdTensorSpec& s = source.spec;
    Tensor t = Tensor::create(s.precision, MemoryType::EXTERNAL,
                              s.rows, s.cols, 1, 1,
                              const_cast<uint8_t*>(data.data()));
    if (s.precision == Precision::INT8 || s.precision == Precision::INT4) {
        t.scales = reinterpret_cast<const float*>(scales.data());
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
    if (!gate_up || !down || !valid_pair(gate_up, down, 0)) {
        std::fprintf(stderr, "MoE SSD: invalid expert pair request\n");
        return false;
    }
    std::vector<uint8_t> seen((size_t)gate_up->spec.num_experts, 0);
    bool queued = false;
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
            entry = reserve_entry_locked(gate_up, down, expert);
            // Do not reserve beyond the per-layer byte budget. acquire() will
            // submit this expert later after an in-flight slot becomes ready.
            if (!entry) continue;
            io_jobs_.push_back({entry});
            ++misses_;
            queued = true;
        }
    }
    if (queued) io_cv_.notify_all();
    return true;
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
            io_jobs_.push_back({entry});
            ++misses_;
            lock.unlock();
            io_cv_.notify_one();
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
    if (entry->fresh_miss) {
        entry->fresh_miss = false;
    } else {
        ++hits_;
    }
    gate_up_out = make_tensor(*gate_up, entry->gate_up_data, entry->gate_up_scales);
    down_out = make_tensor(*down, entry->down_data, entry->down_scales);
    return true;
}

MoeSsdCache::Stats MoeSsdCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {hits_, misses_, evictions_, bytes_read_, resident_bytes_};
}

void MoeSsdCache::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    hits_ = 0;
    misses_ = 0;
    evictions_ = 0;
    bytes_read_ = 0;
}
