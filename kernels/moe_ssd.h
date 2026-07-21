#pragma once

#include "kernels/tensor.h"

#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// CPU-side, demand-paged storage for the two matrices belonging to a routed
// MoE expert.  The package converter emits byte-accurate slices for these
// matrices in `moe_expert_storage`; this class reads only selected slices with
// pread(2) and retains recently used expert pairs in an LRU cache.
struct MoeSsdTensorSpec {
    std::string weight_ref;
    int layer = -1;
    int num_experts = 0;
    int rows = 0;
    int cols = 0;
    Precision precision = Precision::FP16;
    uint32_t flags = 0;
    uint32_t group_size = 0;
    uint32_t groups_per_row = 0;
    uint64_t data_offset = 0;       // absolute package-file offset
    uint64_t data_bytes = 0;        // per-expert bytes
    uint64_t scales_offset = 0;     // absolute package-file offset
    uint64_t scales_bytes = 0;      // per-expert bytes
};

class MoeSsdCache;

// Attached to an otherwise data-less aggregate expert Tensor.  It is owned by
// MoeSsdCache and remains valid for the Engine lifetime.
struct MoeSsdTensorSource {
    MoeSsdTensorSpec spec;
    MoeSsdCache* cache = nullptr;
};

class MoeSsdCache {
public:
    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        uint64_t bytes_read = 0;
        size_t resident_bytes = 0;
    };

    MoeSsdCache();
    ~MoeSsdCache();
    MoeSsdCache(const MoeSsdCache&) = delete;
    MoeSsdCache& operator=(const MoeSsdCache&) = delete;

    bool open(const std::string& package_path, size_t capacity_bytes,
              int io_workers = 4);
    bool add_source(const MoeSsdTensorSpec& spec);
    const MoeSsdTensorSource* find_source(const std::string& weight_ref) const;

    // Fetch a single expert pair. The output Tensors borrow cache-owned
    // storage and are valid until the next acquire() which evicts that entry.
    bool acquire(const MoeSsdTensorSource* gate_up,
                 const MoeSsdTensorSource* down,
                 int expert,
                 Tensor& gate_up_out,
                 Tensor& down_out);

    // Schedule missing expert pairs on independent I/O workers. It returns as
    // soon as jobs are queued; acquire() waits only for the specific pair it
    // needs. Call this immediately after router top-k to overlap disk reads
    // with shared-expert work and cache-hit routed work.
    bool request_many(const MoeSsdTensorSource* gate_up,
                      const MoeSsdTensorSource* down,
                      const std::vector<int>& experts);

    size_t capacity_bytes() const { return capacity_bytes_; }
    Stats stats() const;
    void reset_stats();

private:
    struct Entry;
    struct IoJob { Entry* entry = nullptr; };

    bool valid_pair(const MoeSsdTensorSource* gate_up,
                    const MoeSsdTensorSource* down,
                    int expert) const;
    Entry* find_entry_locked(const MoeSsdTensorSource* gate_up,
                             const MoeSsdTensorSource* down, int expert);
    Entry* reserve_entry_locked(const MoeSsdTensorSource* gate_up,
                                const MoeSsdTensorSource* down, int expert);
    bool read_entry(Entry& entry);
    bool read_exact(uint64_t offset, void* dst, size_t bytes) const;
    void io_worker_main();
    void stop_io_workers();
    static Tensor make_tensor(const MoeSsdTensorSource& source,
                              const std::vector<uint8_t>& data,
                              const std::vector<uint8_t>& scales);

    int fd_ = -1;
    size_t capacity_bytes_ = 0;
    size_t resident_bytes_ = 0;
    uint64_t clock_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;
    uint64_t bytes_read_ = 0;
    mutable std::mutex mutex_;
    std::condition_variable io_cv_;
    std::condition_variable ready_cv_;
    bool stop_io_ = false;
    std::deque<IoJob> io_jobs_;
    std::vector<std::thread> io_workers_;
    std::unordered_map<std::string, MoeSsdTensorSource> sources_;
    std::unordered_set<int> layers_;
    std::vector<std::unique_ptr<Entry>> entries_;
};
