#pragma once

#include "kernels/tensor.h"

#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
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

// Router metadata used by the advisory next-layer SSD prefetch predictor.
struct MoeSsdPredictConfig {
    int hidden_size = 0;
    int num_experts = 0;
    int top_k = 0;
    int router_score_func = 0;
    int n_group = 1;
    int topk_group = 1;
};

// Copy one decode gate input and schedule prediction of the next MoE layer's
// expert reads. The real next-layer router always recomputes the exact route.
bool schedule_moe_cross_layer_prefetch(
    const Tensor& gate_input,
    const Tensor& next_router,
    const Tensor* next_router_bias,
    const MoeSsdTensorSource* next_gate_up,
    const MoeSsdTensorSource* next_down,
    const MoeSsdPredictConfig& config);

class MoeSsdCache {
public:
    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        uint64_t bytes_read = 0;
        uint64_t cross_layer_tasks = 0;
        uint64_t cross_layer_dropped = 0;
        uint64_t cross_layer_experts = 0;
        uint64_t cross_layer_used = 0;
        uint64_t cross_layer_rejected = 0;
        size_t resident_bytes = 0;
        std::vector<uint64_t> cross_layer_rank_attempts;
        std::vector<uint64_t> cross_layer_rank_hits;
    };

    MoeSsdCache();
    ~MoeSsdCache();
    MoeSsdCache(const MoeSsdCache&) = delete;
    MoeSsdCache& operator=(const MoeSsdCache&) = delete;

    bool open(const std::string& package_path, size_t capacity_bytes,
              int io_workers = 8, bool enable_cross_layer_worker = false,
              bool lock_expert_pages = false);
    bool add_source(const MoeSsdTensorSpec& spec);
    // Switch from isolated per-layer quotas to one shared capacity pool.
    // Call before any expert I/O.
    bool set_global_capacity_pool(bool enabled);
    // Favor the first N MoE layers, while retaining one streaming slot for
    // every layer. Call after registering all expert sources and before I/O.
    bool configure_shallow_favoring(int shallow_layers);
    // Mark the start of a graph forward pass. The global cache uses this
    // boundary to distinguish stale entries from current/future entries.
    void begin_forward_pass();
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

    // Speculative reads are only serviced after real router requests.
    bool prefetch_many(const MoeSsdTensorSource* gate_up,
                       const MoeSsdTensorSource* down,
                       const std::vector<int>& experts,
                       const std::vector<float>& confidence = {},
                       size_t prefetch_count = static_cast<size_t>(-1));

    // Drop consistently low-value tail ranks once enough real-route feedback
    // has accumulated. The full prediction is still recorded for adaptation.
    size_t recommended_prefetch_count(size_t predicted_count) const;

    // Queue a tiny gate-prediction task on an otherwise-idle SSD worker. The
    // queue is bounded so stale predictions never accumulate behind decode.
    bool submit_cross_layer_task(std::function<void()> task);

    // A cross-layer prediction must not displace the next layer's real route
    // when its per-layer RAM window is too small for both candidate sets.
    bool can_prefetch_pairs(const MoeSsdTensorSource* gate_up,
                            const MoeSsdTensorSource* down,
                            size_t pairs) const;

    // Number of requested pairs currently resident or loading. Used by the
    // MoE kernel to detect when its route set exceeds the per-layer window.
    size_t resident_count(const MoeSsdTensorSource* gate_up,
                          const MoeSsdTensorSource* down,
                          const std::vector<int>& experts) const;

    bool contains(const MoeSsdTensorSource* gate_up,
                  const MoeSsdTensorSource* down,
                  int expert) const;

    // Drop a pair once its borrowed Tensor views are no longer in use. This
    // lets a small cache slide its async prefetch window forward.
    bool release(const MoeSsdTensorSource* gate_up,
                 const MoeSsdTensorSource* down,
                 int expert);

    size_t capacity_bytes() const { return capacity_bytes_; }
    size_t layer_capacity_bytes(int layer) const;
    Stats stats() const;
    void reset_stats();

private:
    struct Entry;
    struct ByteBuffer;
    struct LayerLayout {
        int num_experts = 0;
        size_t pair_bytes = 0;
    };
    struct PredictionRecord {
        uint64_t forward_epoch = 0;
        std::vector<int> experts;
    };
    // One job reads a physically contiguous run of the same component
    // (gate data/scales or down data/scales) for adjacent expert ids. Keeping
    // components separate exposes a deep queue even for a single decode token.
    struct IoJob {
        std::vector<Entry*> entries;
        uint8_t component = 0;
        uint64_t trace_id = 0;
        bool speculative = false;
    };

    bool valid_pair(const MoeSsdTensorSource* gate_up,
                    const MoeSsdTensorSource* down,
                    int expert) const;
    bool request_many_impl(const MoeSsdTensorSource* gate_up,
                           const MoeSsdTensorSource* down,
                           const std::vector<int>& experts,
                           bool speculative,
                           const std::vector<float>& confidence = {},
                           size_t request_count = static_cast<size_t>(-1));
    Entry* find_entry_locked(const MoeSsdTensorSource* gate_up,
                             const MoeSsdTensorSource* down, int expert);
    const Entry* find_entry_locked(const MoeSsdTensorSource* gate_up,
                                   const MoeSsdTensorSource* down,
                                   int expert) const;
    Entry* reserve_entry_locked(const MoeSsdTensorSource* gate_up,
                                const MoeSsdTensorSource* down, int expert,
                                bool speculative = false,
                                float prediction_confidence = 0.0f);
    std::unique_ptr<Entry> remove_entry_locked(Entry* entry,
                                               bool count_eviction);
    size_t layer_capacity_bytes_locked(int layer) const;
    bool global_victim_before_locked(const Entry* candidate, const Entry* current) const;
    static ByteBuffer& component_buffer(Entry& entry, uint8_t component);
    static const ByteBuffer& component_buffer(const Entry& entry, uint8_t component);
    static uint64_t component_offset(const Entry& entry, uint8_t component);
    void enqueue_entry_reads_locked(const std::vector<Entry*>& entries,
                                    bool low_priority = false);
    bool read_job(const IoJob& job);
    bool read_exact(uint64_t offset, void* dst, size_t bytes) const;
    void io_worker_main(int worker_index);
    void cross_layer_worker_main();
    void stop_io_workers();
    static Tensor make_tensor(const MoeSsdTensorSource& source,
                              const uint8_t* data,
                              const uint8_t* scales);

    int fd_ = -1;
    uint64_t file_size_ = 0;
    int io_workers_count_ = 0;
    uint64_t next_trace_id_ = 1;
    size_t capacity_bytes_ = 0;
    size_t resident_bytes_ = 0;
    uint64_t clock_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;
    uint64_t bytes_read_ = 0;
    uint64_t cross_layer_tasks_count_ = 0;
    uint64_t cross_layer_dropped_ = 0;
    uint64_t cross_layer_experts_ = 0;
    uint64_t cross_layer_used_ = 0;
    uint64_t cross_layer_rejected_ = 0;
    std::vector<uint64_t> cross_layer_rank_attempts_;
    std::vector<uint64_t> cross_layer_rank_hits_;
    std::vector<uint64_t> prediction_policy_attempts_;
    std::vector<uint64_t> prediction_policy_hits_;
    mutable std::mutex mutex_;
    std::condition_variable io_cv_;
    std::condition_variable cross_layer_cv_;
    std::condition_variable ready_cv_;
    bool stop_io_ = false;
    bool lock_expert_pages_ = false;
    std::deque<IoJob> io_jobs_;
    std::deque<IoJob> low_priority_io_jobs_;
    std::deque<std::function<void()>> cross_layer_tasks_;
    std::vector<std::thread> io_workers_;
    std::thread cross_layer_worker_;
    std::unordered_map<std::string, MoeSsdTensorSource> sources_;
    std::unordered_set<int> layers_;
    std::unordered_map<int, LayerLayout> layer_layouts_;
    // Empty means the legacy equal-per-layer layout. Otherwise each value is
    // a fixed per-layer quota computed by configure_shallow_favoring().
    std::unordered_map<int, size_t> layer_capacity_bytes_;
    bool global_capacity_pool_ = false;
    int shallow_favoring_layers_ = 0;
    uint64_t forward_epoch_ = 1;
    int active_layer_ = -1;
    using EntryList = std::list<std::unique_ptr<Entry>>;
    EntryList entries_;
    std::unordered_map<Entry*, EntryList::iterator> entry_locations_;
    // Expert residency is partitioned by model layer. Keep that partition in
    // the cache too: a 16+ GiB cache can have thousands of global entries,
    // while one MoE layer only needs to inspect its own small route window.
    std::unordered_map<int, std::vector<Entry*>> layer_entries_;
    std::unordered_map<int, size_t> layer_resident_bytes_;
    std::unordered_map<int, PredictionRecord> pending_predictions_;
};
