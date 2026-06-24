#include "kernels/threading.h"

#include <algorithm>

namespace {

bool compute_shard(int begin, int end, int grain_size,
                   int active_threads, int thread_id,
                   int& shard_begin, int& shard_end) {
    int total = end - begin;
    if (total <= 0 || active_threads <= 0 || thread_id < 0 || thread_id >= active_threads) {
        return false;
    }

    int chunk = std::max(grain_size, 1);
    int total_chunks = (total + chunk - 1) / chunk;
    int chunk_begin = (total_chunks * thread_id) / active_threads;
    int chunk_end = (total_chunks * (thread_id + 1)) / active_threads;

    shard_begin = begin + chunk_begin * chunk;
    shard_end = std::min(begin + chunk_end * chunk, end);
    return shard_begin < shard_end;
}

} // namespace

ThreadPool::ThreadPool(int num_threads) {
    resize(num_threads);
}

ThreadPool::~ThreadPool() {
    stop_workers();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void ThreadPool::stop_workers() {
    stop_.store(true, std::memory_order_release);
    // Wake all workers so they can exit
    for (int t = 0; t < num_threads_; t++) {
        if (worker_ready_) worker_ready_[t].store(true, std::memory_order_release);
    }
}

void ThreadPool::resize(int num_threads) {
    num_threads = std::max(num_threads, 1);
    if (num_threads == num_threads_) return;

    stop_workers();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    workers_.clear();
    worker_ready_.reset();

    stop_.store(false, std::memory_order_release);
    pending_workers_.store(0, std::memory_order_release);
    num_threads_ = num_threads;
    job_.fn = nullptr;
    job_.generation.store(0, std::memory_order_release);

    worker_ready_ = std::unique_ptr<std::atomic<bool>[]>(new std::atomic<bool>[num_threads]);
    for (int t = 0; t < num_threads; t++) {
        worker_ready_[t].store(false, std::memory_order_release);
    }

    for (int thread_id = 1; thread_id < num_threads_; thread_id++) {
        workers_.emplace_back(&ThreadPool::worker_loop, this, thread_id);
    }
}

void ThreadPool::parallel_for_impl(int begin, int end, int grain_size, ParallelForFn fn) {
    if (!fn || end <= begin) return;

    grain_size = std::max(grain_size, 1);
    int total = end - begin;
    int total_chunks = (total + grain_size - 1) / grain_size;
    int active_threads = std::min(num_threads_, std::max(total_chunks, 1));
    if (active_threads <= 1) {
        fn(0, begin, end);
        return;
    }

    // Set up job
    job_.begin = begin;
    job_.end = end;
    job_.grain_size = grain_size;
    job_.active_threads = active_threads;
    job_.fn = std::move(fn);
    size_t gen = job_.generation.load(std::memory_order_relaxed) + 1;
    job_.generation.store(gen, std::memory_order_release);

    // Signal workers: set their ready flags
    pending_workers_.store(active_threads - 1, std::memory_order_release);
    for (int t = 1; t < active_threads; t++) {
        worker_ready_[t].store(true, std::memory_order_release);
    }

    // Main thread does shard 0
    int shard_begin = begin, shard_end = end;
    if (compute_shard(begin, end, grain_size, active_threads, 0, shard_begin, shard_end)) {
        job_.fn(0, shard_begin, shard_end);
    }

    // Spin-wait for workers to finish
    while (pending_workers_.load(std::memory_order_acquire) > 0) {
        // busy wait — __builtin_ia32_pause equivalent on ARM is YIELD
        __asm__ __volatile__("yield" ::: "memory");
    }
    job_.fn = nullptr;
}

void ThreadPool::worker_loop(int thread_id) {
    size_t seen_generation = 0;

    while (true) {
        // Spin-wait for job
        while (true) {
            if (stop_.load(std::memory_order_acquire)) return;
            if (worker_ready_[thread_id].load(std::memory_order_acquire)) break;
            __asm__ __volatile__("yield" ::: "memory");
        }

        // Read job
        size_t gen = job_.generation.load(std::memory_order_acquire);
        if (gen == seen_generation) {
            // Already processed this generation, wait for next
            worker_ready_[thread_id].store(false, std::memory_order_release);
            continue;
        }
        seen_generation = gen;

        // Clear ready flag
        worker_ready_[thread_id].store(false, std::memory_order_release);

        if (thread_id >= job_.active_threads) {
            // Not needed for this job
            pending_workers_.fetch_sub(1, std::memory_order_acq_rel);
            continue;
        }

        int shard_begin = job_.begin, shard_end = job_.end;
        if (compute_shard(job_.begin, job_.end, job_.grain_size,
                          job_.active_threads, thread_id,
                          shard_begin, shard_end)) {
            job_.fn(thread_id, shard_begin, shard_end);
        }

        // Signal done
        pending_workers_.fetch_sub(1, std::memory_order_acq_rel);
    }
}
