#include "kernels/threading.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

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
    // Wake all spinning workers via the ready flags.
    for (int t = 0; t < num_threads_; t++) {
        if (worker_ready_) worker_ready_[t].store(true, std::memory_order_release);
    }
    // Wake any workers blocked on park_cv_.wait. Without this notify, the
    // worker join in ~ThreadPool / resize() would block forever if a worker
    // was parked when stop was requested.
    {
        std::lock_guard<std::mutex> lk(park_mtx_);
        // No state change needed — the wait predicate already checks stop_.
    }
    park_cv_.notify_all();
}

void ThreadPool::ensure_workers_started() {
    if (workers_started_) return;
    workers_started_ = true;
    parked_.store(false, std::memory_order_release);
    for (int thread_id = 1; thread_id < num_threads_; thread_id++) {
        workers_.emplace_back(&ThreadPool::worker_loop, this, thread_id);
    }
}

void ThreadPool::park() {
    // Writers to parked_ take the lock so that workers re-checking parked_
    // inside park_cv_.wait's predicate (also under the lock) never miss a
    // transition. Readers in worker_loop (line 279) and parallel_for_impl
    // use atomic loads so they don't race with this write.
    std::lock_guard<std::mutex> lk(park_mtx_);
    parked_.store(true, std::memory_order_release);
}

void ThreadPool::resume() {
    {
        std::lock_guard<std::mutex> lk(park_mtx_);
        parked_.store(false, std::memory_order_release);
    }
    park_cv_.notify_all();
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
    workers_started_ = false;
    job_.fn = nullptr;
    job_.generation.store(0, std::memory_order_release);

    worker_ready_ = std::unique_ptr<std::atomic<bool>[]>(new std::atomic<bool>[num_threads]);
    for (int t = 0; t < num_threads; t++) {
        worker_ready_[t].store(false, std::memory_order_release);
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

    // Lazy start + auto-resume if parked.
    ensure_workers_started();
    if (parked_.load(std::memory_order_acquire)) resume();

    // Set up job
    job_.begin = begin;
    job_.end = end;
    job_.grain_size = grain_size;
    job_.active_threads = active_threads;
    job_.fn = std::move(fn);
    job_.total_2d_jobs = 0;  // mark as 1D job
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

    // Spin-wait for workers to finish.
    while (pending_workers_.load(std::memory_order_acquire) > 0) {
        __asm__ __volatile__("yield" ::: "memory");
    }
    job_.fn = nullptr;
}

// Helper: decode 2D job index to [m_begin,m_end) × [n_begin,n_end).
static inline void decode_2d_job(int job_idx,
                                   int m_total, int m_tile_size,
                                   int n_total, int n_block_size,
                                   int& m_begin, int& m_end,
                                   int& n_begin, int& n_end) {
    int m_tiles = (m_total + m_tile_size - 1) / m_tile_size;
    int m_idx = job_idx % m_tiles;
    int n_idx = job_idx / m_tiles;
    m_begin = m_idx * m_tile_size;
    m_end = std::min(m_begin + m_tile_size, m_total);
    n_begin = n_idx * n_block_size;
    n_end = std::min(n_begin + n_block_size, n_total);
}

void ThreadPool::parallel_for_2d_impl(int m_total, int m_tile_size,
                                        int n_total, int n_block_size,
                                        ParallelFor2dFn fn) {
    if (!fn || m_total <= 0 || n_total <= 0) return;

    m_tile_size = std::max(m_tile_size, 1);
    n_block_size = std::max(n_block_size, 1);
    int m_tiles = (m_total + m_tile_size - 1) / m_tile_size;
    int n_blocks = (n_total + n_block_size - 1) / n_block_size;
    int total_jobs = m_tiles * n_blocks;
    int active_threads = std::min(num_threads_, total_jobs);
    if (active_threads <= 1) {
        int m_begin, m_end, n_begin, n_end;
        decode_2d_job(0, m_total, m_tile_size, n_total, n_block_size,
                       m_begin, m_end, n_begin, n_end);
        // Single-threaded: do all jobs sequentially.
        for (int j = 0; j < total_jobs; j++) {
            decode_2d_job(j, m_total, m_tile_size, n_total, n_block_size,
                          m_begin, m_end, n_begin, n_end);
            fn(0, m_begin, m_end, n_begin, n_end);
        }
        return;
    }

    ensure_workers_started();
    if (parked_.load(std::memory_order_acquire)) resume();

    // Set up 2D job
    job_.m_total = m_total;
    job_.m_tile_size = m_tile_size;
    job_.n_total = n_total;
    job_.n_block_size = n_block_size;
    job_.total_2d_jobs = total_jobs;
    job_.fn_2d = std::move(fn);
    job_.active_threads = active_threads;
    job_.current_chunk.store(0, std::memory_order_release);
    size_t gen = job_.generation.load(std::memory_order_relaxed) + 1;
    job_.generation.store(gen, std::memory_order_release);

    // Signal workers
    pending_workers_.store(active_threads - 1, std::memory_order_release);
    for (int t = 1; t < active_threads; t++) {
        worker_ready_[t].store(true, std::memory_order_release);
    }

    // Main thread starts at chunk = active_threads (workers grab 0..active_threads-1).
    // Actually: workers will spin up and atomic-fetch-add from 0. Main also
    // participates from a high index to avoid stealing early jobs.
    int job_idx = job_.current_chunk.fetch_add(1, std::memory_order_acq_rel);
    while (job_idx < total_jobs) {
        int m_begin, m_end, n_begin, n_end;
        decode_2d_job(job_idx, m_total, m_tile_size, n_total, n_block_size,
                       m_begin, m_end, n_begin, n_end);
        job_.fn_2d(0, m_begin, m_end, n_begin, n_end);
        job_idx = job_.current_chunk.fetch_add(1, std::memory_order_acq_rel);
    }

    // Spin-wait for workers
    while (pending_workers_.load(std::memory_order_acquire) > 0) {
        __asm__ __volatile__("yield" ::: "memory");
    }
    job_.fn_2d = nullptr;
}

void ThreadPool::worker_loop(int thread_id) {
    size_t seen_generation = 0;

    while (true) {
        // Spin-wait for job: pure spin for minimum dispatch latency.
        // Idle CPU is handled by explicit park()/resume() between generation
        // steps, so workers only exist while inference is active.
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
            goto wait_park;
        }

        if (job_.total_2d_jobs > 0) {
            // 2D atomic-steal: grab jobs until exhausted.
            int job_idx = job_.current_chunk.fetch_add(1, std::memory_order_acq_rel);
            while (job_idx < job_.total_2d_jobs) {
                int m_begin, m_end, n_begin, n_end;
                decode_2d_job(job_idx, job_.m_total, job_.m_tile_size,
                              job_.n_total, job_.n_block_size,
                              m_begin, m_end, n_begin, n_end);
                job_.fn_2d(thread_id, m_begin, m_end, n_begin, n_end);
                job_idx = job_.current_chunk.fetch_add(1, std::memory_order_acq_rel);
            }
        } else {
            // 1D static shard
            int shard_begin = job_.begin, shard_end = job_.end;
            if (compute_shard(job_.begin, job_.end, job_.grain_size,
                              job_.active_threads, thread_id,
                              shard_begin, shard_end)) {
                job_.fn(thread_id, shard_begin, shard_end);
            }
        }

        // Signal done
        pending_workers_.fetch_sub(1, std::memory_order_acq_rel);

    wait_park:
        // If parked, block until resume() is called. This drops idle CPU
        // to 0% between generation steps without sacrificing dispatch
        // latency during active inference.
        //
        // We re-check parked_ inside the lock so the cv predicate and the
        // writer (park()/resume()) are mutually exclusive — eliminates the
        // data race that previously existed when worker read parked_ here
        // without the lock while main thread wrote it in park().
        if (parked_.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lk(park_mtx_);
            park_cv_.wait(lk, [this] {
                return !parked_.load(std::memory_order_acquire) ||
                       stop_.load(std::memory_order_acquire);
            });
        }
    }
}
