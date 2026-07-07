#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

class ThreadPool {
public:
    using ParallelForFn = std::function<void(int thread_id, int begin, int end)>;
    // 2D job: fn(job_id, m_begin, m_end, n_begin, n_end)
    using ParallelFor2dFn = std::function<void(int thread_id,
                                                int m_begin, int m_end,
                                                int n_begin, int n_end)>;

    explicit ThreadPool(int num_threads = 1);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    void resize(int num_threads);
    int num_threads() const { return num_threads_; }

    // Park all workers: they exit their spin loop and block on a CV until
    // the next parallel_for. Use between generation steps to drop idle CPU.
    void park();
    // Resume workers (no-op if already running). Next parallel_for also
    // implicitly resumes.
    void resume();

    template <typename Fn>
    void parallel_for(int begin, int end, int grain_size, Fn&& fn) {
        parallel_for_impl(begin, end, grain_size, ParallelForFn(std::forward<Fn>(fn)));
    }

    // 2D dynamic scheduling: total_jobs = m_tiles * n_blocks, atomic-steal.
    // Each job covers [m_begin,m_end) × [n_begin,n_end).
    // m_tile_size is the M-tile size (e.g. 8). n_block_size is the N-block size.
    // fn signature: void(int thread_id, int m_begin, int m_end, int n_begin, int n_end)
    template <typename Fn>
    void parallel_for_2d(int m_total, int m_tile_size,
                          int n_total, int n_block_size,
                          Fn&& fn) {
        parallel_for_2d_impl(m_total, m_tile_size, n_total, n_block_size,
                              ParallelFor2dFn(std::forward<Fn>(fn)));
    }

private:
    struct Job {
        int begin = 0;
        int end = 0;
        int grain_size = 1;
        int active_threads = 1;
        std::atomic<size_t> generation{0};
        ParallelForFn fn;
        // 2D job fields
        int m_total = 0, m_tile_size = 1;
        int n_total = 0, n_block_size = 1;
        int total_2d_jobs = 0;
        std::atomic<int> current_chunk{0};
        ParallelFor2dFn fn_2d;
    };

    void parallel_for_impl(int begin, int end, int grain_size, ParallelForFn fn);
    void parallel_for_2d_impl(int m_total, int m_tile_size,
                                int n_total, int n_block_size,
                                ParallelFor2dFn fn);
    void worker_loop(int thread_id);
    void stop_workers();
    void ensure_workers_started();

    std::vector<std::thread> workers_;
    Job job_;
    int num_threads_ = 1;
    bool workers_started_ = false;
    std::atomic<bool> parked_{false};  // workers blocked on park_cv_

    // Sync primitives (spin-wait based for low dispatch latency)
    std::atomic<bool> stop_{false};
    std::atomic<int>  pending_workers_{0};
    std::unique_ptr<std::atomic<bool>[]> worker_ready_;
    // Park support: workers block on park_cv_ when idle instead of spinning.
    // parked_ is atomic — readers in worker_loop and parallel_for_impl
    // observe it consistently without racing with park()/resume() writers.
    // The cv predicate re-reads parked_ under park_mtx_ for the actual
    // wait decision (see worker_loop in threading.cpp).
    std::mutex park_mtx_;
    std::condition_variable park_cv_;
};
