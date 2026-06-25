#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

class ThreadPool {
public:
    using ParallelForFn = std::function<void(int thread_id, int begin, int end)>;

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

private:
    struct Job {
        int begin = 0;
        int end = 0;
        int grain_size = 1;
        int active_threads = 1;
        std::atomic<size_t> generation{0};
        ParallelForFn fn;
    };

    void parallel_for_impl(int begin, int end, int grain_size, ParallelForFn fn);
    void worker_loop(int thread_id);
    void stop_workers();
    void ensure_workers_started();

    std::vector<std::thread> workers_;
    Job job_;
    int num_threads_ = 1;
    bool workers_started_ = false;
    bool parked_ = false;  // workers blocked on park_cv_

    // Sync primitives (spin-wait based for low dispatch latency)
    std::atomic<bool> stop_{false};
    std::atomic<int>  pending_workers_{0};
    std::unique_ptr<std::atomic<bool>[]> worker_ready_;
    // Park support: workers block on park_cv_ when idle instead of spinning.
    std::mutex park_mtx_;
    std::condition_variable park_cv_;
};
