#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
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
        size_t generation = 0;
        ParallelForFn fn;
    };

    void parallel_for_impl(int begin, int end, int grain_size, ParallelForFn fn);
    void worker_loop(int thread_id);
    void stop_locked();

    std::mutex mutex_;
    std::condition_variable cv_job_;
    std::condition_variable cv_done_;
    std::vector<std::thread> workers_;
    Job job_;
    int num_threads_ = 1;
    int pending_workers_ = 0;
    bool stop_ = false;
};
