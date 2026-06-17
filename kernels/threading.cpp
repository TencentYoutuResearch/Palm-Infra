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
    std::unique_lock<std::mutex> lock(mutex_);
    stop_locked();
    lock.unlock();

    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void ThreadPool::stop_locked() {
    stop_ = true;
    cv_job_.notify_all();
}

void ThreadPool::resize(int num_threads) {
    num_threads = std::max(num_threads, 1);
    if (num_threads == num_threads_) return;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_locked();
    }
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    workers_.clear();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = false;
        pending_workers_ = 0;
        job_ = Job();
        num_threads_ = num_threads;
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

    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_.begin = begin;
        job_.end = end;
        job_.grain_size = grain_size;
        job_.active_threads = active_threads;
        job_.fn = std::move(fn);
        job_.generation++;
        pending_workers_ = active_threads - 1;
    }
    cv_job_.notify_all();

    int shard_begin = begin;
    int shard_end = end;
    if (compute_shard(begin, end, grain_size, active_threads, 0, shard_begin, shard_end)) {
        job_.fn(0, shard_begin, shard_end);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] { return pending_workers_ == 0; });
    job_.fn = nullptr;
}

void ThreadPool::worker_loop(int thread_id) {
    size_t seen_generation = 0;

    while (true) {
        Job local_job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_job_.wait(lock, [&] { return stop_ || job_.generation != seen_generation; });
            if (stop_) return;
            local_job = job_;
            seen_generation = job_.generation;
        }

        if (!local_job.fn || thread_id >= local_job.active_threads) {
            continue;
        }

        int shard_begin = local_job.begin;
        int shard_end = local_job.end;
        if (compute_shard(local_job.begin, local_job.end, local_job.grain_size,
                          local_job.active_threads, thread_id,
                          shard_begin, shard_end)) {
            local_job.fn(thread_id, shard_begin, shard_end);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        pending_workers_--;
        if (pending_workers_ == 0) {
            cv_done_.notify_one();
        }
    }
}
