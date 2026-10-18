#pragma once
#include "runtime/cuda_host_memory_budget.h"
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#if defined(__linux__)
#include <termios.h>
#include <unistd.h>
#endif

namespace mollm_memory {

// CLI-only Linux UMA protection. No GPU calls, allocations of model buffers,
// external signals or process groups. Sampling is not an OOM guarantee.
class CudaHostMemoryWatchdog {
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread worker_;
    bool stop_ = false;
#if defined(__linux__)
    termios terminal_{};
    bool restore_terminal_ = false;
#endif
public:
    using Reader = bool (*)(size_t&);
    CudaHostMemoryWatchdog() = default;
    CudaHostMemoryWatchdog(const CudaHostMemoryWatchdog&) = delete;
    CudaHostMemoryWatchdog& operator=(const CudaHostMemoryWatchdog&) = delete;
    ~CudaHostMemoryWatchdog() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        wake_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    bool start(size_t headroom, Reader reader = read_host_available_memory) {
        if (!headroom) return true;
        size_t available = 0;
        if (worker_.joinable() || !reader || !reader(available) || available < headroom) {
            std::fprintf(stderr,"CUDA host watchdog: cannot start with available=%zu headroom=%zu bytes\n",
                         available,headroom);
            return false;
        }
#if defined(__linux__)
        // Capture before REPL line editing. Emergency _Exit skips its scoped
        // terminal destructor, including when memory falls while waiting idle.
        restore_terminal_ = isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO,&terminal_) == 0;
#endif
        try {
            worker_ = std::thread([this,headroom,reader] {
                std::unique_lock<std::mutex> lock(mutex_);
                while (!wake_.wait_for(lock,std::chrono::milliseconds(250),[this]{return stop_;})) {
                    size_t current = 0;
                    const bool readable = reader(current);
                    if (!readable || current < headroom) {
                        std::fprintf(stderr,
                            "CUDA host watchdog: stopping this model process; MemAvailable=%zu headroom=%zu bytes readable=%d (exit137)\n",
                            current,headroom,int(readable));
                        std::fflush(stderr);
#if defined(__linux__)
                        if (restore_terminal_) tcsetattr(STDIN_FILENO,TCSANOW,&terminal_);
#endif
                        // No GPU synchronization/destructors while memory is
                        // exhausted. Exit ONLY this process, not its parent.
                        std::_Exit(137);
                    }
                }
            });
        } catch (const std::exception& e) {
            std::fprintf(stderr,"CUDA host watchdog: failed to start: %s\n",e.what());
            return false;
        }
        std::fprintf(stderr,"CUDA host watchdog: monitoring every250ms, headroom=%zu bytes\n",headroom);
        return true;
    }
    bool start_from_environment(bool cuda) {
#if defined(__linux__)
        if (!cuda) return true;
        const char* option = std::getenv("MOLLM_CUDA_HOST_HEADROOM_MB");
        if (!option) return true;
        size_t headroom = 0;
        if (!parse_cuda_host_headroom(option,headroom)) {
            std::fprintf(stderr,"CUDA host watchdog: invalid MOLLM_CUDA_HOST_HEADROOM_MB\n");
            return false;
        }
        return start(headroom);
#else
        (void)cuda;
        return true;
#endif
    }
};
} // namespace mollm_memory
