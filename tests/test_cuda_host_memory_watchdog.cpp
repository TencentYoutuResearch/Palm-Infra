#include "runtime/cuda_host_memory_watchdog.h"
#include <atomic>
#include <cstdio>
#include <cstring>

static std::atomic<int> reads{0};
static bool high(size_t& bytes) { bytes=1000;return true; }
static bool low(size_t& bytes) { bytes=99;return true; }
static bool dropping(size_t& bytes) { bytes=reads++==0?1000:99;return true; }
static bool unreadable_after_start(size_t& bytes) { bytes=1000;return reads++==0; }
static bool unreadable(size_t&) { return false; }

int main(int argc,char** argv) {
    if(argc==2) {
        mollm_memory::CudaHostMemoryWatchdog guard;
        const bool missing=!std::strcmp(argv[1],"--unreadable");
        const bool raw=!std::strcmp(argv[1],"--drop-raw");
        if(!missing&&!raw&&std::strcmp(argv[1],"--drop"))return 2;
        if(!guard.start(100,missing?unreadable_after_start:dropping))return 3;
#if defined(__linux__)
        if(raw) {
            termios state{};
            if(tcgetattr(STDIN_FILENO,&state))return 5;
            state.c_lflag &= ~(ICANON|ECHO|ISIG);
            if(tcsetattr(STDIN_FILENO,TCSANOW,&state))return 5;
        }
#else
        if(raw)return 5;
#endif
        std::puts("watchdog child ready");std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        return 4; // Expected to terminate137 first, without GPU/OOM pressure.
    }
    if(argc!=1)return 2;
    { mollm_memory::CudaHostMemoryWatchdog guard;
      if(guard.start(100,low)||guard.start(100,unreadable)||guard.start(100,nullptr))return 1;
      if(!guard.start(0,unreadable))return 1; }
    const auto begin=std::chrono::steady_clock::now();
    { mollm_memory::CudaHostMemoryWatchdog guard;
      if(!guard.start(100,high)||guard.start(100,high))return 1;
      std::this_thread::sleep_for(std::chrono::milliseconds(300)); }
    if(std::chrono::steady_clock::now()-begin>std::chrono::seconds(2))return 1;
    { mollm_memory::CudaHostMemoryWatchdog guard;
      if(!guard.start_from_environment(false))return 1; }
    std::puts("host watchdog startup/admission/disable/shutdown PASS");
    return 0;
}
