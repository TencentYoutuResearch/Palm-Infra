// Small, bounded diagnostic; cgroup deltas do not prove enforcement of a limit.
// Build: nvcc -O2 -arch=sm_121 tools/probe_cuda_memory_accounting.cu -o build_cuda/probe_cuda_memory_accounting
// Run in a separate scope (not the interactive session's scope):
// systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -- /absolute/path/to/probe_cuda_memory_accounting
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

static long long available() {
    std::ifstream input("/proc/meminfo");
    std::string key, rest;
    long long kb;
    while (input >> key) {
        if (key == "MemAvailable:" && input >> kb) return kb * 1024;
        std::getline(input, rest);
    }
    return -1;
}

static std::string cgroup_path() {
    std::ifstream input("/proc/self/cgroup");
    std::string line;
    while (std::getline(input, line))
        if (line.rfind("0::/", 0) == 0) return "/sys/fs/cgroup" + line.substr(3);
    return {};
}

static long long current(const std::string& path) {
    std::ifstream input(path + "/memory.current");
    long long bytes;
    return input >> bytes ? bytes : -1;
}

static void snapshot(const char* phase, const std::string& path) {
    const auto host = available();
    const auto group = current(path);
    std::printf("phase=%s host_available_bytes=%lld cgroup_current_bytes=%lld\n",
                phase, host, group);
    std::fflush(stdout);
}

int main() {
    // Fixed allocation, no arbitrary user-controlled large-allocation option.
    constexpr size_t bytes = 256ULL * 1024 * 1024;
    constexpr long long minimum = 100LL * 1024 * 1024 * 1024;
    if (available() < minimum) {
        std::fprintf(stderr, "requires at least 100 GiB MemAvailable\n");
        return 2;
    }
    const auto path = cgroup_path();
    if (path.empty() || current(path) < 0) return 2;
    std::printf("pid=%d allocation_bytes=%zu cgroup=%s\n", getpid(), bytes, path.c_str());
    if (cudaFree(nullptr) != cudaSuccess) return 3;
    snapshot("context", path);
    void* buffer = nullptr;
    if (cudaMalloc(&buffer, bytes) != cudaSuccess) return 3;
    snapshot("allocated", path);
    const auto fill = cudaMemset(buffer, 0x5a, bytes);
    const auto sync = cudaDeviceSynchronize();
    snapshot("touched", path);
    const auto released = cudaFree(buffer);
    snapshot("freed", path);
    return fill == cudaSuccess && sync == cudaSuccess && released == cudaSuccess ? 0 : 3;
}
