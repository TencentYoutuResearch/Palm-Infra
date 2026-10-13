#include "backends/cuda/device_runtime.h"
#include "runtime/cuda_host_memory_budget.h"

#include <cstdio>
#include <algorithm>
#include <map>
#include <new>
#include <unordered_map>
#include <vector>

namespace mollm_cuda {

struct DeviceBufferPool {
    std::vector<void*> allocations;
    std::unordered_map<void*, size_t> sizes;
    std::multimap<size_t, void*> free;
};
size_t device_buffer_pool_reserved_bytes(const DeviceBufferPool* pool) {
    size_t bytes=0;
    if(pool)for(const auto& entry:pool->sizes)bytes+=entry.second;
    return bytes;
}
bool device_buffer_pool_can_reuse(const DeviceBufferPool* pool, size_t bytes) {
    return pool && bytes && pool->free.lower_bound(bytes) != pool->free.end();
}

bool report_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess)
        return true;
    std::fprintf(stderr, "CudaBackend: %s failed: %s\n", operation,
                 cudaGetErrorString(error));
    return false;
}

bool report_cublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return true;
    std::fprintf(stderr, "CudaBackend: %s failed (cuBLAS status %d)\n",
                 operation, static_cast<int>(status));
    return false;
}

bool malloc_device(void** pointer, size_t bytes, const char* operation) {
    if (!mollm_memory::cuda_host_allocation_allowed(bytes, operation)) return false;
    return report_cuda(cudaMalloc(pointer, bytes), operation);
}

bool malloc_managed(void** pointer, size_t bytes, const char* operation) {
    if (!mollm_memory::cuda_host_allocation_allowed(bytes, operation)) return false;
    return report_cuda(cudaMallocManaged(pointer, bytes), operation);
}

bool copy_memory(void* destination, const void* source, size_t bytes,
                 cudaMemcpyKind kind, const char* operation) {
    return report_cuda(cudaMemcpy(destination, source, bytes, kind), operation);
}

bool zero_memory(void* destination, size_t bytes, const char* operation) {
    return report_cuda(cudaMemset(destination, 0, bytes), operation);
}

DeviceBufferPool* create_device_buffer_pool() {
    return new (std::nothrow) DeviceBufferPool();
}

void destroy_device_buffer_pool(DeviceBufferPool* pool) {
    if (!pool)
        return;
    for (void* allocation : pool->allocations)
        cudaFree(allocation);
    delete pool;
}

void* acquire_device_buffer(DeviceBufferPool* pool, size_t bytes) {
    if (!pool || bytes == 0)
        return nullptr;
    auto found = pool->free.lower_bound(bytes);
    if (found != pool->free.end()) {
        void* pointer = found->second;
        pool->free.erase(found);
        return pointer;
    }
    void* pointer = nullptr;
    if (!malloc_device(&pointer, bytes, "cudaMalloc output"))
        return nullptr;
    pool->allocations.push_back(pointer);
    pool->sizes.emplace(pointer, bytes);
    return pointer;
}

void release_device_buffer(DeviceBufferPool* pool, void* pointer) {
    if (!pool || !pointer)
        return;
    const auto found = pool->sizes.find(pointer);
    if (found != pool->sizes.end())
        pool->free.emplace(found->second, pointer);
}

bool trim_free_device_buffers(DeviceBufferPool* pool, cudaStream_t stream,
                              size_t* reclaimed_bytes) {
    if (reclaimed_bytes) *reclaimed_bytes = 0;
    if (!pool || !reclaimed_bytes) return false;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (!report_cuda(cudaStreamIsCapturing(stream, &capture), "pool trim capture check") ||
        capture != cudaStreamCaptureStatusNone) return false;
    while (!pool->free.empty()) {
        auto entry = pool->free.begin();
        void* pointer = entry->second;
        const size_t bytes = entry->first;
        if (!report_cuda(cudaFree(pointer), "pool trim free")) return false;
        *reclaimed_bytes += bytes;
        pool->sizes.erase(pointer);
        pool->allocations.erase(std::remove(pool->allocations.begin(),
            pool->allocations.end(), pointer), pool->allocations.end());
        pool->free.erase(entry);
    }
    return true;
}

}  // namespace mollm_cuda
