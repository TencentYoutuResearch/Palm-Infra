#include "backends/cuda/device_runtime.h"

#include <cstdio>
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
    return report_cuda(cudaMalloc(pointer, bytes), operation);
}

bool malloc_managed(void** pointer, size_t bytes, const char* operation) {
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

}  // namespace mollm_cuda
