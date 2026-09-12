#pragma once

#include <cstddef>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace mollm_cuda {

struct DeviceBufferPool;

bool report_cuda(cudaError_t error, const char* operation);
bool report_cublas(cublasStatus_t status, const char* operation);
bool malloc_device(void** pointer, size_t bytes, const char* operation);
bool malloc_managed(void** pointer, size_t bytes, const char* operation);
bool copy_memory(void* destination, const void* source, size_t bytes,
                 cudaMemcpyKind kind, const char* operation);
bool zero_memory(void* destination, size_t bytes, const char* operation);

DeviceBufferPool* create_device_buffer_pool();
void destroy_device_buffer_pool(DeviceBufferPool* pool);
void* acquire_device_buffer(DeviceBufferPool* pool, size_t bytes);
void release_device_buffer(DeviceBufferPool* pool, void* pointer);

}  // namespace mollm_cuda
