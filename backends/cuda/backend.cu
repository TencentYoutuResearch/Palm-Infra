#include <cuda_runtime.h>

#include "backends/cuda/backend.h"
#include "backends/cuda/backend_internal.h"
#include "backends/cuda/device_runtime.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

using mollm::cuda_backend_detail::device_pointer;
using mollm::cuda_backend_detail::device_pointer_const;

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {
    int count = 0;
    if (!mollm_cuda::report_cuda(
            cudaGetDeviceCount(&count), "cudaGetDeviceCount") ||
        count <= 0)
        return;
    if (!mollm_cuda::report_cuda(cudaSetDevice(0), "cudaSetDevice") ||
        !mollm_cuda::report_cublas(
            cublasCreate(&impl_->cublas), "cublasCreate"))
        return;
    impl_->ok = true;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess)
        std::fprintf(stderr, "CudaBackend: using %s (sm_%d%d)\n",
                     properties.name, properties.major, properties.minor);
}

CudaBackend::~CudaBackend() = default;

bool CudaBackend::available() const { return impl_ && impl_->ok; }

void CudaBackend::begin_execution() {
    impl_->cpu.begin_execution();
}

void CudaBackend::clear_dispatch_error() {
    impl_->failed = false;
    impl_->cpu.clear_dispatch_error();
}

bool CudaBackend::dispatch_failed() const {
    return impl_->failed || impl_->cpu.dispatch_failed();
}

bool CudaBackend::set_operator_fallback_policy(
    OperatorFallbackPolicy policy) {
    impl_->operator_fallback = policy;
    return true;
}

bool CudaBackend::register_weight_region(void*, size_t) { return true; }

void CudaBackend::wrap_weight(Tensor& tensor) {
    if (!available() || !tensor.data || tensor.shape[0] <= 0 ||
        tensor.shape[1] <= 0)
        return;
    const int n = static_cast<int>(tensor.shape[0]);
    const int k = static_cast<int>(tensor.shape[1]);
    if (tensor.prec == Precision::FP16) {
        impl_->upload_weight(tensor, tensor.data, tensor.data,
                             static_cast<size_t>(n) * k * sizeof(__half),
                             CUDA_R_16F, n, k);
    } else if (tensor.prec == Precision::FP32) {
        impl_->upload_weight(tensor, tensor.data, tensor.data,
                             static_cast<size_t>(n) * k * sizeof(float),
                             CUDA_R_32F, n, k);
    } else if (tensor.prec == Precision::INT8 && tensor.scales &&
               tensor.group_size > 0) {
        const void* quantized =
            tensor.rowmajor_data ? tensor.rowmajor_data : tensor.data;
        const int group_size = static_cast<int>(tensor.group_size);
        const int groups_per_row = static_cast<int>(tensor.groups_per_row);
        if (quantized && groups_per_row ==
                (k + group_size - 1) / group_size) {
            impl_->upload_quantized_weight(
                tensor, quantized, quantized, static_cast<size_t>(n) * k,
                tensor.scales, static_cast<size_t>(n) * groups_per_row,
                n, k, group_size, groups_per_row);
        }
    }
}

void CudaBackend::wrap_weight_int4(Tensor& tensor,
                                   bool keep_native_experts) {
    if (!available() || keep_native_experts || tensor.prec != Precision::INT4 ||
        tensor.shape[0] <= 0 || tensor.shape[1] <= 0)
        return;
    const int n = static_cast<int>(tensor.shape[0]);
    const int k = static_cast<int>(tensor.shape[1]);

    if (tensor.is_q4_g32_packed && tensor.q4_g32_data && k % 32 == 0 &&
        tensor.group_size == 32 &&
        tensor.groups_per_row == static_cast<uint32_t>(k / 32)) {
        const size_t bytes = static_cast<size_t>((n + 7) / 8) * (k / 32) *
            sizeof(Q4B8G32Block);
        impl_->upload_weight(
            tensor, tensor.q4_g32_data, tensor.q4_g32_data, bytes,
            CUDA_R_8I, n, k, Impl::WeightLayout::Q4Bg32Biased);
    } else if (tensor.is_q4_g128_packed && tensor.q4_g128_data &&
               k % 128 == 0 && tensor.group_size == 128 &&
               tensor.groups_per_row == static_cast<uint32_t>(k / 128)) {
        const size_t bytes = static_cast<size_t>((n + 7) / 8) * (k / 128) *
            sizeof(Q4B8G128Block);
        impl_->upload_weight(
            tensor, tensor.q4_g128_data, tensor.q4_g128_data, bytes,
            CUDA_R_8I, n, k, Impl::WeightLayout::Q4Bg128);
    }
}

void* CudaBackend::alloc_output(Tensor& output, size_t nbytes, BufferPool*) {
    if (!available() || nbytes == 0)
        return nullptr;
    void* pointer = mollm_cuda::acquire_device_buffer(
        impl_->output_pool, nbytes);
    if (!pointer) {
        impl_->failed = true;
        return nullptr;
    }
    output.data = pointer;
    output.device.buffer = pointer;
    output.device.offset = 0;
    output.mem_type = MemoryType::POOLED;
    output.owner_id = 0;
    output.storage_id = 0;
    return pointer;
}

void CudaBackend::free_output(Tensor& tensor, BufferPool*) {
    if (tensor.device.buffer)
        mollm_cuda::release_device_buffer(
            impl_->output_pool, tensor.device.buffer);
}

bool CudaBackend::copy_to_host(const Tensor& source, void* destination,
                               size_t nbytes, size_t source_offset) {
    if (!destination || source_offset > source.view_span_bytes() ||
        nbytes > source.view_span_bytes() - source_offset) {
        impl_->failed = true;
        return false;
    }
    size_t host_prefix_bytes = 0;
    const auto mirror = impl_->persistent_host_mirrors.find(
        source.device.buffer);
    const size_t absolute_offset = source.device.offset + source_offset;
    if (mirror != impl_->persistent_host_mirrors.end() &&
        !mirror->second.device_coherent &&
        absolute_offset < mirror->second.size) {
        host_prefix_bytes = std::min(
            nbytes, mirror->second.size - absolute_offset);
        std::memcpy(
            destination, mirror->second.data() + absolute_offset,
            host_prefix_bytes);
        if (host_prefix_bytes == nbytes)
            return true;
    }
    const auto* device = device_pointer_const<uint8_t>(source);
    if (!device) {
        if (!source.data) {
            impl_->failed = true;
            return false;
        }
        std::memcpy(
            static_cast<uint8_t*>(destination) + host_prefix_bytes,
            static_cast<const uint8_t*>(source.data) + source_offset +
                host_prefix_bytes,
            nbytes - host_prefix_bytes);
        return true;
    }
    if (!mollm_cuda::copy_memory(
            static_cast<uint8_t*>(destination) + host_prefix_bytes,
            device + source_offset + host_prefix_bytes,
            nbytes - host_prefix_bytes,
            cudaMemcpyDeviceToHost, "cudaMemcpy tensor to host")) {
        impl_->failed = true;
        return false;
    }
    return true;
}

bool CudaBackend::copy_from_host(const void* source, Tensor& destination,
                                 size_t nbytes,
                                 size_t destination_offset) {
    if (!source || destination_offset > destination.view_span_bytes() ||
        nbytes > destination.view_span_bytes() - destination_offset) {
        impl_->failed = true;
        return false;
    }
    const auto mirror = impl_->persistent_host_mirrors.find(
        destination.device.buffer);
    const size_t absolute_offset =
        destination.device.offset + destination_offset;
    size_t host_prefix_bytes = 0;
    if (mirror != impl_->persistent_host_mirrors.end() &&
        absolute_offset < mirror->second.size) {
        host_prefix_bytes = std::min(
            nbytes, mirror->second.size - absolute_offset);
        if (!mirror->second.device_coherent) {
            std::memcpy(
                mirror->second.data() + absolute_offset, source,
                host_prefix_bytes);
            if (host_prefix_bytes == nbytes)
                return true;
        }
    }
    auto* device = device_pointer<uint8_t>(destination);
    if (!device) {
        if (!destination.data) {
            impl_->failed = true;
            return false;
        }
        std::memcpy(
            static_cast<uint8_t*>(destination.data) + destination_offset,
            source, nbytes);
        return true;
    }
    const size_t device_skip =
        mirror != impl_->persistent_host_mirrors.end() &&
            !mirror->second.device_coherent
        ? host_prefix_bytes : 0;
    if (!mollm_cuda::copy_memory(
            device + destination_offset + device_skip,
            static_cast<const uint8_t*>(source) + device_skip,
            nbytes - device_skip,
            cudaMemcpyHostToDevice, "cudaMemcpy tensor from host")) {
        impl_->failed = true;
        return false;
    }
    if (mirror != impl_->persistent_host_mirrors.end() &&
        mirror->second.device_coherent && host_prefix_bytes > 0)
        std::memcpy(
            mirror->second.data() + absolute_offset, source,
            host_prefix_bytes);
    return true;
}

bool CudaBackend::zero_tensor(Tensor& tensor, size_t nbytes,
                              size_t destination_offset) {
    if (destination_offset > tensor.view_span_bytes() ||
        nbytes > tensor.view_span_bytes() - destination_offset) {
        impl_->failed = true;
        return false;
    }
    const auto mirror = impl_->persistent_host_mirrors.find(
        tensor.device.buffer);
    const size_t absolute_offset = tensor.device.offset + destination_offset;
    size_t host_prefix_bytes = 0;
    if (mirror != impl_->persistent_host_mirrors.end() &&
        absolute_offset < mirror->second.size) {
        host_prefix_bytes = std::min(
            nbytes, mirror->second.size - absolute_offset);
        if (!mirror->second.device_coherent) {
            std::memset(
                mirror->second.data() + absolute_offset, 0,
                host_prefix_bytes);
            if (host_prefix_bytes == nbytes)
                return true;
        }
    }
    auto* device = device_pointer<uint8_t>(tensor);
    if (!device) {
        if (!tensor.data) {
            impl_->failed = true;
            return false;
        }
        std::memset(
            static_cast<uint8_t*>(tensor.data) + destination_offset,
            0, nbytes);
        return true;
    }
    const size_t device_skip =
        mirror != impl_->persistent_host_mirrors.end() &&
            !mirror->second.device_coherent
        ? host_prefix_bytes : 0;
    if (!mollm_cuda::zero_memory(
            device + destination_offset + device_skip, nbytes - device_skip,
            "cudaMemset tensor")) {
        impl_->failed = true;
        return false;
    }
    if (mirror != impl_->persistent_host_mirrors.end() &&
        mirror->second.device_coherent && host_prefix_bytes > 0)
        std::memset(
            mirror->second.data() + absolute_offset, 0, host_prefix_bytes);
    return true;
}

bool CudaBackend::round_to_bf16(Tensor& tensor) {
    float* values = device_pointer<float>(tensor);
    if (!values || tensor.prec != Precision::FP32 ||
        !tensor.is_contiguous()) {
        impl_->failed = true;
        return false;
    }
    if (!mollm_cuda::launch_round_to_bf16(
            values, static_cast<size_t>(tensor.nelements()))) {
        impl_->failed = true;
        return false;
    }
    return true;
}

void CudaBackend::synchronize_for_host_read() {
    if (!mollm_cuda::report_cuda(
            cudaDeviceSynchronize(), "cudaDeviceSynchronize"))
        impl_->failed = true;
}

void CudaBackend::begin_graph() {}

void CudaBackend::end_graph() { synchronize_for_host_read(); }

void CudaBackend::alloc_persistent(
    Tensor& tensor, size_t nbytes, PersistentHostAccess host_access,
    size_t host_prefix_bytes) {
    void* storage = nullptr;
    const bool has_host_prefix =
        host_access == PersistentHostAccess::MIRRORED_PREFIX ||
        host_access == PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX;
    if (!available() || nbytes == 0 ||
        (has_host_prefix &&
         (host_prefix_bytes == 0 || host_prefix_bytes > nbytes))) {
        impl_->failed = true;
        return;
    }
    if (host_access == PersistentHostAccess::FULL) {
        if (!mollm_cuda::malloc_managed(
                &storage, nbytes,
                "cudaMallocManaged host-coherent persistent")) {
            impl_->failed = true;
            return;
        }
        impl_->managed_allocations.push_back(storage);
        std::memset(storage, 0, nbytes);
        tensor.data = storage;
    } else {
        if (!mollm_cuda::malloc_device(
                &storage, nbytes, "cudaMalloc persistent") ||
            !mollm_cuda::zero_memory(
                storage, nbytes, "cudaMemset persistent")) {
            if (storage)
                cudaFree(storage);
            impl_->failed = true;
            return;
        }
        impl_->device_allocations.push_back(storage);
        if (has_host_prefix) {
            auto [entry, inserted] =
                impl_->persistent_host_mirrors.emplace(
                    storage,
                    Impl::PersistentHostMirror(
                        host_prefix_bytes,
                        host_access ==
                            PersistentHostAccess::MIRRORED_PREFIX));
            (void)inserted;
            tensor.data = entry->second.data();
        } else {
            // Tensor::data remains a non-null storage handle for executor
            // ownership checks. PersistentHostAccess::NONE forbids host
            // dereference; all access goes through explicit transfer methods.
            tensor.data = storage;
        }
    }
    tensor.device.buffer = storage;
    tensor.device.offset = 0;
    tensor.mem_type = MemoryType::EXTERNAL;
}

void CudaBackend::upload_input(Tensor& tensor, const std::string& key,
                               const void* host_source, size_t nbytes) {
    if (!available() || key.empty() || !host_source || nbytes == 0)
        return;
    auto& buffer = impl_->boundary_buffers[key];
    if (buffer.capacity < nbytes) {
        if (buffer.data)
            cudaFree(buffer.data);
        buffer.data = nullptr;
        buffer.capacity = 0;
        if (!mollm_cuda::malloc_device(
                &buffer.data, nbytes, "cudaMalloc input")) {
            impl_->failed = true;
            return;
        }
        buffer.capacity = nbytes;
    }
    if (!mollm_cuda::copy_memory(
            buffer.data, host_source, nbytes, cudaMemcpyHostToDevice,
            "cudaMemcpy input")) {
        impl_->failed = true;
        return;
    }
    tensor.device.buffer = buffer.data;
    tensor.device.offset = 0;
}
