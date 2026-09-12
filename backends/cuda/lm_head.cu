#include <cuda_runtime.h>

#include "backends/cuda/backend_internal.h"

#include <algorithm>

using mollm::cuda_backend_detail::device_pointer_const;

bool CudaBackend::supports_lm_head(const Tensor& weight) const {
    return impl_->find_weight(weight) != nullptr;
}

bool CudaBackend::supports_lm_head_argmax(const Tensor& weight) const {
    return supports_lm_head(weight);
}

void CudaBackend::lm_head_gemv(const float* activation_host,
                               const Tensor& weight, float* output_host,
                               int n, int k, int activation) {
    if (!impl_->run_matmul(
            activation_host, k, weight, output_host, n, 1, n, k,
            static_cast<Activation>(activation), 0, -1))
        impl_->failed = true;
}

void CudaBackend::lm_head_gemv_device_and_end_graph(
    const Tensor& activation, size_t activation_element_offset,
    const Tensor& weight, float* output_host, int n, int k,
    int activation_kind) {
    const float* activation_base = device_pointer_const<float>(activation);
    const float* source = activation_base
        ? activation_base + activation_element_offset : nullptr;
    const size_t output_size = static_cast<size_t>(n) * sizeof(float);
    if (!source ||
        !impl_->reserve(impl_->output, impl_->output_bytes, output_size) ||
        !impl_->run_matmul_device(
            source, k, weight, static_cast<float*>(impl_->output), n,
            1, n, k, static_cast<Activation>(activation_kind), 0, -1) ||
        !mollm_cuda::copy_memory(
            output_host, impl_->output, output_size, cudaMemcpyDeviceToHost,
            "cudaMemcpy lm_head output")) {
        impl_->failed = true;
        return;
    }
    ++impl_->native_ops[static_cast<uint32_t>(OpType::MATMUL)];
}

int CudaBackend::lm_head_argmax_device_and_end_graph(
    const Tensor& activation, size_t activation_element_offset,
    const Tensor& weight, int n, int k, int activation_kind,
    Tensor* hidden_copy) {
    const float* activation_base = device_pointer_const<float>(activation);
    const float* source = activation_base
        ? activation_base + activation_element_offset : nullptr;
    const size_t output_size = static_cast<size_t>(n) * sizeof(float);
    const int groups = std::min(256, (n + 255) / 256);
    const size_t partial_size =
        static_cast<size_t>(groups) * sizeof(mollm_cuda::ArgMaxPair);
    if (!source || n <= 0 || groups <= 0 ||
        !impl_->reserve(impl_->output, impl_->output_bytes, output_size) ||
        !impl_->reserve(
            impl_->argmax_partial, impl_->argmax_partial_bytes,
            partial_size) ||
        !impl_->reserve(
            impl_->argmax_result, impl_->argmax_result_bytes,
            sizeof(mollm_cuda::ArgMaxPair)) ||
        !impl_->run_matmul_device(
            source, k, weight, static_cast<float*>(impl_->output), n,
            1, n, k, static_cast<Activation>(activation_kind), 0, -1)) {
        impl_->failed = true;
        return -1;
    }
    if (hidden_copy && hidden_copy->device.buffer &&
        hidden_copy->nbytes() >= static_cast<size_t>(k) * sizeof(float) &&
        !mollm_cuda::copy_memory(
            hidden_copy->device.buffer, source,
            static_cast<size_t>(k) * sizeof(float),
            cudaMemcpyDeviceToDevice, "cudaMemcpy lm_head hidden")) {
        impl_->failed = true;
        return -1;
    }
    mollm_cuda::launch_argmax(
        static_cast<const float*>(impl_->output), n,
        static_cast<mollm_cuda::ArgMaxPair*>(impl_->argmax_partial), groups,
        static_cast<mollm_cuda::ArgMaxPair*>(impl_->argmax_result));
    mollm_cuda::ArgMaxPair result{};
    if (!mollm_cuda::report_cuda(cudaGetLastError(), "argmax_cuda") ||
        !mollm_cuda::copy_memory(
            &result, impl_->argmax_result, sizeof(result),
            cudaMemcpyDeviceToHost, "cudaMemcpy lm_head argmax") ||
        result.index < 0 || result.index >= n) {
        impl_->failed = true;
        return -1;
    }
    ++impl_->native_ops[static_cast<uint32_t>(OpType::MATMUL)];
    return result.index;
}
