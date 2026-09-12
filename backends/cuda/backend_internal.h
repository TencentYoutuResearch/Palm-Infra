#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "backends/cpu/backend.h"
#include "backends/cuda/backend.h"
#include "backends/cuda/device_runtime.h"
#include "core/activation.h"
#include "core/quant_layouts.h"
#include "graph/graph.h"
#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/matmul.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

struct CudaBackend::Impl {
    enum class WeightLayout : uint8_t {
        Dense = 0,
        Q8RowMajor,
        Q4Bg32Biased,
        Q4Bg128,
    };

    struct DeviceWeight {
        void* data = nullptr;
        float* scales = nullptr;
        cudaDataType type = CUDA_R_16F;
        int n = 0;
        int k = 0;
        int group_size = 0;
        int groups_per_row = 0;
        WeightLayout layout = WeightLayout::Dense;
    };

    struct BoundaryBuffer {
        void* data = nullptr;
        size_t capacity = 0;
    };

    struct PersistentHostMirror {
        PersistentHostMirror(size_t bytes, bool coherent)
            : size(bytes),
              device_coherent(coherent),
              words((bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t)) {}

        uint8_t* data() {
            return reinterpret_cast<uint8_t*>(words.data());
        }

        size_t size = 0;
        bool device_coherent = true;
        std::vector<uint64_t> words;
    };

    bool ok = false;
    bool failed = false;
    cublasHandle_t cublas = nullptr;
    CPUBackend cpu;
    std::unordered_map<const void*, DeviceWeight> weights;
    std::unordered_map<const void*, const DeviceWeight*> weights_by_device;
    std::vector<void*> device_allocations;
    mollm_cuda::DeviceBufferPool* output_pool =
        mollm_cuda::create_device_buffer_pool();
    std::vector<void*> managed_allocations;
    std::unordered_map<void*, PersistentHostMirror> persistent_host_mirrors;
    std::unordered_map<std::string, BoundaryBuffer> boundary_buffers;
    std::unordered_map<uint32_t, uint64_t> native_ops;
    std::unordered_map<uint32_t, uint64_t> fallback_ops;
    OperatorFallbackPolicy operator_fallback =
        OperatorFallbackPolicy::ALLOW_REFERENCE;
    void* activation = nullptr;
    size_t activation_bytes = 0;
    void* activation_fp16 = nullptr;
    size_t activation_fp16_bytes = 0;
    void* quantized_weight_scratch = nullptr;
    size_t quantized_weight_scratch_bytes = 0;
    void* output = nullptr;
    size_t output_bytes = 0;
    void* argmax_partial = nullptr;
    size_t argmax_partial_bytes = 0;
    void* argmax_result = nullptr;
    size_t argmax_result_bytes = 0;
    void* attention_scores = nullptr;
    size_t attention_scores_bytes = 0;
    void* norm_scratch = nullptr;
    size_t norm_scratch_bytes = 0;
    void* recurrent_scratch = nullptr;
    size_t recurrent_scratch_bytes = 0;

    ~Impl() {
        if (cublas)
            cublasDestroy(cublas);
        for (void* allocation : device_allocations)
            cudaFree(allocation);
        mollm_cuda::destroy_device_buffer_pool(output_pool);
        for (void* allocation : managed_allocations)
            cudaFree(allocation);
        for (auto& entry : boundary_buffers)
            if (entry.second.data)
                cudaFree(entry.second.data);
        if (activation)
            cudaFree(activation);
        if (activation_fp16)
            cudaFree(activation_fp16);
        if (quantized_weight_scratch)
            cudaFree(quantized_weight_scratch);
        if (output)
            cudaFree(output);
        if (argmax_partial)
            cudaFree(argmax_partial);
        if (argmax_result)
            cudaFree(argmax_result);
        if (attention_scores)
            cudaFree(attention_scores);
        if (norm_scratch)
            cudaFree(norm_scratch);
        if (recurrent_scratch)
            cudaFree(recurrent_scratch);
        if (std::getenv("MOLLM_CUDA_PROFILE")) {
            std::fprintf(stderr, "\nCudaBackend operator coverage:\n");
            for (const auto& entry : native_ops)
                std::fprintf(stderr, "  native   %-24s %llu\n",
                             op_type_name(static_cast<OpType>(entry.first)),
                             static_cast<unsigned long long>(entry.second));
            for (const auto& entry : fallback_ops)
                std::fprintf(stderr, "  fallback %-24s %llu\n",
                             op_type_name(static_cast<OpType>(entry.first)),
                             static_cast<unsigned long long>(entry.second));
        }
    }

    bool reserve(void*& pointer, size_t& capacity, size_t requested) {
        if (capacity >= requested)
            return true;
        if (pointer)
            cudaFree(pointer);
        pointer = nullptr;
        capacity = 0;
        if (!mollm_cuda::malloc_device(
                &pointer, requested, "cudaMalloc scratch"))
            return false;
        capacity = requested;
        return true;
    }

    const DeviceWeight* find_weight(const Tensor& tensor) const {
        if (!tensor.device.buffer)
            return nullptr;
        const auto found = weights_by_device.find(tensor.device.buffer);
        return found == weights_by_device.end() ? nullptr : found->second;
    }

    bool upload_weight(Tensor& tensor, const void* cache_key,
                       const void* source, size_t bytes, cudaDataType type,
                       int n, int k,
                       WeightLayout layout = WeightLayout::Dense) {
        if (!cache_key || !source || bytes == 0)
            return false;
        auto found = weights.find(cache_key);
        if (found == weights.end()) {
            void* device = nullptr;
            if (!mollm_cuda::malloc_device(
                    &device, bytes, "cudaMalloc weight") ||
                !mollm_cuda::copy_memory(
                    device, source, bytes, cudaMemcpyHostToDevice,
                    "cudaMemcpy weight")) {
                if (device)
                    cudaFree(device);
                return false;
            }
            // CUDA's hot W4G32 kernels consume an unsigned nibble minus 8.
            // Bias the private device copy once so every decode FMA avoids
            // recovering two's-complement INT4 signs. The package and all
            // other backends retain the canonical representation.
            if (layout == WeightLayout::Q4Bg32Biased) {
                mollm_cuda::launch_bias_q4_g32_weight(
                    static_cast<Q4B8G32Block*>(device),
                    bytes / sizeof(Q4B8G32Block));
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), "bias_q4_g32_weight_cuda")) {
                    cudaFree(device);
                    return false;
                }
            }
            device_allocations.push_back(device);
            found = weights.emplace(
                cache_key,
                DeviceWeight{device, nullptr, type, n, k, 0, 0, layout})
                        .first;
            weights_by_device.emplace(device, &found->second);
        }
        tensor.device.buffer = found->second.data;
        tensor.device.offset = 0;
        return true;
    }

    bool upload_quantized_weight(
        Tensor& tensor, const void* cache_key, const void* source,
        size_t bytes, const float* host_scales, size_t scale_count,
        int n, int k, int group_size, int groups_per_row) {
        if (!host_scales || scale_count == 0 || group_size <= 0 ||
            groups_per_row <= 0 ||
            !upload_weight(
                tensor, cache_key, source, bytes, CUDA_R_8I, n, k,
                WeightLayout::Q8RowMajor))
            return false;
        auto found = weights.find(cache_key);
        if (found == weights.end())
            return false;
        DeviceWeight& prepared = found->second;
        if (!prepared.scales) {
            float* device_scales = nullptr;
            const size_t scale_bytes = scale_count * sizeof(float);
            if (!mollm_cuda::malloc_device(
                    reinterpret_cast<void**>(&device_scales), scale_bytes,
                    "cudaMalloc weight scales") ||
                !mollm_cuda::copy_memory(
                    device_scales, host_scales, scale_bytes,
                    cudaMemcpyHostToDevice, "cudaMemcpy weight scales")) {
                if (device_scales)
                    cudaFree(device_scales);
                return false;
            }
            device_allocations.push_back(device_scales);
            prepared.scales = device_scales;
        }
        prepared.group_size = group_size;
        prepared.groups_per_row = groups_per_row;
        return true;
    }

    bool run_matmul_device(const float* device_a, int lda,
                           const Tensor& weight, float* device_c, int ldc,
                           int m, int n, int k,
                           Activation activation_kind, int act_begin,
                           int act_len) {
        const DeviceWeight* prepared = find_weight(weight);
        if (!prepared || prepared->n != n || prepared->k != k || !device_a ||
            !device_c || ldc != n)
            return false;

        const bool valid_q8 =
            prepared->layout == WeightLayout::Q8RowMajor &&
            prepared->scales && prepared->group_size > 0 &&
            prepared->groups_per_row ==
                (k + prepared->group_size - 1) / prepared->group_size;
        const bool valid_q4_g32 =
            prepared->layout == WeightLayout::Q4Bg32Biased && k % 32 == 0;
        const bool valid_q4_g128 =
            prepared->layout == WeightLayout::Q4Bg128 && k % 128 == 0;
        const bool valid_quantized =
            valid_q8 || valid_q4_g32 || valid_q4_g128;
        if (valid_quantized && m == 1) {
            const char* label = nullptr;
            if (valid_q8) {
                mollm_cuda::launch_q8_dense_gemv(
                    device_a, static_cast<const int8_t*>(prepared->data),
                    prepared->scales, prepared->group_size,
                    prepared->groups_per_row, device_c, n, k);
                label = "q8_dense_gemv_cuda";
            } else if (valid_q4_g32) {
                mollm_cuda::launch_q4_g32_dense_gemv(
                    device_a,
                    static_cast<const Q4B8G32Block*>(prepared->data),
                    device_c, n, k);
                label = "q4_g32_dense_gemv_cuda";
            } else {
                mollm_cuda::launch_q4_g128_dense_gemv(
                    device_a,
                    static_cast<const Q4B8G128Block*>(prepared->data),
                    device_c, n, k);
                label = "q4_g128_dense_gemv_cuda";
            }
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), label))
                return false;
        } else {
            const void* linear_weight = prepared->data;
            cudaDataType linear_weight_type = prepared->type;
            const size_t a_elements = static_cast<size_t>(m) * lda;
            bool activation_prepared = false;
            if (valid_quantized) {
                const size_t weight_elements = static_cast<size_t>(n) * k;
                if (!reserve(
                        quantized_weight_scratch,
                        quantized_weight_scratch_bytes,
                        weight_elements * sizeof(__half)))
                    return false;
                const char* label = nullptr;
                if (valid_q8) {
                    mollm_cuda::launch_dequantize_q8_dense_weight(
                        static_cast<const int8_t*>(prepared->data),
                        prepared->scales, prepared->group_size,
                        prepared->groups_per_row,
                        static_cast<__half*>(quantized_weight_scratch),
                        weight_elements, k);
                    label = "dequantize_q8_dense_weight_cuda";
                } else if (valid_q4_g32) {
                    if (!reserve(
                            activation_fp16, activation_fp16_bytes,
                            a_elements * sizeof(__half)))
                        return false;
                    mollm_cuda::launch_prepare_q4_g32_dense_gemm(
                        static_cast<const Q4B8G32Block*>(prepared->data),
                        reinterpret_cast<__half2*>(quantized_weight_scratch),
                        device_a, static_cast<__half*>(activation_fp16),
                        a_elements, n, k / 32);
                    activation_prepared = true;
                    label = "prepare_q4_g32_dense_gemm_cuda";
                } else {
                    const size_t packed_count =
                        static_cast<size_t>((n + 7) / 8) * (k / 128) *
                        4 * 8 * 16;
                    mollm_cuda::launch_dequantize_q4_g128_dense_weight(
                        static_cast<const Q4B8G128Block*>(prepared->data),
                        reinterpret_cast<__half2*>(quantized_weight_scratch),
                        packed_count, n, k / 128);
                    label = "dequantize_q4_g128_dense_weight_cuda";
                }
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), label))
                    return false;
                linear_weight = quantized_weight_scratch;
                linear_weight_type = CUDA_R_16F;
            } else if (prepared->layout != WeightLayout::Dense) {
                return false;
            }

            const void* gemm_activation = device_a;
            cudaDataType activation_type = CUDA_R_32F;
            if (linear_weight_type == CUDA_R_16F) {
                if (!activation_prepared) {
                    if (!reserve(activation_fp16, activation_fp16_bytes,
                                 a_elements * sizeof(__half)))
                        return false;
                    mollm_cuda::launch_fp32_to_fp16(
                        device_a, static_cast<__half*>(activation_fp16),
                        a_elements);
                    if (!mollm_cuda::report_cuda(
                            cudaGetLastError(), "fp32_to_fp16"))
                        return false;
                }
                gemm_activation = activation_fp16;
                activation_type = CUDA_R_16F;
            }

            if (!mollm_cuda::run_dense_matmul(
                    cublas, linear_weight, linear_weight_type,
                    gemm_activation, activation_type, device_c, m, n, k,
                    lda))
                return false;
        }

        if (activation_kind != Activation::NONE && act_len != 0) {
            const int begin = std::max(0, act_begin);
            const int end = act_len < 0 ? n : std::min(n, begin + act_len);
            mollm_cuda::launch_apply_activation(device_c, m, n,
                           static_cast<int>(activation_kind), begin, end);
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "apply_activation_cuda"))
                return false;
        }
        return true;
    }

    bool run_matmul(const float* host_a, int lda, const Tensor& weight,
                    float* host_c, int ldc, int m, int n, int k,
                    Activation activation_kind, int act_begin, int act_len) {
        const size_t a_bytes = static_cast<size_t>(m) * lda * sizeof(float);
        const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
        if (!host_a || !host_c ||
            !reserve(activation, activation_bytes, a_bytes) ||
            !reserve(output, output_bytes, c_bytes) ||
            !mollm_cuda::copy_memory(
                activation, host_a, a_bytes, cudaMemcpyHostToDevice,
                "cudaMemcpy activation") ||
            !run_matmul_device(
                static_cast<const float*>(activation), lda, weight,
                static_cast<float*>(output), ldc, m, n, k, activation_kind,
                act_begin, act_len) ||
            !mollm_cuda::copy_memory(
                host_c, output, c_bytes, cudaMemcpyDeviceToHost,
                "cudaMemcpy output"))
            return false;
        return true;
    }
};


namespace mollm::cuda_backend_detail {

template <typename T>
T* device_pointer(const Tensor& tensor) {
    if (!tensor.device.buffer)
        return nullptr;
    return reinterpret_cast<T*>(
        static_cast<uint8_t*>(tensor.device.buffer) + tensor.device.offset);
}

template <typename T>
const T* device_pointer_const(const Tensor& tensor) {
    if (!tensor.device.buffer)
        return nullptr;
    return reinterpret_cast<const T*>(
        static_cast<const uint8_t*>(tensor.device.buffer) +
        tensor.device.offset);
}

inline bool fp32_contiguous(const Tensor& tensor) {
    return tensor.prec == Precision::FP32 && tensor.is_contiguous();
}

inline bool same_shape(const Tensor& lhs, const Tensor& rhs) {
    for (int dimension = 0; dimension < 4; ++dimension)
        if (lhs.shape[dimension] != rhs.shape[dimension])
            return false;
    return true;
}

}  // namespace mollm::cuda_backend_detail
