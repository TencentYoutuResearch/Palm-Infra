#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "engine/cuda_backend.h"

#include "kernels/activations.h"
#include "kernels/quant_layouts.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

const char* cuda_error(cudaError_t error) {
    return cudaGetErrorString(error);
}

bool report_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess)
        return true;
    std::fprintf(stderr, "CudaBackend: %s failed: %s\n", operation,
                 cuda_error(error));
    return false;
}

bool report_cublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return true;
    std::fprintf(stderr, "CudaBackend: %s failed (cuBLAS status %d)\n",
                 operation, static_cast<int>(status));
    return false;
}

__global__ void fp32_to_fp16(const float* source, __half* destination,
                             size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        destination[index] = __float2half(source[index]);
}

int signed_nibble(uint8_t value) {
    const int nibble = value & 0x0f;
    return nibble >= 8 ? nibble - 16 : nibble;
}

}  // namespace

struct CudaBackend::Impl {
    struct DeviceWeight {
        void* data = nullptr;
        cudaDataType type = CUDA_R_16F;
        int n = 0;
        int k = 0;
    };

    bool ok = false;
    bool failed = false;
    cublasHandle_t cublas = nullptr;
    CPUBackend cpu;
    std::unordered_map<const void*, DeviceWeight> weights;
    std::unordered_map<const void*, const DeviceWeight*> weights_by_device;
    std::vector<void*> device_allocations;
    std::vector<std::unique_ptr<uint8_t[]>> persistent_host;
    void* activation = nullptr;
    size_t activation_bytes = 0;
    void* activation_fp16 = nullptr;
    size_t activation_fp16_bytes = 0;
    void* output = nullptr;
    size_t output_bytes = 0;

    ~Impl() {
        if (cublas)
            cublasDestroy(cublas);
        for (void* allocation : device_allocations)
            cudaFree(allocation);
        if (activation)
            cudaFree(activation);
        if (activation_fp16)
            cudaFree(activation_fp16);
        if (output)
            cudaFree(output);
    }

    bool reserve(void*& pointer, size_t& capacity, size_t requested) {
        if (capacity >= requested)
            return true;
        if (pointer)
            cudaFree(pointer);
        pointer = nullptr;
        capacity = 0;
        if (!report_cuda(cudaMalloc(&pointer, requested), "cudaMalloc scratch"))
            return false;
        capacity = requested;
        return true;
    }

    const DeviceWeight* find_weight(const Tensor& tensor) const {
        if (!tensor.device_data)
            return nullptr;
        const auto found = weights_by_device.find(tensor.device_data);
        return found == weights_by_device.end() ? nullptr : found->second;
    }

    bool upload_weight(Tensor& tensor, const void* cache_key,
                       const void* source, size_t bytes, cudaDataType type,
                       int n, int k) {
        if (!cache_key || !source || bytes == 0)
            return false;
        auto found = weights.find(cache_key);
        if (found == weights.end()) {
            void* device = nullptr;
            if (!report_cuda(cudaMalloc(&device, bytes),
                             "cudaMalloc weight") ||
                !report_cuda(cudaMemcpy(device, source, bytes,
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy weight")) {
                if (device)
                    cudaFree(device);
                return false;
            }
            device_allocations.push_back(device);
            found = weights.emplace(
                cache_key, DeviceWeight{device, type, n, k}).first;
            weights_by_device.emplace(device, &found->second);
        }
        tensor.device_data = found->second.data;
        tensor.device_offset = 0;
        return true;
    }

    bool run_matmul(const float* host_a, int lda, const Tensor& weight,
                    float* host_c, int ldc, int m, int n, int k,
                    Activation activation_kind, int act_begin, int act_len) {
        const DeviceWeight* prepared = find_weight(weight);
        if (!prepared || prepared->n != n || prepared->k != k || !host_a ||
            !host_c || ldc != n)
            return false;

        const size_t a_bytes = static_cast<size_t>(m) * lda * sizeof(float);
        const size_t a_elements = static_cast<size_t>(m) * lda;
        const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
        if (!reserve(activation, activation_bytes, a_bytes) ||
            !reserve(output, output_bytes, c_bytes) ||
            !report_cuda(cudaMemcpy(activation, host_a, a_bytes,
                                    cudaMemcpyHostToDevice),
                         "cudaMemcpy activation"))
            return false;

        const void* gemm_activation = activation;
        cudaDataType activation_type = CUDA_R_32F;
        if (prepared->type == CUDA_R_16F) {
            if (!reserve(activation_fp16, activation_fp16_bytes,
                         a_elements * sizeof(__half)))
                return false;
            constexpr int threads = 256;
            fp32_to_fp16<<<
                static_cast<unsigned>((a_elements + threads - 1) / threads),
                threads>>>(static_cast<const float*>(activation),
                           static_cast<__half*>(activation_fp16), a_elements);
            if (!report_cuda(cudaGetLastError(), "fp32_to_fp16"))
                return false;
            gemm_activation = activation_fp16;
            activation_type = CUDA_R_16F;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Row-major C[M,N] = A[M,K] * W[N,K]^T is the equivalent
        // column-major operation C_col[N,M] = W_col[K,N]^T * A_col[K,M].
        if (!report_cublas(
                cublasGemmEx(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                    prepared->data, prepared->type, k, gemm_activation,
                    activation_type, lda, &beta, output, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx") ||
            !report_cuda(cudaMemcpy(host_c, output, c_bytes,
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy output"))
            return false;

        if (activation_kind != Activation::NONE && act_len != 0) {
            const int begin = std::max(0, act_begin);
            const int end = act_len < 0 ? n : std::min(n, begin + act_len);
            for (int row = 0; row < m; ++row)
                for (int col = begin; col < end; ++col)
                    host_c[static_cast<size_t>(row) * ldc + col] =
                        apply_activation_scalar(
                            host_c[static_cast<size_t>(row) * ldc + col],
                            activation_kind);
        }
        return true;
    }
};

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {
    int count = 0;
    if (!report_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount") ||
        count <= 0)
        return;
    if (!report_cuda(cudaSetDevice(0), "cudaSetDevice") ||
        !report_cublas(cublasCreate(&impl_->cublas), "cublasCreate"))
        return;
    impl_->ok = true;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess)
        std::fprintf(stderr, "CudaBackend: using %s (sm_%d%d)\n",
                     properties.name, properties.major, properties.minor);
}

CudaBackend::~CudaBackend() = default;

bool CudaBackend::available() const { return impl_ && impl_->ok; }

void CudaBackend::clear_dispatch_error() {
    impl_->failed = false;
    impl_->cpu.clear_dispatch_error();
}

bool CudaBackend::dispatch_failed() const {
    return impl_->failed || impl_->cpu.dispatch_failed();
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
    }
}

void CudaBackend::wrap_weight_int4(Tensor& tensor,
                                   bool keep_native_experts) {
    if (!available() || keep_native_experts || tensor.prec != Precision::INT4 ||
        tensor.shape[0] <= 0 || tensor.shape[1] <= 0)
        return;
    const int n = static_cast<int>(tensor.shape[0]);
    const int k = static_cast<int>(tensor.shape[1]);
    std::vector<__half> dequantized(static_cast<size_t>(n) * k);

    if (tensor.is_q4_g32_packed && tensor.q4_g32_data && k % 32 == 0) {
        const auto* blocks =
            static_cast<const Q4B8G32Block*>(tensor.q4_g32_data);
        const int groups = k / 32;
        for (int row = 0; row < n; ++row) {
            for (int group = 0; group < groups; ++group) {
                const auto& block =
                    blocks[static_cast<size_t>(row / 8) * groups + group];
                const int lane = row % 8;
                for (int inner = 0; inner < 32; ++inner) {
                    const uint8_t packed = block.q[lane][inner / 2];
                    const int value = signed_nibble(
                        inner & 1 ? packed >> 4 : packed);
                    dequantized[static_cast<size_t>(row) * k +
                                group * 32 + inner] =
                        __float2half(static_cast<float>(value) *
                                     block.scales[lane]);
                }
            }
        }
    } else if (tensor.is_q4_g128_packed && tensor.q4_g128_data &&
               k % 128 == 0) {
        const auto* blocks =
            static_cast<const Q4B8G128Block*>(tensor.q4_g128_data);
        const int groups = k / 128;
        for (int row = 0; row < n; ++row) {
            for (int group = 0; group < groups; ++group) {
                const auto& block =
                    blocks[static_cast<size_t>(row / 8) * groups + group];
                const int lane = row % 8;
                for (int inner = 0; inner < 128; ++inner) {
                    const int qgroup = inner / 32;
                    const int qinner = inner % 32;
                    const uint8_t packed =
                        block.q[qgroup][lane][qinner / 2];
                    const int value = signed_nibble(
                        qinner & 1 ? packed >> 4 : packed);
                    dequantized[static_cast<size_t>(row) * k +
                                group * 128 + inner] =
                        __float2half(static_cast<float>(value) *
                                     block.scales[lane]);
                }
            }
        }
    } else {
        return;
    }

    const void* cache_key = tensor.rowmajor_data
        ? tensor.rowmajor_data : tensor.data;
    impl_->upload_weight(tensor, cache_key, dequantized.data(),
                         dequantized.size() * sizeof(__half), CUDA_R_16F,
                         n, k);
}

void CudaBackend::alloc_persistent(Tensor& tensor, size_t nbytes) {
    auto storage = std::make_unique<uint8_t[]>(nbytes);
    std::memset(storage.get(), 0, nbytes);
    tensor.data = storage.get();
    tensor.device_data = nullptr;
    tensor.device_offset = 0;
    tensor.mem_type = MemoryType::EXTERNAL;
    impl_->persistent_host.push_back(std::move(storage));
}

void CudaBackend::upload_input(Tensor&, const std::string&, const void*,
                               size_t) {
    // Hybrid correctness mode uploads activations lazily at CUDA MATMUL.
}

bool CudaBackend::supports_lm_head(const Tensor& weight) const {
    return impl_->find_weight(weight) != nullptr;
}

void CudaBackend::dispatch(const GraphNode& node,
                           const std::vector<const Tensor*>& inputs,
                           Tensor* output, ThreadPool* thread_pool) {
    if (node.op_type == OpType::MATMUL && inputs.size() >= 2 && inputs[0] &&
        inputs[1] && output && inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 &&
        impl_->find_weight(*inputs[1])) {
        const Tensor& a = *inputs[0];
        const Tensor& weight = *inputs[1];
        const int m = static_cast<int>(a.shape[1]);
        const int k = static_cast<int>(a.shape[0]);
        const int n = static_cast<int>(weight.shape[0]);
        const int lda = static_cast<int>(a.stride[1] / sizeof(float));
        const int ldc = static_cast<int>(output->stride[1] / sizeof(float));
        const Activation activation_kind = static_cast<Activation>(
            graph_params::get_i32(node.params, 0, 0));
        if (impl_->run_matmul(
                a.ptr<float>(), lda, weight, output->ptr<float>(), ldc,
                m, n, k, activation_kind,
                graph_params::get_i32(node.params, 1, 0),
                graph_params::get_i32(node.params, 2, -1)))
            return;
        impl_->failed = true;
        return;
    }
    impl_->cpu.dispatch(node, inputs, output, thread_pool);
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
    std::vector<float> host(static_cast<size_t>(k));
    const auto* source = static_cast<const uint8_t*>(activation.device_data) +
        activation.device_offset + activation_element_offset * sizeof(float);
    if (!report_cuda(cudaMemcpy(host.data(), source,
                                static_cast<size_t>(k) * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy lm_head activation")) {
        impl_->failed = true;
        return;
    }
    lm_head_gemv(host.data(), weight, output_host, n, k, activation_kind);
}
