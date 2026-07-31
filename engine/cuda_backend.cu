#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "engine/cuda_backend.h"

#include "kernels/activations.h"
#include "kernels/quant_layouts.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
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

__device__ float cuda_activation(float value, int kind) {
    switch (kind) {
    case 1: return value / (1.0f + expf(-value));
    case 2: {
        const float inner = 0.7978845608f *
            (value + 0.044715f * value * value * value);
        return 0.5f * value * (1.0f + tanhf(inner));
    }
    case 3: return fmaxf(value, 0.0f);
    case 4: {
        const float positive = fmaxf(value, 0.0f);
        return positive * positive;
    }
    default: return value;
    }
}

__global__ void apply_activation_cuda(float* values, int rows, int columns,
                                      int kind, int begin, int end) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(rows) * columns;
    if (index >= count)
        return;
    const int column = static_cast<int>(index % columns);
    if (column >= begin && column < end)
        values[index] = cuda_activation(values[index], kind);
}

__global__ void binary_cuda(const float* lhs, const float* rhs, float* output,
                            size_t count, bool multiply) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        output[index] = multiply ? lhs[index] * rhs[index]
                                 : lhs[index] + rhs[index];
}

__global__ void sigmoid_mul_cuda(const float* value, const float* gate,
                                 float* output, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        output[index] = value[index] / (1.0f + expf(-gate[index]));
}

__global__ void unary_cuda(const float* input, float* output, size_t count,
                           int operation) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    const float value = input[index];
    switch (operation) {
    case 0: output[index] = value / (1.0f + expf(-value)); break;
    case 1: {
        const float inner = 0.7978845608f *
            (value + 0.044715f * value * value * value);
        output[index] = 0.5f * value * (1.0f + tanhf(inner));
        break;
    }
    case 2: output[index] = tanhf(value); break;
    case 3: output[index] = 1.0f / (1.0f + expf(-value)); break;
    case 4: output[index] = expf(value); break;
    case 5:
        output[index] = value > 20.0f ? value : log1pf(expf(value));
        break;
    }
}

__global__ void swiglu_cuda(const float* input, float* output,
                            size_t output_count, size_t half) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < output_count) {
        const size_t row = index / half;
        const size_t column = index % half;
        const size_t base = row * half * 2;
        const float gate = input[base + column];
        output[index] = gate / (1.0f + expf(-gate)) *
            input[base + half + column];
    }
}

__global__ void rms_norm_cuda(const float* input, const float* weight,
                              float* output, int width, int rows,
                              float epsilon) {
    const int row = blockIdx.x;
    if (row >= rows)
        return;
    float sum = 0.0f;
    const float* source = input + static_cast<size_t>(row) * width;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = source[column];
        sum += value * value;
    }
    __shared__ float reduction[256];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduction[0] / width + epsilon);
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        output[static_cast<size_t>(row) * width + column] =
            source[column] * inverse * weight[column];
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

    struct BoundaryBuffer {
        void* data = nullptr;
        size_t capacity = 0;
    };

    bool ok = false;
    bool failed = false;
    cublasHandle_t cublas = nullptr;
    CPUBackend cpu;
    std::unordered_map<const void*, DeviceWeight> weights;
    std::unordered_map<const void*, const DeviceWeight*> weights_by_device;
    std::vector<void*> device_allocations;
    std::vector<void*> managed_allocations;
    std::unordered_map<void*, size_t> managed_sizes;
    std::multimap<size_t, void*> free_managed;
    std::unordered_map<std::string, BoundaryBuffer> boundary_buffers;
    std::unordered_map<uint32_t, uint64_t> native_ops;
    std::unordered_map<uint32_t, uint64_t> fallback_ops;
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
        for (void* allocation : managed_allocations)
            cudaFree(allocation);
        for (auto& entry : boundary_buffers)
            if (entry.second.data)
                cudaFree(entry.second.data);
        if (activation)
            cudaFree(activation);
        if (activation_fp16)
            cudaFree(activation_fp16);
        if (output)
            cudaFree(output);
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

    void* acquire_managed(size_t bytes) {
        auto found = free_managed.lower_bound(bytes);
        if (found != free_managed.end()) {
            void* pointer = found->second;
            free_managed.erase(found);
            return pointer;
        }
        void* pointer = nullptr;
        if (!report_cuda(cudaMallocManaged(&pointer, bytes),
                         "cudaMallocManaged output"))
            return nullptr;
        managed_allocations.push_back(pointer);
        managed_sizes.emplace(pointer, bytes);
        return pointer;
    }

    void release_managed(void* pointer) {
        const auto found = managed_sizes.find(pointer);
        if (found != managed_sizes.end())
            free_managed.emplace(found->second, pointer);
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

    bool run_matmul_device(const float* device_a, int lda,
                           const Tensor& weight, float* device_c, int ldc,
                           int m, int n, int k,
                           Activation activation_kind, int act_begin,
                           int act_len) {
        const DeviceWeight* prepared = find_weight(weight);
        if (!prepared || prepared->n != n || prepared->k != k || !device_a ||
            !device_c || ldc != n)
            return false;

        const size_t a_elements = static_cast<size_t>(m) * lda;
        const void* gemm_activation = device_a;
        cudaDataType activation_type = CUDA_R_32F;
        if (prepared->type == CUDA_R_16F) {
            if (!reserve(activation_fp16, activation_fp16_bytes,
                         a_elements * sizeof(__half)))
                return false;
            constexpr int threads = 256;
            fp32_to_fp16<<<
                static_cast<unsigned>((a_elements + threads - 1) / threads),
                threads>>>(device_a,
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
                    activation_type, lda, &beta, device_c, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx"))
            return false;

        if (activation_kind != Activation::NONE && act_len != 0) {
            const int begin = std::max(0, act_begin);
            const int end = act_len < 0 ? n : std::min(n, begin + act_len);
            const size_t count = static_cast<size_t>(m) * n;
            constexpr int threads = 256;
            apply_activation_cuda<<<
                static_cast<unsigned>((count + threads - 1) / threads),
                threads>>>(device_c, m, n,
                           static_cast<int>(activation_kind), begin, end);
            if (!report_cuda(cudaGetLastError(), "apply_activation_cuda"))
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
            !report_cuda(cudaMemcpy(activation, host_a, a_bytes,
                                    cudaMemcpyHostToDevice),
                         "cudaMemcpy activation") ||
            !run_matmul_device(
                static_cast<const float*>(activation), lda, weight,
                static_cast<float*>(output), ldc, m, n, k, activation_kind,
                act_begin, act_len) ||
            !report_cuda(cudaMemcpy(host_c, output, c_bytes,
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy output"))
            return false;
        return true;
    }
};

namespace {

template <typename T>
T* device_pointer(const Tensor& tensor) {
    if (!tensor.device_data)
        return nullptr;
    return reinterpret_cast<T*>(
        static_cast<uint8_t*>(tensor.device_data) + tensor.device_offset);
}

template <typename T>
const T* device_pointer_const(const Tensor& tensor) {
    if (!tensor.device_data)
        return nullptr;
    return reinterpret_cast<const T*>(
        static_cast<const uint8_t*>(tensor.device_data) +
        tensor.device_offset);
}

bool fp32_contiguous(const Tensor& tensor) {
    return tensor.prec == Precision::FP32 && tensor.is_contiguous();
}

}  // namespace

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

void* CudaBackend::alloc_output(Tensor& output, size_t nbytes, BufferPool*) {
    if (!available() || nbytes == 0)
        return nullptr;
    void* pointer = impl_->acquire_managed(nbytes);
    if (!pointer) {
        impl_->failed = true;
        return nullptr;
    }
    output.data = pointer;
    output.device_data = pointer;
    output.device_offset = 0;
    output.mem_type = MemoryType::POOLED;
    output.owner_id = 0;
    output.storage_id = 0;
    return pointer;
}

void CudaBackend::free_output(Tensor& tensor, BufferPool*) {
    if (tensor.device_data)
        impl_->release_managed(tensor.device_data);
}

void CudaBackend::synchronize_for_host_read() {
    if (!report_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"))
        impl_->failed = true;
}

void CudaBackend::begin_graph() {}

void CudaBackend::end_graph() { synchronize_for_host_read(); }

void CudaBackend::alloc_persistent(Tensor& tensor, size_t nbytes) {
    void* storage = nullptr;
    if (!available() || nbytes == 0 ||
        !report_cuda(cudaMallocManaged(&storage, nbytes),
                     "cudaMallocManaged persistent")) {
        impl_->failed = true;
        return;
    }
    impl_->managed_allocations.push_back(storage);
    std::memset(storage, 0, nbytes);
    tensor.data = storage;
    tensor.device_data = storage;
    tensor.device_offset = 0;
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
        if (!report_cuda(cudaMalloc(&buffer.data, nbytes),
                         "cudaMalloc input")) {
            impl_->failed = true;
            return;
        }
        buffer.capacity = nbytes;
    }
    if (!report_cuda(cudaMemcpy(buffer.data, host_source, nbytes,
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy input")) {
        impl_->failed = true;
        return;
    }
    tensor.device_data = buffer.data;
    tensor.device_offset = 0;
}

bool CudaBackend::supports_lm_head(const Tensor& weight) const {
    return impl_->find_weight(weight) != nullptr;
}

void CudaBackend::dispatch(const GraphNode& node,
                           const std::vector<const Tensor*>& inputs,
                           Tensor* output, ThreadPool* thread_pool) {
    auto record_native = [&]() {
        ++impl_->native_ops[static_cast<uint32_t>(node.op_type)];
    };
    constexpr int threads = 256;

    if (node.op_type == OpType::INPUT ||
        node.op_type == OpType::CONSTANT) {
        record_native();
        return;
    }

    if (node.op_type == OpType::RESHAPE && !inputs.empty() && inputs[0] &&
        output && inputs[0]->is_contiguous()) {
        const int64_t shape[4] = {output->shape[0], output->shape[1],
                                  output->shape[2], output->shape[3]};
        *output = *inputs[0];
        for (int dimension = 0; dimension < 4; ++dimension)
            output->shape[dimension] = shape[dimension];
        output->compute_strides();
        record_native();
        return;
    }

    if (node.op_type == OpType::PERMUTE && !inputs.empty() && inputs[0] &&
        output) {
        const Tensor& source = *inputs[0];
        const int axis[4] = {
            graph_params::get_i32(node.params, 0, 0),
            graph_params::get_i32(node.params, 1, 1),
            graph_params::get_i32(node.params, 2, 2),
            graph_params::get_i32(node.params, 3, 3),
        };
        Tensor view = source;
        for (int dimension = 0; dimension < 4; ++dimension) {
            view.shape[axis[dimension]] = source.shape[dimension];
            view.stride[axis[dimension]] = source.stride[dimension];
        }
        *output = view;
        record_native();
        return;
    }

    if (node.op_type == OpType::SLICE && !inputs.empty() && inputs[0] &&
        output) {
        const Tensor& source = *inputs[0];
        const int dimension = graph_params::get_i32(node.params, 0, 0);
        const int offset = graph_params::get_i32(node.params, 1, 0);
        const int size = graph_params::get_i32(
            node.params, 2, static_cast<int>(source.shape[dimension]));
        *output = source;
        output->device_offset = source.device_offset +
            static_cast<size_t>(offset) * source.stride[dimension];
        output->shape[dimension] = size;
        record_native();
        return;
    }

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
        const float* device_a = device_pointer_const<float>(a);
        float* device_c = device_pointer<float>(*output);
        const bool ok = device_a && device_c
            ? impl_->run_matmul_device(
                  device_a, lda, weight, device_c, ldc, m, n, k,
                  activation_kind,
                  graph_params::get_i32(node.params, 1, 0),
                  graph_params::get_i32(node.params, 2, -1))
            : impl_->run_matmul(
                  a.ptr<float>(), lda, weight, output->ptr<float>(), ldc,
                  m, n, k, activation_kind,
                  graph_params::get_i32(node.params, 1, 0),
                  graph_params::get_i32(node.params, 2, -1));
        if (ok) {
            record_native();
            return;
        }
        impl_->failed = true;
        return;
    }

    if (node.op_type == OpType::RMS_NORM && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output && fp32_contiguous(*inputs[0]) &&
        fp32_contiguous(*inputs[1]) && fp32_contiguous(*output)) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        const float* weight = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(inputs[0]->shape[0]);
        const int rows = width > 0
            ? static_cast<int>(inputs[0]->nelements() / width) : 0;
        if (source && weight && destination && rows > 0) {
            rms_norm_cuda<<<rows, threads>>>(
                source, weight, destination, width, rows,
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!report_cuda(cudaGetLastError(), "rms_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::ADD || node.op_type == OpType::MUL) &&
        inputs.size() >= 2 && inputs[0] && inputs[1] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        fp32_contiguous(*output) &&
        inputs[0]->nelements() == inputs[1]->nelements()) {
        const float* lhs = device_pointer_const<float>(*inputs[0]);
        const float* rhs = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (lhs && rhs && destination) {
            binary_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                                threads), threads>>>(
                lhs, rhs, destination, count,
                node.op_type == OpType::MUL);
            if (!report_cuda(cudaGetLastError(), "binary_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SIGMOID_MUL && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output && fp32_contiguous(*inputs[0]) &&
        fp32_contiguous(*inputs[1]) && fp32_contiguous(*output) &&
        inputs[0]->nelements() == inputs[1]->nelements()) {
        const float* value = device_pointer_const<float>(*inputs[0]);
        const float* gate = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (value && gate && destination) {
            sigmoid_mul_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                                     threads), threads>>>(
                value, gate, destination, count);
            if (!report_cuda(cudaGetLastError(), "sigmoid_mul_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    int unary_operation = -1;
    switch (node.op_type) {
    case OpType::SILU: unary_operation = 0; break;
    case OpType::GELU: unary_operation = 1; break;
    case OpType::TANH: unary_operation = 2; break;
    case OpType::SIGMOID:
    case OpType::SIGMOID_EXACT: unary_operation = 3; break;
    case OpType::EXP:
    case OpType::EXP_EXACT: unary_operation = 4; break;
    case OpType::SOFTPLUS: unary_operation = 5; break;
    default: break;
    }
    if (unary_operation >= 0 && !inputs.empty() && inputs[0] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*output)) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (source && destination) {
            unary_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                               threads), threads>>>(
                source, destination, count, unary_operation);
            if (!report_cuda(cudaGetLastError(), "unary_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SWIGLU && !inputs.empty() && inputs[0] &&
        output && fp32_contiguous(*inputs[0]) && fp32_contiguous(*output) &&
        inputs[0]->shape[0] == output->shape[0] * 2) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (source && destination) {
            swiglu_cuda<<<static_cast<unsigned>((count + threads - 1) /
                                               threads), threads>>>(
                source, destination, count,
                static_cast<size_t>(output->shape[0]));
            if (!report_cuda(cudaGetLastError(), "swiglu_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    synchronize_for_host_read();
    if (impl_->failed)
        return;
    std::vector<Tensor> host_tensors;
    std::vector<const Tensor*> host_inputs;
    host_tensors.reserve(inputs.size());
    host_inputs.reserve(inputs.size());
    for (const Tensor* input : inputs) {
        if (!input) {
            host_inputs.push_back(nullptr);
            continue;
        }
        host_tensors.push_back(*input);
        Tensor& host = host_tensors.back();
        if (host.data && host.device_offset)
            host.data = static_cast<uint8_t*>(host.data) +
                host.device_offset;
        host.device_data = nullptr;
        host.device_offset = 0;
        host_inputs.push_back(&host);
    }
    impl_->cpu.clear_dispatch_error();
    impl_->cpu.dispatch(node, host_inputs, output, thread_pool);
    if (impl_->cpu.dispatch_failed()) {
        impl_->failed = true;
        return;
    }
    ++impl_->fallback_ops[static_cast<uint32_t>(node.op_type)];
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
        !report_cuda(cudaMemcpy(output_host, impl_->output, output_size,
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy lm_head output")) {
        impl_->failed = true;
        return;
    }
    ++impl_->native_ops[static_cast<uint32_t>(OpType::MATMUL)];
}
