#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/matmul/matmul_internal.h"
#include "kernels/cpu/matmul/matmul_profile.h"

MatmulConfig g_matmul_config;

// Debug override used by precision-comparison tests and tools.
bool g_mollm_force_fp32_acc = false;

void kernel_matmul_batch(const std::vector<const Tensor*>& pairs,
                         Tensor& output, ThreadPool* thread_pool) {
    if ((pairs.size() & 1) != 0 || pairs.empty())
        return;

    const size_t batch = pairs.size() / 2;
    std::vector<Tensor> inputs;
    std::vector<Tensor> weights;
    std::vector<Tensor> outputs;
    inputs.reserve(batch);
    weights.reserve(batch);
    outputs.reserve(batch);

    size_t output_offset = 0;
    bool valid = output.prec == Precision::FP32 && output.data;
    for (size_t i = 0; i < batch; ++i) {
        const Tensor* input = pairs[i * 2];
        const Tensor* weight = pairs[i * 2 + 1];
        if (!input || !weight || input->shape[0] != weight->shape[1] ||
            input->shape[1] != output.shape[1]) {
            valid = false;
            break;
        }
        inputs.push_back(*input);
        weights.push_back(*weight);
        Tensor slice = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, weight->shape[0],
            input->shape[1], 1, 1,
            output.ptr<float>() + output_offset);
        outputs.push_back(slice);
        output_offset +=
            (size_t)weight->shape[0] * (size_t)input->shape[1];
    }
    if (!valid || output.shape[2] != (int64_t)batch ||
        output_offset != output.nelements())
        return;

    if (thread_pool && output.shape[1] == 1 &&
        kernel_matmul_int4_gemv_batch(inputs, weights, outputs, thread_pool)) {
        return;
    }
    if (thread_pool && output.shape[1] == 1 &&
        kernel_matmul_int8_gemv_batch(inputs, weights, outputs, thread_pool)) {
        return;
    }
    if (thread_pool && output.shape[1] == 1 &&
        kernel_matmul_mxfp4_gemv_batch(
            inputs, weights, outputs, thread_pool)) {
        return;
    }
    if (thread_pool && output.shape[1] == 1 &&
        kernel_matmul_nvfp4_gemv_batch(
            inputs, weights, outputs, thread_pool)) {
        return;
    }
    if (thread_pool && output.shape[1] == 1 &&
        kernel_matmul_fp32_gemv_batch(
            inputs, weights, outputs, thread_pool)) {
        return;
    }
    for (size_t i = 0; i < batch; ++i)
        kernel_matmul_fp32(inputs[i], weights[i], outputs[i], thread_pool);
}

void kernel_matmul_fp32(const Tensor& A, const Tensor& B, Tensor& C,
                        ThreadPool* thread_pool, Activation act,
                        int act_n_begin, int act_n_len,
                        bool force_fp32_acc) {
    MatmulTimer timer;

    switch (B.prec) {
    case Precision::INT4:
        matmul_dispatch_int4(A, B, C, thread_pool, act, act_n_begin, act_n_len,
                             timer);
        return;
    case Precision::INT8:
        matmul_dispatch_int8(A, B, C, thread_pool, act, act_n_begin, act_n_len,
                             timer, false);
        return;
    case Precision::MXFP4:
        matmul_dispatch_mxfp4(A, B, C, thread_pool, act, act_n_begin,
                              act_n_len, timer);
        return;
    case Precision::FP8_E4M3:
        matmul_dispatch_fp8_e4m3(A, B, C, thread_pool, act, act_n_begin,
                                 act_n_len, timer);
        return;
    case Precision::NVFP4:
        matmul_dispatch_nvfp4(A, B, C, thread_pool, act, act_n_begin,
                              act_n_len, timer);
        return;
    default:
        matmul_dispatch_dense(A, B, C, thread_pool, act, act_n_begin, act_n_len,
                              timer, force_fp32_acc);
        return;
    }
}
