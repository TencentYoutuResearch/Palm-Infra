#include "kernels/gated_residual.h"

#include "kernels/bf16.h"
#include "runtime/threading.h"

#include <cmath>
#include <cstdio>

namespace {

template <typename Fn>
void parallel_tokens(int tokens, ThreadPool* pool, Fn&& fn) {
    if (pool && pool->num_threads() > 1 && tokens > 1) {
        pool->parallel_for(0, tokens, 1,
                           [&](int, int begin, int end) { fn(begin, end); });
    } else {
        fn(0, tokens);
    }
}

bool dense_fp32_2d(const Tensor& tensor) {
    return tensor.prec == Precision::FP32 && tensor.data &&
           tensor.shape[2] == 1 && tensor.shape[3] == 1;
}

bool same_shape(const Tensor& left, const Tensor& right) {
    for (int dim = 0; dim < 4; ++dim) {
        if (left.shape[dim] != right.shape[dim]) return false;
    }
    return true;
}

} // namespace

bool kernel_group_rms_norm(const Tensor& input, const Tensor& weight,
                           Tensor& output, int group_size, float eps,
                           ThreadPool* thread_pool) {
    if (!dense_fp32_2d(input) || !dense_fp32_2d(weight) ||
        !dense_fp32_2d(output) || group_size <= 0 ||
        input.shape[0] % group_size != 0 ||
        weight.shape[0] != input.shape[0] || !same_shape(output, input)) {
        std::fprintf(
            stderr,
            "GROUP_RMS_NORM: invalid input=[%lld,%lld,%lld,%lld] p=%u "
            "weight=[%lld,%lld,%lld,%lld] p=%u output=[%lld,%lld,%lld,%lld] "
            "p=%u group=%d data=%d/%d/%d\n",
            static_cast<long long>(input.shape[0]),
            static_cast<long long>(input.shape[1]),
            static_cast<long long>(input.shape[2]),
            static_cast<long long>(input.shape[3]),
            static_cast<unsigned>(input.prec),
            static_cast<long long>(weight.shape[0]),
            static_cast<long long>(weight.shape[1]),
            static_cast<long long>(weight.shape[2]),
            static_cast<long long>(weight.shape[3]),
            static_cast<unsigned>(weight.prec),
            static_cast<long long>(output.shape[0]),
            static_cast<long long>(output.shape[1]),
            static_cast<long long>(output.shape[2]),
            static_cast<long long>(output.shape[3]),
            static_cast<unsigned>(output.prec), group_size,
            input.data != nullptr, weight.data != nullptr,
            output.data != nullptr);
        return false;
    }
    const int width = static_cast<int>(input.shape[0]);
    const int tokens = static_cast<int>(input.shape[1]);
    const int groups = width / group_size;
    const int in_stride = static_cast<int>(input.stride[1] / sizeof(float));
    const int out_stride = static_cast<int>(output.stride[1] / sizeof(float));
    const float* gamma = weight.ptr<float>();
    parallel_tokens(tokens, thread_pool, [&](int begin, int end) {
        for (int token = begin; token < end; ++token) {
            const float* src = input.ptr<float>() +
                static_cast<size_t>(token) * in_stride;
            float* dst = output.ptr<float>() +
                static_cast<size_t>(token) * out_stride;
            for (int group = 0; group < groups; ++group) {
                const int base = group * group_size;
                float square_sum = 0.0f;
                for (int dim = 0; dim < group_size; ++dim) {
                    const float value = src[base + dim];
                    square_sum += value * value;
                }
                const float inv_rms = 1.0f / std::sqrt(
                    square_sum / static_cast<float>(group_size) + eps);
                for (int dim = 0; dim < group_size; ++dim)
                    dst[base + dim] = src[base + dim] * inv_rms *
                                      gamma[base + dim];
            }
        }
    });
    return true;
}

bool kernel_gr_reduce(const Tensor& normalized, const Tensor& gates,
                      Tensor& output, int hidden_size, int hc_count,
                      ThreadPool* thread_pool) {
    const int wide = hidden_size * hc_count;
    if (!dense_fp32_2d(normalized) || !dense_fp32_2d(gates) ||
        !dense_fp32_2d(output) || hidden_size <= 0 || hc_count <= 0 ||
        normalized.shape[0] != wide || !same_shape(gates, normalized) ||
        output.shape[0] != hidden_size ||
        output.shape[1] != normalized.shape[1]) {
        std::fprintf(stderr, "GR_REDUCE: invalid tensor shape or precision\n");
        return false;
    }
    const int tokens = static_cast<int>(output.shape[1]);
    const int x_stride = static_cast<int>(normalized.stride[1] / sizeof(float));
    const int gate_stride = static_cast<int>(gates.stride[1] / sizeof(float));
    const int out_stride = static_cast<int>(output.stride[1] / sizeof(float));
    const float inv_count = 1.0f / static_cast<float>(hc_count);
    parallel_tokens(tokens, thread_pool, [&](int begin, int end) {
        for (int token = begin; token < end; ++token) {
            const float* x = normalized.ptr<float>() +
                static_cast<size_t>(token) * x_stride;
            const float* gate = gates.ptr<float>() +
                static_cast<size_t>(token) * gate_stride;
            float* dst = output.ptr<float>() +
                static_cast<size_t>(token) * out_stride;
            for (int dim = 0; dim < hidden_size; ++dim) {
                float sum = 0.0f;
                for (int stream = 0; stream < hc_count; ++stream) {
                    const int index = stream * hidden_size + dim;
                    sum += gate[index] * x[index];
                }
                // Qwen4-Exp keeps the residual stream in BF16. The read
                // reduction is the branch boundary, so match the reference
                // cast instead of leaking FP32 precision into attention/MoE.
                dst[dim] = mollm_round_to_bf16(sum * inv_count);
            }
        }
    });
    return true;
}

bool kernel_gr_inject(const Tensor& branch, const Tensor& residual,
                      const Tensor& gates, Tensor& output,
                      int hidden_size, int hc_count,
                      ThreadPool* thread_pool) {
    const int wide = hidden_size * hc_count;
    if (!dense_fp32_2d(branch) || !dense_fp32_2d(residual) ||
        !dense_fp32_2d(gates) || !dense_fp32_2d(output) ||
        hidden_size <= 0 || hc_count <= 0 ||
        branch.shape[0] != hidden_size || residual.shape[0] != wide ||
        !same_shape(output, residual) || gates.shape[0] != hc_count ||
        branch.shape[1] != residual.shape[1] ||
        gates.shape[1] != residual.shape[1]) {
        std::fprintf(stderr, "GR_INJECT: invalid tensor shape or precision\n");
        return false;
    }
    const int tokens = static_cast<int>(output.shape[1]);
    const int branch_stride = static_cast<int>(branch.stride[1] / sizeof(float));
    const int residual_stride = static_cast<int>(residual.stride[1] / sizeof(float));
    const int gate_stride = static_cast<int>(gates.stride[1] / sizeof(float));
    const int out_stride = static_cast<int>(output.stride[1] / sizeof(float));
    parallel_tokens(tokens, thread_pool, [&](int begin, int end) {
        for (int token = begin; token < end; ++token) {
            const float* update = branch.ptr<float>() +
                static_cast<size_t>(token) * branch_stride;
            const float* src = residual.ptr<float>() +
                static_cast<size_t>(token) * residual_stride;
            const float* gate = gates.ptr<float>() +
                static_cast<size_t>(token) * gate_stride;
            float* dst = output.ptr<float>() +
                static_cast<size_t>(token) * out_stride;
            for (int stream = 0; stream < hc_count; ++stream) {
                const int base = stream * hidden_size;
                for (int dim = 0; dim < hidden_size; ++dim)
                    dst[base + dim] = mollm_round_to_bf16(
                        src[base + dim] + gate[stream] * update[dim]);
            }
        }
    });
    return true;
}
