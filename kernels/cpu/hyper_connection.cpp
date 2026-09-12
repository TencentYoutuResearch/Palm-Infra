#include "kernels/cpu/hyper_connection.h"

#include "core/bf16.h"
#include "runtime/threading.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

namespace {

inline float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

bool validate_common(const Tensor& x, const Tensor& fn,
                     const Tensor& scale, const Tensor& base,
                     int hidden_size, int hc_mult, int mix_size) {
    return x.prec == Precision::FP32 && fn.prec == Precision::FP32 &&
           scale.prec == Precision::FP32 && base.prec == Precision::FP32 &&
           x.data && fn.data && scale.data && base.data &&
           hidden_size > 0 && hc_mult > 0 &&
           x.shape[0] == static_cast<int64_t>(hidden_size) * hc_mult &&
           fn.shape[0] == mix_size && fn.shape[1] == x.shape[0] &&
           scale.shape[0] >= (mix_size == hc_mult ? 1 : 3) &&
           base.shape[0] == mix_size;
}

float mean_square(const float* values, int count) {
    float sum = 0.0f;
    for (int index = 0; index < count; ++index)
        sum += values[index] * values[index];
    return sum / static_cast<float>(count);
}

void reduce_hidden(const float* wide, const float* coefficients,
                   float* output, int hidden_size, int hc_mult) {
    for (int dim = 0; dim < hidden_size; ++dim) {
        float sum = 0.0f;
        for (int h = 0; h < hc_mult; ++h) {
            sum += coefficients[h] *
                   wide[static_cast<size_t>(h) * hidden_size + dim];
        }
        output[dim] = mollm_round_to_bf16(sum);
    }
}

void project_hc(const Tensor& x, const Tensor& fn, float* output,
                int rows, ThreadPool* pool) {
    const int tokens = static_cast<int>(x.shape[1]);
    const int width = static_cast<int>(x.shape[0]);
    const int x_stride = static_cast<int>(x.stride[1] / sizeof(float));
    const float* x_data = x.ptr<float>();
    const float* weight_data = fn.ptr<float>();
    auto project_rows = [&](int token, int row_begin, int row_end) {
        const float* token_x =
            x_data + static_cast<size_t>(token) * x_stride;
        float* token_output =
            output + static_cast<size_t>(token) * rows;
        for (int row = row_begin; row < row_end; ++row) {
            const float* weight =
                weight_data + static_cast<size_t>(row) * width;
            float sum = 0.0f;
            // Preserve the checkpoint-validated accumulation order exactly.
            for (int index = 0; index < width; ++index)
                sum += token_x[index] * weight[index];
            token_output[row] = sum;
        }
    };
    if (tokens == 1 && pool && pool->num_threads() > 1 && rows > 1) {
        const int grain =
            std::max(1, (rows + pool->num_threads() - 1) /
                            pool->num_threads());
        pool->parallel_for(
            0, rows, grain,
            [&](int, int begin, int end) {
                project_rows(0, begin, end);
            });
    } else if (tokens > 1 && pool && pool->num_threads() > 1) {
        pool->parallel_for(
            0, tokens, 1,
            [&](int, int begin, int end) {
                for (int token = begin; token < end; ++token)
                    project_rows(token, 0, rows);
            });
    } else {
        for (int token = 0; token < tokens; ++token)
            project_rows(token, 0, rows);
    }
}

void parallel_tokens(int tokens, ThreadPool* pool,
                     const std::function<void(int, int)>& run) {
    if (pool && pool->num_threads() > 1 && tokens > 1) {
        pool->parallel_for(
            0, tokens, 1,
            [&](int, int begin, int end) { run(begin, end); });
    } else {
        run(0, tokens);
    }
}

} // namespace

bool kernel_hc_pre(const Tensor& x, const Tensor& fn, const Tensor& scale,
                   const Tensor& base, Tensor& packed, int hidden_size,
                   int hc_mult, int sinkhorn_iters, float norm_eps,
                   float sinkhorn_eps, ThreadPool* thread_pool) {
    const int mix_size = (2 + hc_mult) * hc_mult;
    const int packed_size = hidden_size + hc_mult + hc_mult * hc_mult;
    if (!validate_common(
            x, fn, scale, base, hidden_size, hc_mult, mix_size) ||
        packed.prec != Precision::FP32 || !packed.data ||
        packed.shape[0] != packed_size ||
        packed.shape[1] != x.shape[1] || sinkhorn_iters <= 0) {
        std::fprintf(stderr, "HC_PRE: invalid tensor shape or precision\n");
        return false;
    }
    const int tokens = static_cast<int>(x.shape[1]);
    const int wide = hidden_size * hc_mult;
    const float* x_data = x.ptr<float>();
    const float* scale_data = scale.ptr<float>();
    const float* base_data = base.ptr<float>();
    float* output = packed.ptr<float>();
    const int x_stride = static_cast<int>(x.stride[1] / sizeof(float));
    const int out_stride =
        static_cast<int>(packed.stride[1] / sizeof(float));

    // Decode has one token but 24 independent projection rows. Parallelize
    // those rows while retaining each row's original scalar reduction order.
    static thread_local std::vector<float> projected;
    projected.resize(static_cast<size_t>(mix_size) * tokens);
    project_hc(x, fn, projected.data(), mix_size, thread_pool);
    // Resolve TLS on the caller. Worker lambdas must not look up their own
    // empty thread-local vector.
    float* projected_data = projected.data();

    parallel_tokens(tokens, thread_pool, [&](int begin, int end) {
        std::vector<float> pre(static_cast<size_t>(hc_mult));
        std::vector<float> combination(
            static_cast<size_t>(hc_mult) * hc_mult);
        std::vector<float> row_sums(static_cast<size_t>(hc_mult));
        std::vector<float> col_sums(static_cast<size_t>(hc_mult));
        for (int token = begin; token < end; ++token) {
            const float* token_x =
                x_data + static_cast<size_t>(token) * x_stride;
            const float rsqrt = 1.0f / std::sqrt(
                mean_square(token_x, wide) + norm_eps);
            float* mixes =
                projected_data + static_cast<size_t>(token) * mix_size;
            for (int row = 0; row < mix_size; ++row)
                mixes[row] *= rsqrt;

            float* token_out =
                output + static_cast<size_t>(token) * out_stride;
            float* post_out = token_out + hidden_size;
            float* comb_out = post_out + hc_mult;
            for (int h = 0; h < hc_mult; ++h) {
                pre[h] = sigmoid(
                    mixes[h] * scale_data[0] + base_data[h]) +
                    sinkhorn_eps;
                post_out[h] = 2.0f * sigmoid(
                    mixes[hc_mult + h] * scale_data[1] +
                    base_data[hc_mult + h]);
            }
            for (int row = 0; row < hc_mult; ++row) {
                float maximum = -INFINITY;
                for (int col = 0; col < hc_mult; ++col) {
                    const int index =
                        2 * hc_mult + row * hc_mult + col;
                    const float value =
                        mixes[index] * scale_data[2] + base_data[index];
                    combination[row * hc_mult + col] = value;
                    maximum = std::max(maximum, value);
                }
                float sum = 0.0f;
                for (int col = 0; col < hc_mult; ++col) {
                    float& value = combination[row * hc_mult + col];
                    value = std::exp(value - maximum);
                    sum += value;
                }
                for (int col = 0; col < hc_mult; ++col)
                    combination[row * hc_mult + col] =
                        combination[row * hc_mult + col] / sum +
                        sinkhorn_eps;
            }
            for (int iteration = 0; iteration < sinkhorn_iters; ++iteration) {
                if (iteration != 0) {
                    std::fill(row_sums.begin(), row_sums.end(), 0.0f);
                    for (int row = 0; row < hc_mult; ++row) {
                        for (int col = 0; col < hc_mult; ++col)
                            row_sums[row] +=
                                combination[row * hc_mult + col];
                    }
                    for (int row = 0; row < hc_mult; ++row) {
                        for (int col = 0; col < hc_mult; ++col)
                            combination[row * hc_mult + col] /=
                                row_sums[row] + sinkhorn_eps;
                    }
                }
                std::fill(col_sums.begin(), col_sums.end(), 0.0f);
                for (int row = 0; row < hc_mult; ++row) {
                    for (int col = 0; col < hc_mult; ++col)
                        col_sums[col] +=
                            combination[row * hc_mult + col];
                }
                for (int row = 0; row < hc_mult; ++row) {
                    for (int col = 0; col < hc_mult; ++col)
                        combination[row * hc_mult + col] /=
                            col_sums[col] + sinkhorn_eps;
                }
            }
            // Official DeepSeek-V4 returns HC-pre's reduced state in the
            // original BF16 activation dtype.
            reduce_hidden(
                token_x, pre.data(), token_out, hidden_size, hc_mult);
            std::copy(combination.begin(), combination.end(), comb_out);
        }
    });
    return true;
}

bool kernel_hc_post(const Tensor& branch, const Tensor& residual,
                    const Tensor& packed, Tensor& output, int hidden_size,
                    int hc_mult, ThreadPool* thread_pool) {
    const int packed_size = hidden_size + hc_mult + hc_mult * hc_mult;
    if (branch.prec != Precision::FP32 ||
        residual.prec != Precision::FP32 ||
        packed.prec != Precision::FP32 ||
        output.prec != Precision::FP32 || !branch.data || !residual.data ||
        !packed.data || !output.data || hidden_size <= 0 || hc_mult <= 0 ||
        branch.shape[0] != hidden_size ||
        residual.shape[0] != static_cast<int64_t>(hidden_size) * hc_mult ||
        packed.shape[0] != packed_size ||
        output.shape[0] != residual.shape[0] ||
        branch.shape[1] != residual.shape[1] ||
        packed.shape[1] != residual.shape[1] ||
        output.shape[1] != residual.shape[1]) {
        std::fprintf(stderr, "HC_POST: invalid tensor shape or precision\n");
        return false;
    }
    const int tokens = static_cast<int>(output.shape[1]);
    const int branch_stride =
        static_cast<int>(branch.stride[1] / sizeof(float));
    const int residual_stride =
        static_cast<int>(residual.stride[1] / sizeof(float));
    const int packed_stride =
        static_cast<int>(packed.stride[1] / sizeof(float));
    const int output_stride =
        static_cast<int>(output.stride[1] / sizeof(float));
    parallel_tokens(tokens, thread_pool, [&](int begin, int end) {
        for (int token = begin; token < end; ++token) {
            const float* branch_row =
                branch.ptr<float>() + static_cast<size_t>(token) *
                                          branch_stride;
            const float* residual_row =
                residual.ptr<float>() + static_cast<size_t>(token) *
                                            residual_stride;
            const float* post =
                packed.ptr<float>() + static_cast<size_t>(token) *
                                          packed_stride +
                hidden_size;
            const float* combination = post + hc_mult;
            float* output_row =
                output.ptr<float>() + static_cast<size_t>(token) *
                                          output_stride;
            for (int out_h = 0; out_h < hc_mult; ++out_h) {
                for (int dim = 0; dim < hidden_size; ++dim) {
                    float residual_sum = 0.0f;
                    for (int in_h = 0; in_h < hc_mult; ++in_h) {
                        residual_sum +=
                            combination[in_h * hc_mult + out_h] *
                            residual_row[in_h * hidden_size + dim];
                    }
                    const float value =
                        post[out_h] * branch_row[dim] + residual_sum;
                    // HC-post also casts the wide residual state back to the
                    // branch activation dtype (BF16 in the official runtime).
                    output_row[out_h * hidden_size + dim] =
                        mollm_round_to_bf16(value);
                }
            }
        }
    });
    return true;
}

bool kernel_hc_head(const Tensor& x, const Tensor& fn, const Tensor& scale,
                    const Tensor& base, Tensor& output, int hidden_size,
                    int hc_mult, float norm_eps, float hc_eps,
                    ThreadPool* thread_pool) {
    if (!validate_common(
            x, fn, scale, base, hidden_size, hc_mult, hc_mult) ||
        output.prec != Precision::FP32 || !output.data ||
        output.shape[0] != hidden_size || output.shape[1] != x.shape[1]) {
        std::fprintf(stderr, "HC_HEAD: invalid tensor shape or precision\n");
        return false;
    }
    const int tokens = static_cast<int>(x.shape[1]);
    const int wide = hidden_size * hc_mult;
    const int x_stride = static_cast<int>(x.stride[1] / sizeof(float));
    const int output_stride =
        static_cast<int>(output.stride[1] / sizeof(float));
    static thread_local std::vector<float> projected;
    projected.resize(static_cast<size_t>(hc_mult) * tokens);
    project_hc(x, fn, projected.data(), hc_mult, thread_pool);
    const float* projected_data = projected.data();

    parallel_tokens(tokens, thread_pool, [&](int begin, int end) {
        std::vector<float> pre(static_cast<size_t>(hc_mult));
        for (int token = begin; token < end; ++token) {
            const float* token_x =
                x.ptr<float>() + static_cast<size_t>(token) * x_stride;
            const float rsqrt = 1.0f / std::sqrt(
                mean_square(token_x, wide) + norm_eps);
            const float* token_projected =
                projected_data + static_cast<size_t>(token) * hc_mult;
            for (int h = 0; h < hc_mult; ++h) {
                pre[h] = sigmoid(
                    token_projected[h] * rsqrt *
                        scale.ptr<float>()[0] +
                    base.ptr<float>()[h]) + hc_eps;
            }
            float* output_row =
                output.ptr<float>() + static_cast<size_t>(token) *
                                          output_stride;
            reduce_hidden(
                token_x, pre.data(), output_row, hidden_size, hc_mult);
        }
    });
    return true;
}
