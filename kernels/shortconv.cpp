#include "kernels/shortconv.h"

#include <cmath>
#include <vector>

#if HAS_NEON
#include <arm_neon.h>
#endif

void kernel_shortconv(const OpParams& params,
                      const std::vector<const Tensor*>& inputs, Tensor& output,
                      ThreadPool* thread_pool) {
    if (inputs.size() < 3 || !inputs[0] || !inputs[1] || !inputs[2])
        return;

    const Tensor& x = *inputs[0];
    const Tensor& weight = *inputs[1];
    const int kernel_size = graph_params::get_i32(params, 0, 4);
    const int groups = static_cast<int>(x.shape[0]);
    const int seq_len = static_cast<int>(x.shape[1]);

    const float* x_data = x.ptr<float>();
    const float* weight_data = weight.ptr<float>();
    float* state_data = static_cast<float*>(inputs[2]->data);
    float* output_data = output.ptr<float>();

#if HAS_NEON
    // The decode path is common enough to avoid allocating the prefill
    // staging buffer. Qwen3.5 uses a fixed four-element convolution.
    if (seq_len == 1 && kernel_size == 4) {
        constexpr int prefix_len = 3;
        for (int group = 0; group < groups; ++group) {
            float* state = state_data + group * prefix_len;
            const float value = x_data[group];
            const float* weights = weight_data + group * kernel_size;
            const float sum = state[0] * weights[0] + state[1] * weights[1] +
                              state[2] * weights[2] + value * weights[3];
            const float sigmoid = 1.f / (1.f + std::exp(-sum));
            output_data[group] = sum * sigmoid;

            state[0] = state[1];
            state[1] = state[2];
            state[2] = value;
        }
        return;
    }
#endif

    const int n_real = graph_params::get_i32(params, 1, seq_len);
    const int prefix_len = kernel_size - 1;
    const int total_len = prefix_len + seq_len;
    const int process_len = (n_real > 0 && n_real < seq_len) ? n_real : seq_len;
    std::vector<float> staged(groups * total_len);

    // Transpose [seq, groups] into group-major staging storage while reading
    // the input sequentially.
    for (int token = 0; token < seq_len; ++token) {
        const float* input_row = x_data + token * groups;
        float* staged_row = staged.data() + prefix_len + token;
        for (int group = 0; group < groups; ++group)
            staged_row[group * total_len] = input_row[group];
    }

    auto process_groups = [&](int /*thread_id*/, int begin, int end) {
        for (int group = begin; group < end; ++group) {
            float* values = staged.data() + group * total_len;
            float* state = state_data + group * prefix_len;
            for (int i = 0; i < prefix_len; ++i)
                values[i] = state[i];

            const float* weights = weight_data + group * kernel_size;
            float* result = output_data + group * seq_len;
            for (int i = 0; i < seq_len; ++i)
                result[i] = 0.f;

#if HAS_NEON
            if (kernel_size == 4) {
                const float32x4_t weight4 = vld1q_f32(weights);
                for (int token = 0; token < process_len; ++token) {
                    const float32x4_t value4 = vld1q_f32(values + token);
                    const float sum = vaddvq_f32(vmulq_f32(value4, weight4));
                    const float sigmoid = 1.f / (1.f + std::exp(-sum));
                    result[token] = sum * sigmoid;
                }
            } else
#endif
            {
                for (int token = 0; token < process_len; ++token) {
                    float sum = 0.f;
                    for (int i = 0; i < kernel_size; ++i)
                        sum += values[token + i] * weights[i];
                    const float sigmoid = 1.f / (1.f + std::exp(-sum));
                    result[token] = sum * sigmoid;
                }
            }

            if (process_len > 0) {
                const int last_real = prefix_len + process_len - 1;
                for (int i = 0; i < prefix_len; ++i)
                    state[i] = values[last_real - prefix_len + 1 + i];
            }
        }
    };

    if (thread_pool && groups >= 4) {
        int chunk = (groups + thread_pool->num_threads() - 1) /
                    thread_pool->num_threads();
        if (chunk < 1)
            chunk = 1;
        thread_pool->parallel_for(0, groups, chunk, process_groups);
    } else {
        process_groups(0, 0, groups);
    }
}
