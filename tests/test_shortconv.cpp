#include "kernels/shortconv.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition, message)                                              \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::fprintf(stderr, "FAIL: %s\n", message);                       \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

void reference_shortconv(const std::vector<float>& input,
                         const std::vector<float>& weights,
                         std::vector<float>& state, std::vector<float>& output,
                         int groups, int seq_len, int kernel_size, int n_real) {
    const int prefix_len = kernel_size - 1;
    const int process_len = (n_real > 0 && n_real < seq_len) ? n_real : seq_len;
    std::vector<float> values(prefix_len + seq_len);

    for (int group = 0; group < groups; ++group) {
        for (int i = 0; i < prefix_len; ++i)
            values[i] = state[group * prefix_len + i];
        for (int token = 0; token < seq_len; ++token)
            values[prefix_len + token] = input[token * groups + group];

        for (int token = 0; token < process_len; ++token) {
            float sum = 0.f;
            for (int i = 0; i < kernel_size; ++i)
                sum += values[token + i] * weights[group * kernel_size + i];
            output[group * seq_len + token] = sum / (1.f + std::exp(-sum));
        }

        if (process_len > 0) {
            const int first = process_len;
            for (int i = 0; i < prefix_len; ++i)
                state[group * prefix_len + i] = values[first + i];
        }
    }
}

void run_case(int groups, int seq_len, int kernel_size, int n_real,
              ThreadPool* thread_pool, const char* label) {
    std::vector<float> input(groups * seq_len);
    std::vector<float> weights(groups * kernel_size);
    std::vector<float> state(groups * (kernel_size - 1));
    for (size_t i = 0; i < input.size(); ++i)
        input[i] =
            static_cast<float>(static_cast<int>((i * 7) % 19) - 9) * 0.03f;
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] =
            static_cast<float>(static_cast<int>((i * 5) % 13) - 6) * 0.02f;
    for (size_t i = 0; i < state.size(); ++i)
        state[i] =
            static_cast<float>(static_cast<int>((i * 3) % 11) - 5) * 0.01f;

    std::vector<float> expected_state = state;
    std::vector<float> expected_output(groups * seq_len, 0.f);
    reference_shortconv(input, weights, expected_state, expected_output, groups,
                        seq_len, kernel_size, n_real);

    std::vector<float> output(groups * seq_len, -1.f);
    Tensor input_tensor = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                         groups, seq_len, 1, 1, input.data());
    Tensor weight_tensor =
        Tensor::create(Precision::FP32, MemoryType::EXTERNAL, kernel_size,
                       groups, 1, 1, weights.data());
    Tensor state_tensor =
        Tensor::create(Precision::FP32, MemoryType::EXTERNAL, kernel_size - 1,
                       groups, 1, 1, state.data());
    Tensor output_tensor = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                          groups, seq_len, 1, 1, output.data());
    std::vector<const Tensor*> inputs = {&input_tensor, &weight_tensor,
                                         &state_tensor};
    OpParams params;
    params.i32 = {kernel_size, n_real};
    kernel_shortconv(params, inputs, output_tensor, thread_pool);

    bool matches = true;
    for (size_t i = 0; i < output.size(); ++i)
        matches &= std::fabs(output[i] - expected_output[i]) < 1e-5f;
    for (size_t i = 0; i < state.size(); ++i)
        matches &= std::fabs(state[i] - expected_state[i]) < 1e-6f;
    CHECK(matches, label);
}

} // namespace

int main() {
    ThreadPool pool(4);
    run_case(8, 1, 4, 1, &pool, "decode k=4");
    run_case(7, 5, 4, 3, &pool, "padded prefill k=4");
    run_case(5, 4, 3, 4, &pool, "generic kernel size");

    if (failures == 0)
        std::printf("All shortconv tests passed!\n");
    return failures;
}
