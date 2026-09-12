#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/ple.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
bool close(float a, float b) { return std::fabs(a - b) < 1e-6f; }
} // namespace

int main() {
    constexpr int eos = 99;
    constexpr int seq = 5;
    constexpr int heads = 4;
    const int32_t tokens[seq] = {10, 20, 30, eos, 40};
    int32_t history[2] = {eos, eos};
    const int32_t sizes[heads] = {101, 103, 107, 109};
    const int32_t offsets[heads] = {0, 101, 204, 311};
    const int32_t expected[seq][heads] = {
        {72, 188, 241, 334},
        {82, 180, 233, 399},
        {79, 171, 269, 345},
        {4, 199, 303, 414},
        {39, 138, 301, 337},
    };
    std::vector<int32_t> indices(seq * heads);
    check(ple_build_indices(tokens, seq, history, 2, sizes, offsets,
                            3, 2, eos, 100, 0, 1234, indices.data()),
          "PLE index builder executes");
    bool indices_match = true;
    for (int token = 0; token < seq; ++token)
        for (int head = 0; head < heads; ++head)
            indices_match &= indices[token * heads + head] ==
                             expected[token][head];
    check(indices_match, "PLE indices match the official hash formula");
    check(history[0] == eos && history[1] == 40,
          "PLE history resets at EOS and continues into decode");

    constexpr int rows = 420;
    constexpr int row_dim = 2;
    std::vector<uint8_t> table(rows * row_dim);
    for (int row = 0; row < rows; ++row) {
        table[row * row_dim] = encode_fp8_e4m3fn((row % 7) - 3.0f);
        table[row * row_dim + 1] = encode_fp8_e4m3fn((row % 5) * 0.5f);
    }
    std::vector<float> embedding(seq * heads * row_dim);
    check(ple_gather_fp8_rows(table.data(), rows, row_dim, indices.data(),
                              seq, heads, 0.25f, embedding.data()),
          "PLE FP8 row gather executes");
    bool gather_matches = true;
    for (int token = 0; token < seq; ++token) {
        for (int head = 0; head < heads; ++head) {
            const int row = indices[token * heads + head];
            for (int dim = 0; dim < row_dim; ++dim) {
                const float expected_value = decode_fp8_e4m3fn(
                    table[row * row_dim + dim]) * 0.25f;
                gather_matches &= close(
                    embedding[(token * heads + head) * row_dim + dim],
                    expected_value);
            }
        }
    }
    check(gather_matches, "PLE FP8 gather preserves head concatenation order");

    int32_t wrapper_history[3] = {0, 0, 0};
    std::vector<float> wrapper_embedding(embedding.size());
    float table_scale = 0.25f;
    Tensor token_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, seq, 1, 1, 1,
        const_cast<int32_t*>(tokens));
    Tensor history_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, 3, 1, 1, 1,
        wrapper_history);
    Tensor table_tensor = Tensor::create(
        Precision::RAW_U8, MemoryType::EXTERNAL, row_dim, rows, 1, 1,
        table.data());
    Tensor scale_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1,
        &table_scale);
    Tensor sizes_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, heads, 1, 1, 1,
        const_cast<int32_t*>(sizes));
    Tensor offsets_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, heads, 1, 1, 1,
        const_cast<int32_t*>(offsets));
    Tensor output_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, heads * row_dim, seq, 1, 1,
        wrapper_embedding.data());
    check(kernel_ple_lookup(
              token_tensor, history_tensor, table_tensor, scale_tensor,
              sizes_tensor, offsets_tensor, output_tensor,
              3, 2, eos, 100, 0, 1234),
          "graph-facing PLE lookup executes");
    bool wrapper_matches = wrapper_history[0] == eos &&
                           wrapper_history[1] == 40 &&
                           wrapper_history[2] == 1;
    for (size_t index = 0; index < embedding.size(); ++index)
        wrapper_matches &= close(wrapper_embedding[index], embedding[index]);
    check(wrapper_matches, "graph-facing PLE lookup matches split helpers");

    // A static-padded prefill must not hash padding tokens or advance history.
    const int32_t real_tokens[2] = {10, 20};
    const int32_t padded_tokens[5] = {10, 20, 0, 0, 0};
    int32_t real_history[3] = {};
    int32_t padded_history[3] = {};
    std::vector<float> real_embedding(2 * heads * row_dim);
    std::vector<float> padded_embedding(5 * heads * row_dim, -1.0f);
    Tensor real_token_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, 2, 1, 1, 1,
        const_cast<int32_t*>(real_tokens));
    Tensor padded_token_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, 5, 1, 1, 1,
        const_cast<int32_t*>(padded_tokens));
    Tensor real_history_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, 3, 1, 1, 1, real_history);
    Tensor padded_history_tensor = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, 3, 1, 1, 1, padded_history);
    Tensor real_output_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, heads * row_dim, 2, 1, 1,
        real_embedding.data());
    Tensor padded_output_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, heads * row_dim, 5, 1, 1,
        padded_embedding.data());
    const bool real_lookup = kernel_ple_lookup(
        real_token_tensor, real_history_tensor, table_tensor, scale_tensor,
        sizes_tensor, offsets_tensor, real_output_tensor,
        3, 2, eos, 100, 0, 1234, 2);
    const bool padded_lookup = kernel_ple_lookup(
        padded_token_tensor, padded_history_tensor, table_tensor, scale_tensor,
        sizes_tensor, offsets_tensor, padded_output_tensor,
        3, 2, eos, 100, 0, 1234, 2);
    bool padded_lookup_matches = real_lookup && padded_lookup;
    for (int index = 0; index < 3; ++index)
        padded_lookup_matches &= real_history[index] == padded_history[index];
    for (size_t index = 0; index < real_embedding.size(); ++index)
        padded_lookup_matches &= close(
            real_embedding[index], padded_embedding[index]);
    for (size_t index = real_embedding.size();
         index < padded_embedding.size(); ++index)
        padded_lookup_matches &= close(padded_embedding[index], 0.0f);
    check(padded_lookup_matches,
          "PLE lookup ignores static padding and preserves history");

    // Gate scores are reduced independently for every residual stream.
    constexpr int gate_hidden = 2;
    constexpr int gate_streams = 2;
    float query_data[4] = {1.0f, 0.0f, -1.0f, 0.0f};
    float key_data[4] = {2.0f, 0.0f, 2.0f, 0.0f};
    float value_data[2] = {3.0f, -4.0f};
    float gated_data[4] = {};
    Tensor query_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1, query_data);
    Tensor key_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1, key_data);
    Tensor value_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 2, 1, 1, 1, value_data);
    Tensor gated_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1, gated_data);
    check(kernel_ple_gate(query_tensor, key_tensor, value_tensor,
                          gated_tensor, gate_hidden, gate_streams),
          "PLE stream gate executes");
    const float positive_score = std::sqrt(2.0f / std::sqrt(2.0f));
    const float positive_gate = 1.0f / (1.0f + std::exp(-positive_score));
    const float negative_gate = 1.0f - positive_gate;
    check(close(gated_data[0], positive_gate * 3.0f) &&
          close(gated_data[1], positive_gate * -4.0f) &&
          close(gated_data[2], negative_gate * 3.0f) &&
          close(gated_data[3], negative_gate * -4.0f),
          "PLE stream gate matches signed-root sigmoid reference");

    // Dilation=2, kernel=3 reads delays 4, 2 and 0 and retains four states.
    float conv_input[3] = {1.0f, 2.0f, 3.0f};
    float conv_weight[3] = {10.0f, 1.0f, 0.1f};
    float conv_state[4] = {4.0f, 5.0f, 6.0f, 7.0f};
    float conv_output[3] = {};
    Tensor conv_input_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 3, 1, 1, conv_input);
    Tensor conv_weight_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 3, 1, 1, conv_weight);
    Tensor conv_state_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 4, 1, 1, conv_state);
    Tensor conv_output_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 3, 1, 1, conv_output);
    check(kernel_ple_dilated_conv(
              conv_input_tensor, conv_weight_tensor, conv_state_tensor,
              conv_output_tensor, 3, 2),
          "PLE dilated convolution executes");
    const float sums[3] = {46.1f, 57.2f, 61.3f};
    bool conv_matches = true;
    for (int i = 0; i < 3; ++i)
        conv_matches &= close(
            conv_output[i], sums[i] / (1.0f + std::exp(-sums[i])));
    check(conv_matches && close(conv_state[0], 7.0f) &&
          close(conv_state[1], 1.0f) && close(conv_state[2], 2.0f) &&
          close(conv_state[3], 3.0f),
          "PLE dilated convolution updates its causal state");

    float short_input[1] = {1.0f};
    float padded_conv_input[3] = {1.0f, 0.0f, 0.0f};
    float short_state[4] = {4.0f, 5.0f, 6.0f, 7.0f};
    float padded_state[4] = {4.0f, 5.0f, 6.0f, 7.0f};
    float short_output[1] = {};
    float padded_conv_output[3] = {-1.0f, -1.0f, -1.0f};
    Tensor short_input_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1, short_input);
    Tensor padded_conv_input_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 3, 1, 1,
        padded_conv_input);
    Tensor short_state_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 4, 1, 1, short_state);
    Tensor padded_state_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 4, 1, 1, padded_state);
    Tensor short_output_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1, short_output);
    Tensor padded_conv_output_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 1, 3, 1, 1,
        padded_conv_output);
    const bool short_conv = kernel_ple_dilated_conv(
        short_input_tensor, conv_weight_tensor, short_state_tensor,
        short_output_tensor, 3, 2, 1);
    const bool padded_conv = kernel_ple_dilated_conv(
        padded_conv_input_tensor, conv_weight_tensor, padded_state_tensor,
        padded_conv_output_tensor, 3, 2, 1);
    bool padded_conv_matches = short_conv && padded_conv &&
        close(short_output[0], padded_conv_output[0]) &&
        close(padded_conv_output[1], 0.0f) &&
        close(padded_conv_output[2], 0.0f);
    for (int index = 0; index < 4; ++index)
        padded_conv_matches &= close(short_state[index], padded_state[index]);
    check(padded_conv_matches,
          "PLE convolution ignores static padding and preserves state");

    if (failures) {
        std::fprintf(stderr, "%d PLE test(s) failed\n", failures);
        return 1;
    }
    std::printf("All PLE tests passed!\n");
    return 0;
}
