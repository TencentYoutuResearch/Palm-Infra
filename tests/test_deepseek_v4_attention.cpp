#include "kernels/cpu/models/deepseek_v4_attention.h"
#include "core/bf16.h"
#include "kernels/cpu/matmul/matmul.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL: %s\n", message);                    \
            ++failures;                                                     \
        } else {                                                            \
            std::printf("  PASS: %s\n", message);                           \
        }                                                                   \
    } while (0)

bool close(float actual, float expected, float tolerance = 1e-5f) {
    return std::fabs(actual - expected) <= tolerance;
}

Tensor tensor(float* data, int dim0, int dim1 = 1) {
    return Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        dim0, dim1, 1, 1, data);
}

struct CompressorBuffers {
    std::vector<float> kv_state;
    std::vector<float> score_state;
    std::vector<float> cache;
    Tensor kv_state_tensor;
    Tensor score_state_tensor;
    Tensor cache_tensor;

    CompressorBuffers(int projected, int state_rows,
                      int head_dim, int capacity)
        : kv_state(static_cast<size_t>(projected) * state_rows),
          score_state(
              static_cast<size_t>(projected) * state_rows,
              -std::numeric_limits<float>::infinity()),
          cache(static_cast<size_t>(head_dim) * capacity),
          kv_state_tensor(
              tensor(kv_state.data(), projected, state_rows)),
          score_state_tensor(
              tensor(score_state.data(), projected, state_rows)),
          cache_tensor(tensor(cache.data(), head_dim, capacity)) {}
};

}  // namespace

int main() {
    {
        float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        dsv4_hadamard_rotate(values, 4);
        CHECK(close(values[0], 5.0f) &&
                  close(values[1], -1.0f) &&
                  close(values[2], -2.0f) &&
                  close(values[3], 0.0f),
              "normalized Hadamard rotation");
    }

    {
        Dsv4RopeConfig rope;
        rope.rope_dim = 2;
        rope.theta = 10000.0f;
        rope.original_context = 0;
        float values[2] = {1.0f, 0.0f};
        dsv4_apply_rope(values, 2, 1, rope);
        CHECK(close(values[0], std::cos(1.0f)) &&
                  close(values[1], std::sin(1.0f)),
              "DeepSeek adjacent-pair RoPE");
        dsv4_apply_rope(values, 2, 1, rope, true);
        CHECK(close(values[0], 1.0f, 2e-5f) &&
                  close(values[1], 0.0f, 2e-5f),
              "DeepSeek inverse RoPE");
    }

    // Non-overlapping ratio-2 compression. Running [2 tokens] in one call
    // must match two arbitrarily split chunks.
    {
        Dsv4CompressorConfig config;
        config.hidden_size = 2;
        config.head_dim = 2;
        config.ratio = 2;
        config.overlap = false;
        config.rope.rope_dim = 0;

        float hidden_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float wkv_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        float wgate_data[4] = {};
        float ape_data[4] = {};
        float norm_data[2] = {1.0f, 1.0f};
        Tensor hidden = tensor(hidden_data, 2, 2);
        Tensor wkv = tensor(wkv_data, 2, 2);
        Tensor wgate = tensor(wgate_data, 2, 2);
        Tensor ape = tensor(ape_data, 2, 2);
        Tensor norm = tensor(norm_data, 2);

        CompressorBuffers whole(2, 2, 2, 4);
        const int whole_emitted = kernel_dsv4_compressor(
            hidden, wkv, wgate, ape, norm,
            whole.kv_state_tensor, whole.score_state_tensor,
            whole.cache_tensor, 0, config);

        CompressorBuffers split(2, 2, 2, 4);
        Tensor first = tensor(hidden_data, 2, 1);
        Tensor second = tensor(hidden_data + 2, 2, 1);
        const int first_emitted = kernel_dsv4_compressor(
            first, wkv, wgate, ape, norm,
            split.kv_state_tensor, split.score_state_tensor,
            split.cache_tensor, 0, config);
        const int second_emitted = kernel_dsv4_compressor(
            second, wkv, wgate, ape, norm,
            split.kv_state_tensor, split.score_state_tensor,
            split.cache_tensor, 1, config);

        const float inverse = 1.0f / std::sqrt(6.5f + 1e-6f);
        const float expected0 =
            mollm_round_to_bf16(2.0f * inverse);
        const float expected1 =
            mollm_round_to_bf16(3.0f * inverse);
        CHECK(whole_emitted == 1 && first_emitted == 0 &&
                  second_emitted == 1,
              "compressor emits on absolute ratio boundary");
        CHECK(close(whole.cache[0], expected0) &&
                  close(whole.cache[1], expected1),
              "learned uniform pooling followed by RMSNorm");
        CHECK(close(whole.cache[0], split.cache[0]) &&
                  close(whole.cache[1], split.cache[1]),
              "compressor state is invariant to chunk boundaries");
    }

    // Ratio-4 overlapping state: compare a monolithic prefill with uneven
    // chunks crossing both compression boundaries.
    {
        Dsv4CompressorConfig config;
        config.hidden_size = 2;
        config.head_dim = 2;
        config.ratio = 4;
        config.overlap = true;
        config.rope.rope_dim = 0;
        const int projected = 4;
        float hidden_data[16];
        for (int token = 0; token < 8; ++token) {
            hidden_data[token * 2] = static_cast<float>(token + 1);
            hidden_data[token * 2 + 1] =
                static_cast<float>(token + 2);
        }
        float wkv_data[8] = {
            1, 0, 0, 1,
            2, 0, 0, 2,
        };
        float wgate_data[8] = {};
        float ape_data[16] = {};
        float norm_data[2] = {1, 1};
        Tensor hidden = tensor(hidden_data, 2, 8);
        Tensor wkv = tensor(wkv_data, projected, 2);
        Tensor wgate = tensor(wgate_data, projected, 2);
        Tensor ape = tensor(ape_data, projected, 4);
        Tensor norm = tensor(norm_data, 2);

        CompressorBuffers whole(projected, 8, 2, 4);
        CHECK(kernel_dsv4_compressor(
                  hidden, wkv, wgate, ape, norm,
                  whole.kv_state_tensor, whole.score_state_tensor,
                  whole.cache_tensor, 0, config) == 2,
              "overlap compressor emits two groups");

        CompressorBuffers split(projected, 8, 2, 4);
        Tensor chunk0 = tensor(hidden_data, 2, 3);
        Tensor chunk1 = tensor(hidden_data + 6, 2, 2);
        Tensor chunk2 = tensor(hidden_data + 10, 2, 3);
        CHECK(kernel_dsv4_compressor(
                  chunk0, wkv, wgate, ape, norm,
                  split.kv_state_tensor, split.score_state_tensor,
                  split.cache_tensor, 0, config) == 0 &&
                  kernel_dsv4_compressor(
                      chunk1, wkv, wgate, ape, norm,
                      split.kv_state_tensor, split.score_state_tensor,
                      split.cache_tensor, 3, config) == 1 &&
                  kernel_dsv4_compressor(
                      chunk2, wkv, wgate, ape, norm,
                      split.kv_state_tensor, split.score_state_tensor,
                      split.cache_tensor, 5, config) == 1,
              "overlap compressor supports uneven continuation chunks");
        bool same = true;
        for (int i = 0; i < 4; ++i)
            same = same && close(whole.cache[i], split.cache[i]);
        CHECK(same, "overlap compressor output is chunk invariant");
    }

    // Indexer availability is causal in compressed-block units. A query may
    // select a compressed block only once the block is complete.
    {
        Dsv4IndexerConfig config;
        config.hidden_size = 2;
        config.q_lora_rank = 2;
        config.num_heads = 2;
        config.head_dim = 2;
        config.top_k = 2;
        config.compressor.hidden_size = 2;
        config.compressor.head_dim = 2;
        config.compressor.ratio = 2;
        config.compressor.overlap = false;
        config.compressor.rotate = true;
        config.compressor.rope.rope_dim = 0;

        float hidden_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float wq_b_data[8] = {
            1, 0, 0, 1,
            1, 0, 0, 1,
        };
        float weights_projection_data[4] = {};
        float compressor_wkv_data[4] = {1, 0, 0, 1};
        float compressor_wgate_data[4] = {};
        float compressor_ape_data[4] = {};
        float compressor_norm_data[2] = {1, 1};
        Tensor hidden = tensor(hidden_data, 2, 4);
        Tensor q_lora = tensor(hidden_data, 2, 4);
        Tensor wq_b = tensor(wq_b_data, 4, 2);
        Tensor weights_projection =
            tensor(weights_projection_data, 2, 2);
        Tensor compressor_wkv =
            tensor(compressor_wkv_data, 2, 2);
        Tensor compressor_wgate =
            tensor(compressor_wgate_data, 2, 2);
        Tensor compressor_ape =
            tensor(compressor_ape_data, 2, 2);
        Tensor compressor_norm =
            tensor(compressor_norm_data, 2);
        CompressorBuffers state(2, 2, 2, 4);
        int32_t index_data[8];
        Tensor indices = Tensor::create(
            Precision::INT32, MemoryType::EXTERNAL,
            2, 4, 1, 1, index_data);

        CHECK(kernel_dsv4_indexer(
                  hidden, q_lora, wq_b, weights_projection,
                  compressor_wkv, compressor_wgate,
                  compressor_ape, compressor_norm,
                  state.kv_state_tensor, state.score_state_tensor,
                  state.cache_tensor, indices, 0, config),
              "DeepSeek learned indexer executes");
        CHECK(index_data[0] == -1 && index_data[1] == -1 &&
                  index_data[2] == 0 && index_data[3] == -1 &&
                  index_data[4] == 0 && index_data[5] == -1 &&
                  index_data[6] == 0 && index_data[7] == 1,
              "indexer compressed-history mask is causal");
    }

    // Sparse attention must preserve the exact sliding window when a prefill
    // is split across calls; later tokens in the same chunk come from the
    // current KV tensor while earlier tokens come from the ring.
    {
        Dsv4SparseAttentionConfig config;
        config.num_heads = 1;
        config.head_dim = 2;
        config.window_size = 2;
        config.rope.rope_dim = 0;
        float query_data[6] = {1, 0, 0, 1, 1, 1};
        float kv_data[6] = {1, 0, 0, 1, 2, 1};
        float sink_data[1] = {-100.0f};
        float whole_window_data[4] = {};
        float whole_output_data[6] = {};
        Tensor query = tensor(query_data, 2, 3);
        Tensor kv = tensor(kv_data, 2, 3);
        Tensor sink = tensor(sink_data, 1);
        Tensor whole_window = tensor(whole_window_data, 2, 2);
        Tensor whole_output = tensor(whole_output_data, 2, 3);
        CHECK(kernel_dsv4_sparse_attention(
                  query, kv, sink, whole_window, nullptr, nullptr,
                  0, whole_output, config),
              "DeepSeek sparse sliding attention executes");

        float split_window_data[4] = {};
        float split_output_data[6] = {};
        Tensor split_window = tensor(split_window_data, 2, 2);
        Tensor query0 = tensor(query_data, 2, 1);
        Tensor kv0 = tensor(kv_data, 2, 1);
        Tensor output0 = tensor(split_output_data, 2, 1);
        Tensor query1 = tensor(query_data + 2, 2, 2);
        Tensor kv1 = tensor(kv_data + 2, 2, 2);
        Tensor output1 = tensor(split_output_data + 2, 2, 2);
        CHECK(kernel_dsv4_sparse_attention(
                  query0, kv0, sink, split_window, nullptr, nullptr,
                  0, output0, config) &&
                  kernel_dsv4_sparse_attention(
                      query1, kv1, sink, split_window, nullptr, nullptr,
                      1, output1, config),
              "sparse attention accepts continuation chunks");
        bool same = true;
        for (int i = 0; i < 6; ++i)
            same = same &&
                   close(whole_output_data[i], split_output_data[i], 2e-5f);
        CHECK(same, "sparse sliding attention is chunk invariant");
    }

    {
        float input_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float weight_data[8] = {
            1, 0, 0, 1,
            1, 0, 0, 1,
        };
        float output_data[8] = {};
        Tensor input = tensor(input_data, 4, 2);
        Tensor weight = tensor(weight_data, 4, 2);
        Tensor output = tensor(output_data, 4, 2);
        CHECK(kernel_dsv4_grouped_linear(
                  input, weight, output, 2),
              "DeepSeek grouped output projection executes");
        bool same = true;
        for (int i = 0; i < 8; ++i)
            same = same && close(output_data[i], input_data[i]);
        CHECK(same, "grouped output projection keeps group boundaries");
    }

    // The production wo_a path materializes dequantized FP8 values with BF16
    // rounding into an interleaved FP16 sidecar. It must remain numerically
    // equivalent to the exact scalar reference path.
    {
        constexpr int groups = 2;
        constexpr int group_width = 128;
        constexpr int rank = 8;
        constexpr int rows = groups * rank;
        std::vector<float> input_data(groups * group_width);
        for (size_t i = 0; i < input_data.size(); ++i)
            input_data[i] = static_cast<float>((int)(i % 11) - 5) / 16.0f;
        std::vector<uint8_t> weight_data(
            static_cast<size_t>(rows) * group_width);
        for (size_t i = 0; i < weight_data.size(); ++i) {
            const float value =
                static_cast<float>((int)(i % 7) - 3) / 8.0f;
            weight_data[i] = encode_fp8_e4m3fn(value);
        }
        uint8_t scales[1] = {127};
        Tensor input = tensor(input_data.data(), groups * group_width);
        Tensor weight = Tensor::create(
            Precision::FP8_E4M3, MemoryType::EXTERNAL,
            rows, group_width, 1, 1, weight_data.data());
        weight.e8m0_scales = scales;
        weight.is_fp8_block128 = true;
        std::vector<float> reference_data(rows);
        std::vector<float> optimized_data(rows);
        Tensor reference = tensor(reference_data.data(), rows);
        Tensor optimized = tensor(optimized_data.data(), rows);
        Tensor exact_weight = weight;
        CHECK(kernel_dsv4_grouped_linear(
                  input, exact_weight, reference, groups),
              "exact grouped FP8 reference executes");
        PackedWeightMap packed;
        const bool prepared = prepare_fp8_bf16_fp16_weight(
            weight, "grouped_fp8_fixture", weight_data.data(), packed);
        CHECK(prepared == mollm::cpu::capabilities().fp16_interleaved_weights,
              "grouped FP8 sidecar follows CPU provider capability");
        CHECK(kernel_dsv4_grouped_linear(
                  input, weight, optimized, groups),
              "optimized grouped FP8 projection executes");
        bool same = true;
        for (int i = 0; i < rows; ++i)
            same = same &&
                   close(optimized_data[i], reference_data[i], 2e-5f);
        CHECK(same,
              "optimized grouped FP8 projection matches exact reference");
    }

    if (failures) {
        std::fprintf(stderr, "%d DeepSeek-V4 attention test(s) failed\n",
                     failures);
        return 1;
    }
    std::printf("All DeepSeek-V4 attention tests passed!\n");
    return 0;
}
