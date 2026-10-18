#include "kernels/cpu/models/deepseek_v4_attention.h"
#include "core/bf16.h"
#include "backends/cpu/backend.h"
#include "graph/graph.h"
#include "kernels/cpu/matmul/matmul.h"

#include <algorithm>
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


    // ---------------------------------------------------------------------
    // Hy4 indexer (Gated DSA)
    //
    // Unlike the DeepSeek-V4 indexer, Hy4 has no learned compressor: keys are
    // `wk @ hidden` directly, so there is no compressor state and no ratio.
    //
    // Fixture (num_heads=1, head_dim=2, rope_dim=0 so scores stay exact):
    //   hidden columns  t0 = (1,0), t1 = (0,1)
    //   wk, wq_b        identity  ->  projected keys = hidden, queries = q_lora
    //   q_lora          = hidden
    //   weights_proj    [1,1]     ->  weight = h0 + h1 = 1 for both tokens
    //   k_norm          weight 1, bias 0
    //   weight_scale    = 1/sqrt(head_dim * num_heads) = 1/sqrt(2) = 0.7071
    //
    //   LayerNorm maps keys to (1,-1), (-1,1), after BF16 rounding.
    //   token 0 sees only key 0: q.k = 1 -> score 0.7071, selects key 0
    //   token 1 sees keys 0,1:   key0 scores 0 (ReLU), key1 scores
    //                            0.7071, so key 1 must rank STRICTLY first
    //                            and key 0 second.
    // ---------------------------------------------------------------------
    {
        Hy4IndexerConfig cfg;
        cfg.hidden_size = 2;
        cfg.q_lora_rank = 2;
        cfg.num_heads = 1;
        cfg.head_dim = 2;
        cfg.rope_dim = 0;
        cfg.top_k = 2;
        cfg.capacity = 4;
        cfg.rope.rope_dim = 0;

        float hidden_data[4] = {1, 0, 0, 1};
        float q_lora_data[4] = {1, 0, 0, 1};
        float wk_data[4] = {1, 0, 0, 1};
        float wq_b_data[4] = {1, 0, 0, 1};
        float weights_proj_data[2] = {1, 1};
        float k_norm_w[2] = {1, 1};
        float k_norm_b[2] = {0, 0};
        std::vector<float> key_cache(2 * 4, -7.0f);
        int32_t index_data[4] = {9, 9, 9, 9};

        Tensor hidden = tensor(hidden_data, 2, 2);
        Tensor q_lora = tensor(q_lora_data, 2, 2);
        Tensor wk = tensor(wk_data, 2, 2);
        Tensor wq_b = tensor(wq_b_data, 2, 2);
        Tensor weights_proj = tensor(weights_proj_data, 1, 2);
        Tensor k_norm_weight = tensor(k_norm_w, 2);
        Tensor k_norm_bias = tensor(k_norm_b, 2);
        Tensor key_cache_t = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            2, 4, 1, 1, key_cache.data());
        Tensor indices = Tensor::create(
            Precision::INT32, MemoryType::EXTERNAL,
            2, 2, 1, 1, index_data);

        CHECK(kernel_hy4_indexer(
                  hidden, q_lora, wq_b, wk, weights_proj,
                  k_norm_weight, k_norm_bias, key_cache_t, indices,
                  0, cfg),
              "Hy4 indexer executes");

        // Keys land in the cache at their absolute positions.
        CHECK(close(key_cache[0], 1.0f) && close(key_cache[1], -1.0f),
              "Hy4 indexer caches token 0 key at position 0");
        CHECK(close(key_cache[2], -1.0f) && close(key_cache[3], 1.0f),
              "Hy4 indexer caches token 1 key at position 1");

        // token 0 has one available key, so slot 1 stays at the -1 sentinel.
        CHECK(index_data[0] == 0 && index_data[1] == -1,
              "Hy4 indexer selects key 0 for token 0 and leaves the rest -1");

        // token 1 must rank key 1 above key 0 (strict, not a tie).
        CHECK(index_data[2] == 1 && index_data[3] == 0,
              "Hy4 indexer ranks key 1 strictly above key 0 for token 1");
    }

    // top_k smaller than the visible window must truncate, not overflow.
    {
        Hy4IndexerConfig cfg;
        cfg.hidden_size = 2;
        cfg.q_lora_rank = 2;
        cfg.num_heads = 1;
        cfg.head_dim = 2;
        cfg.rope_dim = 0;
        cfg.top_k = 1;
        cfg.capacity = 4;
        cfg.rope.rope_dim = 0;

        float hidden_data[4] = {1, 0, 0, 1};
        float q_lora_data[4] = {1, 0, 0, 1};
        float wk_data[4] = {1, 0, 0, 1};
        float wq_b_data[4] = {1, 0, 0, 1};
        float weights_proj_data[2] = {1, 1};
        float k_norm_w[2] = {1, 1};
        float k_norm_b[2] = {0, 0};
        std::vector<float> key_cache(2 * 4, 0.0f);
        int32_t index_data[2] = {9, 9};

        Tensor hidden = tensor(hidden_data, 2, 2);
        Tensor q_lora = tensor(q_lora_data, 2, 2);
        Tensor wk = tensor(wk_data, 2, 2);
        Tensor wq_b = tensor(wq_b_data, 2, 2);
        Tensor weights_proj = tensor(weights_proj_data, 1, 2);
        Tensor k_norm_weight = tensor(k_norm_w, 2);
        Tensor k_norm_bias = tensor(k_norm_b, 2);
        Tensor key_cache_t = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            2, 4, 1, 1, key_cache.data());
        Tensor indices = Tensor::create(
            Precision::INT32, MemoryType::EXTERNAL,
            1, 2, 1, 1, index_data);

        CHECK(kernel_hy4_indexer(
                  hidden, q_lora, wq_b, wk, weights_proj,
                  k_norm_weight, k_norm_bias, key_cache_t, indices,
                  0, cfg),
              "Hy4 indexer honours a smaller top_k");
        CHECK(index_data[0] == 0 && index_data[1] == 1,
              "Hy4 indexer writes one index per position when top_k is 1");
    }

    // Offsets/unequal variances distinguish LayerNorm from affine/RMS norm.
    // Raw keys rank key 1 first (dot 4 vs 2); normalized keys rank key 0 first.
    for (float eps : {1e-5f, 0.25f}) {
        float raw[12] = {101, 99, 100, 100, 4, 0, 0, 0, 7, 7, 7, 7};
        float queries[12] = {1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0};
        float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                              0, 0, 1, 0, 0, 0, 0, 1};
        float projection[4] = {0, 0, 0, 1};
        float gamma[4] = {1, 1, 2, 0.5f}, beta[4] = {0.25f, -0.125f, 0.5f, -0.25f};
        float cache[16] = {}, chunk_cache[16] = {};
        int32_t selected[6] = {}, chunk_selected[6] = {};
        Tensor hidden = tensor(raw, 4, 3), q = tensor(queries, 4, 3);
        Tensor w = tensor(identity, 4, 4), wp = tensor(projection, 1, 4);
        Tensor nw = tensor(gamma, 4), nb = tensor(beta, 4);
        Tensor kt = tensor(cache, 4, 4), chunk_kt = tensor(chunk_cache, 4, 4);
        Tensor indices = Tensor::create(Precision::INT32, MemoryType::EXTERNAL,
                                        2, 3, 1, 1, selected);
        Hy4IndexerConfig cfg;
        cfg.hidden_size = cfg.q_lora_rank = cfg.head_dim = 4;
        cfg.num_heads = 1;
        cfg.top_k = 2;
        cfg.capacity = 4;
        cfg.rope_dim = cfg.rope.rope_dim = 0;
        cfg.norm_eps = eps;
        CHECK(kernel_hy4_indexer(hidden, q, w, w, wp, nw, nb, kt, indices, 0, cfg),
              "Hy4 LayerNorm fixture executes");
        for (int row = 0; row < 3; ++row) {
            double mean = 0.0, variance = 0.0;
            for (int d = 0; d < 4; ++d) mean += raw[4 * row + d] / 4.0;
            for (int d = 0; d < 4; ++d)
                variance += std::pow(raw[4 * row + d] - mean, 2) / 4.0;
            for (int d = 0; d < 4; ++d) {
                const float expected = mollm_round_to_bf16(static_cast<float>(
                    (raw[4 * row + d] - mean) / std::sqrt(variance + eps) *
                    gamma[d] + beta[d]));
                CHECK(close(cache[4 * row + d], expected),
                      "Hy4 cached key matches independent LayerNorm formula");
            }
        }
        CHECK(selected[4] == 0 && selected[5] == 1,
              "Hy4 normalized top-k keeps key 0 first and excludes constant key");
        for (int row = 0; row < 3; ++row) {
            Tensor h = tensor(raw + 4 * row, 4), qr = tensor(queries + 4 * row, 4);
            Tensor out = Tensor::create(Precision::INT32, MemoryType::EXTERNAL,
                                        2, 1, 1, 1, chunk_selected + 2 * row);
            CHECK(kernel_hy4_indexer(h, qr, w, w, wp, nw, nb, chunk_kt, out, row, cfg),
                  "Hy4 indexer executes incremental positions");
        }
        bool same = true;
        for (int j = 0; j < 16; ++j) same &= cache[j] == chunk_cache[j];
        for (int j = 0; j < 6; ++j) same &= selected[j] == chunk_selected[j];
        CHECK(same, "Hy4 normalized cache and selections agree for batch and decode");

        // Exercise the graph parameter and the legacy four-float default.
        GraphNode node;
        node.op_type = OpType::HY4_INDEXER;
        node.params.i32 = {4, 4, 1, 4, 0, 2, 4, 0};
        node.params.f32 = {1e7f, 1.0f, 32.0f, 1.0f};
        if (eps != 1e-5f) node.params.f32.push_back(eps);
        int32_t position = 0, count = 3;
        Tensor pos = Tensor::create(Precision::INT32, MemoryType::EXTERNAL,
                                   1, 1, 1, 1, &position);
        Tensor n = Tensor::create(Precision::INT32, MemoryType::EXTERNAL,
                                 1, 1, 1, 1, &count);
        Tensor out = Tensor::create(Precision::INT32, MemoryType::EXTERNAL,
                                   2, 3, 1, 1, chunk_selected);
        CPUBackend backend;
        backend.dispatch(node, {&hidden, &q, &w, &w, &wp, &nw, &nb, &chunk_kt,
                               &pos, &n}, &out, nullptr);
        same = !backend.dispatch_failed();
        for (int j = 0; j < 12; ++j) same &= cache[j] == chunk_cache[j];
        for (int j = 0; j < 6; ++j) same &= selected[j] == chunk_selected[j];
        CHECK(same, "Hy4 dispatch honors explicit epsilon and legacy default");
    }

    // BF16 collapses these two distinct best scores into a tie. FP32 must
    // retain the higher score of key 1, rather than choosing key 0 by index.
    {
        float raw[12] = {1, .01f, 0, 0, 1, 0, 0, 0, 1, .1f, 0, 0};
        float query[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
        float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        float projection[4] = {1, 0, 0, 0}, gamma[4] = {1, 1, 1, 1}, beta[4] = {};
        float cache[12] = {};
        int32_t chosen[3] = {}, pos_data = 0, count_data = 3;
        Tensor h = tensor(raw, 4, 3), q = tensor(query, 4, 3), w = tensor(identity, 4, 4);
        Tensor wp = tensor(projection, 1, 4), nw = tensor(gamma, 4), nb = tensor(beta, 4);
        Tensor k = tensor(cache, 4, 3);
        Tensor out = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 1, 3, 1, 1, chosen);
        Tensor pos = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 1, 1, 1, 1, &pos_data);
        Tensor count = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 1, 1, 1, 1, &count_data);
        for (bool bf16 : {false, true}) {
            GraphNode node;
            node.op_type = OpType::HY4_INDEXER;
            node.params.i32 = {4, 4, 1, 4, 0, 1, 3, 0};
            if (!bf16) node.params.i32.push_back(0);
            node.params.f32 = {1e7f, 1, 32, 1, 1e-5f};
            CPUBackend backend;
            backend.dispatch(node,{&h, &q, &w, &w, &wp, &nw, &nb, &k, &pos, &count}, &out, nullptr);
            CHECK(!backend.dispatch_failed(), "Hy4 indexer precision dispatch executes");
            CHECK(chosen[2] == (bf16 ? 0 : 1),
                  "Hy4 FP32 preserves score ordering lost to legacy BF16 rounding");
            CHECK(bf16 ? cache[0] == cache[4] : cache[0] < cache[4],
                  "Hy4 indexer cache retains the selected numerical precision");
            Hy4IndexerConfig cfg;
            cfg.hidden_size = cfg.q_lora_rank = cfg.head_dim = 4;
            cfg.num_heads = cfg.top_k = 1;
            cfg.capacity = 3;
            cfg.rope_dim = cfg.rope.rope_dim = 0;
            cfg.round_bf16 = bf16;
            int32_t one_result = 0;
            Tensor one = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 1, 1, 1, 1, &one_result);
            for (int row = 0; row < 3; ++row) {
                Tensor hr = tensor(raw + 4 * row, 4), qr = tensor(query + 4 * row, 4);
                CHECK(kernel_hy4_indexer(hr, qr, w, w, wp, nw, nb, k, one, row, cfg) &&
                          one_result == chosen[row],
                      "Hy4 indexer precision agrees for batch and incremental decode");
            }
        }
    }

    // Exercise the production top2048 transition with a preloaded cache,
    // without running a 2050-token model. One chunk sees 2047, 2048, 2049 keys.
    {
        constexpr int start = 2046, sequence = 3, capacity = 2052, top_k = 2048;
        Hy4IndexerConfig cfg;
        cfg.hidden_size = cfg.q_lora_rank = cfg.head_dim = 2;
        cfg.num_heads = 1;
        cfg.rope_dim = cfg.rope.rope_dim = 0;
        cfg.top_k = top_k;
        cfg.capacity = capacity;
        cfg.round_bf16 = false;
        float identity[] = {1, 0, 0, 1};
        float projection[] = {1, 1}, norm[] = {1, 1}, bias[] = {0, 0};
        float hidden_data[] = {1, 0, 0, 1, 1, 0};
        auto hidden = tensor(hidden_data, 2, sequence);
        auto weight = tensor(identity, 2, 2);
        auto wp = tensor(projection, 1, 2);
        auto nw = tensor(norm, 2), nb = tensor(bias, 2);
        // Future positions contain very high scores, making causal leakage
        // observable. Repeated integer keys also exercise stable tie handling.
        std::vector<float> initial(capacity * 2, 10000.0f);
        for (int i = 0; i < start; ++i) {
            initial[2 * i] = float((i * 37) % 101 - 50);
            initial[2 * i + 1] = -initial[2 * i];
        }
        auto cache = initial;
        auto incremental = initial;
        constexpr int32_t guard = 123456789;
        std::vector<int32_t> actual(top_k * sequence + 2, guard);
        auto kt = tensor(cache.data(), 2, capacity);
        auto output = Tensor::create(Precision::INT32, MemoryType::EXTERNAL,
                                     top_k, sequence, 1, 1, actual.data() + 1);
        CHECK(kernel_hy4_indexer(hidden, hidden, weight, weight, wp, nw, nb,
                                kt, output, start, cfg),
              "Hy4 indexer executes across top2048 boundary");
        const float normalized = 0.5f / std::sqrt(0.25f + cfg.norm_eps);
        auto expected_cache = initial;
        for (int token = 0; token < sequence; ++token) {
            const float sign = token == 1 ? -1.0f : 1.0f;
            expected_cache[2 * (start + token)] = sign * normalized;
            expected_cache[2 * (start + token) + 1] = -sign * normalized;
        }
        bool cache_ok = true, ranks_ok = true, incremental_ok = true;
        for (size_t i = 0; i < cache.size(); ++i)
            cache_ok &= close(cache[i], expected_cache[i]);
        for (int token = 0; token < sequence; ++token) {
            const int available = start + token + 1;
            const int coordinate = token == 1 ? 1 : 0;
            // One-hot queries make the mathematical score proportional to
            // ReLU(key[coordinate]); sorting all candidates is the oracle.
            std::vector<int> expected(available);
            for (int i = 0; i < available; ++i) expected[i] = i;
            std::stable_sort(expected.begin(), expected.end(), [&](int a, int b) {
                return std::max(expected_cache[2 * a + coordinate], 0.0f) >
                       std::max(expected_cache[2 * b + coordinate], 0.0f);
            });
            for (int rank = 0; rank < top_k; ++rank) {
                const int want = rank < available ? expected[rank] : -1;
                ranks_ok &= actual[1 + token * top_k + rank] == want;
            }
            std::vector<int32_t> one(top_k, guard);
            auto row = tensor(hidden_data + 2 * token, 2);
            auto ik = tensor(incremental.data(), 2, capacity);
            auto io = Tensor::create(Precision::INT32, MemoryType::EXTERNAL,
                                     top_k, 1, 1, 1, one.data());
            incremental_ok &= kernel_hy4_indexer(row, row, weight, weight, wp,
                nw, nb, ik, io, start + token, cfg);
            incremental_ok &= std::equal(one.begin(), one.end(),
                                         actual.begin() + 1 + token * top_k);
        }
        CHECK(ranks_ok, "top2048 matches full-sort oracle, causal mask, ties and padding");
        CHECK(cache_ok && actual.front() == guard && actual.back() == guard,
              "top2048 updates only current cache slots and respects output bounds");
        CHECK(incremental_ok && cache == incremental,
              "top2048 chunk and incremental decode choose identical keys");
    }

    // A wrong-shaped wk must be rejected rather than read out of bounds.
    {
        Hy4IndexerConfig cfg;
        cfg.hidden_size = 2;
        cfg.q_lora_rank = 2;
        cfg.num_heads = 1;
        cfg.head_dim = 2;
        cfg.rope_dim = 0;
        cfg.top_k = 1;
        cfg.capacity = 4;
        cfg.rope.rope_dim = 0;

        float hidden_data[4] = {1, 0, 0, 1};
        float q_lora_data[4] = {1, 0, 0, 1};
        float wrong_wk[4] = {1, 0, 0, 1};   // declared as [3, 2], not [2, 2]
        float wq_b_data[4] = {1, 0, 0, 1};
        float weights_proj_data[2] = {1, 1};
        float k_norm_w[2] = {1, 1};
        float k_norm_b[2] = {0, 0};
        std::vector<float> key_cache(2 * 4, 0.0f);
        int32_t index_data[1] = {9};

        Tensor hidden = tensor(hidden_data, 2, 2);
        Tensor q_lora = tensor(q_lora_data, 2, 2);
        Tensor wk_bad = tensor(wrong_wk, 3, 2);
        Tensor wq_b = tensor(wq_b_data, 2, 2);
        Tensor weights_proj = tensor(weights_proj_data, 1, 2);
        Tensor k_norm_weight = tensor(k_norm_w, 2);
        Tensor k_norm_bias = tensor(k_norm_b, 2);
        Tensor key_cache_t = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            2, 4, 1, 1, key_cache.data());
        Tensor indices = Tensor::create(
            Precision::INT32, MemoryType::EXTERNAL,
            1, 1, 1, 1, index_data);

        CHECK(!kernel_hy4_indexer(
                  hidden, q_lora, wq_b, wk_bad, weights_proj,
                  k_norm_weight, k_norm_bias, key_cache_t, indices,
                  0, cfg),
              "Hy4 indexer rejects a mis-shaped wk");
    }

    if (failures) {
        std::fprintf(stderr, "%d DeepSeek-V4 attention test(s) failed\n",
                     failures);
        return 1;
    }
    std::printf("All DeepSeek-V4 attention tests passed!\n");
    return 0;
}
