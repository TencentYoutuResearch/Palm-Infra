#include "kernels/cpu/models/deepseek_v4_attention.h"

#include "core/bf16.h"
#include "kernels/cpu_platform.h"
#include "kernels/cpu/matmul/matmul.h"
#include "runtime/threading.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

#if HAS_NEON

bool grouped_fp16_interleaved_gemv(
    const float* input, const __fp16* packed_weight, float* output,
    int groups, int group_width, int rank, ThreadPool* thread_pool) {
    if (!input || !packed_weight || !output || groups <= 0 ||
        group_width <= 0 || rank <= 0 || (rank % 8) != 0) {
        return false;
    }
    // Process two adjacent 8-row interleaved tiles together when possible.
    // Both projections use the same group activation, so this halves its loads
    // and loop/control work without changing any output's accumulation order.
    const int rows_per_task = (rank % 16) == 0 ? 16 : 8;
    const int tiles_per_group = rank / rows_per_task;
    const int total_tiles = groups * tiles_per_group;
    auto run = [&](int begin, int end) {
        for (int tile = begin; tile < end; ++tile) {
            const int group = tile / tiles_per_group;
            const int local_tile = tile % tiles_per_group;
            const int output_row =
                group * rank + local_tile * rows_per_task;
            const float* group_input =
                input + static_cast<size_t>(group) * group_width;
            const __fp16* weight_tile0 =
                packed_weight +
                static_cast<size_t>(output_row) * group_width;
            const __fp16* weight_tile1 =
                rows_per_task == 16
                    ? weight_tile0 +
                          static_cast<size_t>(8) * group_width
                    : nullptr;
            float32x4_t low0 = vdupq_n_f32(0.0f);
            float32x4_t high0 = vdupq_n_f32(0.0f);
            float32x4_t low1 = vdupq_n_f32(0.0f);
            float32x4_t high1 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 1 < group_width; k += 2) {
                const float activation0 = group_input[k];
                const float activation1 = group_input[k + 1];
                const float16x8_t values00 =
                    vld1q_f16(weight_tile0 + static_cast<size_t>(k) * 8);
                const float16x8_t values01 =
                    vld1q_f16(
                        weight_tile0 + static_cast<size_t>(k + 1) * 8);
                low0 = vfmaq_n_f32(
                    low0, vcvt_f32_f16(vget_low_f16(values00)), activation0);
                high0 = vfmaq_n_f32(
                    high0, vcvt_f32_f16(vget_high_f16(values00)), activation0);
                low0 = vfmaq_n_f32(
                    low0, vcvt_f32_f16(vget_low_f16(values01)), activation1);
                high0 = vfmaq_n_f32(
                    high0, vcvt_f32_f16(vget_high_f16(values01)), activation1);
                if (weight_tile1) {
                    const float16x8_t values10 =
                        vld1q_f16(
                            weight_tile1 + static_cast<size_t>(k) * 8);
                    const float16x8_t values11 =
                        vld1q_f16(
                            weight_tile1 + static_cast<size_t>(k + 1) * 8);
                    low1 = vfmaq_n_f32(
                        low1, vcvt_f32_f16(vget_low_f16(values10)),
                        activation0);
                    high1 = vfmaq_n_f32(
                        high1, vcvt_f32_f16(vget_high_f16(values10)),
                        activation0);
                    low1 = vfmaq_n_f32(
                        low1, vcvt_f32_f16(vget_low_f16(values11)),
                        activation1);
                    high1 = vfmaq_n_f32(
                        high1, vcvt_f32_f16(vget_high_f16(values11)),
                        activation1);
                }
            }
            for (; k < group_width; ++k) {
                const float16x8_t values0 =
                    vld1q_f16(weight_tile0 + static_cast<size_t>(k) * 8);
                const float activation = group_input[k];
                low0 = vfmaq_n_f32(
                    low0, vcvt_f32_f16(vget_low_f16(values0)), activation);
                high0 = vfmaq_n_f32(
                    high0, vcvt_f32_f16(vget_high_f16(values0)), activation);
                if (weight_tile1) {
                    const float16x8_t values1 =
                        vld1q_f16(
                            weight_tile1 + static_cast<size_t>(k) * 8);
                    low1 = vfmaq_n_f32(
                        low1, vcvt_f32_f16(vget_low_f16(values1)),
                        activation);
                    high1 = vfmaq_n_f32(
                        high1, vcvt_f32_f16(vget_high_f16(values1)),
                        activation);
                }
            }
            vst1q_f32(output + output_row, low0);
            vst1q_f32(output + output_row + 4, high0);
            if (weight_tile1) {
                vst1q_f32(output + output_row + 8, low1);
                vst1q_f32(output + output_row + 12, high1);
            }
        }
    };
    if (thread_pool && thread_pool->num_threads() > 1) {
        const int grain = std::max(
            1, (total_tiles + thread_pool->num_threads() - 1) /
                   thread_pool->num_threads());
        thread_pool->parallel_for(
            0, total_tiles, grain,
            [&](int, int begin, int end) { run(begin, end); });
    } else {
        run(0, total_tiles);
    }
    return true;
}

#endif

bool tensor_is_fp32_matrix(const Tensor& tensor, int rows, int columns) {
    return tensor.prec == Precision::FP32 && tensor.data &&
           tensor.shape[0] == rows && tensor.shape[1] == columns;
}

float yarn_frequency(int pair, const Dsv4RopeConfig& config) {
    const int half = config.rope_dim / 2;
    float frequency = 1.0f / std::pow(
        config.theta,
        2.0f * static_cast<float>(pair) /
            static_cast<float>(config.rope_dim));
    if (config.original_context <= 0 || config.factor <= 1.0f)
        return frequency;

    auto correction_dim = [&](float rotations) {
        return static_cast<float>(config.rope_dim) *
               std::log(
                   static_cast<float>(config.original_context) /
                   (rotations * 2.0f * kPi)) /
               (2.0f * std::log(config.theta));
    };
    const int low = std::max(
        0, static_cast<int>(std::floor(
               correction_dim(config.beta_fast))));
    const int high = std::min(
        config.rope_dim - 1,
        static_cast<int>(std::ceil(
            correction_dim(config.beta_slow))));
    const float denominator =
        high == low ? 0.001f : static_cast<float>(high - low);
    const float ramp = std::clamp(
        (static_cast<float>(pair) - low) / denominator, 0.0f, 1.0f);
    const float smooth = 1.0f - ramp;
    return frequency / config.factor * (1.0f - smooth) +
           frequency * smooth;
}

float nearest_e2m1(float value) {
    static constexpr float magnitudes[] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float sign = std::signbit(value) ? -1.0f : 1.0f;
    const float magnitude = std::fabs(value);
    float nearest = magnitudes[0];
    float distance = std::fabs(magnitude - nearest);
    int nearest_code = 0;
    for (int code = 1; code < 8; ++code) {
        const float candidate = magnitudes[code];
        const float candidate_distance =
            std::fabs(magnitude - candidate);
        if (candidate_distance < distance ||
            (candidate_distance == distance &&
             (code & 1) == 0 && (nearest_code & 1) != 0)) {
            distance = candidate_distance;
            nearest = candidate;
            nearest_code = code;
        }
    }
    return sign * nearest;
}

float nearest_e4m3(float value) {
    return decode_fp8_e4m3fn(encode_fp8_e4m3fn(value));
}

void reset_compressor_state(Tensor& kv_state, Tensor& score_state,
                            Tensor& cache) {
    std::fill(
        kv_state.ptr<float>(),
        kv_state.ptr<float>() + kv_state.nelements(), 0.0f);
    std::fill(
        score_state.ptr<float>(),
        score_state.ptr<float>() + score_state.nelements(),
        -std::numeric_limits<float>::infinity());
    std::fill(
        cache.ptr<float>(),
        cache.ptr<float>() + cache.nelements(), 0.0f);
}

float dot_product(const float* left, const float* right, int count) {
#if HAS_NEON
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < count; i += 8) {
        sum0 = vfmaq_f32(
            sum0, vld1q_f32(left + i), vld1q_f32(right + i));
        sum1 = vfmaq_f32(
            sum1, vld1q_f32(left + i + 4),
            vld1q_f32(right + i + 4));
    }
    float result = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < count; ++i)
        result += left[i] * right[i];
    return result;
#else
    float result = 0.0f;
    for (int i = 0; i < count; ++i)
        result += left[i] * right[i];
    return result;
#endif
}

}  // namespace

void dsv4_apply_rope(float* values, int width, int position,
                     const Dsv4RopeConfig& config, bool inverse) {
    if (!values || width < config.rope_dim || config.rope_dim <= 0 ||
        (config.rope_dim & 1)) {
        return;
    }
    float* rotary = values + width - config.rope_dim;
    const float direction = inverse ? -1.0f : 1.0f;
    for (int pair = 0; pair < config.rope_dim / 2; ++pair) {
        const float angle =
            direction * static_cast<float>(position) *
            yarn_frequency(pair, config);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float real = rotary[pair * 2];
        const float imaginary = rotary[pair * 2 + 1];
        rotary[pair * 2] = real * cosine - imaginary * sine;
        rotary[pair * 2 + 1] =
            real * sine + imaginary * cosine;
    }
}

void dsv4_hadamard_rotate(float* values, int width) {
    if (!values || width <= 0 || (width & (width - 1)) != 0)
        return;
    for (int span = 1; span < width; span *= 2) {
        for (int base = 0; base < width; base += span * 2) {
            for (int offset = 0; offset < span; ++offset) {
                const float a = values[base + offset];
                const float b = values[base + span + offset];
                values[base + offset] = a + b;
                values[base + span + offset] = a - b;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(width));
    for (int i = 0; i < width; ++i)
        values[i] *= scale;
}

void dsv4_fp4_simulate_inplace(float* values, int width, int group_size) {
    if (!values || width <= 0 || group_size <= 0 ||
        width % group_size != 0) {
        return;
    }
    for (int begin = 0; begin < width; begin += group_size) {
        float maximum = 0.0f;
        for (int i = 0; i < group_size; ++i)
            maximum = std::max(maximum, std::fabs(values[begin + i]));
        maximum = std::max(
            maximum, 6.0f * std::ldexp(1.0f, -126));
        const float raw_scale = maximum / 6.0f;
        const float scale = std::exp2(std::ceil(std::log2(raw_scale)));
        for (int i = 0; i < group_size; ++i) {
            const float normalized = std::clamp(
                values[begin + i] / scale, -6.0f, 6.0f);
            values[begin + i] =
                nearest_e2m1(normalized) * scale;
        }
    }
}

void dsv4_fp8_simulate_inplace(float* values, int width, int group_size) {
    if (!values || width <= 0 || group_size <= 0 ||
        width % group_size != 0) {
        return;
    }
    for (int begin = 0; begin < width; begin += group_size) {
        float maximum = 0.0f;
        for (int i = 0; i < group_size; ++i)
            maximum = std::max(maximum, std::fabs(values[begin + i]));
        maximum = std::max(maximum, 1e-4f);
        const float raw_scale = maximum / 448.0f;
        const float scale = std::exp2(std::ceil(std::log2(raw_scale)));
        for (int i = 0; i < group_size; ++i) {
            const float normalized = std::clamp(
                values[begin + i] / scale, -448.0f, 448.0f);
            values[begin + i] =
                nearest_e4m3(normalized) * scale;
        }
    }
}

int kernel_dsv4_compressor(
    const Tensor& hidden,
    const Tensor& wkv,
    const Tensor& wgate,
    const Tensor& ape,
    const Tensor& norm_weight,
    Tensor& kv_state,
    Tensor& score_state,
    Tensor& cache,
    int start_pos,
    const Dsv4CompressorConfig& config,
    ThreadPool* thread_pool) {
    const int sequence = static_cast<int>(hidden.shape[1]);
    const int coff = config.overlap ? 2 : 1;
    const int projected = coff * config.head_dim;
    const int state_rows = coff * config.ratio;
    if (start_pos < 0 || sequence <= 0 || config.hidden_size <= 0 ||
        config.head_dim <= 0 || config.ratio <= 0 ||
        hidden.prec != Precision::FP32 || !hidden.data ||
        hidden.shape[0] != config.hidden_size ||
        wkv.shape[0] != projected ||
        wkv.shape[1] != config.hidden_size ||
        wgate.shape[0] != projected ||
        wgate.shape[1] != config.hidden_size ||
        !tensor_is_fp32_matrix(ape, projected, config.ratio) ||
        norm_weight.prec != Precision::FP32 || !norm_weight.data ||
        norm_weight.nelements() !=
            static_cast<size_t>(config.head_dim) ||
        !tensor_is_fp32_matrix(kv_state, projected, state_rows) ||
        !tensor_is_fp32_matrix(score_state, projected, state_rows) ||
        cache.prec != Precision::FP32 || !cache.data ||
        cache.shape[0] != config.head_dim) {
        return -1;
    }

    if (start_pos == 0)
        reset_compressor_state(kv_state, score_state, cache);

    const size_t projection_elements =
        static_cast<size_t>(projected) * sequence;
    std::vector<float> projection_values(projection_elements * 2);
    float* kv_values = projection_values.data();
    float* gate_values = kv_values + projection_elements;
    Tensor kv_output = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        projected, sequence, 1, 1, kv_values);
    Tensor gate_output = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        projected, sequence, 1, 1, gate_values);
    if (sequence == 1 && thread_pool) {
        Tensor projection_output = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            projected, sequence, 2, 1, projection_values.data());
        const std::vector<const Tensor*> projection_pairs = {
            &hidden, &wkv, &hidden, &wgate};
        kernel_matmul_batch(
            projection_pairs, projection_output, thread_pool);
    } else {
        kernel_matmul_fp32(hidden, wkv, kv_output, thread_pool);
        kernel_matmul_fp32(hidden, wgate, gate_output, thread_pool);
    }
    // Compressor wkv/wgate are explicitly FP32 in the reference and stay
    // FP32 through the learned pooling.  Only the pooled KV is cast back to
    // the model activation dtype below.

    const float* ape_data = ape.ptr<float>();
    const float* norm = norm_weight.ptr<float>();
    float* kv_state_data = kv_state.ptr<float>();
    float* score_state_data = score_state.ptr<float>();
    float* cache_data = cache.ptr<float>();
    const int cache_capacity = static_cast<int>(cache.shape[1]);
    std::vector<float> compressed(
        static_cast<size_t>(config.head_dim));
    int emitted = 0;

    for (int token = 0; token < sequence; ++token) {
        const int position = start_pos + token;
        const int within = position % config.ratio;
        const int destination_row =
            config.overlap ? config.ratio + within : within;
        const float* token_kv =
            kv_values + static_cast<size_t>(token) * projected;
        const float* token_gate =
            gate_values + static_cast<size_t>(token) * projected;
        float* state_kv =
            kv_state_data +
            static_cast<size_t>(destination_row) * projected;
        float* state_score =
            score_state_data +
            static_cast<size_t>(destination_row) * projected;
        const float* position_ape =
            ape_data + static_cast<size_t>(within) * projected;
        std::copy(token_kv, token_kv + projected, state_kv);
        for (int dim = 0; dim < projected; ++dim)
            state_score[dim] = token_gate[dim] + position_ape[dim];

        if ((position + 1) % config.ratio != 0)
            continue;
        const int cache_index = position / config.ratio;
        if (cache_index < 0 || cache_index >= cache_capacity)
            return -1;

        for (int dim = 0; dim < config.head_dim; ++dim) {
            float maximum =
                -std::numeric_limits<float>::infinity();
            const int row_count =
                config.overlap ? config.ratio * 2 : config.ratio;
            for (int row = 0; row < row_count; ++row) {
                const int source_dim =
                    config.overlap && row >= config.ratio
                        ? config.head_dim + dim
                        : dim;
                maximum = std::max(
                    maximum,
                    score_state_data[
                        static_cast<size_t>(row) * projected +
                        source_dim]);
            }
            float denominator = 0.0f;
            float numerator = 0.0f;
            for (int row = 0; row < row_count; ++row) {
                const int source_dim =
                    config.overlap && row >= config.ratio
                        ? config.head_dim + dim
                        : dim;
                const size_t index =
                    static_cast<size_t>(row) * projected + source_dim;
                const float probability =
                    std::exp(score_state_data[index] - maximum);
                denominator += probability;
                numerator += probability * kv_state_data[index];
            }
            compressed[dim] =
                denominator > 0.0f ? numerator / denominator : 0.0f;
        }

        // weighted_pool(...).to(hidden.dtype)
        mollm_round_to_bf16(compressed.data(), compressed.size());
        float mean_square = 0.0f;
        for (float value : compressed)
            mean_square += value * value;
        mean_square /= static_cast<float>(config.head_dim);
        const float inverse =
            1.0f / std::sqrt(mean_square + config.norm_eps);
        for (int dim = 0; dim < config.head_dim; ++dim)
            compressed[dim] *= inverse * norm[dim];
        mollm_round_to_bf16(compressed.data(), compressed.size());

        const int compressed_position = position + 1 - config.ratio;
        dsv4_apply_rope(
            compressed.data(), config.head_dim, compressed_position,
            config.rope);
        mollm_round_to_bf16(compressed.data(), compressed.size());
        if (config.rotate) {
            dsv4_hadamard_rotate(
                compressed.data(), config.head_dim);
            mollm_round_to_bf16(
                compressed.data(), compressed.size());
            dsv4_fp4_simulate_inplace(
                compressed.data(), config.head_dim);
        } else {
            const int non_rotary =
                config.head_dim - config.rope.rope_dim;
            if (non_rotary > 0)
                dsv4_fp8_simulate_inplace(
                    compressed.data(), non_rotary, 64);
        }
        std::copy(
            compressed.begin(), compressed.end(),
            cache_data +
                static_cast<size_t>(cache_index) * config.head_dim);
        ++emitted;

        if (config.overlap) {
            const size_t half_bytes =
                static_cast<size_t>(config.ratio) * projected *
                sizeof(float);
            std::memmove(
                kv_state_data,
                kv_state_data +
                    static_cast<size_t>(config.ratio) * projected,
                half_bytes);
            std::memmove(
                score_state_data,
                score_state_data +
                    static_cast<size_t>(config.ratio) * projected,
                half_bytes);
            std::fill(
                score_state_data +
                    static_cast<size_t>(config.ratio) * projected,
                score_state_data +
                    static_cast<size_t>(state_rows) * projected,
                -std::numeric_limits<float>::infinity());
        }
    }
    return emitted;
}

bool kernel_dsv4_indexer(
    const Tensor& hidden,
    const Tensor& q_lora,
    const Tensor& wq_b,
    const Tensor& weights_projection,
    const Tensor& compressor_wkv,
    const Tensor& compressor_wgate,
    const Tensor& compressor_ape,
    const Tensor& compressor_norm,
    Tensor& compressor_kv_state,
    Tensor& compressor_score_state,
    Tensor& index_cache,
    Tensor& indices,
    int start_pos,
    const Dsv4IndexerConfig& config,
    ThreadPool* thread_pool) {
    const int sequence = static_cast<int>(hidden.shape[1]);
    const int query_width = config.num_heads * config.head_dim;
    if (sequence <= 0 || start_pos < 0 ||
        hidden.prec != Precision::FP32 ||
        hidden.shape[0] != config.hidden_size ||
        q_lora.prec != Precision::FP32 ||
        q_lora.shape[0] != config.q_lora_rank ||
        q_lora.shape[1] != sequence ||
        wq_b.shape[0] != query_width ||
        wq_b.shape[1] != config.q_lora_rank ||
        weights_projection.shape[0] != config.num_heads ||
        weights_projection.shape[1] != config.hidden_size ||
        index_cache.prec != Precision::FP32 ||
        index_cache.shape[0] != config.head_dim ||
        indices.prec != Precision::INT32 ||
        indices.shape[0] != config.top_k ||
        indices.shape[1] != sequence ||
        config.compressor.head_dim != config.head_dim ||
        config.compressor.hidden_size != config.hidden_size) {
        return false;
    }

    std::vector<float> query_values(
        static_cast<size_t>(query_width) * sequence);
    std::vector<float> head_weights(
        static_cast<size_t>(config.num_heads) * sequence);
    Tensor query = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        query_width, sequence, 1, 1, query_values.data());
    Tensor weights = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        config.num_heads, sequence, 1, 1, head_weights.data());
    kernel_matmul_fp32(q_lora, wq_b, query, thread_pool);
    kernel_matmul_fp32(
        hidden, weights_projection, weights, thread_pool);
    mollm_round_to_bf16(query_values.data(), query_values.size());
    mollm_round_to_bf16(head_weights.data(), head_weights.size());

    const float weight_scale =
        1.0f / std::sqrt(
            static_cast<float>(
                config.head_dim * config.num_heads));
    for (float& value : head_weights) {
        value *= weight_scale;
        value = mollm_round_to_bf16(value);
    }
    for (int token = 0; token < sequence; ++token) {
        const int position = start_pos + token;
        float* token_query =
            query_values.data() +
            static_cast<size_t>(token) * query_width;
        for (int head = 0; head < config.num_heads; ++head) {
            float* head_query =
                token_query + static_cast<size_t>(head) * config.head_dim;
            dsv4_apply_rope(
                head_query, config.head_dim, position,
                config.compressor.rope);
            mollm_round_to_bf16(
                head_query, static_cast<size_t>(config.head_dim));
            dsv4_hadamard_rotate(head_query, config.head_dim);
            mollm_round_to_bf16(
                head_query, static_cast<size_t>(config.head_dim));
            dsv4_fp4_simulate_inplace(head_query, config.head_dim);
        }
    }

    if (kernel_dsv4_compressor(
            hidden, compressor_wkv, compressor_wgate, compressor_ape,
            compressor_norm, compressor_kv_state,
            compressor_score_state, index_cache, start_pos,
            config.compressor, thread_pool) < 0) {
        return false;
    }

    int32_t* output = indices.ptr<int32_t>();
    std::fill(
        output, output + indices.nelements(), static_cast<int32_t>(-1));
    const float* cache_data = index_cache.ptr<float>();
    std::vector<float> scores;
    std::vector<int> order;
    for (int token = 0; token < sequence; ++token) {
        const int position = start_pos + token;
        const int available =
            std::min(
                static_cast<int>(index_cache.shape[1]),
                (position + 1) / config.compressor.ratio);
        if (available <= 0)
            continue;
        scores.assign(static_cast<size_t>(available), 0.0f);
        const float* token_query =
            query_values.data() +
            static_cast<size_t>(token) * query_width;
        const float* token_weights =
            head_weights.data() +
            static_cast<size_t>(token) * config.num_heads;
        for (int cache_index = 0; cache_index < available;
             ++cache_index) {
            const float* key =
                cache_data +
                static_cast<size_t>(cache_index) * config.head_dim;
            float score = 0.0f;
            for (int head = 0; head < config.num_heads; ++head) {
                // einsum writes BF16, and the following elementwise
                // relu/multiply also stays in BF16 in the reference.
                const float dot = mollm_round_to_bf16(dot_product(
                    token_query +
                        static_cast<size_t>(head) * config.head_dim,
                    key, config.head_dim));
                const float weighted = mollm_round_to_bf16(
                    std::max(dot, 0.0f) * token_weights[head]);
                score += weighted;
            }
            scores[cache_index] = mollm_round_to_bf16(score);
        }

        const int selected = std::min(config.top_k, available);
        order.resize(static_cast<size_t>(available));
        std::iota(order.begin(), order.end(), 0);
        if (selected < available) {
            std::nth_element(
                order.begin(), order.begin() + selected, order.end(),
                [&](int left, int right) {
                    return scores[left] > scores[right];
                });
            order.resize(static_cast<size_t>(selected));
        }
        std::sort(
            order.begin(), order.end(),
            [&](int left, int right) {
                if (scores[left] == scores[right])
                    return left < right;
                return scores[left] > scores[right];
            });
        int32_t* token_output =
            output + static_cast<size_t>(token) * config.top_k;
        for (int rank = 0; rank < selected; ++rank) {
            token_output[rank] = static_cast<int32_t>(order[rank]);
        }
    }
    return true;
}

bool kernel_dsv4_sparse_attention(
    const Tensor& query,
    const Tensor& current_kv,
    const Tensor& attention_sink,
    Tensor& window_cache,
    const Tensor* compressed_cache,
    const Tensor* compressed_indices,
    int start_pos,
    Tensor& output,
    const Dsv4SparseAttentionConfig& config,
    ThreadPool* thread_pool) {
    const int sequence = static_cast<int>(query.shape[1]);
    const int query_width = config.num_heads * config.head_dim;
    if (sequence <= 0 || start_pos < 0 ||
        query.prec != Precision::FP32 || !query.data ||
        query.shape[0] != query_width ||
        current_kv.prec != Precision::FP32 || !current_kv.data ||
        current_kv.shape[0] != config.head_dim ||
        current_kv.shape[1] != sequence ||
        attention_sink.prec != Precision::FP32 ||
        !attention_sink.data ||
        attention_sink.nelements() <
            static_cast<size_t>(config.num_heads) ||
        window_cache.prec != Precision::FP32 || !window_cache.data ||
        window_cache.shape[0] != config.head_dim ||
        window_cache.shape[1] != config.window_size ||
        output.prec != Precision::FP32 || !output.data ||
        output.shape[0] != query_width ||
        output.shape[1] != sequence ||
        config.window_size <= 0 || config.head_dim <= 0 ||
        config.num_heads <= 0) {
        return false;
    }
    if (config.compress_ratio > 0) {
        if (!compressed_cache ||
            compressed_cache->prec != Precision::FP32 ||
            !compressed_cache->data ||
            compressed_cache->shape[0] != config.head_dim) {
            return false;
        }
        if (compressed_indices &&
            (compressed_indices->prec != Precision::INT32 ||
             !compressed_indices->data ||
             compressed_indices->shape[1] != sequence)) {
            return false;
        }
    }

    if (start_pos == 0) {
        std::fill(
            window_cache.ptr<float>(),
            window_cache.ptr<float>() + window_cache.nelements(), 0.0f);
    }

    std::vector<float> rotated_query(
        static_cast<size_t>(query_width) * sequence);
    std::vector<float> rotated_kv(
        static_cast<size_t>(config.head_dim) * sequence);
    std::copy(
        query.ptr<float>(),
        query.ptr<float>() + query.nelements(),
        rotated_query.begin());
    std::copy(
        current_kv.ptr<float>(),
        current_kv.ptr<float>() + current_kv.nelements(),
        rotated_kv.begin());

    for (int token = 0; token < sequence; ++token) {
        const int position = start_pos + token;
        float* token_kv =
            rotated_kv.data() +
            static_cast<size_t>(token) * config.head_dim;
        dsv4_apply_rope(
            token_kv, config.head_dim, position, config.rope);
        mollm_round_to_bf16(
            token_kv, static_cast<size_t>(config.head_dim));
        const int non_rotary =
            config.head_dim - config.rope.rope_dim;
        if (non_rotary > 0)
            dsv4_fp8_simulate_inplace(token_kv, non_rotary, 64);
        float* token_query =
            rotated_query.data() +
            static_cast<size_t>(token) * query_width;
        for (int head = 0; head < config.num_heads; ++head) {
            float* head_query =
                token_query +
                static_cast<size_t>(head) * config.head_dim;
            if (config.normalize_query) {
                float square_sum = 0.0f;
                for (int dim = 0; dim < config.head_dim; ++dim) {
                    square_sum += mollm_round_to_bf16(
                        head_query[dim] * head_query[dim]);
                }
                const float mean = mollm_round_to_bf16(
                    square_sum / static_cast<float>(config.head_dim));
                const float variance = mollm_round_to_bf16(
                    mean + config.query_norm_eps);
                const float inverse = mollm_round_to_bf16(
                    1.0f / std::sqrt(variance));
                for (int dim = 0; dim < config.head_dim; ++dim) {
                    head_query[dim] = mollm_round_to_bf16(
                        head_query[dim] * inverse);
                }
            }
            dsv4_apply_rope(
                head_query,
                config.head_dim, position, config.rope);
            mollm_round_to_bf16(
                head_query, static_cast<size_t>(config.head_dim));
        }
    }

    const float scale =
        config.softmax_scale > 0.0f
            ? config.softmax_scale
            : 1.0f / std::sqrt(static_cast<float>(config.head_dim));
    const float* sink = attention_sink.ptr<float>();
    const float* old_window = window_cache.ptr<float>();
    const float* compressed =
        compressed_cache ? compressed_cache->ptr<float>() : nullptr;
    const int compressed_capacity =
        compressed_cache
            ? static_cast<int>(compressed_cache->shape[1])
            : 0;
    const int32_t* selected =
        compressed_indices
            ? compressed_indices->ptr<int32_t>()
            : nullptr;
    const int selected_width =
        compressed_indices
            ? static_cast<int>(compressed_indices->shape[0])
            : 0;
    float* output_data = output.ptr<float>();
    const int tasks = sequence * config.num_heads;

    auto process = [&](int, int begin, int end) {
        std::vector<const float*> keys;
        std::vector<float> scores;
        for (int task = begin; task < end; ++task) {
            const int token = task / config.num_heads;
            const int head = task % config.num_heads;
            const int position = start_pos + token;
            const int first_position =
                std::max(0, position - config.window_size + 1);
            keys.clear();
            keys.reserve(
                static_cast<size_t>(config.window_size) +
                std::max(selected_width, 0));
            for (int key_position = first_position;
                 key_position <= position; ++key_position) {
                if (key_position >= start_pos) {
                    keys.push_back(
                        rotated_kv.data() +
                        static_cast<size_t>(
                            key_position - start_pos) *
                            config.head_dim);
                } else {
                    keys.push_back(
                        old_window +
                        static_cast<size_t>(
                            key_position % config.window_size) *
                            config.head_dim);
                }
            }

            if (config.compress_ratio > 0) {
                if (selected) {
                    const int32_t* token_indices =
                        selected +
                        static_cast<size_t>(token) * selected_width;
                    for (int rank = 0; rank < selected_width; ++rank) {
                        if (token_indices[rank] < 0)
                            continue;
                        const int cache_index = token_indices[rank];
                        if (cache_index >= 0 &&
                            cache_index < compressed_capacity) {
                            keys.push_back(
                                compressed +
                                static_cast<size_t>(cache_index) *
                                    config.head_dim);
                        }
                    }
                } else {
                    const int available = std::min(
                        compressed_capacity,
                        (position + 1) / config.compress_ratio);
                    for (int cache_index = 0;
                         cache_index < available; ++cache_index) {
                        keys.push_back(
                            compressed +
                            static_cast<size_t>(cache_index) *
                                config.head_dim);
                    }
                }
            }

            scores.resize(keys.size());
            const float* head_query =
                rotated_query.data() +
                static_cast<size_t>(token) * query_width +
                static_cast<size_t>(head) * config.head_dim;
            // Match the published sparse-attention kernel: the online
            // softmax maximum is reduced over KV scores only.  The learned
            // sink contributes to the denominator afterwards.  Including the
            // sink in this maximum is algebraically equivalent in FP32, but
            // not after probabilities are cast to BF16 for the value GEMM.
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t key = 0; key < keys.size(); ++key) {
                scores[key] =
                    dot_product(
                        head_query, keys[key], config.head_dim) *
                    scale;
                maximum = std::max(maximum, scores[key]);
            }
            float denominator = std::exp(sink[head] - maximum);
            float* head_output =
                output_data +
                static_cast<size_t>(token) * query_width +
                static_cast<size_t>(head) * config.head_dim;
            std::fill(
                head_output, head_output + config.head_dim, 0.0f);
            for (size_t key = 0; key < keys.size(); ++key) {
                const float probability =
                    std::exp(scores[key] - maximum);
                denominator += probability;
                // The reference sparse-attention kernel casts the softmax
                // probabilities to BF16 before the probability/value GEMM,
                // while keeping the denominator in FP32.
                const float value_probability =
                    mollm_round_to_bf16(probability);
                const float* value = keys[key];
#if HAS_NEON
                const float32x4_t probability4 =
                    vdupq_n_f32(value_probability);
                int dim = 0;
                for (; dim + 7 < config.head_dim; dim += 8) {
                    vst1q_f32(
                        head_output + dim,
                        vfmaq_f32(
                            vld1q_f32(head_output + dim),
                            vld1q_f32(value + dim), probability4));
                    vst1q_f32(
                        head_output + dim + 4,
                        vfmaq_f32(
                            vld1q_f32(head_output + dim + 4),
                            vld1q_f32(value + dim + 4),
                            probability4));
                }
                for (; dim < config.head_dim; ++dim)
                    head_output[dim] += value_probability * value[dim];
#else
                for (int dim = 0; dim < config.head_dim; ++dim)
                    head_output[dim] += value_probability * value[dim];
#endif
            }
            if (denominator > 0.0f) {
                const float inverse = 1.0f / denominator;
                for (int dim = 0; dim < config.head_dim; ++dim)
                    head_output[dim] *= inverse;
            }
            mollm_round_to_bf16(
                head_output, static_cast<size_t>(config.head_dim));
            dsv4_apply_rope(
                head_output, config.head_dim, position,
                config.rope, true);
            mollm_round_to_bf16(
                head_output, static_cast<size_t>(config.head_dim));
        }
    };
    if (thread_pool && thread_pool->num_threads() > 1 && tasks > 1) {
        thread_pool->parallel_for(
            0, tasks,
            std::max(1, tasks / (thread_pool->num_threads() * 4)),
            process);
    } else {
        process(0, 0, tasks);
    }

    float* window = window_cache.ptr<float>();
    for (int token = 0; token < sequence; ++token) {
        const int position = start_pos + token;
        std::copy(
            rotated_kv.data() +
                static_cast<size_t>(token) * config.head_dim,
            rotated_kv.data() +
                static_cast<size_t>(token + 1) * config.head_dim,
            window +
                static_cast<size_t>(
                    position % config.window_size) *
                    config.head_dim);
    }
    return true;
}

bool kernel_dsv4_grouped_linear(
    const Tensor& input, const Tensor& weight, Tensor& output,
    int groups, ThreadPool* thread_pool) {
    if (input.prec != Precision::FP32 || !input.data ||
        output.prec != Precision::FP32 || !output.data ||
        groups <= 0 || input.shape[0] % groups != 0 ||
        weight.shape[0] % groups != 0 ||
        weight.shape[1] != input.shape[0] / groups ||
        output.shape[0] != weight.shape[0] ||
        output.shape[1] != input.shape[1]) {
        return false;
    }
    const int group_width =
        static_cast<int>(input.shape[0] / groups);
    const int rank =
        static_cast<int>(weight.shape[0] / groups);
    const int sequence = static_cast<int>(input.shape[1]);
    const size_t weight_element_bytes =
        weight.prec == Precision::FP16
            ? sizeof(__fp16)
            : weight.prec == Precision::FP32
                  ? sizeof(float)
                  : 1;
    if (weight.prec != Precision::FP32 &&
        weight.prec != Precision::FP16 &&
        weight.prec != Precision::INT8 &&
        weight.prec != Precision::FP8_E4M3) {
        return false;
    }

#if HAS_NEON
    if (sequence == 1 && weight.prec == Precision::FP8_E4M3 &&
        weight.fp8_bf16_fp16_data &&
        grouped_fp16_interleaved_gemv(
            input.ptr<float>(),
            static_cast<const __fp16*>(weight.fp8_bf16_fp16_data),
            output.ptr<float>(), groups, group_width, rank, thread_pool)) {
        return true;
    }
#endif

    for (int group = 0; group < groups; ++group) {
        Tensor group_input = input;
        group_input.data = const_cast<float*>(
            input.ptr<float>() +
            static_cast<size_t>(group) * group_width);
        group_input.shape[0] = group_width;
        group_input.shape[1] = sequence;
        group_input.stride[0] = sizeof(float);
        group_input.stride[1] = input.stride[1];

        Tensor group_output = output;
        group_output.data =
            output.ptr<float>() +
            static_cast<size_t>(group) * rank;
        group_output.shape[0] = rank;
        group_output.shape[1] = sequence;
        group_output.stride[0] = sizeof(float);
        group_output.stride[1] = output.stride[1];

        Tensor group_weight = weight;
        const int row_begin = group * rank;
        const size_t row_elements =
            static_cast<size_t>(row_begin) * group_width;
        group_weight.data = const_cast<uint8_t*>(
            static_cast<const uint8_t*>(weight.data) +
            row_elements * weight_element_bytes);
        group_weight.shape[0] = rank;
        group_weight.shape[1] = group_width;
        group_weight.compute_strides();
        if (weight.scales && weight.groups_per_row > 0) {
            group_weight.scales =
                weight.scales +
                static_cast<size_t>(row_begin) *
                    weight.groups_per_row;
        }
        if (weight.e8m0_scales &&
            weight.prec == Precision::FP8_E4M3) {
            const int k_blocks =
                (group_width + 127) / 128;
            group_weight.e8m0_scales =
                weight.e8m0_scales +
                static_cast<size_t>(row_begin / 128) * k_blocks;
        }
        if (weight.prec == Precision::FP8_E4M3) {
            if (weight.fp8_bf16_fp16_data && (row_begin % 8) == 0) {
                group_weight.prec = Precision::FP16;
                group_weight.data = const_cast<uint8_t*>(
                    static_cast<const uint8_t*>(
                        weight.fp8_bf16_fp16_data) +
                    static_cast<size_t>(row_begin) * group_width *
                        sizeof(__fp16));
                group_weight.rowmajor_data = nullptr;
                group_weight.is_interleaved = true;
                group_weight.compute_strides();
                kernel_matmul_fp32(
                    group_input, group_weight, group_output, thread_pool,
                    Activation::NONE, 0, -1, true);
            } else if (!kernel_matmul_fp8_weight_f32_activation(
                           group_input, group_weight, group_output,
                           thread_pool)) {
                return false;
            }
        } else {
            kernel_matmul_fp32(
                group_input, group_weight, group_output, thread_pool);
        }
    }
    return true;
}
