#include "kernels/cpu/ple.h"

#include "kernels/cpu/matmul/matmul.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

constexpr uint64_t kSplitmixGamma = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kSplitmixM1 = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kSplitmixM2 = 0x94D049BB133111EBULL;
constexpr uint64_t kLayerPrime = 10007ULL;

const std::array<float, 256> kFp8E4m3fnDecodeTable = [] {
    std::array<float, 256> values{};
    for (int code = 0; code < 256; ++code)
        values[code] = decode_fp8_e4m3fn(static_cast<uint8_t>(code));
    return values;
}();

uint64_t splitmix64(uint64_t value) {
    value += kSplitmixGamma;
    value = ((value ^ (value >> 30)) * kSplitmixM1);
    value = ((value ^ (value >> 27)) * kSplitmixM2);
    return value ^ (value >> 31);
}

std::vector<uint64_t> layer_multipliers(int unigram_vocab_size,
                                        int ngram_size,
                                        int ple_layer_index,
                                        uint64_t seed) {
    const uint64_t max_long =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const uint64_t divisor =
        static_cast<uint64_t>(std::max(unigram_vocab_size, 1));
    const uint64_t multiplier_max = max_long / divisor;
    const uint64_t half_bound = std::max<uint64_t>(1, multiplier_max / 2);
    const uint64_t base_seed = seed + kLayerPrime *
        static_cast<uint64_t>(ple_layer_index);
    std::vector<uint64_t> result(static_cast<size_t>(ngram_size));
    for (int index = 0; index < ngram_size; ++index) {
        const uint64_t value = base_seed + kSplitmixGamma *
            static_cast<uint64_t>(index + 1);
        result[index] = 2 * (splitmix64(value) % half_bound) + 1;
    }
    return result;
}

int64_t positive_remainder(int64_t value, int32_t divisor) {
    int64_t remainder = value % static_cast<int64_t>(divisor);
    if (remainder < 0) remainder += divisor;
    return remainder;
}

void prefetch_ple_rows(const uint8_t* table, int row_dim,
                       const int32_t* indices, int rows) {
#if defined(MADV_WILLNEED)
    static const bool enabled = [] {
        const char* value = std::getenv("MOLLM_PLE_MADVISE");
        return !value || std::strcmp(value, "0") != 0;
    }();
    if (!enabled) return;
    static const uintptr_t page_size = [] {
        const long value = sysconf(_SC_PAGESIZE);
        return static_cast<uintptr_t>(value > 0 ? value : 4096);
    }();
    const uintptr_t page_mask = page_size - 1;
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    ranges.reserve(static_cast<size_t>(rows));
    for (int index = 0; index < rows; ++index) {
        const uintptr_t source = reinterpret_cast<uintptr_t>(table) +
            static_cast<size_t>(indices[index]) * row_dim;
        const uintptr_t begin = source & ~page_mask;
        const uintptr_t end =
            (source + static_cast<uintptr_t>(row_dim) + page_mask) &
            ~page_mask;
        ranges.emplace_back(begin, end);
    }
    std::sort(ranges.begin(), ranges.end());
    size_t merged = 0;
    for (const auto& range : ranges) {
        if (merged != 0 && range.first <= ranges[merged - 1].second) {
            ranges[merged - 1].second =
                std::max(ranges[merged - 1].second, range.second);
        } else {
            ranges[merged++] = range;
        }
    }
    for (size_t index = 0; index < merged; ++index) {
        madvise(reinterpret_cast<void*>(ranges[index].first),
                ranges[index].second - ranges[index].first,
                MADV_WILLNEED);
    }
#else
    (void)table;
    (void)row_dim;
    (void)indices;
    (void)rows;
#endif
}

} // namespace

bool ple_build_indices(const int32_t* token_ids, int seq_len,
                       int32_t* history, int context_len,
                       const int32_t* head_vocab_sizes,
                       const int32_t* head_offsets,
                       int ngram_size, int heads_per_ngram,
                       int eos_token_id, int unigram_vocab_size,
                       int ple_layer_index, uint64_t seed,
                       int32_t* output_indices) {
    const int num_heads = (ngram_size - 1) * heads_per_ngram;
    if (!token_ids || !history || !head_vocab_sizes || !head_offsets ||
        !output_indices || seq_len < 0 || ngram_size < 2 ||
        heads_per_ngram <= 0 || context_len != ngram_size - 1 ||
        unigram_vocab_size <= 0 || ple_layer_index < 0) {
        return false;
    }
    for (int head = 0; head < num_heads; ++head) {
        if (head_vocab_sizes[head] <= 0 || head_offsets[head] < 0)
            return false;
    }
    const std::vector<uint64_t> multipliers = layer_multipliers(
        unigram_vocab_size, ngram_size, ple_layer_index, seed);
    std::vector<int32_t> local_history(
        history, history + static_cast<size_t>(context_len));
    std::vector<int32_t> values(static_cast<size_t>(ngram_size));

    for (int token_index = 0; token_index < seq_len; ++token_index) {
        const int32_t token = token_ids[token_index];
        values[0] = token;
        for (int shift = 1; shift < ngram_size; ++shift)
            values[shift] = local_history[context_len - shift];

        for (int ngram = 2; ngram <= ngram_size; ++ngram) {
            uint64_t mixed = 0;
            for (int position = 0; position < ngram; ++position) {
                mixed ^= static_cast<uint64_t>(
                    static_cast<int64_t>(values[position])) *
                    multipliers[position];
            }
            const int64_t signed_mixed = static_cast<int64_t>(mixed);
            const int first_head = (ngram - 2) * heads_per_ngram;
            for (int local_head = 0; local_head < heads_per_ngram;
                 ++local_head) {
                const int head = first_head + local_head;
                const int64_t row = positive_remainder(
                    signed_mixed, head_vocab_sizes[head]) +
                    head_offsets[head];
                if (row > std::numeric_limits<int32_t>::max()) return false;
                output_indices[static_cast<size_t>(token_index) * num_heads +
                               head] = static_cast<int32_t>(row);
            }
        }

        if (token == eos_token_id) {
            std::fill(local_history.begin(), local_history.end(),
                      eos_token_id);
        } else {
            for (int index = 0; index + 1 < context_len; ++index)
                local_history[index] = local_history[index + 1];
            local_history[context_len - 1] = token;
        }
    }
    std::copy(local_history.begin(), local_history.end(), history);
    return true;
}

bool ple_gather_fp8_rows(const uint8_t* table, int64_t total_rows,
                         int row_dim, const int32_t* indices,
                         int seq_len, int num_heads, float table_scale,
                         float* output) {
    if (!table || !indices || !output || total_rows <= 0 || row_dim <= 0 ||
        seq_len < 0 || num_heads <= 0) {
        return false;
    }
    const int rows = seq_len * num_heads;
    for (int index = 0; index < rows; ++index) {
        const int32_t row = indices[index];
        if (row < 0 || row >= total_rows) return false;
    }
    prefetch_ple_rows(table, row_dim, indices, rows);
    for (int index = 0; index < rows; ++index) {
        const int32_t row = indices[index];
        const uint8_t* source =
            table + static_cast<size_t>(row) * row_dim;
        float* destination =
            output + static_cast<size_t>(index) * row_dim;
        for (int dim = 0; dim < row_dim; ++dim)
            destination[dim] =
                kFp8E4m3fnDecodeTable[source[dim]] * table_scale;
    }
    return true;
}

bool kernel_ple_lookup(const Tensor& token_ids, Tensor& history,
                       const Tensor& table, const Tensor& table_scale,
                       const Tensor& head_vocab_sizes,
                       const Tensor& head_offsets, Tensor& output,
                       int ngram_size, int heads_per_ngram,
                       int eos_token_id, int unigram_vocab_size,
                       int ple_layer_index, uint64_t seed,
                       int n_real_tokens,
                       ThreadPool*) {
    const int num_heads = (ngram_size - 1) * heads_per_ngram;
    if (token_ids.prec != Precision::INT32 ||
        history.prec != Precision::INT32 ||
        table.prec != Precision::RAW_U8 ||
        table_scale.prec != Precision::FP32 ||
        head_vocab_sizes.prec != Precision::INT32 ||
        head_offsets.prec != Precision::INT32 ||
        output.prec != Precision::FP32 ||
        !token_ids.data || !history.data || !table.data ||
        !table_scale.data || !head_vocab_sizes.data || !head_offsets.data ||
        !output.data || history.nelements() != ngram_size ||
        head_vocab_sizes.nelements() != num_heads ||
        head_offsets.nelements() != num_heads || table.shape[0] <= 0 ||
        output.shape[0] != table.shape[0] * num_heads ||
        output.shape[1] != static_cast<int64_t>(token_ids.nelements())) {
        return false;
    }
    const int seq_len = static_cast<int>(token_ids.nelements());
    const int process_len =
        n_real_tokens >= 0 && n_real_tokens < seq_len
            ? n_real_tokens : seq_len;
    int32_t* history_data = history.ptr<int32_t>();
    if (history_data[ngram_size - 1] == 0) {
        std::fill(history_data, history_data + ngram_size - 1,
                  eos_token_id);
        history_data[ngram_size - 1] = 1;
    }
    std::memset(output.data, 0, output.nbytes());
    if (process_len == 0) return true;
    std::vector<int32_t> indices(
        static_cast<size_t>(process_len) * num_heads);
    if (!ple_build_indices(
            token_ids.ptr<int32_t>(), process_len, history_data,
            ngram_size - 1, head_vocab_sizes.ptr<int32_t>(),
            head_offsets.ptr<int32_t>(), ngram_size, heads_per_ngram,
            eos_token_id, unigram_vocab_size, ple_layer_index, seed,
            indices.data())) {
        return false;
    }
    return ple_gather_fp8_rows(
        table.ptr<uint8_t>(), table.shape[1],
        static_cast<int>(table.shape[0]), indices.data(), process_len,
        num_heads, table_scale.ptr<float>()[0], output.ptr<float>());
}

bool kernel_ple_gate(const Tensor& query, const Tensor& key,
                     const Tensor& value, Tensor& output,
                     int hidden_size, int hc_count, ThreadPool*) {
    const int wide = hidden_size * hc_count;
    if (query.prec != Precision::FP32 || key.prec != Precision::FP32 ||
        value.prec != Precision::FP32 || output.prec != Precision::FP32 ||
        !query.data || !key.data || !value.data || !output.data ||
        hidden_size <= 0 || hc_count <= 0 || query.shape[0] != wide ||
        key.shape[0] != wide || output.shape[0] != wide ||
        value.shape[0] != hidden_size || key.shape[1] != query.shape[1] ||
        value.shape[1] != query.shape[1] || output.shape[1] != query.shape[1]) {
        return false;
    }
    const int tokens = static_cast<int>(query.shape[1]);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    for (int token = 0; token < tokens; ++token) {
        const float* q = query.ptr<float>() + static_cast<size_t>(token) * wide;
        const float* k = key.ptr<float>() + static_cast<size_t>(token) * wide;
        const float* v = value.ptr<float>() +
            static_cast<size_t>(token) * hidden_size;
        float* out = output.ptr<float>() + static_cast<size_t>(token) * wide;
        for (int stream = 0; stream < hc_count; ++stream) {
            const int base = stream * hidden_size;
            float dot = 0.0f;
            for (int dim = 0; dim < hidden_size; ++dim)
                dot += q[base + dim] * k[base + dim];
            float score = dot * inv_sqrt;
            const float sign = score < 0.0f ? -1.0f : 1.0f;
            score = sign * std::sqrt(std::max(std::fabs(score), 1e-6f));
            const float gate = 1.0f / (1.0f + std::exp(-score));
            for (int dim = 0; dim < hidden_size; ++dim)
                out[base + dim] = gate * v[dim];
        }
    }
    return true;
}

bool kernel_ple_dilated_conv(const Tensor& input, const Tensor& weight,
                             Tensor& state, Tensor& output,
                             int kernel_size, int dilation,
                             int n_real_tokens, ThreadPool*) {
    const int state_len = (kernel_size - 1) * dilation;
    if (input.prec != Precision::FP32 || weight.prec != Precision::FP32 ||
        state.prec != Precision::FP32 || output.prec != Precision::FP32 ||
        !input.data || !weight.data || !state.data || !output.data ||
        kernel_size <= 0 || dilation <= 0 || input.shape[0] <= 0 ||
        weight.shape[0] != input.shape[0] ||
        weight.shape[1] != kernel_size || state.shape[0] != input.shape[0] ||
        state.shape[1] != state_len || output.shape[0] != input.shape[0] ||
        output.shape[1] != input.shape[1]) {
        return false;
    }
    const int channels = static_cast<int>(input.shape[0]);
    const int tokens = static_cast<int>(input.shape[1]);
    const int process_len =
        n_real_tokens >= 0 && n_real_tokens < tokens
            ? n_real_tokens : tokens;
    std::memset(output.data, 0, output.nbytes());
    std::vector<float> combined(static_cast<size_t>(state_len + tokens));
    for (int channel = 0; channel < channels; ++channel) {
        for (int index = 0; index < state_len; ++index)
            combined[index] = state.ptr<float>()[
                static_cast<size_t>(channel) * state_len + index];
        for (int token = 0; token < tokens; ++token)
            combined[state_len + token] = input.ptr<float>()[
                static_cast<size_t>(token) * channels + channel];
        const float* kernel = weight.ptr<float>() +
            static_cast<size_t>(channel) * kernel_size;
        for (int token = 0; token < process_len; ++token) {
            float sum = 0.0f;
            for (int tap = 0; tap < kernel_size; ++tap) {
                const int source = token + tap * dilation;
                sum += combined[source] * kernel[tap];
            }
            output.ptr<float>()[static_cast<size_t>(token) * channels +
                                channel] = sum / (1.0f + std::exp(-sum));
        }
        for (int index = 0; index < state_len; ++index)
            state.ptr<float>()[static_cast<size_t>(channel) * state_len +
                               index] = combined[process_len + index];
    }
    return true;
}
