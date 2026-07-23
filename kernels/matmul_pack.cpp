#include "kernels/matmul_internal.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace {

bool is_2d_linear_weight(const Tensor& weight) {
    return weight.shape[2] == 1 && weight.shape[3] == 1;
}

bool int8_q8dot_repack_supported(const Tensor& weight) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    return is_2d_linear_weight(weight) && weight.prec == Precision::INT8 &&
           weight.group_size > 0 && (weight.group_size % 32) == 0;
#else
    (void)weight;
    return false;
#endif
}

bool int4_q4dot_repack_supported(const Tensor& weight) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    return is_2d_linear_weight(weight) && weight.prec == Precision::INT4 &&
           weight.group_size > 0 && (weight.group_size % 32) == 0 &&
           weight.shape[1] > 0 && (weight.shape[1] % 32) == 0;
#else
    (void)weight;
    return false;
#endif
}

struct Int8PackingPlan {
    bool build_interleaved = false;
    bool build_q8dot = false;
};

Int8PackingPlan plan_int8_packing(const Tensor& weight) {
    Int8PackingPlan plan;
    if (weight.prec != Precision::INT8 ||
        !g_matmul_config.use_interleave_pack || !is_2d_linear_weight(weight)) {
        return plan;
    }

    const bool can_q8dot = int8_q8dot_repack_supported(weight);
    plan.build_q8dot = can_q8dot;
    // Keep the older interleaved layout only where DOTPROD q8-dot repacking is
    // unavailable. The production path has one canonical W8 layout.
    plan.build_interleaved = !plan.build_q8dot;
    return plan;
}

void maybe_pack_fp16_weight(Tensor& weight, const std::string& key,
                            const void* rowmajor_data,
                            PackedWeightMap& packed_weights) {
    if (weight.prec != Precision::FP16 ||
        !g_matmul_config.use_interleave_pack || !is_2d_linear_weight(weight)) {
        return;
    }

    auto it = packed_weights.find(key);
    if (it == packed_weights.end()) {
        const int N = (int)weight.shape[0];
        const int K = (int)weight.shape[1];
        const auto* b_orig = reinterpret_cast<const __fp16*>(rowmajor_data);
        __fp16* b_packed = pack_b_interleaved_full(b_orig, N, K, K);
        const size_t buf_size = (size_t)((N + 7) / 8) * 8 * K * sizeof(__fp16);
        std::vector<uint8_t> buf((uint8_t*)b_packed,
                                 (uint8_t*)b_packed + buf_size);
        delete[] b_packed;
        it = packed_weights.emplace(key, std::move(buf)).first;
    }
    weight.data = it->second.data();
    weight.is_interleaved = true;
}

void maybe_pack_int8_weight(Tensor& weight, const std::string& key,
                            const void* rowmajor_data,
                            PackedWeightMap& packed_weights) {
#if HAS_NEON
    const Int8PackingPlan plan = plan_int8_packing(weight);
    if (!plan.build_interleaved && !plan.build_q8dot)
        return;

    const int N = (int)weight.shape[0];
    const int K = (int)weight.shape[1];
    const auto* b_orig = reinterpret_cast<const int8_t*>(rowmajor_data);

    if (plan.build_interleaved) {
        const std::string pack_key = key + "#int8_interleaved";
        auto it = packed_weights.find(pack_key);
        if (it == packed_weights.end()) {
            int8_t* b_packed = pack_b_interleaved_int8_full(b_orig, N, K, K);
            const size_t buf_size =
                (size_t)((N + 7) / 8) * 8 * K * sizeof(int8_t);
            std::vector<uint8_t> buf((uint8_t*)b_packed,
                                     (uint8_t*)b_packed + buf_size);
            delete[] b_packed;
            it = packed_weights.emplace(pack_key, std::move(buf)).first;
        }
        weight.data = it->second.data();
        weight.is_interleaved = true;
        // The interleaved W8 buffer is exactly the [N/8,K,8] layout consumed
        // by sparse-A GEMV, so expose it without building a second copy.
        if (key.find("_ffn_value_weight.weights") != std::string::npos) {
            weight.sparse_data = weight.data;
        }
    }

    if (plan.build_q8dot) {
        const std::string q8_key = key + "#int8_q8dot";
        auto it = packed_weights.find(q8_key);
        if (it == packed_weights.end()) {
            const int K_blocks = (K + 31) / 32;
            int8_t* b_q8 = pack_b_q8dot_int8_full(b_orig, N, K, K);
            const size_t buf_size =
                (size_t)((N + 7) / 8) * 8 * K_blocks * 32 * sizeof(int8_t);
            std::vector<uint8_t> buf((uint8_t*)b_q8, (uint8_t*)b_q8 + buf_size);
            delete[] b_q8;
            it = packed_weights.emplace(q8_key, std::move(buf)).first;
        }
        weight.q8_repack_data = it->second.data();
    }
#else
    (void)weight;
    (void)key;
    (void)rowmajor_data;
    (void)packed_weights;
#endif
}

void maybe_pack_int4_g128_weight(Tensor& weight, const std::string& key,
                                 const void* q4dot_data,
                                 PackedWeightMap& packed_weights) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    if (!is_2d_linear_weight(weight))
        return;
    if (weight.prec != Precision::INT4 || !q4dot_data || !weight.scales ||
        weight.group_size != 128 || weight.shape[1] <= 0 ||
        (weight.shape[1] % 128) != 0) {
        return;
    }

    const int N = (int)weight.shape[0];
    const int K = (int)weight.shape[1];
    const std::string g128_key = key + "#int4_q4g128";
    auto it = packed_weights.find(g128_key);
    if (it == packed_weights.end()) {
        uint8_t* b_g128 = pack_b_q4dot_g128_full(
            reinterpret_cast<const uint8_t*>(q4dot_data), weight.scales, N, K,
            (int)weight.groups_per_row);
        if (!b_g128)
            return;
        const size_t buf_size = pack_b_q4dot_g128_bytes(N, K);
        std::vector<uint8_t> buf(b_g128, b_g128 + buf_size);
        delete[] b_g128;
        it = packed_weights.emplace(g128_key, std::move(buf)).first;
    }
    weight.q4_g128_data = it->second.data();
#else
    (void)weight;
    (void)key;
    (void)q4dot_data;
    (void)packed_weights;
#endif
}

void maybe_pack_int4_weight(Tensor& weight, const std::string& key,
                            const void* weight_data,
                            PackedWeightMap& packed_weights) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    if (!is_2d_linear_weight(weight))
        return;
    if (weight.is_q4_g128_packed) {
        weight.q4_g128_data = weight_data;
        return;
    }
    if (weight.is_q4_repacked) {
        weight.q4_repack_data = weight_data;
        maybe_pack_int4_g128_weight(weight, key, weight_data, packed_weights);
        return;
    }
    if (!g_matmul_config.use_interleave_pack ||
        !int4_q4dot_repack_supported(weight)) {
        return;
    }
    const int N = (int)weight.shape[0];
    const int K = (int)weight.shape[1];
    const auto* b_orig = reinterpret_cast<const uint8_t*>(weight_data);
    const std::string q4_key = key + "#int4_q4dot";
    auto it = packed_weights.find(q4_key);
    if (it == packed_weights.end()) {
        const int K_blocks = (K + 31) / 32;
        uint8_t* b_q4 = pack_b_q4dot_int4_full(b_orig, N, K, K);
        const size_t buf_size = (size_t)((N + 7) / 8) * 8 * K_blocks * 16;
        std::vector<uint8_t> buf(b_q4, b_q4 + buf_size);
        delete[] b_q4;
        it = packed_weights.emplace(q4_key, std::move(buf)).first;
    }
    weight.q4_repack_data = it->second.data();
    maybe_pack_int4_g128_weight(weight, key, weight.q4_repack_data,
                                packed_weights);
#else
    (void)weight;
    (void)key;
    (void)weight_data;
    (void)packed_weights;
#endif
}

} // namespace

bool matmul_int4_q4dot_kernel_available() {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    return true;
#else
    return false;
#endif
}

void prepare_matmul_weight(Tensor& weight, const std::string& key,
                           const void* weight_data,
                           PackedWeightMap& packed_weights, bool pack_fp16) {
    if (pack_fp16)
        maybe_pack_fp16_weight(weight, key, weight_data, packed_weights);
    maybe_pack_int8_weight(weight, key, weight_data, packed_weights);
    maybe_pack_int4_weight(weight, key, weight_data, packed_weights);
}

int8_t* pack_b_interleaved_int8_full(const int8_t* B_original, int N, int K,
                                     int K_weight) {
    int N_padded = ((N + 7) / 8) * 8;
    int8_t* dst = new int8_t[(size_t)N_padded * K];
    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0)
            tile_valid = 0;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < tile_valid; j++) {
                dst[n_tile * K + k * 8 + j] =
                    B_original[(n_tile + j) * K_weight + k];
            }
            for (int j = tile_valid; j < 8; j++) {
                dst[n_tile * K + k * 8 + j] = 0;
            }
        }
    }
    return dst;
}

int8_t* pack_b_q8dot_int8_full(const int8_t* B_original, int N, int K,
                               int K_weight) {
    int N_padded = ((N + 7) / 8) * 8;
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    int8_t* dst = new int8_t[(size_t)(N_padded / 8) * blocks_per_row * 8 *
                             MATMUL_Q8_BLOCK];
    std::memset(dst, 0,
                (size_t)(N_padded / 8) * blocks_per_row * 8 * MATMUL_Q8_BLOCK);

    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0)
            tile_valid = 0;
        int8_t* tile =
            dst + (size_t)(n_tile / 8) * blocks_per_row * 8 * MATMUL_Q8_BLOCK;
        for (int qb = 0; qb < blocks_per_row; qb++) {
            int k_begin = qb * MATMUL_Q8_BLOCK;
            int k_end = std::min(k_begin + MATMUL_Q8_BLOCK, K);
            int8_t* block = tile + (size_t)qb * 8 * MATMUL_Q8_BLOCK;
            for (int j = 0; j < tile_valid; j++) {
                int8_t* col = block + j * MATMUL_Q8_BLOCK;
                const int8_t* src =
                    B_original + (size_t)(n_tile + j) * K_weight;
                for (int k = k_begin; k < k_end; k++) {
                    col[k - k_begin] = src[k];
                }
            }
        }
    }
    return dst;
}

uint8_t* pack_b_q4dot_int4_full(const uint8_t* B_original, int N, int K,
                                int K_weight) {
    int N_padded = ((N + 7) / 8) * 8;
    int blocks_per_row = (K + MATMUL_Q8_BLOCK - 1) / MATMUL_Q8_BLOCK;
    int src_row_stride = (K_weight + 1) / 2;
    constexpr int bytes_per_block = MATMUL_Q8_BLOCK / 2;
    size_t total_bytes =
        (size_t)(N_padded / 8) * blocks_per_row * 8 * bytes_per_block;
    uint8_t* dst = new uint8_t[total_bytes];
    std::memset(dst, 0, total_bytes);

    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0)
            tile_valid = 0;
        uint8_t* tile =
            dst + (size_t)(n_tile / 8) * blocks_per_row * 8 * bytes_per_block;
        for (int qb = 0; qb < blocks_per_row; qb++) {
            int k_begin = qb * MATMUL_Q8_BLOCK;
            int k_end = std::min(k_begin + MATMUL_Q8_BLOCK, K);
            int nbytes = (k_end - k_begin + 1) / 2;
            uint8_t* block = tile + (size_t)qb * 8 * bytes_per_block;
            for (int j = 0; j < tile_valid; j++) {
                const uint8_t* src =
                    B_original + (size_t)(n_tile + j) * src_row_stride;
                std::memcpy(block + (size_t)j * bytes_per_block,
                            src + k_begin / 2, (size_t)nbytes);
            }
        }
    }
    return dst;
}

size_t pack_b_q4dot_g128_bytes(int N, int K) {
    int N_padded = ((N + 7) / 8) * 8;
    int groups_per_row = K / 128;
    return (size_t)(N_padded / 8) * groups_per_row * sizeof(Q4B8G128Block);
}

uint8_t* pack_b_q4dot_g128_full(const uint8_t* B_q4dot, const float* scales,
                                int N, int K, int groups_per_row) {
    if (!B_q4dot || !scales || K % 128 != 0)
        return nullptr;
    int N_padded = ((N + 7) / 8) * 8;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    int g128_per_row = K / 128;
    if (groups_per_row != g128_per_row)
        return nullptr;

    size_t total_bytes = pack_b_q4dot_g128_bytes(N, K);
    uint8_t* raw = new uint8_t[total_bytes];
    std::memset(raw, 0, total_bytes);
    auto* dst = reinterpret_cast<Q4B8G128Block*>(raw);

    constexpr int bytes_per_block = MATMUL_Q8_BLOCK / 2;
    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0)
            tile_valid = 0;
        const uint8_t* src_tile = B_q4dot + (size_t)(n_tile / 8) *
                                                blocks_per_row * 8 *
                                                bytes_per_block;
        Q4B8G128Block* dst_tile = dst + (size_t)(n_tile / 8) * g128_per_row;
        for (int g = 0; g < g128_per_row; g++) {
            Q4B8G128Block& block = dst_tile[g];
            for (int c = 0; c < 8; c++) {
                block.scales[c] =
                    (c < tile_valid)
                        ? scales[(size_t)(n_tile + c) * groups_per_row + g]
                        : 0.f;
            }
            for (int qgi = 0; qgi < 4; qgi++) {
                int qb = g * 4 + qgi;
                const uint8_t* src_block =
                    src_tile + (size_t)qb * 8 * bytes_per_block;
                std::memcpy(block.q[qgi], src_block, 8 * bytes_per_block);
            }
        }
    }
    return raw;
}

#if HAS_NEON
__fp16* pack_b_interleaved_full(const __fp16* B_original, int N, int K,
                                int K_weight) {
    int N_padded = ((N + 7) / 8) * 8; // round up to multiple of 8
    __fp16* dst = new __fp16[(size_t)N_padded * K];
    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0)
            tile_valid = 0;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < tile_valid; j++) {
                dst[n_tile * K + k * 8 + j] =
                    B_original[(n_tile + j) * K_weight + k];
            }
            for (int j = tile_valid; j < 8; j++) {
                dst[n_tile * K + k * 8 + j] = (__fp16)0.f;
            }
        }
    }
    return dst;
}

// ---------------------------------------------------------------------------
// A interleaved packing — FP32 [K, M] column-major → FP16 [M/8, K, 8].
//
// For each M-tile of 8 rows, 8 M values at the same k are stored consecutively.
// Enables vld1q_f16 contiguous load of A + vfmlalq_laneq_f16 lane-broadcast.
// FP32→FP16 conversion happens during pack (one-time precision loss).
// ---------------------------------------------------------------------------
#endif
