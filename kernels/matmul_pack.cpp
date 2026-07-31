#include "kernels/matmul_internal.h"
#include "kernels/bf16.h"

#include <algorithm>
#include <cstring>
#include <limits>
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
        !mollm::cpu::capabilities().fp16_interleaved_weights ||
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

void maybe_pack_fp8_weight(Tensor& weight, const std::string& key,
                           const void* rowmajor_data,
                           PackedWeightMap& packed_weights) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    if (weight.prec != Precision::FP8_E4M3 ||
        !g_matmul_config.use_interleave_pack ||
        !is_2d_linear_weight(weight) || !rowmajor_data ||
        !weight.e8m0_scales || !weight.is_fp8_block128 ||
        weight.shape[1] <= 0 || weight.shape[1] % MATMUL_Q8_BLOCK != 0) {
        return;
    }
    const int N = static_cast<int>(weight.shape[0]);
    const int K = static_cast<int>(weight.shape[1]);
    const int groups = K / MATMUL_Q8_BLOCK;
    const size_t data_bytes = pack_fp8_e4m3_q8dot_bytes(N, K);
    const size_t scale_bytes =
        static_cast<size_t>(N) * groups * sizeof(float);
    const std::string pack_key = key + "#fp8_q8dot";
    auto it = packed_weights.find(pack_key);
    if (it == packed_weights.end()) {
        std::vector<uint8_t> buffer(data_bytes + scale_bytes);
        auto* packed = reinterpret_cast<int8_t*>(buffer.data());
        auto* scales = reinterpret_cast<float*>(
            buffer.data() + data_bytes);
        if (!pack_fp8_e4m3_q8dot(
                static_cast<const uint8_t*>(rowmajor_data),
                weight.e8m0_scales, N, K, packed, scales)) {
            return;
        }
        it = packed_weights.emplace(pack_key, std::move(buffer)).first;
    }
    weight.q8_repack_data = it->second.data();
    weight.fp8_q8_scales = reinterpret_cast<const float*>(
        it->second.data() + data_bytes);
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

void maybe_pack_int4_g32_weight(Tensor& weight, const std::string& key,
                                const void* q4dot_data,
                                PackedWeightMap& packed_weights) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    if (!is_2d_linear_weight(weight))
        return;
    if (weight.prec != Precision::INT4 || !q4dot_data || !weight.scales ||
        weight.group_size != 32 || weight.shape[1] <= 0 ||
        (weight.shape[1] % 32) != 0) {
        return;
    }

    const int N = (int)weight.shape[0];
    const int K = (int)weight.shape[1];
    const std::string g32_key = key + "#int4_q4g32";
    auto it = packed_weights.find(g32_key);
    if (it == packed_weights.end()) {
        uint8_t* b_g32 = pack_b_q4dot_g32_full(
            reinterpret_cast<const uint8_t*>(q4dot_data), weight.scales, N, K,
            (int)weight.groups_per_row);
        if (!b_g32)
            return;
        const size_t buf_size = pack_b_q4dot_g32_bytes(N, K);
        std::vector<uint8_t> buf(b_g32, b_g32 + buf_size);
        delete[] b_g32;
        it = packed_weights.emplace(g32_key, std::move(buf)).first;
    }
    weight.q4_g32_data = it->second.data();
#else
    (void)weight;
    (void)key;
    (void)q4dot_data;
    (void)packed_weights;
#endif
}

void maybe_pack_int4_weight(Tensor& weight, const std::string& key,
                            const void* weight_data,
                            PackedWeightMap& packed_weights,
                            PreparedWeightMap& prepared_weights) {
    if (mollm::cpu::capabilities().x86_w4_q8_activations &&
        is_2d_linear_weight(weight) && weight.prec == Precision::INT4 &&
        weight.is_q4_g32_packed && weight_data && weight.group_size == 32 &&
        weight.shape[1] > 0 && (weight.shape[1] % 32) == 0 &&
        key.find("lm_head") == std::string::npos &&
        key.find("_experts_") == std::string::npos) {
        const int N = static_cast<int>(weight.shape[0]);
        const int K = static_cast<int>(weight.shape[1]);
        PreparedWeight& prepared = prepared_weights[key];
        auto& vnni = prepared.layout(WeightLayout::X86_VNNI_Q4_G32);
        if (vnni.empty()) {
            uint8_t* packed = pack_b_q4_vnni_full(weight_data, N, K);
            if (packed) {
                const size_t bytes = pack_b_q4_vnni_bytes(N, K);
                vnni.assign(packed, packed + bytes);
                delete[] packed;
            }
        }
        if (!vnni.empty())
            weight.prepared_weight = &prepared;
    }
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    if (!is_2d_linear_weight(weight))
        return;
    if (weight.is_q4_g32_packed) {
        weight.q4_g32_data = weight_data;
        return;
    }
    if (weight.is_q4_g128_packed) {
        weight.q4_g128_data = weight_data;
        return;
    }
    if (weight.is_q4_repacked) {
        weight.q4_repack_data = weight_data;
        maybe_pack_int4_g32_weight(weight, key, weight_data, packed_weights);
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
    maybe_pack_int4_g32_weight(weight, key, weight.q4_repack_data,
                               packed_weights);
    maybe_pack_int4_g128_weight(weight, key, weight.q4_repack_data,
                                packed_weights);
#else
    (void)weight;
    (void)key;
    (void)weight_data;
    (void)packed_weights;
    (void)prepared_weights;
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
                           PackedWeightMap& packed_weights,
                           PreparedWeightMap& prepared_weights,
                           bool pack_fp16, bool pack_fp8) {
    if (pack_fp16)
        maybe_pack_fp16_weight(weight, key, weight_data, packed_weights);
    if (pack_fp8)
        maybe_pack_fp8_weight(weight, key, weight_data, packed_weights);
    maybe_pack_int8_weight(weight, key, weight_data, packed_weights);
    maybe_pack_int4_weight(weight, key, weight_data, packed_weights,
                           prepared_weights);
}

bool prepare_fp8_bf16_fp16_weight(
    Tensor& weight, const std::string& key, const void* weight_data,
    PackedWeightMap& packed_weights) {
#if HAS_NEON
    if (weight.prec != Precision::FP8_E4M3 ||
        !is_2d_linear_weight(weight) || !weight_data ||
        !weight.e8m0_scales || !weight.is_fp8_block128 ||
        weight.shape[0] <= 0 || weight.shape[1] <= 0) {
        return false;
    }
    const int N = static_cast<int>(weight.shape[0]);
    const int K = static_cast<int>(weight.shape[1]);
    const int padded_n = ((N + 7) / 8) * 8;
    const int k_blocks = (K + 127) / 128;
    const size_t elements = static_cast<size_t>(padded_n) * K;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(__fp16))
        return false;

    const std::string pack_key = key + "#fp8_bf16_fp16";
    auto it = packed_weights.find(pack_key);
    if (it == packed_weights.end()) {
        std::vector<uint8_t> buffer(elements * sizeof(__fp16));
        auto* packed = reinterpret_cast<__fp16*>(buffer.data());
        const auto* source = static_cast<const uint8_t*>(weight_data);
        for (int n_tile = 0; n_tile < padded_n; n_tile += 8) {
            for (int k = 0; k < K; ++k) {
                for (int lane = 0; lane < 8; ++lane) {
                    const int n = n_tile + lane;
                    float value = 0.0f;
                    if (n < N) {
                        value =
                            decode_fp8_e4m3fn(
                                source[static_cast<size_t>(n) * K + k]) *
                            decode_e8m0(
                                weight.e8m0_scales[
                                    static_cast<size_t>(n / 128) *
                                        k_blocks +
                                    k / 128]);
                        value = mollm_round_to_bf16(value);
                    }
                    packed[
                        static_cast<size_t>(n_tile) * K +
                        static_cast<size_t>(k) * 8 + lane] =
                        static_cast<__fp16>(value);
                }
            }
        }
        it = packed_weights.emplace(pack_key, std::move(buffer)).first;
    }
    weight.fp8_bf16_fp16_data = it->second.data();
    return true;
#else
    (void)weight;
    (void)key;
    (void)weight_data;
    (void)packed_weights;
    return false;
#endif
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

size_t pack_b_q4dot_g32_bytes(int N, int K) {
    int N_padded = ((N + 7) / 8) * 8;
    int groups_per_row = K / 32;
    return (size_t)(N_padded / 8) * groups_per_row * sizeof(Q4B8G32Block);
}

uint8_t* pack_b_q4dot_g32_full(const uint8_t* B_q4dot, const float* scales,
                               int N, int K, int groups_per_row) {
    if (!B_q4dot || !scales || K % 32 != 0)
        return nullptr;
    int N_padded = ((N + 7) / 8) * 8;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    if (groups_per_row != blocks_per_row)
        return nullptr;

    size_t total_bytes = pack_b_q4dot_g32_bytes(N, K);
    uint8_t* raw = new uint8_t[total_bytes];
    std::memset(raw, 0, total_bytes);
    auto* dst = reinterpret_cast<Q4B8G32Block*>(raw);

    constexpr int bytes_per_block = MATMUL_Q8_BLOCK / 2;
    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0)
            tile_valid = 0;
        const uint8_t* src_tile =
            B_q4dot + (size_t)(n_tile / 8) * blocks_per_row * 8 *
                          bytes_per_block;
        Q4B8G32Block* dst_tile =
            dst + (size_t)(n_tile / 8) * blocks_per_row;
        for (int qb = 0; qb < blocks_per_row; qb++) {
            Q4B8G32Block& block = dst_tile[qb];
            for (int c = 0; c < 8; c++) {
                block.scales[c] =
                    (c < tile_valid)
                        ? scales[(size_t)(n_tile + c) * groups_per_row + qb]
                        : 0.f;
            }
            const uint8_t* src_block =
                src_tile + (size_t)qb * 8 * bytes_per_block;
            std::memcpy(block.q, src_block, 8 * bytes_per_block);
        }
    }
    return raw;
}

size_t pack_b_q4_vnni_bytes(int N, int K) {
    const int N_padded = ((N + 7) / 8) * 8;
    return static_cast<size_t>(N_padded / 8) * (K / 32) *
           sizeof(Q4B8G32VnniBlock);
}

uint8_t* pack_b_q4_vnni_full(const void* B_q4_g32, int N, int K) {
    if (!B_q4_g32 || N <= 0 || K <= 0 || (K % 32) != 0)
        return nullptr;
    const int N_padded = ((N + 7) / 8) * 8;
    const int groups = K / 32;
    const auto* source = static_cast<const uint8_t*>(B_q4_g32);
    auto* raw = new uint8_t[pack_b_q4_vnni_bytes(N, K)];
    auto* destination = reinterpret_cast<Q4B8G32VnniBlock*>(raw);

    for (int n_tile = 0; n_tile < N_padded / 8; ++n_tile) {
        for (int group = 0; group < groups; ++group) {
            Q4B8G32Block src;
            const size_t block_index =
                static_cast<size_t>(n_tile) * groups + group;
            std::memcpy(&src,
                        source + block_index * sizeof(Q4B8G32Block),
                        sizeof(src));
            auto& dst =
                destination[block_index];
            for (int chunk = 0; chunk < 8; ++chunk) {
                for (int n = 0; n < 8; ++n) {
                    for (int pair = 0; pair < 2; ++pair) {
                        const int k = chunk * 4 + pair * 2;
                        const uint8_t source_byte = src.q[n][k / 2];
                        const uint8_t low =
                            static_cast<uint8_t>(
                                ((source_byte & 0x0f) + 8) & 0x0f);
                        const uint8_t high =
                            static_cast<uint8_t>(
                                (((source_byte >> 4) & 0x0f) + 8) & 0x0f);
                        dst.q[chunk][n * 2 + pair] =
                            static_cast<uint8_t>(low | (high << 4));
                    }
                }
            }
        }
    }
    return raw;
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
