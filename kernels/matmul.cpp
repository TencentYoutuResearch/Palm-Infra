#include "kernels/matmul.h"
#include "kernels/threading.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

// ---------------------------------------------------------------------------
// Post-hoc activation helpers (used by FP32 acc fallback kernels + standard
// FP32 path + parallel M-sharded paths that can't fuse activation into their
// writeback). Applied as a separate pass over the matmul output.
//
// For the fused path (FP16 acc lane-FMA kernel and GEMV FP16 acc kernel),
// activation is applied in the kernel writeback directly — see
// matmul_fp16_neon_8x8_range_packed_a_fp16acc / matmul_fp16_neon_gemv_range_fp16acc.
// ---------------------------------------------------------------------------

// Apply activation to C[m_begin..m_end, n_begin..n_end) where C is row-major
// with leading dim ldc (in float elements, not bytes).
// act_n_begin/act_n_len are LOCAL coords (already translated by caller).
static void apply_activation_to_range(float* C, int M, int N, int ldc,
                                       int m_begin, int m_end,
                                       Activation act,
                                       int act_n_begin, int act_n_len) {
    if (act == Activation::NONE || act_n_len == 0) return;

    bool full_N = (act_n_len < 0) || (act_n_begin == 0 && act_n_len >= N);
#if HAS_NEON
    for (int m = m_begin; m < m_end; m++) {
        float* row = C + m * ldc;
        if (full_N) {
            int n = 0;
            for (; n + 3 < N; n += 4) {
                float32x4_t v = vld1q_f32(row + n);
                v = apply_activation_f32_neon(v, act);
                vst1q_f32(row + n, v);
            }
            for (; n < N; n++) {
                row[n] = apply_activation_scalar(row[n], act);
            }
        } else {
            int n_end_act = std::min(act_n_begin + act_n_len, N);
            int n = act_n_begin;
            // Vectorized 4-wide run when fully inside active range.
            for (; n + 3 < n_end_act; n += 4) {
                float32x4_t v = vld1q_f32(row + n);
                v = apply_activation_f32_neon(v, act);
                vst1q_f32(row + n, v);
            }
            for (; n < n_end_act; n++) {
                row[n] = apply_activation_scalar(row[n], act);
            }
        }
    }
#else
    for (int m = m_begin; m < m_end; m++) {
        float* row = C + m * ldc;
        int n_end_act = full_N ? N : std::min(act_n_begin + act_n_len, N);
        int n = full_N ? 0 : act_n_begin;
        for (; n < n_end_act; n++) {
            row[n] = apply_activation_scalar(row[n], act);
        }
    }
#endif
}

// Apply activation to a GEMV output (M=1, single row of N elements).
// C is contiguous (GEMV output always is).
static void apply_activation_to_range_gemv(float* C, int N,
                                            Activation act,
                                            int act_n_begin, int act_n_len) {
    apply_activation_to_range(C, 1, N, N, 0, 1, act, act_n_begin, act_n_len);
}

// Profiling counters for A pack overhead (reset by bench).
// Read via pack_a_total_ms() / matmul_total_ms().
// std::atomic<double> has no fetch_add, use mutex-protected doubles.
static double g_pack_a_ms = 0;
static double g_matmul_ms = 0;
static double g_q8_quant_a_ms = 0;
static long long g_pack_a_calls = 0;
static long long g_q8_quant_a_calls = 0;
static std::mutex g_prof_mtx;

static bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "0") != 0;
}

static int env_int_or(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0) return fallback;
    return (int)parsed;
}

static bool matmul_shape_profile_enabled() {
    static bool enabled = env_flag_enabled("MOLLM_MATMUL_SHAPE_PROFILE");
    return enabled;
}

struct MatmulShapeProfileRow {
    const char* phase = "unscoped";
    const char* path = "unknown";
    int M = 0;
    int N = 0;
    int K = 0;
    int group_size = 0;
    int groups_per_row = 0;
    int threads = 1;
    bool has_q8_repack = false;
    bool b_interleaved = false;
    long long calls = 0;
    double total_ms = 0.0;
};

static const char* g_matmul_profile_phase = "unscoped";
static std::vector<MatmulShapeProfileRow> g_matmul_shape_profile;

static bool same_shape_profile_key(const MatmulShapeProfileRow& row,
                                   const MatmulShapeProfileRow& key) {
    return std::strcmp(row.phase, key.phase) == 0 &&
           std::strcmp(row.path, key.path) == 0 &&
           row.M == key.M && row.N == key.N && row.K == key.K &&
           row.group_size == key.group_size &&
           row.groups_per_row == key.groups_per_row &&
           row.threads == key.threads &&
           row.has_q8_repack == key.has_q8_repack &&
           row.b_interleaved == key.b_interleaved;
}

static void record_matmul_shape_profile_locked(const MatmulShapeProfileRow& key,
                                               double elapsed_ms) {
    for (auto& row : g_matmul_shape_profile) {
        if (same_shape_profile_key(row, key)) {
            row.calls += 1;
            row.total_ms += elapsed_ms;
            return;
        }
    }
    MatmulShapeProfileRow row = key;
    row.calls = 1;
    row.total_ms = elapsed_ms;
    g_matmul_shape_profile.push_back(row);
}

extern "C" double mollm_pack_a_total_ms() { return g_pack_a_ms; }
extern "C" long long mollm_pack_a_calls() { return g_pack_a_calls; }
extern "C" double mollm_matmul_total_ms() { return g_matmul_ms; }
extern "C" double mollm_q8_quant_a_total_ms() { return g_q8_quant_a_ms; }
extern "C" long long mollm_q8_quant_a_calls() { return g_q8_quant_a_calls; }
extern "C" int mollm_matmul_shape_profile_enabled() {
    return matmul_shape_profile_enabled() ? 1 : 0;
}
extern "C" void mollm_set_matmul_profile_phase(const char* phase) {
    if (!matmul_shape_profile_enabled()) return;
    std::lock_guard<std::mutex> lk(g_prof_mtx);
    g_matmul_profile_phase = (phase && phase[0]) ? phase : "unscoped";
}
extern "C" void mollm_reset_matmul_shape_profile() {
    std::lock_guard<std::mutex> lk(g_prof_mtx);
    g_matmul_shape_profile.clear();
    g_matmul_profile_phase = "unscoped";
}
extern "C" void mollm_print_matmul_shape_profile(const char* title, int top_n) {
    if (!matmul_shape_profile_enabled()) return;

    std::vector<MatmulShapeProfileRow> rows;
    {
        std::lock_guard<std::mutex> lk(g_prof_mtx);
        rows = g_matmul_shape_profile;
    }
    if (rows.empty()) return;

    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.total_ms > b.total_ms;
    });

    double total_ms = 0.0;
    for (const auto& row : rows) total_ms += row.total_ms;
    if (top_n <= 0 || top_n > (int)rows.size()) top_n = (int)rows.size();

    std::printf("\n[%s]\n", title && title[0] ? title : "matmul_shape_profile");
    std::printf("  %-16s %-28s %5s %7s %6s %7s %7s %4s %3s %3s %8s %10s %9s %7s %9s\n",
                "phase", "path", "M", "N", "K", "group", "groups", "thr",
                "q8r", "int", "calls", "total_ms", "avg_ms", "pct", "GMAC/s");
    std::printf("  %-16s %-28s %5s %7s %6s %7s %7s %4s %3s %3s %8s %10s %9s %7s %9s\n",
                "---", "---", "---", "---", "---", "---", "---", "---",
                "---", "---", "---", "---", "---", "---", "---");
    for (int i = 0; i < top_n; i++) {
        const auto& row = rows[i];
        double avg_ms = row.calls > 0 ? row.total_ms / row.calls : 0.0;
        double pct = total_ms > 0.0 ? row.total_ms * 100.0 / total_ms : 0.0;
        double gmac = (double)row.calls * (double)row.M * (double)row.N * (double)row.K / 1e9;
        double gmac_s = row.total_ms > 0.0 ? gmac / (row.total_ms / 1000.0) : 0.0;
        std::printf("  %-16s %-28s %5d %7d %6d %7d %7d %4d %3d %3d %8lld %10.2f %9.3f %6.1f%% %9.2f\n",
                    row.phase, row.path, row.M, row.N, row.K,
                    row.group_size, row.groups_per_row, row.threads,
                    row.has_q8_repack ? 1 : 0,
                    row.b_interleaved ? 1 : 0,
                    row.calls, row.total_ms, avg_ms, pct, gmac_s);
    }
}
extern "C" void mollm_reset_pack_counters() {
    std::lock_guard<std::mutex> lk(g_prof_mtx);
    g_pack_a_ms = 0;
    g_pack_a_calls = 0;
    g_q8_quant_a_ms = 0;
    g_q8_quant_a_calls = 0;
    g_matmul_ms = 0;
}

// RAII timer for kernel_matmul_fp32 — captures all return paths.
struct MatmulTimer {
    std::chrono::steady_clock::time_point t0;
    MatmulShapeProfileRow shape;
    bool shape_valid = false;

    MatmulTimer() : t0(std::chrono::steady_clock::now()) {}
    void set_shape(const char* path, int M, int N, int K,
                   int group_size = 0, int groups_per_row = 0,
                   bool has_q8_repack = false, bool b_interleaved = false,
                   int threads = 1) {
        if (!matmul_shape_profile_enabled()) return;
        shape.path = path ? path : "unknown";
        shape.M = M;
        shape.N = N;
        shape.K = K;
        shape.group_size = group_size;
        shape.groups_per_row = groups_per_row;
        shape.has_q8_repack = has_q8_repack;
        shape.b_interleaved = b_interleaved;
        shape.threads = threads;
        shape_valid = true;
    }
    ~MatmulTimer() {
        auto t1 = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::lock_guard<std::mutex> lk(g_prof_mtx);
        g_matmul_ms += elapsed_ms;
        if (shape_valid && matmul_shape_profile_enabled()) {
            shape.phase = g_matmul_profile_phase;
            record_matmul_shape_profile_locked(shape, elapsed_ms);
        }
    }
};

MatmulConfig g_matmul_config;

// Debug override: force FP32 accumulation (for precision comparison)
bool g_mollm_force_fp32_acc = false;

static constexpr int W8_Q8_BLOCK = 32;

struct alignas(16) Q4B8G128Block {
    float scales[8];
    uint8_t q[4][8][16];
};
static_assert(sizeof(Q4B8G128Block) == 544, "unexpected Q4B8G128Block size");

enum class W8ScaleMode {
    PerChannel,
    PerBlock32,
    PerGroup,
};

static inline W8ScaleMode w8_scale_mode(int group_size, int groups_per_row) {
    if (groups_per_row == 1) return W8ScaleMode::PerChannel;
    if (group_size == W8_Q8_BLOCK) return W8ScaleMode::PerBlock32;
    return W8ScaleMode::PerGroup;
}

static inline int w8_scale_group(W8ScaleMode mode, int qb, int group_size) {
    switch (mode) {
        case W8ScaleMode::PerChannel:
            return 0;
        case W8ScaleMode::PerBlock32:
            return qb;
        case W8ScaleMode::PerGroup:
            return (qb * W8_Q8_BLOCK) / group_size;
    }
    return 0;
}

#if HAS_NEON
static inline void load_w8_b_scales8(const float* scales, int n, int c_valid,
                                     int groups_per_row, int group,
                                     float32x4_t& lo, float32x4_t& hi) {
    float tmp[8];
    for (int c = 0; c < 8; c++) {
        tmp[c] = (c < c_valid) ? scales[(n + c) * groups_per_row + group] : 0.f;
    }
    lo = vld1q_f32(tmp);
    hi = vld1q_f32(tmp + 4);
}
#endif

static void debug_w8_path_once(const char* path, int M, int N, int K,
                               int group_size, bool has_q8_repack) {
    static bool enabled = env_flag_enabled("MOLLM_W8_DEBUG_PATHS");
    if (!enabled) return;
    static std::mutex mu;
    static int count = 0;
    std::lock_guard<std::mutex> lock(mu);
    if (count++ >= 80) return;
    std::fprintf(stderr, "W8_PATH path=%s M=%d N=%d K=%d group=%d q8_repack=%d\n",
                 path, M, N, K, group_size, has_q8_repack ? 1 : 0);
}

// ---------------------------------------------------------------------------
// scalar matmul (fallback)
// ---------------------------------------------------------------------------

// After weight dim swap: B has shape [N, K] (N=output, K=input).
// Weight file stores row-major [N, K]: data[n*K + k] = W[n, k].
// Tensor access: B.at<float>(n, k) = data[n*K + k].
// C[m,n] = sum_k A[k + m*lda] * B[n*K_weight + k] = A[m,:] @ W[n,:].
static void matmul_fp32_scalar_range(const float* A, const float* B, float* C,
                                     int M, int N, int K,
                                     int lda, int K_weight, int ldc,
                                     int m_begin, int m_end) {
    (void)M;
    for (int m = m_begin; m < m_end; m++) {
        float* c_row = C + m * ldc;
        for (int n = 0; n < N; n++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += A[k + m * lda] * B[n * K_weight + k];
            }
            c_row[n] = sum;
        }
    }
}

int8_t* pack_b_interleaved_int8_full(const int8_t* B_original, int N, int K, int K_weight) {
    int N_padded = ((N + 7) / 8) * 8;
    int8_t* dst = new int8_t[(size_t)N_padded * K];
    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0) tile_valid = 0;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < tile_valid; j++) {
                dst[n_tile * K + k * 8 + j] = B_original[(n_tile + j) * K_weight + k];
            }
            for (int j = tile_valid; j < 8; j++) {
                dst[n_tile * K + k * 8 + j] = 0;
            }
        }
    }
    return dst;
}

int8_t* pack_b_q8dot_int8_full(const int8_t* B_original, int N, int K, int K_weight) {
    int N_padded = ((N + 7) / 8) * 8;
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;
    int8_t* dst = new int8_t[(size_t)(N_padded / 8) * blocks_per_row * 8 * W8_Q8_BLOCK];
    std::memset(dst, 0, (size_t)(N_padded / 8) * blocks_per_row * 8 * W8_Q8_BLOCK);

    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0) tile_valid = 0;
        int8_t* tile = dst + (size_t)(n_tile / 8) * blocks_per_row * 8 * W8_Q8_BLOCK;
        for (int qb = 0; qb < blocks_per_row; qb++) {
            int k_begin = qb * W8_Q8_BLOCK;
            int k_end = std::min(k_begin + W8_Q8_BLOCK, K);
            int8_t* block = tile + (size_t)qb * 8 * W8_Q8_BLOCK;
            for (int j = 0; j < tile_valid; j++) {
                int8_t* col = block + j * W8_Q8_BLOCK;
                const int8_t* src = B_original + (size_t)(n_tile + j) * K_weight;
                for (int k = k_begin; k < k_end; k++) {
                    col[k - k_begin] = src[k];
                }
            }
        }
    }
    return dst;
}

uint8_t* pack_b_q4dot_int4_full(const uint8_t* B_original, int N, int K, int K_weight) {
    int N_padded = ((N + 7) / 8) * 8;
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;
    int src_row_stride = (K_weight + 1) / 2;
    constexpr int bytes_per_block = W8_Q8_BLOCK / 2;
    size_t total_bytes = (size_t)(N_padded / 8) * blocks_per_row * 8 * bytes_per_block;
    uint8_t* dst = new uint8_t[total_bytes];
    std::memset(dst, 0, total_bytes);

    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0) tile_valid = 0;
        uint8_t* tile = dst + (size_t)(n_tile / 8) * blocks_per_row * 8 * bytes_per_block;
        for (int qb = 0; qb < blocks_per_row; qb++) {
            int k_begin = qb * W8_Q8_BLOCK;
            int k_end = std::min(k_begin + W8_Q8_BLOCK, K);
            int nbytes = (k_end - k_begin + 1) / 2;
            uint8_t* block = tile + (size_t)qb * 8 * bytes_per_block;
            for (int j = 0; j < tile_valid; j++) {
                const uint8_t* src = B_original + (size_t)(n_tile + j) * src_row_stride;
                std::memcpy(block + (size_t)j * bytes_per_block,
                            src + k_begin / 2,
                            (size_t)nbytes);
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
    if (!B_q4dot || !scales || K % 128 != 0) return nullptr;
    int N_padded = ((N + 7) / 8) * 8;
    int blocks_per_row = K / W8_Q8_BLOCK;
    int g128_per_row = K / 128;
    if (groups_per_row != g128_per_row) return nullptr;

    size_t total_bytes = pack_b_q4dot_g128_bytes(N, K);
    uint8_t* raw = new uint8_t[total_bytes];
    std::memset(raw, 0, total_bytes);
    auto* dst = reinterpret_cast<Q4B8G128Block*>(raw);

    constexpr int bytes_per_block = W8_Q8_BLOCK / 2;
    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0) tile_valid = 0;
        const uint8_t* src_tile =
            B_q4dot + (size_t)(n_tile / 8) * blocks_per_row * 8 * bytes_per_block;
        Q4B8G128Block* dst_tile = dst + (size_t)(n_tile / 8) * g128_per_row;
        for (int g = 0; g < g128_per_row; g++) {
            Q4B8G128Block& block = dst_tile[g];
            for (int c = 0; c < 8; c++) {
                block.scales[c] = (c < tile_valid)
                    ? scales[(size_t)(n_tile + c) * groups_per_row + g]
                    : 0.f;
            }
            for (int qgi = 0; qgi < 4; qgi++) {
                int qb = g * 4 + qgi;
                const uint8_t* src_block = src_tile + (size_t)qb * 8 * bytes_per_block;
                std::memcpy(block.q[qgi], src_block, 8 * bytes_per_block);
            }
        }
    }
    return raw;
}

int8_t* pack_b_sparse_int4_g128_full(const void* source, int N, int K) {
    if (!source || K % 128 != 0) return nullptr;
    int N_padded = ((N + 7) / 8) * 8;
    int groups_per_row = K / 128;
    int8_t* dst = new int8_t[(size_t)N_padded * K];
    std::memset(dst, 0, (size_t)N_padded * K);
    const auto* blocks = reinterpret_cast<const Q4B8G128Block*>(source);
    for (int n = 0; n < N_padded; n += 8) {
        const Q4B8G128Block* tile = blocks + (size_t)(n / 8) * groups_per_row;
        int valid = std::min(8, N - n);
        for (int g = 0; g < groups_per_row; ++g) {
            for (int p = 0; p < 128; ++p) {
                int qgi = p >> 5;
                int pos = p & 31;
                int k = g * 128 + p;
                for (int j = 0; j < valid; ++j) {
                    uint8_t packed = tile[g].q[qgi][j][pos >> 1];
                    int value = (pos & 1) ? (packed >> 4) : (packed & 15);
                    dst[(size_t)n * K + (size_t)k * 8 + j] =
                        (int8_t)(value >= 8 ? value - 16 : value);
                }
            }
        }
    }
    return dst;
}

static void matmul_int8_scalar_range(const float* A, const int8_t* B, const float* scales,
                                     int group_size, int groups_per_row,
                                     float* C, int M, int N, int K,
                                     int lda, int K_weight, int ldc,
                                     int m_begin, int m_end,
                                     int n_begin, int n_end,
                                     bool b_interleaved = false) {
    (void)M;
    if (group_size <= 0) group_size = K;
    if (groups_per_row <= 0) groups_per_row = 1;

    for (int m = m_begin; m < m_end; m++) {
        float* c_row = C + m * ldc;
        for (int n = n_begin; n < n_end; n++) {
            const int8_t* b_row = B + n * K_weight;
            const float* s_row = scales + n * groups_per_row;
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                int g = k / group_size;
                int8_t q = b_interleaved
                    ? B[(n & ~7) * K_weight + k * 8 + (n & 7)]
                    : b_row[k];
                sum += A[k + m * lda] * ((float)q * s_row[g]);
            }
            c_row[n] = sum;
        }
    }
}

static inline int8_t unpack_int4_signed(uint8_t byte, bool high_nibble) {
    int v = high_nibble ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    if (v >= 8) v -= 16;
    return (int8_t)v;
}

#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
static inline int8x16_t sign_extend_int4_nibbles(uint8x16_t nibbles) {
    // Move the 4-bit sign bit into bit 7, then arithmetic shift it back.
    return vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(nibbles, 4)), 4);
}

static inline void unpack_int4x32_signed(const uint8_t* src,
                                         int8x16_t& even,
                                         int8x16_t& odd) {
    uint8x16_t packed = vld1q_u8(src);
    uint8x16_t lo = vandq_u8(packed, vdupq_n_u8(0x0F));
    uint8x16_t hi = vshrq_n_u8(packed, 4);
    even = sign_extend_int4_nibbles(lo);
    odd = sign_extend_int4_nibbles(hi);
}

static inline void load_int4x32_signed_scaled16(const uint8_t* src,
                                                int8x16_t& even_scaled,
                                                int8x16_t& odd_scaled) {
    uint8x16_t packed = vld1q_u8(src);
    even_scaled = vreinterpretq_s8_u8(vshlq_n_u8(packed, 4));
    odd_scaled = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0xF0)));
}

static inline int32x4_t q4_q8_dot32(int8x16_t q4_even, int8x16_t q4_odd,
                                    int8x16_t qa_even, int8x16_t qa_odd) {
    int32x4_t d = vdupq_n_s32(0);
    d = vdotq_s32(d, q4_even, qa_even);
    d = vdotq_s32(d, q4_odd, qa_odd);
    return d;
}

static inline float32x4_t q4_scaled16_dot_to_f32(int32x4_t dots) {
#if defined(__aarch64__)
    return vcvtq_n_f32_s32(dots, 4);
#else
    return vmulq_n_f32(vcvtq_f32_s32(dots), 1.0f / 16.0f);
#endif
}

struct alignas(16) Q8A4Block {
    float scales[4];
    int8_t even[4][16];
    int8_t odd[4][16];
};

static void matmul_int4_q8dot_neon_gemv_range(
    const int8_t* qA, const int8_t* qA_even_pre, const int8_t* qA_odd_pre,
    const float* a_scales,
    const uint8_t* B, const uint8_t* B_repack, const float* scales,
    int group_size, int groups_per_row,
    float* C, int K, int K_weight,
    int n_begin, int n_end)
{
    if (group_size <= 0) group_size = K;
    int row_stride = (K_weight + 1) / 2;
    int blocks_per_row = K / W8_Q8_BLOCK;
    constexpr int bytes_per_block = W8_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        int c_valid = n_tile_end - n;
        bool full_n_tile = (c_valid == 8);
        float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
        float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
        if (scale_per_channel) {
            load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                              bscale_lo_pc, bscale_hi_pc);
        }

        float32x4_t acc_lo = vdupq_n_f32(0.f);
        float32x4_t acc_hi = vdupq_n_f32(0.f);

        auto run_qblock = [&](int qb, float32x4_t bscale_lo, float32x4_t bscale_hi) {
            int byte_off = qb * (W8_Q8_BLOCK / 2);
            const uint8_t* b_repack_block = B_repack
                ? B_repack + ((size_t)(n / 8) * blocks_per_row + qb) * 8 * bytes_per_block
                : nullptr;

            int8x16_t qa_even;
            int8x16_t qa_odd;
            if (qA_even_pre && qA_odd_pre) {
                qa_even = vld1q_s8(qA_even_pre + (size_t)qb * 16);
                qa_odd = vld1q_s8(qA_odd_pre + (size_t)qb * 16);
            } else {
                int8x16_t qa0 = vld1q_s8(qA + (size_t)qb * W8_Q8_BLOCK);
                int8x16_t qa1 = vld1q_s8(qA + (size_t)qb * W8_Q8_BLOCK + 16);
                qa_even = vuzp1q_s8(qa0, qa1);
                qa_odd = vuzp2q_s8(qa0, qa1);
            }

            int32x4_t d0 = vdupq_n_s32(0);
            int32x4_t d1 = vdupq_n_s32(0);
            int32x4_t d2 = vdupq_n_s32(0);
            int32x4_t d3 = vdupq_n_s32(0);
            int32x4_t d4 = vdupq_n_s32(0);
            int32x4_t d5 = vdupq_n_s32(0);
            int32x4_t d6 = vdupq_n_s32(0);
            int32x4_t d7 = vdupq_n_s32(0);

            auto dot_col = [&](int c, int32x4_t& d) {
                int8x16_t q4_even, q4_odd;
                const uint8_t* src = b_repack_block
                    ? b_repack_block + (size_t)c * bytes_per_block
                    : B + (size_t)(n + c) * row_stride + byte_off;
                load_int4x32_signed_scaled16(src,
                                             q4_even, q4_odd);
                d = q4_q8_dot32(q4_even, q4_odd, qa_even, qa_odd);
            };
            if (full_n_tile) {
                dot_col(0, d0);
                dot_col(1, d1);
                dot_col(2, d2);
                dot_col(3, d3);
                dot_col(4, d4);
                dot_col(5, d5);
                dot_col(6, d6);
                dot_col(7, d7);
            } else {
                if (c_valid > 0) dot_col(0, d0);
                if (c_valid > 1) dot_col(1, d1);
                if (c_valid > 2) dot_col(2, d2);
                if (c_valid > 3) dot_col(3, d3);
                if (c_valid > 4) dot_col(4, d4);
                if (c_valid > 5) dot_col(5, d5);
                if (c_valid > 6) dot_col(6, d6);
                if (c_valid > 7) dot_col(7, d7);
            }

            int32x4_t p01 = vpaddq_s32(d0, d1);
            int32x4_t p23 = vpaddq_s32(d2, d3);
            int32x4_t p45 = vpaddq_s32(d4, d5);
            int32x4_t p67 = vpaddq_s32(d6, d7);
            int32x4_t dots_lo = vpaddq_s32(p01, p23);
            int32x4_t dots_hi = vpaddq_s32(p45, p67);

            float a_scale = a_scales[qb];
            acc_lo = vfmaq_f32(acc_lo, q4_scaled16_dot_to_f32(dots_lo),
                               vmulq_n_f32(bscale_lo, a_scale));
            acc_hi = vfmaq_f32(acc_hi, q4_scaled16_dot_to_f32(dots_hi),
                               vmulq_n_f32(bscale_hi, a_scale));
        };

        if (scale_per_channel) {
            for (int qb = 0; qb < blocks_per_row; qb++) {
                run_qblock(qb, bscale_lo_pc, bscale_hi_pc);
            }
        } else if (scale_mode == W8ScaleMode::PerGroup) {
            int qblocks_per_group = std::max(1, group_size / W8_Q8_BLOCK);
            for (int group = 0; group < groups_per_row; group++) {
                float32x4_t bscale_lo;
                float32x4_t bscale_hi;
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                  bscale_lo, bscale_hi);
                int qb_begin = group * qblocks_per_group;
                int qb_end = std::min(qb_begin + qblocks_per_group, blocks_per_row);
                for (int qb = qb_begin; qb < qb_end; qb++) {
                    run_qblock(qb, bscale_lo, bscale_hi);
                }
            }
        } else {
            for (int qb = 0; qb < blocks_per_row; qb++) {
                float32x4_t bscale_lo;
                float32x4_t bscale_hi;
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, qb,
                                  bscale_lo, bscale_hi);
                run_qblock(qb, bscale_lo, bscale_hi);
            }
        }

        if (full_n_tile) {
            vst1q_f32(C + n, acc_lo);
            vst1q_f32(C + n + 4, acc_hi);
        } else {
            float tmp[4];
            vst1q_f32(tmp, acc_lo);
            for (int c = 0; c < 4 && c < c_valid; c++) C[n + c] = tmp[c];
            vst1q_f32(tmp, acc_hi);
            for (int c = 0; c < 4 && c + 4 < c_valid; c++) C[n + 4 + c] = tmp[c];
        }
    }
}

static void matmul_int4_q8dot_neon_gemv_g128_range(
    const int8_t* qA_even_pre, const int8_t* qA_odd_pre,
    const float* a_scales,
    const Q4B8G128Block* B_g128,
    float* C, int K,
    int n_begin, int n_end)
{
    int g128_per_row = K / 128;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        int c_valid = n_tile_end - n;

        float32x4_t acc_lo = vdupq_n_f32(0.f);
        float32x4_t acc_hi = vdupq_n_f32(0.f);
        const Q4B8G128Block* b_tile =
            B_g128 + (size_t)(n / 8) * g128_per_row;

        for (int g = 0; g < g128_per_row; g++) {
            const Q4B8G128Block& b_group = b_tile[g];
            float32x4_t bscale_lo = vld1q_f32(b_group.scales);
            float32x4_t bscale_hi = vld1q_f32(b_group.scales + 4);

            for (int qgi = 0; qgi < 4; qgi++) {
                int qb = g * 4 + qgi;
                int8x16_t qa_even = vld1q_s8(qA_even_pre + (size_t)qb * 16);
                int8x16_t qa_odd = vld1q_s8(qA_odd_pre + (size_t)qb * 16);

                int32x4_t d0 = vdupq_n_s32(0);
                int32x4_t d1 = vdupq_n_s32(0);
                int32x4_t d2 = vdupq_n_s32(0);
                int32x4_t d3 = vdupq_n_s32(0);
                int32x4_t d4 = vdupq_n_s32(0);
                int32x4_t d5 = vdupq_n_s32(0);
                int32x4_t d6 = vdupq_n_s32(0);
                int32x4_t d7 = vdupq_n_s32(0);

                auto dot_col = [&](int c, int32x4_t& d) {
                    int8x16_t q4_even;
                    int8x16_t q4_odd;
                    load_int4x32_signed_scaled16(b_group.q[qgi][c],
                                                 q4_even, q4_odd);
                    d = q4_q8_dot32(q4_even, q4_odd, qa_even, qa_odd);
                };

                if (c_valid > 0) dot_col(0, d0);
                if (c_valid > 1) dot_col(1, d1);
                if (c_valid > 2) dot_col(2, d2);
                if (c_valid > 3) dot_col(3, d3);
                if (c_valid > 4) dot_col(4, d4);
                if (c_valid > 5) dot_col(5, d5);
                if (c_valid > 6) dot_col(6, d6);
                if (c_valid > 7) dot_col(7, d7);

                int32x4_t p01 = vpaddq_s32(d0, d1);
                int32x4_t p23 = vpaddq_s32(d2, d3);
                int32x4_t p45 = vpaddq_s32(d4, d5);
                int32x4_t p67 = vpaddq_s32(d6, d7);
                int32x4_t dots_lo = vpaddq_s32(p01, p23);
                int32x4_t dots_hi = vpaddq_s32(p45, p67);

                float a_scale = a_scales[qb];
                acc_lo = vfmaq_f32(acc_lo, q4_scaled16_dot_to_f32(dots_lo),
                                   vmulq_n_f32(bscale_lo, a_scale));
                acc_hi = vfmaq_f32(acc_hi, q4_scaled16_dot_to_f32(dots_hi),
                                   vmulq_n_f32(bscale_hi, a_scale));
            }
        }

        float tmp[4];
        vst1q_f32(tmp, acc_lo);
        for (int c = 0; c < 4 && c < c_valid; c++) C[n + c] = tmp[c];
        vst1q_f32(tmp, acc_hi);
        for (int c = 0; c < 4 && c + 4 < c_valid; c++) C[n + 4 + c] = tmp[c];
    }
}

static void matmul_int4_q8dot_neon_4x8_range(
    const int8_t* qA, const float* a_scales,
    const uint8_t* B, const uint8_t* B_repack, const float* scales,
    int group_size, int groups_per_row,
    float* C, int M, int N, int K,
    int K_padded, int K_weight, int ldc,
    int m_begin, int m_end)
{
    (void)M;
    if (group_size <= 0) group_size = K;
    int row_stride = (K_weight + 1) / 2;
    int blocks_per_row = K / W8_Q8_BLOCK;
    constexpr int bytes_per_block = W8_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);
        int r_valid = m_tile_end - m;

        for (int n = 0; n < N; n += 8) {
            int n_tile_end = std::min(n + 8, N);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float acc[4][8] = {};

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int group = w8_scale_group(scale_mode, qb, group_size);
                int byte_off = qb * (W8_Q8_BLOCK / 2);
                const uint8_t* b_repack_block = B_repack
                    ? B_repack + ((size_t)(n / 8) * blocks_per_row + qb) * 8 * bytes_per_block
                    : nullptr;

                int8x16_t q4_even[8];
                int8x16_t q4_odd[8];
                int c_load_end = full_n_tile ? 8 : c_valid;
                for (int c = 0; c < c_load_end; c++) {
                    const uint8_t* src = b_repack_block
                        ? b_repack_block + (size_t)c * bytes_per_block
                        : B + (size_t)(n + c) * row_stride + byte_off;
                    load_int4x32_signed_scaled16(src,
                                                 q4_even[c], q4_odd[c]);
                }

                int32_t dots[4][8] = {};
                for (int r = 0; r < r_valid; r++) {
                    const int8_t* qa = qA + (size_t)(m + r) * K_padded + qb * W8_Q8_BLOCK;
                    int8x16_t qa0 = vld1q_s8(qa);
                    int8x16_t qa1 = vld1q_s8(qa + 16);
                    int8x16_t qa_even = vuzp1q_s8(qa0, qa1);
                    int8x16_t qa_odd = vuzp2q_s8(qa0, qa1);
                    for (int c = 0; c < c_valid; c++) {
                        int32x4_t d = q4_q8_dot32(q4_even[c], q4_odd[c], qa_even, qa_odd);
                        dots[r][c] = vaddvq_s32(d);
                    }
                }

                float32x4_t bscale_lo = bscale_lo_pc;
                float32x4_t bscale_hi = bscale_hi_pc;
                if (!scale_per_channel) {
                    load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                      bscale_lo, bscale_hi);
                }

                for (int r = 0; r < r_valid; r++) {
                    float a_scale = a_scales[(size_t)(m + r) * blocks_per_row + qb];
                    float32x4_t acc_lo = vld1q_f32(acc[r]);
                    float32x4_t acc_hi = vld1q_f32(acc[r] + 4);
                    acc_lo = vfmaq_f32(acc_lo, q4_scaled16_dot_to_f32(vld1q_s32(dots[r])),
                                       vmulq_n_f32(bscale_lo, a_scale));
                    acc_hi = vfmaq_f32(acc_hi, q4_scaled16_dot_to_f32(vld1q_s32(dots[r] + 4)),
                                       vmulq_n_f32(bscale_hi, a_scale));
                    vst1q_f32(acc[r], acc_lo);
                    vst1q_f32(acc[r] + 4, acc_hi);
                }
            }

            for (int r = 0; r < r_valid; r++) {
                float* c_row = C + (m + r) * ldc;
                for (int c = 0; c < c_valid; c++) {
                    c_row[n + c] = acc[r][c];
                }
            }
        }
    }
}

template <bool PackedA4>
static void matmul_int4_q8dot_neon_8x8_range(
    const int8_t* qA, const float* a_scales,
    const uint8_t* B, const uint8_t* B_repack, const float* scales,
    int group_size, int groups_per_row,
    float* C, int M, int N, int K,
    int K_padded, int K_weight, int ldc,
    int m_begin, int m_end,
    int n_begin, int n_end,
    const Q8A4Block* qA4)
{
    (void)M;
    (void)N;
    if (group_size <= 0) group_size = K;
    int row_stride = (K_weight + 1) / 2;
    int blocks_per_row = K / W8_Q8_BLOCK;
    constexpr int bytes_per_block = W8_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 8) {
        int m_tile_end = std::min(m + 8, m_end);
        int r_valid = m_tile_end - m;

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float32x4_t acc_lo[8];
            float32x4_t acc_hi[8];
            for (int r = 0; r < r_valid; r++) {
                acc_lo[r] = vdupq_n_f32(0.f);
                acc_hi[r] = vdupq_n_f32(0.f);
            }

            int qblocks_per_group = (scale_mode == W8ScaleMode::PerGroup)
                ? std::max(1, group_size / W8_Q8_BLOCK)
                : 1;
            int cached_group = -1;
            float32x4_t cached_bscale_lo = vdupq_n_f32(0.f);
            float32x4_t cached_bscale_hi = vdupq_n_f32(0.f);

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int byte_off = qb * (W8_Q8_BLOCK / 2);
                const uint8_t* b_repack_block = B_repack
                    ? B_repack + ((size_t)(n / 8) * blocks_per_row + qb) * 8 * bytes_per_block
                    : nullptr;

                int8x16_t q4_even[8];
                int8x16_t q4_odd[8];
                int c_load_end = full_n_tile ? 8 : c_valid;
                for (int c = 0; c < c_load_end; c++) {
                    const uint8_t* src = b_repack_block
                        ? b_repack_block + (size_t)c * bytes_per_block
                        : B + (size_t)(n + c) * row_stride + byte_off;
                    load_int4x32_signed_scaled16(src, q4_even[c], q4_odd[c]);
                }

                float32x4_t bscale_lo = bscale_lo_pc;
                float32x4_t bscale_hi = bscale_hi_pc;
                if (!scale_per_channel) {
                    int group = (scale_mode == W8ScaleMode::PerGroup)
                        ? qb / qblocks_per_group
                        : qb;
                    if (group != cached_group) {
                        load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                          cached_bscale_lo, cached_bscale_hi);
                        cached_group = group;
                    }
                    bscale_lo = cached_bscale_lo;
                    bscale_hi = cached_bscale_hi;
                }

                for (int r = 0; r < r_valid; r++) {
                    int8x16_t qa_even;
                    int8x16_t qa_odd;
                    float a_scale;
                    if constexpr (PackedA4) {
                        int row = m + r;
                        const Q8A4Block& a_block =
                            qA4[(size_t)(row / 4) * blocks_per_row + qb];
                        int ar = row & 3;
                        qa_even = vld1q_s8(a_block.even[ar]);
                        qa_odd = vld1q_s8(a_block.odd[ar]);
                        a_scale = a_block.scales[ar];
                    } else {
                        const int8_t* qa =
                            qA + (size_t)(m + r) * K_padded + qb * W8_Q8_BLOCK;
                        int8x16_t qa0 = vld1q_s8(qa);
                        int8x16_t qa1 = vld1q_s8(qa + 16);
                        qa_even = vuzp1q_s8(qa0, qa1);
                        qa_odd = vuzp2q_s8(qa0, qa1);
                        a_scale = a_scales[(size_t)(m + r) * blocks_per_row + qb];
                    }
                    int32x4_t d0 = vdupq_n_s32(0);
                    int32x4_t d1 = vdupq_n_s32(0);
                    int32x4_t d2 = vdupq_n_s32(0);
                    int32x4_t d3 = vdupq_n_s32(0);
                    int32x4_t d4 = vdupq_n_s32(0);
                    int32x4_t d5 = vdupq_n_s32(0);
                    int32x4_t d6 = vdupq_n_s32(0);
                    int32x4_t d7 = vdupq_n_s32(0);
                    if (full_n_tile) {
                        d0 = q4_q8_dot32(q4_even[0], q4_odd[0], qa_even, qa_odd);
                        d1 = q4_q8_dot32(q4_even[1], q4_odd[1], qa_even, qa_odd);
                        d2 = q4_q8_dot32(q4_even[2], q4_odd[2], qa_even, qa_odd);
                        d3 = q4_q8_dot32(q4_even[3], q4_odd[3], qa_even, qa_odd);
                        d4 = q4_q8_dot32(q4_even[4], q4_odd[4], qa_even, qa_odd);
                        d5 = q4_q8_dot32(q4_even[5], q4_odd[5], qa_even, qa_odd);
                        d6 = q4_q8_dot32(q4_even[6], q4_odd[6], qa_even, qa_odd);
                        d7 = q4_q8_dot32(q4_even[7], q4_odd[7], qa_even, qa_odd);
                    } else {
                        if (c_valid > 0) d0 = q4_q8_dot32(q4_even[0], q4_odd[0], qa_even, qa_odd);
                        if (c_valid > 1) d1 = q4_q8_dot32(q4_even[1], q4_odd[1], qa_even, qa_odd);
                        if (c_valid > 2) d2 = q4_q8_dot32(q4_even[2], q4_odd[2], qa_even, qa_odd);
                        if (c_valid > 3) d3 = q4_q8_dot32(q4_even[3], q4_odd[3], qa_even, qa_odd);
                        if (c_valid > 4) d4 = q4_q8_dot32(q4_even[4], q4_odd[4], qa_even, qa_odd);
                        if (c_valid > 5) d5 = q4_q8_dot32(q4_even[5], q4_odd[5], qa_even, qa_odd);
                        if (c_valid > 6) d6 = q4_q8_dot32(q4_even[6], q4_odd[6], qa_even, qa_odd);
                        if (c_valid > 7) d7 = q4_q8_dot32(q4_even[7], q4_odd[7], qa_even, qa_odd);
                    }

                    int32x4_t p01 = vpaddq_s32(d0, d1);
                    int32x4_t p23 = vpaddq_s32(d2, d3);
                    int32x4_t p45 = vpaddq_s32(d4, d5);
                    int32x4_t p67 = vpaddq_s32(d6, d7);
                    int32x4_t dots_lo = vpaddq_s32(p01, p23);
                    int32x4_t dots_hi = vpaddq_s32(p45, p67);

                    acc_lo[r] = vfmaq_f32(acc_lo[r], q4_scaled16_dot_to_f32(dots_lo),
                                          vmulq_n_f32(bscale_lo, a_scale));
                    acc_hi[r] = vfmaq_f32(acc_hi[r], q4_scaled16_dot_to_f32(dots_hi),
                                          vmulq_n_f32(bscale_hi, a_scale));
                }
            }

            for (int r = 0; r < r_valid; r++) {
                float* c_row = C + (m + r) * ldc;
                if (full_n_tile) {
                    vst1q_f32(c_row + n, acc_lo[r]);
                    vst1q_f32(c_row + n + 4, acc_hi[r]);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, acc_lo[r]);
                    for (int c = 0; c < 4 && c < c_valid; c++) c_row[n + c] = tmp[c];
                    vst1q_f32(tmp, acc_hi[r]);
                    for (int c = 0; c < 4 && c + 4 < c_valid; c++) c_row[n + 4 + c] = tmp[c];
                }
            }
        }
    }
}

template <bool PackedA4>
static void matmul_int4_q8dot_neon_8x8_g128packed_range(
    const int8_t* qA, const float* a_scales,
    const Q4B8G128Block* B_g128,
    float* C, int M, int N, int K,
    int K_padded, int ldc,
    int m_begin, int m_end,
    int n_begin, int n_end,
    const Q8A4Block* qA4)
{
    (void)M;
    (void)N;
    int blocks_per_row = K / W8_Q8_BLOCK;
    int g128_per_row = K / 128;

    for (int m = m_begin; m < m_end; m += 8) {
        int m_tile_end = std::min(m + 8, m_end);
        int r_valid = m_tile_end - m;

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);

            float32x4_t acc_lo[8];
            float32x4_t acc_hi[8];
            for (int r = 0; r < r_valid; r++) {
                acc_lo[r] = vdupq_n_f32(0.f);
                acc_hi[r] = vdupq_n_f32(0.f);
            }

            const Q4B8G128Block* b_tile =
                B_g128 + (size_t)(n / 8) * g128_per_row;
            for (int g = 0; g < g128_per_row; g++) {
                const Q4B8G128Block& b_group = b_tile[g];
                float32x4_t bscale_lo = vld1q_f32(b_group.scales);
                float32x4_t bscale_hi = vld1q_f32(b_group.scales + 4);

                for (int qgi = 0; qgi < 4; qgi++) {
                    int qb = g * 4 + qgi;
                    int8x16_t q4_even[8];
                    int8x16_t q4_odd[8];
                    for (int c = 0; c < 8; c++) {
                        load_int4x32_signed_scaled16(b_group.q[qgi][c],
                                                     q4_even[c], q4_odd[c]);
                    }

                    for (int r = 0; r < r_valid; r++) {
                        int8x16_t qa_even;
                        int8x16_t qa_odd;
                        float a_scale;
                        if constexpr (PackedA4) {
                            int row = m + r;
                            const Q8A4Block& a_block =
                                qA4[(size_t)(row / 4) * blocks_per_row + qb];
                            int ar = row & 3;
                            qa_even = vld1q_s8(a_block.even[ar]);
                            qa_odd = vld1q_s8(a_block.odd[ar]);
                            a_scale = a_block.scales[ar];
                        } else {
                            const int8_t* qa =
                                qA + (size_t)(m + r) * K_padded + qb * W8_Q8_BLOCK;
                            int8x16_t qa0 = vld1q_s8(qa);
                            int8x16_t qa1 = vld1q_s8(qa + 16);
                            qa_even = vuzp1q_s8(qa0, qa1);
                            qa_odd = vuzp2q_s8(qa0, qa1);
                            a_scale = a_scales[(size_t)(m + r) * blocks_per_row + qb];
                        }

                        int32x4_t d0 = q4_q8_dot32(q4_even[0], q4_odd[0],
                                                    qa_even, qa_odd);
                        int32x4_t d1 = q4_q8_dot32(q4_even[1], q4_odd[1],
                                                    qa_even, qa_odd);
                        int32x4_t d2 = q4_q8_dot32(q4_even[2], q4_odd[2],
                                                    qa_even, qa_odd);
                        int32x4_t d3 = q4_q8_dot32(q4_even[3], q4_odd[3],
                                                    qa_even, qa_odd);
                        int32x4_t d4 = q4_q8_dot32(q4_even[4], q4_odd[4],
                                                    qa_even, qa_odd);
                        int32x4_t d5 = q4_q8_dot32(q4_even[5], q4_odd[5],
                                                    qa_even, qa_odd);
                        int32x4_t d6 = q4_q8_dot32(q4_even[6], q4_odd[6],
                                                    qa_even, qa_odd);
                        int32x4_t d7 = q4_q8_dot32(q4_even[7], q4_odd[7],
                                                    qa_even, qa_odd);

                        int32x4_t p01 = vpaddq_s32(d0, d1);
                        int32x4_t p23 = vpaddq_s32(d2, d3);
                        int32x4_t p45 = vpaddq_s32(d4, d5);
                        int32x4_t p67 = vpaddq_s32(d6, d7);
                        int32x4_t dots_lo = vpaddq_s32(p01, p23);
                        int32x4_t dots_hi = vpaddq_s32(p45, p67);

                        acc_lo[r] = vfmaq_f32(acc_lo[r], q4_scaled16_dot_to_f32(dots_lo),
                                              vmulq_n_f32(bscale_lo, a_scale));
                        acc_hi[r] = vfmaq_f32(acc_hi[r], q4_scaled16_dot_to_f32(dots_hi),
                                              vmulq_n_f32(bscale_hi, a_scale));
                    }
                }
            }

            for (int r = 0; r < r_valid; r++) {
                float* c_row = C + (m + r) * ldc;
                if (full_n_tile) {
                    vst1q_f32(c_row + n, acc_lo[r]);
                    vst1q_f32(c_row + n + 4, acc_hi[r]);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, acc_lo[r]);
                    for (int c = 0; c < 4 && c < c_valid; c++) c_row[n + c] = tmp[c];
                    vst1q_f32(tmp, acc_hi[r]);
                    for (int c = 0; c < 4 && c + 4 < c_valid; c++) c_row[n + 4 + c] = tmp[c];
                }
            }
        }
    }
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
template <bool PackedA4>
static void matmul_int4_q8dot_neon_4x8_repacked_i8mm_range(
    const int8_t* qA, const float* a_scales,
    const uint8_t* B_repack, const float* scales,
    int group_size, int groups_per_row,
    float* C, int M, int N, int K,
    int K_padded, int ldc, int m_begin, int m_end,
    const Q8A4Block* qA4)
{
    (void)M;
    if (group_size <= 0) group_size = K;
    int blocks_per_row = K / W8_Q8_BLOCK;
    constexpr int bytes_per_block = W8_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);

        for (int n = 0; n < N; n += 8) {
            int n_tile_end = std::min(n + 8, N);
            int c_valid = n_tile_end - n;
            const uint8_t* b_tile = B_repack +
                (size_t)(n / 8) * blocks_per_row * 8 * bytes_per_block;

            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float32x4_t c0_lo = vdupq_n_f32(0.f);
            float32x4_t c0_hi = vdupq_n_f32(0.f);
            float32x4_t c1_lo = vdupq_n_f32(0.f);
            float32x4_t c1_hi = vdupq_n_f32(0.f);
            float32x4_t c2_lo = vdupq_n_f32(0.f);
            float32x4_t c2_hi = vdupq_n_f32(0.f);
            float32x4_t c3_lo = vdupq_n_f32(0.f);
            float32x4_t c3_hi = vdupq_n_f32(0.f);

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int group = w8_scale_group(scale_mode, qb, group_size);
                const uint8_t* b_block = b_tile + (size_t)qb * 8 * bytes_per_block;

                int8x16_t b_even[8];
                int8x16_t b_odd[8];
                for (int c = 0; c < 8; c++) {
                    if (c < c_valid) {
                        load_int4x32_signed_scaled16(b_block + (size_t)c * bytes_per_block,
                                                     b_even[c], b_odd[c]);
                    } else {
                        b_even[c] = vdupq_n_s8(0);
                        b_odd[c] = vdupq_n_s8(0);
                    }
                }

                int32x4_t acc01_01 = vdupq_n_s32(0);
                int32x4_t acc01_23 = vdupq_n_s32(0);
                int32x4_t acc23_01 = vdupq_n_s32(0);
                int32x4_t acc23_23 = vdupq_n_s32(0);
                int32x4_t acc01_45 = vdupq_n_s32(0);
                int32x4_t acc01_67 = vdupq_n_s32(0);
                int32x4_t acc23_45 = vdupq_n_s32(0);
                int32x4_t acc23_67 = vdupq_n_s32(0);

                auto load_even_odd = [&](int row, int8x16_t& even, int8x16_t& odd,
                                          float& a_scale) {
                    int row_load = (row < m_tile_end) ? row : m;
                    if constexpr (PackedA4) {
                        const Q8A4Block& a_block =
                            qA4[(size_t)(row_load / 4) * blocks_per_row + qb];
                        int ar = row_load & 3;
                        even = vld1q_s8(a_block.even[ar]);
                        odd = vld1q_s8(a_block.odd[ar]);
                        a_scale = a_block.scales[ar];
                    } else {
                        const int8_t* qa =
                            qA + (size_t)row_load * K_padded + qb * W8_Q8_BLOCK;
                        int8x16_t qa0v = vld1q_s8(qa);
                        int8x16_t qa1v = vld1q_s8(qa + 16);
                        even = vuzp1q_s8(qa0v, qa1v);
                        odd = vuzp2q_s8(qa0v, qa1v);
                        a_scale = a_scales[(size_t)row_load * blocks_per_row + qb];
                    }
                };

                int8x16_t a0_even, a0_odd, a1_even, a1_odd;
                int8x16_t a2_even, a2_odd, a3_even, a3_odd;
                float a_scale0, a_scale1, a_scale2, a_scale3;
                load_even_odd(m + 0, a0_even, a0_odd, a_scale0);
                load_even_odd(m + 1, a1_even, a1_odd, a_scale1);
                load_even_odd(m + 2, a2_even, a2_odd, a_scale2);
                load_even_odd(m + 3, a3_even, a3_odd, a_scale3);

                auto run_half = [&](bool high_half, bool odd_lane) {
                    auto half8 = [&](int8x16_t v) -> int8x8_t {
                        return high_half ? vget_high_s8(v) : vget_low_s8(v);
                    };
                    int8x16_t a01 = vcombine_s8(
                        half8(odd_lane ? a0_odd : a0_even),
                        half8(odd_lane ? a1_odd : a1_even));
                    int8x16_t a23 = vcombine_s8(
                        half8(odd_lane ? a2_odd : a2_even),
                        half8(odd_lane ? a3_odd : a3_even));

                    const int8x16_t* b = odd_lane ? b_odd : b_even;
                    int8x16_t b01 = vcombine_s8(half8(b[0]), half8(b[1]));
                    int8x16_t b23 = vcombine_s8(half8(b[2]), half8(b[3]));
                    int8x16_t b45 = vcombine_s8(half8(b[4]), half8(b[5]));
                    int8x16_t b67 = vcombine_s8(half8(b[6]), half8(b[7]));

                    acc01_01 = vmmlaq_s32(acc01_01, a01, b01);
                    acc01_23 = vmmlaq_s32(acc01_23, a01, b23);
                    acc23_01 = vmmlaq_s32(acc23_01, a23, b01);
                    acc23_23 = vmmlaq_s32(acc23_23, a23, b23);
                    acc01_45 = vmmlaq_s32(acc01_45, a01, b45);
                    acc01_67 = vmmlaq_s32(acc01_67, a01, b67);
                    acc23_45 = vmmlaq_s32(acc23_45, a23, b45);
                    acc23_67 = vmmlaq_s32(acc23_67, a23, b67);
                };

                run_half(false, false);
                run_half(true, false);
                run_half(false, true);
                run_half(true, true);

                int32x4_t row0_lo = vcombine_s32(vget_low_s32(acc01_01), vget_low_s32(acc01_23));
                int32x4_t row0_hi = vcombine_s32(vget_low_s32(acc01_45), vget_low_s32(acc01_67));
                int32x4_t row1_lo = vcombine_s32(vget_high_s32(acc01_01), vget_high_s32(acc01_23));
                int32x4_t row1_hi = vcombine_s32(vget_high_s32(acc01_45), vget_high_s32(acc01_67));
                int32x4_t row2_lo = vcombine_s32(vget_low_s32(acc23_01), vget_low_s32(acc23_23));
                int32x4_t row2_hi = vcombine_s32(vget_low_s32(acc23_45), vget_low_s32(acc23_67));
                int32x4_t row3_lo = vcombine_s32(vget_high_s32(acc23_01), vget_high_s32(acc23_23));
                int32x4_t row3_hi = vcombine_s32(vget_high_s32(acc23_45), vget_high_s32(acc23_67));

                float32x4_t bs0 = bscale_lo_pc;
                float32x4_t bs1 = bscale_hi_pc;
                if (!scale_per_channel) {
                    load_w8_b_scales8(scales, n, c_valid, groups_per_row, group, bs0, bs1);
                }

                auto add_row = [&](int row, int32x4_t lo, int32x4_t hi,
                                   float32x4_t& dst_lo, float32x4_t& dst_hi,
                                   float a_scale) {
                    if (row >= m_tile_end) return;
                    dst_lo = vfmaq_f32(dst_lo, q4_scaled16_dot_to_f32(lo),
                                       vmulq_n_f32(bs0, a_scale));
                    dst_hi = vfmaq_f32(dst_hi, q4_scaled16_dot_to_f32(hi),
                                       vmulq_n_f32(bs1, a_scale));
                };
                add_row(m + 0, row0_lo, row0_hi, c0_lo, c0_hi, a_scale0);
                add_row(m + 1, row1_lo, row1_hi, c1_lo, c1_hi, a_scale1);
                add_row(m + 2, row2_lo, row2_hi, c2_lo, c2_hi, a_scale2);
                add_row(m + 3, row3_lo, row3_hi, c3_lo, c3_hi, a_scale3);
            }

            auto store_row = [&](int row, float32x4_t lo, float32x4_t hi) {
                if (row >= m_tile_end) return;
                float* c_row = C + row * ldc;
                float tmp[4];
                vst1q_f32(tmp, lo);
                for (int c = 0; c < 4 && n + c < n_tile_end; c++) c_row[n + c] = tmp[c];
                vst1q_f32(tmp, hi);
                for (int c = 0; c < 4 && n + 4 + c < n_tile_end; c++) c_row[n + 4 + c] = tmp[c];
            };
            store_row(m + 0, c0_lo, c0_hi);
            store_row(m + 1, c1_lo, c1_hi);
            store_row(m + 2, c2_lo, c2_hi);
            store_row(m + 3, c3_lo, c3_hi);
        }
    }
}

#endif
#endif

static void matmul_int4_scalar_range(const float* A, const uint8_t* B, const float* scales,
                                     int group_size, int groups_per_row,
                                     float* C, int M, int N, int K,
                                     int lda, int K_weight, int ldc,
                                     int m_begin, int m_end,
                                     int n_begin, int n_end) {
    (void)M;
    if (group_size <= 0) group_size = K;
    if (groups_per_row <= 0) groups_per_row = 1;
    int row_stride = (K_weight + 1) / 2;

    for (int m = m_begin; m < m_end; m++) {
        float* c_row = C + m * ldc;
        for (int n = n_begin; n < n_end; n++) {
            const uint8_t* b_row = B + (size_t)n * row_stride;
            const float* s_row = scales + n * groups_per_row;
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                uint8_t byte = b_row[k >> 1];
                int8_t q = unpack_int4_signed(byte, (k & 1) != 0);
                int g = k / group_size;
                sum += A[k + m * lda] * ((float)q * s_row[g]);
            }
            c_row[n] = sum;
        }
    }
}

#if HAS_NEON && defined(__aarch64__)
static inline void quantize_q8_block32_neon(const float* src,
                                            float& scale,
                                            int8x16_t& q_lo,
                                            int8x16_t& q_hi) {
    float32x4_t v0 = vld1q_f32(src + 0);
    float32x4_t v1 = vld1q_f32(src + 4);
    float32x4_t v2 = vld1q_f32(src + 8);
    float32x4_t v3 = vld1q_f32(src + 12);
    float32x4_t v4 = vld1q_f32(src + 16);
    float32x4_t v5 = vld1q_f32(src + 20);
    float32x4_t v6 = vld1q_f32(src + 24);
    float32x4_t v7 = vld1q_f32(src + 28);
    float32x4_t m0 = vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1));
    float32x4_t m1 = vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3));
    float32x4_t m2 = vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5));
    float32x4_t m3 = vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7));
    float amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));

    scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
    float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;

    int32x4_t q0 = vcvtnq_s32_f32(vmulq_n_f32(v0, inv_scale));
    int32x4_t q1 = vcvtnq_s32_f32(vmulq_n_f32(v1, inv_scale));
    int32x4_t q2 = vcvtnq_s32_f32(vmulq_n_f32(v2, inv_scale));
    int32x4_t q3 = vcvtnq_s32_f32(vmulq_n_f32(v3, inv_scale));
    int32x4_t q4 = vcvtnq_s32_f32(vmulq_n_f32(v4, inv_scale));
    int32x4_t q5 = vcvtnq_s32_f32(vmulq_n_f32(v5, inv_scale));
    int32x4_t q6 = vcvtnq_s32_f32(vmulq_n_f32(v6, inv_scale));
    int32x4_t q7 = vcvtnq_s32_f32(vmulq_n_f32(v7, inv_scale));
    int32x4_t qmin = vdupq_n_s32(-127);
    int32x4_t qmax = vdupq_n_s32(127);
    q0 = vmaxq_s32(qmin, vminq_s32(qmax, q0));
    q1 = vmaxq_s32(qmin, vminq_s32(qmax, q1));
    q2 = vmaxq_s32(qmin, vminq_s32(qmax, q2));
    q3 = vmaxq_s32(qmin, vminq_s32(qmax, q3));
    q4 = vmaxq_s32(qmin, vminq_s32(qmax, q4));
    q5 = vmaxq_s32(qmin, vminq_s32(qmax, q5));
    q6 = vmaxq_s32(qmin, vminq_s32(qmax, q6));
    q7 = vmaxq_s32(qmin, vminq_s32(qmax, q7));

    int16x8_t q01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
    int16x8_t q23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
    int16x8_t q45 = vcombine_s16(vqmovn_s32(q4), vqmovn_s32(q5));
    int16x8_t q67 = vcombine_s16(vqmovn_s32(q6), vqmovn_s32(q7));
    q_lo = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));
    q_hi = vcombine_s8(vqmovn_s16(q45), vqmovn_s16(q67));
}
#endif

static void quantize_a_q8_blocks(const float* A, int M, int K, int lda,
                                 int K_storage,
                                 std::vector<int8_t>& qA,
                                 std::vector<float>& a_scales) {
    auto t0 = std::chrono::steady_clock::now();
    if (K_storage < K) K_storage = K;
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;
    if (K_storage == K) {
        qA.resize((size_t)M * K_storage);
    } else {
        qA.assign((size_t)M * K_storage, 0);
    }
    a_scales.resize((size_t)M * blocks_per_row);
    static const bool force_scalar_qa = env_flag_enabled("MOLLM_Q8_GEMM_SCALAR_QA");

    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * lda;
        int8_t* qa_row = qA.data() + (size_t)m * K_storage;
        float* s_row = a_scales.data() + (size_t)m * blocks_per_row;
        for (int qb = 0; qb < blocks_per_row; qb++) {
            int k_begin = qb * W8_Q8_BLOCK;
            int k_end = std::min(k_begin + W8_Q8_BLOCK, K);
            float amax = 0.f;
#if HAS_NEON && defined(__aarch64__)
            if (!force_scalar_qa && k_end - k_begin == W8_Q8_BLOCK) {
                float scale = 1.f;
                int8x16_t q_lo;
                int8x16_t q_hi;
                quantize_q8_block32_neon(a_row + k_begin, scale, q_lo, q_hi);
                s_row[qb] = scale;
                vst1q_s8(qa_row + k_begin, q_lo);
                vst1q_s8(qa_row + k_begin + 16, q_hi);
                continue;
            }
#endif
            for (int k = k_begin; k < k_end; k++) {
                amax = std::max(amax, std::fabs(a_row[k]));
            }
            float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
            float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;
            s_row[qb] = scale;
            for (int k = k_begin; k < k_end; k++) {
                int q = (int)std::nearbyint(a_row[k] * inv_scale);
                q = std::max(-127, std::min(127, q));
                qa_row[k] = (int8_t)q;
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::lock_guard<std::mutex> lk(g_prof_mtx);
    g_q8_quant_a_ms += ms;
    g_q8_quant_a_calls++;
}

#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
static void quantize_a_q8_blocks_a4(const float* A, int M, int K, int lda,
                                    std::vector<Q8A4Block>& qA4) {
    auto t0 = std::chrono::steady_clock::now();
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;
    int m_tiles = (M + 3) / 4;
    qA4.resize((size_t)m_tiles * blocks_per_row);
    static const bool force_scalar_qa = env_flag_enabled("MOLLM_Q8_GEMM_SCALAR_QA");

    for (int mt = 0; mt < m_tiles; mt++) {
        for (int qb = 0; qb < blocks_per_row; qb++) {
            int k_begin = qb * W8_Q8_BLOCK;
            int k_end = std::min(k_begin + W8_Q8_BLOCK, K);
            Q8A4Block& block = qA4[(size_t)mt * blocks_per_row + qb];
            for (int ar = 0; ar < 4; ar++) {
                int m = mt * 4 + ar;
                if (m >= M) continue;

                const float* a_row = A + (size_t)m * lda;
#if HAS_NEON && defined(__aarch64__)
                if (!force_scalar_qa && k_end - k_begin == W8_Q8_BLOCK) {
                    float scale = 1.f;
                    int8x16_t q_lo;
                    int8x16_t q_hi;
                    quantize_q8_block32_neon(a_row + k_begin, scale, q_lo, q_hi);
                    block.scales[ar] = scale;
                    vst1q_s8(block.even[ar], vuzp1q_s8(q_lo, q_hi));
                    vst1q_s8(block.odd[ar], vuzp2q_s8(q_lo, q_hi));
                    continue;
                }
#endif
                float amax = 0.f;
                for (int k = k_begin; k < k_end; k++) {
                    amax = std::max(amax, std::fabs(a_row[k]));
                }
                float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
                float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;
                block.scales[ar] = scale;
                for (int i = 0; i < 16; i++) {
                    int k0 = k_begin + i * 2;
                    int k1 = k0 + 1;
                    int q0 = (k0 < k_end) ? (int)std::nearbyint(a_row[k0] * inv_scale) : 0;
                    int q1 = (k1 < k_end) ? (int)std::nearbyint(a_row[k1] * inv_scale) : 0;
                    q0 = std::max(-127, std::min(127, q0));
                    q1 = std::max(-127, std::min(127, q1));
                    block.even[ar][i] = (int8_t)q0;
                    block.odd[ar][i] = (int8_t)q1;
                }
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::lock_guard<std::mutex> lk(g_prof_mtx);
    g_q8_quant_a_ms += ms;
    g_q8_quant_a_calls++;
}
#endif

static void quantize_a_q8_blocks_even_odd(const float* A, int K,
                                          std::vector<int8_t>& qA_even,
                                          std::vector<int8_t>& qA_odd,
                                          std::vector<float>& a_scales) {
    auto t0 = std::chrono::steady_clock::now();
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;
    qA_even.resize((size_t)blocks_per_row * 16);
    qA_odd.resize((size_t)blocks_per_row * 16);
    a_scales.resize((size_t)blocks_per_row);
    static const bool force_scalar_qa = env_flag_enabled("MOLLM_W4_GEMV_SCALAR_QA");

    for (int qb = 0; qb < blocks_per_row; qb++) {
        int k_begin = qb * W8_Q8_BLOCK;
        int k_end = std::min(k_begin + W8_Q8_BLOCK, K);
        float amax = 0.f;
#if HAS_NEON && defined(__aarch64__)
        if (!force_scalar_qa && k_end - k_begin == W8_Q8_BLOCK) {
            float scale = 1.f;
            int8x16_t q_lo;
            int8x16_t q_hi;
            quantize_q8_block32_neon(A + k_begin, scale, q_lo, q_hi);
            a_scales[qb] = scale;
            vst1q_s8(qA_even.data() + (size_t)qb * 16, vuzp1q_s8(q_lo, q_hi));
            vst1q_s8(qA_odd.data() + (size_t)qb * 16, vuzp2q_s8(q_lo, q_hi));
            continue;
        }
#endif
        for (int k = k_begin; k < k_end; k++) {
            amax = std::max(amax, std::fabs(A[k]));
        }
        float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
        float inv_scale = (amax > 0.f) ? (127.f / amax) : 0.f;
        a_scales[qb] = scale;

        int8_t* even = qA_even.data() + (size_t)qb * 16;
        int8_t* odd = qA_odd.data() + (size_t)qb * 16;
        for (int i = 0; i < 16; i++) {
            int k0 = k_begin + i * 2;
            int k1 = k0 + 1;
            int q0 = (k0 < k_end) ? (int)std::nearbyint(A[k0] * inv_scale) : 0;
            int q1 = (k1 < k_end) ? (int)std::nearbyint(A[k1] * inv_scale) : 0;
            q0 = std::max(-127, std::min(127, q0));
            q1 = std::max(-127, std::min(127, q1));
            even[i] = (int8_t)q0;
            odd[i] = (int8_t)q1;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::lock_guard<std::mutex> lk(g_prof_mtx);
    g_q8_quant_a_ms += ms;
    g_q8_quant_a_calls++;
}

struct Q4GemvScratch {
    std::vector<int8_t> qA_even;
    std::vector<int8_t> qA_odd;
    std::vector<float> a_scales;
};

static inline int8_t load_b_int8_value(const int8_t* B, int n, int k,
                                       int K_weight, bool b_interleaved) {
    return b_interleaved
        ? B[(n & ~7) * K_weight + k * 8 + (n & 7)]
        : B[n * K_weight + k];
}

static void matmul_int8_q8dot_scalar_range(
    const int8_t* qA, const float* a_scales,
    const int8_t* B, const float* scales,
    int group_size, int groups_per_row,
    float* C, int M, int N, int K,
    int K_weight, int ldc,
    int m_begin, int m_end,
    int n_begin, int n_end,
    bool b_interleaved)
{
    (void)M;
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;
    for (int m = m_begin; m < m_end; m++) {
        const int8_t* qa_row = qA + (size_t)m * K;
        const float* as_row = a_scales + (size_t)m * blocks_per_row;
        float* c_row = C + m * ldc;
        for (int n = n_begin; n < n_end; n++) {
            const float* bs_row = scales + n * groups_per_row;
            float sum = 0.f;
            int k0 = 0;
            while (k0 < K) {
                int q_block = k0 / W8_Q8_BLOCK;
                int group = k0 / group_size;
                int k_end = std::min({K, (q_block + 1) * W8_Q8_BLOCK,
                                      (group + 1) * group_size});
                int32_t dot = 0;
                for (int k = k0; k < k_end; k++) {
                    dot += (int32_t)qa_row[k] *
                           (int32_t)load_b_int8_value(B, n, k, K_weight, b_interleaved);
                }
                sum += (float)dot * as_row[q_block] * bs_row[group];
                k0 = k_end;
            }
            c_row[n] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// NEON matmul — TILE_M=8, TILE_N=8, FP16 storage + FP32 accumulate
//
// B is stored as float16 ([N, K] or [K, N] repacked).
// Each inner iteration loads 8 float16 values (one vld1q_f16 or gather),
// converts to float32 via vcvt_f32_f16, then FMA into float32 accumulators.
// This halves the memory bandwidth for B compared to FP32 storage.
//
// For gather (row-major) layout, we load via uint16x4_t + vget_lane + vcvt
// to avoid stack temporaries.
// ---------------------------------------------------------------------------
#if HAS_NEON

// ---------------------------------------------------------------------------
// B interleaved packing — transform B[N,K] to tile-of-8 transposed layout.
//
// For each N-tile of 8 rows, transpose so that for fixed k,
// B_packed[tile_base + k*8 + 0..7] are 8 consecutive FP16 values.
// This enables vld1q_f16 contiguous load instead of strided gather.
//
// Full-matrix version: pack entire B [N, K] → interleaved [N/8, K, 8].
// Caller owns the returned buffer (must delete[]).
// ---------------------------------------------------------------------------
__fp16* pack_b_interleaved_full(const __fp16* B_original, int N, int K, int K_weight) {
    int N_padded = ((N + 7) / 8) * 8;  // round up to multiple of 8
    __fp16* dst = new __fp16[(size_t)N_padded * K];
    for (int n_tile = 0; n_tile < N_padded; n_tile += 8) {
        int tile_valid = std::min(8, N - n_tile);
        if (tile_valid < 0) tile_valid = 0;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < tile_valid; j++) {
                dst[n_tile * K + k * 8 + j] = B_original[(n_tile + j) * K_weight + k];
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
__fp16* pack_a_interleaved_full(const float* A_original, int M, int K, int lda) {
    auto t0 = std::chrono::steady_clock::now();
    int M_padded = ((M + 7) / 8) * 8;  // round up to multiple of 8
    __fp16* dst = new __fp16[(size_t)M_padded * K];
    for (int m_tile = 0; m_tile < M_padded; m_tile += 8) {
        int tile_valid = std::min(8, M - m_tile);
        if (tile_valid < 0) tile_valid = 0;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < tile_valid; j++) {
                dst[m_tile * K + k * 8 + j] = (__fp16)A_original[k + (m_tile + j) * lda];
            }
            for (int j = tile_valid; j < 8; j++) {
                dst[m_tile * K + k * 8 + j] = (__fp16)0.f;
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    { std::lock_guard<std::mutex> lk(g_prof_mtx); g_pack_a_ms += ms; g_pack_a_calls++; }
    return dst;
}

static void matmul_fp16_neon_8x8_range(const float* A, const __fp16* B, float* C,
                                       int M, int N, int K,
                                       int lda, int K_weight, int ldc,
                                       int m_begin, int m_end) {
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    // FP16: twice as many elements fit in cache, so double K_BLOCK.
    const int K_BLOCK_FP16 = K_BLOCK * 2;

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK_FP16) {
        int k_end = std::min(k_outer + K_BLOCK_FP16, K);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                float32x4_t c[8][2];
                bool first_block = (k_outer == 0);
                if (first_block) {
                    for (int r = 0; r < 8; r++) {
                        c[r][0] = vdupq_n_f32(0.f);
                        c[r][1] = vdupq_n_f32(0.f);
                    }
                } else {
                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            c[r][0] = vld1q_f32(&C[row * ldc + n]);
                            if (n + 4 < n_end) {
                                c[r][1] = vld1q_f32(&C[row * ldc + n + 4]);
                            } else {
                                c[r][1] = vdupq_n_f32(0.f);
                            }
                        } else {
                            c[r][0] = vdupq_n_f32(0.f);
                            c[r][1] = vdupq_n_f32(0.f);
                        }
                    }
                }

                for (int k = k_outer; k < k_end; k++) {
                    // Load 8 FP16 B values, convert to FP32.
                    float32x4_t b0, b1;
                    {
                        __fp16 tmp[4] = {(__fp16)0.f, (__fp16)0.f, (__fp16)0.f, (__fp16)0.f};
                        for (int j = 0; j < 4 && n + j < n_end; j++) {
                            tmp[j] = B[(n + j) * K_weight + k];
                        }
                        b0 = vcvt_f32_f16(vld1_f16(tmp));
                    }
                    {
                        __fp16 tmp[4] = {(__fp16)0.f, (__fp16)0.f, (__fp16)0.f, (__fp16)0.f};
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++) {
                            tmp[j] = B[(n + 4 + j) * K_weight + k];
                        }
                        b1 = vcvt_f32_f16(vld1_f16(tmp));
                    }

                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            float a_val = A[k + row * lda];
                            c[r][0] = vfmaq_n_f32(c[r][0], b0, a_val);
                            c[r][1] = vfmaq_n_f32(c[r][1], b1, a_val);
                        }
                    }
                }

                // Write back
                for (int r = 0; r < 8; r++) {
                    int row = m + r;
                    if (row < m_tile_end && row < m_global_end) {
                        float tmp[4];
                        vst1q_f32(tmp, c[r][0]);
                        for (int j = 0; j < 4 && n + j < n_end; j++) C[row * ldc + n + j] = tmp[j];
                        vst1q_f32(tmp, c[r][1]);
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++) C[row * ldc + n + 4 + j] = tmp[j];
                    }
                }
            }
        }
    }
}

// ---- FP16 packed kernel (B pre-packed interleaved, contiguous load) ----
//
// B_packed: load-time interleaved layout [N/8, K, 8].
// For fixed k, B_packed[(n & ~7) * K + k * 8 + 0..7] are 8 consecutive FP16.
// K-blocking loop is internal (cache blocking only, no packing).
static void matmul_fp16_neon_8x8_range_packed(
    const float* A, const __fp16* B_packed, float* C,
    int M, int N, int K,
    int lda, int ldc,
    int m_begin, int m_end)
{
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2;  // FP16: 2x cache density

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK_FP16) {
        int k_end = std::min(k_outer + K_BLOCK_FP16, K);
        bool first_block = (k_outer == 0);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                float32x4_t c[8][2];
                if (first_block) {
                    for (int r = 0; r < 8; r++) {
                        c[r][0] = vdupq_n_f32(0.f);
                        c[r][1] = vdupq_n_f32(0.f);
                    }
                } else {
                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            c[r][0] = vld1q_f32(&C[row * ldc + n]);
                            if (n + 4 < n_end) {
                                c[r][1] = vld1q_f32(&C[row * ldc + n + 4]);
                            } else {
                                c[r][1] = vdupq_n_f32(0.f);
                            }
                        } else {
                            c[r][0] = vdupq_n_f32(0.f);
                            c[r][1] = vdupq_n_f32(0.f);
                        }
                    }
                }

                for (int k = k_outer; k < k_end; k++) {
                    // Load 8 contiguous FP16 values from pre-packed B
                    float16x8_t b_vec = vld1q_f16(&B_packed[(n & ~7) * K + k * 8]);
                    float32x4_t b0 = vcvt_f32_f16(vget_low_f16(b_vec));
                    float32x4_t b1 = vcvt_f32_f16(vget_high_f16(b_vec));

                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            float a_val = A[k + row * lda];
                            c[r][0] = vfmaq_n_f32(c[r][0], b0, a_val);
                            c[r][1] = vfmaq_n_f32(c[r][1], b1, a_val);
                        }
                    }
                }

                // Write back
                for (int r = 0; r < 8; r++) {
                    int row = m + r;
                    if (row < m_tile_end && row < m_global_end) {
                        float tmp[4];
                        vst1q_f32(tmp, c[r][0]);
                        for (int j = 0; j < 4 && n + j < n_end; j++) C[row * ldc + n + j] = tmp[j];
                        vst1q_f32(tmp, c[r][1]);
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++) C[row * ldc + n + 4 + j] = tmp[j];
                    }
                }
            }
        }
    }
}

// ---- FP16 GEMV kernel (M=1, B pre-packed interleaved) ----
//
// C[n] = sum_k A[k] * B[n, k]
//
// A:        FP32 [K]              (single row, M=1)
// B_packed: interleaved FP16 [N/8, K, 8]
//           For fixed k, B_packed[(n & ~7) * K + k * 8 + 0..7] = 8 N values
// C:        FP32 [N] output
//
// n_begin must be 8-aligned (parallel_for grain_size is aligned). n_end may
// be any value; the trailing partial tile is masked on store.
//
// Two accumulate modes:
//   FP32 acc (default fallback): 1 vld + 2 vcvt + 2 vfmaq_n_f32 = 5 instr/K-step
//     FMA-bound at ~48 GF/s (4 threads). Higher precision.
//   FP16 acc (use_fp16_accumulate=true): 1 vld + 1 vfmaq_n_f16 = 2 instr/K-step
//     Bandwidth-bound at ~60 GF/s (4 threads). K_BLOCK store/reload for precision.

static void matmul_fp16_neon_gemv_range(
    const float* A, const __fp16* B_packed, float* C,
    int K, int n_begin, int n_end)
{
    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);

        float32x4_t acc0 = vdupq_n_f32(0.f);  // n+0..n+3
        float32x4_t acc1 = vdupq_n_f32(0.f);  // n+4..n+7

        const __fp16* b_tile = &B_packed[(n & ~7) * K];
        for (int k = 0; k < K; k++) {
            float a_val = A[k];
            float16x8_t b_vec = vld1q_f16(b_tile + k * 8);
            acc0 = vfmaq_n_f32(acc0, vcvt_f32_f16(vget_low_f16(b_vec)), a_val);
            acc1 = vfmaq_n_f32(acc1, vcvt_f32_f16(vget_high_f16(b_vec)), a_val);
        }

        // Store partial tile (handles n_end not 8-aligned).
        float tmp[4];
        vst1q_f32(tmp, acc0);
        for (int j = 0; j < 4 && n + j < n_tile_end; j++) C[n + j] = tmp[j];
        vst1q_f32(tmp, acc1);
        for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++) C[n + 4 + j] = tmp[j];
    }
}

// FP16 accumulate variant: vfmaq_n_f16 does 8-lane FP16×FP16→FP16 FMA with
// scalar broadcast. K_BLOCK store/reload (FP16→FP32→FP16) between blocks
// limits within-block FP16 accumulation range for precision.
//
// 8-way K-unroll with 8 independent FP16 accumulators to fully hide FMA
// latency. Apple M5 Pro FP16 FMA latency = 2 cycles, throughput = 2/cycle.
// With N independent acc chains, CPI = max(1, 2/N) cycles/K-step.
//   N=1 (single-acc):  CPI=2  → latency-bound
//   N=2 (2-way unroll): CPI=1  → at FMA throughput ceiling
//   N=8 (8-way unroll): CPI=1  → same throughput, but lower loop overhead
//                                 per FMA, better ILP for issue stage.
// Inspired by ggml's SVE GEMV which uses 4-acc × 8-way K-unroll.
static void matmul_fp16_neon_gemv_range_fp16acc(
    const float* A, const __fp16* B_packed, float* C,
    int K, int n_begin, int n_end,
    Activation act = Activation::NONE,
    int act_n_begin = 0, int act_n_len = -1)
{
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2;  // FP16: 2x cache density

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        const __fp16* b_tile = &B_packed[(n & ~7) * K];

        // 8 independent FP16 acc chains to fully hide FMA latency.
        // Each K step writes to one acc, rotating through acc[0..7].
        // After 8 K steps, every acc has exactly 1 FMA → fully independent.
        float16x8_t acc[8];

        int k_outer = 0;
        for (; k_outer < K; k_outer += K_BLOCK_FP16) {
            int k_end = std::min(k_outer + K_BLOCK_FP16, K);

            // Initialize acc for this K-block.
            if (k_outer == 0) {
                for (int i = 0; i < 8; i++) acc[i] = vdupq_n_f16((__fp16)0.f);
            } else {
                // Reload partial sums from C (FP32 → FP16) — split into 8 chains.
                // acc[0] carries the merged partial from prior block.
                float tmp[8];
                for (int j = 0; j < 8; j++) {
                    tmp[j] = (n + j < n_tile_end) ? C[n + j] : 0.f;
                }
                acc[0] = vcombine_f16(
                    vcvt_f16_f32(vld1q_f32(tmp)),
                    vcvt_f16_f32(vld1q_f32(tmp + 4)));
                for (int i = 1; i < 8; i++) acc[i] = vdupq_n_f16((__fp16)0.f);
            }

            // 8-way K-unroll: 8 independent FMA chains.
            // Each K step: load A[k], load B[k*8], FMA into acc[k%8].
            // 8 acc chains → FMA latency (2 cycles) fully hidden.
            int k = k_outer;
            int k_unrolled_end = k_outer + ((k_end - k_outer) & ~7);
            for (; k < k_unrolled_end; k += 8) {
                __fp16 a0 = (__fp16)A[k + 0];
                __fp16 a1 = (__fp16)A[k + 1];
                __fp16 a2 = (__fp16)A[k + 2];
                __fp16 a3 = (__fp16)A[k + 3];
                __fp16 a4 = (__fp16)A[k + 4];
                __fp16 a5 = (__fp16)A[k + 5];
                __fp16 a6 = (__fp16)A[k + 6];
                __fp16 a7 = (__fp16)A[k + 7];
                float16x8_t b0 = vld1q_f16(b_tile + (k + 0) * 8);
                float16x8_t b1 = vld1q_f16(b_tile + (k + 1) * 8);
                float16x8_t b2 = vld1q_f16(b_tile + (k + 2) * 8);
                float16x8_t b3 = vld1q_f16(b_tile + (k + 3) * 8);
                float16x8_t b4 = vld1q_f16(b_tile + (k + 4) * 8);
                float16x8_t b5 = vld1q_f16(b_tile + (k + 5) * 8);
                float16x8_t b6 = vld1q_f16(b_tile + (k + 6) * 8);
                float16x8_t b7 = vld1q_f16(b_tile + (k + 7) * 8);
                acc[0] = vfmaq_n_f16(acc[0], b0, a0);
                acc[1] = vfmaq_n_f16(acc[1], b1, a1);
                acc[2] = vfmaq_n_f16(acc[2], b2, a2);
                acc[3] = vfmaq_n_f16(acc[3], b3, a3);
                acc[4] = vfmaq_n_f16(acc[4], b4, a4);
                acc[5] = vfmaq_n_f16(acc[5], b5, a5);
                acc[6] = vfmaq_n_f16(acc[6], b6, a6);
                acc[7] = vfmaq_n_f16(acc[7], b7, a7);
            }
            // Tail (K % 8 != 0 within block): fall back to rotating acc[0..7].
            for (; k < k_end; k++) {
                __fp16 a0 = (__fp16)A[k];
                float16x8_t b0 = vld1q_f16(b_tile + k * 8);
                int idx = (k - k_outer) & 7;
                acc[idx] = vfmaq_n_f16(acc[idx], b0, a0);
            }

            // Merge 8 acc chains into acc[0] (pairwise reduction).
            acc[0] = vaddq_f16(acc[0], acc[1]);
            acc[2] = vaddq_f16(acc[2], acc[3]);
            acc[4] = vaddq_f16(acc[4], acc[5]);
            acc[6] = vaddq_f16(acc[6], acc[7]);
            acc[0] = vaddq_f16(acc[0], acc[2]);
            acc[4] = vaddq_f16(acc[4], acc[6]);
            acc[0] = vaddq_f16(acc[0], acc[4]);

            // Store partial sums to C as FP32 (reloaded if more blocks remain).
            // On the last K-block, also apply fused activation in-place.
            // Single-store path: convert FP16 acc → FP32, apply act if needed,
            // store. Avoids a separate activation pass after the K-block loop.
            bool is_last_block = (k_outer + K_BLOCK_FP16 >= K);
            bool apply_act = (act != Activation::NONE) && is_last_block;

            float32x4_t lo = vcvt_f32_f16(vget_low_f16(acc[0]));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(acc[0]));
            if (apply_act) {
                // Apply activation per-column.
                // For typical 8-wide N-tile, columns are either all-active or
                // all-inactive (since act_n_len is multiple of 8 for SwiGLU gate).
                // Fast path: if all 4 in [act_n_begin, act_n_begin+act_n_len),
                // vectorize; else scalar per-column.
                bool lo_active = (act_n_len < 0) ||
                                 (activation_applies_at(n + 0, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 1, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 2, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 3, act_n_begin, act_n_len));
                if (lo_active) {
                    lo = apply_activation_f32_neon(lo, act);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, lo);
                    for (int jj = 0; jj < 4; jj++) {
                        if (activation_applies_at(n + jj, act_n_begin, act_n_len)) {
                            tmp[jj] = apply_activation_scalar(tmp[jj], act);
                        }
                    }
                    lo = vld1q_f32(tmp);
                }
                bool hi_active = (act_n_len < 0) ||
                                 (activation_applies_at(n + 4, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 5, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 6, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 7, act_n_begin, act_n_len));
                if (hi_active) {
                    hi = apply_activation_f32_neon(hi, act);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, hi);
                    for (int jj = 0; jj < 4; jj++) {
                        if (activation_applies_at(n + 4 + jj, act_n_begin, act_n_len)) {
                            tmp[jj] = apply_activation_scalar(tmp[jj], act);
                        }
                    }
                    hi = vld1q_f32(tmp);
                }
            }

            float tmp[4];
            vst1q_f32(tmp, lo);
            for (int j = 0; j < 4 && n + j < n_tile_end; j++) C[n + j] = tmp[j];
            vst1q_f32(tmp, hi);
            for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++) C[n + 4 + j] = tmp[j];
        }

        // Note: fused activation is applied inline above on the last K-block
        // store, so no separate post-loop pass is needed here.
    }
}

static void matmul_int8_neon_gemv_range(
    const float* A, const int8_t* B_packed, const float* scales,
    int group_size, int groups_per_row,
    float* C, int K, int n_begin, int n_end)
{
    if (group_size <= 0) group_size = K;
    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        const int8_t* b_tile = &B_packed[(n & ~7) * K];

        float32x4_t acc0 = vdupq_n_f32(0.f);
        float32x4_t acc1 = vdupq_n_f32(0.f);

        for (int g = 0; g < groups_per_row; g++) {
            int k_begin = g * group_size;
            int k_end = std::min(k_begin + group_size, K);

            float scale_tmp[8];
            for (int j = 0; j < 8; j++) {
                scale_tmp[j] = (n + j < n_tile_end)
                    ? scales[(n + j) * groups_per_row + g]
                    : 0.f;
            }
            float32x4_t scale0 = vld1q_f32(scale_tmp);
            float32x4_t scale1 = vld1q_f32(scale_tmp + 4);

            for (int k = k_begin; k < k_end; k++) {
                int8x8_t b8 = vld1_s8(b_tile + k * 8);
                int16x8_t b16 = vmovl_s8(b8);
                float32x4_t b0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
                float32x4_t b1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));
                float a = A[k];
                acc0 = vfmaq_f32(acc0, b0, vmulq_n_f32(scale0, a));
                acc1 = vfmaq_f32(acc1, b1, vmulq_n_f32(scale1, a));
            }
        }

        float tmp[4];
        vst1q_f32(tmp, acc0);
        for (int j = 0; j < 4 && n + j < n_tile_end; j++) C[n + j] = tmp[j];
        vst1q_f32(tmp, acc1);
        for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++) C[n + 4 + j] = tmp[j];
    }
}

static void matmul_int8_neon_4x8_range(
    const float* A, const int8_t* B_packed, const float* scales,
    int group_size, int groups_per_row,
    float* C, int M, int N, int K,
    int lda, int ldc, int m_begin, int m_end)
{
    if (group_size <= 0) group_size = K;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);

        for (int n = 0; n < N; n += 8) {
            int n_tile_end = std::min(n + 8, N);
            const int8_t* b_tile = &B_packed[(n & ~7) * K];

            float32x4_t c0_lo = vdupq_n_f32(0.f);
            float32x4_t c0_hi = vdupq_n_f32(0.f);
            float32x4_t c1_lo = vdupq_n_f32(0.f);
            float32x4_t c1_hi = vdupq_n_f32(0.f);
            float32x4_t c2_lo = vdupq_n_f32(0.f);
            float32x4_t c2_hi = vdupq_n_f32(0.f);
            float32x4_t c3_lo = vdupq_n_f32(0.f);
            float32x4_t c3_hi = vdupq_n_f32(0.f);

            for (int g = 0; g < groups_per_row; g++) {
                int k_begin = g * group_size;
                int k_end = std::min(k_begin + group_size, K);

                float scale_tmp[8];
                for (int j = 0; j < 8; j++) {
                    scale_tmp[j] = (n + j < n_tile_end)
                        ? scales[(n + j) * groups_per_row + g]
                        : 0.f;
                }
                float32x4_t scale0 = vld1q_f32(scale_tmp);
                float32x4_t scale1 = vld1q_f32(scale_tmp + 4);

                for (int k = k_begin; k < k_end; k++) {
                    int8x8_t b8 = vld1_s8(b_tile + k * 8);
                    int16x8_t b16 = vmovl_s8(b8);
                    float32x4_t b0 = vmulq_f32(
                        vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16))), scale0);
                    float32x4_t b1 = vmulq_f32(
                        vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16))), scale1);

                    float a0 = A[k + (m + 0) * lda];
                    c0_lo = vfmaq_n_f32(c0_lo, b0, a0);
                    c0_hi = vfmaq_n_f32(c0_hi, b1, a0);
                    if (m + 1 < m_tile_end) {
                        float a1 = A[k + (m + 1) * lda];
                        c1_lo = vfmaq_n_f32(c1_lo, b0, a1);
                        c1_hi = vfmaq_n_f32(c1_hi, b1, a1);
                    }
                    if (m + 2 < m_tile_end) {
                        float a2 = A[k + (m + 2) * lda];
                        c2_lo = vfmaq_n_f32(c2_lo, b0, a2);
                        c2_hi = vfmaq_n_f32(c2_hi, b1, a2);
                    }
                    if (m + 3 < m_tile_end) {
                        float a3 = A[k + (m + 3) * lda];
                        c3_lo = vfmaq_n_f32(c3_lo, b0, a3);
                        c3_hi = vfmaq_n_f32(c3_hi, b1, a3);
                    }
                }
            }

            auto store_row = [&](int row, float32x4_t lo, float32x4_t hi) {
                if (row >= m_tile_end) return;
                float* c_row = C + row * ldc;
                float tmp[4];
                vst1q_f32(tmp, lo);
                for (int j = 0; j < 4 && n + j < n_tile_end; j++) c_row[n + j] = tmp[j];
                vst1q_f32(tmp, hi);
                for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++) c_row[n + 4 + j] = tmp[j];
            };
            store_row(m + 0, c0_lo, c0_hi);
            store_row(m + 1, c1_lo, c1_hi);
            store_row(m + 2, c2_lo, c2_hi);
            store_row(m + 3, c3_lo, c3_hi);
        }
    }
}

static void matmul_int8_q8dot_neon_gemv_range(
    const int8_t* qA, const float* a_scales,
    const int8_t* B_packed, const float* scales,
    int group_size, int groups_per_row,
    float* C, int K, int n_begin, int n_end)
{
    if (group_size <= 0) group_size = K;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        const int8_t* b_tile = &B_packed[(n & ~7) * K];
        float32x4_t acc0 = vdupq_n_f32(0.f);
        float32x4_t acc1 = vdupq_n_f32(0.f);

        int k0 = 0;
        while (k0 < K) {
            int q_block = k0 / W8_Q8_BLOCK;
            int group = k0 / group_size;
            int k_end = std::min({K, (q_block + 1) * W8_Q8_BLOCK,
                                  (group + 1) * group_size});
            int32x4_t dot0 = vdupq_n_s32(0);
            int32x4_t dot1 = vdupq_n_s32(0);

            for (int k = k0; k < k_end; k++) {
                int16x8_t b16 = vmovl_s8(vld1_s8(b_tile + k * 8));
                int16x8_t prod = vmulq_n_s16(b16, (int16_t)qA[k]);
                dot0 = vaddw_s16(dot0, vget_low_s16(prod));
                dot1 = vaddw_s16(dot1, vget_high_s16(prod));
            }

            float scale_tmp[8];
            float a_scale = a_scales[q_block];
            for (int j = 0; j < 8; j++) {
                scale_tmp[j] = (n + j < n_tile_end)
                    ? a_scale * scales[(n + j) * groups_per_row + group]
                    : 0.f;
            }
            acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(dot0), vld1q_f32(scale_tmp));
            acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(dot1), vld1q_f32(scale_tmp + 4));
            k0 = k_end;
        }

        float tmp[4];
        vst1q_f32(tmp, acc0);
        for (int j = 0; j < 4 && n + j < n_tile_end; j++) C[n + j] = tmp[j];
        vst1q_f32(tmp, acc1);
        for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++) C[n + 4 + j] = tmp[j];
    }
}

static void matmul_int8_q8dot_neon_4x8_range(
    const int8_t* qA, const float* a_scales,
    const int8_t* B_packed, const float* scales,
    int group_size, int groups_per_row,
    float* C, int M, int N, int K,
    int ldc, int m_begin, int m_end)
{
    if (group_size <= 0) group_size = K;
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);

        for (int n = 0; n < N; n += 8) {
            int n_tile_end = std::min(n + 8, N);
            const int8_t* b_tile = &B_packed[(n & ~7) * K];

            float32x4_t c0_lo = vdupq_n_f32(0.f);
            float32x4_t c0_hi = vdupq_n_f32(0.f);
            float32x4_t c1_lo = vdupq_n_f32(0.f);
            float32x4_t c1_hi = vdupq_n_f32(0.f);
            float32x4_t c2_lo = vdupq_n_f32(0.f);
            float32x4_t c2_hi = vdupq_n_f32(0.f);
            float32x4_t c3_lo = vdupq_n_f32(0.f);
            float32x4_t c3_hi = vdupq_n_f32(0.f);

            int k0 = 0;
            while (k0 < K) {
                int q_block = k0 / W8_Q8_BLOCK;
                int group = k0 / group_size;
                int k_end = std::min({K, (q_block + 1) * W8_Q8_BLOCK,
                                      (group + 1) * group_size});

                int32x4_t d0_lo = vdupq_n_s32(0);
                int32x4_t d0_hi = vdupq_n_s32(0);
                int32x4_t d1_lo = vdupq_n_s32(0);
                int32x4_t d1_hi = vdupq_n_s32(0);
                int32x4_t d2_lo = vdupq_n_s32(0);
                int32x4_t d2_hi = vdupq_n_s32(0);
                int32x4_t d3_lo = vdupq_n_s32(0);
                int32x4_t d3_hi = vdupq_n_s32(0);

                const int8_t* qa0 = qA + (size_t)(m + 0) * K;
                const int8_t* qa1 = (m + 1 < m_tile_end) ? qA + (size_t)(m + 1) * K : qa0;
                const int8_t* qa2 = (m + 2 < m_tile_end) ? qA + (size_t)(m + 2) * K : qa0;
                const int8_t* qa3 = (m + 3 < m_tile_end) ? qA + (size_t)(m + 3) * K : qa0;

                for (int k = k0; k < k_end; k++) {
                    int16x8_t b16 = vmovl_s8(vld1_s8(b_tile + k * 8));
                    int16x4_t b_lo = vget_low_s16(b16);
                    int16x4_t b_hi = vget_high_s16(b16);

                    int16x8_t p0 = vmulq_n_s16(b16, (int16_t)qa0[k]);
                    d0_lo = vaddw_s16(d0_lo, vget_low_s16(p0));
                    d0_hi = vaddw_s16(d0_hi, vget_high_s16(p0));
                    if (m + 1 < m_tile_end) {
                        d1_lo = vaddw_s16(d1_lo, vmul_n_s16(b_lo, (int16_t)qa1[k]));
                        d1_hi = vaddw_s16(d1_hi, vmul_n_s16(b_hi, (int16_t)qa1[k]));
                    }
                    if (m + 2 < m_tile_end) {
                        d2_lo = vaddw_s16(d2_lo, vmul_n_s16(b_lo, (int16_t)qa2[k]));
                        d2_hi = vaddw_s16(d2_hi, vmul_n_s16(b_hi, (int16_t)qa2[k]));
                    }
                    if (m + 3 < m_tile_end) {
                        d3_lo = vaddw_s16(d3_lo, vmul_n_s16(b_lo, (int16_t)qa3[k]));
                        d3_hi = vaddw_s16(d3_hi, vmul_n_s16(b_hi, (int16_t)qa3[k]));
                    }
                }

                float b_scale_tmp[8];
                for (int j = 0; j < 8; j++) {
                    b_scale_tmp[j] = (n + j < n_tile_end)
                        ? scales[(n + j) * groups_per_row + group]
                        : 0.f;
                }
                float32x4_t bs0 = vld1q_f32(b_scale_tmp);
                float32x4_t bs1 = vld1q_f32(b_scale_tmp + 4);

                auto add_row = [&](int row, int32x4_t lo, int32x4_t hi,
                                   float32x4_t& c_lo, float32x4_t& c_hi) {
                    float a_scale = a_scales[(size_t)row * blocks_per_row + q_block];
                    c_lo = vfmaq_f32(c_lo, vcvtq_f32_s32(lo), vmulq_n_f32(bs0, a_scale));
                    c_hi = vfmaq_f32(c_hi, vcvtq_f32_s32(hi), vmulq_n_f32(bs1, a_scale));
                };
                add_row(m + 0, d0_lo, d0_hi, c0_lo, c0_hi);
                if (m + 1 < m_tile_end) add_row(m + 1, d1_lo, d1_hi, c1_lo, c1_hi);
                if (m + 2 < m_tile_end) add_row(m + 2, d2_lo, d2_hi, c2_lo, c2_hi);
                if (m + 3 < m_tile_end) add_row(m + 3, d3_lo, d3_hi, c3_lo, c3_hi);

                k0 = k_end;
            }

            auto store_row = [&](int row, float32x4_t lo, float32x4_t hi) {
                if (row >= m_tile_end) return;
                float* c_row = C + row * ldc;
                float tmp[4];
                vst1q_f32(tmp, lo);
                for (int j = 0; j < 4 && n + j < n_tile_end; j++) c_row[n + j] = tmp[j];
                vst1q_f32(tmp, hi);
                for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++) c_row[n + 4 + j] = tmp[j];
            };
            store_row(m + 0, c0_lo, c0_hi);
            store_row(m + 1, c1_lo, c1_hi);
            store_row(m + 2, c2_lo, c2_hi);
            store_row(m + 3, c3_lo, c3_hi);
        }
    }
}

#if defined(__ARM_FEATURE_DOTPROD)
static void matmul_int8_q8dot_neon_gemv_repacked_range(
    const int8_t* qA, const float* a_scales,
    const int8_t* B_repack, const float* scales,
    int group_size, int groups_per_row,
    float* C, int K, int K_padded, int n_begin, int n_end)
{
    (void)K_padded;
    if (group_size <= 0) group_size = K;
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        int c_valid = n_tile_end - n;
        const int8_t* b_tile = B_repack + (size_t)(n / 8) * blocks_per_row * 8 * W8_Q8_BLOCK;
        float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
        float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
        if (scale_per_channel) {
            load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                              bscale_lo_pc, bscale_hi_pc);
        }

        float32x4_t acc_lo = vdupq_n_f32(0.f);
        float32x4_t acc_hi = vdupq_n_f32(0.f);

        for (int qb = 0; qb < blocks_per_row; qb++) {
            int group = w8_scale_group(scale_mode, qb, group_size);
            const int8_t* b_block = b_tile + (size_t)qb * 8 * W8_Q8_BLOCK;
            int32x4_t d0 = vdupq_n_s32(0);
            int32x4_t d1 = vdupq_n_s32(0);
            int32x4_t d2 = vdupq_n_s32(0);
            int32x4_t d3 = vdupq_n_s32(0);
            int32x4_t d4 = vdupq_n_s32(0);
            int32x4_t d5 = vdupq_n_s32(0);
            int32x4_t d6 = vdupq_n_s32(0);
            int32x4_t d7 = vdupq_n_s32(0);

            for (int half = 0; half < W8_Q8_BLOCK; half += 16) {
                const int8_t* qa = qA + (size_t)qb * W8_Q8_BLOCK + half;
                int8x16_t a_vec = vld1q_s8(qa);
                d0 = vdotq_s32(d0, vld1q_s8(b_block + 0 * W8_Q8_BLOCK + half), a_vec);
                d1 = vdotq_s32(d1, vld1q_s8(b_block + 1 * W8_Q8_BLOCK + half), a_vec);
                d2 = vdotq_s32(d2, vld1q_s8(b_block + 2 * W8_Q8_BLOCK + half), a_vec);
                d3 = vdotq_s32(d3, vld1q_s8(b_block + 3 * W8_Q8_BLOCK + half), a_vec);
                d4 = vdotq_s32(d4, vld1q_s8(b_block + 4 * W8_Q8_BLOCK + half), a_vec);
                d5 = vdotq_s32(d5, vld1q_s8(b_block + 5 * W8_Q8_BLOCK + half), a_vec);
                d6 = vdotq_s32(d6, vld1q_s8(b_block + 6 * W8_Q8_BLOCK + half), a_vec);
                d7 = vdotq_s32(d7, vld1q_s8(b_block + 7 * W8_Q8_BLOCK + half), a_vec);
            }

            int32x4_t p01 = vpaddq_s32(d0, d1);
            int32x4_t p23 = vpaddq_s32(d2, d3);
            int32x4_t p45 = vpaddq_s32(d4, d5);
            int32x4_t p67 = vpaddq_s32(d6, d7);
            int32x4_t dots_lo = vpaddq_s32(p01, p23);
            int32x4_t dots_hi = vpaddq_s32(p45, p67);

            float a_scale = a_scales[qb];
            float32x4_t bscale_lo = bscale_lo_pc;
            float32x4_t bscale_hi = bscale_hi_pc;
            if (!scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                  bscale_lo, bscale_hi);
            }
            float32x4_t scale_lo = vmulq_n_f32(bscale_lo, a_scale);
            float32x4_t scale_hi = vmulq_n_f32(bscale_hi, a_scale);

            acc_lo = vfmaq_f32(acc_lo, vcvtq_f32_s32(dots_lo), scale_lo);
            acc_hi = vfmaq_f32(acc_hi, vcvtq_f32_s32(dots_hi), scale_hi);
        }

        float tmp[4];
        vst1q_f32(tmp, acc_lo);
        for (int c = 0; c < 4 && c < c_valid; c++) C[n + c] = tmp[c];
        vst1q_f32(tmp, acc_hi);
        for (int c = 0; c < 4 && c + 4 < c_valid; c++) C[n + 4 + c] = tmp[c];
    }
}

static void matmul_int8_q8dot_neon_4x8_repacked_range(
    const int8_t* qA, const float* a_scales,
    const int8_t* B_repack, const float* scales,
    int group_size, int groups_per_row,
    float* C, int M, int N, int K,
    int K_padded, int ldc, int m_begin, int m_end,
    int n_begin, int n_end)
{
    (void)N;
    if (group_size <= 0) group_size = K;
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);
        int r_valid = m_tile_end - m;

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            const int8_t* b_tile = B_repack + (size_t)(n / 8) * blocks_per_row * 8 * W8_Q8_BLOCK;
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float acc[4][8] = {};

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int group = w8_scale_group(scale_mode, qb, group_size);
                const int8_t* b_block = b_tile + (size_t)qb * 8 * W8_Q8_BLOCK;
                int32_t dots[4][8] = {};

                for (int half = 0; half < W8_Q8_BLOCK; half += 16) {
                    int8x16_t b_vec[8];
                    for (int c = 0; c < 8; c++) {
                        b_vec[c] = vld1q_s8(b_block + c * W8_Q8_BLOCK + half);
                    }
                    for (int r = 0; r < r_valid; r++) {
                        const int8_t* qa = qA + (size_t)(m + r) * K_padded + qb * W8_Q8_BLOCK + half;
                        int8x16_t a_vec = vld1q_s8(qa);
                        for (int c = 0; c < 8; c++) {
                            int32x4_t dot = vdotq_s32(vdupq_n_s32(0), b_vec[c], a_vec);
                            dots[r][c] += vaddvq_s32(dot);
                        }
                    }
                }

                float32x4_t bscale_lo = bscale_lo_pc;
                float32x4_t bscale_hi = bscale_hi_pc;
                if (!scale_per_channel) {
                    load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                      bscale_lo, bscale_hi);
                }

                for (int r = 0; r < r_valid; r++) {
                    float a_scale = a_scales[(size_t)(m + r) * blocks_per_row + qb];
                    float32x4_t acc_lo = vld1q_f32(acc[r]);
                    float32x4_t acc_hi = vld1q_f32(acc[r] + 4);
                    acc_lo = vfmaq_f32(acc_lo, vcvtq_f32_s32(vld1q_s32(dots[r])),
                                       vmulq_n_f32(bscale_lo, a_scale));
                    acc_hi = vfmaq_f32(acc_hi, vcvtq_f32_s32(vld1q_s32(dots[r] + 4)),
                                       vmulq_n_f32(bscale_hi, a_scale));
                    vst1q_f32(acc[r], acc_lo);
                    vst1q_f32(acc[r] + 4, acc_hi);
                }
            }

            for (int r = 0; r < r_valid; r++) {
                float* c_row = C + (m + r) * ldc;
                for (int c = 0; c < c_valid; c++) {
                    c_row[n + c] = acc[r][c];
                }
            }
        }
    }
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
static void matmul_int8_q8dot_neon_4x8_repacked_i8mm_range(
    const int8_t* qA, const float* a_scales,
    const int8_t* B_repack, const float* scales,
    int group_size, int groups_per_row,
    float* C, int M, int N, int K,
    int K_padded, int ldc, int m_begin, int m_end,
    int n_begin, int n_end)
{
    (void)M;
    (void)N;
    if (group_size <= 0) group_size = K;
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);
        bool full_m_tile = (m + 4 <= m_end);

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);
            const int8_t* b_tile = B_repack + (size_t)(n / 8) * blocks_per_row * 8 * W8_Q8_BLOCK;
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float32x4_t c0_lo = vdupq_n_f32(0.f);
            float32x4_t c0_hi = vdupq_n_f32(0.f);
            float32x4_t c1_lo = vdupq_n_f32(0.f);
            float32x4_t c1_hi = vdupq_n_f32(0.f);
            float32x4_t c2_lo = vdupq_n_f32(0.f);
            float32x4_t c2_hi = vdupq_n_f32(0.f);
            float32x4_t c3_lo = vdupq_n_f32(0.f);
            float32x4_t c3_hi = vdupq_n_f32(0.f);

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int group = w8_scale_group(scale_mode, qb, group_size);
                const int8_t* b_block = b_tile + (size_t)qb * 8 * W8_Q8_BLOCK;

                int32x4_t acc01_01 = vdupq_n_s32(0);
                int32x4_t acc01_23 = vdupq_n_s32(0);
                int32x4_t acc23_01 = vdupq_n_s32(0);
                int32x4_t acc23_23 = vdupq_n_s32(0);
                int32x4_t acc01_45 = vdupq_n_s32(0);
                int32x4_t acc01_67 = vdupq_n_s32(0);
                int32x4_t acc23_45 = vdupq_n_s32(0);
                int32x4_t acc23_67 = vdupq_n_s32(0);

                const int8_t* qa0 = qA + (size_t)(m + 0) * K_padded + qb * W8_Q8_BLOCK;
                const int8_t* qa1 = (m + 1 < m_tile_end)
                    ? qA + (size_t)(m + 1) * K_padded + qb * W8_Q8_BLOCK
                    : qa0;
                const int8_t* qa2 = (m + 2 < m_tile_end)
                    ? qA + (size_t)(m + 2) * K_padded + qb * W8_Q8_BLOCK
                    : qa0;
                const int8_t* qa3 = (m + 3 < m_tile_end)
                    ? qA + (size_t)(m + 3) * K_padded + qb * W8_Q8_BLOCK
                    : qa0;

                for (int off = 0; off < W8_Q8_BLOCK; off += 8) {
                    int8x16_t a01 = vcombine_s8(vld1_s8(qa0 + off), vld1_s8(qa1 + off));
                    int8x16_t a23 = vcombine_s8(vld1_s8(qa2 + off), vld1_s8(qa3 + off));

                    int8x16_t b01 = vcombine_s8(vld1_s8(b_block + 0 * W8_Q8_BLOCK + off),
                                                 vld1_s8(b_block + 1 * W8_Q8_BLOCK + off));
                    int8x16_t b23 = vcombine_s8(vld1_s8(b_block + 2 * W8_Q8_BLOCK + off),
                                                 vld1_s8(b_block + 3 * W8_Q8_BLOCK + off));
                    int8x16_t b45 = vcombine_s8(vld1_s8(b_block + 4 * W8_Q8_BLOCK + off),
                                                 vld1_s8(b_block + 5 * W8_Q8_BLOCK + off));
                    int8x16_t b67 = vcombine_s8(vld1_s8(b_block + 6 * W8_Q8_BLOCK + off),
                                                 vld1_s8(b_block + 7 * W8_Q8_BLOCK + off));

                    acc01_01 = vmmlaq_s32(acc01_01, a01, b01);
                    acc01_23 = vmmlaq_s32(acc01_23, a01, b23);
                    acc23_01 = vmmlaq_s32(acc23_01, a23, b01);
                    acc23_23 = vmmlaq_s32(acc23_23, a23, b23);
                    acc01_45 = vmmlaq_s32(acc01_45, a01, b45);
                    acc01_67 = vmmlaq_s32(acc01_67, a01, b67);
                    acc23_45 = vmmlaq_s32(acc23_45, a23, b45);
                    acc23_67 = vmmlaq_s32(acc23_67, a23, b67);
                }

                int32x4_t row0_lo = vcombine_s32(vget_low_s32(acc01_01), vget_low_s32(acc01_23));
                int32x4_t row0_hi = vcombine_s32(vget_low_s32(acc01_45), vget_low_s32(acc01_67));
                int32x4_t row1_lo = vcombine_s32(vget_high_s32(acc01_01), vget_high_s32(acc01_23));
                int32x4_t row1_hi = vcombine_s32(vget_high_s32(acc01_45), vget_high_s32(acc01_67));
                int32x4_t row2_lo = vcombine_s32(vget_low_s32(acc23_01), vget_low_s32(acc23_23));
                int32x4_t row2_hi = vcombine_s32(vget_low_s32(acc23_45), vget_low_s32(acc23_67));
                int32x4_t row3_lo = vcombine_s32(vget_high_s32(acc23_01), vget_high_s32(acc23_23));
                int32x4_t row3_hi = vcombine_s32(vget_high_s32(acc23_45), vget_high_s32(acc23_67));

                float32x4_t bs0 = bscale_lo_pc;
                float32x4_t bs1 = bscale_hi_pc;
                if (!scale_per_channel) {
                    load_w8_b_scales8(scales, n, c_valid, groups_per_row, group, bs0, bs1);
                }

                if (full_m_tile) {
                    float a0 = a_scales[(size_t)(m + 0) * blocks_per_row + qb];
                    float a1 = a_scales[(size_t)(m + 1) * blocks_per_row + qb];
                    float a2 = a_scales[(size_t)(m + 2) * blocks_per_row + qb];
                    float a3 = a_scales[(size_t)(m + 3) * blocks_per_row + qb];
                    c0_lo = vfmaq_f32(c0_lo, vcvtq_f32_s32(row0_lo), vmulq_n_f32(bs0, a0));
                    c0_hi = vfmaq_f32(c0_hi, vcvtq_f32_s32(row0_hi), vmulq_n_f32(bs1, a0));
                    c1_lo = vfmaq_f32(c1_lo, vcvtq_f32_s32(row1_lo), vmulq_n_f32(bs0, a1));
                    c1_hi = vfmaq_f32(c1_hi, vcvtq_f32_s32(row1_hi), vmulq_n_f32(bs1, a1));
                    c2_lo = vfmaq_f32(c2_lo, vcvtq_f32_s32(row2_lo), vmulq_n_f32(bs0, a2));
                    c2_hi = vfmaq_f32(c2_hi, vcvtq_f32_s32(row2_hi), vmulq_n_f32(bs1, a2));
                    c3_lo = vfmaq_f32(c3_lo, vcvtq_f32_s32(row3_lo), vmulq_n_f32(bs0, a3));
                    c3_hi = vfmaq_f32(c3_hi, vcvtq_f32_s32(row3_hi), vmulq_n_f32(bs1, a3));
                } else {
                    auto add_row = [&](int row, int32x4_t lo, int32x4_t hi,
                                       float32x4_t& dst_lo, float32x4_t& dst_hi) {
                        if (row >= m_tile_end) return;
                        float a_scale = a_scales[(size_t)row * blocks_per_row + qb];
                        dst_lo = vfmaq_f32(dst_lo, vcvtq_f32_s32(lo), vmulq_n_f32(bs0, a_scale));
                        dst_hi = vfmaq_f32(dst_hi, vcvtq_f32_s32(hi), vmulq_n_f32(bs1, a_scale));
                    };
                    add_row(m + 0, row0_lo, row0_hi, c0_lo, c0_hi);
                    add_row(m + 1, row1_lo, row1_hi, c1_lo, c1_hi);
                    add_row(m + 2, row2_lo, row2_hi, c2_lo, c2_hi);
                    add_row(m + 3, row3_lo, row3_hi, c3_lo, c3_hi);
                }
            }

            if (full_m_tile && full_n_tile) {
                vst1q_f32(C + (m + 0) * ldc + n, c0_lo);
                vst1q_f32(C + (m + 0) * ldc + n + 4, c0_hi);
                vst1q_f32(C + (m + 1) * ldc + n, c1_lo);
                vst1q_f32(C + (m + 1) * ldc + n + 4, c1_hi);
                vst1q_f32(C + (m + 2) * ldc + n, c2_lo);
                vst1q_f32(C + (m + 2) * ldc + n + 4, c2_hi);
                vst1q_f32(C + (m + 3) * ldc + n, c3_lo);
                vst1q_f32(C + (m + 3) * ldc + n + 4, c3_hi);
            } else {
                auto store_row = [&](int row, float32x4_t lo, float32x4_t hi) {
                    if (row >= m_tile_end) return;
                    float* c_row = C + row * ldc;
                    float tmp[4];
                    vst1q_f32(tmp, lo);
                    for (int c = 0; c < 4 && n + c < n_tile_end; c++) c_row[n + c] = tmp[c];
                    vst1q_f32(tmp, hi);
                    for (int c = 0; c < 4 && n + 4 + c < n_tile_end; c++) c_row[n + 4 + c] = tmp[c];
                };
                store_row(m + 0, c0_lo, c0_hi);
                store_row(m + 1, c1_lo, c1_hi);
                store_row(m + 2, c2_lo, c2_hi);
                store_row(m + 3, c3_lo, c3_hi);
            }
        }
    }
}

static inline void w8_i8mm_8x4_dot_rows(
    const int8_t* qa_block, int K_padded,
    const int8_t* b_block, int col_base,
    int32x4_t rows[8])
{
    const int8_t* qa0 = qa_block;
    const int8_t* qa1 = qa0 + K_padded;
    const int8_t* qa2 = qa1 + K_padded;
    const int8_t* qa3 = qa2 + K_padded;
    const int8_t* qa4 = qa3 + K_padded;
    const int8_t* qa5 = qa4 + K_padded;
    const int8_t* qa6 = qa5 + K_padded;
    const int8_t* qa7 = qa6 + K_padded;

    int32x4_t acc01_01 = vdupq_n_s32(0);
    int32x4_t acc01_23 = vdupq_n_s32(0);
    int32x4_t acc23_01 = vdupq_n_s32(0);
    int32x4_t acc23_23 = vdupq_n_s32(0);
    int32x4_t acc45_01 = vdupq_n_s32(0);
    int32x4_t acc45_23 = vdupq_n_s32(0);
    int32x4_t acc67_01 = vdupq_n_s32(0);
    int32x4_t acc67_23 = vdupq_n_s32(0);

    const int8_t* b0 = b_block + (col_base + 0) * W8_Q8_BLOCK;
    const int8_t* b1 = b_block + (col_base + 1) * W8_Q8_BLOCK;
    const int8_t* b2 = b_block + (col_base + 2) * W8_Q8_BLOCK;
    const int8_t* b3 = b_block + (col_base + 3) * W8_Q8_BLOCK;

    for (int off = 0; off < W8_Q8_BLOCK; off += 8) {
        int8x16_t a01 = vcombine_s8(vld1_s8(qa0 + off), vld1_s8(qa1 + off));
        int8x16_t a23 = vcombine_s8(vld1_s8(qa2 + off), vld1_s8(qa3 + off));
        int8x16_t a45 = vcombine_s8(vld1_s8(qa4 + off), vld1_s8(qa5 + off));
        int8x16_t a67 = vcombine_s8(vld1_s8(qa6 + off), vld1_s8(qa7 + off));

        int8x16_t b01 = vcombine_s8(vld1_s8(b0 + off), vld1_s8(b1 + off));
        int8x16_t b23 = vcombine_s8(vld1_s8(b2 + off), vld1_s8(b3 + off));

        acc01_01 = vmmlaq_s32(acc01_01, a01, b01);
        acc01_23 = vmmlaq_s32(acc01_23, a01, b23);
        acc23_01 = vmmlaq_s32(acc23_01, a23, b01);
        acc23_23 = vmmlaq_s32(acc23_23, a23, b23);
        acc45_01 = vmmlaq_s32(acc45_01, a45, b01);
        acc45_23 = vmmlaq_s32(acc45_23, a45, b23);
        acc67_01 = vmmlaq_s32(acc67_01, a67, b01);
        acc67_23 = vmmlaq_s32(acc67_23, a67, b23);
    }

    rows[0] = vcombine_s32(vget_low_s32(acc01_01), vget_low_s32(acc01_23));
    rows[1] = vcombine_s32(vget_high_s32(acc01_01), vget_high_s32(acc01_23));
    rows[2] = vcombine_s32(vget_low_s32(acc23_01), vget_low_s32(acc23_23));
    rows[3] = vcombine_s32(vget_high_s32(acc23_01), vget_high_s32(acc23_23));
    rows[4] = vcombine_s32(vget_low_s32(acc45_01), vget_low_s32(acc45_23));
    rows[5] = vcombine_s32(vget_high_s32(acc45_01), vget_high_s32(acc45_23));
    rows[6] = vcombine_s32(vget_low_s32(acc67_01), vget_low_s32(acc67_23));
    rows[7] = vcombine_s32(vget_high_s32(acc67_01), vget_high_s32(acc67_23));
}

static void matmul_int8_q8dot_neon_8x8_repacked_i8mm_range(
    const int8_t* qA, const float* a_scales,
    const int8_t* B_repack, const float* scales,
    int group_size, int groups_per_row,
    float* C, int M, int N, int K,
    int K_padded, int ldc, int m_begin, int m_end,
    int n_begin, int n_end)
{
    (void)M;
    (void)N;
    if (group_size <= 0) group_size = K;
    int blocks_per_row = (K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 8) {
        if (m + 8 > m_end) {
            matmul_int8_q8dot_neon_4x8_repacked_i8mm_range(
                qA, a_scales, B_repack, scales, group_size, groups_per_row,
                C, M, N, K, K_padded, ldc, m, m_end, n_begin, n_end);
            break;
        }

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);
            const int8_t* b_tile = B_repack + (size_t)(n / 8) * blocks_per_row * 8 * W8_Q8_BLOCK;
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float32x4_t c_lo[8];
            float32x4_t c_hi[8];
            for (int r = 0; r < 8; r++) {
                c_lo[r] = vdupq_n_f32(0.f);
                c_hi[r] = vdupq_n_f32(0.f);
            }

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int group = w8_scale_group(scale_mode, qb, group_size);
                const int8_t* b_block = b_tile + (size_t)qb * 8 * W8_Q8_BLOCK;
                const int8_t* qa_block = qA + (size_t)m * K_padded + qb * W8_Q8_BLOCK;

                float32x4_t bs0 = bscale_lo_pc;
                float32x4_t bs1 = bscale_hi_pc;
                if (!scale_per_channel) {
                    load_w8_b_scales8(scales, n, c_valid, groups_per_row, group, bs0, bs1);
                }

                float a0 = a_scales[(size_t)(m + 0) * blocks_per_row + qb];
                float a1 = a_scales[(size_t)(m + 1) * blocks_per_row + qb];
                float a2 = a_scales[(size_t)(m + 2) * blocks_per_row + qb];
                float a3 = a_scales[(size_t)(m + 3) * blocks_per_row + qb];
                float a4 = a_scales[(size_t)(m + 4) * blocks_per_row + qb];
                float a5 = a_scales[(size_t)(m + 5) * blocks_per_row + qb];
                float a6 = a_scales[(size_t)(m + 6) * blocks_per_row + qb];
                float a7 = a_scales[(size_t)(m + 7) * blocks_per_row + qb];

                int32x4_t rows[8];
                w8_i8mm_8x4_dot_rows(qa_block, K_padded, b_block, 0, rows);
                c_lo[0] = vfmaq_f32(c_lo[0], vcvtq_f32_s32(rows[0]), vmulq_n_f32(bs0, a0));
                c_lo[1] = vfmaq_f32(c_lo[1], vcvtq_f32_s32(rows[1]), vmulq_n_f32(bs0, a1));
                c_lo[2] = vfmaq_f32(c_lo[2], vcvtq_f32_s32(rows[2]), vmulq_n_f32(bs0, a2));
                c_lo[3] = vfmaq_f32(c_lo[3], vcvtq_f32_s32(rows[3]), vmulq_n_f32(bs0, a3));
                c_lo[4] = vfmaq_f32(c_lo[4], vcvtq_f32_s32(rows[4]), vmulq_n_f32(bs0, a4));
                c_lo[5] = vfmaq_f32(c_lo[5], vcvtq_f32_s32(rows[5]), vmulq_n_f32(bs0, a5));
                c_lo[6] = vfmaq_f32(c_lo[6], vcvtq_f32_s32(rows[6]), vmulq_n_f32(bs0, a6));
                c_lo[7] = vfmaq_f32(c_lo[7], vcvtq_f32_s32(rows[7]), vmulq_n_f32(bs0, a7));

                w8_i8mm_8x4_dot_rows(qa_block, K_padded, b_block, 4, rows);
                c_hi[0] = vfmaq_f32(c_hi[0], vcvtq_f32_s32(rows[0]), vmulq_n_f32(bs1, a0));
                c_hi[1] = vfmaq_f32(c_hi[1], vcvtq_f32_s32(rows[1]), vmulq_n_f32(bs1, a1));
                c_hi[2] = vfmaq_f32(c_hi[2], vcvtq_f32_s32(rows[2]), vmulq_n_f32(bs1, a2));
                c_hi[3] = vfmaq_f32(c_hi[3], vcvtq_f32_s32(rows[3]), vmulq_n_f32(bs1, a3));
                c_hi[4] = vfmaq_f32(c_hi[4], vcvtq_f32_s32(rows[4]), vmulq_n_f32(bs1, a4));
                c_hi[5] = vfmaq_f32(c_hi[5], vcvtq_f32_s32(rows[5]), vmulq_n_f32(bs1, a5));
                c_hi[6] = vfmaq_f32(c_hi[6], vcvtq_f32_s32(rows[6]), vmulq_n_f32(bs1, a6));
                c_hi[7] = vfmaq_f32(c_hi[7], vcvtq_f32_s32(rows[7]), vmulq_n_f32(bs1, a7));
            }

            for (int r = 0; r < 8; r++) {
                float* c_row = C + (m + r) * ldc;
                if (full_n_tile) {
                    vst1q_f32(c_row + n, c_lo[r]);
                    vst1q_f32(c_row + n + 4, c_hi[r]);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, c_lo[r]);
                    for (int c = 0; c < 4 && c < c_valid; c++) c_row[n + c] = tmp[c];
                    vst1q_f32(tmp, c_hi[r]);
                    for (int c = 0; c < 4 && c + 4 < c_valid; c++) c_row[n + 4 + c] = tmp[c];
                }
            }
        }
    }
}
#endif
#endif

static inline float16x8_t dequant_int8x8_to_fp16(
    const int8_t* q, const float* scales, int n, int n_tile_end,
    int groups_per_row, int group)
{
    int8x8_t b8 = vld1_s8(q);
    int16x8_t b16 = vmovl_s8(b8);
    float32x4_t b0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
    float32x4_t b1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));

    float scale_tmp[8];
    for (int j = 0; j < 8; j++) {
        scale_tmp[j] = (n + j < n_tile_end)
            ? scales[(n + j) * groups_per_row + group]
            : 0.f;
    }
    b0 = vmulq_f32(b0, vld1q_f32(scale_tmp));
    b1 = vmulq_f32(b1, vld1q_f32(scale_tmp + 4));
    return vcombine_f16(vcvt_f16_f32(b0), vcvt_f16_f32(b1));
}

static void matmul_int8_neon_gemv_range_fp16acc(
    const float* A, const int8_t* B_packed, const float* scales,
    int group_size, int groups_per_row,
    float* C, int K, int n_begin, int n_end,
    Activation act = Activation::NONE,
    int act_n_begin = 0, int act_n_len = -1)
{
    if (group_size <= 0) group_size = K;
    if (groups_per_row <= 0) groups_per_row = 1;
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        const int8_t* b_tile = &B_packed[(n & ~7) * K];

        float16x8_t acc[8];
        for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK_FP16) {
            int k_end = std::min(k_outer + K_BLOCK_FP16, K);
            if (k_outer == 0) {
                for (int i = 0; i < 8; i++) acc[i] = vdupq_n_f16((__fp16)0.f);
            } else {
                float tmp[8];
                for (int j = 0; j < 8; j++) tmp[j] = (n + j < n_tile_end) ? C[n + j] : 0.f;
                acc[0] = vcombine_f16(
                    vcvt_f16_f32(vld1q_f32(tmp)),
                    vcvt_f16_f32(vld1q_f32(tmp + 4)));
                for (int i = 1; i < 8; i++) acc[i] = vdupq_n_f16((__fp16)0.f);
            }

            int k = k_outer;
            int k_unrolled_end = k_outer + ((k_end - k_outer) & ~7);
            for (; k < k_unrolled_end; k += 8) {
                __fp16 a0 = (__fp16)A[k + 0];
                __fp16 a1 = (__fp16)A[k + 1];
                __fp16 a2 = (__fp16)A[k + 2];
                __fp16 a3 = (__fp16)A[k + 3];
                __fp16 a4 = (__fp16)A[k + 4];
                __fp16 a5 = (__fp16)A[k + 5];
                __fp16 a6 = (__fp16)A[k + 6];
                __fp16 a7 = (__fp16)A[k + 7];
                float16x8_t b0 = dequant_int8x8_to_fp16(b_tile + (k + 0) * 8, scales, n, n_tile_end, groups_per_row, (k + 0) / group_size);
                float16x8_t b1 = dequant_int8x8_to_fp16(b_tile + (k + 1) * 8, scales, n, n_tile_end, groups_per_row, (k + 1) / group_size);
                float16x8_t b2 = dequant_int8x8_to_fp16(b_tile + (k + 2) * 8, scales, n, n_tile_end, groups_per_row, (k + 2) / group_size);
                float16x8_t b3 = dequant_int8x8_to_fp16(b_tile + (k + 3) * 8, scales, n, n_tile_end, groups_per_row, (k + 3) / group_size);
                float16x8_t b4 = dequant_int8x8_to_fp16(b_tile + (k + 4) * 8, scales, n, n_tile_end, groups_per_row, (k + 4) / group_size);
                float16x8_t b5 = dequant_int8x8_to_fp16(b_tile + (k + 5) * 8, scales, n, n_tile_end, groups_per_row, (k + 5) / group_size);
                float16x8_t b6 = dequant_int8x8_to_fp16(b_tile + (k + 6) * 8, scales, n, n_tile_end, groups_per_row, (k + 6) / group_size);
                float16x8_t b7 = dequant_int8x8_to_fp16(b_tile + (k + 7) * 8, scales, n, n_tile_end, groups_per_row, (k + 7) / group_size);
                acc[0] = vfmaq_n_f16(acc[0], b0, a0);
                acc[1] = vfmaq_n_f16(acc[1], b1, a1);
                acc[2] = vfmaq_n_f16(acc[2], b2, a2);
                acc[3] = vfmaq_n_f16(acc[3], b3, a3);
                acc[4] = vfmaq_n_f16(acc[4], b4, a4);
                acc[5] = vfmaq_n_f16(acc[5], b5, a5);
                acc[6] = vfmaq_n_f16(acc[6], b6, a6);
                acc[7] = vfmaq_n_f16(acc[7], b7, a7);
            }
            for (; k < k_end; k++) {
                int idx = (k - k_outer) & 7;
                __fp16 a0 = (__fp16)A[k];
                float16x8_t b0 = dequant_int8x8_to_fp16(
                    b_tile + k * 8, scales, n, n_tile_end,
                    groups_per_row, k / group_size);
                acc[idx] = vfmaq_n_f16(acc[idx], b0, a0);
            }

            acc[0] = vaddq_f16(acc[0], acc[1]);
            acc[2] = vaddq_f16(acc[2], acc[3]);
            acc[4] = vaddq_f16(acc[4], acc[5]);
            acc[6] = vaddq_f16(acc[6], acc[7]);
            acc[0] = vaddq_f16(acc[0], acc[2]);
            acc[4] = vaddq_f16(acc[4], acc[6]);
            acc[0] = vaddq_f16(acc[0], acc[4]);

            bool is_last_block = (k_outer + K_BLOCK_FP16 >= K);
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(acc[0]));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(acc[0]));
            if (is_last_block && act != Activation::NONE && act_n_len != 0) {
                bool lo_active = (act_n_len < 0) ||
                                 (activation_applies_at(n + 0, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 1, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 2, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 3, act_n_begin, act_n_len));
                if (lo_active) lo = apply_activation_f32_neon(lo, act);
                bool hi_active = (act_n_len < 0) ||
                                 (activation_applies_at(n + 4, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 5, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 6, act_n_begin, act_n_len) &&
                                  activation_applies_at(n + 7, act_n_begin, act_n_len));
                if (hi_active) hi = apply_activation_f32_neon(hi, act);
            }

            float tmp[4];
            vst1q_f32(tmp, lo);
            for (int j = 0; j < 4 && n + j < n_tile_end; j++) C[n + j] = tmp[j];
            vst1q_f32(tmp, hi);
            for (int j = 0; j < 4 && n + 4 + j < n_tile_end; j++) C[n + 4 + j] = tmp[j];
        }
    }
}

static void matmul_int8_neon_8x8_range_packed_a_fp16acc(
    const __fp16* A_packed, const int8_t* B_packed, const float* scales,
    int group_size, int groups_per_row,
    float* C, int M, int N, int K,
    int ldc, int m_begin, int m_end,
    Activation act = Activation::NONE,
    int act_n_begin = 0, int act_n_len = -1)
{
    if (group_size <= 0) group_size = K;
    if (groups_per_row <= 0) groups_per_row = 1;
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2;
    bool act_enabled = (act != Activation::NONE) && (act_n_len != 0);
    bool act_full_N = act_enabled && (act_n_len < 0 ||
                                      (act_n_begin == 0 && act_n_len >= N));

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK_FP16) {
        int k_end = std::min(k_outer + K_BLOCK_FP16, K);
        bool first_block = (k_outer == 0);
        bool last_block = (k_outer + K_BLOCK_FP16 >= K);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);
                const int8_t* b_tile = &B_packed[(n & ~7) * K];

                float16x8_t c[8];
                if (first_block) {
                    for (int j = 0; j < 8; j++) c[j] = vdupq_n_f16((__fp16)0.f);
                } else {
                    for (int j = 0; j < 8; j++) {
                        if (n + j < n_end) {
                            float tmp[8];
                            for (int r = 0; r < 8; r++) {
                                int row = m + r;
                                tmp[r] = (row < m_tile_end && row < m_global_end)
                                       ? C[row * ldc + n + j] : 0.f;
                            }
                            c[j] = vcombine_f16(
                                vcvt_f16_f32(vld1q_f32(tmp)),
                                vcvt_f16_f32(vld1q_f32(tmp + 4)));
                        } else {
                            c[j] = vdupq_n_f16((__fp16)0.f);
                        }
                    }
                }

                float16x8_t c1[8];
                for (int j = 0; j < 8; j++) c1[j] = vdupq_n_f16((__fp16)0.f);

                int k = k_outer;
                for (; k + 1 < k_end; k += 2) {
                    float16x8_t a0 = vld1q_f16(&A_packed[(m & ~7) * K + k * 8]);
                    float16x8_t a1 = vld1q_f16(&A_packed[(m & ~7) * K + (k + 1) * 8]);
                    float16x8_t b0 = dequant_int8x8_to_fp16(
                        b_tile + k * 8, scales, n, n_end,
                        groups_per_row, k / group_size);
                    float16x8_t b1 = dequant_int8x8_to_fp16(
                        b_tile + (k + 1) * 8, scales, n, n_end,
                        groups_per_row, (k + 1) / group_size);
                    float16x4_t b0_low = vget_low_f16(b0), b0_high = vget_high_f16(b0);
                    float16x4_t b1_low = vget_low_f16(b1), b1_high = vget_high_f16(b1);

                    c[0] = vfmaq_lane_f16(c[0], a0, b0_low, 0);
                    c[1] = vfmaq_lane_f16(c[1], a0, b0_low, 1);
                    c[2] = vfmaq_lane_f16(c[2], a0, b0_low, 2);
                    c[3] = vfmaq_lane_f16(c[3], a0, b0_low, 3);
                    c[4] = vfmaq_lane_f16(c[4], a0, b0_high, 0);
                    c[5] = vfmaq_lane_f16(c[5], a0, b0_high, 1);
                    c[6] = vfmaq_lane_f16(c[6], a0, b0_high, 2);
                    c[7] = vfmaq_lane_f16(c[7], a0, b0_high, 3);

                    c1[0] = vfmaq_lane_f16(c1[0], a1, b1_low, 0);
                    c1[1] = vfmaq_lane_f16(c1[1], a1, b1_low, 1);
                    c1[2] = vfmaq_lane_f16(c1[2], a1, b1_low, 2);
                    c1[3] = vfmaq_lane_f16(c1[3], a1, b1_low, 3);
                    c1[4] = vfmaq_lane_f16(c1[4], a1, b1_high, 0);
                    c1[5] = vfmaq_lane_f16(c1[5], a1, b1_high, 1);
                    c1[6] = vfmaq_lane_f16(c1[6], a1, b1_high, 2);
                    c1[7] = vfmaq_lane_f16(c1[7], a1, b1_high, 3);
                }
                if (k < k_end) {
                    float16x8_t a0 = vld1q_f16(&A_packed[(m & ~7) * K + k * 8]);
                    float16x8_t b0 = dequant_int8x8_to_fp16(
                        b_tile + k * 8, scales, n, n_end,
                        groups_per_row, k / group_size);
                    float16x4_t b0_low = vget_low_f16(b0), b0_high = vget_high_f16(b0);
                    c[0] = vfmaq_lane_f16(c[0], a0, b0_low, 0);
                    c[1] = vfmaq_lane_f16(c[1], a0, b0_low, 1);
                    c[2] = vfmaq_lane_f16(c[2], a0, b0_low, 2);
                    c[3] = vfmaq_lane_f16(c[3], a0, b0_low, 3);
                    c[4] = vfmaq_lane_f16(c[4], a0, b0_high, 0);
                    c[5] = vfmaq_lane_f16(c[5], a0, b0_high, 1);
                    c[6] = vfmaq_lane_f16(c[6], a0, b0_high, 2);
                    c[7] = vfmaq_lane_f16(c[7], a0, b0_high, 3);
                }
                for (int j = 0; j < 8; j++) c[j] = vaddq_f16(c[j], c1[j]);

                bool tile_active = act_enabled && last_block &&
                                   (act_full_N ||
                                    (activation_applies_at(n + 0, act_n_begin, act_n_len) &&
                                     activation_applies_at(n + 7, act_n_begin, act_n_len)));
                for (int j = 0; j < 8 && n + j < n_end; j++) {
                    float32x4_t lo = vcvt_f32_f16(vget_low_f16(c[j]));
                    float32x4_t hi = vcvt_f32_f16(vget_high_f16(c[j]));
                    if (tile_active) {
                        lo = apply_activation_f32_neon(lo, act);
                        hi = apply_activation_f32_neon(hi, act);
                    } else if (act_enabled && last_block &&
                               activation_applies_at(n + j, act_n_begin, act_n_len)) {
                        lo = apply_activation_f32_neon(lo, act);
                        hi = apply_activation_f32_neon(hi, act);
                    }
                    float tmp[8];
                    vst1q_f32(tmp, lo);
                    vst1q_f32(tmp + 4, hi);
                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            C[row * ldc + n + j] = tmp[r];
                        }
                    }
                }
            }
        }
    }
}

// ---- FP16 lane-FMA kernel (A + B both pre-packed interleaved) ----
//
// A_packed: [M/8, K, 8] FP16 — 8 M values at same k are contiguous.
// B_packed: [N/8, K, 8] FP16 — 8 N values at same k are contiguous.
// Uses vfmlalq_laneq_low/high_f16 for FP16×FP16→FP32 widening FMA with
// lane broadcast: one A vector × one B lane → 4-way FP32 accumulate.
// Per K step: 2 loads + 16 FMLAL = 64 FLOPs in 18 instructions.
//
// Accumulator layout (column-major for efficient lane-FMA):
//   c[j][0] = rows 0..3 for N column n+j  (float32x4_t)
//   c[j][1] = rows 4..7 for N column n+j  (float32x4_t)
// C is row-major: C[row * ldc + col]. Transpose needed at init/writeback.
static void matmul_fp16_neon_8x8_range_packed_a(
    const __fp16* A_packed, const __fp16* B_packed, float* C,
    int M, int N, int K,
    int ldc, int m_begin, int m_end)
{
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2;

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK_FP16) {
        int k_end = std::min(k_outer + K_BLOCK_FP16, K);
        bool first_block = (k_outer == 0);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                // Column-major accumulators: c[j][half]
                // c[j][0] = [C[m+0][n+j], C[m+1][n+j], C[m+2][n+j], C[m+3][n+j]]
                // c[j][1] = [C[m+4][n+j], C[m+5][n+j], C[m+6][n+j], C[m+7][n+j]]
                float32x4_t c[8][2];
                if (first_block) {
                    for (int j = 0; j < 8; j++) {
                        c[j][0] = vdupq_n_f32(0.f);
                        c[j][1] = vdupq_n_f32(0.f);
                    }
                } else {
                    // Gather C columns into column-major accumulators.
                    // C is row-major: C[(m+r) * ldc + (n+j)].
                    // Build c[j][0] = [C[m+0][n+j], C[m+1][n+j], C[m+2][n+j], C[m+3][n+j]]
                    // Use vld4q_f32 to de-interleave 4 rows × 4 cols block.
                    // For each 4x4 block, load 4 rows and transpose.
                    auto load_col4 = [&](int col, int row_start) -> float32x4_t {
                        // Gather C[row_start..row_start+3][col] into a vector
                        float tmp[4];
                        for (int r = 0; r < 4; r++) {
                            int row = row_start + r;
                            if (row < m_tile_end && row < m_global_end && col < n_end) {
                                tmp[r] = C[row * ldc + col];
                            } else {
                                tmp[r] = 0.f;
                            }
                        }
                        return vld1q_f32(tmp);
                    };
                    for (int j = 0; j < 8; j++) {
                        c[j][0] = load_col4(n + j, m + 0);
                        c[j][1] = load_col4(n + j, m + 4);
                    }
                }

                // K loop: lane-FMA
                // A_packed: A_packed[(m & ~7) * K + k * 8 + 0..7] = 8 M values at K=k
                // B_packed: B_packed[(n & ~7) * K + k * 8 + 0..7] = 8 N values at K=k
                for (int k = k_outer; k < k_end; k++) {
                    float16x8_t a_vec = vld1q_f16(&A_packed[(m & ~7) * K + k * 8]);
                    float16x8_t b_vec = vld1q_f16(&B_packed[(n & ~7) * K + k * 8]);

                    // For each N column j, broadcast b_vec[j] and FMA with a_vec
                    // low half (rows 0..3) and high half (rows 4..7)
                    c[0][0] = vfmlalq_laneq_low_f16 (c[0][0], a_vec, b_vec, 0);
                    c[0][1] = vfmlalq_laneq_high_f16(c[0][1], a_vec, b_vec, 0);
                    c[1][0] = vfmlalq_laneq_low_f16 (c[1][0], a_vec, b_vec, 1);
                    c[1][1] = vfmlalq_laneq_high_f16(c[1][1], a_vec, b_vec, 1);
                    c[2][0] = vfmlalq_laneq_low_f16 (c[2][0], a_vec, b_vec, 2);
                    c[2][1] = vfmlalq_laneq_high_f16(c[2][1], a_vec, b_vec, 2);
                    c[3][0] = vfmlalq_laneq_low_f16 (c[3][0], a_vec, b_vec, 3);
                    c[3][1] = vfmlalq_laneq_high_f16(c[3][1], a_vec, b_vec, 3);
                    c[4][0] = vfmlalq_laneq_low_f16 (c[4][0], a_vec, b_vec, 4);
                    c[4][1] = vfmlalq_laneq_high_f16(c[4][1], a_vec, b_vec, 4);
                    c[5][0] = vfmlalq_laneq_low_f16 (c[5][0], a_vec, b_vec, 5);
                    c[5][1] = vfmlalq_laneq_high_f16(c[5][1], a_vec, b_vec, 5);
                    c[6][0] = vfmlalq_laneq_low_f16 (c[6][0], a_vec, b_vec, 6);
                    c[6][1] = vfmlalq_laneq_high_f16(c[6][1], a_vec, b_vec, 6);
                    c[7][0] = vfmlalq_laneq_low_f16 (c[7][0], a_vec, b_vec, 7);
                    c[7][1] = vfmlalq_laneq_high_f16(c[7][1], a_vec, b_vec, 7);
                }

                // Write back: column-major → row-major C
                // c[j][0] = [C[m+0][n+j], C[m+1][n+j], C[m+2][n+j], C[m+3][n+j]]
                // Need: C[(m+r) * ldc + (n+j)] = c[j][half][r]
                auto store_col4 = [&](int col, int row_start, float32x4_t val) {
                    float tmp[4];
                    vst1q_f32(tmp, val);
                    for (int r = 0; r < 4; r++) {
                        int row = row_start + r;
                        if (row < m_tile_end && row < m_global_end && col < n_end) {
                            C[row * ldc + col] = tmp[r];
                        }
                    }
                };
                for (int j = 0; j < 8; j++) {
                    store_col4(n + j, m + 0, c[j][0]);
                    store_col4(n + j, m + 4, c[j][1]);
                }
            }
        }
    }
}

// ---- FP16 lane-FMA kernel with FP16 accumulation ----
//
// Same interface as above but accumulates in float16x8_t using vfmaq_lane_f16.
// 2x FMA throughput vs FP32 widening path. Accumulator is row-major
// (c[j] = 8 M rows for N column j), writeback is direct store.
//
// Precision: FP16 accumulation over K_BLOCK (1024) terms. Between K-blocks,
// partial results are stored to C (FP32) and reloaded (FP16), providing
// FP32 intermediate precision across blocks.
static void matmul_fp16_neon_8x8_range_packed_a_fp16acc(
    const __fp16* A_packed, const __fp16* B_packed, float* C,
    int M, int N, int K,
    int ldc, int m_begin, int m_end,
    Activation act = Activation::NONE,
    int act_n_begin = 0, int act_n_len = -1)
{
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;
    const int K_BLOCK_FP16 = K_BLOCK * 2;

    // Fused activation fast path: when act == NONE or act_n_len covers whole N,
    // we can skip per-column check. We still need to apply after all K-blocks
    // (activation must not be applied to intermediate partial sums).
    bool act_enabled = (act != Activation::NONE) && (act_n_len != 0);
    bool act_full_N = act_enabled && (act_n_len < 0 ||
                                      (act_n_begin == 0 && act_n_len >= N));

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK_FP16) {
        int k_end = std::min(k_outer + K_BLOCK_FP16, K);
        bool first_block = (k_outer == 0);
        bool last_block = (k_outer + K_BLOCK_FP16 >= K);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                // Row-major accumulators: c[j] = 8 M values for N column n+j
                // Layout matches C: c[j][r] = C[(m+r)][n+j]
                float16x8_t c[8];
                if (first_block) {
                    for (int j = 0; j < 8; j++) c[j] = vdupq_n_f16((__fp16)0.f);
                } else {
                    // Reload from C (FP32 → FP16)
                    for (int j = 0; j < 8; j++) {
                        if (n + j < n_end) {
                            float tmp[8];
                            for (int r = 0; r < 8; r++) {
                                int row = m + r;
                                tmp[r] = (row < m_tile_end && row < m_global_end)
                                       ? C[row * ldc + n + j] : 0.f;
                            }
                            c[j] = vcombine_f16(
                                vcvt_f16_f32(vld1q_f32(tmp)),
                                vcvt_f16_f32(vld1q_f32(tmp + 4)));
                        } else {
                            c[j] = vdupq_n_f16((__fp16)0.f);
                        }
                    }
                }

                // K loop: FP16 lane-FMA, 2-way unroll.
                // 8 N-columns already provide 8 independent acc chains (>>2),
                // enough to hide FMA latency (lat=2, throughput=2/cycle).
                // 2-way unroll mainly amortizes loop branch overhead.
                // Note: 8-way unroll (64 acc) was tried and reverted — it caused
                // register spill on Apple Silicon (32 NEON Q regs), hurting prefill.
                float16x8_t c1[8];
                for (int j = 0; j < 8; j++) c1[j] = vdupq_n_f16((__fp16)0.f);

                int k = k_outer;
                for (; k + 1 < k_end; k += 2) {
                    float16x8_t a0 = vld1q_f16(&A_packed[(m & ~7) * K + k * 8]);
                    float16x8_t a1 = vld1q_f16(&A_packed[(m & ~7) * K + (k + 1) * 8]);
                    float16x8_t b0 = vld1q_f16(&B_packed[(n & ~7) * K + k * 8]);
                    float16x8_t b1 = vld1q_f16(&B_packed[(n & ~7) * K + (k + 1) * 8]);
                    float16x4_t b0_low = vget_low_f16(b0), b0_high = vget_high_f16(b0);
                    float16x4_t b1_low = vget_low_f16(b1), b1_high = vget_high_f16(b1);

                    c[0] = vfmaq_lane_f16(c[0], a0, b0_low, 0);
                    c[1] = vfmaq_lane_f16(c[1], a0, b0_low, 1);
                    c[2] = vfmaq_lane_f16(c[2], a0, b0_low, 2);
                    c[3] = vfmaq_lane_f16(c[3], a0, b0_low, 3);
                    c[4] = vfmaq_lane_f16(c[4], a0, b0_high, 0);
                    c[5] = vfmaq_lane_f16(c[5], a0, b0_high, 1);
                    c[6] = vfmaq_lane_f16(c[6], a0, b0_high, 2);
                    c[7] = vfmaq_lane_f16(c[7], a0, b0_high, 3);

                    c1[0] = vfmaq_lane_f16(c1[0], a1, b1_low, 0);
                    c1[1] = vfmaq_lane_f16(c1[1], a1, b1_low, 1);
                    c1[2] = vfmaq_lane_f16(c1[2], a1, b1_low, 2);
                    c1[3] = vfmaq_lane_f16(c1[3], a1, b1_low, 3);
                    c1[4] = vfmaq_lane_f16(c1[4], a1, b1_high, 0);
                    c1[5] = vfmaq_lane_f16(c1[5], a1, b1_high, 1);
                    c1[6] = vfmaq_lane_f16(c1[6], a1, b1_high, 2);
                    c1[7] = vfmaq_lane_f16(c1[7], a1, b1_high, 3);
                }
                // Tail (odd K within block)
                if (k < k_end) {
                    float16x8_t a0 = vld1q_f16(&A_packed[(m & ~7) * K + k * 8]);
                    float16x8_t b0 = vld1q_f16(&B_packed[(n & ~7) * K + k * 8]);
                    float16x4_t b0_low = vget_low_f16(b0), b0_high = vget_high_f16(b0);
                    c[0] = vfmaq_lane_f16(c[0], a0, b0_low, 0);
                    c[1] = vfmaq_lane_f16(c[1], a0, b0_low, 1);
                    c[2] = vfmaq_lane_f16(c[2], a0, b0_low, 2);
                    c[3] = vfmaq_lane_f16(c[3], a0, b0_low, 3);
                    c[4] = vfmaq_lane_f16(c[4], a0, b0_high, 0);
                    c[5] = vfmaq_lane_f16(c[5], a0, b0_high, 1);
                    c[6] = vfmaq_lane_f16(c[6], a0, b0_high, 2);
                    c[7] = vfmaq_lane_f16(c[7], a0, b0_high, 3);
                }
                // Merge c += c1
                for (int j = 0; j < 8; j++) c[j] = vaddq_f16(c[j], c1[j]);

                // Write back: FP16 → FP32, store to C
                // On the last K-block, also apply fused activation if enabled.
                // Batch-check activation status per N-tile (8 columns): SwiGLU
                // gate (act_n_len multiple of 8) → all 8 columns same status,
                // so we can skip per-column check entirely.
                bool tile_active = act_enabled && last_block &&
                                   (act_full_N ||
                                    (activation_applies_at(n + 0, act_n_begin, act_n_len) &&
                                     activation_applies_at(n + 7, act_n_begin, act_n_len)));
                for (int j = 0; j < 8 && n + j < n_end; j++) {
                    float32x4_t lo = vcvt_f32_f16(vget_low_f16(c[j]));
                    float32x4_t hi = vcvt_f32_f16(vget_high_f16(c[j]));
                    if (tile_active) {
                        lo = apply_activation_f32_neon(lo, act);
                        hi = apply_activation_f32_neon(hi, act);
                    } else if (act_enabled && last_block) {
                        // Mixed tile (rare — only at gate/up boundary).
                        // Per-column scalar apply via lane extract/insert.
                        if (activation_applies_at(n + j, act_n_begin, act_n_len)) {
                            // Apply to all 8 M-rows for this N column.
                            // lo = [m0,m1,m2,m3], hi = [m4,m5,m6,m7]
                            float32x4_t a_lo = apply_activation_f32_neon(lo, act);
                            float32x4_t a_hi = apply_activation_f32_neon(hi, act);
                            lo = a_lo; hi = a_hi;
                        }
                    }
                    float tmp[8];
                    vst1q_f32(tmp, lo);
                    vst1q_f32(tmp + 4, hi);
                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            C[row * ldc + n + j] = tmp[r];
                        }
                    }
                }
            }
        }
    }
}

// ---- FP32 kernel (kept for backward compat) ----
static void matmul_fp32_neon_8x8_range(const float* A, const float* B, float* C,
                                       int M, int N, int K,
                                       int lda, int K_weight, int ldc,
                                       int m_begin, int m_end) {
    const int K_BLOCK = g_matmul_config.k_block > 0 ? g_matmul_config.k_block : K;

    for (int k_outer = 0; k_outer < K; k_outer += K_BLOCK) {
        int k_end = std::min(k_outer + K_BLOCK, K);

        for (int m = m_begin; m < m_end; m += 8) {
            int m_tile_end = std::min(m + 8, m_end);
            int m_global_end = std::min(m + 8, M);

            for (int n = 0; n < N; n += 8) {
                int n_end = std::min(n + 8, N);

                float32x4_t c[8][2]; // c[row][col_hi/lo]: hi=cols 0-3, lo=cols 4-7
                bool first_block = (k_outer == 0);
                if (first_block) {
                    for (int r = 0; r < 8; r++) {
                        c[r][0] = vdupq_n_f32(0.f);
                        c[r][1] = vdupq_n_f32(0.f);
                    }
                } else {
                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            c[r][0] = vld1q_f32(&C[row * ldc + n]);
                            if (n + 4 < n_end) {
                                c[r][1] = vld1q_f32(&C[row * ldc + n + 4]);
                            } else {
                                c[r][1] = vdupq_n_f32(0.f);
                            }
                        } else {
                            c[r][0] = vdupq_n_f32(0.f);
                            c[r][1] = vdupq_n_f32(0.f);
                        }
                    }
                }

                for (int k = k_outer; k < k_end; k++) {
                    // Load B columns n..n+3
                    float tmp_b0[4] = {0.f, 0.f, 0.f, 0.f};
                    float tmp_b1[4] = {0.f, 0.f, 0.f, 0.f};
                    for (int j = 0; j < 4 && n + j < n_end; j++) {
                        tmp_b0[j] = B[(n + j) * K_weight + k];
                    }
                    for (int j = 0; j < 4 && n + 4 + j < n_end; j++) {
                        tmp_b1[j] = B[(n + 4 + j) * K_weight + k];
                    }
                    float32x4_t b0 = vld1q_f32(tmp_b0);
                    float32x4_t b1 = vld1q_f32(tmp_b1);

                    for (int r = 0; r < 8; r++) {
                        int row = m + r;
                        if (row < m_tile_end && row < m_global_end) {
                            float a_val = A[k + row * lda];
                            c[r][0] = vfmaq_n_f32(c[r][0], b0, a_val);
                            c[r][1] = vfmaq_n_f32(c[r][1], b1, a_val);
                        }
                    }
                }

                // Write back
                for (int r = 0; r < 8; r++) {
                    int row = m + r;
                    if (row < m_tile_end && row < m_global_end) {
                        float tmp[4];
                        vst1q_f32(tmp, c[r][0]);
                        for (int j = 0; j < 4 && n + j < n_end; j++) C[row * ldc + n + j] = tmp[j];
                        vst1q_f32(tmp, c[r][1]);
                        for (int j = 0; j < 4 && n + 4 + j < n_end; j++) C[row * ldc + n + 4 + j] = tmp[j];
                    }
                }
            }
        }
    }
}

#endif // HAS_NEON

static void matmul_fp32_range(const float* A, const float* B, float* C,
                              int M, int N, int K,
                              int lda, int K_weight, int ldc,
                              int m_begin, int m_end) {
#if HAS_NEON
    matmul_fp32_neon_8x8_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin, m_end);
#else
    matmul_fp32_scalar_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin, m_end);
#endif
}

// FP16 variant: B is __fp16*, A and C are float*.
static void matmul_fp16_range(const float* A, const __fp16* B, float* C,
                              int M, int N, int K,
                              int lda, int K_weight, int ldc,
                              int m_begin, int m_end) {
#if HAS_NEON
    matmul_fp16_neon_8x8_range(A, B, C, M, N, K, lda, K_weight, ldc, m_begin, m_end);
#else
    // Scalar fallback: convert each FP16 to FP32 on the fly.
    for (int m = m_begin; m < m_end; m++) {
        float* c_row = C + m * ldc;
        for (int n = 0; n < N; n++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += A[k + m * lda] * (float)B[n * K_weight + k];
            }
            c_row[n] = sum;
        }
    }
#endif
}

// Like matmul_fp32_range but shards by N (output dimension) instead of M.
// Used when N >> M (e.g. lm_head where M=1, N=vocab_size).
static void matmul_fp32_range_n(const float* A, const float* B, float* C,
                                int M, int N, int K,
                                int lda, int K_weight, int ldc,
                                int n_begin, int n_end) {
#if HAS_NEON
    // Decompose into the existing 8x8 NEON kernel by limiting N range.
    // We pass the full M range but only the [n_begin, n_end) columns of C.
    // The NEON kernel writes to C[row * ldc + col], so we offset C by n_begin.
    matmul_fp32_neon_8x8_range(A, B + n_begin * K_weight, C + n_begin,
                               M, n_end - n_begin, K,
                               lda, K_weight, ldc,
                               0, M);
#else
    matmul_fp32_scalar_range(A, B + n_begin * K_weight, C + n_begin,
                             M, n_end - n_begin, K,
                             lda, K_weight, ldc,
                             0, M);
#endif
}

// ---------------------------------------------------------------------------
// kernel_matmul_fp32
// ---------------------------------------------------------------------------

void kernel_matmul_fp32(const Tensor& A, const Tensor& B, Tensor& C,
                        ThreadPool* thread_pool,
                        Activation act,
                        int act_n_begin,
                        int act_n_len) {
    MatmulTimer _timer;  // captures all return paths
    int M = (int)A.shape[1];
    int K = (int)A.shape[0];
    int N = (int)B.shape[0];

    int lda = (int)(A.stride[1] / sizeof(float));
    int ldc = (int)(C.stride[1] / sizeof(float));
    int K_weight = (int)B.shape[1];
    const float* a_ptr = A.ptr<float>();
    float* c_ptr = C.ptr<float>();

    // Detect quantized/FP16 weight.
    bool is_int8 = (B.prec == Precision::INT8);
    bool is_int4 = (B.prec == Precision::INT4);
    bool is_fp16 = (B.prec == Precision::FP16);
    const int8_t* b_int8 = is_int8 ? reinterpret_cast<const int8_t*>(B.data) : nullptr;
    const uint8_t* b_int4 = is_int4 ? reinterpret_cast<const uint8_t*>(B.data) : nullptr;
    const __fp16* b_fp16 = is_fp16 ? reinterpret_cast<const __fp16*>(B.data) : nullptr;
    const float* b_fp32 = (is_fp16 || is_int8 || is_int4) ? nullptr : B.ptr<float>();

    // K_weight is the stride between consecutive k rows in the repacked layout.
    // For repacked [K, N]: K_weight = original N.
    // For non-repacked [N, K]: K_weight = original K (the inner dim of B).
    // We determine this by comparing K_weight with K: if they differ, it's repacked.
    bool is_repacked = (K_weight != K);

    if (is_int4) {
        const float* scales = B.scales;
        int group_size = (int)B.group_size;
        int groups_per_row = (int)B.groups_per_row;
        const uint8_t* b_q4_repack = reinterpret_cast<const uint8_t*>(B.q4_repack_data);
        const auto* b_q4_g128 = reinterpret_cast<const Q4B8G128Block*>(B.q4_g128_data);
        int n_threads = thread_pool ? thread_pool->num_threads() : 1;
        if (!scales || group_size <= 0 || groups_per_row <= 0) {
            _timer.set_shape("int4_invalid_scales", M, N, K, group_size, groups_per_row,
                             false, false, n_threads);
            return;
        }

        constexpr int tile_m = HAS_NEON ? 8 : 1;
        bool shard_by_n = (N > M * 8 && M == 1);
        int chunk_size = (M == 1 || N == 1) ? g_matmul_config.gemv_chunk_size : tile_m;
        int total_dim = shard_by_n ? N : M;
        int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
        bool use_parallel = n_threads > 1 && n_chunks > 1;

#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
        static const bool w4_q8_dot_enabled = !env_flag_enabled("MOLLM_W4_NO_Q8_DOT");
        static const bool w4_q8_dot_gemm_disabled = env_flag_enabled("MOLLM_W4_NO_Q8_DOT_GEMM");
        static const bool w4_q8_dot_gemm_4x8_enabled =
            env_flag_enabled("MOLLM_W4_Q8_DOT_GEMM_4X8");
        static const bool w4_q8_dot_bg128_enabled =
            env_flag_enabled("MOLLM_W4_PACKED_BG128");
        static const bool w4_q4_repack_enabled =
            !env_flag_enabled("MOLLM_W4_NO_Q4_REPACK") &&
            !env_flag_enabled("MOLLM_W4_NO_Q8_REPACK");
        bool has_direct_q4_g128 = B.is_q4_g128_packed && b_q4_g128;
        bool can_use_q4_dot = (B.is_q4_repacked || has_direct_q4_g128 || w4_q8_dot_enabled) &&
                              (K % W8_Q8_BLOCK == 0) &&
                              (group_size % W8_Q8_BLOCK == 0);
        bool use_q4_repack = can_use_q4_dot && b_q4_repack &&
                              (B.is_q4_repacked || w4_q4_repack_enabled);
        bool can_use_q4_bg128 =
            can_use_q4_dot && b_q4_g128 && group_size == 128 && (K % 128 == 0) &&
            (w4_q8_dot_bg128_enabled || has_direct_q4_g128);
        if (M == 1 && can_use_q4_dot) {
            bool use_q4_gemv_bg128 = can_use_q4_bg128;
            const char* path = use_q4_gemv_bg128
                ? "q4dot_gemv_bg128"
                : (use_q4_repack ? "q4dot_gemv_repack" : "q4dot_gemv");
            _timer.set_shape(path, M, N, K, group_size, groups_per_row,
                             use_q4_gemv_bg128 || use_q4_repack, false, n_threads);
            static thread_local Q4GemvScratch scratch;
            quantize_a_q8_blocks_even_odd(
                a_ptr, K, scratch.qA_even, scratch.qA_odd, scratch.a_scales);
            const int8_t* qA_even_data = scratch.qA_even.data();
            const int8_t* qA_odd_data = scratch.qA_odd.data();
            const float* a_scales_data = scratch.a_scales.data();
            if (!use_parallel) {
                if (use_q4_gemv_bg128) {
                    matmul_int4_q8dot_neon_gemv_g128_range(
                        qA_even_data, qA_odd_data, a_scales_data,
                        b_q4_g128, c_ptr, K, 0, N);
                } else {
                    matmul_int4_q8dot_neon_gemv_range(
                        nullptr, qA_even_data, qA_odd_data, a_scales_data, b_int4,
                        use_q4_repack ? b_q4_repack : nullptr, scales,
                        group_size, groups_per_row, c_ptr, K, K_weight, 0, N);
                }
            } else {
                int n_chunk = std::max(N / (n_threads * 8), 64);
                n_chunk = ((n_chunk + 7) / 8) * 8;
                thread_pool->parallel_for(0, N, n_chunk,
                    [&](int, int n_begin, int n_end) {
                        if (use_q4_gemv_bg128) {
                            matmul_int4_q8dot_neon_gemv_g128_range(
                                qA_even_data, qA_odd_data, a_scales_data,
                                b_q4_g128, c_ptr, K, n_begin, n_end);
                        } else {
                            matmul_int4_q8dot_neon_gemv_range(
                                nullptr, qA_even_data, qA_odd_data,
                                a_scales_data, b_int4,
                                use_q4_repack ? b_q4_repack : nullptr, scales,
                                group_size, groups_per_row, c_ptr, K, K_weight,
                                n_begin, n_end);
                        }
                    });
            }
            if (act != Activation::NONE && act_n_len != 0) {
                apply_activation_to_range_gemv(c_ptr, N, act, act_n_begin, act_n_len);
            }
            return;
        }
        if (can_use_q4_dot && (!w4_q8_dot_gemm_disabled || has_direct_q4_g128)) {
            bool force_4x8 = w4_q8_dot_gemm_4x8_enabled && !has_direct_q4_g128;
#if defined(__ARM_FEATURE_MATMUL_INT8)
            static const bool w4_i8mm_enabled = env_flag_enabled("MOLLM_W4_I8MM") &&
                                                !env_flag_enabled("MOLLM_W4_NO_I8MM");
            bool use_q4_dot_gemm_i8mm =
                use_q4_repack && w4_i8mm_enabled && !force_4x8;
#else
            bool use_q4_dot_gemm_i8mm = false;
#endif
            static const bool w4_q8_dot_gemm_a4_enabled =
                !env_flag_enabled("MOLLM_W4_NO_PACKED_A4");
            bool use_q4_dot_gemm_a4 = w4_q8_dot_gemm_a4_enabled &&
                                      !force_4x8;
            bool use_q4_dot_gemm_bg128 =
                can_use_q4_bg128 && !use_q4_dot_gemm_i8mm && !force_4x8;
            static const bool w4_gemm_2d_enabled = env_flag_enabled("MOLLM_W4_GEMM_2D");
            static const bool w4_gemm_1d_forced = env_flag_enabled("MOLLM_W4_GEMM_1D");
            const char* path = use_q4_dot_gemm_i8mm
                ? (use_q4_dot_gemm_a4
                    ? "q4dot_gemm_repack_i8mm_a4"
                    : "q4dot_gemm_repack_i8mm")
                : (use_q4_dot_gemm_bg128
                    ? (use_q4_dot_gemm_a4 ? "q4dot_gemm_bg128_a4" : "q4dot_gemm_bg128")
                    : (use_q4_dot_gemm_a4
                        ? (use_q4_repack ? "q4dot_gemm_repack_a4" : "q4dot_gemm_a4")
                        : (use_q4_repack ? "q4dot_gemm_repack" : "q4dot_gemm")));
            _timer.set_shape(path, M, N, K, group_size, groups_per_row,
                             use_q4_repack || use_q4_dot_gemm_bg128, false, n_threads);
            std::vector<float> a_scales;
            std::vector<int8_t> qA;
            std::vector<Q8A4Block> qA4;
            if (use_q4_dot_gemm_a4) {
                quantize_a_q8_blocks_a4(a_ptr, M, K, lda, qA4);
            } else {
                quantize_a_q8_blocks(a_ptr, M, K, lda, K, qA, a_scales);
            }
            const int8_t* qA_data = qA.data();
            const float* a_scales_data = a_scales.data();
            const Q8A4Block* qA4_data = qA4.data();

            auto run_q4_gemm = [&](int m_begin, int m_end,
                                   int n_begin, int n_end) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
                if (use_q4_dot_gemm_i8mm) {
                    if (use_q4_dot_gemm_a4) {
                        matmul_int4_q8dot_neon_4x8_repacked_i8mm_range<true>(
                            nullptr, nullptr,
                            b_q4_repack,
                            scales, group_size, groups_per_row,
                            c_ptr, M, N, K, K, ldc, m_begin, m_end, qA4_data);
                    } else {
                        matmul_int4_q8dot_neon_4x8_repacked_i8mm_range<false>(
                            qA_data, a_scales_data,
                            b_q4_repack,
                            scales, group_size, groups_per_row,
                            c_ptr, M, N, K, K, ldc, m_begin, m_end, nullptr);
                    }
                    return;
                } else
#endif
                if (force_4x8) {
                    matmul_int4_q8dot_neon_4x8_range(
                        qA_data, a_scales_data, b_int4,
                        use_q4_repack ? b_q4_repack : nullptr, scales,
                        group_size, groups_per_row, c_ptr, M, N, K, K, K_weight, ldc,
                        m_begin, m_end);
                } else if (use_q4_dot_gemm_bg128) {
                    if (use_q4_dot_gemm_a4) {
                        matmul_int4_q8dot_neon_8x8_g128packed_range<true>(
                            nullptr, nullptr, b_q4_g128,
                            c_ptr, M, N, K, K, ldc, m_begin, m_end,
                            n_begin, n_end, qA4_data);
                    } else {
                        matmul_int4_q8dot_neon_8x8_g128packed_range<false>(
                            qA_data, a_scales_data, b_q4_g128,
                            c_ptr, M, N, K, K, ldc, m_begin, m_end,
                            n_begin, n_end, nullptr);
                    }
                } else if (use_q4_dot_gemm_a4) {
                    matmul_int4_q8dot_neon_8x8_range<true>(
                        nullptr, nullptr, b_int4,
                        use_q4_repack ? b_q4_repack : nullptr, scales,
                        group_size, groups_per_row, c_ptr, M, N, K, K, K_weight, ldc,
                        m_begin, m_end, n_begin, n_end, qA4_data);
                } else {
                    matmul_int4_q8dot_neon_8x8_range<false>(
                        qA_data, a_scales_data, b_int4,
                        use_q4_repack ? b_q4_repack : nullptr, scales,
                        group_size, groups_per_row, c_ptr, M, N, K, K, K_weight, ldc,
                        m_begin, m_end, n_begin, n_end, nullptr);
                }
            };

            if (!use_parallel) {
                run_q4_gemm(0, M, 0, N);
            } else {
                bool use_q4_gemm_2d = !w4_gemm_1d_forced &&
                                      w4_gemm_2d_enabled &&
                                      !use_q4_dot_gemm_i8mm &&
                                      !force_4x8;
                if (use_q4_gemm_2d) {
                    static const int w4_gemm_n_block = env_int_or("MOLLM_W4_GEMM_N_BLOCK", 1024);
                    int n_block = ((w4_gemm_n_block + 7) / 8) * 8;
                    if (n_block > N) n_block = ((N + 7) / 8) * 8;
                    thread_pool->parallel_for_2d(
                        M, tile_m, N, n_block,
                        [&](int, int m_begin, int m_end, int n_begin, int n_end) {
                            run_q4_gemm(m_begin, m_end, n_begin, n_end);
                        });
                } else {
                    thread_pool->parallel_for(0, M, tile_m,
                        [&](int, int m_begin, int m_end) {
                            run_q4_gemm(m_begin, m_end, 0, N);
                        });
                }
            }
            if (act != Activation::NONE && act_n_len != 0) {
                apply_activation_to_range(c_ptr, M, N, ldc, 0, M, act, act_n_begin, act_n_len);
            }
            return;
        }
#endif

        _timer.set_shape("int4_scalar", M, N, K, group_size, groups_per_row,
                         false, false, n_threads);
        if (!use_parallel) {
            matmul_int4_scalar_range(a_ptr, b_int4, scales, group_size, groups_per_row,
                                     c_ptr, M, N, K, lda, K_weight, ldc, 0, M, 0, N);
        } else if (shard_by_n) {
            thread_pool->parallel_for(0, N, chunk_size,
                                      [&](int, int n_begin, int n_end) {
                                          matmul_int4_scalar_range(a_ptr, b_int4, scales,
                                                                   group_size, groups_per_row,
                                                                   c_ptr, M, N, K, lda, K_weight, ldc,
                                                                   0, M, n_begin, n_end);
                                      });
        } else {
            thread_pool->parallel_for(0, M, chunk_size,
                                      [&](int, int m_begin, int m_end) {
                                          matmul_int4_scalar_range(a_ptr, b_int4, scales,
                                                                   group_size, groups_per_row,
                                                                   c_ptr, M, N, K, lda, K_weight, ldc,
                                                                   m_begin, m_end, 0, N);
                                      });
        }
        if (act != Activation::NONE && act_n_len != 0) {
            apply_activation_to_range(c_ptr, M, N, ldc, 0, M, act, act_n_begin, act_n_len);
        }
        return;
    }

    if (is_int8) {
        const float* scales = B.scales;
        int group_size = (int)B.group_size;
        int groups_per_row = (int)B.groups_per_row;
        bool b_interleaved = B.is_interleaved;
        const int8_t* b_q8_repack = reinterpret_cast<const int8_t*>(B.q8_repack_data);
        if (!scales || group_size <= 0 || groups_per_row <= 0) {
            _timer.set_shape("int8_invalid_scales", M, N, K, group_size, groups_per_row,
                             b_q8_repack != nullptr, b_interleaved,
                             thread_pool ? thread_pool->num_threads() : 1);
            return;
        }

        constexpr int tile_m = HAS_NEON ? 8 : 1;
        int n_threads = thread_pool ? thread_pool->num_threads() : 1;
        bool shard_by_n = (N > M * 8 && M == 1);
        int chunk_size = (M == 1 || N == 1) ? g_matmul_config.gemv_chunk_size : tile_m;
        int total_dim = shard_by_n ? N : M;
        int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
        bool use_parallel = n_threads > 1 && n_chunks > 1;

#if HAS_NEON
        static const bool w8_onfly_fp16_enabled = env_flag_enabled("MOLLM_W8_ONFLY_FP16");
        static const bool w8_q8_dot_enabled = !env_flag_enabled("MOLLM_W8_NO_Q8_DOT");
        static const bool w8_q8_dot_gemm_disabled = env_flag_enabled("MOLLM_W8_NO_Q8_DOT_GEMM");
        static const bool w8_q8_dot_gemm_legacy_enabled = env_flag_enabled("MOLLM_W8_Q8_DOT_GEMM");
        static const bool w8_q8_dot_repack_enabled =
            !env_flag_enabled("MOLLM_W8_NO_Q8_DOT_REPACK") &&
            !env_flag_enabled("MOLLM_W8_NO_Q8_REPACK");
        bool use_onfly_fp16 = b_interleaved && w8_onfly_fp16_enabled;
#if defined(__ARM_FEATURE_DOTPROD)
        bool can_use_q8_repack = w8_q8_dot_repack_enabled &&
                                 b_q8_repack && (group_size % W8_Q8_BLOCK == 0);
#else
        bool can_use_q8_repack = false;
#endif
        bool use_q8_dot_gemv = w8_q8_dot_enabled && (can_use_q8_repack || b_interleaved);
        bool use_q8_dot_gemm = w8_q8_dot_enabled && !w8_q8_dot_gemm_disabled &&
                               (can_use_q8_repack ||
                                (b_interleaved && w8_q8_dot_gemm_legacy_enabled));
        bool use_q8_dot_gemv_repack = use_q8_dot_gemv && can_use_q8_repack;
        bool use_q8_dot_gemm_repack = use_q8_dot_gemm && can_use_q8_repack;
        if (M == 1 && use_onfly_fp16) {
            _timer.set_shape("onfly_fp16_gemv", M, N, K, group_size, groups_per_row,
                             b_q8_repack != nullptr, b_interleaved, n_threads);
            debug_w8_path_once("onfly_fp16_gemv", M, N, K, group_size, b_q8_repack != nullptr);
            if (!use_parallel) {
                matmul_int8_neon_gemv_range_fp16acc(
                    a_ptr, b_int8, scales, group_size, groups_per_row,
                    c_ptr, K, 0, N);
            } else {
                int n_chunk = std::max(N / (n_threads * 8), 64);
                n_chunk = ((n_chunk + 7) / 8) * 8;
                thread_pool->parallel_for(0, N, n_chunk,
                    [&](int, int n_begin, int n_end) {
                        matmul_int8_neon_gemv_range_fp16acc(
                            a_ptr, b_int8, scales, group_size, groups_per_row,
                            c_ptr, K, n_begin, n_end);
                    });
            }
            if (act != Activation::NONE && act_n_len != 0) {
                apply_activation_to_range_gemv(c_ptr, N, act, act_n_begin, act_n_len);
            }
            return;
        }
        if (use_onfly_fp16) {
            _timer.set_shape("onfly_fp16_gemm", M, N, K, group_size, groups_per_row,
                             b_q8_repack != nullptr, b_interleaved, n_threads);
            debug_w8_path_once("onfly_fp16_gemm", M, N, K, group_size, b_q8_repack != nullptr);
            if (!use_parallel) {
                __fp16* a_packed = pack_a_interleaved_full(a_ptr, M, K, lda);
                matmul_int8_neon_8x8_range_packed_a_fp16acc(
                    a_packed, b_int8, scales, group_size, groups_per_row,
                    c_ptr, M, N, K, ldc, 0, M);
                delete[] a_packed;
            } else {
                int m_tiles = (M + tile_m - 1) / tile_m;
                std::vector<__fp16*> a_packed_tiles(m_tiles, nullptr);
                for (int i = 0; i < m_tiles; i++) {
                    int m_begin = i * tile_m;
                    int m_len = std::min(tile_m, M - m_begin);
                    a_packed_tiles[i] = pack_a_interleaved_full(
                        a_ptr + m_begin * lda, m_len, K, lda);
                }
                thread_pool->parallel_for(0, M, tile_m,
                    [&](int, int m_begin, int m_end) {
                        int tile_idx = m_begin / tile_m;
                        int m_len = m_end - m_begin;
                        matmul_int8_neon_8x8_range_packed_a_fp16acc(
                            a_packed_tiles[tile_idx], b_int8, scales,
                            group_size, groups_per_row,
                            c_ptr + m_begin * ldc, m_len, N, K, ldc, 0, m_len);
                    });
                for (int i = 0; i < m_tiles; i++) delete[] a_packed_tiles[i];
            }
            if (act != Activation::NONE && act_n_len != 0) {
                apply_activation_to_range(c_ptr, M, N, ldc, 0, M, act, act_n_begin, act_n_len);
            }
            return;
        }
        if (M == 1 && use_q8_dot_gemv) {
            const char* path = use_q8_dot_gemv_repack ? "q8dot_gemv_repack" : "q8dot_gemv";
            _timer.set_shape(path, M, N, K, group_size, groups_per_row,
                             b_q8_repack != nullptr, b_interleaved, n_threads);
            debug_w8_path_once(path,
                               M, N, K, group_size, b_q8_repack != nullptr);
            std::vector<int8_t> qA;
            std::vector<float> a_scales;
            int K_padded = ((K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK) * W8_Q8_BLOCK;
            quantize_a_q8_blocks(a_ptr, M, K, lda,
                                 use_q8_dot_gemv_repack ? K_padded : K,
                                 qA, a_scales);
            const int8_t* qA_data = qA.data();
            const float* a_scales_data = a_scales.data();
            if (!use_parallel) {
#if defined(__ARM_FEATURE_DOTPROD)
                if (use_q8_dot_gemv_repack) {
                    matmul_int8_q8dot_neon_gemv_repacked_range(
                        qA_data, a_scales_data, b_q8_repack, scales,
                        group_size, groups_per_row, c_ptr, K, K_padded, 0, N);
                } else
#endif
                {
                    matmul_int8_q8dot_neon_gemv_range(
                        qA_data, a_scales_data, b_int8, scales,
                        group_size, groups_per_row, c_ptr, K, 0, N);
                }
            } else {
                int n_chunk = std::max(N / (n_threads * 8), 64);
                n_chunk = ((n_chunk + 7) / 8) * 8;
                thread_pool->parallel_for(0, N, n_chunk,
                    [&](int, int n_begin, int n_end) {
#if defined(__ARM_FEATURE_DOTPROD)
                        if (use_q8_dot_gemv_repack) {
                            matmul_int8_q8dot_neon_gemv_repacked_range(
                                qA_data, a_scales_data, b_q8_repack, scales,
                                group_size, groups_per_row, c_ptr, K, K_padded, n_begin, n_end);
                        } else
#endif
                        {
                            matmul_int8_q8dot_neon_gemv_range(
                                qA_data, a_scales_data, b_int8, scales,
                                group_size, groups_per_row, c_ptr, K, n_begin, n_end);
                        }
                    });
            }
            if (act != Activation::NONE && act_n_len != 0) {
                apply_activation_to_range_gemv(c_ptr, N, act, act_n_begin, act_n_len);
            }
            return;
        }
        if (use_q8_dot_gemm) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
            static const bool w8_i8mm_disabled = env_flag_enabled("MOLLM_W8_NO_I8MM");
            static const bool w8_i8mm_4x8_forced = env_flag_enabled("MOLLM_W8_I8MM_4X8");
            bool use_q8_dot_gemm_i8mm = use_q8_dot_gemm_repack && !w8_i8mm_disabled;
            bool use_q8_dot_gemm_i8mm_8x8 = use_q8_dot_gemm_i8mm && !w8_i8mm_4x8_forced;
#else
            bool use_q8_dot_gemm_i8mm = false;
            bool use_q8_dot_gemm_i8mm_8x8 = false;
#endif
            static const bool w8_gemm_1d_forced = env_flag_enabled("MOLLM_W8_GEMM_1D");
            const char* path = use_q8_dot_gemm_i8mm_8x8
                ? "q8dot_gemm_repack_i8mm_8x8"
                : (use_q8_dot_gemm_i8mm
                    ? "q8dot_gemm_repack_i8mm"
                    : (use_q8_dot_gemm_repack ? "q8dot_gemm_repack" : "q8dot_gemm"));
            _timer.set_shape(path, M, N, K, group_size, groups_per_row,
                             b_q8_repack != nullptr, b_interleaved, n_threads);
            debug_w8_path_once(path,
                               M, N, K, group_size, b_q8_repack != nullptr);
            std::vector<int8_t> qA;
            std::vector<float> a_scales;
            int K_padded = ((K + W8_Q8_BLOCK - 1) / W8_Q8_BLOCK) * W8_Q8_BLOCK;
            quantize_a_q8_blocks(a_ptr, M, K, lda,
                                 use_q8_dot_gemm_repack ? K_padded : K,
                                 qA, a_scales);
            const int8_t* qA_data = qA.data();
            const float* a_scales_data = a_scales.data();
            auto run_legacy_q8_gemm = [&](int m_begin, int m_end) {
                matmul_int8_q8dot_neon_4x8_range(
                    qA_data, a_scales_data, b_int8, scales,
                    group_size, groups_per_row, c_ptr, M, N, K, ldc,
                    m_begin, m_end);
            };
#if defined(__ARM_FEATURE_DOTPROD)
            auto run_repacked_q8_gemm = [&](int m_begin, int m_end,
                                            int n_begin, int n_end) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
                if (use_q8_dot_gemm_i8mm_8x8) {
                    matmul_int8_q8dot_neon_8x8_repacked_i8mm_range(
                        qA_data, a_scales_data, b_q8_repack, scales,
                        group_size, groups_per_row, c_ptr, M, N, K, K_padded, ldc,
                        m_begin, m_end, n_begin, n_end);
                    return;
                }
                if (use_q8_dot_gemm_i8mm) {
                    matmul_int8_q8dot_neon_4x8_repacked_i8mm_range(
                        qA_data, a_scales_data, b_q8_repack, scales,
                        group_size, groups_per_row, c_ptr, M, N, K, K_padded, ldc,
                        m_begin, m_end, n_begin, n_end);
                    return;
                }
#endif
                matmul_int8_q8dot_neon_4x8_repacked_range(
                    qA_data, a_scales_data, b_q8_repack, scales,
                    group_size, groups_per_row, c_ptr, M, N, K, K_padded, ldc,
                    m_begin, m_end, n_begin, n_end);
            };
#endif
            if (!use_parallel) {
#if defined(__ARM_FEATURE_DOTPROD)
                if (use_q8_dot_gemm_repack) {
                    run_repacked_q8_gemm(0, M, 0, N);
                } else
#endif
                {
                    run_legacy_q8_gemm(0, M);
                }
            } else {
#if defined(__ARM_FEATURE_DOTPROD)
                if (use_q8_dot_gemm_repack && !w8_gemm_1d_forced) {
                    static const int w8_gemm_n_block = env_int_or("MOLLM_W8_GEMM_N_BLOCK", 1024);
                    int n_block = ((w8_gemm_n_block + 7) / 8) * 8;
                    if (n_block > N) n_block = ((N + 7) / 8) * 8;
                    thread_pool->parallel_for_2d(
                        M, tile_m, N, n_block,
                        [&](int, int m_begin, int m_end, int n_begin, int n_end) {
                            run_repacked_q8_gemm(m_begin, m_end, n_begin, n_end);
                        });
                } else
#endif
                {
                    thread_pool->parallel_for(0, M, tile_m,
                        [&](int, int m_begin, int m_end) {
#if defined(__ARM_FEATURE_DOTPROD)
                            if (use_q8_dot_gemm_repack) {
                                run_repacked_q8_gemm(m_begin, m_end, 0, N);
                            } else
#endif
                            {
                                run_legacy_q8_gemm(m_begin, m_end);
                            }
                        });
                }
            }
            if (act != Activation::NONE && act_n_len != 0) {
                apply_activation_to_range(c_ptr, M, N, ldc, 0, M, act, act_n_begin, act_n_len);
            }
            return;
        }
        if (M == 1 && b_interleaved) {
            _timer.set_shape("native_w8_gemv", M, N, K, group_size, groups_per_row,
                             b_q8_repack != nullptr, b_interleaved, n_threads);
            debug_w8_path_once("native_w8_gemv", M, N, K, group_size, b_q8_repack != nullptr);
            if (!use_parallel) {
                matmul_int8_neon_gemv_range(a_ptr, b_int8, scales, group_size, groups_per_row,
                                             c_ptr, K, 0, N);
            } else {
                int n_chunk = std::max(N / (n_threads * 8), 64);
                n_chunk = ((n_chunk + 7) / 8) * 8;
                thread_pool->parallel_for(0, N, n_chunk,
                    [&](int, int n_begin, int n_end) {
                        matmul_int8_neon_gemv_range(a_ptr, b_int8, scales,
                                                     group_size, groups_per_row,
                                                     c_ptr, K, n_begin, n_end);
                    });
            }
            if (act != Activation::NONE && act_n_len != 0) {
                apply_activation_to_range_gemv(c_ptr, N, act, act_n_begin, act_n_len);
            }
            return;
        }
        if (b_interleaved) {
            _timer.set_shape("native_w8_gemm", M, N, K, group_size, groups_per_row,
                             b_q8_repack != nullptr, b_interleaved, n_threads);
            debug_w8_path_once("native_w8_gemm", M, N, K, group_size, b_q8_repack != nullptr);
            if (!use_parallel) {
                matmul_int8_neon_4x8_range(a_ptr, b_int8, scales, group_size, groups_per_row,
                                            c_ptr, M, N, K, lda, ldc, 0, M);
            } else {
                thread_pool->parallel_for(0, M, tile_m,
                    [&](int, int m_begin, int m_end) {
                        matmul_int8_neon_4x8_range(a_ptr, b_int8, scales,
                                                    group_size, groups_per_row,
                                                    c_ptr, M, N, K, lda, ldc,
                                                    m_begin, m_end);
                    });
            }
            if (act != Activation::NONE && act_n_len != 0) {
                apply_activation_to_range(c_ptr, M, N, ldc, 0, M, act, act_n_begin, act_n_len);
            }
            return;
        }
#endif

        static const bool w8_q8_dot_enabled_scalar = !env_flag_enabled("MOLLM_W8_NO_Q8_DOT");
        static const bool w8_q8_dot_gemm_enabled_scalar = env_flag_enabled("MOLLM_W8_Q8_DOT_GEMM");
        if (b_interleaved && w8_q8_dot_enabled_scalar &&
            (M == 1 || w8_q8_dot_gemm_enabled_scalar)) {
            _timer.set_shape("q8dot_scalar", M, N, K, group_size, groups_per_row,
                             b_q8_repack != nullptr, b_interleaved, n_threads);
            debug_w8_path_once("q8dot_scalar", M, N, K, group_size, b_q8_repack != nullptr);
            std::vector<int8_t> qA;
            std::vector<float> a_scales;
            quantize_a_q8_blocks(a_ptr, M, K, lda, K, qA, a_scales);
            const int8_t* qA_data = qA.data();
            const float* a_scales_data = a_scales.data();
            if (!use_parallel) {
                matmul_int8_q8dot_scalar_range(
                    qA_data, a_scales_data, b_int8, scales,
                    group_size, groups_per_row, c_ptr, M, N, K,
                    K_weight, ldc, 0, M, 0, N, b_interleaved);
            } else if (shard_by_n) {
                thread_pool->parallel_for(0, N, chunk_size,
                    [&](int, int n_begin, int n_end) {
                        matmul_int8_q8dot_scalar_range(
                            qA_data, a_scales_data, b_int8, scales,
                            group_size, groups_per_row, c_ptr, M, N, K,
                            K_weight, ldc, 0, M, n_begin, n_end, b_interleaved);
                    });
            } else {
                thread_pool->parallel_for(0, M, chunk_size,
                    [&](int, int m_begin, int m_end) {
                        matmul_int8_q8dot_scalar_range(
                            qA_data, a_scales_data, b_int8, scales,
                            group_size, groups_per_row, c_ptr, M, N, K,
                            K_weight, ldc, m_begin, m_end, 0, N, b_interleaved);
                    });
            }
            if (act != Activation::NONE && act_n_len != 0) {
                apply_activation_to_range(c_ptr, M, N, ldc, 0, M, act, act_n_begin, act_n_len);
            }
            return;
        }

        _timer.set_shape("int8_scalar", M, N, K, group_size, groups_per_row,
                         b_q8_repack != nullptr, b_interleaved, n_threads);
        if (!use_parallel) {
            debug_w8_path_once("scalar", M, N, K, group_size, b_q8_repack != nullptr);
            matmul_int8_scalar_range(a_ptr, b_int8, scales, group_size, groups_per_row,
                                     c_ptr, M, N, K, lda, K_weight, ldc, 0, M, 0, N,
                                     b_interleaved);
        } else if (shard_by_n) {
            thread_pool->parallel_for(0, N, chunk_size,
                                      [&](int, int n_begin, int n_end) {
                                          matmul_int8_scalar_range(a_ptr, b_int8, scales,
                                                                   group_size, groups_per_row,
                                                                   c_ptr, M, N, K, lda, K_weight, ldc,
                                                                   0, M, n_begin, n_end,
                                                                   b_interleaved);
                                      });
        } else {
            thread_pool->parallel_for(0, M, chunk_size,
                                      [&](int, int m_begin, int m_end) {
                                          matmul_int8_scalar_range(a_ptr, b_int8, scales,
                                                                   group_size, groups_per_row,
                                                                   c_ptr, M, N, K, lda, K_weight, ldc,
                                                                   m_begin, m_end, 0, N,
                                                                   b_interleaved);
                                      });
        }
        if (act != Activation::NONE && act_n_len != 0) {
            apply_activation_to_range(c_ptr, M, N, ldc, 0, M, act, act_n_begin, act_n_len);
        }
        return;
    }

    // ---- Interleaved packing path (FP16 + NEON) ----
    // B is pre-packed at load time (engine) or by the caller (bench/test).
    // A is packed per-call (small overhead, M×K << N×K).
    // - M >= 8: lane-FMA kernel (A+B both packed)
    //   - FP16 accumulate: vfmaq_lane_f16 (2x throughput, default)
    //   - FP32 widen accumulate: vfmlalq_laneq_f16 (precision fallback)
    // - M <  8: scalar-A kernel (vfmaq_n_f32, only B packed) — GEMV path
    // Accelerate's SGEMM reaches the Apple AMX path for large prefill
    // matrices. Like llama.cpp's BLAS backend, convert the immutable FP16
    // weight to a temporary FP32 plane per call. mollm's NEON kernel wins for
    // short prompts; model-level sweeps put the crossover at roughly M=96.
#if defined(__APPLE__)
    static const bool accelerate_gemm = !env_flag_enabled("MOLLM_NO_ACCELERATE_GEMM");
    static const int accelerate_min_m=env_int_or("MOLLM_ACCELERATE_GEMM_MIN_M",96);
    if(accelerate_gemm && is_fp16 && B.rowmajor_data &&
       M>=accelerate_min_m && N>=32 && K>=32) {
        int n_threads=thread_pool?thread_pool->num_threads():1;
        _timer.set_shape("accelerate_sgemm",M,N,K,0,0,false,false,n_threads);
        const __fp16* src=reinterpret_cast<const __fp16*>(B.rowmajor_data);
        std::vector<float> bf((size_t)N*K);
        auto convert_rows=[&](int,int begin,int end) {
            for(int n=begin;n<end;++n) {
                const __fp16* in=src+(size_t)n*K;
                float* out=bf.data()+(size_t)n*K;
                int k=0;
#if HAS_NEON
                for(;k+7<K;k+=8) {
                    float16x8_t h=vld1q_f16(in+k);
                    vst1q_f32(out+k,vcvt_f32_f16(vget_low_f16(h)));
                    vst1q_f32(out+k+4,vcvt_f32_f16(vget_high_f16(h)));
                }
#endif
                for(;k<K;++k) out[k]=(float)in[k];
            }
        };
        if(thread_pool&&n_threads>1) {
            int chunk=std::max(1,(N+n_threads-1)/n_threads);
            thread_pool->parallel_for(0,N,chunk,convert_rows);
        } else convert_rows(0,0,N);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                    M,N,K,1.f,a_ptr,lda,bf.data(),K,0.f,c_ptr,ldc);
        if(act!=Activation::NONE&&act_n_len!=0) {
            if(thread_pool&&n_threads>1) {
                int chunk=std::max(1,(M+n_threads-1)/n_threads);
                thread_pool->parallel_for(0,M,chunk,[&](int,int begin,int end) {
                    apply_activation_to_range(c_ptr,M,N,ldc,begin,end,
                                              act,act_n_begin,act_n_len);
                });
            } else apply_activation_to_range(c_ptr,M,N,ldc,0,M,act,act_n_begin,act_n_len);
        }
        return;
    }
#endif
    bool use_interleave = is_fp16 && HAS_NEON && !is_repacked
                       && g_matmul_config.use_interleave_pack;
    bool use_lane_fma = use_interleave && (M >= 8);
    bool use_fp16_acc = g_matmul_config.use_fp16_accumulate && !g_mollm_force_fp32_acc;

    // Select lane-FMA kernel based on accumulation mode.
    // `act`/`act_n_begin`/`act_n_len` are in local shard coords (caller already
    // translated global → local).
    auto dispatch_lane_fma = [&](const __fp16* a_packed, const __fp16* b_packed,
                                 float* c, int m_len, int n_len,
                                 int ldc_, int m_begin_, int m_end_,
                                 Activation act_, int act_n_begin_, int act_n_len_) {
        if (use_fp16_acc) {
            matmul_fp16_neon_8x8_range_packed_a_fp16acc(
                a_packed, b_packed, c, m_len, n_len, K, ldc_, m_begin_, m_end_,
                act_, act_n_begin_, act_n_len_);
        } else {
            // FP32 acc lane-FMA kernel — activation not yet supported here.
            // Fall back: run kernel without act, then apply act in a separate pass.
            matmul_fp16_neon_8x8_range_packed_a(
                a_packed, b_packed, c, m_len, n_len, K, ldc_, m_begin_, m_end_);
            if (act_ != Activation::NONE && act_n_len_ != 0) {
                apply_activation_to_range(c, m_len, n_len, ldc_, m_begin_, m_end_,
                                          act_, act_n_begin_, act_n_len_);
            }
        }
    };

    // Helper: apply activation post-hoc to a [M, N] C row-major block,
    // for columns in [act_n_begin, act_n_begin + act_n_len).
    // Used by FP32 acc fallback kernels that don't fuse activation in writeback.
    // (Defined below as a lambda for closure over nothing — pure function.)

    if (use_interleave) {
        constexpr int tile_m = HAS_NEON ? 8 : 1;
        int n_threads = thread_pool ? thread_pool->num_threads() : 1;
        bool shard_by_n = (N > M * 8 && M == 1);
        int chunk_size = tile_m;
        if (M == 1 || N == 1) chunk_size = g_matmul_config.gemv_chunk_size;

        int total_dim = shard_by_n ? N : M;
        int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
        bool use_parallel = n_threads > 1 && n_chunks > 1;

        const __fp16* b_packed = b_fp16;

        // ---- GEMV path: M == 1, dedicated kernel ----
        if (M == 1) {
            _timer.set_shape(use_fp16_acc ? "fp16_gemv_interleaved_fp16acc"
                                          : "fp16_gemv_interleaved_fp32acc",
                             M, N, K, 0, 0, false, false, n_threads);
            if (use_fp16_acc) {
                if (!use_parallel) {
                    matmul_fp16_neon_gemv_range_fp16acc(a_ptr, b_packed, c_ptr, K, 0, N,
                                                         act, act_n_begin, act_n_len);
                } else {
                    int n_chunk = std::max(N / (n_threads * 8), 64);
                    n_chunk = ((n_chunk + 7) / 8) * 8;
                    thread_pool->parallel_for(0, N, n_chunk,
                        [&](int, int n_begin, int n_end) {
                            // Translate global act range to local shard coords.
                            // GEMV kernel uses absolute n indices (C + n_begin offset
                            // is added by kernel itself), so pass global range
                            // intersected with shard.
                            int local_act_begin = std::max(0, act_n_begin - n_begin);
                            int local_act_end;
                            if (act_n_len < 0) {
                                local_act_end = n_end - n_begin;
                            } else {
                                local_act_end = std::min(n_end - n_begin,
                                                          act_n_begin + act_n_len - n_begin);
                            }
                            int local_act_len = (act_n_len < 0) ? -1 :
                                                  std::max(0, local_act_end - local_act_begin);
                            matmul_fp16_neon_gemv_range_fp16acc(a_ptr, b_packed, c_ptr,
                                                                  K, n_begin, n_end,
                                                                  act, n_begin + local_act_begin,
                                                                  local_act_len);
                        });
                }
            } else {
                if (!use_parallel) {
                    matmul_fp16_neon_gemv_range(a_ptr, b_packed, c_ptr, K, 0, N);
                    if (act != Activation::NONE && act_n_len != 0) {
                        apply_activation_to_range_gemv(c_ptr, N, act, act_n_begin, act_n_len);
                    }
                } else {
                    int n_chunk = std::max(N / (n_threads * 8), 64);
                    n_chunk = ((n_chunk + 7) / 8) * 8;
                    thread_pool->parallel_for(0, N, n_chunk,
                        [&](int, int n_begin, int n_end) {
                            matmul_fp16_neon_gemv_range(a_ptr, b_packed, c_ptr,
                                                        K, n_begin, n_end);
                            int local_act_begin = std::max(0, act_n_begin - n_begin);
                            int local_act_end;
                            if (act_n_len < 0) {
                                local_act_end = n_end - n_begin;
                            } else {
                                local_act_end = std::min(n_end - n_begin,
                                                          act_n_begin + act_n_len - n_begin);
                            }
                            int local_act_len = (act_n_len < 0) ? -1 :
                                                  std::max(0, local_act_end - local_act_begin);
                            apply_activation_to_range_gemv(c_ptr + n_begin, n_end - n_begin,
                                                             act, local_act_begin, local_act_len);
                        });
                }
            }
            return;
        }

        _timer.set_shape(use_lane_fma
                             ? (use_fp16_acc ? "fp16_gemm_interleaved_fp16acc"
                                             : "fp16_gemm_interleaved_fp32acc")
                             : "fp16_gemm_interleaved_scalar_a",
                         M, N, K, 0, 0, false, false, n_threads);

        __fp16* a_packed = nullptr;
        if (use_lane_fma && !use_parallel) {
            a_packed = pack_a_interleaved_full(a_ptr, M, K, lda);
        }

        if (!use_parallel) {
            if (use_lane_fma) {
                dispatch_lane_fma(a_packed, b_packed, c_ptr,
                                  M, N, ldc, 0, M,
                                  act, act_n_begin, act_n_len);
                delete[] a_packed;
            } else {
                matmul_fp16_neon_8x8_range_packed(
                    a_ptr, b_packed, c_ptr,
                    M, N, K, lda, ldc, 0, M);
                if (act != Activation::NONE && act_n_len != 0) {
                    apply_activation_to_range(c_ptr, M, N, ldc, 0, M,
                                                act, act_n_begin, act_n_len);
                }
            }
        } else if (shard_by_n) {
            // GEMV-ish: shard_by_n with M<8 (use the scalar-A kernel).
            int n_chunk = std::max(chunk_size, 8);
            n_chunk = ((n_chunk + 7) / 8) * 8;
            thread_pool->parallel_for(0, N, n_chunk,
                [&](int, int n_begin, int n_end) {
                    int n_begin_aligned = n_begin & ~7;
                    matmul_fp16_neon_8x8_range_packed(
                        a_ptr,
                        b_packed + n_begin_aligned * K,
                        c_ptr + n_begin,
                        M, n_end - n_begin,
                        K, lda, ldc, 0, M);
                    int local_act_begin = std::max(0, act_n_begin - n_begin);
                    int local_act_end;
                    if (act_n_len < 0) {
                        local_act_end = n_end - n_begin;
                    } else {
                        local_act_end = std::min(n_end - n_begin,
                                                  act_n_begin + act_n_len - n_begin);
                    }
                    int local_act_len = (act_n_len < 0) ? -1 :
                                          std::max(0, local_act_end - local_act_begin);
                    if (act != Activation::NONE && local_act_len != 0) {
                        apply_activation_to_range(c_ptr + n_begin, M, n_end - n_begin,
                                                    ldc, 0, M, act, local_act_begin, local_act_len);
                    }
                });
        } else {
            // GEMM (M>=8): 2D atomic-steal scheduling.
            // Jobs = (M/8) × (N/N_BLOCK), each job = 8 rows × N_BLOCK cols.
            // N_BLOCK=256 reduces job count + atomic contention.
            constexpr int N_BLOCK = 256;
            int n_block = ((N_BLOCK + 7) / 8) * 8;  // align to 8
            if (n_block > N) n_block = ((N + 7) / 8) * 8;

            // Pre-pack A per M-tile ONCE (not per job).
            int m_tiles = (M + tile_m - 1) / tile_m;
            std::vector<__fp16*> a_packed_tiles(m_tiles, nullptr);
            for (int i = 0; i < m_tiles; i++) {
                int m_begin = i * tile_m;
                int m_len = std::min(tile_m, M - m_begin);
                a_packed_tiles[i] = pack_a_interleaved_full(
                    a_ptr + m_begin * lda, m_len, K, lda);
            }

            thread_pool->parallel_for_2d(
                M, tile_m, N, n_block,
                [&](int, int m_begin, int m_end, int n_begin, int n_end) {
                    int m_len = m_end - m_begin;
                    int n_len = n_end - n_begin;
                    int tile_idx = m_begin / tile_m;
                    __fp16* a_slice_packed = a_packed_tiles[tile_idx];
                    int n_begin_aligned = n_begin & ~7;

                    // Translate global act range to local shard coords.
                    int local_act_begin = std::max(0, act_n_begin - n_begin);
                    int local_act_end;
                    if (act_n_len < 0) {
                        local_act_end = n_len;
                    } else {
                        local_act_end = std::min(n_len,
                                                  act_n_begin + act_n_len - n_begin);
                    }
                    int local_act_len = (act_n_len < 0) ? -1 :
                                          std::max(0, local_act_end - local_act_begin);

                    dispatch_lane_fma(a_slice_packed,
                                      b_packed + n_begin_aligned * K,
                                      c_ptr + m_begin * ldc + n_begin,
                                      m_len, n_len, ldc, 0, m_len,
                                      act, local_act_begin, local_act_len);
                });

            // Free pre-packed A slices.
            for (int i = 0; i < m_tiles; i++) {
                if (a_packed_tiles[i]) delete[] a_packed_tiles[i];
            }
        }
        return;
    }

    // ---- Standard path (FP32 or non-packed FP16) ----

    constexpr int tile_m = HAS_NEON ? 8 : 1;
    int n_threads = thread_pool ? thread_pool->num_threads() : 1;

    // Decide sharding dimension adaptively, similar to ggml:
    //   If N >> M, shard by N (e.g. lm_head: M=1, N=vocab_size).
    //   Otherwise shard by M (the common case).
    bool shard_by_n = (N > M * 8 && M == 1);

    // Decide chunk size adaptively.
    // For GEMV-like shapes, use a larger chunk to reduce per-chunk overhead.
    int chunk_size = tile_m;
    if (M == 1 || N == 1) {
        chunk_size = g_matmul_config.gemv_chunk_size;
    }

    int total_dim = shard_by_n ? N : M;
    int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
    bool use_parallel = n_threads > 1 && n_chunks > 1;

    _timer.set_shape(is_fp16 ? (is_repacked ? "fp16_standard_repacked" : "fp16_standard")
                             : "fp32_standard",
                     M, N, K, 0, 0, false, false, n_threads);

    if (!use_parallel) {
        if (is_fp16) {
            matmul_fp16_range(a_ptr, b_fp16, c_ptr, M, N, K, lda, K_weight, ldc, 0, M);
        } else {
            matmul_fp32_range(a_ptr, b_fp32, c_ptr, M, N, K, lda, K_weight, ldc, 0, M);
        }
        if (act != Activation::NONE && act_n_len != 0) {
            apply_activation_to_range(c_ptr, M, N, ldc, 0, M, act, act_n_begin, act_n_len);
        }
        return;
    }

    if (shard_by_n) {
        if (is_fp16) {
            thread_pool->parallel_for(0, N, chunk_size,
                                      [&](int, int n_begin, int n_end) {
                                          matmul_fp16_range(a_ptr, b_fp16 + n_begin * K_weight, c_ptr + n_begin,
                                                            M, n_end - n_begin, K, lda, K_weight, ldc, 0, M);
                                          int local_act_begin = std::max(0, act_n_begin - n_begin);
                                          int local_act_end;
                                          if (act_n_len < 0) {
                                              local_act_end = n_end - n_begin;
                                          } else {
                                              local_act_end = std::min(n_end - n_begin,
                                                                        act_n_begin + act_n_len - n_begin);
                                          }
                                          int local_act_len = (act_n_len < 0) ? -1 :
                                                                std::max(0, local_act_end - local_act_begin);
                                          if (act != Activation::NONE && local_act_len != 0) {
                                              apply_activation_to_range(c_ptr + n_begin, M, n_end - n_begin,
                                                                        ldc, 0, M, act, local_act_begin, local_act_len);
                                          }
                                      });
        } else {
            thread_pool->parallel_for(0, N, chunk_size,
                                      [&](int, int n_begin, int n_end) {
                                          matmul_fp32_range_n(a_ptr, b_fp32, c_ptr,
                                                              M, N, K, lda, K_weight, ldc,
                                                              n_begin, n_end);
                                          int local_act_begin = std::max(0, act_n_begin - n_begin);
                                          int local_act_end;
                                          if (act_n_len < 0) {
                                              local_act_end = n_end - n_begin;
                                          } else {
                                              local_act_end = std::min(n_end - n_begin,
                                                                        act_n_begin + act_n_len - n_begin);
                                          }
                                          int local_act_len = (act_n_len < 0) ? -1 :
                                                                std::max(0, local_act_end - local_act_begin);
                                          if (act != Activation::NONE && local_act_len != 0) {
                                              apply_activation_to_range(c_ptr + n_begin, M, n_end - n_begin,
                                                                        ldc, 0, M, act, local_act_begin, local_act_len);
                                          }
                                      });
        }
    } else {
        if (is_fp16) {
            thread_pool->parallel_for(0, M, chunk_size,
                                      [&](int, int m_begin, int m_end) {
                                          matmul_fp16_range(a_ptr, b_fp16, c_ptr,
                                                            M, N, K, lda, K_weight, ldc,
                                                            m_begin, m_end);
                                      });
        } else {
            thread_pool->parallel_for(0, M, chunk_size,
                                      [&](int, int m_begin, int m_end) {
                                          matmul_fp32_range(a_ptr, b_fp32, c_ptr,
                                                            M, N, K, lda, K_weight, ldc,
                                                            m_begin, m_end);
                                      });
        }
    }

    // M-sharded path: activation can't be applied per-shard (shards split M,
    // not N). Apply once after all shards complete.
    // (shard_by_n branches above already apply per-shard.)
    if (!shard_by_n && act != Activation::NONE && act_n_len != 0) {
        apply_activation_to_range(c_ptr, M, N, ldc, 0, M, act, act_n_begin, act_n_len);
    }
}

// Sparse-A decode GEMV used by RWKV FFN down projections. ReLU-squared makes
// roughly half of A exactly zero, so an output-tile outer-product traversal
// can omit both the weight load and FMA for those entries.
void kernel_gemv_sparse_a(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* thread_pool) {
    const int M = (int)A.shape[1];
    const int K = (int)A.shape[0];
    const int N = (int)B.shape[0];
    if (M != 1 || (int)B.shape[1] != K) {
        kernel_matmul_fp32(A, B, C, thread_pool);
        return;
    }

    const float* a = A.ptr<float>();
    std::vector<int> nonzero;
    nonzero.reserve(K / 2);
    for (int k = 0; k < K; ++k) {
        if (a[k] != 0.0f) nonzero.push_back(k);
    }
    if (std::getenv("MOLLM_SPARSE_DEBUG")) {
        static int debug_count = 0;
        if (debug_count++ < 48)
            std::fprintf(stderr, "sparseA precision=%d density=%.4f nnz=%zu K=%d\n",
                         (int)B.prec, (double)nonzero.size() / K, nonzero.size(), K);
    }
    // Quantized dot-product GEMV has a much lower dense cost than FP16 GEMV;
    // only use scalar-sparse dequantization for the exceptionally sparse
    // early FFN layers. FP16 crosses over at a much higher density.
    size_t density_limit = B.prec == Precision::FP16 ? (size_t)K * 4 / 5
        : (B.prec == Precision::INT4 ? (size_t)K * 3 / 100 : 0);
    if (std::getenv("MOLLM_SPARSE_A_FORCE")) density_limit = K;
    if (nonzero.size() >= density_limit) {
        kernel_matmul_fp32(A, B, C, thread_pool);
        return;
    }

#if HAS_NEON
    float* c = C.ptr<float>();
    const bool fp16 = B.prec == Precision::FP16;
    const bool w8 = B.prec == Precision::INT8 && B.sparse_data && B.scales;
    const bool w4 = B.prec == Precision::INT4 && B.sparse_data && B.scales &&
                    B.group_size == 128 && (K % 128) == 0;
    if (!fp16 && !w8 && !w4) {
        kernel_matmul_fp32(A, B, C, thread_pool);
        return;
    }

    const __fp16* b16 = fp16 ? reinterpret_cast<const __fp16*>(B.data) : nullptr;
    const int8_t* b8 = w8 ? reinterpret_cast<const int8_t*>(B.sparse_data) : nullptr;
    const int8_t* b4 = w4 ? reinterpret_cast<const int8_t*>(B.sparse_data) : nullptr;
    const int gpr = (int)B.groups_per_row;

    auto run_range = [&](int n_begin, int n_end) {
        for (int n = n_begin; n < n_end; n += 8) {
            int valid = std::min(8, N - n);
            float16x8_t acc[4] = {
                vdupq_n_f16((__fp16)0), vdupq_n_f16((__fp16)0),
                vdupq_n_f16((__fp16)0), vdupq_n_f16((__fp16)0)};
            int slot = 0;
            int last_group = -1;
            float16x8_t scale = vdupq_n_f16((__fp16)0);
            const __fp16* tile16 = fp16 ? b16 + (size_t)(n & ~7) * K : nullptr;
            const int8_t* tile8 = !fp16 ? (w8 ? b8 : b4) + (size_t)(n & ~7) * K : nullptr;
            for (int k : nonzero) {
                float16x8_t weight;
                if (fp16) {
                    weight = vld1q_f16(tile16 + (size_t)k * 8);
                } else {
                    int group = std::min(k / (int)B.group_size, gpr - 1);
                    if (group != last_group) {
                        float st[8] = {};
                        for (int j = 0; j < valid; ++j)
                            st[j] = B.scales[(size_t)(n + j) * gpr + group];
                        scale = vcombine_f16(vcvt_f16_f32(vld1q_f32(st)),
                                             vcvt_f16_f32(vld1q_f32(st + 4)));
                        last_group = group;
                    }
                    weight = vcvtq_f16_s16(vmovl_s8(vld1_s8(tile8 + (size_t)k * 8)));
                    weight = vmulq_f16(weight, scale);
                }
                acc[slot] = vfmaq_n_f16(acc[slot], weight, (__fp16)a[k]);
                slot = (slot + 1) & 3;
            }
            float16x8_t sum = vaddq_f16(vaddq_f16(acc[0], acc[1]),
                                         vaddq_f16(acc[2], acc[3]));
            float tmp[8];
            vst1q_f32(tmp, vcvt_f32_f16(vget_low_f16(sum)));
            vst1q_f32(tmp + 4, vcvt_f32_f16(vget_high_f16(sum)));
            for (int j = 0; j < valid; ++j) c[n + j] = tmp[j];
        }
    };

    int threads = thread_pool ? thread_pool->num_threads() : 1;
    if (threads > 1 && N >= 64) {
        int chunk = std::max(64, ((N + threads * 8 - 1) / (threads * 8)) * 8);
        thread_pool->parallel_for(0, N, chunk,
            [&](int, int begin, int end) { run_range(begin, end); });
    } else {
        run_range(0, N);
    }
#else
    kernel_matmul_fp32(A, B, C, thread_pool);
#endif
}
