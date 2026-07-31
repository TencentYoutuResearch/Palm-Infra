#pragma once

#include "kernels/matmul.h"
#include "kernels/matmul_profile.h"
#include "kernels/quant_layouts.h"

#include <algorithm>
#include <cstdint>
#include <vector>

class ThreadPool;

constexpr int MATMUL_Q8_BLOCK = 32;

enum class W8ScaleMode {
    PerChannel,
    PerBlock32,
    PerGroup,
};

inline W8ScaleMode w8_scale_mode(int group_size, int groups_per_row) {
    if (groups_per_row == 1)
        return W8ScaleMode::PerChannel;
    if (group_size == MATMUL_Q8_BLOCK)
        return W8ScaleMode::PerBlock32;
    return W8ScaleMode::PerGroup;
}

inline int w8_scale_group(W8ScaleMode mode, int qb, int group_size) {
    switch (mode) {
    case W8ScaleMode::PerChannel:
        return 0;
    case W8ScaleMode::PerBlock32:
        return qb;
    case W8ScaleMode::PerGroup:
        return (qb * MATMUL_Q8_BLOCK) / group_size;
    }
    return 0;
}

#if HAS_NEON
inline void load_w8_b_scales8(const float* scales, int n, int c_valid,
                              int groups_per_row, int group, float32x4_t& lo,
                              float32x4_t& hi) {
    float tmp[8];
    for (int c = 0; c < 8; c++) {
        tmp[c] = c < c_valid ? scales[(n + c) * groups_per_row + group] : 0.f;
    }
    lo = vld1q_f32(tmp);
    hi = vld1q_f32(tmp + 4);
}
#endif

struct alignas(16) Q8A4Block {
    float scales[4];
    int8_t even[4][16];
    int8_t odd[4][16];
};

struct alignas(16) Q8A8I8MMBlock {
    float scale_pairs[4][4];
    int8_t q[4][8][8];
};
static_assert(sizeof(Q8A8I8MMBlock) == 320,
              "unexpected Q8A8I8MMBlock size");

struct Q4GemvScratch {
    std::vector<int8_t> qA_even;
    std::vector<int8_t> qA_odd;
    std::vector<float> a_scales;
};

inline void matmul_apply_activation(float* C, int M, int N, int ldc,
                                    int m_begin, int m_end, Activation act,
                                    int act_n_begin, int act_n_len) {
    (void)M;
    if (act == Activation::NONE || act_n_len == 0)
        return;

    bool full_n = act_n_len < 0 || (act_n_begin == 0 && act_n_len >= N);
#if HAS_NEON
    for (int m = m_begin; m < m_end; m++) {
        float* row = C + m * ldc;
        if (full_n) {
            int n = 0;
            for (; n + 3 < N; n += 4) {
                float32x4_t value = vld1q_f32(row + n);
                value = apply_activation_f32_neon(value, act);
                vst1q_f32(row + n, value);
            }
            for (; n < N; n++)
                row[n] = apply_activation_scalar(row[n], act);
        } else {
            int n_end = std::min(act_n_begin + act_n_len, N);
            int n = act_n_begin;
            for (; n + 3 < n_end; n += 4) {
                float32x4_t value = vld1q_f32(row + n);
                value = apply_activation_f32_neon(value, act);
                vst1q_f32(row + n, value);
            }
            for (; n < n_end; n++)
                row[n] = apply_activation_scalar(row[n], act);
        }
    }
#else
    for (int m = m_begin; m < m_end; m++) {
        float* row = C + m * ldc;
        int n_end = full_n ? N : std::min(act_n_begin + act_n_len, N);
        int n = full_n ? 0 : act_n_begin;
        for (; n < n_end; n++)
            row[n] = apply_activation_scalar(row[n], act);
    }
#endif
}

inline void matmul_apply_activation_gemv(float* C, int N, Activation act,
                                         int act_n_begin, int act_n_len) {
    matmul_apply_activation(C, 1, N, N, 0, 1, act, act_n_begin, act_n_len);
}

void quantize_a_q8_blocks(const float* A, int M, int K, int lda, int K_padded,
                          std::vector<int8_t>& qA,
                          std::vector<float>& a_scales);
// DeepSeek-V4 quantized linears first round activations to E4M3 with one
// power-of-two E8M0 scale per 128 K elements. These helpers preserve that
// model-visible quantization before mapping the coefficients to Q8 for ARM
// SDOT/i8mm execution.
void quantize_a_fp8_q8_blocks(const float* A, int M, int K, int lda,
                              int K_padded, std::vector<int8_t>& qA,
                              std::vector<float>& a_scales);
void quantize_a_fp8_dequant(const float* A, int M, int K, int lda,
                            std::vector<float>& output);
void quantize_a_q8_blocks_a4(const float* A, int M, int K, int lda,
                             std::vector<Q8A4Block>& qA4);
void quantize_a_q8_blocks_i8mm_a8(const float* A, int M, int K, int lda,
                                  std::vector<Q8A8I8MMBlock>& qA8);
#if HAS_NEON && MOLLM_ARM_I8MM_KERNELS
void matmul_int4_i8mm_g128(const Q8A8I8MMBlock* qA8,
                           const Q4B8G128Block* B_g128, float* C, int M,
                           int N, int K, int ldc, int m_begin, int m_end,
                           int n_begin, int n_end);
void matmul_int4_i8mm_g32(const Q8A8I8MMBlock* qA8,
                          const Q4B8G32Block* B_g32, float* C, int M, int N,
                          int K, int ldc, int m_begin, int m_end, int n_begin,
                          int n_end);
void matmul_int8_q8dot_neon_4x8_repacked_i8mm_range(
    const int8_t* qA, const float* a_scales, const int8_t* B_repack,
    const float* scales, int group_size, int groups_per_row, float* C, int M,
    int N, int K, int K_padded, int ldc, int m_begin, int m_end, int n_begin,
    int n_end);
void matmul_int8_q8dot_neon_8x8_repacked_i8mm_range(
    const int8_t* qA, const float* a_scales, const int8_t* B_repack,
    const float* scales, int group_size, int groups_per_row, float* C, int M,
    int N, int K, int K_padded, int ldc, int m_begin, int m_end, int n_begin,
    int n_end);
#endif
void quantize_a_q8_blocks_even_odd(const float* A, int K,
                                   std::vector<int8_t>& qA_even,
                                   std::vector<int8_t>& qA_odd,
                                   std::vector<float>& a_scales);
void quantize_a_fp8_q8_even_odd(const float* A, int K,
                               std::vector<int8_t>& qA_even,
                               std::vector<int8_t>& qA_odd,
                               std::vector<float>& a_scales,
                               std::vector<float>* fp8_dequant = nullptr);
// BG128 decode variant: one activation scale covers four 32-value dot blocks.
void quantize_a_q8_g128_even_odd(const float* A, int K,
                                 std::vector<int8_t>& qA_even,
                                 std::vector<int8_t>& qA_odd,
                                 std::vector<float>& a_scales);

void matmul_dispatch_int4(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* thread_pool, Activation act,
                          int act_n_begin, int act_n_len, MatmulTimer& timer);
void matmul_dispatch_int8(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* thread_pool, Activation act,
                          int act_n_begin, int act_n_len, MatmulTimer& timer,
                          bool fp8_activation = false);
void matmul_dispatch_dense(const Tensor& A, const Tensor& B, Tensor& C,
                           ThreadPool* thread_pool, Activation act,
                           int act_n_begin, int act_n_len, MatmulTimer& timer,
                           bool force_fp32_acc = false);
void matmul_dispatch_mxfp4(const Tensor& A, const Tensor& B, Tensor& C,
                           ThreadPool* thread_pool, Activation act,
                           int act_n_begin, int act_n_len, MatmulTimer& timer);
void matmul_dispatch_fp8_e4m3(const Tensor& A, const Tensor& B, Tensor& C,
                              ThreadPool* thread_pool, Activation act,
                              int act_n_begin, int act_n_len,
                              MatmulTimer& timer);
