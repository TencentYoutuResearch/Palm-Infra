#pragma once

#include "kernels/cpu_platform.h"

namespace mollm::cpu::x86 {

void matmul_fp32_avx2_range(const float* A, const float* B, float* C, int N,
                            int K, int lda, int K_weight, int ldc,
                            int m_begin, int m_end);
void matmul_fp16_avx2_range(const float* A, const fp16_t* B, float* C, int N,
                            int K, int lda, int K_weight, int ldc,
                            int m_begin, int m_end);
void matmul_fp16_m2_avx2_range_n(const float* A, const fp16_t* B, float* C,
                                  int N, int K, int lda, int K_weight, int ldc,
                                  int n_begin, int n_end);
void matmul_int4_bg_avx2_range(const Tensor& A, const Tensor& B, Tensor& C,
                               int lda, int ldc, int m_begin, int m_end,
                               int n_begin, int n_end);
void matmul_int8_avx2_range(const float* A, const int8_t* B,
                            const float* scales, float* C, int N, int K,
                            int group_size, int groups_per_row, int lda,
                            int K_weight, int ldc, int m_begin, int m_end,
                            int n_begin, int n_end);

}  // namespace mollm::cpu::x86
