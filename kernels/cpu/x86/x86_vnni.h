#pragma once

#include "kernels/tensor.h"

#include <cstdint>

namespace mollm::cpu::x86 {

void quantize_q8_vnni_range(const float* A, int8_t* qA, float* scales,
                            int16_t* sums, int K, int lda, int m_begin,
                            int m_end);
void matmul_int4_bg32_vnni_range(
    const int8_t* qA, const float* a_scales, const int16_t* a_sums,
    const void* vnni_data, const Tensor& B, Tensor& C, int K, int ldc,
    int m_begin, int m_end, int n_begin, int n_end);

}  // namespace mollm::cpu::x86
