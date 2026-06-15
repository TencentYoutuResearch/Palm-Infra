#pragma once

#include "kernels/tensor.h"

// ---------------------------------------------------------------------------
// PROJECT_NAME — Matmul kernels
//
// C[M,N] = A[M,K] * B[K,N]
//   A: M×K, row-major (stride[1] = row stride)
//   B: K×N, row-major
//   C: M×N, row-major
//
// kernel_matmul_fp32 dispatches to the best available implementation
// (NEON SIMD for ARM, scalar fallback otherwise).
// ---------------------------------------------------------------------------

void kernel_matmul_fp32(const Tensor& A, const Tensor& B, Tensor& C);
