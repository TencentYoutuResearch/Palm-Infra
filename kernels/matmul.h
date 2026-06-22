#pragma once

#include "kernels/tensor.h"

class ThreadPool;

// Runtime-configurable matmul parameters (for benchmarking).
struct MatmulConfig {
    int k_block = 512;          // 0 = disable K-blocking
    int gemv_chunk_size = 64;   // chunk size for M==1 or N==1 shapes
    bool use_interleave_pack = true;  // B interleaved packing for FP16
};

extern MatmulConfig g_matmul_config;

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

void kernel_matmul_fp32(const Tensor& A, const Tensor& B, Tensor& C,
                        ThreadPool* thread_pool = nullptr);
