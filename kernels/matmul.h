#pragma once

#include "kernels/tensor.h"
#include "kernels/activations.h"  // for Activation enum

class ThreadPool;

// Runtime-configurable matmul parameters (for benchmarking).
struct MatmulConfig {
    int k_block = 2048;          // 0 = disable K-blocking (sweep: 512→2048 = +4-8%)
    int gemv_chunk_size = 64;   // chunk size for M==1 or N==1 shapes
    bool use_interleave_pack = true;  // B interleaved packing for FP16
    bool use_fp16_accumulate = true;  // FP16 accumulate (2x throughput, may lose precision)
};

extern MatmulConfig g_matmul_config;
extern bool g_mollm_force_fp32_acc;  // debug: force FP32 accumulation

extern "C" {
int mollm_matmul_shape_profile_enabled();
void mollm_set_matmul_profile_phase(const char* phase);
void mollm_reset_matmul_shape_profile();
void mollm_print_matmul_shape_profile(const char* title, int top_n);
}

// ---------------------------------------------------------------------------
// mollm — Matmul kernels
//
// C[M,N] = A[M,K] * B[K,N]
//   A: M×K, row-major (stride[1] = row stride)
//   B: K×N, row-major
//   C: M×N, row-major
//
// `act` is an optional fused activation applied to C at writeback time
// (avoids a separate SILU/GELU op + memory round-trip). When act != NONE,
// only output columns in [act_n_begin, act_n_begin + act_n_len) get the
// activation applied; the rest are written raw. Set act_n_len = -1 for
// "apply to whole N" (fast path, no per-column check).
//
// kernel_matmul_fp32 dispatches to the best available implementation
// (NEON SIMD for ARM, scalar fallback otherwise).
// ---------------------------------------------------------------------------

void kernel_matmul_fp32(const Tensor& A, const Tensor& B, Tensor& C,
                        ThreadPool* thread_pool = nullptr,
                        Activation act = Activation::NONE,
                        int act_n_begin = 0,
                        int act_n_len = -1);

// Pack full B [N, K] row-major → interleaved [N/8, K, 8] layout.
// For each N-tile of 8 rows, transpose so that for fixed k,
// B_packed[tile_base + k*8 + 0..7] are 8 consecutive FP16 values.
// Enables vld1q_f16 contiguous load instead of strided gather.
// Returns newly allocated buffer (caller owns, must delete[]).
// K_weight is the stride between consecutive k rows in B_original
// (typically == K for row-major).
__fp16* pack_b_interleaved_full(const __fp16* B_original, int N, int K, int K_weight);

// Pack full int8 B [N, K] row-major -> interleaved [N/8, K, 8] layout.
// Same layout as FP16 B packing, but with int8 elements. Padding rows are zero.
int8_t* pack_b_interleaved_int8_full(const int8_t* B_original, int N, int K, int K_weight);

// Pack full int8 B [N, K] row-major -> Q8-dot layout [N/8, K/32, 8, 32].
// Padding output rows and K tail are zero.
int8_t* pack_b_q8dot_int8_full(const int8_t* B_original, int N, int K, int K_weight);

// Pack full int4 B [N, ceil(K/2)] row-major -> Q4-dot layout
// [N/8, K/32, 8, 16 packed bytes]. Padding output rows and K tail are zero.
uint8_t* pack_b_q4dot_int4_full(const uint8_t* B_original, int N, int K, int K_weight);

// Pack A [K, M] column-major FP32 → interleaved [M/8, K, 8] FP16.
// For each M-tile of 8 rows, 8 M values at the same k are stored consecutively.
// Enables vld1q_f16 contiguous load + vfmlalq_laneq_f16 lane-broadcast FMA.
// Returns newly allocated buffer (caller owns, must delete[]).
__fp16* pack_a_interleaved_full(const float* A_original, int M, int K, int lda);
