#pragma once

#include "kernels/activations.h" // for Activation enum
#include "kernels/tensor.h"

#include <string>
#include <unordered_map>
#include <vector>

class ThreadPool;

// Runtime-configurable matmul parameters (for benchmarking).
struct MatmulConfig {
    int k_block = 2048;              // 0 = disable K-blocking (sweep: 512→2048 = +4-8%)
    int gemv_chunk_size = 64;        // chunk size for M==1 or N==1 shapes
    bool use_interleave_pack = true; // B interleaved packing for FP16
    bool use_fp16_accumulate =
        true; // FP16 accumulate (2x throughput, may lose precision)
};

extern MatmulConfig g_matmul_config;
extern bool g_mollm_force_fp32_acc; // debug: force FP32 accumulation

// Engine-lifetime buffers that own load-time matmul repacks. The key is the
// package/file weight path plus a layout suffix where one source weight needs
// multiple layouts.
using PackedWeightMap = std::unordered_map<std::string, std::vector<uint8_t>>;

// Whether this build can consume the packed INT4 Q4-dot/BG128 layouts emitted
// by the converter.
bool matmul_int4_q4dot_kernel_available();

// Prepare load-time layouts consumed by the CPU matmul kernels. `weight_data`
// must point at the source row-major (or prepacked INT4) bytes for `weight`.
// Embedding tables pass false for `pack_fp16` because lookup requires their
// original row-major layout.
void prepare_matmul_weight(Tensor& weight, const std::string& key,
                           const void* weight_data,
                           PackedWeightMap& packed_weights,
                           bool pack_fp16 = true);

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
                        Activation act = Activation::NONE, int act_n_begin = 0,
                        int act_n_len = -1);

void kernel_gemv_sparse_a(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* thread_pool = nullptr);

// Pack full B [N, K] row-major → interleaved [N/8, K, 8] layout.
// For each N-tile of 8 rows, transpose so that for fixed k,
// B_packed[tile_base + k*8 + 0..7] are 8 consecutive FP16 values.
// Enables vld1q_f16 contiguous load instead of strided gather.
// Returns newly allocated buffer (caller owns, must delete[]).
// K_weight is the stride between consecutive k rows in B_original
// (typically == K for row-major).
__fp16* pack_b_interleaved_full(const __fp16* B_original, int N, int K,
                                int K_weight);

// Pack full int8 B [N, K] row-major -> interleaved [N/8, K, 8] layout.
// Same layout as FP16 B packing, but with int8 elements. Padding rows are zero.
int8_t* pack_b_interleaved_int8_full(const int8_t* B_original, int N, int K,
                                     int K_weight);

// Pack full int8 B [N, K] row-major -> Q8-dot layout [N/8, K/32, 8, 32].
// Padding output rows and K tail are zero.
int8_t* pack_b_q8dot_int8_full(const int8_t* B_original, int N, int K,
                               int K_weight);

// Pack full int4 B [N, ceil(K/2)] row-major -> Q4-dot layout
// [N/8, K/32, 8, 16 packed bytes]. Padding output rows and K tail are zero.
uint8_t* pack_b_q4dot_int4_full(const uint8_t* B_original, int N, int K,
                                int K_weight);

// Pack Q4-dot B plus W4G128 scales -> [N/8, K/128] blocks.
// Each block stores float scales[8] then q4dot q[4][8][16].
size_t pack_b_q4dot_g128_bytes(int N, int K);
uint8_t* pack_b_q4dot_g128_full(const uint8_t* B_q4dot, const float* scales,
                                int N, int K, int groups_per_row);

// Expand G128 nibbles into the [N/8,K,8] signed-byte layout consumed by
// sparse-A GEMV. Exposed for sparse kernel validation.
int8_t* pack_b_sparse_int4_g128_full(const void* B_g128, int N, int K);
