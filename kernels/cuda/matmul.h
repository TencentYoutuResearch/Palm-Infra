#pragma once

#include "core/quant_layouts.h"

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mollm_cuda {

// Diagnostic private layout. Codes are BF16 maxima, not rounded FP16 scales.
struct Q4CompactG32 {
    uint16_t scales[8];
    uint8_t q[8][16];
};
static_assert(sizeof(Q4CompactG32) == 144);
void launch_prepare_q4_compact_g32_dense_gemm(
    const Q4CompactG32* weight, __half2* weight_output,
    const float* activation, __half* activation_output,
    size_t activation_count, int rows, int groups);
void launch_q4_g32_compact_gemv(const float* activation, const Q4CompactG32* weight,
    float* output, int columns, int inner, cudaStream_t stream = nullptr);
// Diagnostic half-warp compact path, mode 0/1/2: none/weights/inputs;
// mode 3 predecodes scales into CTA shared memory (K <= 4096).
cudaError_t launch_q4_compact_pair_probe(const float* activation,
    const Q4CompactG32* weight,float* output,int columns,int inner,int pipeline,
    cudaStream_t stream = nullptr);
// Diagnostic only: exact cached-wide compact arithmetic. Hints 1/2 use
// cg/cs reads; hint 3 combines table/activation initialization barriers
// (table decoder only). Hints 4/5 use two/eight warps with the table decoder.
// Hint 6 overlaps the next compact packet/scale load with current-group work.
// Hint 7 uses that overlap without the baseline's explicit L2 prefetch.
cudaError_t launch_q4_compact_cache_probe(const float* activation,
    const Q4CompactG32* weight, float* output, int columns, int inner,
    bool integer_scale, int hint, cudaStream_t stream = nullptr);
// Cached-wide compact decoder without a shared mantissa table. Runtime uses
// this only for the separately opted-in N13312/K2560 private weight layout.
cudaError_t launch_q4_compact_integer_gemv(const float* activation,
    const Q4CompactG32* weight, float* output, int columns, int inner,
    cudaStream_t stream = nullptr);
// Diagnostic only; narrow exact warp-register table, K multiple of512.
cudaError_t launch_q4_compact_warp_table_gemv(const float* activation,
    const Q4CompactG32* weight, float* output, int columns, int inner,
    cudaStream_t stream = nullptr);

struct Q4RowG32 {
    float scale;
    uint8_t q[16];
};
static_assert(sizeof(Q4RowG32) == 20);
void launch_q4_row_g32_gemv(const float* activation, const Q4RowG32* weight,
    float* output, int columns, int inner, cudaStream_t stream = nullptr);
void launch_repack_q4_row_g32(const Q4B8G32Block* source, Q4RowG32* target,
    int columns, int inner);

void launch_fp32_to_fp16(const float* source, __half* destination,
                         size_t count);
// Diagnostic: double-buffer four weight groups using asynchronous copies.
cudaError_t launch_q4_g32_async_wide_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    cudaStream_t stream = nullptr);
// Diagnostic: same cached-wide arithmetic and four-group prefetch; hint32
// is an explicit-load control, others request L2 prefetch sizes in bytes.
cudaError_t launch_q4_g32_load_hint_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int hint,
    cudaStream_t stream = nullptr);
void launch_dequantize_q8_dense_weight(
    const int8_t* weight, const float* scales, int group_size,
    int groups_per_row, __half* output, size_t count, int width);
// Diagnostic: aligned, per-channel weights and inner divisible by four.
// Diagnostic only: mode1 next char4, mode2 next float4+char4 lookahead;
// both retain original FP32 arithmetic.
cudaError_t launch_q8_pipeline_gemv(const float* activation, const int8_t* weight,
    const float* scales, float* output, int columns, int inner, int mode = 2, cudaStream_t stream = nullptr);
void launch_q8_prefetch_gemv(const float* activation, const int8_t* weight,
    const float* scales, float* output, int columns, int inner, int distance);
// Diagnostic only; retains the production per-lane FMA and reduction order.
cudaError_t launch_q8_tuned_gemv(const float* activation, const int8_t* weight,
    const float* scales, float* output, int columns, int inner, int warps,
    bool cache_activation, cudaStream_t stream = nullptr);

// Diagnostic short-batch per-channel Q8, bit-exact serial GEMV reduction.
cudaError_t launch_q8_small_batch_exact(const float* activation,
    const int8_t* weight, const float* scales, float* output, int tokens,
    int columns, int inner, int lda, int ldc, cudaStream_t stream = nullptr);
void launch_q8_dense_gemv(
    const float* activation, const int8_t* weight, const float* scales,
    int group_size, int groups_per_row, float* output, int columns,
    int inner);
void launch_bias_q4_g32_weight(Q4B8G32Block* weight, size_t block_count);
// Experimental in-place packet layout for the exact wide-row reduction tree.
void launch_swizzle_q4_g32_weight(Q4B8G32Block* weight, size_t block_count);
void launch_q4_g32_swizzled_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    cudaStream_t stream = nullptr);
void launch_prepare_q4_g32_dense_gemm(
    const Q4B8G32Block* weight, __half2* weight_output,
    const float* activation, __half* activation_output,
    size_t activation_count, int rows, int groups);
void launch_dequantize_q4_g128_dense_weight(
    const Q4B8G128Block* weight, __half2* output, size_t packed_count,
    int rows, int groups);
void launch_q4_g32_dense_gemv(
    const float* activation, const Q4B8G32Block* weight, float* output,
    int columns, int inner, cudaStream_t stream = nullptr);
// Small-token FP32 projection; shares each decoded weight across 2..8 tokens.
// Weight nibble codes must already be biased, like the resident GEMV layout.
cudaError_t launch_q4_g32_small_batch(const float* activation,
    const Q4B8G32Block* weight, float* output, int tokens, int columns,
    int inner, int lda, int ldc, cudaStream_t stream = nullptr);
cudaError_t launch_q4_g32_small_batch_wide(const float* activation,
    const Q4B8G32Block* weight, float* output, int tokens, int columns,
    int inner, int lda, int ldc, cudaStream_t stream = nullptr);
cudaError_t launch_q4_g32_small_batch_reduced(const float* activation,
    const Q4B8G32Block* weight, float* output, int tokens, int columns,
    int inner, int lda, int ldc, cudaStream_t stream = nullptr);
// Reuse weights only when the batch kernel preserves the decode reduction;
// otherwise fall back to independent decode launches.
cudaError_t launch_q4_g32_small_batch_exact(const float* activation,
    const Q4B8G32Block* weight, float* output, int tokens, int columns,
    int inner, int lda, int ldc, cudaStream_t stream = nullptr);
cudaError_t launch_q4_compact_small_batch_exact(const float* activation,
    const Q4CompactG32* weight, float* output, int tokens, int columns,
    int inner, int lda, int ldc, bool integer_scale, cudaStream_t stream = nullptr);
// Diagnostic: preserve the narrow-row reduction tree without end-of-kernel barriers.
void launch_q4_g32_grouped_narrow_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int rows,
    cudaStream_t stream = nullptr);
void launch_q4_g32_prefetch_narrow_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int distance);

void launch_q4_g32_pair_narrow_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, bool cache,
    cudaStream_t stream = nullptr);
// Diagnostic half-warp iteration lookahead, canonical BG32, K<=2560.
// Original FP32 dot and reduction order; no allocation or synchronization.
cudaError_t launch_q4_g32_pipelined_pair_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    bool pipeline_inputs = true, cudaStream_t stream = nullptr);

void launch_q4_g32_split_pair_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int splits);

void launch_q4_g32_terminal_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner);
// Two independent canonical narrow projections; no weight concatenation.
cudaError_t launch_q4_g32_paired_projections(const float* activation,
    const Q4B8G32Block* first_weight, const Q4B8G32Block* second_weight,
    float* first_output, float* second_output, int columns, int inner,
    cudaStream_t stream = nullptr);
// Canonical narrow shared gate/up plus the existing BF16 SiLU epilogue.
cudaError_t launch_q4_g32_shared_activation(const float* activation,
    const Q4B8G32Block* gate_weight, const Q4B8G32Block* up_weight,
    float* output, int columns, int inner, cudaStream_t stream = nullptr);
// Diagnostic unequal projections sharing an input; independent canonical weights.
cudaError_t launch_q4_g32_unequal_projections(const float* activation,
    const Q4B8G32Block* first_weight, const Q4B8G32Block* second_weight,
    float* first_output, float* second_output, int first_columns, int second_columns,
    int inner, cudaStream_t stream = nullptr);
// Diagnostic register lookahead of one/two/four original narrow iterations.
cudaError_t launch_q4_g32_pipelined_narrow_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int depth,
    bool pipeline_inputs = true, cudaStream_t stream = nullptr, bool single_barrier = false);
// Diagnostic exact one-barrier reconstruction of the original 256-thread tree.
cudaError_t launch_q4_compact_pipelined_narrow_gemv(const float* activation,
    const Q4CompactG32* weight,float* output,int columns,int inner,cudaStream_t stream=nullptr);
cudaError_t launch_q4_g32_single_barrier_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    cudaStream_t stream = nullptr);
// Spatial diagnostic: 256 threads interleave two/four/eight rows while retaining
// all original logical accumulator chains and their FP32 reduction tree.
cudaError_t launch_q4_g32_spatial_narrow_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int rows_per_cta,
    cudaStream_t stream = nullptr);
// Diagnostic only: preserve the 256-lane arithmetic with 128 physical threads.
cudaError_t launch_q4_g32_dual_lane_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    cudaStream_t stream = nullptr, bool pipeline_inputs = false);
void launch_q4_g32_uncached_wide_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner);
void launch_q4_g32_prefetch_wide_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner, int distance,
    cudaStream_t stream = nullptr);
// Diagnostic L1 packet prefetch, distances 1/2/4/8, unchanged wide arithmetic.
cudaError_t launch_q4_g32_l1_prefetch_wide_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    int distance, cudaStream_t stream = nullptr);
// Diagnostic next-group register lookahead, original wide arithmetic/tree.
cudaError_t launch_q4_g32_pipelined_wide_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    cudaStream_t stream = nullptr);
// Diagnostic scale+packet prefetch, keeping wide-row decode arithmetic.
cudaError_t launch_q4_g32_prefetch_scale_gemv(const float* activation,
    const Q4B8G32Block* weight, float* output, int columns, int inner,
    int distance, cudaStream_t stream = nullptr);
void launch_q4_g128_dense_gemv(
    const float* activation, const Q4B8G128Block* weight, float* output,
    int columns, int inner);
bool run_dense_matmul(
    cublasHandle_t cublas, const void* weight, cudaDataType weight_type,
    const void* activation, cudaDataType activation_type, float* output,
    int m, int n, int k, int lda, bool pedantic = false,
    int hc_projection_row_tile = 0);
// Diagnostic exact expansion of existing half-rounded weights, not a new
// dequantization policy. Source/destination must not overlap.
void launch_fp16_to_fp32(const __half* source, float* destination, size_t count);

}  // namespace mollm_cuda
