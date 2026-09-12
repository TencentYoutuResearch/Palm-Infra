#pragma once

#include "core/quant_layouts.h"

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mollm_cuda {

void launch_fp32_to_fp16(const float* source, __half* destination,
                         size_t count);
void launch_dequantize_q8_dense_weight(
    const int8_t* weight, const float* scales, int group_size,
    int groups_per_row, __half* output, size_t count, int width);
void launch_q8_dense_gemv(
    const float* activation, const int8_t* weight, const float* scales,
    int group_size, int groups_per_row, float* output, int columns,
    int inner);
void launch_bias_q4_g32_weight(Q4B8G32Block* weight, size_t block_count);
void launch_prepare_q4_g32_dense_gemm(
    const Q4B8G32Block* weight, __half2* weight_output,
    const float* activation, __half* activation_output,
    size_t activation_count, int rows, int groups);
void launch_dequantize_q4_g128_dense_weight(
    const Q4B8G128Block* weight, __half2* output, size_t packed_count,
    int rows, int groups);
void launch_q4_g32_dense_gemv(
    const float* activation, const Q4B8G32Block* weight, float* output,
    int columns, int inner);
void launch_q4_g128_dense_gemv(
    const float* activation, const Q4B8G128Block* weight, float* output,
    int columns, int inner);
bool run_dense_matmul(
    cublasHandle_t cublas, const void* weight, cudaDataType weight_type,
    const void* activation, cudaDataType activation_type, float* output,
    int m, int n, int k, int lda);

}  // namespace mollm_cuda
