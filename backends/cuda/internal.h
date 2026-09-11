#pragma once

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/quant_layouts.h"

// CUDA backend resources and kernel-launch declarations.
namespace mollm_cuda {

struct ArgMaxPair {
    float value;
    int index;
};

struct DeviceBufferPool;

bool report_cuda(cudaError_t error, const char* operation);
bool report_cublas(cublasStatus_t status, const char* operation);
bool malloc_device(void** pointer, size_t bytes, const char* operation);
bool malloc_managed(void** pointer, size_t bytes, const char* operation);
bool copy_memory(void* destination, const void* source, size_t bytes,
                 cudaMemcpyKind kind, const char* operation);
bool zero_memory(void* destination, size_t bytes, const char* operation);
bool launch_round_to_bf16(float* values, size_t count);
DeviceBufferPool* create_device_buffer_pool();
void destroy_device_buffer_pool(DeviceBufferPool* pool);
void* acquire_device_buffer(DeviceBufferPool* pool, size_t bytes);
void release_device_buffer(DeviceBufferPool* pool, void* pointer);

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

void launch_rope(
    const float* input, const float* cosine, const float* sine, float* output,
    int feature_dim, int sequence_length, int channels, int shape2,
    int rope_dim, bool interleave, size_t x_s0, size_t x_s1, size_t x_s2,
    size_t x_s3, size_t c_s0, size_t c_s1, size_t s_s0, size_t s_s1,
    size_t o_s0, size_t o_s1, size_t o_s2, size_t o_s3);
void launch_qk_rms_norm_rope(
    const float* query, const float* key, const float* query_weight,
    const float* key_weight, const float* cosine, const float* sine,
    float* output, int feature_dim, int sequence_length, int query_heads,
    int total_heads, int rope_dim, bool interleave,
    size_t query_feature_stride, size_t query_row_stride,
    size_t key_feature_stride, size_t key_row_stride,
    size_t cosine_feature_stride, size_t cosine_position_stride,
    size_t sine_feature_stride, size_t sine_position_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride, float epsilon);
void launch_append_kv(
    const float* key, const float* value, void* key_cache,
    void* value_cache, bool fp16_cache, int num_kv_heads,
    int current_length, int past_length, int max_length,
    int key_dim, int value_dim, size_t key_position_stride,
    size_t key_head_stride, size_t value_position_stride,
    size_t value_head_stride);
void launch_sdpa_scores(
    const float* query, const void* key, float* scores, const float* mask,
    int num_heads, int num_kv_heads, int query_length, int key_length,
    int past_length, int key_dim, int key_capacity, bool cached,
    bool fp16_cache, bool causal, float scale,
    size_t query_feature_stride, size_t query_position_stride,
    size_t query_head_stride, size_t key_feature_stride,
    size_t key_position_stride, size_t key_head_stride,
    size_t mask_column_stride, size_t mask_row_stride);
void launch_sdpa_output(
    const float* scores, const void* value, float* output, int num_heads,
    int num_kv_heads, int query_length, int key_length, int value_dim,
    int value_capacity, bool cached, bool fp16_cache,
    size_t value_feature_stride, size_t value_position_stride,
    size_t value_head_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride);
bool try_launch_sdpa_prefill(
    const float* query, const void* key, const void* value, float* scores,
    float* output, const float* mask, int num_heads, int num_kv_heads,
    int query_length, int key_length, int past_length, int key_dim,
    int value_dim, int key_capacity, bool cached, bool fp16_cache,
    bool causal, float scale, size_t query_feature_stride,
    size_t query_position_stride, size_t query_head_stride,
    size_t mask_column_stride, size_t mask_row_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride);
bool try_launch_sdpa_decode_fp16_cached(
    const float* query, const float* current_key,
    const float* current_value, void* key_cache, void* value_cache,
    float* scores, float* output, const float* mask, int num_heads,
    int num_kv_heads, int key_length, int past_length, int key_dim,
    int value_dim, int key_capacity, bool causal, float scale,
    size_t query_feature_stride, size_t query_head_stride,
    size_t current_key_feature_stride, size_t current_key_head_stride,
    size_t current_value_feature_stride, size_t current_value_head_stride,
    size_t mask_column_stride, size_t output_feature_stride,
    size_t output_head_stride);
void launch_sdpa_decode(
    const float* query, const void* key, const void* value, float* scores,
    float* output, const float* mask, int num_heads, int num_kv_heads,
    int key_length, int past_length, int key_dim, int value_dim,
    int key_capacity, bool cached, bool fp16_cache, bool causal, float scale,
    size_t query_feature_stride, size_t query_head_stride,
    size_t key_feature_stride, size_t key_position_stride,
    size_t key_head_stride, size_t value_feature_stride,
    size_t value_position_stride, size_t value_head_stride,
    size_t mask_column_stride, size_t output_feature_stride,
    size_t output_head_stride);

void launch_apply_activation(float* values, int rows, int columns, int kind,
                             int begin, int end);
void launch_binary(const float* lhs, const float* rhs, float* output,
                   size_t count, bool multiply);
void launch_sigmoid_mul(const float* value, const float* gate, float* output,
                        size_t count);
void launch_sigmoid_mul_strided(
    const float* value, const float* gate, float* output, size_t count,
    int64_t d0, int64_t d1, int64_t d2, size_t value_s0, size_t value_s1,
    size_t value_s2, size_t value_s3, size_t gate_s0, size_t gate_s1,
    size_t gate_s2, size_t gate_s3);
void launch_unary(const float* input, float* output, size_t count,
                  int operation);
void launch_swiglu(const float* input, float* output, size_t output_count,
                   size_t half);
void launch_rms_norm(const float* input, const float* weight, float* output,
                     int width, int rows, float epsilon);
bool launch_shortconv(
    const float* input, const float* weight, float* state, float* output,
    int groups, int sequence_length, int kernel_size, int real_tokens,
    size_t input_feature_stride, size_t input_row_stride);
size_t gdn_prefill_scratch_bytes(
    int num_heads, int num_value_heads, int key_dimension,
    int value_dimension, int sequence_length);
bool launch_gdn(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, int num_heads, int num_value_heads,
    int key_dimension, int value_dimension, int sequence_length,
    int real_tokens, bool normalize_qk, float rms_epsilon, float l2_epsilon,
    float scale, size_t a_row_stride, size_t b_row_stride,
    size_t z_row_stride, void* prefill_scratch = nullptr,
    size_t prefill_scratch_bytes = 0);
void launch_add_rms_norm(
    float* residual, const float* update, const float* weight, float* output,
    int width, int rows, size_t residual_row_stride,
    size_t update_row_stride, size_t output_row_stride, float epsilon);
void launch_contiguous(
    const float* input, float* output, size_t count, int64_t d0, int64_t d1,
    int64_t d2, size_t s0, size_t s1, size_t s2, size_t s3);
void launch_argmax(const float* input, int count, ArgMaxPair* partial,
                   int groups, ArgMaxPair* result);

}  // namespace mollm_cuda
