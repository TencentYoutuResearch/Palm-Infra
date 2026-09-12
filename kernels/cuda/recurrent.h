#pragma once

#include <cstddef>

namespace mollm_cuda {

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

}  // namespace mollm_cuda
