#pragma once

#include <cstddef>

namespace mollm_cuda {

// output_mapping: 0 original, 1 diagnostic token-contiguous, 2 tiled transpose.
// Alternate mappings affect parallel prefill only; decode stays serial.
bool launch_shortconv(
    const float* input, const float* weight, float* state, float* output,
    int groups, int sequence_length, int kernel_size, int real_tokens,
    size_t input_feature_stride, size_t input_row_stride, int output_mapping = 0);
// Diagnostic short-batch history snapshots, token-major then group/history.
// Caller supplies real_tokens * groups * (kernel_size-1) floats.
bool launch_shortconv_prefixes(
    const float* input, const float* weight, float* state, float* output,
    float* prefixes, int groups, int sequence_length, int kernel_size,
    int real_tokens, size_t input_feature_stride, size_t input_row_stride);
size_t gdn_prefill_scratch_bytes(
    int num_heads, int num_value_heads, int key_dimension,
    int value_dimension, int sequence_length);
// Independent prepared-recurrence probe. All input/state/output ranges must
// be disjoint. No production selector; arithmetic/reduction order unchanged.
bool launch_gdn_recurrence_probe(
    const float* qkv, const float* query, const float* key, const float* dot,
    const float* decay, const float* beta, float* state, float* output,
    int heads, int value_heads, int sequence, int real_tokens, float scale,
    bool prefetch_qk, int value_warps = 2, bool butterfly_key = false,
    bool broadcast_gates = false, bool paired_values = false, bool quad_values = false);
bool launch_gdn(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, int num_heads, int num_value_heads,
    int key_dimension, int value_dimension, int sequence_length,
    int real_tokens, int flags, float rms_epsilon, float l2_epsilon,
    float scale, size_t a_row_stride, size_t b_row_stride,
    size_t z_row_stride, void* prefill_scratch = nullptr,
    size_t prefill_scratch_bytes = 0);
// Diagnostic token-major snapshots of the complete post-token recurrent state.
// Caller supplies real_tokens * num_value_heads * 128 * 128 floats.
bool launch_gdn_prefixes(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, float* prefixes, int num_heads, int num_value_heads,
    int sequence_length, int real_tokens, int flags, float rms_epsilon,
    float l2_epsilon, float scale, size_t a_row_stride, size_t b_row_stride,
    size_t z_row_stride);

}  // namespace mollm_cuda
