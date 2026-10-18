#pragma once

#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>

namespace mollm_cuda {

// Diagnostic only. Feature strides are one; cache is FP16/head-position-feature.
// Caller owns disjoint scores H*Q*K and two FP32 workspaces KVH*K*D floats.
// Accumulation order differs from SIMT; no bit-exact guarantee or backend selector.
// Handle must use host pointer mode and the explicitly supplied stream.
// Optional device pointer workspace contains 5*heads float* entries, disjoint
// from all other ranges. Null keeps the independent per-head reference path.
struct SdpaGemmPrefillParams {
    int heads, kv_heads, queries, keys, past, dimension, capacity;
    bool causal;
    float scale;
    size_t query_position_stride, query_head_stride;
    size_t mask_column_stride, mask_row_stride;
    size_t output_position_stride, output_head_stride;
};
// ordered_pv is diagnostic stage isolation: GEMM QK/softmax, ordered FP32
// PV FMAs. It is not a runtime selector and does not promise bit equivalence.
cublasStatus_t launch_sdpa_prefill_fp32_gemm(cublasHandle_t handle,
    const float* query, const void* key, const void* value, float* scores,
    float* output, const float* mask, float* key_fp32, float* value_fp32,
    const SdpaGemmPrefillParams& params, cudaStream_t stream = nullptr,
    float** pointer_workspace = nullptr, bool ordered_pv = false);
// Default-off runtime entry: validates the explicit0/-1e38 causal mask and Q/K bounds
// on device before skipping future QK. Caller owns a four-byte flag.
// skip_zero_pv also requires finite V; zero/subnormal PV results retain
// original future FMAs. Defaults preserve the existing runtime behavior.
// Backend admission for grouped-query optimization, independent of engine
// chunk size. The scalar validated entry already supports larger query grids.
inline constexpr int sdpa_prefill_group_max_queries = 2048;
bool try_launch_sdpa_prefill_validated_prefix(const float* query,
    const void* key, const void* value, float* scores, float* output,
    const float* mask, uint32_t* valid, const SdpaGemmPrefillParams& params,
    cudaStream_t stream = nullptr, cudaError_t* launch_error = nullptr,
    int shared_score_limit = 1024, bool skip_zero_pv = false, bool unroll_pv = false,
    bool pair_pv = false, bool pair_queries = false, int query_group = 2,
    bool grouped_long_shared = false, bool grouped_heads = false,
    bool probabilities_only = false, int head_lanes = 1,
    bool elide_validated_mask = false);
// Native-probability PV component. Caller owns disjoint KVH*K*D FP32 values,
// 3*H device pointers, and a separate output span. No allocation or selection.
cublasStatus_t launch_sdpa_prefill_fp32_pv(cublasHandle_t handle,const void* value,
    const float* probabilities,float* output,float* value_fp32,float** pointers,
    const SdpaGemmPrefillParams& params,cudaStream_t stream);
// Private composite candidate: full-batch validation once, compact score slices,
// V expansion once. Caller supplies H*min(query_tile,Q)*K scores and the same
// full output/value/pointer spans as the unsliced PV path. No allocation.
bool try_launch_sdpa_prefill_query_slices(cublasHandle_t handle,const float* query,
    const void* key,const void* value,float* scores,float* output,const float* mask,
    uint32_t* valid,float* value_fp32,float* pv_output,float** pointers,
    const SdpaGemmPrefillParams& params,int query_tile,int head_lanes,
    cudaStream_t stream,cudaError_t* launch_error=nullptr);
void launch_sdpa_validated_pv_copy(const float* source,float* destination,
    const uint32_t* valid,const SdpaGemmPrefillParams& params,cudaStream_t stream);

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
// skip_causal_qk is a diagnostic specialization for implicit-causal head256
// only. Production dispatch retains false: this model uses explicit masks.
bool try_launch_sdpa_prefill(
    const float* query, const void* key, const void* value, float* scores,
    float* output, const float* mask, int num_heads, int num_kv_heads,
    int query_length, int key_length, int past_length, int key_dim,
    int value_dim, int key_capacity, bool cached, bool fp16_cache,
    bool causal, float scale, size_t query_feature_stride,
    size_t query_position_stride, size_t query_head_stride,
    size_t mask_column_stride, size_t mask_row_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride, bool skip_causal_qk = false,
    bool paired_queries = false, bool paired_qk = false,
    bool reload_paired_query = false); // Diagnostic D256 exact-order query-pair reuse.
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

}  // namespace mollm_cuda
