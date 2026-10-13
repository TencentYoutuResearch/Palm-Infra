#pragma once
#include "graph/hc_prefill_limits.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace mollm_cuda {

struct ArgMaxPair {
    float value;
    int index;
};

bool launch_round_to_bf16(float* values, size_t count);
void launch_apply_activation(float* values, int rows, int columns, int kind,
                             int begin, int end);
void launch_binary(const float* lhs, const float* rhs, float* output,
                   size_t count, bool multiply,
                   bool lhs_scalar = false, bool rhs_scalar = false);
void launch_sigmoid_mul(const float* value, const float* gate, float* output,
                        size_t count);
void launch_sigmoid_mul_strided(
    const float* value, const float* gate, float* output, size_t count,
    int64_t d0, int64_t d1, int64_t d2, size_t value_s0, size_t value_s1,
    size_t value_s2, size_t value_s3, size_t gate_s0, size_t gate_s1,
    size_t gate_s2, size_t gate_s3);
void launch_unary(const float* input, float* output, size_t count,
                  int operation);
void launch_unary_strided(
    const float* input, float* output, size_t count, int operation,
    int64_t d0, int64_t d1, int64_t d2,
    size_t s0, size_t s1, size_t s2, size_t s3);
void launch_swiglu(const float* input, float* output, size_t output_count,
                   size_t half);
void launch_rms_norm(const float* input, const float* weight, float* output,
                     int width, int rows, float epsilon);
void launch_group_rms_norm(const float* input, const float* weight,
                           float* output, int group_size, int groups,
                           int rows, float epsilon);
// Optional FP16 shadow; original FP32 output and reduction order are preserved.
// cache_input is diagnostic-only and requires width2560; default is unchanged.
bool launch_group_rms_norm_half(const float* input, const float* weight,
    float* output, uint16_t* half, int group_size, int groups, int rows,
    float epsilon, bool cache_input = false);
// Bounded four-group producer: exact existing FP16 shadow plus RMS inverses,
// no FP32 materialization. Inputs must outlive consumers/reconstruction.
bool launch_group_rms_norm_deferred(const float* input, const float* weight,
    uint16_t* half, float* inverses, int rows, float epsilon,
    cudaStream_t stream = nullptr);
// Diagnostic entry; runtime opt-in only selects single-row, four-group decode.
bool launch_group_rms_norm_cached(const float* input, const float* weight,
    float* output, int group_size, int groups, int rows, float epsilon,
    cudaStream_t stream = nullptr);
// Diagnostic row-tiled sigmoid reduction; unsupported geometry returns false.
bool launch_gr_reduce_sigmoid_vector4(const float* input, const float* logits,
                                    float* output, int hidden, int streams,
                                    int rows, cudaStream_t stream = nullptr);
bool launch_gr_reduce_sigmoid_tiled(const float* input, const float* logits,
                                  float* output, int hidden, int streams,
                                  int rows, cudaStream_t stream = nullptr,
                                  bool specialize_four = false);
void launch_gr_reduce(const float* input, const float* gates, float* output,
                       int hidden, int streams, int rows);
void launch_gr_reduce_sigmoid(const float* input, const float* logits,
                              float* output, int hidden, int streams,
                              int rows, cudaStream_t stream = nullptr);
void launch_gr_inject(const float* branch, const float* residual,
                       const float* gates, float* output,
                       int hidden, int streams, int rows);
// Diagnostic geometry specialization; production dispatch is unchanged.
bool launch_gr_inject_tiled(const float* branch, const float* residual,
    const float* gates, float* output, int hidden, int streams, int rows,
    cudaStream_t stream = nullptr);
// Diagnostic only: width2560/streams4/rows<=4096; all output spans must be
// disjoint from each other and every input. Publishes the original rounded
// residual AND RMS outputs. No inference selector yet.
bool launch_gr_inject_group_rms_half(const float* branch, const float* residual,
    const float* gates, const float* weight, float* carry, float* normalized,
    uint16_t* half, int hidden, int streams, int rows, float epsilon,
    cudaStream_t stream = nullptr);
// Diagnostic deferred representation: caller retains carry/weight/inverses
// until the consumer completes. No normalized FP32 output is materialized.
// Optional compact flags use high16 bit storage in the first half of logical
// FP32 carry/residual allocations; FULL logical capacity/spans remain required.
// Only known BF16-rounded producer outputs may be tagged compact. Rows16..prefill_max_rows;
// compact_residual requires compact_carry. Defaults preserve FP32 storage.
bool launch_gr_inject_rms_deferred(const float* branch,const float* residual,
    const float* gates,const float* weight,float* carry,uint16_t* half,
    float* inverses,int rows,float epsilon,cudaStream_t stream=nullptr,
    bool compact_carry=false,bool compact_residual=false);
// Fixed hidden2560/top10/four streams. Compact carry AND compact residual;
// both retain full FP32 owner extents. BF16 routed outputs, FP32 shared branch.
// Default materializes original FP32 branch as well as compact carry/half/inverses.
// publish_branch=false preserves branch bytes; only executor-proven unexposed
// single-use branches may be omitted. Branch allocation/alias validation stays.
// Rows16..2048, positive finite epsilon, disjoint outputs/inputs; no alloc/sync.
// Diagnostic interface only: caller proves scratch and compact-owner lifetime.
bool launch_moe_combine_gr_inject_rms_deferred(const uint16_t* routed,
    const float* shared,const float* shared_scale,const float* residual,
    const float* gates,const float* weight,float* branch,float* carry,
    uint16_t* half,float* inverses,int rows,float epsilon,
    cudaStream_t stream=nullptr,bool publish_branch=true);
bool launch_gr_reduce_sigmoid_deferred_rms(const float* carry,const float* logits,
    const float* weight,const float* inverses,float* output,int rows,
    cudaStream_t stream=nullptr,bool compact_carry=false);
// Prepared FP16 HC up10240/K320 fused with deferred-RMS sigmoid/reduction.
// Fixed hidden2560/four streams; rows16..prefill_max_rows; FP32 BF16-rounded output.
// FP16 operands align32, FP32 operands align4; output may not alias inputs.
// Production default pitch320; optional pitch328/336 pad shared input only.
// Optional columns32 reuses each loaded input fragment for two output tiles.
bool launch_hc_up_reduce_fused(const uint16_t* activation,const uint16_t* weight,
    const float* carry,const float* gamma,const float* inverses,float* output,
    int rows,cudaStream_t stream=nullptr,int input_pitch=320,int tile_columns=16,
    bool compact_carry=false);
// Diagnostic fallback: reconstruct a complete normalized FP32 tensor.
bool launch_gr_rms_deferred_materialize(const float* carry,const float* weight,
    const float* inverses,float* output,int rows,cudaStream_t stream=nullptr,
    bool compact_carry=false);
// Restore a known compact HC carry (high16 bits of BF16-rounded FP32 values)
// occupying the first half of its logical FP32 allocation. Caller guarantees
// rows*10240 FP32 capacity and keeps scratch alive through stream completion.
// Scratch needs rows*10240*2 bytes, disjoint from the ENTIRE logical buffer.
// No allocation/synchronization; copy-before-expand is safe under graph capture.
// Rows1..prefill_max_rows supports fallback on small views/windows, not a decode selector.
bool launch_hc_compact_carry_materialize(float* logical_buffer,uint16_t* scratch,
    size_t scratch_bytes,int rows,cudaStream_t stream=nullptr);
void launch_ple_gate(const float* query, const float* key, const float* value,
                     float* output, int hidden, int streams, int rows);
bool launch_ple_dilated_conv_prefixes(const float* input,const float* weight,
    float* state,float* output,float* prefixes,int channels,int tokens,int kernel_size,
    int dilation,int real_tokens);
void launch_ple_dilated_conv(const float* input, const float* weight,
    float* state, float* output, int channels, int tokens, int kernel_size,
    int dilation, int real_tokens);
void launch_add_rms_norm(
    float* residual, const float* update, const float* weight, float* output,
    int width, int rows, size_t residual_row_stride,
    size_t update_row_stride, size_t output_row_stride, float epsilon);
void launch_contiguous(
    const float* input, float* output, size_t count, int64_t d0, int64_t d1,
    int64_t d2, size_t s0, size_t s1, size_t s2, size_t s3);
void launch_tile(const float* input, float* output, size_t count,
    int64_t o0, int64_t o1, int64_t o2,
    int64_t i0, int64_t i1, int64_t i2, int64_t i3,
    size_t s0, size_t s1, size_t s2, size_t s3);
void launch_argmax(const float* input, int count, ArgMaxPair* partial,
                   int groups, ArgMaxPair* result);

}  // namespace mollm_cuda
