#pragma once

#include "kernels/tensor.h"

class ThreadPool;

struct Dsv4RopeConfig {
    int rope_dim = 64;
    float theta = 160000.0f;
    int original_context = 65536;
    float factor = 16.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
};

struct Dsv4CompressorConfig {
    int hidden_size = 4096;
    int head_dim = 512;
    int ratio = 4;
    bool overlap = true;
    bool rotate = false;
    float norm_eps = 1e-6f;
    Dsv4RopeConfig rope;
};

struct Dsv4IndexerConfig {
    int hidden_size = 4096;
    int q_lora_rank = 1024;
    int num_heads = 64;
    int head_dim = 128;
    int top_k = 512;
    Dsv4CompressorConfig compressor;
};

struct Dsv4SparseAttentionConfig {
    int num_heads = 64;
    int head_dim = 512;
    int window_size = 128;
    int compress_ratio = 0;
    int compressed_top_k = 512;
    float softmax_scale = 0.0f;
    float query_norm_eps = 1e-6f;
    bool normalize_query = true;
    Dsv4RopeConfig rope;
};

// Stateful learned KV compression used by DeepSeek-V4.
//
// Tensor layouts follow mollm's dim0-fast convention:
//   hidden:       [hidden_size, sequence]
//   wkv/wgate:    [coff * head_dim, hidden_size]
//   ape:          [coff * head_dim, ratio]
//   norm_weight:  [head_dim]
//   kv_state:     [coff * head_dim, coff * ratio]
//   score_state:  [coff * head_dim, coff * ratio]
//   cache:        [head_dim, compressed_capacity]
//
// The state/cache tensors are mutated. `start_pos == 0` resets them. The
// function accepts arbitrary chunk sizes at any start position and returns
// the number of newly emitted compressed vectors, or -1 on invalid input.
int kernel_dsv4_compressor(
    const Tensor& hidden,
    const Tensor& wkv,
    const Tensor& wgate,
    const Tensor& ape,
    const Tensor& norm_weight,
    Tensor& kv_state,
    Tensor& score_state,
    Tensor& cache,
    int start_pos,
    const Dsv4CompressorConfig& config,
    ThreadPool* thread_pool = nullptr);

// Learned compressed-history selector used by ratio-4 attention layers.
// `indices` is INT32 [top_k, sequence]; unused slots are set to -1.
// The indexer compressor state/cache is updated before scoring, matching the
// reference implementation.
bool kernel_dsv4_indexer(
    const Tensor& hidden,
    const Tensor& q_lora,
    const Tensor& wq_b,
    const Tensor& weights_projection,
    const Tensor& compressor_wkv,
    const Tensor& compressor_wgate,
    const Tensor& compressor_ape,
    const Tensor& compressor_norm,
    Tensor& compressor_kv_state,
    Tensor& compressor_score_state,
    Tensor& index_cache,
    Tensor& indices,
    int start_pos,
    const Dsv4IndexerConfig& config,
    ThreadPool* thread_pool = nullptr);

// Shared-KV sparse attention. Q is [num_heads * head_dim, sequence], current
// KV is [head_dim, sequence], and output has Q's shape. The window cache is a
// plain FP32 [head_dim, window_size] ring. Compressed indices, when present,
// directly index `compressed_cache`; the serialized graph does not need to
// emulate the reference kernel's temporary combined-KV array.
bool kernel_dsv4_sparse_attention(
    const Tensor& query,
    const Tensor& current_kv,
    const Tensor& attention_sink,
    Tensor& window_cache,
    const Tensor* compressed_cache,
    const Tensor* compressed_indices,
    int start_pos,
    Tensor& output,
    const Dsv4SparseAttentionConfig& config,
    ThreadPool* thread_pool = nullptr);

// Group-specific low-rank output projection:
// input [groups * group_width, sequence],
// weight [groups * rank, group_width],
// output [groups * rank, sequence].
bool kernel_dsv4_grouped_linear(
    const Tensor& input, const Tensor& weight, Tensor& output,
    int groups, ThreadPool* thread_pool = nullptr);

// Model-specific numerical helpers exposed for deterministic unit/reference
// validation.
void dsv4_apply_rope(float* values, int width, int position,
                     const Dsv4RopeConfig& config, bool inverse = false);
void dsv4_hadamard_rotate(float* values, int width);
void dsv4_fp4_simulate_inplace(float* values, int width,
                               int group_size = 32);
void dsv4_fp8_simulate_inplace(float* values, int width,
                               int group_size = 64);
