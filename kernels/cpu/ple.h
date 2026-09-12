#pragma once

#include <cstdint>

#include "kernels/tensor.h"

class ThreadPool;

// Build Qwen4Exp PLE embedding row indices. `history` is ordered oldest to
// newest and is updated in-place, allowing the same function to serve prefill
// chunks and single-token decode.
bool ple_build_indices(const int32_t* token_ids, int seq_len,
                       int32_t* history, int context_len,
                       const int32_t* head_vocab_sizes,
                       const int32_t* head_offsets,
                       int ngram_size, int heads_per_ngram,
                       int eos_token_id, int unigram_vocab_size,
                       int ple_layer_index, uint64_t seed,
                       int32_t* output_indices);

// Gather raw E4M3 rows into token-major FP32 embeddings. This intentionally
// accepts a plain byte table so PLE can later source rows from mmap or a small
// SSD page cache without pretending the table is a block-scaled matmul weight.
bool ple_gather_fp8_rows(const uint8_t* table, int64_t total_rows,
                         int row_dim, const int32_t* indices,
                         int seq_len, int num_heads, float table_scale,
                         float* output);

bool kernel_ple_lookup(const Tensor& token_ids, Tensor& history,
                       const Tensor& table, const Tensor& table_scale,
                       const Tensor& head_vocab_sizes,
                       const Tensor& head_offsets, Tensor& output,
                       int ngram_size, int heads_per_ngram,
                       int eos_token_id, int unigram_vocab_size,
                       int ple_layer_index, uint64_t seed,
                       int n_real_tokens = -1,
                       ThreadPool* thread_pool = nullptr);

bool kernel_ple_gate(const Tensor& query, const Tensor& key,
                     const Tensor& value, Tensor& output,
                     int hidden_size, int hc_count,
                     ThreadPool* thread_pool = nullptr);

bool kernel_ple_dilated_conv(const Tensor& input, const Tensor& weight,
                             Tensor& state, Tensor& output,
                             int kernel_size, int dilation,
                             int n_real_tokens = -1,
                             ThreadPool* thread_pool = nullptr);
