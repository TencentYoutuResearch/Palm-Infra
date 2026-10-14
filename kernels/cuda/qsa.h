#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace mollm_cuda {
// Selected-key attention primitive, not yet a graph/backend selector.
// Indices are unique cache token IDs, padded with -1, already intersected
// with the indexer's visible mask. This primitive also bounds reads by causal
// position and actual key length (not capacity). Query/output features are
// contiguous FP32; cache features contiguous FP16 or FP32. All strides below
// are in ELEMENTS of the corresponding storage, not bytes.
struct QsaSelectedAttentionParams {
    int heads=0, kv_heads=0, queries=0, keys=0, past=0;
    int key_dim=0, value_dim=0, slots=0;
    float scale=0;
    size_t query_position_stride=0, query_head_stride=0;
    size_t key_position_stride=0, key_head_stride=0;
    size_t value_position_stride=0, value_head_stride=0;
    size_t index_position_stride=0;
    size_t output_position_stride=0, output_head_stride=0;
};

// No allocation, host readback, full-history score buffer or expanded KV copy.
// Invalid/padded/future indices are inactive, so they cannot cause an OOB read;
// an entirely inactive row returns zero. Caller owns all distinct buffers.
// This is a correctness-first implementation, not a throughput qualification.
cudaError_t launch_qsa_selected_attention(const float* query, const void* key,
    const void* value, bool fp16_cache, const int32_t* indices, float* output,
    const QsaSelectedAttentionParams& p, cudaStream_t stream=nullptr);

// Contiguous-causal block-selection primitive. Caller provides finite FP32
// block scores [queries, score_stride] and output [queries, index_stride].
// Scores for invisible/future blocks are ignored. Selects budget/ratio whole
// groups plus ALL incomplete visible tail, sorts resulting IDs chronologically,
// fills unused slots with -1. It does not compute projections/indexer scores.
// Current sorting geometry supports <=8192 groups (32768 keys at ratio4).
// Cutoff ties use stable ascending block IDs; official topk tie order is not
// specified. No GPU-to-CPU readback and no per-call scratch allocation.
cudaError_t launch_qsa_causal_block_topk(const float* scores,int32_t* indices,
    int queries,int keys,int past,int budget,int ratio,size_t score_stride,
    size_t index_stride,cudaStream_t stream=nullptr);
}
