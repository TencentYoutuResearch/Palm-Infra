#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace mollm_cuda {
// Selected-key attention primitive used by the graph/backend selector.
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
    size_t mask_column_stride=0, mask_row_stride=0;
    // Opt-in ordered FP32 GQA kernel: stages each selected KV tile
    // once for all query heads sharing that KV head. D256, groups2..16.
    bool reuse_gqa=false;
    // Split independent QK scores for small batches; PV keeps original order.
    int partitions=1;
    // Ordered optimized path: caller owns [queries,heads,slots] FP32 scores.
    // Preserve legacy normalization/PV FMA order, important around subsequent
    // low-precision quantization boundaries. No full-history dense scores.
    float* selected_scores=nullptr;
};

// No allocation, host readback, full-history score buffer or expanded KV copy.
// Invalid/padded/future indices are inactive, so they cannot cause an OOB read;
// an entirely inactive row returns zero. Output must not alias any input;
// read-only key/value buffers may alias each other.
// If reuse_gqa or partitions>1, selected_scores is required, disjoint from
// all inputs/output, with queries*heads*slots FP32 elements. No host readback.
// This is a correctness-first implementation, not a throughput qualification.
cudaError_t launch_qsa_selected_attention(const float* query, const void* key,
    const void* value, bool fp16_cache, const int32_t* indices, float* output,
    const QsaSelectedAttentionParams& p, cudaStream_t stream=nullptr,
    const float* mask=nullptr);

// Flag is reset by this launch. 0=canonical contiguous causal visibility,
// 1=unsupported gaps/future visibility/nonnegative or NaN future mask.
// Visible values must equal0; future values must be negative (-inf allowed).
// No CPU readback inside this allocation-free primitive.
cudaError_t launch_qsa_validate_causal_mask(const float* mask,uint32_t* invalid,
    int queries,int keys,int past,size_t column_stride,size_t row_stride,
    cudaStream_t stream=nullptr);

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

// FP32-activation indexer path consistent with the existing FP32 graph.
// It consumes a separate dense projection's [Q, (heads+1)*dimension] result;
// norm weights are ORIGINAL zero-centered weights (the kernel applies 1+w).
// inv_frequency is derived from the MAIN attention rope width, has rope_dim/2
// FP32 entries, and is shared by query and pooled-key rotation.
struct QsaIndexerParams {
    int heads=4, dimension=128, rope_dim=64, ratio=4;
    int queries=0, past=0, capacity=0, position_offset=0;
    float eps=1e-6f;
    size_t projection_stride=0, score_stride=0;
};
// Contiguous causal text only; no masked-gap grouping. Persistent raw_cache
// [capacity,dimension] MUST contain unnormalized/unrotated prior K. Persistent
// pooled_cache [capacity/ratio,dimension] contains only COMPLETE normalized,
// rotated groups. Only newly completed groups (including a cross-chunk group)
// are recomputed. Caller resets logical past to0; stale future cache is never
// read. Caller owns query [Q,heads,dimension] and scores [Q,score_stride].
// All buffers distinct FP32; no allocation/readback or metadata mutation.
cudaError_t launch_qsa_indexer_update(const float* projection,const float* q_norm,
    const float* k_norm,const float* inv_frequency,float* raw_cache,
    float* pooled_cache,float* query,float* scores,const QsaIndexerParams& p,
    cudaStream_t stream=nullptr);
}
