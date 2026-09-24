#pragma once

#include "core/tensor.h"
#include "core/attention_params.h"
#include <vector>

class ThreadPool;

// ---------------------------------------------------------------------------
// mollm — SDPA kernels
//
// Standard SDPA (kv_cache=2):
//   Q:       [head_dim, src_seqlen, num_heads]
//   K_cur:   [head_dim, cur_seqlen, num_kv_heads]
//   V_cur:   [v_head_dim, cur_seqlen, num_kv_heads]
//   mask:    [dst_seqlen, src_seqlen]  or empty (no mask)
//   K_cache: [head_dim, capacity, num_kv_heads]  (h = past_seqlen)
//   V_cache: [v_head_dim, capacity, num_kv_heads] (h = past_seqlen)
//
//   output:  [v_head_dim, src_seqlen, num_heads]
//
// The kernel appends K_cur/V_cur to K_cache/V_cache in-place,
// then computes Q*K^T → softmax → *V.
//
// MLA SDPA: same interface, but handles:
//   - K_cache stores compressed representation (kv_lora_rank instead of
//     expanded head_dim).  Phase 1 treats MLA identically to standard SDPA
//     (the up-projection is done as separate Matmul nodes before this kernel).
// ---------------------------------------------------------------------------

/// Standard SDPA with in-place KV cache append.
void kernel_sdpa(const SdpaParams& params,
                 const std::vector<const Tensor*>& inputs,
                 std::vector<Tensor*>& outputs,
                 ThreadPool* thread_pool = nullptr);

/// Sparse SDPA driven by a precomputed key selection.
///
/// Identical to kernel_sdpa in every respect except that each query position
/// attends only to the cache positions named by `selection`. This is Hy4's
/// Gated DeepSeek Sparse Attention consumer: the selection comes from
/// kernel_hy4_indexer and is shared across consecutive layers by IndexCache.
///
/// Inputs match kernel_sdpa, plus one trailing entry:
///   [.. , K_cache, V_cache, selection]
/// where `selection` is INT32 [top_k, seq]; -1 marks an unused slot and rank
/// order is by descending score.
///
/// kernel_sdpa itself is untouched, so the dense path carries no risk from
/// this addition. The two share the underlying flash-attention kernels.
void kernel_sdpa_sparse(const SdpaParams& params,
                        const std::vector<const Tensor*>& inputs,
                        std::vector<Tensor*>& outputs,
                        ThreadPool* thread_pool = nullptr);
