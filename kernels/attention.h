#pragma once

#include "kernels/tensor.h"
#include "graph/graph.h"
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
/// params layout:
///   i32[0] = kv_cache mode (0/1/2)
///   i32[1] = causal (0/1)
///   i32[2] = num_heads
///   i32[3] = num_kv_heads
///   i32[4] = head_dim
///   i32[5] = v_head_dim
///   f32[0] = scale (0 → 1/sqrt(head_dim))
void kernel_sdpa(const OpParams& params,
                 const std::vector<const Tensor*>& inputs,
                 std::vector<Tensor*>& outputs,
                 ThreadPool* thread_pool = nullptr);
