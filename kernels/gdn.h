#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"
#include "kernels/threading.h"

// ---------------------------------------------------------------------------
// Gated Delta Rule (GDN) kernel for Qwen3.5 linear attention layers.
//
// Delta rule recurrence:
//   state_t = state_{t-1} * exp(g_t) + outer(k_t, (v_t - state_{t-1} @ k_t) * beta_t)
//   out_t   = (state_t @ q_t) * scale
//
// State is [num_heads, k_head_dim, v_head_dim] (FP32 for precision).
// Q and K are L2-normalized before the recurrence (if use_qk_l2norm=1).
//
// Params (i32): [num_heads, k_head_dim, v_head_dim, use_qk_l2norm]
//
// Inputs:
//   [0] q       [k_head_dim, seq, num_heads] FP32
//   [1] k       [k_head_dim, seq, num_heads] FP32
//   [2] v       [v_head_dim, seq, num_heads] FP32
//   [3] g       [seq, num_heads] FP32
//   [4] beta    [seq, num_heads] FP32
//   [5] state   [v_head_dim, k_head_dim, num_heads] FP32 (in-place modified)
//
// Output:
//   [0] out     [v_head_dim, seq, num_heads] FP32
//
// State is modified in-place (like KV cache in SDPA). The caller allocates
// and persists the state buffer across prefill/decode calls.
// ---------------------------------------------------------------------------

void kernel_gdn_prefill(const OpParams& params,
                        const std::vector<const Tensor*>& inputs,
                        std::vector<Tensor*>& outputs,
                        ThreadPool* thread_pool = nullptr);

void kernel_gdn_decode(const OpParams& params,
                       const std::vector<const Tensor*>& inputs,
                       std::vector<Tensor*>& outputs,
                       ThreadPool* thread_pool = nullptr);
