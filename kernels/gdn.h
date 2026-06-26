#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"
#include "kernels/threading.h"

// ---------------------------------------------------------------------------
// Gated Delta Rule (GDN) kernel for Qwen3.5 linear attention layers.
//
// Ported from ncnn_llm/src/utils/gdr.cpp. The delta rule recurrence is:
//   state_t = state_{t-1} * exp(g_t) + outer(k_t, (v_t - state_{t-1} @ k_t) * beta_t)
//   out_t   = (state_t @ q_t) * scale
//
// State is [num_heads, k_head_dim, v_head_dim] (FP32 for precision).
// Q and K are L2-normalized before the recurrence.
//
// Two variants:
//   - prefill: processes seq_len > 1 tokens (loop over time)
//   - decode: processes single token (seq_len=1, state passed in/out)
//
// Params (i32): [num_heads, k_head_dim, v_head_dim, use_qk_l2norm]
// Inputs (prefill):
//   [0] q       [k_head_dim, seq, num_heads] FP32 (already L2-normed if use_qk_l2norm)
//   [1] k       [k_head_dim, seq, num_heads] FP32
//   [2] v       [v_head_dim, seq, num_heads] FP32
//   [3] g       [seq, num_heads] FP32 (decay gate, pre-computed as -exp(A)*softplus(a+dt_bias))
//   [4] beta    [seq, num_heads] FP32 (sigmoid(b))
//   [5] state   [v_head_dim, k_head_dim, num_heads] FP32 (recurrent, in/out)
// Outputs (prefill):
//   [0] out     [v_head_dim, seq, num_heads] FP32
//   [1] state   [v_head_dim, k_head_dim, num_heads] FP32 (updated)
// ---------------------------------------------------------------------------

void kernel_gdn_prefill(const OpParams& params,
                        const std::vector<const Tensor*>& inputs,
                        std::vector<Tensor*>& outputs,
                        ThreadPool* thread_pool = nullptr);

// Decode: seq_len=1, same params/inputs but q/k/v are single-token.
void kernel_gdn_decode(const OpParams& params,
                       const std::vector<const Tensor*>& inputs,
                       std::vector<Tensor*>& outputs,
                       ThreadPool* thread_pool = nullptr);
