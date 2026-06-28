#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"
#include "kernels/threading.h"

// ---------------------------------------------------------------------------
// Fused Gated Delta Rule (GDN) linear-attention core for Qwen3.5.
//
// Replaces the post-shortconv block: split qkv, g/beta compute, GDN recurrence,
// RMSNormGated. Fusing into one op sidesteps the shape-vs-data-layout ambiguity
// of zero-copy reshape/permute/slice views on matmul outputs (the root cause of
// the original garbled-text bug).
//
// Delta-rule recurrence (matches ncnn_llm gdr.cpp torch_recurrent_gated_delta_rule):
//   state_t = state_{t-1} * exp(g_t) + outer(k_t, (v_t - state_{t-1} @ k_t) * beta_t)
//   out_t   = (state_t @ q_t) * scale
// with
//   g_t    = -exp(A_log[h]) * softplus(a_t + dt_bias[h])
//   beta_t = sigmoid(b_t)
//   q_t, k_t L2-normalized per (head, token) when use_qk_l2norm=1
//
// RMSNormGated: out = rms_norm(core_attn_out, norm_weight) * silu(z)
//
// ---------------------------------------------------------------------------
// Input layout contract (CRITICAL):
//
// All four matmul-derived inputs are consumed in their NATIVE [seq, dim] row-major
// data layout — i.e. ptr[t * dim + d]. The declared Tensor shape ([dim, seq]) is
// NOT consulted by the kernel; seq_len and dims come from params. The builder
// emits a `reshape` materialize on qkv_conv (shortconv output) to bring it to
// [seq, qkv_total] so all four inputs share this convention.
//
//   inputs[0] qkv_conv     FP32  data [seq, qkv_total=3*num_heads*k_dim]  (post-shortconv, post-reshape)
//   inputs[1] a_out        FP32  data [seq, num_heads]
//   inputs[2] b_out        FP32  data [seq, num_heads]
//   inputs[3] z_out        FP32  data [seq, num_v_heads*v_dim]  (z_proj output)
//   inputs[4] A_log        FP32  [num_heads]               (CONSTANT)
//   inputs[5] dt_bias      FP32  [num_heads]               (CONSTANT)
//   inputs[6] norm_weight  FP32  [v_dim]                    (CONSTANT, RMSNorm gamma)
//   inputs[7] gdn_state    FP32  [num_heads, k_dim, v_dim]  (INPUT, in-place modified)
//                          layout: state[h*k_dim*v_dim + dk*v_dim + dv]
//
// Output:
//   outputs[0] out         FP32  data [seq, num_v_heads*v_dim] row-major
//                          (= RMSNormGated result, ready for out_proj matmul)
//
// Params:
//   i32[0] = num_heads         (key heads, used for q/k/a/b)
//   i32[1] = k_head_dim
//   i32[2] = v_head_dim
//   i32[3] = seq_len           (prefill: N, decode: 1)
//   i32[4] = use_qk_l2norm     (1 for Qwen3.5)
//   i32[5] = conv_kernel       (informational, unused — shortconv already done)
//   i32[6] = n_real_tokens     (runtime-injected by engine, 0 = all)
//   i32[7] = num_v_heads       (value heads, for z/out dim; defaults to num_heads)
//
//   f32[0] = rms_eps           (1e-6, RMSNorm eps)
//   f32[1] = l2norm_eps        (1e-6, L2 norm eps)
//   f32[2] = scale             (1/sqrt(k_dim))
//
// State (inputs[7]) is modified in-place through its data pointer (same pattern
// as SDPA's KV cache). The engine allocates and persists the buffer.
// ---------------------------------------------------------------------------

void kernel_gdn_prefill(const OpParams& params,
                        const std::vector<const Tensor*>& inputs,
                        std::vector<Tensor*>& outputs,
                        ThreadPool* thread_pool = nullptr);

void kernel_gdn_decode(const OpParams& params,
                       const std::vector<const Tensor*>& inputs,
                       std::vector<Tensor*>& outputs,
                       ThreadPool* thread_pool = nullptr);
