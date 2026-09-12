#pragma once

#include "kernels/tensor.h"

#include <vector>

class ThreadPool;

// Compute only the dense/shared expert branch of an MoE layer. Keeping this
// independent from routed execution lets SSD device backends overlap routed
// expert I/O and compute with the CPU shared MLP.
bool kernel_moe_shared_expert(const Tensor& hidden,
                              const Tensor& gate,
                              const Tensor& up,
                              const Tensor& down,
                              const Tensor* scale_weight,
                              Tensor& output,
                              ThreadPool* thread_pool,
                              int intermediate_size,
                              bool emulate_bf16,
                              float swiglu_limit = 0.0f);

// Fused Qwen-style sparse MLP.
//
// Inputs:
//   0 hidden             FP32 [hidden, seq]
//   1 router             FP16/FP32 [num_experts, hidden]
//   2 experts_gate_up    FP16/FP32/INT8/INT4 [num_experts*2*intermediate, hidden]
//   3 experts_down       FP16/FP32/INT8/INT4 [num_experts*hidden, intermediate]
// Optional inputs after slot 3 are described by the input-index parameters:
// shared expert tensors occupy slots 4..6, followed by an optional shared
// expert gate, router bias, token ids, and hash table as selected by the graph.
//
// Output:
//   FP32 [hidden, seq]
// Returns false instead of leaving a partial output when routing or SSD I/O
// fails.
bool kernel_qwen3_moe(const std::vector<const Tensor*>& inputs,
                      Tensor& output,
                      ThreadPool* thread_pool,
                      int hidden_size,
                      int num_experts,
                      int top_k,
                      int intermediate_size,
                      int shared_intermediate_size,
                      int router_score_func = 0,
                      bool norm_topk_prob = true,
                      bool has_shared_expert = true,
                      int n_group = 1,
                      int topk_group = 1,
                      float routed_scaling_factor = 1.0f,
                      bool shared_expert_has_gate = true,
                      int router_bias_input = -1,
                      int token_ids_input = -1,
                      int hash_table_input = -1,
                      float swiglu_limit = 0.0f);

extern "C" int mollm_moe_profile_enabled();
extern "C" void mollm_reset_moe_profile();
extern "C" void mollm_print_moe_profile(const char* title);
