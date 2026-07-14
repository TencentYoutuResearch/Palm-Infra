#pragma once

#include "kernels/tensor.h"

#include <vector>

class ThreadPool;

// Fused Qwen-style sparse MLP.
//
// Inputs:
//   0 hidden             FP32 [hidden, seq]
//   1 router             FP16/FP32 [num_experts, hidden]
//   2 experts_gate_up    FP16/FP32/INT8/INT4 [num_experts*2*intermediate, hidden]
//   3 experts_down       FP16/FP32/INT8/INT4 [num_experts*hidden, intermediate]
//   4 shared_gate        FP16/FP32/INT8/INT4 [shared_intermediate, hidden]
//   5 shared_up          FP16/FP32/INT8/INT4 [shared_intermediate, hidden]
//   6 shared_down        FP16/FP32/INT8/INT4 [hidden, shared_intermediate]
//   7 shared_expert_gate FP16/FP32 [1, hidden]
//   8 router_bias        optional FP32 [num_experts], used by sigmoid routers
//
// Output:
//   FP32 [hidden, seq]
void kernel_qwen3_moe(const std::vector<const Tensor*>& inputs,
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
                      float routed_scaling_factor = 1.0f);

extern "C" int mollm_moe_profile_enabled();
extern "C" void mollm_reset_moe_profile();
extern "C" void mollm_print_moe_profile(const char* title);
