#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"
#include "kernels/threading.h"

void kernel_rwkv_token_shift(const OpParams&, const std::vector<const Tensor*>&,
                             Tensor&);
void kernel_rwkv_mix(const OpParams&, const std::vector<const Tensor*>&, Tensor&);
void kernel_rwkv_l2_norm(const OpParams&, const std::vector<const Tensor*>&,
                         Tensor&);
void kernel_rwkv_group_norm(const OpParams&, const std::vector<const Tensor*>&,
                            Tensor&);
void kernel_rwkv_bonus(const OpParams&, const std::vector<const Tensor*>&,
                       Tensor&);
void kernel_rwkv_post(const OpParams&, const std::vector<const Tensor*>&, Tensor&,
                      ThreadPool* = nullptr);
void kernel_rwkv7(const OpParams&, const std::vector<const Tensor*>&, Tensor&,
                  ThreadPool* = nullptr);
