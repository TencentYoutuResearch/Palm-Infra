#pragma once

#include "core/rwkv_params.h"
#include "kernels/tensor.h"
#include "runtime/threading.h"

void kernel_rwkv_token_shift(const RwkvTokenShiftParams&, const std::vector<const Tensor*>&,
                             Tensor&);
void kernel_rwkv_mix(const std::vector<const Tensor*>&, Tensor&);
void kernel_rwkv_l2_norm(const RwkvL2NormParams&, const std::vector<const Tensor*>&,
                         Tensor&);
void kernel_rwkv_post(const RwkvPostParams&, const std::vector<const Tensor*>&, Tensor&,
                      ThreadPool* = nullptr);
void kernel_rwkv7(const Rwkv7Params&, const std::vector<const Tensor*>&, Tensor&,
                  ThreadPool* = nullptr);
