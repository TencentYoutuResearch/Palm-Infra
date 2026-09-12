#pragma once

#include "kernels/tensor.h"

class ThreadPool;

// DeepSeek-V4 Hyper-Connections. Tensors flatten [hc, hidden] into dim 0.
// HC_PRE packs [reduced_hidden, post, combination] into one graph output so
// the single-output graph format can retain all values needed by HC_POST.
bool kernel_hc_pre(const Tensor& x, const Tensor& fn, const Tensor& scale,
                   const Tensor& base, Tensor& packed, int hidden_size,
                   int hc_mult, int sinkhorn_iters, float norm_eps,
                   float sinkhorn_eps, ThreadPool* thread_pool);

bool kernel_hc_post(const Tensor& branch, const Tensor& residual,
                    const Tensor& packed, Tensor& output, int hidden_size,
                    int hc_mult, ThreadPool* thread_pool);

bool kernel_hc_head(const Tensor& x, const Tensor& fn, const Tensor& scale,
                    const Tensor& base, Tensor& output, int hidden_size,
                    int hc_mult, float norm_eps, float hc_eps,
                    ThreadPool* thread_pool);
