#pragma once

#include "core/tensor.h"

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

// ---------------------------------------------------------------------------
// Hy4 identity Hyper-Connections (iHC)
//
// Same tensor layouts as the DeepSeek-V4 kernels above, but a different
// formula: the inter-stream mixing matrix is dropped, so `fn` supplies only
// 2*hc coefficients instead of (2+hc)*hc, and no Sinkhorn iteration runs.
// The reduced pre-activation is scaled by `hc_magnitude` before the post
// gate. See docs_internal/HY4_OPS_SPEC.md section 3.
//
// HC_PRE_ID packs [reduced_hidden, post] (no combination block), because
// HC_POST_ID applies the gate per stream instead of mixing streams.
bool kernel_hc_pre_id(const Tensor& x, const Tensor& fn, const Tensor& scale,
                      const Tensor& base, Tensor& packed, int hidden_size,
                      int hc_mult, float hc_magnitude, float norm_eps,
                      float hc_eps, ThreadPool* thread_pool);

bool kernel_hc_post_id(const Tensor& branch, const Tensor& residual,
                       const Tensor& packed, Tensor& output, int hidden_size,
                       int hc_mult, ThreadPool* thread_pool);

bool kernel_hc_head_id(const Tensor& x, const Tensor& fn, const Tensor& scale,
                       const Tensor& base, Tensor& output, int hidden_size,
                       int hc_mult, float norm_eps, float hc_eps,
                       ThreadPool* thread_pool);
