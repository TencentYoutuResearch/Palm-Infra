#pragma once

#include "kernels/tensor.h"

class ThreadPool;

// ---------------------------------------------------------------------------
// mollm — Normalization kernels
//
// RMSNorm: out = x / rms(x) * weight
//   rms(x) = sqrt(mean(x^2) + eps)
//
//   x:      [D, N]  — D features, N rows
//   weight: [D]     — per-feature gamma
//   out:    [D, N]
// ---------------------------------------------------------------------------

void kernel_rms_norm(const Tensor& x, const Tensor& weight,
                     float eps, Tensor& out);

// Update residual in place, then normalize the updated value into out.
void kernel_add_rms_norm(Tensor& residual, const Tensor& update,
                         const Tensor& weight, float eps, Tensor& out,
                         ThreadPool* thread_pool = nullptr);

void kernel_layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                       float eps, Tensor& out, ThreadPool* thread_pool = nullptr);
