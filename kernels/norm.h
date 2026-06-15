#pragma once

#include "kernels/tensor.h"

// ---------------------------------------------------------------------------
// PROJECT_NAME — Normalization kernels
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
