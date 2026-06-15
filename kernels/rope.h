#pragma once

#include "kernels/tensor.h"

// ---------------------------------------------------------------------------
// PROJECT_NAME — Rotary Embedding
//
// Applies RoPE to input tensor x.
//
//   x:      [D, N]  — D features, N rows.  Last rope_dim features are
//                      rotated in pairs (interleave mode) or halves.
//   cos:    [rope_dim/2, N] — precomputed cos values
//   sin:    [rope_dim/2, N] — precomputed sin values
//   out:    [D, N]
//
// interleave = true:  pairs are (0,1), (2,3), ... (GPT-NeoX style)
// interleave = false: pairs are (0, rope_dim/2), (1, rope_dim/2+1), ...
// ---------------------------------------------------------------------------

void kernel_rope(const Tensor& x, const Tensor& cos, const Tensor& sin,
                 int rope_dim, bool interleave, Tensor& out);
