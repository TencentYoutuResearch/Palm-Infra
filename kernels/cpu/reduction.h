#pragma once

// ---------------------------------------------------------------------------
// mollm — Reduction kernels
//
// kernel_argmax: index of the first maximum of values[0..count). count >= 1.
// ---------------------------------------------------------------------------
int kernel_argmax(const float* values, int count);
