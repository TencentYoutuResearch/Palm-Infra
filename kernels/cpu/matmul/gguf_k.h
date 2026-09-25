#pragma once

#include "core/tensor.h"
#include "kernels/cpu/activations.h"

#include <cstddef>
#include <cstdint>

class ThreadPool;

// Decode a raw GGUF Q8_0 or K-quant block without a dense sidecar.
size_t gguf_k_block_bytes(Precision precision);
size_t gguf_k_block_values(Precision precision);
bool gguf_k_dequantize_block(Precision precision, const uint8_t* block,
                             float* output);
void kernel_matmul_gguf_k(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* pool, Activation act,
                          int act_n_begin, int act_n_len);
