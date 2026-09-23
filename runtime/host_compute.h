#pragma once

#include "core/prepared_weight.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class ThreadPool;

using HostPackedWeightMap =
    std::unordered_map<std::string, std::vector<uint8_t>>;

// Host calculations used by model loading and accelerator fallback. The CPU
// backend owns the implementation; engine code depends only on this contract.
void host_matmul_fp32(const Tensor& input, const Tensor& weight,
                      Tensor& output, ThreadPool* thread_pool);
void host_prepare_matmul_weight(
    Tensor& weight, const std::string& key, const void* data,
    HostPackedWeightMap& packed_weights, PreparedWeightMap& prepared_weights,
    bool pack_fp16, bool pack_fp8);
bool host_prepare_fp8_bf16_fp16_weight(
    Tensor& weight, const std::string& key, const void* data,
    HostPackedWeightMap& packed_weights);
bool host_packed_int4_supported();
bool host_arm_neon_available();
size_t host_packed_int4_g32_bytes(int rows, int columns);
size_t host_packed_int4_g128_bytes(int rows, int columns);
void host_set_matmul_profile_phase(const char* phase);
int host_argmax_token(const float* logits, int vocab_size);
