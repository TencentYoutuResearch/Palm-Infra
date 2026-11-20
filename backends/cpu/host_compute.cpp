#include "runtime/host_compute.h"

#include "backends/cpu/platform.h"
#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/reduction.h"

void host_matmul_fp32(const Tensor& input, const Tensor& weight,
                      Tensor& output, ThreadPool* thread_pool,
                      bool force_fp32_acc) {
    kernel_matmul_fp32(input, weight, output, thread_pool,
                      Activation::NONE, 0, -1, force_fp32_acc);
}

void host_prepare_matmul_weight(
    Tensor& weight, const std::string& key, const void* data,
    HostPackedWeightMap& packed_weights, PreparedWeightMap& prepared_weights,
    bool pack_fp16, bool pack_fp8) {
    prepare_matmul_weight(weight, key, data, packed_weights,
                          prepared_weights, pack_fp16, pack_fp8);
}

bool host_prepare_fp8_bf16_fp16_weight(
    Tensor& weight, const std::string& key, const void* data,
    HostPackedWeightMap& packed_weights) {
    return prepare_fp8_bf16_fp16_weight(weight, key, data, packed_weights);
}

bool host_packed_int4_supported() {
    return matmul_int4_q4dot_kernel_available();
}

bool host_arm_neon_available() {
    return mollm::cpu::capabilities().arm_neon;
}

size_t host_packed_int4_g32_bytes(int rows, int columns) {
    return pack_b_q4dot_g32_bytes(rows, columns);
}

size_t host_packed_int4_g128_bytes(int rows, int columns) {
    return pack_b_q4dot_g128_bytes(rows, columns);
}

void host_set_matmul_profile_phase(const char* phase) {
    mollm_set_matmul_profile_phase(phase);
}

int host_argmax_token(const float* logits, int vocab_size) {
    return kernel_argmax(logits, vocab_size);
}
