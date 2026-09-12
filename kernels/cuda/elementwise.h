#pragma once

#include <cstddef>
#include <cstdint>

namespace mollm_cuda {

struct ArgMaxPair {
    float value;
    int index;
};

bool launch_round_to_bf16(float* values, size_t count);
void launch_apply_activation(float* values, int rows, int columns, int kind,
                             int begin, int end);
void launch_binary(const float* lhs, const float* rhs, float* output,
                   size_t count, bool multiply);
void launch_sigmoid_mul(const float* value, const float* gate, float* output,
                        size_t count);
void launch_sigmoid_mul_strided(
    const float* value, const float* gate, float* output, size_t count,
    int64_t d0, int64_t d1, int64_t d2, size_t value_s0, size_t value_s1,
    size_t value_s2, size_t value_s3, size_t gate_s0, size_t gate_s1,
    size_t gate_s2, size_t gate_s3);
void launch_unary(const float* input, float* output, size_t count,
                  int operation);
void launch_swiglu(const float* input, float* output, size_t output_count,
                   size_t half);
void launch_rms_norm(const float* input, const float* weight, float* output,
                     int width, int rows, float epsilon);
void launch_add_rms_norm(
    float* residual, const float* update, const float* weight, float* output,
    int width, int rows, size_t residual_row_stride,
    size_t update_row_stride, size_t output_row_stride, float epsilon);
void launch_contiguous(
    const float* input, float* output, size_t count, int64_t d0, int64_t d1,
    int64_t d2, size_t s0, size_t s1, size_t s2, size_t s3);
void launch_argmax(const float* input, int count, ArgMaxPair* partial,
                   int groups, ArgMaxPair* result);

}  // namespace mollm_cuda
