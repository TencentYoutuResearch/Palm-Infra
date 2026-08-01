#include "engine/cuda_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool close_enough(const float* actual, const float* expected, size_t count,
                  float tolerance, const char* label) {
    float maximum = 0.0f;
    size_t maximum_index = 0;
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(actual[index]) ||
            !std::isfinite(expected[index])) {
            std::fprintf(
                stderr,
                "%s non-finite value at %zu: actual=%g expected=%g\n",
                label, index, actual[index], expected[index]);
            return false;
        }
        const float error = std::fabs(actual[index] - expected[index]);
        if (error > maximum) {
            maximum = error;
            maximum_index = index;
        }
    }
    if (maximum <= tolerance)
        return true;
    std::fprintf(stderr,
                 "%s mismatch at %zu: actual=%g expected=%g error=%g\n",
                 label, maximum_index, actual[maximum_index],
                 expected[maximum_index], maximum);
    return false;
}

Tensor device_tensor(CudaBackend& backend, int64_t d0, int64_t d1 = 1,
                     int64_t d2 = 1);
Tensor physical_float_state(CudaBackend& backend, int64_t elements);
Tensor device_constant(CudaBackend& backend, std::vector<float>& values,
                       int64_t d0, int64_t d1 = 1);
void fill_values(float* values, size_t count, int modulus, float divisor);
void gdn_reference(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, int num_heads, int key_dim, int value_dim,
    int sequence_length, int real_length, int num_value_heads,
    int a_row_stride, int b_row_stride, int z_row_stride,
    float rms_epsilon, float l2_epsilon, float scale);

bool test_gdn_long_prefill_decode(CudaBackend& backend) {
    constexpr int heads = 16;
    constexpr int key_dim = 128;
    constexpr int value_dim = 128;
    constexpr int value_heads = 16;
    constexpr int sequence = 224;
    constexpr int qkv_dim = heads * key_dim;
    constexpr int qkv_total = 2 * qkv_dim + value_heads * value_dim;
    constexpr int output_dim = value_heads * value_dim;
    constexpr size_t state_elements =
        static_cast<size_t>(value_heads) * key_dim * value_dim;
    constexpr float rms_epsilon = 1e-6f;
    constexpr float l2_epsilon = 1e-6f;
    const float scale = 1.0f / std::sqrt(static_cast<float>(key_dim));

    Tensor qkv = device_tensor(backend, qkv_total, sequence);
    Tensor a = device_tensor(backend, value_heads, sequence);
    Tensor b = device_tensor(backend, value_heads, sequence);
    Tensor z = device_tensor(backend, output_dim, sequence);
    fill_values(qkv.ptr<float>(), qkv.nelements(), 257, 181.0f);
    fill_values(a.ptr<float>(), a.nelements(), 31, 23.0f);
    fill_values(b.ptr<float>(), b.nelements(), 37, 29.0f);
    fill_values(z.ptr<float>(), z.nelements(), 257, 197.0f);

    std::vector<float> a_log(value_heads), dt_bias(value_heads),
        norm(value_dim);
    for (int index = 0; index < value_heads; ++index) {
        a_log[index] = -0.7f + 0.025f * index;
        dt_bias[index] = -0.3f + 0.04f * index;
    }
    for (int index = 0; index < value_dim; ++index)
        norm[index] = 0.8f + 0.003f * index;
    Tensor a_log_tensor = device_constant(backend, a_log, value_heads);
    Tensor dt_bias_tensor = device_constant(backend, dt_bias, value_heads);
    Tensor norm_tensor = device_constant(backend, norm, value_dim);
    Tensor state = physical_float_state(backend, state_elements);
    std::fill_n(static_cast<float*>(state.data), state_elements, 0.0f);
    std::vector<float> expected_state(state_elements, 0.0f);
    std::vector<float> expected_output(
        static_cast<size_t>(sequence) * output_dim);
    gdn_reference(
        qkv.ptr<float>(), a.ptr<float>(), b.ptr<float>(), z.ptr<float>(),
        a_log.data(), dt_bias.data(), norm.data(), expected_state.data(),
        expected_output.data(), heads, key_dim, value_dim, sequence,
        sequence, value_heads, value_heads, value_heads, output_dim,
        rms_epsilon, l2_epsilon, scale);

    Tensor output = device_tensor(backend, output_dim, sequence);
    GraphNode prefill;
    prefill.op_type = OpType::GATED_DELTANET_PREFILL;
    prefill.params.i32 = {heads, key_dim, value_dim, sequence,
                          1, 4, sequence, value_heads};
    prefill.params.f32 = {rms_epsilon, l2_epsilon, scale};
    backend.clear_dispatch_error();
    backend.dispatch(
        prefill,
        {&qkv, &a, &b, &z, &a_log_tensor, &dt_bias_tensor, &norm_tensor,
         &state},
        &output, nullptr);
    backend.end_graph();
    if (backend.dispatch_failed() ||
        !close_enough(output.ptr<float>(), expected_output.data(),
                      expected_output.size(), 3e-4f,
                      "GDN long prefill output") ||
        !close_enough(static_cast<float*>(state.data), expected_state.data(),
                      expected_state.size(), 3e-4f,
                      "GDN long prefill state")) {
        return false;
    }

    Tensor decode_qkv = device_tensor(backend, qkv_total);
    Tensor decode_a = device_tensor(backend, value_heads);
    Tensor decode_b = device_tensor(backend, value_heads);
    Tensor decode_z = device_tensor(backend, output_dim);
    fill_values(decode_qkv.ptr<float>(), decode_qkv.nelements(), 251, 173.0f);
    fill_values(decode_a.ptr<float>(), decode_a.nelements(), 29, 31.0f);
    fill_values(decode_b.ptr<float>(), decode_b.nelements(), 31, 37.0f);
    fill_values(decode_z.ptr<float>(), decode_z.nelements(), 251, 211.0f);
    std::vector<float> expected_decode(output_dim);
    gdn_reference(
        decode_qkv.ptr<float>(), decode_a.ptr<float>(), decode_b.ptr<float>(),
        decode_z.ptr<float>(), a_log.data(), dt_bias.data(), norm.data(),
        expected_state.data(), expected_decode.data(), heads, key_dim,
        value_dim, 1, 1, value_heads, value_heads, value_heads, output_dim,
        rms_epsilon, l2_epsilon, scale);
    Tensor decode_output = device_tensor(backend, output_dim);
    GraphNode decode;
    decode.op_type = OpType::GATED_DELTANET_DECODE;
    decode.params.i32 =
        {heads, key_dim, value_dim, 1, 1, 4, 1, value_heads};
    decode.params.f32 = {rms_epsilon, l2_epsilon, scale};
    backend.clear_dispatch_error();
    backend.dispatch(
        decode,
        {&decode_qkv, &decode_a, &decode_b, &decode_z, &a_log_tensor,
         &dt_bias_tensor, &norm_tensor, &state},
        &decode_output, nullptr);
    backend.end_graph();
    return !backend.dispatch_failed() &&
        close_enough(decode_output.ptr<float>(), expected_decode.data(),
                     expected_decode.size(), 3e-4f,
                     "GDN long continuation output") &&
        close_enough(static_cast<float*>(state.data), expected_state.data(),
                     expected_state.size(), 3e-4f,
                     "GDN long continuation state");
}

Tensor device_tensor(CudaBackend& backend, int64_t d0, int64_t d1,
                     int64_t d2) {
    Tensor tensor = Tensor::create(Precision::FP32, MemoryType::NONE,
                                   d0, d1, d2);
    backend.alloc_persistent(tensor, tensor.nbytes());
    return tensor;
}

Tensor physical_float_state(CudaBackend& backend, int64_t elements) {
    // Package graphs historically label recurrent state inputs FP16, while
    // Engine::allocate_caches gives them physical FP32 storage.
    Tensor tensor = Tensor::create(Precision::FP16, MemoryType::EXTERNAL,
                                   elements);
    backend.alloc_persistent(tensor,
                             static_cast<size_t>(elements) * sizeof(float));
    return tensor;
}

Tensor device_constant(CudaBackend& backend, std::vector<float>& values,
                       int64_t d0, int64_t d1) {
    Tensor tensor = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                   d0, d1, 1, 1, values.data());
    backend.wrap_weight(tensor);
    return tensor;
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

float softplus(float value) {
    if (value > 20.0f)
        return value;
    if (value < -20.0f)
        return std::exp(value);
    return std::log1p(std::exp(value));
}

void shortconv_reference(const float* input, const float* weight,
                         float* state, float* output, int groups,
                         int sequence_length, int kernel_size,
                         int real_length, int input_row_stride) {
    const int prefix = kernel_size - 1;
    const int process = real_length > 0 && real_length < sequence_length
        ? real_length : sequence_length;
    std::vector<float> window(prefix + sequence_length);
    for (int group = 0; group < groups; ++group) {
        for (int index = 0; index < prefix; ++index)
            window[index] = state[group * prefix + index];
        for (int token = 0; token < sequence_length; ++token)
            window[prefix + token] =
                input[token * input_row_stride + group];
        for (int token = 0; token < sequence_length; ++token) {
            float sum = 0.0f;
            for (int inner = 0; inner < kernel_size; ++inner)
                sum += window[token + inner] *
                    weight[group * kernel_size + inner];
            output[group * sequence_length + token] = token < process
                ? sum * sigmoid(sum) : 0.0f;
        }
        for (int index = 0; index < prefix; ++index)
            state[group * prefix + index] = window[process + index];
    }
}

void gdn_reference(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, int num_heads, int key_dim, int value_dim,
    int sequence_length, int real_length, int num_value_heads,
    int a_row_stride, int b_row_stride, int z_row_stride,
    float rms_epsilon, float l2_epsilon, float scale) {
    const int process = real_length > 0 && real_length < sequence_length
        ? real_length : sequence_length;
    const int repeat = num_value_heads / num_heads;
    const int qkv_dim = num_heads * key_dim;
    const int output_dim = num_value_heads * value_dim;
    const int state_size = key_dim * value_dim;
    std::vector<float> query(key_dim), key(key_dim);
    std::vector<float> attention(value_dim);

    std::fill(output, output + static_cast<size_t>(sequence_length) *
                             output_dim, 0.0f);
    for (int value_head = 0; value_head < num_value_heads; ++value_head) {
        const int key_head = value_head / repeat;
        float* state_head = state + value_head * state_size;
        for (int token = 0; token < process; ++token) {
            float query_sum = 0.0f;
            float key_sum = 0.0f;
            for (int dimension = 0; dimension < key_dim; ++dimension) {
                query[dimension] = qkv[
                    (key_head * key_dim + dimension) * sequence_length +
                    token];
                key[dimension] = qkv[
                    (qkv_dim + key_head * key_dim + dimension) *
                        sequence_length + token];
                query_sum += query[dimension] * query[dimension];
                key_sum += key[dimension] * key[dimension];
            }
            const float query_inverse =
                1.0f / std::sqrt(query_sum + l2_epsilon);
            const float key_inverse =
                1.0f / std::sqrt(key_sum + l2_epsilon);
            float query_key = 0.0f;
            for (int dimension = 0; dimension < key_dim; ++dimension) {
                query[dimension] *= query_inverse;
                key[dimension] *= key_inverse;
                query_key += query[dimension] * key[dimension];
            }
            const float decay = std::exp(
                -std::exp(a_log[value_head]) *
                softplus(a[token * a_row_stride + value_head] +
                         dt_bias[value_head]));
            const float beta = sigmoid(
                b[token * b_row_stride + value_head]);
            float attention_sum = 0.0f;
            for (int value_column = 0; value_column < value_dim;
                 ++value_column) {
                float memory = 0.0f;
                float decayed_attention = 0.0f;
                for (int key_row = 0; key_row < key_dim; ++key_row) {
                    const float decayed =
                        state_head[key_row * value_dim + value_column] * decay;
                    memory += decayed * key[key_row];
                    decayed_attention += decayed * query[key_row];
                }
                const float current_value = qkv[
                    (2 * qkv_dim + value_head * value_dim + value_column) *
                        sequence_length + token];
                const float delta = (current_value - memory) * beta;
                for (int key_row = 0; key_row < key_dim; ++key_row)
                    state_head[key_row * value_dim + value_column] =
                        state_head[key_row * value_dim + value_column] * decay +
                        key[key_row] * delta;
                attention[value_column] =
                    (decayed_attention + delta * query_key) * scale;
                attention_sum +=
                    attention[value_column] * attention[value_column];
            }
            const float rms = 1.0f /
                std::sqrt(attention_sum / value_dim + rms_epsilon);
            for (int value_column = 0; value_column < value_dim;
                 ++value_column) {
                const float z_value = z[
                    token * z_row_stride + value_head * value_dim +
                    value_column];
                output[token * output_dim + value_head * value_dim +
                       value_column] = attention[value_column] * rms *
                    norm_weight[value_column] * z_value * sigmoid(z_value);
            }
        }
    }
}

void fill_values(float* values, size_t count, int modulus, float divisor) {
    for (size_t index = 0; index < count; ++index)
        values[index] =
            static_cast<float>(static_cast<int>(index % modulus) -
                               modulus / 2) / divisor;
}

bool test_shortconv(CudaBackend& backend) {
    constexpr int groups = 7;
    constexpr int sequence = 5;
    constexpr int kernel = 4;
    constexpr int real = 3;
    Tensor input = device_tensor(backend, groups, sequence);
    fill_values(input.ptr<float>(), input.nelements(), 17, 19.0f);
    std::vector<float> weight_values(groups * kernel);
    fill_values(weight_values.data(), weight_values.size(), 13, 23.0f);
    Tensor weight =
        device_constant(backend, weight_values, groups, kernel);
    Tensor state = physical_float_state(backend, groups * (kernel - 1));
    fill_values(static_cast<float*>(state.data), groups * (kernel - 1),
                11, 29.0f);
    Tensor output = device_tensor(backend, groups, sequence);
    std::vector<float> expected_state(
        static_cast<float*>(state.data),
        static_cast<float*>(state.data) + groups * (kernel - 1));
    std::vector<float> expected_output(groups * sequence);
    shortconv_reference(input.ptr<float>(), weight_values.data(),
                        expected_state.data(), expected_output.data(), groups,
                        sequence, kernel, real, groups);
    GraphNode node;
    node.op_type = OpType::SHORTCONV;
    node.params.i32 = {kernel, real};
    backend.clear_dispatch_error();
    backend.dispatch(node, {&input, &weight, &state}, &output, nullptr);
    backend.end_graph();
    return !backend.dispatch_failed() &&
        close_enough(output.ptr<float>(), expected_output.data(),
                     expected_output.size(), 2e-6f, "shortconv output") &&
        close_enough(static_cast<float*>(state.data), expected_state.data(),
                     expected_state.size(), 1e-7f, "shortconv state");
}

struct GdnFixture {
    static constexpr int heads = 2;
    static constexpr int value_heads = 4;
    static constexpr int key_dim = 8;
    static constexpr int value_dim = 8;
    static constexpr int qkv_dim = heads * key_dim;
    static constexpr int qkv_total = 2 * qkv_dim + value_heads * value_dim;
    static constexpr int output_dim = value_heads * value_dim;
};

bool test_gdn_prefill(CudaBackend& backend) {
    using F = GdnFixture;
    constexpr int sequence = 5;
    constexpr int real = 3;
    Tensor qkv = device_tensor(backend, F::qkv_total, sequence);
    Tensor a = device_tensor(backend, F::value_heads, sequence);
    Tensor b = device_tensor(backend, F::value_heads, sequence);
    Tensor z = device_tensor(backend, F::output_dim, sequence);
    fill_values(qkv.ptr<float>(), qkv.nelements(), 19, 31.0f);
    fill_values(a.ptr<float>(), a.nelements(), 11, 17.0f);
    fill_values(b.ptr<float>(), b.nelements(), 13, 21.0f);
    fill_values(z.ptr<float>(), z.nelements(), 17, 27.0f);
    std::vector<float> a_log(F::value_heads), dt_bias(F::value_heads),
        norm(F::value_dim);
    fill_values(a_log.data(), a_log.size(), 7, 15.0f);
    fill_values(dt_bias.data(), dt_bias.size(), 9, 19.0f);
    for (int i = 0; i < F::value_dim; ++i)
        norm[i] = 0.8f + i * 0.025f;
    Tensor a_log_tensor =
        device_constant(backend, a_log, F::value_heads);
    Tensor dt_bias_tensor =
        device_constant(backend, dt_bias, F::value_heads);
    Tensor norm_tensor = device_constant(backend, norm, F::value_dim);
    Tensor state = physical_float_state(
        backend, F::value_heads * F::key_dim * F::value_dim);
    fill_values(static_cast<float*>(state.data),
                F::value_heads * F::key_dim * F::value_dim, 23, 101.0f);
    std::vector<float> expected_state(
        static_cast<float*>(state.data),
        static_cast<float*>(state.data) +
            F::value_heads * F::key_dim * F::value_dim);
    std::vector<float> expected_output(sequence * F::output_dim);
    constexpr float rms_epsilon = 1e-6f;
    constexpr float l2_epsilon = 1e-6f;
    const float scale = 1.0f / std::sqrt(static_cast<float>(F::key_dim));
    gdn_reference(qkv.ptr<float>(), a.ptr<float>(), b.ptr<float>(),
                  z.ptr<float>(), a_log.data(), dt_bias.data(), norm.data(),
                  expected_state.data(), expected_output.data(), F::heads,
                  F::key_dim, F::value_dim, sequence, real, F::value_heads,
                  F::value_heads, F::value_heads, F::output_dim,
                  rms_epsilon, l2_epsilon, scale);
    Tensor output = device_tensor(backend, F::output_dim, sequence);
    GraphNode node;
    node.op_type = OpType::GATED_DELTANET_PREFILL;
    node.params.i32 = {F::heads, F::key_dim, F::value_dim, sequence,
                       1, 4, real, F::value_heads};
    node.params.f32 = {rms_epsilon, l2_epsilon, scale};
    std::vector<const Tensor*> inputs = {
        &qkv, &a, &b, &z, &a_log_tensor, &dt_bias_tensor, &norm_tensor,
        &state};
    backend.clear_dispatch_error();
    backend.dispatch(node, inputs, &output, nullptr);
    backend.end_graph();
    return !backend.dispatch_failed() &&
        close_enough(output.ptr<float>(), expected_output.data(),
                     expected_output.size(), 4e-5f, "GDN prefill output") &&
        close_enough(static_cast<float*>(state.data), expected_state.data(),
                     expected_state.size(), 4e-5f, "GDN prefill state");
}

bool test_gdn_conv_decode(CudaBackend& backend) {
    using F = GdnFixture;
    constexpr int kernel = 4;
    Tensor raw_qkv = device_tensor(backend, F::qkv_total);
    Tensor a = device_tensor(backend, F::value_heads);
    Tensor b = device_tensor(backend, F::value_heads);
    Tensor z = device_tensor(backend, F::output_dim);
    fill_values(raw_qkv.ptr<float>(), raw_qkv.nelements(), 23, 37.0f);
    fill_values(a.ptr<float>(), a.nelements(), 7, 13.0f);
    fill_values(b.ptr<float>(), b.nelements(), 11, 17.0f);
    fill_values(z.ptr<float>(), z.nelements(), 13, 19.0f);
    std::vector<float> a_log(F::value_heads), dt_bias(F::value_heads),
        norm(F::value_dim), conv_weight(F::qkv_total * kernel);
    fill_values(a_log.data(), a_log.size(), 7, 17.0f);
    fill_values(dt_bias.data(), dt_bias.size(), 9, 23.0f);
    fill_values(conv_weight.data(), conv_weight.size(), 17, 41.0f);
    for (int i = 0; i < F::value_dim; ++i)
        norm[i] = 0.9f + i * 0.015f;
    Tensor a_log_tensor =
        device_constant(backend, a_log, F::value_heads);
    Tensor dt_bias_tensor =
        device_constant(backend, dt_bias, F::value_heads);
    Tensor norm_tensor = device_constant(backend, norm, F::value_dim);
    Tensor conv_weight_tensor =
        device_constant(backend, conv_weight, F::qkv_total, kernel);
    Tensor state = physical_float_state(
        backend, F::value_heads * F::key_dim * F::value_dim);
    Tensor conv_state =
        physical_float_state(backend, F::qkv_total * (kernel - 1));
    fill_values(static_cast<float*>(state.data),
                F::value_heads * F::key_dim * F::value_dim, 29, 113.0f);
    fill_values(static_cast<float*>(conv_state.data),
                F::qkv_total * (kernel - 1), 19, 67.0f);
    std::vector<float> initial_conv_state(
        static_cast<float*>(conv_state.data),
        static_cast<float*>(conv_state.data) +
            F::qkv_total * (kernel - 1));
    std::vector<float> expected_state(
        static_cast<float*>(state.data),
        static_cast<float*>(state.data) +
            F::value_heads * F::key_dim * F::value_dim);
    std::vector<float> expected_conv_state(
        static_cast<float*>(conv_state.data),
        static_cast<float*>(conv_state.data) +
            F::qkv_total * (kernel - 1));
    std::vector<float> convolved(F::qkv_total);
    shortconv_reference(raw_qkv.ptr<float>(), conv_weight.data(),
                        expected_conv_state.data(), convolved.data(),
                        F::qkv_total, 1, kernel, 1, F::qkv_total);
    Tensor conv_probe_state =
        physical_float_state(backend, F::qkv_total * (kernel - 1));
    std::memcpy(conv_probe_state.data, initial_conv_state.data(),
                initial_conv_state.size() * sizeof(float));
    Tensor conv_probe_output = device_tensor(backend, F::qkv_total);
    GraphNode conv_probe;
    conv_probe.op_type = OpType::SHORTCONV;
    conv_probe.params.i32 = {kernel, 1};
    backend.dispatch(conv_probe,
                     {&raw_qkv, &conv_weight_tensor, &conv_probe_state},
                     &conv_probe_output, nullptr);
    backend.end_graph();
    if (backend.dispatch_failed() ||
        !close_enough(conv_probe_output.ptr<float>(), convolved.data(),
                      convolved.size(), 2e-6f,
                      "GDN decode ShortConv staging"))
        return false;
    std::vector<float> expected_output(F::output_dim);
    const float scale = 1.0f / std::sqrt(static_cast<float>(F::key_dim));
    gdn_reference(convolved.data(), a.ptr<float>(), b.ptr<float>(),
                  z.ptr<float>(), a_log.data(), dt_bias.data(), norm.data(),
                  expected_state.data(), expected_output.data(), F::heads,
                  F::key_dim, F::value_dim, 1, 1, F::value_heads,
                  F::value_heads, F::value_heads, F::output_dim,
                  1e-6f, 1e-6f, scale);
    Tensor output = device_tensor(backend, F::output_dim);
    GraphNode node;
    node.op_type = OpType::GATED_DELTANET_CONV_DECODE;
    node.params.i32 = {F::heads, F::key_dim, F::value_dim, 1,
                       1, kernel, 1, F::value_heads};
    node.params.f32 = {1e-6f, 1e-6f, scale};
    std::vector<const Tensor*> inputs = {
        &raw_qkv, &a, &b, &z, &a_log_tensor, &dt_bias_tensor, &norm_tensor,
        &state, &conv_weight_tensor, &conv_state};
    backend.clear_dispatch_error();
    backend.dispatch(node, inputs, &output, nullptr);
    backend.end_graph();
    bool matches = !backend.dispatch_failed();
    matches &= close_enough(output.ptr<float>(), expected_output.data(),
                            expected_output.size(), 4e-5f,
                            "GDN conv decode output");
    matches &= close_enough(static_cast<float*>(state.data),
                            expected_state.data(), expected_state.size(),
                            4e-5f, "GDN conv decode state");
    matches &= close_enough(static_cast<float*>(conv_state.data),
                            expected_conv_state.data(),
                            expected_conv_state.size(), 1e-7f,
                            "GDN conv decode conv state");
    return matches;
}

}  // namespace

int main() {
    CudaBackend availability_backend;
    if (!availability_backend.available()) {
        std::fprintf(stderr, "CUDA device unavailable; skipping\n");
        return 77;
    }
    {
        CudaBackend backend;
        if (!test_shortconv(backend))
            return 1;
    }
    {
        CudaBackend backend;
        if (!test_gdn_prefill(backend))
            return 1;
    }
    {
        CudaBackend backend;
        if (!test_gdn_long_prefill_decode(backend))
            return 1;
    }
    {
        CudaBackend backend;
        if (!test_gdn_conv_decode(backend))
            return 1;
    }
    std::printf("CUDA recurrent ShortConv/Gated DeltaNet tests passed\n");
    return 0;
}
