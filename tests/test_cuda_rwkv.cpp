#include "engine/cuda_backend.h"
#include "kernels/cpu_platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool close_enough(const float* actual, const float* expected, size_t count,
                  float absolute_tolerance, float relative_tolerance,
                  const char* label) {
    float maximum = 0.0f;
    size_t maximum_index = 0;
    for (size_t index = 0; index < count; ++index) {
        const float error = std::fabs(actual[index] - expected[index]);
        const float tolerance = absolute_tolerance +
            relative_tolerance * std::fabs(expected[index]);
        if (error > tolerance && error > maximum) {
            maximum = error;
            maximum_index = index;
        }
    }
    if (maximum == 0.0f)
        return true;
    std::fprintf(stderr,
                 "%s mismatch at %zu: actual=%g expected=%g error=%g\n",
                 label, maximum_index, actual[maximum_index],
                 expected[maximum_index], maximum);
    return false;
}

Tensor device_tensor(CudaBackend& backend, int64_t d0, int64_t d1 = 1,
                     int64_t d2 = 1) {
    Tensor tensor = Tensor::create(
        Precision::FP32, MemoryType::NONE, d0, d1, d2);
    backend.alloc_output(tensor, tensor.nbytes(), nullptr);
    return tensor;
}

Tensor device_constant(CudaBackend& backend, std::vector<float>& values,
                       int64_t d0) {
    Tensor tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, d0, 1, 1, 1, values.data());
    backend.wrap_weight(tensor);
    return tensor;
}

void fill_values(float* values, size_t count, int modulus, float divisor) {
    for (size_t index = 0; index < count; ++index)
        values[index] =
            static_cast<float>(static_cast<int>(index % modulus) -
                               modulus / 2) / divisor;
}

bool test_layer_norm(CudaBackend& backend) {
    constexpr int parent_width = 131;
    constexpr int width = 127;
    constexpr int rows = 5;
    Tensor parent = device_tensor(backend, parent_width, rows);
    fill_values(parent.ptr<float>(), parent.nelements(), 37, 19.0f);
    Tensor input;
    GraphNode slice;
    slice.op_type = OpType::SLICE;
    slice.params.i32 = {0, 2, width};
    backend.dispatch(slice, {&parent}, &input, nullptr);
    std::vector<float> weight(width), bias(width);
    fill_values(weight.data(), weight.size(), 29, 23.0f);
    fill_values(bias.data(), bias.size(), 17, 31.0f);
    Tensor weight_tensor = device_constant(backend, weight, width);
    Tensor bias_tensor = device_constant(backend, bias, width);
    Tensor output = device_tensor(backend, width, rows);
    GraphNode node;
    node.op_type = OpType::LAYER_NORM;
    node.params.f32 = {1e-5f};
    backend.dispatch(
        node, {&input, &weight_tensor, &bias_tensor}, &output, nullptr);
    backend.end_graph();

    std::vector<float> expected(width * rows);
    for (int row = 0; row < rows; ++row) {
        float mean = 0.0f;
        for (int dimension = 0; dimension < width; ++dimension)
            mean += parent.ptr<float>()[
                row * parent_width + dimension + 2];
        mean /= width;
        float variance = 0.0f;
        for (int dimension = 0; dimension < width; ++dimension) {
            const float centered = parent.ptr<float>()[
                row * parent_width + dimension + 2] - mean;
            variance += centered * centered;
        }
        const float inverse = 1.0f /
            std::sqrt(variance / width + 1e-5f);
        for (int dimension = 0; dimension < width; ++dimension)
            expected[row * width + dimension] =
                (parent.ptr<float>()[
                     row * parent_width + dimension + 2] - mean) *
                    inverse * weight[dimension] + bias[dimension];
    }
    return !backend.dispatch_failed() && close_enough(
        output.ptr<float>(), expected.data(), expected.size(), 3e-6f, 2e-5f,
        "RWKV LayerNorm");
}

bool test_token_shift(CudaBackend& backend) {
    constexpr int hidden = 67;
    constexpr int sequence = 5;
    constexpr int real = 3;
    Tensor input = device_tensor(backend, hidden, sequence);
    fill_values(input.ptr<float>(), input.nelements(), 41, 27.0f);
    Tensor state = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, hidden);
    backend.alloc_persistent(state, state.nbytes());
    auto* state_data = static_cast<mollm::cpu::fp16_t*>(state.data);
    std::vector<float> initial(hidden);
    fill_values(initial.data(), initial.size(), 31, 21.0f);
    for (int dimension = 0; dimension < hidden; ++dimension)
        state_data[dimension] =
            static_cast<mollm::cpu::fp16_t>(initial[dimension]);
    std::vector<float> expected(hidden * sequence, 0.0f);
    std::vector<float> previous(hidden);
    for (int dimension = 0; dimension < hidden; ++dimension)
        previous[dimension] = static_cast<float>(state_data[dimension]);
    for (int token = 0; token < real; ++token)
        for (int dimension = 0; dimension < hidden; ++dimension) {
            const float current = input.ptr<float>()[token * hidden +
                                                     dimension];
            expected[token * hidden + dimension] =
                previous[dimension] - current;
            previous[dimension] = static_cast<float>(
                static_cast<mollm::cpu::fp16_t>(current));
        }
    Tensor output = device_tensor(backend, hidden, sequence);
    GraphNode node;
    node.op_type = OpType::RWKV_TOKEN_SHIFT;
    node.params.i32 = {hidden, sequence, real};
    backend.dispatch(node, {&input, &state}, &output, nullptr);
    backend.end_graph();
    bool state_matches = true;
    for (int dimension = 0; dimension < hidden; ++dimension)
        state_matches &= static_cast<float>(state_data[dimension]) ==
            previous[dimension];
    return !backend.dispatch_failed() && state_matches && close_enough(
        output.ptr<float>(), expected.data(), expected.size(), 0.0f, 0.0f,
        "RWKV token shift");
}

bool test_mix(CudaBackend& backend) {
    constexpr int hidden = 65;
    constexpr int tokens = 4;
    Tensor input = device_tensor(backend, hidden, tokens);
    Tensor shift = device_tensor(backend, hidden, tokens);
    Tensor mix = device_tensor(backend, hidden);
    fill_values(input.ptr<float>(), input.nelements(), 43, 29.0f);
    fill_values(shift.ptr<float>(), shift.nelements(), 37, 31.0f);
    fill_values(mix.ptr<float>(), mix.nelements(), 23, 17.0f);
    std::vector<float> expected(hidden * tokens);
    for (int token = 0; token < tokens; ++token)
        for (int dimension = 0; dimension < hidden; ++dimension) {
            const int index = token * hidden + dimension;
            expected[index] = input.ptr<float>()[index] +
                shift.ptr<float>()[index] * mix.ptr<float>()[dimension];
        }
    Tensor output = device_tensor(backend, hidden, tokens);
    GraphNode node;
    node.op_type = OpType::RWKV_MIX;
    backend.dispatch(node, {&input, &shift, &mix}, &output, nullptr);
    backend.end_graph();
    return !backend.dispatch_failed() && close_enough(
        output.ptr<float>(), expected.data(), expected.size(), 2e-7f, 2e-6f,
        "RWKV mix");
}

bool test_l2_norm(CudaBackend& backend) {
    constexpr int heads = 3;
    constexpr int head_size = 64;
    constexpr int tokens = 4;
    constexpr int hidden = heads * head_size;
    Tensor input = device_tensor(backend, hidden, tokens);
    fill_values(input.ptr<float>(), input.nelements(), 47, 33.0f);
    std::vector<float> expected(input.nelements());
    constexpr float epsilon = 1e-6f;
    for (int token = 0; token < tokens; ++token)
        for (int head = 0; head < heads; ++head) {
            const int base = token * hidden + head * head_size;
            float sum = 0.0f;
            for (int dimension = 0; dimension < head_size; ++dimension) {
                const float value = input.ptr<float>()[base + dimension];
                sum += value * value;
            }
            const float inverse = 1.0f / (std::sqrt(sum) + epsilon);
            for (int dimension = 0; dimension < head_size; ++dimension)
                expected[base + dimension] =
                    input.ptr<float>()[base + dimension] * inverse;
        }
    Tensor output = device_tensor(backend, hidden, tokens);
    GraphNode node;
    node.op_type = OpType::RWKV_L2_NORM;
    node.params.i32 = {heads, head_size};
    node.params.f32 = {epsilon};
    backend.dispatch(node, {&input}, &output, nullptr);
    backend.end_graph();
    return !backend.dispatch_failed() && close_enough(
        output.ptr<float>(), expected.data(), expected.size(), 2e-6f, 2e-5f,
        "RWKV L2 norm");
}

bool test_post(CudaBackend& backend) {
    constexpr int heads = 3;
    constexpr int head_size = 64;
    constexpr int tokens = 2;
    constexpr int hidden = heads * head_size;
    constexpr int total = hidden * tokens;
    Tensor raw = device_tensor(backend, hidden, tokens);
    Tensor receptance = device_tensor(backend, hidden, tokens);
    Tensor key = device_tensor(backend, hidden, tokens);
    Tensor value = device_tensor(backend, hidden, tokens);
    Tensor gate = device_tensor(backend, hidden, tokens);
    Tensor receptance_key = device_tensor(backend, hidden);
    Tensor weight = device_tensor(backend, hidden);
    Tensor bias = device_tensor(backend, hidden);
    fill_values(raw.ptr<float>(), total, 53, 37.0f);
    fill_values(receptance.ptr<float>(), total, 47, 41.0f);
    fill_values(key.ptr<float>(), total, 43, 39.0f);
    fill_values(value.ptr<float>(), total, 41, 35.0f);
    fill_values(gate.ptr<float>(), total, 37, 31.0f);
    fill_values(receptance_key.ptr<float>(), hidden, 31, 29.0f);
    fill_values(weight.ptr<float>(), hidden, 29, 23.0f);
    fill_values(bias.ptr<float>(), hidden, 23, 27.0f);
    constexpr float epsilon = 64e-5f;
    std::vector<float> expected(total);
    for (int token = 0; token < tokens; ++token)
        for (int head = 0; head < heads; ++head) {
            const int base = token * hidden + head * head_size;
            const int parameter_base = head * head_size;
            float mean = 0.0f;
            float bonus = 0.0f;
            for (int dimension = 0; dimension < head_size; ++dimension) {
                mean += raw.ptr<float>()[base + dimension];
                bonus += receptance.ptr<float>()[base + dimension] *
                    key.ptr<float>()[base + dimension] *
                    receptance_key.ptr<float>()[parameter_base + dimension];
            }
            mean /= head_size;
            float variance = 0.0f;
            for (int dimension = 0; dimension < head_size; ++dimension) {
                const float centered =
                    raw.ptr<float>()[base + dimension] - mean;
                variance += centered * centered;
            }
            const float inverse = 1.0f /
                std::sqrt(variance / head_size + epsilon);
            for (int dimension = 0; dimension < head_size; ++dimension) {
                const int index = base + dimension;
                const int parameter_index = parameter_base + dimension;
                const float normalized =
                    (raw.ptr<float>()[index] - mean) * inverse *
                        weight.ptr<float>()[parameter_index] +
                    bias.ptr<float>()[parameter_index];
                expected[index] =
                    (normalized + bonus * value.ptr<float>()[index]) *
                    gate.ptr<float>()[index];
            }
        }
    Tensor output = device_tensor(backend, hidden, tokens);
    GraphNode node;
    node.op_type = OpType::RWKV_POST;
    node.params.i32 = {heads, head_size};
    node.params.f32 = {epsilon};
    backend.dispatch(
        node, {&raw, &receptance, &key, &value, &receptance_key, &weight,
               &bias, &gate},
        &output, nullptr);
    backend.end_graph();
    return !backend.dispatch_failed() && close_enough(
        output.ptr<float>(), expected.data(), expected.size(), 3e-6f, 4e-5f,
        "RWKV post");
}

bool test_rwkv7(CudaBackend& backend) {
    constexpr int heads = 2;
    constexpr int head_size = 8;
    constexpr int sequence = 4;
    constexpr int real = 3;
    constexpr int hidden = heads * head_size;
    constexpr int total = hidden * sequence;
    constexpr int state_size = heads * head_size * head_size;
    Tensor receptance = device_tensor(backend, hidden, sequence);
    Tensor decay = device_tensor(backend, hidden, sequence);
    Tensor key = device_tensor(backend, hidden, sequence);
    Tensor value = device_tensor(backend, hidden, sequence);
    Tensor a = device_tensor(backend, hidden, sequence);
    Tensor b = device_tensor(backend, hidden, sequence);
    Tensor state = device_tensor(backend, state_size);
    fill_values(receptance.ptr<float>(), total, 37, 43.0f);
    fill_values(key.ptr<float>(), total, 31, 39.0f);
    fill_values(value.ptr<float>(), total, 29, 37.0f);
    fill_values(a.ptr<float>(), total, 23, 41.0f);
    fill_values(b.ptr<float>(), total, 19, 47.0f);
    fill_values(state.ptr<float>(), state_size, 17, 53.0f);
    for (int index = 0; index < total; ++index)
        decay.ptr<float>()[index] =
            0.9f + 0.09f * static_cast<float>(index % 11) / 10.0f;
    const std::vector<float> initial_state(
        state.ptr<float>(), state.ptr<float>() + state_size);
    auto reference = [&](std::vector<float>& expected_state,
                         bool fp16_state) {
        std::vector<float> expected(total, 0.0f);
        for (int head = 0; head < heads; ++head) {
            float* state_head = expected_state.data() +
                head * head_size * head_size;
            for (int token = 0; token < real; ++token) {
                const int base = token * hidden + head * head_size;
                for (int row = 0; row < head_size; ++row) {
                    float state_a = 0.0f;
                    for (int column = 0; column < head_size; ++column)
                        state_a += state_head[row * head_size + column] *
                            a.ptr<float>()[base + column];
                    float result = 0.0f;
                    for (int column = 0; column < head_size; ++column) {
                        float& state_value =
                            state_head[row * head_size + column];
                        state_value =
                            state_value * decay.ptr<float>()[base + column] +
                            value.ptr<float>()[base + row] *
                                key.ptr<float>()[base + column] +
                            state_a * b.ptr<float>()[base + column];
                        result += state_value *
                            receptance.ptr<float>()[base + column];
                    }
                    expected[base + row] = result;
                }
                if (fp16_state)
                    for (int index = 0; index < head_size * head_size;
                         ++index)
                        state_head[index] = static_cast<float>(
                            static_cast<mollm::cpu::fp16_t>(
                                state_head[index]));
            }
        }
        return expected;
    };
    std::vector<float> expected_state = initial_state;
    std::vector<float> expected = reference(expected_state, false);
    Tensor output = device_tensor(backend, hidden, sequence);
    GraphNode node;
    node.op_type = OpType::RWKV7;
    node.params.i32 = {heads, head_size, sequence, real};
    backend.dispatch(
        node, {&receptance, &decay, &key, &value, &a, &b, &state}, &output,
        nullptr);
    backend.end_graph();
    const bool output_matches = close_enough(
        output.ptr<float>(), expected.data(), expected.size(), 3e-6f, 4e-5f,
        "RWKV7 output");
    const bool state_matches_fp32 = close_enough(
        state.ptr<float>(), expected_state.data(), expected_state.size(),
        3e-6f, 4e-5f, "RWKV7 state");
    if (backend.dispatch_failed() || !output_matches ||
        !state_matches_fp32)
        return false;

    Tensor state_fp16 = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, state_size);
    backend.alloc_persistent(state_fp16, state_fp16.nbytes());
    auto* state_fp16_data =
        static_cast<mollm::cpu::fp16_t*>(state_fp16.data);
    expected_state.resize(state_size);
    for (int index = 0; index < state_size; ++index) {
        state_fp16_data[index] = static_cast<mollm::cpu::fp16_t>(
            initial_state[index]);
        expected_state[index] = static_cast<float>(state_fp16_data[index]);
    }
    expected = reference(expected_state, true);
    Tensor output_fp16 = device_tensor(backend, hidden, sequence);
    backend.clear_dispatch_error();
    backend.dispatch(
        node,
        {&receptance, &decay, &key, &value, &a, &b, &state_fp16},
        &output_fp16, nullptr);
    backend.end_graph();
    bool state_matches = true;
    for (int index = 0; index < state_size; ++index)
        state_matches &= static_cast<float>(state_fp16_data[index]) ==
            expected_state[index];
    return !backend.dispatch_failed() && state_matches && close_enough(
        output_fp16.ptr<float>(), expected.data(), expected.size(), 3e-6f,
        4e-5f, "RWKV7 FP16-state output");
}

}  // namespace

int main() {
    CudaBackend backend;
    if (!backend.available()) {
        std::fprintf(stderr, "CUDA device unavailable; skipping\n");
        return 77;
    }
    if (!test_layer_norm(backend) || !test_token_shift(backend) ||
        !test_mix(backend) || !test_l2_norm(backend) ||
        !test_post(backend) || !test_rwkv7(backend))
        return 1;
    std::printf("CUDA RWKV normalization, state and recurrence tests passed\n");
    return 0;
}
