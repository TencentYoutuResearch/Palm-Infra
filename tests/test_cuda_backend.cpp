#include "engine/cuda_backend.h"
#include "engine/engine.h"
#include "kernels/moe.h"
#include "kernels/quant_layouts.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool close_enough(const std::vector<float>& actual,
                  const std::vector<float>& expected, float tolerance) {
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr,
                         "CUDA mismatch at %zu: actual=%g expected=%g\n",
                         i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

void reference(const std::vector<float>& activation,
               const std::vector<float>& weight, std::vector<float>& output,
               int m, int n, int k) {
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner)
                sum += activation[static_cast<size_t>(row) * k + inner] *
                       weight[static_cast<size_t>(col) * k + inner];
            output[static_cast<size_t>(row) * n + col] = sum;
        }
}

bool dispatch_matmul(CudaBackend& backend, Tensor& weight,
                     const std::vector<float>& activation,
                     std::vector<float>& output, int m, int n, int k) {
    Tensor a = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                              k, m, 1, 1,
                              const_cast<float*>(activation.data()));
    Tensor c = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                              n, m, 1, 1, output.data());
    GraphNode node;
    node.op_type = OpType::MATMUL;
    std::vector<const Tensor*> inputs = {&a, &weight};
    backend.clear_dispatch_error();
    backend.dispatch(node, inputs, &c, nullptr);
    return !backend.dispatch_failed();
}

Tensor device_tensor(CudaBackend& backend, int64_t d0, int64_t d1 = 1,
                     int64_t d2 = 1, int64_t d3 = 1) {
    Tensor tensor = Tensor::create(Precision::FP32, MemoryType::NONE,
                                   d0, d1, d2, d3);
    if (!backend.alloc_output(tensor, tensor.nbytes(), nullptr))
        std::fprintf(stderr, "CUDA managed allocation failed\n");
    return tensor;
}

Tensor device_tensor(CudaBackend& backend, Precision precision,
                     int64_t d0, int64_t d1 = 1,
                     int64_t d2 = 1, int64_t d3 = 1) {
    Tensor tensor = Tensor::create(
        precision, MemoryType::NONE, d0, d1, d2, d3);
    if (!backend.alloc_output(tensor, tensor.nbytes(), nullptr))
        std::fprintf(stderr, "CUDA managed allocation failed\n");
    return tensor;
}

bool test_device_resident_ops(CudaBackend& backend, Tensor& weight,
                              const std::vector<float>& activation,
                              const std::vector<float>& expected,
                              int m, int n, int k) {
    Tensor a = device_tensor(backend, k, m);
    Tensor c = device_tensor(backend, n, m);
    if (!a.data || !c.data)
        return false;
    std::memcpy(a.data, activation.data(), a.nbytes());

    GraphNode matmul;
    matmul.op_type = OpType::MATMUL;
    backend.clear_dispatch_error();
    backend.dispatch(matmul, {&a, &weight}, &c, nullptr);
    backend.end_graph();
    std::vector<float> actual(static_cast<size_t>(m) * n);
    std::memcpy(actual.data(), c.data, c.nbytes());
    if (backend.dispatch_failed() ||
        !close_enough(actual, expected, 3e-3f))
        return false;

    constexpr int width = 8;
    constexpr int rows = 2;
    std::vector<float> norm_input(width * rows);
    std::vector<float> norm_weight(width);
    std::vector<float> norm_expected(width * rows);
    for (size_t i = 0; i < norm_input.size(); ++i)
        norm_input[i] = static_cast<float>(static_cast<int>(i) - 7) / 9.0f;
    for (int i = 0; i < width; ++i)
        norm_weight[i] = 0.75f + i * 0.025f;
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < width; ++column) {
            const float value = norm_input[row * width + column];
            sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum / width + 1e-6f);
        for (int column = 0; column < width; ++column)
            norm_expected[row * width + column] =
                norm_input[row * width + column] * scale *
                norm_weight[column];
    }
    Tensor norm_source = device_tensor(backend, width, rows);
    Tensor norm_output = device_tensor(backend, width, rows);
    std::memcpy(norm_source.data, norm_input.data(), norm_source.nbytes());
    Tensor norm_scale = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, width, 1, 1, 1,
        norm_weight.data());
    backend.wrap_weight(norm_scale);
    GraphNode rms_norm;
    rms_norm.op_type = OpType::RMS_NORM;
    backend.dispatch(rms_norm, {&norm_source, &norm_scale},
                     &norm_output, nullptr);
    backend.end_graph();
    actual.resize(norm_expected.size());
    std::memcpy(actual.data(), norm_output.data, norm_output.nbytes());
    if (backend.dispatch_failed() ||
        !close_enough(actual, norm_expected, 2e-5f))
        return false;

    Tensor lhs = device_tensor(backend, width, rows);
    Tensor rhs = device_tensor(backend, width, rows);
    Tensor sum = device_tensor(backend, width, rows);
    Tensor silu = device_tensor(backend, width, rows);
    auto* lhs_data = lhs.ptr<float>();
    auto* rhs_data = rhs.ptr<float>();
    std::vector<float> elementwise_expected(width * rows);
    for (int i = 0; i < width * rows; ++i) {
        lhs_data[i] = (i - 5) / 7.0f;
        rhs_data[i] = (3 - i) / 11.0f;
        const float value = lhs_data[i] + rhs_data[i];
        elementwise_expected[i] = value / (1.0f + std::exp(-value));
    }
    GraphNode add;
    add.op_type = OpType::ADD;
    GraphNode silu_node;
    silu_node.op_type = OpType::SILU;
    backend.dispatch(add, {&lhs, &rhs}, &sum, nullptr);
    backend.dispatch(silu_node, {&sum}, &silu, nullptr);
    backend.end_graph();
    actual.resize(elementwise_expected.size());
    std::memcpy(actual.data(), silu.data, silu.nbytes());
    if (backend.dispatch_failed() ||
        !close_enough(actual, elementwise_expected, 2e-6f))
        return false;

    Tensor view_source = device_tensor(backend, 6, 2);
    for (int i = 0; i < 12; ++i)
        view_source.ptr<float>()[i] = (i - 4) / 5.0f;
    Tensor slice;
    GraphNode slice_node;
    slice_node.op_type = OpType::SLICE;
    slice_node.params.i32 = {0, 2, 2};
    backend.dispatch(slice_node, {&view_source}, &slice, nullptr);
    Tensor view_result = device_tensor(backend, 2, 2);
    backend.dispatch(silu_node, {&slice}, &view_result, nullptr);
    backend.end_graph();
    std::vector<float> view_expected(4);
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 2; ++column) {
            const float value = view_source.ptr<float>()[row * 6 + column + 2];
            view_expected[row * 2 + column] =
                value / (1.0f + std::exp(-value));
        }
    actual.resize(view_expected.size());
    std::memcpy(actual.data(), view_result.data, view_result.nbytes());
    return !backend.dispatch_failed() &&
        close_enough(actual, view_expected, 2e-6f);
}

bool test_rwkv_matrix_ops(CudaBackend& backend) {
    constexpr int m = 2;
    constexpr int n = 5;
    constexpr int k = 7;
    constexpr int batch = 2;
    std::vector<std::vector<float>> activation(
        batch, std::vector<float>(static_cast<size_t>(m) * k));
    std::vector<std::vector<float>> weight(
        batch, std::vector<float>(static_cast<size_t>(n) * k));
    std::vector<std::vector<mollm::cpu::fp16_t>> weight_fp16(
        batch, std::vector<mollm::cpu::fp16_t>(
                   static_cast<size_t>(n) * k));
    std::vector<Tensor> activation_device;
    std::vector<Tensor> weights;
    activation_device.reserve(batch);
    weights.reserve(batch);
    std::vector<float> expected(static_cast<size_t>(batch) * m * n);
    for (int item = 0; item < batch; ++item) {
        for (size_t index = 0; index < activation[item].size(); ++index)
            activation[item][index] =
                (static_cast<int>((index + item * 3) % 11) - 5) / 13.0f;
        for (size_t index = 0; index < weight[item].size(); ++index) {
            weight[item][index] =
                (static_cast<int>((index + item * 5) % 9) - 4) / 17.0f;
            weight_fp16[item][index] =
                static_cast<mollm::cpu::fp16_t>(weight[item][index]);
        }
        activation_device.push_back(device_tensor(backend, k, m));
        std::memcpy(activation_device.back().data, activation[item].data(),
                    activation_device.back().nbytes());
        weights.push_back(Tensor::create(
            Precision::FP16, MemoryType::EXTERNAL, n, k, 1, 1,
            weight_fp16[item].data()));
        backend.wrap_weight(weights.back());
        std::vector<float> item_expected(static_cast<size_t>(m) * n);
        reference(activation[item], weight[item], item_expected, m, n, k);
        std::copy(item_expected.begin(), item_expected.end(),
                  expected.begin() + static_cast<size_t>(item) * m * n);
    }

    Tensor batch_output = device_tensor(backend, n, m, batch);
    GraphNode batch_node;
    batch_node.op_type = OpType::MATMUL_BATCH;
    std::vector<const Tensor*> batch_inputs;
    for (int item = 0; item < batch; ++item) {
        batch_inputs.push_back(&activation_device[item]);
        batch_inputs.push_back(&weights[item]);
    }
    backend.clear_dispatch_error();
    backend.dispatch(batch_node, batch_inputs, &batch_output, nullptr);
    backend.end_graph();
    std::vector<float> actual(expected.size());
    std::memcpy(actual.data(), batch_output.data, batch_output.nbytes());
    if (backend.dispatch_failed() ||
        !close_enough(actual, expected, 3e-3f))
        return false;

    for (int row = 0; row < m; ++row)
        for (int inner = 0; inner < k; ++inner)
            if ((row * k + inner) % 3 != 0)
                activation_device[0].ptr<float>()[row * k + inner] = 0.0f;
    std::memcpy(activation[0].data(), activation_device[0].data,
                activation_device[0].nbytes());
    reference(activation[0], weight[0], expected, m, n, k);
    Tensor sparse_output = device_tensor(backend, n, m);
    GraphNode sparse;
    sparse.op_type = OpType::GEMV_SPARSE_A;
    backend.clear_dispatch_error();
    backend.dispatch(sparse, {&activation_device[0], &weights[0]},
                     &sparse_output, nullptr);
    backend.end_graph();
    actual.resize(static_cast<size_t>(m) * n);
    std::memcpy(actual.data(), sparse_output.data, sparse_output.nbytes());
    if (backend.dispatch_failed() ||
        !close_enough(actual, expected, 3e-3f))
        return false;

    constexpr int group_size = 4;
    constexpr int groups_per_row = 2;
    std::vector<int8_t> quantized(static_cast<size_t>(n) * k);
    std::vector<float> scales(static_cast<size_t>(n) * groups_per_row);
    std::vector<float> dequantized(static_cast<size_t>(n) * k);
    for (int row = 0; row < n; ++row) {
        for (int group = 0; group < groups_per_row; ++group)
            scales[row * groups_per_row + group] =
                0.025f + 0.005f * (row + group);
        for (int inner = 0; inner < k; ++inner) {
            const int8_t value = static_cast<int8_t>(
                (row * 7 + inner * 3) % 31 - 15);
            quantized[static_cast<size_t>(row) * k + inner] = value;
            dequantized[static_cast<size_t>(row) * k + inner] =
                value * scales[row * groups_per_row +
                               inner / group_size];
        }
    }
    Tensor int8 = Tensor::create(Precision::INT8, MemoryType::EXTERNAL,
                                 n, k, 1, 1, quantized.data());
    int8.rowmajor_data = quantized.data();
    int8.scales = scales.data();
    int8.group_size = group_size;
    int8.groups_per_row = groups_per_row;
    int8.num_groups = n * groups_per_row;
    backend.wrap_weight_int4(int8);
    reference(activation[1], dequantized, expected, m, n, k);
    Tensor int8_output = device_tensor(backend, n, m);
    backend.clear_dispatch_error();
    GraphNode matmul;
    matmul.op_type = OpType::MATMUL;
    backend.dispatch(matmul, {&activation_device[1], &int8},
                     &int8_output, nullptr);
    backend.end_graph();
    std::memcpy(actual.data(), int8_output.data, int8_output.nbytes());
    return !backend.dispatch_failed() &&
        close_enough(actual, expected, 3e-3f);
}

bool test_layout_rope_and_sdpa(CudaBackend& backend) {
    backend.clear_dispatch_error();
    Tensor rope_storage = device_tensor(backend, 10, 2);
    for (int i = 0; i < 20; ++i)
        rope_storage.ptr<float>()[i] = (i - 8) / 17.0f;
    Tensor rope_input;
    GraphNode slice;
    slice.op_type = OpType::SLICE;
    slice.params.i32 = {0, 1, 8};
    backend.dispatch(slice, {&rope_storage}, &rope_input, nullptr);
    Tensor cosine = device_tensor(backend, 4, 2);
    Tensor sine = device_tensor(backend, 4, 2);
    for (int i = 0; i < 8; ++i) {
        cosine.ptr<float>()[i] = std::cos((i + 1) * 0.13f);
        sine.ptr<float>()[i] = std::sin((i + 1) * 0.13f);
    }
    Tensor rope_output = device_tensor(backend, 8, 2);
    GraphNode rope;
    rope.op_type = OpType::ROTARY_EMBED;
    rope.params.i32 = {8, 0};
    backend.dispatch(rope, {&rope_input, &cosine, &sine},
                     &rope_output, nullptr);
    backend.end_graph();
    std::vector<float> expected(16);
    for (int row = 0; row < 2; ++row)
        for (int pair = 0; pair < 4; ++pair) {
            const float x0 = rope_storage.ptr<float>()[row * 10 + pair + 1];
            const float x1 =
                rope_storage.ptr<float>()[row * 10 + pair + 5];
            const float c = cosine.ptr<float>()[row * 4 + pair];
            const float s = sine.ptr<float>()[row * 4 + pair];
            expected[row * 8 + pair] = x0 * c - x1 * s;
            expected[row * 8 + pair + 4] = x0 * s + x1 * c;
        }
    std::vector<float> actual(expected.size());
    std::memcpy(actual.data(), rope_output.data, rope_output.nbytes());
    if (backend.dispatch_failed() || !close_enough(actual, expected, 2e-6f))
        return false;

    Tensor contiguous = device_tensor(backend, 8, 2);
    GraphNode contiguous_node;
    contiguous_node.op_type = OpType::CONTIGUOUS;
    backend.dispatch(contiguous_node, {&rope_input}, &contiguous, nullptr);
    backend.end_graph();
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 8; ++column)
            expected[row * 8 + column] =
                rope_storage.ptr<float>()[row * 10 + column + 1];
    std::memcpy(actual.data(), contiguous.data, contiguous.nbytes());
    if (backend.dispatch_failed() || !close_enough(actual, expected, 0.0f))
        return false;

    constexpr int norm_width = 8;
    constexpr int norm_rows = 2;
    Tensor residual = device_tensor(backend, norm_width, norm_rows);
    Tensor update = device_tensor(backend, norm_width, norm_rows);
    Tensor add_norm_output = device_tensor(backend, norm_width, norm_rows);
    std::vector<float> norm_weight(norm_width);
    std::vector<float> residual_expected(norm_width * norm_rows);
    std::vector<float> add_norm_expected(norm_width * norm_rows);
    for (int column = 0; column < norm_width; ++column)
        norm_weight[column] = 0.8f + column * 0.02f;
    for (int i = 0; i < norm_width * norm_rows; ++i) {
        residual.ptr<float>()[i] = (i - 6) / 9.0f;
        update.ptr<float>()[i] = (4 - i) / 13.0f;
        residual_expected[i] =
            residual.ptr<float>()[i] + update.ptr<float>()[i];
    }
    for (int row = 0; row < norm_rows; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < norm_width; ++column) {
            const float value = residual_expected[row * norm_width + column];
            sum += value * value;
        }
        const float inverse = 1.0f /
            std::sqrt(sum / norm_width + 1e-6f);
        for (int column = 0; column < norm_width; ++column)
            add_norm_expected[row * norm_width + column] =
                residual_expected[row * norm_width + column] * inverse *
                norm_weight[column];
    }
    Tensor norm_scale = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, norm_width, 1, 1, 1,
        norm_weight.data());
    backend.wrap_weight(norm_scale);
    GraphNode add_norm;
    add_norm.op_type = OpType::ADD_RMS_NORM;
    backend.dispatch(add_norm, {&residual, &update, &norm_scale},
                     &add_norm_output, nullptr);
    backend.end_graph();
    actual.resize(add_norm_expected.size());
    std::memcpy(actual.data(), add_norm_output.data,
                add_norm_output.nbytes());
    std::vector<float> residual_actual(residual_expected.size());
    std::memcpy(residual_actual.data(), residual.data, residual.nbytes());
    if (backend.dispatch_failed() ||
        !close_enough(actual, add_norm_expected, 2e-5f) ||
        !close_enough(residual_actual, residual_expected, 2e-6f))
        return false;

    constexpr int rope_sequence = 2;
    constexpr int rope_heads = 3;
    Tensor flat_norm_input =
        device_tensor(backend, norm_width, rope_sequence * rope_heads);
    for (int64_t i = 0; i < flat_norm_input.nelements(); ++i)
        flat_norm_input.ptr<float>()[i] =
            (static_cast<int>(i % 17) - 8) / 15.0f;
    Tensor fused_rope_output =
        device_tensor(backend, norm_width, rope_sequence, rope_heads);
    GraphNode fused_rope;
    fused_rope.op_type = OpType::RMS_NORM_ROPE;
    fused_rope.params.i32 = {norm_width, 1};
    backend.dispatch(fused_rope,
                     {&flat_norm_input, &norm_scale, &cosine, &sine},
                     &fused_rope_output, nullptr);
    backend.end_graph();
    expected.assign(static_cast<size_t>(norm_width) * rope_sequence *
                        rope_heads,
                    0.0f);
    for (int head = 0; head < rope_heads; ++head)
        for (int position = 0; position < rope_sequence; ++position) {
            const int row = head * rope_sequence + position;
            float sum = 0.0f;
            for (int dimension = 0; dimension < norm_width; ++dimension) {
                const float value =
                    flat_norm_input.ptr<float>()[row * norm_width + dimension];
                sum += value * value;
            }
            const float inverse = 1.0f /
                std::sqrt(sum / norm_width + 1e-6f);
            for (int pair = 0; pair < norm_width / 2; ++pair) {
                const float x0 = flat_norm_input.ptr<float>()[
                    row * norm_width + pair * 2] * inverse *
                    norm_weight[pair * 2];
                const float x1 = flat_norm_input.ptr<float>()[
                    row * norm_width + pair * 2 + 1] * inverse *
                    norm_weight[pair * 2 + 1];
                const float c = cosine.ptr<float>()[position * 4 + pair];
                const float s = sine.ptr<float>()[position * 4 + pair];
                expected[row * norm_width + pair * 2] = x0 * c - x1 * s;
                expected[row * norm_width + pair * 2 + 1] = x0 * s + x1 * c;
            }
        }
    actual.resize(expected.size());
    std::memcpy(actual.data(), fused_rope_output.data,
                fused_rope_output.nbytes());
    if (backend.dispatch_failed() || !close_enough(actual, expected, 3e-5f))
        return false;

    constexpr int query_heads = 2;
    constexpr int key_heads = 1;
    Tensor qk_query = device_tensor(
        backend, norm_width, rope_sequence * query_heads);
    Tensor qk_key = device_tensor(
        backend, norm_width, rope_sequence * key_heads);
    for (int64_t i = 0; i < qk_query.nelements(); ++i)
        qk_query.ptr<float>()[i] =
            (static_cast<int>(i % 19) - 9) / 17.0f;
    for (int64_t i = 0; i < qk_key.nelements(); ++i)
        qk_key.ptr<float>()[i] =
            (static_cast<int>(i % 13) - 6) / 11.0f;
    std::vector<float> key_norm_weight(norm_width);
    for (int dimension = 0; dimension < norm_width; ++dimension)
        key_norm_weight[dimension] = 0.65f + dimension * 0.015f;
    Tensor key_norm_scale = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, norm_width, 1, 1, 1,
        key_norm_weight.data());
    backend.wrap_weight(key_norm_scale);
    Tensor qk_output = device_tensor(
        backend, norm_width, rope_sequence, query_heads + key_heads);
    GraphNode qk_fused;
    qk_fused.op_type = OpType::QK_RMS_NORM_ROPE;
    qk_fused.params.i32 = {norm_width, 1, query_heads};
    qk_fused.params.f32 = {1e-6f};
    backend.dispatch(
        qk_fused,
        {&qk_query, &qk_key, &norm_scale, &key_norm_scale, &cosine, &sine},
        &qk_output, nullptr);
    backend.end_graph();
    expected.assign(
        static_cast<size_t>(norm_width) * rope_sequence *
            (query_heads + key_heads),
        0.0f);
    for (int head = 0; head < query_heads + key_heads; ++head) {
        const bool is_query = head < query_heads;
        const float* source = is_query
            ? qk_query.ptr<float>() : qk_key.ptr<float>();
        const float* scale_weight = is_query
            ? norm_weight.data() : key_norm_weight.data();
        const int local_head = is_query ? head : head - query_heads;
        for (int position = 0; position < rope_sequence; ++position) {
            const float* row = source +
                static_cast<size_t>(local_head * rope_sequence + position) *
                    norm_width;
            float sum = 0.0f;
            for (int dimension = 0; dimension < norm_width; ++dimension)
                sum += row[dimension] * row[dimension];
            const float inverse = 1.0f /
                std::sqrt(sum / norm_width + 1e-6f);
            for (int pair = 0; pair < norm_width / 2; ++pair) {
                const float x0 = row[pair * 2] * inverse *
                    scale_weight[pair * 2];
                const float x1 = row[pair * 2 + 1] * inverse *
                    scale_weight[pair * 2 + 1];
                const float c = cosine.ptr<float>()[position * 4 + pair];
                const float s = sine.ptr<float>()[position * 4 + pair];
                const size_t base =
                    static_cast<size_t>(head * rope_sequence + position) *
                    norm_width;
                expected[base + pair * 2] = x0 * c - x1 * s;
                expected[base + pair * 2 + 1] = x0 * s + x1 * c;
            }
        }
    }
    actual.resize(expected.size());
    std::memcpy(actual.data(), qk_output.data, qk_output.nbytes());
    if (backend.dispatch_failed() || !close_enough(actual, expected, 3e-5f))
        return false;

    constexpr int heads = 4;
    constexpr int kv_heads = 2;
    constexpr int key_dim = 4;
    constexpr int value_dim = 3;
    constexpr int query_length = 2;
    constexpr int current_length = 2;
    constexpr int past_length = 1;
    constexpr int capacity = 4;
    Tensor query = device_tensor(backend, key_dim, query_length, heads);
    Tensor key = device_tensor(backend, key_dim, current_length, kv_heads);
    Tensor value =
        device_tensor(backend, value_dim, current_length, kv_heads);
    for (int64_t i = 0; i < query.nelements(); ++i)
        query.ptr<float>()[i] = (static_cast<int>(i % 13) - 6) / 11.0f;
    for (int64_t i = 0; i < key.nelements(); ++i)
        key.ptr<float>()[i] = (static_cast<int>(i % 9) - 4) / 7.0f;
    for (int64_t i = 0; i < value.nelements(); ++i)
        value.ptr<float>()[i] = (static_cast<int>(i % 7) - 3) / 5.0f;

    const size_t key_cache_bytes = CacheMetadata::SIZE +
        static_cast<size_t>(kv_heads) * capacity * key_dim * sizeof(float);
    const size_t value_cache_bytes = CacheMetadata::SIZE +
        static_cast<size_t>(kv_heads) * capacity * value_dim * sizeof(float);
    Tensor key_cache = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, key_cache_bytes / sizeof(float));
    Tensor value_cache = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        value_cache_bytes / sizeof(float));
    backend.alloc_persistent(key_cache, key_cache_bytes);
    backend.alloc_persistent(value_cache, value_cache_bytes);
    auto* key_metadata = cache_meta(key_cache.data);
    key_metadata->current_seq_len = past_length;
    key_metadata->max_seq_len = capacity;
    key_metadata->num_kv_heads = kv_heads;
    key_metadata->head_dim = key_dim;
    auto* value_metadata = cache_meta(value_cache.data);
    value_metadata->current_seq_len = past_length;
    value_metadata->max_seq_len = capacity;
    value_metadata->num_kv_heads = kv_heads;
    value_metadata->v_head_dim = value_dim;
    auto* cached_key = static_cast<float*>(cache_data(key_cache.data));
    auto* cached_value = static_cast<float*>(cache_data(value_cache.data));
    std::vector<float> initial_key(
        static_cast<size_t>(kv_heads) * capacity * key_dim, 0.0f);
    std::vector<float> initial_value(
        static_cast<size_t>(kv_heads) * capacity * value_dim, 0.0f);
    for (int head = 0; head < kv_heads; ++head) {
        for (int dimension = 0; dimension < key_dim; ++dimension)
            initial_key[(head * capacity) * key_dim + dimension] =
                (head * key_dim + dimension - 3) / 8.0f;
        for (int dimension = 0; dimension < value_dim; ++dimension)
            initial_value[(head * capacity) * value_dim + dimension] =
                (head * value_dim + dimension - 2) / 6.0f;
    }
    std::memcpy(cached_key, initial_key.data(),
                initial_key.size() * sizeof(float));
    std::memcpy(cached_value, initial_value.data(),
                initial_value.size() * sizeof(float));

    Tensor attention_output =
        device_tensor(backend, value_dim, query_length, heads);
    GraphNode sdpa;
    sdpa.op_type = OpType::SDPA;
    sdpa.params.i32 = {2, 1, heads, kv_heads, key_dim, value_dim};
    const float scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
    sdpa.params.f32 = {scale};
    std::vector<const Tensor*> attention_inputs = {
        &query, &key, &value, nullptr, &key_cache, &value_cache};
    backend.dispatch(sdpa, attention_inputs, &attention_output, nullptr);
    backend.end_graph();

    expected.assign(static_cast<size_t>(heads) * query_length * value_dim,
                    0.0f);
    const int heads_per_group = heads / kv_heads;
    const int total_length = past_length + current_length;
    for (int head = 0; head < heads; ++head) {
        const int key_head = head / heads_per_group;
        for (int position = 0; position < query_length; ++position) {
            float scores[total_length];
            float maximum = -INFINITY;
            for (int key_position = 0; key_position < total_length;
                 ++key_position) {
                float score = 0.0f;
                for (int dimension = 0; dimension < key_dim; ++dimension) {
                    const float key_value = key_position < past_length
                        ? initial_key[(key_head * capacity + key_position) *
                                      key_dim + dimension]
                        : key.ptr<float>()[
                              (key_head * current_length + key_position -
                               past_length) * key_dim + dimension];
                    score += query.ptr<float>()[
                                 (head * query_length + position) * key_dim +
                                 dimension] * key_value;
                }
                score *= scale;
                if (key_position > past_length + position)
                    score = -INFINITY;
                scores[key_position] = score;
                maximum = std::max(maximum, score);
            }
            float sum = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            for (int key_position = 0; key_position < total_length;
                 ++key_position) {
                for (int dimension = 0; dimension < value_dim; ++dimension) {
                    const float current = key_position < past_length
                        ? initial_value[
                              (key_head * capacity + key_position) *
                                  value_dim + dimension]
                        : value.ptr<float>()[
                              (key_head * current_length + key_position -
                               past_length) * value_dim + dimension];
                    expected[(head * query_length + position) * value_dim +
                             dimension] += scores[key_position] / sum * current;
                }
            }
        }
    }
    actual.resize(expected.size());
    std::memcpy(actual.data(), attention_output.data,
                attention_output.nbytes());
    if (backend.dispatch_failed() || !close_enough(actual, expected, 3e-5f))
        return false;
    for (int head = 0; head < kv_heads; ++head)
        for (int position = 0; position < current_length; ++position)
            for (int dimension = 0; dimension < key_dim; ++dimension)
                if (std::fabs(cached_key[
                        (head * capacity + past_length + position) * key_dim +
                        dimension] - key.ptr<float>()[
                            (head * current_length + position) * key_dim +
                            dimension]) > 1e-6f)
                    return false;
    return true;
}

bool test_strided_layout_ops(CudaBackend& backend) {
    backend.clear_dispatch_error();

    constexpr int norm_parent_width = 10;
    constexpr int norm_width = 8;
    constexpr int rows = 3;
    Tensor norm_parent = device_tensor(backend, norm_parent_width, rows);
    for (int i = 0; i < norm_parent_width * rows; ++i)
        norm_parent.ptr<float>()[i] = (i - 11) / 13.0f;
    Tensor norm_input;
    GraphNode slice;
    slice.op_type = OpType::SLICE;
    slice.params.i32 = {0, 1, norm_width};
    backend.dispatch(slice, {&norm_parent}, &norm_input, nullptr);
    std::vector<float> norm_weight(norm_width);
    for (int i = 0; i < norm_width; ++i)
        norm_weight[i] = 0.7f + i * 0.03f;
    Tensor norm_scale = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, norm_width, 1, 1, 1,
        norm_weight.data());
    backend.wrap_weight(norm_scale);
    Tensor norm_output = device_tensor(backend, norm_width, rows);
    GraphNode rms;
    rms.op_type = OpType::RMS_NORM;
    backend.dispatch(rms, {&norm_input, &norm_scale}, &norm_output, nullptr);
    backend.end_graph();
    std::vector<float> expected(norm_width * rows);
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < norm_width; ++column) {
            const float value = norm_parent.ptr<float>()[
                row * norm_parent_width + column + 1];
            sum += value * value;
        }
        const float inverse = 1.0f /
            std::sqrt(sum / norm_width + 1e-6f);
        for (int column = 0; column < norm_width; ++column)
            expected[row * norm_width + column] =
                norm_parent.ptr<float>()[
                    row * norm_parent_width + column + 1] *
                inverse * norm_weight[column];
    }
    std::vector<float> actual(expected.size());
    std::memcpy(actual.data(), norm_output.data, norm_output.nbytes());
    if (backend.dispatch_failed() ||
        !close_enough(actual, expected, 2e-5f))
        return false;

    constexpr int merged_width = 8;
    constexpr int half = merged_width / 2;
    Tensor merged = device_tensor(backend, merged_width, rows);
    for (int i = 0; i < merged_width * rows; ++i)
        merged.ptr<float>()[i] = (i - 9) / 11.0f;
    Tensor gate;
    Tensor up;
    slice.params.i32 = {0, 0, half};
    backend.dispatch(slice, {&merged}, &gate, nullptr);
    slice.params.i32 = {0, half, half};
    backend.dispatch(slice, {&merged}, &up, nullptr);
    Tensor product = device_tensor(backend, half, rows);
    GraphNode multiply;
    multiply.op_type = OpType::MUL;
    backend.dispatch(multiply, {&gate, &up}, &product, nullptr);
    backend.end_graph();
    expected.resize(half * rows);
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < half; ++column)
            expected[row * half + column] =
                merged.ptr<float>()[row * merged_width + column] *
                merged.ptr<float>()[row * merged_width + half + column];
    actual.resize(expected.size());
    std::memcpy(actual.data(), product.data, product.nbytes());
    if (backend.dispatch_failed() ||
        !close_enough(actual, expected, 0.0f))
        return false;

    Tensor factors = device_tensor(backend, half);
    for (int column = 0; column < half; ++column)
        factors.ptr<float>()[column] = 0.5f + column * 0.25f;
    Tensor broadcast_product = device_tensor(backend, half, rows);
    backend.dispatch(
        multiply, {&gate, &factors}, &broadcast_product, nullptr);
    backend.end_graph();
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < half; ++column)
            expected[row * half + column] =
                merged.ptr<float>()[row * merged_width + column] *
                factors.ptr<float>()[column];
    std::memcpy(actual.data(), broadcast_product.data,
                broadcast_product.nbytes());
    if (backend.dispatch_failed() ||
        !close_enough(actual, expected, 0.0f))
        return false;

    constexpr int tile_width = 3;
    constexpr int tile_rows = 2;
    constexpr int tile_heads = 4;
    Tensor tile_input = device_tensor(backend, tile_width, tile_rows, 1);
    for (int i = 0; i < tile_width * tile_rows; ++i)
        tile_input.ptr<float>()[i] = (i - 2) / 7.0f;
    Tensor tiled = device_tensor(
        backend, tile_width, tile_rows, tile_heads);
    GraphNode tile;
    tile.op_type = OpType::TILE;
    tile.params.i32 = {1, 1, tile_heads, 1};
    backend.dispatch(tile, {&tile_input}, &tiled, nullptr);
    backend.end_graph();
    expected.resize(tile_width * tile_rows * tile_heads);
    for (int head = 0; head < tile_heads; ++head)
        for (int row = 0; row < tile_rows; ++row)
            for (int column = 0; column < tile_width; ++column)
                expected[(head * tile_rows + row) * tile_width + column] =
                    tile_input.ptr<float>()[row * tile_width + column];
    actual.resize(expected.size());
    std::memcpy(actual.data(), tiled.data, tiled.nbytes());
    if (backend.dispatch_failed() ||
        !close_enough(actual, expected, 0.0f))
        return false;

    constexpr int concat_width = 6;
    constexpr int concat_heads = 2;
    constexpr int concat_rows = 3;
    Tensor concat_storage = device_tensor(
        backend, concat_width, concat_heads, concat_rows);
    for (int64_t i = 0; i < concat_storage.nelements(); ++i)
        concat_storage.ptr<float>()[i] =
            (static_cast<int>(i) - 15) / 17.0f;
    Tensor permuted;
    GraphNode permute;
    permute.op_type = OpType::PERMUTE;
    permute.params.i32 = {0, 2, 1, 3};
    backend.dispatch(permute, {&concat_storage}, &permuted, nullptr);
    Tensor first;
    Tensor second;
    slice.params.i32 = {0, 0, 2};
    backend.dispatch(slice, {&permuted}, &first, nullptr);
    slice.params.i32 = {0, 2, 4};
    backend.dispatch(slice, {&permuted}, &second, nullptr);
    Tensor concatenated = device_tensor(
        backend, concat_width, concat_rows, concat_heads);
    GraphNode concat;
    concat.op_type = OpType::CONCAT;
    concat.params.i32 = {0};
    backend.dispatch(concat, {&first, &second}, &concatenated, nullptr);
    backend.end_graph();
    expected.resize(concatenated.nelements());
    for (int head = 0; head < concat_heads; ++head)
        for (int row = 0; row < concat_rows; ++row)
            for (int column = 0; column < concat_width; ++column)
                expected[(head * concat_rows + row) * concat_width +
                         column] = concat_storage.ptr<float>()[
                    (row * concat_heads + head) * concat_width + column];
    actual.resize(expected.size());
    std::memcpy(actual.data(), concatenated.data, concatenated.nbytes());
    return !backend.dispatch_failed() &&
        close_enough(actual, expected, 0.0f);
}

bool test_moe_fp16_shared(CudaBackend& backend) {
    constexpr int hidden_size = 8;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int intermediate_size = 5;
    constexpr int shared_intermediate_size = 6;
    constexpr int tokens = 3;
    auto fill = [](size_t index, int offset) {
        return static_cast<float>(
            static_cast<int>((index + offset) % 19) - 9) / 23.0f;
    };

    std::vector<float> hidden(hidden_size * tokens);
    std::vector<float> router_f32(num_experts * hidden_size);
    std::vector<float> gate_up_f32(
        num_experts * 2 * intermediate_size * hidden_size);
    std::vector<float> down_f32(
        num_experts * hidden_size * intermediate_size);
    std::vector<float> shared_gate_f32(
        shared_intermediate_size * hidden_size);
    std::vector<float> shared_up_f32(
        shared_intermediate_size * hidden_size);
    std::vector<float> shared_down_f32(
        hidden_size * shared_intermediate_size);
    std::vector<float> shared_scale_f32(hidden_size);
    std::vector<float> router_bias(num_experts);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = fill(i, 1);
    for (size_t i = 0; i < router_f32.size(); ++i)
        router_f32[i] = fill(i, 3);
    for (size_t i = 0; i < gate_up_f32.size(); ++i)
        gate_up_f32[i] = fill(i, 5);
    for (size_t i = 0; i < down_f32.size(); ++i)
        down_f32[i] = fill(i, 7);
    for (size_t i = 0; i < shared_gate_f32.size(); ++i)
        shared_gate_f32[i] = fill(i, 9);
    for (size_t i = 0; i < shared_up_f32.size(); ++i)
        shared_up_f32[i] = fill(i, 11);
    for (size_t i = 0; i < shared_down_f32.size(); ++i)
        shared_down_f32[i] = fill(i, 13);
    for (size_t i = 0; i < shared_scale_f32.size(); ++i)
        shared_scale_f32[i] = fill(i, 15);
    router_bias = {0.03f, -0.07f, 0.11f, -0.02f};

    auto to_fp16 = [](const std::vector<float>& source) {
        std::vector<mollm::cpu::fp16_t> result(source.size());
        for (size_t i = 0; i < source.size(); ++i)
            result[i] = static_cast<mollm::cpu::fp16_t>(source[i]);
        return result;
    };
    auto router_data = to_fp16(router_f32);
    auto gate_up_data = to_fp16(gate_up_f32);
    auto down_data = to_fp16(down_f32);
    auto shared_gate_data = to_fp16(shared_gate_f32);
    auto shared_up_data = to_fp16(shared_up_f32);
    auto shared_down_data = to_fp16(shared_down_f32);
    auto shared_scale_data = to_fp16(shared_scale_f32);

    Tensor hidden_host = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, hidden_size, tokens, 1, 1,
        hidden.data());
    Tensor router = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, num_experts, hidden_size,
        1, 1, router_data.data());
    Tensor gate_up = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL,
        num_experts * 2 * intermediate_size, hidden_size, 1, 1,
        gate_up_data.data());
    Tensor down = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL,
        num_experts * hidden_size, intermediate_size, 1, 1,
        down_data.data());
    Tensor shared_gate = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, shared_intermediate_size,
        hidden_size, 1, 1, shared_gate_data.data());
    Tensor shared_up = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, shared_intermediate_size,
        hidden_size, 1, 1, shared_up_data.data());
    Tensor shared_down = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, hidden_size,
        shared_intermediate_size, 1, 1, shared_down_data.data());
    Tensor shared_scale = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, 1, hidden_size, 1, 1,
        shared_scale_data.data());
    Tensor bias = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, num_experts, 1, 1, 1,
        router_bias.data());
    std::vector<const Tensor*> inputs = {
        &hidden_host, &router, &gate_up, &down, &shared_gate, &shared_up,
        &shared_down, &shared_scale, &bias,
    };
    std::vector<float> expected(hidden_size * tokens);
    Tensor expected_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, hidden_size, tokens, 1, 1,
        expected.data());
    if (!kernel_qwen3_moe(
            inputs, expected_tensor, nullptr, hidden_size, num_experts,
            top_k, intermediate_size, shared_intermediate_size, 1, true,
            true, 2, 1, 0.75f, true, 8))
        return false;

    backend.wrap_weight(router);
    backend.wrap_weight(gate_up);
    backend.wrap_weight(down);
    backend.wrap_weight(shared_gate);
    backend.wrap_weight(shared_up);
    backend.wrap_weight(shared_down);
    backend.wrap_weight(shared_scale);
    backend.wrap_weight(bias);
    Tensor hidden_device = device_tensor(backend, hidden_size, tokens);
    Tensor output_device = device_tensor(backend, hidden_size, tokens);
    std::memcpy(hidden_device.data, hidden.data(), hidden_device.nbytes());
    inputs = {
        &hidden_device, &router, &gate_up, &down, &shared_gate, &shared_up,
        &shared_down, &shared_scale, &bias,
    };
    GraphNode node;
    node.op_type = OpType::MOE;
    node.params.i32 = {
        hidden_size, num_experts, top_k, intermediate_size,
        shared_intermediate_size, 1, 1, 1, 2, 1, 1, 8, -1, -1,
    };
    node.params.f32 = {0.75f, 0.0f};
    // A host fallback must fail after this point; success proves the CUDA path
    // handled every routed and shared-expert stage.
    router.data = gate_up.data = down.data = nullptr;
    shared_gate.data = shared_up.data = shared_down.data = nullptr;
    shared_scale.data = bias.data = nullptr;
    backend.clear_dispatch_error();
    backend.dispatch(node, inputs, &output_device, nullptr);
    backend.end_graph();
    std::vector<float> actual(expected.size());
    std::memcpy(actual.data(), output_device.data, output_device.nbytes());
    return !backend.dispatch_failed() &&
        close_enough(actual, expected, 2.5e-2f);
}

bool test_moe_hash_routing(CudaBackend& backend) {
    constexpr int hidden_size = 8;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int intermediate_size = 5;
    constexpr int shared_intermediate_size = 6;
    constexpr int tokens = 3;
    constexpr int vocab_size = 4;
    auto fill = [](size_t index, int offset) {
        return static_cast<float>(
            static_cast<int>((index * 7 + offset) % 23) - 11) / 29.0f;
    };
    auto to_fp16 = [](const std::vector<float>& source) {
        std::vector<mollm::cpu::fp16_t> result(source.size());
        for (size_t i = 0; i < source.size(); ++i)
            result[i] = static_cast<mollm::cpu::fp16_t>(source[i]);
        return result;
    };

    std::vector<float> hidden(hidden_size * tokens);
    std::vector<float> router_f32(num_experts * hidden_size);
    std::vector<float> gate_up_f32(
        num_experts * 2 * intermediate_size * hidden_size);
    std::vector<float> down_f32(
        num_experts * hidden_size * intermediate_size);
    std::vector<float> shared_gate_f32(
        shared_intermediate_size * hidden_size);
    std::vector<float> shared_up_f32(
        shared_intermediate_size * hidden_size);
    std::vector<float> shared_down_f32(
        hidden_size * shared_intermediate_size);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = fill(i, 1);
    for (size_t i = 0; i < router_f32.size(); ++i)
        router_f32[i] = fill(i, 3);
    for (size_t i = 0; i < gate_up_f32.size(); ++i)
        gate_up_f32[i] = fill(i, 5);
    for (size_t i = 0; i < down_f32.size(); ++i)
        down_f32[i] = fill(i, 7);
    for (size_t i = 0; i < shared_gate_f32.size(); ++i)
        shared_gate_f32[i] = fill(i, 9);
    for (size_t i = 0; i < shared_up_f32.size(); ++i)
        shared_up_f32[i] = fill(i, 11);
    for (size_t i = 0; i < shared_down_f32.size(); ++i)
        shared_down_f32[i] = fill(i, 13);

    auto router_data = to_fp16(router_f32);
    auto gate_up_data = to_fp16(gate_up_f32);
    auto down_data = to_fp16(down_f32);
    auto shared_gate_data = to_fp16(shared_gate_f32);
    auto shared_up_data = to_fp16(shared_up_f32);
    auto shared_down_data = to_fp16(shared_down_f32);
    std::vector<int32_t> token_ids_data = {2, 0, 3};
    // Tensor storage is [vocab, top_k] in flat row-major terms: each token id
    // owns top_k consecutive expert ids. Descending pairs exercise the CUDA
    // accumulation-order normalization independently of lookup order.
    std::vector<int32_t> hash_data = {
        3, 1,
        2, 0,
        1, 3,
        3, 0,
    };

    Tensor hidden_host = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, hidden_size, tokens, 1, 1,
        hidden.data());
    Tensor router = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, num_experts, hidden_size,
        1, 1, router_data.data());
    Tensor gate_up = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL,
        num_experts * 2 * intermediate_size, hidden_size, 1, 1,
        gate_up_data.data());
    Tensor down = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL,
        num_experts * hidden_size, intermediate_size, 1, 1,
        down_data.data());
    Tensor shared_gate = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, shared_intermediate_size,
        hidden_size, 1, 1, shared_gate_data.data());
    Tensor shared_up = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, shared_intermediate_size,
        hidden_size, 1, 1, shared_up_data.data());
    Tensor shared_down = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, hidden_size,
        shared_intermediate_size, 1, 1, shared_down_data.data());
    Tensor token_ids = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, tokens, 1, 1, 1,
        token_ids_data.data());
    Tensor hash_table = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, top_k, vocab_size, 1, 1,
        hash_data.data());
    std::vector<const Tensor*> inputs = {
        &hidden_host, &router, &gate_up, &down, &shared_gate, &shared_up,
        &shared_down, &token_ids, &hash_table,
    };
    std::vector<float> expected(hidden_size * tokens);
    Tensor expected_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, hidden_size, tokens, 1, 1,
        expected.data());
    if (!kernel_qwen3_moe(
            inputs, expected_tensor, nullptr, hidden_size, num_experts,
            top_k, intermediate_size, shared_intermediate_size, 2, true,
            true, 1, 1, 0.7f, false, -1, 7, 8, 0.5f))
        return false;

    backend.wrap_weight(router);
    backend.wrap_weight(gate_up);
    backend.wrap_weight(down);
    backend.wrap_weight(shared_gate);
    backend.wrap_weight(shared_up);
    backend.wrap_weight(shared_down);
    backend.wrap_weight(hash_table);
    Tensor hidden_device = device_tensor(backend, hidden_size, tokens);
    Tensor token_ids_device = device_tensor(
        backend, Precision::INT32, tokens);
    Tensor output_device = device_tensor(backend, hidden_size, tokens);
    std::memcpy(hidden_device.data, hidden.data(), hidden_device.nbytes());
    std::memcpy(token_ids_device.data, token_ids_data.data(),
                token_ids_device.nbytes());
    inputs = {
        &hidden_device, &router, &gate_up, &down, &shared_gate, &shared_up,
        &shared_down, &token_ids_device, &hash_table,
    };
    GraphNode node;
    node.op_type = OpType::MOE;
    node.params.i32 = {
        hidden_size, num_experts, top_k, intermediate_size,
        shared_intermediate_size, 2, 1, 1, 1, 1, 0, -1, 7, 8,
    };
    node.params.f32 = {0.7f, 0.5f};
    // Remove every host-side weight, including the INT32 lookup table. A
    // successful dispatch therefore proves that hash routing stayed on CUDA.
    router.data = gate_up.data = down.data = nullptr;
    shared_gate.data = shared_up.data = shared_down.data = nullptr;
    hash_table.data = nullptr;
    backend.clear_dispatch_error();
    backend.dispatch(node, inputs, &output_device, nullptr);
    backend.end_graph();
    std::vector<float> actual(expected.size());
    std::memcpy(actual.data(), output_device.data, output_device.nbytes());
    return !backend.dispatch_failed() &&
        close_enough(actual, expected, 2.5e-2f);
}

template <int group_size, typename Block>
bool test_moe_packed(CudaBackend& backend) {
    static_assert(group_size == 32 || group_size == 128);
    constexpr int hidden_size = group_size;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int intermediate_size = group_size;
    constexpr int tokens = 2;
    const int gate_up_rows = num_experts * 2 * intermediate_size;
    const int down_rows = num_experts * hidden_size;
    auto make_blocks = [](int rows, int width, int seed) {
        std::vector<Block> blocks(
            static_cast<size_t>(rows / 8) * (width / group_size));
        const int groups = width / group_size;
        for (int row_block = 0; row_block < rows / 8; ++row_block) {
            for (int group = 0; group < groups; ++group) {
                auto& block = blocks[
                    static_cast<size_t>(row_block) * groups + group];
                for (int lane = 0; lane < 8; ++lane) {
                    block.scales[lane] = 0.0125f * (1 + (lane % 3));
                    for (int inner = 0; inner < group_size; inner += 2) {
                        const int row = row_block * 8 + lane;
                        const int low = (row + group + inner + seed) % 15 - 7;
                        const int high =
                            (row + group + inner + seed + 3) % 15 - 7;
                        const uint8_t packed = static_cast<uint8_t>(
                            (low & 0xf) | ((high & 0xf) << 4));
                        if constexpr (group_size == 32) {
                            block.q[lane][inner / 2] = packed;
                        } else {
                            const int subgroup = inner / 32;
                            const int subgroup_inner = inner % 32;
                            block.q[subgroup][lane][subgroup_inner / 2] =
                                packed;
                        }
                    }
                }
            }
        }
        return blocks;
    };
    auto gate_up_blocks = make_blocks(gate_up_rows, hidden_size, 2);
    auto down_blocks = make_blocks(down_rows, intermediate_size, 7);
    std::vector<float> hidden(hidden_size * tokens);
    std::vector<float> router_f32(num_experts * hidden_size);
    for (size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 31.0f;
    for (size_t i = 0; i < router_f32.size(); ++i)
        router_f32[i] =
            static_cast<float>(static_cast<int>(i % 13) - 6) / 29.0f;
    std::vector<mollm::cpu::fp16_t> router_data(router_f32.size());
    for (size_t i = 0; i < router_f32.size(); ++i)
        router_data[i] =
            static_cast<mollm::cpu::fp16_t>(router_f32[i]);

    Tensor hidden_host = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, hidden_size, tokens, 1, 1,
        hidden.data());
    Tensor router = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, num_experts, hidden_size,
        1, 1, router_data.data());
    Tensor gate_up = Tensor::create(
        Precision::INT4, MemoryType::EXTERNAL, gate_up_rows, hidden_size,
        1, 1, gate_up_blocks.data());
    if constexpr (group_size == 32) {
        gate_up.q4_g32_data = gate_up_blocks.data();
        gate_up.is_q4_g32_packed = true;
    } else {
        gate_up.q4_g128_data = gate_up_blocks.data();
        gate_up.is_q4_g128_packed = true;
    }
    gate_up.group_size = group_size;
    gate_up.groups_per_row = 1;
    Tensor down = Tensor::create(
        Precision::INT4, MemoryType::EXTERNAL, down_rows,
        intermediate_size, 1, 1, down_blocks.data());
    if constexpr (group_size == 32) {
        down.q4_g32_data = down_blocks.data();
        down.is_q4_g32_packed = true;
    } else {
        down.q4_g128_data = down_blocks.data();
        down.is_q4_g128_packed = true;
    }
    down.group_size = group_size;
    down.groups_per_row = 1;
    std::vector<const Tensor*> inputs = {
        &hidden_host, &router, &gate_up, &down,
    };
    std::vector<float> expected(hidden_size * tokens);
    Tensor expected_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, hidden_size, tokens, 1, 1,
        expected.data());
    if (!kernel_qwen3_moe(
            inputs, expected_tensor, nullptr, hidden_size, num_experts,
            top_k, intermediate_size, 0, 0, true, false, 2, 1))
        return false;
    if (std::none_of(expected.begin(), expected.end(),
                     [](float value) { return std::fabs(value) > 1e-6f; }))
        return false;

    backend.wrap_weight(router);
    backend.wrap_weight_int4(gate_up, true);
    backend.wrap_weight_int4(down, true);
    Tensor hidden_device = device_tensor(backend, hidden_size, tokens);
    Tensor output_device = device_tensor(backend, hidden_size, tokens);
    std::memcpy(hidden_device.data, hidden.data(), hidden_device.nbytes());
    inputs = {&hidden_device, &router, &gate_up, &down};
    GraphNode node;
    node.op_type = OpType::MOE;
    node.params.i32 = {
        hidden_size, num_experts, top_k, intermediate_size, 0,
        0, 1, 0, 2, 1, 0, -1, -1, -1,
    };
    node.params.f32 = {1.0f, 0.0f};
    router.data = gate_up.data = down.data = nullptr;
    backend.clear_dispatch_error();
    backend.dispatch(node, inputs, &output_device, nullptr);
    backend.end_graph();
    std::vector<float> actual(expected.size());
    std::memcpy(actual.data(), output_device.data, output_device.nbytes());
    return !backend.dispatch_failed() &&
        close_enough(actual, expected, 3.0e-2f);
}

bool test_moe_w8(CudaBackend& backend) {
    constexpr int hidden_size = 32;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int intermediate_size = 24;
    constexpr int tokens = 2;
    constexpr int group_size = 8;
    const int gate_up_rows = num_experts * 2 * intermediate_size;
    const int down_rows = num_experts * hidden_size;
    auto make_weight = [](int rows, int width, int seed,
                          std::vector<int8_t>& values,
                          std::vector<float>& scales) {
        const int groups_per_row = width / group_size;
        values.resize(static_cast<size_t>(rows) * width);
        scales.resize(static_cast<size_t>(rows) * groups_per_row);
        for (int row = 0; row < rows; ++row) {
            for (int group = 0; group < groups_per_row; ++group) {
                scales[static_cast<size_t>(row) * groups_per_row + group] =
                    0.003f * (1 + (row + group + seed) % 5);
                for (int inner = 0; inner < group_size; ++inner) {
                    const int column = group * group_size + inner;
                    values[static_cast<size_t>(row) * width + column] =
                        static_cast<int8_t>(
                            (row * 3 + column + seed) % 31 - 15);
                }
            }
        }
    };
    std::vector<int8_t> gate_up_values;
    std::vector<float> gate_up_scales;
    std::vector<int8_t> down_values;
    std::vector<float> down_scales;
    make_weight(gate_up_rows, hidden_size, 3,
                gate_up_values, gate_up_scales);
    make_weight(down_rows, intermediate_size, 11,
                down_values, down_scales);

    std::vector<float> hidden(hidden_size * tokens);
    std::vector<float> router_f32(num_experts * hidden_size);
    for (size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 23.0f;
    for (size_t i = 0; i < router_f32.size(); ++i)
        router_f32[i] =
            static_cast<float>(static_cast<int>(i % 13) - 6) / 19.0f;
    std::vector<mollm::cpu::fp16_t> router_data(router_f32.size());
    for (size_t i = 0; i < router_f32.size(); ++i)
        router_data[i] =
            static_cast<mollm::cpu::fp16_t>(router_f32[i]);

    Tensor hidden_host = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, hidden_size, tokens, 1, 1,
        hidden.data());
    Tensor router = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, num_experts, hidden_size,
        1, 1, router_data.data());
    Tensor gate_up = Tensor::create(
        Precision::INT8, MemoryType::EXTERNAL, gate_up_rows, hidden_size,
        1, 1, gate_up_values.data());
    gate_up.rowmajor_data = gate_up_values.data();
    gate_up.scales = gate_up_scales.data();
    gate_up.group_size = group_size;
    gate_up.groups_per_row = hidden_size / group_size;
    Tensor down = Tensor::create(
        Precision::INT8, MemoryType::EXTERNAL, down_rows, intermediate_size,
        1, 1, down_values.data());
    down.rowmajor_data = down_values.data();
    down.scales = down_scales.data();
    down.group_size = group_size;
    down.groups_per_row = intermediate_size / group_size;
    std::vector<const Tensor*> inputs = {
        &hidden_host, &router, &gate_up, &down,
    };
    std::vector<float> expected(hidden_size * tokens);
    Tensor expected_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, hidden_size, tokens, 1, 1,
        expected.data());
    if (!kernel_qwen3_moe(
            inputs, expected_tensor, nullptr, hidden_size, num_experts,
            top_k, intermediate_size, 0, 1, true, false, 2, 1, 0.8f))
        return false;
    if (std::none_of(expected.begin(), expected.end(),
                     [](float value) { return std::fabs(value) > 1e-6f; }))
        return false;

    backend.wrap_weight(router);
    backend.wrap_weight_int4(gate_up, true);
    backend.wrap_weight_int4(down, true);
    Tensor hidden_device = device_tensor(backend, hidden_size, tokens);
    Tensor output_device = device_tensor(backend, hidden_size, tokens);
    std::memcpy(hidden_device.data, hidden.data(), hidden_device.nbytes());
    inputs = {&hidden_device, &router, &gate_up, &down};
    GraphNode node;
    node.op_type = OpType::MOE;
    node.params.i32 = {
        hidden_size, num_experts, top_k, intermediate_size, 0,
        1, 1, 0, 2, 1, 0, -1, -1, -1,
    };
    node.params.f32 = {0.8f, 0.0f};
    router.data = gate_up.data = down.data = nullptr;
    backend.clear_dispatch_error();
    backend.dispatch(node, inputs, &output_device, nullptr);
    backend.end_graph();
    std::vector<float> actual(expected.size());
    std::memcpy(actual.data(), output_device.data, output_device.nbytes());
    return !backend.dispatch_failed() &&
        close_enough(actual, expected, 3.0e-2f);
}

}  // namespace

int main() {
    CudaBackend backend;
    if (!backend.available()) {
        std::fprintf(stderr, "CUDA device unavailable; skipping\n");
        return 77;
    }

    constexpr int m = 3;
    constexpr int n = 8;
    constexpr int k = 32;
    std::vector<float> activation(static_cast<size_t>(m) * k);
    std::vector<float> weight_f32(static_cast<size_t>(n) * k);
    for (size_t i = 0; i < activation.size(); ++i)
        activation[i] = static_cast<float>(static_cast<int>(i % 13) - 6) /
                        11.0f;
    for (size_t i = 0; i < weight_f32.size(); ++i)
        weight_f32[i] = static_cast<float>(static_cast<int>(i % 9) - 4) /
                        17.0f;

    std::vector<mollm::cpu::fp16_t> weight_fp16(weight_f32.size());
    for (size_t i = 0; i < weight_f32.size(); ++i)
        weight_fp16[i] = static_cast<mollm::cpu::fp16_t>(weight_f32[i]);
    Tensor fp16 = Tensor::create(Precision::FP16, MemoryType::EXTERNAL,
                                 n, k, 1, 1, weight_fp16.data());
    backend.wrap_weight(fp16);
    std::vector<float> actual(static_cast<size_t>(m) * n);
    std::vector<float> expected(actual.size());
    reference(activation, weight_f32, expected, m, n, k);
    if (!dispatch_matmul(backend, fp16, activation, actual, m, n, k) ||
        !close_enough(actual, expected, 3e-3f))
        return 1;
    if (!backend.is_device_resident() ||
        !test_device_resident_ops(backend, fp16, activation, expected,
                                  m, n, k))
        return 1;
    if (!test_rwkv_matrix_ops(backend))
        return 1;
    if (!test_layout_rope_and_sdpa(backend))
        return 1;
    {
        CudaBackend layout_backend;
        if (!layout_backend.available() ||
            !test_strided_layout_ops(layout_backend))
            return 1;
    }

    Q4B8G32Block block{};
    std::vector<float> q4_weight(static_cast<size_t>(n) * k);
    for (int row = 0; row < n; ++row) {
        block.scales[row] = 0.125f + row * 0.01f;
        for (int inner = 0; inner < k; inner += 2) {
            const int low = (row + inner) % 15 - 7;
            const int high = (row + inner + 1) % 15 - 7;
            block.q[row][inner / 2] =
                static_cast<uint8_t>((low & 0xf) | ((high & 0xf) << 4));
            q4_weight[static_cast<size_t>(row) * k + inner] =
                low * block.scales[row];
            q4_weight[static_cast<size_t>(row) * k + inner + 1] =
                high * block.scales[row];
        }
    }
    Tensor q4 = Tensor::create(Precision::INT4, MemoryType::EXTERNAL,
                               n, k, 1, 1, &block);
    q4.rowmajor_data = &block;
    q4.q4_g32_data = &block;
    q4.is_q4_g32_packed = true;
    q4.group_size = 32;
    q4.groups_per_row = 1;
    backend.wrap_weight_int4(q4);
    std::fill(actual.begin(), actual.end(), 0.0f);
    reference(activation, q4_weight, expected, m, n, k);
    if (!dispatch_matmul(backend, q4, activation, actual, m, n, k) ||
        !close_enough(actual, expected, 8e-3f))
        return 1;

    constexpr int k128 = 128;
    std::vector<float> activation128(static_cast<size_t>(m) * k128);
    for (size_t i = 0; i < activation128.size(); ++i)
        activation128[i] =
            static_cast<float>(static_cast<int>(i % 17) - 8) / 19.0f;
    Q4B8G128Block block128{};
    std::vector<float> q4_weight128(static_cast<size_t>(n) * k128);
    for (int row = 0; row < n; ++row) {
        block128.scales[row] = 0.0625f + row * 0.005f;
        for (int inner = 0; inner < k128; inner += 2) {
            const int low = (row + inner) % 15 - 7;
            const int high = (row + inner + 1) % 15 - 7;
            const int qgroup = inner / 32;
            const int qinner = inner % 32;
            block128.q[qgroup][row][qinner / 2] =
                static_cast<uint8_t>((low & 0xf) | ((high & 0xf) << 4));
            q4_weight128[static_cast<size_t>(row) * k128 + inner] =
                low * block128.scales[row];
            q4_weight128[static_cast<size_t>(row) * k128 + inner + 1] =
                high * block128.scales[row];
        }
    }
    Tensor q4_128 = Tensor::create(Precision::INT4, MemoryType::EXTERNAL,
                                   n, k128, 1, 1, &block128);
    q4_128.rowmajor_data = &block128;
    q4_128.q4_g128_data = &block128;
    q4_128.is_q4_g128_packed = true;
    q4_128.group_size = 128;
    q4_128.groups_per_row = 1;
    backend.wrap_weight_int4(q4_128);
    std::fill(actual.begin(), actual.end(), 0.0f);
    reference(activation128, q4_weight128, expected, m, n, k128);
    if (!dispatch_matmul(backend, q4_128, activation128, actual,
                         m, n, k128) ||
        !close_enough(actual, expected, 1.5e-2f))
        return 1;

    {
        CudaBackend moe_backend;
        if (!moe_backend.available() ||
            !test_moe_fp16_shared(moe_backend))
            return 1;
    }
    {
        CudaBackend moe_backend;
        if (!moe_backend.available() ||
            !test_moe_hash_routing(moe_backend))
            return 1;
    }
    {
        CudaBackend moe_backend;
        if (!moe_backend.available() ||
            !test_moe_packed<32, Q4B8G32Block>(moe_backend))
            return 1;
    }
    {
        CudaBackend moe_backend;
        if (!moe_backend.available() ||
            !test_moe_packed<128, Q4B8G128Block>(moe_backend))
            return 1;
    }
    {
        CudaBackend moe_backend;
        if (!moe_backend.available() || !test_moe_w8(moe_backend))
            return 1;
    }

    std::printf("CUDA device-resident ops, RWKV and standard/hash MoE "
                "variants, strided views, FP16, W8, W4G32 and W4G128 "
                "matmul tests passed\n");
    return 0;
}
