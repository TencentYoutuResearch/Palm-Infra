#include "backends/cuda/backend.h"
#include "backends/cuda/prefill_batch_policy.h"
#include "backends/cuda/device_runtime.h"
#include "engine/engine.h"
#include "graph/execute.h"
#include "backends/cpu/backend.h"
#include "core/quant_layouts.h"
#include "core/bf16.h"
#include "runtime/expert_provider.h"
#include "kernels/cuda/moe.h"
#include "kernels/cuda/matmul.h"
#include "kernels/cpu/moe/moe_routing.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool close_enough(const std::vector<float>& actual,
                  const std::vector<float>& expected, float tolerance,
                  int line = __builtin_LINE()) {
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            std::fabs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr,
                         "CUDA mismatch at %zu (test line %d): actual=%g expected=%g\n",
                         i, line, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

bool close_enough_relative(
    const std::vector<float>& actual, const std::vector<float>& expected,
    float absolute_tolerance, float relative_tolerance,
    int line = __builtin_LINE()) {
    for (size_t index = 0; index < actual.size(); ++index) {
        const float difference = std::fabs(actual[index] - expected[index]);
        const float tolerance = absolute_tolerance +
            relative_tolerance * std::fabs(expected[index]);
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) || difference > tolerance) {
            std::fprintf(
                stderr,
                "CUDA mismatch at %zu (test line %d): actual=%g expected=%g "
                "tolerance=%g\n",
                index, line, actual[index], expected[index], tolerance);
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

bool upload(CudaBackend& backend, Tensor& tensor,
            const std::vector<float>& values) {
    return values.size() * sizeof(float) == tensor.nbytes() &&
        backend.copy_from_host(values.data(), tensor, tensor.nbytes());
}

bool download(CudaBackend& backend, const Tensor& tensor,
              std::vector<float>& values) {
    values.resize(static_cast<size_t>(tensor.nelements()));
    return backend.copy_to_host(
        tensor, values.data(), values.size() * sizeof(float));
}

bool test_device_resident_ops(CudaBackend& backend, Tensor& weight,
                              const std::vector<float>& activation,
                              const std::vector<float>& expected,
                              int m, int n, int k) {
    Tensor a = device_tensor(backend, k, m);
    Tensor c = device_tensor(backend, n, m);
    if (!a.data || !c.data)
        return false;
    if (!upload(backend, a, activation))
        return false;

    GraphNode matmul;
    matmul.op_type = OpType::MATMUL;
    backend.clear_dispatch_error();
    backend.dispatch(matmul, {&a, &weight}, &c, nullptr);
    backend.end_graph();
    std::vector<float> actual;
    if (!download(backend, c, actual))
        return false;
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
    if (!upload(backend, norm_source, norm_input))
        return false;
    Tensor norm_scale = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, width, 1, 1, 1,
        norm_weight.data());
    backend.wrap_weight(norm_scale);
    GraphNode rms_norm;
    rms_norm.op_type = OpType::RMS_NORM;
    backend.dispatch(rms_norm, {&norm_source, &norm_scale},
                     &norm_output, nullptr);
    backend.end_graph();
    if (!download(backend, norm_output, actual))
        return false;
    if (backend.dispatch_failed() ||
        !close_enough(actual, norm_expected, 2e-5f))
        return false;

    Tensor lhs = device_tensor(backend, width, rows);
    Tensor rhs = device_tensor(backend, width, rows);
    Tensor sum = device_tensor(backend, width, rows);
    Tensor silu = device_tensor(backend, width, rows);
    std::vector<float> lhs_data(width * rows);
    std::vector<float> rhs_data(width * rows);
    std::vector<float> elementwise_expected(width * rows);
    for (int i = 0; i < width * rows; ++i) {
        lhs_data[i] = (i - 5) / 7.0f;
        rhs_data[i] = (3 - i) / 11.0f;
        const float value = lhs_data[i] + rhs_data[i];
        elementwise_expected[i] = value / (1.0f + std::exp(-value));
    }
    if (!upload(backend, lhs, lhs_data) || !upload(backend, rhs, rhs_data))
        return false;
    GraphNode add;
    add.op_type = OpType::ADD;
    GraphNode silu_node;
    silu_node.op_type = OpType::SILU;
    backend.dispatch(add, {&lhs, &rhs}, &sum, nullptr);
    backend.dispatch(silu_node, {&sum}, &silu, nullptr);
    backend.end_graph();
    if (!download(backend, silu, actual))
        return false;
    if (backend.dispatch_failed() ||
        !close_enough(actual, elementwise_expected, 2e-6f))
        return false;

    Tensor view_source = device_tensor(backend, 6, 2);
    // Both scalar operand positions must remain native, including constants
    // used for Qwen3.8 residual gates. Use exact values to isolate broadcasting.
    Tensor scalar = device_tensor(backend, 1);
    if (!upload(backend, scalar, {0.25f}) ||
        !backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE))
        return false;
    for (OpType operation : {OpType::MUL, OpType::ADD}) {
        for (bool scalar_left : {false, true}) {
            GraphNode binary;
            binary.op_type = operation;
            backend.dispatch(binary, scalar_left
                ? std::vector<const Tensor*>{&scalar, &lhs}
                : std::vector<const Tensor*>{&lhs, &scalar}, &sum, nullptr);
            std::vector<float> broadcast_expected(lhs_data.size());
            for (size_t i = 0; i < lhs_data.size(); ++i)
                broadcast_expected[i] = operation == OpType::MUL
                    ? lhs_data[i] * 0.25f : lhs_data[i] + 0.25f;
            if (backend.dispatch_failed() || !download(backend, sum, actual) ||
                !close_enough(actual, broadcast_expected, 0.0f))
                return false;
        }
    }
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::ALLOW_REFERENCE);
    std::vector<float> view_source_data(12);
    for (int i = 0; i < 12; ++i)
        view_source_data[i] = (i - 4) / 5.0f;
    if (!upload(backend, view_source, view_source_data))
        return false;
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
            const float value = view_source_data[row * 6 + column + 2];
            view_expected[row * 2 + column] =
                value / (1.0f + std::exp(-value));
        }
    if (!download(backend, view_result, actual))
        return false;
    return !backend.dispatch_failed() &&
        close_enough(actual, view_expected, 2e-6f);
}

bool test_layout_rope_and_sdpa(CudaBackend& backend,
                               Precision cache_precision) {
    backend.clear_dispatch_error();
    Tensor rope_storage = device_tensor(backend, 10, 2);
    std::vector<float> rope_storage_data(20);
    for (int i = 0; i < 20; ++i)
        rope_storage_data[i] = (i - 8) / 17.0f;
    if (!upload(backend, rope_storage, rope_storage_data))
        return false;
    Tensor rope_input;
    GraphNode slice;
    slice.op_type = OpType::SLICE;
    slice.params.i32 = {0, 1, 8};
    backend.dispatch(slice, {&rope_storage}, &rope_input, nullptr);
    Tensor cosine = device_tensor(backend, 4, 2);
    Tensor sine = device_tensor(backend, 4, 2);
    std::vector<float> cosine_data(8);
    std::vector<float> sine_data(8);
    for (int i = 0; i < 8; ++i) {
        cosine_data[i] = std::cos((i + 1) * 0.13f);
        sine_data[i] = std::sin((i + 1) * 0.13f);
    }
    if (!upload(backend, cosine, cosine_data) ||
        !upload(backend, sine, sine_data))
        return false;
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
            const float x0 = rope_storage_data[row * 10 + pair + 1];
            const float x1 =
                rope_storage_data[row * 10 + pair + 5];
            const float c = cosine_data[row * 4 + pair];
            const float s = sine_data[row * 4 + pair];
            expected[row * 8 + pair] = x0 * c - x1 * s;
            expected[row * 8 + pair + 4] = x0 * s + x1 * c;
        }
    std::vector<float> actual;
    if (!download(backend, rope_output, actual))
        return false;
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
                rope_storage_data[row * 10 + column + 1];
    if (!download(backend, contiguous, actual))
        return false;
    if (backend.dispatch_failed() || !close_enough(actual, expected, 0.0f))
        return false;

    // TILE must replicate a sliced, row-strided input on every axis without
    // staging the view through CPU memory.
    Tensor tiled = device_tensor(backend, 16, 6, 2, 2);
    GraphNode tile_node;
    tile_node.op_type = OpType::TILE;
    tile_node.params.i32 = {2, 3, 2, 2};
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    backend.dispatch(tile_node, {&rope_input}, &tiled, nullptr);
    backend.end_graph();
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::ALLOW_REFERENCE);
    if (!download(backend, tiled, actual) || backend.dispatch_failed()) return false;
    expected.resize(16 * 6 * 2 * 2);
    for (size_t index = 0; index < expected.size(); ++index)
        expected[index] = rope_storage_data[((index / 16) % 6 % 2) * 10 + (index % 16 % 8) + 1];
    if (!close_enough(actual, expected, 0.0f)) return false;

    // Growing boundary inputs must refresh contents without reallocating
    // within the same geometric capacity bucket.
    void* previous_boundary = nullptr;
    const int boundary_sizes[] = {1, 1025, 1100, 2049, 2050, 16};
    for (int step = 0; step < 6; ++step) {
        const int count = boundary_sizes[step];
        std::vector<float> values(count);
        for (int i = 0; i < count; ++i) values[i] = static_cast<float>(i + step * 7);
        Tensor boundary = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
            count, 1, 1, 1, values.data());
        backend.upload_input(boundary, "growth-regression", values.data(), values.size() * sizeof(float));
        if (backend.dispatch_failed() || !download(backend, boundary, actual) ||
            !close_enough(actual, values, 0.0f)) return false;
        if (!std::getenv("MOLLM_CUDA_DISABLE_GEOMETRIC_GROWTH") &&
            (step == 2 || step >= 4) && boundary.device.buffer != previous_boundary) return false;
        previous_boundary = boundary.device.buffer;
    }

    constexpr int norm_width = 8;
    constexpr int norm_rows = 2;
    Tensor residual = device_tensor(backend, norm_width, norm_rows);
    Tensor update = device_tensor(backend, norm_width, norm_rows);
    Tensor add_norm_output = device_tensor(backend, norm_width, norm_rows);
    std::vector<float> norm_weight(norm_width);
    std::vector<float> residual_data(norm_width * norm_rows);
    std::vector<float> update_data(norm_width * norm_rows);
    std::vector<float> residual_expected(norm_width * norm_rows);
    std::vector<float> add_norm_expected(norm_width * norm_rows);
    for (int column = 0; column < norm_width; ++column)
        norm_weight[column] = 0.8f + column * 0.02f;
    for (int i = 0; i < norm_width * norm_rows; ++i) {
        residual_data[i] = (i - 6) / 9.0f;
        update_data[i] = (4 - i) / 13.0f;
        residual_expected[i] =
            residual_data[i] + update_data[i];
    }
    if (!upload(backend, residual, residual_data) ||
        !upload(backend, update, update_data))
        return false;
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
    if (!download(backend, add_norm_output, actual))
        return false;
    std::vector<float> residual_actual;
    if (!download(backend, residual, residual_actual))
        return false;
    if (backend.dispatch_failed() ||
        !close_enough(actual, add_norm_expected, 2e-5f) ||
        !close_enough(residual_actual, residual_expected, 2e-6f))
        return false;

    constexpr int rope_sequence = 2;
    constexpr int rope_heads = 3;
    Tensor flat_norm_input =
        device_tensor(backend, norm_width, rope_sequence * rope_heads);
    std::vector<float> flat_norm_data(
        static_cast<size_t>(flat_norm_input.nelements()));
    for (int64_t i = 0; i < flat_norm_input.nelements(); ++i)
        flat_norm_data[static_cast<size_t>(i)] =
            (static_cast<int>(i % 17) - 8) / 15.0f;
    if (!upload(backend, flat_norm_input, flat_norm_data))
        return false;
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
                    flat_norm_data[row * norm_width + dimension];
                sum += value * value;
            }
            const float inverse = 1.0f /
                std::sqrt(sum / norm_width + 1e-6f);
            for (int pair = 0; pair < norm_width / 2; ++pair) {
                const float x0 = flat_norm_data[
                    row * norm_width + pair * 2] * inverse *
                    norm_weight[pair * 2];
                const float x1 = flat_norm_data[
                    row * norm_width + pair * 2 + 1] * inverse *
                    norm_weight[pair * 2 + 1];
                const float c = cosine_data[position * 4 + pair];
                const float s = sine_data[position * 4 + pair];
                expected[row * norm_width + pair * 2] = x0 * c - x1 * s;
                expected[row * norm_width + pair * 2 + 1] = x0 * s + x1 * c;
            }
        }
    if (!download(backend, fused_rope_output, actual))
        return false;
    if (backend.dispatch_failed() || !close_enough(actual, expected, 3e-5f))
        return false;

    constexpr int qk_query_heads = 2;
    constexpr int qk_key_heads = 1;
    constexpr int qk_rope_dim = 6;
    constexpr int qk_query_rows = rope_sequence * qk_query_heads;
    constexpr int qk_key_rows = rope_sequence * qk_key_heads;
    Tensor query_storage =
        device_tensor(backend, norm_width + 3, qk_query_rows);
    Tensor key_storage =
        device_tensor(backend, norm_width + 5, qk_key_rows);
    Tensor qk_query = query_storage.view_2d(norm_width, qk_query_rows);
    Tensor qk_key = key_storage.view_2d(norm_width, qk_key_rows);
    std::vector<float> query_storage_data(
        static_cast<size_t>(query_storage.nelements()), 91.0f);
    std::vector<float> key_storage_data(
        static_cast<size_t>(key_storage.nelements()), -83.0f);
    std::vector<float> qk_query_data(norm_width * qk_query_rows);
    std::vector<float> qk_key_data(norm_width * qk_key_rows);
    for (size_t i = 0; i < qk_query_data.size(); ++i)
        qk_query_data[i] =
            (static_cast<int>((i * 7) % 23) - 11) / 13.0f;
    for (size_t i = 0; i < qk_key_data.size(); ++i)
        qk_key_data[i] =
            (static_cast<int>((i * 5) % 19) - 9) / 11.0f;
    for (int row = 0; row < qk_query_rows; ++row)
        std::copy_n(
            qk_query_data.data() + row * norm_width, norm_width,
            query_storage_data.data() + row * (norm_width + 3));
    for (int row = 0; row < qk_key_rows; ++row)
        std::copy_n(
            qk_key_data.data() + row * norm_width, norm_width,
            key_storage_data.data() + row * (norm_width + 5));
    if (!upload(backend, query_storage, query_storage_data) ||
        !upload(backend, key_storage, key_storage_data))
        return false;
    std::vector<float> key_norm_weight(norm_width);
    for (int dimension = 0; dimension < norm_width; ++dimension)
        key_norm_weight[dimension] = 1.1f - dimension * 0.03f;
    Tensor key_norm_scale = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, norm_width, 1, 1, 1,
        key_norm_weight.data());
    backend.wrap_weight(key_norm_scale);
    Tensor qk_output = device_tensor(
        backend, norm_width, rope_sequence,
        qk_query_heads + qk_key_heads);
    GraphNode qk_fused;
    qk_fused.op_type = OpType::QK_RMS_NORM_ROPE;
    qk_fused.params.i32 = {qk_rope_dim, 0, qk_query_heads};
    qk_fused.params.f32 = {1e-6f};
    backend.dispatch(
        qk_fused,
        {&qk_query, &qk_key, &norm_scale, &key_norm_scale, &cosine, &sine},
        &qk_output, nullptr);
    backend.end_graph();
    expected.assign(
        static_cast<size_t>(norm_width) * rope_sequence *
            (qk_query_heads + qk_key_heads),
        0.0f);
    auto qk_reference = [&](const std::vector<float>& source,
                            const std::vector<float>& weight, int rows,
                            int output_row_offset) {
        for (int row = 0; row < rows; ++row) {
            float square_sum = 0.0f;
            for (int dimension = 0; dimension < norm_width; ++dimension) {
                const float value = source[row * norm_width + dimension];
                square_sum += value * value;
            }
            const float inverse = 1.0f /
                std::sqrt(square_sum / norm_width + 1e-6f);
            const int position = row % rope_sequence;
            for (int pair = 0; pair < qk_rope_dim / 2; ++pair) {
                const int first = pair;
                const int second = pair + qk_rope_dim / 2;
                const float x0 = source[row * norm_width + first] * inverse *
                    weight[first];
                const float x1 = source[row * norm_width + second] * inverse *
                    weight[second];
                const float c = cosine_data[position * 4 + pair];
                const float s = sine_data[position * 4 + pair];
                expected[(output_row_offset + row) * norm_width + first] =
                    x0 * c - x1 * s;
                expected[(output_row_offset + row) * norm_width + second] =
                    x1 * c + x0 * s;
            }
            for (int dimension = qk_rope_dim; dimension < norm_width;
                 ++dimension) {
                expected[(output_row_offset + row) * norm_width + dimension] =
                    source[row * norm_width + dimension] * inverse *
                    weight[dimension];
            }
        }
    };
    qk_reference(qk_query_data, norm_weight, qk_query_rows, 0);
    qk_reference(
        qk_key_data, key_norm_weight, qk_key_rows, qk_query_rows);
    if (!download(backend, qk_output, actual))
        return false;
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
    std::vector<float> query_data(
        static_cast<size_t>(query.nelements()));
    std::vector<float> key_data(static_cast<size_t>(key.nelements()));
    std::vector<float> value_data(static_cast<size_t>(value.nelements()));
    for (int64_t i = 0; i < query.nelements(); ++i)
        query_data[static_cast<size_t>(i)] =
            (static_cast<int>(i % 13) - 6) / 11.0f;
    for (int64_t i = 0; i < key.nelements(); ++i)
        key_data[static_cast<size_t>(i)] =
            (static_cast<int>(i % 9) - 4) / 7.0f;
    for (int64_t i = 0; i < value.nelements(); ++i)
        value_data[static_cast<size_t>(i)] =
            (static_cast<int>(i % 7) - 3) / 5.0f;
    if (!upload(backend, query, query_data) ||
        !upload(backend, key, key_data) ||
        !upload(backend, value, value_data))
        return false;

    const bool fp16_cache = cache_precision == Precision::FP16;
    const size_t cache_element_size = fp16_cache
        ? sizeof(mollm::cpu::fp16_t) : sizeof(float);
    const size_t key_cache_bytes = CacheMetadata::SIZE +
        static_cast<size_t>(kv_heads) * capacity * key_dim *
            cache_element_size;
    const size_t value_cache_bytes = CacheMetadata::SIZE +
        static_cast<size_t>(kv_heads) * capacity * value_dim *
            cache_element_size;
    Tensor key_cache = Tensor::create(
        cache_precision, MemoryType::EXTERNAL,
        key_cache_bytes / cache_element_size);
    Tensor value_cache = Tensor::create(
        cache_precision, MemoryType::EXTERNAL,
        value_cache_bytes / cache_element_size);
    backend.alloc_persistent(
        key_cache, key_cache_bytes,
        PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX,
        CacheMetadata::SIZE);
    backend.alloc_persistent(
        value_cache, value_cache_bytes,
        PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX,
        CacheMetadata::SIZE);
    CacheMetadata key_metadata;
    key_metadata.current_seq_len = past_length;
    key_metadata.max_seq_len = capacity;
    key_metadata.num_kv_heads = kv_heads;
    key_metadata.head_dim = key_dim;
    CacheMetadata value_metadata;
    value_metadata.current_seq_len = past_length;
    value_metadata.max_seq_len = capacity;
    value_metadata.num_kv_heads = kv_heads;
    value_metadata.v_head_dim = value_dim;
    if (!backend.copy_from_host(
            &key_metadata, key_cache, sizeof(key_metadata)) ||
        !backend.copy_from_host(
            &value_metadata, value_cache, sizeof(value_metadata)))
        return false;
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
    auto round_for_cache = [fp16_cache](float value) {
        return fp16_cache
            ? static_cast<float>(mollm::cpu::fp16_t(value)) : value;
    };
    if (fp16_cache) {
        std::vector<mollm::cpu::fp16_t> cached_key(initial_key.size());
        std::vector<mollm::cpu::fp16_t> cached_value(initial_value.size());
        for (size_t index = 0; index < initial_key.size(); ++index) {
            cached_key[index] =
                static_cast<mollm::cpu::fp16_t>(initial_key[index]);
            initial_key[index] = static_cast<float>(cached_key[index]);
        }
        for (size_t index = 0; index < initial_value.size(); ++index) {
            cached_value[index] =
                static_cast<mollm::cpu::fp16_t>(initial_value[index]);
            initial_value[index] = static_cast<float>(cached_value[index]);
        }
        if (!backend.copy_from_host(
                cached_key.data(), key_cache,
                cached_key.size() * sizeof(cached_key[0]),
                CacheMetadata::SIZE) ||
            !backend.copy_from_host(
                cached_value.data(), value_cache,
                cached_value.size() * sizeof(cached_value[0]),
                CacheMetadata::SIZE))
            return false;
    } else if (
        !backend.copy_from_host(
            initial_key.data(), key_cache,
            initial_key.size() * sizeof(float), CacheMetadata::SIZE) ||
        !backend.copy_from_host(
            initial_value.data(), value_cache,
            initial_value.size() * sizeof(float), CacheMetadata::SIZE)) {
        return false;
    }

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
                        : round_for_cache(key_data[
                              (key_head * current_length + key_position -
                               past_length) * key_dim + dimension]);
                    score += query_data[
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
                        : round_for_cache(value_data[
                              (key_head * current_length + key_position -
                               past_length) * value_dim + dimension]);
                    expected[(head * query_length + position) * value_dim +
                             dimension] += scores[key_position] / sum * current;
                }
            }
        }
    }
    if (!download(backend, attention_output, actual))
        return false;
    if (backend.dispatch_failed() || !close_enough(actual, expected, 3e-5f))
        return false;
    std::vector<uint8_t> cached_key_bytes(
        static_cast<size_t>(kv_heads) * capacity * key_dim *
        cache_element_size);
    if (!backend.copy_to_host(
            key_cache, cached_key_bytes.data(), cached_key_bytes.size(),
            CacheMetadata::SIZE))
        return false;
    auto cached_key_value = [&](size_t index) {
        if (fp16_cache) {
            mollm::cpu::fp16_t value;
            std::memcpy(
                &value, cached_key_bytes.data() + index * sizeof(value),
                sizeof(value));
            return static_cast<float>(value);
        }
        float value;
        std::memcpy(
            &value, cached_key_bytes.data() + index * sizeof(value),
            sizeof(value));
        return value;
    };
    for (int head = 0; head < kv_heads; ++head)
        for (int position = 0; position < current_length; ++position)
            for (int dimension = 0; dimension < key_dim; ++dimension)
                if (std::fabs(cached_key_value(
                        (head * capacity + past_length + position) * key_dim +
                        dimension) - round_for_cache(key_data[
                            (head * current_length + position) * key_dim +
                            dimension])) > 1e-6f)
                    return false;

    // Decode from the nonzero prefill length and append at the final slot.
    Tensor decode_query = device_tensor(backend, key_dim, 1, heads);
    Tensor decode_key = device_tensor(backend, key_dim, 1, kv_heads);
    Tensor decode_value = device_tensor(backend, value_dim, 1, kv_heads);
    Tensor decode_output = device_tensor(backend, value_dim, 1, heads);
    std::vector<float> decode_query_data(
        static_cast<size_t>(decode_query.nelements()));
    std::vector<float> decode_key_data(
        static_cast<size_t>(decode_key.nelements()));
    std::vector<float> decode_value_data(
        static_cast<size_t>(decode_value.nelements()));
    for (size_t index = 0; index < decode_query_data.size(); ++index)
        decode_query_data[index] =
            (static_cast<int>(index) - 5) / 13.0f;
    for (size_t index = 0; index < decode_key_data.size(); ++index)
        decode_key_data[index] =
            (static_cast<int>(index) + 2) / 9.0f;
    for (size_t index = 0; index < decode_value_data.size(); ++index)
        decode_value_data[index] =
            (static_cast<int>(index) - 1) / 7.0f;
    if (!upload(backend, decode_query, decode_query_data) ||
        !upload(backend, decode_key, decode_key_data) ||
        !upload(backend, decode_value, decode_value_data))
        return false;
    const uint64_t decode_past = total_length;
    if (!backend.copy_from_host(
            &decode_past, key_cache, sizeof(decode_past),
            offsetof(CacheMetadata, current_seq_len)) ||
        !backend.copy_from_host(
            &decode_past, value_cache, sizeof(decode_past),
            offsetof(CacheMetadata, current_seq_len)))
        return false;
    backend.dispatch(
        sdpa,
        {&decode_query, &decode_key, &decode_value, nullptr,
         &key_cache, &value_cache},
        &decode_output, nullptr);
    backend.end_graph();
    if (backend.dispatch_failed())
        return false;
    std::vector<float> decode_actual;
    if (!download(backend, decode_output, decode_actual))
        return false;
    std::vector<float> decode_expected(
        static_cast<size_t>(heads) * value_dim, 0.0f);
    for (int head = 0; head < heads; ++head) {
        const int key_head = head / heads_per_group;
        float scores[total_length + 1];
        float maximum = -INFINITY;
        for (int key_position = 0; key_position <= total_length;
             ++key_position) {
            float score = 0.0f;
            for (int dimension = 0; dimension < key_dim; ++dimension) {
                float key_element;
                if (key_position < past_length) {
                    key_element = initial_key[
                        (key_head * capacity + key_position) * key_dim +
                        dimension];
                } else if (key_position < total_length) {
                    key_element = round_for_cache(key_data[
                        (key_head * current_length + key_position -
                         past_length) * key_dim + dimension]);
                } else {
                    key_element = round_for_cache(
                        decode_key_data[key_head * key_dim + dimension]);
                }
                score += decode_query_data[head * key_dim + dimension] *
                    key_element;
            }
            scores[key_position] = score * scale;
            maximum = std::max(maximum, scores[key_position]);
        }
        float sum = 0.0f;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            sum += score;
        }
        for (int key_position = 0; key_position <= total_length;
             ++key_position) {
            for (int dimension = 0; dimension < value_dim; ++dimension) {
                float value_element;
                if (key_position < past_length) {
                    value_element = initial_value[
                        (key_head * capacity + key_position) * value_dim +
                        dimension];
                } else if (key_position < total_length) {
                    value_element = round_for_cache(value_data[
                        (key_head * current_length + key_position -
                         past_length) * value_dim + dimension]);
                } else {
                    value_element = round_for_cache(
                        decode_value_data[key_head * value_dim + dimension]);
                }
                decode_expected[head * value_dim + dimension] +=
                    scores[key_position] / sum * value_element;
            }
        }
    }
    if (!close_enough(decode_actual, decode_expected, 3e-5f))
        return false;

    // FP16 cached attention vectorizes aligned, contiguous query rows. A
    // borrowed view may still begin at a scalar-float offset, which must use
    // the numerically equivalent scalar path instead of a misaligned float2
    // load.
    if (fp16_cache) {
        Tensor misaligned_storage = device_tensor(
            backend, static_cast<int64_t>(decode_query_data.size()) + 1);
        std::vector<float> padded_query(
            decode_query_data.size() + 1, 0.0f);
        std::copy(
            decode_query_data.begin(), decode_query_data.end(),
            padded_query.begin() + 1);
        if (!upload(backend, misaligned_storage, padded_query))
            return false;
        Tensor misaligned_query = misaligned_storage;
        misaligned_query.shape[0] = key_dim;
        misaligned_query.shape[1] = 1;
        misaligned_query.shape[2] = heads;
        misaligned_query.shape[3] = 1;
        misaligned_query.compute_strides();
        misaligned_query.device.offset += sizeof(float);
        backend.dispatch(
            sdpa,
            {&misaligned_query, &decode_key, &decode_value, nullptr,
             &key_cache, &value_cache},
            &decode_output, nullptr);
        backend.end_graph();
        if (backend.dispatch_failed() ||
            !download(backend, decode_output, decode_actual) ||
            !close_enough(decode_actual, decode_expected, 3e-5f))
            return false;
    }
    if (!backend.copy_to_host(
            key_cache, cached_key_bytes.data(), cached_key_bytes.size(),
            CacheMetadata::SIZE))
        return false;
    for (int head = 0; head < kv_heads; ++head)
        for (int dimension = 0; dimension < key_dim; ++dimension)
            if (std::fabs(
                    cached_key_value(
                        (head * capacity + total_length) * key_dim +
                        dimension) -
                    round_for_cache(
                        decode_key_data[head * key_dim + dimension])) >
                1e-6f)
                return false;

    // Reset only the length metadata. A one-token attention result must then
    // depend solely on the newly written value, not stale cache payload.
    const uint64_t reset_length = 0;
    if (!backend.copy_from_host(
            &reset_length, key_cache, sizeof(reset_length),
            offsetof(CacheMetadata, current_seq_len)) ||
        !backend.copy_from_host(
            &reset_length, value_cache, sizeof(reset_length),
            offsetof(CacheMetadata, current_seq_len)))
        return false;
    for (float& value : decode_key_data)
        value += 0.375f;
    for (float& value : decode_value_data)
        value -= 0.25f;
    if (!upload(backend, decode_key, decode_key_data) ||
        !upload(backend, decode_value, decode_value_data))
        return false;
    backend.dispatch(
        sdpa,
        {&decode_query, &decode_key, &decode_value, nullptr,
         &key_cache, &value_cache},
        &decode_output, nullptr);
    backend.end_graph();
    if (backend.dispatch_failed() ||
        !download(backend, decode_output, decode_actual))
        return false;
    for (int head = 0; head < heads; ++head)
        for (int dimension = 0; dimension < value_dim; ++dimension)
            if (std::fabs(
                    decode_actual[head * value_dim + dimension] -
                    round_for_cache(decode_value_data[
                        (head / heads_per_group) * value_dim + dimension])) >
                3e-5f)
                return false;

    // Force SDPA through the explicit device-to-host fallback bridge by
    // making only the native metadata validation fail. The CPU reference
    // ignores head_dim metadata and must append the staged FP16/FP32 cache
    // back to device storage.
    key_metadata.current_seq_len = 0;
    key_metadata.head_dim = key_dim + 1;
    value_metadata.current_seq_len = 0;
    if (!backend.copy_from_host(
            &key_metadata, key_cache, sizeof(key_metadata)) ||
        !backend.copy_from_host(
            &value_metadata, value_cache, sizeof(value_metadata)))
        return false;
    for (float& value : decode_key_data)
        value -= 0.125f;
    if (!upload(backend, decode_key, decode_key_data))
        return false;
    backend.dispatch(
        sdpa,
        {&decode_query, &decode_key, &decode_value, nullptr,
         &key_cache, &value_cache},
        &decode_output, nullptr);
    backend.end_graph();
    if (backend.dispatch_failed() ||
        !backend.copy_to_host(
            key_cache, cached_key_bytes.data(), cached_key_bytes.size(),
            CacheMetadata::SIZE))
        return false;
    for (int head = 0; head < kv_heads; ++head)
        for (int dimension = 0; dimension < key_dim; ++dimension)
            if (std::fabs(
                    cached_key_value(head * capacity * key_dim + dimension) -
                    round_for_cache(
                        decode_key_data[head * key_dim + dimension])) >
                1e-6f)
                return false;
    return true;
}

bool test_wide_cached_sdpa(CudaBackend& backend,
                           Precision cache_precision, int past_length,
                           int current_length = 1,
                           bool explicit_mask = false,
                           int kv_heads = 1, int head_dim = 128,
                           bool canonical_mask = false, int heads_per_kv = 2,
                           int capacity_override = 0) {
    backend.clear_dispatch_error();
    const int heads = heads_per_kv * kv_heads;
    // Exercise common Qwen head widths so decode and prefill specializations
    // can test their parallel reductions against the same reference.
    const int key_dim = head_dim;
    const int value_dim = head_dim;
    const int key_length = past_length + current_length;
    const int capacity = capacity_override ? capacity_override : key_length + 1;
    if(capacity<key_length)return false;

    Tensor query = device_tensor(backend, key_dim, current_length, heads);
    Tensor key = device_tensor(backend, key_dim, current_length, kv_heads);
    Tensor value = device_tensor(
        backend, value_dim, current_length, kv_heads);
    Tensor output = device_tensor(
        backend, value_dim, current_length, heads);
    std::vector<float> query_data(query.nelements());
    std::vector<float> key_data(key.nelements());
    std::vector<float> value_data(value.nelements());
    for (size_t index = 0; index < query_data.size(); ++index)
        query_data[index] =
            (static_cast<int>(index % 11) - 5) / 13.0f;
    for (size_t index = 0; index < key_data.size(); ++index)
        key_data[index] =
            (static_cast<int>(index % 7) - 3) / 9.0f;
    for (size_t index = 0; index < value_data.size(); ++index)
        value_data[index] =
            (static_cast<int>(index % 9) - 4) / 10.0f;
    if (!upload(backend, query, query_data) ||
        !upload(backend, key, key_data) ||
        !upload(backend, value, value_data))
        return false;

    Tensor mask;
    const Tensor* mask_input = nullptr;
    std::vector<float> mask_data;
    if (explicit_mask) {
        mask = device_tensor(backend, key_length, current_length);
        mask_data.resize(mask.nelements());
        for (int query_position = 0; query_position < current_length;
             ++query_position) {
            for (int position = 0; position < key_length; ++position) {
                mask_data[static_cast<size_t>(query_position) * key_length +
                          position] =
                    canonical_mask ? (position>past_length+query_position?-1e38f:0.f) :
                    position > past_length + query_position
                        ? -INFINITY
                        : ((query_position * 3 + position) % 5 - 2) / 100.0f;
            }
        }
        if (!upload(backend, mask, mask_data))
            return false;
        mask_input = &mask;
    }

    std::vector<float> cached_key(
        static_cast<size_t>(kv_heads) * capacity * key_dim);
    std::vector<float> cached_value(
        static_cast<size_t>(kv_heads) * capacity * value_dim);
    for (int key_head = 0; key_head < kv_heads; ++key_head) {
        for (int position = 0; position < capacity; ++position) {
            for (int dimension = 0; dimension < key_dim; ++dimension) {
                cached_key[
                    (static_cast<size_t>(key_head) * capacity + position) *
                        key_dim + dimension] =
                    ((key_head * 11 + position * 5 + dimension * 3) % 23 -
                     11) / 17.0f;
            }
            for (int dimension = 0; dimension < value_dim; ++dimension) {
                cached_value[
                    (static_cast<size_t>(key_head) * capacity + position) *
                        value_dim + dimension] =
                    ((key_head * 13 + position * 7 + dimension * 2) % 19 -
                     9) / 14.0f;
            }
        }
    }

    const bool fp16_cache = cache_precision == Precision::FP16;
    const size_t cache_element_size = fp16_cache
        ? sizeof(mollm::cpu::fp16_t) : sizeof(float);
    const size_t key_cache_bytes = CacheMetadata::SIZE +
        cached_key.size() * cache_element_size;
    const size_t value_cache_bytes = CacheMetadata::SIZE +
        cached_value.size() * cache_element_size;
    Tensor key_cache = Tensor::create(
        cache_precision, MemoryType::EXTERNAL,
        key_cache_bytes / cache_element_size);
    Tensor value_cache = Tensor::create(
        cache_precision, MemoryType::EXTERNAL,
        value_cache_bytes / cache_element_size);
    backend.alloc_persistent(
        key_cache, key_cache_bytes,
        PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX,
        CacheMetadata::SIZE);
    backend.alloc_persistent(
        value_cache, value_cache_bytes,
        PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX,
        CacheMetadata::SIZE);

    CacheMetadata key_metadata;
    key_metadata.current_seq_len = past_length;
    key_metadata.max_seq_len = capacity;
    key_metadata.num_kv_heads = kv_heads;
    key_metadata.head_dim = key_dim;
    CacheMetadata value_metadata;
    value_metadata.current_seq_len = past_length;
    value_metadata.max_seq_len = capacity;
    value_metadata.num_kv_heads = kv_heads;
    value_metadata.v_head_dim = value_dim;
    if (!backend.copy_from_host(
            &key_metadata, key_cache, sizeof(key_metadata)) ||
        !backend.copy_from_host(
            &value_metadata, value_cache, sizeof(value_metadata)))
        return false;
    if (fp16_cache) {
        std::vector<mollm::cpu::fp16_t> key_fp16(cached_key.size());
        std::vector<mollm::cpu::fp16_t> value_fp16(cached_value.size());
        for (size_t index = 0; index < cached_key.size(); ++index) {
            key_fp16[index] =
                static_cast<mollm::cpu::fp16_t>(cached_key[index]);
            cached_key[index] = static_cast<float>(key_fp16[index]);
        }
        for (size_t index = 0; index < cached_value.size(); ++index) {
            value_fp16[index] =
                static_cast<mollm::cpu::fp16_t>(cached_value[index]);
            cached_value[index] = static_cast<float>(value_fp16[index]);
        }
        if (!backend.copy_from_host(
                key_fp16.data(), key_cache,
                key_fp16.size() * sizeof(key_fp16[0]),
                CacheMetadata::SIZE) ||
            !backend.copy_from_host(
                value_fp16.data(), value_cache,
                value_fp16.size() * sizeof(value_fp16[0]),
                CacheMetadata::SIZE))
            return false;
    } else if (!backend.copy_from_host(
                   cached_key.data(), key_cache,
                   cached_key.size() * sizeof(float), CacheMetadata::SIZE) ||
               !backend.copy_from_host(
                   cached_value.data(), value_cache,
                   cached_value.size() * sizeof(float),
                   CacheMetadata::SIZE)) {
        return false;
    }
    const auto round_for_cache = [fp16_cache](float value) {
        return fp16_cache
            ? static_cast<float>(mollm::cpu::fp16_t(value)) : value;
    };

    GraphNode sdpa;
    sdpa.op_type = OpType::SDPA;
    sdpa.params.i32 = {
        2, 1, heads, kv_heads, key_dim, value_dim};
    const float scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
    sdpa.params.f32 = {scale};
    backend.dispatch(
        sdpa, {&query, &key, &value, mask_input, &key_cache, &value_cache},
        &output, nullptr);
    backend.end_graph();
    if (backend.dispatch_failed())
        return false;

    std::vector<float> expected(
        static_cast<size_t>(heads) * current_length * value_dim, 0.0f);
    std::vector<float> scores(key_length);
    for (int head = 0; head < heads; ++head) {
        const int key_head = head / (heads / kv_heads);
        for (int query_position = 0; query_position < current_length;
             ++query_position) {
            float maximum = -INFINITY;
            for (int position = 0; position < key_length; ++position) {
                if (!explicit_mask &&
                    position > past_length + query_position) {
                    scores[position] = -INFINITY;
                    continue;
                }
                float dot = 0.0f;
                for (int dimension = 0; dimension < key_dim; ++dimension) {
                    const int current_position = position - past_length;
                    const float key_element = position < past_length
                        ? cached_key[
                              (static_cast<size_t>(key_head) * capacity +
                               position) * key_dim + dimension]
                        : round_for_cache(key_data[
                              (static_cast<size_t>(key_head) *
                                   current_length +
                               current_position) * key_dim + dimension]);
                    const size_t query_index =
                        (static_cast<size_t>(head) * current_length +
                         query_position) * key_dim + dimension;
                    dot += query_data[query_index] * key_element;
                }
                scores[position] = dot * scale;
                if (explicit_mask)
                    scores[position] += mask_data[
                        static_cast<size_t>(query_position) * key_length +
                        position];
                maximum = std::max(maximum, scores[position]);
            }
            float sum = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            for (int position = 0; position < key_length; ++position) {
                for (int dimension = 0; dimension < value_dim; ++dimension) {
                    const int current_position = position - past_length;
                    const float value_element = position < past_length
                        ? cached_value[
                              (static_cast<size_t>(key_head) * capacity +
                               position) * value_dim + dimension]
                        : round_for_cache(value_data[
                              (static_cast<size_t>(key_head) *
                                   current_length +
                               current_position) * value_dim + dimension]);
                    const size_t output_index =
                        (static_cast<size_t>(head) * current_length +
                         query_position) * value_dim + dimension;
                    expected[output_index] +=
                        scores[position] / sum * value_element;
                }
            }
        }
    }
    std::vector<float> actual;
    if (!download(backend, output, actual) ||
        !close_enough(actual, expected, 3e-5f))
        return false;

    std::vector<uint8_t> key_payload(
        cached_key.size() * cache_element_size);
    std::vector<uint8_t> value_payload(
        cached_value.size() * cache_element_size);
    if (!backend.copy_to_host(
            key_cache, key_payload.data(), key_payload.size(),
            CacheMetadata::SIZE) ||
        !backend.copy_to_host(
            value_cache, value_payload.data(), value_payload.size(),
            CacheMetadata::SIZE))
        return false;
    const auto payload_value = [fp16_cache](
        const std::vector<uint8_t>& payload, size_t index) {
        if (fp16_cache) {
            mollm::cpu::fp16_t value;
            std::memcpy(
                &value, payload.data() + index * sizeof(value),
                sizeof(value));
            return static_cast<float>(value);
        }
        float value;
        std::memcpy(
            &value, payload.data() + index * sizeof(value), sizeof(value));
        return value;
    };
    for (int key_head = 0; key_head < kv_heads; ++key_head) {
        for (int position = 0; position < current_length; ++position) {
            for (int dimension = 0; dimension < key_dim; ++dimension) {
                const size_t cache_index =
                    (static_cast<size_t>(key_head) * capacity + past_length +
                     position) * key_dim + dimension;
                const size_t source_index =
                    (static_cast<size_t>(key_head) * current_length +
                     position) * key_dim + dimension;
                if (std::fabs(payload_value(key_payload, cache_index) -
                              round_for_cache(key_data[source_index])) >
                    1e-6f)
                    return false;
            }
            for (int dimension = 0; dimension < value_dim; ++dimension) {
                const size_t cache_index =
                    (static_cast<size_t>(key_head) * capacity + past_length +
                     position) * value_dim + dimension;
                const size_t source_index =
                    (static_cast<size_t>(key_head) * current_length +
                     position) * value_dim + dimension;
                if (std::fabs(payload_value(value_payload, cache_index) -
                              round_for_cache(value_data[source_index])) >
                    1e-6f)
                    return false;
            }
        }
    }
    return true;
}

bool test_shortconv(CudaBackend& backend) {
    backend.clear_dispatch_error();
    constexpr int groups = 7;
    constexpr int sequence_length = 65;
    constexpr int kernel_size = 4;
    constexpr int real_tokens = 61;
    constexpr int row_padding = 3;
    constexpr int row_stride = groups + row_padding;

    Tensor input_storage =
        device_tensor(backend, row_stride, sequence_length);
    Tensor input = input_storage;
    input.shape[0] = groups;
    Tensor weight = device_tensor(backend, kernel_size, groups);
    Tensor state = device_tensor(backend, kernel_size - 1, groups);
    Tensor output = device_tensor(backend, groups, sequence_length);
    std::vector<float> input_storage_data(input_storage.nelements(), 99.0f);
    std::vector<float> input_data(
        static_cast<size_t>(groups) * sequence_length);
    std::vector<float> weight_data(weight.nelements());
    std::vector<float> state_data(state.nelements());
    for (int token = 0; token < sequence_length; ++token) {
        for (int group = 0; group < groups; ++group) {
            const float value =
                ((token * 7 + group * 3) % 17 - 8) / 19.0f;
            input_data[static_cast<size_t>(token) * groups + group] = value;
            input_storage_data[
                static_cast<size_t>(token) * row_stride + group] = value;
        }
    }
    for (size_t index = 0; index < weight_data.size(); ++index)
        weight_data[index] =
            (static_cast<int>(index % 13) - 6) / 23.0f;
    for (size_t index = 0; index < state_data.size(); ++index)
        state_data[index] =
            (static_cast<int>(index % 11) - 5) / 29.0f;
    if (!upload(backend, input_storage, input_storage_data) ||
        !upload(backend, weight, weight_data) ||
        !upload(backend, state, state_data))
        return false;

    const auto reference = [&](
        const std::vector<float>& source, int tokens, int process_tokens,
        std::vector<float>& state_reference,
        std::vector<float>& output_reference) {
        for (int group = 0; group < groups; ++group) {
            float window[kernel_size - 1];
            for (int index = 0; index < kernel_size - 1; ++index)
                window[index] = state_reference[
                    static_cast<size_t>(group) * (kernel_size - 1) + index];
            for (int token = 0; token < process_tokens; ++token) {
                const float value =
                    source[static_cast<size_t>(token) * groups + group];
                float sum = value * weight_data[
                    static_cast<size_t>(group) * kernel_size +
                    kernel_size - 1];
                for (int index = 0; index < kernel_size - 1; ++index)
                    sum += window[index] * weight_data[
                        static_cast<size_t>(group) * kernel_size + index];
                output_reference[
                    static_cast<size_t>(group) * tokens + token] =
                    sum / (1.0f + std::exp(-sum));
                for (int index = 0; index < kernel_size - 2; ++index)
                    window[index] = window[index + 1];
                window[kernel_size - 2] = value;
            }
            for (int index = 0; index < kernel_size - 1; ++index)
                state_reference[
                    static_cast<size_t>(group) * (kernel_size - 1) + index] =
                    window[index];
        }
    };

    std::vector<float> expected_state = state_data;
    std::vector<float> expected_output(
        static_cast<size_t>(groups) * sequence_length, 0.0f);
    reference(input_data, sequence_length, real_tokens, expected_state,
              expected_output);
    GraphNode shortconv;
    shortconv.op_type = OpType::SHORTCONV;
    shortconv.params.i32 = {kernel_size, real_tokens};
    backend.dispatch(
        shortconv, {&input, &weight, &state}, &output, nullptr);
    backend.end_graph();
    std::vector<float> actual_output;
    std::vector<float> actual_state;
    if (backend.dispatch_failed() ||
        !download(backend, output, actual_output) ||
        !download(backend, state, actual_state) ||
        !close_enough(actual_output, expected_output, 1e-6f) ||
        !close_enough(actual_state, expected_state, 1e-6f))
        return false;

    // Exercise the parallel prefill state commit when fewer real tokens than
    // history elements are present; the untouched tail must shift forward.
    expected_state = state_data;
    std::fill(expected_output.begin(), expected_output.end(), 0.0f);
    reference(input_data, sequence_length, 1, expected_state,
              expected_output);
    if (!upload(backend, state, state_data))
        return false;
    shortconv.params.i32 = {kernel_size, 1};
    backend.dispatch(
        shortconv, {&input, &weight, &state}, &output, nullptr);
    backend.end_graph();
    if (backend.dispatch_failed() ||
        !download(backend, output, actual_output) ||
        !download(backend, state, actual_state) ||
        !close_enough(actual_output, expected_output, 1e-6f) ||
        !close_enough(actual_state, expected_state, 1e-6f))
        return false;

    Tensor decode_input = device_tensor(backend, groups);
    Tensor decode_output = device_tensor(backend, groups);
    std::vector<float> decode_data(groups);
    for (int group = 0; group < groups; ++group)
        decode_data[group] = (group - 3) / 17.0f;
    if (!upload(backend, decode_input, decode_data))
        return false;
    std::vector<float> expected_decode(groups, 0.0f);
    reference(decode_data, 1, 1, expected_state, expected_decode);
    shortconv.params.i32 = {kernel_size, 1};
    backend.dispatch(
        shortconv, {&decode_input, &weight, &state}, &decode_output, nullptr);
    backend.end_graph();
    return !backend.dispatch_failed() &&
        download(backend, decode_output, actual_output) &&
        download(backend, state, actual_state) &&
        close_enough(actual_output, expected_decode, 1e-6f) &&
        close_enough(actual_state, expected_state, 1e-6f);
}

bool test_nvfp4_moe_provider() {
    int pageable = 0, host_tables = 0;
    cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0);
    cudaDeviceGetAttribute(&host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
    if (!pageable || !host_tables) return true;
    struct Provider final : ExpertProvider {
        ExpertSource gate_source, down_source;
        std::vector<uint8_t> gate_bytes[2], down_bytes[2], gs, ds;
        std::vector<float> row_scales;
        int pins = 0, borrows = 0;
        bool miss_nonblocking = false;
        bool resident_capability = false;
        Provider() : gs(64, 56), ds(32, 56), row_scales(32, 1.f / 32) {
            gate_source.provider = down_source.provider = this;
            for (int e = 0; e < 2; ++e) {
                gate_bytes[e].assign(512, (e + 1) * 0x11);
                down_bytes[e].assign(256, (e + 1) * 0x11);
            }
        }
        bool borrow(const ExpertSource*, const ExpertSource*, int expert,
                    ExpertLease& lease, bool wait) override {
            lease.reset();
            if (expert < 0 || expert >= 2) return false;
            if (!wait && miss_nonblocking && expert == 1) return false;
            Tensor gate = Tensor::create(Precision::NVFP4, MemoryType::EXTERNAL,
                32, 32, 1, 1, gate_bytes[expert].data());
            Tensor down = Tensor::create(Precision::NVFP4, MemoryType::EXTERNAL,
                32, 16, 1, 1, down_bytes[expert].data());
            gate.nvfp4_scales = gs.data(); down.nvfp4_scales = ds.data();
            gate.nvfp4_row_scales = down.nvfp4_row_scales = row_scales.data();
            gate.group_size = down.group_size = 16;
            gate.groups_per_row = 2; down.groups_per_row = 1;
            ++pins; ++borrows;
            lease = ExpertLease(*this, this, gate, down);
            return true;
        }
        bool request_many(const ExpertSource*, const ExpertSource*, const std::vector<int>&) override { return true; }
        bool borrow_resident_layer(const ExpertSource* gate, const ExpertSource* down,
                                   std::vector<ExpertLease>& leases) override {
            leases.clear();
            if (!resident_capability) return false;
            leases.resize(2);
            return borrow(gate, down, 0, leases[0], false) && borrow(gate, down, 1, leases[1], false);
        }
        bool retain_for_next_forward(const ExpertSource*, const ExpertSource*, const std::vector<int>&, bool) override { return true; }
        size_t resident_count(const ExpertSource*, const ExpertSource*, const std::vector<int>&) const override { return 2; }
        bool contains(const ExpertSource*, const ExpertSource*, int) const override { return true; }
        bool is_ready(const ExpertSource*, const ExpertSource*, int) const override { return true; }
        bool evict(const ExpertSource*, const ExpertSource*, int) override { return false; }
        void release_lease(void*) noexcept override { --pins; }
    } provider;
    CudaBackend backend;
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    std::vector<float> router_data(64, 0), shared_g(512, 1.f/32), shared_u(512, 1.f/16);
    std::vector<float> shared_d(512, 1.f/32), shared_scale(32, 0);
    auto weight = [&](std::vector<float>& data, int n, int k) {
        Tensor t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, n, k, 1, 1, data.data());
        backend.wrap_weight(t);
        return t;
    };
    Tensor router = weight(router_data, 2, 32), sg = weight(shared_g, 16, 32);
    Tensor su = weight(shared_u, 16, 32), sd = weight(shared_d, 32, 16);
    Tensor scale = weight(shared_scale, 1, 32);
    Tensor gate = Tensor::create(Precision::NVFP4, MemoryType::EXTERNAL, 32, 32, 2);
    Tensor down = Tensor::create(Precision::NVFP4, MemoryType::EXTERNAL, 32, 16, 2);
    gate.moe_ssd_source = &provider.gate_source;
    down.moe_ssd_source = &provider.down_source;
    for (int rows : {1, 3}) {
        std::vector<float> input(rows * 32), expected(rows * 32), actual;
        for (int row = 0; row < rows; ++row)
            std::fill(input.begin() + row * 32, input.begin() + (row + 1) * 32, (row + 1) * .25f);
        Tensor host_x = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, rows, 1, 1, input.data());
        Tensor host_y = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, rows, 1, 1, expected.data());
        GraphNode node;
        node.op_type = OpType::MOE;
        node.params.i32 = {32, 2, 2, 16, 16, 0, 1, 1, 1, 1, 1};
        CPUBackend cpu;
        cpu.dispatch(node, {&host_x, &router, &gate, &down, &sg, &su, &sd, &scale}, &host_y, nullptr);
        Tensor x = device_tensor(backend, 32, rows), y = device_tensor(backend, 32, rows);
        if (!upload(backend, x, input)) return false;
        const int before = provider.borrows;
        backend.dispatch(node, {&x, &router, &gate, &down, &sg, &su, &sd, &scale}, &y, nullptr);
        if (cpu.dispatch_failed() || backend.dispatch_failed() || provider.pins != 0 ||
            provider.borrows - before != rows * 2 || !download(backend, y, actual) ||
            !close_enough(actual, expected, 1e-6f)) return false;
        // A partial nonblocking acquisition must release its pins before
        // entering the CPU bridge; reference execution may then wait for I/O.
        provider.miss_nonblocking = true;
        backend.set_operator_fallback_policy(OperatorFallbackPolicy::ALLOW_REFERENCE);
        backend.dispatch(node, {&x, &router, &gate, &down, &sg, &su, &sd, &scale}, &y, nullptr);
        if (backend.dispatch_failed() || provider.pins != 0 ||
            !download(backend, y, actual) ||
            !close_enough(actual, expected, 1e-6f)) return false;
        provider.miss_nonblocking = false;
        backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    }
    provider.resident_capability = true;
    {
        CudaBackend resident_backend;
        resident_backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
        for (Tensor* t : {&router, &sg, &su, &sd, &scale}) resident_backend.wrap_weight(*t);
        constexpr int resident_rows[] = {1, 2, 3, 4, 5, 7, 8};
        for (int pass = 0; pass < 21; ++pass) {
            const int rows = resident_rows[pass % 7];
            std::vector<float> input(rows * 32, .25f + pass / 32.f), expected(rows * 32), actual;
            Tensor host_x = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, rows, 1, 1, input.data());
            Tensor host_y = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, rows, 1, 1, expected.data());
            GraphNode node;
            node.op_type = OpType::MOE;
            node.params.i32 = {32, 2, 2, 16, 16, pass / 7, 1, 1, 1, 1, 1};
            CPUBackend cpu;
            cpu.dispatch(node, {&host_x, &router, &gate, &down, &sg, &su, &sd, &scale}, &host_y, nullptr);
            Tensor x = device_tensor(resident_backend, 32, rows), y = device_tensor(resident_backend, 32, rows);
            if (!upload(resident_backend, x, input)) return false;
            const int before = provider.borrows;
            resident_backend.dispatch(node, {&x, &router, &gate, &down, &sg, &su, &sd, &scale}, &y, nullptr);
            if (cpu.dispatch_failed() || resident_backend.dispatch_failed() || provider.pins != 2 ||
                provider.borrows - before != (pass == 0 ? 2 : 0) ||
                !download(resident_backend, y, actual) || !close_enough(actual, expected, 1e-6f)) return false;
        }
    }
    if (provider.pins != 0) return false;
    {
        // Q4 router/shared weights make decode eligible for graph replay.
        // Reuse pointers, grow scratch with prefill, then change routing
        // parameters: every cached graph must bind the current invocation.
        const char* previous_limit = std::getenv("MOLLM_CUDA_OUTPUT_POOL_MB");
        struct RestoreGraphPoolLimit {
            bool had; std::string value;
            ~RestoreGraphPoolLimit() {
                if (had) setenv("MOLLM_CUDA_OUTPUT_POOL_MB", value.c_str(), 1);
                else unsetenv("MOLLM_CUDA_OUTPUT_POOL_MB");
            }
        } restore_limit{previous_limit != nullptr, previous_limit ? previous_limit : ""};
        setenv("MOLLM_CUDA_OUTPUT_POOL_MB", "1", 1);
        CudaBackend graph_backend;
        graph_backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
        std::vector<Q4B8G32Block> rb(1), gb(4), ub(4), db(4), sb(1);
        const auto q4_weight = [&](std::vector<Q4B8G32Block>& blocks, int n, float s) {
            for (auto& b : blocks) for (int row = 0; row < 8; ++row) {
                b.scales[row] = s;
                std::fill(std::begin(b.q[row]), std::end(b.q[row]), uint8_t{0x11});
            }
            Tensor t = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, n, 32, 1, 1, blocks.data());
            t.is_q4_g32_packed = true; t.q4_g32_data = blocks.data();
            t.group_size = 32; t.groups_per_row = 1;
            graph_backend.wrap_weight_int4(t);
            return t;
        };
        Tensor qr = q4_weight(rb, 2, 0), qg = q4_weight(gb, 32, 1.f/32);
        Tensor qu = q4_weight(ub, 32, 1.f/16), qd = q4_weight(db, 32, 1.f/32), qs = q4_weight(sb, 1, 0);
        std::vector<float> hg(1024, 1.f/32), hu(1024, 1.f/16), hd(1024, 1.f/32);
        Tensor fg = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, 32, 1, 1, hg.data());
        Tensor fu = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, 32, 1, 1, hu.data());
        Tensor fd = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, 32, 1, 1, hd.data());
        Tensor x1 = device_tensor(graph_backend, 32), y1 = device_tensor(graph_backend, 32);
        Tensor x3 = device_tensor(graph_backend, 32, 3), y3 = device_tensor(graph_backend, 32, 3);
        for (int pass = 0; pass < 6; ++pass) {
            const int rows = pass == 2 ? 3 : 1;
            std::vector<float> input(rows * 32, .25f + pass / 32.f), expected(rows * 32), actual;
            Tensor hx = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, rows, 1, 1, input.data());
            Tensor hy = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, rows, 1, 1, expected.data());
            GraphNode node; node.op_type = OpType::MOE;
            node.params.i32 = {32, 2, 2, 16, 32, pass >= 4 ? 1 : 0, 1, 1, 1, 1, 1};
            CPUBackend cpu;
            cpu.dispatch(node, {&hx, &router, &gate, &down, &fg, &fu, &fd, &scale}, &hy, nullptr);
            Tensor& x = rows == 1 ? x1 : x3;
            Tensor& y = rows == 1 ? y1 : y3;
            if (!upload(graph_backend, x, input)) return false;
            graph_backend.dispatch(node, {&x, &qr, &gate, &down, &qg, &qu, &qd, &qs}, &y, nullptr);
            if (cpu.dispatch_failed() || graph_backend.dispatch_failed() || provider.pins != 2 ||
                !download(graph_backend, y, actual) || !close_enough(actual, expected, 1e-6f)) return false;
        }
        // Exercise segment capture/replay, external BF16 rounding, a late
        // parameter mismatch after a skipped prefix, and scratch growth from
        // interleaved prefill. A zero-real-token PLE lookup is a valid CPU
        // boundary without allocating a model-sized embedding table.
        if(std::getenv("MOLLM_CUDA_MOE_PREFILL_INPUT_HALF_REUSE")) {
            setenv("MOLLM_CUDA_OUTPUT_POOL_MB","8",1);
            CudaBackend scoped_backend;
            setenv("MOLLM_CUDA_OUTPUT_POOL_MB","1",1);
            scoped_backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
            Tensor sr=qr,sgate=qg,sup=qu,sdown=qd,sscale=qs;
            for(Tensor* t:{&sr,&sgate,&sup,&sdown,&sscale})scoped_backend.wrap_weight_int4(*t);
            // One scoped shadow must follow changed shapes/values, and must
            // not survive into the small-row/decode fallback. This is only a
            // tiny two-expert native fixture, not a model-sized graph window.
            for(int rows:{15,16,17,65,1024,2048,2049,17,1}) {
                Tensor tx=device_tensor(scoped_backend,32,rows),ty=device_tensor(scoped_backend,32,rows);
                for(int variant=0;variant<3;++variant) {
                    std::vector<float> input(rows*32,.25f+variant/32.f),expected(rows*32),actual;
                    Tensor hx=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,32,rows,1,1,input.data());
                    Tensor hy=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,32,rows,1,1,expected.data());
                    GraphNode node;node.op_type=OpType::MOE;
                    node.params.i32={32,2,2,16,32,variant%2,1,1,1,1,1};
                    CPUBackend cpu;
                    cpu.dispatch(node,{&hx,&router,&gate,&down,&fg,&fu,&fd,&scale},&hy,nullptr);
                    if(!upload(scoped_backend,tx,input))return false;
                    scoped_backend.dispatch(node,{&tx,&sr,&gate,&down,&sgate,&sup,&sdown,&sscale},&ty,nullptr);
                    if(cpu.dispatch_failed()||scoped_backend.dispatch_failed()||
                       !download(scoped_backend,ty,actual)||!close_enough(actual,expected,1e-6f))return false;
                }
                scoped_backend.free_output(tx,nullptr);scoped_backend.free_output(ty,nullptr);
            }
        }
        Tensor tiled = device_tensor(graph_backend, 32), activated = device_tensor(graph_backend, 32);
        Tensor ple_output = device_tensor(graph_backend, 32);
        int32_t token = 0, history_data[2] = {}, vocabulary = 2, offset = 0;
        uint8_t table_data[32] = {};
        float table_scale = 1;
        Tensor ids = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 1, 1, 1, 1, &token);
        Tensor history = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 2, 1, 1, 1, history_data);
        Tensor table = Tensor::create(Precision::RAW_U8, MemoryType::EXTERNAL, 32, 1, 1, 1, table_data);
        Tensor table_scales = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1, &table_scale);
        Tensor vocabularies = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 1, 1, 1, 1, &vocabulary);
        Tensor offsets = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 1, 1, 1, 1, &offset);
        GraphNode tile, lookup, sigmoid, segment_moe;
        tile.op_type = OpType::TILE; tile.params.i32 = {1, 1, 1, 1};
        lookup.op_type = OpType::PLE_LOOKUP; lookup.params.i32 = {2, 1, 0, 2, 0, 1234, 0, 0};
        sigmoid.op_type = OpType::SIGMOID;
        segment_moe.op_type = OpType::MOE;
        for (int pass = 0; pass < 12; ++pass) {
            if (pass == 9) {
                // Force automatic limit-triggered reclamation while resident
                // and segment caches from the preceding passes exist.
                Tensor cached = device_tensor(graph_backend, 768 * 1024 / sizeof(float));
                if (!cached.data) return false;
                graph_backend.free_output(cached, nullptr);
                Tensor larger = device_tensor(graph_backend, 896 * 1024 / sizeof(float));
                if (!larger.data || graph_backend.dispatch_failed()) return false;
                graph_backend.free_output(larger, nullptr);
                Tensor large_x = device_tensor(graph_backend, 32, 33), large_y = device_tensor(graph_backend, 32, 33);
                std::vector<float> values(32 * 33, .25f);
                if (!upload(graph_backend, large_x, values)) return false;
                GraphNode prefill; prefill.op_type = OpType::MOE;
                prefill.params.i32 = {32, 2, 2, 16, 32, 1, 1, 1, 1, 1, 1};
                graph_backend.begin_graph();
                graph_backend.dispatch(prefill, {&large_x, &qr, &gate, &down, &qg, &qu, &qd, &qs}, &large_y, nullptr);
                graph_backend.end_graph();
                if (graph_backend.dispatch_failed()) return false;
            }
            std::vector<float> input(32, .25f + pass / 32.f), transformed(32), expected(32), actual;
            for (int i = 0; i < 32; ++i) transformed[i] = 1.f / (1.f + std::exp(-input[i]));
            Tensor hx = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, 1, 1, 1, transformed.data());
            Tensor hy = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 32, 1, 1, 1, expected.data());
            segment_moe.params.i32 = {32, 2, 2, 16, 32, pass >= 3 ? 1 : 0, 1, 1, 1, 1, 1};
            segment_moe.params.f32 = {pass >= 5 ? .75f : 1.f};
            CPUBackend cpu;
            cpu.dispatch(segment_moe, {&hx, &router, &gate, &down, &fg, &fu, &fd, &scale}, &hy, nullptr);
            for (float& value : expected) value = mollm_round_to_bf16(value);
            if (!upload(graph_backend, x1, input)) return false;
            graph_backend.begin_graph();
            graph_backend.dispatch(tile, {&x1}, &tiled, nullptr);
            graph_backend.set_operator_fallback_policy(OperatorFallbackPolicy::ALLOW_REFERENCE);
            graph_backend.dispatch(lookup, {&ids, &history, &table, &table_scales, &vocabularies, &offsets}, &ple_output, nullptr);
            graph_backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
            graph_backend.dispatch(sigmoid, {&tiled}, &activated, nullptr);
            graph_backend.dispatch(segment_moe, {&activated, &qr, &gate, &down, &qg, &qu, &qd, &qs}, &y1, nullptr);
            if (!graph_backend.round_to_bf16(y1)) return false;
            graph_backend.end_graph();
            if (cpu.dispatch_failed() || graph_backend.dispatch_failed() ||
                !download(graph_backend, y1, actual) || !close_enough(actual, expected, 0.f)) return false;
            if (pass == 7) {
                // Both resident MoE and (when enabled) segment graph caches
                // already exist. Preserve all live operands and invalidate
                // cached executables before releasing this returned block.
                Tensor cached = device_tensor(graph_backend, 2048);
                if (!cached.data) return false;
                graph_backend.free_output(cached, nullptr);
                size_t reclaimed = 0, invalidated = 0;
                if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
                const bool rejected = !graph_backend.trim_cached_outputs(&reclaimed, &invalidated) &&
                    reclaimed == 0 && invalidated == 0;
                cudaGraph_t capture = nullptr;
                const cudaError_t ended = cudaStreamEndCapture(cudaStreamPerThread, &capture);
                if (capture) cudaGraphDestroy(capture);
                if (!rejected || ended != cudaSuccess) return false;
                if (!graph_backend.trim_cached_outputs(&reclaimed, &invalidated) || reclaimed < 8192) return false;
                std::vector<float> survived;
                if (!download(graph_backend, y1, survived) || survived.size() != actual.size() ||
                    std::memcmp(survived.data(), actual.data(), actual.size() * sizeof(float))) return false;
                std::printf("cuda_backend_output_trim reclaimed=%zu invalidated_graphs=%zu live_output_exact=1 capture_rejected=1\n",
                    reclaimed, invalidated);
                // Later changed-input passes must rebuild caches and still
                // match the complete CPU output, including BF16 rounding.
            }
        }
    }
    if (provider.pins != 0) return false;
    return true;
}

bool test_ple_device_ops(CudaBackend& backend) {
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    for (int hidden : {7, 2560}) for (int rows : {1, 5}) {
        constexpr int streams = 4;
        const int wide = hidden * streams;
        std::vector<float> q(wide * rows), k(wide * rows), v(hidden * rows), expected(wide * rows), actual;
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = (static_cast<int>(i % 43) - 21) / 16.f;
            k[i] = (static_cast<int>((i * 17) % 59) - 29) / 32.f;
        }
        for (size_t i = 0; i < v.size(); ++i) v[i] = (static_cast<int>(i % 23) - 11) / 32.f;
        Tensor hq = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, wide, rows, 1, 1, q.data());
        Tensor hk = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, wide, rows, 1, 1, k.data());
        Tensor hv = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hidden, rows, 1, 1, v.data());
        Tensor hy = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, wide, rows, 1, 1, expected.data());
        GraphNode node; node.op_type = OpType::PLE_GATE; node.params.i32 = {hidden, streams};
        CPUBackend cpu; cpu.dispatch(node, {&hq, &hk, &hv}, &hy, nullptr);
        Tensor dq = device_tensor(backend, wide, rows), dk = device_tensor(backend, wide, rows);
        Tensor dv = device_tensor(backend, hidden, rows), dy = device_tensor(backend, wide, rows);
        if (!upload(backend, dq, q) || !upload(backend, dk, k) || !upload(backend, dv, v)) return false;
        backend.dispatch(node, {&dq, &dk, &dv}, &dy, nullptr);
        if (cpu.dispatch_failed() || backend.dispatch_failed() || !download(backend, dy, actual) ||
            !close_enough(actual, expected, 3e-6f)) return false;
    }
    for (int channels : {7, 10240}) for (int kernel : {2, 4}) for (int dilation : {1, 3}) {
        const int history_len = (kernel - 1) * dilation;
        std::vector<float> w(channels * kernel), history(channels * history_len), actual, actual_history;
        for (size_t i = 0; i < w.size(); ++i) w[i] = (static_cast<int>(i % 17) - 8) / 32.f;
        for (size_t i = 0; i < history.size(); ++i) history[i] = (static_cast<int>(i % 13) - 6) / 16.f;
        Tensor hw = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, channels, kernel, 1, 1, w.data());
        Tensor hs = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, channels, history_len, 1, 1, history.data());
        Tensor dw = device_tensor(backend, channels, kernel), ds = device_tensor(backend, channels, history_len);
        if (!upload(backend, dw, w) || !upload(backend, ds, history)) return false;
        // Keep state across padded prefill, decode, and zero-token forwards.
        for (int rows : {17, 1, 3}) for (int real : {0, 1, rows - 1, -1, rows + 3}) {
            std::vector<float> x(channels * rows), expected(channels * rows);
            for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>((i * 7 + rows) % 31) - 15) / 32.f;
            Tensor hx = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, channels, rows, 1, 1, x.data());
            Tensor hy = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, channels, rows, 1, 1, expected.data());
            GraphNode node; node.op_type = OpType::PLE_DILATED_CONV; node.params.i32 = {kernel, dilation, real};
            const char* prefix_flag=std::getenv("MOLLM_CUDA_MTP_STATE_PREFIXES");
            const bool prefix_test=prefix_flag && !std::strcmp(prefix_flag,"1") && rows==3 && real==-1;
            const auto initial_history=history;
            if(prefix_test && !backend.begin_state_prefixes({&ds},rows))return false;
            CPUBackend cpu; cpu.dispatch(node, {&hx, &hw, &hs}, &hy, nullptr);
            Tensor dx = device_tensor(backend, channels, rows), dy = device_tensor(backend, channels, rows);
            if (!upload(backend, dx, x)) return false;
            backend.dispatch(node, {&dx, &dw, &ds}, &dy, nullptr);
            if (cpu.dispatch_failed() || backend.dispatch_failed() || !download(backend, dy, actual) ||
                !download(backend, ds, actual_history) || !close_enough(actual, expected, 1e-6f) ||
                !close_enough(actual_history, history, 0.f)) return false;
            if(prefix_test) {
                for(int kept=1;kept<=rows;++kept) {
                    if(!backend.restore_state_prefix(kept)||!download(backend,ds,actual_history))return false;
                    for(int channel=0;channel<channels;++channel)for(int j=0;j<history_len;++j) {
                        const int source=j+kept;
                        const float value=source<history_len?initial_history[size_t(channel)*history_len+source]
                            :x[size_t(source-history_len)*channels+channel];
                        if(actual_history[size_t(channel)*history_len+j]!=value)return false;
                    }
                }
                backend.end_state_prefixes();
            }
        }
    }
    return true;
}

bool test_resident_routes(CudaBackend& backend) {
    using mollm_cuda::Nvfp4Projection;
    const bool local16_probe=std::getenv("MOLLM_ROUTE_LOCAL16_PROBE")!=nullptr;
    for (int experts : {1, 7, 31, 32, 33, 511, 512, 513, 1023, 1024}) {
        for (int top_k : {1, std::min(10, experts), std::min(32, experts)}) {
          for(int rows : local16_probe?std::vector<int>{1,3}:std::vector<int>{3}) {
            constexpr int hidden = 32, inter = 16;
            const int routes = rows * top_k;
            std::vector<uint8_t> identities(2 * experts);
            std::vector<Nvfp4Projection> tables(2 * experts), jobs(2 * routes);
            for (int i = 0; i < 2 * experts; ++i)
                tables[i] = {&identities[i], &identities[i], nullptr, nullptr, nullptr, i % 2 != 0};
            Tensor dt = device_tensor(backend, 2 * experts * sizeof(Nvfp4Projection) / sizeof(float));
            Tensor dj = device_tensor(backend, 2 * routes * sizeof(Nvfp4Projection) / sizeof(float));
            Tensor logits = device_tensor(backend, experts, rows);
            Tensor weights = device_tensor(backend, top_k, rows);
            Tensor input = device_tensor(backend, hidden, rows);
            Tensor gate = device_tensor(backend, 2 * inter, routes);
            Tensor activated = device_tensor(backend, inter, routes);
            Tensor down = device_tensor(backend, hidden, routes);
            std::vector<float> data(rows * experts), actual;
            for (int row = 0; row < rows; ++row)
                for (int i = 0; i < experts; ++i)
                    data[row * experts + i] = row == 0 && rows!=1 ? 0.f :
                        (static_cast<int>((i * 37 + row * 11) % 97) - 48) / 8.f;
            if (!upload(backend, logits, data) ||
                cudaMemcpy(dt.device.buffer, tables.data(), dt.nbytes(), cudaMemcpyHostToDevice) != cudaSuccess)
                return false;
            for (int score_func : {0, 1, 2}) for (bool normalize : {false, true})
            for (int cache_mode : local16_probe?std::vector<int>{-1,0,1,2}:std::vector<int>{0,1}) {
                mollm::detail::MoeRoutingParams params;
                params.num_experts = experts; params.top_k = top_k;
                params.score_func = score_func; params.normalize_topk = normalize;
                params.scaling_factor = 1.25f;
                std::vector<int> ids;
                std::vector<float> expected;
                if (!mollm::detail::select_moe_routes(data.data(), rows, nullptr, params, ids, expected)) return false;
                for (int row = 0; row < rows; ++row)
                    for (int k = 1; k < top_k; ++k)
                        for (int j = k; j > 0 && ids[row * top_k + j] < ids[row * top_k + j - 1]; --j) {
                            std::swap(ids[row * top_k + j], ids[row * top_k + j - 1]);
                            std::swap(expected[row * top_k + j], expected[row * top_k + j - 1]);
                        }
                auto* gate_jobs = static_cast<Nvfp4Projection*>(dj.device.buffer);
                mollm_cuda::launch_moe_resident_routes(static_cast<const float*>(logits.device.buffer),
                    rows, experts, top_k, score_func, normalize, params.scaling_factor,
                    static_cast<const Nvfp4Projection*>(dt.device.buffer),
                    static_cast<const float*>(input.device.buffer), static_cast<float*>(gate.device.buffer),
                    static_cast<float*>(activated.device.buffer), static_cast<float*>(down.device.buffer),
                    hidden, inter, gate_jobs, gate_jobs + routes, static_cast<float*>(weights.device.buffer),nullptr,cache_mode);
                if (!download(backend, weights, actual) || !close_enough_relative(actual, expected, 1e-6f, 1e-6f) ||
                    cudaMemcpy(jobs.data(), dj.device.buffer, dj.nbytes(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
                for (int i = 0; i < routes; ++i) {
                    if (jobs[i].weight != tables[ids[i]].weight || jobs[i].pair_layout != tables[ids[i]].pair_layout ||
                        jobs[i + routes].weight != tables[experts + ids[i]].weight ||
                        jobs[i].input != static_cast<float*>(input.device.buffer) + (i / top_k) * hidden ||
                        jobs[i].output != static_cast<float*>(gate.device.buffer) + i * 2 * inter ||
                        jobs[i + routes].input != static_cast<float*>(activated.device.buffer) + i * inter ||
                        jobs[i + routes].output != static_cast<float*>(down.device.buffer) + i * hidden) return false;
                }
                if(local16_probe && (cache_mode==2 || cache_mode==-1)) {
                    cudaStream_t replay_stream;
                    if(cudaStreamCreateWithFlags(&replay_stream,cudaStreamNonBlocking)!=cudaSuccess)return false;
                    auto launch=[&](int mode) {
                        mollm_cuda::launch_moe_resident_routes(static_cast<const float*>(logits.device.buffer),
                            rows,experts,top_k,score_func,normalize,params.scaling_factor,
                            static_cast<const Nvfp4Projection*>(dt.device.buffer),static_cast<const float*>(input.device.buffer),
                            static_cast<float*>(gate.device.buffer),static_cast<float*>(activated.device.buffer),
                            static_cast<float*>(down.device.buffer),hidden,inter,gate_jobs,gate_jobs+routes,
                            static_cast<float*>(weights.device.buffer),replay_stream,mode);
                    };
                    cudaGraph_t graph;cudaGraphExec_t executable;
                    if(cudaStreamBeginCapture(replay_stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return false;
                    launch(cache_mode);
                    if(cudaStreamEndCapture(replay_stream,&graph)!=cudaSuccess ||
                        cudaGraphInstantiateWithFlags(&executable,graph,0)!=cudaSuccess)return false;
                    std::vector<Nvfp4Projection> reference_jobs(2*routes),replayed_jobs(2*routes);
                    std::vector<float> changed(data.size()),reference_weights(routes),replayed_weights(routes);
                    for(int replay=0;replay<6;++replay) {
                        for(size_t i=0;i<changed.size();++i) {
                            if(replay==0)changed[i]=(int((i*19+7)%61)-30)*.125f;
                            else if(replay==1)changed[i]=.25f;
                            else if(replay==2)changed[i]=i%2?-0.f:0.f;
                            else if(replay==3)changed[i]=i%2?-1e20f:1e20f;
                            else if(replay==4)changed[i]=i%3==0?NAN:i%3==1?INFINITY:-INFINITY;
                            else changed[i]=NAN;
                        }
                        if(cudaMemcpyAsync(logits.device.buffer,changed.data(),logits.nbytes(),cudaMemcpyHostToDevice,replay_stream)!=cudaSuccess ||
                            cudaMemsetAsync(dj.device.buffer,0xa5,dj.nbytes(),replay_stream)!=cudaSuccess)return false;
                        launch(0);
                        if(cudaMemcpyAsync(reference_jobs.data(),dj.device.buffer,dj.nbytes(),cudaMemcpyDeviceToHost,replay_stream)!=cudaSuccess ||
                            cudaMemcpyAsync(reference_weights.data(),weights.device.buffer,weights.nbytes(),cudaMemcpyDeviceToHost,replay_stream)!=cudaSuccess ||
                            cudaStreamSynchronize(replay_stream)!=cudaSuccess ||
                            cudaMemsetAsync(dj.device.buffer,0xa5,dj.nbytes(),replay_stream)!=cudaSuccess ||
                            cudaGraphLaunch(executable,replay_stream)!=cudaSuccess ||
                            cudaMemcpyAsync(replayed_jobs.data(),dj.device.buffer,dj.nbytes(),cudaMemcpyDeviceToHost,replay_stream)!=cudaSuccess ||
                            cudaMemcpyAsync(replayed_weights.data(),weights.device.buffer,weights.nbytes(),cudaMemcpyDeviceToHost,replay_stream)!=cudaSuccess ||
                            cudaStreamSynchronize(replay_stream)!=cudaSuccess)return false;
                        if(std::memcmp(reference_jobs.data(),replayed_jobs.data(),dj.nbytes()) ||
                            std::memcmp(reference_weights.data(),replayed_weights.data(),weights.nbytes())) {
                            std::fprintf(stderr,"local16 route replay mismatch rows=%d experts=%d top_k=%d score=%d normalize=%d replay=%d\n",
                                rows,experts,top_k,score_func,int(normalize),replay);return false;
                        }
                    }
                    if(cudaGraphExecDestroy(executable)!=cudaSuccess || cudaGraphDestroy(graph)!=cudaSuccess ||
                        cudaStreamDestroy(replay_stream)!=cudaSuccess || !upload(backend,logits,data))return false;
                }
                if (std::getenv("MOLLM_ROUTE_BENCH") && cache_mode==0 && experts==512 && top_k==10 && score_func==1 && normalize) {
                    auto require_cuda = [](cudaError_t status) {
                        if (status != cudaSuccess) {
                            std::fprintf(stderr,"route benchmark CUDA failure: %s\n",cudaGetErrorString(status));
                            std::abort();
                        }
                    };
                    cudaStream_t stream; if(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess)return false;
                    auto time = [&](int mode) {
                        cudaGraph_t graph; cudaGraphExec_t executable;
                        require_cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                        for(int iteration=0;iteration<128;++iteration)
                            mollm_cuda::launch_moe_resident_routes(static_cast<const float*>(logits.device.buffer),
                                rows,experts,top_k,score_func,normalize,params.scaling_factor,
                                static_cast<const Nvfp4Projection*>(dt.device.buffer),static_cast<const float*>(input.device.buffer),
                                static_cast<float*>(gate.device.buffer),static_cast<float*>(activated.device.buffer),
                                static_cast<float*>(down.device.buffer),hidden,inter,gate_jobs,gate_jobs+routes,
                                static_cast<float*>(weights.device.buffer),stream,mode);
                        require_cuda(cudaStreamEndCapture(stream,&graph));require_cuda(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
                        for(int warm=0;warm<2;++warm)require_cuda(cudaGraphLaunch(executable,stream));
                        cudaEvent_t start,stop;require_cuda(cudaEventCreate(&start));require_cuda(cudaEventCreate(&stop));
                        require_cuda(cudaEventRecord(start,stream));for(int replay=0;replay<4;++replay)require_cuda(cudaGraphLaunch(executable,stream));
                        require_cuda(cudaEventRecord(stop,stream));require_cuda(cudaEventSynchronize(stop));float ms=0;require_cuda(cudaEventElapsedTime(&ms,start,stop));
                        std::vector<float> measured;
                        if (!download(backend,weights,measured) || measured.size()!=actual.size() ||
                            std::memcmp(measured.data(),actual.data(),actual.size()*sizeof(float))) ms=-1;
                        require_cuda(cudaEventDestroy(start));require_cuda(cudaEventDestroy(stop));require_cuda(cudaGraphExecDestroy(executable));require_cuda(cudaGraphDestroy(graph));
                        return ms*1000./512;
                    };
                    for(int repeat=0;repeat<3;++repeat) {
                        double original,cached;
                        const int candidate_mode=local16_probe?2:1;
                        if(repeat%2){cached=time(candidate_mode);original=time(0);}else{original=time(0);cached=time(candidate_mode);}
                        if(original<0||cached<0)return false;
                        std::printf("route_cached_best rows=%d repeat=%d original_us=%.3f cached_us=%.3f bit_exact=yes candidate_mode=%d\n",rows,repeat,original,cached,candidate_mode);
                    }
                    cudaStreamDestroy(stream);
                }
            }
          }
        }
    }
    return true;
}

bool test_nvfp4_system_memory(CudaBackend& backend) {
    int pageable = 0, host_tables = 0;
    if (cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0) != cudaSuccess ||
        cudaDeviceGetAttribute(&host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0) != cudaSuccess)
        return false;
    if (!pageable || !host_tables) {
        std::printf("SKIP: direct system-memory NVFP4 requires coherent host memory\n");
        return true;
    }
    {
        constexpr int rows = 3, width = 17, top_k = 2;
        std::vector<float> gate_up(rows * top_k * width * 2);
        std::vector<float> weights(rows * top_k);
        std::vector<float> expected(rows * top_k * width), actual;
        for (size_t i = 0; i < gate_up.size(); ++i)
            gate_up[i] = (static_cast<int>(i % 29) - 14) / 8.f;
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] = (i + 1) / 16.f;
        for (int row = 0; row < rows * top_k; ++row) {
            for (int col = 0; col < width; ++col) {
                const float gate = mollm_round_to_bf16(gate_up[row * 2 * width + col]);
                const float up = mollm_round_to_bf16(gate_up[row * 2 * width + width + col]);
                expected[row * width + col] = mollm_round_to_bf16(
                    (gate * (1.f / (1.f + std::exp(-gate)))) * up * weights[row]);
            }
        }
        Tensor activated = device_tensor(backend, width, rows * top_k);
        mollm_cuda::launch_moe_activation(gate_up.data(), gate_up.data() + width,
            weights.data(), static_cast<float*>(activated.device.buffer),
            rows * top_k, width, 2 * width);
        if (!download(backend, activated, actual) ||
            !close_enough(actual, expected, 0.0f)) return false;
        std::vector<float> shared(rows * width), shared_gate(rows);
        for (size_t i = 0; i < shared.size(); ++i) shared[i] = (i % 7) / 31.f;
        for (int i = 0; i < rows; ++i) shared_gate[i] = (i - 1) / 3.f;
        std::vector<float> combined(rows * width);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < width; ++col) {
                float sum = 0;
                for (int route = 0; route < top_k; ++route)
                    sum += mollm_round_to_bf16(expected[(row * top_k + route) * width + col]);
                const float contribution = mollm_round_to_bf16(shared[row * width + col]) *
                    (1.f / (1.f + std::exp(-mollm_round_to_bf16(shared_gate[row]))));
                combined[row * width + col] = sum + contribution;
            }
        }
        Tensor output = device_tensor(backend, width, rows);
        mollm_cuda::launch_moe_combine(expected.data(), shared.data(), shared_gate.data(),
            static_cast<float*>(output.device.buffer), rows, width, top_k);
        if (!download(backend, output, actual) ||
            !close_enough(actual, combined, 1e-6f)) return false;
    }
    constexpr float values[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6,
                                  0, -.5f, -1, -1.5f, -2, -3, -4, -6};
    constexpr uint8_t scale_codes[] = {0, 1, 7, 8, 48, 56, 64, 126};
    constexpr float scale_values[] = {0, 1.f/512, 7.f/512, 1.f/64, .5f, 1, 2, 448};
    for (int rows : {7, 12}) {
        for (int inner : {16, 32, 80, 640, 2560}) {
            const int groups = inner / 16;
            std::vector<float> activation(inner), row_scales(rows), expected(rows);
            std::vector<uint8_t> row_major(rows * inner / 2), pair_layout(
                ((rows + 3) / 4) * 4 * inner / 2), scales(rows * groups);
            for (int k = 0; k < inner; ++k)
                activation[k] = (k % 19 - 9) / 16.f;
            for (int row = 0; row < rows; ++row) {
                row_scales[row] = (row + 1) / 1024.f;
                double sum = 0;
                for (int group = 0; group < groups; ++group) {
                    const int scale_index = (row + group) % 8;
                    scales[row * groups + group] = scale_codes[scale_index];
                    for (int p = 0; p < 8; ++p) {
                        const uint8_t packed = static_cast<uint8_t>(row * 29 + group * 7 + p * 13);
                        row_major[row * (inner / 2) + group * 8 + p] = packed;
                        pair_layout[((row / 4) * groups + group) * 32 + (row % 4) * 8 + p] = packed;
                        sum += activation[group * 16 + p * 2] *
                            values[packed & 15] * scale_values[scale_index];
                        sum += activation[group * 16 + p * 2 + 1] *
                            values[packed >> 4] * scale_values[scale_index];
                    }
                }
                expected[row] = static_cast<float>(sum * row_scales[row]);
            }
            Tensor input = device_tensor(backend, inner);
            Tensor output0 = device_tensor(backend, rows);
            Tensor output1 = device_tensor(backend, rows);
            if (!upload(backend, input, activation)) return false;
            const mollm_cuda::Nvfp4Projection jobs[] = {
                {row_major.data(), scales.data(), row_scales.data(),
                 static_cast<const float*>(input.device.buffer),
                 static_cast<float*>(output0.device.buffer), false},
                {pair_layout.data(), scales.data(), row_scales.data(),
                 static_cast<const float*>(input.device.buffer),
                 static_cast<float*>(output1.device.buffer), true}};
            // Descriptors and all weights deliberately stay in ordinary CPU
            // allocations. The caller must keep them alive until completion.
            mollm_cuda::launch_nvfp4_projections(jobs, 2, rows, inner);
            if (cudaGetLastError() != cudaSuccess) return false;
            std::vector<float> actual;
            for (const Tensor* output : {&output0, &output1}) {
                if (!download(backend, *output, actual) ||
                    !close_enough_relative(actual, expected, 1e-5f, 2e-6f)) return false;
            }
            std::vector<float> gold0, gold1;
            if (!download(backend, output0, gold0) || !download(backend, output1, gold1)) return false;
            // Unequal layouts/weights exercise the independent-pair fallback.
            if (mollm_cuda::launch_nvfp4_paired_projections(jobs, 2, rows, inner) != cudaSuccess ||
                !download(backend, output0, actual) || actual != gold0 ||
                !download(backend, output1, actual) || actual != gold1) return false;
            // Equal immutable weights, distinct destinations; odd last job.
            Tensor output2 = device_tensor(backend, rows);
            const mollm_cuda::Nvfp4Projection same[] = {jobs[0],
                {jobs[0].weight, jobs[0].block_scales, jobs[0].row_scales,
                 jobs[0].input, jobs[1].output, false},
                {jobs[1].weight, jobs[1].block_scales, jobs[1].row_scales,
                 jobs[1].input, static_cast<float*>(output2.device.buffer), true}};
            if (mollm_cuda::launch_nvfp4_paired_projections(same, 2, rows, inner) != cudaSuccess ||
                !download(backend, output1, actual) || actual != gold0 ||
                mollm_cuda::launch_nvfp4_paired_projections(same, 1, rows, inner) != cudaSuccess ||
                !download(backend, output0, actual) || actual != gold0) return false;
            mollm_cuda::Nvfp4MmaBatch* pair_plan = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&pair_plan), 4 * sizeof(*pair_plan)) != cudaSuccess) return false;
            mollm_cuda::Nvfp4Projection* ranked_jobs = nullptr;
            if(cudaMalloc(reinterpret_cast<void**>(&ranked_jobs),6*sizeof(*ranked_jobs))!=cudaSuccess)return false;
            for (int mode : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}) {
            if((mode==14 || mode==18) && inner%128)continue;
            cudaStream_t stream = nullptr;
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            auto ranked_launch = [&] {
                mollm_cuda::launch_group_nvfp4_jobs(same,same,ranked_jobs,ranked_jobs+3,3,stream,pair_plan);
                const auto status=cudaGetLastError();
                return status!=cudaSuccess?status:mollm_cuda::launch_nvfp4_ranked_pairs(ranked_jobs,3,pair_plan,rows,inner,stream);
            };
            if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
                cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess ||
                (mode >= 21 ? mollm_cuda::launch_nvfp4_pipelined_projections(same,3,rows,inner,true,stream,false,mode-16)
                 : mode == 20 ? mollm_cuda::launch_nvfp4_pipelined_projections(same,3,rows,inner,true,stream,false,4)
                 : mode == 19 ? mollm_cuda::launch_nvfp4_pipelined_projections(same,3,rows,inner,true,stream,false,3)
                 : mode == 18 ? mollm_cuda::launch_nvfp4_pipelined_projections(same,3,rows,inner,true,stream,false,0,false,false,false,false,false,false,false,true)
                 : mode >= 16 ? mollm_cuda::launch_nvfp4_exact_subwarp(same,3,rows,inner,mode==16?16:8,stream)
                 : mode == 15 ? mollm_cuda::launch_nvfp4_pipelined_projections(same,3,rows,inner,true,stream,false,0,false,false,false,false,false,true)
                 : mode == 14 ? mollm_cuda::launch_nvfp4_pipelined_projections(same,3,rows,inner,true,stream,false,0,false,false,true)
                 : mode >= 12 ? mollm_cuda::launch_nvfp4_row_reuse(same,3,rows,inner,mode==12?2:4,stream)
                 : mode == 11 ? ranked_launch()
                 : mode == 10 ? mollm_cuda::launch_nvfp4_inline_pairs(same,3,rows,inner,stream)
                 : mode >= 6 ? mollm_cuda::launch_nvfp4_pipelined_projections(same, 3, rows, inner, true, stream, mode == 6, mode >= 7 && mode <= 8 ? mode - 6 : 0, mode == 9)
                 : mode >= 4 ? mollm_cuda::launch_nvfp4_deep_pipelined_projections(same, 3, rows, inner, mode == 4 ? 2 : 4, stream)
                 : mode >= 2 ? mollm_cuda::launch_nvfp4_pipelined_projections(same, 3, rows, inner, mode == 3, stream)
                 : mode == 1 ? mollm_cuda::launch_nvfp4_grouped_pairs(same, 3, pair_plan, rows, inner, stream)
                             : mollm_cuda::launch_nvfp4_paired_projections(same, 3, rows, inner, stream)) != cudaSuccess ||
                cudaStreamEndCapture(stream, &graph) != cudaSuccess ||
                cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) != cudaSuccess) return false;
            bool exact = true;
            for (int replay = 0; replay < 2; ++replay) {
                if (replay) {
                    for (float& value : activation) value = value * -.75f + .125f;
                    if (!upload(backend, input, activation)) { exact = false; break; }
                    mollm_cuda::launch_nvfp4_projections(jobs, 2, rows, inner);
                    if (!download(backend, output0, gold0) || !download(backend, output1, gold1)) {
                        exact = false; break;
                    }
                }
                if (cudaGraphLaunch(executable, stream) != cudaSuccess ||
                    cudaStreamSynchronize(stream) != cudaSuccess ||
                    !download(backend, output0, actual) || actual != gold0 ||
                    !download(backend, output1, actual) || actual != gold0 ||
                    !download(backend, output2, actual) || actual != gold1) { exact = false; break; }
            }
            cudaGraphExecDestroy(executable);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(stream);
            if (!exact) return false;
            }
            cudaFree(pair_plan);
            cudaFree(ranked_jobs);
            if(mollm_cuda::launch_nvfp4_ranked_pairs(nullptr,0,nullptr,rows,inner)!=cudaSuccess ||
                mollm_cuda::launch_nvfp4_ranked_pairs(same,1,nullptr,rows,inner)!=cudaErrorInvalidValue)return false;
            if (mollm_cuda::launch_nvfp4_deep_pipelined_projections(nullptr, 0, rows, inner, 2) != cudaSuccess ||
                mollm_cuda::launch_nvfp4_deep_pipelined_projections(nullptr, 1, rows, inner, 2) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_deep_pipelined_projections(same, 1, rows, inner, 3) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_deep_pipelined_projections(same, -1, rows, inner, 4) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_deep_pipelined_projections(same, 65536, rows, inner, 4) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_deep_pipelined_projections(same, 1, 0, inner, 2) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_deep_pipelined_projections(same, 1, rows, inner - 1, 4) != cudaErrorInvalidValue)
                return false;
            if (mollm_cuda::launch_nvfp4_pipelined_projections(same, 1, rows, inner, true, nullptr, false, 12) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_pipelined_projections(same, 1, rows, inner, false, nullptr, false, 2) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_pipelined_projections(same, 1, rows, inner, true, nullptr, true, 2) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_pipelined_projections(nullptr, 0, rows, inner) != cudaSuccess ||
                mollm_cuda::launch_nvfp4_pipelined_projections(nullptr, 1, rows, inner) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_pipelined_projections(same, -1, rows, inner) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_pipelined_projections(same, 65536, rows, inner) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_pipelined_projections(same, 1, 0, inner) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_pipelined_projections(same, 1, rows, inner - 1) != cudaErrorInvalidValue)
                return false;
            if (mollm_cuda::launch_nvfp4_grouped_pairs(nullptr, 0, nullptr, rows, inner) != cudaSuccess ||
                mollm_cuda::launch_nvfp4_grouped_pairs(nullptr, 1, nullptr, rows, inner) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_grouped_pairs(same, 4097, nullptr, rows, inner) != cudaErrorInvalidValue ||
                mollm_cuda::launch_nvfp4_grouped_pairs(same, 1, nullptr, rows, inner) != cudaErrorInvalidValue)
                return false;
        }
    }
    return true;
}

bool test_gated_residual(CudaBackend& backend) {
    backend.clear_dispatch_error();
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    for (int hidden : {7, 2560}) {
        for (int rows : {1, 5}) {
            for (int streams : {3, 4}) {
                const int wide = hidden * streams;
                std::vector<float> x_data(wide * rows), gates_data(wide * rows);
                std::vector<float> gamma_data(wide), branch_data(hidden * rows);
                std::vector<float> inject_gates_data(streams * rows);
                for (size_t i = 0; i < x_data.size(); ++i) {
                    x_data[i] = (static_cast<int>(i % 29) - 14) / 8.0f;
                    gates_data[i] = static_cast<float>(i % 17) / 16.0f;
                }
                for (int i = 0; i < wide; ++i)
                    gamma_data[i] = 0.5f + (i % 23) / 32.0f;
                for (size_t i = 0; i < branch_data.size(); ++i)
                    branch_data[i] = (static_cast<int>(i % 31) - 15) / 16.0f;
                for (size_t i = 0; i < inject_gates_data.size(); ++i)
                    inject_gates_data[i] = static_cast<float>(i % 13) / 16.0f;
                Tensor x = device_tensor(backend, wide, rows);
                Tensor gates = device_tensor(backend, wide, rows);
                Tensor gamma = device_tensor(backend, wide);
                Tensor branch = device_tensor(backend, hidden, rows);
                Tensor inject_gates = device_tensor(backend, streams, rows);
                Tensor wide_output = device_tensor(backend, wide, rows);
                Tensor narrow_output = device_tensor(backend, hidden, rows);
                if (!upload(backend, x, x_data) || !upload(backend, gates, gates_data) ||
                    !upload(backend, gamma, gamma_data) ||
                    !upload(backend, branch, branch_data) ||
                    !upload(backend, inject_gates, inject_gates_data)) return false;
                Tensor hx = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                    wide, rows, 1, 1, x_data.data());
                Tensor hg = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                    wide, rows, 1, 1, gates_data.data());
                Tensor hw = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                    wide, 1, 1, 1, gamma_data.data());
                Tensor hb = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                    hidden, rows, 1, 1, branch_data.data());
                Tensor hi = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                    streams, rows, 1, 1, inject_gates_data.data());
                for (OpType op : {OpType::GROUP_RMS_NORM,
                                  OpType::GR_REDUCE, OpType::GR_INJECT}) {
                    const bool reduce = op == OpType::GR_REDUCE;
                    Tensor& out = reduce ? narrow_output : wide_output;
                    std::vector<float> expected(out.nelements()), actual;
                    Tensor host_out = Tensor::create(Precision::FP32,
                        MemoryType::EXTERNAL, out.shape[0], rows, 1, 1, expected.data());
                    GraphNode node;
                    node.op_type = op;
                    node.params.i32 = {hidden, streams};
                    node.params.f32 = {1e-6f};
                    const std::vector<const Tensor*> host_inputs =
                        op == OpType::GROUP_RMS_NORM ? std::vector<const Tensor*>{&hx, &hw} :
                        reduce ? std::vector<const Tensor*>{&hx, &hg} :
                                 std::vector<const Tensor*>{&hb, &hx, &hi};
                    const std::vector<const Tensor*> device_inputs =
                        op == OpType::GROUP_RMS_NORM ? std::vector<const Tensor*>{&x, &gamma} :
                        reduce ? std::vector<const Tensor*>{&x, &gates} :
                                 std::vector<const Tensor*>{&branch, &x, &inject_gates};
                    CPUBackend cpu;
                    cpu.dispatch(node, host_inputs, &host_out, nullptr);
                    backend.dispatch(node, device_inputs, &out, nullptr);
                    if (cpu.dispatch_failed() || backend.dispatch_failed() ||
                        !download(backend, out, actual) ||
                        !close_enough(actual, expected,
                            op == OpType::GROUP_RMS_NORM ? 1e-5f : 0.0f)) return false;
                }
            }
        }
    }
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::ALLOW_REFERENCE);
    return true;
}

bool test_gdn(CudaBackend& backend, bool sigmoid_output_gate = false) {
    backend.clear_dispatch_error();
    constexpr int key_heads = 3;
    constexpr int value_heads = 6;
    constexpr int dimension = 128;
    constexpr int sequence_length = 65;
    constexpr int real_tokens = 61;
    constexpr int qkv_width =
        2 * key_heads * dimension + value_heads * dimension;
    constexpr int output_width = value_heads * dimension;

    std::vector<float> qkv_data(
        static_cast<size_t>(qkv_width) * sequence_length);
    std::vector<float> a_data(
        static_cast<size_t>(value_heads) * sequence_length);
    std::vector<float> b_data(a_data.size());
    std::vector<float> z_data(
        static_cast<size_t>(output_width) * sequence_length);
    std::vector<float> a_log(value_heads);
    std::vector<float> dt_bias(value_heads);
    std::vector<float> norm_weight(dimension);
    std::vector<float> initial_state(
        static_cast<size_t>(value_heads) * dimension * dimension);
    for (size_t index = 0; index < qkv_data.size(); ++index)
        qkv_data[index] =
            (static_cast<int>(index % 29) - 14) / 41.0f;
    for (size_t index = 0; index < a_data.size(); ++index) {
        a_data[index] = (static_cast<int>(index % 7) - 3) / 13.0f;
        b_data[index] = (static_cast<int>(index % 5) - 2) / 11.0f;
    }
    for (size_t index = 0; index < z_data.size(); ++index)
        z_data[index] =
            (static_cast<int>(index % 17) - 8) / 23.0f;
    for (int head = 0; head < value_heads; ++head) {
        a_log[head] = -1.1f + head * 0.15f;
        dt_bias[head] = -0.2f + head * 0.1f;
    }
    for (int index = 0; index < dimension; ++index)
        norm_weight[index] = 0.75f + (index % 11) / 32.0f;
    for (size_t index = 0; index < initial_state.size(); ++index)
        initial_state[index] =
            (static_cast<int>(index % 19) - 9) / 97.0f;

    OpParams params;
    params.i32 = {key_heads, dimension, dimension, sequence_length,
                  sigmoid_output_gate ? 3 : 1, 4, real_tokens, value_heads};
    params.f32 = {1e-6f, 1e-6f, 1.0f / std::sqrt(128.0f)};
    std::vector<float> expected_state = initial_state;
    std::vector<float> expected_output(
        static_cast<size_t>(output_width) * sequence_length);
    Tensor h_qkv = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        qkv_width, sequence_length, 1, 1, qkv_data.data());
    Tensor h_a = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        value_heads, sequence_length, 1, 1, a_data.data());
    Tensor h_b = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        value_heads, sequence_length, 1, 1, b_data.data());
    Tensor h_z = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        output_width, sequence_length, 1, 1, z_data.data());
    Tensor h_a_log = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        value_heads, 1, 1, 1, a_log.data());
    Tensor h_dt_bias = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        value_heads, 1, 1, 1, dt_bias.data());
    Tensor h_norm = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        dimension, 1, 1, 1, norm_weight.data());
    Tensor h_state = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        dimension, dimension, value_heads, 1, expected_state.data());
    Tensor h_output = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        output_width, sequence_length, 1, 1, expected_output.data());
    std::vector<const Tensor*> host_inputs = {
        &h_qkv, &h_a, &h_b, &h_z, &h_a_log, &h_dt_bias, &h_norm, &h_state};
    {
        GraphNode reference;
        reference.op_type = OpType::GATED_DELTANET_PREFILL;
        reference.params = params;
        CPUBackend{}.dispatch(reference, host_inputs, &h_output, nullptr);
    }

    Tensor qkv = device_tensor(backend, qkv_width, sequence_length);
    Tensor a = device_tensor(backend, value_heads, sequence_length);
    Tensor b = device_tensor(backend, value_heads, sequence_length);
    Tensor z = device_tensor(backend, output_width, sequence_length);
    Tensor device_a_log = device_tensor(backend, value_heads);
    Tensor device_dt_bias = device_tensor(backend, value_heads);
    Tensor norm = device_tensor(backend, dimension);
    Tensor state = device_tensor(
        backend, dimension, dimension, value_heads);
    Tensor output = device_tensor(backend, output_width, sequence_length);
    if (!upload(backend, qkv, qkv_data) ||
        !upload(backend, a, a_data) || !upload(backend, b, b_data) ||
        !upload(backend, z, z_data) ||
        !upload(backend, device_a_log, a_log) ||
        !upload(backend, device_dt_bias, dt_bias) ||
        !upload(backend, norm, norm_weight) ||
        !upload(backend, state, initial_state))
        return false;
    GraphNode gdn;
    gdn.op_type = OpType::GATED_DELTANET_PREFILL;
    gdn.params = params;
    backend.dispatch(
        gdn, {&qkv, &a, &b, &z, &device_a_log, &device_dt_bias,
              &norm, &state},
        &output, nullptr);
    backend.end_graph();
    std::vector<float> actual_output;
    std::vector<float> actual_state;
    if (backend.dispatch_failed() ||
        !download(backend, output, actual_output) ||
        !download(backend, state, actual_state) ||
        !close_enough_relative(
            actual_output, expected_output, 5e-6f, 2e-6f) ||
        !close_enough_relative(
            actual_state, expected_state, 5e-4f, 2e-6f))
        return false;

    // Reusing shared reduction storage used to race between consecutive
    // reductions. Reset the recurrent state and require deterministic output
    // across several identical launches so that regression cannot hide behind
    // a one-shot numerical tolerance check.
    const std::vector<float> deterministic_output = actual_output;
    const std::vector<float> deterministic_state = actual_state;
    {
        // Bounded graph-level liveness test around the real GDN state update.
        // Padded tail stays present, but only real_tokens may advance state.
        for (bool owned_state : {false,true}) {
        std::vector<std::vector<float>> graph_outputs, graph_states;
        for (int mode : {0,1,2}) {
            const bool planned=mode!=0,contract=mode==2;
            Graph graph; graph.nodes.resize(owned_state ? 16 : 14);
            graph.runtime.tensors.resize(graph.nodes.size());
            const std::vector<Tensor> sources = {qkv, a, b, z, device_a_log,
                                                device_dt_bias, norm, state};
            for (int id = 0; id < 14; ++id) {
                auto& node = graph.nodes[id]; node.id = id;
                node.out_shape[0] = id == 8 || id == 9 ? qkv_width : output_width;
                node.out_shape[1] = sequence_length;
            }
            for (int id = 0; id < 8; ++id) {
                graph.nodes[id].op_type = OpType::INPUT;
                graph.runtime.tensors[id] = sources[id];
            }
            graph.nodes[8].op_type = OpType::ADD; graph.nodes[8].inputs = {0, 0};
            graph.nodes[9].op_type = OpType::RESHAPE; graph.nodes[9].inputs = {8};
            graph.nodes[10] = gdn; graph.nodes[10].id = 10;
            graph.nodes[10].out_shape[0] = output_width;
            graph.nodes[10].out_shape[1] = sequence_length;
            graph.nodes[10].inputs = {9, 1, 2, 3, 4, 5, 6, 7};
            const bool reclaim=contract && backend.independent_stateful_mutable_input(graph.nodes[10])==7;
            graph.nodes[11].op_type = OpType::ADD; graph.nodes[11].inputs = {3, 3};
            graph.nodes[12].op_type = OpType::ADD; graph.nodes[12].inputs = {11, 11};
            graph.nodes[13].op_type = OpType::ADD; graph.nodes[13].inputs = {10, 10};
            if (owned_state) {
                auto& owner=graph.nodes[14]; owner.id=14;
                owner.op_type=OpType::ADD; owner.inputs={7,7};
                for (int dim=0;dim<4;++dim) owner.out_shape[dim]=state.shape[dim];
                graph.nodes[15]=owner; graph.nodes[15].id=15;
                graph.nodes[15].op_type=OpType::RESHAPE; graph.nodes[15].inputs={14};
                graph.nodes[10].inputs[7]=15;
                std::rotate(graph.nodes.begin()+8,graph.nodes.begin()+14,graph.nodes.end());
            }
            graph.graph_outputs = {13};
            BufferPool pool; ExecContext ctx;
            ctx.graph = &graph; ctx.backend = &backend; ctx.pool = &pool;
            ctx.conservative_stateful_liveness = planned;
            ctx.stateful_readonly_last_use = contract;
            ctx.runtime_seq_len = real_tokens; ctx.static_padded = true;
            ctx.padded_seq_len = sequence_length;
            prepare_execution(ctx);
            if (!upload(backend, state, initial_state)) return false;
            for (int pass = 0; pass < 3; ++pass) {
                auto changed = qkv_data;
                for (size_t i = 0; i < changed.size(); ++i) changed[i] += .01f * pass;
                if (!upload(backend, qkv, changed)) return false;
                inject_runtime_shapes(ctx);
                backend.begin_graph(); execute_graph(ctx); backend.end_graph();
                std::vector<float> result, final_state;
                if (ctx.execution_failed || backend.dispatch_failed() ||
                    !download(backend, graph.runtime.tensors[13], result) ||
                    !download(backend,owned_state ? graph.runtime.tensors[15] : state,final_state)) return false;
                if (owned_state && (!graph.runtime.tensors[14].data ||
                    graph.runtime.tensors[15].data!=graph.runtime.tensors[14].data)) return false;
                if (!planned) { graph_outputs.push_back(result); graph_states.push_back(final_state); }
                else if (result.size() != graph_outputs[pass].size() || final_state.size() != graph_states[pass].size() ||
                    std::memcmp(result.data(), graph_outputs[pass].data(), result.size() * sizeof(float)) ||
                    std::memcmp(final_state.data(), graph_states[pass].data(), final_state.size() * sizeof(float)) ||
                    (reclaim ? (graph.runtime.tensors[8].data || graph.runtime.tensors[9].data || graph.runtime.tensors[10].data)
                              : (!graph.runtime.tensors[8].data || !graph.runtime.tensors[10].data)) ||
                    graph.runtime.tensors[11].data || graph.runtime.tensors[12].data) return false;
            }
            for (int id : {8, 10, 11, 12, 13})
                if (graph.runtime.tensors[id].data) backend.free_output(graph.runtime.tensors[id], nullptr);
            if (owned_state && graph.runtime.tensors[14].data)
                backend.free_output(graph.runtime.tensors[14],nullptr);
        }
        std::printf("conservative_gdn_state_owner sigmoid_gate=%d owned_state=%d exact_output_state=1 passes=3 policies=3\n",sigmoid_output_gate,owned_state);
        }
        if (!upload(backend, qkv, qkv_data)) return false;
        std::printf("conservative_gdn_graph sigmoid_gate=%d exact_output_state=1 padded=65 real=61 replays=3 released_branch=1\n", sigmoid_output_gate);
    }
    for (int iteration = 0; iteration < 3; ++iteration) {
        if (!upload(backend, state, initial_state))
            return false;
        backend.dispatch(
            gdn, {&qkv, &a, &b, &z, &device_a_log, &device_dt_bias,
                  &norm, &state},
            &output, nullptr);
        backend.end_graph();
        if (backend.dispatch_failed() ||
            !download(backend, output, actual_output) ||
            !download(backend, state, actual_state) ||
            !close_enough(actual_output, deterministic_output, 0.0f) ||
            !close_enough(actual_state, deterministic_state, 0.0f))
            return false;
    }

    // The optimized prefill path must preserve the optional-normalization
    // contract as well as Qwen3.5's normal use_qk_l2norm=true case.
    std::vector<float> expected_raw_state = initial_state;
    std::vector<float> expected_raw_output(
        static_cast<size_t>(output_width) * sequence_length);
    h_state.data = expected_raw_state.data();
    h_output.data = expected_raw_output.data();
    params.i32[4] = sigmoid_output_gate ? 2 : 0;
    {
        GraphNode reference;
        reference.op_type = OpType::GATED_DELTANET_PREFILL;
        reference.params = params;
        CPUBackend{}.dispatch(reference, host_inputs, &h_output, nullptr);
    }
    if (!upload(backend, state, initial_state))
        return false;
    gdn.params = params;
    backend.dispatch(
        gdn, {&qkv, &a, &b, &z, &device_a_log, &device_dt_bias,
              &norm, &state},
        &output, nullptr);
    backend.end_graph();
    if (backend.dispatch_failed() ||
        !download(backend, output, actual_output) ||
        !download(backend, state, actual_state) ||
        !close_enough_relative(
            actual_output, expected_raw_output, 5e-6f, 2e-6f) ||
        !close_enough_relative(
            actual_state, expected_raw_state, 5e-4f, 2e-6f))
        return false;

    params.i32[4] = sigmoid_output_gate ? 3 : 1;
    if (!upload(backend, state, expected_state))
        return false;

    std::vector<float> decode_qkv(qkv_width);
    std::vector<float> decode_a(value_heads);
    std::vector<float> decode_b(value_heads);
    std::vector<float> decode_z(output_width);
    for (size_t index = 0; index < decode_qkv.size(); ++index)
        decode_qkv[index] =
            (static_cast<int>(index % 23) - 11) / 37.0f;
    for (int head = 0; head < value_heads; ++head) {
        decode_a[head] = (head - 1) / 9.0f;
        decode_b[head] = (1 - head) / 7.0f;
    }
    for (size_t index = 0; index < decode_z.size(); ++index)
        decode_z[index] =
            (static_cast<int>(index % 13) - 6) / 19.0f;
    std::vector<float> expected_decode(output_width);
    h_qkv = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        qkv_width, 1, 1, 1, decode_qkv.data());
    h_a = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        value_heads, 1, 1, 1, decode_a.data());
    h_b = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        value_heads, 1, 1, 1, decode_b.data());
    h_z = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        output_width, 1, 1, 1, decode_z.data());
    h_state.data = expected_state.data();
    h_output = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        output_width, 1, 1, 1, expected_decode.data());
    params.i32[3] = 1;
    params.i32[6] = 1;
    host_inputs = {
        &h_qkv, &h_a, &h_b, &h_z, &h_a_log, &h_dt_bias, &h_norm, &h_state};
    {
        GraphNode reference;
        reference.op_type = OpType::GATED_DELTANET_DECODE;
        reference.params = params;
        CPUBackend{}.dispatch(reference, host_inputs, &h_output, nullptr);
    }

    qkv = device_tensor(backend, qkv_width);
    a = device_tensor(backend, value_heads);
    b = device_tensor(backend, value_heads);
    z = device_tensor(backend, output_width);
    output = device_tensor(backend, output_width);
    if (!upload(backend, qkv, decode_qkv) ||
        !upload(backend, a, decode_a) || !upload(backend, b, decode_b) ||
        !upload(backend, z, decode_z))
        return false;
    gdn.op_type = OpType::GATED_DELTANET_DECODE;
    gdn.params = params;
    backend.dispatch(
        gdn, {&qkv, &a, &b, &z, &device_a_log, &device_dt_bias,
              &norm, &state},
        &output, nullptr);
    backend.end_graph();
    if (backend.dispatch_failed() ||
        !download(backend, output, actual_output) ||
        !download(backend, state, actual_state) ||
        !close_enough(actual_output, expected_decode, 2e-4f) ||
        !close_enough(actual_state, expected_state, 2e-4f))
        return false;

    constexpr int convolution_kernel = 4;
    std::vector<float> raw_qkv(qkv_width);
    std::vector<float> convolution_weight(
        static_cast<size_t>(qkv_width) * convolution_kernel);
    std::vector<float> convolution_state(
        static_cast<size_t>(qkv_width) * (convolution_kernel - 1));
    for (size_t index = 0; index < raw_qkv.size(); ++index)
        raw_qkv[index] =
            (static_cast<int>(index % 31) - 15) / 43.0f;
    for (size_t index = 0; index < convolution_weight.size(); ++index)
        convolution_weight[index] =
            (static_cast<int>(index % 11) - 5) / 27.0f;
    for (size_t index = 0; index < convolution_state.size(); ++index)
        convolution_state[index] =
            (static_cast<int>(index % 13) - 6) / 39.0f;
    std::vector<float> expected_convolution_state = convolution_state;
    std::vector<float> expected_conv_output(output_width);
    Tensor h_raw_qkv = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        qkv_width, 1, 1, 1, raw_qkv.data());
    Tensor h_conv_weight = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        convolution_kernel, qkv_width, 1, 1, convolution_weight.data());
    Tensor h_conv_state = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        convolution_kernel - 1, qkv_width, 1, 1,
        expected_convolution_state.data());
    h_state.data = expected_state.data();
    h_output = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        output_width, 1, 1, 1, expected_conv_output.data());
    host_inputs = {
        &h_raw_qkv, &h_a, &h_b, &h_z, &h_a_log, &h_dt_bias, &h_norm,
        &h_state, &h_conv_weight, &h_conv_state};
    {
        GraphNode reference;
        reference.op_type = OpType::GATED_DELTANET_CONV_DECODE;
        reference.params = params;
        CPUBackend{}.dispatch(reference, host_inputs, &h_output, nullptr);
    }

    Tensor raw_qkv_tensor = device_tensor(backend, qkv_width);
    Tensor conv_weight =
        device_tensor(backend, convolution_kernel, qkv_width);
    Tensor conv_state =
        device_tensor(backend, convolution_kernel - 1, qkv_width);
    output = device_tensor(backend, output_width);
    if (!upload(backend, raw_qkv_tensor, raw_qkv) ||
        !upload(backend, conv_weight, convolution_weight) ||
        !upload(backend, conv_state, convolution_state))
        return false;
    gdn.op_type = OpType::GATED_DELTANET_CONV_DECODE;
    gdn.params = params;
    backend.dispatch(
        gdn, {&raw_qkv_tensor, &a, &b, &z, &device_a_log,
              &device_dt_bias, &norm, &state, &conv_weight, &conv_state},
        &output, nullptr);
    backend.end_graph();
    std::vector<float> actual_convolution_state;
    return !backend.dispatch_failed() &&
        download(backend, output, actual_output) &&
        download(backend, state, actual_state) &&
        download(backend, conv_state, actual_convolution_state) &&
        close_enough(actual_output, expected_conv_output, 3e-4f) &&
        close_enough(actual_state, expected_state, 3e-4f) &&
        close_enough(
            actual_convolution_state, expected_convolution_state, 1e-6f);
}

bool test_decode_add_rms_norm(CudaBackend& backend) {
    backend.clear_dispatch_error();
    constexpr int width = 1024;
    std::vector<float> residual_data(width);
    std::vector<float> update_data(width);
    std::vector<float> weight_data(width);
    std::vector<float> residual_expected(width);
    std::vector<float> output_expected(width);
    float square_sum = 0.0f;
    for (int column = 0; column < width; ++column) {
        residual_data[column] =
            (static_cast<int>(column % 29) - 14) / 19.0f;
        update_data[column] =
            (static_cast<int>(column % 17) - 8) / 23.0f;
        weight_data[column] = 0.75f + (column % 13) / 32.0f;
        residual_expected[column] =
            residual_data[column] + update_data[column];
        square_sum += residual_expected[column] * residual_expected[column];
    }
    const float inverse =
        1.0f / std::sqrt(square_sum / width + 1e-6f);
    for (int column = 0; column < width; ++column) {
        output_expected[column] =
            residual_expected[column] * inverse * weight_data[column];
    }

    Tensor residual = device_tensor(backend, width);
    Tensor update = device_tensor(backend, width);
    Tensor output = device_tensor(backend, width);
    Tensor weight = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, width, 1, 1, 1,
        weight_data.data());
    backend.wrap_weight(weight);
    if (!upload(backend, residual, residual_data) ||
        !upload(backend, update, update_data))
        return false;

    GraphNode add_norm;
    add_norm.op_type = OpType::ADD_RMS_NORM;
    add_norm.params.f32 = {1e-6f};
    backend.dispatch(add_norm, {&residual, &update, &weight}, &output,
                     nullptr);
    backend.end_graph();
    std::vector<float> residual_actual;
    std::vector<float> output_actual;
    return !backend.dispatch_failed() &&
        download(backend, residual, residual_actual) &&
        download(backend, output, output_actual) &&
        close_enough(residual_actual, residual_expected, 2e-6f) &&
        close_enough(output_actual, output_expected, 3e-5f);
}

bool test_lm_head_argmax(CudaBackend& backend) {
    backend.clear_dispatch_error();
    constexpr int n = 7;
    constexpr int k = 8;
    std::vector<float> activation_data(k, 1.0f);
    std::vector<mollm::cpu::fp16_t> weight_data(
        static_cast<size_t>(n) * k);
    for (int row = 0; row < n; ++row) {
        const float value = row == 1 || row == 2
            ? 0.25f : (row == 3 ? 0.125f : -0.25f);
        for (int column = 0; column < k; ++column) {
            weight_data[static_cast<size_t>(row) * k + column] =
                static_cast<mollm::cpu::fp16_t>(value);
        }
    }
    Tensor activation = device_tensor(backend, k);
    Tensor weight = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, n, k, 1, 1,
        weight_data.data());
    backend.wrap_weight(weight);
    if (!upload(backend, activation, activation_data) ||
        !backend.supports_lm_head_argmax(weight))
        return false;

    std::vector<float> logits(n);
    backend.lm_head_gemv_device_and_end_graph(
        activation, 0, weight, logits.data(), n, k);
    if (backend.dispatch_failed())
        return false;
    const int expected = static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    const int actual = backend.lm_head_argmax_device_and_end_graph(
        activation, 0, weight, n, k);
    return !backend.dispatch_failed() && expected == 1 && actual == expected;
}

bool test_ple_history_prefix(CudaBackend&) {
    const char* flag=std::getenv("MOLLM_CUDA_MTP_STATE_PREFIXES");
    if(!flag||std::strcmp(flag,"1"))return true;
    CudaBackend backend;
    int32_t ids_data[4]={4,0,5,6},sizes_data[2]={7,11},offset_data[2]={0,7};
    uint8_t table_data[36]={};float scale_data=1.f;
    Tensor ids=Tensor::create(Precision::INT32,MemoryType::EXTERNAL,4,1,1,1,ids_data);
    Tensor table=Tensor::create(Precision::RAW_U8,MemoryType::EXTERNAL,2,18,1,1,table_data);
    Tensor scale=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,1,1,1,1,&scale_data);
    Tensor sizes=Tensor::create(Precision::INT32,MemoryType::EXTERNAL,2,1,1,1,sizes_data);
    Tensor offsets=Tensor::create(Precision::INT32,MemoryType::EXTERNAL,2,1,1,1,offset_data);
    Tensor history=Tensor::create(Precision::INT32,MemoryType::EXTERNAL,3);
    backend.alloc_persistent(history,12,PersistentHostAccess::NONE);
    Tensor output=device_tensor(backend,4,4);
    GraphNode node;node.op_type=OpType::PLE_LOOKUP;node.params.i32={3,1,0,100,0,1234,0,4};
    for(int initialized:{0,1}) {
        int32_t initial[3]={1,2,initialized};
        if(!backend.copy_from_host(initial,history,12)||!backend.begin_state_prefixes({&history},4))return false;
        backend.dispatch(node,{&ids,&history,&table,&scale,&sizes,&offsets},&output,nullptr);
        backend.end_graph();if(backend.dispatch_failed())return false;
        for(int kept=1;kept<=4;++kept) {
            int32_t expected[3];std::memcpy(expected,initial,12);
            Tensor host_history=Tensor::create(Precision::INT32,MemoryType::EXTERNAL,3,1,1,1,expected);
            Tensor prefix_ids=ids;prefix_ids.shape[0]=kept;prefix_ids.compute_strides();
            std::vector<float> values(4*kept);
            Tensor host_output=Tensor::create(Precision::FP32,MemoryType::EXTERNAL,4,kept,1,1,values.data());
            GraphNode prefix=node;prefix.params.i32[7]=kept;
            CPUBackend cpu;cpu.dispatch(prefix,{&prefix_ids,&host_history,&table,&scale,&sizes,&offsets},&host_output,nullptr);
            int32_t actual[3];
            if(cpu.dispatch_failed()||!backend.restore_state_prefix(kept)||
                !backend.copy_to_host(history,actual,12)||std::memcmp(expected,actual,12))return false;
        }
        backend.end_state_prefixes();
    }
    return true;
}

bool test_state_prefix_coverage(CudaBackend&) {
    CudaBackend backend;
    Tensor input=device_tensor(backend,7,3),weight=device_tensor(backend,4,7);
    Tensor state=device_tensor(backend,3,7),output=device_tensor(backend,7,3),unused=device_tensor(backend,4);
    std::vector<float> x(21),w(28,.25f),initial(21,.5f),dummy(4,9.f);
    for(int i=0;i<21;++i)x[i]=float(i-10)/17.f;
    if(!upload(backend,input,x)||!upload(backend,weight,w)||!upload(backend,state,initial)||!upload(backend,unused,dummy))return false;
    const char* flag=std::getenv("MOLLM_CUDA_MTP_STATE_PREFIXES");
    if(!flag||std::strcmp(flag,"1"))return !backend.begin_state_prefixes({&state},3);
    GraphNode node;node.op_type=OpType::SHORTCONV;node.params.i32={4,3};
    for(bool incomplete:{true,false}) {
        if(!upload(backend,state,initial))return false;
        std::vector<Tensor*> states={&state,&state};if(incomplete)states.push_back(&unused);
        if(!backend.begin_state_prefixes(states,3)||backend.restore_state_prefix(1))return false;
        backend.dispatch(node,{&input,&weight,&state},&output,nullptr);backend.end_graph();
        std::vector<float> before,after;
        if(backend.dispatch_failed()||!download(backend,state,before))return false;
        if(incomplete) {
            if(backend.restore_state_prefix(1)||!download(backend,state,after)||after!=before)return false;
            std::vector<float> untouched;if(!download(backend,unused,untouched)||untouched!=dummy)return false;
        } else {
            for(int kept:{1,2,3}) {
                if(!backend.restore_state_prefix(kept)||!download(backend,state,after))return false;
                for(int group=0;group<7;++group)for(int tap=0;tap<3;++tap) {
                    const int token=kept-3+tap;
                    const float expected=token<0?.5f:x[token*7+group];
                    if(after[group*3+tap]!=expected)return false;
                }
            }
            if(backend.restore_state_prefix(0)||backend.restore_state_prefix(4))return false;
        }
        backend.end_state_prefixes();if(backend.restore_state_prefix(1))return false;
    }
    return true;
}

bool test_q8_batch_head(CudaBackend&) {
    CudaBackend backend;
    constexpr int n=39,k=256;
    std::vector<int8_t> weights(n*k);
    std::vector<float> scales(n,.125f);
    for(int r=0;r<n;++r)for(int c=0;c<k;++c)
        weights[r*k+c]=static_cast<int8_t>((r*13+c*7)%127-63);
    // Duplicate rows exercise the same lowest-index argmax tie handling.
    std::copy_n(weights.data(),k,weights.data()+k);
    Tensor weight=Tensor::create(Precision::INT8,MemoryType::EXTERNAL,n,k,1,1,weights.data());
    weight.rowmajor_data=weights.data();weight.scales=scales.data();
    weight.group_size=k;weight.groups_per_row=1;backend.wrap_weight(weight);
    const char* flag=std::getenv("MOLLM_CUDA_MTP_VERIFY_Q8_BATCH");
    if(!flag||std::strcmp(flag,"1"))return !backend.supports_lm_head_small_batch(weight);
    if(!backend.supports_lm_head_small_batch(weight)) {std::fprintf(stderr,"q8 batch unsupported\n");return false;}
    {
        Tensor activation=device_tensor(backend,k,2);
        int output[2]={-1,-1};
        for(int bad_m:{0,1,9}) {
            if(backend.lm_head_small_batch_argmax_device_and_end_graph(
                activation,weight,output,bad_m,n,k) || !backend.dispatch_failed() ||
                output[0]!=-1 || output[1]!=-1)return false;
            backend.clear_dispatch_error();
        }
        if(backend.lm_head_small_batch_argmax_device_and_end_graph(
            activation,weight,output,2,INT32_MAX,k) || !backend.dispatch_failed())return false;
        backend.clear_dispatch_error();
        if(backend.lm_head_small_batch_argmax_device_and_end_graph(
            activation,weight,output,2,n,k,1) || !backend.dispatch_failed())return false;
        backend.clear_dispatch_error();
    }
    for(int m:{2,3,4,5,7,8}) {
        Tensor activation=device_tensor(backend,k,m);
        std::vector<float> inputs(m*k),logits(n);
        std::vector<int> expected(m),actual(m,-1);
        for(int replay=0;replay<2;++replay) {
            for(int t=0;t<m;++t)for(int c=0;c<k;++c)
                inputs[t*k+c]=replay?((c*11+t*17)%97-48)/101.f:0.f;
            if(!upload(backend,activation,inputs))return false;
            for(int t=0;t<m;++t) {
                backend.lm_head_gemv_device_and_end_graph(activation,size_t(t)*k,weight,logits.data(),n,k);
                if(backend.dispatch_failed())return false;
                expected[t]=std::max_element(logits.begin(),logits.end())-logits.begin();
            }
            if(!backend.lm_head_small_batch_argmax_device_and_end_graph(
                activation,weight,actual.data(),m,n,k)||actual!=expected) {
                std::fprintf(stderr,"q8 batch head failed m=%d replay=%d failed=%d\n",m,replay,backend.dispatch_failed());
                for(int t=0;t<m;++t)std::fprintf(stderr,"t=%d got=%d expected=%d\n",t,actual[t],expected[t]);
                return false;
            }
        }
    }
    return true;
}

bool test_memory_and_fallback_bridge(CudaBackend& backend) {
    backend.clear_dispatch_error();
    Tensor storage = device_tensor(backend, 6, 2);
    std::vector<float> source(12);
    for (int i = 0; i < 12; ++i)
        source[i] = static_cast<float>(i + 1);
    if (!upload(backend, storage, source))
        return false;

    Tensor view = storage;
    view.device.offset += 2 * sizeof(float);
    view.shape[0] = 2;
    const size_t span = view.view_span_bytes();
    std::vector<float> raw_span(span / sizeof(float));
    if (!backend.copy_to_host(view, raw_span.data(), span) ||
        raw_span.front() != 3.0f || raw_span[1] != 4.0f ||
        raw_span[6] != 9.0f || raw_span[7] != 10.0f)
        return false;
    float second_row[2] = {};
    if (!backend.copy_to_host(
            view, second_row, sizeof(second_row), 6 * sizeof(float)) ||
        second_row[0] != 9.0f || second_row[1] != 10.0f)
        return false;

    std::vector<float> replacement(raw_span.size(), -3.0f);
    replacement[0] = 21.0f;
    replacement[1] = 22.0f;
    replacement[6] = 23.0f;
    replacement[7] = 24.0f;
    if (!backend.copy_from_host(replacement.data(), view, span) ||
        !download(backend, storage, source) ||
        source[2] != 21.0f || source[3] != 22.0f ||
        source[8] != 23.0f || source[9] != 24.0f)
        return false;

    Tensor layer_output = device_tensor(backend, 2, 2);
    std::vector<float> scale_data = {-91.0f, 83.0f, 1.5f, 0.75f};
    std::vector<float> bias_data = {0.25f, -0.5f};
    Tensor scale_storage = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1,
        scale_data.data());
    Tensor scale = scale_storage.view_1d(2, 2 * sizeof(float));
    Tensor bias = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 2, 1, 1, 1,
        bias_data.data());
    GraphNode layer_norm;
    layer_norm.op_type = OpType::LAYER_NORM;
    layer_norm.params.f32 = {1e-5f};
    backend.dispatch(layer_norm, {&view, &scale, &bias}, &layer_output, nullptr);
    std::vector<float> layer_actual;
    if (backend.dispatch_failed() ||
        !download(backend, layer_output, layer_actual))
        return false;
    std::vector<float> layer_expected(4);
    for (int row = 0; row < 2; ++row) {
        const float x0 = row == 0 ? 21.0f : 23.0f;
        const float x1 = row == 0 ? 22.0f : 24.0f;
        const float mean = (x0 + x1) * 0.5f;
        const float inv = 1.0f /
            std::sqrt(((x0 - mean) * (x0 - mean) +
                       (x1 - mean) * (x1 - mean)) * 0.5f + 1e-5f);
        layer_expected[row * 2] = (x0 - mean) * inv * scale_data[2] +
            bias_data[0];
        layer_expected[row * 2 + 1] =
            (x1 - mean) * inv * scale_data[3] + bias_data[1];
    }
    if (!close_enough(layer_actual, layer_expected, 2e-5f))
        return false;

    Tensor update = device_tensor(backend, 2, 2);
    Tensor add_norm_output = device_tensor(backend, 2, 2);
    std::vector<float> update_data = {1.0f, -2.0f, 3.0f, -4.0f};
    if (!upload(backend, update, update_data))
        return false;
    GraphNode add_norm;
    add_norm.op_type = OpType::ADD_RMS_NORM;
    add_norm.params.f32 = {1e-6f};
    backend.dispatch(add_norm, {&view, &update, &scale},
                     &add_norm_output, nullptr);
    if (backend.dispatch_failed() ||
        !download(backend, storage, source) ||
        source[2] != 22.0f || source[3] != 20.0f ||
        source[8] != 26.0f || source[9] != 20.0f)
        return false;

    Tensor rounded = device_tensor(backend, 4);
    std::vector<float> round_source = {
        1.001f, -2.003f, INFINITY, std::nanf("")};
    if (!upload(backend, rounded, round_source) ||
        !backend.round_to_bf16(rounded))
        return false;
    std::vector<float> round_actual;
    if (!download(backend, rounded, round_actual))
        return false;
    for (size_t i = 0; i < 3; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &round_source[i], sizeof(bits));
        if ((bits & 0x7f800000u) != 0x7f800000u)
            bits += 0x7fffu + ((bits >> 16) & 1u);
        bits &= 0xffff0000u;
        uint32_t actual_bits = 0;
        std::memcpy(&actual_bits, &round_actual[i], sizeof(actual_bits));
        if (actual_bits != bits)
            return false;
    }

    Tensor full = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, 16);
    Tensor mirrored = full;
    Tensor host_authoritative = full;
    Tensor device_only = full;
    backend.alloc_persistent(
        full, full.nbytes(), PersistentHostAccess::FULL);
    backend.alloc_persistent(
        mirrored, mirrored.nbytes(),
        PersistentHostAccess::MIRRORED_PREFIX, 16);
    backend.alloc_persistent(
        host_authoritative, host_authoritative.nbytes(),
        PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX, 16);
    backend.alloc_persistent(
        device_only, device_only.nbytes(), PersistentHostAccess::NONE);
    if (!full.data || full.data != full.device.buffer ||
        !mirrored.data || mirrored.data == mirrored.device.buffer ||
        !host_authoritative.data ||
        host_authoritative.data == host_authoritative.device.buffer ||
        !device_only.data || device_only.data != device_only.device.buffer)
        return false;
    const uint32_t prefix[4] = {7, 8, 9, 10};
    if (!backend.copy_from_host(prefix, mirrored, sizeof(prefix)) ||
        std::memcmp(mirrored.data, prefix, sizeof(prefix)) != 0 ||
        !backend.zero_tensor(mirrored, sizeof(uint32_t), sizeof(uint32_t)) ||
        static_cast<const uint32_t*>(mirrored.data)[1] != 0)
        return false;
    const uint32_t split_source[6] = {21, 22, 23, 24, 25, 26};
    uint32_t split_copy[6] = {};
    if (!backend.copy_from_host(
            split_source, host_authoritative, sizeof(split_source)) ||
        !backend.copy_to_host(
            host_authoritative, split_copy, sizeof(split_copy)) ||
        std::memcmp(split_source, split_copy, sizeof(split_source)) != 0 ||
        !backend.zero_tensor(
            host_authoritative, 2 * sizeof(uint32_t),
            3 * sizeof(uint32_t)) ||
        !backend.copy_to_host(
            host_authoritative, split_copy, sizeof(split_copy)) ||
        split_copy[0] != 21 || split_copy[1] != 22 || split_copy[2] != 23 ||
        split_copy[3] != 0 || split_copy[4] != 0 || split_copy[5] != 26)
        return false;
    const float payload[2] = {31.0f, 32.0f};
    float payload_copy[2] = {};
    if (!backend.copy_from_host(
            payload, device_only, sizeof(payload), 4 * sizeof(float)) ||
        !backend.copy_to_host(
            device_only, payload_copy, sizeof(payload_copy),
            4 * sizeof(float)) ||
        payload_copy[0] != payload[0] || payload_copy[1] != payload[1])
        return false;

    Tensor pooled_a = device_tensor(backend, 64);
    void* pooled_pointer = pooled_a.device.buffer;
    backend.free_output(pooled_a, nullptr);
    Tensor pooled_b = device_tensor(backend, 32);
    if (pooled_b.device.buffer != pooled_pointer)
        return false;

    backend.clear_dispatch_error();
    if (!backend.set_operator_fallback_policy(
            OperatorFallbackPolicy::REQUIRE_NATIVE))
        return false;
    backend.dispatch(layer_norm, {&view, &scale, &bias},
                     &layer_output, nullptr);
    const bool rejected_before_staging = backend.dispatch_failed();
    backend.set_operator_fallback_policy(
        OperatorFallbackPolicy::ALLOW_REFERENCE);
    backend.clear_dispatch_error();
    return rejected_before_staging;
}

bool test_grouped_narrow_stream() {
    for (int rows : {1, 4, 7, 19, 21, 257, 319, 320, 321, 512, 640}) {
        const int inner = rows == 19 ? 2080 : rows == 21 ? 2144 : rows == 257 ? 6144 :
            (rows == 1 || rows == 512 || rows == 640) ? 2560 : 10240;
        const size_t projection_blocks = static_cast<size_t>((rows + 7) / 8) * (inner / 32);
        std::vector<Q4B8G32Block> host(2 * projection_blocks);
        for (size_t i = 0; i < host.size(); ++i)
            for (int row = 0; row < 8; ++row) {
                host[i].scales[row] = (1 + (i + row) % 17) / 37.f;
                for (int byte = 0; byte < 16; ++byte)
                    host[i].q[row][byte] = (i * 19 + row * 7 + byte * 13) & 255;
            }
        std::vector<float> activation(inner), reference(rows), second_reference(rows), actual(2 * (rows + 8), 42.f);
        for (int i = 0; i < inner; ++i) activation[i] = (i % 31 - 15) / 19.f;
        Q4B8G32Block* weight = nullptr;
        float* input = nullptr; float* output = nullptr;
        cudaStream_t stream = nullptr;
        const auto cleanup = [&]() {
            if (stream) cudaStreamDestroy(stream);
            cudaFree(weight); cudaFree(input); cudaFree(output);
        };
        if (cudaMalloc(reinterpret_cast<void**>(&weight), host.size() * sizeof(host[0])) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&input), inner * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&output), actual.size() * sizeof(float)) != cudaSuccess ||
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaMemcpy(weight, host.data(), host.size() * sizeof(host[0]), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(input, activation.data(), inner * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            cleanup(); return false;
        }
        mollm_cuda::launch_q4_g32_dense_gemv(input, weight, output, rows, inner);
        if (cudaMemcpy(reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            cleanup(); return false;
        }
        mollm_cuda::launch_q4_g32_dense_gemv(input, weight + projection_blocks, output, rows, inner);
        if (cudaMemcpy(second_reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            cleanup(); return false;
        }
        if (mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(nullptr, weight, output, rows, inner, 1, true, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(input + 1, weight, output, rows, inner, 2, true, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(input, weight, output, rows, inner, 3, false, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(input, weight, output, rows, inner - 1, 4, true, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_spatial_narrow_gemv(nullptr, weight, output, rows, inner, 2, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_spatial_narrow_gemv(input + 1, weight, output, rows, inner, 4, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_spatial_narrow_gemv(input, weight, output, rows, inner, 3, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_spatial_narrow_gemv(input, weight, output, rows, inner - 1, 8, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_dual_lane_gemv(nullptr, weight, output, rows, inner, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_dual_lane_gemv(input + 1, weight, output, rows, inner, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_dual_lane_gemv(input, weight, output, rows, inner - 1, stream) != cudaErrorInvalidValue) {
            cleanup(); return false;
        }
        if (mollm_cuda::launch_q4_g32_paired_projections(input + 1, weight, weight + projection_blocks,
                output, output + rows + 8, rows, inner, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_paired_projections(input, weight, nullptr,
                output, output + rows + 8, rows, inner, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_paired_projections(input, weight, weight + projection_blocks,
                output, output + rows + 8, rows, 2016, stream) != cudaErrorInvalidValue) {
            cleanup(); return false;
        }
        if (mollm_cuda::launch_q4_g32_shared_activation(input + 1, weight, weight + projection_blocks,
                output, rows, inner, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_shared_activation(input, weight, nullptr,
                output, rows, inner, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_shared_activation(input, weight, weight + projection_blocks,
                output, rows, 2016, stream) != cudaErrorInvalidValue) {
            cleanup(); return false;
        }
        if (mollm_cuda::launch_q4_g32_unequal_projections(input + 1, weight, weight + projection_blocks,
                output, output + rows + 8, rows, 1, inner, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_unequal_projections(input, weight, nullptr,
                output, output + rows + 8, rows, 1, inner, stream) != cudaErrorInvalidValue ||
            mollm_cuda::launch_q4_g32_unequal_projections(input, weight, weight + projection_blocks,
                output, output + rows + 8, rows, 0, inner, stream) != cudaErrorInvalidValue) {
            cleanup(); return false;
        }
        for (int group : {-16, -15, -14, -13, -12, -11, -10, -9, -8, -4, -3, -2, -1, 0, 2, 4, 8}) {
            std::fill(actual.begin(), actual.end(), 42.f);
            if (cudaMemcpyAsync(output, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess ||
                cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
                cleanup(); return false;
            }
            // -2 fuses shared activation; -1 pairs projections; zero uses dual lanes.
            // Both retain byte-exact references and explicit graph streams.
            if (group == -16) {
                mollm_cuda::launch_q4_g32_unequal_projections(input, weight, weight + projection_blocks,
                    output, output + rows + 8, rows, std::min(rows, 4), inner, stream);
            } else if (group <= -10) {
                const int ordinal = (-group - 10) % 3;
                mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(input, weight, output, rows, inner,
                    1 << ordinal, group >= -12, stream);
            } else if (group == -9)
                mollm_cuda::launch_q4_g32_single_barrier_gemv(input, weight, output, rows, inner, stream);
            else if (group < -2)
                mollm_cuda::launch_q4_g32_spatial_narrow_gemv(input, weight, output, rows, inner,
                    group == -3 ? 2 : -group, stream);
            else if (group == -2)
                mollm_cuda::launch_q4_g32_shared_activation(input, weight, weight + projection_blocks,
                    output, rows, inner, stream);
            else if (group == -1)
                mollm_cuda::launch_q4_g32_paired_projections(input, weight, weight + projection_blocks,
                    output, output + rows + 8, rows, inner, stream);
            else if (group == 0)
                mollm_cuda::launch_q4_g32_dual_lane_gemv(input, weight, output, rows, inner, stream);
            else
                mollm_cuda::launch_q4_g32_grouped_narrow_gemv(input, weight, output, rows, inner, group, stream);
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            bool ok = cudaStreamEndCapture(stream, &graph) == cudaSuccess && graph;
            if (ok) ok = cudaGraphInstantiateWithFlags(&executable, graph, 0) == cudaSuccess;
            for (int replay = 0; ok && replay < 2; ++replay) {
                for (int j = 0; j < inner; ++j)
                    activation[j] = (j % 31 - 15) / (replay ? 23.f : 19.f);
                ok = cudaMemcpy(input, activation.data(), inner * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
                if (ok) {
                    mollm_cuda::launch_q4_g32_dense_gemv(input, weight, output, rows, inner);
                    ok = cudaMemcpy(reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
                    mollm_cuda::launch_q4_g32_dense_gemv(input, weight + projection_blocks, output + rows + 8, rows, inner);
                    ok = ok && cudaMemcpy(second_reference.data(), output + rows + 8, rows * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
                    if (group == -2 && ok) {
                        mollm_cuda::launch_moe_activation(output, output + rows + 8, nullptr,
                            output, 1, rows, rows);
                        ok = cudaMemcpy(reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
                    }
                }
                std::fill(actual.begin(), actual.end(), 42.f);
                // Reset on the replay stream, after reference copies complete.
                ok = ok && cudaMemcpyAsync(output, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                    cudaGraphLaunch(executable, stream) == cudaSuccess &&
                    cudaMemcpyAsync(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                    cudaStreamSynchronize(stream) == cudaSuccess;
                ok = ok && std::memcmp(reference.data(), actual.data(), rows * sizeof(float)) == 0 &&
                    std::all_of(actual.begin() + rows, actual.begin() + rows + 8, [](float x) { return x == 42.f; });
                if (group == -1 || group == -16) {
                    const int second_rows = group == -16 ? std::min(rows, 4) : rows;
                    ok = ok && std::memcmp(second_reference.data(), actual.data() + rows + 8, second_rows * sizeof(float)) == 0 &&
                        std::all_of(actual.begin() + rows + 8 + second_rows, actual.end(), [](float x) { return x == 42.f; });
                } else
                    ok = ok && std::all_of(actual.begin() + rows + 8, actual.end(), [](float x) { return x == 42.f; });
            }
            if (executable) cudaGraphExecDestroy(executable);
            if (graph) cudaGraphDestroy(graph);
            if (!ok) {
                std::fprintf(stderr, "grouped narrow graph mismatch rows=%d group=%d\n", rows, group);
                cleanup(); return false;
            }
        }
        cleanup();
    }
    return true;
}

bool test_q4_wide_pipeline_stream(int pair_mode = 0) {
    std::vector<std::pair<int, int>> shapes = {{1, 32}, {19, 96}, {31, 160}, {33, 2560}, {16480, 2560}};
    if (pair_mode == 1 || pair_mode == 2) { shapes.emplace_back(10240, 320); shapes.emplace_back(2560, 640); }
    const auto launch = [&](const float* input, const Q4B8G32Block* weight, float* output,
                            int rows, int inner, cudaStream_t stream) {
        return pair_mode >= 3 ? mollm_cuda::launch_q4_g32_l1_prefetch_wide_gemv(input, weight, output, rows, inner, 1 << (pair_mode - 3), stream)
                         : pair_mode ? mollm_cuda::launch_q4_g32_pipelined_pair_gemv(input, weight, output, rows, inner, pair_mode == 2, stream)
                         : mollm_cuda::launch_q4_g32_pipelined_wide_gemv(input, weight, output, rows, inner, stream);
    };
    for (const auto shape : shapes) {
        const int rows = shape.first, inner = shape.second;
        std::vector<Q4B8G32Block> host(((rows + 7) / 8) * (inner / 32));
        for (size_t i = 0; i < host.size(); ++i)
            for (int row = 0; row < 8; ++row) {
                host[i].scales[row] = (1 + (i + row) % 17) / 37.f;
                for (int byte = 0; byte < 16; ++byte) host[i].q[row][byte] = (i * 19 + row * 7 + byte * 13) & 255;
            }
        Q4B8G32Block* weight = nullptr;
        float* input = nullptr; float* output = nullptr;
        cudaStream_t stream = nullptr; cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
        const auto cleanup = [&]() {
            if (executable) cudaGraphExecDestroy(executable);
            if (graph) cudaGraphDestroy(graph);
            if (stream) cudaStreamDestroy(stream);
            cudaFree(weight); cudaFree(input); cudaFree(output);
        };
        if (cudaMalloc(&weight, host.size() * sizeof(host[0])) != cudaSuccess ||
            cudaMalloc(&input, inner * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&output, (rows + 8) * sizeof(float)) != cudaSuccess ||
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaMemcpy(weight, host.data(), host.size() * sizeof(host[0]), cudaMemcpyHostToDevice) != cudaSuccess) {
            cleanup(); return false;
        }
        if (pair_mode >= 3 && mollm_cuda::launch_q4_g32_l1_prefetch_wide_gemv(
                input, weight, output, rows, inner, 3, stream) != cudaErrorInvalidValue) {
            cleanup(); return false;
        }
        if (launch(nullptr, weight, output, rows, inner, stream) != cudaErrorInvalidValue ||
            launch(input + 1, weight, output, rows, inner, stream) != cudaErrorInvalidValue ||
            launch(input, weight, output, rows, inner - 1, stream) != cudaErrorInvalidValue ||
            launch(input, weight, output, rows, 2592, stream) != cudaErrorInvalidValue) {
            cleanup(); return false;
        }
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess ||
            launch(input, weight, output, rows, inner, stream) != cudaSuccess ||
            cudaStreamEndCapture(stream, &graph) != cudaSuccess ||
            cudaGraphInstantiateWithFlags(&executable, graph, 0) != cudaSuccess) {
            cleanup(); return false;
        }
        std::vector<float> activation(inner), reference(rows), actual(rows + 8);
        for (int replay = 0; replay < 2; ++replay) {
            for (int i = 0; i < inner; ++i) activation[i] = (i % 41 - 20 + replay) / (37.f + replay);
            bool ok = cudaMemcpyAsync(input, activation.data(), inner * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess;
            if (pair_mode == 1 || pair_mode == 2) mollm_cuda::launch_q4_g32_pair_narrow_gemv(input, weight, output, rows, inner, false, stream);
            else mollm_cuda::launch_q4_g32_prefetch_wide_gemv(input, weight, output, rows, inner, 4, stream);
            ok = ok && cudaMemcpyAsync(reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                cudaStreamSynchronize(stream) == cudaSuccess;
            std::fill(actual.begin(), actual.end(), 42.f);
            ok = ok && cudaMemcpyAsync(output, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                cudaGraphLaunch(executable, stream) == cudaSuccess &&
                cudaMemcpyAsync(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                cudaStreamSynchronize(stream) == cudaSuccess &&
                std::memcmp(actual.data(), reference.data(), rows * sizeof(float)) == 0 &&
                std::all_of(actual.begin() + rows, actual.end(), [](float x) { return x == 42.f; });
            if (!ok) {
                std::fprintf(stderr,"Q4 wide pipeline graph check failed pair_mode=%d rows=%d inner=%d replay=%d\n",
                    pair_mode,rows,inner,replay);
                cleanup(); return false;
            }
        }
        cleanup();
    }
    return true;
}

bool test_q4_async_stream() {
    for (int rows : {1, 19, 31, 33}) for (int inner : {32, 96, 128, 160, 2560})
    for (int variant : {0, 4, 8}) {
        std::vector<Q4B8G32Block> host(((rows + 7) / 8) * (inner / 32));
        for (size_t i = 0; i < host.size(); ++i) for (int r = 0; r < 8; ++r) {
            host[i].scales[r] = (1 + (i + r) % 17) / 37.f;
            for (int k = 0; k < 16; ++k) host[i].q[r][k] = (i * 19 + r * 7 + k * 13) & 255;
        }
        std::vector<float> x(inner), reference(rows), actual(rows + 8), initial(rows + 8, 42.f);
        for (int k = 0; k < inner; ++k) x[k] = (k % 41 - 20) / 37.f;
        Q4B8G32Block* weight = nullptr;
        float *input = nullptr, *output = nullptr;
        cudaStream_t stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        bool ok = cudaMalloc(reinterpret_cast<void**>(&weight), host.size() * sizeof(host[0])) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&input), inner * sizeof(float)) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&output), initial.size() * sizeof(float)) == cudaSuccess &&
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess;
        if (ok) ok = cudaMemcpy(weight, host.data(), host.size() * sizeof(host[0]), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(input, x.data(), inner * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok) {
            mollm_cuda::launch_q4_g32_prefetch_wide_gemv(input, weight, output, rows, inner, 4);
            ok = cudaMemcpy(reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
            if (ok) {
                const auto launch = variant == 0
                    ? mollm_cuda::launch_q4_g32_async_wide_gemv(input, weight, output, rows, inner, stream)
                    : mollm_cuda::launch_q4_g32_prefetch_scale_gemv(input, weight, output, rows, inner, variant, stream);
                const auto end = cudaStreamEndCapture(stream, &graph);
                ok = launch == cudaSuccess && end == cudaSuccess;
            }
            if (ok) ok = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) == cudaSuccess;
            for (int replay = 0; ok && replay < 2; ++replay) {
                ok = cudaMemcpyAsync(output, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                    cudaGraphLaunch(executable, stream) == cudaSuccess &&
                    cudaMemcpyAsync(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                    cudaStreamSynchronize(stream) == cudaSuccess;
                ok = ok && std::memcmp(actual.data(), reference.data(), rows * sizeof(float)) == 0 &&
                    std::all_of(actual.begin() + rows, actual.end(), [](float y) { return y == 42.f; });
            }
        }
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        if (output) cudaFree(output);
        if (input) cudaFree(input);
        if (weight) cudaFree(weight);
        if (!ok) {
            std::fprintf(stderr, "Q4 async/scale-prefetch graph mismatch rows=%d inner=%d variant=%d\n", rows, inner, variant);
            return false;
        }
    }
    return true;
}

bool test_fp16_to_fp32_reference() {
    for (size_t count : {size_t(1), size_t(255), size_t(256), size_t(257)}) {
        std::vector<__half> source(count);
        std::vector<float> expected(count), actual(count);
        const uint16_t codes[] = {0x0000, 0x8000, 0x0001, 0x03ff, 0x0400,
                                  0x3c01, 0xbc00, 0x7bff, 0x7c00, 0xfc00};
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(&source[i], &codes[i % 10], sizeof(__half));
            expected[i] = __half2float(source[i]);
        }
        __half* input = nullptr;
        float* output = nullptr;
        bool ok = cudaMalloc(reinterpret_cast<void**>(&input), count * sizeof(__half)) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&output), count * sizeof(float)) == cudaSuccess;
        if (ok) ok = cudaMemcpy(input, source.data(), count * sizeof(__half), cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok) {
            mollm_cuda::launch_fp16_to_fp32(input, output, count);
            ok = cudaGetLastError() == cudaSuccess &&
                cudaMemcpy(actual.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
                std::memcmp(actual.data(), expected.data(), count * sizeof(float)) == 0;
        }
        if (input) cudaFree(input);
        if (output) cudaFree(output);
        if (!ok) return false;
    }
    mollm_cuda::launch_fp16_to_fp32(nullptr, nullptr, 0);
    return cudaGetLastError() == cudaSuccess;
}

bool test_nvfp4_sort_stream() {
    using Job = mollm_cuda::Nvfp4Projection;
    const auto same_job = [](const Job& a, const Job& b) {
        return a.weight == b.weight && a.block_scales == b.block_scales &&
            a.row_scales == b.row_scales && a.input == b.input && a.output == b.output &&
            a.pair_layout == b.pair_layout && a.mma_layout == b.mma_layout &&
            std::memcmp(&a.input_global, &b.input_global, sizeof(float)) == 0 &&
            std::memcmp(&a.input_global_second, &b.input_global_second, sizeof(float)) == 0 &&
            std::memcmp(&a.output_scale, &b.output_scale, sizeof(float)) == 0;
    };
    size_t zero_bytes = 1;
    if (mollm_cuda::nvfp4_radix_group_scratch_size(0, &zero_bytes) != cudaSuccess || zero_bytes ||
        mollm_cuda::nvfp4_radix_group_scratch_size(-1, &zero_bytes) != cudaErrorInvalidValue ||
        mollm_cuda::nvfp4_radix_group_scratch_size(65536, &zero_bytes) != cudaErrorInvalidValue ||
        mollm_cuda::nvfp4_radix_group_scratch_size(1, nullptr) != cudaErrorInvalidValue ||
        mollm_cuda::launch_group_nvfp4_jobs_radix(nullptr, nullptr, nullptr, nullptr,
            0, nullptr, 0) != cudaSuccess) return false;
    for (int method : {0, 1, 2})
    for (bool paired : {false, true})
    for (int count : {1, 127, 128, 129, 2048, 2560, 4096, 5120, 10240, 65535}) {
        const bool tiled = method == 1, radix = method == 2;
        if (radix && paired) continue;
        if (!radix && count > 10240) continue;
        if (!tiled && !radix && count > 4096) continue;
        std::vector<Job> gate(count), down(count), actual(2 * count);
        std::vector<int> order(count);
        for (int i = 0; i < count; ++i) {
            gate[i] = {};
            down[i] = {};
            // Opaque keys are sorted but never dereferenced by this kernel.
            gate[i].weight = reinterpret_cast<const uint8_t*>(uintptr_t(1 + (i * 37) % 113));
            gate[i].input_global = float(i + 1);
            down[i].output_scale = float(-i - 1);
            if (!paired) {
                gate[i].block_scales = reinterpret_cast<const uint8_t*>(uintptr_t(16 + i * 8));
                gate[i].input = reinterpret_cast<const float*>(uintptr_t(64 + i * 4));
                gate[i].output = reinterpret_cast<float*>(uintptr_t(128 + i * 4));
                gate[i].pair_layout = i % 2;
                gate[i].mma_layout = i % 3 == 0;
                gate[i].input_global_second = float(-i - 1);
                gate[i].output_scale = i % 2 ? -0.0f : 0.0f;
                down[i].weight = reinterpret_cast<const uint8_t*>(uintptr_t(1 + i % 41));
                down[i].block_scales = reinterpret_cast<const uint8_t*>(uintptr_t(32 + i * 8));
                down[i].row_scales = reinterpret_cast<const float*>(uintptr_t(256 + i * 4));
                down[i].input = reinterpret_cast<const float*>(uintptr_t(512 + i * 4));
                down[i].output = reinterpret_cast<float*>(uintptr_t(1024 + i * 4));
                down[i].pair_layout = i % 3 == 0;
                down[i].mma_layout = i % 2;
                down[i].input_global = float(i % 7 + 1);
                down[i].input_global_second = float(-i % 11 - 1);
            }
            order[i] = i;
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return reinterpret_cast<uintptr_t>(gate[a].weight) <
                   reinterpret_cast<uintptr_t>(gate[b].weight);
        });
        Job* device = nullptr;
        cudaStream_t stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        mollm_cuda::Nvfp4MmaBatch* plan = nullptr;
        void* radix_scratch = nullptr;
        size_t radix_bytes = 0;
        std::vector<mollm_cuda::Nvfp4MmaBatch> actual_plan(count);
        bool ok = cudaMalloc(reinterpret_cast<void**>(&device), 4 * count * sizeof(Job)) == cudaSuccess &&
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess;
        if (ok && paired) ok = cudaMalloc(reinterpret_cast<void**>(&plan), count * sizeof(*plan)) == cudaSuccess;
        if (ok && radix) ok = mollm_cuda::nvfp4_radix_group_scratch_size(count, &radix_bytes) == cudaSuccess &&
            cudaMalloc(&radix_scratch, radix_bytes) == cudaSuccess;
        if (ok) ok = cudaMemcpyAsync(device, gate.data(), count * sizeof(Job), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemcpyAsync(device + count, down.data(), count * sizeof(Job), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        if (ok) {
            if (radix) {
                ok = mollm_cuda::launch_group_nvfp4_jobs_radix(device, device + count,
                    device + 2 * count, device + 3 * count, count, radix_scratch,
                    radix_bytes - 1, stream) == cudaErrorInvalidValue &&
                    mollm_cuda::launch_group_nvfp4_jobs_radix(device, device + count,
                        device + 2 * count, device + 3 * count, count,
                        static_cast<uint8_t*>(radix_scratch) + 1, radix_bytes, stream) == cudaErrorInvalidValue &&
                    mollm_cuda::launch_group_nvfp4_jobs_radix(device, device + count,
                        device + 2 * count, device + 2 * count, count,
                        radix_scratch, radix_bytes, stream) == cudaErrorInvalidValue &&
                    mollm_cuda::launch_group_nvfp4_jobs_radix(device, device + count,
                        device, device + 3 * count, count,
                        radix_scratch, radix_bytes, stream) == cudaErrorInvalidValue &&
                    mollm_cuda::launch_group_nvfp4_jobs_radix(device, device + count,
                        device + 2 * count, device + 3 * count, count,
                        device, radix_bytes, stream) == cudaErrorInvalidValue;
            }
        }
        if (ok) {
            ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
            if (ok) {
                if (radix) ok = mollm_cuda::launch_group_nvfp4_jobs_radix(device, device + count,
                    device + 2 * count, device + 3 * count, count, radix_scratch, radix_bytes, stream) == cudaSuccess;
                else mollm_cuda::launch_group_nvfp4_jobs(device, device + count,
                    device + 2 * count, device + 3 * count, count, stream, plan, tiled);
                const auto ended = cudaStreamEndCapture(stream, &graph);
                ok = ok && ended == cudaSuccess;
            }
            if (ok) ok = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) == cudaSuccess;
            for (int replay = 0; ok && replay < 3; ++replay) {
                // Replay with changed keys and layout metadata, not just outputs.
                for (int i = 0; i < count; ++i) {
                    uintptr_t key = 1 + ((replay ? count - 1 - i : i) * 37) % 113;
                    if (replay == 1) {
                        if (i % 3 == 0) key |= uintptr_t(1) << (sizeof(uintptr_t) * 8 - 1);
                        if (i % 5 == 0) key |= uintptr_t(1) << (sizeof(uintptr_t) * 4);
                    } else if (replay == 2) key = 0;  // all-equal keys must stay stable
                    gate[i].weight = reinterpret_cast<const uint8_t*>(key);
                    gate[i].row_scales = replay && i % 257 == 0
                        ? reinterpret_cast<const float*>(uintptr_t(4)) : nullptr;
                    order[i] = i;
                }
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    return reinterpret_cast<uintptr_t>(gate[a].weight) < reinterpret_cast<uintptr_t>(gate[b].weight);
                });
                ok = cudaMemcpyAsync(device, gate.data(), count * sizeof(Job), cudaMemcpyHostToDevice, stream) == cudaSuccess;
                if (!ok) break;
                ok = cudaMemsetAsync(device + 2 * count, 0xff, 2 * count * sizeof(Job), stream) == cudaSuccess &&
                    cudaGraphLaunch(executable, stream) == cudaSuccess &&
                    cudaMemcpyAsync(actual.data(), device + 2 * count, 2 * count * sizeof(Job), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                    cudaStreamSynchronize(stream) == cudaSuccess;
                for (int i = 0; ok && i < count; ++i)
                    ok = same_job(actual[i], gate[order[i]]) &&
                        same_job(actual[count + i], down[order[i]]);
                if (ok && paired) {
                    ok = cudaMemcpyAsync(actual_plan.data(), plan, count * sizeof(*plan), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                        cudaStreamSynchronize(stream) == cudaSuccess;
                    for (int start = 0; ok && start < count;) {
                        int end = start + 1;
                        bool compatible = true;
                        while (end < count && gate[order[end]].weight == gate[order[start]].weight) {
                            compatible &= gate[order[end]].row_scales == gate[order[start]].row_scales;
                            ++end;
                        }
                        for (int i = start; ok && i < end; ++i) {
                            const int first = !compatible ? i : (i - start) % 2 ? -1 : i;
                            const int size = !compatible ? 1 : (i - start) % 2 ? 0 : i + 1 < end ? 2 : 1;
                            ok = actual_plan[i].first == first && actual_plan[i].size == size;
                        }
                        start = end;
                    }
                }
            }
        }
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        if (device) cudaFree(device);
        if (plan) cudaFree(plan);
        if (radix_scratch) cudaFree(radix_scratch);
        if (!ok) {
            std::fprintf(stderr, "NVFP4 stable sort stream/graph mismatch method=%d paired=%d count=%d\n", method, paired, count);
            return false;
        }
    }
    return true;
}

bool test_conservative_shortconv_graph(CudaBackend& backend, bool ple_conv = false,
                                      bool returned_view = false, bool owned_state = false) {
    std::vector<std::vector<float>> outputs, states, view_outputs, prefix_states;
    const char* prefix_flag=std::getenv("MOLLM_CUDA_MTP_STATE_PREFIXES");
    const bool prefixes=!ple_conv && !owned_state && prefix_flag && !std::strcmp(prefix_flag,"1");
    for (int mode : {0,1,2}) {
        const bool planned=mode!=0,contract=mode==2;
        backend.clear_dispatch_error();
        Graph graph; graph.nodes.resize(9 + (owned_state ? 2 : 0) + (returned_view ? 1 : 0));
        graph.runtime.tensors.resize(graph.nodes.size());
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            auto& node = graph.nodes[i]; node.id = i;
            node.out_shape[0] = 4; node.out_shape[1] = 3;
            node.dim_expr[1].kind = DIM_SEQ;
        }
        for (int i = 0; i < 3; ++i) graph.nodes[i].op_type = OpType::INPUT;
        graph.nodes[1].dim_expr[1].kind = DIM_CONST;
        graph.nodes[2].dim_expr[1].kind = DIM_CONST;
        graph.nodes[3].op_type = OpType::ADD; graph.nodes[3].inputs = {0, 0};
        graph.nodes[4].op_type = OpType::RESHAPE; graph.nodes[4].inputs = {3};
        graph.nodes[5].op_type = ple_conv ? OpType::PLE_DILATED_CONV : OpType::SHORTCONV;
        graph.nodes[5].inputs = {4, 1, 2};
        graph.nodes[5].params.i32 = ple_conv ? std::vector<int32_t>{3, 2, 3} : std::vector<int32_t>{4, 3};
        const bool reclaim=contract && backend.independent_stateful_mutable_input(graph.nodes[5])==2;
        graph.nodes[6].op_type = OpType::ADD; graph.nodes[6].inputs = {0, 0};
        graph.nodes[7].op_type = OpType::ADD; graph.nodes[7].inputs = {6, 6};
        graph.nodes[8].op_type = OpType::ADD; graph.nodes[8].inputs = {5, 5};
        graph.graph_inputs = {0, 1, 2}; graph.graph_outputs = {8};
        if (owned_state) {
            // Unlike the read-only activation view, this alias and its freshly
            // materialized owner must survive the state-mutating dispatch.
            auto& owner = graph.nodes[9];
            owner.op_type = OpType::ADD; owner.inputs = {2, 2};
            owner.out_shape[0] = ple_conv ? 4 : 3; owner.out_shape[1] = 4;
            owner.dim_expr[1].kind = DIM_CONST;
            graph.nodes[10] = owner; graph.nodes[10].id = 10;
            graph.nodes[10].op_type = OpType::RESHAPE; graph.nodes[10].inputs = {9};
            graph.nodes[5].inputs[2] = 10;
            // Node IDs index tensors, whereas node order is topological.
            std::rotate(graph.nodes.begin()+3, graph.nodes.begin()+9, graph.nodes.begin()+11);
        }
        if(returned_view) {
            auto& view=graph.nodes.back();
            view.op_type=OpType::RESHAPE;view.inputs={5};
            graph.graph_outputs.push_back(view.id);
        }
        auto& tensors = graph.runtime.tensors;
        tensors[1] = device_tensor(backend, 4, ple_conv ? 3 : 4);
        tensors[2] = device_tensor(backend, ple_conv ? 4 : 3, 4);
        std::vector<float> weights(ple_conv ? 12 : 16), state(ple_conv ? 16 : 12);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = .03f * (i + 1);
        for (size_t i = 0; i < state.size(); ++i) state[i] = .01f * i;
        if (!upload(backend, tensors[1], weights) || !upload(backend, tensors[2], state)) return false;
        BufferPool pool; ExecContext ctx;
        ctx.graph = &graph; ctx.backend = &backend; ctx.pool = &pool;
        ctx.conservative_stateful_liveness = planned;
        ctx.stateful_readonly_last_use = contract;
        prepare_execution(ctx);
        for (int pass = 0; pass < 3; ++pass) {
            const int rows = pass == 1 ? 5 : 3;
            if (tensors[0].data) backend.free_output(tensors[0], nullptr);
            tensors[0] = device_tensor(backend, 4, rows);
            std::vector<float> input(4 * rows);
            for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(i * .2f + pass);
            if (!upload(backend, tensors[0], input)) return false;
            ctx.runtime_seq_len = rows; inject_runtime_shapes(ctx);
            if (owned_state) {
                // Stop after the materialized state + alias, before the
                // convolution. A partial call must also preserve both. The
                // subsequent full call rematerializes, so do not bind prefix
                // snapshots across those two execution lifetimes.
                backend.begin_graph(); execute_graph(ctx,4); backend.end_graph();
                if (ctx.execution_failed || !tensors[9].data ||
                    tensors[10].data!=tensors[9].data) return false;
            }
            if (prefixes) {
                if (!backend.begin_state_prefixes({&tensors[owned_state ? 10 : 2]},rows)) return false;
            }
            backend.begin_graph(); execute_graph(ctx); backend.end_graph();
            std::vector<float> result, final_state, view_result;
            if (ctx.execution_failed || backend.dispatch_failed() ||
                !download(backend, tensors[8], result) ||
                !download(backend, tensors[owned_state ? 10 : 2], final_state)) return false;
            if (owned_state && (!tensors[9].data || tensors[10].data!=tensors[9].data)) return false;
            if(returned_view && !download(backend,tensors[graph.graph_outputs.back()],view_result))return false;
            if (!planned) { outputs.push_back(result); states.push_back(final_state);view_outputs.push_back(view_result); }
            else if (result.size() != outputs[pass].size() || final_state.size() != states[pass].size() ||
                     std::memcmp(result.data(), outputs[pass].data(), result.size() * sizeof(float)) ||
                     std::memcmp(final_state.data(), states[pass].data(), final_state.size() * sizeof(float)) ||
                     (returned_view && (view_result.size()!=view_outputs[pass].size() ||
                         std::memcmp(view_result.data(),view_outputs[pass].data(),view_result.size()*sizeof(float)))) ||
                     (reclaim ? (tensors[3].data || tensors[4].data ||
                         (returned_view ? !tensors[5].data : tensors[5].data!=nullptr))
                                             : (!tensors[3].data || !tensors[5].data)) ||
                     tensors[6].data || tensors[7].data) return false;
            if (prefixes) {
                // Return each accepted prefix after read-only operands have
                // been released, then restore the full prefix for the next pass.
                for (int kept=1;kept<=rows;++kept) {
                    std::vector<float> restored;
                    if (!backend.restore_state_prefix(kept) ||
                        !download(backend,tensors[owned_state ? 10 : 2],restored)) return false;
                    if (mode==0) prefix_states.push_back(restored);
                    else {
                        const size_t index=(pass==0 ? 0 : pass==1 ? 3 : 8)+kept-1;
                        if (restored.size()!=prefix_states[index].size() ||
                            std::memcmp(restored.data(),prefix_states[index].data(),restored.size()*sizeof(float))) return false;
                    }
                }
                backend.end_state_prefixes();
                std::vector<float> restored_full;
                if (!download(backend,tensors[owned_state ? 10 : 2],restored_full) || restored_full!=final_state) return false;
            }
        }
        // Node4 is a borrowed view; release only owners that remain live.
        for (int id : {0, 1, 2, 3, 5, 6, 7, 8})
            if (tensors[id].data) backend.free_output(tensors[id], nullptr);
        if (owned_state && tensors[9].data) backend.free_output(tensors[9], nullptr);
    }
    std::printf("conservative_convolution_graph ple_conv=%d returned_view=%d owned_state=%d prefixes=%d contract_modes=3 exact_output_state=1 changed_shapes=3 released_branch=1\n", ple_conv,returned_view,owned_state,prefixes);
    return true;
}

bool test_conservative_sdpa_graph(CudaBackend& backend, bool fp16_cache = false, int dim = 4, bool contiguous_view = false) {
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<uint8_t>> caches;
    constexpr int capacity = 16;
    for (bool planned : {false, true}) {
        backend.clear_dispatch_error();
        Graph graph; graph.nodes.resize(12); graph.runtime.tensors.resize(12);
        for (int i = 0; i < 12; ++i) {
            auto& node = graph.nodes[i]; node.id = i;
            node.out_shape[0] = dim; node.out_shape[1] = 3;
            node.dim_expr[1].kind = DIM_SEQ;
        }
        for (int i = 0; i < 6; ++i) graph.nodes[i].op_type = OpType::INPUT;
        for (int i : {3, 4, 5}) graph.nodes[i].dim_expr[1].kind = DIM_CONST;
        graph.nodes[6].op_type = OpType::ADD; graph.nodes[6].inputs = {0, 0};
        graph.nodes[7].op_type = contiguous_view ? OpType::CONTIGUOUS : OpType::RESHAPE;
        graph.nodes[7].inputs = {6};
        graph.nodes[8].op_type = OpType::SDPA;
        graph.nodes[8].inputs = {7, 1, 2, 3, 4, 5};
        graph.nodes[8].params.i32 = {2, 1, 1, 1, dim, dim};
        graph.nodes[8].params.f32 = {.5f};
        graph.nodes[9].op_type = OpType::ADD; graph.nodes[9].inputs = {0, 0};
        graph.nodes[10].op_type = OpType::ADD; graph.nodes[10].inputs = {9, 9};
        graph.nodes[11].op_type = OpType::ADD; graph.nodes[11].inputs = {8, 8};
        graph.graph_inputs = {0, 1, 2, 3, 4, 5}; graph.graph_outputs = {11};
        auto& tensors = graph.runtime.tensors;
        const size_t element_bytes = fp16_cache ? sizeof(uint16_t) : sizeof(float);
        const size_t payload_bytes = dim * capacity * element_bytes;
        const size_t bytes = CacheMetadata::SIZE + payload_bytes;
        for (int id : {4, 5}) {
            tensors[id] = Tensor::create(fp16_cache ? Precision::FP16 : Precision::FP32,
                MemoryType::EXTERNAL, bytes / element_bytes);
            backend.alloc_persistent(tensors[id], bytes,
                PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX, CacheMetadata::SIZE);
            std::vector<uint8_t> zero(payload_bytes, 0);
            if (!backend.copy_from_host(zero.data(), tensors[id], zero.size(), CacheMetadata::SIZE)) return false;
        }
        BufferPool pool; ExecContext ctx;
        ctx.graph = &graph; ctx.backend = &backend; ctx.pool = &pool;
        ctx.conservative_stateful_liveness = planned; prepare_execution(ctx);
        int past = 0;
        for (int pass = 0; pass < 3; ++pass) {
            const int rows = pass == 1 ? 5 : 3;
            for (int id : {0, 1, 2}) {
                if (tensors[id].data) backend.free_output(tensors[id], nullptr);
                tensors[id] = device_tensor(backend, dim, rows);
                std::vector<float> values(dim * rows);
                for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(i * .2f + pass + id);
                if (!upload(backend, tensors[id], values)) return false;
            }
            CacheMetadata metadata{};
            metadata.current_seq_len = past; metadata.max_seq_len = capacity;
            metadata.num_kv_heads = 1; metadata.head_dim = dim; metadata.v_head_dim = dim;
            for (int id : {4, 5}) if (!backend.copy_from_host(&metadata, tensors[id], sizeof(metadata))) return false;
            ctx.runtime_seq_len = rows; inject_runtime_shapes(ctx);
            backend.begin_graph(); execute_graph(ctx); backend.end_graph();
            std::vector<float> result;
            std::vector<uint8_t> payload(2 * payload_bytes);
            if (ctx.execution_failed || backend.dispatch_failed() || !download(backend, tensors[11], result)) return false;
            for (int id : {4, 5}) if (!backend.copy_to_host(tensors[id],
                payload.data() + (id - 4) * payload_bytes, payload_bytes, CacheMetadata::SIZE)) return false;
            if (!planned) { outputs.push_back(result); caches.push_back(payload); }
            else if (result.size() != outputs[pass].size() ||
                std::memcmp(result.data(), outputs[pass].data(), result.size() * sizeof(float)) ||
                std::memcmp(payload.data(), caches[pass].data(), payload.size()) ||
                !tensors[6].data || !tensors[8].data || tensors[9].data || tensors[10].data) return false;
            past += rows;
        }
        for (int id : {0, 1, 2, 6, 8, 9, 10, 11})
            if (tensors[id].data) backend.free_output(tensors[id], nullptr);
    }
    std::printf("conservative_sdpa_graph dim=%d fp16_cache=%d contiguous_view=%d exact_output_kv=1 changed_shapes=3 released_branch=1\n", dim, fp16_cache, contiguous_view);
    return true;
}

bool test_conservative_materialized_view_chain(CudaBackend& backend, bool dynamic = false) {
    std::vector<std::vector<float>> reference;
    for (bool planned : {false, true}) {
        Graph graph; graph.nodes.resize(8); graph.runtime.tensors.resize(8);
        const int64_t shapes[8][4] = {{8,3,1,1}, {8,3,1,1}, {4,3,1,1},
            {2,2,3,1}, {2,3,2,1}, {4,3,1,1}, {2,6,1,1}, {2,6,1,1}};
        for (int i = 0; i < 8; ++i) {
            graph.nodes[i].id = i;
            std::copy(shapes[i], shapes[i] + 4, graph.nodes[i].out_shape);
            if (dynamic) {
                const int axis = i == 3 ? 2 : 1;
                graph.nodes[i].dim_expr[axis].kind = i >= 6 ? DIM_MUL : DIM_SEQ;
                graph.nodes[i].dim_expr[axis].coeff = i >= 6 ? 2 : 0;
            }
        }
        graph.nodes[0].op_type = OpType::INPUT;
        graph.nodes[1].op_type = OpType::ADD; graph.nodes[1].inputs = {0,0};
        graph.nodes[2].op_type = OpType::SLICE; graph.nodes[2].inputs = {1};
        graph.nodes[2].params.i32 = {0,2,4};
        graph.nodes[3].op_type = OpType::RESHAPE; graph.nodes[3].inputs = {2};
        graph.nodes[4].op_type = OpType::PERMUTE; graph.nodes[4].inputs = {3};
        graph.nodes[4].params.i32 = {0,2,1,3};
        // Same-sized allocation between the borrowed view and its consumer:
        // an early release of materialized reshape 3 corrupts view 4.
        graph.nodes[5].op_type = OpType::SIGMOID_MUL; graph.nodes[5].inputs = {2,2};
        graph.nodes[6].op_type = OpType::RESHAPE; graph.nodes[6].inputs = {4};
        graph.nodes[7].op_type = OpType::ADD; graph.nodes[7].inputs = {6,6};
        graph.graph_inputs = {0}; graph.graph_outputs = {7};
        auto& tensors = graph.runtime.tensors;
        tensors[0] = device_tensor(backend,8,3);
        BufferPool pool; ExecContext ctx;
        ctx.graph = &graph; ctx.backend = &backend; ctx.pool = &pool;
        ctx.conservative_stateful_liveness = planned;
        ctx.reuse_static_workspace = !planned;
        prepare_execution(ctx);
        if (planned) {
            for (size_t i = 0; i < 6; ++i)
                if (std::find(ctx.release_queue[i].begin(), ctx.release_queue[i].end(), 3) !=
                    ctx.release_queue[i].end()) {
                    std::fprintf(stderr,"materialized_view_chain premature_release node=3 at=%zu\n",i);
                    return false;
                }
        }
        for (int pass = 0; pass < 3; ++pass) {
            const int rows = dynamic && pass == 1 ? 5 : 3;
            if (dynamic) {
                backend.free_output(tensors[0], nullptr);
                tensors[0] = device_tensor(backend, 8, rows);
                ctx.runtime_seq_len = rows;
                inject_runtime_shapes(ctx);
            }
            std::vector<float> values(8 * rows), actual;
            for (size_t i = 0; i < values.size(); ++i) values[i] = float(i + pass * 31) / 8.f;
            if (!upload(backend,tensors[0],values)) return false;
            backend.begin_graph(); execute_graph(ctx); backend.end_graph();
            if (ctx.execution_failed || backend.dispatch_failed() ||
                !download(backend,tensors[7],actual)) return false;
            if (actual.size() != size_t(4 * rows)) return false;
            for (int h = 0; h < 2; ++h)
                for (int s = 0; s < rows; ++s)
                    for (int d = 0; d < 2; ++d)
                        if (actual[d + 2 * (s + rows * h)] !=
                            4.f * values[2 + d + 2 * h + 8 * s]) {
                            std::fprintf(stderr, "reshape analytic mismatch dynamic=%d planned=%d pass=%d d=%d s=%d h=%d actual=%g expected=%g\n",
                                dynamic, planned, pass, d, s, h, actual[d + 2 * (s + rows * h)],
                                4.f * values[2 + d + 2 * h + 8 * s]);
                            return false;
                        }
            if (!planned && !dynamic) {
                size_t strides[4];
                const bool view = backend.reshape_view_strides(tensors[2], tensors[3], strides);
                if (tensors[3].shares_storage_with(tensors[2]) != view ||
                    tensors[6].shares_storage_with(tensors[4])) {
                    std::fprintf(stderr, "reshape alias mismatch dynamic=%d planned=%d pass=%d view=%d\n", dynamic, planned, pass, view);
                    return false;
                }
            }
            if (!planned) reference.push_back(actual);
            else if (actual.size() != reference[pass].size() ||
                std::memcmp(actual.data(),reference[pass].data(),actual.size()*sizeof(float))) {
                std::fprintf(stderr,"materialized_view_chain mismatch pass=%d\n",pass);
                return false;
            }
        }
        const bool borrowed3 = tensors[3].data && tensors[3].shares_storage_with(tensors[2]);
        for (int id : {0,1,3,5,6,7})
            if (!(id == 3 && borrowed3))
            if (tensors[id].data) backend.free_output(tensors[id],nullptr);
        backend.release_graph_plan(graph);
    }
    std::printf("conservative_materialized_view_chain exact_output=1 changed_inputs=3 dynamic=%d\n", dynamic);
    return true;
}

bool test_conservative_ple_graph(CudaBackend&) {
    CudaBackend backend;
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::ALLOW_REFERENCE);
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int32_t>> histories;
    for (bool planned : {false, true}) {
        Graph graph; graph.nodes.resize(11); graph.runtime.tensors.resize(11);
        for (int i = 0; i < 11; ++i) {
            auto& node = graph.nodes[i]; node.id = i;
            node.out_shape[0] = 4; node.out_shape[1] = 4;
        }
        for (int i = 0; i < 7; ++i) graph.nodes[i].op_type = OpType::INPUT;
        graph.nodes[7].op_type = OpType::PLE_LOOKUP;
        graph.nodes[7].inputs = {0, 1, 2, 3, 4, 5};
        graph.nodes[7].params.i32 = {3, 1, 0, 100, 0, 1234, 0, 4};
        graph.nodes[8].op_type = OpType::ADD; graph.nodes[8].inputs = {6, 6};
        graph.nodes[9].op_type = OpType::ADD; graph.nodes[9].inputs = {8, 8};
        graph.nodes[10].op_type = OpType::ADD; graph.nodes[10].inputs = {7, 7};
        graph.graph_outputs = {10};
        int32_t ids[4] = {4, 0, 5, 6}, history[3] = {1, 2, 1};
        int32_t sizes[2] = {7, 11}, offsets[2] = {0, 7};
        uint8_t table[36]; float scale = .5f;
        for (int i = 0; i < 36; ++i) table[i] = uint8_t(0x30 + i % 8);
        auto& tensors = graph.runtime.tensors;
        tensors[0] = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 4, 1, 1, 1, ids);
        tensors[1] = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 3, 1, 1, 1, history);
        tensors[2] = Tensor::create(Precision::RAW_U8, MemoryType::EXTERNAL, 2, 18, 1, 1, table);
        tensors[3] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1, &scale);
        tensors[4] = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 2, 1, 1, 1, sizes);
        tensors[5] = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, 2, 1, 1, 1, offsets);
        tensors[6] = device_tensor(backend, 4, 4);
        if (!upload(backend, tensors[6], std::vector<float>(16, .25f))) return false;
        BufferPool pool; ExecContext ctx;
        ctx.graph = &graph; ctx.backend = &backend; ctx.pool = &pool;
        ctx.conservative_stateful_liveness = planned;
        ctx.static_padded = true; ctx.padded_seq_len = 4;
        prepare_execution(ctx);
        for (int pass = 0; pass < 3; ++pass) {
            ids[0] = 4 + pass; ids[2] = 5 - pass;
            ctx.runtime_seq_len = pass == 1 ? 4 : 3;
            inject_runtime_shapes(ctx);
            backend.begin_graph(); execute_graph(ctx); backend.end_graph();
            std::vector<float> result;
            if (ctx.execution_failed || backend.dispatch_failed() || !download(backend, tensors[10], result)) return false;
            std::vector<int32_t> current(history, history + 3);
            if (!planned) { outputs.push_back(result); histories.push_back(current); }
            else if (result.size() != outputs[pass].size() ||
                std::memcmp(result.data(), outputs[pass].data(), result.size() * sizeof(float)) ||
                current != histories[pass] || !tensors[7].data || tensors[8].data || tensors[9].data) return false;
        }
        for (int id : {6, 7, 8, 9, 10})
            if (tensors[id].data) backend.free_output(tensors[id], nullptr);
    }
    std::puts("conservative_ple_graph host_table=1 exact_output_history=1 changed_real_tokens=3 released_branch=1");
    return true;
}

bool test_sigmoid_gr_reduce_graph(CudaBackend& backend) {
    constexpr int hidden = 19, streams = 4, wide = hidden * streams;
    Graph graph; graph.nodes.resize(5); graph.runtime.tensors.resize(5);
    for (int i = 0; i < 5; ++i) {
        auto& node = graph.nodes[i]; node.id = i;
        node.out_shape[0] = i == 4 ? hidden : wide;
        node.out_shape[1] = 3; node.dim_expr[1].kind = DIM_SEQ;
        graph.runtime.tensors[i] = Tensor::create(Precision::FP32,
            MemoryType::NONE, node.out_shape[0], 3, 1, 1, nullptr);
    }
    graph.nodes[0].op_type = graph.nodes[1].op_type = OpType::INPUT;
    graph.nodes[2].op_type = OpType::ADD; graph.nodes[2].inputs = {1, 1};
    graph.nodes[3].op_type = OpType::SIGMOID; graph.nodes[3].inputs = {2};
    graph.nodes[4].op_type = OpType::GR_REDUCE;
    graph.nodes[4].inputs = {0, 3}; graph.nodes[4].params.i32 = {hidden, streams};
    graph.graph_inputs = {0, 1}; graph.graph_outputs = {4};
    BufferPool pool; ExecContext ctx;
    ctx.graph = &graph; ctx.pool = &pool; ctx.backend = &backend;
    prepare_execution(ctx);
    for (int rows : {3, 5, 1, 5}) {
        std::vector<float> x(wide * rows), logits(wide * rows), doubled(wide * rows);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = std::sin(float(i) * .17f + rows);
            logits[i] = std::cos(float(i) * .11f + rows);
            doubled[i] = logits[i] + logits[i];
        }
        for (int id : {0, 1}) {
            if (graph.runtime.tensors[id].data) backend.free_output(graph.runtime.tensors[id], nullptr);
            graph.runtime.tensors[id] = device_tensor(backend, wide, rows);
        }
        if (!upload(backend, graph.runtime.tensors[0], x) ||
            !upload(backend, graph.runtime.tensors[1], logits)) return false;
        ctx.runtime_seq_len = rows; inject_runtime_shapes(ctx);
        backend.begin_graph(); execute_graph(ctx); backend.end_graph();
        std::vector<float> actual;
        if (ctx.execution_failed || backend.dispatch_failed() ||
            !download(backend, graph.runtime.tensors[4], actual)) return false;
        Tensor raw = device_tensor(backend, wide, rows);
        Tensor gate = device_tensor(backend, wide, rows);
        Tensor output = device_tensor(backend, hidden, rows);
        if (!upload(backend, raw, doubled)) return false;
        backend.begin_graph();
        backend.dispatch(graph.nodes[3], {&raw}, &gate, nullptr);
        backend.dispatch(graph.nodes[4], {&graph.runtime.tensors[0], &gate}, &output, nullptr);
        backend.end_graph();
        std::vector<float> expected;
        if (backend.dispatch_failed() || !download(backend, output, expected) ||
            actual.size() != expected.size() ||
            std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float))) {
            std::fprintf(stderr, "sigmoid GR graph bit-exact mismatch rows=%d\n", rows);
            return false;
        }
        if (rows > 1 && backend.supports_sigmoid_gr_reduce_fusion() &&
            graph.runtime.tensors[3].data) return false;
        for (Tensor* tensor : {&raw, &gate, &output}) backend.free_output(*tensor, nullptr);
    }
    for (auto& tensor : graph.runtime.tensors)
        if (tensor.data) backend.free_output(tensor, nullptr);
    return true;
}

}  // namespace

bool test_shared_fp16_cache(CudaBackend& backend) {
    const char* trace_flag = std::getenv("MOLLM_TEST_CUDA_SHARED_CACHE_TRACE");
    const bool trace = trace_flag && !std::strcmp(trace_flag,"1");
    const auto stage = [&](const char* label,int n=0,int k=0,int rows=0,int pass=0) {
        if (trace) std::fprintf(stderr,"shared_cache_stage=%s n=%d k=%d rows=%d pass=%d\n",label,n,k,rows,pass);
    };
    stage("begin");
    if (!backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE)) return false;
    const char* flag = std::getenv("MOLLM_CUDA_PREFILL_FP16_CACHE_SHARED");
    const char* budget = std::getenv("MOLLM_CUDA_PREFILL_FP16_CACHE_MB");
    const bool enabled = flag && !std::strcmp(flag, "1");
    const size_t limit = budget ? size_t(std::strtoull(budget, nullptr, 10)) * 1024 * 1024 : 0;
    const bool small = std::getenv("MOLLM_TEST_CUDA_SHARED_CACHE_SMALL") != nullptr;
    const std::vector<int> row_cases = small ? std::vector<int>{17} : std::vector<int>{17,719,1743};
    std::vector<std::vector<Q4B8G32Block>> sources;
    std::vector<std::vector<mollm::cpu::fp16_t>> dense_sources;
    size_t expected_bytes = backend.prefill_shared_cache_bytes(), expected_hits = backend.prefill_shared_cache_hits();
    for (auto shape : {std::pair<int,int>{640,2560}, {2560,640}, {512,2560}}) {
        const int n = shape.first, k = shape.second, groups = k / 32;
        sources.emplace_back(size_t(n / 8) * groups);
        dense_sources.emplace_back(size_t(n) * k);
        auto& blocks = sources.back(); auto& dense = dense_sources.back();
        for (int row = 0; row < n; ++row) for (int group = 0; group < groups; ++group) {
            auto& b = blocks[size_t(row / 8) * groups + group];
            const float scale = (1 + (row + group) % 13) / 7.f;
            b.scales[row % 8] = scale;
            for (int pair = 0; pair < 16; ++pair) {
                const unsigned low = (row + group * 7 + pair * 3) % 16;
                const unsigned high = (row * 3 + group + pair * 11) % 16;
                b.q[row % 8][pair] = uint8_t((low ^ 8u) | ((high ^ 8u) << 4));
                dense[size_t(row) * k + group * 32 + 2 * pair] = mollm::cpu::fp16_t((int(low)-8)*scale);
                dense[size_t(row) * k + group * 32 + 2 * pair + 1] = mollm::cpu::fp16_t((int(high)-8)*scale);
            }
        }
        Tensor q = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, n, k, 1, 1, blocks.data());
        q.rowmajor_data = blocks.data(); q.q4_g32_data = blocks.data(); q.is_q4_g32_packed = true;
        stage("wrap_q4",n,k);
        q.group_size = 32; q.groups_per_row = groups; backend.wrap_weight_int4(q);
        Tensor h = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, n, k, 1, 1, dense.data());
        stage("wrap_fp16",n,k);
        backend.wrap_weight(h);
        stage("weights_ready",n,k);
        const size_t bytes = size_t(n) * k * 2;
        const bool cached = enabled && n != 512 && expected_bytes <= limit && bytes <= limit - expected_bytes;
        if (cached) expected_bytes += size_t(n) * k * 2;
        if (backend.prefill_shared_cache_bytes() != expected_bytes) {
            std::fprintf(stderr,"shared cache bytes mismatch actual=%zu expected=%zu\n",backend.prefill_shared_cache_bytes(),expected_bytes);
            return false;
        }
        for (int rows : row_cases) for (int pass = 0; pass < 3; ++pass) {
            stage("gemm_begin",n,k,rows,pass);
            std::vector<float> x(size_t(rows)*k), actual(size_t(rows)*n), reference(size_t(rows)*n);
            for (size_t i=0;i<x.size();++i) x[i] = float(int((i*7+pass*19)%97)-48)/83.f;
            Tensor dx = device_tensor(backend,k,rows), dy = device_tensor(backend,n,rows), dr = device_tensor(backend,n,rows);
            GraphNode gemm; gemm.op_type=OpType::MATMUL;
            if (!upload(backend,dx,x)) { stage("upload_failed",n,k,rows,pass); return false; }
            backend.clear_dispatch_error(); backend.dispatch(gemm,{&dx,&q},&dy,nullptr);
            if (backend.dispatch_failed()) { stage("q4_dispatch_failed",n,k,rows,pass); return false; }
            stage("q4_dispatched",n,k,rows,pass);
            backend.dispatch(gemm,{&dx,&h},&dr,nullptr);
            stage("fp16_dispatched",n,k,rows,pass);
            const bool okay = !backend.dispatch_failed() && download(backend,dy,actual) && download(backend,dr,reference);
            for (Tensor* t : {&dx,&dy,&dr}) backend.free_output(*t,nullptr);
            if (!okay ||
                actual.size()!=reference.size() || std::memcmp(actual.data(),reference.data(),actual.size()*4)) {
                std::fprintf(stderr,"shared cache/reference mismatch n=%d k=%d rows=%d pass=%d\n",n,k,rows,pass);
                return false;
            }
            if (cached) ++expected_hits;
            if (backend.prefill_shared_cache_hits()!=expected_hits ||
                backend.prefill_shared_cache_bytes()!=expected_bytes) {
                std::fprintf(stderr,"shared cache counters mismatch hits=%zu/%zu bytes=%zu/%zu\n",backend.prefill_shared_cache_hits(),expected_hits,backend.prefill_shared_cache_bytes(),expected_bytes);
                return false;
            }
            stage("gemm_exact",n,k,rows,pass);
        }
        stage("decode_begin",n,k);
        std::vector<float> x(k,.125f), decode(n);
        Tensor dx = device_tensor(backend,k), dy = device_tensor(backend,n);
        GraphNode gemm; gemm.op_type=OpType::MATMUL;
        if (!upload(backend,dx,x)) return false;
        backend.clear_dispatch_error(); backend.dispatch(gemm,{&dx,&q},&dy,nullptr);
        const bool okay = !backend.dispatch_failed() && download(backend,dy,decode);
        backend.free_output(dx,nullptr); backend.free_output(dy,nullptr);
        if (!okay || backend.prefill_shared_cache_hits()!=expected_hits) return false;
        stage("decode_done",n,k);
    }
    std::printf("shared_fp16_cache enabled=%d bytes=%zu hits=%zu shapes=3 rows=%s changed_inputs=3 exact_fp16_reference=1 decode_not_using_cache=1\n",
                enabled,expected_bytes,expected_hits,small?"17":"17,719,1743");
    return true;
}

int main(int argc, char** argv) {
    const bool shared_cache_only=argc==2 && !std::strcmp(argv[1],"--shared-cache-only");
    const bool reshape_only=argc==2 && !std::strcmp(argv[1],"--strided-reshape-only");
    const bool contracts_only=argc==2 && !std::strcmp(argv[1],"--stateful-contract-only");
    const bool head_group_only=argc==2 && !std::strcmp(argv[1],"--sdpa-head-group-only");
    const bool native_pv_only=argc==2 && !std::strcmp(argv[1],"--sdpa-native-pv-only");
    const bool full_batch_only=argc==2 && !std::strcmp(argv[1],"--sdpa-full-batch-only");
    if (argc!=1 && !contracts_only && !head_group_only && !native_pv_only && !full_batch_only && !reshape_only && !shared_cache_only) {
        std::fprintf(stderr,"usage: %s [--stateful-contract-only|--sdpa-head-group-only|--sdpa-native-pv-only|--sdpa-full-batch-only|--strided-reshape-only|--shared-cache-only]\n",argv[0]);
        return 2;
    }
    {
        const char* prior = std::getenv("MOLLM_CUDA_HOST_HEADROOM_MB");
        const bool had_prior = prior != nullptr;
        const std::string saved = prior ? prior : "";
        void* pointer = nullptr;
        setenv("MOLLM_CUDA_HOST_HEADROOM_MB", "bad", 1);
        const bool rejected_device = !mollm_cuda::malloc_device(&pointer, 4096, "test guard device") && !pointer;
        const bool rejected_managed = !mollm_cuda::malloc_managed(&pointer, 4096, "test guard managed") && !pointer;
        setenv("MOLLM_CUDA_HOST_HEADROOM_MB", "1048576", 1);
        const bool rejected_budget = !mollm_cuda::malloc_device(&pointer, SIZE_MAX, "test guard impossible") && !pointer;
        if (had_prior) setenv("MOLLM_CUDA_HOST_HEADROOM_MB", saved.c_str(), 1);
        else unsetenv("MOLLM_CUDA_HOST_HEADROOM_MB");
        if (!rejected_device || !rejected_managed || !rejected_budget) return 1;
        std::puts("cuda_host_headroom_guard rejected_device_managed=1 allocation_pointer_null=1");
    }
    CudaBackend backend;
    if (!backend.available()) {
        std::fprintf(stderr, "CUDA device unavailable; skipping\n");
        return 77;
    }
    if (shared_cache_only) return test_shared_fp16_cache(backend) ? 0 : 1;
    if (reshape_only) {
        if (!backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE)) return 1;
        Tensor source = Tensor::create(Precision::FP32, MemoryType::NONE, 4, 3);
        source.stride[1] = 32;
        Graph geometry;
        geometry.nodes.resize(5); geometry.runtime.tensors.resize(5);
        for (int i = 0; i < 5; ++i) geometry.nodes[i].id = i;
        geometry.nodes[0].op_type = OpType::INPUT;
        geometry.nodes[1].op_type = OpType::SLICE; geometry.nodes[1].inputs = {0};
        geometry.nodes[1].params.i32 = {0, 0, 4};
        geometry.nodes[2].op_type = OpType::RESHAPE; geometry.nodes[2].inputs = {1};
        geometry.nodes[3].op_type = OpType::PERMUTE; geometry.nodes[3].inputs = {2};
        geometry.nodes[3].params.i32 = {0, 2, 1, 3};
        geometry.nodes[4].op_type = OpType::RESHAPE; geometry.nodes[4].inputs = {3};
        geometry.graph_outputs = {4};
        Tensor& target = geometry.runtime.tensors[2];
        target = Tensor::create(Precision::FP32, MemoryType::NONE, 2, 2, 3);
        backend.prepare_graph(geometry);
        size_t strides[4] = {11, 22, 33, 44};
        const bool enabled = backend.reshape_view_strides(source, target, strides);
        GraphNode conv; conv.op_type = OpType::SHORTCONV; conv.inputs = {0, 1, 2};
        const char* flag = std::getenv("MOLLM_CUDA_STRIDED_RESHAPE_VIEWS");
        if (enabled != bool(flag && !std::strcmp(flag, "1") &&
                            backend.independent_stateful_mutable_input(conv) == 2)) return 1;
        if (enabled && (strides[0] != 4 || strides[1] != 8 || strides[2] != 32 || strides[3] != 96)) return 1;
        const Tensor original_target = target;
        for (int invalid = 0; invalid < 8; ++invalid) {
            Tensor s = source;
            target = original_target;
            if (invalid == 0) s.stride[0] = 8;
            if (invalid == 1) s.stride[1] = 8;
            if (invalid == 2) s.stride[1] = 31;
            if (invalid == 3) s.shape[2] = 2;
            if (invalid == 4) target.shape[0] = 3;
            if (invalid == 5) target.shape[2] = 4;
            if (invalid == 6) target.prec = Precision::FP16;
            if (invalid == 7) s.stride[1] = SIZE_MAX - 3;
            size_t untouched[4] = {11, 22, 33, 44};
            if (backend.reshape_view_strides(s, target, untouched) || untouched[0] != 11 ||
                untouched[1] != 22 || untouched[2] != 33 || untouched[3] != 44) return 1;
        }
        target = original_target;
        for (int rejected = 0; rejected < 4; ++rejected) {
            geometry.nodes[3].op_type = rejected == 0 ? OpType::ADD : OpType::PERMUTE;
            geometry.nodes[3].inputs = rejected == 0 ? std::vector<uint32_t>({2, 2}) : std::vector<uint32_t>({2});
            geometry.nodes[2].inputs = rejected == 3 ? std::vector<uint32_t>({0}) : std::vector<uint32_t>({1});
            geometry.graph_outputs = rejected == 1 ? std::vector<uint32_t>({2, 4}) :
                                     rejected == 2 ? std::vector<uint32_t>({3, 4}) : std::vector<uint32_t>({4});
            backend.prepare_graph(geometry);
            if (backend.reshape_view_strides(source, target, strides)) return 1;
        }
        backend.release_graph_plan(geometry);
        const auto before = backend.strided_reshape_view_calls();
        if (!test_conservative_materialized_view_chain(backend) ||
            !test_conservative_materialized_view_chain(backend, true) ||
            backend.strided_reshape_view_calls() - before != size_t(enabled ? 12 : 0)) return 1;
        std::printf("strided_reshape_views enabled=%d exact_analytic_output=1 static_dynamic_reuse=1 calls=%zu invalid=8\n",
                    enabled, backend.strided_reshape_view_calls() - before);
        return 0;
    }
    if(full_batch_only) {
        for(int rows:{1025,1743,2048,2049}) {
            const auto before=backend.attention_query_group_calls();
            if(!test_wide_cached_sdpa(backend,Precision::FP16,0,rows,true,1,256,true,2)||
                backend.attention_query_group_calls()-before!=size_t(rows<=2048))return 1;
            std::printf("CUDA SDPA logical batch rows=%d reference/cache checks PASS grouped=%d\n",rows,int(rows<=2048));
            std::fflush(stdout);
        }
        return 0;
    }
    if(native_pv_only) {
        if(!backend.set_logical_prefill_batch(true))return 1;
        const bool enabled=
            mollm_cuda::logical_prefill_option(std::getenv("MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX"),true)&&
            mollm_cuda::logical_prefill_option(std::getenv("MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV"),true)&&
            mollm_cuda::logical_prefill_option(std::getenv("MOLLM_CUDA_SDPA_PREFILL_PV_PAIR"),true);
        cudaDeviceProp properties{};int current_device=0;
        if(cudaGetDevice(&current_device)!=cudaSuccess||cudaGetDeviceProperties(&properties,current_device)!=cudaSuccess)return 1;
        const bool long_pv_qualified=properties.major==12&&properties.minor==1&&
            properties.maxThreadsPerBlock>=512&&properties.regsPerBlock>=32768;
        const bool actual_long_selected=long_pv_qualified;
        const int head_max_keys=16384;
        const bool head_pair=long_pv_qualified;
        const bool mid_pair=long_pv_qualified;
        size_t expected_score_peak=0;
        auto run=[&](int keys,int rows,int ratio,bool selected,int capacity=0) {
            expected_score_peak=std::max(expected_score_peak,size_t(2*ratio)*rows*keys*sizeof(float));
            const auto before=backend.attention_native_pv_calls();
            const auto head_before=backend.attention_head_lane_calls();
            return test_wide_cached_sdpa(backend,Precision::FP16,keys-rows,rows,true,2,256,true,ratio,capacity)&&
                backend.attention_score_workspace_bytes()==expected_score_peak&&
                backend.attention_native_pv_calls()-before==size_t(enabled&&selected)&&
                backend.attention_head_lane_calls()-head_before==size_t(enabled&&selected&&head_pair&&keys>4096&&keys<=head_max_keys);
        };
        const bool passed=run(4097,32,12,true)&&run(4097,33,12,true)&&
            run(8192,32,12,true)&&run(8193,32,12,true)&&
            run(16384,32,12,true,16384)&&run(16384,32,12,true)&&run(16385,32,12,actual_long_selected)&&
            run(32768,32,12,actual_long_selected,32768)&&
            run(4097,32,12,true,32768)&&run(16384,32,12,true,32768)&&run(4097,32,12,false,32769)&&
            run(4097,31,12,false)&&run(4096,32,12,mid_pair)&&run(4097,32,4,false)&&
            run(2080,32,12,mid_pair)&&run(3072,1024,12,mid_pair)&&
            run(3072,1025,12,false)&&run(2080,31,12,false)&&run(2048,32,12,false);
        if(!backend.set_logical_prefill_batch(false))return 1;
        const auto before=backend.attention_native_pv_calls();
        const auto ordinary_head_before=backend.attention_head_lane_calls();
        const bool ordinary=test_wide_cached_sdpa(backend,Precision::FP16,4065,32,true,2,256,true,12)&&
            backend.attention_native_pv_calls()==before&&backend.attention_head_lane_calls()==ordinary_head_before;
        if(!backend.set_logical_prefill_batch(true))return 1;
        const bool restored=run(4097,32,12,true);
        std::puts(passed&&ordinary&&restored?"CUDA native-probability PV selector/reference fixtures passed":"CUDA native-probability PV fixtures FAILED");
        return passed&&ordinary&&restored?0:1;
    }
    if(head_group_only) {
        constexpr bool enabled=true;
        auto run=[&](int past,int current,int dim,int ratio,bool selected) {
            const auto before=backend.attention_head_group_calls();
            return test_wide_cached_sdpa(backend,Precision::FP16,past,current,true,2,dim,true,ratio)&&
                backend.attention_head_group_calls()-before==size_t(enabled&&selected);
        };
        const bool narrow_passed=run(0,17,256,4,false)&&run(1007,17,256,4,false)&&
            run(1008,17,256,4,true)&&run(1280,17,256,2,false)&&
            run(1280,1,256,4,false)&&run(1280,17,128,4,false)&&
            run(4080,17,256,4,false)&&run(16381,3,256,4,false);
        const auto before_full=backend.attention_head_group_calls();
        const bool passed=narrow_passed&&
            test_wide_cached_sdpa(backend,Precision::FP16,0,1743,true,1,256,true,4)&&
            backend.attention_head_group_calls()-before_full==size_t(enabled);
        std::puts(passed?"CUDA long head-group selector/reference fixtures passed":"CUDA long head-group selector/reference fixtures FAILED");
        return passed?0:1;
    }
    {
        GraphNode conv;conv.op_type=OpType::SHORTCONV;conv.inputs={0,1,2};
        GraphNode gdn;gdn.op_type=OpType::GATED_DELTANET_PREFILL;gdn.inputs.resize(8);
        bool uncaptured=true;
#ifdef MOLLM_CUDA_DECODE_SEGMENTS
        uncaptured=std::getenv("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS")!=nullptr;
#endif
        if (backend.independent_stateful_mutable_input(conv)!=(uncaptured ? 2 : -1) ||
            backend.independent_stateful_mutable_input(gdn)!=(uncaptured ? 7 : -1)) return 1;
        std::printf("stateful_contract_capture_guard uncaptured=%d contracts_available=%d\n",uncaptured,uncaptured);
    }
    if (contracts_only) {
        const bool passed=test_conservative_shortconv_graph(backend) &&
            test_conservative_shortconv_graph(backend,true) &&
            test_conservative_shortconv_graph(backend,false,true) &&
            test_conservative_shortconv_graph(backend,false,false,true) &&
            test_conservative_shortconv_graph(backend,false,true,true) &&
            test_gdn(backend) && test_gdn(backend,true);
        std::puts(passed ? "CUDA stateful contract fixtures passed" : "CUDA stateful contract fixtures FAILED");
        return passed ? 0 : 1;
    }

    {
        const char* prior = std::getenv("MOLLM_CUDA_OUTPUT_POOL_MB");
        struct RestoreOption {
            bool had;
            std::string value;
            ~RestoreOption() {
                if (had) setenv("MOLLM_CUDA_OUTPUT_POOL_MB", value.c_str(), 1);
                else unsetenv("MOLLM_CUDA_OUTPUT_POOL_MB");
            }
        } restore{prior != nullptr, prior ? prior : ""};
        for (const char* bad : {"", "bad", "-1", "+1", " 1", "1 ", "18446744073709551615"}) {
            setenv("MOLLM_CUDA_OUTPUT_POOL_MB", bad, 1);
            CudaBackend invalid;
            if (invalid.available()) return 1;
        }
        setenv("MOLLM_CUDA_OUTPUT_POOL_MB", "1", 1);
        CudaBackend limited;
        if (!limited.available()) return 1;
        Tensor active, cached, larger, rejected;
        void* live = limited.alloc_output(active, 128 * 1024, nullptr);
        if (!live || !limited.alloc_output(cached, 512 * 1024, nullptr) ||
            cudaMemset(live, 0x5a, 128 * 1024) != cudaSuccess) return 1;
        limited.free_output(cached, nullptr);
        // 640KiB reserved +768KiB new exceeds1MiB; reclaim512KiB first.
        if (!limited.alloc_output(larger, 768 * 1024, nullptr) || limited.dispatch_failed()) return 1;
        std::vector<unsigned char> bytes(128 * 1024);
        if (cudaMemcpy(bytes.data(), live, bytes.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
            !std::all_of(bytes.begin(), bytes.end(), [](unsigned char x) { return x == 0x5a; })) return 1;
        // 896KiB active +256KiB cannot fit, even after all idle blocks free.
        if (limited.alloc_output(rejected, 256 * 1024, nullptr) || !limited.dispatch_failed() || rejected.data) return 1;
        std::puts("cuda_output_pool_limit reclaim=1 live_output_exact=1 active_over_limit_rejected=1 invalid_options_rejected=1");
    }

    {
        auto* pool = mollm_cuda::create_device_buffer_pool();
        if (!pool) return 1;
        struct PoolOwner {
            mollm_cuda::DeviceBufferPool* pool;
            ~PoolOwner() { mollm_cuda::destroy_device_buffer_pool(pool); }
        } owner{pool};
        void* active = mollm_cuda::acquire_device_buffer(pool, 4096);
        void* cached = mollm_cuda::acquire_device_buffer(pool, 8192);
        if (!active || !cached || cudaMemset(active, 0x5a, 4096) != cudaSuccess) return 1;
        mollm_cuda::release_device_buffer(pool, cached);
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return 1;
        size_t reclaimed = 123;
        const bool capture_rejected = !mollm_cuda::trim_free_device_buffers(pool, stream, &reclaimed) &&
            reclaimed == 0 && mollm_cuda::device_buffer_pool_reserved_bytes(pool) == 12288;
        cudaGraph_t graph = nullptr;
        const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
        if (graph) cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        if (!capture_rejected || ended != cudaSuccess ||
            !mollm_cuda::trim_free_device_buffers(pool, cudaStreamPerThread, &reclaimed) ||
            reclaimed != 8192 || mollm_cuda::device_buffer_pool_reserved_bytes(pool) != 4096) return 1;
        std::vector<unsigned char> bytes(4096);
        if (cudaMemcpy(bytes.data(), active, bytes.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
            !std::all_of(bytes.begin(), bytes.end(), [](unsigned char value) { return value == 0x5a; })) return 1;
        if (!mollm_cuda::trim_free_device_buffers(pool, cudaStreamPerThread, &reclaimed) || reclaimed != 0) return 1;
        mollm_cuda::release_device_buffer(pool, active);
        if (!mollm_cuda::trim_free_device_buffers(pool, cudaStreamPerThread, &reclaimed) || reclaimed != 4096 ||
            mollm_cuda::device_buffer_pool_reserved_bytes(pool) != 0) return 1;
        void* fresh = mollm_cuda::acquire_device_buffer(pool, 2048);
        if (!fresh) return 1;
        std::puts("cuda_output_pool_trim active_bytes_preserved=1 capture_rejected=1 reacquire=1");
    }

    // Preserve the test's early-return behavior, but identify the failing
    // phase even when a reference/helper check has no detailed diagnostic.
    struct FailurePhase {
        const char* name = "FP16 and device-resident operations";
        bool passed = false;
        ~FailurePhase() {
            if (!passed) std::fprintf(stderr, "CUDA backend test failed in phase: %s\n", name);
        }
    } phase;

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
    {
        constexpr int sliced_n = n / 2;
        Graph graph;
        GraphNode input_node;
        input_node.id = 0;
        input_node.op_type = OpType::INPUT;
        input_node.out_shape[0] = k;
        input_node.out_shape[1] = m;
        input_node.out_prec = Precision::FP32;
        GraphNode weight_node;
        weight_node.id = 1;
        weight_node.op_type = OpType::CONSTANT;
        weight_node.out_shape[0] = n;
        weight_node.out_shape[1] = k;
        weight_node.out_prec = Precision::FP16;
        GraphNode slice_weight;
        slice_weight.id = 2;
        slice_weight.op_type = OpType::SLICE;
        slice_weight.inputs = {1};
        slice_weight.out_shape[0] = sliced_n;
        slice_weight.out_shape[1] = k;
        slice_weight.out_prec = Precision::FP16;
        slice_weight.params.i32 = {0, sliced_n, sliced_n};
        GraphNode sliced_matmul;
        sliced_matmul.id = 3;
        sliced_matmul.op_type = OpType::MATMUL;
        sliced_matmul.inputs = {0, 2};
        sliced_matmul.out_shape[0] = sliced_n;
        sliced_matmul.out_shape[1] = m;
        sliced_matmul.out_prec = Precision::FP32;
        graph.nodes = {
            input_node, weight_node, slice_weight, sliced_matmul};
        graph.graph_inputs = {0};
        graph.graph_outputs = {3};
        graph.runtime.tensors.resize(4);
        Tensor sliced_input = device_tensor(backend, k, m);
        graph.runtime.tensors[0] = sliced_input;
        graph.runtime.tensors[1] = fp16;
        const size_t sliced_weight_offset =
            static_cast<size_t>(sliced_n) * fp16.stride[0] /
            sizeof(mollm::cpu::fp16_t);
        std::vector<float> sliced_reference_weight(
            weight_f32.begin() + sliced_weight_offset,
            weight_f32.begin() + sliced_weight_offset +
                static_cast<size_t>(sliced_n) * k);
        std::vector<float> sliced_expected(
            static_cast<size_t>(m) * sliced_n);
        std::vector<float> sliced_actual;
        reference(activation, sliced_reference_weight, sliced_expected,
                  m, sliced_n, k);
        backend.clear_dispatch_error();
        if (!upload(backend, sliced_input, activation))
            return 1;
        graph.runtime.tensors[0] = sliced_input;
        BufferPool pool;
        ExecContext context;
        context.graph = &graph;
        context.pool = &pool;
        context.backend = &backend;
        prepare_execution(context);
        execute_graph(context);
        if (backend.dispatch_failed() ||
            context.execution_failed ||
            !download(backend, graph.runtime.tensors[3], sliced_actual) ||
            !close_enough(sliced_actual, sliced_expected, 3e-3f))
            return 1;
    }
    if (backend.kv_cache_precision(Precision::FP16) != Precision::FP16 ||
        backend.kv_cache_precision(Precision::FP32) != Precision::FP32 ||
        !test_layout_rope_and_sdpa(backend, Precision::FP32) ||
        !test_layout_rope_and_sdpa(backend, Precision::FP16) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 0) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 255) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 511) ||
        !test_wide_cached_sdpa(backend, Precision::FP32, 512) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 512) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 1023) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 1280) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 2047, 1, false, 2, 256) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 4095, 1, false, 2, 256) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 4096, 1, false, 2, 256) ||
        !test_wide_cached_sdpa(backend, Precision::FP32, 2047) ||
        !test_wide_cached_sdpa(
            backend, Precision::FP16, 255, 1, false, 4) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 0, 4) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 257, 8) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 3, 5, true) ||
        !test_wide_cached_sdpa(
            backend, Precision::FP16, 255, 1, false, 2, 256) ||
        !test_wide_cached_sdpa(
            backend, Precision::FP16, 0, 17, false, 2, 256) ||
        !test_wide_cached_sdpa(
            backend, Precision::FP16, 1023, 2, true, 1, 256) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 0, 17, true, 2, 256, true) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 1023, 2, true, 1, 256, true) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 4094, 2, true, 1, 256, true) ||
        !test_wide_cached_sdpa(backend, Precision::FP16, 4095, 2, true, 1, 256, true) ||
        !test_shortconv(backend) ||
        !test_conservative_shortconv_graph(backend) ||
        !test_conservative_shortconv_graph(backend, true) ||
        !test_conservative_shortconv_graph(backend, false, true) ||
        !test_conservative_shortconv_graph(backend, false, false, true) ||
        !test_conservative_shortconv_graph(backend, false, true, true) ||
        !test_conservative_sdpa_graph(backend) ||
        !test_conservative_sdpa_graph(backend, true) ||
        !test_conservative_sdpa_graph(backend, true, 256) ||
        !test_conservative_sdpa_graph(backend, true, 256, true) ||
        !test_conservative_materialized_view_chain(backend) ||
        !test_conservative_ple_graph(backend) ||
        !test_ple_device_ops(backend) ||
        !test_resident_routes(backend) ||
        !test_nvfp4_system_memory(backend) ||
        !test_nvfp4_moe_provider() ||
        !test_gated_residual(backend) ||
        !test_sigmoid_gr_reduce_graph(backend) ||
        !test_gdn(backend) ||
        !test_gdn(backend, true) ||
        !test_decode_add_rms_norm(backend) ||
        !test_lm_head_argmax(backend) || !test_q8_batch_head(backend) || !test_state_prefix_coverage(backend) ||
        !test_ple_history_prefix(backend))
        return 1;
    phase.name = "memory and fallback bridge";
    if (!test_memory_and_fallback_bridge(backend))
        return 1;

    phase.name = "INT8 grouped matmul";
    constexpr int int8_group_size = 16;
    constexpr int int8_groups_per_row = k / int8_group_size;
    std::vector<int8_t> weight_int8(static_cast<size_t>(n) * k);
    std::vector<float> weight_int8_scales(
        static_cast<size_t>(n) * int8_groups_per_row);
    std::vector<float> weight_int8_reference(static_cast<size_t>(n) * k);
    for (int row = 0; row < n; ++row) {
        for (int group = 0; group < int8_groups_per_row; ++group) {
            const float scale = 0.01f * (1 + (row + group) % 5);
            weight_int8_scales[
                static_cast<size_t>(row) * int8_groups_per_row + group] =
                    scale;
            for (int inner = 0; inner < int8_group_size; ++inner) {
                const int column = group * int8_group_size + inner;
                const int8_t value = static_cast<int8_t>(
                    (row * 7 + column * 3) % 31 - 15);
                weight_int8[static_cast<size_t>(row) * k + column] = value;
                weight_int8_reference[
                    static_cast<size_t>(row) * k + column] = value * scale;
            }
        }
    }
    Tensor int8 = Tensor::create(
        Precision::INT8, MemoryType::EXTERNAL, n, k, 1, 1,
        weight_int8.data());
    int8.rowmajor_data = weight_int8.data();
    int8.scales = weight_int8_scales.data();
    int8.group_size = int8_group_size;
    int8.groups_per_row = int8_groups_per_row;
    backend.wrap_weight(int8);
    std::fill(actual.begin(), actual.end(), 0.0f);
    reference(activation, weight_int8_reference, expected, m, n, k);
    if (!dispatch_matmul(backend, int8, activation, actual, m, n, k) ||
        !close_enough(actual, expected, 8e-3f))
        return 1;
    std::vector<float> decode_actual(n);
    std::vector<float> decode_expected(n);
    reference(activation, weight_int8_reference, decode_expected, 1, n, k);
    if (!dispatch_matmul(
            backend, int8, activation, decode_actual, 1, n, k) ||
        !close_enough(decode_actual, decode_expected, 3e-3f))
        return 1;

    // Exercise the vectorized W8 per-channel decode path with more than one
    // vector-load iteration per lane.  Grouped W8 above continues to cover the
    // generic kernel.
    phase.name = "W8 per-channel matmul";
    constexpr int w8pc_k = 256;
    std::vector<float> w8pc_activation(w8pc_k);
    std::vector<int8_t> w8pc_weight(static_cast<size_t>(n) * w8pc_k);
    std::vector<float> w8pc_scales(n);
    std::vector<float> w8pc_reference_weight(
        static_cast<size_t>(n) * w8pc_k);
    for (int inner = 0; inner < w8pc_k; ++inner)
        w8pc_activation[inner] =
            static_cast<float>(inner % 29 - 14) / 31.0f;
    for (int row = 0; row < n; ++row) {
        w8pc_scales[row] = 0.0075f * (row + 1);
        for (int inner = 0; inner < w8pc_k; ++inner) {
            const int8_t value = static_cast<int8_t>(
                (row * 11 + inner * 5) % 63 - 31);
            const size_t index = static_cast<size_t>(row) * w8pc_k + inner;
            w8pc_weight[index] = value;
            w8pc_reference_weight[index] = value * w8pc_scales[row];
        }
    }
    Tensor w8pc = Tensor::create(
        Precision::INT8, MemoryType::EXTERNAL, n, w8pc_k, 1, 1,
        w8pc_weight.data());
    w8pc.rowmajor_data = w8pc_weight.data();
    w8pc.scales = w8pc_scales.data();
    w8pc.group_size = w8pc_k;
    w8pc.groups_per_row = 1;
    backend.wrap_weight(w8pc);
    std::vector<float> w8pc_actual(n);
    std::vector<float> w8pc_expected(n);
    reference(
        w8pc_activation, w8pc_reference_weight, w8pc_expected,
        1, n, w8pc_k);
    if (!dispatch_matmul(
            backend, w8pc, w8pc_activation, w8pc_actual,
            1, n, w8pc_k) ||
        !close_enough(w8pc_actual, w8pc_expected, 3e-3f))
        return 1;

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
    // The fused W4G32 preparation vectorizes aligned activation rows but must
    // preserve the scalar conversion fallback for borrowed device views.
    Tensor misaligned_activation_storage = device_tensor(
        backend, static_cast<int64_t>(activation.size()) + 1);
    std::vector<float> padded_activation(
        activation.size() + 1, 0.0f);
    std::copy(
        activation.begin(), activation.end(),
        padded_activation.begin() + 1);
    if (!upload(
            backend, misaligned_activation_storage, padded_activation))
        return 1;
    Tensor misaligned_activation = misaligned_activation_storage;
    misaligned_activation.shape[0] = k;
    misaligned_activation.shape[1] = m;
    misaligned_activation.compute_strides();
    misaligned_activation.device.offset += sizeof(float);
    Tensor misaligned_output = device_tensor(backend, n, m);
    GraphNode q4_matmul;
    q4_matmul.op_type = OpType::MATMUL;
    backend.clear_dispatch_error();
    backend.dispatch(
        q4_matmul, {&misaligned_activation, &q4},
        &misaligned_output, nullptr);
    backend.end_graph();
    if (backend.dispatch_failed() ||
        !download(backend, misaligned_output, actual) ||
        !close_enough(actual, expected, 8e-3f))
        return 1;
    reference(activation, q4_weight, decode_expected, 1, n, k);
    if (!dispatch_matmul(
            backend, q4, activation, decode_actual, 1, n, k) ||
        !close_enough(decode_actual, decode_expected, 8e-3f))
        return 1;

    constexpr int padded_n = n - 1;
    Q4B8G32Block padded_block = block;
    Tensor padded_q4 = Tensor::create(
        Precision::INT4, MemoryType::EXTERNAL, padded_n, k, 1, 1,
        &padded_block);
    padded_q4.rowmajor_data = &padded_block;
    padded_q4.q4_g32_data = &padded_block;
    padded_q4.is_q4_g32_packed = true;
    padded_q4.group_size = 32;
    padded_q4.groups_per_row = 1;
    backend.wrap_weight_int4(padded_q4);
    std::vector<float> padded_q4_weight(
        q4_weight.begin(), q4_weight.begin() + padded_n * k);
    std::vector<float> padded_actual(static_cast<size_t>(m) * padded_n);
    std::vector<float> padded_expected(padded_actual.size());
    reference(
        activation, padded_q4_weight, padded_expected, m, padded_n, k);
    if (!dispatch_matmul(
            backend, padded_q4, activation, padded_actual, m, padded_n, k) ||
        !close_enough(padded_actual, padded_expected, 8e-3f))
        return 1;
    padded_actual.resize(padded_n);
    padded_expected.resize(padded_n);
    reference(
        activation, padded_q4_weight, padded_expected, 1, padded_n, k);
    if (!dispatch_matmul(
            backend, padded_q4, activation, padded_actual, 1, padded_n, k) ||
        !close_enough(padded_actual, padded_expected, 8e-3f))
        return 1;
    Tensor invalid_q4 = Tensor::create(
        Precision::INT4, MemoryType::EXTERNAL, n, k, 1, 1, &block);
    invalid_q4.q4_g32_data = &block;
    invalid_q4.is_q4_g32_packed = true;
    invalid_q4.group_size = 16;
    invalid_q4.groups_per_row = 2;
    backend.wrap_weight_int4(invalid_q4);
    if (invalid_q4.device.buffer)
        return 1;

    phase.name = "W4G32 matmul and activation caching";
    constexpr int k64 = 64;
    constexpr int groups32 = k64 / 32;
    std::vector<float> activation64(static_cast<size_t>(m) * k64);
    for (size_t i = 0; i < activation64.size(); ++i)
        activation64[i] =
            static_cast<float>(static_cast<int>(i % 19) - 9) / 23.0f;
    std::vector<Q4B8G32Block> blocks32(groups32);
    std::vector<float> q4_weight64(static_cast<size_t>(n) * k64);
    for (int group = 0; group < groups32; ++group) {
        auto& packed_block = blocks32[group];
        for (int row = 0; row < n; ++row) {
            packed_block.scales[row] =
                0.03125f + 0.002f * (row + group);
            for (int inner = 0; inner < 32; inner += 2) {
                const int low = (row + group * 5 + inner) % 16 - 8;
                const int high =
                    (row + group * 5 + inner + 1) % 16 - 8;
                packed_block.q[row][inner / 2] =
                    static_cast<uint8_t>(
                        (low & 0xf) | ((high & 0xf) << 4));
                q4_weight64[static_cast<size_t>(row) * k64 +
                            group * 32 + inner] =
                    low * packed_block.scales[row];
                q4_weight64[static_cast<size_t>(row) * k64 +
                            group * 32 + inner + 1] =
                    high * packed_block.scales[row];
            }
        }
    }
    Tensor q4_multi = Tensor::create(
        Precision::INT4, MemoryType::EXTERNAL, n, k64, 1, 1,
        blocks32.data());
    q4_multi.rowmajor_data = blocks32.data();
    q4_multi.q4_g32_data = blocks32.data();
    q4_multi.is_q4_g32_packed = true;
    q4_multi.group_size = 32;
    q4_multi.groups_per_row = groups32;
    backend.wrap_weight_int4(q4_multi);
    std::memset(
        blocks32.data(), 0, blocks32.size() * sizeof(Q4B8G32Block));
    actual.resize(static_cast<size_t>(m) * n);
    expected.resize(actual.size());
    reference(activation64, q4_weight64, expected, m, n, k64);
    if (!dispatch_matmul(
            backend, q4_multi, activation64, actual, m, n, k64) ||
        !close_enough(actual, expected, 8e-3f))
        return 1;
    reference(activation64, q4_weight64, decode_expected, 1, n, k64);
    // A repeated multi-row call must refresh its activations while retaining
    // the immutable uploaded matrix, including after a preceding GEMV.
    auto changed_activation64 = activation64;
    for (float& value : changed_activation64) value = value * -0.5f + 0.125f;
    reference(changed_activation64, q4_weight64, expected, m, n, k64);
    if (!dispatch_matmul(
            backend, q4_multi, changed_activation64, actual, m, n, k64) ||
        !close_enough(actual, expected, 8e-3f))
        return 1;
    if (!dispatch_matmul(
            backend, q4_multi, activation64, decode_actual, 1, n, k64) ||
        !close_enough(decode_actual, decode_expected, 8e-3f))
        return 1;

    // The pair-row activation-cache specialization covers both the 4096-wide
    // GDN projection and the 9216-wide down projection. Keep each dynamic
    // shared-memory size and its following quantized dot under reference and
    // sanitizer coverage without allocating a model-sized matrix.
    const auto check_cached_pair = [&](int cached_k) {
        const int cached_groups = cached_k / 32;
        std::vector<float> cached_activation(cached_k);
        for (int inner = 0; inner < cached_k; ++inner)
            cached_activation[inner] =
                static_cast<float>(inner % 37 - 18) / 41.0f;
        std::vector<Q4B8G32Block> cached_blocks(cached_groups);
        std::vector<float> cached_weight(
            static_cast<size_t>(n) * cached_k);
        for (int group = 0; group < cached_groups; ++group) {
            auto& packed_block = cached_blocks[group];
            for (int row = 0; row < n; ++row) {
                const float scale = 0.015625f +
                    ((group + row) % 11) * 0.00075f;
                packed_block.scales[row] = scale;
                for (int pair = 0; pair < 16; ++pair) {
                    const int low =
                        (group * 7 + row + pair * 2) % 16 - 8;
                    const int high =
                        (group * 7 + row + pair * 2 + 1) % 16 - 8;
                    packed_block.q[row][pair] = static_cast<uint8_t>(
                        (low & 0xf) | ((high & 0xf) << 4));
                    const int inner = group * 32 + pair * 2;
                    cached_weight[
                        static_cast<size_t>(row) * cached_k + inner] =
                        low * scale;
                    cached_weight[
                        static_cast<size_t>(row) * cached_k + inner + 1] =
                        high * scale;
                }
            }
        }
        Tensor cached_q4 = Tensor::create(
            Precision::INT4, MemoryType::EXTERNAL,
            n, cached_k, 1, 1, cached_blocks.data());
        cached_q4.rowmajor_data = cached_blocks.data();
        cached_q4.q4_g32_data = cached_blocks.data();
        cached_q4.is_q4_g32_packed = true;
        cached_q4.group_size = 32;
        cached_q4.groups_per_row = cached_groups;
        backend.wrap_weight_int4(cached_q4);
        std::memset(
            cached_blocks.data(), 0,
            cached_blocks.size() * sizeof(Q4B8G32Block));
        std::vector<float> cached_expected(n);
        std::vector<float> cached_actual(n);
        reference(
            cached_activation, cached_weight, cached_expected,
            1, n, cached_k);
        return dispatch_matmul(
                   backend, cached_q4, cached_activation, cached_actual,
                   1, n, cached_k) &&
            close_enough(cached_actual, cached_expected, 8e-3f);
    };
    if (!check_cached_pair(4096) || !check_cached_pair(9216))
        return 1;

    // The narrow-prefetch selector needs the actual model shape: smaller
    // correctness cases do not exercise its opt-in branch.
    // The backend keys immutable source identities by address. Do not recycle
    // these fixture identities before constructing same-shaped wide weights.
    std::vector<std::vector<Q4B8G32Block>> retained_narrow_sources;
    for (const auto shape : {std::pair<int, int>{2560, 6144}, {320, 10240}, {640, 2560}}) {
        const int narrow_n = shape.first, narrow_k = shape.second;
        std::vector<Q4B8G32Block> packed((narrow_n / 8) * (narrow_k / 32));
        for (size_t i = 0; i < packed.size(); ++i)
            for (int row = 0; row < 8; ++row) {
                packed[i].scales[row] = (1 + (i + row) % 17) / 37.f;
                for (int pair = 0; pair < 16; ++pair)
                    packed[i].q[row][pair] = (i * 19 + row * 7 + pair * 13) & 255;
            }
        std::vector<float> x(narrow_k), original, candidate;
        for (int i = 0; i < narrow_k; ++i) x[i] = (i % 41 - 20) / 37.f;
        Tensor dx = device_tensor(backend, narrow_k);
        Tensor dw = device_tensor(backend, packed.size() * sizeof(Q4B8G32Block) / sizeof(float));
        Tensor dy = device_tensor(backend, narrow_n);
        if (!upload(backend, dx, x) || cudaMemcpy(dw.device.buffer, packed.data(),
            packed.size() * sizeof(Q4B8G32Block), cudaMemcpyHostToDevice) != cudaSuccess) return 1;
        auto* weights = static_cast<const Q4B8G32Block*>(dw.device.buffer);
        auto* input = static_cast<const float*>(dx.device.buffer);
        auto* output = static_cast<float*>(dy.device.buffer);
        mollm_cuda::launch_q4_g32_prefetch_narrow_gemv(input, weights, output, narrow_n, narrow_k, 0);
        if (!download(backend, dy, original)) return 1;
        for (int distance : {1, 2, 4}) {
            mollm_cuda::launch_q4_g32_prefetch_narrow_gemv(input, weights, output, narrow_n, narrow_k, distance);
            if (!download(backend, dy, candidate) || std::memcmp(original.data(), candidate.data(), narrow_n * sizeof(float))) return 1;
        }
        mollm_cuda::launch_q4_g32_dense_gemv(input, weights, output, narrow_n, narrow_k);
        if (!download(backend, dy, candidate) || std::memcmp(original.data(), candidate.data(), narrow_n * sizeof(float))) return 1;
        Tensor dr = device_tensor(backend, packed.size() * sizeof(Q4B8G32Block) / sizeof(float));
        auto* row_weights = static_cast<mollm_cuda::Q4RowG32*>(dr.device.buffer);
        mollm_cuda::launch_repack_q4_row_g32(weights, row_weights, narrow_n, narrow_k);
        mollm_cuda::launch_q4_row_g32_gemv(input, row_weights, output, narrow_n, narrow_k);
        if (!download(backend, dy, candidate) || std::memcmp(original.data(), candidate.data(), narrow_n * sizeof(float))) return 1;
        retained_narrow_sources.push_back(packed);
        auto& canonical = retained_narrow_sources.back();
        for (auto& block : canonical)
            for (auto& row : block.q)
                for (auto& byte : row) byte ^= 0x88;
        Tensor narrow_weight = Tensor::create(Precision::INT4, MemoryType::EXTERNAL,
            narrow_n, narrow_k, 1, 1, canonical.data());
        narrow_weight.rowmajor_data = canonical.data();
        narrow_weight.q4_g32_data = canonical.data();
        narrow_weight.is_q4_g32_packed = true;
        narrow_weight.group_size = 32;
        narrow_weight.groups_per_row = narrow_k / 32;
        backend.wrap_weight_int4(narrow_weight);
        std::memset(canonical.data(), 0, canonical.size() * sizeof(canonical[0]));
        if (!dispatch_matmul(backend, narrow_weight, x, candidate, 1, narrow_n, narrow_k) ||
            std::memcmp(original.data(), candidate.data(), narrow_n * sizeof(float))) {
            std::fprintf(stderr, "narrow decode fixture mismatch n=%d k=%d\n", narrow_n, narrow_k);
            return 1;
        }
        for (bool cache : {false, true}) {
            mollm_cuda::launch_q4_g32_pair_narrow_gemv(input, weights, output, narrow_n, narrow_k, cache);
            if (!download(backend, dy, candidate) || !close_enough(candidate, original, 1e-3f)) return 1;
        }
        for (int splits : {2, 4, 8}) {
            mollm_cuda::launch_q4_g32_split_pair_gemv(input, weights, output, narrow_n, narrow_k, splits);
            if (!download(backend, dy, candidate) || !close_enough(candidate, original, 1e-3f)) return 1;
        }
    }

    // Exercise the wide-output CUDA kernel selector at the model's K=2560
    // with a ragged final
    // storage block. Compute the reference directly from the packed weights
    // so the test does not need a large expanded FP32 matrix.
    const bool narrow_cache_fixture=std::getenv("MOLLM_TEST_CUDA_NARROW_CACHE")!=nullptr;
    const bool hc_up_fixture=std::getenv("MOLLM_TEST_CUDA_HC_UP")!=nullptr;
    const bool hc_down_fixture=std::getenv("MOLLM_TEST_CUDA_HC_DOWN_CACHE")!=nullptr;
    const int wide_n = hc_up_fixture ? 10240 : hc_down_fixture ? 320 : narrow_cache_fixture ? 2560 : 18439;
    const int wide_k = hc_up_fixture ? 320 : hc_down_fixture ? 10240 : narrow_cache_fixture ? 6144 : 2560;
    const int wide_groups = wide_k / 32;
    constexpr int wide_rows_per_block = 8;
    const int wide_row_blocks =
        (wide_n + wide_rows_per_block - 1) / wide_rows_per_block;
    std::vector<float> wide_activation(wide_k);
    for (int inner = 0; inner < wide_k; ++inner)
        wide_activation[inner] =
            static_cast<float>(inner % 31 - 15) / 37.0f;
    std::vector<Q4B8G32Block> wide_blocks(
        static_cast<size_t>(wide_row_blocks) * wide_groups);
    std::vector<float> wide_expected(wide_n, 0.0f);
    const int reference_lanes = hc_up_fixture ? 16 : 8;
    std::vector<float> wide_lane_sums(static_cast<size_t>(wide_n) * reference_lanes, 0.0f);
    for (int row_block = 0; row_block < wide_row_blocks; ++row_block) {
        for (int group = 0; group < wide_groups; ++group) {
            auto& packed_block = wide_blocks[
                static_cast<size_t>(row_block) * wide_groups + group];
            for (int row_lane = 0; row_lane < wide_rows_per_block;
                 ++row_lane) {
                const int row =
                    row_block * wide_rows_per_block + row_lane;
                const float group_scale = hc_up_fixture
                    ? (1.f + ((row_block + group + row_lane) % 13) * .125f) / 7.f
                    : 0.0078125f + ((row_block + group + row_lane) % 13) * 0.0005f;
                packed_block.scales[row_lane] = group_scale;
                for (int pair = 0; pair < 16; ++pair) {
                    const int low =
                        (row + group * 3 + pair * 2) % 16 - 8;
                    const int high =
                        (row + group * 3 + pair * 2 + 1) % 16 - 8;
                    packed_block.q[row_lane][pair] =
                        static_cast<uint8_t>(
                            (low & 0xf) | ((high & 0xf) << 4));
                    if (row < wide_n) {
                        const int inner = group * 32 + pair * 2;
                        wide_expected[row] +=
                            wide_activation[inner] * low * group_scale;
                        wide_expected[row] +=
                            wide_activation[inner + 1] * high * group_scale;
                        if (hc_up_fixture) {
                            const float dot=std::fma(wide_activation[inner],static_cast<float>(low),wide_activation[inner+1]*static_cast<float>(high));
                            float& partial=wide_lane_sums[static_cast<size_t>(row)*16+pair];
                            partial=std::fma(dot,group_scale,partial);
                        } else if (pair & 1) {
                            const int previous_low = (row + group * 3 + (pair - 1) * 2) % 16 - 8;
                            const int previous_high = (row + group * 3 + (pair - 1) * 2 + 1) % 16 - 8;
                            float dot = std::fma(wide_activation[inner - 2], static_cast<float>(previous_low),
                                wide_activation[inner - 1] * static_cast<float>(previous_high));
                            dot = std::fma(wide_activation[inner], static_cast<float>(low), dot);
                            dot = std::fma(wide_activation[inner + 1], static_cast<float>(high), dot);
                            float& partial = wide_lane_sums[static_cast<size_t>(row) * 8 + pair / 2];
                            partial = std::fma(dot, group_scale, partial);
                        }
                    }
                }
            }
        }
    }
    Tensor wide_q4 = Tensor::create(
        Precision::INT4, MemoryType::EXTERNAL,
        wide_n, wide_k, 1, 1, wide_blocks.data());
    wide_q4.rowmajor_data = wide_blocks.data();
    wide_q4.q4_g32_data = wide_blocks.data();
    wide_q4.is_q4_g32_packed = true;
    wide_q4.group_size = 32;
    wide_q4.groups_per_row = wide_groups;
    backend.wrap_weight_int4(wide_q4);
    std::memset(
        wide_blocks.data(), 0,
        wide_blocks.size() * sizeof(Q4B8G32Block));
    std::vector<float> wide_actual(wide_n);
    if (!dispatch_matmul(
            backend, wide_q4, wide_activation, wide_actual,
            1, wide_n, wide_k) ||
        !close_enough(wide_actual, wide_expected, 5e-3f)) {
        std::fprintf(stderr, "wide decode fixture mismatch n=%d k=%d\n", wide_n, wide_k);
        return 1;
    }
    int wide_device = 0;
    cudaDeviceProp wide_properties{};
    cudaGetDevice(&wide_device);
    cudaGetDeviceProperties(&wide_properties, wide_device);
    if ((wide_n + 31) / 32 >= wide_properties.multiProcessorCount * 6) {
        for (int row = 0; row < wide_n; ++row) {
            const float* partial = wide_lane_sums.data() + static_cast<size_t>(row) * reference_lanes;
            float old_result = ((partial[0] + partial[4]) + (partial[2] + partial[6])) +
                ((partial[1] + partial[5]) + (partial[3] + partial[7]));
            if(hc_up_fixture){
                float tree[16];std::copy(partial,partial+16,tree);
                for(int offset=8;offset;offset/=2)
                    for(int lane=0;lane<offset;++lane)tree[lane]+=tree[lane+offset];
                old_result=tree[0];
            }
            if (wide_actual[row] != old_result) {
                std::fprintf(stderr, "wide W4 reduction-order mismatch at row %d: %.9g != %.9g\n",
                    row, wide_actual[row], old_result);
                return 1;
            }
        }
    }

    // The same private weight representation must support prefill after
    // decode, including a ragged output tile and destroyed CPU backing.
    std::vector<float> wide_prefill_activation(2 * wide_k);
    std::vector<float> wide_prefill_expected(2 * wide_n), wide_prefill_actual(2 * wide_n);
    for (int i = 0; i < wide_k; ++i) {
        wide_prefill_activation[i] = wide_activation[i];
        wide_prefill_activation[wide_k + i] = wide_activation[i] * 0.5f;
    }
    for (int i = 0; i < wide_n; ++i) {
        wide_prefill_expected[i] = wide_expected[i];
        wide_prefill_expected[wide_n + i] = wide_expected[i] * 0.5f;
    }
    if (!dispatch_matmul(backend, wide_q4, wide_prefill_activation, wide_prefill_actual,
            2, wide_n, wide_k) ||
        !close_enough(wide_prefill_actual, wide_prefill_expected, 5e-3f)) {
        std::fprintf(stderr, "wide prefill fixture failed n=%d k=%d\n", wide_n, wide_k);
        return 1;
    }

    // Reuse immutable uploaded weights across changed inputs and batch sizes.
    // Enabled FP16 prefill caching must not retain activations or stale shapes;
    // a small cache budget must also preserve the original scratch fallback.
    for (int batch : {3, 17}) {
        std::vector<float> changed_input(static_cast<size_t>(batch) * wide_k);
        std::vector<float> changed_expected(static_cast<size_t>(batch) * wide_n);
        std::vector<float> changed_output(changed_expected.size());
        for (int token = 0; token < batch; ++token) {
            const float multiplier = (token % 3 - 1) * .5f;
            for (int col = 0; col < wide_k; ++col)
                changed_input[static_cast<size_t>(token) * wide_k + col] = wide_activation[col] * multiplier;
            for (int col = 0; col < wide_n; ++col)
                changed_expected[static_cast<size_t>(token) * wide_n + col] = wide_expected[col] * multiplier;
        }
        if (!dispatch_matmul(backend, wide_q4, changed_input, changed_output,
                batch, wide_n, wide_k) ||
            !close_enough(changed_output, changed_expected, 5e-3f)) {
            std::fprintf(stderr, "wide changed-input fixture failed n=%d k=%d batch=%d\n", wide_n, wide_k, batch);
            return 1;
        }
    }

    phase.name = "W4G128 single-group matmul";
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
    reference(
        std::vector<float>(activation128.begin(),
                           activation128.begin() + k128),
        q4_weight128, decode_expected, 1, n, k128);
    if (!dispatch_matmul(
            backend, q4_128, activation128, decode_actual, 1, n, k128) ||
        !close_enough(decode_actual, decode_expected, 1.5e-2f))
        return 1;

    Q4B8G128Block padded_block128 = block128;
    Tensor padded_q4_128 = Tensor::create(
        Precision::INT4, MemoryType::EXTERNAL, padded_n, k128, 1, 1,
        &padded_block128);
    padded_q4_128.rowmajor_data = &padded_block128;
    padded_q4_128.q4_g128_data = &padded_block128;
    padded_q4_128.is_q4_g128_packed = true;
    padded_q4_128.group_size = 128;
    padded_q4_128.groups_per_row = 1;
    backend.wrap_weight_int4(padded_q4_128);
    std::vector<float> padded_q4_weight128(
        q4_weight128.begin(), q4_weight128.begin() + padded_n * k128);
    padded_actual.resize(static_cast<size_t>(m) * padded_n);
    padded_expected.resize(padded_actual.size());
    reference(
        activation128, padded_q4_weight128, padded_expected,
        m, padded_n, k128);
    if (!dispatch_matmul(
            backend, padded_q4_128, activation128, padded_actual,
            m, padded_n, k128) ||
        !close_enough(padded_actual, padded_expected, 1.5e-2f))
        return 1;
    padded_actual.resize(padded_n);
    padded_expected.resize(padded_n);
    reference(
        activation128, padded_q4_weight128, padded_expected,
        1, padded_n, k128);
    if (!dispatch_matmul(
            backend, padded_q4_128, activation128, padded_actual,
            1, padded_n, k128) ||
        !close_enough(padded_actual, padded_expected, 1.5e-2f))
        return 1;

    phase.name = "W4G128 multi-group matmul";
    constexpr int k256 = 256;
    constexpr int groups128 = k256 / 128;
    std::vector<float> activation256(static_cast<size_t>(m) * k256);
    for (size_t i = 0; i < activation256.size(); ++i)
        activation256[i] =
            static_cast<float>(static_cast<int>(i % 23) - 11) / 29.0f;
    std::vector<Q4B8G128Block> blocks128(groups128);
    std::vector<float> q4_weight256(static_cast<size_t>(n) * k256);
    for (int group = 0; group < groups128; ++group) {
        auto& packed_block = blocks128[group];
        for (int row = 0; row < n; ++row) {
            packed_block.scales[row] =
                0.015625f + 0.001f * (row + group);
            for (int inner = 0; inner < 128; inner += 2) {
                const int low = (row + group * 7 + inner) % 16 - 8;
                const int high =
                    (row + group * 7 + inner + 1) % 16 - 8;
                const int subgroup = inner / 32;
                const int subgroup_inner = inner % 32;
                packed_block.q[subgroup][row][subgroup_inner / 2] =
                    static_cast<uint8_t>(
                        (low & 0xf) | ((high & 0xf) << 4));
                q4_weight256[static_cast<size_t>(row) * k256 +
                             group * 128 + inner] =
                    low * packed_block.scales[row];
                q4_weight256[static_cast<size_t>(row) * k256 +
                             group * 128 + inner + 1] =
                    high * packed_block.scales[row];
            }
        }
    }
    Tensor q4_multi128 = Tensor::create(
        Precision::INT4, MemoryType::EXTERNAL, n, k256, 1, 1,
        blocks128.data());
    q4_multi128.rowmajor_data = blocks128.data();
    q4_multi128.q4_g128_data = blocks128.data();
    q4_multi128.is_q4_g128_packed = true;
    q4_multi128.group_size = 128;
    q4_multi128.groups_per_row = groups128;
    backend.wrap_weight_int4(q4_multi128);
    std::memset(
        blocks128.data(), 0, blocks128.size() * sizeof(Q4B8G128Block));
    reference(activation256, q4_weight256, expected, m, n, k256);
    if (!dispatch_matmul(
            backend, q4_multi128, activation256, actual, m, n, k256) ||
        !close_enough(actual, expected, 1.5e-2f))
        return 1;
    reference(activation256, q4_weight256, decode_expected, 1, n, k256);
    if (!dispatch_matmul(
            backend, q4_multi128, activation256, decode_actual, 1, n, k256) ||
        !close_enough(decode_actual, decode_expected, 1.5e-2f))
        return 1;

    phase.name = "grouped narrow stream";
    if (!test_grouped_narrow_stream()) return 1;
    phase.name = "FP16 conversion and NVFP4 sort streams";
    if (!test_fp16_to_fp32_reference() || !test_nvfp4_sort_stream()) return 1;
    phase.name = "Q4 pipeline and async streams";
    if (!test_q4_wide_pipeline_stream() || !test_q4_wide_pipeline_stream(1) ||
        !test_q4_wide_pipeline_stream(2) || !test_q4_async_stream()) return 1;
    for (int mode = 3; mode <= 6; ++mode) if (!test_q4_wide_pipeline_stream(mode)) return 1;
    std::printf("CUDA device-resident ops, fallback views, FP16, W8, W4G32 "
                "and W4G128 matmul tests passed\n");
    phase.passed = true;
    return 0;
}
