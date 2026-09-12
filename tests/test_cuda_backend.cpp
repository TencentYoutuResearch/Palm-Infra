#include "backends/cuda/backend.h"
#include "engine/engine.h"
#include "graph/execute.h"
#include "backends/cpu/backend.h"
#include "core/quant_layouts.h"

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

bool close_enough_relative(
    const std::vector<float>& actual, const std::vector<float>& expected,
    float absolute_tolerance, float relative_tolerance) {
    for (size_t index = 0; index < actual.size(); ++index) {
        const float difference = std::fabs(actual[index] - expected[index]);
        const float tolerance = absolute_tolerance +
            relative_tolerance * std::fabs(expected[index]);
        if (difference > tolerance) {
            std::fprintf(
                stderr,
                "CUDA mismatch at %zu: actual=%g expected=%g "
                "tolerance=%g\n",
                index, actual[index], expected[index], tolerance);
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
                           int kv_heads = 1, int head_dim = 128) {
    backend.clear_dispatch_error();
    const int heads = 2 * kv_heads;
    // Exercise common Qwen head widths so decode and prefill specializations
    // can test their parallel reductions against the same reference.
    const int key_dim = head_dim;
    const int value_dim = head_dim;
    const int key_length = past_length + current_length;
    const int capacity = key_length + 1;

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

bool test_gdn(CudaBackend& backend) {
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
                  1, 4, real_tokens, value_heads};
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
    params.i32[4] = 0;
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

    params.i32[4] = 1;
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
        !test_shortconv(backend) ||
        !test_gdn(backend) ||
        !test_decode_add_rms_norm(backend) ||
        !test_lm_head_argmax(backend))
        return 1;
    if (!test_memory_and_fallback_bridge(backend))
        return 1;

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

    // Exercise the wide-output CUDA kernel selector with a ragged final
    // storage block. Compute the reference directly from the packed weights
    // so the test does not need a large expanded FP32 matrix.
    constexpr int wide_n = 18439;
    constexpr int wide_k = 2048;
    constexpr int wide_groups = wide_k / 32;
    constexpr int wide_rows_per_block = 8;
    constexpr int wide_row_blocks =
        (wide_n + wide_rows_per_block - 1) / wide_rows_per_block;
    std::vector<float> wide_activation(wide_k);
    for (int inner = 0; inner < wide_k; ++inner)
        wide_activation[inner] =
            static_cast<float>(inner % 31 - 15) / 37.0f;
    std::vector<Q4B8G32Block> wide_blocks(
        static_cast<size_t>(wide_row_blocks) * wide_groups);
    std::vector<float> wide_expected(wide_n, 0.0f);
    for (int row_block = 0; row_block < wide_row_blocks; ++row_block) {
        for (int group = 0; group < wide_groups; ++group) {
            auto& packed_block = wide_blocks[
                static_cast<size_t>(row_block) * wide_groups + group];
            for (int row_lane = 0; row_lane < wide_rows_per_block;
                 ++row_lane) {
                const int row =
                    row_block * wide_rows_per_block + row_lane;
                const float group_scale = 0.0078125f +
                    ((row_block + group + row_lane) % 13) * 0.0005f;
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
        !close_enough(wide_actual, wide_expected, 5e-3f))
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

    std::printf("CUDA device-resident ops, fallback views, FP16, W8, W4G32 "
                "and W4G128 matmul tests passed\n");
    return 0;
}
