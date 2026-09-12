#include <cuda_runtime.h>

#include "backends/cuda/backend_internal.h"
#include "core/cache_layout.h"
#include "graph/graph.h"
#include "kernels/cuda/attention.h"
#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/recurrent.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using mollm::cuda_backend_detail::device_pointer;
using mollm::cuda_backend_detail::device_pointer_const;
using mollm::cuda_backend_detail::fp32_contiguous;
using mollm::cuda_backend_detail::same_shape;

void CudaBackend::dispatch(const GraphNode& node,
                           const std::vector<const Tensor*>& inputs,
                           Tensor* output, ThreadPool* thread_pool) {
    auto record_native = [&]() {
        ++impl_->native_ops[static_cast<uint32_t>(node.op_type)];
    };
    if (node.op_type == OpType::INPUT ||
        node.op_type == OpType::CONSTANT) {
        record_native();
        return;
    }

    if (node.op_type == OpType::RESHAPE && !inputs.empty() && inputs[0] &&
        output && inputs[0]->is_contiguous()) {
        const int64_t shape[4] = {output->shape[0], output->shape[1],
                                  output->shape[2], output->shape[3]};
        *output = *inputs[0];
        for (int dimension = 0; dimension < 4; ++dimension)
            output->shape[dimension] = shape[dimension];
        output->compute_strides();
        record_native();
        return;
    }

    if (node.op_type == OpType::PERMUTE && !inputs.empty() && inputs[0] &&
        output) {
        const Tensor& source = *inputs[0];
        const int axis[4] = {
            graph_params::get_i32(node.params, 0, 0),
            graph_params::get_i32(node.params, 1, 1),
            graph_params::get_i32(node.params, 2, 2),
            graph_params::get_i32(node.params, 3, 3),
        };
        Tensor view = source;
        for (int dimension = 0; dimension < 4; ++dimension) {
            view.shape[axis[dimension]] = source.shape[dimension];
            view.stride[axis[dimension]] = source.stride[dimension];
        }
        *output = view;
        record_native();
        return;
    }

    if (node.op_type == OpType::SLICE && !inputs.empty() && inputs[0] &&
        output) {
        const Tensor& source = *inputs[0];
        const int dimension = graph_params::get_i32(node.params, 0, 0);
        const int offset = graph_params::get_i32(node.params, 1, 0);
        const int size = graph_params::get_i32(
            node.params, 2, static_cast<int>(source.shape[dimension]));
        *output = source;
        const size_t byte_offset =
            static_cast<size_t>(offset) * source.stride[dimension];
        if (source.data)
            output->data = static_cast<uint8_t*>(source.data) + byte_offset;
        output->device.offset = source.device.offset + byte_offset;
        output->shape[dimension] = size;
        record_native();
        return;
    }

    if ((node.op_type == OpType::CONTIGUOUS ||
         node.op_type == OpType::RESHAPE) &&
        !inputs.empty() && inputs[0] && output &&
        inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 && output->is_contiguous()) {
        const Tensor& source = *inputs[0];
        const float* input = device_pointer_const<float>(source);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (input && destination && source.nelements() == output->nelements()) {
            mollm_cuda::launch_contiguous(
                input, destination, count, source.shape[0], source.shape[1],
                source.shape[2], source.stride[0] / sizeof(float),
                source.stride[1] / sizeof(float),
                source.stride[2] / sizeof(float),
                source.stride[3] / sizeof(float));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "contiguous_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::ROTARY_EMBED && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        inputs[2]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& source = *inputs[0];
        const Tensor& cosine = *inputs[1];
        const Tensor& sine = *inputs[2];
        const float* input = device_pointer_const<float>(source);
        const float* cos_data = device_pointer_const<float>(cosine);
        const float* sin_data = device_pointer_const<float>(sine);
        float* destination = device_pointer<float>(*output);
        const int feature_dim = static_cast<int>(source.shape[0]);
        const int sequence_length = static_cast<int>(source.shape[1]);
        const int channels = static_cast<int>(source.shape[2] * source.shape[3]);
        const int rope_dim = graph_params::get_i32(node.params, 0, 64);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        if (input && cos_data && sin_data && destination &&
            rope_dim > 0 && rope_dim <= feature_dim && rope_dim % 2 == 0 &&
            cosine.shape[0] >= rope_dim / 2 &&
            sine.shape[0] >= rope_dim / 2) {
            mollm_cuda::launch_rope(
                input, cos_data, sin_data, destination, feature_dim,
                sequence_length, channels, static_cast<int>(source.shape[2]),
                rope_dim, interleave,
                source.stride[0] / sizeof(float),
                source.stride[1] / sizeof(float),
                source.stride[2] / sizeof(float),
                source.stride[3] / sizeof(float),
                cosine.stride[0] / sizeof(float),
                cosine.stride[1] / sizeof(float),
                sine.stride[0] / sizeof(float),
                sine.stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                output->stride[3] / sizeof(float));
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::SDPA ||
         node.op_type == OpType::SDPA_MLA) &&
        inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] &&
        output && inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        inputs[2]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& query = *inputs[0];
        const Tensor& current_key = *inputs[1];
        const Tensor& current_value = *inputs[2];
        const Tensor* mask = inputs.size() > 3 && inputs[3] &&
                inputs[3]->data && inputs[3]->nelements() > 0
            ? inputs[3] : nullptr;
        const Tensor* key_cache = inputs.size() > 4 && inputs[4] &&
                inputs[4]->data
            ? inputs[4] : nullptr;
        const Tensor* value_cache = inputs.size() > 5 && inputs[5] &&
                inputs[5]->data
            ? inputs[5] : nullptr;
        const int cache_mode = graph_params::get_i32(node.params, 0, 2);
        const bool causal = graph_params::get_i32(node.params, 1, 1) != 0;
        const int num_heads = graph_params::get_i32(
            node.params, 2, static_cast<int>(query.shape[2]));
        const int num_kv_heads = graph_params::get_i32(
            node.params, 3, static_cast<int>(current_key.shape[2]));
        const int key_dim = graph_params::get_i32(
            node.params, 4, static_cast<int>(query.shape[0]));
        const int value_dim = graph_params::get_i32(
            node.params, 5, static_cast<int>(current_value.shape[0]));
        float scale = graph_params::get_f32(node.params, 0, 0.0f);
        if (scale == 0.0f)
            scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
        const int query_length = static_cast<int>(query.shape[1]);
        const int current_length = static_cast<int>(current_key.shape[1]);
        int past_length = 0;
        int key_capacity = current_length;
        bool cached = false;
        const float* current_key_data =
            device_pointer_const<float>(current_key);
        const float* current_value_data =
            device_pointer_const<float>(current_value);
        const void* key_data = current_key_data;
        const void* value_data = current_value_data;
        void* key_cache_data = nullptr;
        void* value_cache_data = nullptr;
        bool fp16_cache = false;
        if (cache_mode == 2 && key_cache && value_cache &&
            key_cache->prec != value_cache->prec) {
            std::fprintf(
                stderr,
                "CudaBackend: SDPA requires matching K/V cache precision\n");
            impl_->failed = true;
            return;
        }
        if (cache_mode == 2 && key_cache && value_cache &&
            key_cache->prec == value_cache->prec &&
            (key_cache->prec == Precision::FP16 ||
             key_cache->prec == Precision::FP32) &&
            key_cache->device.buffer && value_cache->device.buffer) {
            const auto* key_metadata = cache_meta(
                static_cast<const uint8_t*>(key_cache->data) +
                key_cache->device.offset);
            const auto* value_metadata = cache_meta(
                static_cast<const uint8_t*>(value_cache->data) +
                value_cache->device.offset);
            past_length = static_cast<int>(key_metadata->current_seq_len);
            key_capacity = static_cast<int>(key_metadata->max_seq_len);
            key_cache_data =
                static_cast<uint8_t*>(key_cache->device.buffer) +
                key_cache->device.offset + CacheMetadata::SIZE;
            value_cache_data =
                static_cast<uint8_t*>(value_cache->device.buffer) +
                value_cache->device.offset + CacheMetadata::SIZE;
            fp16_cache = key_cache->prec == Precision::FP16;
            cached =
                value_metadata->current_seq_len ==
                    key_metadata->current_seq_len &&
                value_metadata->max_seq_len == key_metadata->max_seq_len &&
                key_metadata->num_kv_heads ==
                    static_cast<uint64_t>(num_kv_heads) &&
                value_metadata->num_kv_heads ==
                    static_cast<uint64_t>(num_kv_heads) &&
                key_metadata->head_dim == static_cast<uint64_t>(key_dim) &&
                value_metadata->v_head_dim ==
                    static_cast<uint64_t>(value_dim);
        }
        const int key_length = past_length + current_length;
        const float* query_data = device_pointer_const<float>(query);
        float* output_data = device_pointer<float>(*output);
        const float* mask_data = mask
            ? device_pointer_const<float>(*mask) : nullptr;
        const bool valid = query_data && current_key_data &&
            current_value_data &&
            output_data && num_heads > 0 && num_kv_heads > 0 &&
            num_heads % num_kv_heads == 0 && query_length > 0 &&
            current_length > 0 && key_dim > 0 && value_dim > 0 &&
            query.shape[0] >= key_dim && current_key.shape[0] >= key_dim &&
            current_value.shape[0] >= value_dim &&
            output->shape[0] >= value_dim &&
            (cache_mode == 0 || cached) &&
            (!cached || (key_cache_data && value_cache_data &&
                         key_length <= key_capacity)) &&
            (!mask || mask_data);
        if (valid) {
            bool cache_appended = false;
            const auto append_cache = [&]() {
                if (!cached || cache_appended)
                    return true;
                mollm_cuda::launch_append_kv(
                    current_key_data, current_value_data, key_cache_data,
                    value_cache_data, fp16_cache, num_kv_heads,
                    current_length, past_length, key_capacity, key_dim,
                    value_dim,
                    current_key.stride[1] / sizeof(float),
                    current_key.stride[2] / sizeof(float),
                    current_value.stride[1] / sizeof(float),
                    current_value.stride[2] / sizeof(float));
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), "append_kv_cuda"))
                    return false;
                cache_appended = true;
                return true;
            };
            if (cached) {
                key_data = key_cache_data;
                value_data = value_cache_data;
                // Decode may fold the one-token append into its score stage.
                // Prefill and generic decode layouts keep the standalone
                // append path below.
                if (query_length != 1 && !append_cache()) {
                    impl_->failed = true;
                    return;
                }
            }
            const size_t score_count = static_cast<size_t>(num_heads) *
                query_length * key_length;
            const size_t score_bytes = score_count * sizeof(float);
            if (!impl_->reserve(impl_->attention_scores,
                                impl_->attention_scores_bytes,
                                score_bytes)) {
                impl_->failed = true;
                return;
            }
            auto* scores = static_cast<float*>(impl_->attention_scores);
            if (query_length == 1) {
                const bool fused_append = cached && fp16_cache &&
                    mollm_cuda::try_launch_sdpa_decode_fp16_cached(
                        query_data, current_key_data, current_value_data,
                        key_cache_data, value_cache_data, scores, output_data,
                        mask_data, num_heads, num_kv_heads, key_length,
                        past_length, key_dim, value_dim, key_capacity, causal,
                        scale, query.stride[0] / sizeof(float),
                        query.stride[2] / sizeof(float),
                        current_key.stride[0] / sizeof(float),
                        current_key.stride[2] / sizeof(float),
                        current_value.stride[0] / sizeof(float),
                        current_value.stride[2] / sizeof(float),
                        mask ? mask->stride[0] / sizeof(float) : 0,
                        output->stride[0] / sizeof(float),
                        output->stride[2] / sizeof(float));
                if (!fused_append) {
                    if (!append_cache()) {
                        impl_->failed = true;
                        return;
                    }
                    mollm_cuda::launch_sdpa_decode(
                        query_data, key_data, value_data, scores, output_data,
                        mask_data, num_heads, num_kv_heads, key_length,
                        past_length, key_dim, value_dim, key_capacity, cached,
                        fp16_cache, causal, scale,
                        query.stride[0] / sizeof(float),
                        query.stride[2] / sizeof(float),
                        cached ? 1 : current_key.stride[0] / sizeof(float),
                        cached ? static_cast<size_t>(key_dim)
                               : current_key.stride[1] / sizeof(float),
                        cached ? static_cast<size_t>(key_capacity) * key_dim
                               : current_key.stride[2] / sizeof(float),
                        cached ? 1 : current_value.stride[0] / sizeof(float),
                        cached ? static_cast<size_t>(value_dim)
                               : current_value.stride[1] / sizeof(float),
                        cached
                            ? static_cast<size_t>(key_capacity) * value_dim
                            : current_value.stride[2] / sizeof(float),
                        mask ? mask->stride[0] / sizeof(float) : 0,
                        output->stride[0] / sizeof(float),
                        output->stride[2] / sizeof(float));
                }
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), "sdpa_decode_cuda")) {
                    impl_->failed = true;
                    return;
                }
                record_native();
                return;
            }
            if (mollm_cuda::try_launch_sdpa_prefill(
                    query_data, key_data, value_data, scores, output_data,
                    mask_data, num_heads, num_kv_heads, query_length,
                    key_length, past_length, key_dim, value_dim,
                    key_capacity, cached, fp16_cache, causal, scale,
                    query.stride[0] / sizeof(float),
                    query.stride[1] / sizeof(float),
                    query.stride[2] / sizeof(float),
                    mask ? mask->stride[0] / sizeof(float) : 0,
                    mask ? mask->stride[1] / sizeof(float) : 0,
                    output->stride[0] / sizeof(float),
                    output->stride[1] / sizeof(float),
                    output->stride[2] / sizeof(float))) {
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), "sdpa_prefill_fp16_cuda")) {
                    impl_->failed = true;
                    return;
                }
                record_native();
                return;
            }
            mollm_cuda::launch_sdpa_scores(
                query_data, key_data, scores, mask_data, num_heads,
                num_kv_heads, query_length, key_length, past_length, key_dim,
                key_capacity, cached, fp16_cache, causal, scale,
                query.stride[0] / sizeof(float),
                query.stride[1] / sizeof(float),
                query.stride[2] / sizeof(float),
                cached ? 1 : current_key.stride[0] / sizeof(float),
                cached ? static_cast<size_t>(key_dim)
                       : current_key.stride[1] / sizeof(float),
                cached ? static_cast<size_t>(key_capacity) * key_dim
                       : current_key.stride[2] / sizeof(float),
                mask ? mask->stride[0] / sizeof(float) : 0,
                mask ? mask->stride[1] / sizeof(float) : 0);
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "sdpa_scores_cuda")) {
                impl_->failed = true;
                return;
            }
            mollm_cuda::launch_sdpa_output(
                scores, value_data, output_data, num_heads, num_kv_heads,
                query_length, key_length, value_dim, key_capacity, cached,
                fp16_cache,
                cached ? 1 : current_value.stride[0] / sizeof(float),
                cached ? static_cast<size_t>(value_dim)
                       : current_value.stride[1] / sizeof(float),
                cached ? static_cast<size_t>(key_capacity) * value_dim
                       : current_value.stride[2] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "sdpa_output_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    const auto* matmul_weight =
        inputs.size() >= 2 && inputs[1] ? impl_->find_weight(*inputs[1])
                                        : nullptr;
    if (node.op_type == OpType::MATMUL && inputs.size() >= 2 && inputs[0] &&
        inputs[1] && output && inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 &&
        matmul_weight &&
        matmul_weight->n == inputs[1]->shape[0] &&
        matmul_weight->k == inputs[0]->shape[0] &&
        inputs[1]->device.offset == 0) {
        const Tensor& a = *inputs[0];
        const Tensor& weight = *inputs[1];
        const int m = static_cast<int>(a.shape[1]);
        const int k = static_cast<int>(a.shape[0]);
        const int n = static_cast<int>(weight.shape[0]);
        const int lda = static_cast<int>(a.stride[1] / sizeof(float));
        const int ldc = static_cast<int>(output->stride[1] / sizeof(float));
        const Activation activation_kind = static_cast<Activation>(
            graph_params::get_i32(node.params, 0, 0));
        const float* device_a = device_pointer_const<float>(a);
        float* device_c = device_pointer<float>(*output);
        const bool ok = device_a && device_c
            ? impl_->run_matmul_device(
                  device_a, lda, weight, device_c, ldc, m, n, k,
                  activation_kind,
                  graph_params::get_i32(node.params, 1, 0),
                  graph_params::get_i32(node.params, 2, -1))
            : impl_->run_matmul(
                  a.ptr<float>(), lda, weight, output->ptr<float>(), ldc,
                  m, n, k, activation_kind,
                  graph_params::get_i32(node.params, 1, 0),
                  graph_params::get_i32(node.params, 2, -1));
        if (ok) {
            record_native();
            return;
        }
        impl_->failed = true;
        return;
    }

    if (node.op_type == OpType::QK_RMS_NORM_ROPE && inputs.size() >= 6 &&
        inputs[0] && inputs[1] && inputs[2] && inputs[3] && inputs[4] &&
        inputs[5] && output && inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        fp32_contiguous(*inputs[2]) && fp32_contiguous(*inputs[3]) &&
        inputs[4]->prec == Precision::FP32 &&
        inputs[5]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& query = *inputs[0];
        const Tensor& key = *inputs[1];
        const Tensor& query_weight = *inputs[2];
        const Tensor& key_weight = *inputs[3];
        const Tensor& cosine = *inputs[4];
        const Tensor& sine = *inputs[5];
        const int width = static_cast<int>(output->shape[0]);
        const int sequence_length = static_cast<int>(output->shape[1]);
        const int total_heads = static_cast<int>(output->shape[2]);
        const int query_heads = graph_params::get_i32(
            node.params, 2, total_heads);
        const int key_heads = total_heads - query_heads;
        const int rope_dim = graph_params::get_i32(node.params, 0, width);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        const auto element_aligned = [](const Tensor& tensor) {
            for (int dimension = 0; dimension < 4; ++dimension)
                if (tensor.stride[dimension] % sizeof(float) != 0)
                    return false;
            return true;
        };
        const float* query_data = device_pointer_const<float>(query);
        const float* key_data = device_pointer_const<float>(key);
        const float* query_weight_data =
            device_pointer_const<float>(query_weight);
        const float* key_weight_data =
            device_pointer_const<float>(key_weight);
        const float* cosine_data = device_pointer_const<float>(cosine);
        const float* sine_data = device_pointer_const<float>(sine);
        float* destination = device_pointer<float>(*output);
        if (query_data && key_data && query_weight_data && key_weight_data &&
            cosine_data && sine_data && destination && width > 0 &&
            sequence_length > 0 && query_heads > 0 && key_heads > 0 &&
            rope_dim > 0 && rope_dim <= width && rope_dim % 2 == 0 &&
            query.shape[0] == width && key.shape[0] == width &&
            query.shape[1] >= sequence_length * query_heads &&
            key.shape[1] >= sequence_length * key_heads &&
            query.shape[2] == 1 && query.shape[3] == 1 &&
            key.shape[2] == 1 && key.shape[3] == 1 &&
            query_weight.shape[0] >= width && key_weight.shape[0] >= width &&
            cosine.shape[0] >= rope_dim / 2 &&
            sine.shape[0] >= rope_dim / 2 &&
            cosine.shape[1] >= sequence_length &&
            sine.shape[1] >= sequence_length && output->shape[3] == 1 &&
            element_aligned(query) && element_aligned(key) &&
            element_aligned(cosine) && element_aligned(sine) &&
            element_aligned(*output)) {
            mollm_cuda::launch_qk_rms_norm_rope(
                query_data, key_data, query_weight_data, key_weight_data,
                cosine_data, sine_data, destination, width, sequence_length,
                query_heads, total_heads, rope_dim, interleave,
                query.stride[0] / sizeof(float),
                query.stride[1] / sizeof(float),
                key.stride[0] / sizeof(float),
                key.stride[1] / sizeof(float),
                cosine.stride[0] / sizeof(float),
                cosine.stride[1] / sizeof(float),
                sine.stride[0] / sizeof(float),
                sine.stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "qk_rms_norm_rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RMS_NORM_ROPE && inputs.size() >= 4 &&
        inputs[0] && inputs[1] && inputs[2] && inputs[3] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        inputs[2]->prec == Precision::FP32 &&
        inputs[3]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& source = *inputs[0];
        const Tensor& weight = *inputs[1];
        const Tensor& cosine = *inputs[2];
        const Tensor& sine = *inputs[3];
        const float* source_data = device_pointer_const<float>(source);
        const float* weight_data = device_pointer_const<float>(weight);
        const float* cosine_data = device_pointer_const<float>(cosine);
        const float* sine_data = device_pointer_const<float>(sine);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(source.shape[0]);
        const int sequence_length = static_cast<int>(output->shape[1]);
        const int channels =
            static_cast<int>(output->shape[2] * output->shape[3]);
        const int rows = width > 0
            ? static_cast<int>(source.nelements() / width) : 0;
        const int rope_dim = graph_params::get_i32(node.params, 0, width);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        const size_t count = static_cast<size_t>(source.nelements());
        const size_t scratch_bytes = count * sizeof(float);
        if (source_data && weight_data && cosine_data && sine_data &&
            destination && width > 0 && rows > 0 && rope_dim > 0 &&
            rope_dim <= width && rope_dim % 2 == 0 &&
            rows == sequence_length * channels &&
            cosine.shape[0] >= rope_dim / 2 &&
            sine.shape[0] >= rope_dim / 2 &&
            impl_->reserve(impl_->norm_scratch,
                           impl_->norm_scratch_bytes, scratch_bytes)) {
            auto* normalized = static_cast<float*>(impl_->norm_scratch);
            mollm_cuda::launch_rms_norm(
                source_data, weight_data, normalized, width, rows,
                graph_params::get_f32(node.params, 0, 1e-6f));
            mollm_cuda::launch_rope(
                normalized, cosine_data, sine_data, destination, width,
                sequence_length, channels, static_cast<int>(output->shape[2]),
                rope_dim, interleave, 1, static_cast<size_t>(width),
                static_cast<size_t>(width) * sequence_length,
                static_cast<size_t>(width) * sequence_length *
                    output->shape[2],
                cosine.stride[0] / sizeof(float),
                cosine.stride[1] / sizeof(float),
                sine.stride[0] / sizeof(float),
                sine.stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                output->stride[3] / sizeof(float));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "rms_norm_rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SHORTCONV && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        inputs[2]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& input = *inputs[0];
        const Tensor& weight = *inputs[1];
        const Tensor& state = *inputs[2];
        const int groups = static_cast<int>(input.shape[0]);
        const int sequence_length = static_cast<int>(input.shape[1]);
        const int kernel_size = graph_params::get_i32(node.params, 0, 4);
        const int real_tokens = graph_params::get_i32(
            node.params, 1, sequence_length);
        const float* input_data = device_pointer_const<float>(input);
        const float* weight_data = device_pointer_const<float>(weight);
        float* state_data = device_pointer<float>(state);
        float* destination = device_pointer<float>(*output);
        if (input_data && weight_data && state_data && destination &&
            groups > 0 && sequence_length > 0 && kernel_size > 0 &&
            input.shape[2] == 1 && input.shape[3] == 1 &&
            output->shape[0] == input.shape[0] &&
            output->shape[1] == input.shape[1] &&
            output->shape[2] == 1 && output->shape[3] == 1 &&
            weight.nelements() >=
                static_cast<int64_t>(groups) * kernel_size &&
            state.nelements() >=
                static_cast<int64_t>(groups) * (kernel_size - 1) &&
            output->nelements() >=
                static_cast<int64_t>(groups) * sequence_length &&
            input.stride[0] % sizeof(float) == 0 &&
            input.stride[1] % sizeof(float) == 0 &&
            mollm_cuda::launch_shortconv(
                input_data, weight_data, state_data, destination, groups,
                sequence_length, kernel_size, real_tokens,
                input.stride[0] / sizeof(float),
                input.stride[1] / sizeof(float))) {
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "shortconv_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    const bool gdn_core =
        node.op_type == OpType::GATED_DELTANET_PREFILL ||
        node.op_type == OpType::GATED_DELTANET_DECODE;
    const bool gdn_conv_decode =
        node.op_type == OpType::GATED_DELTANET_CONV_DECODE;
    if ((gdn_core || gdn_conv_decode) &&
        inputs.size() >= (gdn_conv_decode ? 10u : 8u) && output &&
        std::all_of(
            inputs.begin(), inputs.begin() + 8,
            [](const Tensor* tensor) {
                return tensor && tensor->prec == Precision::FP32;
            }) &&
        (!gdn_conv_decode ||
         (inputs[8] && inputs[8]->prec == Precision::FP32 &&
          inputs[9] && inputs[9]->prec == Precision::FP32)) &&
        output->prec == Precision::FP32) {
        const int num_heads = graph_params::get_i32(node.params, 0, 16);
        const int key_dimension = graph_params::get_i32(node.params, 1, 128);
        const int value_dimension =
            graph_params::get_i32(node.params, 2, 128);
        const int sequence_length = graph_params::get_i32(node.params, 3, 1);
        const int real_tokens =
            graph_params::get_i32(node.params, 6, sequence_length);
        const int num_value_heads =
            graph_params::get_i32(node.params, 7, num_heads);
        const int qkv_key_width = num_heads * key_dimension;
        const int qkv_width =
            2 * qkv_key_width + num_value_heads * value_dimension;
        const int output_width = num_value_heads * value_dimension;
        const float* qkv = device_pointer_const<float>(*inputs[0]);
        const float* a = device_pointer_const<float>(*inputs[1]);
        const float* b = device_pointer_const<float>(*inputs[2]);
        const float* z = device_pointer_const<float>(*inputs[3]);
        const float* a_log = device_pointer_const<float>(*inputs[4]);
        const float* dt_bias = device_pointer_const<float>(*inputs[5]);
        const float* norm_weight = device_pointer_const<float>(*inputs[6]);
        float* state = device_pointer<float>(*inputs[7]);
        float* destination = device_pointer<float>(*output);
        float scale = graph_params::get_f32(node.params, 2, 0.0f);
        if (scale == 0.0f && key_dimension > 0)
            scale = 1.0f / std::sqrt(static_cast<float>(key_dimension));
        const bool valid = qkv && a && b && z && a_log && dt_bias &&
            norm_weight && state && destination && num_heads > 0 &&
            num_value_heads > 0 && num_value_heads % num_heads == 0 &&
            key_dimension == 128 && value_dimension == 128 &&
            sequence_length > 0 && inputs[0]->nelements() >=
                static_cast<int64_t>(qkv_width) * sequence_length &&
            inputs[1]->shape[0] >= num_value_heads &&
            inputs[2]->shape[0] >= num_value_heads &&
            inputs[3]->shape[0] >= output_width &&
            inputs[4]->nelements() >= num_value_heads &&
            inputs[5]->nelements() >= num_value_heads &&
            inputs[6]->nelements() >= value_dimension &&
            inputs[7]->nelements() >=
                static_cast<int64_t>(num_value_heads) * key_dimension *
                    value_dimension &&
            output->nelements() >=
                static_cast<int64_t>(output_width) * sequence_length &&
            inputs[1]->stride[0] == sizeof(float) &&
            inputs[2]->stride[0] == sizeof(float) &&
            inputs[3]->stride[0] == sizeof(float) &&
            output->is_contiguous();
        bool launched = false;
        if (valid && gdn_conv_decode) {
            const int convolution_kernel =
                graph_params::get_i32(node.params, 5, 4);
            const float* convolution_weight =
                device_pointer_const<float>(*inputs[8]);
            float* convolution_state = device_pointer<float>(*inputs[9]);
            const size_t scratch_bytes =
                static_cast<size_t>(qkv_width) * sizeof(float);
            const bool valid_convolution = sequence_length == 1 &&
                convolution_weight && convolution_state &&
                convolution_kernel > 0 && inputs[8]->nelements() >=
                    static_cast<int64_t>(qkv_width) * convolution_kernel &&
                inputs[9]->nelements() >=
                    static_cast<int64_t>(qkv_width) *
                        (convolution_kernel - 1) &&
                inputs[0]->stride[0] % sizeof(float) == 0 &&
                inputs[0]->stride[1] % sizeof(float) == 0;
            if (valid_convolution &&
                impl_->reserve(
                    impl_->recurrent_scratch,
                    impl_->recurrent_scratch_bytes, scratch_bytes)) {
                auto* convolved =
                    static_cast<float*>(impl_->recurrent_scratch);
                const bool convolution_launched =
                    mollm_cuda::launch_shortconv(
                        qkv, convolution_weight, convolution_state,
                        convolved, qkv_width, 1, convolution_kernel, 1,
                        inputs[0]->stride[0] / sizeof(float),
                        inputs[0]->stride[1] / sizeof(float));
                launched = convolution_launched && mollm_cuda::launch_gdn(
                    convolved, a, b, z, a_log, dt_bias, norm_weight, state,
                    destination, num_heads, num_value_heads, key_dimension,
                    value_dimension, 1, 1,
                    graph_params::get_i32(node.params, 4, 1) != 0,
                    graph_params::get_f32(node.params, 0, 1e-6f),
                    graph_params::get_f32(node.params, 1, 1e-6f), scale,
                    inputs[1]->stride[1] / sizeof(float),
                    inputs[2]->stride[1] / sizeof(float),
                    inputs[3]->stride[1] / sizeof(float));
                if (convolution_launched && !launched) {
                    impl_->failed = true;
                    return;
                }
            }
        } else if (valid) {
            const bool prefill =
                node.op_type == OpType::GATED_DELTANET_PREFILL;
            const size_t prefill_scratch_bytes = prefill
                ? mollm_cuda::gdn_prefill_scratch_bytes(
                      num_heads, num_value_heads, key_dimension,
                      value_dimension, sequence_length)
                : 0;
            void* prefill_scratch = nullptr;
            if (prefill_scratch_bytes > 0) {
                if (!impl_->reserve(
                        impl_->recurrent_scratch,
                        impl_->recurrent_scratch_bytes,
                        prefill_scratch_bytes)) {
                    impl_->failed = true;
                    return;
                }
                prefill_scratch = impl_->recurrent_scratch;
            }
            launched = mollm_cuda::launch_gdn(
                qkv, a, b, z, a_log, dt_bias, norm_weight, state,
                destination, num_heads, num_value_heads, key_dimension,
                value_dimension, sequence_length, real_tokens,
                graph_params::get_i32(node.params, 4, 1) != 0,
                graph_params::get_f32(node.params, 0, 1e-6f),
                graph_params::get_f32(node.params, 1, 1e-6f), scale,
                inputs[1]->stride[1] / sizeof(float),
                inputs[2]->stride[1] / sizeof(float),
                inputs[3]->stride[1] / sizeof(float), prefill_scratch,
                prefill_scratch_bytes);
        }
        if (launched) {
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "gdn_128_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::ADD_RMS_NORM && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        fp32_contiguous(*inputs[2]) && output->prec == Precision::FP32 &&
        inputs[0]->stride[0] == sizeof(float) &&
        inputs[1]->stride[0] == sizeof(float) &&
        output->stride[0] == sizeof(float) &&
        inputs[0]->shape[2] == 1 && inputs[0]->shape[3] == 1 &&
        inputs[1]->shape[2] == 1 && inputs[1]->shape[3] == 1 &&
        output->shape[2] == 1 && output->shape[3] == 1 &&
        inputs[0]->nelements() == inputs[1]->nelements() &&
        inputs[0]->nelements() == output->nelements()) {
        float* residual = device_pointer<float>(*inputs[0]);
        const float* update = device_pointer_const<float>(*inputs[1]);
        const float* weight = device_pointer_const<float>(*inputs[2]);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(inputs[0]->shape[0]);
        const int rows = static_cast<int>(inputs[0]->shape[1]);
        if (residual && update && weight && destination && width > 0 &&
            rows > 0) {
            mollm_cuda::launch_add_rms_norm(
                residual, update, weight, destination, width, rows,
                inputs[0]->stride[1] / sizeof(float),
                inputs[1]->stride[1] / sizeof(float),
                output->stride[1] / sizeof(float),
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "add_rms_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RMS_NORM && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output && fp32_contiguous(*inputs[0]) &&
        fp32_contiguous(*inputs[1]) && fp32_contiguous(*output)) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        const float* weight = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(inputs[0]->shape[0]);
        const int rows = width > 0
            ? static_cast<int>(inputs[0]->nelements() / width) : 0;
        if (source && weight && destination && rows > 0) {
            mollm_cuda::launch_rms_norm(
                source, weight, destination, width, rows,
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "rms_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::ADD || node.op_type == OpType::MUL) &&
        inputs.size() >= 2 && inputs[0] && inputs[1] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        fp32_contiguous(*output) &&
        inputs[0]->nelements() == inputs[1]->nelements()) {
        const float* lhs = device_pointer_const<float>(*inputs[0]);
        const float* rhs = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (lhs && rhs && destination) {
            mollm_cuda::launch_binary(
                lhs, rhs, destination, count,
                node.op_type == OpType::MUL);
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "binary_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SIGMOID_MUL && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 && fp32_contiguous(*output) &&
        same_shape(*inputs[0], *inputs[1]) &&
        inputs[0]->nelements() == output->nelements()) {
        const float* value = device_pointer_const<float>(*inputs[0]);
        const float* gate = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (value && gate && destination) {
            if (inputs[0]->is_contiguous() && inputs[1]->is_contiguous()) {
                mollm_cuda::launch_sigmoid_mul(value, gate, destination, count);
            } else {
                mollm_cuda::launch_sigmoid_mul_strided(
                    value, gate, destination, count, inputs[0]->shape[0],
                    inputs[0]->shape[1], inputs[0]->shape[2],
                    inputs[0]->stride[0] / sizeof(float),
                    inputs[0]->stride[1] / sizeof(float),
                    inputs[0]->stride[2] / sizeof(float),
                    inputs[0]->stride[3] / sizeof(float),
                    inputs[1]->stride[0] / sizeof(float),
                    inputs[1]->stride[1] / sizeof(float),
                    inputs[1]->stride[2] / sizeof(float),
                    inputs[1]->stride[3] / sizeof(float));
            }
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "sigmoid_mul_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    int unary_operation = -1;
    switch (node.op_type) {
    case OpType::SILU: unary_operation = 0; break;
    case OpType::GELU: unary_operation = 1; break;
    case OpType::TANH: unary_operation = 2; break;
    case OpType::SIGMOID:
    case OpType::SIGMOID_EXACT: unary_operation = 3; break;
    case OpType::EXP:
    case OpType::EXP_EXACT: unary_operation = 4; break;
    case OpType::SOFTPLUS: unary_operation = 5; break;
    default: break;
    }
    if (unary_operation >= 0 && !inputs.empty() && inputs[0] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*output)) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (source && destination) {
            mollm_cuda::launch_unary(
                source, destination, count, unary_operation);
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "unary_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SWIGLU && !inputs.empty() && inputs[0] &&
        output && fp32_contiguous(*inputs[0]) && fp32_contiguous(*output) &&
        inputs[0]->shape[0] == output->shape[0] * 2) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (source && destination) {
            mollm_cuda::launch_swiglu(
                source, destination, count,
                static_cast<size_t>(output->shape[0]));
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "swiglu_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (impl_->operator_fallback ==
        OperatorFallbackPolicy::REQUIRE_NATIVE) {
        std::fprintf(
            stderr,
            "CudaBackend: native-only mode rejected %s operator fallback\n",
            op_type_name(node.op_type));
        impl_->failed = true;
        return;
    }

    synchronize_for_host_read();
    if (impl_->failed)
        return;
    std::vector<Tensor> host_tensors;
    std::vector<const Tensor*> host_inputs;
    std::vector<std::vector<uint64_t>> host_storage;
    std::vector<int> staged_storage(inputs.size(), -1);
    host_tensors.reserve(inputs.size());
    host_inputs.reserve(inputs.size());
    host_storage.reserve(inputs.size());
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        const Tensor* input = inputs[input_index];
        if (!input) {
            host_inputs.push_back(nullptr);
            continue;
        }
        host_tensors.push_back(*input);
        Tensor& host = host_tensors.back();
        // Prepared constants retain their package-native host representation.
        // Device intermediates and persistent state are staged from the first
        // byte of the tensor view, preserving their original shape/strides.
        const bool device_weight = impl_->find_weight(*input) != nullptr;
        if (input->device.buffer && !device_weight) {
            const size_t bytes = input->view_span_bytes();
            host_storage.emplace_back(
                (bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
            staged_storage[input_index] =
                static_cast<int>(host_storage.size() - 1);
            void* destination = host_storage.back().data();
            if (!copy_to_host(*input, destination, bytes)) {
                impl_->failed = true;
                return;
            }
            host.data = destination;
        } else if (device_weight && !host.data) {
            std::fprintf(
                stderr, "CudaBackend: %s fallback lacks host weight storage\n",
                op_type_name(node.op_type));
            impl_->failed = true;
            return;
        }
        host.device.buffer = nullptr;
        host.device.offset = 0;
        host_inputs.push_back(&host);
    }
    if (!output || !output->device.buffer) {
        impl_->failed = true;
        return;
    }
    const size_t output_bytes = output->view_span_bytes();
    std::vector<uint64_t> host_output_storage(
        (output_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    Tensor host_output = *output;
    host_output.data = host_output_storage.data();
    host_output.device.buffer = nullptr;
    host_output.device.offset = 0;
    // Staging allocations can reuse addresses/storage IDs across fallback
    // operators within one device graph. Their contents have a new lifetime.
    impl_->cpu.begin_execution();
    impl_->cpu.clear_dispatch_error();
    impl_->cpu.dispatch(node, host_inputs, &host_output, thread_pool);
    if (impl_->cpu.dispatch_failed()) {
        impl_->failed = true;
        return;
    }
    // Reference kernels may mutate cache or recurrent-state inputs. Copy every
    // staged input back; this is intentionally centralized so new fallback
    // operators cannot silently lose in-place updates.
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        const int storage_index = staged_storage[input_index];
        if (storage_index < 0)
            continue;
        Tensor& input = *const_cast<Tensor*>(inputs[input_index]);
        const size_t bytes = input.view_span_bytes();
        if (!copy_from_host(
                host_storage[static_cast<size_t>(storage_index)].data(),
                input, bytes)) {
            impl_->failed = true;
            return;
        }
    }
    if (host_output.data != host_output_storage.data() ||
        !copy_from_host(host_output.data, *output, output_bytes)) {
        impl_->failed = true;
        return;
    }
    ++impl_->fallback_ops[static_cast<uint32_t>(node.op_type)];
}
