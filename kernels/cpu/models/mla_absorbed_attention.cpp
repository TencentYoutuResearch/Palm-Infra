#include "kernels/cpu/models/mla_absorbed_attention.h"

#include "core/cache_layout.h"
#include "graph/graph.h"
#include "runtime/threading.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

bool kernel_mla_absorbed_attention(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* thread_pool) {
    if (!output || inputs.size() < 8)
        return false;
    for (int i = 0; i < 8; ++i)
        if (!inputs[i] || !inputs[i]->data)
            return false;

    const Tensor& q_nope = *inputs[0];
    const Tensor& q_rope = *inputs[1];
    const Tensor& kv_cur = *inputs[2];
    const Tensor& rope_cur = *inputs[3];
    const Tensor& k_b = *inputs[4];
    const Tensor& v_b = *inputs[5];
    const Tensor& mask = *inputs[6];
    const Tensor& cache = *inputs[7];
    const auto& p = node.params;
    const int heads = graph_params::get_i32(p, 2, 0);
    const int nope = (int)q_nope.shape[0];
    const int rope = graph_params::get_i32(p, 9, 0);
    const int rank = graph_params::get_i32(p, 8, 0);
    const int vdim = graph_params::get_i32(p, 5, 0);
    const int seq = (int)q_nope.shape[1];
    const float scale = graph_params::get_f32(p, 0, 0.f);
    const int sink_idx = graph_params::get_i32(p, 6, -1);
    const int selection_idx = graph_params::get_i32(p, 7, -1);
    if (heads <= 0 || nope <= 0 || rope <= 0 || rank <= 0 || vdim <= 0 ||
        seq <= 0 || scale <= 0 ||
        graph_params::get_i32(p, 0, 0) != 2 ||
        graph_params::get_i32(p, 1, 0) != 1 ||
        q_nope.prec != Precision::FP32 || q_rope.prec != Precision::FP32 ||
        kv_cur.prec != Precision::FP32 || rope_cur.prec != Precision::FP32 ||
        k_b.prec != Precision::FP16 || v_b.prec != Precision::FP16 ||
        cache.prec != Precision::FP16 || output->prec != Precision::FP32 ||
        mask.prec != Precision::FP32 ||
        q_rope.shape[0] != rope || q_rope.shape[1] != seq ||
        kv_cur.shape[0] != rank || kv_cur.shape[1] != seq ||
        rope_cur.shape[0] != rope || rope_cur.shape[1] != seq ||
        k_b.shape[0] != heads || k_b.shape[1] != rank || k_b.shape[2] != nope ||
        v_b.shape[0] != heads || v_b.shape[1] != vdim || v_b.shape[2] != rank ||
        output->shape[0] != vdim || output->shape[1] != seq ||
        output->shape[2] != heads)
        return false;

    const float* sink = nullptr;
    if (sink_idx >= 0) {
        if ((size_t)sink_idx >= inputs.size() || !inputs[sink_idx] ||
            !inputs[sink_idx]->data ||
            inputs[sink_idx]->prec != Precision::FP32 ||
            inputs[sink_idx]->nelements() < heads)
            return false;
        sink = inputs[sink_idx]->ptr<float>();
    }
    const Tensor* selection = nullptr;
    if (selection_idx >= 0) {
        if ((size_t)selection_idx >= inputs.size() ||
            !inputs[selection_idx] || !inputs[selection_idx]->data ||
            inputs[selection_idx]->prec != Precision::INT32 ||
            inputs[selection_idx]->shape[0] <= 0 ||
            inputs[selection_idx]->shape[1] < seq)
            return false;
        selection = inputs[selection_idx];
    }
    CacheMetadata* meta = cache_meta(cache.data);
    const int past = (int)meta->current_seq_len;
    const int capacity = (int)meta->max_seq_len;
    const int total = past + seq;
    if (past < 0 || total > capacity || meta->head_dim != (uint64_t)(rank + rope) ||
        meta->num_kv_heads != 1 || mask.shape[0] < total || mask.shape[1] < seq)
        return false;

    auto* latent = static_cast<__fp16*>(cache_data(cache.data));
    const int width = rank + rope;
    for (int s = 0; s < seq; ++s) {
        const float* kv = reinterpret_cast<const float*>(
            static_cast<const char*>(kv_cur.data) + (size_t)s * kv_cur.stride[1]);
        const float* kr = reinterpret_cast<const float*>(
            static_cast<const char*>(rope_cur.data) + (size_t)s * rope_cur.stride[1]);
        __fp16* dest = latent + (size_t)(past + s) * width;
        for (int c = 0; c < rank; ++c) dest[c] = (__fp16)kv[c];
        for (int c = 0; c < rope; ++c) dest[rank + c] = (__fp16)kr[c];
    }

    const auto* kb = k_b.ptr<__fp16>();
    const auto* vb = v_b.ptr<__fp16>();
    auto run_heads = [&](int, int first, int end) {
        std::vector<float> q_abs(rank);
        std::vector<float> weighted(rank);
        for (int h = first; h < end; ++h) {
            const __fp16* kh = kb + (size_t)h * rank * nope;
            const __fp16* vh = vb + (size_t)h * vdim * rank;
            for (int s = 0; s < seq; ++s) {
                const float* mask_row = reinterpret_cast<const float*>(
                    static_cast<const char*>(mask.data) +
                    (size_t)s * mask.stride[1]);
                const float* qn = reinterpret_cast<const float*>(
                    static_cast<const char*>(q_nope.data) +
                    (size_t)h * q_nope.stride[2] + (size_t)s * q_nope.stride[1]);
                const float* qr = reinterpret_cast<const float*>(
                    static_cast<const char*>(q_rope.data) +
                    (size_t)h * q_rope.stride[2] + (size_t)s * q_rope.stride[1]);
                for (int c = 0; c < rank; ++c) {
                    float sum = 0.f;
                    const __fp16* row = kh + (size_t)c * nope;
                    for (int d = 0; d < nope; ++d) sum += qn[d] * (float)row[d];
                    q_abs[c] = sum;
                }
                std::fill(weighted.begin(), weighted.end(), 0.f);
                float maximum = -std::numeric_limits<float>::infinity();
                float denominator = 0.f;
                const int visible = past + s + 1;
                const bool dense = !selection || selection->shape[0] >= visible;
                const int candidates = dense ? visible : (int)selection->shape[0];
                const int32_t* selected = dense ? nullptr :
                    reinterpret_cast<const int32_t*>(
                        static_cast<const char*>(selection->data) +
                        (size_t)s * selection->stride[1]);
                for (int slot = 0; slot < candidates; ++slot) {
                    const int t = dense ? slot : selected[slot];
                    // The indexer is a ranking heuristic. The attention
                    // kernel must re-apply causality to every selected key.
                    if (t < 0 || t >= visible) continue;
                    const __fp16* key = latent + (size_t)t * width;
                    float score = 0.f;
                    for (int c = 0; c < rank; ++c) score += q_abs[c] * (float)key[c];
                    for (int d = 0; d < rope; ++d)
                        score += qr[d] * (float)key[rank + d];
                    score = score * scale + mask_row[t];
                    if (!std::isfinite(score)) continue;
                    const float next_max = std::max(maximum, score);
                    const float old_factor = std::isfinite(maximum)
                        ? std::exp(maximum - next_max) : 0.f;
                    const float new_factor = std::exp(score - next_max);
                    denominator = denominator * old_factor + new_factor;
                    for (int c = 0; c < rank; ++c)
                        weighted[c] = weighted[c] * old_factor +
                                      new_factor * (float)key[c];
                    maximum = next_max;
                }
                if (sink) {
                    const float next_max = std::max(maximum, sink[h]);
                    const float old_factor = std::isfinite(maximum)
                        ? std::exp(maximum - next_max) : 0.f;
                    denominator = denominator * old_factor +
                                  std::exp(sink[h] - next_max);
                    for (float& value : weighted) value *= old_factor;
                }
                float* out = reinterpret_cast<float*>(
                    static_cast<char*>(output->data) +
                    (size_t)h * output->stride[2] + (size_t)s * output->stride[1]);
                if (denominator == 0.f) {
                    std::fill(out, out + vdim, 0.f);
                    continue;
                }
                for (int d = 0; d < vdim; ++d) {
                    const __fp16* row = vh + (size_t)d * rank;
                    float sum = 0.f;
                    for (int c = 0; c < rank; ++c)
                        sum += (weighted[c] / denominator) * (float)row[c];
                    out[d] = sum;
                }
            }
        }
    };
    if (thread_pool && heads > 1)
        thread_pool->parallel_for(0, heads, 1, run_heads);
    else
        run_heads(0, 0, heads);
    meta->current_seq_len = (uint64_t)total;
    return true;
}
