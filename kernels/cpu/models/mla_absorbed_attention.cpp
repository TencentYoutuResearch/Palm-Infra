#include "kernels/cpu/models/mla_absorbed_attention.h"

#include "core/cache_layout.h"
#include "graph/graph.h"
#include "runtime/threading.h"
#include "kernels/cpu/matmul/gguf_k.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
template <typename Weight, int Rows>
void dot_rows(const float* input, const Weight* const* rows,
              float* output, int columns) {
#pragma clang fp contract(off)
    static_assert(Rows == 4 || Rows == 8);
    float32x4_t sums[Rows / 4];
    for (auto& sum : sums) sum = vdupq_n_f32(0.f);
    const auto load = [](const Weight* p) {
        if constexpr (sizeof(Weight) == 2)
            return vcvt_f32_f16(vld1_f16(p));
        else
            return vld1q_f32(p);
    };
    int k = 0;
    for (; k + 4 <= columns; k += 4) {
        const float32x4_t x = vld1q_f32(input + k);
#pragma unroll
        for (int g = 0; g < Rows / 4; ++g) {
            const float32x4_t r0 = load(rows[g * 4] + k);
            const float32x4_t r1 = load(rows[g * 4 + 1] + k);
            const float32x4_t r2 = load(rows[g * 4 + 2] + k);
            const float32x4_t r3 = load(rows[g * 4 + 3] + k);
            const float32x4_t t0 = vtrn1q_f32(r0, r1), t1 = vtrn2q_f32(r0, r1);
            const float32x4_t t2 = vtrn1q_f32(r2, r3), t3 = vtrn2q_f32(r2, r3);
            // Each lane owns one output row. Transpose in registers so its
            // products retain the scalar K order, with no horizontal sum.
            const float32x4_t v0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
            const float32x4_t v1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
            const float32x4_t v2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
            const float32x4_t v3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));
            sums[g] = vaddq_f32(sums[g], vmulq_laneq_f32(v0, x, 0));
            sums[g] = vaddq_f32(sums[g], vmulq_laneq_f32(v1, x, 1));
            sums[g] = vaddq_f32(sums[g], vmulq_laneq_f32(v2, x, 2));
            sums[g] = vaddq_f32(sums[g], vmulq_laneq_f32(v3, x, 3));
        }
    }
    for (; k < columns; ++k) {
#pragma unroll
        for (int g = 0; g < Rows / 4; ++g) {
            const float32x4_t w = {(float)rows[g * 4][k],
                (float)rows[g * 4 + 1][k], (float)rows[g * 4 + 2][k],
                (float)rows[g * 4 + 3][k]};
            sums[g] = vaddq_f32(sums[g], vmulq_n_f32(w, input[k]));
        }
    }
    for (int g = 0; g < Rows / 4; ++g) vst1q_f32(output + g * 4, sums[g]);
}
#endif

template <typename Weight>
void project_rows(const float* input, const Weight* weights,
                  float* output, int rows, int columns) {
    int row = 0;
#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
    // Interleave independent rows to hide the reduction dependency while
    // preserving each row's product rounding and accumulation order.
    for (; row + 8 <= rows; row += 8) {
        const Weight* group[8];
        for (int i = 0; i < 8; ++i)
            group[i] = weights + (size_t)(row + i) * columns;
        dot_rows<Weight, 8>(input, group, output + row, columns);
    }
    for (; row + 4 <= rows; row += 4) {
        const Weight* w0 = weights + (size_t)row * columns;
        const Weight* group[] = {w0, w0 + columns,
                                 w0 + 2 * columns, w0 + 3 * columns};
        dot_rows<Weight, 4>(input, group, output + row, columns);
    }
#endif
    for (; row < rows; ++row) {
        float sum = 0.f;
        const Weight* weight = weights + (size_t)row * columns;
        for (int k = 0; k < columns; ++k)
            sum += input[k] * (float)weight[k];
        output[row] = sum;
    }
}

// Project from native blocks without materializing a full FP32 matrix.
// Heads are already parallelized by the caller; keep this GEMV single-threaded.
void project_q8_rows(const float* input, const uint8_t* weights,
                     float* output, int rows, int columns) {
    Tensor a = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                             columns, 1, 1, 1, const_cast<float*>(input));
    Tensor b = Tensor::create(Precision::GGUF_Q8_0, MemoryType::EXTERNAL,
                             rows, columns, 1, 1, const_cast<uint8_t*>(weights));
    Tensor c = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                             rows, 1, 1, 1, output);
    kernel_matmul_gguf_k(a, b, c, nullptr, Activation::NONE, 0, 0);
}

template <typename CacheValue>
bool mla_absorbed_attention(
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
        (k_b.prec != Precision::FP16 &&
         k_b.prec != Precision::GGUF_Q8_0) ||
        (v_b.prec != Precision::FP16 &&
         v_b.prec != Precision::GGUF_Q8_0) ||
        output->prec != Precision::FP32 ||
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

    auto* latent = static_cast<CacheValue*>(cache_data(cache.data));
    const int width = rank + rope;
    int score_batch = 1;
#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
    // Preserve scalar tails at the latent/RoPE boundary on generic shapes.
    if (rank % 4 == 0 && rope % 4 == 0) score_batch = 8;
#endif
    for (int s = 0; s < seq; ++s) {
        const float* kv = reinterpret_cast<const float*>(
            static_cast<const char*>(kv_cur.data) + (size_t)s * kv_cur.stride[1]);
        const float* kr = reinterpret_cast<const float*>(
            static_cast<const char*>(rope_cur.data) + (size_t)s * rope_cur.stride[1]);
        CacheValue* dest = latent + (size_t)(past + s) * width;
        for (int c = 0; c < rank; ++c) dest[c] = (CacheValue)kv[c];
        for (int c = 0; c < rope; ++c) dest[rank + c] = (CacheValue)kr[c];
    }

    const auto* kb = k_b.prec == Precision::FP16 ? k_b.ptr<__fp16>() : nullptr;
    const auto* vb = v_b.prec == Precision::FP16 ? v_b.ptr<__fp16>() : nullptr;
    const auto* kb_q8 = static_cast<const uint8_t*>(k_b.data);
    const auto* vb_q8 = static_cast<const uint8_t*>(v_b.data);
    // Prefill amortizes decoding across tokens. Direct GEMV requires complete
    // Q8 blocks per row; generic shapes retain the decoded projection path.
    const bool direct_k = seq == 1 && nope % 32 == 0;
    const bool direct_v = seq == 1 && rank % 32 == 0;
    auto run_heads = [&](int, int first, int end) {
        std::vector<float> q_abs(width);
        std::vector<float> weighted(rank);
        std::vector<float> k_decoded, v_decoded;
        if (!kb && !direct_k) k_decoded.resize((size_t)rank * nope);
        if (!vb && !direct_v) v_decoded.resize((size_t)vdim * rank);
        for (int h = first; h < end; ++h) {
            const __fp16* kh = kb ? kb + (size_t)h * rank * nope : nullptr;
            const __fp16* vh = vb ? vb + (size_t)h * vdim * rank : nullptr;
            if (!kb && !direct_k) {
                const size_t blocks = (size_t)rank * nope / 32;
                const uint8_t* source = kb_q8 + (size_t)h * blocks * 34;
                for (size_t b = 0; b < blocks; ++b)
                    gguf_k_dequantize_block(
                        Precision::GGUF_Q8_0, source + b * 34,
                        k_decoded.data() + b * 32);
            }
            if (!vb && !direct_v) {
                const size_t blocks = (size_t)vdim * rank / 32;
                const uint8_t* source = vb_q8 + (size_t)h * blocks * 34;
                for (size_t b = 0; b < blocks; ++b)
                    gguf_k_dequantize_block(
                        Precision::GGUF_Q8_0, source + b * 34,
                        v_decoded.data() + b * 32);
            }
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
                if (kh)
                    project_rows(qn, kh, q_abs.data(), rank, nope);
                else if (direct_k)
                    project_q8_rows(qn, kb_q8 + (size_t)h * rank * nope / 32 * 34,
                                    q_abs.data(), rank, nope);
                else
                    project_rows(qn, k_decoded.data(), q_abs.data(), rank, nope);
                // The joined query lets the score batch reuse the row dot.
                std::copy_n(qr, rope, q_abs.data() + rank);
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
                for (int slot = 0; slot < candidates; slot += score_batch) {
                    const int count = std::min(score_batch, candidates - slot);
                    int positions[8];
                    const CacheValue* keys[8];
                    float scores[8] = {};
                    for (int lane = 0; lane < count; ++lane) {
                        const int t = dense ? slot + lane : selected[slot + lane];
                        positions[lane] = t;
                        // Invalid selections use a safe pointer for the dot
                        // product and are discarded before the softmax update.
                        const int safe_t = t >= 0 && t < visible ? t : 0;
                        keys[lane] = latent + (size_t)safe_t * width;
                    }
#if defined(__APPLE__) && defined(__clang__) && defined(__aarch64__)
                    if (count == 8) {
                        dot_rows<CacheValue, 8>(q_abs.data(), keys, scores, width);
                    } else if (count == 4) {
                        dot_rows<CacheValue, 4>(q_abs.data(), keys, scores, width);
                    } else
#endif
                    {
                        for (int lane = 0; lane < count; ++lane) {
                            float score = 0.f;
                            const CacheValue* key = keys[lane];
                            for (int c = 0; c < rank; ++c)
                                score += q_abs[c] * (float)key[c];
                            for (int d = 0; d < rope; ++d)
                                score += qr[d] * (float)key[rank + d];
                            scores[lane] = score;
                        }
                    }
                    // Preserve the original key order for online softmax.
                    for (int lane = 0; lane < count; ++lane) {
                        const int t = positions[lane];
                        if (t < 0 || t >= visible) continue;
                        const CacheValue* key = keys[lane];
                        const float score = scores[lane] * scale + mask_row[t];
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
                // Every output row uses the same normalized latent vector.
                // Divide once per element, retaining FP32 division rounding.
                for (float& value : weighted) value /= denominator;
                if (vh)
                    project_rows(weighted.data(), vh, out, vdim, rank);
                else if (direct_v)
                    project_q8_rows(weighted.data(), vb_q8 + (size_t)h * vdim * rank / 32 * 34,
                                    out, vdim, rank);
                else
                    project_rows(weighted.data(), v_decoded.data(), out, vdim, rank);
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

} // namespace

bool kernel_mla_absorbed_attention(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, ThreadPool* thread_pool) {
    if (inputs.size() < 8 || !inputs[7]) return false;
    switch (inputs[7]->prec) {
    case Precision::FP16:
        return mla_absorbed_attention<__fp16>(node, inputs, output, thread_pool);
    case Precision::FP32:
        return mla_absorbed_attention<float>(node, inputs, output, thread_pool);
    default:
        return false;
    }
}
