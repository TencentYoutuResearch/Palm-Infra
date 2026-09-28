#include "kernels/cpu/models/mla_absorbed_attention.h"
#include "core/cache_layout.h"
#include "graph/graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// Exercise eight/four-row projections, a V tail, and score batch boundaries.
constexpr int H = 2, C = 8, D = 2, R = 4, V = 13, CAP = 13;

Tensor fp32(float* data, int a, int b = 1, int c = 1) {
    return Tensor::create(Precision::FP32, MemoryType::EXTERNAL, a, b, c, 1, data);
}
Tensor fp16(__fp16* data, int a, int b = 1, int c = 1) {
    return Tensor::create(Precision::FP16, MemoryType::EXTERNAL, a, b, c, 1, data);
}

float explicit_output(
        int head, int query, int dim, int visible,
        const float* qn, const float* qr,
        const __fp16* kb, const __fp16* vb,
        const std::vector<float>& latent, float sink,
        const std::vector<int>* positions = nullptr) {
    float scores[CAP] = {};
    float maximum = sink;
    const std::vector<int> all_positions = [&] {
        std::vector<int> result;
        for (int t = 0; t < visible; ++t) result.push_back(t);
        return result;
    }();
    const auto& candidates = positions ? *positions : all_positions;
    for (int t : candidates) {
        if (t < 0 || t >= visible) continue;
        const float* key = latent.data() + t * (C + R);
        float score = 0.f;
        for (int d = 0; d < D; ++d) {
            float expanded = 0.f;
            for (int c = 0; c < C; ++c)
                expanded += (float)kb[(head * C + c) * D + d] * key[c];
            score += qn[(head * 3 + query) * D + d] * expanded;
        }
        for (int r = 0; r < R; ++r)
            score += qr[(head * 3 + query) * R + r] * key[C + r];
        scores[t] = score * 0.5f;
        maximum = std::max(maximum, scores[t]);
    }
    float denominator = std::exp(sink - maximum);
    float numerator = 0.f;
    for (int t : candidates) {
        if (t < 0 || t >= visible) continue;
        const float weight = std::exp(scores[t] - maximum);
        denominator += weight;
        const float* key = latent.data() + t * (C + R);
        float value = 0.f;
        for (int c = 0; c < C; ++c)
            value += (float)vb[(head * V + dim) * C + c] * key[c];
        numerator += weight * value;
    }
    return numerator / denominator;
}

bool close(float a, float b) { return std::fabs(a - b) < 0.003f; }

template <typename CacheValue, int Seq = 3, int Nope = 4>
bool q8_weights_match_fp16() {
    constexpr int heads = 2, rank = 32, nope = Nope, rope = 4;
    constexpr int vdim = 5, seq = Seq, past = 7, capacity = past + seq;
    std::vector<__fp16> kb(heads * rank * nope), vb(heads * vdim * rank);
    auto encode = [](std::vector<__fp16>& weights) {
        std::vector<uint8_t> bytes(weights.size() / 32 * 34);
        for (size_t b = 0; b < weights.size() / 32; ++b) {
            const __fp16 scale = (__fp16)(b % 2 ? -0.125f : 0.0625f);
            std::memcpy(bytes.data() + b * 34, &scale, sizeof(scale));
            for (int k = 0; k < 32; ++k) {
                const int8_t q = static_cast<int8_t>(int((b * 13 + k * 7) % 255) - 127);
                std::memcpy(bytes.data() + b * 34 + 2 + k, &q, sizeof(q));
                // Powers-of-two scales make both representations exact.
                weights[b * 32 + k] = (__fp16)((float)scale * q);
            }
        }
        return bytes;
    };
    auto kb_bytes = encode(kb), vb_bytes = encode(vb);
    Tensor kb_half = fp16(kb.data(), heads, rank, nope);
    Tensor vb_half = fp16(vb.data(), heads, vdim, rank);
    Tensor kb_q8 = Tensor::create(Precision::GGUF_Q8_0, MemoryType::EXTERNAL,
                                  heads, rank, nope, 1, kb_bytes.data());
    Tensor vb_q8 = Tensor::create(Precision::GGUF_Q8_0, MemoryType::EXTERNAL,
                                  heads, vdim, rank, 1, vb_bytes.data());
    std::vector<float> qn(heads * seq * nope), qr(heads * seq * rope);
    std::vector<float> kv(seq * rank), kr(seq * rope), mask(seq * capacity, 0.f);
    for (auto* values : {&qn, &qr, &kv, &kr})
        for (size_t i = 0; i < values->size(); ++i)
            (*values)[i] = (int(i * 7 % 19) - 9) * 0.1f;
    for (int s = 0; s < seq; ++s)
        mask[s * capacity + 2] = -std::numeric_limits<float>::infinity();
    std::vector<unsigned char> initial(CacheMetadata::SIZE + capacity * (rank + rope) * sizeof(CacheValue));
    *cache_meta(initial.data()) = CacheMetadata{};
    cache_meta(initial.data())->max_seq_len = capacity;
    cache_meta(initial.data())->current_seq_len = past;
    cache_meta(initial.data())->head_dim = rank + rope;
    cache_meta(initial.data())->num_kv_heads = 1;
    auto* latent = static_cast<CacheValue*>(cache_data(initial.data()));
    for (int i = 0; i < past * (rank + rope); ++i)
        latent[i] = (CacheValue)((i * 3 % 17 - 8) * 0.125f);
    float sink_data[heads] = {0.3f, -0.4f};
    int32_t indices[15] = {7, 2, -1, capacity, 0,
                               8, 2, 3, 0, 1,
                               9, 7, 2, 1, 0};
    Tensor qn_t = fp32(qn.data(), nope, seq, heads);
    Tensor qr_t = fp32(qr.data(), rope, seq, heads);
    Tensor kv_t = fp32(kv.data(), rank, seq), kr_t = fp32(kr.data(), rope, seq);
    Tensor mask_t = fp32(mask.data(), capacity, seq), sink_t = fp32(sink_data, heads);
    Tensor selection = Tensor::create(Precision::INT32, MemoryType::EXTERNAL,
                                      5, seq, 1, 1, indices);
    GraphNode node;
    node.op_type = OpType::SDPA_MLA_ABSORB;
    node.params.f32 = {0.5f};
    for (bool sparse : {false, true}) {
        node.params.i32 = {2, 1, heads, 1, nope + rope, vdim, 8,
                           sparse ? 9 : -1, rank, rope};
        std::vector<float> expected;
        for (int format = 0; format < 4; ++format) {
            auto storage = initial;
            std::vector<float> result(heads * seq * vdim);
            Tensor cache = Tensor::create(
                sizeof(CacheValue) == 4 ? Precision::FP32 : Precision::FP16,
                MemoryType::EXTERNAL, storage.size() / sizeof(CacheValue),
                1, 1, 1, storage.data());
            Tensor out = fp32(result.data(), vdim, seq, heads);
            std::vector<const Tensor*> inputs = {
                &qn_t, &qr_t, &kv_t, &kr_t,
                format & 1 ? &kb_q8 : &kb_half,
                format & 2 ? &vb_q8 : &vb_half,
                &mask_t, &cache, &sink_t, &selection};
            if (!kernel_mla_absorbed_attention(node, inputs, &out, nullptr) ||
                cache_meta(storage.data())->current_seq_len != capacity)
                return false;
            const auto* saved = static_cast<const CacheValue*>(cache_data(storage.data()));
            for (int t = 0; t < seq; ++t) {
                for (int c = 0; c < rank; ++c)
                    if (saved[(past + t) * (rank + rope) + c] !=
                        (CacheValue)kv[t * rank + c]) return false;
                for (int c = 0; c < rope; ++c)
                    if (saved[(past + t) * (rank + rope) + rank + c] !=
                        (CacheValue)kr[t * rope + c]) return false;
            }
            if (format == 0) expected = result;
            for (size_t i = 0; i < result.size(); ++i) {
                if (!std::isfinite(result[i])) return false;
                // Q8 block-scale GEMV keeps the exact encoded weights but
                // changes the FP32 reduction order relative to FP16 GEMV.
                const bool match = format == 0 ||
                    std::fabs(result[i] - expected[i]) <=
                        1e-5f * std::max(1.f, std::fabs(expected[i]));
                if (!match) {
                    std::fprintf(stderr, "Q8 mismatch cache=%zu seq=%d nope=%d sparse=%d format=%d index=%zu got=%g want=%g diff=%g\n",
                                 sizeof(CacheValue), seq, nope, int(sparse), format, i,
                                 result[i], expected[i], result[i] - expected[i]);
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

int main() {
    __fp16 k_b[H * C * D], v_b[H * V * C];
    for (int i = 0; i < H * C * D; ++i)
        k_b[i] = (__fp16)((i * 7 % 13 - 6) * 0.125f);
    for (int i = 0; i < H * V * C; ++i)
        v_b[i] = (__fp16)((i * 3 % 11 - 5) * 0.125f);
    float q_nope[H * 3 * D] = {
        1, -2, 0.5f, 1.5f, -1, 0.25f,
        -0.5f, 2, 1, -1, 0.75f, 1.25f};
    float q_rope[H * 3 * R], kv[3 * C], kr[3 * R];
    for (int i = 0; i < H * 3 * R; ++i)
        q_rope[i] = (i * 3 % 9 - 4) * 0.25f;
    for (int i = 0; i < 3 * C; ++i)
        kv[i] = (i * 5 % 13 - 6) * 0.25f;
    for (int i = 0; i < 3 * R; ++i)
        kr[i] = (i * 7 % 9 - 4) * 0.25f;
    float sink[H] = {0.3f, -0.4f};
    std::vector<unsigned char> cache_storage(CacheMetadata::SIZE + CAP * (C + R) * 2);
    auto* meta = cache_meta(cache_storage.data());
    *meta = CacheMetadata{};
    meta->max_seq_len = CAP;
    meta->head_dim = C + R;
    meta->num_kv_heads = 1;
    Tensor cache = fp16(static_cast<__fp16*>(static_cast<void*>(cache_storage.data())),
                        (int)cache_storage.size() / 2);
    Tensor kb_t = fp16(k_b, H, C, D), vb_t = fp16(v_b, H, V, C);
    Tensor sink_t = fp32(sink, H);
    GraphNode node;
    node.op_type = OpType::SDPA_MLA_ABSORB;
    node.params.i32 = {2, 1, H, 1, D + R, V, 8, -1, C, R};
    node.params.f32 = {0.5f};
    std::vector<float> all_latent;

    for (int pass = 0; pass < 2; ++pass) {
        const int start = pass == 0 ? 0 : 2;
        const int seq = pass == 0 ? 2 : 1;
        float mask_data[6] = {};
        float output_data[H * 2 * V] = {};
        float qn_packed[H * 2 * D] = {};
        float qr_packed[H * 2 * R] = {};
        for (int h = 0; h < H; ++h)
            for (int s = 0; s < seq; ++s) {
                std::copy_n(q_nope + (h * 3 + start + s) * D, D,
                            qn_packed + (h * seq + s) * D);
                std::copy_n(q_rope + (h * 3 + start + s) * R, R,
                            qr_packed + (h * seq + s) * R);
            }
        Tensor qn_t = fp32(qn_packed, D, seq, H);
        Tensor qr_t = fp32(qr_packed, R, seq, H);
        Tensor kv_t = fp32(kv + start * C, C, seq);
        Tensor kr_t = fp32(kr + start * R, R, seq);
        Tensor mask_t = fp32(mask_data, start + seq, seq);
        Tensor output = fp32(output_data, V, seq, H);
        std::vector<const Tensor*> inputs = {
            &qn_t, &qr_t, &kv_t, &kr_t, &kb_t, &vb_t, &mask_t, &cache, &sink_t};
        if (!kernel_mla_absorbed_attention(node, inputs, &output, nullptr)) {
            std::fprintf(stderr, "MLA kernel rejected pass %d\n", pass);
            return 1;
        }
        for (int s = 0; s < seq; ++s) {
            for (int c = 0; c < C; ++c)
                all_latent.push_back((float)(__fp16)kv[(start + s) * C + c]);
            for (int r = 0; r < R; ++r)
                all_latent.push_back((float)(__fp16)kr[(start + s) * R + r]);
        }
        if (meta->current_seq_len != (uint64_t)(start + seq)) return 2;
        for (int h = 0; h < H; ++h)
            for (int s = 0; s < seq; ++s)
                for (int d = 0; d < V; ++d) {
                    const float expected = explicit_output(
                        h, start + s, d, start + s + 1,
                        q_nope, q_rope, k_b, v_b, all_latent, sink[h]);
                    const float actual = output_data[(h * seq + s) * V + d];
                    if (!close(actual, expected)) {
                        std::fprintf(stderr,
                            "pass=%d head=%d token=%d dim=%d got=%g want=%g\n",
                            pass, h, s, d, actual, expected);
                        return 3;
                    }
                }
    }
    // SLICE(q, nope) retains the original 256-wide row stride in the real
    // graph. Verify the kernel reads Tensor strides instead of assuming that
    // its logical [nope, seq, heads] shape is contiguous.
    meta->current_seq_len = 0;
    float qn_strided[H * 2 * (D + R)] = {};
    float qr_strided[H * 2 * (R + 1)] = {};
    for (int h = 0; h < H; ++h)
        for (int s = 0; s < 2; ++s) {
            std::copy_n(q_nope + (h * 3 + s) * D, D,
                        qn_strided + (h * 2 + s) * (D + R));
            std::copy_n(q_rope + (h * 3 + s) * R, R,
                        qr_strided + (h * 2 + s) * (R + 1));
        }
    Tensor qn_t = fp32(qn_strided, D, 2, H);
    qn_t.stride[1] = (D + R) * sizeof(float);
    qn_t.stride[2] = 2 * qn_t.stride[1];
    Tensor qr_t = fp32(qr_strided, R, 2, H);
    qr_t.stride[1] = (R + 1) * sizeof(float);
    qr_t.stride[2] = 2 * qr_t.stride[1];
    Tensor kv_t = fp32(kv, C, 2), kr_t = fp32(kr, R, 2);
    float mask_data[4] = {}, output_data[H * 2 * V] = {};
    Tensor mask_t = fp32(mask_data, 2, 2), output = fp32(output_data, V, 2, H);
    std::vector<const Tensor*> inputs = {
        &qn_t, &qr_t, &kv_t, &kr_t, &kb_t, &vb_t, &mask_t, &cache, &sink_t};
    if (!kernel_mla_absorbed_attention(node, inputs, &output, nullptr)) return 4;
    for (int h = 0; h < H; ++h)
        for (int s = 0; s < 2; ++s)
            for (int d = 0; d < V; ++d) {
                const float want = explicit_output(
                    h, s, d, s + 1, q_nope, q_rope,
                    k_b, v_b, all_latent, sink[h]);
                if (!close(output_data[(h * 2 + s) * V + d], want)) return 5;
            }
    // With three visible keys and width two, selection must change the
    // attention result. A future index is ignored even if the indexer names it.
    float qn_decode[H * D], qr_decode[H * R];
    for (int h = 0; h < H; ++h) {
        std::copy_n(q_nope + (h * 3 + 2) * D, D, qn_decode + h * D);
        std::copy_n(q_rope + (h * 3 + 2) * R, R, qr_decode + h * R);
    }
    Tensor qn_dec = fp32(qn_decode, D, 1, H);
    Tensor qr_dec = fp32(qr_decode, R, 1, H);
    Tensor kv_dec = fp32(kv + 2 * C, C), kr_dec = fp32(kr + 2 * R, R);
    float mask_dec[3] = {}, out_dec[H * V] = {};
    Tensor mask_dec_t = fp32(mask_dec, 3), out_dec_t = fp32(out_dec, V, 1, H);
    int32_t selected_data[2] = {2, 0};
    Tensor selected = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, 2, 1, 1, 1, selected_data);
    std::vector<const Tensor*> sparse_inputs = {
        &qn_dec, &qr_dec, &kv_dec, &kr_dec, &kb_t, &vb_t,
        &mask_dec_t, &cache, &sink_t, &selected};
    node.params.i32[7] = 9;
    for (int variant = 0; variant < 2; ++variant) {
        meta->current_seq_len = 2;
        if (variant == 1) selected_data[0] = 3; // future key: reject it
        if (!kernel_mla_absorbed_attention(
                node, sparse_inputs, &out_dec_t, nullptr)) return 6;
        const std::vector<int> positions = {
            selected_data[0], selected_data[1]};
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < V; ++d) {
                const float want = explicit_output(
                    h, 2, d, 3, q_nope, q_rope,
                    k_b, v_b, all_latent, sink[h], &positions);
                if (!close(out_dec[h * V + d], want)) return 7;
            }
    }
    // A longer cache exercises batched scores, masking, invalid sparse
    // indices, and a partial final batch against explicit attention.
    std::vector<float> long_latent(CAP * (C + R));
    auto* latent_data = static_cast<__fp16*>(cache_data(cache.data));
    for (int t = 0; t < CAP; ++t) {
        const int source = t == CAP - 1 ? 2 : t % 3;
        for (int c = 0; c < C + R; ++c) {
            const float value = all_latent[source * (C + R) + c];
            long_latent[t * (C + R) + c] = value;
            latent_data[t * (C + R) + c] = (__fp16)value;
        }
    }
    float long_mask[CAP] = {};
    long_mask[6] = -std::numeric_limits<float>::infinity();
    Tensor long_mask_t = fp32(long_mask, CAP);
    int32_t selected_long[9] = {6, 2, -1, CAP, 0, 8, 10, 3, 12};
    Tensor long_selection = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, 9, 1, 1, 1, selected_long);
    sparse_inputs[6] = &long_mask_t;
    sparse_inputs[9] = &long_selection;
    for (int selection_width : {0, 4, 5, 9}) {
        node.params.i32[7] = selection_width ? 9 : -1;
        long_selection.shape[0] = selection_width;
        meta->current_seq_len = CAP - 1;
        if (!kernel_mla_absorbed_attention(
                node, sparse_inputs, &out_dec_t, nullptr)) return 8;
        // Position 6 has zero softmax mass due to its -inf mask.
        std::vector<int> positions;
        for (int i = 0; i < (selection_width ? selection_width : CAP); ++i) {
            const int t = selection_width ? selected_long[i] : i;
            if (t != 6) positions.push_back(t);
        }
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < V; ++d) {
                const float want = explicit_output(
                    h, 2, d, CAP, q_nope, q_rope,
                    k_b, v_b, long_latent, sink[h], &positions);
                if (!close(out_dec[h * V + d], want)) return 9;
            }
    }
    // Decode covers both an unaligned K fallback and native Q8 projections,
    // with mixed K/V formats, both cache types, and a five-row V tail.
    if (!q8_weights_match_fp16<__fp16>() || !q8_weights_match_fp16<float>() ||
        !q8_weights_match_fp16<__fp16, 1, 4>() ||
        !q8_weights_match_fp16<float, 1, 4>() ||
        !q8_weights_match_fp16<__fp16, 1, 32>() ||
        !q8_weights_match_fp16<float, 1, 32>()) {
        std::fprintf(stderr, "MLA Q8_0 weights differ from exact FP16 weights\n");
        return 12;
    }
    std::puts("absorbed MLA dense/sparse prefill/decode matches explicit reference; Q8_0 matches FP16 within tolerance");
    return 0;
}
