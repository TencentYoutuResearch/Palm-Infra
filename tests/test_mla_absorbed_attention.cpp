#include "kernels/cpu/models/mla_absorbed_attention.h"
#include "core/cache_layout.h"
#include "graph/graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr int H = 2, C = 3, D = 2, R = 2, V = 2, CAP = 4;

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

} // namespace

int main() {
    __fp16 k_b[H * C * D] = {
        0.5f, -0.25f, 0.125f, 0.75f, -0.5f, 0.25f,
        -0.125f, 0.5f, 0.375f, -0.25f, 0.75f, -0.5f};
    __fp16 v_b[H * V * C] = {
        0.5f, 0.25f, -0.25f, -0.5f, 0.125f, 0.75f,
        0.25f, -0.375f, 0.5f, 0.625f, 0.125f, -0.25f};
    float q_nope[H * 3 * D] = {
        1, -2, 0.5f, 1.5f, -1, 0.25f,
        -0.5f, 2, 1, -1, 0.75f, 1.25f};
    float q_rope[H * 3 * R] = {
        0.25f, -0.5f, 1, 0.25f, -0.75f, 0.5f,
        0.5f, 0.25f, -1, 0.5f, 0.25f, -0.75f};
    float kv[3 * C] = {1, 0.5f, -1, -0.5f, 1.5f, 0.25f, 0.75f, -0.25f, 1};
    float kr[3 * R] = {0.25f, 1, -0.5f, 0.75f, 1, -0.25f};
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
    std::puts("absorbed MLA dense/sparse prefill/decode matches explicit reference");
    return 0;
}
