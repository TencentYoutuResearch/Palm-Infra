#include "kernels/moe.h"
#include "kernels/matmul.h"
#include "kernels/threading.h"

#include <algorithm>
#include <cassert>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

enum class MoeProfileStage : int {
    Router = 0,
    TopK,
    RoutedGather,
    RoutedGateUp,
    RoutedDown,
    RoutedScatter,
    SharedGateUp,
    SharedDown,
    Count,
};

static constexpr int MOE_PROFILE_STAGE_COUNT = (int)MoeProfileStage::Count;
static std::atomic<uint64_t> g_moe_profile_ns[MOE_PROFILE_STAGE_COUNT];
static std::atomic<uint64_t> g_moe_profile_calls[MOE_PROFILE_STAGE_COUNT];

static const char* moe_profile_stage_name(int stage) {
    switch ((MoeProfileStage)stage) {
    case MoeProfileStage::Router: return "router_matmul";
    case MoeProfileStage::TopK: return "topk_selector";
    case MoeProfileStage::RoutedGather: return "routed_gather";
    case MoeProfileStage::RoutedGateUp: return "routed_gate_up";
    case MoeProfileStage::RoutedDown: return "routed_down";
    case MoeProfileStage::RoutedScatter: return "routed_scatter";
    case MoeProfileStage::SharedGateUp: return "shared_gate_up";
    case MoeProfileStage::SharedDown: return "shared_down";
    case MoeProfileStage::Count: break;
    }
    return "unknown";
}

static bool moe_profile_enabled() {
    static bool enabled = [] {
        const char* value = std::getenv("MOLLM_MOE_PROFILE");
        return value && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static inline std::chrono::steady_clock::time_point moe_profile_now() {
    return std::chrono::steady_clock::now();
}

static inline void moe_profile_add(MoeProfileStage stage,
                                   std::chrono::steady_clock::time_point start) {
    if (!moe_profile_enabled()) return;
    auto end = std::chrono::steady_clock::now();
    uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    int idx = (int)stage;
    g_moe_profile_ns[idx].fetch_add(ns, std::memory_order_relaxed);
    g_moe_profile_calls[idx].fetch_add(1, std::memory_order_relaxed);
}

static inline int8_t unpack_int4_signed(uint8_t byte, bool high) {
    uint8_t q = high ? (byte >> 4) : (byte & 0x0f);
    return (q & 0x08) ? (int8_t)(q | 0xf0) : (int8_t)q;
}

static inline float load_weight_at(const Tensor& t, int64_t row, int col, int K) {
    int64_t idx = row * (int64_t)K + col;
    if (t.prec == Precision::FP32) {
        return static_cast<const float*>(t.data)[idx];
    }
    if (t.prec == Precision::FP16) {
        return static_cast<float>(static_cast<const __fp16*>(t.data)[idx]);
    }
    int group_size = t.group_size > 0 ? (int)t.group_size : K;
    int groups_per_row = t.groups_per_row > 0 ? (int)t.groups_per_row : 1;
    int group = col / group_size;
    const float* scales = t.scales;
    if (!scales) return 0.0f;

    if (t.prec == Precision::INT8) {
        const int8_t* q = static_cast<const int8_t*>(t.data);
        return (float)q[idx] * scales[row * groups_per_row + group];
    }

    if (t.prec == Precision::INT4) {
        const uint8_t* q = static_cast<const uint8_t*>(t.data);
        uint8_t byte = 0;
        float scale = scales[row * groups_per_row + group];

        if (t.is_q4_g128_packed && t.q4_g128_data) {
            constexpr int BG128_BLOCK_BYTES = 544;
            const uint8_t* bg = static_cast<const uint8_t*>(t.q4_g128_data);
            int lane = (int)(row & 7);
            int g128 = col / 128;
            int qgi = (col & 127) / 32;
            const uint8_t* block =
                bg + (((size_t)(row / 8) * groups_per_row + (size_t)g128) *
                      BG128_BLOCK_BYTES);
            std::memcpy(&scale, block + (size_t)lane * sizeof(float), sizeof(float));
            byte = block[32 + ((qgi * 8 + lane) * 16) + ((col & 31) >> 1)];
        } else if (t.is_q4_repacked) {
            int k_blocks = (K + 31) / 32;
            size_t qidx = (((size_t)(row / 8) * k_blocks + (size_t)(col / 32)) * 8 +
                           (size_t)(row & 7)) * 16 + (size_t)((col & 31) >> 1);
            byte = q[qidx];
        } else {
            int row_stride = (K + 1) / 2;
            byte = q[(size_t)row * row_stride + (size_t)(col >> 1)];
        }

        return (float)unpack_int4_signed(byte, (col & 1) != 0) * scale;
    }

    return 0.0f;
}

static inline float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float silu_scalar(float x) {
    return x * sigmoid_scalar(x);
}

static Tensor make_fp32_tensor(float* data, int cols, int rows) {
    return Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                          cols, rows, 1, 1, data);
}

static Tensor make_weight_view_2d(const Tensor& src, int rows, int K) {
    Tensor view = src;
    view.shape[0] = rows;
    view.shape[1] = K;
    view.shape[2] = 1;
    view.shape[3] = 1;
    return view;
}

static bool make_weight_rows_view(const Tensor& src, int64_t row0, int rows, int K,
                                  Tensor& view) {
    if (!src.data || row0 < 0 || rows <= 0 || K <= 0) return false;

    view = src;
    view.mem_type = MemoryType::EXTERNAL;
    view.shape[0] = rows;
    view.shape[1] = K;
    view.shape[2] = 1;
    view.shape[3] = 1;
    view.stride[0] = view.element_size();
    view.stride[1] = view.stride[0] * (size_t)K;
    view.stride[2] = view.stride[1] * (size_t)rows;
    view.stride[3] = view.stride[2];

    view.scales = nullptr;
    view.num_groups = 0;
    view.is_interleaved = false;
    view.is_q4_repacked = false;
    view.is_q4_g128_packed = false;
    view.q8_repack_data = nullptr;
    view.q4_repack_data = nullptr;
    view.q4_g128_data = nullptr;

    if (src.prec == Precision::FP32 || src.prec == Precision::FP16) {
        size_t elem = precision_size(src.prec);
        view.data = static_cast<char*>(src.data) + (size_t)row0 * (size_t)K * elem;
        return true;
    }

    int groups_per_row = src.groups_per_row > 0
        ? (int)src.groups_per_row
        : (src.group_size > 0 ? (K + (int)src.group_size - 1) / (int)src.group_size : 0);
    if (!src.scales || src.group_size == 0 || groups_per_row <= 0) return false;
    view.scales = src.scales + (size_t)row0 * (size_t)groups_per_row;
    view.group_size = src.group_size;
    view.groups_per_row = (uint32_t)groups_per_row;
    view.num_groups = (uint32_t)((size_t)rows * (size_t)groups_per_row);

    if (src.prec == Precision::INT8) {
        view.data = static_cast<char*>(src.data) + (size_t)row0 * (size_t)K;
        if (src.is_interleaved) {
            if ((row0 & 7) != 0) return false;
            view.data = static_cast<char*>(src.data) + (size_t)(row0 / 8) * (size_t)K * 8;
            view.is_interleaved = true;
        }
        if (src.q8_repack_data) {
            if ((row0 & 7) != 0) return false;
            int k_blocks = (K + 31) / 32;
            view.q8_repack_data = static_cast<const char*>(src.q8_repack_data) +
                (size_t)(row0 / 8) * (size_t)k_blocks * 8 * 32;
        }
        return true;
    }

    if (src.prec == Precision::INT4) {
        int row_stride = (K + 1) / 2;
        view.data = static_cast<char*>(src.data) + (size_t)row0 * (size_t)row_stride;

        if (src.is_q4_g128_packed && src.q4_g128_data) {
            if ((row0 & 7) != 0 || src.group_size != 128 || (K % 128) != 0) return false;
            constexpr size_t BG128_BLOCK_BYTES = 544;
            const char* p = static_cast<const char*>(src.q4_g128_data) +
                (size_t)(row0 / 8) * (size_t)groups_per_row * BG128_BLOCK_BYTES;
            view.data = const_cast<char*>(p);
            view.q4_g128_data = p;
            view.is_q4_g128_packed = true;
            return true;
        }

        if (src.q4_repack_data) {
            if ((row0 & 7) != 0) return false;
            int k_blocks = (K + 31) / 32;
            const char* p = static_cast<const char*>(src.q4_repack_data) +
                (size_t)(row0 / 8) * (size_t)k_blocks * 8 * 16;
            view.q4_repack_data = p;
            if (src.is_q4_repacked) {
                view.data = const_cast<char*>(p);
                view.is_q4_repacked = true;
            }
        }
        return true;
    }

    return false;
}

static float dot_row(const Tensor& w, int64_t row_offset, const float* x, int K) {
    float sum = 0.0f;
    int64_t row = row_offset / K;
    for (int k = 0; k < K; k++) sum += load_weight_at(w, row, k, K) * x[k];
    return sum;
}

static bool validate_inputs(const std::vector<const Tensor*>& inputs,
                            const Tensor& output,
                            int hidden_size,
                            int num_experts,
                            int top_k,
                            int intermediate_size,
                            int shared_intermediate_size,
                            bool has_shared_expert) {
    size_t required_inputs = has_shared_expert ? 8 : 4;
    if (inputs.size() < required_inputs) {
        std::fprintf(stderr, "MOE: expected at least %zu inputs, got %zu\n",
                     required_inputs, inputs.size());
        return false;
    }
    for (size_t i = 0; i < required_inputs; i++) {
        if (!inputs[i] || !inputs[i]->data) {
            std::fprintf(stderr, "MOE: missing input %zu\n", i);
            return false;
        }
    }
    if (hidden_size <= 0 || num_experts <= 0 || top_k <= 0 ||
        intermediate_size <= 0 ||
        (has_shared_expert && shared_intermediate_size <= 0) ||
        top_k > num_experts) {
        std::fprintf(stderr,
                     "MOE: bad params hidden=%d experts=%d top_k=%d intermediate=%d shared=%d\n",
                     hidden_size, num_experts, top_k, intermediate_size,
                     shared_intermediate_size);
        return false;
    }
    if (output.shape[0] != hidden_size) {
        std::fprintf(stderr, "MOE: output hidden mismatch got=%lld expected=%d\n",
                     (long long)output.shape[0], hidden_size);
        return false;
    }
    return true;
}

static bool routed_ffn_scalar_fallback(const Tensor& experts_gate_up,
                                       const Tensor& experts_down,
                                       const float* x,
                                       float* y,
                                       int expert_id,
                                       float route_w,
                                       int hidden_size,
                                       int intermediate_size) {
    std::vector<float> gate_up(2 * intermediate_size);
    std::vector<float> inter(intermediate_size);
    int64_t gu_base = (int64_t)expert_id * (int64_t)(2 * intermediate_size) * hidden_size;
    for (int r = 0; r < 2 * intermediate_size; r++) {
        gate_up[r] = dot_row(experts_gate_up,
                             gu_base + (int64_t)r * hidden_size,
                             x, hidden_size);
    }
    for (int j = 0; j < intermediate_size; j++) {
        inter[j] = silu_scalar(gate_up[j]) * gate_up[intermediate_size + j];
    }
    for (int d = 0; d < hidden_size; d++) {
        float sum = 0.0f;
        int64_t row = (int64_t)expert_id * hidden_size + d;
        for (int j = 0; j < intermediate_size; j++) {
            sum += load_weight_at(experts_down, row, j, intermediate_size) * inter[j];
        }
        y[d] += route_w * sum;
    }
    return true;
}

} // namespace

extern "C" int mollm_moe_profile_enabled() {
    return moe_profile_enabled() ? 1 : 0;
}

extern "C" void mollm_reset_moe_profile() {
    for (int i = 0; i < MOE_PROFILE_STAGE_COUNT; i++) {
        g_moe_profile_ns[i].store(0, std::memory_order_relaxed);
        g_moe_profile_calls[i].store(0, std::memory_order_relaxed);
    }
}

extern "C" void mollm_print_moe_profile(const char* title) {
    if (!moe_profile_enabled()) return;

    struct Row {
        int stage;
        uint64_t calls;
        uint64_t ns;
    };
    std::vector<Row> rows;
    uint64_t total_ns = 0;
    for (int i = 0; i < MOE_PROFILE_STAGE_COUNT; i++) {
        uint64_t calls = g_moe_profile_calls[i].load(std::memory_order_relaxed);
        uint64_t ns = g_moe_profile_ns[i].load(std::memory_order_relaxed);
        if (calls == 0) continue;
        rows.push_back({i, calls, ns});
        total_ns += ns;
    }
    if (rows.empty()) return;
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return a.ns > b.ns;
    });

    std::printf("\n[%s]\n", title && title[0] ? title : "moe_profile");
    std::printf("  %-22s %10s %10s %10s %7s\n", "stage", "calls", "total_ms", "avg_ms", "pct");
    std::printf("  %-22s %10s %10s %10s %7s\n", "---", "---", "---", "---", "---");
    for (const auto& row : rows) {
        double total_ms = row.ns / 1e6;
        double avg_ms = row.calls ? total_ms / row.calls : 0.0;
        double pct = total_ns ? 100.0 * (double)row.ns / (double)total_ns : 0.0;
        std::printf("  %-22s %10llu %10.2f %10.3f %6.1f%%\n",
                    moe_profile_stage_name(row.stage),
                    (unsigned long long)row.calls,
                    total_ms,
                    avg_ms,
                    pct);
    }
}

void kernel_qwen3_moe(const std::vector<const Tensor*>& inputs,
                      Tensor& output,
                      ThreadPool* thread_pool,
                      int hidden_size,
                      int num_experts,
                      int top_k,
                      int intermediate_size,
                      int shared_intermediate_size,
                      int router_score_func,
                      bool norm_topk_prob,
                      bool has_shared_expert,
                      int n_group,
                      int topk_group,
                      float routed_scaling_factor) {
    if (!validate_inputs(inputs, output, hidden_size, num_experts, top_k,
                         intermediate_size, shared_intermediate_size,
                         has_shared_expert)) {
        return;
    }

    const Tensor& hidden = *inputs[0];
    const Tensor& router = *inputs[1];
    const Tensor& experts_gate_up = *inputs[2];
    const Tensor& experts_down = *inputs[3];
    const Tensor* router_bias = inputs.size() > 8 ? inputs[8] : nullptr;
    const Tensor* shared_gate = has_shared_expert ? inputs[4] : nullptr;
    const Tensor* shared_up = has_shared_expert ? inputs[5] : nullptr;
    const Tensor* shared_down = has_shared_expert ? inputs[6] : nullptr;
    const Tensor* shared_expert_gate = has_shared_expert ? inputs[7] : nullptr;

    assert(hidden.prec == Precision::FP32);
    const int seq_len = (int)hidden.shape[1];
    const int ldx = (int)(hidden.stride[1] / sizeof(float));
    const int ldo = (int)(output.stride[1] / sizeof(float));
    const float* x_data = hidden.ptr<float>();
    float* out_data = output.ptr<float>();

    for (int t = 0; t < seq_len; t++) {
        std::fill(out_data + (int64_t)t * ldo,
                  out_data + (int64_t)t * ldo + hidden_size, 0.0f);
    }

    bool profile = moe_profile_enabled();
    std::chrono::steady_clock::time_point stage_start;

    std::vector<float> router_logits((size_t)seq_len * (size_t)num_experts);
    Tensor router_b = make_weight_view_2d(router, num_experts, hidden_size);
    Tensor router_out = make_fp32_tensor(router_logits.data(), num_experts, seq_len);
    if (profile) stage_start = moe_profile_now();
    kernel_matmul_fp32(hidden, router_b, router_out, thread_pool);
    if (profile) moe_profile_add(MoeProfileStage::Router, stage_start);

    std::vector<int> top_idx((size_t)seq_len * (size_t)top_k);
    std::vector<float> top_w((size_t)seq_len * (size_t)top_k);
    std::vector<float> router_scores(router_score_func == 1
        ? (size_t)seq_len * (size_t)num_experts : 0);
    std::vector<float> choice_scores(router_score_func == 1
        ? (size_t)num_experts : 0);
    const float* bias_data = router_bias && router_bias->data
        ? router_bias->ptr<float>() : nullptr;
    if (n_group <= 0) n_group = 1;
    if (topk_group <= 0 || topk_group > n_group) topk_group = n_group;
    int experts_per_group = num_experts / n_group;
    if (profile) stage_start = moe_profile_now();
    for (int t = 0; t < seq_len; t++) {
        const float* logits = router_logits.data() + (size_t)t * num_experts;
        int* idx = top_idx.data() + (size_t)t * top_k;
        float* weights = top_w.data() + (size_t)t * top_k;

        for (int k = 0; k < top_k; k++) {
            idx[k] = 0;
            weights[k] = -std::numeric_limits<float>::infinity();
        }

        const float* weight_source = logits;
        if (router_score_func == 1) {
            float* scores = router_scores.data() + (size_t)t * num_experts;
            for (int e = 0; e < num_experts; e++) {
                scores[e] = sigmoid_scalar(logits[e]);
                choice_scores[e] = scores[e] + (bias_data ? bias_data[e] : 0.0f);
            }
            if (n_group > 1 && experts_per_group > 0 && topk_group < n_group) {
                std::vector<float> group_scores(n_group, -std::numeric_limits<float>::infinity());
                std::vector<int> group_idx(topk_group, 0);
                std::vector<float> group_top(topk_group, -std::numeric_limits<float>::infinity());
                for (int g = 0; g < n_group; g++) {
                    float best0 = -std::numeric_limits<float>::infinity();
                    float best1 = -std::numeric_limits<float>::infinity();
                    int begin = g * experts_per_group;
                    int end = begin + experts_per_group;
                    for (int e = begin; e < end; e++) {
                        float v = choice_scores[e];
                        if (v > best0) {
                            best1 = best0;
                            best0 = v;
                        } else if (v > best1) {
                            best1 = v;
                        }
                    }
                    group_scores[g] = best0 + best1;
                    for (int k = 0; k < topk_group; k++) {
                        if (group_scores[g] > group_top[k]) {
                            for (int s = topk_group - 1; s > k; s--) {
                                group_top[s] = group_top[s - 1];
                                group_idx[s] = group_idx[s - 1];
                            }
                            group_top[k] = group_scores[g];
                            group_idx[k] = g;
                            break;
                        }
                    }
                }
                std::vector<unsigned char> keep_group(n_group, 0);
                for (int k = 0; k < topk_group; k++) keep_group[group_idx[k]] = 1;
                for (int e = 0; e < num_experts; e++) {
                    int g = e / experts_per_group;
                    if (!keep_group[g]) choice_scores[e] = -std::numeric_limits<float>::infinity();
                }
            }
            weight_source = scores;
        }

        const float* select_source = router_score_func == 1 ? choice_scores.data() : logits;
        for (int e = 0; e < num_experts; e++) {
            float v = select_source[e];
            for (int k = 0; k < top_k; k++) {
                if (v > weights[k]) {
                    for (int s = top_k - 1; s > k; s--) {
                        weights[s] = weights[s - 1];
                        idx[s] = idx[s - 1];
                    }
                    weights[k] = v;
                    idx[k] = e;
                    break;
                }
            }
        }

        if (router_score_func == 1) {
            float sum_top = 0.0f;
            for (int k = 0; k < top_k; k++) {
                weights[k] = weight_source[idx[k]];
                sum_top += weights[k];
            }
            if (norm_topk_prob) {
                float inv_top = sum_top > 0.0f ? 1.0f / sum_top : 0.0f;
                for (int k = 0; k < top_k; k++) weights[k] *= inv_top;
            }
            for (int k = 0; k < top_k; k++) weights[k] *= routed_scaling_factor;
        } else {
            float max_top = weights[0];
            float sum_top = 0.0f;
            for (int k = 0; k < top_k; k++) {
                weights[k] = std::exp(weights[k] - max_top);
                sum_top += weights[k];
            }
            float inv_top = sum_top > 0.0f ? 1.0f / sum_top : 0.0f;
            for (int k = 0; k < top_k; k++) weights[k] *= inv_top;
        }
    }
    if (profile) moe_profile_add(MoeProfileStage::TopK, stage_start);

    // Shared expert: shared_down(silu(shared_gate(x)) * shared_up(x))
    // multiplied by sigmoid(shared_expert_gate(x)).
    if (has_shared_expert) {
        std::vector<float> shared_gate_out((size_t)seq_len * (size_t)shared_intermediate_size);
        std::vector<float> shared_up_out((size_t)seq_len * (size_t)shared_intermediate_size);
        std::vector<float> shared_inter((size_t)seq_len * (size_t)shared_intermediate_size);
        std::vector<float> shared_down_out((size_t)seq_len * (size_t)hidden_size);
        std::vector<float> shared_scale((size_t)seq_len);

        Tensor shared_gate_b = make_weight_view_2d(*shared_gate, shared_intermediate_size, hidden_size);
        Tensor shared_up_b = make_weight_view_2d(*shared_up, shared_intermediate_size, hidden_size);
        Tensor shared_gate_t = make_fp32_tensor(shared_gate_out.data(), shared_intermediate_size, seq_len);
        Tensor shared_up_t = make_fp32_tensor(shared_up_out.data(), shared_intermediate_size, seq_len);

        if (profile) stage_start = moe_profile_now();
        kernel_matmul_fp32(hidden, shared_gate_b, shared_gate_t, thread_pool);
        kernel_matmul_fp32(hidden, shared_up_b, shared_up_t, thread_pool);
        for (int t = 0; t < seq_len; t++) {
            float* dst = shared_inter.data() + (size_t)t * shared_intermediate_size;
            const float* gate = shared_gate_out.data() + (size_t)t * shared_intermediate_size;
            const float* up = shared_up_out.data() + (size_t)t * shared_intermediate_size;
            for (int j = 0; j < shared_intermediate_size; j++) {
                dst[j] = silu_scalar(gate[j]) * up[j];
            }
        }
        if (profile) moe_profile_add(MoeProfileStage::SharedGateUp, stage_start);

        Tensor shared_inter_t = make_fp32_tensor(shared_inter.data(), shared_intermediate_size, seq_len);
        Tensor shared_down_b = make_weight_view_2d(*shared_down, hidden_size, shared_intermediate_size);
        Tensor shared_down_t = make_fp32_tensor(shared_down_out.data(), hidden_size, seq_len);
        std::vector<float> shared_gate_scale_out((size_t)seq_len);
        Tensor shared_scale_b = make_weight_view_2d(*shared_expert_gate, 1, hidden_size);
        Tensor shared_scale_t = make_fp32_tensor(shared_gate_scale_out.data(), 1, seq_len);

        if (profile) stage_start = moe_profile_now();
        kernel_matmul_fp32(shared_inter_t, shared_down_b, shared_down_t, thread_pool);
        kernel_matmul_fp32(hidden, shared_scale_b, shared_scale_t, thread_pool);
        for (int t = 0; t < seq_len; t++) {
            shared_scale[t] = sigmoid_scalar(shared_gate_scale_out[t]);
            const float* src = shared_down_out.data() + (size_t)t * hidden_size;
            float* dst = out_data + (int64_t)t * ldo;
            float scale = shared_scale[t];
            for (int d = 0; d < hidden_size; d++) dst[d] += scale * src[d];
        }
        if (profile) moe_profile_add(MoeProfileStage::SharedDown, stage_start);
    }

    std::vector<int> counts(num_experts, 0);
    for (int t = 0; t < seq_len; t++) {
        for (int k = 0; k < top_k; k++) counts[top_idx[(size_t)t * top_k + k]]++;
    }
    std::vector<int> offsets(num_experts + 1, 0);
    for (int e = 0; e < num_experts; e++) offsets[e + 1] = offsets[e] + counts[e];
    std::vector<int> cursor = offsets;
    int total_routes = offsets[num_experts];
    std::vector<int> route_tokens(total_routes);
    std::vector<float> route_weights(total_routes);
    for (int t = 0; t < seq_len; t++) {
        for (int k = 0; k < top_k; k++) {
            int e = top_idx[(size_t)t * top_k + k];
            int pos = cursor[e]++;
            route_tokens[pos] = t;
            route_weights[pos] = top_w[(size_t)t * top_k + k];
        }
    }

    int max_count = 0;
    for (int c : counts) max_count = std::max(max_count, c);
    std::vector<float> expert_x((size_t)max_count * (size_t)hidden_size);
    std::vector<float> gate_up_out((size_t)max_count * (size_t)(2 * intermediate_size));
    std::vector<float> routed_inter((size_t)max_count * (size_t)intermediate_size);
    std::vector<float> down_out((size_t)max_count * (size_t)hidden_size);

    for (int e = 0; e < num_experts; e++) {
        int count = counts[e];
        if (count == 0) continue;
        int begin = offsets[e];

        if (profile) stage_start = moe_profile_now();
        for (int i = 0; i < count; i++) {
            int t = route_tokens[begin + i];
            std::memcpy(expert_x.data() + (size_t)i * hidden_size,
                        x_data + (int64_t)t * ldx,
                        (size_t)hidden_size * sizeof(float));
        }
        if (profile) moe_profile_add(MoeProfileStage::RoutedGather, stage_start);

        Tensor gate_up_b;
        Tensor down_b;
        bool has_gate_up_view = make_weight_rows_view(
            experts_gate_up, (int64_t)e * (2 * intermediate_size),
            2 * intermediate_size, hidden_size, gate_up_b);
        bool has_down_view = make_weight_rows_view(
            experts_down, (int64_t)e * hidden_size,
            hidden_size, intermediate_size, down_b);

        if (!has_gate_up_view || !has_down_view) {
            for (int i = 0; i < count; i++) {
                int t = route_tokens[begin + i];
                routed_ffn_scalar_fallback(experts_gate_up, experts_down,
                                           x_data + (int64_t)t * ldx,
                                           out_data + (int64_t)t * ldo,
                                           e, route_weights[begin + i],
                                           hidden_size, intermediate_size);
            }
            continue;
        }

        Tensor expert_x_t = make_fp32_tensor(expert_x.data(), hidden_size, count);
        Tensor gate_up_t = make_fp32_tensor(gate_up_out.data(), 2 * intermediate_size, count);
        if (profile) stage_start = moe_profile_now();
        kernel_matmul_fp32(expert_x_t, gate_up_b, gate_up_t, thread_pool);
        for (int i = 0; i < count; i++) {
            const float* gu = gate_up_out.data() + (size_t)i * (2 * intermediate_size);
            float* dst = routed_inter.data() + (size_t)i * intermediate_size;
            for (int j = 0; j < intermediate_size; j++) {
                dst[j] = silu_scalar(gu[j]) * gu[intermediate_size + j];
            }
        }
        if (profile) moe_profile_add(MoeProfileStage::RoutedGateUp, stage_start);

        Tensor inter_t = make_fp32_tensor(routed_inter.data(), intermediate_size, count);
        Tensor down_t = make_fp32_tensor(down_out.data(), hidden_size, count);
        if (profile) stage_start = moe_profile_now();
        kernel_matmul_fp32(inter_t, down_b, down_t, thread_pool);
        if (profile) moe_profile_add(MoeProfileStage::RoutedDown, stage_start);

        if (profile) stage_start = moe_profile_now();
        for (int i = 0; i < count; i++) {
            int t = route_tokens[begin + i];
            float weight = route_weights[begin + i];
            const float* src = down_out.data() + (size_t)i * hidden_size;
            float* dst = out_data + (int64_t)t * ldo;
            for (int d = 0; d < hidden_size; d++) dst[d] += weight * src[d];
        }
        if (profile) moe_profile_add(MoeProfileStage::RoutedScatter, stage_start);
    }
}
