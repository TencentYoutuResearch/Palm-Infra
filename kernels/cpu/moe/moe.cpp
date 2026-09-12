#include "kernels/cpu/moe/moe.h"

#include "core/bf16.h"
#include "backends/cpu/platform.h"
#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/moe/moe_routing.h"
#include "runtime/expert_provider.h"
#include "runtime/trace.h"
#include "runtime/threading.h"

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
    const bool has_embedded_bg128_scales =
        t.prec == Precision::INT4 && t.is_q4_g128_packed && t.q4_g128_data;
    if (!scales && !has_embedded_bg128_scales &&
        !(t.prec == Precision::MXFP4 && t.e8m0_scales)) {
        return 0.0f;
    }

    if (t.prec == Precision::INT8) {
        const int8_t* q = static_cast<const int8_t*>(t.data);
        return (float)q[idx] * scales[row * groups_per_row + group];
    }

    if (t.prec == Precision::INT4) {
        const uint8_t* q = static_cast<const uint8_t*>(t.data);
        uint8_t byte = 0;
        float scale = 0.0f;

        if (has_embedded_bg128_scales) {
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
            scale = scales[row * groups_per_row + group];
            int k_blocks = (K + 31) / 32;
            size_t qidx = (((size_t)(row / 8) * k_blocks + (size_t)(col / 32)) * 8 +
                           (size_t)(row & 7)) * 16 + (size_t)((col & 31) >> 1);
            byte = q[qidx];
        } else {
            scale = scales[row * groups_per_row + group];
            int row_stride = (K + 1) / 2;
            byte = q[(size_t)row * row_stride + (size_t)(col >> 1)];
        }

        return (float)unpack_int4_signed(byte, (col & 1) != 0) * scale;
    }

    if (t.prec == Precision::MXFP4 && t.e8m0_scales &&
        t.group_size == 32) {
        const uint8_t* q = static_cast<const uint8_t*>(t.data);
        const int row_stride = (K + 1) / 2;
        const uint8_t byte =
            q[static_cast<size_t>(row) * row_stride +
              static_cast<size_t>(col >> 1)];
        const uint8_t nibble = (col & 1) ? byte >> 4 : byte & 0x0f;
        return decode_mxfp4_e2m1(nibble) *
               decode_e8m0(
                   t.e8m0_scales[row * groups_per_row + group]);
    }

    return 0.0f;
}

static inline float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float silu_scalar(float x) {
    return x * sigmoid_scalar(x);
}

static void apply_swiglu(const float* gate,
                         const float* up,
                         float* output,
                         int rows,
                         int width,
                         int gate_stride,
                         int up_stride,
                         int output_stride,
                         float limit = 0.0f) {
    for (int row = 0; row < rows; ++row) {
        const float* gate_row = gate + static_cast<size_t>(row) * gate_stride;
        const float* up_row = up + static_cast<size_t>(row) * up_stride;
        float* output_row = output + static_cast<size_t>(row) * output_stride;
        for (int col = 0; col < width; ++col) {
            float gate_value = gate_row[col];
            float up_value = up_row[col];
            if (limit > 0.0f) {
                gate_value = std::min(gate_value, limit);
                up_value = std::clamp(up_value, -limit, limit);
            }
            output_row[col] = silu_scalar(gate_value) * up_value;
        }
    }
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
    view.e8m0_scales = nullptr;
    view.nvfp4_scales = nullptr;
    view.nvfp4_row_scales = nullptr;
    view.nvfp4_q8_pair_data = nullptr;
    view.num_groups = 0;
    view.is_interleaved = false;
    view.is_q4_repacked = false;
    view.is_q4_g32_packed = false;
    view.is_q4_g128_packed = false;
    view.q8_repack_data = nullptr;
    view.q4_repack_data = nullptr;
    view.q4_g32_data = nullptr;
    view.q4_g128_data = nullptr;
    view.prepared.weight = src.prepared.weight;
    view.prepared.row_offset =
        src.prepared.row_offset + static_cast<size_t>(row0);

    if (src.prec == Precision::FP32 || src.prec == Precision::FP16) {
        size_t elem = precision_size(src.prec);
        if (src.prec == Precision::FP16 && src.is_interleaved) {
            // CPU FP16 weights are packed in 8-row tiles at load time.  An
            // aggregate MoE tensor still uses that layout, so selecting one
            // expert must advance by whole tiles rather than pretending the
            // packed buffer is row-major.  Production expert widths are
            // multiples of eight; reject an unaligned slice so callers do not
            // silently consume scrambled weights.
            if ((row0 & 7) != 0)
                return false;
            view.data = static_cast<char*>(src.data) +
                        (size_t)(row0 / 8) * (size_t)K * 8 * elem;
            view.is_interleaved = true;
        } else {
            view.data = static_cast<char*>(src.data) +
                        (size_t)row0 * (size_t)K * elem;
        }
        return true;
    }

    int groups_per_row = src.groups_per_row > 0
        ? (int)src.groups_per_row
        : (src.group_size > 0 ? (K + (int)src.group_size - 1) / (int)src.group_size : 0);
    if (src.prec == Precision::MXFP4) {
        if (!src.e8m0_scales || src.group_size != 32 ||
            groups_per_row <= 0 || (K % 32) != 0) {
            return false;
        }
        view.data = static_cast<char*>(src.data) +
                    static_cast<size_t>(row0) * (K / 2);
        view.e8m0_scales =
            src.e8m0_scales +
            static_cast<size_t>(row0) * groups_per_row;
        view.group_size = 32;
        view.groups_per_row = static_cast<uint32_t>(groups_per_row);
        view.num_groups =
            static_cast<uint32_t>(
                static_cast<size_t>(rows) * groups_per_row);
        return true;
    }

    if (src.prec == Precision::NVFP4) {
        if (!src.nvfp4_scales || src.group_size != 16 ||
            groups_per_row <= 0 || (K % 16) != 0) {
            return false;
        }
        view.data = static_cast<char*>(src.data) +
                    static_cast<size_t>(row0) * (K / 2);
        view.nvfp4_scales = src.nvfp4_scales +
            static_cast<size_t>(row0) * groups_per_row;
        view.nvfp4_row_scales = src.nvfp4_row_scales + row0;
        if (src.nvfp4_q8_pair_data) {
            if ((row0 & 3) != 0) return false;
            view.nvfp4_q8_pair_data =
                src.nvfp4_q8_pair_data +
                static_cast<size_t>(row0 / 4) * groups_per_row * 32;
        }
        view.group_size = 16;
        view.groups_per_row = static_cast<uint32_t>(groups_per_row);
        view.num_groups = static_cast<uint32_t>(
            static_cast<size_t>(rows) * groups_per_row);
        return true;
    }

    const bool embeds_scales =
        (src.is_q4_g32_packed && src.q4_g32_data) ||
        (src.is_q4_g128_packed && src.q4_g128_data);
    if ((!src.scales && !embeds_scales) || src.group_size == 0 ||
        groups_per_row <= 0)
        return false;
    view.scales = src.scales
        ? src.scales + (size_t)row0 * (size_t)groups_per_row
        : nullptr;
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

        if (src.q4_g32_data) {
            if ((row0 & 7) != 0 || src.group_size != 32 ||
                (K % 32) != 0)
                return false;
            const char* p =
                static_cast<const char*>(src.q4_g32_data) +
                (size_t)(row0 / 8) * pack_b_q4dot_g32_bytes(8, K);
            view.q4_g32_data = p;
            if (src.is_q4_g32_packed) {
                view.data = const_cast<char*>(p);
                view.is_q4_g32_packed = true;
                return true;
            }
        }

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
                            bool has_shared_expert,
                            bool shared_expert_has_gate) {
    size_t required_inputs =
        has_shared_expert ? (shared_expert_has_gate ? 8 : 7) : 4;
    if (inputs.size() < required_inputs) {
        std::fprintf(stderr, "MOE: expected at least %zu inputs, got %zu\n",
                     required_inputs, inputs.size());
        return false;
    }
    for (size_t i = 0; i < required_inputs; i++) {
        // Routed expert aggregates may be data-less: their provider supplies
        // one selected expert pair at a time.
        bool is_provider_expert = (i == 2 || i == 3) && inputs[i] &&
                             inputs[i]->moe_ssd_source != nullptr;
        if (!inputs[i] || (!inputs[i]->data && !is_provider_expert)) {
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
                                       int intermediate_size,
                                       float swiglu_limit) {
    std::vector<float> gate_up(2 * intermediate_size);
    std::vector<float> inter(intermediate_size);
    int64_t gu_base = (int64_t)expert_id * (int64_t)(2 * intermediate_size) * hidden_size;
    for (int r = 0; r < 2 * intermediate_size; r++) {
        gate_up[r] = dot_row(experts_gate_up,
                             gu_base + (int64_t)r * hidden_size,
                             x, hidden_size);
    }
    apply_swiglu(gate_up.data(), gate_up.data() + intermediate_size,
                 inter.data(), 1, intermediate_size,
                 2 * intermediate_size, 2 * intermediate_size,
                 intermediate_size, swiglu_limit);
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

bool kernel_moe_shared_expert(const Tensor& hidden,
                              const Tensor& gate,
                              const Tensor& up,
                              const Tensor& down,
                              const Tensor* scale_weight,
                              Tensor& output,
                              ThreadPool* thread_pool,
                              int intermediate_size,
                              bool emulate_bf16,
                              float swiglu_limit) {
    if (hidden.prec != Precision::FP32 || output.prec != Precision::FP32 ||
        !hidden.data || !gate.data || !up.data || !down.data ||
        !output.data || intermediate_size <= 0 ||
        hidden.shape[1] != output.shape[1]) {
        return false;
    }
    const int hidden_size = static_cast<int>(hidden.shape[0]);
    const int seq_len = static_cast<int>(hidden.shape[1]);
    if (output.shape[0] != hidden_size)
        return false;

    std::vector<float> gate_values(
        static_cast<size_t>(seq_len) * intermediate_size);
    std::vector<float> up_values(
        static_cast<size_t>(seq_len) * intermediate_size);
    std::vector<float> activated(
        static_cast<size_t>(seq_len) * intermediate_size);
    Tensor gate_weight = make_weight_view_2d(
        gate, intermediate_size, hidden_size);
    Tensor up_weight = make_weight_view_2d(
        up, intermediate_size, hidden_size);
    Tensor gate_output = make_fp32_tensor(
        gate_values.data(), intermediate_size, seq_len);
    Tensor up_output = make_fp32_tensor(
        up_values.data(), intermediate_size, seq_len);

    bool batched_gate_up = false;
    if (seq_len == 1) {
        std::vector<Tensor> shared_inputs = {hidden, hidden};
        std::vector<Tensor> shared_weights = {gate_weight, up_weight};
        std::vector<Tensor> shared_outputs = {gate_output, up_output};
        batched_gate_up =
            kernel_matmul_int4_gemv_batch(
                shared_inputs, shared_weights, shared_outputs, thread_pool) ||
            kernel_matmul_int8_gemv_batch(
                shared_inputs, shared_weights, shared_outputs, thread_pool) ||
            kernel_matmul_nvfp4_gemv_batch(
                shared_inputs, shared_weights, shared_outputs, thread_pool);
    }
    if (!batched_gate_up) {
        kernel_matmul_fp32(hidden, gate_weight, gate_output, thread_pool);
        kernel_matmul_fp32(hidden, up_weight, up_output, thread_pool);
    }
    if (emulate_bf16) {
        mollm_round_to_bf16(gate_values.data(), gate_values.size());
        mollm_round_to_bf16(up_values.data(), up_values.size());
    }
    apply_swiglu(
        gate_values.data(), up_values.data(), activated.data(), seq_len,
        intermediate_size, intermediate_size, intermediate_size,
        intermediate_size, swiglu_limit);
    if (emulate_bf16)
        mollm_round_to_bf16(activated.data(), activated.size());

    Tensor activated_tensor = make_fp32_tensor(
        activated.data(), intermediate_size, seq_len);
    Tensor down_weight = make_weight_view_2d(
        down, hidden_size, intermediate_size);
    kernel_matmul_fp32(
        activated_tensor, down_weight, output, thread_pool);
    if (emulate_bf16) {
        mollm_round_to_bf16(
            output.ptr<float>(), static_cast<size_t>(output.nelements()));
    }

    if (scale_weight) {
        std::vector<float> scales(static_cast<size_t>(seq_len));
        Tensor scale_output = make_fp32_tensor(scales.data(), 1, seq_len);
        Tensor scale_matrix = make_weight_view_2d(
            *scale_weight, 1, hidden_size);
        kernel_matmul_fp32(
            hidden, scale_matrix, scale_output, thread_pool);
        if (emulate_bf16)
            mollm_round_to_bf16(scales.data(), scales.size());
        const int output_stride =
            static_cast<int>(output.stride[1] / sizeof(float));
        for (int token = 0; token < seq_len; ++token) {
            const float multiplier = sigmoid_scalar(scales[token]);
            float* row = output.ptr<float>() +
                static_cast<size_t>(token) * output_stride;
            for (int dim = 0; dim < hidden_size; ++dim)
                row[dim] *= multiplier;
        }
    }
    return true;
}

bool kernel_qwen3_moe(const std::vector<const Tensor*>& inputs,
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
                      float routed_scaling_factor,
                      bool shared_expert_has_gate,
                      int router_bias_input,
                      int token_ids_input,
                      int hash_table_input,
                      float swiglu_limit) {
    if (!validate_inputs(inputs, output, hidden_size, num_experts, top_k,
                         intermediate_size, shared_intermediate_size,
                         has_shared_expert, shared_expert_has_gate)) {
        return false;
    }

    const Tensor& hidden = *inputs[0];
    const Tensor& router = *inputs[1];
    const Tensor& experts_gate_up = *inputs[2];
    const Tensor& experts_down = *inputs[3];
    const bool use_bf16_activations =
        experts_gate_up.prec == Precision::MXFP4 ||
        experts_gate_up.prec == Precision::NVFP4;
    const auto* gate_up_source = experts_gate_up.moe_ssd_source;
    const auto* down_source = experts_down.moe_ssd_source;
    const bool use_provider = gate_up_source || down_source;
    if (use_provider && (!gate_up_source || !down_source || !gate_up_source->provider ||
                    gate_up_source->provider != down_source->provider)) {
        std::fprintf(stderr, "MOE: incomplete expert provider source pair\n");
        return false;
    }
    std::string trace_layer_args;
    if (mollm_trace::enabled()) {
        trace_layer_args = use_provider
            ? "{\"layer\":" + std::to_string(gate_up_source->layer) + "}"
            : "{}";
    }
    mollm_trace::ScopedEvent trace_moe("compute", "moe", trace_layer_args);
    const Tensor* router_bias =
        router_bias_input >= 0 &&
                static_cast<size_t>(router_bias_input) < inputs.size()
            ? inputs[router_bias_input]
            : nullptr;
    const Tensor* shared_gate = has_shared_expert ? inputs[4] : nullptr;
    const Tensor* shared_up = has_shared_expert ? inputs[5] : nullptr;
    const Tensor* shared_down = has_shared_expert ? inputs[6] : nullptr;
    const Tensor* shared_expert_gate =
        has_shared_expert && shared_expert_has_gate ? inputs[7] : nullptr;

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
    {
        mollm_trace::ScopedEvent trace_router("compute", "moe.router", trace_layer_args);
        kernel_matmul_fp32(hidden, router_b, router_out, thread_pool);
    }
    if (profile) moe_profile_add(MoeProfileStage::Router, stage_start);

    std::vector<int> top_idx;
    std::vector<float> top_w;
    const float* bias_data = router_bias && router_bias->data
        ? router_bias->ptr<float>() : nullptr;
    mollm::detail::MoeRoutingParams routing;
    routing.num_experts = num_experts;
    routing.top_k = top_k;
    routing.score_func = router_score_func;
    routing.normalize_topk = norm_topk_prob;
    routing.num_groups = n_group;
    routing.topk_groups = topk_group;
    routing.scaling_factor = routed_scaling_factor;
    const bool exact_hash_route =
        token_ids_input >= 0 && hash_table_input >= 0 &&
        static_cast<size_t>(token_ids_input) < inputs.size() &&
        static_cast<size_t>(hash_table_input) < inputs.size() &&
        inputs[token_ids_input] && inputs[hash_table_input] &&
        inputs[token_ids_input]->prec == Precision::INT32 &&
        inputs[hash_table_input]->prec == Precision::INT32;
    if (profile) stage_start = moe_profile_now();
    {
        mollm_trace::ScopedEvent trace_topk("compute", "moe.topk", trace_layer_args);
        bool selected = false;
        if (exact_hash_route) {
            selected = mollm::detail::select_moe_hash_routes(
                router_logits.data(), seq_len,
                inputs[token_ids_input]->ptr<int32_t>(),
                inputs[hash_table_input]->ptr<int32_t>(),
                static_cast<int>(inputs[hash_table_input]->shape[1]),
                routing, top_idx, top_w);
        } else {
            selected = mollm::detail::select_moe_routes(
                router_logits.data(), seq_len, bias_data, routing,
                top_idx, top_w);
        }
        if (!selected) {
            std::fprintf(stderr, "MOE: failed to select routed experts\n");
            return false;
        }
    }
    if (profile) moe_profile_add(MoeProfileStage::TopK, stage_start);

    std::vector<int> selected_experts;
    bool stream_expert_window = false;
    if (use_provider) {
        selected_experts.reserve((size_t)seq_len * (size_t)top_k);
        for (int t = 0; t < seq_len; t++) {
            for (int k = 0; k < top_k; k++) {
                selected_experts.push_back(top_idx[(size_t)t * top_k + k]);
            }
        }
        // Routed computation walks experts in ascending id order below. Queue
        // the same order so a cache too small for every routed pair holds the
        // next experts to be consumed rather than an arbitrary router order.
        std::sort(selected_experts.begin(), selected_experts.end());
        selected_experts.erase(std::unique(selected_experts.begin(), selected_experts.end()),
                               selected_experts.end());
        if (seq_len == 1 && !top_idx.empty()) {
            std::vector<int> ranked_experts(
                top_idx.begin(), top_idx.begin() + top_k);
            gate_up_source->provider->retain_for_next_forward(
                gate_up_source, down_source, ranked_experts,
                !exact_hash_route);
        }
        // Queue all cache misses before shared-expert work. The I/O workers
        // can now fill them while this CPU thread executes the independent
        // shared MLP below; acquire() only waits for a particular expert when
        // its routed matmul is about to run.
        if (!gate_up_source->provider->request_many(gate_up_source, down_source,
                                                  selected_experts)) {
            std::fprintf(stderr, "MOE: failed to request expert weights\n");
            return false;
        }
        stream_expert_window = gate_up_source->provider->resident_count(
            gate_up_source, down_source, selected_experts) < selected_experts.size();
    }

    // Compute the shared expert while routed-expert reads are in flight, but
    // defer its accumulation.  The reference accumulates routed experts first
    // and adds the shared expert last; preserving that FP32 summation order
    // avoids a small but systematic logits drift without sacrificing overlap.
    std::vector<float> shared_contribution;
    if (has_shared_expert) {
        mollm_trace::ScopedEvent trace_shared("compute", "moe.shared", trace_layer_args);
        shared_contribution.resize(
            (size_t)seq_len * (size_t)hidden_size);
        Tensor shared_output = make_fp32_tensor(
            shared_contribution.data(), hidden_size, seq_len);
        if (profile) stage_start = moe_profile_now();
        if (!kernel_moe_shared_expert(
                hidden, *shared_gate, *shared_up, *shared_down,
                shared_expert_has_gate ? shared_expert_gate : nullptr,
                shared_output, thread_pool, shared_intermediate_size,
                use_bf16_activations, swiglu_limit)) {
            return false;
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

    // Batch whichever decode experts are already materialized in RAM. Each
    // GEMV remains sharded across the full thread pool, while the batch pays
    // one dispatch barrier instead of one per expert. Experts still loading
    // are deliberately excluded: their reads continue in parallel with this
    // work and fall through to the streaming path below. Results are scattered
    // later in expert-id order, preserving the original accumulation order.
    std::vector<int> batched_slots;
    std::vector<float> batch_down_out;
    const bool can_batch_ready_decode =
        use_provider && seq_len == 1 && thread_pool &&
        thread_pool->num_threads() > 1 && selected_experts.size() > 1;
    if (can_batch_ready_decode)
        batched_slots.assign(num_experts, -1);

    auto process_ready_batch = [&](const std::vector<int>& experts) {
        const size_t batch = experts.size();
        const size_t output_base =
            batch_down_out.size() / static_cast<size_t>(hidden_size);
        std::vector<ExpertLease> leases(batch);
        std::vector<Tensor> gate_up_weights(batch);
        std::vector<Tensor> down_weights(batch);
        for (size_t i = 0; i < batch; ++i) {
            if (!gate_up_source->provider->borrow(
                    gate_up_source, down_source, experts[i], leases[i], false))
                return false;  // Readiness changed; fall back to sequential borrowing.
            gate_up_weights[i] = leases[i].gate_up;
            down_weights[i] = leases[i].down;
        }
        for (size_t i = 0; i < batch; ++i) {
            batched_slots[experts[i]] =
                static_cast<int>(output_base + i);
        }

        std::vector<float> batch_gate_up_out(
            batch * static_cast<size_t>(2 * intermediate_size));
        std::vector<float> batch_inter(
            batch * static_cast<size_t>(intermediate_size));
        batch_down_out.resize(
            (output_base + batch) * static_cast<size_t>(hidden_size));
        Tensor hidden_one = make_fp32_tensor(
            const_cast<float*>(x_data), hidden_size, 1);
        std::vector<Tensor> gate_inputs(batch, hidden_one);
        std::vector<Tensor> gate_outputs;
        gate_outputs.reserve(batch);
        for (size_t i = 0; i < batch; ++i) {
            gate_outputs.push_back(make_fp32_tensor(
                batch_gate_up_out.data() +
                    i * static_cast<size_t>(2 * intermediate_size),
                2 * intermediate_size, 1));
        }

        const std::string batch_trace_args =
            mollm_trace::enabled()
                ? "{\"layer\":" +
                      std::to_string(gate_up_source->layer) +
                      ",\"experts\":" + std::to_string(batch) + "}"
                : std::string();
        if (profile) stage_start = moe_profile_now();
        {
            mollm_trace::ScopedEvent trace_batch(
                "compute", "moe.routed_gate_up_batch", batch_trace_args);
            if (!kernel_matmul_int4_gemv_batch(
                    gate_inputs, gate_up_weights, gate_outputs, thread_pool) &&
                !kernel_matmul_mxfp4_gemv_batch(
                    gate_inputs, gate_up_weights, gate_outputs, thread_pool) &&
                !kernel_matmul_nvfp4_gemv_batch(
                    gate_inputs, gate_up_weights, gate_outputs, thread_pool)) {
                for (size_t i = 0; i < batch; ++i) {
                    kernel_matmul_fp32(
                        gate_inputs[i], gate_up_weights[i], gate_outputs[i],
                        thread_pool);
                }
            }
            for (size_t i = 0; i < batch; ++i) {
                float* gate_up = batch_gate_up_out.data() +
                    i * static_cast<size_t>(2 * intermediate_size);
                if (use_bf16_activations) {
                    mollm_round_to_bf16(
                        gate_up,
                        static_cast<size_t>(2 * intermediate_size));
                }
                apply_swiglu(
                    gate_up, gate_up + intermediate_size,
                    batch_inter.data() +
                        i * static_cast<size_t>(intermediate_size),
                    1, intermediate_size, 2 * intermediate_size,
                    2 * intermediate_size, intermediate_size,
                    swiglu_limit);
                if (use_bf16_activations) {
                    const int route = offsets[experts[i]];
                    const float route_weight = route_weights[route];
                    float* intermediate =
                        batch_inter.data() +
                        i * static_cast<size_t>(intermediate_size);
                    for (int dim = 0; dim < intermediate_size; ++dim)
                        intermediate[dim] *= route_weight;
                    mollm_round_to_bf16(
                        intermediate,
                        static_cast<size_t>(intermediate_size));
                }
            }
        }
        if (profile)
            moe_profile_add(MoeProfileStage::RoutedGateUp, stage_start);

        std::vector<Tensor> down_inputs;
        std::vector<Tensor> down_outputs;
        down_inputs.reserve(batch);
        down_outputs.reserve(batch);
        for (size_t i = 0; i < batch; ++i) {
            down_inputs.push_back(make_fp32_tensor(
                batch_inter.data() +
                    i * static_cast<size_t>(intermediate_size),
                intermediate_size, 1));
            down_outputs.push_back(make_fp32_tensor(
                batch_down_out.data() +
                    (output_base + i) * static_cast<size_t>(hidden_size),
                hidden_size, 1));
        }
        if (profile) stage_start = moe_profile_now();
        {
            mollm_trace::ScopedEvent trace_batch(
                "compute", "moe.routed_down_batch", batch_trace_args);
            if (!kernel_matmul_int4_gemv_batch(
                    down_inputs, down_weights, down_outputs, thread_pool) &&
                !kernel_matmul_mxfp4_gemv_batch(
                    down_inputs, down_weights, down_outputs, thread_pool) &&
                !kernel_matmul_nvfp4_gemv_batch(
                    down_inputs, down_weights, down_outputs, thread_pool)) {
                for (size_t i = 0; i < batch; ++i) {
                    kernel_matmul_fp32(
                        down_inputs[i], down_weights[i], down_outputs[i],
                        thread_pool);
                }
            }
        }
        if (profile)
            moe_profile_add(MoeProfileStage::RoutedDown, stage_start);
        if (use_bf16_activations) {
            mollm_round_to_bf16(
                batch_down_out.data() +
                    output_base * static_cast<size_t>(hidden_size),
                batch * static_cast<size_t>(hidden_size));
        }
        return true;
    };

    if (can_batch_ready_decode) {
        // The first batch overlaps its compute with reads that were not ready
        // after the shared expert. Re-scan after each non-trivial wave so those
        // newly completed experts also share two thread-pool dispatches instead
        // of falling back to two dispatches per expert.
        for (;;) {
            std::vector<int> ready;
            for (int expert : selected_experts) {
                if (counts[expert] == 1 && batched_slots[expert] < 0 &&
                    gate_up_source->provider->is_ready(
                        gate_up_source, down_source, expert)) {
                    ready.push_back(expert);
                }
            }
            if (ready.size() < 2)
                break;
            if (!process_ready_batch(ready))
                break;
        }
    }

    auto advance_stream_window = [&](int expert) {
        if (!use_provider || !stream_expert_window) return;
        gate_up_source->provider->evict(
            gate_up_source, down_source, expert);
        auto next = std::upper_bound(
            selected_experts.begin(), selected_experts.end(), expert);
        for (; next != selected_experts.end(); ++next) {
            if (!gate_up_source->provider->contains(
                    gate_up_source, down_source, *next)) {
                // Submit one exact replacement. Passing every remaining route
                // could evict a nearer ready expert while reserving the tail.
                gate_up_source->provider->request_many(
                    gate_up_source, down_source, {*next});
                break;
            }
        }
    };

    for (int e = 0; e < num_experts; e++) {
        int count = counts[e];
        if (count == 0) continue;
        int begin = offsets[e];
        std::string trace_expert_args;
        if (mollm_trace::enabled()) {
            trace_expert_args = "{\"expert\":" + std::to_string(e) +
                                (use_provider ? ",\"layer\":" +
                                               std::to_string(gate_up_source->layer)
                                         : std::string()) + "}";
        }
        const int batched_slot =
            batched_slots.empty() ? -1 : batched_slots[e];
        if (batched_slot >= 0) {
            if (profile) stage_start = moe_profile_now();
            const int route = offsets[e];
            const float* src = batch_down_out.data() +
                (size_t)batched_slot * hidden_size;
            float* dst =
                out_data + (int64_t)route_tokens[route] * ldo;
            const float output_scale =
                use_bf16_activations ? 1.0f : route_weights[route];
            for (int d = 0; d < hidden_size; ++d)
                dst[d] += output_scale * src[d];
            if (profile)
                moe_profile_add(MoeProfileStage::RoutedScatter, stage_start);

            advance_stream_window(e);
            continue;
        }
        if (profile) stage_start = moe_profile_now();
        {
            mollm_trace::ScopedEvent trace_gather("compute", "moe.routed_gather",
                                                   trace_expert_args);
            for (int i = 0; i < count; i++) {
                int t = route_tokens[begin + i];
                std::memcpy(expert_x.data() + (size_t)i * hidden_size,
                            x_data + (int64_t)t * ldx,
                            (size_t)hidden_size * sizeof(float));
            }
        }
        if (profile) moe_profile_add(MoeProfileStage::RoutedGather, stage_start);

        ExpertLease lease;
        Tensor gate_up_b;
        Tensor down_b;
        bool has_gate_up_view = false;
        bool has_down_view = false;
        if (use_provider) {
            if (!gate_up_source->provider->borrow(gate_up_source, down_source, e,
                                                 lease)) {
                std::fprintf(stderr, "MOE: failed to borrow expert %d\n", e);
                return false;
            }
            gate_up_b = lease.gate_up;
            down_b = lease.down;
            has_gate_up_view = true;
            has_down_view = true;
        } else {
            has_gate_up_view = make_weight_rows_view(
                experts_gate_up, (int64_t)e * (2 * intermediate_size),
                2 * intermediate_size, hidden_size, gate_up_b);
            has_down_view = make_weight_rows_view(
                experts_down, (int64_t)e * hidden_size,
                hidden_size, intermediate_size, down_b);
        }

        if (!has_gate_up_view || !has_down_view) {
            if (use_bf16_activations) {
                std::fprintf(
                    stderr,
                    "MOE: MXFP4 expert does not support the scalar fallback\n");
                return false;
            }
            for (int i = 0; i < count; i++) {
                int t = route_tokens[begin + i];
                routed_ffn_scalar_fallback(experts_gate_up, experts_down,
                                           x_data + (int64_t)t * ldx,
                                           out_data + (int64_t)t * ldo,
                                           e, route_weights[begin + i],
                                           hidden_size, intermediate_size,
                                           swiglu_limit);
            }
            continue;
        }

        {
            // Keep the compute interval disjoint from acquire(): in a trace this
            // makes I/O wait, input gather, and expert matmuls visually additive
            // rather than a misleading pair of overlapping compute bars.
            mollm_trace::ScopedEvent trace_compute("compute", "moe.routed_compute",
                                                    trace_expert_args);
            Tensor expert_x_t = make_fp32_tensor(expert_x.data(), hidden_size, count);
            Tensor gate_up_t = make_fp32_tensor(gate_up_out.data(), 2 * intermediate_size, count);
            if (profile) stage_start = moe_profile_now();
            kernel_matmul_fp32(expert_x_t, gate_up_b, gate_up_t, thread_pool);
            if (use_bf16_activations) {
                mollm_round_to_bf16(
                    gate_up_out.data(),
                    static_cast<size_t>(count) *
                        static_cast<size_t>(2 * intermediate_size));
            }
            apply_swiglu(gate_up_out.data(),
                         gate_up_out.data() + intermediate_size,
                         routed_inter.data(), count, intermediate_size,
                         2 * intermediate_size, 2 * intermediate_size,
                         intermediate_size, swiglu_limit);
            if (use_bf16_activations) {
                for (int i = 0; i < count; ++i) {
                    float* intermediate =
                        routed_inter.data() +
                        static_cast<size_t>(i) * intermediate_size;
                    const float route_weight =
                        route_weights[begin + i];
                    for (int dim = 0; dim < intermediate_size; ++dim)
                        intermediate[dim] *= route_weight;
                }
                mollm_round_to_bf16(
                    routed_inter.data(),
                    static_cast<size_t>(count) *
                        static_cast<size_t>(intermediate_size));
            }
            if (profile) moe_profile_add(MoeProfileStage::RoutedGateUp, stage_start);

            Tensor inter_t = make_fp32_tensor(routed_inter.data(), intermediate_size, count);
            Tensor down_t = make_fp32_tensor(down_out.data(), hidden_size, count);
            if (profile) stage_start = moe_profile_now();
            kernel_matmul_fp32(inter_t, down_b, down_t, thread_pool);
            if (use_bf16_activations) {
                mollm_round_to_bf16(
                    down_out.data(),
                    static_cast<size_t>(count) *
                        static_cast<size_t>(hidden_size));
            }
            if (profile) moe_profile_add(MoeProfileStage::RoutedDown, stage_start);

            if (profile) stage_start = moe_profile_now();
            for (int i = 0; i < count; i++) {
                int t = route_tokens[begin + i];
                const float* src = down_out.data() + (size_t)i * hidden_size;
                float* dst = out_data + (int64_t)t * ldo;
                const float output_scale =
                    use_bf16_activations
                        ? 1.0f
                        : route_weights[begin + i];
                for (int d = 0; d < hidden_size; d++)
                    dst[d] += output_scale * src[d];
            }
            if (profile) moe_profile_add(MoeProfileStage::RoutedScatter, stage_start);
        }

        lease.reset();
        // Weight views are no longer used. Advance a bounded cache immediately
        // so its replacement read overlaps the next ready expert.
        advance_stream_window(e);
    }

    if (has_shared_expert) {
        for (int t = 0; t < seq_len; ++t) {
            const float* src =
                shared_contribution.data() + (size_t)t * hidden_size;
            float* dst = out_data + (int64_t)t * ldo;
            for (int d = 0; d < hidden_size; ++d)
                dst[d] += src[d];
        }
    }
    return true;
}
