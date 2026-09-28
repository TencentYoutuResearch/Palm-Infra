#include "graph/execute.h"
#include "backends/cpu/backend.h"
#include "backends/cpu/moe_ops.h"
#include "core/fp16.h"
#include "kernels/cpu/moe/moe.h"
#include "storage/ssd_expert_cache/cache.h"
#include "core/tensor.h"
#include "runtime/threading.h"
#include "runtime/expert_provider.h"
#include <thread>
#include <chrono>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { std::printf("  PASS: %s\n", msg); } \
} while (0)

static float fill_value(int i) {
    return 0.17f * std::sin(0.37f * (float)i) + 0.11f * std::cos(0.13f * (float)i);
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float silu(float x) {
    return x * sigmoid(x);
}

static float dot(const std::vector<float>& w, int row_offset,
                 const float* x, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += w[row_offset + k] * x[k];
    return sum;
}

static void ref_moe(const std::vector<float>& hidden,
                    const std::vector<float>& router,
                    const std::vector<float>& experts_gate_up,
                    const std::vector<float>& experts_down,
                    const std::vector<float>& shared_gate,
                    const std::vector<float>& shared_up,
                    const std::vector<float>& shared_down,
                    const std::vector<float>& shared_expert_gate,
                    std::vector<float>& out,
                    int H, int M, int E, int top_k, int I, int SI) {
    std::vector<float> logits(E), probs(E), gate_up(2 * I), inter(I);
    std::vector<float> s_gate(SI), s_up(SI), s_inter(SI);
    std::vector<int> top_idx(top_k);
    std::vector<float> top_w(top_k);
    std::fill(out.begin(), out.end(), 0.0f);

    for (int m = 0; m < M; m++) {
        const float* x = hidden.data() + m * H;
        float* y = out.data() + m * H;

        float max_logit = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < E; e++) {
            logits[e] = dot(router, e * H, x, H);
            max_logit = std::max(max_logit, logits[e]);
        }
        float sum_exp = 0.0f;
        for (int e = 0; e < E; e++) {
            probs[e] = std::exp(logits[e] - max_logit);
            sum_exp += probs[e];
        }
        for (int e = 0; e < E; e++) probs[e] /= sum_exp;

        float top_sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            int best_e = 0;
            float best = -1.0f;
            for (int e = 0; e < E; e++) {
                if (probs[e] > best) {
                    best = probs[e];
                    best_e = e;
                }
            }
            top_idx[k] = best_e;
            top_w[k] = best;
            top_sum += best;
            probs[best_e] = -1.0f;
        }
        for (int k = 0; k < top_k; k++) top_w[k] /= top_sum;

        for (int kk = 0; kk < top_k; kk++) {
            int e = top_idx[kk];
            int gu_base = e * (2 * I * H);
            int down_base = e * (H * I);
            for (int r = 0; r < 2 * I; r++) {
                gate_up[r] = dot(experts_gate_up, gu_base + r * H, x, H);
            }
            for (int j = 0; j < I; j++) {
                inter[j] = silu(gate_up[j]) * gate_up[I + j];
            }
            for (int d = 0; d < H; d++) {
                float sum = 0.0f;
                int row = down_base + d * I;
                for (int j = 0; j < I; j++) sum += experts_down[row + j] * inter[j];
                y[d] += top_w[kk] * sum;
            }
        }

        if (SI == 0) continue;
        for (int j = 0; j < SI; j++) {
            s_gate[j] = dot(shared_gate, j * H, x, H);
            s_up[j] = dot(shared_up, j * H, x, H);
            s_inter[j] = silu(s_gate[j]) * s_up[j];
        }
        float scale = sigmoid(dot(shared_expert_gate, 0, x, H));
        for (int d = 0; d < H; d++) {
            float sum = 0.0f;
            int row = d * SI;
            for (int j = 0; j < SI; j++) sum += shared_down[row + j] * s_inter[j];
            y[d] += scale * sum;
        }
    }
}

static bool close_enough(const std::vector<float>& got,
                         const std::vector<float>& ref,
                         float tol) {
    for (size_t i = 0; i < got.size(); i++) {
        float diff = std::fabs(got[i] - ref[i]);
        if (diff > tol) {
            std::fprintf(stderr, "  mismatch at %zu: got %.8f expected %.8f diff %.8f\n",
                         i, got[i], ref[i], diff);
            return false;
        }
    }
    return true;
}

static std::vector<__fp16> to_fp16(const std::vector<float>& src) {
    std::vector<__fp16> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) dst[i] = (__fp16)src[i];
    return dst;
}

static std::vector<float> fp16_to_float(const std::vector<__fp16>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) dst[i] = (float)src[i];
    return dst;
}

static std::vector<__fp16> interleave_fp16_rows(
        const std::vector<__fp16>& src, int rows, int cols) {
    const int padded_rows = ((rows + 7) / 8) * 8;
    std::vector<__fp16> dst((size_t)padded_rows * cols, (__fp16)0.0f);
    for (int row_base = 0; row_base < padded_rows; row_base += 8) {
        for (int col = 0; col < cols; ++col) {
            for (int lane = 0; lane < 8; ++lane) {
                const int row = row_base + lane;
                if (row < rows) {
                    dst[(size_t)row_base * cols + (size_t)col * 8 + lane] =
                        src[(size_t)row * cols + col];
                }
            }
        }
    }
    return dst;
}

static bool gguf_router_prefill_matches_single_token() {
    constexpr int H = 256, M = 33, E = 8, I = 256, TOP_K = 2;
    constexpr int Q8_ROW_BYTES = (H / 32) * 34;
    std::vector<float> hidden(H * M), router_fp32(E * H);
    for (size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = fill_value(static_cast<int>(i) + 101);
    for (size_t i = 0; i < router_fp32.size(); ++i)
        router_fp32[i] = fill_value(static_cast<int>(i) + 701);
    std::vector<__fp16> router_packed = interleave_fp16_rows(
        to_fp16(router_fp32), E, H);
    auto make_q8 = [](size_t rows) {
        std::vector<uint8_t> bytes(rows * Q8_ROW_BYTES);
        for (size_t block = 0; block < rows * (H / 32); ++block) {
            uint8_t* dst = bytes.data() + block * 34;
            dst[0] = 0;
            dst[1] = 0x2c;  // FP16 scale 1/16.
            for (int lane = 0; lane < 32; ++lane)
                dst[2 + lane] = static_cast<uint8_t>(
                    static_cast<int>((block * 5 + lane * 3) % 7) - 3);
        }
        return bytes;
    };
    std::vector<uint8_t> gate_up = make_q8(E * 2 * I);
    std::vector<uint8_t> down = make_q8(E * H);
    std::vector<float> all_output(H * M), single_output(H);
    Tensor all_hidden = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, H, M, 1, 1, hidden.data());
    Tensor last_hidden = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, H, 1, 1, 1,
        hidden.data() + (M - 1) * H);
    Tensor router = Tensor::create(
        Precision::FP16, MemoryType::EXTERNAL, E, H, 1, 1,
        router_packed.data());
    router.is_interleaved = true;
    Tensor gate_up_weight = Tensor::create(
        Precision::GGUF_Q8_0, MemoryType::EXTERNAL, E * 2 * I, H, 1, 1,
        gate_up.data());
    Tensor down_weight = Tensor::create(
        Precision::GGUF_Q8_0, MemoryType::EXTERNAL, E * H, I, 1, 1,
        down.data());
    Tensor all_result = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, H, M, 1, 1,
        all_output.data());
    Tensor single_result = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL, H, 1, 1, 1,
        single_output.data());
    std::vector<const Tensor*> inputs = {
        &all_hidden, &router, &gate_up_weight, &down_weight};
    ThreadPool pool(4);
    if (!kernel_qwen3_moe(inputs, all_result, &pool,
                          H, E, TOP_K, I, 0, 0, true, false))
        return false;
    inputs[0] = &last_hidden;
    if (!kernel_qwen3_moe(inputs, single_result, &pool,
                          H, E, TOP_K, I, 0, 0, true, false))
        return false;
    // Q8 singleton GEMV and multi-token kernels use different reduction
    // orders. Validate both against independently decoded expert weights and
    // the scalar MoE reference, using the existing FP32 MoE tolerance.
    auto decode_q8 = [](const std::vector<uint8_t>& encoded) {
        std::vector<float> decoded(encoded.size() / 34 * 32);
        for (size_t block = 0; block < encoded.size() / 34; ++block) {
            __fp16 scale;
            std::memcpy(&scale, encoded.data() + block * 34, sizeof(scale));
            for (int k = 0; k < 32; ++k) {
                int8_t value;
                std::memcpy(&value, encoded.data() + block * 34 + 2 + k, 1);
                decoded[block * 32 + k] = float(scale) * float(value);
            }
        }
        return decoded;
    };
    std::vector<float> reference(H * M);
    ref_moe(hidden, fp16_to_float(to_fp16(router_fp32)),
            decode_q8(gate_up), decode_q8(down), {}, {}, {}, {}, reference,
            H, M, E, TOP_K, I, 0);
    float max_error = 0.f;
    for (size_t i = 0; i < all_output.size(); ++i) {
        if (!std::isfinite(all_output[i]) || !std::isfinite(reference[i])) return false;
        max_error = std::max(max_error, std::abs(all_output[i] - reference[i]));
    }
    for (int i = 0; i < H; ++i) {
        if (!std::isfinite(single_output[i])) return false;
        max_error = std::max(max_error,
            std::abs(single_output[i] - reference[(M - 1) * H + i]));
        max_error = std::max(max_error,
            std::abs(single_output[i] - all_output[(M - 1) * H + i]));
    }
    std::printf("  Q8 MoE prefill/single reference maximum error: %.9g\n", max_error);
    return max_error <= 1e-5f;
}

// A controlled provider exercises the wait path without SSD timing races.
struct DelayedProvider : ExpertProvider {
    ExpertSource gate, down;
    Tensor gate_weight, down_weight;
    bool fail = false, ready = false;
    int probes = 0, waits = 0, releases = 0, live = 0;

    DelayedProvider(Tensor gate_tensor, Tensor down_tensor)
        : gate_weight(gate_tensor), down_weight(down_tensor) {
        gate.provider = down.provider = this;
    }
    bool borrow(const ExpertSource*, const ExpertSource*, int,
                ExpertLease& out, bool wait) override {
        out.reset();
        if (!wait) {
            ++probes;
            if (!ready) return false;
        } else {
            ++waits;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (fail) return false;
        }
        ++live;
        out = ExpertLease(*this, this, gate_weight, down_weight);
        return true;
    }
    bool request_many(const ExpertSource*, const ExpertSource*,
                      const std::vector<int>&) override { return true; }
    bool retain_for_next_forward(const ExpertSource*, const ExpertSource*,
                                 const std::vector<int>&, bool) override {
        return true;
    }
    size_t resident_count(const ExpertSource*, const ExpertSource*,
                          const std::vector<int>& experts) const override {
        return experts.size();
    }
    bool contains(const ExpertSource*, const ExpertSource*, int) const override {
        return true;
    }
    bool is_ready(const ExpertSource*, const ExpertSource*, int) const override {
        return ready;
    }
    bool evict(const ExpertSource*, const ExpertSource*, int) override {
        return false;
    }
    void release_lease(void*) noexcept override {
        --live;
        ++releases;
    }
};

static int test_delayed_provider_wait() {
    constexpr int H = 128, I = 128, E = 1;
    auto tensor = [](int a, int b, int c, void* data) {
        return Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                              a, b, c, 1, data);
    };
    std::vector<float> x(H), router(H), gate(2 * I * H), down(H * I);
    std::vector<float> reference(H), output(H);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i)) * 0.1f;
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = std::cos(float(i)) * 0.02f;
    for (size_t i = 0; i < down.size(); ++i) down[i] = std::sin(float(i)) * 0.03f;
    auto hidden = tensor(H, 1, 1, x.data());
    auto router_tensor = tensor(E, H, 1, router.data());
    auto gate_tensor = tensor(E, 2 * I, H, gate.data());
    auto down_tensor = tensor(E, H, I, down.data());
    auto output_tensor = tensor(H, 1, 1, output.data());
    auto reference_tensor = tensor(H, 1, 1, reference.data());
    std::vector<const Tensor*> inputs{
        &hidden, &router_tensor, &gate_tensor, &down_tensor};
    ThreadPool pool(4);
    pool.resize(4, 1024);  // Combine idle yields with provider park/resume.
    if (!kernel_qwen3_moe(inputs, reference_tensor, &pool,
                          H, E, 1, I, 0, 0, true, false)) return 1;

    DelayedProvider provider(tensor(2 * I, H, 1, gate.data()),
                             tensor(H, I, 1, down.data()));
    gate_tensor.data = down_tensor.data = nullptr;
    gate_tensor.moe_ssd_source = &provider.gate;
    down_tensor.moe_ssd_source = &provider.down;
    // Missing, already ready, and failed I/O; every call must leave the pool usable.
    for (int mode = 0; mode < 3; ++mode) {
        provider.ready = mode == 1;
        provider.fail = mode == 2;
        const int old_waits = provider.waits;
        for (int iteration = 0; iteration < 20; ++iteration) {
            const bool ok = kernel_qwen3_moe(inputs, output_tensor, &pool,
                                             H, E, 1, I, 0, 0, true, false);
            if (ok == provider.fail || provider.live) return 2;
            if (ok) {
                for (int i = 0; i < H; ++i) {
                    if (std::abs(output[i] - reference[i]) > 1e-7f) return 3;
                }
            }
            std::vector<int> seen(256);
            pool.parallel_for(0, 256, 1, [&](int, int begin, int end) {
                for (int i = begin; i < end; ++i) seen[i] = i + 1;
            });
            for (int i = 0; i < 256; ++i) {
                if (seen[i] != i + 1) return 4;
            }
        }
        const int expected_waits = mode == 1 ? 0 : 20;
        if (provider.waits - old_waits != expected_waits) return 5;
    }
    if (provider.probes != 60 || provider.releases != 40) return 6;
    pool.park();  // Destruction must also wake and join parked workers.
    return 0;
}

int main() {
    CHECK(test_delayed_provider_wait() == 0,
          "delayed/ready/failed expert borrow preserves output and worker recovery");
    CHECK(gguf_router_prefill_matches_single_token(),
          "GGUF MoE router agrees between 33-token prefill and one token");
    const int H = 8;
    const int M = 3;
    const int E = 5;
    const int KTOP = 2;
    const int I = 8;
    const int SI = 4;

    std::vector<float> hidden(H * M);
    std::vector<float> router(E * H);
    std::vector<float> experts_gate_up(E * 2 * I * H);
    std::vector<float> experts_down(E * H * I);
    std::vector<float> shared_gate(SI * H);
    std::vector<float> shared_up(SI * H);
    std::vector<float> shared_down(H * SI);
    std::vector<float> shared_expert_gate(H);

    for (size_t i = 0; i < hidden.size(); i++) hidden[i] = fill_value((int)i + 1);
    for (size_t i = 0; i < router.size(); i++) router[i] = fill_value((int)i + 101);
    for (size_t i = 0; i < experts_gate_up.size(); i++) experts_gate_up[i] = fill_value((int)i + 301);
    for (size_t i = 0; i < experts_down.size(); i++) experts_down[i] = fill_value((int)i + 701);
    for (size_t i = 0; i < shared_gate.size(); i++) shared_gate[i] = fill_value((int)i + 1101);
    for (size_t i = 0; i < shared_up.size(); i++) shared_up[i] = fill_value((int)i + 1301);
    for (size_t i = 0; i < shared_down.size(); i++) shared_down[i] = fill_value((int)i + 1701);
    for (size_t i = 0; i < shared_expert_gate.size(); i++) shared_expert_gate[i] = fill_value((int)i + 1901);

    std::vector<float> out(H * M, 0.0f);
    std::vector<float> ref(H * M, 0.0f);

    Tensor hidden_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, H, M, 1, 1, hidden.data());
    Tensor router_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, E, H, 1, 1, router.data());
    Tensor gu_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, E, 2 * I, H, 1, experts_gate_up.data());
    Tensor down_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, E, H, I, 1, experts_down.data());
    Tensor sg_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, SI, H, 1, 1, shared_gate.data());
    Tensor su_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, SI, H, 1, 1, shared_up.data());
    Tensor sd_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, H, SI, 1, 1, shared_down.data());
    Tensor seg_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 1, H, 1, 1, shared_expert_gate.data());
    Tensor out_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, H, M, 1, 1, out.data());

    std::vector<const Tensor*> inputs = {
        &hidden_t, &router_t, &gu_t, &down_t, &sg_t, &su_t, &sd_t, &seg_t,
    };
    CHECK(kernel_qwen3_moe(inputs, out_t, nullptr, H, E, KTOP, I, SI),
          "kernel_qwen3_moe reports successful execution");
    ref_moe(hidden, router, experts_gate_up, experts_down,
            shared_gate, shared_up, shared_down, shared_expert_gate,
            ref, H, M, E, KTOP, I, SI);
    CHECK(close_enough(out, ref, 1e-5f), "kernel_qwen3_moe FP32 matches reference");

    std::vector<__fp16> router_h = to_fp16(router);
    std::vector<__fp16> experts_gate_up_h = to_fp16(experts_gate_up);
    std::vector<__fp16> experts_down_h = to_fp16(experts_down);
    std::vector<__fp16> shared_gate_h = to_fp16(shared_gate);
    std::vector<__fp16> shared_up_h = to_fp16(shared_up);
    std::vector<__fp16> shared_down_h = to_fp16(shared_down);
    std::vector<__fp16> shared_expert_gate_h = to_fp16(shared_expert_gate);
    std::vector<float> out_h(H * M, 0.0f);
    std::vector<float> ref_h(H * M, 0.0f);

    Tensor router_ht = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, E, H, 1, 1, router_h.data());
    Tensor gu_ht = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, E, 2 * I, H, 1, experts_gate_up_h.data());
    Tensor down_ht = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, E, H, I, 1, experts_down_h.data());
    Tensor sg_ht = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, SI, H, 1, 1, shared_gate_h.data());
    Tensor su_ht = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, SI, H, 1, 1, shared_up_h.data());
    Tensor sd_ht = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, H, SI, 1, 1, shared_down_h.data());
    Tensor seg_ht = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, 1, H, 1, 1, shared_expert_gate_h.data());
    Tensor out_ht = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, H, M, 1, 1, out_h.data());
    std::vector<const Tensor*> inputs_h = {
        &hidden_t, &router_ht, &gu_ht, &down_ht, &sg_ht, &su_ht, &sd_ht, &seg_ht,
    };
    kernel_qwen3_moe(inputs_h, out_ht, nullptr, H, E, KTOP, I, SI);
    ref_moe(hidden,
            fp16_to_float(router_h),
            fp16_to_float(experts_gate_up_h),
            fp16_to_float(experts_down_h),
            fp16_to_float(shared_gate_h),
            fp16_to_float(shared_up_h),
            fp16_to_float(shared_down_h),
            fp16_to_float(shared_expert_gate_h),
            ref_h, H, M, E, KTOP, I, SI);
    CHECK(close_enough(out_h, ref_h, 2e-2f), "kernel_qwen3_moe FP16 weights match rounded reference");

    // Model loading replaces row-major FP16 linear weights with an 8-row
    // interleaved working layout. Aggregate expert tensors must preserve that
    // layout when kernel_qwen3_moe selects one expert's row range.
    std::vector<__fp16> experts_gate_up_i = interleave_fp16_rows(
        experts_gate_up_h, E * 2 * I, H);
    std::vector<__fp16> experts_down_i = interleave_fp16_rows(
        experts_down_h, E * H, I);
    Tensor gu_it = gu_ht;
    gu_it.data = experts_gate_up_i.data();
    gu_it.is_interleaved = true;
    Tensor down_it = down_ht;
    down_it.data = experts_down_i.data();
    down_it.is_interleaved = true;
    std::vector<float> out_i(H * M, 0.0f);
    Tensor out_it = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                   H, M, 1, 1, out_i.data());
    std::vector<const Tensor*> inputs_i = {
        &hidden_t, &router_ht, &gu_it, &down_it,
        &sg_ht, &su_ht, &sd_ht, &seg_ht,
    };
    CHECK(kernel_qwen3_moe(inputs_i, out_it, nullptr,
                           H, E, KTOP, I, SI),
          "kernel_qwen3_moe accepts interleaved FP16 expert weights");
    CHECK(close_enough(out_i, ref_h, 2e-2f),
          "interleaved FP16 MoE expert views match rounded reference");

    // The SSD route must be numerically identical to the ordinary aggregate
    // expert path. Write the same FP16 expert slices to a temporary package
    // stand-in, then leave the aggregate Tensor pointers intentionally null.
    const char* ssd_path = "/tmp/mollm_test_moe_ssd_e2e.bin";
    {
        std::ofstream file(ssd_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(experts_gate_up_h.data()),
                   (std::streamsize)(experts_gate_up_h.size() * sizeof(__fp16)));
        file.write(reinterpret_cast<const char*>(experts_down_h.data()),
                   (std::streamsize)(experts_down_h.size() * sizeof(__fp16)));
    }
    MoeSsdTensorSpec ssd_gu;
    ssd_gu.weight_ref = "./gate";
    ssd_gu.layer = 0;
    ssd_gu.num_experts = E;
    ssd_gu.rows = 2 * I;
    ssd_gu.cols = H;
    ssd_gu.precision = Precision::FP16;
    ssd_gu.data_offset = 0;
    ssd_gu.data_bytes = (uint64_t)(2 * I * H * sizeof(__fp16));
    MoeSsdTensorSpec ssd_down;
    ssd_down.weight_ref = "./down";
    ssd_down.layer = 0;
    ssd_down.num_experts = E;
    ssd_down.rows = H;
    ssd_down.cols = I;
    ssd_down.precision = Precision::FP16;
    ssd_down.data_offset = (uint64_t)(experts_gate_up_h.size() * sizeof(__fp16));
    ssd_down.data_bytes = (uint64_t)(H * I * sizeof(__fp16));
    MoeSsdCache ssd_cache;
    CHECK(ssd_cache.open(ssd_path, (size_t)(ssd_gu.data_bytes + ssd_down.data_bytes)),
          "open MoE SSD numerical cache");
    CHECK(ssd_cache.add_source(ssd_gu) && ssd_cache.add_source(ssd_down),
          "register MoE SSD numerical sources");
    Tensor gu_ssd = gu_ht;
    Tensor down_ssd = down_ht;
    gu_ssd.data = nullptr;
    down_ssd.data = nullptr;
    gu_ssd.moe_ssd_source = ssd_cache.find_source("./gate");
    down_ssd.moe_ssd_source = ssd_cache.find_source("./down");
    std::vector<float> out_ssd(H * M, 0.0f);
    Tensor out_ssd_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                      H, M, 1, 1, out_ssd.data());
    std::vector<const Tensor*> inputs_ssd = {
        &hidden_t, &router_ht, &gu_ssd, &down_ssd, &sg_ht, &su_ht, &sd_ht, &seg_ht,
    };
    kernel_qwen3_moe(inputs_ssd, out_ssd_t, nullptr, H, E, KTOP, I, SI);
    CHECK(close_enough(out_ssd, ref_h, 2e-2f), "MoE SSD expert paging matches FP16 reference");
    std::remove(ssd_path);

    // Exercise graph dispatch/allocation path.
    Graph g;
    g.nodes.resize(9);
    for (uint32_t i = 0; i < 8; i++) {
        g.nodes[i].id = i;
        g.nodes[i].op_type = OpType::INPUT;
        g.nodes[i].out_prec = Precision::FP32;
    }
    g.nodes[0].out_shape[0] = H; g.nodes[0].out_shape[1] = M;
    g.nodes[1].out_shape[0] = E; g.nodes[1].out_shape[1] = H;
    g.nodes[2].out_shape[0] = E; g.nodes[2].out_shape[1] = 2 * I; g.nodes[2].out_shape[2] = H;
    g.nodes[3].out_shape[0] = E; g.nodes[3].out_shape[1] = H; g.nodes[3].out_shape[2] = I;
    g.nodes[4].out_shape[0] = SI; g.nodes[4].out_shape[1] = H;
    g.nodes[5].out_shape[0] = SI; g.nodes[5].out_shape[1] = H;
    g.nodes[6].out_shape[0] = H; g.nodes[6].out_shape[1] = SI;
    g.nodes[7].out_shape[0] = 1; g.nodes[7].out_shape[1] = H;
    g.nodes[8].id = 8;
    g.nodes[8].op_type = OpType::MOE;
    g.nodes[8].inputs = {0, 1, 2, 3, 4, 5, 6, 7};
    g.nodes[8].out_shape[0] = H;
    g.nodes[8].out_shape[1] = M;
    g.nodes[8].out_prec = Precision::FP32;
    g.nodes[8].params.i32 = {H, E, KTOP, I, SI};
    g.graph_outputs = {8};
    g.runtime.tensors.resize(9);
    g.runtime.tensors[0] = hidden_t;
    g.runtime.tensors[1] = router_t;
    g.runtime.tensors[2] = gu_t;
    g.runtime.tensors[3] = down_t;
    g.runtime.tensors[4] = sg_t;
    g.runtime.tensors[5] = su_t;
    g.runtime.tensors[6] = sd_t;
    g.runtime.tensors[7] = seg_t;

    CPUBackend backend;
    ExecContext ctx;
    ctx.graph = &g;
    ctx.pool = &g.runtime.pool;
    ctx.backend = &backend;
    prepare_execution(ctx);
    execute_graph(ctx);
    Tensor& graph_out = g.runtime.tensors[8];
    std::vector<float> graph_vals(graph_out.ptr<float>(), graph_out.ptr<float>() + H * M);
    CHECK(close_enough(graph_vals, ref, 1e-5f), "OpType::MOE graph dispatch matches reference");

    // DeepSeek-V4 variant: hash-selected routes, sqrt-softplus weights,
    // ungated shared expert, and clipped SwiGLU inputs.
    {
        float hidden_data[] = {20.0f};
        float router_data[] = {0.0f, 0.0f};
        float expert_gate_up_data[] = {
            20.0f, 20.0f, // expert 0 gate/up
            0.0f, 0.0f,   // expert 1 gate/up
        };
        float expert_down_data[] = {1.0f, 0.0f};
        float shared_gate_data[] = {20.0f};
        float shared_up_data[] = {20.0f};
        float shared_down_data[] = {1.0f};
        int32_t token_data[] = {0};
        int32_t hash_data[] = {0};
        float output_data[] = {0.0f};
        Tensor hidden = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1,
            hidden_data);
        Tensor router = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, 2, 1, 1, 1,
            router_data);
        Tensor expert_gate_up = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1,
            expert_gate_up_data);
        Tensor expert_down = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, 2, 1, 1, 1,
            expert_down_data);
        Tensor shared_gate = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1,
            shared_gate_data);
        Tensor shared_up = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1,
            shared_up_data);
        Tensor shared_down = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1,
            shared_down_data);
        Tensor token_ids = Tensor::create(
            Precision::INT32, MemoryType::EXTERNAL, 1, 1, 1, 1,
            token_data);
        Tensor hash_table = Tensor::create(
            Precision::INT32, MemoryType::EXTERNAL, 1, 1, 1, 1,
            hash_data);
        Tensor output = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1,
            output_data);
        std::vector<const Tensor*> deepseek_inputs = {
            &hidden, &router, &expert_gate_up, &expert_down,
            &shared_gate, &shared_up, &shared_down,
            &token_ids, &hash_table,
        };
        CHECK(kernel_qwen3_moe(
                  deepseek_inputs, output, nullptr,
                  1, 2, 1, 1, 1, 2, true, true, 1, 1, 1.0f,
                  false, -1, 7, 8, 10.0f),
              "DeepSeek hash MoE reports successful execution");
        const float clipped =
            10.0f / (1.0f + std::exp(-10.0f)) * 10.0f;
        CHECK(std::abs(output_data[0] - 2.0f * clipped) < 1e-3f,
              "DeepSeek hash MoE applies ungated shared expert and SwiGLU limit");

        // Hy4 clips routed gate/up logits, but its shared branch must retain
        // values above that threshold. Exercise the serialized graph fields.
        GraphNode hy4;
        hy4.op_type = OpType::MOE;
        hy4.params.i32 = {1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1};
        hy4.params.f32 = {1.0f, 10.0f, 0.0f};
        CHECK(dispatch_cpu_moe(hy4, deepseek_inputs, &output, nullptr),
              "Hy4 MoE accepts independent shared clamp");
        CHECK(std::abs(output_data[0] - (clipped + 160000.0f)) < 1e-3f,
              "Hy4 clips routed experts without clipping shared expert");
        hy4.params.f32[2] = 20.0f;
        CHECK(dispatch_cpu_moe(hy4, deepseek_inputs, &output, nullptr) &&
                  std::abs(output_data[0] - (clipped + 400.0f)) < 1e-3f,
              "shared clamp has its own positive threshold");
        hy4.params.f32.resize(2);
        CHECK(dispatch_cpu_moe(hy4, deepseek_inputs, &output, nullptr) &&
                  std::abs(output_data[0] - 2.0f * clipped) < 1e-3f,
              "legacy generic MoE retains inherited clamp semantics");
    }

    return failures == 0 ? 0 : 1;
}
