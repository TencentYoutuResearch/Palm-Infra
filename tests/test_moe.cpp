#include "graph/execute.h"
#include "engine/backend.h"
#include "kernels/moe.h"
#include "kernels/moe_ssd.h"
#include "kernels/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

int main() {
    const int H = 8;
    const int M = 3;
    const int E = 5;
    const int KTOP = 2;
    const int I = 6;
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
    kernel_qwen3_moe(inputs, out_t, nullptr, H, E, KTOP, I, SI);
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

    return failures == 0 ? 0 : 1;
}
