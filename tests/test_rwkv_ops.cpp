#include "backends/cpu/backend.h"
#include "graph/graph.h"
#include "runtime/threading.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
int failures = 0;

void check(bool ok, const char* label) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}

bool near(float a, float b) { return std::fabs(a - b) < 2e-5f; }

Tensor tensor(std::vector<float>& values, int width, int rows = 1) {
    return Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                          width, rows, 1, 1, values.data());
}

void test_shift_and_mix(bool padded) {
    std::vector<float> x{1, 2, 3, 4, 5, 6}, state{-1, -2}, shifted(6, 99.f);
    Tensor tx = tensor(x, 2, 3), ts = tensor(state, 2), ty = tensor(shifted, 2, 3);
    GraphNode node;
    node.op_type = OpType::RWKV_TOKEN_SHIFT;
    node.params.i32 = {2, 3};
    if (padded) node.params.i32.push_back(2);
    CPUBackend{}.dispatch(node, {&tx, &ts}, &ty, nullptr);
    check(shifted == (padded ? std::vector<float>{-2, -4, -2, -2, 0, 0}
                             : std::vector<float>{-2, -4, -2, -2, -2, -2}),
          "token shift output and padded tail");
    check(state == (padded ? std::vector<float>{3, 4} : std::vector<float>{5, 6}),
          "token shift persists the last real token");

    std::vector<float> mix{0.5f, 2.f}, output(6);
    Tensor tm = tensor(mix, 2), tout = tensor(output, 2, 3);
    node.op_type = OpType::RWKV_MIX;
    node.params = {};
    CPUBackend{}.dispatch(node, {&tx, &ty, &tm}, &tout, nullptr);
    for (int i = 0; i < 6; ++i)
        check(near(output[i], x[i] + shifted[i] * mix[i % 2]),
              "RWKV mix broadcasts channel weights across tokens");
}

void test_norms(bool explicit_epsilon) {
    std::vector<float> x{3, 4, 5, 12}, output(4);
    Tensor tx = tensor(x, 4), tout = tensor(output, 4);
    GraphNode node;
    node.op_type = OpType::RWKV_L2_NORM;
    node.params.i32 = {2, 2};
    if (explicit_epsilon) node.params.f32 = {0.5f};
    CPUBackend{}.dispatch(node, {&tx}, &tout, nullptr);
    const float eps = explicit_epsilon ? 0.5f : 1e-12f;
    for (int i = 0; i < 4; ++i)
        check(near(output[i], x[i] / ((i < 2 ? 5.f : 13.f) + eps)),
              "RWKV L2 normalization uses per-head dimensions and epsilon");

    std::vector<float> raw{1, 3, 2, 4}, r(4, 1), k(4, 1), v(4, 0.4f);
    std::vector<float> rk(2, 0.25f), weight(2, 2), bias(2, 0.1f), gate(4, 0.5f);
    Tensor traw = tensor(raw, 2, 2), tr = tensor(r, 2, 2), tk = tensor(k, 2, 2);
    Tensor tv = tensor(v, 2, 2), trk = tensor(rk, 2), tw = tensor(weight, 2);
    Tensor tb = tensor(bias, 2), tg = tensor(gate, 2, 2);
    tout = tensor(output, 2, 2);
    node.op_type = OpType::RWKV_POST;
    node.params.i32 = {1, 2};
    node.params.f32 = explicit_epsilon ? std::vector<float>{0.25f}
                                       : std::vector<float>{};
    CPUBackend{}.dispatch(node, {&traw, &tr, &tk, &tv, &trk, &tw, &tb, &tg},
                           &tout, nullptr);
    const float inv = 1.f / std::sqrt(1.f + (explicit_epsilon ? 0.25f : 64e-5f));
    for (int i = 0; i < 4; ++i)
        check(near(output[i], ((i % 2 ? 1.f : -1.f) * inv * 2.f + 0.3f) * 0.5f),
              "RWKV post combines normalization, affine, bonus and gate");
}

void test_recurrence(bool padded) {
    constexpr int heads = 4, dim = 4, seq = 3, hidden = heads * dim;
    const int real = padded ? 2 : seq;
    std::vector<float> r(hidden * seq, 1), decay(hidden * seq, 0.5f);
    std::vector<float> k(hidden * seq, 1), v(hidden * seq);
    std::vector<float> a(hidden * seq, 0.1f), b(hidden * seq, 0.2f);
    std::vector<float> state(heads * dim * dim, 0), output(hidden * seq, 99);
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < heads; ++h)
            for (int row = 0; row < dim; ++row)
                v[t * hidden + h * dim + row] = 0.125f * (t + 1) * (h + 1) * (row + 1);
    Tensor tr = tensor(r, hidden, seq), td = tensor(decay, hidden, seq);
    Tensor tk = tensor(k, hidden, seq), tv = tensor(v, hidden, seq);
    Tensor ta = tensor(a, hidden, seq), tb = tensor(b, hidden, seq);
    Tensor ts = tensor(state, state.size()), tout = tensor(output, hidden, seq);
    GraphNode node;
    node.op_type = OpType::RWKV7;
    node.params.i32 = {heads, dim, seq};
    if (padded) node.params.i32.push_back(real);
    ThreadPool pool(2);
    CPUBackend{}.dispatch(node, {&tr, &td, &tk, &tv, &ta, &tb, &ts}, &tout, &pool);

    // Each state row stays constant across columns. Its closed-form response
    // to the ramp input includes decay and the rank-one state update.
    const double growth = 0.5 + dim * 0.1 * 0.2;
    for (int t = 0; t < seq; ++t) {
        double response = 0;
        for (int j = 0; j <= t; ++j)
            response += (j + 1) * std::pow(growth, t - j);
        for (int h = 0; h < heads; ++h)
            for (int row = 0; row < dim; ++row) {
                const float expected = 0.125 * (h + 1) * (row + 1) * response;
                check(near(output[t * hidden + h * dim + row], t < real ? dim * expected : 0.f),
                      "RWKV7 ramp response and padded output");
                if (t == real - 1)
                    for (int col = 0; col < dim; ++col)
                        check(near(state[(h * dim + row) * dim + col], expected),
                              "RWKV7 state excludes padded tokens");
            }
    }
}
}  // namespace

int main() {
    for (bool explicit_params : {false, true}) {
        test_shift_and_mix(explicit_params);
        test_norms(explicit_params);
        test_recurrence(explicit_params);
    }
    if (!failures) std::puts("All RWKV operator tests passed!");
    return failures ? 1 : 0;
}
