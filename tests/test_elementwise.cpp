#include "kernels/elementwise.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

bool close(float actual, float expected, float tolerance = 1e-4f) {
    return std::fabs(actual - expected) <=
           tolerance * std::max(1.f, std::fabs(expected));
}

Tensor external_tensor(std::vector<float>& data, int64_t d0, int64_t d1 = 1) {
    return Tensor::create(Precision::FP32, MemoryType::EXTERNAL, d0, d1, 1, 1,
                          data.data());
}

void test_binary_broadcast(ThreadPool& pool) {
    std::vector<float> a(12), b = {1.f, 2.f, 3.f, 4.f}, output(12);
    for (int i = 0; i < 12; ++i)
        a[i] = static_cast<float>(i);
    Tensor ta = external_tensor(a, 4, 3);
    Tensor tb = external_tensor(b, 4, 1);
    Tensor out = external_tensor(output, 4, 3);
    std::vector<const Tensor*> inputs = {&ta, &tb};

    kernel_elementwise(OpType::ADD, inputs, &out, &pool);
    bool add_ok = true;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            add_ok &= output[row * 4 + col] == a[row * 4 + col] + b[col];
    check(add_ok, "ADD singleton broadcast");

    kernel_elementwise(OpType::MUL, inputs, &out, &pool);
    bool mul_ok = true;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            mul_ok &= output[row * 4 + col] == a[row * 4 + col] * b[col];
    check(mul_ok, "MUL singleton broadcast");
}

void test_strided_mul(ThreadPool& pool) {
    std::vector<float> parent(24), output(12);
    for (int i = 0; i < 24; ++i)
        parent[i] = static_cast<float>(i + 1);
    Tensor base = external_tensor(parent, 8, 3);
    Tensor left = base.view_2d(4, 3);
    Tensor right = base.view_2d(4, 3, 4 * sizeof(float));
    Tensor out = external_tensor(output, 4, 3);
    std::vector<const Tensor*> inputs = {&left, &right};
    kernel_elementwise(OpType::MUL, inputs, &out, &pool);

    bool matches = true;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            matches &= output[row * 4 + col] ==
                       parent[row * 8 + col] * parent[row * 8 + col + 4];
    check(matches, "MUL strided views");
}

void test_unary(ThreadPool& pool) {
    std::vector<float> input = {-3.f, -1.f, -0.25f, 0.f, 0.25f, 1.f, 2.f, 3.f};
    std::vector<float> output(input.size());
    Tensor in = external_tensor(input, input.size());
    Tensor out = external_tensor(output, output.size());
    std::vector<const Tensor*> inputs = {&in};

    auto run = [&](OpType op, auto reference, float tolerance,
                   const char* label) {
        kernel_elementwise(op, inputs, &out, &pool);
        bool matches = true;
        for (size_t i = 0; i < input.size(); ++i)
            matches &= close(output[i], reference(input[i]), tolerance);
        check(matches, label);
    };

    run(
        OpType::SILU, [](float x) { return x / (1.f + std::exp(-x)); }, 0.02f,
        "SILU");
    run(OpType::TANH, [](float x) { return std::tanh(x); }, 1e-6f, "TANH");
    run(
        OpType::SIGMOID, [](float x) { return 1.f / (1.f + std::exp(-x)); },
        0.02f, "SIGMOID approximate");
    run(
        OpType::SIGMOID_EXACT,
        [](float x) { return 1.f / (1.f + std::exp(-x)); }, 1e-6f,
        "SIGMOID exact");
    run(
        OpType::EXP, [](float x) { return std::exp(x); }, 0.02f,
        "EXP approximate");
    run(
        OpType::EXP_EXACT, [](float x) { return std::exp(x); }, 1e-6f,
        "EXP exact");
    run(
        OpType::SOFTPLUS, [](float x) { return std::log1pf(std::exp(x)); },
        0.02f, "SOFTPLUS");
}

void test_swiglu(ThreadPool& pool) {
    std::vector<float> merged = {
        -1.f, 0.f, 1.f,  2.f, 2.f,  3.f,  4.f, 5.f,
        0.5f, 1.f, 1.5f, 2.f, -2.f, -1.f, 1.f, 2.f,
    };
    std::vector<float> output(8);
    Tensor in = external_tensor(merged, 8, 2);
    Tensor out = external_tensor(output, 4, 2);
    std::vector<const Tensor*> inputs = {&in};
    kernel_elementwise(OpType::SWIGLU, inputs, &out, &pool);

    bool matches = true;
    for (int row = 0; row < 2; ++row) {
        for (int i = 0; i < 4; ++i) {
            const float gate = merged[row * 8 + i];
            const float up = merged[row * 8 + 4 + i];
            const float expected = gate / (1.f + std::exp(-gate)) * up;
            matches &= close(output[row * 4 + i], expected);
        }
    }
    check(matches, "SWIGLU");
}

void test_parallel_exact(ThreadPool& pool) {
    std::vector<float> input(40000, 0.25f), output(input.size());
    Tensor in = external_tensor(input, input.size());
    Tensor out = external_tensor(output, output.size());
    std::vector<const Tensor*> inputs = {&in};
    kernel_elementwise(OpType::EXP_EXACT, inputs, &out, &pool);
    check(close(output.front(), std::exp(0.25f), 1e-6f) &&
              close(output.back(), std::exp(0.25f), 1e-6f),
          "EXP exact parallel path");
}

} // namespace

int main() {
    ThreadPool pool(4);
    test_binary_broadcast(pool);
    test_strided_mul(pool);
    test_unary(pool);
    test_swiglu(pool);
    test_parallel_exact(pool);

    if (failures == 0)
        std::printf("All elementwise tests passed!\n");
    return failures;
}
