#include "kernels/cpu/gated_residual.h"
#include "kernels/bf16.h"

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

bool close(float a, float b, float tolerance = 1e-5f) {
    return std::fabs(a - b) <= tolerance;
}

} // namespace

int main() {
    constexpr int hidden = 2;
    constexpr int streams = 2;
    constexpr int tokens = 2;
    std::vector<float> x = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -2.0f, 1.0f, 4.0f, -3.0f,
    };
    std::vector<float> gamma = {1.0f, 2.0f, 0.5f, 1.5f};
    std::vector<float> normalized(x.size());
    Tensor xt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                               hidden * streams, tokens, 1, 1, x.data());
    Tensor wt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                               hidden * streams, 1, 1, 1, gamma.data());
    Tensor nt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                               hidden * streams, tokens, 1, 1,
                               normalized.data());
    check(kernel_group_rms_norm(xt, wt, nt, hidden, 0.0f),
          "grouped RMSNorm executes");
    const float inv0 = 1.0f / std::sqrt(2.5f);
    const float inv1 = 1.0f / std::sqrt(12.5f);
    check(close(normalized[0], 1.0f * inv0) &&
          close(normalized[1], 4.0f * inv0) &&
          close(normalized[2], 1.5f * inv1) &&
          close(normalized[3], 6.0f * inv1),
          "grouped RMSNorm uses independent streams and per-element gamma");

    std::vector<float> gates = {
        0.2f, 0.4f, 0.6f, 0.8f,
        0.5f, 0.25f, 1.0f, 0.75f,
    };
    std::vector<float> reduced(hidden * tokens);
    Tensor gt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                               hidden * streams, tokens, 1, 1, gates.data());
    Tensor rt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                               hidden, tokens, 1, 1, reduced.data());
    check(kernel_gr_reduce(nt, gt, rt, hidden, streams),
          "GR read reduction executes");
    check(close(reduced[0], mollm_round_to_bf16(
                (normalized[0] * gates[0] + normalized[2] * gates[2]) / 2)) &&
          close(reduced[1], mollm_round_to_bf16(
                (normalized[1] * gates[1] + normalized[3] * gates[3]) / 2)),
          "GR read reduction matches reference");

    std::vector<float> injection = {0.25f, 1.5f, 0.5f, 1.25f};
    std::vector<float> output(x.size());
    Tensor it = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                               streams, tokens, 1, 1, injection.data());
    Tensor ot = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                               hidden * streams, tokens, 1, 1, output.data());
    check(kernel_gr_inject(rt, xt, it, ot, hidden, streams),
          "GR write injection executes");
    check(close(output[0], mollm_round_to_bf16(
                    x[0] + injection[0] * reduced[0])) &&
          close(output[1], mollm_round_to_bf16(
                    x[1] + injection[0] * reduced[1])) &&
          close(output[2], mollm_round_to_bf16(
                    x[2] + injection[1] * reduced[0])) &&
          close(output[3], mollm_round_to_bf16(
                    x[3] + injection[1] * reduced[1])),
          "GR write injection matches reference");

    if (failures) {
        std::fprintf(stderr, "%d gated residual test(s) failed\n", failures);
        return 1;
    }
    std::printf("All gated residual tests passed!\n");
    return 0;
}
