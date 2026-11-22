// W4G128 verification batches and routed-expert projections must reproduce
// per-token decode bit for bit: each result is compared with an M=1 decode
// GEMV on the same inputs.
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/moe.h"

#include "cuda_test_support.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using mollm_test::check;
using mollm_test::GuardedBuffer;
using mollm_test::require;
using mollm_test::same_bits;

bool batch_matches_decode(int columns, int inner, int tokens, int lda, int ldc,
                          int rows_per_warp, bool special_values, uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
    std::uniform_int_distribution<int> byte(0, 255);
    const int groups = inner / 128;
    std::vector<Q4B8G128Block> weight(static_cast<size_t>((columns + 7) / 8) * groups);
    for (auto& block : weight) {
        for (float& scale : block.scales) scale = uniform(random) * 0.01f;
        for (auto& subgroup : block.q)
            for (auto& row : subgroup)
                for (auto& packed : row) packed = static_cast<uint8_t>(byte(random));
    }
    std::vector<float> activation(static_cast<size_t>(tokens) * lda);
    for (float& value : activation) value = uniform(random);
    if (special_values) {
        const auto specials = mollm_test::special_fp32_values();
        for (size_t i = 0; i < specials.size(); ++i)
            activation[(i * 977) % activation.size()] = specials[i];
    }

    GuardedBuffer device_weight(weight.size() * sizeof(Q4B8G128Block));
    GuardedBuffer device_activation(activation.size() * sizeof(float));
    GuardedBuffer batch(static_cast<size_t>(tokens) * ldc * sizeof(float));
    GuardedBuffer single(static_cast<size_t>(columns) * sizeof(float));
    device_weight.upload(weight.data());
    device_activation.upload(activation.data());
    batch.fill(0);

    check(mollm_cuda::launch_q4_g128_small_batch_exact(
              device_activation.as<float>(), device_weight.as<Q4B8G128Block>(),
              batch.as<float>(), tokens, columns, inner, lda, ldc, rows_per_warp),
          "W4G128 batch launch");
    check(cudaDeviceSynchronize(), "W4G128 batch");
    const auto batched = batch.download<float>();
    for (int token = 0; token < tokens; ++token) {
        mollm_cuda::launch_q4_g128_dense_gemv(
            device_activation.as<float>() + static_cast<size_t>(token) * lda,
            device_weight.as<Q4B8G128Block>(), single.as<float>(), columns, inner);
        check(cudaGetLastError(), "W4G128 decode launch");
        check(cudaDeviceSynchronize(), "W4G128 decode");
        const auto decoded = single.download<float>();
        const std::vector<float> row(batched.begin() + static_cast<size_t>(token) * ldc,
                                     batched.begin() + static_cast<size_t>(token) * ldc + columns);
        if (!same_bits(row, decoded)) {
            std::fprintf(stderr, "mismatch: columns=%d inner=%d tokens=%d rows=%d token=%d\n",
                         columns, inner, tokens, rows_per_warp, token);
            return false;
        }
    }
    device_weight.check_guards();
    device_activation.check_guards();
    batch.check_guards();
    single.check_guards();
    return true;
}

// Each route's projection equals a decode GEMV on its expert's rows.
bool experts_match_decode(int experts, int rows, int inner, uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
    std::uniform_int_distribution<int> byte(0, 255);
    const size_t expert_blocks = static_cast<size_t>(rows / 8) * (inner / 128);
    std::vector<Q4B8G128Block> weight(expert_blocks * experts);
    for (auto& block : weight) {
        for (float& scale : block.scales) scale = uniform(random) * 0.01f;
        for (auto& subgroup : block.q)
            for (auto& row : subgroup)
                for (auto& packed : row) packed = static_cast<uint8_t>(byte(random));
    }
    // Routes repeat experts and inputs, as tokens sharing experts do.
    const int routes = 2 * experts + 1;
    std::vector<float> input(static_cast<size_t>(routes) * inner);
    for (float& value : input) value = uniform(random);
    GuardedBuffer device_weight(weight.size() * sizeof(Q4B8G128Block));
    GuardedBuffer device_input(input.size() * sizeof(float));
    GuardedBuffer device_output(static_cast<size_t>(routes) * rows * sizeof(float));
    GuardedBuffer single(static_cast<size_t>(rows) * sizeof(float));
    device_weight.upload(weight.data());
    device_input.upload(input.data());
    device_output.fill(0);
    std::vector<mollm_cuda::Nvfp4Projection> jobs(routes);
    std::vector<int> expert_of(routes);
    for (int route = 0; route < routes; ++route) {
        expert_of[route] = (route * 7 + 3) % experts;
        jobs[route].weight = reinterpret_cast<const uint8_t*>(
            device_weight.as<Q4B8G128Block>() + expert_of[route] * expert_blocks);
        jobs[route].input = device_input.as<float>() + static_cast<size_t>(route) * inner;
        jobs[route].output = device_output.as<float>() + static_cast<size_t>(route) * rows;
    }
    GuardedBuffer device_jobs(jobs.size() * sizeof(jobs[0]));
    device_jobs.upload(jobs.data());
    check(mollm_cuda::launch_q4_g128_expert_projections(
              device_jobs.as<mollm_cuda::Nvfp4Projection>(), routes, rows, inner),
          "W4G128 expert projections");
    check(cudaDeviceSynchronize(), "W4G128 expert projections");
    const auto projected = device_output.download<float>();
    for (int route = 0; route < routes; ++route) {
        mollm_cuda::launch_q4_g128_dense_gemv(
            device_input.as<float>() + static_cast<size_t>(route) * inner,
            device_weight.as<Q4B8G128Block>() + expert_of[route] * expert_blocks,
            single.as<float>(), rows, inner);
        check(cudaGetLastError(), "W4G128 decode launch");
        check(cudaDeviceSynchronize(), "W4G128 decode");
        const auto decoded = single.download<float>();
        const std::vector<float> row(projected.begin() + static_cast<size_t>(route) * rows,
                                     projected.begin() + static_cast<size_t>(route + 1) * rows);
        if (!same_bits(row, decoded)) {
            std::fprintf(stderr, "expert mismatch: rows=%d inner=%d route=%d\n", rows, inner, route);
            return false;
        }
    }
    device_weight.check_guards();
    device_input.check_guards();
    device_output.check_guards();
    return true;
}

}  // namespace

int main() {
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    bool passed = true;
    for (int rows : {1, 2, 4}) {
        for (int tokens = 2; tokens <= 8; ++tokens) {
            // Dense projection shape, partial 8-row tiles, and padded strides.
            const uint32_t seed = 16u * tokens + rows;
            passed &= batch_matches_decode(2560, 2560, tokens, 2560, 2560, rows, false, 11u + seed);
            passed &= batch_matches_decode(20, 256, tokens, 260, 24, rows, false, 31u + seed);
            passed &= batch_matches_decode(38, 128, tokens, 128, 38, rows, true, 53u + seed);
        }
    }
    // Qwen3.5-MoE gate/up and down geometry, and a minimal expert.
    passed &= experts_match_decode(4, 1024, 2048, 71u);
    passed &= experts_match_decode(3, 2048, 512, 73u);
    passed &= experts_match_decode(2, 8, 128, 79u);
    require(mollm_cuda::launch_q4_g128_expert_projections(nullptr, 1, 12, 128) == cudaErrorInvalidValue,
            "expert launcher rejects partial 8-row tiles");
    const float* activation = nullptr;
    require(mollm_cuda::launch_q4_g128_small_batch_exact(
                activation, nullptr, nullptr, 9, 8, 128, 128, 8, 1) == cudaErrorInvalidValue,
            "batch launcher rejects more than eight tokens");
    require(mollm_cuda::launch_q4_g128_small_batch_exact(
                activation, nullptr, nullptr, 2, 8, 96, 96, 8, 1) == cudaErrorInvalidValue,
            "batch launcher rejects partial 128-column groups");
    require(mollm_cuda::launch_q4_g128_small_batch_exact(
                activation, nullptr, nullptr, 2, 8, 128, 128, 8, 8) == cudaErrorInvalidValue,
            "batch launcher rejects unsupported rows per warp");
    if (!passed) return 1;
    std::printf("W4G128 batches and expert projections match decode bit for bit\n");
    return 0;
}
