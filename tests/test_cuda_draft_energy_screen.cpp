#include "kernels/cuda/draft_compact_head.h"

#include "cuda_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

using namespace mollm_cuda;
using mollm_test::check;
using mollm_test::GuardedBuffer;
using mollm_test::require;

constexpr int rows = 248320;
constexpr int inner = 2560;
constexpr int candidates = 1024;

uint32_t ordered_key(float value) {
    uint32_t bits = mollm_test::to_bits(value);
    if (!(bits & 0x7fffffffU)) bits = 0;  // Stable +0/-0 equivalence in CUB.
    return bits & 0x80000000U ? ~bits : bits ^ 0x80000000U;
}

void test_screen(int group_count) {
    std::vector<Q4CompactG32> weights(size_t(rows / 8) * (inner / 32));
    for (size_t i = 0; i < weights.size(); ++i) {
        for (int row = 0; row < 8; ++row) {
            auto& block = weights[i];
            block.scales[row] = uint16_t(0x3d00 + (i * 13 + row * 7) % 512);
            uint32_t value = uint32_t(i * 1664525 + row * 1013904223u);
            for (int j = 0; j < 16; ++j) {
                value = value * 1664525 + 1013904223;
                block.q[row][j] = uint8_t(value >> 24);
            }
        }
    }
    GuardedBuffer weight(weights.size() * sizeof(weights[0]));
    GuardedBuffer input(inner * sizeof(float));
    GuardedBuffer output(rows * sizeof(float));
    GuardedBuffer full(rows * sizeof(float));
    GuardedBuffer coarse(rows * sizeof(float));
    GuardedBuffer sorted(rows * sizeof(float));
    GuardedBuffer ids(rows * sizeof(int));
    GuardedBuffer selected(rows * sizeof(int));
    GuardedBuffer groups(32 * sizeof(int));
    weight.upload(weights.data());
    DraftHeadScreenScratch scratch{coarse.as<float>(), sorted.as<float>(), ids.as<int>(),
                                   selected.as<int>(), groups.as<int>(), nullptr, 0};
    size_t bytes = 0;
    check(draft_head_screen_temp_bytes(scratch, bytes));
    require(bytes > 0, "temporary query");
    GuardedBuffer temporary(bytes);
    scratch.sort_temp = temporary.data();
    scratch.sort_bytes = bytes;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(initialize_draft_head_screen(scratch, stream));
    check(cudaStreamSynchronize(stream));
    const auto original_ids = ids.download<int>();
    for (int i = 0; i < rows; ++i) require(original_ids[i] == i, "initialized stable IDs");

    const auto* w = weight.as<Q4CompactG32>();
    std::vector<float> values(inner, 0.f);
    input.upload(values.data());
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    check(launch_draft_head_screen(input.as<float>(), w, output.as<float>(), rows, inner, scratch,
                                   stream, group_count));
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int fixture = 0; fixture < 10; ++fixture) {
        for (int i = 0; i < inner; ++i) {
            if (fixture == 0)
                values[i] = 0.f;
            else if (fixture == 1)
                values[i] = i == 73 * 32 + 7 ? -2.f : 0.f;
            else
                values[i] = float((i * 29 + fixture * 7) % 251 - 125) * (fixture == 3 ? 1e-40f : .013f);
        }
        if (fixture == 9) std::fill(values.begin(), values.end(), 1.f);
        if (fixture == 5) std::fill(values.begin(), values.end(), mollm_test::from_bits(0xffc00001U));
        if (fixture == 6) values[73 * 32 + 7] = INFINITY;
        if (fixture == 7) values[17] = -INFINITY;
        // Energy overflows, logits need not.
        if (fixture == 8) std::fill(values.begin(), values.end(), 1e30f);
        if (fixture == 4) {
            // Replay must observe changed weight contents without repacking.
            weights[0].scales[0] = 0x3f00;
            weights[0].q[0][0] ^= 0x5a;
            check(cudaMemcpy(weight.data(), weights.data(), sizeof(weights[0]), cudaMemcpyHostToDevice));
        }
        input.upload(values.data());
        launch_q4_g32_compact_gemv(input.as<float>(), w, full.as<float>(), rows, inner, stream);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(stream));
        const auto reference = full.download<float>();
        check(launch_draft_head_screen(input.as<float>(), w, output.as<float>(), rows, inner, scratch,
                                       stream, group_count));
        check(cudaStreamSynchronize(stream));
        const auto eager = output.download<float>();
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        const auto actual = output.download<float>();
        require(mollm_test::same_bits(eager, actual), "mutable graph equals eager");

        float energies[80];
        std::vector<int> expected_groups(80);
        std::iota(expected_groups.begin(), expected_groups.end(), 0);
        for (int group = 0; group < 80; ++group) {
            float sum = 0;
            for (int j = 0; j < 32; ++j) {
                volatile float product = values[group * 32 + j] * values[group * 32 + j];
                sum += product;
            }
            energies[group] = std::isnan(sum) ? INFINITY : sum;
        }
        std::stable_sort(expected_groups.begin(), expected_groups.end(),
                         [&](int a, int b) { return energies[a] > energies[b]; });
        const auto actual_groups = groups.download<int>();
        require(std::equal(actual_groups.begin(), actual_groups.end(), expected_groups.begin()),
                "CPU group selection");
        const auto scores = coarse.download<float>();
        const auto selected_ids = selected.download<int>();
        std::vector<int> expected(rows);
        std::iota(expected.begin(), expected.end(), 0);
        std::partial_sort(expected.begin(), expected.begin() + candidates, expected.end(),
                          [&](int a, int b) {
                              const auto left = ordered_key(scores[a]);
                              const auto right = ordered_key(scores[b]);
                              return left > right || (left == right && a < b);
                          });
        std::vector<bool> included(rows, false);
        for (int j = 0; j < candidates; ++j) {
            require(expected[j] == selected_ids[j], "CPU vocabulary selection and ties");
            const int row = selected_ids[j];
            require(row >= 0 && row < rows && !included[row], "candidate bounds and uniqueness");
            included[row] = true;
            require((std::isnan(actual[row]) && std::isnan(reference[row])) ||
                        mollm_test::to_bits(actual[row]) == mollm_test::to_bits(reference[row]),
                    "selected full-K native bits / NaN semantics");
        }
        for (int row = 0; row < rows; ++row) {
            if (!included[row]) require(actual[row] == -INFINITY, "non-candidate sentinel");
        }
        if (fixture == 9) {
            // Equal energies select ascending K groups. A full-K projection
            // with the unselected input groups zeroed independently checks
            // that the requested group budget is used by the coarse pass.
            auto masked = values;
            std::fill(masked.begin() + group_count * 32, masked.end(), 0.f);
            input.upload(masked.data());
            launch_q4_g32_compact_gemv(input.as<float>(), w, full.as<float>(), rows, inner, stream);
            check(cudaGetLastError());
            check(cudaStreamSynchronize(stream));
            const auto coarse_reference = full.download<float>();
            require(mollm_test::same_bits(scores, coarse_reference),
                    "coarse group budget matches masked full projection");
            input.upload(values.data());
        }
        require(ids.download<int>() == original_ids, "stable IDs immutable");
        require(mollm_test::same_bits(input.download<float>(), values), "input immutable");
        for (const auto* buffer :
             {&weight, &input, &output, &full, &coarse, &sorted, &ids, &selected, &groups, &temporary})
            buffer->check_guards();
    }
    require(mollm_test::same_bits(weight.download<Q4CompactG32>(), weights), "weights immutable");
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(graph));

    // Rejected geometry, aliasing and capacity must not enqueue partial work.
    const float* x = input.as<float>();
    float* y = output.as<float>();
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    auto invalid = scratch;
    invalid.sort_bytes = bytes - 1;
    require(launch_draft_head_screen(x, w, y, rows, inner, invalid, stream) == cudaErrorInvalidValue,
            "short temporary");
    invalid = scratch;
    invalid.sorted = invalid.coarse;
    require(initialize_draft_head_screen(invalid, stream) == cudaErrorInvalidValue, "aliased scratch");
    invalid = scratch;
    invalid.sort_temp = coarse.data();
    require(launch_draft_head_screen(x, w, y, rows, inner, invalid, stream) == cudaErrorInvalidValue,
            "aliased temporary");
    require(launch_draft_head_screen(x, w, coarse.as<float>(), rows, inner, scratch, stream) ==
                cudaErrorInvalidValue,
            "aliased output");
    require(launch_draft_head_screen(nullptr, w, y, rows, inner, scratch, stream) == cudaErrorInvalidValue,
            "null input");
    require(launch_draft_head_screen(x + 1, w, y, rows, inner, scratch, stream) == cudaErrorInvalidValue,
            "unaligned input");
    require(launch_draft_head_screen(x, w, y, rows - 1, inner, scratch, stream) == cudaErrorInvalidValue,
            "unsupported rows");
    require(launch_draft_head_screen(x, w, y, rows, inner - 32, scratch, stream) == cudaErrorInvalidValue,
            "unsupported inner");
    for (int unsupported : {0, 15, 17, 20, 25, 80}) {
        require(launch_draft_head_screen(x, w, y, rows, inner, scratch, stream, unsupported) ==
                    cudaErrorInvalidValue,
                "unsupported group budget");
    }
    check(cudaStreamEndCapture(stream, &graph));
    size_t nodes = 1;
    check(cudaGraphGetNodes(graph, nullptr, &nodes));
    require(nodes == 0, "invalid calls submit no kernels");
    check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));
    std::printf("draft energy screen groups=%d: native bits, CPU selections, mutable graphs, guards, "
                "immutable inputs and rejection PASS\n",
                group_count);
}

}  // namespace

int main() {
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    test_screen(32);
    test_screen(24);
    test_screen(16);
}
