// Correctness of the scalar Q8 greedy head screen (pack_q8_head_screen and
// launch_q8_head_screen). For every fixture the screen runs as a CUDA graph
// replay with changed inputs and is compared against the full Q8 GEMV:
// approximate scores stay inside the conservative bound, every retained
// candidate carries the bit-exact Q8 score, and the argmax matches exactly.
// --large uses the production head shape (248320 rows).
#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/q8_head_screen.h"

#include "cuda_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using mollm_cuda::ArgMaxPair;
using mollm_test::GuardedBuffer;
using mollm_test::check;
using mollm_test::require;

constexpr int inner = 2560;
constexpr int fixture_count = 15;

// Deterministic pseudo-random Q8 code for flat weight index `index`.
int8_t weight_code(size_t index) {
    uint32_t x = uint32_t(index);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return int8_t(x & 255);
}

// Rows 0 and 1 are identical so the argmax has to resolve an exact tie.
std::vector<int8_t> make_weights(int rows) {
    std::vector<int8_t> weights(size_t(rows) * inner);
    for (int row = 0; row < rows; ++row) {
        const size_t source = size_t(row == 1 ? 0 : row) * inner;
        for (int column = 0; column < inner; ++column)
            weights[size_t(row) * inner + column] = weight_code(source + column);
    }
    return weights;
}

std::vector<float> make_scales(int rows) {
    std::vector<float> scales(rows);
    for (int row = 0; row < rows; ++row)
        scales[row] = row < 2 ? .001f : (row % 13 == 0 ? -.001f : .001f);
    return scales;
}

// Fixture 0 is all zeros, fixture 1 is correlated with row 0, and every other
// fixture starts from a sine wave before its edge case is applied.
std::vector<float> make_input(int fixture, const std::vector<int8_t>& weights) {
    std::vector<float> input(inner, 0.f);
    if (fixture == 1) {
        for (int i = 0; i < inner; ++i)
            input[i] = float(weights[i]) / 128.f;
    } else if (fixture >= 2) {
        for (int i = 0; i < inner; ++i)
            input[i] = std::sin(float(i) * .31f);
    }
    return input;
}

void apply_edge_case(int fixture, std::vector<float>& input, std::vector<float>& scales) {
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float infinity = std::numeric_limits<float>::infinity();
    switch (fixture) {
    case 3:  // Large finite input that is still eligible for screening.
        for (float& value : input)
            value *= 1e16f;
        break;
    case 4:  // Out-of-range input that must fall back.
        for (float& value : input)
            value *= 1e20f;
        break;
    case 5:  // Subnormal input.
        for (float& value : input)
            value *= 1e-40f;
        break;
    case 6:
        std::fill(scales.begin(), scales.end(), 1e-40f);
        break;
    case 7:
        std::fill(scales.begin(), scales.end(), 1e10f);
        break;
    case 8:
        std::fill(scales.begin(), scales.end(), 1e11f);
        break;
    case 9:
        input[17] = nan;
        break;
    case 10:
        input[17] = infinity;
        break;
    case 11:
        scales[0] = infinity;
        scales[1] = nan;
        break;
    case 12:
        std::fill(input.begin(), input.end(), 0.f);
        std::fill(scales.begin(), scales.end(), std::numeric_limits<float>::max());
        break;
    case 13:
        std::fill(scales.begin(), scales.end(), nan);
        break;
    case 14:
        std::fill(scales.begin(), scales.end(), -0.f);
        break;
    default:
        break;
    }
}

bool same_score(float a, float b) {
    return (std::isnan(a) && std::isnan(b)) || mollm_test::to_bits(a) == mollm_test::to_bits(b);
}

void run(bool large) {
    const int rows = large ? 248320 : 4099;
    const std::vector<int8_t> weights = make_weights(rows);
    const std::vector<float> base_scales = make_scales(rows);

    GuardedBuffer w(weights.size());
    GuardedBuffer packed(size_t(rows) * inner / 4 * 3);
    GuardedBuffer scales(size_t(rows) * sizeof(float));
    GuardedBuffer x(inner * sizeof(float));
    GuardedBuffer approximate(size_t(rows) * sizeof(float));
    GuardedBuffer output(size_t(rows) * sizeof(float));
    GuardedBuffer reference(size_t(rows) * sizeof(float));
    GuardedBuffer norm(sizeof(double));
    GuardedBuffer lower(257 * sizeof(float));
    GuardedBuffer count(sizeof(unsigned));
    GuardedBuffer ids(size_t(rows) * sizeof(int));
    GuardedBuffer argmax_partial(256 * sizeof(ArgMaxPair));
    GuardedBuffer argmax_result(sizeof(ArgMaxPair));
    w.upload(weights.data());
    check(cudaDeviceSynchronize(), "weight upload");

    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    check(mollm_cuda::pack_q8_head_screen(w.as<int8_t>(), packed.as<uint8_t>(), rows, stream), "pack");
    check(cudaStreamSynchronize(stream), "pack");

    const mollm_cuda::Q8HeadScreenScratch scratch{approximate.as<float>(), lower.as<float>(),
                                                  norm.as<double>(), count.as<unsigned>(),
                                                  ids.as<int>()};
    auto launch_reference = [&] {
        check(mollm_cuda::launch_q8_tuned_gemv(x.as<float>(), w.as<int8_t>(), scales.as<float>(),
                                               reference.as<float>(), rows, inner, 8, false, stream),
              "Q8 reference GEMV");
    };
    auto launch_screen = [&] {
        check(mollm_cuda::launch_q8_head_screen(x.as<float>(), w.as<int8_t>(), packed.as<uint8_t>(),
                                                scales.as<float>(), output.as<float>(), rows, scratch,
                                                stream),
              "Q8 head screen");
    };
    auto argmax = [&](const float* values) {
        mollm_cuda::launch_argmax(values, rows, argmax_partial.as<ArgMaxPair>(), 256,
                                  argmax_result.as<ArgMaxPair>());
        check(cudaGetLastError(), "argmax");
        return argmax_result.download<ArgMaxPair>()[0].index;
    };

    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    launch_screen();
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");

    for (int fixture = 0; fixture < fixture_count; ++fixture) {
        std::vector<float> input = make_input(fixture, weights);
        std::vector<float> row_scales = base_scales;
        apply_edge_case(fixture, input, row_scales);
        check(cudaMemcpyAsync(scales.data(), row_scales.data(), scales.bytes(), cudaMemcpyHostToDevice,
                              stream),
              "upload scales");
        check(cudaMemcpyAsync(x.data(), input.data(), x.bytes(), cudaMemcpyHostToDevice, stream),
              "upload input");
        launch_reference();
        check(cudaGraphLaunch(executable, stream), "graph replay");
        check(cudaStreamSynchronize(stream), "fixture");

        const unsigned retained = count.download<unsigned>()[0];
        require(retained > 0 && retained <= unsigned(rows), "retained candidate count");
        const std::vector<float> screened = output.download<float>();
        const std::vector<float> exact = reference.download<float>();
        const std::vector<float> coarse = approximate.download<float>();
        const std::vector<int> selected = ids.download<int>();

        double l1 = 0;
        for (float value : input)
            l1 += std::fabs(double(value));
        for (int row = 0; row < rows; ++row) {
            const bool bounded = std::isfinite(coarse[row]) && std::isfinite(l1) && l1 <= 1e20 &&
                                 std::isfinite(row_scales[row]) && std::fabs(row_scales[row]) <= 1e10f;
            if (!bounded)
                continue;
            const double bound = 2.04 * std::fabs(double(row_scales[row])) * l1 + 1e-5;
            require(std::fabs(double(coarse[row]) - exact[row]) <= bound,
                    "approximate score outside the conservative bound");
        }
        for (unsigned i = 0; i < retained; ++i) {
            const int row = selected[i];
            require(row >= 0 && row < rows, "candidate index out of range");
            require(same_score(screened[row], exact[row]), "retained score differs from full Q8");
        }
        require(argmax(output.as<float>()) == argmax(reference.as<float>()),
                "screened argmax differs from full Q8");
        for (const GuardedBuffer* buffer : {&w, &packed, &scales, &x, &approximate, &output, &reference,
                                            &norm, &lower, &count, &ids, &argmax_partial, &argmax_result})
            buffer->check_guards();
        std::printf("fixture=%d rows=%d retained=%u exact_argmax=pass\n", fixture, rows, retained);
    }

    check(cudaGraphExecDestroy(executable), "destroy graph exec");
    check(cudaGraphDestroy(graph), "destroy graph");
    check(cudaStreamDestroy(stream), "destroy stream");
    std::printf("Q8 head screen: rows=%d fixtures=%d bounds, retained bits, argmax and guards PASS\n", rows,
                fixture_count);
}

}  // namespace

int main(int argc, char** argv) {
    const bool large = argc == 2 && !std::strcmp(argv[1], "--large");
    if (argc > 2 || (argc == 2 && !large)) {
        std::fprintf(stderr, "usage: %s [--large]\n", argv[0]);
        return 2;
    }
    if (!mollm_test::cuda_device_available())
        return mollm_test::skip_status;
    run(large);
    return 0;
}
