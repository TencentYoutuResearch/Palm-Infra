// Scalar Q8 greedy head screen with the 4-bit packed approximation
// (pack_q8_head_screen_q4 / launch_q8_head_screen_q4): packed codes and their
// quantization bound, eager/graph equality, exact argmax, bit-exact retained
// scores, the -infinity sentinel for discarded rows, the conservative interval
// bound, immutability of inputs and weights, guard bands, the scale-disparity
// anchor regression and rejection of invalid launches. --large uses the
// production head shape (248320 rows) on a subset of fixtures.
#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/q8_head_screen.h"

#include "cuda_test_support.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using namespace mollm_cuda;
using mollm_test::GuardedBuffer;
using mollm_test::check;
using mollm_test::require;

constexpr int inner = 2560;
constexpr int packed_bytes_per_row = inner / 2;
constexpr int fixture_count = 16;

bool same_score(float a, float b) {
    return (std::isnan(a) && std::isnan(b)) || mollm_test::to_bits(a) == mollm_test::to_bits(b);
}

void run(bool large) {
    const int rows = large ? 248320 : 521;
    std::vector<int8_t> weights(size_t(rows) * inner);
    std::vector<float> scales(rows);
    std::vector<float> values(inner);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = int8_t((i * 29 + i / 257) % 256);
    for (int i = 0; i < 256; ++i)
        weights[i] = int8_t(i);
    std::copy_n(weights.begin(), inner, weights.begin() + inner);
    for (int i = 0; i < rows; ++i)
        scales[i] = i % 13 ? .001f : -.001f;
    scales[1] = scales[0];

    GuardedBuffer w(weights.size());
    GuardedBuffer packed(size_t(rows) * packed_bytes_per_row);
    GuardedBuffer sf(rows * sizeof(float));
    GuardedBuffer x(inner * sizeof(float));
    GuardedBuffer output(rows * sizeof(float));
    GuardedBuffer full(rows * sizeof(float));
    GuardedBuffer approximate(rows * sizeof(float));
    GuardedBuffer lower(257 * sizeof(float));
    GuardedBuffer norm(sizeof(double));
    GuardedBuffer count(sizeof(unsigned));
    GuardedBuffer ids(rows * sizeof(int));
    GuardedBuffer partial(256 * sizeof(ArgMaxPair));
    GuardedBuffer result(sizeof(ArgMaxPair));
    w.upload(weights.data());
    sf.upload(scales.data());
    check(cudaDeviceSynchronize(), "initial upload");

    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    // Packs the current weights and verifies every 4-bit code (all of them for
    // the small shape, the first and last 512 groups for the large shape).
    auto repack = [&] {
        check(pack_q8_head_screen_q4(w.as<int8_t>(), packed.as<uint8_t>(), rows, stream), "pack");
        check(cudaStreamSynchronize(stream), "pack");
        const std::vector<uint8_t> codes = packed.download<uint8_t>();
        const size_t groups = weights.size() / 4;
        for (size_t g = 0; g < groups; ++g) {
            if (large && g >= 512 && g + 512 < groups)
                continue;
            const unsigned packet = unsigned(codes[g * 2]) | unsigned(codes[g * 2 + 1]) << 8;
            for (int j = 0; j < 4; ++j) {
                const int weight = int(weights[g * 4 + j]);
                const int code = int(packet >> (4 * j)) & 15;
                require(code == (weight + 128) / 16, "packed Q4 codes");
                require(std::fabs(weight - (16 * code - 120.5)) <= 7.5, "Q4 half-center quantization bound");
            }
        }
        return codes;
    };
    std::vector<uint8_t> original_packed = repack();

    const Q8HeadScreenScratch scratch{approximate.as<float>(), lower.as<float>(), norm.as<double>(),
                                      count.as<unsigned>(), ids.as<int>()};
    auto launch = [&] {
        check(launch_q8_head_screen_q4(x.as<float>(), w.as<int8_t>(), packed.as<uint8_t>(), sf.as<float>(),
                                       output.as<float>(), rows, scratch, stream),
              "Q4 head screen");
    };
    auto argmax = [&](const float* data) {
        launch_argmax(data, rows, partial.as<ArgMaxPair>(), 256, result.as<ArgMaxPair>());
        check(cudaGetLastError(), "argmax");
        check(cudaDeviceSynchronize(), "argmax");
        return result.download<ArgMaxPair>()[0].index;
    };

    x.upload(values.data());
    check(cudaDeviceSynchronize(), "input upload");
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    launch();
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");

    int fixtures = 0;
    for (int fixture = 0; fixture < fixture_count; ++fixture) {
        if (large && fixture != 0 && fixture != 2 && fixture != 15)
            continue;
        ++fixtures;
        for (int j = 0; j < inner; ++j)
            values[j] = std::sin(float(j) * .31f);
        std::vector<float> ss = scales;
        switch (fixture) {
        case 0:
            std::fill(values.begin(), values.end(), 0.f);
            break;
        case 1:
            for (int j = 0; j < inner; ++j)
                values[j] = float(weights[j]) / 128.f;
            break;
        case 3:
            for (float& v : values)
                v *= 1e16f;
            break;
        case 4:
            for (float& v : values)
                v *= 1e20f;
            break;
        case 5:
            for (float& v : values)
                v *= 1e-40f;
            break;
        case 6:
            std::fill(ss.begin(), ss.end(), 1e-40f);
            break;
        case 7:
            std::fill(ss.begin(), ss.end(), 1e10f);
            break;
        case 8:
            std::fill(ss.begin(), ss.end(), 1e11f);
            break;
        case 9:
            values[17] = NAN;
            break;
        case 10:
            values[17] = INFINITY;
            break;
        case 11:
            ss[0] = INFINITY;
            ss[1] = NAN;
            break;
        case 12:
            std::fill(values.begin(), values.end(), 0.f);
            std::fill(ss.begin(), ss.end(), std::numeric_limits<float>::max());
            break;
        case 13:
            std::fill(ss.begin(), ss.end(), NAN);
            break;
        case 14:
            std::fill(ss.begin(), ss.end(), -0.f);
            break;
        case 15:
            // Scale disparity makes the max-lower-bound anchor score zero.
            // The coarse-center anchor scores 20; every zero row's certified
            // upper bound is < 20, so precisely one candidate must survive.
            std::fill(values.begin(), values.end(), 1.f);
            std::fill(ss.begin(), ss.end(), 1e-4f);
            ss[0] = 1e-8f;
            ss[rows - 1] = 1.f;
            std::fill(weights.begin(), weights.end(), 0);
            weights[size_t(rows - 1) * inner] = 20;
            w.upload(weights.data());
            check(cudaDeviceSynchronize(), "weight upload");
            original_packed = repack();
            break;
        default:
            break;
        }
        x.upload(values.data());
        sf.upload(ss.data());
        check(cudaDeviceSynchronize(), "fixture upload");
        check(launch_q8_tuned_gemv(x.as<float>(), w.as<int8_t>(), sf.as<float>(), full.as<float>(), rows,
                                   inner, 8, false, stream),
              "Q8 reference GEMV");
        check(cudaStreamSynchronize(stream), "reference");
        const std::vector<float> reference = full.download<float>();
        const int expected = argmax(full.as<float>());

        check(cudaMemsetAsync(output.data(), 0xa5, output.bytes(), stream), "poison output");
        launch();
        check(cudaStreamSynchronize(stream), "eager screen");
        const std::vector<float> eager = output.download<float>();
        check(cudaMemsetAsync(output.data(), 0xa5, output.bytes(), stream), "poison output");
        check(cudaGraphLaunch(executable, stream), "graph replay");
        check(cudaStreamSynchronize(stream), "graph screen");
        const std::vector<float> actual = output.download<float>();
        const std::vector<float> coarse = approximate.download<float>();
        require(mollm_test::same_bits(eager, actual), "changed-input graph equals eager");
        require(argmax(output.as<float>()) == expected, "original full-Q8 argmax");

        const std::vector<int> selected = ids.download<int>();
        const unsigned kept = count.download<unsigned>()[0];
        require(kept > 0 && kept <= unsigned(rows), "candidate count");
        std::vector<bool> included(rows, false);
        for (unsigned j = 0; j < kept; ++j) {
            const int row = selected[j];
            require(row >= 0 && row < rows && !included[row], "candidate uniqueness and bounds");
            included[row] = true;
            require(same_score(actual[row], reference[row]), "selected original-Q8 score bits");
        }
        double l1 = 0;
        for (float v : values)
            l1 += std::fabs(double(v));
        for (int row = 0; row < rows; ++row) {
            if (!included[row])
                require(actual[row] == -INFINITY, "discarded sentinel");
            const bool bounded = std::isfinite(coarse[row]) && std::isfinite(l1) && l1 <= 1e20 &&
                                 std::isfinite(ss[row]) && std::fabs(ss[row]) <= 1e10f;
            if (bounded) {
                const double bound = 7.51 * std::fabs(double(ss[row])) * l1 + 1e-5;
                require(std::fabs(double(coarse[row]) - reference[row]) <= bound,
                        "conservative interval bound");
            }
        }
        if (fixture == 15)
            require(expected == rows - 1 && kept == 1, "scale-disparity anchor regression");
        require(mollm_test::same_bits(x.download<float>(), values) &&
                    mollm_test::same_bits(sf.download<float>(), ss),
                "input/scales immutable");
        for (const GuardedBuffer* buffer : {&w, &packed, &sf, &x, &output, &full, &approximate, &lower,
                                            &norm, &count, &ids, &partial, &result})
            buffer->check_guards();
    }
    require(w.download<int8_t>() == weights && packed.download<uint8_t>() == original_packed,
            "weights immutable");
    check(cudaGraphExecDestroy(executable), "destroy graph exec");
    check(cudaGraphDestroy(graph), "destroy graph");

    // Every invalid call must fail before enqueueing work, so the capture
    // below has to end with an empty graph.
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    auto reject = [&](const float* input, float* out, int n, Q8HeadScreenScratch s) {
        const cudaError_t status = launch_q8_head_screen_q4(input, w.as<int8_t>(), packed.as<uint8_t>(),
                                                            sf.as<float>(), out, n, s, stream);
        require(status == cudaErrorInvalidValue, "invalid launch rejected");
    };
    reject(x.as<float>(), output.as<float>(), 511, scratch);
    reject(x.as<float>(), output.as<float>(), INT_MAX, scratch);
    reject(nullptr, output.as<float>(), rows, scratch);
    reject(x.as<float>() + 1, output.as<float>(), rows, scratch);
    reject(x.as<float>(), approximate.as<float>(), rows, scratch);
    reject(reinterpret_cast<const float*>(UINTPTR_MAX - 15), output.as<float>(), rows, scratch);
    Q8HeadScreenScratch invalid = scratch;
    invalid.norm = reinterpret_cast<double*>(norm.as<unsigned char>() + 4);
    reject(x.as<float>(), output.as<float>(), rows, invalid);
    invalid = scratch;
    invalid.indices = lower.as<int>();
    reject(x.as<float>(), output.as<float>(), rows, invalid);
    require(pack_q8_head_screen_q4(w.as<int8_t>(), w.as<uint8_t>(), rows, stream) == cudaErrorInvalidValue,
            "packing aliases rejected");
    require(pack_q8_head_screen_q4(w.as<int8_t>(), packed.as<uint8_t>(), 511, stream) ==
                cudaErrorInvalidValue,
            "packing geometry rejected");
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    size_t nodes = 1;
    check(cudaGraphGetNodes(graph, nullptr, &nodes), "graph nodes");
    require(nodes == 0, "rejections enqueue no kernels");
    check(cudaGraphDestroy(graph), "destroy graph");
    check(cudaStreamDestroy(stream), "destroy stream");
    std::printf("Q4 screen: rows=%d fixtures=%d native bits, argmax, bounds, graph, packing, guards, "
                "anchor and rejection PASS\n",
                rows, fixtures);
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
