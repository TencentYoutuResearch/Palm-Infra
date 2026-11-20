// Multi-query Q8 greedy head screen: launch_q8_head_screen_batch3 (width 3,
// the default) or launch_q8_head_screen_batch (width 4-8, first argument).
// Each fixture runs eagerly and as a graph replay next to the exact small-batch
// Q8 GEMM and the scalar screen per query. Retained scores must match the exact
// GEMM bit for bit, the batch outputs and approximations must equal the scalar
// screen, argmax must match, guards stay intact and weights, packed weights,
// inputs and scales stay unchanged. Invalid launches must be rejected.
#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/q8_head_screen.h"

#include "cuda_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

using mollm_cuda::ArgMaxPair;
using mollm_cuda::Q8HeadScreenBatch3Scratch;
using mollm_cuda::Q8HeadScreenBatchScratch;
using mollm_test::GuardedBuffer;
using mollm_test::check;
using mollm_test::require;

constexpr int inner = 2560;
constexpr int rows = 263;  // Exercises row-grid tails with distinct queries.
constexpr int fixture_count = 20;

uint32_t mix32(uint32_t v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    return v ^ (v >> 16);
}

bool same_score(float a, float b) {
    return (std::isnan(a) && std::isnan(b)) || mollm_test::to_bits(a) == mollm_test::to_bits(b);
}

// Guard bands intact and payload equal to the bytes that were uploaded.
template<class T>
void require_unchanged(const GuardedBuffer& buffer, const std::vector<T>& expected, const char* what) {
    buffer.check_guards();
    require(mollm_test::same_bits(buffer.download<T>(), expected), what);
}

void apply_edge_case(int fixture, std::vector<float>& input, std::vector<float>& scales) {
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float infinity = std::numeric_limits<float>::infinity();
    switch (fixture) {
    case 3:
        for (float& v : input)
            v *= 1e16f;
        break;
    case 4:
        for (float& v : input)
            v *= 1e20f;
        break;
    case 5:
        for (float& v : input)
            v *= 1e-40f;
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
        input[inner + 17] = infinity;
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
    case 15:
        for (float& v : input)
            v *= 1e-40f;
        std::fill(scales.begin(), scales.end(), 1e10f);
        break;
    case 16:
        for (float& v : input)
            v *= 1e-30f;
        std::fill(scales.begin(), scales.end(), 1e-10f);
        break;
    case 17:
        for (float& v : input)
            v *= 1e18f;
        std::fill(scales.begin(), scales.end(), 1e-40f);
        break;
    case 18:
    case 19: {
        // Alternating-sign input just below (18) and just above (19) the
        // 1e20 L1 limit with tiny negative scales.
        const float magnitude = (1e20f / 2560) * (fixture == 18 ? .999f : 1.001f);
        for (size_t j = 0; j < input.size(); ++j)
            input[j] = (j % 2 ? -1.f : 1.f) * magnitude;
        std::fill(scales.begin(), scales.end(), -1e-20f);
        break;
    }
    default:
        break;
    }
}

int run(int m) {
    std::vector<unsigned char> weights(size_t(rows) * inner);
    std::vector<float> scales(rows);
    for (size_t i = 0; i < weights.size(); ++i) {
        const size_t j = i / inner == 1 ? i - inner : i;
        weights[i] = mix32(uint32_t(j)) & 255;
    }
    for (int i = 0; i < rows; ++i)
        scales[i] = i % 13 == 0 ? -.001f : .001f;
    scales[0] = scales[1] = .001f;  // Equal winning rows for the correlation fixture.

    GuardedBuffer w(weights.size());
    GuardedBuffer packed(size_t(rows) * 1920);
    GuardedBuffer sf(rows * sizeof(float));
    GuardedBuffer x(size_t(m) * inner * sizeof(float));
    GuardedBuffer reference(size_t(m) * rows * sizeof(float));
    GuardedBuffer output(size_t(m) * rows * sizeof(float));
    GuardedBuffer scalar(size_t(m) * rows * sizeof(float));
    GuardedBuffer approx(size_t(m) * rows * sizeof(float));
    GuardedBuffer lower(size_t(m) * 257 * sizeof(float));
    GuardedBuffer norm(size_t(m) * sizeof(double));
    GuardedBuffer count(size_t(m) * sizeof(unsigned));
    GuardedBuffer indices(size_t(m) * rows * sizeof(int));
    GuardedBuffer scalar_approx(size_t(m) * rows * sizeof(float));
    GuardedBuffer scalar_lower(size_t(m) * 257 * sizeof(float));
    GuardedBuffer scalar_norm(size_t(m) * sizeof(double));
    GuardedBuffer scalar_count(size_t(m) * sizeof(unsigned));
    GuardedBuffer scalar_indices(size_t(m) * rows * sizeof(int));
    GuardedBuffer partial(256 * sizeof(ArgMaxPair));
    GuardedBuffer result(sizeof(ArgMaxPair));
    w.upload(weights.data());
    sf.upload(scales.data());
    check(cudaDeviceSynchronize(), "initial upload");

    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    const int8_t* weight_codes = w.as<int8_t>();
    uint8_t* packed_codes = packed.as<uint8_t>();
    check(mollm_cuda::pack_q8_head_screen(weight_codes, packed_codes, rows, stream), "pack");
    check(cudaStreamSynchronize(stream), "pack");
    const std::vector<unsigned char> packed_original = packed.download<unsigned char>();

    const Q8HeadScreenBatch3Scratch scratch{approx.as<float>(), lower.as<float>(), norm.as<double>(),
                                            count.as<unsigned>(), indices.as<int>()};
    auto wide_scratch = [&](const Q8HeadScreenBatch3Scratch& work) {
        return Q8HeadScreenBatchScratch{work.approximate, work.lower, work.norm, work.count, work.indices,
                                        size_t(m) * rows, size_t(m) * 257, size_t(m), size_t(m),
                                        size_t(m) * rows};
    };
    auto screen = [&](const float* source, float* target, int screen_rows, Q8HeadScreenBatch3Scratch work) {
        if (m == 3) {
            return mollm_cuda::launch_q8_head_screen_batch3(
                source, weight_codes, packed_codes, sf.as<float>(), target, screen_rows, work, stream);
        }
        return mollm_cuda::launch_q8_head_screen_batch(source, weight_codes, packed_codes, sf.as<float>(),
                                                       target, m, screen_rows, wide_scratch(work), stream);
    };
    // Mode 0: exact small-batch Q8 GEMM. Mode 1: batch screen. Mode 2: the
    // scalar screen once per query.
    auto launch = [&](int mode) {
        if (mode == 0) {
            check(mollm_cuda::launch_q8_small_batch_exact(x.as<float>(), weight_codes, sf.as<float>(),
                                                          reference.as<float>(), m, rows, inner, inner, rows,
                                                          stream),
                  "exact small-batch Q8");
        } else if (mode == 1) {
            check(screen(x.as<float>(), output.as<float>(), rows, scratch), "batch screen");
        } else {
            for (int t = 0; t < m; ++t) {
                const mollm_cuda::Q8HeadScreenScratch query_scratch{
                    scalar_approx.as<float>() + t * rows, scalar_lower.as<float>() + t * 257,
                    scalar_norm.as<double>() + t, scalar_count.as<unsigned>() + t,
                    scalar_indices.as<int>() + t * rows};
                check(mollm_cuda::launch_q8_head_screen(x.as<float>() + t * inner, weight_codes, packed_codes,
                                                        sf.as<float>(), scalar.as<float>() + t * rows, rows,
                                                        query_scratch, stream),
                      "scalar screen");
            }
        }
    };

    auto reject_batch3 = [&](const float* input, const uint8_t* codes, int screen_rows,
                             Q8HeadScreenBatch3Scratch work) {
        const cudaError_t status = mollm_cuda::launch_q8_head_screen_batch3(
            input, weight_codes, codes, sf.as<float>(), output.as<float>(), screen_rows, work, stream);
        require(status == cudaErrorInvalidValue, "invalid batch3 launch rejected");
    };
    reject_batch3(x.as<float>(), packed_codes, 0, scratch);
    reject_batch3(x.as<float>() + 1, packed_codes, rows, scratch);
    reject_batch3(x.as<float>(), packed_codes + 1, rows, scratch);
    Q8HeadScreenBatch3Scratch incomplete = scratch;
    incomplete.lower = nullptr;
    reject_batch3(x.as<float>(), packed_codes, rows, incomplete);
    if (m != 3) {
        const Q8HeadScreenBatchScratch wide = wide_scratch(scratch);
        auto reject = [&](const float* input, const uint8_t* codes, int queries, int screen_rows,
                          Q8HeadScreenBatchScratch work) {
            const cudaError_t status = mollm_cuda::launch_q8_head_screen_batch(
                input, weight_codes, codes, sf.as<float>(), output.as<float>(), queries, screen_rows, work,
                stream);
            require(status == cudaErrorInvalidValue, "invalid batch launch rejected");
        };
        for (int width : {0, 1, 2, 3, 9, 16})
            reject(x.as<float>(), packed_codes, width, rows, wide);
        reject(x.as<float>(), packed_codes, m, 0, wide);
        reject(x.as<float>(), packed_codes, m, INT32_MAX, wide);
        for (size_t Q8HeadScreenBatchScratch::*capacity :
             {&Q8HeadScreenBatchScratch::approximate_capacity, &Q8HeadScreenBatchScratch::lower_capacity,
              &Q8HeadScreenBatchScratch::norm_capacity, &Q8HeadScreenBatchScratch::count_capacity,
              &Q8HeadScreenBatchScratch::indices_capacity}) {
            Q8HeadScreenBatchScratch bad = wide;
            --(bad.*capacity);
            reject(x.as<float>(), packed_codes, m, rows, bad);
        }
        Q8HeadScreenBatchScratch bad = wide;
        bad.norm = reinterpret_cast<double*>(reinterpret_cast<char*>(wide.norm) + 4);
        reject(x.as<float>(), packed_codes, m, rows, bad);
        bad = wide;
        bad.lower = nullptr;
        reject(x.as<float>(), packed_codes, m, rows, bad);
        reject(x.as<float>() + 1, packed_codes, m, rows, wide);
        reject(x.as<float>(), packed_codes + 1, m, rows, wide);
    }

    auto argmax = [&](const float* values) {
        mollm_cuda::launch_argmax(values, rows, partial.as<ArgMaxPair>(), 256, result.as<ArgMaxPair>());
        check(cudaGetLastError(), "argmax");
        return result.download<ArgMaxPair>()[0].index;
    };

    cudaGraph_t graphs[3];
    cudaGraphExec_t executable[3];
    for (int mode = 0; mode < 3; ++mode) {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        launch(mode);
        check(cudaStreamEndCapture(stream, &graphs[mode]), "end capture");
        check(cudaGraphInstantiate(&executable[mode], graphs[mode], nullptr, nullptr, 0), "instantiate");
    }

    for (int fixture = 0; fixture < fixture_count; ++fixture) {
        std::vector<float> input(size_t(m) * inner);
        std::vector<float> current_scales = scales;
        for (int t = 0; t < m; ++t) {
            for (int j = 0; j < inner; ++j)
                input[t * inner + j] = std::sin(j * .31f + t * .07f);
        }
        if (fixture == 0) {
            std::fill(input.begin(), input.end(), 0.f);
        } else if (fixture == 1) {
            for (int t = 0; t < m; ++t) {
                for (int j = 0; j < inner; ++j)
                    input[t * inner + j] = float(int8_t(weights[j])) / 128.f;
            }
        } else {
            apply_edge_case(fixture, input, current_scales);
        }
        x.upload(input.data());
        sf.upload(current_scales.data());
        check(cudaDeviceSynchronize(), "fixture upload");

        // Phase 0 launches eagerly, phase 1 replays the captured graphs.
        for (int phase = 0; phase < 2; ++phase) {
            for (int mode = 0; mode < 3; ++mode) {
                if (phase)
                    check(cudaGraphLaunch(executable[mode], stream), "graph replay");
                else
                    launch(mode);
            }
            check(cudaStreamSynchronize(stream), "fixture");
            const std::vector<float> ref = reference.download<float>();
            const std::vector<float> got = output.download<float>();
            const std::vector<float> old = scalar.download<float>();
            const std::vector<float> ap = approx.download<float>();
            const std::vector<float> old_ap = scalar_approx.download<float>();
            const std::vector<unsigned> retained = count.download<unsigned>();
            const std::vector<unsigned> old_count = scalar_count.download<unsigned>();
            const std::vector<int> selected = indices.download<int>();
            require(retained == old_count, "batch and scalar retained counts match");
            for (int t = 0; t < m; ++t) {
                require(retained[t] > 0 && retained[t] <= unsigned(rows), "retained candidate count");
                std::vector<bool> seen(rows);
                for (unsigned j = 0; j < retained[t]; ++j) {
                    const int row = selected[t * rows + j];
                    require(row >= 0 && row < rows && !seen[row], "candidate uniqueness and bounds");
                    seen[row] = true;
                    require(same_score(ref[t * rows + row], got[t * rows + row]),
                            "retained score differs from exact Q8");
                }
                for (int row = 0; row < rows; ++row) {
                    require(same_score(ap[t * rows + row], old_ap[t * rows + row]),
                            "batch approximation differs from scalar screen");
                    require(same_score(old[t * rows + row], got[t * rows + row]),
                            "batch output differs from scalar screen");
                }
                require(argmax(reference.as<float>() + t * rows) == argmax(output.as<float>() + t * rows),
                        "screened argmax differs from exact Q8");
                if (fixture == 0 || fixture == 1)
                    require(argmax(output.as<float>() + t * rows) == 0, "tie selects the first row");
            }
            for (const GuardedBuffer* buffer :
                 {&output, &reference, &scalar, &approx, &lower, &norm, &count, &indices, &scalar_approx,
                  &scalar_lower, &scalar_norm, &scalar_count, &scalar_indices})
                buffer->check_guards();
            require_unchanged(w, weights, "weights immutable");
            require_unchanged(packed, packed_original, "packed weights immutable");
            require_unchanged(x, input, "input immutable");
            require_unchanged(sf, current_scales, "scales immutable");
        }
        std::printf("batch_screen_case frame=%d eager_graph=1 retained_bits_exact=1 "
                    "scalar_screen_equivalent=1 target_argmax_exact=1 guards=1\n",
                    fixture);
    }
    for (int mode = 0; mode < 3; ++mode) {
        check(cudaGraphExecDestroy(executable[mode]), "destroy graph exec");
        check(cudaGraphDestroy(graphs[mode]), "destroy graph");
    }
    check(cudaStreamDestroy(stream), "destroy stream");
    std::printf("batch_screen_all cases=%d rows=%d width=%d all_exact_argmax=1 immutable=1 guards=1\n",
                fixture_count, rows, m);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const int m = argc == 2 ? std::atoi(argv[1]) : 3;
    if (argc > 2 || m < 3 || m > 8) {
        std::fprintf(stderr, "usage: %s [query-width 3..8]\n", argv[0]);
        return 2;
    }
    if (!mollm_test::cuda_device_available())
        return mollm_test::skip_status;
    return run(m);
}
