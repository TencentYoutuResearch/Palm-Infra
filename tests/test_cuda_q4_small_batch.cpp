#include "kernels/cuda/matmul.h"
#include "backends/cuda/backend.h"
#include "graph/graph.h"

#include "cuda_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

namespace {

using mollm_test::check;
using mollm_test::GuardedBuffer;
using mollm_test::same_bits;

// Average microseconds per call over 100 calls, after 16 warm-up calls.
template<class F> double measure(F f) {
    for (int i = 0; i < 16; ++i)
        f();
    cudaEvent_t start;
    cudaEvent_t stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    for (int i = 0; i < 100; ++i)
        f();
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float ms;
    check(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms * 10;
}

enum class BatchKernel { plain, wide, reduced, exact };

// Dispatches the projection through CudaBackend and compares it with the
// reference within the batch kernels' FP32 tolerance.
bool backend_matches(const std::vector<Q4B8G32Block>& weights, const std::vector<float>& host_input,
                     const std::vector<float>& reference, int m, int n, int k, int ldc) {
    CudaBackend backend;
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    auto raw = weights;
    for (auto& block : raw) {
        for (auto& row : block.q) {
            for (auto& byte : row)
                byte ^= 0x88;
        }
    }
    Tensor weight = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, n, k, 1, 1, raw.data());
    weight.rowmajor_data = raw.data();
    weight.q4_g32_data = raw.data();
    weight.is_q4_g32_packed = true;
    weight.group_size = 32;
    weight.groups_per_row = k / 32;
    backend.wrap_weight_int4(weight);
    Tensor input = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, k, m);
    Tensor output = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, n, m);
    backend.alloc_persistent(input, input.nbytes(), PersistentHostAccess::NONE);
    backend.alloc_persistent(output, output.nbytes(), PersistentHostAccess::NONE);
    if (!backend.copy_from_host(host_input.data(), input, input.nbytes())) return false;
    GraphNode node;
    node.op_type = OpType::MATMUL;
    backend.dispatch(node, {&input, &weight}, &output, nullptr);
    std::vector<float> got(static_cast<size_t>(m) * n);
    if (backend.dispatch_failed() || !backend.copy_to_host(output, got.data(), output.nbytes())) return false;
    for (int t = 0; t < m; ++t) {
        for (int r = 0; r < n; ++r) {
            const float expected = reference[t * ldc + r];
            if (std::fabs(got[t * n + r] - expected) > 2e-5f * (1 + std::fabs(expected))) return false;
        }
    }
    return true;
}

// Checks one batched BG32 kernel against per-token GEMV (the reduced kernel
// against the plain batch kernel) through a changed-input graph replay. With
// `timing`, large shapes are also checked through CudaBackend, and timings of
// per-token GEMV, the plain batch kernel, this kernel and dequantize+cuBLAS
// are printed.
bool batch_fixture(BatchKernel kernel, int m, int n, int k, bool timing = false, int stride_padding = 4,
                   int alignment_offset = 0) {
    const bool wide = kernel == BatchKernel::wide;
    const bool reduced = kernel == BatchKernel::reduced;
    const bool exact_reuse = kernel == BatchKernel::exact;
    const int lda = k + stride_padding;
    const int ldc = n + 9;
    std::vector<Q4B8G32Block> weights(static_cast<size_t>((n + 7) / 8) * (k / 32));
    for (size_t b = 0; b < weights.size(); ++b) {
        for (int r = 0; r < 8; ++r) {
            weights[b].scales[r] = 0.01317f * (1 + (b + r) % 7);
            for (int p = 0; p < 16; ++p)
                weights[b].q[r][p] = ((b * 7 + r * 3 + p) % 16) | (((b * 3 + r + p * 5) % 16) << 4);
        }
    }
    std::vector<float> x(m * lda, 999);
    std::vector<float> reference(m * ldc, -123);
    std::vector<float> actual = reference;
    for (int t = 0; t < m; ++t) {
        for (int c = 0; c < k; ++c)
            x[t * lda + c] = ((c * 13 + t * 17) % 101 - 50) / 137.f;
    }
    GuardedBuffer weight_buffer(weights.size() * sizeof(Q4B8G32Block));
    GuardedBuffer input_buffer((x.size() + alignment_offset) * sizeof(float));
    GuardedBuffer output_buffer(actual.size() * sizeof(float));
    auto* w = weight_buffer.as<Q4B8G32Block>();
    float* a = input_buffer.as<float>() + alignment_offset;
    float* y = output_buffer.as<float>();
    weight_buffer.upload(weights.data());
    check(cudaMemcpy(a, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    output_buffer.upload(reference.data());

    const auto serial = [&] {
        for (int t = 0; t < m; ++t)
            mollm_cuda::launch_q4_g32_dense_gemv(a + t * lda, w, y + t * ldc, n, k);
    };
    const auto batch = [&](cudaStream_t stream = nullptr) {
        auto launch = exact_reuse ? mollm_cuda::launch_q4_g32_small_batch_exact
                      : reduced   ? mollm_cuda::launch_q4_g32_small_batch_reduced
                      : wide      ? mollm_cuda::launch_q4_g32_small_batch_wide
                                  : mollm_cuda::launch_q4_g32_small_batch;
        check(launch(a, w, y, m, n, k, lda, ldc, stream));
    };
    const auto baseline = [&] {
        if (reduced)
            check(mollm_cuda::launch_q4_g32_small_batch(a, w, y, m, n, k, lda, ldc));
        else
            serial();
    };
    baseline();
    check(cudaMemcpy(reference.data(), y, reference.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    batch(stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    // These shapes must reproduce the baseline bit for bit; others allow a
    // small FP32 tolerance outside the output padding.
    const bool bit_exact_shape = (!wide && n <= 4096 && k >= 2048) || (wide && n >= 9216 && k >= 2048);
    for (int replay = 0; replay < 2; ++replay) {
        if (replay) {
            for (float& value : x)
                value = -value;
            check(cudaMemcpy(a, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
            baseline();
            check(cudaMemcpy(reference.data(), y, reference.size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
        std::fill(actual.begin(), actual.end(), -123.f);
        // Nonblocking graph streams do not wait for the default stream's
        // pageable H2D reset. Queue the sentinel on the graph stream itself.
        check(cudaMemcpyAsync(y, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice,
                              stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(), y, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < actual.size(); ++i) {
            const bool output_padding = i % ldc >= static_cast<size_t>(n);
            const float tolerance = output_padding ? 0 : 2e-5f * (1 + std::fabs(reference[i]));
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - reference[i]) > tolerance ||
                (bit_exact_shape && actual[i] != reference[i])) {
                std::fprintf(stderr,
                             "mismatch M=%d N=%d K=%d exact=%d padding=%d offset=%d replay=%d i=%zu "
                             "%g != %g\n",
                             m, n, k, exact_reuse, stride_padding, alignment_offset, replay, i, actual[i],
                             reference[i]);
                return false;
            }
            if ((reduced || exact_reuse) && actual[i] != reference[i]) return false;
        }
    }

    if (timing) {
        GuardedBuffer dense_weight_buffer(static_cast<size_t>(n) * k * sizeof(__half));
        GuardedBuffer dense_input_buffer(static_cast<size_t>(m) * k * sizeof(__half));
        GuardedBuffer dense_output_buffer(static_cast<size_t>(m) * n * sizeof(float));
        GuardedBuffer packed_input_buffer(static_cast<size_t>(m) * k * sizeof(float));
        auto* d = dense_weight_buffer.as<__half>();
        auto* h = dense_input_buffer.as<__half>();
        float* c = dense_output_buffer.as<float>();
        float* ax = packed_input_buffer.as<float>();
        std::vector<float> host(static_cast<size_t>(m) * k);
        for (int t = 0; t < m; ++t)
            std::copy_n(x.data() + t * lda, k, host.data() + t * k);
        packed_input_buffer.upload(host.data());
        if (k >= 2048 && static_cast<int64_t>(n) * k >= 4 * 1024 * 1024 &&
            !backend_matches(weights, host, reference, m, n, k, ldc))
            return false;
        cublasHandle_t handle;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
        if (cublasSetStream(handle, cudaStreamPerThread) != CUBLAS_STATUS_SUCCESS) return false;
#endif
        const auto gemm = [&] {
            mollm_cuda::launch_prepare_q4_g32_dense_gemm(w, reinterpret_cast<__half2*>(d), ax, h,
                                                         static_cast<size_t>(m) * k, n, k / 32);
            if (!mollm_cuda::run_dense_matmul(handle, d, CUDA_R_16F, h, CUDA_R_16F, c, m, n, k, k))
                mollm_test::fail("dense GEMM failed");
        };
        const auto plain_batch = [&] {
            check(mollm_cuda::launch_q4_g32_small_batch(a, w, y, m, n, k, lda, ldc));
        };
        const double serial_us = measure(serial);
        const double plain_batch_us = measure(plain_batch);
        const double batch_us = measure(batch);
        const double gemm_us = measure(gemm);
        std::printf("wide=%d reduced=%d exact_reuse=%d M=%d N=%d K=%d serial_us=%.2f baseline_batch_us=%.2f "
                    "batch_us=%.2f expand_gemm_us=%.2f\n",
                    wide, reduced, exact_reuse, m, n, k, serial_us, plain_batch_us, batch_us, gemm_us);
        cublasDestroy(handle);
    }
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return true;
}

// Checks the compact (BF16-scale) small-batch kernel, or the cache probe when
// `cache_hint` is set, against per-token compact GEMV through a changed-input
// graph replay. special_values 1 sweeps every BF16 scale code and 2 feeds
// nonfinite activations; both use the original exact kernel as reference.
bool compact_fixture(int m, int n, int padding, bool integer_scale, int cache_hint = 0, bool fixed_m5 = false,
                     int special_values = 0, int output_padding = 9) {
    constexpr int k = 2560;
    const int lda = k + padding;
    const int ldc = n + output_padding;
    std::vector<mollm_cuda::Q4CompactG32> weights(static_cast<size_t>((n + 7) / 8) * (k / 32));
    // Include zero, subnormal maxima, and both normal decoder branches.
    const uint16_t codes[] = {0, 1, 0x0080, 0x0180, 0x3f80, 0x3fe0, 0x4001, 0x417f};
    for (size_t b = 0; b < weights.size(); ++b) {
        for (int r = 0; r < 8; ++r) {
            weights[b].scales[r] =
                special_values == 1 ? static_cast<uint16_t>(b * 8 + r) : codes[(b + r) % 8];
            for (int p = 0; p < 16; ++p)
                weights[b].q[r][p] = ((b + r + p) % 16) | (((b * 3 + r + p * 5) % 16) << 4);
        }
    }
    std::vector<float> x(m * lda, 999.f);
    std::vector<float> reference(m * ldc + 16, -123.f);
    std::vector<float> actual = reference;
    for (int t = 0; t < m; ++t) {
        for (int c = 0; c < k; ++c)
            x[t * lda + c] = ((c * 13 + t * 17) % 101 - 50) / 137.f;
    }
    if (special_values == 2) {
        // Nonfinite activations use ordinary finite maxima. Separately,
        // mode 1 covers all BF16 maxima with finite activations.
        const uint32_t bits[] = {0, 0x80000000u, 1, 0x80000001u, 0x3f800000u, 0x7f800000u, 0xff800000u,
                                 0x7fc12345u};
        for (int t = 0; t < m; ++t) {
            for (int c = 0; c < k; ++c)
                std::memcpy(&x[t * lda + c], &bits[(c + t) % 8], sizeof(float));
        }
    }
    GuardedBuffer weight_buffer(weights.size() * sizeof(mollm_cuda::Q4CompactG32));
    GuardedBuffer input_buffer(x.size() * sizeof(float));
    GuardedBuffer output_buffer(actual.size() * sizeof(float));
    auto* w = weight_buffer.as<mollm_cuda::Q4CompactG32>();
    float* a = input_buffer.as<float>();
    float* y = output_buffer.as<float>();
    // A padded output forces the original M3/M4 path for an independent
    // nonfinite/scale oracle while the contiguous result can use prefetch.
    std::optional<GuardedBuffer> special_reference_buffer;
    if (special_values && (m == 3 || m == 4 || m == 6) && output_padding == 0)
        special_reference_buffer.emplace(size_t(m) * (n + 9) * sizeof(float));
    float* special_reference = special_reference_buffer ? special_reference_buffer->as<float>() : nullptr;
    weight_buffer.upload(weights.data());
    input_buffer.upload(x.data());
    if (cache_hint) {
        output_buffer.upload(actual.data());
        const auto rejected = [&](const float* input, int columns, int inner, int hint) {
            return mollm_cuda::launch_q4_compact_cache_probe(input, w, y, columns, inner, integer_scale,
                                                             hint) == cudaErrorInvalidValue;
        };
        if (!rejected(a + 1, n, k, cache_hint) || !rejected(a, 4096, k, cache_hint) ||
            !rejected(a, n, 4097, cache_hint) || !rejected(a, n, k, 10)) {
            std::fprintf(stderr, "compact cache probe invalid argument accepted N=%d integer=%d hint=%d\n", n,
                         integer_scale, cache_hint);
            return false;
        }
        actual = output_buffer.download<float>();
        if (!same_bits(actual, reference)) return false;
    }
    const auto serial = [&] {
        for (int t = 0; t < m; ++t) {
            if (integer_scale)
                check(mollm_cuda::launch_q4_compact_integer_gemv(a + t * lda, w, y + t * ldc, n, k));
            else
                mollm_cuda::launch_q4_g32_compact_gemv(a + t * lda, w, y + t * ldc, n, k);
        }
    };
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    if (cache_hint) {
        check(mollm_cuda::launch_q4_compact_cache_probe(a, w, y, n, k, integer_scale, cache_hint, stream));
    } else {
        check(mollm_cuda::launch_q4_compact_small_batch_exact(a, w, y, m, n, k, lda, ldc, integer_scale,
                                                              stream, fixed_m5));
    }
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
        if (replay) {
            for (float& value : x)
                value = -value;
        }
        input_buffer.upload(x.data());
        std::fill(reference.begin(), reference.end(), -123.f);
        output_buffer.upload(reference.data());
        if (special_reference) {
            check(mollm_cuda::launch_q4_compact_small_batch_exact(a, w, special_reference, m, n, k, lda,
                                                                  n + 9, integer_scale, nullptr, false));
            check(cudaMemcpy2D(y, size_t(ldc) * sizeof(float), special_reference,
                               size_t(n + 9) * sizeof(float), size_t(n) * sizeof(float), m,
                               cudaMemcpyDeviceToDevice));
        } else if (special_values) {
            check(mollm_cuda::launch_q4_compact_small_batch_exact(a, w, y, m, n, k, lda, ldc, integer_scale,
                                                                  nullptr, false));
        } else {
            serial();
        }
        reference = output_buffer.download<float>();
        std::fill(actual.begin(), actual.end(), -123.f);
        check(cudaMemcpyAsync(y, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice,
                              stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        actual = output_buffer.download<float>();
        for (size_t i = 0; i < actual.size(); ++i) {
            if ((!special_values && !std::isfinite(actual[i])) ||
                std::memcmp(&actual[i], &reference[i], sizeof(float))) {
                std::fprintf(stderr,
                             "compact mismatch M=%d N=%d padding=%d integer=%d replay=%d i=%zu %g != %g\n", m,
                             n, padding, integer_scale, replay, i, actual[i], reference[i]);
                return false;
            }
        }
    }
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return true;
}

// Compares the vector-load dispatch with the independent serial GEMV for each
// token. Keep all finite bits and signed zeros; compare NaNs by classification.
bool narrow_vector_fixture(int m, int n, int k, int padding, int offset, int special) {
    const int lda = k + padding;
    const int ldc = n + 9;
    const int guard = 8;
    const size_t input_start = guard + offset;
    std::vector<Q4B8G32Block> weights(static_cast<size_t>((n + 7) / 8) * (k / 32));
    const uint32_t scale_bits[] = {0, 0x80000000u, 1, 0x00800000u, 0x3f000000u, 0x7f800000u, 0xff800000u,
                                   0x7fc12345u};
    for (size_t b = 0; b < weights.size(); ++b) {
        for (int r = 0; r < 8; ++r) {
            const uint32_t bits = scale_bits[(b + r) % 8];
            if (special == 2)
                std::memcpy(&weights[b].scales[r], &bits, sizeof(float));
            else
                weights[b].scales[r] = 0.01317f * (1 + (b + r) % 7);
            for (int pair = 0; pair < 16; ++pair)
                weights[b].q[r][pair] = ((b + r + pair) % 16) | (((b * 3 + r + pair * 5) % 16) << 4);
        }
    }
    std::vector<float> input(input_start + static_cast<size_t>(m) * lda + guard, -123.f);
    const uint32_t input_bits[] = {0, 0x80000000u, 1, 0x80000001u, 0x3f800000u, 0x7f800000u, 0xff800000u,
                                   0x7fc23456u};
    for (int t = 0; t < m; ++t) {
        for (int c = 0; c < k; ++c) {
            float& value = input[input_start + static_cast<size_t>(t) * lda + c];
            if (special == 1) {
                const uint32_t bits = input_bits[(c + t) % 8];
                std::memcpy(&value, &bits, sizeof(float));
            } else {
                value = ((c * 13 + t * 17) % 101 - 50) / 137.f;
            }
        }
    }
    const size_t output_size = guard + static_cast<size_t>(m) * ldc + guard;
    const std::vector<float> sentinel(output_size, -123.f);
    GuardedBuffer weight_buffer(weights.size() * sizeof(Q4B8G32Block));
    GuardedBuffer input_buffer(input.size() * sizeof(float));
    GuardedBuffer reference_buffer(output_size * sizeof(float));
    GuardedBuffer output_buffer(output_size * sizeof(float));
    auto* w = weight_buffer.as<Q4B8G32Block>();
    float* x = input_buffer.as<float>();
    float* reference = reference_buffer.as<float>();
    float* output = output_buffer.as<float>();
    weight_buffer.upload(weights.data());
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    check(mollm_cuda::launch_q4_g32_small_batch_exact(x + input_start, w, output + guard, m, n, k, lda, ldc,
                                                      stream));
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    bool passed = true;
    for (int replay = 0; replay < 2 && passed; ++replay) {
        if (replay) {
            for (int t = 0; t < m; ++t) {
                for (int c = 0; c < k; ++c) {
                    float& value = input[input_start + static_cast<size_t>(t) * lda + c];
                    value = -value;
                }
            }
        }
        input_buffer.upload(input.data());
        reference_buffer.upload(sentinel.data());
        for (int t = 0; t < m; ++t) {
            mollm_cuda::launch_q4_g32_dense_gemv(x + input_start + static_cast<size_t>(t) * lda, w,
                                                 reference + guard + static_cast<size_t>(t) * ldc, n, k);
        }
        check(cudaGetLastError());
        const std::vector<float> expected = reference_buffer.download<float>();
        check(cudaMemcpyAsync(output, sentinel.data(), output_size * sizeof(float), cudaMemcpyHostToDevice,
                              stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        const std::vector<float> actual = output_buffer.download<float>();
        for (size_t i = 0; i < output_size; ++i) {
            if (std::isnan(actual[i]) && std::isnan(expected[i])) continue;
            if (std::memcmp(&actual[i], &expected[i], sizeof(float))) {
                std::fprintf(stderr,
                             "narrow vector mismatch M=%d N=%d K=%d padding=%d offset=%d special=%d "
                             "replay=%d index=%zu\n",
                             m, n, k, padding, offset, special, replay, i);
                passed = false;
                break;
            }
        }
        if (!same_bits(input_buffer.download<float>(), input) ||
            !same_bits(weight_buffer.download<Q4B8G32Block>(), weights))
            passed = false;
    }
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return passed;
}

bool narrow_vector_fixtures() {
    int fixtures = 0;
    for (int m : {4, 5, 6, 8}) {
        for (int padding : {0, 2, 3}) {
            for (int offset : {0, 1}) {
                for (int special : {0, 1, 2}) {
                    if (!narrow_vector_fixture(m, 2560, 6144, padding, offset, special)) return false;
                    ++fixtures;
                }
            }
        }
    }
    for (auto shape : {std::pair<int, int>{2552, 6144}, {2560, 6112}}) {
        if (!narrow_vector_fixture(5, shape.first, shape.second, 0, 0, 0)) return false;
        ++fixtures;
    }
    std::printf("narrow Q4 vector: %d independent finite/nonfinite, stride/alignment/fallback, "
                "mutable graph, guard and immutable-weight fixtures passed\n",
                fixtures);
    return true;
}

// Reduced fixture set for compute-sanitizer runs of the small-reduction shapes.
bool small_reduction_sanitizer_fixtures() {
    int fixtures = 0;
    for (int m : {5, 6}) {
        for (auto shape : {std::pair<int, int>{320, 10240}, {640, 2560}, {512, 2560}}) {
            for (int special : {0, 1}) {
                if (!narrow_vector_fixture(m, shape.first, shape.second, 2, 0, special)) return false;
                ++fixtures;
            }
        }
    }
    std::printf("small Q4 reduction: %d sanitizer fixtures passed\n", fixtures);
    return true;
}

bool small_reduction_fixtures() {
    int fixtures = 0;
    for (int m : {2, 3, 4, 5, 6, 7, 8}) {
        for (auto shape : {std::pair<int, int>{320, 10240}, {640, 2560}, {512, 2560}}) {
            for (int padding : {0, 2, 3}) {
                for (int offset : {0, 1}) {
                    for (int special : {0, 1, 2}) {
                        if (!narrow_vector_fixture(m, shape.first, shape.second, padding, offset, special))
                            return false;
                        ++fixtures;
                    }
                }
            }
        }
    }
    for (auto shape : {std::pair<int, int>{312, 10240}, {320, 10208}, {632, 2560}, {504, 2560}}) {
        for (int m : {5, 6}) {
            if (!narrow_vector_fixture(m, shape.first, shape.second, 0, 0, 0)) return false;
            ++fixtures;
        }
    }
    std::printf("small Q4 reduction: %d serial-reference finite/nonfinite, stride/alignment, neighboring "
                "shape/width, mutable graph and guard fixtures passed\n",
                fixtures);
    return true;
}

bool m6_fixtures() {
    if (!narrow_vector_fixtures()) return false;
    int fixtures = 0;
    for (int n : {13312, 16480}) {
        for (int padding : {0, 4, 3}) {
            for (bool integer : {false, true}) {
                for (int special : {0, 1, 2}) {
                    if (!compact_fixture(6, n, padding, integer, 0, false, special, 0)) return false;
                    ++fixtures;
                }
            }
        }
    }
    std::printf("M6 Q4: %d compact fixtures (exhaustive SF/nonfinite, padded-original reference), graphs and "
                "guards PASS\n",
                fixtures);
    return true;
}

bool compact_fixed_m5_fixtures() {
    int finite = 0;
    for (int m : {2, 3, 4, 5, 6, 7, 8}) {
        for (int n : {13312, 16480}) {
            for (int padding : {0, 3, 4}) {
                if (!compact_fixture(m, n, padding, n == 13312, 0, true)) return false;
                ++finite;
            }
        }
    }
    int special = 0;
    for (int n : {13312, 16480}) {
        for (int padding : {0, 3, 4}) {
            for (bool integer : {false, true}) {
                for (int special_values : {1, 2}) {
                    if (!compact_fixture(5, n, padding, integer, 0, true, special_values)) return false;
                    ++special;
                }
            }
        }
    }
    std::printf("compact fixed-M5: %d finite and %d exhaustive-SF/nonfinite fixtures passed\n", finite,
                special);
    return true;
}

bool compact_prefetch_fixtures() {
    for (int m : {3, 4}) {
        int fixtures = 0;
        for (int n : {13312, 16480}) {
            for (int padding : {0, 4}) {
                for (bool integer : {false, true}) {
                    for (int special : {0, 1, 2}) {
                        if (!compact_fixture(m, n, padding, integer, 0, false, special, 0)) return false;
                        ++fixtures;
                    }
                }
            }
        }
        std::printf("compact M%d prefetch: %d contiguous/strided, finite/exhaustive-SF/nonfinite "
                    "fixtures passed\n",
                    m, fixtures);
    }
    return true;
}

// Exact kernel over every token count with strided outputs, changed-input
// graph replay, and the unaligned/odd-stride fallbacks around token-parallel
// dispatch, for the given main shapes and nearby fallback shapes.
bool exact_shape_fixtures(const std::vector<std::pair<int, int>>& shapes,
                          const std::vector<std::pair<int, int>>& fallback_shapes) {
    for (int m : {2, 3, 4, 5, 6, 7, 8}) {
        for (const auto& shape : shapes) {
            for (int padding : {0, 3, 4}) {
                if (!batch_fixture(BatchKernel::exact, m, shape.first, shape.second, false, padding))
                    return false;
            }
            if (!batch_fixture(BatchKernel::exact, m, shape.first, shape.second, false, 4, 1)) return false;
        }
    }
    for (int m : {2, 3, 5, 8}) {
        for (const auto& shape : fallback_shapes) {
            if (!batch_fixture(BatchKernel::exact, m, shape.first, shape.second)) return false;
        }
    }
    return true;
}

bool compact_misc_fixtures() {
    if (mollm_cuda::launch_q4_compact_small_batch_exact(nullptr, nullptr, nullptr, 4, 8, 32, 32, 8, false) !=
        cudaErrorInvalidValue)
        return false;
    for (int m : {2, 3, 4, 5, 7, 8}) {
        for (int n : {13312, 16480}) {
            for (int padding : {0, 3}) {
                if (!compact_fixture(m, n, padding, n == 13312)) return false;
            }
        }
    }
    for (int n : {13312, 16480}) {
        for (int padding : {0, 3}) {
            for (int hint : {1, 2}) {
                if (!compact_fixture(1, n, padding, n == 13312, hint)) return false;
            }
        }
    }
    return mollm_cuda::launch_q4_compact_cache_probe(nullptr, nullptr, nullptr, 16480, 2560, false, 2) ==
           cudaErrorInvalidValue;
}

bool invalid_batch_arguments_rejected() {
    if (mollm_cuda::launch_q4_g32_small_batch(nullptr, nullptr, nullptr, 4, 8, 32, 32, 8) !=
        cudaErrorInvalidValue)
        return false;
    if (mollm_cuda::launch_q4_g32_small_batch_exact(nullptr, nullptr, nullptr, 4, 8, 32, 32, 8) !=
        cudaErrorInvalidValue)
        return false;
    GuardedBuffer probe_buffer(128 * sizeof(float));
    float* probe = probe_buffer.as<float>();
    const auto* weight = reinterpret_cast<const Q4B8G32Block*>(probe);
    const auto rejected = [&](const float* input, int columns, int lda, int ldc) {
        return mollm_cuda::launch_q4_g32_small_batch_wide(input, weight, probe, 2, columns, 32, lda, ldc) ==
               cudaErrorInvalidValue;
    };
    return rejected(probe, 1, 33, 1) && rejected(probe + 1, 1, 32, 1) &&
           rejected(probe, INT32_MAX, 32, INT32_MAX);
}

bool default_suite() {
    if (!narrow_vector_fixtures()) return false;
    if (!compact_fixed_m5_fixtures()) return false;
    if (!compact_prefetch_fixtures()) return false;
    if (!exact_shape_fixtures({{1, 2560}, {4, 10240}}, {{1, 32}, {4, 2560}, {7, 10240}, {16, 10240}}))
        return false;
    if (!exact_shape_fixtures({{10240, 320}, {2560, 640}},
                              {{10240, 640}, {2560, 320}, {8199, 320}, {7, 640}}))
        return false;
    if (!compact_misc_fixtures()) return false;
    // The backend checks in the timing fixtures below run with the backend's
    // plain Q4 small-batch route enabled.
    setenv("MOLLM_CUDA_Q4_SMALL_BATCH", "1", 1);
    if (!invalid_batch_arguments_rejected()) return false;

    for (int m : {2, 3, 4, 5, 7, 8}) {
        for (int n : {7, 320, 2560}) {
            for (int k : {32, 2560}) {
                if (!batch_fixture(BatchKernel::plain, m, n, k)) return false;
            }
        }
    }
    for (int m : {4, 8}) {
        for (auto shape : {std::pair<int, int>{320, 10240}, {10240, 2560}, {2560, 6144}, {10240, 320}}) {
            if (!batch_fixture(BatchKernel::plain, m, shape.first, shape.second, true)) return false;
        }
    }
    for (BatchKernel kernel : {BatchKernel::wide, BatchKernel::reduced}) {
        const int middle_n = kernel == BatchKernel::wide ? 8199 : 2560;
        for (int m : {2, 3, 4, 5, 7, 8}) {
            for (int n : {7, 320, middle_n, 10240}) {
                if (!batch_fixture(kernel, m, n, 2560)) return false;
            }
        }
        for (int m : {4, 8}) {
            if (!batch_fixture(kernel, m, 10240, 2560, true) || !batch_fixture(kernel, m, 20608, 2560, true))
                return false;
        }
    }
    for (int m : {2, 3, 4, 5, 7, 8}) {
        for (int n : {7, 320, 8199, 10240}) {
            if (!batch_fixture(BatchKernel::exact, m, n, 2560)) return false;
        }
    }
    for (int m : {4, 8}) {
        if (!batch_fixture(BatchKernel::exact, m, 10240, 2560, true) ||
            !batch_fixture(BatchKernel::exact, m, 2560, 6144, true))
            return false;
    }
    for (int m : {2, 3, 8}) {
        for (int n : {320, 10240}) {
            if (!batch_fixture(BatchKernel::exact, m, n, 2560, false, 3) ||
                !batch_fixture(BatchKernel::exact, m, n, 2560, false, 4, 1))
                return false;
        }
    }
    if (!batch_fixture(BatchKernel::exact, 4, 320, 32) || !batch_fixture(BatchKernel::exact, 4, 10240, 320))
        return false;
    for (int m : {5, 7}) {
        if (!batch_fixture(BatchKernel::exact, m, 2560, 6144) ||
            !batch_fixture(BatchKernel::exact, m, 320, 10240))
            return false;
    }
    for (int n : {7, 320, 8199, 10240}) {
        if (!batch_fixture(BatchKernel::exact, 6, n, 2560)) return false;
    }
    if (!batch_fixture(BatchKernel::exact, 6, 2560, 6144) ||
        !batch_fixture(BatchKernel::exact, 6, 320, 10240))
        return false;
    std::puts("Q4 small-batch numeric/guard/graph cases passed");
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* mode = argc == 2 ? argv[1] : "";
    const bool sanitizer_subset = std::strcmp(mode, "--small-reduction-smoke") == 0;
    const bool small_reduction = std::strcmp(mode, "--small-reduction-only") == 0;
    const bool m6 = std::strcmp(mode, "--m6-only") == 0;
    if (argc > 2 || (argc == 2 && !sanitizer_subset && !small_reduction && !m6)) {
        std::fprintf(stderr, "usage: %s [--small-reduction-only | --small-reduction-smoke | --m6-only]\n",
                     argv[0]);
        return 2;
    }
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    bool passed;
    if (sanitizer_subset)
        passed = small_reduction_sanitizer_fixtures();
    else if (small_reduction)
        passed = small_reduction_fixtures();
    else if (m6)
        passed = m6_fixtures();
    else
        passed = default_suite();
    return passed ? 0 : 1;
}
