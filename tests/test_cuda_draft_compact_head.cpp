#include "backends/cuda/backend.h"
#include "kernels/cuda/draft_compact_head.h"

#include "cuda_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using mollm_test::check;
using mollm_test::GuardedBuffer;
using mollm_test::require;

void test_probability() {
    require(mollm_cuda::launch_draft_head_probability(nullptr, 1, nullptr, nullptr, 1, nullptr) ==
                cudaErrorInvalidValue,
            "probability rejects null");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int n : {1, 17, 257, 65537, 248320}) {
        for (int distribution : {0, 1, 2}) {
            const int groups = std::min(256, (n + 255) / 256);
            GuardedBuffer logits(size_t(n) * sizeof(float));
            GuardedBuffer scratch(size_t(groups) * sizeof(float));
            GuardedBuffer result(sizeof(mollm_cuda::DraftHeadResult));
            const std::vector<float> partials(groups, 123.25f);
            scratch.upload(partials.data());
            auto* result_data = result.as<mollm_cuda::DraftHeadResult>();
            require(mollm_cuda::launch_draft_head_probability(logits.as<float>(), n, result_data,
                                                              scratch.as<float>(), 0, stream) ==
                        cudaErrorInvalidValue,
                    "zero probability groups");
            require(mollm_cuda::launch_draft_head_probability(logits.as<float>(), n, result_data,
                                                              scratch.as<float>(), 257, stream) ==
                        cudaErrorInvalidValue,
                    "too many probability groups");
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            check(mollm_cuda::launch_draft_head_probability(logits.as<float>(), n, result_data,
                                                            scratch.as<float>(), groups, stream));
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 2; ++replay) {
                std::vector<float> values(n);
                for (int i = 0; i < n; ++i) {
                    if (distribution == 0)
                        values[i] = 10000.f;
                    else if (distribution == 1)
                        values[i] = i == (replay ? n - 1 : 0) ? 80.f : -80.f;
                    else
                        values[i] = float((i * 17 + replay * 31) % 1001 - 500) * .031f + replay * 1000.f;
                }
                const int index = int(std::max_element(values.begin(), values.end()) - values.begin());
                const float maximum = values[index];
                double denominator = 0;
                for (float value : values) denominator += std::exp(double(value) - maximum);
                const double expected = 1.0 / denominator;
                const mollm_cuda::DraftHeadResult top{{maximum, index}, -9.f, 0.f};
                logits.upload(values.data());
                result.upload(&top);
                check(cudaGraphLaunch(executable, stream));
                check(cudaStreamSynchronize(stream));
                const auto actual = result.download<mollm_cuda::DraftHeadResult>()[0];
                require(actual.top1.index == index && actual.top1.value == maximum,
                        "probability preserves argmax");
                require(std::abs(double(actual.probability) - expected) <= 3e-6 * expected,
                        "GPU probability versus independent CPU double softmax");
                result.check_guards();
                scratch.check_guards();
            }
            check(cudaGraphExecDestroy(executable));
            check(cudaGraphDestroy(graph));
        }
    }
    check(cudaStreamDestroy(stream));
}

void test_packing() {
    require(mollm_cuda::pack_q8_draft_head_compact(nullptr, nullptr, nullptr, 1, 32,
                                                   cudaStreamPerThread) == cudaErrorInvalidValue,
            "null inputs rejected");
    for (int k : {32, 64, 2560}) {
        for (int n : {1, 17, 257}) {
            std::vector<int8_t> w(size_t(n) * k);
            std::vector<float> s(n);
            for (int r = 0; r < n; ++r) {
                s[r] = (r % 3 == 0 ? -1.f : 1.f) * (.001f * (1 + r % 7));
                for (int j = 0; j < k; ++j)
                    w[size_t(r) * k + j] = r == 0 ? 0 : int8_t((r * 13 + j * 29) % 256 - 128);
            }
            std::vector<mollm_cuda::Q4CompactG32> expected(size_t((n + 7) / 8) * (k / 32));
            for (int r = 0; r < n; ++r) {
                for (int g = 0; g < k / 32; ++g) {
                    const int8_t* group = w.data() + size_t(r) * k + g * 32;
                    int mx = 0;
                    for (int j = 0; j < 32; ++j) mx = std::max(mx, std::abs(int(group[j])));
                    auto& b = expected[size_t(r / 8) * (k / 32) + g];
                    const uint32_t bits = mollm_test::to_bits(float(mx) * s[r]);
                    b.scales[r % 8] = uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
                    for (int j = 0; j < 32; ++j) {
                        int q = mx ? int(std::nearbyint(float(group[j]) * (7.f / float(mx)))) : 0;
                        q = std::clamp(q, -7, 7) + 8;
                        b.q[r % 8][j / 2] |= uint8_t(q << ((j % 2) * 4));
                    }
                }
            }
            GuardedBuffer dw(w.size());
            GuardedBuffer ds(s.size() * sizeof(float));
            GuardedBuffer dq(expected.size() * sizeof(expected[0]));
            dw.upload(w.data());
            ds.upload(s.data());
            check(mollm_cuda::pack_q8_draft_head_compact(dw.as<int8_t>(), ds.as<float>(),
                                                         dq.as<mollm_cuda::Q4CompactG32>(), n, k,
                                                         cudaStreamPerThread));
            require(mollm_test::same_bits(dq.download<mollm_cuda::Q4CompactG32>(), expected),
                    "packed Q4 equals independent CPU quantizer, including padded rows");
            dq.check_guards();
        }
    }
}

void test_backend() {
    std::vector<std::vector<int8_t>> weights_keep;
    std::vector<std::vector<float>> scales_keep;
    CudaBackend backend;
    require(backend.available(), "backend available");
    for (int k : {32, 36}) {
        for (int n : {17, 257, 17}) {
            weights_keep.emplace_back(size_t(n) * k);
            scales_keep.emplace_back(n, 1.f);
            auto& w = weights_keep.back();
            auto& s = scales_keep.back();
            for (int r = 0; r < n; ++r) {
                w[size_t(r) * k] = 100;
                w[size_t(r) * k + 1] = r < 2 ? int8_t(14 + r) : int8_t(-127);
            }
            Tensor weight = Tensor::create(Precision::INT8, MemoryType::EXTERNAL, n, k, 1, 1, w.data());
            weight.rowmajor_data = w.data();
            weight.scales = s.data();
            weight.group_size = k;
            weight.groups_per_row = 1;
            backend.wrap_weight(weight);
            Tensor input = Tensor::create(Precision::FP32, MemoryType::NONE, k, 1);
            require(backend.alloc_output(input, input.nbytes(), nullptr), "input allocation");
            std::vector<float> x(k);
            x[1] = 1;
            require(backend.copy_from_host(x.data(), input, k * sizeof(float)), "upload input");
            // The compact draft head needs whole 32-wide groups; other shapes stay exact Q8.
            const bool compact = k % 32 == 0;
            const int target = backend.lm_head_argmax_device_and_end_graph(input, 0, weight, n, k);
            if (target != 1) {
                std::fprintf(stderr, "target mismatch n=%d k=%d id=%d failed=%d\n", n, k, target,
                             int(backend.dispatch_failed()));
            }
            require(target == 1, "target head starts exact");
            require(backend.lm_head_draft_argmax_device_and_end_graph(input, 0, weight, n, k) ==
                        (compact ? 0 : 1),
                    "only draft role may approximate");
            require(backend.lm_head_argmax_device_and_end_graph(input, 0, weight, n, k) == 1,
                    "target remains exact after draft scope");
            std::fill(x.begin(), x.end(), 0.f);
            require(backend.copy_from_host(x.data(), input, k * sizeof(float)), "changed input");
            require(backend.lm_head_draft_argmax_device_and_end_graph(input, 0, weight, n, k) == 0,
                    "zero draft tie");
            float probability = 99.f;
            require(backend.lm_head_draft_argmax_device_and_end_graph(input, 0, weight, n, k, 0, nullptr,
                                                                      &probability) == 0,
                    "probability keeps stable draft argmax");
            if (compact)
                require(std::abs(double(probability) - 1.0 / n) < 1e-6 / n,
                        "backend uniform draft probability");
            else
                require(probability == -1.f, "unsupported draft confidence is explicit");
            require(backend.lm_head_argmax_device_and_end_graph(input, 0, weight, n, k) == 0,
                    "probability request does not leak to target head");
            backend.free_output(input, nullptr);
        }
    }
}

}  // namespace

int main() {
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    test_probability();
    test_packing();
    test_backend();
    std::puts("draft Q4 head: independent packing/probability oracles, guarded changed-input graph "
              "replay, signed/zero scales, padded rows, weight changes, exact target/draft isolation "
              "passed");
}
