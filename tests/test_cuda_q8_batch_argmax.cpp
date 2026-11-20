// CudaBackend greedy Q8 LM-head argmax for small token batches.
//   (no arguments)            small heads: batch rows match the single-token
//                             argmax, with growth/reuse, changed inputs, ties
//                             and output guards.
//   --screen-shape            the admitted 248320 x 2560 head shape.
//   --screen-cache-lifecycle  weight switches across scalar/batch screen caches.
#include "backends/cuda/backend.h"

#include "cuda_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using mollm_test::require;

Tensor make_q8_weight(CudaBackend& backend, std::vector<int8_t>& codes, std::vector<float>& scales, int n,
                      int k, const char* what) {
    Tensor weight = Tensor::create(Precision::INT8, MemoryType::EXTERNAL, n, k, 1, 1, codes.data());
    weight.rowmajor_data = codes.data();
    weight.scales = scales.data();
    weight.group_size = k;
    weight.groups_per_row = 1;
    backend.wrap_weight(weight);
    require(!backend.dispatch_failed(), what);
    return weight;
}

// Exercise the admitted head shape and scalar/batch scratch reuse at widths
// 2-8, including widths that fall back from the batch screen. Full logits
// provide an independent native-head oracle and must remain dense after a
// sparse argmax call.
int screen_shape() {
    CudaBackend backend;
    if (!backend.available())
        return mollm_test::skip_status;
    constexpr int n = 248320;
    constexpr int k = 2560;
    std::vector<int8_t> weights(size_t(n) * k);
    std::vector<float> scales(n, .001f);
    std::vector<float> x(size_t(8) * k);
    std::vector<float> logits(n);
    for (int r = 0; r < n; ++r) {
        for (int j = 0; j < k; ++j)
            weights[size_t(r) * k + j] = int8_t(((r == 1 ? 0 : r) * 13 + j * 31) % 255 - 127);
    }
    Tensor weight = make_q8_weight(backend, weights, scales, n, k, "prepare large Q8 head");
    Tensor input = Tensor::create(Precision::FP32, MemoryType::NONE, k, 8);
    require(backend.alloc_output(input, input.nbytes(), nullptr) != nullptr, "allocate large-head inputs");
    for (int variant : {0, 1}) {
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = variant ? std::sin(float(i * 7 + 23) * .017f) : 0.f;
        require(backend.copy_from_host(x.data(), input, x.size() * sizeof(float)),
                "upload large-head inputs");
        for (int rows : {2, 3, 5, 4, 3, 8, 5, 2}) {
            std::vector<int> out(rows + 2, -777);
            const bool launched = backend.lm_head_small_batch_argmax_device_and_end_graph(
                input, weight, out.data() + 1, rows, n, k);
            require(launched, "large-head batch top1");
            require(out.front() == -777 && out.back() == -777, "large-head output guards");
            for (int t = 0; t < rows; ++t) {
                const int scalar =
                    backend.lm_head_argmax_device_and_end_graph(input, size_t(t) * k, weight, n, k);
                require(scalar == out[t + 1], "large-head scalar/batch scratch reuse");
                backend.lm_head_gemv(x.data() + size_t(t) * k, weight, logits.data(), n, k, 0);
                require(!backend.dispatch_failed(), "large-head full logits");
                require(std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); }),
                        "full logits remain dense after sparse argmax");
                const int expected = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
                require(out[t + 1] == expected, "large-head independent full-logit argmax");
                if (!variant)
                    require(expected == 0, "large-head tie selects first row");
            }
        }
    }
    backend.free_output(input, nullptr);
    std::puts("Q8 large-head screen: scalar/batch reuse, dense full logits, width fallbacks, changed inputs "
              "and guards passed");
    return 0;
}

// Switch weight identities while alternating scalar Q4 and batch Q6. Recreate
// the backend to exercise per-instance ownership and destruction as well.
int screen_cache_lifecycle() {
    {
        CudaBackend probe;
        if (!probe.available())
            return mollm_test::skip_status;
    }
    constexpr int n = 248320;
    constexpr int k = 2560;
    std::vector<int8_t> a(size_t(n) * k, 0);
    std::vector<int8_t> b(size_t(n) * k, 0);
    std::vector<float> sa(n, 1e-4f);
    std::vector<float> sb(n, 1e-4f);
    std::vector<float> x(size_t(3) * k, 1.f);
    std::vector<float> logits(n);
    a[size_t(7) * k] = 127;
    sa[7] = 1.f;
    b[size_t(11) * k] = 127;
    sb[11] = 1.f;
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
        CudaBackend backend;
        require(backend.available(), "recreated backend");
        Tensor wa = make_q8_weight(backend, a, sa, n, k, "cache-switch weight");
        Tensor wb = make_q8_weight(backend, b, sb, n, k, "cache-switch weight");
        Tensor input = Tensor::create(Precision::FP32, MemoryType::NONE, k, 3);
        require(backend.alloc_output(input, input.nbytes(), nullptr) != nullptr, "cache-switch input");
        std::fill(x.begin(), x.end(), 1.f);
        require(backend.copy_from_host(x.data(), input, x.size() * 4), "cache-switch upload");
        auto scalar = [&](const Tensor& w, int expected) {
            require(backend.lm_head_argmax_device_and_end_graph(input, 0, w, n, k) == expected,
                    "scalar uses the current weight source");
        };
        auto batch = [&](const Tensor& w, int expected) {
            int out[5] = {-777, -777, -777, -777, -777};
            require(backend.lm_head_small_batch_argmax_device_and_end_graph(input, w, out + 1, 3, n, k),
                    "batch after scalar format switch");
            require(out[0] == -777 && out[4] == -777 && out[1] == expected && out[2] == expected &&
                        out[3] == expected,
                    "batch Q6 retains its independent format and source");
        };
        scalar(wa, 7);
        scalar(wa, 7);
        batch(wa, 7);
        scalar(wa, 7);
        scalar(wb, 11);
        batch(wa, 7);
        scalar(wb, 11);
        scalar(wa, 7);
        batch(wb, 11);
        scalar(wa, 7);
        backend.lm_head_gemv(x.data(), wa, logits.data(), n, k, 0);
        require(!backend.dispatch_failed() && logits[7] == 127.f &&
                    std::count(logits.begin(), logits.end(), 0.f) == n - 1,
                "dense logits remain independent");
        std::fill(x.begin(), x.end(), -1.f);
        require(backend.copy_from_host(x.data(), input, x.size() * 4), "changed cached input");
        scalar(wb, 0);
        batch(wb, 0);
        scalar(wa, 0);
        backend.free_output(input, nullptr);
    }
    std::puts("Q8 screen cache: two lifetimes, weight replacement, scalar/batch formats and dense logits "
              "passed");
    return 0;
}

// Small heads whose shapes the screen does not admit, at widths 2-8 with
// workspace growth and reuse, changed inputs and first-row ties.
int small_heads() {
    CudaBackend backend;
    if (!backend.available())
        return mollm_test::skip_status;
    constexpr int k = 64;
    for (int n : {17, 257, 4097}) {
        std::vector<int8_t> w(size_t(n) * k);
        std::vector<float> scales(n);
        for (int r = 0; r < n; ++r) {
            scales[r] = .007f * (1 + r % 7);
            for (int j = 0; j < k; ++j)
                w[size_t(r) * k + j] = int8_t((r * 13 + j * 31 + r * j) % 255 - 127);
        }
        Tensor weight = make_q8_weight(backend, w, scales, n, k, "prepare Q8 weight");
        for (int rows : {2, 3, 4, 8, 3, 2}) {
            Tensor input = Tensor::create(Precision::FP32, MemoryType::NONE, k, rows);
            require(backend.alloc_output(input, input.nbytes(), nullptr) != nullptr, "allocate inputs");
            std::vector<float> x(size_t(rows) * k);
            for (int variant : {0, 1, 2}) {
                for (size_t i = 0; i < x.size(); ++i)
                    x[i] = variant == 0 ? 0.f : std::sin(float(i * 7 + variant * 23) * .017f);
                require(backend.copy_from_host(x.data(), input, x.size() * sizeof(float)),
                        "upload changed input");
                std::vector<int> out(rows + 2, -777);
                const bool launched = backend.lm_head_small_batch_argmax_device_and_end_graph(
                    input, weight, out.data() + 1, rows, n, k);
                require(launched, "batch top1");
                require(out.front() == -777 && out.back() == -777, "output guards");
                for (int r = 0; r < rows; ++r) {
                    const int expected =
                        backend.lm_head_argmax_device_and_end_graph(input, size_t(r) * k, weight, n, k);
                    require(expected >= 0 && !backend.dispatch_failed() && out[r + 1] == expected,
                            "each row matches single-token production argmax");
                    if (variant == 0)
                        require(out[r + 1] == 0, "zero-input tie chooses first row");
                }
            }
            backend.free_output(input, nullptr);
        }
    }
    std::puts("Q8 batch argmax: row outputs, growth/reuse, changed inputs, ties and guards passed");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1)
        return small_heads();
    if (argc == 2 && !std::strcmp(argv[1], "--screen-shape"))
        return screen_shape();
    if (argc == 2 && !std::strcmp(argv[1], "--screen-cache-lifecycle"))
        return screen_cache_lifecycle();
    std::fprintf(stderr, "usage: %s [--screen-shape | --screen-cache-lifecycle]\n", argv[0]);
    return 2;
}
