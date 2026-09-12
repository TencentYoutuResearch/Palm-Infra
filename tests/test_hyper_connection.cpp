#include "kernels/cpu/hyper_connection.h"
#include "runtime/threading.h"

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
}

int main() {
    constexpr int hidden = 2;
    constexpr int hc = 2;
    constexpr int mix = (2 + hc) * hc;
    float x_data[hc * hidden] = {1, 2, 3, 4};
    float fn_data[mix * hc * hidden] = {};
    float scale_data[3] = {1, 1, 1};
    float base_data[mix] = {};
    float packed_data[hidden + hc + hc * hc] = {};
    Tensor x = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                              hc * hidden, 1, 1, 1, x_data);
    Tensor fn = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                               mix, hc * hidden, 1, 1, fn_data);
    Tensor scale = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  3, 1, 1, 1, scale_data);
    Tensor base = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                 mix, 1, 1, 1, base_data);
    Tensor packed = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        hidden + hc + hc * hc, 1, 1, 1, packed_data);
    ThreadPool pool(2);
    kernel_hc_pre(x, fn, scale, base, packed, hidden, hc, 3,
                  1e-6f, 1e-6f, &pool);
    check(std::abs(packed_data[0] - 2.0f) < 1e-4f &&
              std::abs(packed_data[1] - 3.0f) < 1e-4f,
          "HC pre weighted reduction");
    check(std::abs(packed_data[2] - 1.0f) < 1e-6f &&
              std::abs(packed_data[3] - 1.0f) < 1e-6f,
          "HC pre post coefficients");
    check(std::abs(packed_data[4] - 0.5f) < 1e-4f &&
              std::abs(packed_data[7] - 0.5f) < 1e-4f,
          "HC pre Sinkhorn combination");

    float branch_data[hidden] = {10, 20};
    float post_data[hc * hidden] = {};
    Tensor branch = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                   hidden, 1, 1, 1, branch_data);
    Tensor post = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                 hc * hidden, 1, 1, 1, post_data);
    kernel_hc_post(branch, x, packed, post, hidden, hc, &pool);
    check(std::abs(post_data[0] - 12.0f) < 1e-4f &&
              std::abs(post_data[1] - 23.0f) < 1e-4f &&
              std::abs(post_data[2] - 12.0f) < 1e-4f &&
              std::abs(post_data[3] - 23.0f) < 1e-4f,
          "HC post residual mixing");

    // PyTorch evaluates sum(comb[j, k] * residual[j], dim=j), so each
    // output copy consumes a column of the Sinkhorn matrix. A symmetric
    // fixture cannot distinguish this from the transposed implementation.
    float zero_branch[hidden] = {};
    Tensor zero_branch_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        hidden, 1, 1, 1, zero_branch);
    packed_data[hidden] = 0.0f;
    packed_data[hidden + 1] = 0.0f;
    packed_data[hidden + hc + 0] = 0.1f;
    packed_data[hidden + hc + 1] = 0.9f;
    packed_data[hidden + hc + 2] = 0.7f;
    packed_data[hidden + hc + 3] = 0.3f;
    kernel_hc_post(
        zero_branch_tensor, x, packed, post, hidden, hc, &pool);
    check(std::abs(post_data[0] - 2.2f) < 0.01f &&
              std::abs(post_data[1] - 3.0f) < 0.01f &&
              std::abs(post_data[2] - 1.8f) < 0.01f &&
              std::abs(post_data[3] - 3.0f) < 0.01f,
          "HC post uses combination columns");

    float head_fn_data[hc * hc * hidden] = {};
    float head_scale_data[1] = {1};
    float head_base_data[hc] = {};
    float head_data[hidden] = {};
    Tensor head_fn = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        hc, hc * hidden, 1, 1, head_fn_data);
    Tensor head_scale = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        1, 1, 1, 1, head_scale_data);
    Tensor head_base = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        hc, 1, 1, 1, head_base_data);
    Tensor head = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                 hidden, 1, 1, 1, head_data);
    kernel_hc_head(x, head_fn, head_scale, head_base, head,
                   hidden, hc, 1e-6f, 1e-6f, &pool);
    check(std::abs(head_data[0] - 2.0f) < 1e-4f &&
              std::abs(head_data[1] - 3.0f) < 1e-4f,
          "HC head weighted reduction");

    // Exercise the prefill path, where token-parallel workers consume the
    // projection scratch produced on the caller thread.
    {
        constexpr int test_hidden = 8;
        constexpr int test_hc = 2;
        constexpr int test_tokens = 3;
        constexpr int test_wide = test_hidden * test_hc;
        constexpr int test_mix = (2 + test_hc) * test_hc;
        constexpr int test_packed =
            test_hidden + test_hc + test_hc * test_hc;
        std::vector<float> test_x(test_wide * test_tokens);
        std::vector<float> test_fn(test_mix * test_wide);
        std::vector<float> test_base(test_mix);
        for (size_t i = 0; i < test_x.size(); ++i)
            test_x[i] =
                static_cast<float>(static_cast<int>((i * 7) % 19) - 9) /
                16.0f;
        for (size_t i = 0; i < test_fn.size(); ++i)
            test_fn[i] =
                static_cast<float>(static_cast<int>((i * 11) % 23) - 11) /
                32.0f;
        for (size_t i = 0; i < test_base.size(); ++i)
            test_base[i] = static_cast<float>(i) / 64.0f;
        float test_scale[3] = {0.75f, 1.25f, 0.5f};
        std::vector<float> serial(test_packed * test_tokens);
        std::vector<float> parallel(test_packed * test_tokens);
        Tensor test_x_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            test_wide, test_tokens, 1, 1, test_x.data());
        Tensor test_fn_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            test_mix, test_wide, 1, 1, test_fn.data());
        Tensor test_scale_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            3, 1, 1, 1, test_scale);
        Tensor test_base_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            test_mix, 1, 1, 1, test_base.data());
        Tensor serial_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            test_packed, test_tokens, 1, 1, serial.data());
        Tensor parallel_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            test_packed, test_tokens, 1, 1, parallel.data());
        check(kernel_hc_pre(
                  test_x_tensor, test_fn_tensor, test_scale_tensor,
                  test_base_tensor, serial_tensor, test_hidden, test_hc,
                  5, 1e-6f, 1e-6f, nullptr),
              "HC pre serial multi-token path");
        check(kernel_hc_pre(
                  test_x_tensor, test_fn_tensor, test_scale_tensor,
                  test_base_tensor, parallel_tensor, test_hidden, test_hc,
                  5, 1e-6f, 1e-6f, &pool),
              "HC pre parallel multi-token path");
        float max_difference = 0.0f;
        for (size_t i = 0; i < serial.size(); ++i) {
            max_difference = std::max(
                max_difference, std::abs(serial[i] - parallel[i]));
        }
        check(max_difference <= 1e-5f,
              "HC pre serial and parallel multi-token outputs agree");
    }

    if (failures == 0)
        std::printf("All Hyper-Connection tests passed!\n");
    return failures == 0 ? 0 : 1;
}
