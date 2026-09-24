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

    // ---------------------------------------------------------------------
    // Hy4 identity Hyper-Connections (HC_*_ID).
    //
    // Same fixture as above so the two formulas can be compared directly.
    // hc=2, hidden=2, x streams = [1,2] and [3,4]; flattened = [1,2,3,4].
    // fn row 0 selects flat[0], all other mix rows are zero. So
    //   mean_square = 7.5, rsqrt = 0.365148
    //   mixes = [0.365148, 0, 0, 0]
    //   pre   = [sigmoid(0.365148)+eps, sigmoid(0)+eps] = [0.590287, 0.500001]
    //   post  = 2*sigmoid(0)+eps                        = [1.000001, 1.000001]
    //   reduced = [0.590287*1 + 0.500001*3, 0.590287*2 + 0.500001*4]
    //           = [2.090290, 3.180578], then bf16-rounded by the kernel to
    //           [2.093750, 3.187500].
    // ---------------------------------------------------------------------
    {
        constexpr int id_hidden = 2;
        constexpr int id_hc = 2;
        constexpr int id_mix = 2 * id_hc;  // no combination block
        float id_x_data[id_hc * id_hidden] = {1, 2, 3, 4};
        float id_fn_data[id_mix * id_hc * id_hidden] = {};
        id_fn_data[0] = 1.0f;  // mix row 0 = flat[0]
        float id_scale_data[2] = {1.0f, 1.0f};
        float id_base_data[id_mix] = {};
        float id_packed_data[id_hidden + id_hc] = {};
        Tensor id_x = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            id_hc * id_hidden, 1, 1, 1, id_x_data);
        Tensor id_fn = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            id_mix, id_hc * id_hidden, 1, 1, id_fn_data);
        Tensor id_scale = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            2, 1, 1, 1, id_scale_data);
        Tensor id_base = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            id_mix, 1, 1, 1, id_base_data);
        Tensor id_packed = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            id_hidden + id_hc, 1, 1, 1, id_packed_data);
        check(kernel_hc_pre_id(
                  id_x, id_fn, id_scale, id_base, id_packed, id_hidden,
                  id_hc, 2.0f, 1e-6f, 1e-6f, &pool),
              "HC pre id returns true");
        // Tolerance covers one bf16 ulp at this magnitude (~0.0156).
        check(std::abs(id_packed_data[0] - 2.093750f) < 1e-3f &&
                  std::abs(id_packed_data[1] - 3.187500f) < 1e-3f,
              "HC pre id weighted reduction");
        check(std::abs(id_packed_data[2] - 1.000001f) < 1e-4f &&
                  std::abs(id_packed_data[3] - 1.000001f) < 1e-4f,
              "HC pre id post gate with hc_magnitude");

        // Post: each stream is gate*branch + its OWN residual (no mixing).
        float id_branch_data[id_hidden] = {10, 20};
        float id_post_data[id_hc * id_hidden] = {};
        Tensor id_branch = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            id_hidden, 1, 1, 1, id_branch_data);
        Tensor id_post = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            id_hc * id_hidden, 1, 1, 1, id_post_data);
        check(kernel_hc_post_id(
                  id_branch, id_x, id_packed, id_post, id_hidden, id_hc,
                  &pool),
              "HC post id returns true");
        // stream 0: 1.000001*10 + 1 = 11.00001 ; 1.000001*20 + 2 = 22.00002
        // stream 1: 1.000001*10 + 3 = 13.00001 ; 1.000001*20 + 4 = 24.00002
        check(std::abs(id_post_data[0] - 11.00001f) < 1e-3f &&
                  std::abs(id_post_data[1] - 22.00002f) < 1e-3f &&
                  std::abs(id_post_data[2] - 13.00001f) < 1e-3f &&
                  std::abs(id_post_data[3] - 24.00002f) < 1e-3f,
              "HC post id has no cross-stream mixing");

        // The distinguishing property: with a zero branch the post output must
        // equal the residual exactly, so no stream can leak into another.
        float id_zero_branch[id_hidden] = {};
        Tensor id_zero_branch_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            id_hidden, 1, 1, 1, id_zero_branch);
        check(kernel_hc_post_id(
                  id_zero_branch_tensor, id_x, id_packed, id_post, id_hidden,
                  id_hc, &pool),
              "HC post id zero-branch returns true");
        check(std::abs(id_post_data[0] - 1.0f) < 1e-4f &&
                  std::abs(id_post_data[1] - 2.0f) < 1e-4f &&
                  std::abs(id_post_data[2] - 3.0f) < 1e-4f &&
                  std::abs(id_post_data[3] - 4.0f) < 1e-4f,
              "HC post id zero branch preserves each stream");

        // Head: fn is [hc, hc*hidden]; selected rows give pre = [0.590287,
        // 0.500001], so the reduced output matches the pre reduction.
        float id_head_fn_data[id_hc * id_hc * id_hidden] = {};
        id_head_fn_data[0] = 1.0f;  // selects flat[0] like the pre test
        float id_head_scale_data[1] = {1.0f};
        float id_head_base_data[id_hc] = {};
        float id_head_data[id_hidden] = {};
        Tensor id_head_fn = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            id_hc, id_hc * id_hidden, 1, 1, id_head_fn_data);
        Tensor id_head_scale = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            1, 1, 1, 1, id_head_scale_data);
        Tensor id_head_base = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            id_hc, 1, 1, 1, id_head_base_data);
        Tensor id_head = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            id_hidden, 1, 1, 1, id_head_data);
        check(kernel_hc_head_id(
                  id_x, id_head_fn, id_head_scale, id_head_base, id_head,
                  id_hidden, id_hc, 1e-6f, 1e-6f, &pool),
              "HC head id returns true");
        check(std::abs(id_head_data[0] - 2.093750f) < 1e-3f &&
                  std::abs(id_head_data[1] - 3.187500f) < 1e-3f,
              "HC head id weighted reduction");
    }

    // Reject a DeepSeek-V4 shaped fn (2+hc)*hc, which must not be accepted by
    // the identity variant: the mixing-matrix block is not part of Hy4.
    {
        constexpr int bad_hidden = 2;
        constexpr int bad_hc = 2;
        float bad_x_data[bad_hc * bad_hidden] = {1, 2, 3, 4};
        float bad_fn_data[((2 + bad_hc) * bad_hc) * bad_hc * bad_hidden] = {};
        float bad_scale_data[2] = {1.0f, 1.0f};
        float bad_base_data[(2 + bad_hc) * bad_hc] = {};
        float bad_packed_data[bad_hidden + bad_hc] = {};
        Tensor bad_x = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            bad_hc * bad_hidden, 1, 1, 1, bad_x_data);
        Tensor bad_fn = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            (2 + bad_hc) * bad_hc, bad_hc * bad_hidden, 1, 1, bad_fn_data);
        Tensor bad_scale = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            2, 1, 1, 1, bad_scale_data);
        Tensor bad_base = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            (2 + bad_hc) * bad_hc, 1, 1, 1, bad_base_data);
        Tensor bad_packed = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            bad_hidden + bad_hc, 1, 1, 1, bad_packed_data);
        check(!kernel_hc_pre_id(
                  bad_x, bad_fn, bad_scale, bad_base, bad_packed, bad_hidden,
                  bad_hc, 2.0f, 1e-6f, 1e-6f, &pool),
              "HC pre id rejects the DeepSeek-V4 mixing-matrix fn shape");
    }

    // Multi-token parallel path must agree with the serial path.
    {
        constexpr int test_hidden = 8;
        constexpr int test_hc = 2;
        constexpr int test_tokens = 3;
        constexpr int test_wide = test_hidden * test_hc;
        constexpr int test_mix = 2 * test_hc;
        constexpr int test_packed = test_hidden + test_hc;
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
        float test_scale[2] = {0.75f, 1.25f};
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
            2, 1, 1, 1, test_scale);
        Tensor test_base_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            test_mix, 1, 1, 1, test_base.data());
        Tensor serial_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            test_packed, test_tokens, 1, 1, serial.data());
        Tensor parallel_tensor = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            test_packed, test_tokens, 1, 1, parallel.data());
        check(kernel_hc_pre_id(
                  test_x_tensor, test_fn_tensor, test_scale_tensor,
                  test_base_tensor, serial_tensor, test_hidden, test_hc,
                  2.0f, 1e-6f, 1e-6f, nullptr),
              "HC pre id serial multi-token path");
        check(kernel_hc_pre_id(
                  test_x_tensor, test_fn_tensor, test_scale_tensor,
                  test_base_tensor, parallel_tensor, test_hidden, test_hc,
                  2.0f, 1e-6f, 1e-6f, &pool),
              "HC pre id parallel multi-token path");
        float max_difference = 0.0f;
        for (size_t i = 0; i < serial.size(); ++i) {
            max_difference = std::max(
                max_difference, std::abs(serial[i] - parallel[i]));
        }
        check(max_difference <= 1e-5f,
              "HC pre id serial and parallel multi-token outputs agree");
    }

    if (failures == 0)
        std::printf("All Hyper-Connection tests passed!\n");
    return failures == 0 ? 0 : 1;
}
