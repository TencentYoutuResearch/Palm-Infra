// Cross-check harness: encode a vector with the C++ kernel and print the raw
// bytes, so a Python reference can be diffed against it element by element.
//
// Reconstruction error is NOT a usable signal here (ternary on random data
// sits around 55% regardless of correctness), so the comparison must be on
// the encoder's exact output: qs, sign and the fp16 scale.

#include "kernels/cpu/matmul/stq1_0.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <input.f32> [n_blocks]\n", argv[0]);
        return 2;
    }
    const int n_blocks = argc > 2 ? std::atoi(argv[2]) : 1;
    const int k = n_blocks * QK_STQ_K;

    std::vector<float> x(k);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    if (std::fread(x.data(), sizeof(float), k, f) != static_cast<size_t>(k)) {
        std::fprintf(stderr, "short read\n");
        std::fclose(f);
        return 2;
    }
    std::fclose(f);

    for (int b = 0; b < n_blocks; ++b) {
        block_stq1_0 blk;
        float scale = 0.0f;
        stq1_0_quantize_block(x.data() + b * QK_STQ_K, nullptr, blk, &scale);
        std::printf("block %d d=%.9f raw_d=%u\n", b, scale, blk.d);
        std::printf("qs");
        for (int i = 0; i < STQ_QS_BYTES; ++i)
            std::printf(" %u", blk.qs[i]);
        std::printf("\nsign");
        for (int i = 0; i < STQ_SIGN_BYTES; ++i)
            std::printf(" %u", blk.sign[i]);
        std::printf("\n");

        // Also emit the reconstruction so the Python side can compare
        // dequantized values directly.
        std::vector<float> rec(QK_STQ_K);
        stq1_0_dequantize_block(blk, rec.data());
        std::printf("rec");
        for (int i = 0; i < QK_STQ_K; ++i)
            std::printf(" %.9g", rec[i]);
        std::printf("\n");
    }
    return 0;
}
