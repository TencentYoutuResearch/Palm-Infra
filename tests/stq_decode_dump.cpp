// Dump the STQ1_0 decoder's output, in element order, for a raw block file.
//
// Exists so the Python reader (models/gguf_quants.py) can be diffed against the
// kernel the engine actually runs, on real checkpoint bytes. Reconstruction
// error is not a usable signal here -- ternary quantization sits near 55% on
// real weights whether the decoder is right or wrong -- so the comparison has to
// be on exact values.
//
// Usage: stq_decode_dump <blocks.bin>   (file size must be a multiple of 42)
//
// Build (the stq1_0 translation unit is already in libmollm_lib):
//   c++ -std=c++17 -I. -o build/stq_decode_dump tests/stq_decode_dump.cpp \
//       build/CMakeFiles/mollm_lib.dir/kernels/cpu/matmul/stq1_0.cpp.o

#include "kernels/cpu/matmul/stq1_0.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <blocks.bin>\n", argv[0]);
        return 2;
    }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 0 || size % STQ_BLOCK_BYTES) {
        std::fprintf(stderr,
                     "size %ld is not a positive multiple of %d\n",
                     size, STQ_BLOCK_BYTES);
        std::fclose(f);
        return 2;
    }
    const int n = static_cast<int>(size / STQ_BLOCK_BYTES);
    std::vector<block_stq1_0> blocks(n);
    if (std::fread(blocks.data(), STQ_BLOCK_BYTES, n, f) !=
        static_cast<size_t>(n)) {
        std::fprintf(stderr, "short read\n");
        std::fclose(f);
        return 2;
    }
    std::fclose(f);

    for (int b = 0; b < n; ++b) {
        std::vector<float> rec(QK_STQ_K);
        stq1_0_dequantize_block(blocks[b], rec.data());
        for (int i = 0; i < QK_STQ_K; ++i)
            std::printf("%.9g%c", rec[i], i + 1 == QK_STQ_K ? '\n' : ' ');
    }
    return 0;
}
