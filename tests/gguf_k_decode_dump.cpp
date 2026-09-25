// Decode one GGUF quantization block for cross-checking against gguf-py.
#include "kernels/cpu/matmul/gguf_k.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s PRECISION BLOCK_FILE\n", argv[0]);
        return 2;
    }
    const int id = std::atoi(argv[1]);
    const Precision precision = static_cast<Precision>(id);
    const size_t block_bytes = gguf_k_block_bytes(precision);
    const size_t values_count = gguf_k_block_values(precision);
    if (!block_bytes) return 2;
    FILE* f = std::fopen(argv[2], "rb");
    if (!f) return 2;
    std::vector<uint8_t> bytes(block_bytes);
    const size_t read = std::fread(bytes.data(), 1, block_bytes, f);
    std::fclose(f);
    if (read != block_bytes) return 2;
    float values[256];
    if (!gguf_k_dequantize_block(precision, bytes.data(), values)) return 1;
    return std::fwrite(values, sizeof(float), values_count, stdout) ==
                   values_count ? 0 : 1;
}
