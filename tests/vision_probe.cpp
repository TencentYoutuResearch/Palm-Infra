#include "engine/engine.h"
#include "kernels/cpu/matmul/matmul.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool read_input(const char* path, int& t, int& h, int& w,
                std::vector<float>& pixels) {
    std::ifstream in(path, std::ios::binary);
    int32_t grid[3] = {};
    uint64_t count = 0;
    if (!in.read(reinterpret_cast<char*>(grid), sizeof(grid)) ||
        !in.read(reinterpret_cast<char*>(&count), sizeof(count)))
        return false;
    if (grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0 ||
        grid[0] > 1024 || grid[1] > 65536 || grid[2] > 65536 ||
        count > (uint64_t{1} << 34))
        return false;
    pixels.resize(static_cast<size_t>(count));
    if (count &&
        !in.read(reinterpret_cast<char*>(pixels.data()),
                 static_cast<std::streamsize>(count * sizeof(float))))
        return false;
    t = grid[0];
    h = grid[1];
    w = grid[2];
    return true;
}

bool write_output(const char* path, const VisionEmbedding& embedding) {
    std::ofstream out(path, std::ios::binary);
    const int32_t shape[2] = {
        embedding.tokens, embedding.hidden_size};
    return out.write(reinterpret_cast<const char*>(shape), sizeof(shape)) &&
           out.write(reinterpret_cast<const char*>(embedding.values.data()),
                     static_cast<std::streamsize>(
                         embedding.values.size() * sizeof(float)));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s <model.mollm> <patches.bin> <output.bin>\n",
                     argv[0]);
        return 2;
    }
    int grid_t = 0, grid_h = 0, grid_w = 0;
    std::vector<float> pixels;
    EngineConfig cfg;
    cfg.package_path = argv[1];
    cfg.weight_loading = WeightLoadingMode::MMAP;
    cfg.num_threads = 4;
    LLMEngine engine;
    if (!engine.load(cfg))
        return 1;
    g_mollm_force_fp32_acc = true;

    VisionEmbedding output;
    std::string error;
    const bool encoded =
        read_input(argv[2], grid_t, grid_h, grid_w, pixels)
            ? engine.encode_vision_patches(
                  pixels, grid_t, grid_h, grid_w, output, &error)
            : engine.encode_image_file(argv[2], output, &error);
    if (!encoded) {
        std::fprintf(stderr, "vision encode failed: %s\n", error.c_str());
        return 1;
    }
    if (!write_output(argv[3], output)) {
        std::fprintf(stderr, "failed to write %s\n", argv[3]);
        return 1;
    }
    std::printf("vision_tokens=%d hidden=%d\n",
                output.tokens, output.hidden_size);
    return 0;
}
