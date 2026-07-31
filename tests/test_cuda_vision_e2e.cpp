#include "engine/cuda_backend.h"
#include "engine/engine.h"
#include "kernels/matmul.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct VisionResult {
    VisionEmbedding embedding;
    int next_token = -1;
};

int metadata_int(const LLMEngine& engine, const char* key, int fallback) {
    const auto found = engine.package_metadata().find(key);
    if (found == engine.package_metadata().end())
        return fallback;
    try {
        return std::stoi(found->second);
    } catch (...) {
        return fallback;
    }
}

bool run_model(const char* package, Device device, VisionResult& result,
               bool repeat_encoder) {
    EngineConfig config;
    config.package_path = package;
    config.device = device;
    config.device_fallback = device == Device::CPU
        ? DeviceFallbackPolicy::ALLOW_CPU
        : DeviceFallbackPolicy::REQUIRE_REQUESTED;
    config.operator_fallback = device == Device::CUDA
        ? OperatorFallbackPolicy::REQUIRE_NATIVE
        : OperatorFallbackPolicy::ALLOW_REFERENCE;
    config.weight_loading = WeightLoadingMode::MMAP;
    config.num_threads = 4;
    LLMEngine engine;
    if (!engine.load(config) || engine.config().device != device)
        return false;

    constexpr int grid_t = 1;
    constexpr int grid_h = 4;
    constexpr int grid_w = 4;
    const int patch = metadata_int(engine, "vision_patch_size", 16);
    const int temporal = metadata_int(
        engine, "vision_temporal_patch_size", 2);
    const int patch_dim = 3 * temporal * patch * patch;
    std::vector<float> pixels(
        static_cast<size_t>(grid_t) * grid_h * grid_w * patch_dim);
    for (size_t index = 0; index < pixels.size(); ++index)
        pixels[index] =
            (static_cast<int>((index * 7) % 53) - 26) / 53.0f;

    std::string error;
    if (!engine.encode_vision_patches(
            pixels, grid_t, grid_h, grid_w, result.embedding, &error)) {
        std::fprintf(stderr, "vision encode failed: %s\n", error.c_str());
        return false;
    }
    if (repeat_encoder) {
        const VisionEmbedding first = result.embedding;
        if (!engine.encode_vision_patches(
                pixels, grid_t, grid_h, grid_w, result.embedding, &error)) {
            std::fprintf(
                stderr, "repeated vision encode failed: %s\n",
                error.c_str());
            return false;
        }
        if (first.tokens != result.embedding.tokens ||
            first.hidden_size != result.embedding.hidden_size ||
            first.values != result.embedding.values) {
            std::fprintf(stderr, "repeated CUDA vision encode changed output\n");
            return false;
        }
    }

    const int image_token_id = metadata_int(engine, "image_token_id", -1);
    if (image_token_id < 0)
        return false;
    std::vector<int> tokens(
        static_cast<size_t>(result.embedding.tokens), image_token_id);
    result.next_token = engine.prefill_with_image(
        tokens, image_token_id, result.embedding, &error);
    if (result.next_token < 0) {
        std::fprintf(stderr, "vision prefill failed: %s\n", error.c_str());
        return false;
    }
    if (device == Device::CUDA) {
        const auto stats = engine.backend_operator_stats();
        if (!stats.tracked || stats.native_calls == 0 ||
            stats.fallback_calls != 0) {
            std::fprintf(
                stderr,
                "CUDA vision operator coverage is native=%llu fallback=%llu\n",
                static_cast<unsigned long long>(stats.native_calls),
                static_cast<unsigned long long>(stats.fallback_calls));
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* package = argc > 1 ? argv[1] :
                                  std::getenv("MOLLM_QWEN35_VL_PACKAGE");
    if (!package || !*package) {
        std::fprintf(
            stderr, "MOLLM_QWEN35_VL_PACKAGE is unset; skipping\n");
        return 77;
    }
    CudaBackend probe;
    if (!probe.available()) {
        std::fprintf(stderr, "CUDA device unavailable; skipping\n");
        return 77;
    }
    g_mollm_force_fp32_acc = true;
    VisionResult cpu;
    VisionResult cuda;
    if (!run_model(package, Device::CPU, cpu, false) ||
        !run_model(package, Device::CUDA, cuda, true))
        return 1;
    if (cpu.embedding.tokens != cuda.embedding.tokens ||
        cpu.embedding.hidden_size != cuda.embedding.hidden_size ||
        cpu.embedding.values.size() != cuda.embedding.values.size()) {
        std::fprintf(stderr, "CPU/CUDA vision output shape mismatch\n");
        return 1;
    }
    float maximum_error = 0.0f;
    for (size_t index = 0; index < cpu.embedding.values.size(); ++index)
        maximum_error = std::max(
            maximum_error,
            std::fabs(cpu.embedding.values[index] -
                      cuda.embedding.values[index]));
    if (maximum_error > 2e-3f) {
        std::fprintf(
            stderr, "CPU/CUDA vision max error %g exceeds tolerance\n",
            maximum_error);
        return 1;
    }
    if (cpu.next_token != cuda.next_token) {
        std::fprintf(
            stderr, "CPU/CUDA vision next token mismatch: %d vs %d\n",
            cpu.next_token, cuda.next_token);
        return 1;
    }
    std::printf(
        "Qwen3.5 vision CPU/CUDA matched: tokens=%d hidden=%d "
        "max_abs=%g next_token=%d\n",
        cpu.embedding.tokens, cpu.embedding.hidden_size, maximum_error,
        cpu.next_token);
    return 0;
}
