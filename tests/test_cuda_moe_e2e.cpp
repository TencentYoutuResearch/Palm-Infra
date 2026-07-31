#include "engine/cuda_backend.h"
#include "engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool copy_finite(const Tensor& tensor, std::vector<float>& values) {
    if (!tensor.data || tensor.prec != Precision::FP32)
        return false;
    values.assign(tensor.ptr<float>(),
                  tensor.ptr<float>() + tensor.nelements());
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool run(const char* package, const char* expected_architecture, Device device,
         std::vector<float>& prefill, std::vector<float>& decode) {
    LLMEngine engine;
    EngineConfig config;
    config.package_path = package;
    config.device = device;
    config.n_ctx = 8;
    config.num_threads = 1;
    config.weight_loading = WeightLoadingMode::MMAP;
    if (!engine.load(config))
        return false;
    const auto& metadata = engine.package_metadata();
    const auto architecture = metadata.find("architecture");
    if (architecture == metadata.end() ||
        architecture->second != expected_architecture)
        return false;
    Tensor prefill_tensor = engine.prefill_hidden({1, 2, 3});
    if (!copy_finite(prefill_tensor, prefill) || engine.past_len() != 3)
        return false;
    Tensor decode_tensor = engine.decode_hidden(4);
    return copy_finite(decode_tensor, decode) && engine.past_len() == 4;
}

bool close_enough(const std::vector<float>& actual,
                  const std::vector<float>& expected, float tolerance,
                  const char* label) {
    if (actual.size() != expected.size())
        return false;
    float maximum = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i)
        maximum = std::max(maximum, std::fabs(actual[i] - expected[i]));
    std::printf("tiny MoE %s CPU/CUDA max abs error: %.7f\n", label,
                maximum);
    return maximum <= tolerance;
}

bool compare_package(const char* package, const char* architecture,
                     const char* label) {
    std::vector<float> cpu_prefill;
    std::vector<float> cpu_decode;
    std::vector<float> cuda_prefill;
    std::vector<float> cuda_decode;
    if (!run(package, architecture, Device::CPU, cpu_prefill, cpu_decode)) {
        std::fprintf(stderr, "tiny %s CPU reference failed\n", label);
        return false;
    }
    if (!run(package, architecture, Device::CUDA,
             cuda_prefill, cuda_decode)) {
        std::fprintf(stderr, "tiny %s CUDA inference failed\n", label);
        return false;
    }
    char prefill_label[96];
    char decode_label[96];
    std::snprintf(prefill_label, sizeof(prefill_label), "%s prefill", label);
    std::snprintf(decode_label, sizeof(decode_label), "%s decode", label);
    return close_enough(
               cuda_prefill, cpu_prefill, 4e-2f, prefill_label) &&
        close_enough(cuda_decode, cpu_decode, 4e-2f, decode_label);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(
            stderr,
            "usage: %s <qwen3-moe-w4g32.mollm> "
            "<qwen3.5-moe-w4g32.mollm> <qwen3-moe-w8g32.mollm> "
            "<hash-moe-fp16.mollm>\n",
            argv[0]);
        return 2;
    }
    CudaBackend probe;
    if (!probe.available()) {
        std::fprintf(stderr, "CUDA device unavailable; skipping\n");
        return 77;
    }
    if (!compare_package(argv[1], "qwen3-moe", "Qwen3-MoE W4") ||
        !compare_package(
            argv[2], "qwen3.5-moe", "Qwen3.5-MoE W4") ||
        !compare_package(argv[3], "qwen3-moe", "Qwen3-MoE W8") ||
        !compare_package(argv[4], "qwen3-moe", "hash/HC FP16"))
        return 1;
    std::printf("Tiny CUDA MoE E2E tests passed\n");
    return 0;
}
