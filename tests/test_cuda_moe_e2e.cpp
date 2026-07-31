#include "engine/engine.h"
#include "engine/cuda_backend.h"

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

bool run(const char* package, Device device,
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
        architecture->second != "qwen3-moe")
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <qwen3-moe-w4g32.mollm>\n", argv[0]);
        return 2;
    }
    CudaBackend probe;
    if (!probe.available()) {
        std::fprintf(stderr, "CUDA device unavailable; skipping\n");
        return 77;
    }
    std::vector<float> cpu_prefill;
    std::vector<float> cpu_decode;
    std::vector<float> cuda_prefill;
    std::vector<float> cuda_decode;
    if (!run(argv[1], Device::CPU, cpu_prefill, cpu_decode)) {
        std::fprintf(stderr, "tiny MoE CPU reference failed\n");
        return 1;
    }
    if (!run(argv[1], Device::CUDA, cuda_prefill, cuda_decode)) {
        std::fprintf(stderr, "tiny MoE CUDA inference failed\n");
        return 1;
    }
    if (!close_enough(cuda_prefill, cpu_prefill, 4e-2f, "prefill") ||
        !close_enough(cuda_decode, cpu_decode, 4e-2f, "decode"))
        return 1;
    std::printf("Tiny Qwen3-MoE CUDA E2E test passed\n");
    return 0;
}
