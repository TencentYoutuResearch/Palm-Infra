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
    values.assign(
        tensor.ptr<float>(), tensor.ptr<float>() + tensor.nelements());
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

bool run(const char* package, Device device, std::vector<float>& prefill,
         std::vector<float>& first_decode,
         std::vector<float>& second_decode) {
    LLMEngine engine;
    EngineConfig config;
    config.package_path = package;
    config.device = device;
    config.n_ctx = 16;
    config.num_threads = 1;
    config.weight_loading = device == Device::CUDA
        ? WeightLoadingMode::RESIDENT
        : WeightLoadingMode::MMAP;
    if (!engine.load(config))
        return false;
    if (!engine.package_weights_mmap_backed())
        return false;
    const auto architecture = engine.package_metadata().find("architecture");
    if (architecture == engine.package_metadata().end() ||
        architecture->second != "rwkv7")
        return false;

    if (!copy_finite(engine.prefill_hidden({1, 2, 3}), prefill) ||
        engine.past_len() != 3)
        return false;
    if (!copy_finite(engine.decode_hidden(4), first_decode) ||
        engine.past_len() != 4)
        return false;
    return copy_finite(engine.decode_hidden(5), second_decode) &&
        engine.past_len() == 5;
}

bool close_enough(const std::vector<float>& actual,
                  const std::vector<float>& expected, float tolerance,
                  const char* label) {
    if (actual.size() != expected.size())
        return false;
    float maximum = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index)
        maximum = std::max(
            maximum, std::fabs(actual[index] - expected[index]));
    std::printf("tiny RWKV7 %s CPU/CUDA max abs error: %.7f\n", label,
                maximum);
    return maximum <= tolerance;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <rwkv7.mollm>\n", argv[0]);
        return 2;
    }
    CudaBackend probe;
    if (!probe.available()) {
        std::fprintf(stderr, "CUDA device unavailable; skipping\n");
        return 77;
    }

    std::vector<float> cpu_prefill;
    std::vector<float> cpu_first_decode;
    std::vector<float> cpu_second_decode;
    std::vector<float> cuda_prefill;
    std::vector<float> cuda_first_decode;
    std::vector<float> cuda_second_decode;
    if (!run(argv[1], Device::CPU, cpu_prefill, cpu_first_decode,
             cpu_second_decode) ||
        !run(argv[1], Device::CUDA, cuda_prefill, cuda_first_decode,
             cuda_second_decode)) {
        std::fprintf(stderr, "tiny RWKV7 package inference failed\n");
        return 1;
    }
    if (!close_enough(cuda_prefill, cpu_prefill, 2e-3f, "prefill") ||
        !close_enough(cuda_first_decode, cpu_first_decode, 2e-3f,
                      "first decode") ||
        !close_enough(cuda_second_decode, cpu_second_decode, 2e-3f,
                      "second decode"))
        return 1;
    std::printf("Tiny official-layout RWKV7 .pth CUDA E2E passed\n");
    return 0;
}
