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
    if (!copy_finite(engine.decode_hidden(5), second_decode) ||
        engine.past_len() != 5)
        return false;
    engine.reset();
    std::vector<float> reset_prefill;
    std::vector<float> reset_decode;
    return engine.past_len() == 0 &&
        copy_finite(engine.prefill_hidden({1, 2, 3}), reset_prefill) &&
        copy_finite(engine.decode_hidden(4), reset_decode) &&
        reset_prefill == prefill && reset_decode == first_decode &&
        engine.past_len() == 4;
}

bool close_enough(const std::vector<float>& actual,
                  const std::vector<float>& expected, float tolerance,
                  const char* label) {
    if (actual.size() != expected.size())
        return false;
    float maximum = 0.0f;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float error = actual[index] - expected[index];
        maximum = std::max(maximum, std::fabs(error));
        squared_error += static_cast<double>(error) * error;
        squared_reference +=
            static_cast<double>(expected[index]) * expected[index];
    }
    const double rms_error = std::sqrt(
        squared_error / static_cast<double>(actual.size()));
    const double rms_reference = std::sqrt(
        squared_reference / static_cast<double>(actual.size()));
    const double relative_rms = rms_reference > 0.0
        ? rms_error / rms_reference : rms_error;
    std::printf(
        "tiny RWKV7 %s CPU/CUDA max abs error: %.7f, RMS: %.7g "
        "(relative %.7g)\n",
        label, maximum, rms_error, relative_rms);
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
    bool valid = close_enough(
        cuda_prefill, cpu_prefill, 2e-3f, "prefill");
    valid &= close_enough(
        cuda_first_decode, cpu_first_decode, 2e-3f, "first decode");
    valid &= close_enough(
        cuda_second_decode, cpu_second_decode, 2e-3f, "second decode");
    if (!valid)
        return 1;
    std::printf("Tiny official-layout RWKV7 .pth CUDA E2E passed\n");
    return 0;
}
