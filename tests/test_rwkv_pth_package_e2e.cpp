#include "engine/engine.h"

#ifdef MOLLM_CUDA
#include "engine/cuda_backend.h"
#endif

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
    config.device_fallback = device == Device::CPU
        ? DeviceFallbackPolicy::ALLOW_CPU
        : DeviceFallbackPolicy::REQUIRE_REQUESTED;
    config.operator_fallback = device == Device::CUDA
        ? OperatorFallbackPolicy::REQUIRE_NATIVE
        : OperatorFallbackPolicy::ALLOW_REFERENCE;
    config.n_ctx = 16;
    config.num_threads = 1;
    config.weight_loading = device == Device::CUDA
        ? WeightLoadingMode::RESIDENT
        : WeightLoadingMode::MMAP;
    if (!engine.load(config) || engine.config().device != device)
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
    const bool reset_valid = engine.past_len() == 0 &&
        copy_finite(engine.prefill_hidden({1, 2, 3}), reset_prefill) &&
        copy_finite(engine.decode_hidden(4), reset_decode) &&
        reset_prefill == prefill && reset_decode == first_decode &&
        engine.past_len() == 4;
    if (!reset_valid)
        return false;
    if (device == Device::CUDA) {
        const auto stats = engine.backend_operator_stats();
        if (!stats.tracked || stats.native_calls == 0 ||
            stats.fallback_calls != 0) {
            std::fprintf(
                stderr,
                "RWKV7 CUDA operator coverage is native=%llu fallback=%llu\n",
                static_cast<unsigned long long>(stats.native_calls),
                static_cast<unsigned long long>(stats.fallback_calls));
            return false;
        }
    }
    return true;
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

bool test_cuda_fallback_policy(const char* package) {
    EngineConfig config;
    config.package_path = package;
    config.device = Device::CUDA;
    config.n_ctx = 16;
    config.num_threads = 1;
    config.weight_loading = WeightLoadingMode::MMAP;
    config.device_fallback = DeviceFallbackPolicy::REQUIRE_REQUESTED;
    LLMEngine strict;
    if (strict.load(config)) {
        std::fprintf(
            stderr, "unavailable required CUDA device unexpectedly loaded\n");
        return false;
    }

    config.device_fallback = DeviceFallbackPolicy::ALLOW_CPU;
    LLMEngine fallback;
    if (!fallback.load(config) ||
        fallback.config().device != Device::CPU) {
        std::fprintf(stderr, "permitted CUDA-to-CPU fallback failed\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <rwkv7.mollm>\n", argv[0]);
        return 2;
    }
    std::vector<float> cpu_prefill;
    std::vector<float> cpu_first_decode;
    std::vector<float> cpu_second_decode;
    if (!run(argv[1], Device::CPU, cpu_prefill, cpu_first_decode,
             cpu_second_decode)) {
        std::fprintf(stderr, "tiny RWKV7 CPU package inference failed\n");
        return 1;
    }

#ifdef MOLLM_CUDA
    CudaBackend probe;
    if (!probe.available()) {
        if (!test_cuda_fallback_policy(argv[1]))
            return 1;
        std::printf(
            "Tiny official-layout RWKV7 .pth CPU package E2E passed "
            "(CUDA device unavailable)\n");
        return 0;
    }

    std::vector<float> cuda_prefill;
    std::vector<float> cuda_first_decode;
    std::vector<float> cuda_second_decode;
    if (!run(argv[1], Device::CUDA, cuda_prefill, cuda_first_decode,
             cuda_second_decode)) {
        std::fprintf(stderr, "tiny RWKV7 CUDA package inference failed\n");
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
    std::printf("Tiny official-layout RWKV7 .pth CPU/CUDA E2E passed\n");
#else
    if (!test_cuda_fallback_policy(argv[1]))
        return 1;
    std::printf("Tiny official-layout RWKV7 .pth CPU package E2E passed\n");
#endif
    return 0;
}
