#include "engine/cuda_backend.h"
#include "engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct InferenceResult {
    std::vector<float> prefill_hidden;
    std::vector<float> prefill_logits;
    std::vector<float> first_decode_hidden;
    std::vector<float> first_decode_logits;
    std::vector<float> second_decode_hidden;
    std::vector<float> second_decode_logits;
};

bool copy_finite(const Tensor& tensor, std::vector<float>& values) {
    if (!tensor.data || tensor.prec != Precision::FP32)
        return false;
    values.assign(
        tensor.ptr<float>(), tensor.ptr<float>() + tensor.nelements());
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

bool finite(const std::vector<float>& values) {
    return !values.empty() &&
        std::all_of(values.begin(), values.end(), [](float value) {
            return std::isfinite(value);
        });
}

bool run(const char* package, Device device, InferenceResult& result) {
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
    config.n_ctx = 8;
    config.num_threads = 1;
    config.weight_loading = WeightLoadingMode::MMAP;
    if (!engine.load(config) || engine.config().device != device ||
        !engine.package_weights_mmap_backed()) {
        return false;
    }
    if (device == Device::CUDA && engine.cpu_weight_sidecar_bytes() != 0)
        return false;
    const auto& metadata = engine.package_metadata();
    const auto architecture = metadata.find("architecture");
    const auto quantization = metadata.find("quantization");
    const auto package_context = metadata.find("n_ctx");
    if (architecture == metadata.end() ||
        architecture->second != "deepseek-v4" ||
        quantization == metadata.end() ||
        quantization->second != "native-fp8-mxfp4" ||
        package_context == metadata.end() || package_context->second != "8") {
        return false;
    }

    Tensor prefill = engine.prefill_hidden({1, 2, 3, 4});
    if (!copy_finite(prefill, result.prefill_hidden) ||
        engine.past_len() != 4) {
        return false;
    }
    result.prefill_logits = engine.run_lmhead_raw(prefill, 4);
    if (!finite(result.prefill_logits))
        return false;

    Tensor first_decode = engine.decode_hidden(5);
    if (!copy_finite(first_decode, result.first_decode_hidden) ||
        engine.past_len() != 5) {
        return false;
    }
    result.first_decode_logits = engine.run_lmhead_raw(first_decode);
    if (!finite(result.first_decode_logits))
        return false;

    Tensor second_decode = engine.decode_hidden(6);
    if (!copy_finite(second_decode, result.second_decode_hidden) ||
        engine.past_len() != 6) {
        return false;
    }
    result.second_decode_logits = engine.run_lmhead_raw(second_decode);
    if (!finite(result.second_decode_logits))
        return false;

    engine.reset();
    std::vector<float> reset_prefill;
    std::vector<float> reset_decode;
    const bool reset_valid = engine.past_len() == 0 &&
        copy_finite(engine.prefill_hidden({1, 2, 3, 4}), reset_prefill) &&
        copy_finite(engine.decode_hidden(5), reset_decode) &&
        reset_prefill == result.prefill_hidden &&
        reset_decode == result.first_decode_hidden &&
        engine.past_len() == 5;
    if (!reset_valid)
        return false;

    if (device == Device::CUDA) {
        const auto stats = engine.backend_operator_stats();
        if (!stats.tracked || stats.native_calls == 0 ||
            stats.fallback_calls != 0) {
            std::fprintf(
                stderr,
                "DeepSeek-V4 converter CUDA coverage is native=%llu "
                "fallback=%llu\n",
                static_cast<unsigned long long>(stats.native_calls),
                static_cast<unsigned long long>(stats.fallback_calls));
            return false;
        }
    }
    return true;
}

bool close_enough(const std::vector<float>& actual,
                  const std::vector<float>& expected, float tolerance,
                  const char* label, bool require_same_top1 = false) {
    if (actual.size() != expected.size() || actual.empty())
        return false;
    float maximum = 0.0f;
    double squared_error = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float error = actual[index] - expected[index];
        maximum = std::max(maximum, std::fabs(error));
        squared_error += static_cast<double>(error) * error;
    }
    const double rms = std::sqrt(
        squared_error / static_cast<double>(actual.size()));
    const auto actual_top1 = static_cast<size_t>(
        std::max_element(actual.begin(), actual.end()) - actual.begin());
    const auto expected_top1 = static_cast<size_t>(
        std::max_element(expected.begin(), expected.end()) - expected.begin());
    std::printf(
        "tiny DeepSeek-V4 %s CPU/CUDA max abs error: %.7f, RMS: %.7g",
        label, maximum, rms);
    if (require_same_top1)
        std::printf(", top1: %zu/%zu", actual_top1, expected_top1);
    std::printf("\n");
    return maximum <= tolerance &&
        (!require_same_top1 || actual_top1 == expected_top1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <deepseek-v4.mollm>\n", argv[0]);
        return 2;
    }

    InferenceResult cpu;
    if (!run(argv[1], Device::CPU, cpu)) {
        std::fprintf(
            stderr, "tiny converted DeepSeek-V4 CPU inference failed\n");
        return 1;
    }

    CudaBackend probe;
    if (!probe.available()) {
        std::fprintf(stderr, "CUDA device unavailable; skipping\n");
        return 77;
    }

    InferenceResult cuda;
    if (!run(argv[1], Device::CUDA, cuda)) {
        std::fprintf(
            stderr, "tiny converted DeepSeek-V4 CUDA inference failed\n");
        return 1;
    }

    bool valid = close_enough(
        cuda.prefill_hidden, cpu.prefill_hidden, 6e-2f,
        "prefill hidden");
    valid &= close_enough(
        cuda.prefill_logits, cpu.prefill_logits, 6e-2f,
        "prefill logits", true);
    valid &= close_enough(
        cuda.first_decode_hidden, cpu.first_decode_hidden, 6e-2f,
        "first decode hidden");
    valid &= close_enough(
        cuda.first_decode_logits, cpu.first_decode_logits, 6e-2f,
        "first decode logits", true);
    valid &= close_enough(
        cuda.second_decode_hidden, cpu.second_decode_hidden, 6e-2f,
        "second decode hidden");
    valid &= close_enough(
        cuda.second_decode_logits, cpu.second_decode_logits, 6e-2f,
        "second decode logits", true);
    if (!valid)
        return 1;
    std::printf(
        "Native DeepSeek-V4 checkpoint converter CPU/CUDA E2E passed\n");
    return 0;
}
