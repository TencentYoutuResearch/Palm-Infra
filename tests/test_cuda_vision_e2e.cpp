#include "engine/cuda_backend.h"
#include "engine/engine.h"
#include "kernels/matmul.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct VisionResult {
    VisionEmbedding embedding;
    int next_token = -1;
    std::vector<float> decode_hidden;
    std::vector<float> decode_logits;
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

bool copy_finite(const Tensor& tensor, std::vector<float>& values) {
    if (!tensor.data || tensor.prec != Precision::FP32)
        return false;
    values.assign(
        tensor.ptr<float>(), tensor.ptr<float>() + tensor.nelements());
    return !values.empty() &&
        std::all_of(values.begin(), values.end(), [](float value) {
            return std::isfinite(value);
        });
}

bool run_text_continuation(LLMEngine& engine, int image_token_id,
                           VisionResult& result) {
    std::vector<int> tokens(
        static_cast<size_t>(result.embedding.tokens), image_token_id);
    std::string error;
    result.next_token = engine.prefill_with_image(
        tokens, image_token_id, result.embedding, &error);
    if (result.next_token < 0) {
        std::fprintf(stderr, "vision prefill failed: %s\n", error.c_str());
        return false;
    }
    if (engine.past_len() != result.embedding.tokens) {
        std::fprintf(
            stderr, "vision prefill state mismatch: past=%d tokens=%d\n",
            engine.past_len(), result.embedding.tokens);
        return false;
    }
    Tensor hidden = engine.decode_hidden(result.next_token);
    if (!copy_finite(hidden, result.decode_hidden)) {
        std::fprintf(stderr, "vision continuation hidden state is invalid\n");
        return false;
    }
    result.decode_logits = engine.run_lmhead_raw(hidden);
    if (result.decode_logits.empty() ||
        !std::all_of(
            result.decode_logits.begin(), result.decode_logits.end(),
            [](float value) { return std::isfinite(value); })) {
        std::fprintf(stderr, "vision continuation logits are invalid\n");
        return false;
    }
    if (engine.past_len() != result.embedding.tokens + 1) {
        std::fprintf(
            stderr, "vision decode state mismatch: past=%d expected=%d\n",
            engine.past_len(), result.embedding.tokens + 1);
        return false;
    }
    return true;
}

bool exactly_matches(const VisionResult& actual,
                     const VisionResult& expected) {
    return actual.next_token == expected.next_token &&
        actual.decode_hidden == expected.decode_hidden &&
        actual.decode_logits == expected.decode_logits;
}

bool compare_values(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    float max_tolerance, float rms_tolerance,
                    const char* label, bool require_same_top1) {
    if (actual.size() != expected.size() || actual.empty()) {
        std::fprintf(stderr, "%s shape mismatch\n", label);
        return false;
    }
    float maximum = 0.0f;
    double squared_error = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float error = actual[index] - expected[index];
        maximum = std::max(maximum, std::fabs(error));
        squared_error += static_cast<double>(error) * error;
    }
    const double rms = std::sqrt(
        squared_error / static_cast<double>(actual.size()));
    const size_t actual_top1 = static_cast<size_t>(
        std::max_element(actual.begin(), actual.end()) - actual.begin());
    const size_t expected_top1 = static_cast<size_t>(
        std::max_element(expected.begin(), expected.end()) - expected.begin());
    std::printf("  INFO: %s max_abs=%.7g rms=%.7g", label, maximum, rms);
    if (require_same_top1)
        std::printf(" top1=%zu/%zu", actual_top1, expected_top1);
    std::printf("\n");
    if (maximum > max_tolerance || rms > rms_tolerance ||
        (require_same_top1 && actual_top1 != expected_top1)) {
        std::fprintf(stderr, "%s exceeds tolerance\n", label);
        return false;
    }
    return true;
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
    constexpr int grid_w = 6;
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
    const int merge = metadata_int(engine, "vision_spatial_merge_size", 2);
    if (merge <= 0 || grid_h % merge != 0 || grid_w % merge != 0 ||
        result.embedding.tokens !=
            grid_t * grid_h * grid_w / (merge * merge)) {
        const int expected_tokens = merge > 0
            ? grid_t * grid_h * grid_w / (merge * merge) : -1;
        std::fprintf(
            stderr, "non-square vision token count mismatch: got=%d "
                    "expected=%d\n",
            result.embedding.tokens, expected_tokens);
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
    if (image_token_id < 0 ||
        !run_text_continuation(engine, image_token_id, result))
        return false;
    if (repeat_encoder) {
        engine.reset();
        if (engine.past_len() != 0) {
            std::fprintf(stderr, "vision reset did not clear model state\n");
            return false;
        }
        VisionResult replay;
        replay.embedding = result.embedding;
        if (!run_text_continuation(engine, image_token_id, replay) ||
            !exactly_matches(replay, result)) {
            std::fprintf(
                stderr, "CUDA vision continuation changed after reset\n");
            return false;
        }
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
    const bool explicit_real_model = argc == 3 &&
        std::strcmp(argv[1], "--real-model") == 0;
    if (argc > 3 || (argc == 3 && !explicit_real_model)) {
        std::fprintf(
            stderr,
            "usage: %s [tiny.mollm]\n"
            "       %s --real-model <qwen35-vl.mollm>\n",
            argv[0], argv[0]);
        return 2;
    }
    const char* environment_package =
        std::getenv("MOLLM_QWEN35_VL_PACKAGE");
    const char* package = explicit_real_model
        ? argv[2] : (argc == 2 ? argv[1] : environment_package);
    const bool real_model = explicit_real_model ||
        (argc == 1 && environment_package && *environment_package);
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
        cpu.embedding.grid_t != cuda.embedding.grid_t ||
        cpu.embedding.grid_h != cuda.embedding.grid_h ||
        cpu.embedding.grid_w != cuda.embedding.grid_w ||
        cpu.embedding.values.size() != cuda.embedding.values.size()) {
        std::fprintf(stderr, "CPU/CUDA vision output shape mismatch\n");
        return 1;
    }
    if (cpu.next_token != cuda.next_token) {
        std::fprintf(
            stderr, "CPU/CUDA vision next token mismatch: %d vs %d\n",
            cpu.next_token, cuda.next_token);
        return 1;
    }

    // The tiny fixture keeps a strict operator-level contract. A real
    // 12-layer tower compounds the intentional precision boundary difference:
    // the x86 validation path retains FP32 activations for FP16 weights, while
    // cuBLAS consumes checkpoint-style FP16 activations. RMS and top-1 remain
    // the primary real-model guards; the max bound catches isolated corruption.
    const float embedding_max = real_model ? 2e-2f : 2e-3f;
    const float embedding_rms = 1e-3f;
    const float hidden_max = real_model ? 3.5e-1f : 3e-3f;
    const float hidden_rms = real_model ? 3.5e-2f : 2e-3f;
    const float logits_max = real_model ? 1.2e-1f : 2e-3f;
    const float logits_rms = real_model ? 2.5e-2f : 2e-3f;
    bool values_match = compare_values(
        cuda.embedding.values, cpu.embedding.values,
        embedding_max, embedding_rms,
        "vision embedding CPU/CUDA", false);
    values_match = compare_values(
        cuda.decode_hidden, cpu.decode_hidden,
        hidden_max, hidden_rms,
        "vision decode hidden CPU/CUDA", false) &&
        values_match;
    values_match = compare_values(
        cuda.decode_logits, cpu.decode_logits,
        logits_max, logits_rms,
        "vision decode logits CPU/CUDA", true) &&
        values_match;
    if (!values_match) {
        return 1;
    }
    std::printf(
        "Qwen3.5 vision CPU/CUDA matched: tokens=%d hidden=%d "
        "next_token=%d\n",
        cpu.embedding.tokens, cpu.embedding.hidden_size, cpu.next_token);
    return 0;
}
