#include "engine/engine.h"
#ifdef MOLLM_CUDA
#include "engine/cuda_backend.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL: %s\n", message);                  \
            ++failures;                                                     \
        } else {                                                            \
            std::printf("  PASS: %s\n", message);                        \
        }                                                                   \
    } while (0)

struct InferenceResult {
    std::vector<float> prefill_hidden;
    std::vector<float> prefill_logits;
    std::vector<float> decode_hidden;
    std::vector<float> decode_logits;
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

InferenceResult run_package(const char* path, const char* label,
                            const char* architecture, bool use_mmap,
                            Device device = Device::CPU) {
    InferenceResult result;
    LLMEngine engine;
    EngineConfig config;
    config.package_path = path;
    config.device = device;
    config.device_fallback = device == Device::CPU
        ? DeviceFallbackPolicy::ALLOW_CPU
        : DeviceFallbackPolicy::REQUIRE_REQUESTED;
    config.operator_fallback = device == Device::CUDA
        ? OperatorFallbackPolicy::REQUIRE_NATIVE
        : OperatorFallbackPolicy::ALLOW_REFERENCE;
    config.n_ctx = 8;
    config.num_threads = 1;
    config.weight_loading = use_mmap
        ? WeightLoadingMode::MMAP : WeightLoadingMode::RESIDENT;
    config.sampling.temperature = 0.0f;

    CHECK(engine.load(config), label);
    if (failures)
        return result;
    CHECK(engine.config().device == device, "requested device remains active");
    const auto metadata = engine.package_metadata();
    const auto architecture_it = metadata.find("architecture");
    CHECK(architecture_it != metadata.end() &&
              (!architecture || architecture_it->second == architecture),
          "package architecture is preserved");
    CHECK(engine.package_weights_mmap_backed() ==
              (use_mmap || device == Device::CUDA),
          "requested package weight loading mode is honored");
    CHECK(engine.prefill({1, 2, 3}) >= 0, "prefill executes");
    CHECK(engine.past_len() == 3, "prefill advances KV state");

    CHECK(engine.decode(4) >= 0, "decode executes");
    CHECK(engine.past_len() == 4, "decode advances KV state");

    engine.reset();
    CHECK(engine.past_len() == 0, "reset clears KV state");
    Tensor hidden = engine.prefill_hidden({5, 6});
    const int hidden_size = metadata_int(engine, "hidden_size", -1);
    CHECK(hidden.data && hidden.shape[0] == hidden_size &&
              hidden.shape[1] == 2,
          "prefill_hidden returns expected model shape");
    CHECK(copy_finite(hidden, result.prefill_hidden),
          "prefill hidden states are finite");
    result.prefill_logits = engine.run_lmhead_raw(hidden, 2);
    CHECK(finite(result.prefill_logits), "prefill logits are finite");
    Tensor decoded = engine.decode_hidden(7);
    CHECK(copy_finite(decoded, result.decode_hidden),
          "decode hidden states are finite");
    result.decode_logits = engine.run_lmhead_raw(decoded);
    CHECK(finite(result.decode_logits), "decode logits are finite");
    CHECK(engine.past_len() == 3, "hidden inference advances recurrent state");
    return result;
}

bool compare_values(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    float max_tolerance, float rms_tolerance,
                    const char* label, bool require_same_top1) {
    if (actual.size() != expected.size() || actual.empty()) {
        std::fprintf(stderr, "FAIL: %s shape mismatch\n", label);
        ++failures;
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
    const bool valid = maximum <= max_tolerance && rms <= rms_tolerance &&
        (!require_same_top1 || actual_top1 == expected_top1);
    CHECK(valid, label);
    return valid;
}

void compare_cpu_cuda(const InferenceResult& cpu,
                      const InferenceResult& cuda, const char* label,
                      bool real_model) {
    const float hidden_max = real_model ? 3e-2f : 3e-3f;
    const float hidden_rms = real_model ? 6e-3f : 2e-3f;
    const float logits_max = real_model ? 2e-2f : 2e-3f;
    const float logits_rms = real_model ? 5e-3f : 2e-3f;
    const std::string prefix(label);
    compare_values(cuda.prefill_hidden, cpu.prefill_hidden,
                   hidden_max, hidden_rms,
                   (prefix + " prefill hidden CPU/CUDA").c_str(), false);
    compare_values(cuda.prefill_logits, cpu.prefill_logits,
                   logits_max, logits_rms,
                   (prefix + " prefill logits CPU/CUDA").c_str(), true);
    compare_values(cuda.decode_hidden, cpu.decode_hidden,
                   hidden_max, hidden_rms,
                   (prefix + " decode hidden CPU/CUDA").c_str(), false);
    compare_values(cuda.decode_logits, cpu.decode_logits,
                   logits_max, logits_rms,
                   (prefix + " decode logits CPU/CUDA").c_str(), true);
}

void test_unavailable_cuda_policy(const char* path) {
#ifdef MOLLM_CUDA
    CudaBackend probe;
    if (probe.available()) {
        std::printf(
            "  SKIP: CUDA fallback policy needs a host without an available GPU\n");
        return;
    }
#endif

    EngineConfig strict_config;
    strict_config.package_path = path;
    strict_config.device = Device::CUDA;
    strict_config.device_fallback = DeviceFallbackPolicy::REQUIRE_REQUESTED;
    LLMEngine strict_engine;
    CHECK(!strict_engine.load(strict_config),
          "strict CUDA request fails when CUDA is unavailable");

    EngineConfig fallback_config = strict_config;
    fallback_config.device_fallback = DeviceFallbackPolicy::ALLOW_CPU;
    LLMEngine fallback_engine;
    CHECK(fallback_engine.load(fallback_config),
          "permissive CUDA request falls back when CUDA is unavailable");
    CHECK(fallback_engine.config().device == Device::CPU,
          "fallback reports CPU as the active device");
    CHECK(!fallback_engine.package_weights_mmap_backed(),
          "unavailable CUDA fallback preserves resident CPU package storage");
}

std::vector<int> generate_tokens(const char* path, bool use_mtp,
                                 LLMEngine::MtpStats* stats,
                                 bool zero_prompt = false) {
    LLMEngine engine;
    EngineConfig config;
    config.package_path = path;
    config.n_ctx = 64;
    config.num_threads = 1;
    config.sampling.temperature = 0.0f;
    config.mtp_draft_tokens = use_mtp ? 1 : 0;

    CHECK(engine.load(config),
          use_mtp ? "load tiny Qwen3.5 MTP package"
                  : "load tiny Qwen3.5 baseline package");
    if (failures)
        return {};
    CHECK(engine.has_mtp() == use_mtp,
          "MTP graph loads only when speculation is enabled");

    constexpr int kGenerated = 24;
    std::vector<int> tokens;
    const std::vector<int> prompt = zero_prompt
        ? std::vector<int>{0, 0, 0}
        : std::vector<int>{1, 2, 3};
    int next = engine.prefill(prompt);
    CHECK(next >= 0, "Qwen3.5 generation prefill executes");
    if (next < 0)
        return {};
    tokens.push_back(next);

    while (static_cast<int>(tokens.size()) < kGenerated) {
        if (!use_mtp) {
            next = engine.decode(next);
            CHECK(next >= 0, "Qwen3.5 baseline decode executes");
            if (next < 0)
                return {};
            tokens.push_back(next);
            continue;
        }

        const int target_before = engine.past_len();
        const uint64_t accepted_before = engine.mtp_stats().accepted;
        std::vector<int> decoded;
        CHECK(engine.speculative_decode(
                  next, decoded, kGenerated - static_cast<int>(tokens.size())),
              "Qwen3.5 speculative decode executes");
        if (decoded.empty())
            return {};
        CHECK(engine.past_len() ==
                  target_before + static_cast<int>(decoded.size()),
              engine.mtp_stats().accepted > accepted_before
                  ? "accepted draft preserves target state continuity"
                  : "rejected draft restores target state continuity");
        const int remaining = kGenerated - static_cast<int>(tokens.size());
        if (static_cast<int>(decoded.size()) > remaining)
            decoded.resize(static_cast<size_t>(remaining));
        tokens.insert(tokens.end(), decoded.begin(), decoded.end());
        next = tokens.back();
    }

    if (stats)
        *stats = engine.mtp_stats();
    return tokens;
}

void check_mtp_draft_limit(const char* path) {
    LLMEngine engine;
    EngineConfig config;
    config.package_path = path;
    config.n_ctx = 64;
    config.num_threads = 1;
    config.sampling.temperature = 0.0f;
    config.mtp_draft_tokens = 2;
    CHECK(!engine.load(config),
          "Qwen3.5 package rejects unsupported multi-draft speculation");
}

}  // namespace

int main(int argc, char** argv) {
    const bool real_model = argc == 3 &&
        std::strcmp(argv[1], "--real-model") == 0;
    if (real_model) {
#ifdef MOLLM_CUDA
        CudaBackend probe;
        if (!probe.available()) {
            std::fprintf(stderr, "CUDA device unavailable; skipping\n");
            return 77;
        }
        const InferenceResult cpu = run_package(
            argv[2], "load real dense package on CPU", nullptr, true,
            Device::CPU);
        const InferenceResult cuda = run_package(
            argv[2], "load real dense package on CUDA", nullptr, false,
            Device::CUDA);
        compare_cpu_cuda(cpu, cuda, "real dense", true);
        if (failures == 0)
            std::printf("Real dense CPU/CUDA inference test passed\n");
        return failures == 0 ? 0 : 1;
#else
        std::fprintf(stderr, "CUDA backend is not compiled; skipping\n");
        return 77;
#endif
    }
    if (argc != 5) {
        std::fprintf(stderr,
                     "usage: %s <qwen3-fp16.mollm> <qwen3-w4.mollm> "
                     "<qwen35-reject.mollm> <qwen35-accept.mollm>\n"
                     "       %s --real-model <dense.mollm>\n",
                     argv[0], argv[0]);
        return 2;
    }

    const InferenceResult fp16 =
        run_package(argv[1], "load tiny Qwen3 FP16 package", "qwen3", false);
    const InferenceResult w4 =
        run_package(argv[2], "load tiny Qwen3 W4G32 package", "qwen3", true);
    CHECK(!fp16.prefill_hidden.empty() &&
              fp16.prefill_hidden.size() == w4.prefill_hidden.size(),
          "FP16 and W4 packages produce comparable hidden tensors");
    float max_error = 0.0f;
    for (size_t i = 0; i < fp16.prefill_hidden.size() &&
         i < w4.prefill_hidden.size(); ++i) {
        max_error = std::max(
            max_error,
            std::fabs(fp16.prefill_hidden[i] - w4.prefill_hidden[i]));
    }
    std::printf("  INFO: FP16/W4 hidden max error = %.7f\n", max_error);
    // This compares a complete random-weight transformer, so each of its five
    // W4 projections contributes quantization error.  The bound is a smoke
    // guard against layout/scale corruption rather than a W4 quality target.
    CHECK(max_error < 0.30f,
          "W4G32 package remains numerically bounded against FP16 reference");
    run_package(argv[3], "load tiny Qwen3.5 GDN package", "qwen3.5", true);
    check_mtp_draft_limit(argv[3]);

#ifdef MOLLM_CUDA
    CudaBackend probe;
    if (probe.available()) {
        const InferenceResult fp16_cuda = run_package(
            argv[1], "load tiny Qwen3 FP16 package on CUDA", "qwen3",
            false, Device::CUDA);
        const InferenceResult w4_cuda = run_package(
            argv[2], "load tiny Qwen3 W4G32 package on CUDA", "qwen3",
            false, Device::CUDA);
        compare_cpu_cuda(fp16, fp16_cuda, "tiny Qwen3 FP16", false);
        compare_cpu_cuda(w4, w4_cuda, "tiny Qwen3 W4G32", false);
    }
#endif
    test_unavailable_cuda_policy(argv[1]);

    LLMEngine::MtpStats reject_stats;
    const std::vector<int> reject_plain =
        generate_tokens(argv[3], false, nullptr);
    const std::vector<int> reject_mtp =
        generate_tokens(argv[3], true, &reject_stats);
    CHECK(!reject_plain.empty() && reject_plain == reject_mtp,
          "greedy Qwen3.5 rejected-draft output is identical with MTP");
    CHECK(reject_stats.drafted > 0 &&
              reject_stats.accepted < reject_stats.drafted,
          "Qwen3.5 fixture exercises rejected-draft rollback");

    LLMEngine::MtpStats accept_stats;
    const std::vector<int> accept_plain =
        generate_tokens(argv[4], false, nullptr, true);
    const std::vector<int> accept_mtp =
        generate_tokens(argv[4], true, &accept_stats, true);
    CHECK(!accept_plain.empty() && accept_plain == accept_mtp,
          "greedy Qwen3.5 accepted-draft output is identical with MTP");
    CHECK(accept_stats.drafted > 0 && accept_stats.accepted > 0,
          "Qwen3.5 fixture exercises accepted-draft commit");
    if (failures == 0)
        std::printf("Tiny CPU inference test passed\n");
    return failures == 0 ? 0 : 1;
}
