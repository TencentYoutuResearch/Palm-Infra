// Optional real-checkpoint RWKV v7 regression.
//
// Set MOLLM_RWKV_PACKAGE to a package converted from the checked G1H fixture.
// FP16 packages are checked against PyTorch token sequences. Quantized CUDA
// packages are checked against CPU logits from the same package because weight
// quantization can legitimately change a near-tied greedy token.

#include "engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

EngineConfig config_for(const char* package, Device device) {
    EngineConfig config;
    config.package_path = package;
    config.num_threads = 4;
    config.n_ctx = 128;
    config.sampling.temperature = 0.f;
    config.device = device;
    if (device != Device::CPU)
        config.device_fallback = DeviceFallbackPolicy::REQUIRE_REQUESTED;
    if (device == Device::CUDA)
        config.operator_fallback = OperatorFallbackPolicy::REQUIRE_NATIVE;
    return config;
}

int argmax(const std::vector<float>& values) {
    return values.empty()
        ? -1
        : static_cast<int>(std::max_element(values.begin(), values.end()) -
                           values.begin());
}

std::vector<float> prefill_logits(LLMEngine& engine,
                                  const std::vector<int>& tokens) {
    Tensor hidden = engine.prefill_hidden(tokens);
    return hidden.data ? engine.run_lmhead_raw(hidden) : std::vector<float>{};
}

std::vector<float> decode_logits(LLMEngine& engine, int token) {
    Tensor hidden = engine.decode_hidden(token);
    return hidden.data ? engine.run_lmhead_raw(hidden) : std::vector<float>{};
}

bool compare_logits(const std::vector<float>& cpu,
                    const std::vector<float>& cuda, const char* label) {
    // Full-size recurrent models amplify harmless GEMM/GEMV accumulation
    // order differences across layers. These bounds retain a useful guard
    // against scale/layout/state errors while requiring identical greedy
    // choices at every checked step.
    constexpr float kMaximumAbsoluteError = 0.75f;
    constexpr double kMaximumRelativeRmsError = 1.5e-2;
    if (cpu.empty() || cpu.size() != cuda.size()) {
        std::fprintf(stderr,
                     "FAIL: %s logits shape mismatch CPU=%zu CUDA=%zu\n",
                     label, cpu.size(), cuda.size());
        return false;
    }
    double squared_error = 0.0;
    double squared_reference = 0.0;
    float maximum_error = 0.0f;
    for (size_t index = 0; index < cpu.size(); ++index) {
        if (!std::isfinite(cpu[index]) || !std::isfinite(cuda[index])) {
            std::fprintf(stderr, "FAIL: %s has non-finite logit at %zu\n",
                         label, index);
            return false;
        }
        const float error = std::fabs(cpu[index] - cuda[index]);
        maximum_error = std::max(maximum_error, error);
        squared_error += static_cast<double>(error) * error;
        squared_reference += static_cast<double>(cpu[index]) * cpu[index];
    }
    const double rms_error = std::sqrt(squared_error / cpu.size());
    const double reference_rms = std::sqrt(squared_reference / cpu.size());
    const double relative_rms = rms_error /
        std::max(reference_rms, std::numeric_limits<double>::min());
    std::printf("  %s CPU/CUDA logits: max_abs=%.7g rms=%.7g "
                "relative_rms=%.7g "
                "top1=%d/%d\n",
                label, maximum_error, rms_error, relative_rms,
                argmax(cpu), argmax(cuda));
    if (maximum_error > kMaximumAbsoluteError ||
        relative_rms > kMaximumRelativeRmsError ||
        argmax(cpu) != argmax(cuda)) {
        std::fprintf(stderr,
                     "FAIL: %s CPU/CUDA logits exceed tolerance\n", label);
        return false;
    }
    return true;
}

const std::vector<std::pair<std::vector<int>, std::vector<int>>>& cases() {
    static const std::vector<
        std::pair<std::vector<int>, std::vector<int>>> value = {
        // "Hello" in rwkv_vocab_v20230424.txt.
        {{33155}, {45, 308, 459, 332, 22168, 7152, 4811, 22590}},
        // "你好，请用一句话介绍你自己。". This catches non-ASCII tokenizer
        // and multi-token prefill regressions rather than checking only ASCII.
        {{10464, 11685, 19137, 16738, 14589, 10250, 11012, 16713,
          10382, 15484, 10464, 15847, 12144, 10080},
         {9823, 261, 9822, 12605, 13091, 10250, 11043, 15052,
          12217, 11098, 19137, 10264, 13773, 10339, 12266, 10997}},
    };
    return value;
}

bool test_pytorch_sequence(LLMEngine& engine) {
    for (size_t c = 0; c < cases().size(); ++c) {
        const auto& test_case = cases()[c];
        // Compare every prefix as a numerical-stability diagnostic. GEMM and
        // GEMV accumulate in a different order, so a near-tied logit may pick
        // a different token even when both paths match the reference sequence.
        for (size_t prefix_len = 1; prefix_len <= test_case.first.size();
             ++prefix_len) {
            engine.reset();
            int expected = engine.prefill({test_case.first.front()});
            for (size_t i = 1; i < prefix_len; ++i)
                expected = engine.decode(test_case.first[i]);

            engine.reset();
            int actual = engine.prefill(std::vector<int>(
                test_case.first.begin(),
                test_case.first.begin() + prefix_len));
            if (actual != expected) {
                std::printf(
                    "NOTE: case %zu prefix %zu batched got %d decode got %d\n",
                    c, prefix_len, actual, expected);
            }
        }

        engine.reset();
        int token = engine.prefill({test_case.first.front()});
        for (size_t i = 1; i < test_case.first.size(); ++i)
            token = engine.decode(test_case.first[i]);
        if (token != test_case.second.front()) {
            std::fprintf(stderr,
                         "FAIL: case %zu decode path got %d expected %d\n",
                         c, token, test_case.second.front());
            return false;
        }

        engine.reset();
        token = engine.prefill(test_case.first);
        for (size_t i = 0; i < test_case.second.size(); ++i) {
            if (token != test_case.second[i]) {
                std::fprintf(
                    stderr,
                    "FAIL: case %zu token %zu got %d expected %d\n",
                    c, i, token, test_case.second[i]);
                return false;
            }
            if (i + 1 < test_case.second.size())
                token = engine.decode(token);
        }
    }
    return true;
}

bool test_cuda_parity(const char* package, LLMEngine& cuda) {
    LLMEngine cpu;
    if (!cpu.load(config_for(package, Device::CPU))) {
        std::fprintf(stderr, "FAIL: could not load CPU reference package\n");
        return false;
    }
    for (size_t case_index = 0; case_index < cases().size(); ++case_index) {
        cpu.reset();
        cuda.reset();
        auto cpu_logits = prefill_logits(cpu, cases()[case_index].first);
        auto cuda_logits = prefill_logits(cuda, cases()[case_index].first);
        std::string label = "case " + std::to_string(case_index) + " prefill";
        if (!compare_logits(cpu_logits, cuda_logits, label.c_str()))
            return false;

        // Feed the same CPU-selected token into both recurrent states so a
        // numerical difference cannot turn into a different test input.
        for (int step = 0; step < 7; ++step) {
            const int token = argmax(cpu_logits);
            if (token < 0) {
                std::fprintf(stderr, "FAIL: empty CPU logits\n");
                return false;
            }
            cpu_logits = decode_logits(cpu, token);
            cuda_logits = decode_logits(cuda, token);
            label = "case " + std::to_string(case_index) + " decode " +
                    std::to_string(step + 1);
            if (!compare_logits(cpu_logits, cuda_logits, label.c_str()))
                return false;
        }
    }
    const BackendOperatorStats stats = cuda.backend_operator_stats();
    if (!stats.tracked || stats.native_calls == 0 ||
        stats.fallback_calls != 0) {
        std::fprintf(stderr,
                     "FAIL: CUDA coverage native=%llu fallback=%llu\n",
                     static_cast<unsigned long long>(stats.native_calls),
                     static_cast<unsigned long long>(stats.fallback_calls));
        return false;
    }
    return true;
}

} // namespace

int main() {
    const char* package = std::getenv("MOLLM_RWKV_PACKAGE");
    if (!package || !*package) {
        std::printf("SKIP: set MOLLM_RWKV_PACKAGE for RWKV v7 E2E\n");
        return 0;
    }

    const char* device = std::getenv("MOLLM_RWKV_DEVICE");
    Device selected_device = Device::CPU;
    if (device && std::strcmp(device, "metal") == 0)
        selected_device = Device::METAL;
    else if (device && std::strcmp(device, "cuda") == 0)
        selected_device = Device::CUDA;

    LLMEngine engine;
    if (!engine.load(config_for(package, selected_device))) {
        std::fprintf(stderr, "FAIL: could not load RWKV package\n");
        return 1;
    }

    const auto quantization_it =
        engine.package_metadata().find("quantization");
    const std::string quantization =
        quantization_it == engine.package_metadata().end()
            ? "fp16" : quantization_it->second;
    if (quantization == "fp16") {
        if (!test_pytorch_sequence(engine))
            return 1;
        std::printf("PASS: RWKV v7 FP16 PyTorch-reference token sequence\n");
        if (selected_device == Device::CUDA &&
            !test_cuda_parity(package, engine))
            return 1;
    } else if (selected_device == Device::CUDA) {
        if (!test_cuda_parity(package, engine))
            return 1;
        std::printf("PASS: RWKV v7 %s CPU/CUDA logit parity\n",
                    quantization.c_str());
    } else {
        engine.reset();
        const auto logits = prefill_logits(engine, cases().front().first);
        const bool finite = !logits.empty() && std::all_of(
            logits.begin(), logits.end(),
            [](float value) { return std::isfinite(value); });
        if (!finite || argmax(logits) < 0) {
            std::fprintf(stderr, "FAIL: quantized RWKV inference failed\n");
            return 1;
        }
        std::printf("PASS: RWKV v7 %s finite inference smoke test\n",
                    quantization.c_str());
    }
    return 0;
}
