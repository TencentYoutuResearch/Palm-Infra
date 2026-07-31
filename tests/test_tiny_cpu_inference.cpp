#include "engine/engine.h"
#ifdef MOLLM_CUDA
#include "engine/cuda_backend.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
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

std::vector<float> run_package(const char* path, const char* label,
                               const char* architecture, bool use_mmap) {
    LLMEngine engine;
    EngineConfig config;
    config.package_path = path;
    config.n_ctx = 8;
    config.num_threads = 1;
    config.weight_loading =
        use_mmap ? WeightLoadingMode::MMAP : WeightLoadingMode::RESIDENT;
    config.sampling.temperature = 0.0f;

    CHECK(engine.load(config), label);
    if (failures)
        return {};
    const auto metadata = engine.package_metadata();
    const auto architecture_it = metadata.find("architecture");
    CHECK(architecture_it != metadata.end() &&
              architecture_it->second == architecture,
          "package architecture is preserved");
    CHECK(engine.package_weights_mmap_backed() == use_mmap,
          "requested package weight loading mode is honored");
    CHECK(engine.prefill({1, 2, 3}) >= 0, "prefill executes");
    CHECK(engine.past_len() == 3, "prefill advances KV state");

    CHECK(engine.decode(4) >= 0, "decode executes");
    CHECK(engine.past_len() == 4, "decode advances KV state");

    engine.reset();
    CHECK(engine.past_len() == 0, "reset clears KV state");
    Tensor hidden = engine.prefill_hidden({5, 6});
    CHECK(hidden.data && hidden.shape[0] == 32 && hidden.shape[1] == 2,
          "prefill_hidden returns expected tiny-model shape");
    bool finite = hidden.data != nullptr;
    if (finite) {
        const float* values = hidden.ptr<float>();
        for (size_t i = 0; i < static_cast<size_t>(hidden.shape[0] * hidden.shape[1]); ++i)
            finite = finite && std::isfinite(values[i]);
    }
    CHECK(finite, "hidden states are finite");
    if (!finite)
        return {};
    const float* values = hidden.ptr<float>();
    return std::vector<float>(
        values, values + static_cast<size_t>(hidden.shape[0] * hidden.shape[1]));
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
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s <qwen3-fp16.mollm> <qwen3-w4.mollm> "
                     "<qwen35-fp16.mollm>\n",
                     argv[0]);
        return 2;
    }

    const std::vector<float> fp16 =
        run_package(argv[1], "load tiny Qwen3 FP16 package", "qwen3", false);
    const std::vector<float> w4 =
        run_package(argv[2], "load tiny Qwen3 W4G32 package", "qwen3", true);
    CHECK(!fp16.empty() && fp16.size() == w4.size(),
          "FP16 and W4 packages produce comparable hidden tensors");
    float max_error = 0.0f;
    for (size_t i = 0; i < fp16.size() && i < w4.size(); ++i)
        max_error = std::max(max_error, std::fabs(fp16[i] - w4[i]));
    std::printf("  INFO: FP16/W4 hidden max error = %.7f\n", max_error);
    // This compares a complete random-weight transformer, so each of its five
    // W4 projections contributes quantization error.  The bound is a smoke
    // guard against layout/scale corruption rather than a W4 quality target.
    CHECK(max_error < 0.30f,
          "W4G32 package remains numerically bounded against FP16 reference");
    run_package(argv[3], "load tiny Qwen3.5 GDN package", "qwen3.5", true);
    test_unavailable_cuda_policy(argv[1]);
    if (failures == 0)
        std::printf("Tiny CPU inference test passed\n");
    return failures == 0 ? 0 : 1;
}
