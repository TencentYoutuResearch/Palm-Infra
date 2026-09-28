// Hy4 numerical probe for explicit/absorbed and full-model packages.
// Optional chunking supports long-context checks without a large prompt batch.
#include "engine/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s PACKAGE N_CTX OUTPUT_PREFIX [--prefill-only] [--skip-reset] [--pin-gguf-dense] [--ssd-cache-mb N] [--threads N] [--prompt-tokens N] [--chunk-size N]\n", argv[0]);
        return 2;
    }
    bool prefill_only = false;
    bool pin_gguf_dense = false;
    bool skip_reset = false;
    int ssd_cache_mb = 0;
    int threads = 4;
    int prompt_tokens = 0;
    int chunk_size = 0;
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--prefill-only") prefill_only = true;
        else if (std::string(argv[i]) == "--skip-reset") skip_reset = true;
        else if (std::string(argv[i]) == "--pin-gguf-dense") pin_gguf_dense = true;
        else if (std::string(argv[i]) == "--ssd-cache-mb" && i + 1 < argc)
            ssd_cache_mb = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--threads" && i + 1 < argc)
            threads = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--prompt-tokens" && i + 1 < argc) {
            prompt_tokens = std::atoi(argv[++i]);
            if (prompt_tokens < 1) return 2;
        } else if (std::string(argv[i]) == "--chunk-size" && i + 1 < argc) {
            chunk_size = std::atoi(argv[++i]);
            if (chunk_size < 1) return 2;
        }
        else return 2;
    }
    if (threads < 1 || ssd_cache_mb < 0) return 2;
    EngineConfig cfg;
    cfg.package_path = argv[1];
    cfg.n_ctx = std::atoi(argv[2]);
    cfg.num_threads = threads;
    cfg.weight_loading = WeightLoadingMode::MMAP;
    cfg.pin_gguf_dense_weights = pin_gguf_dense;
    cfg.moe_ssd_cache_bytes = static_cast<size_t>(std::max(0, ssd_cache_mb)) *
                              1024 * 1024;
    cfg.moe_ssd_cross_layer_prefetch = false;
    std::vector<int> tokens = {120000, 100, 200};
    if (prompt_tokens) {
        if (prompt_tokens > cfg.n_ctx) return 2;
        tokens.resize(prompt_tokens);
        tokens[0] = 120000;
        for (int i = 1; i < prompt_tokens; ++i) tokens[i] = 99 + i;
    }
    if (cfg.n_ctx < (int)tokens.size() + (prefill_only ? 0 : 1)) return 2;
    if (chunk_size == 0) chunk_size = (int)tokens.size();
    const int last_chunk_tokens = ((int)tokens.size() - 1) % chunk_size + 1;
    LLMEngine engine;
    if (!engine.load(cfg)) return 1;
    auto run_prefill = [&](const char* phase) {
        const auto start = std::chrono::steady_clock::now();
        Tensor hidden;
        for (int offset = 0; offset < (int)tokens.size(); offset += chunk_size) {
            const int end = std::min(offset + chunk_size, (int)tokens.size());
            hidden = engine.prefill_hidden(
                std::vector<int>(tokens.begin() + offset, tokens.begin() + end));
            if (!hidden.data) return Tensor();
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            std::fprintf(stderr, "%s: %d/%zu tokens, %.2f s\n",
                         phase, end, tokens.size(), seconds);
        }
        return hidden;
    };
    std::vector<float> baseline;
    auto dump = [&](const char* suffix, const Tensor& hidden, int n_tokens,
                    bool save_baseline = false) {
        if (!hidden.data) return false;
        std::vector<float> logits = engine.run_lmhead_raw(hidden, n_tokens);
        if (logits.empty()) return false;
        if (!std::all_of(logits.begin(), logits.end(),
                         [](float value) { return std::isfinite(value); })) {
            std::fprintf(stderr, "%s: non-finite logits\n", suffix);
            return false;
        }
        if (save_baseline) baseline = logits;
        std::string path = std::string(argv[3]) + suffix;
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        const size_t written = std::fwrite(logits.data(), sizeof(float), logits.size(), f);
        std::fclose(f);
        std::printf("%s: %zu logits\n", path.c_str(), written);
        return written == logits.size();
    };
    Tensor prefill = run_prefill("prefill");
    if (!dump("_prefill.f32", prefill, last_chunk_tokens, true)) return 1;
    if (prefill_only) return 0;
    Tensor decode = engine.decode_hidden(100);
    if (!dump("_decode.f32", decode, 1)) return 1;
    if (skip_reset) return 0;
    engine.reset();
    Tensor repeated = run_prefill("reset prefill");
    if (!repeated.data) return 1;
    const std::vector<float> again = engine.run_lmhead_raw(
        repeated, last_chunk_tokens);
    if (again.size() != baseline.size()) return 1;
    float max_delta = 0.f;
    for (size_t i = 0; i < again.size(); ++i) {
        if (!std::isfinite(again[i])) return 1;
        max_delta = std::max(max_delta, std::fabs(again[i] - baseline[i]));
    }
    std::printf("reset prefill max logit delta: %g\n", max_delta);
    if (max_delta > 1e-4f) return 1;
    return 0;
}
