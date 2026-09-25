// Two-layer Hy4 numerical probe: dump prefill and decode logits for the same
// fixed token sequence from either explicit or absorbed .mollm package.
#include "engine/engine.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4 || argc > 10) {
        std::fprintf(stderr, "usage: %s PACKAGE N_CTX OUTPUT_PREFIX [--prefill-only] [--pin-gguf-dense] [--ssd-cache-mb N] [--threads N]\n", argv[0]);
        return 2;
    }
    bool prefill_only = false;
    bool pin_gguf_dense = false;
    int ssd_cache_mb = 0;
    int threads = 4;
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--prefill-only") prefill_only = true;
        else if (std::string(argv[i]) == "--pin-gguf-dense") pin_gguf_dense = true;
        else if (std::string(argv[i]) == "--ssd-cache-mb" && i + 1 < argc)
            ssd_cache_mb = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--threads" && i + 1 < argc)
            threads = std::atoi(argv[++i]);
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
    LLMEngine engine;
    if (!engine.load(cfg)) return 1;
    const std::vector<int> tokens = {120000, 100, 200};
    std::vector<float> baseline;
    auto dump = [&](const char* suffix, const Tensor& hidden, int n_tokens) {
        if (!hidden.data) return false;
        std::vector<float> logits = engine.run_lmhead_raw(hidden, n_tokens);
        if (logits.empty()) return false;
        if (n_tokens == (int)tokens.size()) baseline = logits;
        std::string path = std::string(argv[3]) + suffix;
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        const size_t written = std::fwrite(logits.data(), sizeof(float), logits.size(), f);
        std::fclose(f);
        std::printf("%s: %zu logits\n", path.c_str(), written);
        return written == logits.size();
    };
    Tensor prefill = engine.prefill_hidden(tokens);
    if (!dump("_prefill.f32", prefill, (int)tokens.size())) return 1;
    if (prefill_only) return 0;
    Tensor decode = engine.decode_hidden(100);
    if (!dump("_decode.f32", decode, 1)) return 1;
    engine.reset();
    Tensor repeated = engine.prefill_hidden(tokens);
    if (!repeated.data) return 1;
    const std::vector<float> again = engine.run_lmhead_raw(
        repeated, (int)tokens.size());
    if (again.size() != baseline.size()) return 1;
    float max_delta = 0.f;
    for (size_t i = 0; i < again.size(); ++i)
        max_delta = std::max(max_delta, std::fabs(again[i] - baseline[i]));
    std::printf("reset prefill max logit delta: %g\n", max_delta);
    if (max_delta > 1e-4f) return 1;
    return 0;
}
