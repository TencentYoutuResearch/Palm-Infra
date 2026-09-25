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
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "usage: %s PACKAGE N_CTX OUTPUT_PREFIX [--prefill-only]\n", argv[0]);
        return 2;
    }
    EngineConfig cfg;
    cfg.package_path = argv[1];
    cfg.n_ctx = std::atoi(argv[2]);
    cfg.num_threads = 4;
    cfg.weight_loading = WeightLoadingMode::MMAP;
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
    if (argc == 5) return 0;
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
