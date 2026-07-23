// Optional real-checkpoint RWKV v7 regression.
//
// Set MOLLM_RWKV_PACKAGE to a package converted from the checked G1H fixture
// and this verifies both recurrent prefill and decode against PyTorch token
// sequences for ASCII and UTF-8 prompts.

#include "engine/engine.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    const char* package = std::getenv("MOLLM_RWKV_PACKAGE");
    if (!package || !*package) {
        std::printf("SKIP: set MOLLM_RWKV_PACKAGE for RWKV v7 E2E\n");
        return 0;
    }

    LLMEngine engine;
    EngineConfig cfg;
    cfg.package_path = package;
    cfg.num_threads = 4;
    cfg.temperature = 0.f;
    if (!engine.load(cfg)) {
        std::fprintf(stderr, "FAIL: could not load RWKV package\n");
        return 1;
    }

    const std::vector<std::pair<std::vector<int>, std::vector<int>>> cases = {
        // "Hello" in rwkv_vocab_v20230424.txt.
        {{33155}, {45, 308, 459, 332, 22168, 7152, 4811, 22590}},
        // "你好，请用一句话介绍你自己。". This catches non-ASCII tokenizer and
        // multi-token prefill regressions rather than checking only ASCII input.
        {{10464, 11685, 19137, 16738, 14589, 10250, 11012, 16713,
         10382, 15484, 10464, 15847, 12144, 10080},
         {9823, 261, 9822, 12605, 13091, 10250, 11043, 15052,
          12217, 11098, 19137, 10264, 13773, 10339, 12266, 10997}},
    };
    for (size_t c = 0; c < cases.size(); ++c) {
        // Compare every prefix as a numerical-stability diagnostic. GEMM and
        // GEMV accumulate in a different order, so a near-tied logit may pick
        // a different token even when both paths match the reference sequence.
        for (size_t prefix_len = 1; prefix_len <= cases[c].first.size(); ++prefix_len) {
            engine.reset();
            int expected = engine.prefill({cases[c].first.front()});
            for (size_t i = 1; i < prefix_len; ++i)
                expected = engine.decode(cases[c].first[i]);

            engine.reset();
            int actual = engine.prefill(std::vector<int>(cases[c].first.begin(),
                                                          cases[c].first.begin() + prefix_len));
            if (actual != expected) {
                std::printf("NOTE: case %zu prefix %zu batched got %d decode got %d\n",
                            c, prefix_len, actual, expected);
            }
        }

        // Prefill and repeated decode must be equivalent for a recurrent
        // model. Running this first makes the diagnostic independent of the
        // batched prefill path.
        engine.reset();
        int token = engine.prefill({cases[c].first.front()});
        for (size_t i = 1; i < cases[c].first.size(); ++i)
            token = engine.decode(cases[c].first[i]);
        if (token != cases[c].second.front()) {
            std::fprintf(stderr, "FAIL: case %zu decode path got %d expected %d\n",
                         c, token, cases[c].second.front());
            return 1;
        }

        engine.reset();
        token = engine.prefill(cases[c].first);
        for (size_t i = 0; i < cases[c].second.size(); ++i) {
            if (token != cases[c].second[i]) {
                std::fprintf(stderr, "FAIL: case %zu token %zu got %d expected %d\n",
                             c, i, token, cases[c].second[i]);
                return 1;
            }
            if (i + 1 < cases[c].second.size()) token = engine.decode(token);
        }

    }
    std::printf("PASS: RWKV v7 PyTorch-reference token sequence\n");
    return 0;
}
