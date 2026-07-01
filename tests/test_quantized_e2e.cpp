#include "engine/engine.h"
#include "engine/tokenizer.h"
#include "tests/ppl_ref_tokens.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { std::printf("  PASS: %s\n", msg); } \
} while (0)

struct CeResult {
    float ce = -1.f;
    float ppl = -1.f;
    int vocab = 0;
    bool finite = false;
};

static CeResult compute_short_ce(LLMEngine& eng, const std::vector<int>& token_ids) {
    CeResult result;
    Tensor hidden = eng.prefill_hidden(token_ids);
    if (!hidden.data) return result;

    std::vector<float> logits = eng.run_lmhead_raw(
        hidden, (int)token_ids.size(), /*all_positions=*/true);
    if (logits.empty()) return result;

    int n_tokens = (int)token_ids.size();
    int n_targets = n_tokens - 1;
    int vocab = (int)logits.size() / n_tokens;
    if ((int)logits.size() != vocab * n_tokens || vocab <= 0) return result;

    bool finite = true;
    float total_ce = 0.f;
    for (int pos = 0; pos < n_targets; pos++) {
        int target = token_ids[pos + 1];
        const float* p = logits.data() + pos * vocab;

        float max_logit = p[0];
        for (int i = 0; i < vocab; i++) {
            if (!std::isfinite(p[i])) finite = false;
            if (p[i] > max_logit) max_logit = p[i];
        }

        float sum_exp = 0.f;
        for (int i = 0; i < vocab; i++) sum_exp += std::exp(p[i] - max_logit);
        float prob = std::exp(p[target] - max_logit) / sum_exp;
        total_ce += -std::log(prob);
    }

    result.ce = total_ce / n_targets;
    result.ppl = std::exp(result.ce);
    result.vocab = vocab;
    result.finite = finite && std::isfinite(result.ce) && std::isfinite(result.ppl);
    return result;
}

int main(int argc, char** argv) {
    const char* fp16_package = argc > 1 ? argv[1] :
        "/Users/molly/workspace-youtulm-ncnn/mollm/qwen35_0.8b.mollm";
    const char* quant_package = argc > 2 ? argv[2] :
        "/Users/molly/workspace-youtulm-ncnn/mollm/qwen35_0.8b_w8pc.mollm";

    std::vector<int> ppl_ids(PPL_REF_TOKENS, PPL_REF_TOKENS + 32);

    LLMEngine fp16;
    EngineConfig fp16_cfg;
    fp16_cfg.package_path = fp16_package;
    fp16_cfg.n_ctx = 512;
    fp16_cfg.rope_dim = 64;
    fp16_cfg.rope_theta = 1600000.f;
    fp16_cfg.temperature = 0.0f;
    bool fp16_ok = fp16.load(fp16_cfg);
    CHECK(fp16_ok, "FP16 baseline engine load");

    LLMEngine quant;
    EngineConfig quant_cfg;
    quant_cfg.package_path = quant_package;
    quant_cfg.n_ctx = 512;
    quant_cfg.rope_dim = 64;
    quant_cfg.rope_theta = 1600000.f;
    quant_cfg.temperature = 0.0f;
    bool quant_ok = quant.load(quant_cfg);
    CHECK(quant_ok, "W8 engine load");

    if (fp16_ok && fp16.config().tokenizer_path.empty()) {
        CHECK(false, "FP16 package exposes tokenizer");
    }

    if (fp16_ok && quant_ok) {
        CeResult fp16_ce = compute_short_ce(fp16, ppl_ids);
        CeResult quant_ce = compute_short_ce(quant, ppl_ids);
        std::printf("  short CE/PPL: fp16=%.4f/%.2f w8=%.4f/%.2f\n",
                    fp16_ce.ce, fp16_ce.ppl, quant_ce.ce, quant_ce.ppl);
        std::printf("  short CE delta: %.4f\n", quant_ce.ce - fp16_ce.ce);

        CHECK(fp16_ce.finite, "FP16 short logits finite");
        CHECK(quant_ce.finite, "W8 short logits finite");
        CHECK(fp16_ce.vocab > 0 && fp16_ce.vocab == quant_ce.vocab,
              "W8 vocab/logit shape matches FP16");
        CHECK(quant.prefill_pool_stats().active > 0,
              "W8 prefill keeps reusable workspace");

        // This is a correctness smoke bound, not a quality target. W8PC is
        // expected to drift from FP16, but a very large CE jump usually means
        // bad scale indexing, missing metadata, or transposed quantization.
        CHECK(quant_ce.ce < fp16_ce.ce + 1.0f,
              "W8 short CE stays near FP16 baseline");

        Tokenizer tok;
        bool tok_ok = tok.load(fp16.config().tokenizer_path);
        CHECK(tok_ok, "tokenizer load");
        std::vector<int> ids = tok_ok ? tok.encode("Hello, world!") : std::vector<int>();
        CHECK(!ids.empty(), "tokenizer encode non-empty");
        if (!ids.empty() && quant_ce.vocab > 0) {
            quant.reset();
            int token = quant.prefill(ids);
            std::printf("  W8 greedy prefill token: %d\n", token);
            CHECK(token >= 0 && token < quant_ce.vocab, "W8 prefill returns valid token");

            size_t decode_workspace = 0;
            bool stable = true;
            for (int step = 0; step < 2 && token >= 0; step++) {
                token = quant.decode(token);
                std::printf("  W8 decode step %d token: %d\n", step + 1, token);
                size_t active = quant.decode_pool_stats().active;
                if (step == 0) decode_workspace = active;
                else stable = stable && active == decode_workspace;
                CHECK(token >= 0 && token < quant_ce.vocab, "W8 decode returns valid token");
            }
            CHECK(stable && decode_workspace > 0, "W8 decode workspace is stable");
        }
    }

    if (failures == 0) {
        std::printf("\nAll quantized e2e tests passed!\n");
    } else {
        std::printf("\n%d quantized e2e test(s) FAILED\n", failures);
    }
    return failures;
}
