#include "engine/engine.h"
#include "engine/tokenizer.h"
#include "tests/ppl_ref_tokens.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

// HF reference values for Qwen3.5-0.8B, dtype=float16.
static const int REF_DECODE_SEQ[] = {271, 9419, 11, 1814, 0};
static const int REF_DECODE_LEN = 5;
static const int REF_PREFILL_ARGMAX[] = {11, 271, 0};
static const int REF_PREFILL_N = 3;

// PPL reference: 256 tokens from calibration_data_v5_rc.txt
// HF Qwen3.5-0.8B prefill: CE=2.1405, PPL=8.50
static const float REF_PPL = 8.50f;
static const float PPL_TOLERANCE = 0.1f;

static float compute_ppl(LLMEngine& eng, const std::vector<int>& token_ids) {
    Tensor hidden_out = eng.prefill_hidden(token_ids);
    if (!hidden_out.data) return -1.f;
    CHECK(eng.prefill_pool_stats().active > 0,
          "prefill_hidden keeps reusable prefill workspace after output copy");

    int n_tokens = (int)token_ids.size();
    int n_targets = n_tokens - 1;
    std::vector<float> logits = eng.run_lmhead_raw(hidden_out, n_tokens, /*all_positions=*/true);
    if (logits.empty()) return -1.f;

    int vocab_size = (int)logits.size() / n_tokens;

    float total_ce = 0.f;
    for (int pos = 0; pos < n_targets; pos++) {
        int target = token_ids[pos + 1];
        const float* pos_logits = logits.data() + pos * vocab_size;

        float max_logit = pos_logits[0];
        for (int i = 1; i < vocab_size; i++)
            if (pos_logits[i] > max_logit) max_logit = pos_logits[i];

        float sum_exp = 0.f;
        for (int i = 0; i < vocab_size; i++)
            sum_exp += std::exp(pos_logits[i] - max_logit);

        float prob = std::exp(pos_logits[target] - max_logit) / sum_exp;
        total_ce += -std::log(prob);
    }

    return std::exp(total_ce / n_targets);
}

int main(int argc, char** argv) {
    // For Youtu-LLM debugging: can test with FP32 acc
    if (getenv("MOLLM_FP32_ACC")) {
        extern bool g_mollm_force_fp32_acc;
        g_mollm_force_fp32_acc = true;
    }
    // argv[1] = qwen35 .mollm package (Qwen3.5-0.8B transpiled)
    // argv[2] = youtu .mollm package (Youtu-LLM-2B transpiled)
    const char* qwen35_package = argc > 1 ? argv[1] : "../qwen35_0.8b.mollm";
    const char* youtu_package = argc > 2 ? argv[2] : "../youtu-llm-2b.mollm";
    struct stat st{};
    if (stat(qwen35_package, &st) != 0 || stat(youtu_package, &st) != 0) {
        std::printf("SKIP: e2e model packages not found (pass paths as argv[1:2])\n");
        return 0;
    }

    // Load qwen35 tokenizer first (from package via a throwaway engine load)
    Tokenizer tokenizer;
    {
        LLMEngine tmp;
        EngineConfig cfg;
        cfg.package_path = qwen35_package;
        cfg.n_ctx = 64;
        if (tmp.load(cfg) && !tmp.config().tokenizer_path.empty()) {
            tokenizer.load(tmp.config().tokenizer_path);
        }
    }
    CHECK(tokenizer.vocab_size() > 0, "tokenizer load (from package)");

    std::vector<int> ids = tokenizer.encode("Hello, world!");
    CHECK(!ids.empty(), "tokenizer encode non-empty");

    // ---- Test 1: Decode graph only (T=1) ----
    printf("\n=== Test 1: Decode graph only ===\n");
    LLMEngine eng;
    EngineConfig cfg;
    cfg.package_path = qwen35_package;
    cfg.n_ctx = 512;
    cfg.rope_theta = 1600000.f;

    bool ok = eng.load(cfg);
    CHECK(ok, "engine load qwen35 package");
    if (ok) CHECK(eng.prefill_pool_stats().active == 0,
                  "load keeps persistent cache out of prefill graph pool");

    if (ok) {
        // Run single-token "prefill" (simulates decode)
        int token = eng.prefill({ids[0]});
        printf("Prefill returned token: %d\n", token);
        CHECK(token >= 0, "single-token prefill completed");

        size_t decode_workspace = 0;
        bool decode_workspace_stable = true;
        for (int step = 0; step < 3 && token >= 0; step++) {
            token = eng.decode(token);
            printf("Decode step %d returned token: %d\n", step, token);
            size_t active = eng.decode_pool_stats().active;
            if (step == 0) decode_workspace = active;
            else decode_workspace_stable = decode_workspace_stable && (active == decode_workspace);
        }
        size_t prefill_workspace = eng.prefill_pool_stats().active;
        CHECK(prefill_workspace > 0 && decode_workspace_stable,
              "sampling prefill/decode keep bounded reusable workspaces");
        eng.reset();
        CHECK(eng.decode_pool_stats().active == 0,
              "reset releases reusable decode workspace");
        CHECK(eng.prefill_pool_stats().active == prefill_workspace,
              "reset preserves reusable prefill workspace");
        eng.release_prefill_buffers();
        CHECK(eng.prefill_pool_stats().active == 0,
              "release_prefill_buffers releases reusable prefill workspace");
    }
    // ---- Test 2: Prefill graph (T=4) + decode ----
    printf("\n=== Test 2: Prefill graph (T=4) + decode ===\n");
    LLMEngine eng2;
    EngineConfig cfg2;
    cfg2.package_path = qwen35_package;
    cfg2.n_ctx = 512;
    cfg2.rope_dim = 64;
    cfg2.rope_theta = 1600000.f;
    cfg2.temperature = 0.0f;  // greedy for deterministic output

    ok = eng2.load(cfg2);
    CHECK(ok, "engine load qwen35 package (for prefill+decode)");
    if (ok) CHECK(eng2.prefill_pool_stats().active == 0,
                  "load keeps persistent cache out of prefill graph pool (prefill+decode)");

    if (ok) {
        int token = eng2.prefill(ids);
        printf("Prefill (4 tokens) returned token: %d (HF ref: %d)\n", token, REF_DECODE_SEQ[0]);
        CHECK(token == REF_DECODE_SEQ[0], "prefill argmax matches HF reference");

        size_t decode_workspace = 0;
        bool decode_workspace_stable = true;
        for (int step = 0; step < REF_DECODE_LEN - 1; step++) {
            token = eng2.decode(token);
            printf("Decode step %d: token=%d (HF ref: %d)\n",
                   step + 1, token, REF_DECODE_SEQ[step + 1]);
            CHECK(token == REF_DECODE_SEQ[step + 1],
                  "decode argmax matches HF reference");
            size_t active = eng2.decode_pool_stats().active;
            if (step == 0) decode_workspace = active;
            else decode_workspace_stable = decode_workspace_stable && (active == decode_workspace);
        }
        CHECK(eng2.prefill_pool_stats().active > 0 && decode_workspace_stable,
              "prefill+decode sampling path keeps bounded reusable workspaces");
    }
    // ---- Test 3: PPL — prefill vs decode consistency + HF reference ----
    // 256 tokens from calibration_data_v5_rc.txt. HF reference PPL=8.50.
    // Checks: (a) C++ prefill vs HF, (b) C++ decode vs HF, (c) prefill vs decode.
    printf("\n=== Test 3: PPL (prefill vs decode) ===\n");
    {
        std::vector<int> ppl_ids(PPL_REF_TOKENS, PPL_REF_TOKENS + PPL_REF_N);

        // ---- (a) Prefill path: all 256 tokens at once ----
        float prefill_ppl = -1.f;
        {
            LLMEngine eng;
            EngineConfig cfg;
            cfg.package_path = qwen35_package;
            cfg.n_ctx = 512;
            cfg.rope_dim = 64;
            cfg.rope_theta = 1600000.f;

            ok = eng.load(cfg);
            CHECK(ok, "prefill engine load");
            if (ok) CHECK(eng.prefill_pool_stats().active == 0,
                          "prefill engine load has no active graph-pool cache");

            if (ok) {
                prefill_ppl = compute_ppl(eng, ppl_ids);
                printf("  Prefill PPL: %.4f (HF ref: %.2f)\n", prefill_ppl, REF_PPL);
                CHECK(std::fabs(prefill_ppl - REF_PPL) < PPL_TOLERANCE,
                      "prefill PPL matches HF reference");
            }
        }

        // ---- (b) Decode path: token[0] via decode graph (use_decode_as_prefill),
        //         then 255 decode steps ----
        float decode_ppl = -1.f;
        {
            LLMEngine eng;
            EngineConfig cfg;
            cfg.package_path = qwen35_package;
            cfg.use_decode_as_prefill = true;  // load decode graph as prefill
            cfg.n_ctx = 512;
            cfg.rope_dim = 64;
            cfg.rope_theta = 1600000.f;

            ok = eng.load(cfg);
            CHECK(ok, "decode engine load");
            if (ok) CHECK(eng.prefill_pool_stats().active == 0,
                          "decode engine load has no active graph-pool cache");

            if (ok) {
                // Step 0: prefill token[0]
                Tensor hidden = eng.prefill_hidden({ppl_ids[0]});
                size_t prefill_workspace = eng.prefill_pool_stats().active;

                float total_ce = 0.f;
                int n_steps = PPL_REF_N - 1;  // 255 steps
                bool decode_pool_ok = true;
                bool have_decode_workspace = false;
                size_t decode_workspace = 0;

                for (int step = 0; step < n_steps; step++) {
                    int target = ppl_ids[step + 1];

                    // Logits from last hidden position
                    int hd = (int)hidden.shape[0];
                    int sl = (int)hidden.shape[1];
                    int lp = sl - 1;

                    Tensor lh = hidden;
                    lh.shape[1] = 1;
                    lh.data = static_cast<char*>(hidden.data) + lp * hd * sizeof(float);
                    lh.compute_strides();

                    std::vector<float> logits = eng.run_lmhead_raw(lh, 1);
                    int vs = (int)logits.size();

                    float mx = logits[0];
                    for (int i = 1; i < vs; i++) if (logits[i] > mx) mx = logits[i];
                    float se = 0.f;
                    for (int i = 0; i < vs; i++) se += std::exp(logits[i] - mx);
                    float prob = std::exp(logits[target] - mx) / se;
                    total_ce += -std::log(prob);

                    // Next token
                    hidden = eng.decode_hidden(target);
                    size_t active = eng.decode_pool_stats().active;
                    if (!have_decode_workspace) {
                        decode_workspace = active;
                        have_decode_workspace = true;
                    }
                    decode_pool_ok = decode_pool_ok
                        && eng.prefill_pool_stats().active == prefill_workspace
                        && active == decode_workspace;

                    if ((step + 1) % 64 == 0)
                        printf("  decode %d/%d\n", step + 1, n_steps);
                }
                CHECK(decode_pool_ok, "decode_hidden keeps bounded reusable workspaces");

                decode_ppl = std::exp(total_ce / n_steps);
                printf("  Decode PPL:  %.4f (HF ref: %.2f)\n", decode_ppl, REF_PPL);
                CHECK(std::fabs(decode_ppl - REF_PPL) < PPL_TOLERANCE,
                      "decode PPL matches HF reference");
            }
        }

        // ---- (c) Self-consistency ----
        if (prefill_ppl > 0 && decode_ppl > 0) {
            printf("  Prefill: %.4f  Decode: %.4f  Delta: %.4f\n",
                   prefill_ppl, decode_ppl, std::fabs(prefill_ppl - decode_ppl));
            CHECK(std::fabs(prefill_ppl - decode_ppl) < 0.1f,
                  "prefill and decode PPL are self-consistent");
        }
    }

    // ---- Test 4: Youtu-LLM-2B PPL (package mode) ----
    // Tests MLA path correctness. HF reference PPL=45.25 on 256 calibration tokens.
    printf("\n=== Test 4: Youtu-LLM-2B PPL (MLA path) ===\n");
    {
        LLMEngine youtu_eng;
        EngineConfig youtu_cfg;
        youtu_cfg.package_path = youtu_package;
        youtu_cfg.n_ctx = 512;
        youtu_cfg.rope_theta = 500000.f;

        ok = youtu_eng.load(youtu_cfg);
        CHECK(ok, "Youtu engine load (package mode)");
        if (ok) CHECK(youtu_eng.prefill_pool_stats().active == 0,
                      "Youtu load keeps persistent cache out of prefill graph pool");
        if (ok) {
            // Load tokenizer from extracted path (engine sets it)
            Tokenizer youtu_tok;
            CHECK(youtu_tok.load(youtu_eng.config().tokenizer_path), "Youtu tokenizer load");

            // Greedy decode test — print tokens for inspection.
            // NOTE: argmax against HF reference is intentionally NOT checked.
            // Youtu-LLM-2B is a BF16 model; mollm converts BF16→FP32→FP16 which
            // rounds differently than HF's BF16→FP16 path. The greedy path is
            // numerically fragile (top-1 logit gaps are small on "Hello, world!"),
            // so PPL is the authoritative correctness check below.
            {
                std::vector<int> youtu_ids = youtu_tok.encode("Hello, world!");
                int token = youtu_eng.prefill(youtu_ids);
                printf("  Prefill token: %d\n", token);
                size_t decode_workspace = 0;
                bool decode_workspace_stable = true;
                for (int step = 0; step < 4 && token >= 0; step++) {
                    token = youtu_eng.decode(token);
                    printf("  Decode step %d: token=%d\n", step + 1, token);
                    size_t active = youtu_eng.decode_pool_stats().active;
                    if (step == 0) decode_workspace = active;
                    else decode_workspace_stable = decode_workspace_stable && (active == decode_workspace);
                }
                CHECK(youtu_eng.prefill_pool_stats().active > 0 && decode_workspace_stable,
                      "Youtu sampling path keeps bounded reusable workspaces");
            }

            // Hidden state dump for layer comparison
            {
                youtu_eng.reset();  // Clear cache and reusable decode workspace from greedy decode test
                CHECK(youtu_eng.decode_pool_stats().active == 0,
                      "Youtu reset releases reusable decode workspace");
                std::vector<int> ppl_ids(YOUTU_PPL_TOKENS, YOUTU_PPL_TOKENS + YOUTU_PPL_N);
                Tensor hidden = youtu_eng.prefill_hidden(ppl_ids);
                CHECK(youtu_eng.prefill_pool_stats().active > 0,
                      "Youtu prefill_hidden keeps reusable prefill workspace after output copy");
                if (hidden.data) {
                    int hidden_size = (int)hidden.shape[0];  // 2048
                    int n_tokens = (int)hidden.shape[1];      // 256
                    const float* h = hidden.ptr<float>();
                    // Save last token hidden (after final norm, before lm_head)
                    FILE* f = fopen("/tmp/youtu_cpp_hidden.f32", "wb");
                    if (f) {
                        fwrite(h + (n_tokens - 1) * hidden_size, sizeof(float), hidden_size, f);
                        fclose(f);
                        printf("  Dumped last token hidden to /tmp/youtu_cpp_hidden.f32\n");
                    }
                }
            }

            // PPL test — Youtu-LLM-2B MLA path, DYNAMIC mode.
            // HF ref PPL=10.20 on 256 natural-text tokens (BF16 weights).
            // mollm converts BF16→FP32→FP16 (rounds differently), so tolerance 2.0.
            {
                youtu_eng.reset();
                std::vector<int> ppl_ids(YOUTU_PPL_TOKENS, YOUTU_PPL_TOKENS + YOUTU_PPL_N);
                Tensor hidden = youtu_eng.prefill_hidden(ppl_ids);
                CHECK(youtu_eng.prefill_pool_stats().active > 0,
                      "Youtu PPL prefill_hidden keeps reusable prefill workspace");
                bool ppl_ok = false;
                if (hidden.data) {
                    std::vector<float> logits = youtu_eng.run_lmhead_raw(
                        hidden, (int)ppl_ids.size(), /*all_positions=*/true);
                    if (!logits.empty()) {
                        int vocab = (int)logits.size() / (int)ppl_ids.size();
                        int n_targets = (int)ppl_ids.size() - 1;
                        float total_ce = 0.f;
                        for (int pos = 0; pos < n_targets; pos++) {
                            int target = ppl_ids[pos + 1];
                            const float* p = logits.data() + pos * vocab;
                            float mx = p[0];
                            for (int i = 1; i < vocab; i++)
                                if (p[i] > mx) mx = p[i];
                            float s = 0.f;
                            for (int i = 0; i < vocab; i++)
                                s += std::exp(p[i] - mx);
                            total_ce += -std::log(std::exp(p[target] - mx) / s);
                        }
                        float ppl = std::exp(total_ce / n_targets);
                        printf("  Youtu PPL: %.4f (HF ref: %.2f)\n", ppl, YOUTU_REF_PPL);
                        ppl_ok = (std::fabs(ppl - YOUTU_REF_PPL) < 2.0f);
                    }
                }
                CHECK(ppl_ok, "Youtu PPL matches HF reference");
            }
        }
    }

    // ---- Test 5: PPL — chunked prefill consistency ----
    // Split 256 tokens into multiple chunks, each smaller than graph_seq_len,
    // triggering the padding path. Verifies cross-chunk KV/state handoff
    // (GDN state, conv_state, KV cache all need to chain correctly).
    printf("\n=== Test 5: PPL chunked prefill ===\n");
    {
        std::vector<int> ppl_ids(PPL_REF_TOKENS, PPL_REF_TOKENS + PPL_REF_N);

        // Helper: compute PPL from a flat logits vector (seq_len * vocab).
        auto ppl_from_logits = [](const std::vector<float>& all_logits,
                                   const std::vector<int>& tokens) -> float {
            int n_tokens = (int)tokens.size();
            int n_targets = n_tokens - 1;
            int vocab = (int)all_logits.size() / n_tokens;
            float total_ce = 0.f;
            for (int pos = 0; pos < n_targets; pos++) {
                int target = tokens[pos + 1];
                const float* pos_logits = all_logits.data() + pos * vocab;
                float max_logit = pos_logits[0];
                for (int i = 1; i < vocab; i++)
                    if (pos_logits[i] > max_logit) max_logit = pos_logits[i];
                float sum_exp = 0.f;
                for (int i = 0; i < vocab; i++)
                    sum_exp += std::exp(pos_logits[i] - max_logit);
                float prob = std::exp(pos_logits[target] - max_logit) / sum_exp;
                total_ce += -std::log(prob);
            }
            return std::exp(total_ce / n_targets);
        };

        struct ChunkSpec { int chunk_size; const char* name; bool check_ppl; };
        ChunkSpec specs[] = {
            {128, "2 chunks of 128", true},
            {100, "3 chunks (100,100,56)", true},
            {64,  "4 chunks of 64", true},
            {56,  "5 chunks (56,56,56,56,32)", true},
            {256, "1 chunk of 256", true},
            {100, "1 chunk of 100 (non-2pow)", false},  // PPL skip (token count mismatch)
            {57,  "5 chunks (57,57,57,57,28)", true},   // prime chunk size
            {40,  "7 chunks (40,40,40,40,40,40,16)", true},
            {33,  "8 chunks (33,33,...,25)", true},
        };

        for (auto& spec : specs) {
            LLMEngine eng;
            EngineConfig cfg;
            cfg.package_path = qwen35_package;
            cfg.n_ctx = 512;
            cfg.rope_dim = 64;
            cfg.rope_theta = 1600000.f;

            // Build a label like "chunked PPL matches HF: 2 chunks of 128"
            char label[128];
            std::snprintf(label, sizeof(label), "chunked PPL matches HF: %s", spec.name);

            ok = eng.load(cfg);
            if (!ok) {
                printf("  skip %s: load failed\n", spec.name);
                CHECK(false, label);
                continue;
            }
            CHECK(eng.prefill_pool_stats().active == 0,
                  "chunked engine load has no active graph-pool cache");

            // Prefill chunk by chunk, accumulate logits for all positions.
            std::vector<float> all_logits;
            int offset = 0;
            bool chunk_ok = true;
            while (offset < PPL_REF_N) {
                int chunk = std::min(spec.chunk_size, PPL_REF_N - offset);
                std::vector<int> chunk_ids(ppl_ids.begin() + offset,
                                           ppl_ids.begin() + offset + chunk);
                Tensor hidden = eng.prefill_hidden(chunk_ids);
                CHECK(eng.prefill_pool_stats().active > 0,
                      "chunked prefill_hidden keeps bounded reusable prefill workspace");
                if (!hidden.data) {
                    printf("  %s: prefill_hidden failed at offset %d\n",
                           spec.name, offset);
                    chunk_ok = false;
                    break;
                }
                // Extract logits for this chunk's positions
                std::vector<float> logits = eng.run_lmhead_raw(hidden, chunk, true);
                all_logits.insert(all_logits.end(), logits.begin(), logits.end());
                offset += chunk;
            }

            if (!spec.check_ppl) {
                // For non-256-token configs, just check prefill doesn't crash/NaN.
                bool has_nan = false;
                if (!all_logits.empty()) {
                    for (float v : all_logits) {
                        if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; }
                    }
                }
                printf("  %s: %s (logits=%zu)\n",
                       spec.name, has_nan ? "NaN/Inf!" : "ok", all_logits.size());
                CHECK(!has_nan, label);
            } else if (chunk_ok && !all_logits.empty()) {
                int vocab = (int)all_logits.size() / PPL_REF_N;
                if ((int)all_logits.size() != PPL_REF_N * vocab) {
                    printf("  %s: logits size mismatch (got %zu)\n",
                           spec.name, all_logits.size());
                    CHECK(false, label);
                } else {
                    float ppl = ppl_from_logits(all_logits, ppl_ids);
                    printf("  %s: PPL=%.4f (HF ref: %.2f)\n", spec.name, ppl, REF_PPL);
                    CHECK(std::fabs(ppl - REF_PPL) < PPL_TOLERANCE, label);
                }
            } else if (chunk_ok) {
                printf("  %s: no logits produced\n", spec.name);
                CHECK(false, label);
            } else {
                CHECK(false, label);
            }
        }
    }

    if (failures == 0) {
        printf("\nAll e2e tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
