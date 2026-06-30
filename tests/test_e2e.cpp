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
    const char* tokenizer_path = argc > 1 ? argv[1] :
        "/Users/molly/workspace-youtulm-ncnn/Qwen3.5-0.8B/tokenizer.json";
    const char* output_dir = argc > 2 ? argv[2] :
        "/Users/molly/workspace-youtulm-ncnn/mollm/test_output_qwen35_s4";

    Tokenizer tokenizer;
    CHECK(tokenizer.load(tokenizer_path), "tokenizer load");

    std::vector<int> ids = tokenizer.encode("Hello, world!");
    CHECK(!ids.empty(), "tokenizer encode non-empty");

    // ---- Test 1: Decode graph only (T=1) ----
    printf("\n=== Test 1: Decode graph only ===\n");
    LLMEngine eng;
    EngineConfig cfg;
    cfg.prefill_graph_path = std::string(output_dir) + "/model_decode.graph";
    cfg.decode_graph_path  = std::string(output_dir) + "/model_decode.graph";
    cfg.n_ctx = 512;
    cfg.rope_theta = 1600000.f;

    bool ok = eng.load(cfg);
    CHECK(ok, "engine load decode graph");

    if (ok) {
        // Run single-token "prefill" (simulates decode)
        int token = eng.prefill({ids[0]});
        printf("Prefill returned token: %d\n", token);
        CHECK(token >= 0, "decode graph single-token prefill completed");

        for (int step = 0; step < 3 && token >= 0; step++) {
            token = eng.decode(token);
            printf("Decode step %d returned token: %d\n", step, token);
        }
    }
    // ---- Test 2: Prefill graph (T=4) + decode ----
    printf("\n=== Test 2: Prefill graph (T=4) + decode ===\n");
    LLMEngine eng2;
    EngineConfig cfg2;
    cfg2.prefill_graph_path = std::string(output_dir) + "/model_prefill.graph";
    cfg2.decode_graph_path  = std::string(output_dir) + "/model_decode.graph";
    cfg2.n_ctx = 512;
    cfg2.rope_dim = 64;
    cfg2.rope_theta = 1600000.f;
    cfg2.temperature = 0.0f;  // greedy for deterministic output

    ok = eng2.load(cfg2);
    CHECK(ok, "engine load prefill+decode graphs");

    if (ok) {
        int token = eng2.prefill(ids);
        printf("Prefill (4 tokens) returned token: %d (HF ref: %d)\n", token, REF_DECODE_SEQ[0]);
        CHECK(token == REF_DECODE_SEQ[0], "prefill argmax matches HF reference");

        for (int step = 0; step < REF_DECODE_LEN - 1; step++) {
            token = eng2.decode(token);
            printf("Decode step %d: token=%d (HF ref: %d)\n",
                   step + 1, token, REF_DECODE_SEQ[step + 1]);
            CHECK(token == REF_DECODE_SEQ[step + 1],
                  "decode argmax matches HF reference");
        }
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
            cfg.prefill_graph_path = std::string(output_dir) + "/model_prefill.graph";
            cfg.decode_graph_path  = std::string(output_dir) + "/model_decode.graph";
            cfg.n_ctx = 512;
            cfg.rope_dim = 64;
            cfg.rope_theta = 1600000.f;

            ok = eng.load(cfg);
            CHECK(ok, "prefill engine load");

            if (ok) {
                prefill_ppl = compute_ppl(eng, ppl_ids);
                printf("  Prefill PPL: %.4f (HF ref: %.2f)\n", prefill_ppl, REF_PPL);
                CHECK(std::fabs(prefill_ppl - REF_PPL) < PPL_TOLERANCE,
                      "prefill PPL matches HF reference");
            }
        }

        // ---- (b) Decode path: token[0] decode-graph prefill, then 255 decode steps ----
        float decode_ppl = -1.f;
        {
            LLMEngine eng;
            EngineConfig cfg;
            cfg.prefill_graph_path = std::string(output_dir) + "/model_decode.graph";
            cfg.decode_graph_path  = std::string(output_dir) + "/model_decode.graph";
            cfg.n_ctx = 512;
            cfg.rope_dim = 64;
            cfg.rope_theta = 1600000.f;

            ok = eng.load(cfg);
            CHECK(ok, "decode engine load");

            if (ok) {
                // Step 0: prefill token[0]
                Tensor hidden = eng.prefill_hidden({ppl_ids[0]});

                float total_ce = 0.f;
                int n_steps = PPL_REF_N - 1;  // 255 steps

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

                    if ((step + 1) % 64 == 0)
                        printf("  decode %d/%d\n", step + 1, n_steps);
                }

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
        const char* youtu_package = argc > 3 ? argv[3] :
            "/Users/molly/workspace-youtulm-ncnn/mollm/youtu-llm-2b.mollm";
        const char* youtu_tokenizer = argc > 4 ? argv[4] :
            "/Users/molly/workspace-youtulm-ncnn/Youtu-LLM-2B/tokenizer.json";

        Tokenizer youtu_tok;
        CHECK(youtu_tok.load(youtu_tokenizer), "Youtu tokenizer load");

        LLMEngine youtu_eng;
        EngineConfig youtu_cfg;
        youtu_cfg.package_path = youtu_package;
        youtu_cfg.tokenizer_path = youtu_tokenizer;
        youtu_cfg.n_ctx = 512;
        youtu_cfg.rope_theta = 500000.f;

        ok = youtu_eng.load(youtu_cfg);
        CHECK(ok, "Youtu engine load (package mode)");
        if (!ok) goto skip_youtu;

        // Greedy decode test
        {
            std::vector<int> youtu_ids = youtu_tok.encode("Hello, world!");
            int token = youtu_eng.prefill(youtu_ids);
            printf("  Prefill token: %d (HF ref: %d)\n", token, YOUTU_REF_DECODE[0]);
            CHECK(token == YOUTU_REF_DECODE[0], "Youtu prefill argmax matches HF reference");
            for (int step = 0; step < YOUTU_REF_DECODE_LEN - 1 && token >= 0; step++) {
                token = youtu_eng.decode(token);
                printf("  Decode step %d: token=%d (HF ref: %d)\n",
                       step + 1, token, YOUTU_REF_DECODE[step + 1]);
                CHECK(token == YOUTU_REF_DECODE[step + 1],
                      "Youtu decode argmax matches HF reference");
            }
        }

        // Hidden state dump for layer comparison
        {
            youtu_eng.reset();  // Clear cache from greedy decode test
            std::vector<int> ppl_ids(YOUTU_PPL_TOKENS, YOUTU_PPL_TOKENS + YOUTU_PPL_N);
            Tensor hidden = youtu_eng.prefill_hidden(ppl_ids);
            // Dump per-layer ADD outputs (attention residual + MLP residual per layer)
            mkdir("/tmp/youtu_cpp_layers", 0755);
            youtu_eng.dump_prefill_add_outputs("/tmp/youtu_cpp_layers");
            if (hidden.data) {
                int hidden_size = (int)hidden.shape[0];  // 2048
                int n_tokens = (int)hidden.shape[1];      // 256
                const float* h = hidden.ptr<float>();
                // Save last token hidden (after final norm, before lm_head)
                // Actually prefill_hidden returns the output of the final graph node
                // which is after final RMSNorm. Compare with HF layer_31 (last layer output
                // BEFORE final norm — not the same). Let's just dump and compare later.
                FILE* f = fopen("/tmp/youtu_cpp_hidden.f32", "wb");
                if (f) {
                    // Dump last token
                    fwrite(h + (n_tokens - 1) * hidden_size, sizeof(float), hidden_size, f);
                    fclose(f);
                    printf("  Dumped last token hidden to /tmp/youtu_cpp_hidden.f32\n");
                }
                // Also dump logits for all tokens
                std::vector<float> logits = youtu_eng.run_lmhead_raw(hidden, n_tokens, true);
                if (!logits.empty()) {
                    int vocab = (int)logits.size() / n_tokens;
                    f = fopen("/tmp/youtu_cpp_logits.f32", "wb");
                    if (f) {
                        // Last token logits
                        fwrite(logits.data() + (n_tokens - 1) * vocab, sizeof(float), vocab, f);
                        fclose(f);
                        printf("  Dumped last token logits to /tmp/youtu_cpp_logits.f32\n");
                    }
                }
            }
        }

        // PPL test
        {
            youtu_eng.reset();
            std::vector<int> ppl_ids(YOUTU_PPL_TOKENS, YOUTU_PPL_TOKENS + YOUTU_PPL_N);
            float ppl = compute_ppl(youtu_eng, ppl_ids);
            printf("  Youtu PPL: %.4f (HF ref: %.2f)\n", ppl, YOUTU_REF_PPL);
            CHECK(std::fabs(ppl - YOUTU_REF_PPL) < 2.0f,
                  "Youtu PPL matches HF reference (tol=2.0)");
        }

    skip_youtu:;
    }

    if (failures == 0) {
        printf("\nAll e2e tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
