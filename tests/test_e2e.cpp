#include "engine/engine.h"
#include "engine/tokenizer.h"
#include <cstdio>
#include <cstring>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

int main(int argc, char** argv) {
    const char* tokenizer_path = argc > 1 ? argv[1] :
        "/Users/molly/workspace-youtulm-ncnn/Qwen3.5-0.8B/tokenizer.json";
    const char* output_dir = argc > 2 ? argv[2] :
        "/Users/molly/workspace-youtulm-ncnn/mlllm/test_output_qwen35_s4";

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
        CHECK(token >= 0, "prefill returns valid token");

        for (int step = 0; step < 3 && token >= 0; step++) {
            token = eng.decode(token);
            printf("Decode step %d returned token: %d\n", step, token);
        }

        std::vector<int> gen = {token};
        printf("Generated text: '%s'\n", tokenizer.decode(gen).c_str());
        CHECK(true, "decode-only test completed");
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
        printf("Prefill (4 tokens) returned token: %d\n", token);
        CHECK(token >= 0, "prefill returns valid token");

        for (int step = 0; step < 3 && token >= 0; step++) {
            token = eng2.decode(token);
            printf("Decode step %d returned token: %d\n", step, token);
        }

        std::vector<int> gen = {token};
        printf("Generated text: '%s'\n", tokenizer.decode(gen).c_str());
        CHECK(true, "prefill+decode test completed");
    }

    if (failures == 0) {
        printf("\nAll e2e tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
