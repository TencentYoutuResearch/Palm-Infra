#include "examples/cli_common.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    CliCommonOptions opts;
    std::string error;
    if (!parse_common_args(argc, argv, opts, error)) {
        if (error != "help") std::fprintf(stderr, "bench: %s\n", error.c_str());
        print_common_usage(argv[0],
                           "Benchmark-specific notes:\n"
                           "  --warmup <int>            Warmup iterations before timed run\n");
        return error == "help" ? 0 : 1;
    }

    if (opts.prompt.empty()) opts.prompt = "Hello, world!";

    Tokenizer tokenizer;
    LLMEngine engine;
    int prefill_seq_len = 0;

    auto load_start = std::chrono::steady_clock::now();
    if (!load_runtime(opts, tokenizer, engine, prefill_seq_len, error)) {
        std::fprintf(stderr, "bench: %s\n", error.c_str());
        return 1;
    }
    auto load_end = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

    std::vector<int> prompt_ids = tokenizer.apply_chat(opts.prompt);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "bench: prompt is empty after tokenization\n");
        return 1;
    }
    if ((int)prompt_ids.size() > prefill_seq_len) {
        std::fprintf(stderr,
                     "bench: prompt too long (%zu tokens > prefill seq_len %d)\n",
                     prompt_ids.size(), prefill_seq_len);
        return 1;
    }

    for (int i = 0; i < opts.warmup; i++) {
        GenerationResult warmup_result;
        std::string warmup_error;
        if (!generate_greedy(engine, tokenizer, prompt_ids, opts.max_new_tokens,
                             tokenizer.eos_id(), warmup_result, warmup_error)) {
            std::fprintf(stderr, "bench warmup failed: %s\n", warmup_error.c_str());
            return 1;
        }
    }

    GenerationResult result;
    auto total_start = std::chrono::steady_clock::now();
    if (!generate_greedy(engine, tokenizer, prompt_ids, opts.max_new_tokens,
                         tokenizer.eos_id(), result, error)) {
        std::fprintf(stderr, "bench: %s\n", error.c_str());
        return 1;
    }
    auto total_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    GenerationMetrics metrics = compute_generation_metrics(prompt_ids.size(), result);
    metrics.total_ms = total_ms;

    std::printf("load_ms=%.2f\n", load_ms);
    std::printf("prompt_tokens=%d\n", metrics.prompt_tokens);
    std::printf("generated_tokens=%d\n", metrics.generated_tokens);
    std::printf("decode_tokens=%d\n", metrics.decode_tokens);
    std::printf("ttft_ms=%.2f\n", metrics.ttft_ms);
    std::printf("tpot_ms=%.2f\n", metrics.tpot_ms);
    std::printf("prefill_tps=%.2f\n", metrics.prefill_tps);
    std::printf("decode_tps=%.2f\n", metrics.decode_tps);
    std::printf("prefill_ms=%.2f\n", result.prefill_ms);
    std::printf("decode_ms=%.2f\n", result.decode_ms);
    std::printf("total_ms=%.2f\n", metrics.total_ms);
    std::printf("hit_eos=%s\n", result.hit_eos ? "true" : "false");
    std::printf("generated_text=%s\n", result.text.c_str());

    return 0;
}
