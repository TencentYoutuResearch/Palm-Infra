#include "examples/cli_common.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool run_prompt(LLMEngine& engine, const Tokenizer& tokenizer,
                const std::string& prompt, int prefill_seq_len,
                int max_new_tokens) {
    std::vector<int> prompt_ids = tokenizer.apply_chat(prompt);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "chat: prompt is empty after tokenization\n");
        return false;
    }
    if ((int)prompt_ids.size() > prefill_seq_len) {
        std::fprintf(stderr,
                     "chat: prompt too long (%zu tokens > prefill seq_len %d)\n",
                     prompt_ids.size(), prefill_seq_len);
        return false;
    }

    std::printf("prompt_tokens=%zu\n", prompt_ids.size());
    std::printf("assistant> ");
    std::fflush(stdout);

    GenerationResult result;
    std::string error;
    bool ok = generate_greedy(
        engine, tokenizer, prompt_ids, max_new_tokens, tokenizer.eos_id(), result,
        error,
        [](int, const std::string& piece) {
            if (!piece.empty()) {
                std::printf("%s", piece.c_str());
                std::fflush(stdout);
            }
        });

    if (!ok) {
        std::fprintf(stderr, "\nchat: %s\n", error.c_str());
        return false;
    }

    GenerationMetrics metrics = compute_generation_metrics(prompt_ids.size(), result);

    std::printf("\n");
    std::printf("generated_tokens=%d decode_tokens=%d\n",
                metrics.generated_tokens, metrics.decode_tokens);
    std::printf("ttft_ms=%.2f tpot_ms=%.2f\n",
                metrics.ttft_ms, metrics.tpot_ms);
    std::printf("prefill_tps=%.2f decode_tps=%.2f\n",
                metrics.prefill_tps, metrics.decode_tps);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    CliCommonOptions opts;
    std::string error;
    if (!parse_common_args(argc, argv, opts, error)) {
        if (error != "help") std::fprintf(stderr, "chat: %s\n", error.c_str());
        print_common_usage(argv[0],
                           "Commands in REPL:\n"
                           "  /reset   Reset engine state\n"
                           "  /quit    Exit\n");
        return error == "help" ? 0 : 1;
    }

    Tokenizer tokenizer;
    LLMEngine engine;
    int prefill_seq_len = 0;
    if (!load_runtime(opts, tokenizer, engine, prefill_seq_len, error)) {
        std::fprintf(stderr, "chat: %s\n", error.c_str());
        return 1;
    }

    if (!opts.prompt.empty()) {
        return run_prompt(engine, tokenizer, opts.prompt, prefill_seq_len,
                          opts.max_new_tokens)
                   ? 0
                   : 1;
    }

    std::printf("mlllm_chat ready. single-turn mode, prefill_seq_len=%d\n",
                prefill_seq_len);
    std::printf("Type /quit to exit.\n");

    std::string line;
    while (true) {
        std::printf("user> ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/reset") {
            engine.reset();
            std::printf("reset done\n");
            continue;
        }
        run_prompt(engine, tokenizer, line, prefill_seq_len, opts.max_new_tokens);
    }

    return 0;
}
