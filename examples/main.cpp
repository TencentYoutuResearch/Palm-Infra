#include "examples/cli_common.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool run_prompt_single(LLMEngine& engine, const Tokenizer& tokenizer,
                       const std::string& prompt, int /*prefill_seq_len*/,
                       int max_new_tokens) {
    std::vector<int> prompt_ids = tokenizer.apply_chat(prompt);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "Chat: prompt is empty after tokenization\n");
        return false;
    }
    // No length check here — engine.prefill() handles chunked prefill for
    // prompts longer than graph_seq_len.

    std::printf("Assistant> ");
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
        std::fprintf(stderr, "\nChat: %s\n", error.c_str());
        return false;
    }

    GenerationMetrics metrics = compute_generation_metrics(prompt_ids.size(), result);
    std::printf("\n\n[ttft=%.2fs tpot=%.1fms prefill=%.1f tps decode=%.1f tps tokens=%d]\n",
                metrics.ttft_ms / 1000.0, metrics.tpot_ms,
                metrics.prefill_tps, metrics.decode_tps,
                metrics.generated_tokens);
    return true;
}

bool run_turn_multi(LLMEngine& engine, const Tokenizer& tokenizer,
                    std::vector<ChatMessage>& history,
                    const std::string& user_input, int max_new_tokens) {
    history.push_back({"user", user_input});
    std::vector<int> prompt_ids = tokenizer.apply_chat(history);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "Chat: prompt is empty after tokenization\n");
        history.pop_back();
        return false;
    }

    std::printf("Assistant> ");
    std::fflush(stdout);

    GenerationResult result;
    std::string error;
    // reset_context=false: multi-turn, don't wipe KV cache.
    bool ok = generate_greedy(
        engine, tokenizer, prompt_ids, max_new_tokens, tokenizer.eos_id(), result,
        error,
        [](int, const std::string& piece) {
            if (!piece.empty()) {
                std::printf("%s", piece.c_str());
                std::fflush(stdout);
            }
        },
        /*reset_context=*/false);

    if (!ok) {
        std::fprintf(stderr, "\nChat: %s\n", error.c_str());
        history.pop_back();
        return false;
    }

    // Add assistant response to history for next turn.
    history.push_back({"assistant", result.text});

    GenerationMetrics metrics = compute_generation_metrics(prompt_ids.size(), result);
    std::printf("\n\n[ttft=%.2fs tpot=%.1fms prefill=%.1f tps decode=%.1f tps tokens=%d ctx=%d]\n",
                metrics.ttft_ms / 1000.0, metrics.tpot_ms,
                metrics.prefill_tps, metrics.decode_tps,
                metrics.generated_tokens, engine.past_len());
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
                           "  /reset   Reset conversation\n"
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

    // Single-shot mode: --prompt "text" or --prompt-file <path>
    std::string prompt_text = opts.prompt;
    if (prompt_text.empty() && !opts.prompt_file.empty()) {
        std::ifstream f(opts.prompt_file);
        if (!f) {
            std::fprintf(stderr, "chat: cannot open --prompt-file: %s\n", opts.prompt_file.c_str());
            return 1;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        prompt_text = ss.str();
    }
    if (!prompt_text.empty()) {
        return run_prompt_single(engine, tokenizer, prompt_text,
                                  prefill_seq_len, opts.max_new_tokens)
                   ? 0
                   : 1;
    }

    // REPL mode: multi-turn conversation
    std::printf("mollm_chat ready. prefill_seq_len=%d threads=%d ctx=%d\n",
                prefill_seq_len, engine.config().num_threads, opts.n_ctx);
    std::printf("Type /reset to clear context, /quit to exit.\n");

    std::vector<ChatMessage> history;
    history.push_back({"system", "You are a helpful assistant."});

    std::string line;
    while (true) {
        std::printf("\nUser> ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/reset") {
            engine.reset();
            history.clear();
            history.push_back({"system", "You are a helpful assistant."});
            std::printf("\nContext cleared.\n");
            continue;
        }
        std::printf("\n");
        run_turn_multi(engine, tokenizer, history, line, opts.max_new_tokens);
    }

    return 0;
}
