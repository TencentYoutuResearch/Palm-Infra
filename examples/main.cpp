#include "examples/cli_common.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// mollm ASCII logo (figlet "standard" font, pure ASCII).
static const char* const kMollmLogo = R"(
                 _ _
 _ __ ___   ___ | | |_ __ ___
| '_ ` _ \ / _ \| | | '_ ` _ \
| | | | | | (_) | | | | | | | |
|_| |_| |_|\___/|_|_|_| |_| |_|
)";

// Print one banner line " key      : value" only if value is non-empty.
void banner_kv(const char* key, const std::string& value) {
    if (value.empty()) return;
    std::printf(" %-10s: %s\n", key, value.c_str());
}

// Lookup helper for package metadata.
std::string meta_get(const std::unordered_map<std::string, std::string>& meta,
                     const char* key) {
    auto it = meta.find(key);
    return it == meta.end() ? std::string() : it->second;
}

// Print the full REPL banner: logo, model/config fields, command hint.
// No separator lines — grouping is done with blank lines and indentation.
void print_repl_banner(const LLMEngine& engine, int prefill_seq_len, int n_ctx) {
    const auto& meta = engine.package_metadata();
    std::printf("%s\n", kMollmLogo);
    banner_kv("model",    meta_get(meta, "model_name"));
    banner_kv("arch",     meta_get(meta, "architecture"));
    banner_kv("layers",   meta_get(meta, "num_layers"));
    banner_kv("hidden",   meta_get(meta, "hidden_size"));
    banner_kv("quant",    meta_get(meta, "quantization"));
    banner_kv("ctx",      std::to_string(n_ctx));
    banner_kv("threads",  std::to_string(engine.config().num_threads));
    banner_kv("prefill",  std::to_string(prefill_seq_len));
    std::printf("\n");
    std::printf(" /reset   clear context\n");
    std::printf(" /quit    exit\n");
    std::printf("\n");
}

// Compact one-liner header for single-shot mode.
// Prints "--- mollm chat (model, quant) ---" or "--- mollm chat ---" fallback.
void print_single_shot_header(const LLMEngine& engine) {
    const auto& meta = engine.package_metadata();
    std::string model = meta_get(meta, "model_name");
    std::string quant = meta_get(meta, "quantization");
    if (model.empty() && quant.empty()) {
        std::printf("--- mollm chat ---\n");
    } else if (quant.empty()) {
        std::printf("--- mollm chat (%s) ---\n", model.c_str());
    } else {
        std::printf("--- mollm chat (%s, %s) ---\n", model.c_str(), quant.c_str());
    }
}

// Print an aligned metric block (key 10-wide left, value 12-wide right).
// Used at the end of each turn (single-shot and multi-turn).
void print_metric_block(const GenerationMetrics& m, int ctx_len) {
    std::printf("\n");
    std::printf(" %-10s %12.2f s\n",   "ttft",    m.ttft_ms / 1000.0);
    std::printf(" %-10s %12.1f ms\n",  "tpot",    m.tpot_ms);
    std::printf(" %-10s %12.1f t/s\n", "prefill", m.prefill_tps);
    std::printf(" %-10s %12.1f t/s\n", "decode",  m.decode_tps);
    std::printf(" %-10s %12d\n",       "tokens",  m.generated_tokens);
    std::printf(" %-10s %12d\n",       "ctx",     ctx_len);
    std::printf("\n");
}

size_t common_prefix_len(const std::vector<int>& a, const std::vector<int>& b) {
    size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

class TokenStreamPrinter {
public:
    void append(const std::string& piece) {
        if (piece.empty()) return;
        pending_ += piece;

        auto now = std::chrono::steady_clock::now();
        bool time_due = now - last_flush_ >= std::chrono::milliseconds(80);
        bool size_due = pending_.size() >= 512;
        bool newline = piece.find('\n') != std::string::npos;
        if (time_due || size_due || newline) {
            flush();
        }
    }

    void flush() {
        if (pending_.empty()) return;
        std::printf("%s", pending_.c_str());
        std::fflush(stdout);
        pending_.clear();
        last_flush_ = std::chrono::steady_clock::now();
    }

private:
    std::string pending_;
    std::chrono::steady_clock::time_point last_flush_ =
        std::chrono::steady_clock::now();
};

bool run_prompt_single(LLMEngine& engine, const Tokenizer& tokenizer,
                       const std::string& prompt, int /*prefill_seq_len*/,
                       int max_new_tokens) {
    std::vector<int> prompt_ids = tokenizer.apply_chat(prompt);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "[error] prompt is empty after tokenization\n");
        return false;
    }
    // No length check here — engine.prefill() handles chunked prefill for
    // prompts longer than graph_seq_len.

    print_single_shot_header(engine);

    GenerationResult result;
    std::string error;
    TokenStreamPrinter stream;
    bool ok = generate_greedy(
        engine, tokenizer, prompt_ids, max_new_tokens, tokenizer.eos_id(), result,
        error,
        [&](int, const std::string& piece) {
            stream.append(piece);
        });
    stream.flush();

    if (!ok) {
        std::fprintf(stderr, "\n[error] %s\n", error.c_str());
        return false;
    }

    GenerationMetrics metrics = compute_generation_metrics(prompt_ids.size(), result);
    std::printf("\n");
    print_metric_block(metrics, engine.past_len());
    return true;
}

bool run_turn_multi(LLMEngine& engine, const Tokenizer& tokenizer,
                    std::vector<ChatMessage>& history,
                    std::vector<int>& cached_token_ids,
                    const std::string& user_input, int max_new_tokens) {
    history.push_back({"user", user_input});
    std::vector<int> full_prompt_ids = tokenizer.apply_chat(history);
    if (full_prompt_ids.empty()) {
        std::fprintf(stderr, "[error] prompt is empty after tokenization\n");
        history.pop_back();
        return false;
    }

    size_t prefix = common_prefix_len(cached_token_ids, full_prompt_ids);
    if (prefix < cached_token_ids.size()) {
        // The retokenized history diverged from the actual KV prefix. Fall back
        // to a full rebuild rather than appending tokens onto a wrong cache.
        engine.reset();
        cached_token_ids.clear();
        prefix = 0;
    }
    std::vector<int> prompt_delta(full_prompt_ids.begin() + (ptrdiff_t)prefix,
                                  full_prompt_ids.end());

    GenerationResult result;
    std::string error;
    TokenStreamPrinter stream;
    // reset_context=false: multi-turn, keep the KV cache and prefill only the
    // prompt suffix that was not already consumed by previous turns.
    bool ok = generate_greedy(
        engine, tokenizer, prompt_delta, max_new_tokens, tokenizer.eos_id(), result,
        error,
        [&](int, const std::string& piece) {
            stream.append(piece);
        },
        /*reset_context=*/false);
    stream.flush();

    if (!ok) {
        std::fprintf(stderr, "\n[error] %s\n", error.c_str());
        history.pop_back();
        engine.reset();
        cached_token_ids.clear();
        return false;
    }

    // Add assistant response to history for next turn.
    history.push_back({"assistant", result.text});

    cached_token_ids.insert(cached_token_ids.end(),
                            prompt_delta.begin(), prompt_delta.end());
    if (result.token_ids.size() > 1) {
        cached_token_ids.insert(cached_token_ids.end(),
                                result.token_ids.begin(),
                                result.token_ids.end() - 1);
    }
    if ((int)cached_token_ids.size() != engine.past_len()) {
        std::fprintf(stderr,
                     "[warn] prefix cache length mismatch (cached=%zu engine=%d); "
                     "next turn will rebuild context\n",
                     cached_token_ids.size(), engine.past_len());
        engine.reset();
        cached_token_ids.clear();
    }

    GenerationMetrics metrics = compute_generation_metrics(prompt_delta.size(), result);
    std::printf("\n");
    print_metric_block(metrics, engine.past_len());
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
    if (opts.load_warmup && engine.package_weights_mmap_backed()) {
        std::fprintf(stderr, "Engine: warming package weight pages...\n");
        auto warmup_start = std::chrono::steady_clock::now();
        size_t warmup_bytes = engine.warmup_package_weights();
        auto warmup_end = std::chrono::steady_clock::now();
        double warmup_ms =
            std::chrono::duration<double, std::milli>(warmup_end - warmup_start).count();
        std::fprintf(stderr, "Engine: warmed %.1f MB in %.2f ms\n",
                     warmup_bytes / 1e6, warmup_ms);
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
    print_repl_banner(engine, prefill_seq_len, opts.n_ctx);

    std::vector<ChatMessage> history;
    history.push_back({"system", "You are a helpful assistant."});
    std::vector<int> cached_token_ids;

    std::string line;
    while (true) {
        std::printf("\n> ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/reset") {
            engine.reset();
            history.clear();
            history.push_back({"system", "You are a helpful assistant."});
            cached_token_ids.clear();
            std::printf("--- context cleared ---\n");
            continue;
        }
        std::printf("\n");
        run_turn_multi(engine, tokenizer, history, cached_token_ids,
                       line, opts.max_new_tokens);
    }

    return 0;
}
