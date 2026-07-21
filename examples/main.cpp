#include "examples/cli_common.h"

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#define MOLLM_HAVE_POSIX_TTY 1
#include <termios.h>
#include <unistd.h>
#endif

namespace {

// mollm ASCII logo (figlet "standard" font, pure ASCII).
static const char* const kMollmLogo = R"(
                 _ _
 _ __ ___   ___ | | |_ __ ___
| '_ ` _ \ / _ \| | | '_ ` _ \
| | | | | | (_) | | | | | | | |
|_| |_| |_|\___/|_|_|_| |_| |_|
)";

uint32_t decode_utf8_codepoint(const std::string& text, size_t offset) {
    const auto byte = [&](size_t i) { return static_cast<unsigned char>(text[offset + i]); };
    const unsigned char first = byte(0);
    if (first < 0x80) return first;
    const int length = first < 0xe0 ? 2 : first < 0xf0 ? 3 : first < 0xf8 ? 4 : 1;
    if (length == 1) return first;
    if (offset + static_cast<size_t>(length) > text.size()) return first;
    uint32_t codepoint = first & ((1u << (7 - length)) - 1);
    for (int i = 1; i < length; ++i) {
        const unsigned char next = byte(static_cast<size_t>(i));
        if ((next & 0xc0) != 0x80) return first;
        codepoint = (codepoint << 6) | (next & 0x3f);
    }
    return codepoint;
}

int terminal_columns(uint32_t codepoint) {
    // Zero-width combining marks, followed by the common East Asian wide and
    // emoji ranges. This is intentionally local: input editing should not
    // depend on the process locale being configured before main().
    if (codepoint == 0 || (codepoint >= 0x0300 && codepoint <= 0x036f) ||
        (codepoint >= 0x200b && codepoint <= 0x200f) ||
        (codepoint >= 0xfe00 && codepoint <= 0xfe0f)) {
        return 0;
    }
    if ((codepoint >= 0x1100 && codepoint <= 0x115f) ||
        (codepoint >= 0x2329 && codepoint <= 0x232a) ||
        (codepoint >= 0x2e80 && codepoint <= 0xa4cf) ||
        (codepoint >= 0xac00 && codepoint <= 0xd7a3) ||
        (codepoint >= 0xf900 && codepoint <= 0xfaff) ||
        (codepoint >= 0xfe10 && codepoint <= 0xfe6f) ||
        (codepoint >= 0xff00 && codepoint <= 0xffe6) ||
        (codepoint >= 0x1f300 && codepoint <= 0x1faff) ||
        (codepoint >= 0x20000 && codepoint <= 0x3fffd)) {
        return 2;
    }
    return 1;
}

void erase_terminal_columns(int columns) {
    while (columns-- > 0) std::fputs("\b \b", stdout);
    std::fflush(stdout);
}

// The terminal's canonical editor is allowed to erase UTF-8 one byte at a
// time, and in practice several terminal/PTY combinations still display a
// residue for CJK input.  REPL input is small, so handle it directly: retain
// normal signal handling, but echo and erase complete UTF-8 code points.
class TerminalLineReader {
public:
#if defined(MOLLM_HAVE_POSIX_TTY)
    TerminalLineReader() {
        if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &original_) != 0) {
            return;
        }
        termios updated = original_;
        // Handle Ctrl-C below instead of letting a signal terminate us while
        // the terminal is in raw mode (which would leave the caller's shell
        // without canonical input restoration).
        updated.c_lflag &= ~(ICANON | ECHO | ISIG);
        updated.c_cc[VMIN] = 1;
        updated.c_cc[VTIME] = 0;
        active_ = tcsetattr(STDIN_FILENO, TCSANOW, &updated) == 0;
    }

    ~TerminalLineReader() {
        if (active_) tcsetattr(STDIN_FILENO, TCSANOW, &original_);
    }
#else
    TerminalLineReader() = default;
#endif

    TerminalLineReader(const TerminalLineReader&) = delete;
    TerminalLineReader& operator=(const TerminalLineReader&) = delete;

    bool read_line(std::string& line) {
#if !defined(MOLLM_HAVE_POSIX_TTY)
        return static_cast<bool>(std::getline(std::cin, line));
#else
        if (!active_) return static_cast<bool>(std::getline(std::cin, line));
        line.clear();
        int escape_state = 0;
        for (;;) {
            char raw = 0;
            const ssize_t read_bytes = read(STDIN_FILENO, &raw, 1);
            if (read_bytes == 0) return false;
            if (read_bytes < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            const unsigned char byte = static_cast<unsigned char>(raw);
            if (escape_state != 0) {
                if (escape_state == 1 && (byte == '[' || byte == 'O')) {
                    escape_state = 2;
                } else if (escape_state == 2 && byte >= 0x40 && byte <= 0x7e) {
                    escape_state = 0;
                } else {
                    escape_state = 0;
                }
                continue;
            }
            if (byte == 0x1b) {
                escape_state = 1;  // Ignore cursor-key escape sequences.
                continue;
            }
            if (byte == '\r' || byte == '\n') {
                std::fputs("\r\n", stdout);
                std::fflush(stdout);
                return true;
            }
            if (byte == 0x03) {  // Ctrl-C
                std::fputs("^C\r\n", stdout);
                std::fflush(stdout);
                return false;
            }
            if (byte == 0x04) return !line.empty();  // Ctrl-D
            if (byte == 0x08 || byte == 0x7f) {
                if (line.empty()) continue;
                size_t begin = line.size() - 1;
                while (begin > 0 &&
                       (static_cast<unsigned char>(line[begin]) & 0xc0) == 0x80) {
                    --begin;
                }
                const int columns = terminal_columns(decode_utf8_codepoint(line, begin));
                line.erase(begin);
                erase_terminal_columns(columns);
                continue;
            }
            if (byte == 0x15) {  // Ctrl-U
                while (!line.empty()) {
                    size_t begin = line.size() - 1;
                    while (begin > 0 &&
                           (static_cast<unsigned char>(line[begin]) & 0xc0) == 0x80) {
                        --begin;
                    }
                    const int columns = terminal_columns(decode_utf8_codepoint(line, begin));
                    line.erase(begin);
                    erase_terminal_columns(columns);
                }
                continue;
            }
            if (byte < 0x20) continue;
            line.push_back(raw);
            std::fputc(raw, stdout);
            std::fflush(stdout);
        }
#endif
    }

private:
#if defined(MOLLM_HAVE_POSIX_TTY)
    termios original_{};
    bool active_ = false;
#endif
};

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
void print_repl_banner(const LLMEngine& engine, int n_ctx) {
    const auto& meta = engine.package_metadata();
    std::printf("%s\n", kMollmLogo);
    banner_kv("model",    meta_get(meta, "model_name"));
    banner_kv("arch",     meta_get(meta, "architecture"));
    banner_kv("layers",   meta_get(meta, "num_layers"));
    banner_kv("hidden",   meta_get(meta, "hidden_size"));
    banner_kv("quant",    meta_get(meta, "quantization"));
    banner_kv("ctx",      std::to_string(n_ctx));
    banner_kv("threads",  std::to_string(engine.config().num_threads));
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
    print_repl_banner(engine, opts.n_ctx);
    TerminalLineReader line_reader;

    std::vector<ChatMessage> history;
    std::vector<int> cached_token_ids;

    std::string line;
    while (true) {
        std::printf("\n> ");
        std::fflush(stdout);
        if (!line_reader.read_line(line)) break;
        if (line.empty()) continue;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/reset") {
            engine.reset();
            history.clear();
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
