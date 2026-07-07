#include "engine/engine.h"
#include "engine/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string package_path;
    std::string text;
    std::string text_file;
    int max_tokens = 256;
    int chunk_size = 256;
    int n_ctx = 4096;
    int threads = 4;
    bool prepend_bos = false;
    bool tokenize_only = false;
};

bool parse_int(const char* text, int& out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0') return false;
    out = (int)value;
    return true;
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign((std::istreambuf_iterator<char>(f)),
               std::istreambuf_iterator<char>());
    return true;
}

void print_usage(const char* argv0) {
    std::printf("Usage: %s --package <file.mollm> (--text <text> | --text-file <path>) [options]\n", argv0);
    std::printf("Options:\n");
    std::printf("  --max-tokens <int>    Token cap after tokenization (default 256; 0 = no cap)\n");
    std::printf("  --chunk-size <int>    Prefill chunk size for CE passes (default 256)\n");
    std::printf("  --n-ctx <int>         Context size (default 4096)\n");
    std::printf("  --threads <int>       Worker threads (default 4)\n");
    std::printf("  --prepend-bos         Prepend tokenizer BOS before scoring\n");
    std::printf("  --tokenize-only       Print token ids and exit before running the model\n");
}

bool parse_args(int argc, char** argv, Options& opts, std::string& error) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto require_value = [&](const char* flag, const char*& value) -> bool {
            if (i + 1 >= argc) {
                error = std::string("missing value for ") + flag;
                return false;
            }
            value = argv[++i];
            return true;
        };

        const char* value = nullptr;
        if (arg == "--help" || arg == "-h") {
            error = "help";
            return false;
        } else if (arg == "--package") {
            if (!require_value("--package", value)) return false;
            opts.package_path = value;
        } else if (arg == "--text") {
            if (!require_value("--text", value)) return false;
            opts.text = value;
        } else if (arg == "--text-file") {
            if (!require_value("--text-file", value)) return false;
            opts.text_file = value;
        } else if (arg == "--max-tokens") {
            if (!require_value("--max-tokens", value)) return false;
            if (!parse_int(value, opts.max_tokens) || opts.max_tokens < 0) {
                error = "invalid value for --max-tokens";
                return false;
            }
        } else if (arg == "--chunk-size") {
            if (!require_value("--chunk-size", value)) return false;
            if (!parse_int(value, opts.chunk_size) || opts.chunk_size < 1) {
                error = "invalid value for --chunk-size";
                return false;
            }
        } else if (arg == "--n-ctx") {
            if (!require_value("--n-ctx", value)) return false;
            if (!parse_int(value, opts.n_ctx) || opts.n_ctx < 2) {
                error = "invalid value for --n-ctx";
                return false;
            }
        } else if (arg == "--threads") {
            if (!require_value("--threads", value)) return false;
            if (!parse_int(value, opts.threads) || opts.threads < 1) {
                error = "invalid value for --threads";
                return false;
            }
        } else if (arg == "--prepend-bos") {
            opts.prepend_bos = true;
        } else if (arg == "--tokenize-only") {
            opts.tokenize_only = true;
        } else {
            error = std::string("unknown argument: ") + arg;
            return false;
        }
    }

    if (opts.package_path.empty()) {
        error = "missing required --package";
        return false;
    }
    if (opts.text.empty() && opts.text_file.empty()) {
        error = "missing --text or --text-file";
        return false;
    }
    if (!opts.text.empty() && !opts.text_file.empty()) {
        error = "use only one of --text or --text-file";
        return false;
    }
    return true;
}

float cross_entropy_for_target(const float* logits, int vocab, int target, bool& finite) {
    if (target < 0 || target >= vocab) {
        finite = false;
        return 0.0f;
    }

    float max_logit = logits[0];
    for (int i = 0; i < vocab; i++) {
        if (!std::isfinite(logits[i])) finite = false;
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    double sum_exp = 0.0;
    for (int i = 0; i < vocab; i++) {
        sum_exp += std::exp((double)logits[i] - (double)max_logit);
    }
    return (float)((double)max_logit + std::log(sum_exp) - (double)logits[target]);
}

} // namespace

int main(int argc, char** argv) {
    Options opts;
    std::string error;
    if (!parse_args(argc, argv, opts, error)) {
        if (error != "help") std::fprintf(stderr, "ppl: %s\n", error.c_str());
        print_usage(argv[0]);
        return error == "help" ? 0 : 1;
    }

    std::string text = opts.text;
    if (!opts.text_file.empty() && !read_file(opts.text_file, text)) {
        std::fprintf(stderr, "ppl: failed to read %s\n", opts.text_file.c_str());
        return 1;
    }

    LLMEngine engine;
    EngineConfig cfg;
    cfg.package_path = opts.package_path;
    cfg.n_ctx = opts.n_ctx;
    cfg.num_threads = opts.threads;
    cfg.temperature = 0.0f;
    if (!engine.load(cfg)) {
        std::fprintf(stderr, "ppl: failed to load package\n");
        return 1;
    }
    if (engine.config().tokenizer_path.empty()) {
        std::fprintf(stderr, "ppl: package did not expose a tokenizer\n");
        return 1;
    }

    Tokenizer tokenizer;
    if (!tokenizer.load(engine.config().tokenizer_path)) {
        std::fprintf(stderr, "ppl: failed to load tokenizer\n");
        return 1;
    }

    std::vector<int> token_ids = tokenizer.encode(text);
    if (opts.prepend_bos && tokenizer.bos_id() >= 0) {
        token_ids.insert(token_ids.begin(), tokenizer.bos_id());
    }
    if (opts.max_tokens > 0 && (int)token_ids.size() > opts.max_tokens) {
        token_ids.resize(opts.max_tokens);
    }
    if (opts.tokenize_only) {
        std::printf("tokens=%zu\n", token_ids.size());
        std::printf("token_ids=");
        for (size_t i = 0; i < token_ids.size(); i++) {
            if (i) std::printf(",");
            std::printf("%d", token_ids[i]);
        }
        std::printf("\n");
        return 0;
    }
    if ((int)token_ids.size() < 2) {
        std::fprintf(stderr, "ppl: need at least 2 tokens after tokenization\n");
        return 1;
    }
    if ((int)token_ids.size() > opts.n_ctx) {
        std::fprintf(stderr, "ppl: tokens=%zu exceeds n_ctx=%d\n",
                     token_ids.size(), opts.n_ctx);
        return 1;
    }
    engine.reset();
    const int n_tokens = (int)token_ids.size();
    int n_targets = 0;
    int chunks = 0;
    bool finite = true;
    double total_ce = 0.0;

    auto start = std::chrono::steady_clock::now();
    for (int offset = 0; offset < n_tokens - 1; offset += opts.chunk_size) {
        int chunk = std::min(opts.chunk_size, n_tokens - offset);
        std::vector<int> chunk_ids(token_ids.begin() + offset,
                                   token_ids.begin() + offset + chunk);

        Tensor hidden = engine.prefill_hidden(chunk_ids);
        if (!hidden.data) {
            std::fprintf(stderr, "ppl: prefill_hidden failed at offset %d\n", offset);
            return 1;
        }

        std::vector<float> logits = engine.run_lmhead_raw(
            hidden, chunk, /*all_positions=*/true);
        if (logits.empty()) {
            std::fprintf(stderr, "ppl: lm_head failed at offset %d\n", offset);
            return 1;
        }

        int vocab = (int)logits.size() / chunk;
        if (vocab <= 0 || (int)logits.size() != vocab * chunk) {
            std::fprintf(stderr, "ppl: invalid logits shape at offset %d\n", offset);
            return 1;
        }

        for (int pos = 0; pos < chunk; pos++) {
            int global_pos = offset + pos;
            int target_pos = global_pos + 1;
            if (target_pos >= n_tokens) break;
            const float* p = logits.data() + pos * vocab;
            total_ce += cross_entropy_for_target(p, vocab, token_ids[target_pos], finite);
            n_targets++;
        }
        chunks++;
    }
    auto end = std::chrono::steady_clock::now();

    if (n_targets <= 0) {
        std::fprintf(stderr, "ppl: no targets scored\n");
        return 1;
    }

    double ce = total_ce / n_targets;
    double ppl = std::exp(ce);
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::printf("package=%s\n", opts.package_path.c_str());
    std::printf("threads=%d\n", opts.threads);
    std::printf("tokens=%d\n", n_tokens);
    std::printf("targets=%d\n", n_targets);
    std::printf("chunks=%d\n", chunks);
    std::printf("chunk_size=%d\n", opts.chunk_size);
    std::printf("ce=%.6f\n", ce);
    std::printf("ppl=%.6f\n", ppl);
    std::printf("finite=%s\n", finite ? "true" : "false");
    std::printf("elapsed_ms=%.2f\n", ms);
    std::printf("tokens_per_s=%.2f\n", ms > 0.0 ? (1000.0 * n_tokens / ms) : 0.0);

    return finite && std::isfinite(ce) && std::isfinite(ppl) ? 0 : 1;
}
