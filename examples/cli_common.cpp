#include "examples/cli_common.h"

#include "graph/graph.h"
#include "kernels/threading.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

bool parse_int(const char* text, int& out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0') return false;
    out = static_cast<int>(value);
    return true;
}

bool parse_float(const char* text, float& out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    float value = std::strtof(text, &end);
    if (!end || *end != '\0') return false;
    out = value;
    return true;
}

bool require_value(int argc, char** argv, int& i, const char* flag,
                   const char*& value, std::string& error) {
    if (i + 1 >= argc) {
        error = std::string("missing value for ") + flag;
        return false;
    }
    value = argv[++i];
    return true;
}

} // namespace

bool parse_common_args(int argc, char** argv, CliCommonOptions& opts,
                       std::string& error) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        const char* value = nullptr;

        if (arg == "--help" || arg == "-h") {
            error = "help";
            return false;
        } else if (arg == "--tokenizer") {
            if (!require_value(argc, argv, i, "--tokenizer", value, error)) return false;
            opts.tokenizer_path = value;
        } else if (arg == "--artifacts") {
            if (!require_value(argc, argv, i, "--artifacts", value, error)) return false;
            opts.artifacts_dir = value;
        } else if (arg == "--prompt") {
            if (!require_value(argc, argv, i, "--prompt", value, error)) return false;
            opts.prompt = value;
        } else if (arg == "--prompt-file") {
            if (!require_value(argc, argv, i, "--prompt-file", value, error)) return false;
            opts.prompt_file = value;
        } else if (arg == "--prompt-tokens") {
            if (!require_value(argc, argv, i, "--prompt-tokens", value, error)) return false;
            if (!parse_int(value, opts.prompt_tokens) || opts.prompt_tokens < 1) {
                error = "invalid value for --prompt-tokens";
                return false;
            }
        } else if (arg == "--max-new-tokens") {
            if (!require_value(argc, argv, i, "--max-new-tokens", value, error)) return false;
            if (!parse_int(value, opts.max_new_tokens) || opts.max_new_tokens < 1) {
                error = "invalid value for --max-new-tokens";
                return false;
            }
        } else if (arg == "--n-ctx") {
            if (!require_value(argc, argv, i, "--n-ctx", value, error)) return false;
            if (!parse_int(value, opts.n_ctx) || opts.n_ctx < 1) {
                error = "invalid value for --n-ctx";
                return false;
            }
        } else if (arg == "--rope-dim") {
            if (!require_value(argc, argv, i, "--rope-dim", value, error)) return false;
            if (!parse_int(value, opts.rope_dim) || opts.rope_dim < 1) {
                error = "invalid value for --rope-dim";
                return false;
            }
        } else if (arg == "--rope-theta") {
            if (!require_value(argc, argv, i, "--rope-theta", value, error)) return false;
            if (!parse_float(value, opts.rope_theta) || opts.rope_theta <= 0.f) {
                error = "invalid value for --rope-theta";
                return false;
            }
        } else if (arg == "--threads") {
            if (!require_value(argc, argv, i, "--threads", value, error)) return false;
            if (!parse_int(value, opts.num_threads) || opts.num_threads < 1) {
                error = "invalid value for --threads";
                return false;
            }
        } else if (arg == "--profile") {
            opts.profile = true;
        } else if (arg == "--warmup") {
            if (!require_value(argc, argv, i, "--warmup", value, error)) return false;
            if (!parse_int(value, opts.warmup) || opts.warmup < 0) {
                error = "invalid value for --warmup";
                return false;
            }
        } else {
            error = std::string("unknown argument: ") + arg;
            return false;
        }
    }

    if (opts.tokenizer_path.empty()) {
        error = "missing required --tokenizer";
        return false;
    }
    if (opts.artifacts_dir.empty()) {
        error = "missing required --artifacts";
        return false;
    }

    return true;
}

void print_common_usage(const char* program_name, const char* extra_usage) {
    std::printf("Usage: %s --tokenizer <tokenizer.json> --artifacts <dir> [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  --prompt <text>           Run one prompt and exit\n");
    std::printf("  --prompt-file <path>      Read prompt text from file, run and exit\n");
    std::printf("  --prompt-tokens <int>     Use N dummy tokens (skip chat template)\n");
    std::printf("  --max-new-tokens <int>    Default: 128\n");
    std::printf("  --n-ctx <int>             Default: 4096\n");
    std::printf("  --rope-dim <int>          Default: 64\n");
    std::printf("  --rope-theta <float>      Default: 1600000\n");
    std::printf("  --threads <int>          Default: 4\n");
    std::printf("  --profile                Print aggregated per-op profile in bench\n");
    std::printf("  --warmup <int>            Default: 1 (used by benchmark)\n");
    if (extra_usage && *extra_usage) {
        std::printf("%s", extra_usage);
        if (extra_usage[std::strlen(extra_usage) - 1] != '\n') std::printf("\n");
    }
}

EngineConfig make_engine_config(const CliCommonOptions& opts) {
    EngineConfig cfg;
    cfg.prefill_graph_path = opts.artifacts_dir + "/model_prefill.graph";
    cfg.decode_graph_path = opts.artifacts_dir + "/model_decode.graph";
    cfg.n_ctx = opts.n_ctx;
    cfg.rope_dim = opts.rope_dim;
    cfg.rope_theta = opts.rope_theta;
    cfg.num_threads = opts.num_threads;
    return cfg;
}

bool inspect_prefill_seq_len(const std::string& graph_path, int& seq_len,
                             std::string& error) {
    Graph g;
    if (!graph_load(g, graph_path.c_str())) {
        error = std::string("failed to load graph: ") + graph_path;
        return false;
    }

    for (const auto& node : g.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty() &&
            node.params.str[0] == "hidden") {
            seq_len = static_cast<int>(node.out_shape[1]);
            return true;
        }
    }

    error = std::string("graph missing hidden input: ") + graph_path;
    return false;
}

bool load_runtime(const CliCommonOptions& opts, Tokenizer& tokenizer,
                  LLMEngine& engine, int& prefill_seq_len, std::string& error) {
    if (!tokenizer.load(opts.tokenizer_path)) {
        error = std::string("failed to load tokenizer: ") + opts.tokenizer_path;
        return false;
    }

    EngineConfig cfg = make_engine_config(opts);
    if (!inspect_prefill_seq_len(cfg.prefill_graph_path, prefill_seq_len, error)) {
        return false;
    }

    if (!engine.load(cfg)) {
        error = "failed to load engine graphs";
        return false;
    }

    return true;
}

std::string decode_piece(const Tokenizer& tokenizer, int token_id) {
    return tokenizer.decode(std::vector<int>{token_id});
}

GenerationMetrics compute_generation_metrics(size_t prompt_tokens,
                                             const GenerationResult& result) {
    GenerationMetrics metrics;
    metrics.prompt_tokens = static_cast<int>(prompt_tokens);
    metrics.generated_tokens = static_cast<int>(result.token_ids.size());
    metrics.decode_tokens = metrics.generated_tokens > 0 ? metrics.generated_tokens - 1 : 0;
    metrics.ttft_ms = result.prefill_ms;
    metrics.tpot_ms = (metrics.decode_tokens > 0 && result.decode_ms > 0.0)
        ? (result.decode_ms / metrics.decode_tokens)
        : 0.0;
    metrics.prefill_tps = result.prefill_ms > 0.0
        ? (1000.0 * prompt_tokens / result.prefill_ms)
        : 0.0;
    metrics.decode_tps = result.decode_ms > 0.0
        ? (1000.0 * metrics.decode_tokens / result.decode_ms)
        : 0.0;
    metrics.total_ms = result.prefill_ms + result.decode_ms;
    return metrics;
}

bool generate_greedy(LLMEngine& engine, const Tokenizer& tokenizer,
                     const std::vector<int>& prompt_ids, int max_new_tokens,
                     int eos_id, GenerationResult& result, std::string& error,
                     const std::function<void(int, const std::string&)>& on_token,
                     bool reset_context) {
    result = GenerationResult();
    if (prompt_ids.empty()) {
        error = "prompt is empty after tokenization";
        return false;
    }
    if (max_new_tokens < 1) {
        error = "max_new_tokens must be >= 1";
        return false;
    }

    // Reset context for single-turn mode (bench, --prompt).
    // Multi-turn REPL passes reset_context=false to preserve history.
    if (reset_context) {
        engine.reset();
    }

    auto prefill_start = std::chrono::steady_clock::now();
    int next = engine.prefill(prompt_ids);
    auto prefill_end = std::chrono::steady_clock::now();
    result.prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
    if (next < 0) {
        error = "engine prefill failed";
        return false;
    }

    auto append_token = [&](int token_id) {
        result.token_ids.push_back(token_id);
        std::string piece = decode_piece(tokenizer, token_id);
        result.text += piece;
        if (on_token) on_token(token_id, piece);
    };

    append_token(next);
    if (next == eos_id) {
        result.hit_eos = true;
        return true;
    }

    auto decode_start = std::chrono::steady_clock::now();
    while ((int)result.token_ids.size() < max_new_tokens) {
        next = engine.decode(next);
        if (next < 0) {
            error = "engine decode failed";
            return false;
        }
        append_token(next);
        if (next == eos_id) {
            result.hit_eos = true;
            break;
        }
    }
    auto decode_end = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    // Park workers after generation completes to drop idle CPU to ~0%
    // while the user is reading the output / typing the next prompt.
    // Auto-resumes on the next prefill() call.
    engine.park_workers();
    return true;
}
