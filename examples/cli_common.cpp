#include "examples/cli_common.h"

#include "graph/graph.h"
#include "kernels/threading.h"

#include <algorithm>
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
        } else if (arg == "--package") {
            if (!require_value(argc, argv, i, "--package", value, error)) return false;
            opts.package_path = value;
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
        } else if (arg == "--static-padded") {
            opts.static_padded = true;
        } else if (arg == "--device") {
            if (!require_value(argc, argv, i, "--device", value, error)) return false;
            std::string dev(value);
            if (dev == "cpu") {
                opts.device = Device::CPU;
            } else if (dev == "metal") {
#ifdef MOLLM_METAL
                opts.device = Device::METAL;
#else
                error = "--device metal requires a build with -DMOLLM_METAL=ON";
                return false;
#endif
            } else {
                error = std::string("unknown --device value '") + dev + "' (use cpu|metal)";
                return false;
            }
        } else if (arg == "--mmap") {
            opts.weight_loading = WeightLoadingMode::MMAP;
        } else if (arg == "--ssd-cache-mb") {
            if (!require_value(argc, argv, i, "--ssd-cache-mb", value, error)) return false;
            if (!parse_int(value, opts.ssd_cache_mb) || opts.ssd_cache_mb < 1) {
                error = "invalid value for --ssd-cache-mb";
                return false;
            }
        } else if (arg == "--ssd-io-workers") {
            if (!require_value(argc, argv, i, "--ssd-io-workers", value, error)) return false;
            if (!parse_int(value, opts.ssd_io_workers) || opts.ssd_io_workers < 1) {
                error = "invalid value for --ssd-io-workers";
                return false;
            }
        } else if (arg == "--ssd-cross-layer-prefetch") {
            opts.ssd_cross_layer_prefetch = true;
        } else if (arg == "--no-ssd-cross-layer-prefetch") {
            opts.ssd_cross_layer_prefetch = false;
        } else if (arg == "--trace") {
            if (!require_value(argc, argv, i, "--trace", value, error)) return false;
            opts.trace_path = value;
            if (opts.trace_path.empty()) {
                error = "--trace path must not be empty";
                return false;
            }
        } else if (arg == "--load-warmup") {
            opts.load_warmup = true;
            opts.load_warmup_explicit = true;
        } else if (arg == "--no-load-warmup") {
            opts.load_warmup = false;
            opts.load_warmup_explicit = true;
        } else if (arg == "--lock-dense-weights") {
            opts.lock_dense_weights = true;
        } else if (arg == "--no-lock-dense-weights") {
            opts.lock_dense_weights = false;
        } else if (arg == "--warmup") {
            if (!require_value(argc, argv, i, "--warmup", value, error)) return false;
            if (!parse_int(value, opts.warmup) || opts.warmup < 0) {
                error = "invalid value for --warmup";
                return false;
            }
        } else if (arg == "--temperature") {
            if (!require_value(argc, argv, i, "--temperature", value, error)) return false;
            opts.temperature = (float)std::atof(value);
        } else if (arg == "--top-k") {
            if (!require_value(argc, argv, i, "--top-k", value, error)) return false;
            if (!parse_int(value, opts.top_k) || opts.top_k < 0) {
                error = "invalid value for --top-k";
                return false;
            }
        } else if (arg == "--top-p") {
            if (!require_value(argc, argv, i, "--top-p", value, error)) return false;
            opts.top_p = (float)std::atof(value);
        } else if (arg == "--seed") {
            if (!require_value(argc, argv, i, "--seed", value, error)) return false;
            if (!parse_int(value, opts.seed)) {
                error = "invalid value for --seed";
                return false;
            }
        } else if (arg == "--output") {
            if (!require_value(argc, argv, i, "--output", value, error)) return false;
            opts.output_format = value;
            if (opts.output_format != "kv" && opts.output_format != "human") {
                error = "invalid value for --output (expected: kv|human)";
                return false;
            }
        } else {
            error = std::string("unknown argument: ") + arg;
            return false;
        }
    }

    if (opts.package_path.empty()) {
        error = "missing required --package <file.mollm>";
        return false;
    }
    if (opts.ssd_cache_mb > 0) {
        // SSD offload uses explicit pread() rather than touching aggregate
        // expert mappings. Keep default startup light, but honour an explicit
        // --load-warmup: Engine then touches only dense mmap weights.
        opts.weight_loading = WeightLoadingMode::MMAP;
        if (!opts.load_warmup_explicit) opts.load_warmup = false;
    }

    return true;
}

void print_common_usage(const char* program_name, const char* extra_usage) {
    std::printf("Usage: %s --package <file.mollm> [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  --package <file.mollm>    Single-file model package (required)\n");
    std::printf("  --prompt <text>           Run one prompt and exit\n");
    std::printf("  --prompt-file <path>      Read prompt text from file, run and exit\n");
    std::printf("  --prompt-tokens <int>     Use N dummy tokens (skip chat template)\n");
    std::printf("  --max-new-tokens <int>    Default: 2048\n");
    std::printf("  --n-ctx <int>             Default: 16384\n");
    std::printf("  --rope-dim <int>          Default: 64\n");
    std::printf("  --rope-theta <float>      Default: 1600000\n");
    std::printf("  --threads <int>          Default: 4\n");
    std::printf("  --profile                Print aggregated per-op profile in bench\n");
    std::printf("  --static-padded          Pad short prompts to graph_seq_len (A/B vs DYNAMIC)\n");
    std::printf("  --device <cpu|metal>     Compute backend (metal requires MOLLM_METAL build)\n");
    std::printf("  --mmap                  Use mmap-backed package weights (default: resident)\n");
    std::printf("  --ssd-cache-mb <int>    CPU MoE SSD cache; pins dense weights by default\n");
    std::printf("  --ssd-io-workers <int>  Dedicated SSD pread workers (default: 8)\n");
    std::printf("  --ssd-cross-layer-prefetch  Experimental next-layer gate prefetch\n");
    std::printf("  --trace <path.json>     Write Chrome Trace / Perfetto timing data\n");
    std::printf("  --load-warmup           Touch mmap'd package weights after load (dense-only with SSD offload)\n");
    std::printf("  --lock-dense-weights    Pin dense mmap weights in RAM (default with SSD offload)\n");
    std::printf("  --no-lock-dense-weights Keep dense mmap weights pageable\n");
    std::printf("  --no-load-warmup        Skip mmap page-in warmup\n");
    std::printf("  --warmup <int>            Default: 1 (used by benchmark)\n");
    std::printf("  --temperature <float>     Default: 0.6 (0 = greedy)\n");
    std::printf("  --top-k <int>             Default: 50 (0 = disabled)\n");
    std::printf("  --top-p <float>           Default: 0.9 (0 = disabled)\n");
    std::printf("  --seed <int>              Default: 42\n");
    std::printf("  --output <kv|human>       bench output format (default: kv)\n");
    if (extra_usage && *extra_usage) {
        std::printf("%s", extra_usage);
        if (extra_usage[std::strlen(extra_usage) - 1] != '\n') std::printf("\n");
    }
}

EngineConfig make_engine_config(const CliCommonOptions& opts) {
    EngineConfig cfg;
    cfg.package_path = opts.package_path;
    cfg.n_ctx = opts.n_ctx;
    cfg.rope_dim = opts.rope_dim;
    cfg.rope_theta = opts.rope_theta;
    cfg.num_threads = opts.num_threads;
    cfg.temperature = opts.temperature;
    cfg.top_k = opts.top_k;
    cfg.top_p = opts.top_p;
    cfg.seed = (unsigned int)opts.seed;
    cfg.static_padded = opts.static_padded;
    cfg.device = opts.device;
    cfg.weight_loading = opts.weight_loading;
    cfg.moe_ssd_cache_bytes = static_cast<size_t>(opts.ssd_cache_mb) * 1024 * 1024;
    cfg.moe_ssd_io_workers = opts.ssd_io_workers;
    cfg.moe_ssd_cross_layer_prefetch = opts.ssd_cross_layer_prefetch;
    cfg.trace_path = opts.trace_path;
    cfg.lock_dense_weights = opts.lock_dense_weights;
    return cfg;
}

bool load_runtime(const CliCommonOptions& opts, Tokenizer& tokenizer,
                  LLMEngine& engine, int& prefill_seq_len, std::string& error) {
    EngineConfig cfg = make_engine_config(opts);

    if (!engine.load(cfg)) {
        error = "failed to load engine package";
        return false;
    }
    // Load tokenizer from extracted path (engine sets cfg_.tokenizer_path)
    std::string tok_path = engine.config().tokenizer_path;
    if (tok_path.empty()) {
        error = "package did not expose a tokenizer path";
        return false;
    }
    if (!tokenizer.load(tok_path)) {
        error = std::string("failed to load tokenizer: ") + tok_path;
        return false;
    }
    auto rwkv_template = engine.package_metadata().find("rwkv_chat_template");
    if (rwkv_template != engine.package_metadata().end() &&
        rwkv_template->second == "rwkv_legacy") {
        tokenizer.set_rwkv_legacy_chat_template(true);
    }
    prefill_seq_len = 256;  // TODO: expose from engine
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
        engine.park_workers();
        return false;
    }

    const std::vector<std::string> stop_sequences = tokenizer.stop_sequences();
    // Keep the normal Transformer path exactly token-based. This is the hot
    // decode loop for every non-RWKV model, so do not put RWKV string matching
    // or an extra per-token branch on it.
    if (stop_sequences.empty()) {
        auto append_token = [&](int token_id) {
            result.token_ids.push_back(token_id);
            std::string piece = decode_piece(tokenizer, token_id);
            result.text += piece;
            if (on_token) on_token(token_id, piece);
        };
        append_token(next);
        if (next == eos_id) {
            result.hit_eos = true;
            engine.park_workers();
            return true;
        }
        auto decode_start = std::chrono::steady_clock::now();
        while ((int)result.token_ids.size() < max_new_tokens) {
            next = engine.decode(next);
            if (next < 0) {
                error = "engine decode failed";
                engine.park_workers();
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
        engine.park_workers();
        return true;
    }

    // RWKV legacy path only: stop by decoded bytes because the delimiter may
    // be split across several RWKV-world tokens.
    size_t max_stop_len = 0;
    for (const auto& stop : stop_sequences) max_stop_len = std::max(max_stop_len, stop.size());
    std::string pending_text;

    // Keep up to max_stop_len-1 bytes pending so a delimiter split across
    // RWKV byte tokens is neither emitted nor missed in streaming mode.
    auto emit = [&](const std::string& text, int token_id) {
        if (text.empty()) return;
        result.text += text;
        if (on_token) on_token(token_id, text);
    };
    auto append_token = [&](int token_id) {
        result.token_ids.push_back(token_id);
        std::string piece = decode_piece(tokenizer, token_id);
        pending_text += piece;
        size_t stop_at = std::string::npos;
        for (const auto& stop : stop_sequences) {
            size_t at = pending_text.find(stop);
            if (at != std::string::npos && (stop_at == std::string::npos || at < stop_at)) stop_at = at;
        }
        if (stop_at != std::string::npos) {
            emit(pending_text.substr(0, stop_at), token_id);
            pending_text.clear();
            return true;
        }
        if (pending_text.size() > max_stop_len - 1) {
            size_t n = pending_text.size() - (max_stop_len - 1);
            emit(pending_text.substr(0, n), token_id);
            pending_text.erase(0, n);
        }
        return false;
    };

    bool hit_text_stop = append_token(next);
    if (hit_text_stop || next == eos_id) {
        result.hit_eos = true;
        engine.park_workers();
        return true;
    }

    auto decode_start = std::chrono::steady_clock::now();
    while ((int)result.token_ids.size() < max_new_tokens) {
        next = engine.decode(next);
        if (next < 0) {
            error = "engine decode failed";
            engine.park_workers();
            return false;
        }
        hit_text_stop = append_token(next);
        if (hit_text_stop || next == eos_id) {
            result.hit_eos = true;
            break;
        }
    }
    // No delimiter was found: flush the bytes held back for boundary matching.
    emit(pending_text, result.token_ids.empty() ? -1 : result.token_ids.back());
    auto decode_end = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    // Park workers after generation completes to drop idle CPU to ~0%
    // while the user is reading the output / typing the next prompt.
    // Auto-resumes on the next prefill() call.
    engine.park_workers();
    return true;
}

std::vector<int> apply_chat_template(const Tokenizer& tokenizer,
                                      const std::string& user_message,
                                      const std::string& system_prompt) {
    // Qwen3.5 ChatML format:
    // <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
    //
    // Special token IDs (fixed for Qwen3.5 tokenizer):
    //   248045 = <|im_start|>
    //   248046 = <|im_end|>
    //   198    = \n (newline)
    //
    // "system", "user", "assistant" are regular tokens encoded by the tokenizer.
    std::vector<int> ids;

    auto encode_str = [&](const std::string& text) {
        auto encoded = tokenizer.encode(text);
        ids.insert(ids.end(), encoded.begin(), encoded.end());
    };

    // <|im_start|>system\n
    ids.push_back(248045);  // <|im_start|>
    encode_str("system");
    ids.push_back(198);     // \n

    // {system_prompt}
    encode_str(system_prompt);

    // <|im_end|>\n
    ids.push_back(248046);  // <|im_end|>
    ids.push_back(198);     // \n

    // <|im_start|>user\n
    ids.push_back(248045);  // <|im_start|>
    encode_str("user");
    ids.push_back(198);     // \n

    // {user_message}
    encode_str(user_message);

    // <|im_end|>\n
    ids.push_back(248046);  // <|im_end|>
    ids.push_back(198);     // \n

    // <|im_start|>assistant\n
    ids.push_back(248045);  // <|im_start|>
    encode_str("assistant");
    ids.push_back(198);     // \n

    return ids;
}
