#pragma once

#include "engine/engine.h"
#include "engine/tokenizer.h"

#include <functional>
#include <string>
#include <vector>

struct CliCommonOptions {
    std::string package_path;   // .mollm single-file package (required)
    std::string prompt;
    std::string prompt_file;  // read prompt text from file (--prompt-file)
    int prompt_tokens = 0;   // >0: use N dummy tokens instead of --prompt text
    int max_new_tokens = 2048;
    int n_ctx = 16384;
    int rope_dim = 64;
    float rope_theta = 10000000.f;
    int num_threads = 4;
    bool profile = false;
    int warmup = 1;
    bool static_padded = false;  // pad short prompts to graph_seq_len (A/B vs DYNAMIC)
    Device device = Device::CPU;  // compute backend (--device cpu|metal)
    WeightLoadingMode weight_loading = WeightLoadingMode::RESIDENT;
    bool load_warmup = true;     // touch mmap'd package weights after load
    // Needed because SSD offload suppresses the default whole-package warmup,
    // but an explicit --load-warmup can still warm only dense mmap weights.
    bool load_warmup_explicit = false;
    // SSD expert offload benefits from keeping the always-used dense weights
    // resident. Users on constrained systems can opt out explicitly.
    bool lock_dense_weights = true;
    int ssd_cache_mb = 0;        // >0: page routed MoE experts from package
    int ssd_io_workers = 8;      // dedicated pread workers for MoE SSD cache
    bool ssd_cross_layer_prefetch = true;   // next-layer predictor (with global pool)
    int ssd_shallow_cache_layers = 0;  // early MoE layers with priority cache quota
    bool ssd_global_cache = true;   // dynamically shared cache capacity across layers
    std::string trace_path;      // optional Chrome Trace / Perfetto JSON output

    // Sampling
    float temperature = 0.6f;
    int top_k = 50;
    float top_p = 0.9f;
    int seed = 42;

    // bench output format: "kv" (default, machine-parseable) or "human"
    // (aligned summary blocks). Ignored by chat.
    std::string output_format = "kv";
};

struct GenerationResult {
    std::vector<int> token_ids;
    std::string text;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    bool hit_eos = false;
};

struct GenerationMetrics {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int decode_tokens = 0;
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
    double prefill_tps = 0.0;
    double decode_tps = 0.0;
    double total_ms = 0.0;
};

bool parse_common_args(int argc, char** argv, CliCommonOptions& opts,
                       std::string& error);
void print_common_usage(const char* program_name, const char* extra_usage = nullptr);
EngineConfig make_engine_config(const CliCommonOptions& opts);
bool load_runtime(const CliCommonOptions& opts, Tokenizer& tokenizer,
                  LLMEngine& engine, std::string& error);
std::string decode_piece(const Tokenizer& tokenizer, int token_id);
GenerationMetrics compute_generation_metrics(size_t prompt_tokens,
                                             const GenerationResult& result);
bool generate_greedy(LLMEngine& engine, const Tokenizer& tokenizer,
                     const std::vector<int>& prompt_ids, int max_new_tokens,
                     int eos_id, GenerationResult& result, std::string& error,
                     const std::function<void(int, const std::string&)>& on_token = {},
                     bool reset_context = true);

/// Apply Qwen3.5 ChatML template: wrap user message with special tokens.
/// Format: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
std::vector<int> apply_chat_template(const Tokenizer& tokenizer,
                                      const std::string& user_message,
                                      const std::string& system_prompt = "You are a helpful assistant.");
