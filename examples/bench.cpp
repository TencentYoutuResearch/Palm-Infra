#include "examples/cli_common.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

// Pack-A profiling counters (defined in kernels/matmul.cpp)
extern "C" {
double mollm_pack_a_total_ms();
long long mollm_pack_a_calls();
double mollm_matmul_total_ms();
void mollm_reset_pack_counters();
}
#include <string>
#include <vector>

namespace {

struct AggregatedProfileRow {
    OpType op_type = OpType::INPUT;
    uint64_t calls = 0;
    uint64_t total_ns = 0;
};

std::vector<AggregatedProfileRow> aggregate_profile(const ExecContext& ctx) {
    std::vector<AggregatedProfileRow> rows;
    for (const auto& stat : ctx.profile_stats) {
        if (stat.calls == 0 || stat.op_type == OpType::INPUT || stat.op_type == OpType::CONSTANT) {
            continue;
        }

        auto it = std::find_if(rows.begin(), rows.end(), [&](const AggregatedProfileRow& row) {
            return row.op_type == stat.op_type;
        });
        if (it == rows.end()) {
            rows.push_back({stat.op_type, stat.calls, stat.total_ns});
        } else {
            it->calls += stat.calls;
            it->total_ns += stat.total_ns;
        }
    }

    std::sort(rows.begin(), rows.end(), [](const AggregatedProfileRow& a, const AggregatedProfileRow& b) {
        return a.total_ns > b.total_ns;
    });
    return rows;
}

void print_profile_section(const char* title, const ExecContext& ctx) {
    auto rows = aggregate_profile(ctx);
    if (rows.empty()) return;

    uint64_t total_ns = 0;
    for (const auto& row : rows) total_ns += row.total_ns;

    std::printf("\n[%s]\n", title);
    // Aligned table: op name left-aligned, numbers right-aligned
    std::printf("  %-28s %8s %10s %10s %7s\n", "op", "calls", "total_ms", "avg_ms", "pct");
    std::printf("  %-28s %8s %10s %10s %7s\n", "---", "---", "---", "---", "---");
    for (const auto& row : rows) {
        double total_ms = row.total_ns / 1e6;
        double avg_ms = row.calls > 0 ? total_ms / row.calls : 0.0;
        double pct = total_ns > 0 ? (100.0 * row.total_ns / total_ns) : 0.0;
        std::printf("  %-28s %8llu %10.2f %10.3f %6.1f%%\n",
                    op_type_name(row.op_type),
                    (unsigned long long)row.calls,
                    total_ms,
                    avg_ms,
                    pct);
    }
}

} // namespace

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

    if (opts.prompt.empty() && opts.prompt_tokens <= 0) opts.prompt = "Hello, world!";

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

    engine.set_profile_enabled(opts.profile);

    std::vector<int> prompt_ids;
    if (opts.prompt_tokens > 0) {
        // Dummy-token mode: skip chat template, use raw token IDs.
        // Token 0 is always valid (embed() falls back to it for OOB).
        prompt_ids.assign(opts.prompt_tokens, 0);
    } else {
        prompt_ids = tokenizer.apply_chat(opts.prompt);
    }
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

    mollm_reset_pack_counters();

    if (opts.profile) {
        engine.reset_profiles();
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
    std::printf("threads=%d\n", engine.config().num_threads);
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
    // Only show generated_text for real prompts (dummy-token mode produces garbage)
    if (opts.prompt_tokens <= 0) {
        std::printf("generated_text=%s\n", result.text.c_str());
    }

    // Pack-A profiling: show how much of the run is spent packing A.
    double pack_ms = mollm_pack_a_total_ms();
    long long pack_calls = mollm_pack_a_calls();
    double mm_ms = mollm_matmul_total_ms();
    std::printf("pack_a_ms=%.2f pack_a_calls=%lld matmul_ms=%.2f pack_pct=%.1f%%\n",
                pack_ms, pack_calls, mm_ms,
                mm_ms > 0 ? (pack_ms / mm_ms * 100.0) : 0.0);

    if (opts.profile) {
        print_profile_section("prefill_profile", engine.prefill_exec_ctx());
        print_profile_section("decode_profile", engine.decode_exec_ctx());
    }

    return 0;
}
