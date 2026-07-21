#include "examples/cli_common.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

// Pack-A profiling counters (defined in kernels/matmul.cpp)
extern "C" {
double mollm_pack_a_total_ms();
long long mollm_pack_a_calls();
double mollm_matmul_total_ms();
double mollm_q8_quant_a_total_ms();
long long mollm_q8_quant_a_calls();
void mollm_reset_pack_counters();
int mollm_matmul_shape_profile_enabled();
void mollm_reset_matmul_shape_profile();
void mollm_print_matmul_shape_profile(const char* title, int top_n);
int mollm_moe_profile_enabled();
void mollm_reset_moe_profile();
void mollm_print_moe_profile(const char* title);
}
#include <string>
#include <vector>

// Peak RSS reporting (portable: getrusage works on macOS + Linux).
// Reports peak resident set size in bytes across the whole process lifetime,
// including mmap'd weights and all BufferPool allocations.
#include <sys/resource.h>

static double peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    // macOS: ru_maxrss is in bytes
    return ru.ru_maxrss / (1024.0 * 1024.0);
#else
    // Linux: ru_maxrss is in kilobytes
    return ru.ru_maxrss / 1024.0;
#endif
}

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

// 80-column light separator for human mode.
static const char* const kSepLight =
    "--------------------------------------------------------------------------------";

// Print one aligned row " key<10> value<12> unit" in human mode.
void human_row(const char* key, double value, const char* unit) {
    std::printf(" %-14s %12.2f %s\n", key, value, unit);
}
void human_row_int(const char* key, long long value, const char* unit) {
    std::printf(" %-14s %12lld %s\n", key, value, unit);
}

// Default machine-parseable output (byte-identical to pre-polish behavior).
void print_kv_summary(double load_ms, double load_warmup_ms, size_t load_warmup_bytes,
                      const GenerationMetrics& m,
                      const GenerationResult& result, double total_ms,
                      const LLMEngine& engine, const CliCommonOptions& opts,
                      double pack_ms, long long pack_calls,
                      double q8_quant_a_ms, long long q8_quant_a_calls,
                      double mm_ms) {
    std::printf("load_ms=%.2f\n", load_ms);
    std::printf("load_warmup_ms=%.2f\n", load_warmup_ms);
    std::printf("load_warmup_mb=%.1f\n", load_warmup_bytes / 1e6);
    std::printf("threads=%d\n", engine.config().num_threads);
    std::printf("prompt_tokens=%d\n", m.prompt_tokens);
    std::printf("generated_tokens=%d\n", m.generated_tokens);
    std::printf("decode_tokens=%d\n", m.decode_tokens);
    std::printf("ttft_ms=%.2f\n", m.ttft_ms);
    std::printf("tpot_ms=%.2f\n", m.tpot_ms);
    std::printf("prefill_tps=%.2f\n", m.prefill_tps);
    std::printf("decode_tps=%.2f\n", m.decode_tps);
    std::printf("prefill_ms=%.2f\n", result.prefill_ms);
    std::printf("decode_ms=%.2f\n", result.decode_ms);
    std::printf("total_ms=%.2f\n", total_ms);
    std::printf("peak_rss_mb=%.1f\n", peak_rss_mb());
    if (engine.moe_ssd_offload_enabled()) {
        auto ssd = engine.moe_ssd_stats();
        std::printf("moe_ssd_cache_mb=%.1f\n", engine.config().moe_ssd_cache_bytes / 1e6);
        std::printf("moe_ssd_io_workers=%d\n", engine.config().moe_ssd_io_workers);
        std::printf("moe_ssd_hits=%llu moe_ssd_misses=%llu moe_ssd_evictions=%llu moe_ssd_read_mb=%.1f moe_ssd_resident_mb=%.1f\n",
                    (unsigned long long)ssd.hits, (unsigned long long)ssd.misses,
                    (unsigned long long)ssd.evictions, ssd.bytes_read / 1e6,
                    ssd.resident_bytes / 1e6);
    }
    {
        auto pre = engine.prefill_pool_stats();
        auto dec = engine.decode_pool_stats();
        size_t active = pre.active + dec.active;
        size_t peak = pre.peak + dec.peak;
        size_t freelist = pre.freelist + dec.freelist;
        size_t acquires = pre.acquires + dec.acquires;
        size_t releases = pre.releases + dec.releases;
        std::printf("pool_active_mb=%.1f pool_peak_mb=%.1f pool_freelist_mb=%.1f pool_acquires=%zu pool_releases=%zu\n",
                    active / (1024.0 * 1024.0),
                    peak / (1024.0 * 1024.0),
                    freelist / (1024.0 * 1024.0),
                    acquires, releases);
        std::printf("prefill_pool_active_mb=%.1f prefill_pool_peak_mb=%.1f prefill_pool_freelist_mb=%.1f prefill_pool_acquires=%zu prefill_pool_releases=%zu\n",
                    pre.active / (1024.0 * 1024.0),
                    pre.peak / (1024.0 * 1024.0),
                    pre.freelist / (1024.0 * 1024.0),
                    pre.acquires, pre.releases);
        std::printf("decode_pool_active_mb=%.1f decode_pool_peak_mb=%.1f decode_pool_freelist_mb=%.1f decode_pool_acquires=%zu decode_pool_releases=%zu\n",
                    dec.active / (1024.0 * 1024.0),
                    dec.peak / (1024.0 * 1024.0),
                    dec.freelist / (1024.0 * 1024.0),
                    dec.acquires, dec.releases);
    }
    std::printf("hit_eos=%s\n", result.hit_eos ? "true" : "false");
    // Only show generated_text for real prompts (dummy-token mode produces garbage)
    if (opts.prompt_tokens <= 0) {
        std::printf("generated_text=%s\n", result.text.c_str());
    }
    std::printf("pack_a_ms=%.2f pack_a_calls=%lld q8_quant_a_ms=%.2f q8_quant_a_calls=%lld matmul_ms=%.2f pack_pct=%.1f%% q8_quant_a_pct=%.1f%%\n",
                pack_ms, pack_calls, q8_quant_a_ms, q8_quant_a_calls,
                mm_ms,
                mm_ms > 0 ? (pack_ms / mm_ms * 100.0) : 0.0,
                mm_ms > 0 ? (q8_quant_a_ms / mm_ms * 100.0) : 0.0);
}

// Human-readable output: top summary line + grouped aligned sections.
void print_human_summary(double load_ms, double load_warmup_ms, size_t load_warmup_bytes,
                         const GenerationMetrics& m,
                         const GenerationResult& result, double total_ms,
                         const LLMEngine& engine, const CliCommonOptions& opts,
                         double pack_ms, long long pack_calls,
                         double q8_quant_a_ms, long long q8_quant_a_calls,
                         double mm_ms) {
    // Top summary line — one-glance overview.
    std::printf("=== mollm bench ===  pp=%.1f t/s  tg=%.1f t/s  peak_rss=%.1f MB  load=%.1f ms  load_warmup=%.1f ms\n",
                m.prefill_tps, m.decode_tps, peak_rss_mb(), load_ms, load_warmup_ms);

    // load section
    std::printf("%s\n", kSepLight);
    std::printf(" load\n");
    std::printf("%s\n", kSepLight);
    human_row("load_ms",     load_ms,                    "ms");
    human_row("load_warmup_ms", load_warmup_ms,           "ms");
    human_row("load_warmup_mb", load_warmup_bytes / 1e6,  "MB");
    human_row_int("threads", engine.config().num_threads, "");
    if (engine.moe_ssd_offload_enabled()) {
        auto ssd = engine.moe_ssd_stats();
        human_row("moe_ssd_cache_mb", engine.config().moe_ssd_cache_bytes / 1e6, "MB");
        human_row_int("moe_ssd_io_workers", engine.config().moe_ssd_io_workers, "");
        human_row_int("moe_ssd_hits", (long long)ssd.hits, "");
        human_row_int("moe_ssd_misses", (long long)ssd.misses, "");
        human_row_int("moe_ssd_evictions", (long long)ssd.evictions, "");
        human_row("moe_ssd_read_mb", ssd.bytes_read / 1e6, "MB");
    }

    // prefill section
    std::printf("%s\n", kSepLight);
    std::printf(" prefill\n");
    std::printf("%s\n", kSepLight);
    human_row_int("prompt_tokens", m.prompt_tokens,    "");
    human_row("prefill_ms",    result.prefill_ms,      "ms");
    human_row("prefill_tps",   m.prefill_tps,          "t/s");

    // decode section
    std::printf("%s\n", kSepLight);
    std::printf(" decode\n");
    std::printf("%s\n", kSepLight);
    human_row_int("generated_tokens", m.generated_tokens, "");
    human_row_int("decode_tokens",    m.decode_tokens,    "");
    human_row("decode_ms",     result.decode_ms,        "ms");
    human_row("decode_tps",    m.decode_tps,            "t/s");
    human_row("ttft_ms",       m.ttft_ms,               "ms");
    human_row("tpot_ms",       m.tpot_ms,               "ms");
    human_row("total_ms",      total_ms,                "ms");

    // memory section
    std::printf("%s\n", kSepLight);
    std::printf(" memory\n");
    std::printf("%s\n", kSepLight);
    human_row("peak_rss_mb",   peak_rss_mb(),           "MB");
    {
        auto pre = engine.prefill_pool_stats();
        auto dec = engine.decode_pool_stats();
        size_t active = pre.active + dec.active;
        size_t peak = pre.peak + dec.peak;
        size_t freelist = pre.freelist + dec.freelist;
        size_t acquires = pre.acquires + dec.acquires;
        size_t releases = pre.releases + dec.releases;
        human_row("pool_active_mb",   active   / (1024.0 * 1024.0), "MB");
        human_row("pool_peak_mb",     peak     / (1024.0 * 1024.0), "MB");
        human_row("pool_freelist_mb", freelist / (1024.0 * 1024.0), "MB");
        human_row_int("pool_acquires",  (long long)acquires,  "");
        human_row_int("pool_releases",  (long long)releases,  "");

        // pool section (per-graph breakdown)
        std::printf("%s\n", kSepLight);
        std::printf(" pool (prefill / decode)\n");
        std::printf("%s\n", kSepLight);
        human_row("prefill_active_mb",   pre.active   / (1024.0 * 1024.0), "MB");
        human_row("prefill_peak_mb",     pre.peak     / (1024.0 * 1024.0), "MB");
        human_row("prefill_freelist_mb", pre.freelist / (1024.0 * 1024.0), "MB");
        human_row_int("prefill_acquires",  (long long)pre.acquires, "");
        human_row_int("prefill_releases",  (long long)pre.releases, "");
        human_row("decode_active_mb",    dec.active   / (1024.0 * 1024.0), "MB");
        human_row("decode_peak_mb",      dec.peak     / (1024.0 * 1024.0), "MB");
        human_row("decode_freelist_mb",  dec.freelist / (1024.0 * 1024.0), "MB");
        human_row_int("decode_acquires",   (long long)dec.acquires, "");
        human_row_int("decode_releases",   (long long)dec.releases, "");
    }

    // pack section
    std::printf("%s\n", kSepLight);
    std::printf(" pack\n");
    std::printf("%s\n", kSepLight);
    human_row("pack_a_ms",       pack_ms,       "ms");
    human_row_int("pack_a_calls",    pack_calls,    "");
    human_row("q8_quant_a_ms",   q8_quant_a_ms, "ms");
    human_row_int("q8_quant_a_calls", q8_quant_a_calls, "");
    human_row("matmul_ms",       mm_ms,         "ms");
    human_row("pack_pct",        mm_ms > 0 ? (pack_ms       / mm_ms * 100.0) : 0.0, "%");
    human_row("q8_quant_a_pct",  mm_ms > 0 ? (q8_quant_a_ms / mm_ms * 100.0) : 0.0, "%");

    // hit_eos + generated_text
    std::printf("hit_eos=%s\n", result.hit_eos ? "true" : "false");
    if (opts.prompt_tokens <= 0) {
        std::printf("%s\n", kSepLight);
        std::printf(" generated_text\n");
        std::printf("%s\n", kSepLight);
        std::printf(" %s\n", result.text.c_str());
        std::printf("%s\n", kSepLight);
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
                           "  --warmup <int>            Warmup iterations before timed run\n"
                           "  --output <kv|human>       Output format (default: kv)\n");
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

    double load_warmup_ms = 0.0;
    size_t load_warmup_bytes = 0;
    if (opts.load_warmup && engine.package_weights_mmap_backed()) {
        auto warmup_start = std::chrono::steady_clock::now();
        load_warmup_bytes = engine.warmup_package_weights();
        auto warmup_end = std::chrono::steady_clock::now();
        load_warmup_ms =
            std::chrono::duration<double, std::milli>(warmup_end - warmup_start).count();
    }

    engine.set_profile_enabled(opts.profile);

    std::vector<int> prompt_ids;
    if (opts.prompt_tokens > 0) {
        // Dummy-token mode: skip chat template, use raw token IDs.
        // Token 0 is always valid (embed() falls back to it for OOB).
        prompt_ids.assign(opts.prompt_tokens, 0);
        // Benchmarking must consume exactly the requested tg length. RWKV
        // legacy chat stop sequences are irrelevant for raw dummy tokens and
        // may otherwise terminate a run early on a generated "\n\n".
        tokenizer.set_rwkv_legacy_chat_template(false);
    } else {
        prompt_ids = tokenizer.apply_chat(opts.prompt);
    }
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "bench: prompt is empty after tokenization\n");
        return 1;
    }
    // Long prompts are handled by chunked prefill (prefill() splits into
    // graph_seq_len-sized chunks). No upper bound check here — only the
    // per-chunk size matters, and that's enforced inside the engine.

    const int benchmark_eos_id = opts.prompt_tokens > 0 ? -1 : tokenizer.eos_id();
    for (int i = 0; i < opts.warmup; i++) {
        GenerationResult warmup_result;
        std::string warmup_error;
        if (!generate_greedy(engine, tokenizer, prompt_ids, opts.max_new_tokens,
                             benchmark_eos_id, warmup_result, warmup_error)) {
            std::fprintf(stderr, "bench warmup failed: %s\n", warmup_error.c_str());
            return 1;
        }
    }

    mollm_reset_pack_counters();
    mollm_reset_matmul_shape_profile();
    mollm_reset_moe_profile();
    engine.reset_moe_ssd_stats();

    if (opts.profile) {
        engine.reset_profiles();
    }

    GenerationResult result;
    auto total_start = std::chrono::steady_clock::now();
    if (!generate_greedy(engine, tokenizer, prompt_ids, opts.max_new_tokens,
                         benchmark_eos_id, result, error)) {
        std::fprintf(stderr, "bench: %s\n", error.c_str());
        return 1;
    }
    auto total_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    GenerationMetrics metrics = compute_generation_metrics(prompt_ids.size(), result);
    metrics.total_ms = total_ms;

    // Pack-A profiling counters.
    double pack_ms = mollm_pack_a_total_ms();
    long long pack_calls = mollm_pack_a_calls();
    double q8_quant_a_ms = mollm_q8_quant_a_total_ms();
    long long q8_quant_a_calls = mollm_q8_quant_a_calls();
    double mm_ms = mollm_matmul_total_ms();

    // Dispatch to the selected output format. "kv" stays machine-parseable for
    // benchmark scripts.
    if (opts.output_format == "human") {
        print_human_summary(load_ms, load_warmup_ms, load_warmup_bytes,
                            metrics, result, total_ms, engine, opts,
                            pack_ms, pack_calls, q8_quant_a_ms, q8_quant_a_calls, mm_ms);
    } else {
        print_kv_summary(load_ms, load_warmup_ms, load_warmup_bytes,
                         metrics, result, total_ms, engine, opts,
                         pack_ms, pack_calls, q8_quant_a_ms, q8_quant_a_calls, mm_ms);
    }

    if (opts.profile) {
        print_profile_section("prefill_profile", engine.prefill_exec_ctx());
        print_profile_section("decode_profile", engine.decode_exec_ctx());
        if (mollm_matmul_shape_profile_enabled()) {
            mollm_print_matmul_shape_profile("matmul_shape_profile", 24);
        }
        if (mollm_moe_profile_enabled()) {
            mollm_print_moe_profile("moe_profile");
        }
    }

    return 0;
}
