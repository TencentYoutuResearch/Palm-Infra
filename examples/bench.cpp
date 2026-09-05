#include "examples/cli_common.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

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
void human_row_text(const char* key, const char* value) {
    std::printf(" %-14s %12s\n", key, value);
}

uint64_t ssd_wait_ns(const MoeSsdCache::Stats& stats) {
    uint64_t total = 0;
    for (const auto& layer : stats.layers) total += layer.acquire_wait_ns;
    return total;
}

void print_kv_ssd_phase(const char* phase, const MoeSsdCache::Stats& stats,
                        int tokens, const char* token_kind) {
    const double divisor = tokens > 0 ? static_cast<double>(tokens) : 0.0;
    auto mb_per_token = [&](uint64_t bytes) {
        return divisor > 0.0 ? bytes / 1e6 / divisor : 0.0;
    };
    std::printf("moe_ssd_%s_hits=%llu\n", phase,
                static_cast<unsigned long long>(stats.hits));
    std::printf("moe_ssd_%s_misses=%llu\n", phase,
                static_cast<unsigned long long>(stats.misses));
    std::printf("moe_ssd_%s_evictions=%llu\n", phase,
                static_cast<unsigned long long>(stats.evictions));
    std::printf("moe_ssd_%s_logical_load_mb=%.3f\n", phase,
                stats.bytes_read / 1e6);
    std::printf("moe_ssd_%s_demand_origin_load_mb=%.3f\n", phase,
                stats.demand_load_bytes / 1e6);
    std::printf("moe_ssd_%s_prefetch_origin_load_mb=%.3f\n", phase,
                stats.prefetch_load_bytes / 1e6);
    std::printf("moe_ssd_%s_useful_prefetch_mb=%.3f\n", phase,
                stats.useful_prefetch_bytes / 1e6);
    std::printf("moe_ssd_%s_unused_prefetch_evicted_mb=%.3f\n", phase,
                stats.unused_prefetch_bytes / 1e6);
    std::printf("moe_ssd_%s_expert_bytes_acquired_mb=%.3f\n", phase,
                stats.expert_bytes_acquired / 1e6);
    std::printf("moe_ssd_%s_wait_ms=%.3f\n", phase,
                ssd_wait_ns(stats) / 1e6);
    std::printf("moe_ssd_%s_slot_waits=%llu\n", phase,
                static_cast<unsigned long long>(stats.slot_waits));
    std::printf("moe_ssd_%s_slot_wait_ms=%.3f\n", phase,
                stats.slot_wait_ns / 1e6);
    std::printf("moe_ssd_%s_logical_load_mb_per_%s=%.6f\n", phase,
                token_kind, mb_per_token(stats.bytes_read));
    std::printf("moe_ssd_%s_demand_origin_load_mb_per_%s=%.6f\n", phase,
                token_kind, mb_per_token(stats.demand_load_bytes));
    std::printf("moe_ssd_%s_prefetch_origin_load_mb_per_%s=%.6f\n", phase,
                token_kind, mb_per_token(stats.prefetch_load_bytes));
    std::printf("moe_ssd_%s_expert_bytes_acquired_mb_per_%s=%.6f\n", phase,
                token_kind, mb_per_token(stats.expert_bytes_acquired));
    std::printf("moe_ssd_%s_wait_ms_per_%s=%.6f\n", phase,
                token_kind,
                divisor > 0.0 ? ssd_wait_ns(stats) / 1e6 / divisor : 0.0);
}

void print_human_ssd_phase(const MoeSsdCache::Stats& stats, int tokens,
                           const char* token_kind) {
    const double divisor = tokens > 0 ? static_cast<double>(tokens) : 0.0;
    auto mb_per_token = [&](uint64_t bytes) {
        return divisor > 0.0 ? bytes / 1e6 / divisor : 0.0;
    };
    human_row_int("ssd_hits", static_cast<long long>(stats.hits), "");
    human_row_int("ssd_misses", static_cast<long long>(stats.misses), "");
    human_row("ssd_load_mb", stats.bytes_read / 1e6, "logical MB");
    human_row("ssd_demand_mb", stats.demand_load_bytes / 1e6, "logical MB");
    human_row("ssd_prefetch_mb", stats.prefetch_load_bytes / 1e6, "logical MB");
    human_row("ssd_useful_pf_mb", stats.useful_prefetch_bytes / 1e6, "MB");
    human_row("ssd_unused_pf_mb", stats.unused_prefetch_bytes / 1e6, "MB");
    human_row("ssd_acquired_mb", stats.expert_bytes_acquired / 1e6, "logical MB");
    human_row("ssd_wait_ms", ssd_wait_ns(stats) / 1e6, "ms");
    human_row_int("ssd_slot_waits",
                  static_cast<long long>(stats.slot_waits), "");
    human_row("ssd_slot_wait_ms", stats.slot_wait_ns / 1e6, "ms");
    char key[48];
    std::snprintf(key, sizeof(key), "ssd_load_mb/%s", token_kind);
    human_row(key, mb_per_token(stats.bytes_read), "MB");
    std::snprintf(key, sizeof(key), "ssd_acq_mb/%s", token_kind);
    human_row(key, mb_per_token(stats.expert_bytes_acquired), "MB");
}

// Default machine-parseable key/value output.
void print_generated_token_ids(const GenerationResult& result) {
    std::printf("generated_token_ids=");
    for (size_t i = 0; i < result.token_ids.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", result.token_ids[i]);
    }
    std::printf("\n");
}

// Default machine-parseable output (byte-identical to pre-polish behavior).
void print_kv_summary(double load_ms, double load_warmup_ms, size_t load_warmup_bytes,
                      const GenerationMetrics& m,
                      const GenerationResult& result, double total_ms,
                      const LLMEngine& engine, const CliCommonOptions& opts,
                      const MoeSsdCache::Stats& prefill_ssd,
                      const MoeSsdCache::Stats& decode_ssd,
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
        std::printf("moe_ssd_cache_mb=%.1f\n", engine.config().moe_ssd_cache_bytes / 1e6);
        std::printf("moe_ssd_io_workers=%d\n", engine.config().moe_ssd_io_workers);
        std::printf("moe_ssd_resident_mb=%.1f\n", decode_ssd.resident_bytes / 1e6);
        print_kv_ssd_phase("prefill", prefill_ssd, m.prompt_tokens,
                           "prompt_token");
        print_kv_ssd_phase("decode", decode_ssd, m.decode_tokens,
                           "decode_token");
        const auto& ssd = decode_ssd;
        std::printf("moe_ssd_decode_cross_layer_tasks=%llu moe_ssd_decode_cross_layer_dropped=%llu moe_ssd_decode_cross_layer_experts=%llu moe_ssd_decode_cross_layer_used=%llu moe_ssd_decode_cross_layer_rejected=%llu\n",
                    (unsigned long long)ssd.cross_layer_tasks,
                    (unsigned long long)ssd.cross_layer_dropped,
                    (unsigned long long)ssd.cross_layer_experts,
                    (unsigned long long)ssd.cross_layer_used,
                    (unsigned long long)ssd.cross_layer_rejected);
        std::printf("moe_ssd_decode_cross_layer_rank_accuracy=");
        for (size_t rank = 0; rank < ssd.cross_layer_rank_attempts.size(); ++rank) {
            if (rank != 0) std::printf(",");
            const uint64_t attempts = ssd.cross_layer_rank_attempts[rank];
            const uint64_t hits = ssd.cross_layer_rank_hits[rank];
            std::printf("%.3f", attempts == 0 ? 0.0
                                               : static_cast<double>(hits) / attempts);
        }
        std::printf("\n");
        std::printf("moe_ssd_decode_cross_layer_rank_confidence=");
        for (size_t i = 0; i < ssd.cross_layer_rank_attempts.size(); ++i) {
            if (i != 0) std::printf(",");
            const double sum = i < ssd.cross_layer_rank_confidence_sum.size()
                ? ssd.cross_layer_rank_confidence_sum[i] : 0.0;
            const uint64_t attempts = ssd.cross_layer_rank_attempts[i];
            std::printf("%.3f", attempts == 0 ? 0.0 : sum / attempts);
        }
        std::printf("\n");
        uint64_t total_wait_ns = 0;
        uint64_t total_prediction_attempts = 0;
        uint64_t total_prediction_matches = 0;
        uint64_t total_unused_prefetch_evictions = 0;
        uint64_t total_short_term_reloads = 0;
        for (const auto& layer : ssd.layers) {
            total_wait_ns += layer.acquire_wait_ns;
            total_prediction_attempts += layer.prediction_attempts;
            total_prediction_matches += layer.prediction_matches;
            total_unused_prefetch_evictions += layer.unused_prefetch_evictions;
            total_short_term_reloads += layer.short_term_reloads;
        }
        std::printf(
            "moe_ssd_decode_layer_summary_wait_ms=%.3f prediction_accuracy=%.3f "
            "unused_prefetch_evictions=%llu short_term_reloads=%llu\n",
            total_wait_ns / 1e6,
            total_prediction_attempts == 0
                ? 0.0
                : static_cast<double>(total_prediction_matches) /
                      total_prediction_attempts,
            (unsigned long long)total_unused_prefetch_evictions,
            (unsigned long long)total_short_term_reloads);
        std::printf("moe_ssd_decode_layer_stats=");
        for (size_t index = 0; index < ssd.layers.size(); ++index) {
            if (index != 0) std::printf(";");
            const auto& layer = ssd.layers[index];
            std::printf(
                "%d:%llu:%llu:%llu:%.3f:%llu:%llu:%llu:%llu:%llu:%llu",
                layer.layer,
                (unsigned long long)layer.demand_acquires,
                (unsigned long long)layer.demand_hits,
                (unsigned long long)layer.demand_misses,
                layer.acquire_wait_ns / 1e6,
                (unsigned long long)layer.prediction_attempts,
                (unsigned long long)layer.prediction_matches,
                (unsigned long long)layer.prefetch_selected,
                (unsigned long long)layer.prefetch_admitted,
                (unsigned long long)layer.unused_prefetch_evictions,
                (unsigned long long)layer.short_term_reloads);
        }
        std::printf("\n");
    }
    if (engine.config().mtp_draft_tokens > 0) {
        const auto& mtp = engine.mtp_stats();
        const double acceptance = mtp.drafted
            ? 100.0 * static_cast<double>(mtp.accepted) / mtp.drafted
            : 0.0;
        std::printf(
            "mtp_steps=%llu mtp_draft_calls=%llu mtp_drafted=%llu "
            "mtp_accepted=%llu mtp_fallback_steps=%llu "
            "mtp_verify_tokens=%llu mtp_sync_tokens=%llu "
            "mtp_acceptance_pct=%.1f mtp_total_ms=%.2f "
            "mtp_draft_ms=%.2f mtp_draft_model_ms=%.2f "
            "mtp_verify_ms=%.2f mtp_sample_ms=%.2f mtp_sync_ms=%.2f\n",
            static_cast<unsigned long long>(mtp.steps),
            static_cast<unsigned long long>(mtp.draft_calls),
            static_cast<unsigned long long>(mtp.drafted),
            static_cast<unsigned long long>(mtp.accepted),
            static_cast<unsigned long long>(mtp.fallback_steps),
            static_cast<unsigned long long>(mtp.verify_tokens),
            static_cast<unsigned long long>(mtp.sync_tokens), acceptance,
            mtp.total_ms, mtp.draft_ms, mtp.draft_model_ms,
            mtp.verify_ms, mtp.sample_ms, mtp.sync_ms);
        std::printf("mtp_depth_stats=");
        bool first = true;
        for (size_t depth = 0;
             depth < LLMEngine::MtpStats::kMaxDraftDepth; ++depth) {
            if (mtp.attempts_by_depth[depth] == 0)
                break;
            std::printf(
                "%s%zu:%llu/%llu/%llu", first ? "" : ",", depth + 1,
                static_cast<unsigned long long>(
                    mtp.attempts_by_depth[depth]),
                static_cast<unsigned long long>(
                    mtp.drafted_by_depth[depth]),
                static_cast<unsigned long long>(
                    mtp.accepted_by_depth[depth]));
            first = false;
        }
        std::printf("\n");
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
    if (opts.dump_token_ids) {
        print_generated_token_ids(result);
    }
}

// Human-readable output: top summary line + grouped aligned sections.
void print_human_summary(double load_ms, double load_warmup_ms, size_t load_warmup_bytes,
                         const GenerationMetrics& m,
                         const GenerationResult& result, double total_ms,
                         const LLMEngine& engine, const CliCommonOptions& opts,
                         const MoeSsdCache::Stats& prefill_ssd,
                         const MoeSsdCache::Stats& decode_ssd,
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
        human_row("moe_ssd_cache_mb", engine.config().moe_ssd_cache_bytes / 1e6, "MB");
        human_row_int("moe_ssd_io_workers", engine.config().moe_ssd_io_workers, "");
        human_row("moe_ssd_resident_mb", decode_ssd.resident_bytes / 1e6, "MB");
    }
    if (engine.config().mtp_draft_tokens > 0) {
        const auto& mtp = engine.mtp_stats();
        human_row_int("mtp_steps", (long long)mtp.steps, "");
        human_row_int("mtp_draft_calls", (long long)mtp.draft_calls, "");
        human_row_int("mtp_drafted", (long long)mtp.drafted, "");
        human_row_int("mtp_accepted", (long long)mtp.accepted, "");
        human_row_int("mtp_verify_tokens", (long long)mtp.verify_tokens, "");
        human_row_int("mtp_sync_tokens", (long long)mtp.sync_tokens, "");
        human_row("mtp_acceptance_pct",
                  mtp.drafted ? 100.0 * mtp.accepted / mtp.drafted : 0.0,
                  "%");
        human_row("mtp_total_ms", mtp.total_ms, "ms");
        human_row("mtp_draft_ms", mtp.draft_ms, "ms");
        human_row("mtp_draft_model_ms", mtp.draft_model_ms, "ms");
        human_row("mtp_verify_ms", mtp.verify_ms, "ms");
        human_row("mtp_sample_ms", mtp.sample_ms, "ms");
        human_row("mtp_sync_ms", mtp.sync_ms, "ms");
        for (size_t depth = 0;
             depth < LLMEngine::MtpStats::kMaxDraftDepth; ++depth) {
            if (mtp.attempts_by_depth[depth] == 0)
                break;
            char name[48];
            std::snprintf(name, sizeof(name), "mtp_depth_%zu_try/draft/accept",
                          depth + 1);
            char value[64];
            std::snprintf(
                value, sizeof(value), "%llu/%llu/%llu",
                static_cast<unsigned long long>(
                    mtp.attempts_by_depth[depth]),
                static_cast<unsigned long long>(
                    mtp.drafted_by_depth[depth]),
                static_cast<unsigned long long>(
                    mtp.accepted_by_depth[depth]));
            human_row_text(name, value);
        }
    }

    // prefill section
    std::printf("%s\n", kSepLight);
    std::printf(" prefill\n");
    std::printf("%s\n", kSepLight);
    human_row_int("prompt_tokens", m.prompt_tokens,    "");
    human_row("prefill_ms",    result.prefill_ms,      "ms");
    human_row("prefill_tps",   m.prefill_tps,          "t/s");
    if (engine.moe_ssd_offload_enabled())
        print_human_ssd_phase(prefill_ssd, m.prompt_tokens, "prompt_tok");

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
    if (engine.moe_ssd_offload_enabled())
        print_human_ssd_phase(decode_ssd, m.decode_tokens, "decode_tok");

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
    if (opts.dump_token_ids) {
        print_generated_token_ids(result);
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

    if (opts.prompt.empty() && !opts.prompt_file.empty()) {
        std::ifstream prompt_stream(
            opts.prompt_file, std::ios::in | std::ios::binary);
        if (!prompt_stream) {
            std::fprintf(
                stderr, "bench: cannot open --prompt-file: %s\n",
                opts.prompt_file.c_str());
            return 1;
        }
        std::ostringstream contents;
        contents << prompt_stream.rdbuf();
        opts.prompt = contents.str();
    }
    if (opts.prompt.empty() && opts.prompt_tokens <= 0)
        opts.prompt = "Hello, world!";

    Tokenizer tokenizer;
    LLMEngine engine;

    auto load_start = std::chrono::steady_clock::now();
    if (!load_runtime(opts, tokenizer, engine, error)) {
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
        if (!generate_tokens(engine, tokenizer, prompt_ids, opts.max_new_tokens,
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
    MoeSsdCache::Stats prefill_ssd;
    auto total_start = std::chrono::steady_clock::now();
    if (!generate_tokens(engine, tokenizer, prompt_ids, opts.max_new_tokens,
                         benchmark_eos_id, result, error, {}, true, nullptr, -1,
                         [&] {
                             if (!engine.moe_ssd_offload_enabled()) return;
                             prefill_ssd = engine.moe_ssd_stats();
                             engine.reset_moe_ssd_stats();
                         })) {
        std::fprintf(stderr, "bench: %s\n", error.c_str());
        return 1;
    }
    auto total_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    GenerationMetrics metrics = compute_generation_metrics(prompt_ids.size(), result);
    metrics.total_ms = total_ms;
    const MoeSsdCache::Stats decode_ssd = engine.moe_ssd_stats();

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
                            prefill_ssd, decode_ssd,
                            pack_ms, pack_calls, q8_quant_a_ms, q8_quant_a_calls, mm_ms);
    } else {
        print_kv_summary(load_ms, load_warmup_ms, load_warmup_bytes,
                         metrics, result, total_ms, engine, opts,
                         prefill_ssd, decode_ssd,
                         pack_ms, pack_calls, q8_quant_a_ms, q8_quant_a_calls, mm_ms);
    }

    if (opts.profile) {
        print_profile_section("prefill_profile", engine.prefill_exec_ctx());
        print_profile_section("decode_profile", engine.decode_exec_ctx());
        if (engine.has_mtp()) {
            print_profile_section("mtp_draft_profile", engine.mtp_exec_ctx());
            print_profile_section("mtp_verify_profile", engine.mtp_verify_exec_ctx());
        }
        if (mollm_matmul_shape_profile_enabled()) {
            mollm_print_matmul_shape_profile("matmul_shape_profile", 24);
        }
        if (mollm_moe_profile_enabled()) {
            mollm_print_moe_profile("moe_profile");
        }
    }

    return 0;
}
