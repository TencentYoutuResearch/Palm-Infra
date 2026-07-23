#include "kernels/matmul_profile.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

double g_pack_a_ms = 0;
double g_matmul_ms = 0;
double g_q8_quant_a_ms = 0;
long long g_pack_a_calls = 0;
long long g_q8_quant_a_calls = 0;
std::mutex g_prof_mtx;

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "0") != 0;
}

bool matmul_shape_profile_enabled() {
    static bool enabled = env_flag_enabled("MOLLM_MATMUL_SHAPE_PROFILE");
    return enabled;
}

const char* g_matmul_profile_phase = "unscoped";
std::vector<MatmulShapeProfileRow> g_matmul_shape_profile;

bool same_shape_profile_key(const MatmulShapeProfileRow& row,
                            const MatmulShapeProfileRow& key) {
    return std::strcmp(row.phase, key.phase) == 0 &&
           std::strcmp(row.path, key.path) == 0 && row.M == key.M &&
           row.N == key.N && row.K == key.K &&
           row.group_size == key.group_size &&
           row.groups_per_row == key.groups_per_row &&
           row.threads == key.threads &&
           row.has_q8_repack == key.has_q8_repack &&
           row.b_interleaved == key.b_interleaved;
}

void record_matmul_shape_profile_locked(const MatmulShapeProfileRow& key,
                                        double elapsed_ms) {
    for (auto& row : g_matmul_shape_profile) {
        if (same_shape_profile_key(row, key)) {
            row.calls += 1;
            row.total_ms += elapsed_ms;
            return;
        }
    }
    MatmulShapeProfileRow row = key;
    row.calls = 1;
    row.total_ms = elapsed_ms;
    g_matmul_shape_profile.push_back(row);
}

} // namespace

extern "C" double mollm_pack_a_total_ms() { return g_pack_a_ms; }
extern "C" long long mollm_pack_a_calls() { return g_pack_a_calls; }
extern "C" double mollm_matmul_total_ms() { return g_matmul_ms; }
extern "C" double mollm_q8_quant_a_total_ms() { return g_q8_quant_a_ms; }
extern "C" long long mollm_q8_quant_a_calls() { return g_q8_quant_a_calls; }

extern "C" int mollm_matmul_shape_profile_enabled() {
    return matmul_shape_profile_enabled() ? 1 : 0;
}

extern "C" void mollm_set_matmul_profile_phase(const char* phase) {
    if (!matmul_shape_profile_enabled())
        return;
    std::lock_guard<std::mutex> lock(g_prof_mtx);
    g_matmul_profile_phase = (phase && phase[0]) ? phase : "unscoped";
}

extern "C" void mollm_reset_matmul_shape_profile() {
    std::lock_guard<std::mutex> lock(g_prof_mtx);
    g_matmul_shape_profile.clear();
    g_matmul_profile_phase = "unscoped";
}

extern "C" void mollm_print_matmul_shape_profile(const char* title, int top_n) {
    if (!matmul_shape_profile_enabled())
        return;

    std::vector<MatmulShapeProfileRow> rows;
    {
        std::lock_guard<std::mutex> lock(g_prof_mtx);
        rows = g_matmul_shape_profile;
    }
    if (rows.empty())
        return;

    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.total_ms > b.total_ms;
    });

    double total_ms = 0.0;
    for (const auto& row : rows)
        total_ms += row.total_ms;
    if (top_n <= 0 || top_n > (int)rows.size())
        top_n = (int)rows.size();

    std::printf("\n[%s]\n", title && title[0] ? title : "matmul_shape_profile");
    std::printf(
        "  %-16s %-28s %5s %7s %6s %7s %7s %4s %3s %3s %8s %10s %9s %7s %9s\n",
        "phase", "path", "M", "N", "K", "group", "groups", "thr", "q8r", "int",
        "calls", "total_ms", "avg_ms", "pct", "GMAC/s");
    std::printf(
        "  %-16s %-28s %5s %7s %6s %7s %7s %4s %3s %3s %8s %10s %9s %7s %9s\n",
        "---", "---", "---", "---", "---", "---", "---", "---", "---", "---",
        "---", "---", "---", "---", "---");
    for (int i = 0; i < top_n; i++) {
        const auto& row = rows[i];
        double avg_ms = row.calls > 0 ? row.total_ms / row.calls : 0.0;
        double pct = total_ms > 0.0 ? row.total_ms * 100.0 / total_ms : 0.0;
        double gmac = (double)row.calls * (double)row.M * (double)row.N *
                      (double)row.K / 1e9;
        double gmac_s =
            row.total_ms > 0.0 ? gmac / (row.total_ms / 1000.0) : 0.0;
        std::printf("  %-16s %-28s %5d %7d %6d %7d %7d %4d %3d %3d %8lld "
                    "%10.2f %9.3f %6.1f%% %9.2f\n",
                    row.phase, row.path, row.M, row.N, row.K, row.group_size,
                    row.groups_per_row, row.threads, row.has_q8_repack ? 1 : 0,
                    row.b_interleaved ? 1 : 0, row.calls, row.total_ms, avg_ms,
                    pct, gmac_s);
    }
}

extern "C" void mollm_reset_pack_counters() {
    std::lock_guard<std::mutex> lock(g_prof_mtx);
    g_pack_a_ms = 0;
    g_pack_a_calls = 0;
    g_q8_quant_a_ms = 0;
    g_q8_quant_a_calls = 0;
    g_matmul_ms = 0;
}

MatmulTimer::MatmulTimer() : t0_(std::chrono::steady_clock::now()) {}

void MatmulTimer::set_shape(const char* path, int M, int N, int K,
                            int group_size, int groups_per_row,
                            bool has_q8_repack, bool b_interleaved,
                            int threads) {
    if (!matmul_shape_profile_enabled())
        return;
    shape_.path = path ? path : "unknown";
    shape_.M = M;
    shape_.N = N;
    shape_.K = K;
    shape_.group_size = group_size;
    shape_.groups_per_row = groups_per_row;
    shape_.has_q8_repack = has_q8_repack;
    shape_.b_interleaved = b_interleaved;
    shape_.threads = threads;
    shape_valid_ = true;
}

MatmulTimer::~MatmulTimer() {
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0_).count();
    std::lock_guard<std::mutex> lock(g_prof_mtx);
    g_matmul_ms += elapsed_ms;
    if (shape_valid_ && matmul_shape_profile_enabled()) {
        shape_.phase = g_matmul_profile_phase;
        record_matmul_shape_profile_locked(shape_, elapsed_ms);
    }
}

void matmul_record_pack_a(double elapsed_ms) {
    std::lock_guard<std::mutex> lock(g_prof_mtx);
    g_pack_a_ms += elapsed_ms;
    g_pack_a_calls++;
}

void matmul_record_q8_quant_a(double elapsed_ms) {
    std::lock_guard<std::mutex> lock(g_prof_mtx);
    g_q8_quant_a_ms += elapsed_ms;
    g_q8_quant_a_calls++;
}
