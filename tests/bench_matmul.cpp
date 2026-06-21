#include "kernels/tensor.h"
#include "kernels/matmul.h"
#include "kernels/threading.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static void fill_rand(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

struct BenchConfig {
    int M = 128;
    int K = 2048;
    int N = 6144;
    int num_threads = 1;
    int warmup = 3;
    int repeat = 10;
};

struct BenchResult {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double p50_ms = 0.0;
    double gflops = 0.0;
};

static bool parse_int(const char* text, int& out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0') return false;
    out = static_cast<int>(value);
    return true;
}

static bool require_value(int argc, char** argv, int& i, const char* flag,
                          const char*& value) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "bench_matmul: missing value for %s\n", flag);
        return false;
    }
    value = argv[++i];
    return true;
}

static BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        const char* value = nullptr;

        if (arg == "--threads") {
            if (!require_value(argc, argv, i, "--threads", value)) std::exit(1);
            if (!parse_int(value, cfg.num_threads) || cfg.num_threads < 1) {
                std::fprintf(stderr, "bench_matmul: invalid --threads\n");
                std::exit(1);
            }
        } else if (arg == "--warmup") {
            if (!require_value(argc, argv, i, "--warmup", value)) std::exit(1);
            if (!parse_int(value, cfg.warmup) || cfg.warmup < 0) {
                std::fprintf(stderr, "bench_matmul: invalid --warmup\n");
                std::exit(1);
            }
        } else if (arg == "--repeat") {
            if (!require_value(argc, argv, i, "--repeat", value)) std::exit(1);
            if (!parse_int(value, cfg.repeat) || cfg.repeat < 1) {
                std::fprintf(stderr, "bench_matmul: invalid --repeat\n");
                std::exit(1);
            }
        } else if (arg == "--list") {
            std::printf("M=1 K=2048 N=2048\n");
            std::printf("M=1 K=2048 N=6144\n");
            std::printf("M=1 K=6144 N=2048\n");
            std::printf("M=128 K=2048 N=2048\n");
            std::printf("M=128 K=2048 N=6144\n");
            std::printf("M=128 K=6144 N=2048\n");
            std::printf("M=1 K=2048 N=128256\n");
            std::printf("M=128 K=2048 N=128256\n");
            std::exit(0);
        } else {
            // positional: M K N
            static int pos = 0;
            if (!parse_int(argv[i], pos == 0 ? cfg.M : pos == 1 ? cfg.K : cfg.N)) {
                std::fprintf(stderr, "bench_matmul: invalid positional arg: %s\n", argv[i]);
                std::fprintf(stderr, "Usage: bench_matmul [M K N] [--threads N] [--warmup N] [--repeat N] [--list]\n");
                std::exit(1);
            }
            pos++;
        }
    }
    return cfg;
}

static BenchResult run_bench(const BenchConfig& cfg) {
    int M = cfg.M, K = cfg.K, N = cfg.N;

    float* a_data = new float[M * K];
    float* b_data = new float[N * K];
    float* c_data = new float[M * N];
    fill_rand(a_data, M * K);
    fill_rand(b_data, N * K);

    Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
    Tensor B = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, N, K, 1, 1, b_data);
    Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

    ThreadPool pool(cfg.num_threads);
    ThreadPool* tp = (cfg.num_threads > 1) ? &pool : nullptr;

    // warmup
    for (int i = 0; i < cfg.warmup; i++) {
        kernel_matmul_fp32(A, B, C, tp);
    }

    // timed runs
    std::vector<double> times;
    times.reserve(cfg.repeat);
    for (int i = 0; i < cfg.repeat; i++) {
        auto start = std::chrono::steady_clock::now();
        kernel_matmul_fp32(A, B, C, tp);
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());

    BenchResult result;
    result.min_ms = times.front();
    result.max_ms = times.back();
    result.p50_ms = times[times.size() / 2];
    double sum = 0.0;
    for (double t : times) sum += t;
    result.avg_ms = sum / times.size();

    // GFLOPS: 2*M*N*K / time
    double flops = 2.0 * M * N * K;
    result.gflops = (result.avg_ms > 0.0) ? (flops / (result.avg_ms * 1e6)) : 0.0;

    delete[] a_data;
    delete[] b_data;
    delete[] c_data;
    return result;
}

int main(int argc, char** argv) {
    srand(42);
    BenchConfig cfg = parse_args(argc, argv);
    BenchResult result = run_bench(cfg);

    std::printf("M=%d K=%d N=%d threads=%d\n", cfg.M, cfg.K, cfg.N, cfg.num_threads);
    std::printf("  warmup=%d repeat=%d\n", cfg.warmup, cfg.repeat);
    std::printf("  avg=%.2fms min=%.2fms max=%.2fms p50=%.2fms\n",
                result.avg_ms, result.min_ms, result.max_ms, result.p50_ms);
    std::printf("  GFLOPS=%.1f\n", result.gflops);

    return 0;
}
