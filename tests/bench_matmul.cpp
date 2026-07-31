#include "kernels/tensor.h"
#include "kernels/deepseek_v4_attention.h"
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
    int k_block = 0;       // 0 = use default (256)
    int chunk_size = 0;    // 0 = use adaptive
    bool disable_k_block = false;
    bool use_fp16 = false; // FP16 weight storage
    bool use_int8 = false; // INT8 weight storage
    bool use_int4 = false; // INT4 packed weight storage
    bool use_mxfp4 = false;
    bool use_fp32 = false; // explicitly FP32 (default)
    bool interleave_pack = true;   // B interleaved packing (FP16/INT8)
    bool no_interleave_pack = false;
    int group_size = 0;     // quant group size; 0 = per-channel
    bool sparse_a = false;  // benchmark ReLU-squared sparse-A GEMV
    int density_pct = 10;   // nonzero percentage for --sparse-a
    bool dsv4_grouped = false;
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
        } else if (arg == "--k-block") {
            if (!require_value(argc, argv, i, "--k-block", value)) std::exit(1);
            if (!parse_int(value, cfg.k_block) || cfg.k_block < 0) {
                std::fprintf(stderr, "bench_matmul: invalid --k-block\n");
                std::exit(1);
            }
        } else if (arg == "--chunk-size") {
            if (!require_value(argc, argv, i, "--chunk-size", value)) std::exit(1);
            if (!parse_int(value, cfg.chunk_size) || cfg.chunk_size < 1) {
                std::fprintf(stderr, "bench_matmul: invalid --chunk-size\n");
                std::exit(1);
            }
        } else if (arg == "--no-k-block") {
            cfg.disable_k_block = true;
        } else if (arg == "--fp16") {
            cfg.use_fp16 = true;
        } else if (arg == "--int8") {
            cfg.use_int8 = true;
        } else if (arg == "--int4") {
            cfg.use_int4 = true;
        } else if (arg == "--mxfp4") {
            cfg.use_mxfp4 = true;
        } else if (arg == "--fp32") {
            cfg.use_fp32 = true;
        } else if (arg == "--group-size") {
            if (!require_value(argc, argv, i, "--group-size", value)) std::exit(1);
            if (!parse_int(value, cfg.group_size) || cfg.group_size < 0) {
                std::fprintf(stderr, "bench_matmul: invalid --group-size\n");
                std::exit(1);
            }
        } else if (arg == "--sparse-a") {
            cfg.sparse_a = true;
        } else if (arg == "--dsv4-grouped") {
            cfg.dsv4_grouped = true;
        } else if (arg == "--density") {
            if (!require_value(argc, argv, i, "--density", value)) std::exit(1);
            if (!parse_int(value, cfg.density_pct) ||
                cfg.density_pct < 0 || cfg.density_pct > 100) {
                std::fprintf(stderr, "bench_matmul: invalid --density\n");
                std::exit(1);
            }
        } else if (arg == "--interleave-pack") {
            cfg.interleave_pack = true;
        } else if (arg == "--no-interleave-pack") {
            cfg.no_interleave_pack = true;
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

static BenchResult run_dsv4_grouped_bench(const BenchConfig& cfg) {
    constexpr int groups = 8;
    constexpr int group_width = 4096;
    constexpr int rank = 1024;
    constexpr int input_size = groups * group_width;
    constexpr int output_size = groups * rank;
    constexpr int weight_sets = 4;
    constexpr size_t weight_set_size =
        static_cast<size_t>(output_size) * group_width;

    std::vector<float> input(input_size);
    std::vector<__fp16> packed_weight(
        weight_sets * weight_set_size);
    std::vector<float> output(output_size);
    for (int i = 0; i < input_size; ++i)
        input[i] = static_cast<float>((i % 31) - 15) / 32.0f;
    for (size_t i = 0; i < packed_weight.size(); ++i)
        packed_weight[i] =
            static_cast<__fp16>(
                static_cast<float>(
                    static_cast<int>(i % 17) - 8) / 64.0f);

    uint8_t unused_raw_weight = 0;
    Tensor A = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        input_size, 1, 1, 1, input.data());
    Tensor B = Tensor::create(
        Precision::FP8_E4M3, MemoryType::EXTERNAL,
        output_size, group_width, 1, 1, &unused_raw_weight);
    B.fp8_bf16_fp16_data = packed_weight.data();
    Tensor C = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        output_size, 1, 1, 1, output.data());
    ThreadPool pool(cfg.num_threads);
    ThreadPool* thread_pool =
        cfg.num_threads > 1 ? &pool : nullptr;

    for (int i = 0; i < cfg.warmup; ++i) {
        B.fp8_bf16_fp16_data =
            packed_weight.data() +
            static_cast<size_t>(i % weight_sets) * weight_set_size;
        if (!kernel_dsv4_grouped_linear(
                A, B, C, groups, thread_pool)) {
            std::fprintf(
                stderr, "bench_matmul: DSV4 grouped-linear rejected input\n");
            std::exit(1);
        }
    }

    std::vector<double> times;
    times.reserve(cfg.repeat);
    for (int i = 0; i < cfg.repeat; ++i) {
        B.fp8_bf16_fp16_data =
            packed_weight.data() +
            static_cast<size_t>(i % weight_sets) * weight_set_size;
        const auto begin = std::chrono::steady_clock::now();
        kernel_dsv4_grouped_linear(A, B, C, groups, thread_pool);
        const auto end = std::chrono::steady_clock::now();
        times.push_back(
            std::chrono::duration<double, std::milli>(end - begin).count());
    }
    std::sort(times.begin(), times.end());

    BenchResult result;
    result.min_ms = times.front();
    result.max_ms = times.back();
    result.p50_ms = times[times.size() / 2];
    for (double time : times)
        result.avg_ms += time;
    result.avg_ms /= times.size();
    const double operations =
        2.0 * output_size * group_width;
    result.gflops =
        operations / (result.avg_ms * 1.0e6);
    // Keep the output observable and catch accidental all-zero kernels.
    double checksum = 0.0;
    for (float value : output)
        checksum += value;
    std::printf(
        "DSV4 grouped-linear groups=%d group_width=%d rank=%d "
        "threads=%d checksum=%.6f\n",
        groups, group_width, rank, cfg.num_threads, checksum);
    return result;
}

static BenchResult run_bench(const BenchConfig& cfg) {
    int M = cfg.M, K = cfg.K, N = cfg.N;

    // Apply config overrides
    if (cfg.disable_k_block) {
        g_matmul_config.k_block = 0;
    } else if (cfg.k_block > 0) {
        g_matmul_config.k_block = cfg.k_block;
    }
    if (cfg.chunk_size > 0) {
        g_matmul_config.gemv_chunk_size = cfg.chunk_size;
    }
    if (cfg.no_interleave_pack) {
        g_matmul_config.use_interleave_pack = false;
    } else {
        g_matmul_config.use_interleave_pack = cfg.interleave_pack;
    }

    bool is_fp16 = cfg.use_fp16;
    bool is_int8 = cfg.use_int8;
    bool is_int4 = cfg.use_int4;
    bool is_mxfp4 = cfg.use_mxfp4;
    const bool use_interleaved_dense =
        g_matmul_config.use_interleave_pack &&
        mollm::cpu::capabilities().fp16_interleaved_weights;

    float* a_data = new float[M * K];
    float* c_data = new float[M * N];
    fill_rand(a_data, M * K);
    if (cfg.sparse_a) {
        if (M != 1) {
            std::fprintf(stderr, "bench_matmul: --sparse-a requires M=1\n");
            std::exit(1);
        }
        for (int k = 0; k < K; ++k) {
            if ((k * 37 + 11) % 100 >= cfg.density_pct) a_data[k] = 0.f;
        }
    }

    void* b_raw = nullptr;
    float* b_fp32_data = nullptr;
    __fp16* b_fp16_data = nullptr;
    __fp16* b_packed_data = nullptr;  // owns packed buffer if interleave on
    int8_t* b_int8_data = nullptr;
    int8_t* b_int8_packed_data = nullptr;
    int8_t* b_int8_q8dot_data = nullptr;
    uint8_t* b_int4_data = nullptr;
    uint8_t* b_mxfp4_data = nullptr;
    uint8_t* b_mxfp4_scales = nullptr;
    uint8_t* b_int4_q4dot_data = nullptr;
    uint8_t* b_int4_q4g32_data = nullptr;
    uint8_t* b_int4_q4g128_data = nullptr;
    int8_t* b_int4_sparse_data = nullptr;
    float* scales_data = nullptr;
    int group_size = cfg.group_size > 0 ? cfg.group_size : K;
    int groups_per_row = (K + group_size - 1) / group_size;

    if (is_mxfp4) {
        if ((K % 32) != 0) {
            std::fprintf(
                stderr, "bench_matmul: MXFP4 requires K divisible by 32\n");
            std::exit(1);
        }
        b_mxfp4_data = new uint8_t[
            static_cast<size_t>(N) * K / 2];
        b_mxfp4_scales = new uint8_t[
            static_cast<size_t>(N) * K / 32];
        for (size_t i = 0;
             i < static_cast<size_t>(N) * K / 2; ++i) {
            const uint8_t low =
                static_cast<uint8_t>(std::rand() % 16);
            const uint8_t high =
                static_cast<uint8_t>(std::rand() % 16);
            b_mxfp4_data[i] =
                static_cast<uint8_t>(low | (high << 4));
        }
        std::fill(
            b_mxfp4_scales,
            b_mxfp4_scales + static_cast<size_t>(N) * K / 32,
            static_cast<uint8_t>(127));
        b_raw = b_mxfp4_data;
    } else if (is_int8) {
        b_int8_data = new int8_t[N * K];
        scales_data = new float[N * groups_per_row];
        for (int n = 0; n < N; n++) {
            for (int g = 0; g < groups_per_row; g++) {
                scales_data[n * groups_per_row + g] = 0.01f + 0.0001f * (float)((n + g) & 7);
            }
            for (int k = 0; k < K; k++) {
                b_int8_data[n * K + k] = (int8_t)((std::rand() % 255) - 127);
            }
        }
        if (use_interleaved_dense) {
            b_int8_packed_data = pack_b_interleaved_int8_full(b_int8_data, N, K, K);
            b_int8_q8dot_data = pack_b_q8dot_int8_full(b_int8_data, N, K, K);
            b_raw = b_int8_packed_data;
        } else {
            b_raw = b_int8_data;
        }
    } else if (is_int4) {
        int row_stride = (K + 1) / 2;
        b_int4_data = new uint8_t[(size_t)N * row_stride];
        std::memset(b_int4_data, 0, (size_t)N * row_stride);
        scales_data = new float[N * groups_per_row];
        for (int n = 0; n < N; n++) {
            for (int g = 0; g < groups_per_row; g++) {
                scales_data[n * groups_per_row + g] = 0.01f + 0.0001f * (float)((n + g) & 7);
            }
            for (int k = 0; k < K; k++) {
                int q = (std::rand() % 15) - 7;
                uint8_t nibble = (uint8_t)q & 0x0F;
                uint8_t* byte = b_int4_data + (size_t)n * row_stride + (k >> 1);
                if (k & 1) *byte |= (uint8_t)(nibble << 4);
                else *byte |= nibble;
            }
        }
        b_raw = b_int4_data;
        if (g_matmul_config.use_interleave_pack && (K % 32) == 0 && (group_size % 32) == 0) {
            b_int4_q4dot_data = pack_b_q4dot_int4_full(b_int4_data, N, K, K);
            if (group_size == 32) {
                b_int4_q4g32_data = pack_b_q4dot_g32_full(
                    b_int4_q4dot_data, scales_data, N, K, groups_per_row);
            } else if (group_size == 128 && (K % 128) == 0) {
                b_int4_q4g128_data = pack_b_q4dot_g128_full(
                    b_int4_q4dot_data, scales_data, N, K, groups_per_row);
                if (cfg.sparse_a) {
                    b_int4_sparse_data =
                        pack_b_sparse_int4_g128_full(b_int4_q4g128_data, N, K);
                }
            }
        }
    } else if (is_fp16) {
        b_fp16_data = new __fp16[N * K];
        float* tmp = new float[N * K];
        fill_rand(tmp, N * K);
        for (int i = 0; i < N * K; i++) b_fp16_data[i] = (__fp16)tmp[i];
        delete[] tmp;

        // Pre-pack B for interleaved path (mirrors engine load-time packing)
        if (use_interleaved_dense) {
            b_packed_data = pack_b_interleaved_full(b_fp16_data, N, K, K);
            b_raw = b_packed_data;
        } else {
            b_raw = b_fp16_data;
        }
    } else {
        b_fp32_data = new float[N * K];
        fill_rand(b_fp32_data, N * K);
        b_raw = b_fp32_data;
    }

    Tensor A = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, K, M, 1, 1, a_data);
    Tensor B = Tensor::create(is_mxfp4 ? Precision::MXFP4 :
                              is_int4 ? Precision::INT4 :
                              is_int8 ? Precision::INT8 :
                              is_fp16 ? Precision::FP16 : Precision::FP32,
                              MemoryType::EXTERNAL, N, K, 1, 1, b_raw);
    if (is_fp16)
        B.is_interleaved = use_interleaved_dense;
    if (is_mxfp4) {
        B.e8m0_scales = b_mxfp4_scales;
        B.group_size = 32;
        B.groups_per_row = static_cast<uint32_t>(K / 32);
        B.num_groups =
            static_cast<uint32_t>(
                static_cast<size_t>(N) * K / 32);
    } else if (is_int8 || is_int4) {
        B.scales = scales_data;
        B.group_size = (uint32_t)group_size;
        B.groups_per_row = (uint32_t)groups_per_row;
        B.num_groups = (uint32_t)(N * groups_per_row);
        if (is_int8) {
            B.is_interleaved = use_interleaved_dense;
            B.q8_repack_data = b_int8_q8dot_data;
            if (cfg.sparse_a) B.sparse_data = b_int8_packed_data;
        } else {
            B.q4_repack_data = b_int4_q4dot_data;
            B.q4_g32_data = b_int4_q4g32_data;
            B.q4_g128_data = b_int4_q4g128_data;
            B.is_q4_repacked = b_int4_q4dot_data != nullptr;
            B.is_q4_g32_packed = b_int4_q4g32_data != nullptr;
            B.is_q4_g128_packed = b_int4_q4g128_data != nullptr;
            B.sparse_data = b_int4_sparse_data;
        }
    }
    PackedWeightMap benchmark_packed_weights;
    PreparedWeightMap benchmark_prepared_weights;
    if (is_int4 && B.is_q4_g32_packed &&
        mollm::cpu::capabilities().x86_w4_q8_activations) {
        // Mirror engine load-time preparation so this benchmark measures the
        // production VNNI W4A8 path when it is explicitly selected.
        prepare_matmul_weight(
            B, "bench_int4_weight", B.q4_g32_data,
            benchmark_packed_weights, benchmark_prepared_weights,
            false, false);
    }
    Tensor C = Tensor::create(Precision::FP32, MemoryType::OWNED, N, M, 1, 1, c_data);

    ThreadPool pool(cfg.num_threads);
    ThreadPool* tp = (cfg.num_threads > 1) ? &pool : nullptr;

    // warmup
    for (int i = 0; i < cfg.warmup; i++) {
        if (cfg.sparse_a) kernel_gemv_sparse_a(A, B, C, tp);
        else kernel_matmul_fp32(A, B, C, tp);
    }

    // timed runs
    std::vector<double> times;
    times.reserve(cfg.repeat);
    for (int i = 0; i < cfg.repeat; i++) {
        auto start = std::chrono::steady_clock::now();
        if (cfg.sparse_a) kernel_gemv_sparse_a(A, B, C, tp);
        else kernel_matmul_fp32(A, B, C, tp);
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
    delete[] c_data;
    if (b_packed_data) delete[] b_packed_data;
    if (is_fp16) delete[] b_fp16_data;
    if (b_int8_packed_data) delete[] b_int8_packed_data;
    if (b_int8_q8dot_data) delete[] b_int8_q8dot_data;
    if (b_int4_q4dot_data) delete[] b_int4_q4dot_data;
    if (b_int4_q4g32_data) delete[] b_int4_q4g32_data;
    if (b_int4_q4g128_data) delete[] b_int4_q4g128_data;
    if (b_int4_sparse_data) delete[] b_int4_sparse_data;
    if (is_mxfp4) {
        delete[] b_mxfp4_data;
        delete[] b_mxfp4_scales;
    } else if (is_int8) {
        delete[] b_int8_data;
        delete[] scales_data;
    } else if (is_int4) {
        delete[] b_int4_data;
        delete[] scales_data;
    } else if (!is_fp16) {
        delete[] b_fp32_data;
    }
    return result;
}

int main(int argc, char** argv) {
    srand(42);
    BenchConfig cfg = parse_args(argc, argv);
    BenchResult result =
        cfg.dsv4_grouped ? run_dsv4_grouped_bench(cfg) : run_bench(cfg);

    if (!cfg.dsv4_grouped) {
        std::printf("M=%d K=%d N=%d threads=%d isa=%s prec=%s\n",
                    cfg.M, cfg.K, cfg.N, cfg.num_threads,
                    mollm::cpu::isa_name(),
                    cfg.use_mxfp4 ? "MXFP4" :
                    cfg.use_int4 ? "INT4" :
                    cfg.use_int8 ? "INT8" :
                    cfg.use_fp16 ? "FP16" : "FP32");
        if (cfg.use_int4) {
            std::printf(
                "  w4_activation=%s\n",
                mollm::cpu::capabilities().x86_w4_q8_activations
                    ? "q8" : "fp32");
        }
    }
    std::printf("  warmup=%d repeat=%d\n", cfg.warmup, cfg.repeat);
    if (cfg.sparse_a) std::printf("  sparse_a=1 density=%d%%\n", cfg.density_pct);
    std::printf("  avg=%.4fms min=%.4fms max=%.4fms p50=%.4fms\n",
                result.avg_ms, result.min_ms, result.max_ms, result.p50_ms);
    std::printf("  GFLOPS=%.1f\n", result.gflops);

    return 0;
}
