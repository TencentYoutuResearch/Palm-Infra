#include "kernels/attention.h"
#include "kernels/threading.h"
#include "engine/engine.h"  // for CacheMetadata, cache_meta, cache_data

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// bench_sdpa — micro-bench for kernel_sdpa prefill/decode paths.
//
// Mirrors bench_matmul structure. Default shapes match Youtu-LLM-2B MLA:
//   H=16 KV=16 hd=192 vd=128 src=256 cap=512 causal=true
// FP16 KV cache path (kv_cache=2) — same as production prefill.
// ---------------------------------------------------------------------------

static void fill_rand(float* d, int n) {
    for (int i = 0; i < n; i++) d[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

struct BenchConfig {
    int H = 16;
    int KV = 16;
    int hd = 192;
    int vd = 128;
    int src = 256;     // prefill seq_len
    int cur = 256;     // == src for prefill (no past)
    int past = 0;
    int cap = 512;
    bool causal = true;
    bool fp16_cache = true;
    int num_threads = 4;
    int warmup = 3;
    int repeat = 5;
};

struct BenchResult {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double p50_ms = 0.0;
};

static bool parse_int(const char* text, int& out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    long v = std::strtol(text, &end, 10);
    if (!end || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

static bool require_value(int argc, char** argv, int& i, const char* flag,
                          const char*& value) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "bench_sdpa: missing value for %s\n", flag);
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
            if (!parse_int(value, cfg.num_threads) || cfg.num_threads < 1) std::exit(1);
        } else if (arg == "--warmup") {
            if (!require_value(argc, argv, i, "--warmup", value)) std::exit(1);
            if (!parse_int(value, cfg.warmup) || cfg.warmup < 0) std::exit(1);
        } else if (arg == "--repeat") {
            if (!require_value(argc, argv, i, "--repeat", value)) std::exit(1);
            if (!parse_int(value, cfg.repeat) || cfg.repeat < 1) std::exit(1);
        } else if (arg == "--src") {
            if (!require_value(argc, argv, i, "--src", value)) std::exit(1);
            if (!parse_int(value, cfg.src) || cfg.src < 1) std::exit(1);
            cfg.cur = cfg.src;  // prefill: cur == src
        } else if (arg == "--heads") {
            if (!require_value(argc, argv, i, "--heads", value)) std::exit(1);
            if (!parse_int(value, cfg.H) || cfg.H < 1) std::exit(1);
            cfg.KV = cfg.H;  // MLA: KV == H
        } else if (arg == "--hd") {
            if (!require_value(argc, argv, i, "--hd", value)) std::exit(1);
            if (!parse_int(value, cfg.hd) || cfg.hd < 1) std::exit(1);
        } else if (arg == "--vd") {
            if (!require_value(argc, argv, i, "--vd", value)) std::exit(1);
            if (!parse_int(value, cfg.vd) || cfg.vd < 1) std::exit(1);
        } else if (arg == "--fp32-cache") {
            cfg.fp16_cache = false;
        } else if (arg == "--no-causal") {
            cfg.causal = false;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: bench_sdpa [options]\n");
            std::printf("  --threads N     (default 4)\n");
            std::printf("  --warmup N      (default 3)\n");
            std::printf("  --repeat N      (default 5)\n");
            std::printf("  --src N         prefill seq_len (default 256)\n");
            std::printf("  --heads N       num_heads (default 16)\n");
            std::printf("  --hd N          head_dim (default 192)\n");
            std::printf("  --vd N          v_head_dim (default 128)\n");
            std::printf("  --fp32-cache    use FP32 KV cache (default FP16)\n");
            std::printf("  --no-causal     disable causal mask\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "bench_sdpa: unknown arg: %s\n", argv[i]);
            std::exit(1);
        }
    }
    return cfg;
}

static BenchResult run_bench(const BenchConfig& cfg) {
    const int H = cfg.H, KV = cfg.KV, hd = cfg.hd, vd = cfg.vd;
    const int src = cfg.src, cur = cfg.cur, past = cfg.past, cap = cfg.cap;
    const int dst = past + cur;
    float scale = 1.f / std::sqrt((float)hd);

    // Q/K_cur/V_cur are always FP32 (matmul output)
    float* qd = new float[H * src * hd];     fill_rand(qd, H * src * hd);
    float* kd = new float[KV * cur * hd];    fill_rand(kd, KV * cur * hd);
    float* vdata = new float[KV * cur * vd]; fill_rand(vdata, KV * cur * vd);
    float* od = new float[H * src * vd];

    Tensor Q = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, src, H, 1, qd);
    Tensor K_cur = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, cur, KV, 1, kd);
    Tensor V_cur = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, vd, cur, KV, 1, vdata);

    // KV cache buffers: CacheMetadata header + FP16 or FP32 data
    Precision cache_prec = cfg.fp16_cache ? Precision::FP16 : Precision::FP32;
    size_t cache_es = (cache_prec == Precision::FP16) ? 2 : 4;
    size_t k_cache_total = CacheMetadata::SIZE + (size_t)KV * cap * hd * cache_es;
    size_t v_cache_total = CacheMetadata::SIZE + (size_t)KV * cap * vd * cache_es;

    void* kc_buf = calloc(1, k_cache_total);
    void* vc_buf = calloc(1, v_cache_total);

    auto* k_meta = cache_meta(kc_buf);
    k_meta->current_seq_len = (uint64_t)past;
    k_meta->max_seq_len     = (uint64_t)cap;
    k_meta->num_kv_heads    = (uint64_t)KV;
    k_meta->head_dim        = (uint64_t)hd;
    auto* v_meta = cache_meta(vc_buf);
    v_meta->current_seq_len = (uint64_t)past;
    v_meta->max_seq_len     = (uint64_t)cap;
    v_meta->num_kv_heads    = (uint64_t)KV;
    v_meta->v_head_dim      = (uint64_t)vd;

    // Pre-fill cache past portion with random data (matches production path).
    if (cache_prec == Precision::FP16) {
        __fp16* kdst = (__fp16*)cache_data(kc_buf);
        __fp16* vdst = (__fp16*)cache_data(vc_buf);
        for (int g = 0; g < KV; g++) {
            for (int s = 0; s < past; s++) {
                for (int d = 0; d < hd; d++) kdst[g*cap*hd + s*hd + d] = (__fp16)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
                for (int d = 0; d < vd; d++) vdst[g*cap*vd + s*vd + d] = (__fp16)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
            }
        }
    } else {
        float* kdst = (float*)cache_data(kc_buf);
        float* vdst = (float*)cache_data(vc_buf);
        for (int g = 0; g < KV; g++) {
            for (int s = 0; s < past; s++) {
                for (int d = 0; d < hd; d++) kdst[g*cap*hd + s*hd + d] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
                for (int d = 0; d < vd; d++) vdst[g*cap*vd + s*vd + d] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
            }
        }
    }

    // Cache tensors: shape[0] = total_bytes/4 (matches test_attention convention;
    // kernel_sdpa reads K_cache->data + element_size for strides — only data
    // pointer and prec field matter for the cache lookup path).
    Tensor K_cache = Tensor::create(cache_prec, MemoryType::EXTERNAL,
                                     (int64_t)k_cache_total / 4, 1, 1, 1, kc_buf);
    Tensor V_cache = Tensor::create(cache_prec, MemoryType::EXTERNAL,
                                     (int64_t)v_cache_total / 4, 1, 1, 1, vc_buf);
    Tensor out = Tensor::create(Precision::FP32, MemoryType::OWNED, vd, src, H, 1, od);
    Tensor K_out = Tensor::create(cache_prec, MemoryType::EXTERNAL,
                                    (int64_t)k_cache_total / 4, 1, 1, 1, kc_buf);
    Tensor V_out = Tensor::create(cache_prec, MemoryType::EXTERNAL,
                                    (int64_t)v_cache_total / 4, 1, 1, 1, vc_buf);

    // Causal mask (matches production: mask[i*dst + j] = j <= past+i ? 0 : -inf)
    float* mask = new float[src * dst];
    for (int i = 0; i < src; i++) {
        for (int j = 0; j < dst; j++) {
            mask[i * dst + j] = (j <= past + i) ? 0.f : -INFINITY;
        }
    }
    Tensor mask_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    dst, src, 1, 1, mask);

    OpParams p;
    p.i32 = {2 /*kv_cache=2*/, cfg.causal ? 1 : 0, H, KV, hd, vd};
    p.f32 = {scale};

    ThreadPool pool(cfg.num_threads);

    // Reset current_seq_len to `past` before each iter (kernel_sdpa appends
    // cur tokens on top of past, mutating the metadata — reset to keep
    // behavior identical across runs).
    auto reset_meta = [&]() {
        k_meta->current_seq_len = (uint64_t)past;
        v_meta->current_seq_len = (uint64_t)past;
    };

    std::vector<const Tensor*> ins = {&Q, &K_cur, &V_cur, &mask_t, &K_cache, &V_cache};
    std::vector<Tensor*> outs = {&out, &K_out, &V_out};

    // warmup
    for (int i = 0; i < cfg.warmup; i++) {
        reset_meta();
        kernel_sdpa(p, ins, outs, &pool);
    }

    // timed
    std::vector<double> times;
    times.reserve(cfg.repeat);
    for (int i = 0; i < cfg.repeat; i++) {
        reset_meta();
        auto t0 = std::chrono::steady_clock::now();
        kernel_sdpa(p, ins, outs, &pool);
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());

    BenchResult r;
    r.min_ms = times.front();
    r.max_ms = times.back();
    r.p50_ms = times[times.size() / 2];
    double sum = 0; for (double t : times) sum += t;
    r.avg_ms = sum / times.size();

    delete[] qd; delete[] kd; delete[] vdata; delete[] od; delete[] mask;
    free(kc_buf); free(vc_buf);
    return r;
}

int main(int argc, char** argv) {
    srand(42);
    BenchConfig cfg = parse_args(argc, argv);
    BenchResult r = run_bench(cfg);

    const char* cache_str = cfg.fp16_cache ? "FP16" : "FP32";
    std::printf("SDPA: H=%d KV=%d hd=%d vd=%d src=%d past=%d cache=%s causal=%d threads=%d\n",
                cfg.H, cfg.KV, cfg.hd, cfg.vd, cfg.src, cfg.past,
                cache_str, cfg.causal ? 1 : 0, cfg.num_threads);
    std::printf("  warmup=%d repeat=%d\n", cfg.warmup, cfg.repeat);
    std::printf("  avg=%.3fms min=%.3fms max=%.3fms p50=%.3fms\n",
                r.avg_ms, r.min_ms, r.max_ms, r.p50_ms);
    return 0;
}
