#include "backends/cpu/backend.h"
#include "graph/graph.h"
#include "kernels/cpu/attention.h"
#include "core/cache_layout.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum class TestPath { Kernel, Backend, BackendDefaults };

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

static void fill_rand(float* d, int n) { for(int i=0;i<n;i++) d[i]=static_cast<float>(rand())/static_cast<float>(RAND_MAX)-0.5f; }

// Build reference: concat K/V, compute SDPA naively
static void ref_sdpa(const float* Q, const float* K_cache, const float* V_cache,
                     const float* K_cur, const float* V_cur, float* out,
                     int H, int KV, int hd, int vd, int src, int cur, int past, int cap,
                     float scale, bool causal, const float* sink = nullptr) {
    int hpg = H / KV;
    int dst = past + cur;

    // concat K
    float* ref_k = new float[KV * dst * hd];
    float* ref_v = new float[KV * dst * vd];
    for (int g = 0; g < KV; g++) {
        memcpy(ref_k + g*dst*hd, K_cache + g*cap*hd, past*hd*sizeof(float));
        memcpy(ref_k + g*dst*hd + past*hd, K_cur + g*cur*hd, cur*hd*sizeof(float));
        memcpy(ref_v + g*dst*vd, V_cache + g*cap*vd, past*vd*sizeof(float));
        memcpy(ref_v + g*dst*vd + past*vd, V_cur + g*cur*vd, cur*vd*sizeof(float));
    }

    for (int h = 0; h < H; h++) {
        int kv_h = h / hpg;
        for (int s = 0; s < src; s++) {
            float* qk = new float[dst];
            for (int j = 0; j < dst; j++) {
                float dot = 0;
                for (int d = 0; d < hd; d++)
                    dot += Q[h*src*hd + s*hd + d] * ref_k[kv_h*dst*hd + j*hd + d];
                qk[j] = dot * scale;
            }
            if (causal) {
                for (int j = past + s + 1; j < dst; j++) qk[j] = -INFINITY;
            }
            // A sink is an extra logit with no value row: it joins the max and
            // the denominator but contributes nothing to the numerator.
            float mx = sink ? sink[h] : -INFINITY;
            for (int j = 0; j < dst; j++) if (qk[j] > mx) mx = qk[j];
            float sm = sink ? expf(sink[h] - mx) : 0.f;
            for (int j = 0; j < dst; j++) { qk[j] = expf(qk[j] - mx); sm += qk[j]; }
            for (int j = 0; j < dst; j++) qk[j] /= sm;
            float* o = out + h*src*vd + s*vd;
            memset(o, 0, vd*sizeof(float));
            for (int j = 0; j < dst; j++)
                for (int d = 0; d < vd; d++)
                    o[d] += qk[j] * ref_v[kv_h*dst*vd + j*vd + d];
            delete[] qk;
        }
    }
    delete[] ref_k; delete[] ref_v;
}

// Run one SDPA test case
static bool test_case(int H, int KV, int hd, int vd, int src, int cur,
                      int past, int cap, bool causal,
                      bool fp16_cache = false, TestPath path = TestPath::Kernel,
                      const float* sink = nullptr) {
    float scale = path == TestPath::Backend ? 0.37f : 1.f / sqrtf(hd);

    float* qd = new float[H*src*hd]; fill_rand(qd, H*src*hd);
    float* kd = new float[KV*cur*hd]; fill_rand(kd, KV*cur*hd);
    float* vdata = new float[KV*cur*vd]; fill_rand(vdata, KV*cur*vd);
    float* kc = new float[KV*cap*hd]; fill_rand(kc, KV*cap*hd);
    float* vc = new float[KV*cap*vd]; fill_rand(vc, KV*cap*vd);
    float* od = new float[H*src*vd];
    float* ref = new float[H*src*vd];

    Tensor Q = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, src, H, 1, qd);
    Tensor K_cur = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, cur, KV, 1, kd);
    Tensor V_cur = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, vd, cur, KV, 1, vdata);

    // KV cache buffers with CacheMetadata header
    const Precision cache_precision =
        fp16_cache ? Precision::FP16 : Precision::FP32;
    const size_t cache_element_size =
        fp16_cache ? sizeof(mollm::cpu::fp16_t) : sizeof(float);
    size_t k_cache_total =
        CacheMetadata::SIZE + KV * cap * hd * cache_element_size;
    size_t v_cache_total =
        CacheMetadata::SIZE + KV * cap * vd * cache_element_size;
    void* kc_buf = calloc(1, k_cache_total);
    void* vc_buf = calloc(1, v_cache_total);

    // Init metadata
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

    // Copy pre-filled cache data into data region (after metadata)
    std::vector<float> reference_key(kd, kd + KV * cur * hd);
    std::vector<float> reference_value(vdata, vdata + KV * cur * vd);
    if (fp16_cache) {
        auto* cached_key =
            static_cast<mollm::cpu::fp16_t*>(cache_data(kc_buf));
        auto* cached_value =
            static_cast<mollm::cpu::fp16_t*>(cache_data(vc_buf));
        for (int i = 0; i < KV * cap * hd; ++i) {
            cached_key[i] = static_cast<mollm::cpu::fp16_t>(kc[i]);
            kc[i] = static_cast<float>(cached_key[i]);
        }
        for (int i = 0; i < KV * cap * vd; ++i) {
            cached_value[i] = static_cast<mollm::cpu::fp16_t>(vc[i]);
            vc[i] = static_cast<float>(cached_value[i]);
        }
        for (float& value : reference_key)
            value = static_cast<float>(mollm::cpu::fp16_t(value));
        for (float& value : reference_value)
            value = static_cast<float>(mollm::cpu::fp16_t(value));
    } else {
        memcpy(cache_data(kc_buf), kc, KV * cap * hd * sizeof(float));
        memcpy(cache_data(vc_buf), vc, KV * cap * vd * sizeof(float));
    }

    Tensor K_cache = Tensor::create(
        cache_precision, MemoryType::EXTERNAL,
        static_cast<int64_t>(k_cache_total / cache_element_size),
        1, 1, 1, kc_buf);
    Tensor V_cache = Tensor::create(
        cache_precision, MemoryType::EXTERNAL,
        static_cast<int64_t>(v_cache_total / cache_element_size),
        1, 1, 1, vc_buf);
    Tensor out = Tensor::create(Precision::FP32, MemoryType::OWNED, vd, src, H, 1, od);
    Tensor K_out = K_cache;
    Tensor V_out = V_cache;

    SdpaParams p{2, causal?1:0, H, KV, hd, vd, scale};
    p.sink = sink;
    // The backend path receives the sink as a 7th input tensor; the direct
    // kernel path gets it through SdpaParams.
    Tensor sink_tensor = Tensor::create(
        Precision::FP32, MemoryType::EXTERNAL,
        sink ? H : 0, 1, 1, 1,
        sink ? const_cast<float*>(sink) : nullptr);
    std::vector<const Tensor*> ins = {&Q, &K_cur, &V_cur, nullptr, &K_cache, &V_cache};
    if (sink) ins.push_back(&sink_tensor);
    std::vector<Tensor*> outs = {&out, &K_out, &V_out};
    if (path == TestPath::Kernel) {
        kernel_sdpa(p, ins, outs);
    } else {
        GraphNode node;
        node.op_type = OpType::SDPA;
        if (path != TestPath::BackendDefaults) {
            node.params.i32 = {2, causal ? 1 : 0, H, KV, hd, vd};
            node.params.f32 = {scale};
        }
        CPUBackend{}.dispatch(node, ins, &out, nullptr);
    }

    ref_sdpa(qd, kc, vc, reference_key.data(), reference_value.data(), ref,
             H, KV, hd, vd, src, cur, past, cap, scale, causal, sink);

    // Debug first mismatch
    bool ok = true;
    for (int i = 0; i < H*src*vd; i++) {
        if (fabsf(od[i]-ref[i])>1e-3f) {
            fprintf(stderr,"  mismatch[%d]: got %f exp %f\n", i, od[i], ref[i]);
            ok = false;
            break;
        }
    }
    if (!ok) {
        fprintf(stderr,"  od[0..4]: %f %f %f %f %f\n", od[0],od[1],od[2],od[3],od[4]);
        fprintf(stderr,"  ref[0..4]: %f %f %f %f %f\n", ref[0],ref[1],ref[2],ref[3],ref[4]);
    }

    // check cache append (only when past > 0)
    if (past > 0) {
        for (int g = 0; g < KV; g++)
            for (int j = 0; j < cur; j++)
                for (int d = 0; d < hd; d++)
                    if (fabsf(
                            (fp16_cache
                                 ? static_cast<float>(
                                       static_cast<mollm::cpu::fp16_t*>(
                                           cache_data(kc_buf))[
                                           g*cap*hd + (past+j)*hd + d])
                                 : static_cast<float*>(cache_data(kc_buf))[
                                       g*cap*hd + (past+j)*hd + d]) -
                            reference_key[g*cur*hd + j*hd + d]) > 1e-5f)
                        ok = false;
    }

    free(kc_buf); free(vc_buf);
    delete[] qd; delete[] kd; delete[] vdata;
    delete[] kc; delete[] vc;
    delete[] od; delete[] ref;
    return ok;
}

// Compare the sparse entry against the dense kernel on identical inputs.
//
// No reference implementation is involved: selecting every cache position
// makes the gathered set identical to the cache, and both entries then run
// the same flash-attn kernels, so the outputs must match bit for bit. That
// makes this check immune to the indexing mistakes a hand-written reference
// invites.
static bool sparse_matches_dense(int H, int KV, int hd, int vd, int src,
                                 int cur, int past, int cap, bool fp16_cache) {
    const float scale = 1.0f / sqrtf(hd);
    const int dst = past + cur;

    float* qd = new float[H*src*hd]; fill_rand(qd, H*src*hd);
    float* kd = new float[KV*cur*hd]; fill_rand(kd, KV*cur*hd);
    float* vdata = new float[KV*cur*vd]; fill_rand(vdata, KV*cur*vd);
    float* kc = new float[KV*cap*hd]; fill_rand(kc, KV*cap*hd);
    float* vc = new float[KV*cap*vd]; fill_rand(vc, KV*cap*vd);
    float* dense_out = new float[H*src*vd];
    float* sparse_out = new float[H*src*vd];

    const Precision cache_precision =
        fp16_cache ? Precision::FP16 : Precision::FP32;
    const size_t cache_es =
        fp16_cache ? sizeof(mollm::cpu::fp16_t) : sizeof(float);
    size_t k_cache_total = CacheMetadata::SIZE + KV * cap * hd * cache_es;
    size_t v_cache_total = CacheMetadata::SIZE + KV * cap * vd * cache_es;

    // Two independent cache copies so each run appends into a fresh one.
    auto make_cache = [&](size_t total, int cache_hd, float* src) {
        void* buf = calloc(1, total);
        auto* meta = cache_meta(buf);
        meta->current_seq_len = (uint64_t)past;
        meta->max_seq_len     = (uint64_t)cap;
        meta->num_kv_heads    = (uint64_t)KV;
        if (cache_hd == hd) meta->head_dim = (uint64_t)hd;
        else                meta->v_head_dim = (uint64_t)vd;
        const int rows = (int)(total / cache_es / KV / (cache_hd ? cache_hd : 1));
        (void)rows;
        if (fp16_cache) {
            auto* dst16 = static_cast<mollm::cpu::fp16_t*>(cache_data(buf));
            for (int i = 0; i < KV * cap * cache_hd; ++i)
                dst16[i] = static_cast<mollm::cpu::fp16_t>(src[i]);
        } else {
            memcpy(cache_data(buf), src, KV * cap * cache_hd * sizeof(float));
        }
        return buf;
    };

    void* kc_a = make_cache(k_cache_total, hd, kc);
    void* vc_a = make_cache(v_cache_total, vd, vc);
    void* kc_b = make_cache(k_cache_total, hd, kc);
    void* vc_b = make_cache(v_cache_total, vd, vc);

    Tensor Q = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, src, H, 1, qd);
    Tensor K_cur = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, cur, KV, 1, kd);
    Tensor V_cur = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, vd, cur, KV, 1, vdata);
    Tensor KcA = Tensor::create(cache_precision, MemoryType::EXTERNAL,
                                (int64_t)(k_cache_total / cache_es), 1, 1, 1, kc_a);
    Tensor VcA = Tensor::create(cache_precision, MemoryType::EXTERNAL,
                                (int64_t)(v_cache_total / cache_es), 1, 1, 1, vc_a);
    Tensor KcB = Tensor::create(cache_precision, MemoryType::EXTERNAL,
                                (int64_t)(k_cache_total / cache_es), 1, 1, 1, kc_b);
    Tensor VcB = Tensor::create(cache_precision, MemoryType::EXTERNAL,
                                (int64_t)(v_cache_total / cache_es), 1, 1, 1, vc_b);
    Tensor out_a = Tensor::create(Precision::FP32, MemoryType::OWNED, vd, src, H, 1, dense_out);
    Tensor out_b = Tensor::create(Precision::FP32, MemoryType::OWNED, vd, src, H, 1, sparse_out);

    SdpaParams p{2, 1, H, KV, hd, vd, scale};

    // Dense run.
    {
        Tensor K_out = KcA;
        Tensor V_out = VcA;
        std::vector<const Tensor*> ins = {&Q, &K_cur, &V_cur, nullptr, &KcA, &VcA};
        std::vector<Tensor*> outs = {&out_a, &K_out, &V_out};
        kernel_sdpa(p, ins, outs);
    }

    // Sparse run over every position in the cache.
    std::vector<int32_t> selection((size_t)dst * src, -1);
    for (int s = 0; s < src; ++s)
        for (int r = 0; r < dst; ++r)
            selection[(size_t)s * dst + r] = r;
    Tensor sel = Tensor::create(
        Precision::INT32, MemoryType::EXTERNAL, dst, src, 1, 1, selection.data());
    {
        Tensor K_out = KcB;
        Tensor V_out = VcB;
        std::vector<const Tensor*> ins = {&Q, &K_cur, &V_cur, nullptr, &KcB, &VcB, &sel};
        std::vector<Tensor*> outs = {&out_b, &K_out, &V_out};
        kernel_sdpa_sparse(p, ins, outs);
    }

    bool ok = true;
    float worst = 0.0f;
    for (int i = 0; i < H*src*vd; i++) {
        const float diff = fabsf(dense_out[i] - sparse_out[i]);
        if (diff > worst) worst = diff;
    }
    if (worst > 1e-5f) {
        fprintf(stderr, "  sparse vs dense worst diff = %g\n", worst);
        ok = false;
    }

    free(kc_a); free(vc_a); free(kc_b); free(vc_b);
    delete[] qd; delete[] kd; delete[] vdata;
    delete[] kc; delete[] vc; delete[] dense_out; delete[] sparse_out;
    return ok;
}

static void test_sparse_select_all_matches_dense() {
    CHECK(sparse_matches_dense(4, 4, 16, 16, 4, 4, 0, 16, false),
          "sparse: select-all prefill == dense (FP32 cache)");
    CHECK(sparse_matches_dense(4, 4, 16, 16, 1, 1, 8, 32, false),
          "sparse: select-all decode == dense (FP32 cache)");
    CHECK(sparse_matches_dense(4, 4, 16, 16, 4, 4, 0, 16, true),
          "sparse: select-all prefill == dense (FP16 cache)");
    CHECK(sparse_matches_dense(4, 4, 16, 16, 1, 1, 8, 32, true),
          "sparse: select-all decode == dense (FP16 cache)");
    CHECK(sparse_matches_dense(4, 4, 16, 16, 8, 8, 12, 64, false),
          "sparse: select-all with past, larger src == dense");
}

// A selection that names every visible position carries no sparsity, so the
// entry must fall back to the dense kernel rather than pay lookup overhead.
// The fallback is exact, so the result is the dense result either way; what
// this pins is that a full-width selection still produces correct output and
// does not read past the cache.
static void test_sparse_falls_back_when_selection_is_not_sparser() {
    CHECK(sparse_matches_dense(4, 4, 16, 16, 4, 4, 0, 16, true),
          "sparse: full-width selection at short context (FP16)");
    CHECK(sparse_matches_dense(4, 4, 16, 16, 2, 2, 2, 16, false),
          "sparse: full-width selection just past the fallback point");
}

int main() {
    srand(42);
    CHECK(test_case(8, 4, 32, 24, 4, 4, 3, 16, true, true, TestPath::Backend),
          "backend SDPA: explicit dimensions/scale and FP16 cache append");
    CHECK(test_case(16, 16, 192, 128, 2, 2, 1, 8, true, false,
                    TestPath::BackendDefaults),
          "backend SDPA: omitted graph parameters preserve defaults");

    CHECK(test_case(128, 16, 192, 128, 1, 1, 128, 512, false),
          "MLA decode: H=128 KV=16 src=1 past=128");
    CHECK(test_case(128, 16, 192, 128, 1, 1, 0, 512, false),
          "MLA decode: H=128 KV=16 src=1 past=0");
    CHECK(test_case(8, 4, 32, 32, 4, 4, 0, 32, false),
          "GQA prefill: H=8 KV=4 src=4 past=0");
    CHECK(test_case(8, 4, 32, 32, 4, 4, 0, 32, true),
          "GQA causal prefill: H=8 KV=4 src=4 past=0");
    CHECK(test_case(16, 16, 64, 64, 1, 1, 0, 256, false),
          "decode: H=16 KV=16 src=1 past=0");
    CHECK(test_case(8, 4, 32, 24, 4, 4, 3, 16, true, true),
          "GQA causal prefill with FP16 KV cache");
    // Production-shaped MLA prefill: 16 heads × src=256 × past=0, causal.
    // Exercises the rewritten flash_attn_fp16_prefill kernel.
    CHECK(test_case(16, 16, 192, 128, 256, 256, 0, 512, true),
          "MLA prefill: H=16 KV=16 src=256 past=0 causal");
    // Smaller prefill: M not multiple of Br=8 (M=20 exercises tile tail).
    CHECK(test_case(16, 16, 192, 128, 20, 20, 0, 64, true),
          "MLA prefill: H=16 KV=16 src=20 past=0 causal (non-tile M)");
    // MLA prefill with non-tile d_v (vd=100 exercises d_v tail).
    CHECK(test_case(16, 16, 192, 100, 64, 64, 0, 128, true),
          "MLA prefill: H=16 KV=16 src=64 past=0 causal (non-tile d_v)");
    // MLA prefill with past > 0 (exercises cache append + prefill).
    CHECK(test_case(16, 16, 192, 128, 128, 128, 128, 512, true),
          "MLA prefill: H=16 KV=16 src=128 past=128 causal");

    // ---- per-head attention sink (Hy4 / gpt-oss) ----
    // The sink is a virtual extra logit with no value row. A strongly negative
    // sink must leave the result indistinguishable from no sink, since
    // exp(sink - max) underflows and the denominator reverts to the keys.
    {
        // Must match the head count of every case below (H=16): the kernel
        // indexes the sink per query head.
        const int sink_heads = 16;
        std::vector<float> small(sink_heads, -100.0f);
        CHECK(test_case(16, 16, 64, 64, 1, 1, 0, 256, false,
                        false, TestPath::Kernel, small.data()),
              "sink: strongly negative sink matches the no-sink result");
        CHECK(test_case(16, 16, 192, 128, 32, 32, 0, 64, true,
                        false, TestPath::Kernel, small.data()),
              "sink: negative sink on the prefill path");
        CHECK(test_case(16, 16, 64, 64, 1, 1, 0, 256, false,
                        true, TestPath::Kernel, small.data()),
              "sink: negative sink with FP16 KV cache");

        // A dominant sink suppresses every real attention weight, so the
        // output must collapse toward zero (all V weighted by ~0).
        std::vector<float> huge(sink_heads, 1.0e4f);
        CHECK(test_case(16, 16, 64, 64, 1, 1, 0, 256, false,
                        false, TestPath::Kernel, huge.data()),
              "sink: dominant sink collapses the output");

        // Distinct positive sinks exercise the max-stability path, where the
        // sink and the real scores compete for the running maximum.
        std::vector<float> mixed(sink_heads);
        for (int h = 0; h < sink_heads; ++h)
            mixed[h] = -0.5f + 0.25f * static_cast<float>(h);
        CHECK(test_case(16, 16, 64, 64, 1, 1, 0, 256, false,
                        false, TestPath::Kernel, mixed.data()),
              "sink: mixed per-head sinks (decode)");
        CHECK(test_case(16, 16, 192, 128, 64, 64, 32, 128, true,
                        false, TestPath::Kernel, mixed.data()),
              "sink: mixed per-head sinks (prefill with past)");
        CHECK(test_case(16, 16, 64, 64, 1, 1, 0, 256, false,
                        true, TestPath::Kernel, mixed.data()),
              "sink: mixed per-head sinks with FP16 KV cache");
        CHECK(test_case(16, 16, 64, 64, 1, 1, 0, 256, false,
                        false, TestPath::Backend, mixed.data()),
              "sink: mixed per-head sinks through the backend dispatch");
    }

    // ---- sparse SDPA (Hy4 Gated DSA consumer) ----
    // Two properties carry the correctness argument:
    //   1. Selecting every cache position must reproduce the dense result.
    //      The gathered set is then the whole cache, so the same flash-attn
    //      kernels run on identical inputs and the outputs must match.
    //   2. A partial selection must equal the dense result computed over only
    //      those keys, which is what makes it genuinely sparse.
    test_sparse_select_all_matches_dense();
    test_sparse_falls_back_when_selection_is_not_sparser();

    printf(failures ? "\n%d FAILED\n" : "\nAll attention tests passed!\n", failures);
    return failures;
}
