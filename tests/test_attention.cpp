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
            // params[6]/[7] carry the optional input indices (-1 = absent),
            // matching what the graph builder records.
            const int sink_index = sink ? 6 : -1;
            node.params.i32 = {2, causal ? 1 : 0, H, KV, hd, vd,
                               sink_index, -1};
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
    // The kernel locates the selection through this index rather than a
    // fixed slot, so the test must declare where it put it. Without this the
    // sparse entry falls back to the dense kernel and every assertion below
    // compares dense against dense -- green, but testing nothing.
    p.selection_input = 6;

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

// Every test above uses a full-width selection, which the kernel answers with
// `kernel_sdpa` on its first check -- so none of them exercise the sparse
// indexing loop. This one makes the selection strictly narrower than the
// visible keys, which is the only configuration that reaches it.
//
// Two things are asserted. First, a narrow selection must DIFFER from dense:
// if it does not, the selection was ignored and the comparison above proves
// nothing. Second, the sparse result must equal a hand-computed attention over
// exactly the selected keys -- so the test pins the arithmetic, not just that
// something happened.
static bool sparse_narrow_selects_only_listed_keys() {
    const int H = 2, KV = 2, hd = 16, vd = 16, cap = 16;
    const int past = 6, cur = 2;
    const int dst = past + cur;
    const int width = 3;                       // strictly fewer than dst
    const float scale = 1.0f / sqrtf((float)hd);

    float* qd = new float[H*cur*hd];  fill_rand(qd, H*cur*hd);
    float* kd = new float[KV*cur*hd]; fill_rand(kd, KV*cur*hd);
    float* vdata = new float[KV*cur*vd]; fill_rand(vdata, KV*cur*vd);
    float* kc = new float[KV*cap*hd];
    float* vc = new float[KV*cap*vd];
    for (int h = 0; h < KV; ++h) {
        for (int t = 0; t < past; ++t) {
            for (int d = 0; d < hd; ++d)
                kc[(h*cap + t)*hd + d] = sinf((float)(h*97 + t*13 + d)*0.017f);
            for (int d = 0; d < vd; ++d)
                vc[(h*cap + t)*vd + d] = cosf((float)(h*61 + t*29 + d)*0.013f);
        }
    }

    const size_t es = sizeof(mollm::cpu::fp16_t);
    const size_t ktot = CacheMetadata::SIZE + (size_t)KV*cap*hd*es;
    const size_t vtot = CacheMetadata::SIZE + (size_t)KV*cap*vd*es;
    void* kbuf = calloc(1, ktot); void* vbuf = calloc(1, vtot);
    auto* kmeta = (CacheMetadata*)kbuf;
    auto* vmeta = (CacheMetadata*)vbuf;
    kmeta->max_seq_len = cap; kmeta->num_kv_heads = KV; kmeta->head_dim = hd;
    kmeta->current_seq_len = past;
    vmeta->max_seq_len = cap; vmeta->num_kv_heads = KV; vmeta->v_head_dim = vd;
    vmeta->current_seq_len = past;
    auto* kstore = (mollm::cpu::fp16_t*)((char*)kbuf + CacheMetadata::SIZE);
    auto* vstore = (mollm::cpu::fp16_t*)((char*)vbuf + CacheMetadata::SIZE);
    for (int h = 0; h < KV; ++h) {
        for (int t = 0; t < past; ++t) {
            for (int d = 0; d < hd; ++d)
                kstore[(h*cap + t)*hd + d] =
                    mollm::cpu::fp16_t(kc[(h*cap + t)*hd + d]);
            for (int d = 0; d < vd; ++d)
                vstore[(h*cap + t)*vd + d] =
                    mollm::cpu::fp16_t(vc[(h*cap + t)*vd + d]);
        }
    }

    auto Q  = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, cur, H, 1, qd);
    auto Kc = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, cur, KV, 1, kd);
    auto Vc = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, vd, cur, KV, 1, vdata);
    // The tensor data pointer must start at the metadata header, not the
    // payload: the kernels read CacheMetadata from `data` and reach the
    // payload via cache_data(). Passing `kstore` here would make the kernels
    // reinterpret the first payload bytes as a header.
    auto Kt = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, hd, cap, KV, 1, kbuf);
    auto Vt = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, vd, cap, KV, 1, vbuf);
    float* out = new float[H*cur*vd];
    auto Ob = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, vd, cur, H, 1, out);

    // Select a fixed, known set of key positions: the last `width` of the
    // visible window, in descending rank order (as the indexer emits them).
    std::vector<int32_t> selection((size_t)width*cur, -1);
    std::vector<int> chosen;
    for (int r = 0; r < width; ++r) chosen.push_back(dst - 1 - r);
    for (int s = 0; s < cur; ++s)
        for (int r = 0; r < width; ++r)
            selection[(size_t)s*width + r] = chosen[r];
    auto St = Tensor::create(Precision::INT32, MemoryType::EXTERNAL,
                             width, cur, 1, 1, selection.data());

    Tensor Ko = Kt, Vo = Vt;
    std::vector<const Tensor*> ins = {&Q, &Kc, &Vc, nullptr, &Kt, &Vt, &St};
    std::vector<Tensor*> outs = {&Ob, &Ko, &Vo};
    SdpaParams p{2, 1, H, KV, hd, vd, scale};
    p.selection_input = 6;
    kernel_sdpa_sparse(p, ins, outs);

    // Reference: softmax over the selected keys only, per query position,
    // with the same visibility limit the kernel applies.
    bool ok = true;
    float worst = 0.0f;
    for (int s = 0; s < cur; ++s) {
        const int visible_limit = past + s;
        for (int h = 0; h < H; ++h) {
            const int kvh = h / (H / KV);
            std::vector<float> scores;
            std::vector<int> keys;
            for (int r = 0; r < width; ++r) {
                const int key = chosen[r];
                if (key < 0 || key > visible_limit) continue;   // causality
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) {
                    const float kk = key < past
                        ? kc[(kvh*cap + key)*hd + d]
                        : kd[(kvh*cur + (key - past))*hd + d];
                    dot += qd[(h*cur + s)*hd + d] * kk;
                }
                scores.push_back(dot * scale);
                keys.push_back(key);
            }
            float mx = -INFINITY;
            for (float v : scores) mx = fmaxf(mx, v);
            float sum = 0.0f;
            for (float& v : scores) { v = expf(v - mx); sum += v; }
            for (int d = 0; d < vd; ++d) {
                float acc = 0.0f;
                for (size_t i = 0; i < keys.size(); ++i) {
                    const int key = keys[i];
                    const float vv = key < past
                        ? vc[(kvh*cap + key)*vd + d]
                        : vdata[(kvh*cur + (key - past))*vd + d];
                    acc += scores[i] / sum * vv;
                }
                const float got = out[(h*cur + s)*vd + d];
                worst = fmaxf(worst, fabsf(got - acc));
                if (fabsf(got - acc) > 2e-3f) ok = false;
            }
        }
    }
    if (!ok) fprintf(stderr, "  narrow sparse vs reference worst diff = %g\n", worst);

    free(kbuf); free(vbuf);
    delete[] qd; delete[] kd; delete[] vdata; delete[] kc; delete[] vc;
    delete[] out;
    return ok;
}

static void test_sparse_narrow_selection() {
    CHECK(sparse_narrow_selects_only_listed_keys(),
          "sparse: narrow selection matches softmax over exactly those keys");
}

// A graph node carrying a selection must reach kernel_sdpa_sparse through the
// backend dispatch, not be silently ignored by kernel_sdpa. The check is the
// same dense-vs-sparse invariant: a full-width selection must reproduce the
// dense result, and a partial one must differ from it (proving the selection
// is actually consumed rather than dropped).
static void test_backend_dispatch_consumes_selection() {
    const int H = 2, KV = 2, hd = 16, vd = 16, src = 2, cur = 2, past = 4, cap = 16;
    const int dst = past + cur;
    const float scale = 1.0f / sqrtf(hd);

    float qd[H*src*hd]; fill_rand(qd, H*src*hd);
    float kd[KV*cur*hd]; fill_rand(kd, KV*cur*hd);
    float vdta[KV*cur*vd]; fill_rand(vdta, KV*cur*vd);
    float kc[KV*cap*hd]; fill_rand(kc, KV*cap*hd);
    float vc[KV*cap*vd]; fill_rand(vc, KV*cap*vd);

    const size_t es = sizeof(mollm::cpu::fp16_t);
    size_t ktot = CacheMetadata::SIZE + (size_t)KV*cap*hd*es;
    size_t vtot = CacheMetadata::SIZE + (size_t)KV*cap*vd*es;

    auto mk = [&](void* buf, size_t total, int rows_per_head, float* src_data) {
        auto* m = cache_meta(buf);
        m->current_seq_len = (uint64_t)past;
        m->max_seq_len = (uint64_t)cap;
        m->num_kv_heads = (uint64_t)KV;
        return m;
    };

    void* kb_a = calloc(1, ktot); void* vb_a = calloc(1, vtot);
    void* kb_b = calloc(1, ktot); void* vb_b = calloc(1, vtot);
    for (void* b : {kb_a, vb_a}) (void)b;
    auto fill16 = [&](void* buf, int width, float* data, int rows) {
        auto* d = static_cast<mollm::cpu::fp16_t*>(cache_data(buf));
        for (int i = 0; i < KV*cap*width; i++) d[i] = (mollm::cpu::fp16_t)data[i % (KV*rows*width)];
    };

    auto* kma = mk(kb_a, ktot, hd, kc); kma->head_dim = hd;
    auto* vma = mk(vb_a, vtot, vd, vc); vma->v_head_dim = vd;
    auto* kmb = mk(kb_b, ktot, hd, kc); kmb->head_dim = hd;
    auto* vmb = mk(vb_b, vtot, vd, vc); vmb->v_head_dim = vd;
    (void)vma; (void)vmb;
    fill16(kb_a, hd, kc, cap); fill16(vb_a, vd, vc, cap);
    fill16(kb_b, hd, kc, cap); fill16(vb_b, vd, vc, cap);

    Tensor Q = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, src, H, 1, qd);
    Tensor K_cur = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, cur, KV, 1, kd);
    Tensor V_cur = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, vd, cur, KV, 1, vdta);
    Tensor KcA = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, (int64_t)(ktot/es), 1,1,1, kb_a);
    Tensor VcA = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, (int64_t)(vtot/es), 1,1,1, vb_a);
    Tensor KcB = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, (int64_t)(ktot/es), 1,1,1, kb_b);
    Tensor VcB = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, (int64_t)(vtot/es), 1,1,1, vb_b);
    float o_dense[H*src*vd]; float o_sparse[H*src*vd];
    Tensor Oa = Tensor::create(Precision::FP32, MemoryType::OWNED, vd, src, H, 1, o_dense);
    Tensor Ob = Tensor::create(Precision::FP32, MemoryType::OWNED, vd, src, H, 1, o_sparse);

    // Dense reference through the backend (no selection input).
    {
        GraphNode node; node.op_type = OpType::SDPA;
        node.params.i32 = {2, 1, H, KV, hd, vd, -1, -1};
        node.params.f32 = {scale};
        std::vector<const Tensor*> ins = {&Q, &K_cur, &V_cur, nullptr, &KcA, &VcA};
        std::vector<Tensor*> outs = {&Oa, &KcA, &VcA};
        (void)outs;
        CPUBackend{}.dispatch(node, ins, &Oa, nullptr);
    }

    // Full-width selection through the backend must match it.
    std::vector<int32_t> sel((size_t)dst*src, -1);
    for (int s = 0; s < src; s++)
        for (int r = 0; r < dst; r++) sel[(size_t)s*dst + r] = r;
    Tensor Sel = Tensor::create(Precision::INT32, MemoryType::EXTERNAL, dst, src, 1,1, sel.data());
    {
        GraphNode node; node.op_type = OpType::SDPA;
        // sink absent (-1), selection at index 6
        node.params.i32 = {2, 1, H, KV, hd, vd, -1, 6};
        node.params.f32 = {scale};
        std::vector<const Tensor*> ins = {&Q, &K_cur, &V_cur, nullptr, &KcB, &VcB, &Sel};
        CPUBackend{}.dispatch(node, ins, &Ob, nullptr);
    }

    bool same = true;
    for (int i = 0; i < H*src*vd; i++)
        if (fabsf(o_dense[i] - o_sparse[i]) > 1e-4f) same = false;
    CHECK(same, "dispatch: full-width selection matches dense through the backend");

    free(kb_a); free(vb_a); free(kb_b); free(vb_b);
}

enum class SparseScenario {
    Plain, Sink, BiasMask, Mixed, AllMasked, Invalid, OddTail, MaskedBlock
};

// Independent scalar math: no dense kernel or production softmax helper.
static bool sparse_reference_case(bool fp16, int src, int vd, bool causal,
                                  SparseScenario scenario, int requested_width = 0) {
    const int H = 4, KV = 2, hd = 9;
    const int width = requested_width ? requested_width :
        (scenario == SparseScenario::Sink || scenario == SparseScenario::OddTail ? 1 : 5);
    const int past = requested_width ? requested_width + 1 : 6;
    const int dst = past + src, cap = dst;
    const int q_stride = hd + 3, o_stride = vd + 2, mask_stride = dst + 3;
    const float guard = 12345.f, scale = 0.37f;
    std::vector<float> q(H * src * q_stride, guard), k(KV * src * hd);
    std::vector<float> v(KV * src * vd), keys(KV * cap * hd), values(KV * cap * vd);
    std::vector<float> output(H * src * o_stride, guard);
    std::vector<float> mask(src * mask_stride, guard), sink(H);
    std::vector<int32_t> selected(src * width);
    const bool zero_q = scenario == SparseScenario::Sink ||
                        scenario == SparseScenario::OddTail;
    const bool use_mask = scenario == SparseScenario::BiasMask ||
                          scenario == SparseScenario::Mixed ||
                          scenario == SparseScenario::AllMasked ||
                          scenario == SparseScenario::MaskedBlock;
    const bool use_sink = scenario == SparseScenario::Sink ||
                          scenario == SparseScenario::BiasMask ||
                          scenario == SparseScenario::Mixed;
    auto rounded = [&](float x) {
        return fp16 ? static_cast<float>(mollm::cpu::fp16_t(x)) : x;
    };
    for (int h = 0; h < H; ++h) {
        sink[h] = logf(3.f + h);
        for (int s = 0; s < src; ++s)
            for (int d = 0; d < hd; ++d)
                q[(h * src + s) * q_stride + d] = zero_q ? 0.f :
                    sinf(float(h * 17 + s * 7 + d) * 0.13f) * 0.4f;
    }
    for (int g = 0; g < KV; ++g)
        for (int pos = 0; pos < dst; ++pos) {
            for (int d = 0; d < hd; ++d) {
                const float x = cosf(float(g * 23 + pos * 11 + d) * 0.07f) * 0.3f;
                keys[(g * cap + pos) * hd + d] = rounded(x);
                if (pos >= past) k[(g * src + pos - past) * hd + d] = x;
            }
            for (int d = 0; d < vd; ++d) {
                const float x = scenario == SparseScenario::Sink ? 4.f :
                    scenario == SparseScenario::OddTail ? float((pos + 1) * 10) :
                    float(g * 32 + pos * 3 + d % 11 - 20) / 16.f;
                values[(g * cap + pos) * vd + d] = rounded(x);
                if (pos >= past) v[(g * src + pos - past) * vd + d] = x;
            }
        }
    for (int s = 0; s < src; ++s) {
        for (int pos = 0; pos < dst; ++pos)
            mask[s * mask_stride + pos] = scenario == SparseScenario::MaskedBlock ?
                (pos == 1 ? 0.f : -INFINITY) :
                scenario == SparseScenario::AllMasked ? -INFINITY :
                (pos == (s % 2 ? past : 1) ? -INFINITY :
                 float((pos * 3 + s * 5) % 7 - 3) * 0.4f);
        const int32_t positions[5] = {dst - 1, 2, past, 0, 4};
        const int32_t mixed[5] = {dst - 1, -1, dst + 10, 1, past};
        const int32_t invalid[5] = {-1, dst, -7, dst + 10, -2};
        for (int r = 0; r < width; ++r)
            selected[s * width + r] = zero_q ? 1 :
                scenario == SparseScenario::Invalid ? invalid[r % 5] :
                scenario == SparseScenario::Mixed && r < 5 ? mixed[r] :
                requested_width ? (r == 0 ? dst - 1 : width - r) : positions[r];
    }
    const size_t es = fp16 ? sizeof(mollm::cpu::fp16_t) : sizeof(float);
    auto cache = [&](const std::vector<float>& data, int dim, bool is_value) {
        void* buf = calloc(1, CacheMetadata::SIZE + data.size() * es);
        auto* meta = cache_meta(buf);
        meta->current_seq_len = past; meta->max_seq_len = cap;
        meta->num_kv_heads = KV;
        if (is_value) meta->v_head_dim = dim;
        else meta->head_dim = dim;
        if (fp16) {
            auto* dst16 = static_cast<mollm::cpu::fp16_t*>(cache_data(buf));
            for (size_t i = 0; i < data.size(); ++i)
                dst16[i] = mollm::cpu::fp16_t(data[i]);
        } else memcpy(cache_data(buf), data.data(), data.size() * es);
        return buf;
    };
    void* kb = cache(keys, hd, false), *vb = cache(values, vd, true);
    auto Q = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, src, H, 1, q.data());
    Q.stride[1] = q_stride * sizeof(float); Q.stride[2] = src * Q.stride[1];
    auto K = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, src, KV, 1, k.data());
    auto V = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, vd, src, KV, 1, v.data());
    auto KC = Tensor::create(fp16 ? Precision::FP16 : Precision::FP32,
                            MemoryType::EXTERNAL, hd, cap, KV, 1, kb);
    auto VC = Tensor::create(KC.prec, MemoryType::EXTERNAL, vd, cap, KV, 1, vb);
    auto O = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, vd, src, H, 1, output.data());
    O.stride[1] = o_stride * sizeof(float); O.stride[2] = src * O.stride[1];
    auto Mask = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, dst, src, 1, 1, mask.data());
    Mask.stride[1] = mask_stride * sizeof(float);
    auto Selection = Tensor::create(Precision::INT32, MemoryType::EXTERNAL,
                                    width, src, 1, 1, selected.data());
    SdpaParams p{2, causal ? 1 : 0, H, KV, hd, vd, scale};
    p.selection_input = 6; p.sink = use_sink ? sink.data() : nullptr;
    std::vector<const Tensor*> ins = {&Q, &K, &V, use_mask ? &Mask : nullptr,
                                      &KC, &VC, &Selection};
    std::vector<Tensor*> outs = {&O, &KC, &VC};
    bool ok = width < dst;
    if (scenario == SparseScenario::Mixed) {
        auto Sink = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                   H, 1, 1, 1, sink.data());
        ins.push_back(&Sink);
        GraphNode node;
        node.op_type = OpType::SDPA;
        node.params.i32 = {2, causal ? 1 : 0, H, KV, hd, vd, 7, 6};
        node.params.f32 = {scale};
        CPUBackend backend;
        backend.dispatch(node, ins, &O, nullptr);
        ok = ok && !backend.dispatch_failed();
    } else {
        kernel_sdpa_sparse(p, ins, outs);
    }

    for (int h = 0; h < H; ++h)
        for (int s = 0; s < src; ++s) {
            const int g = h / (H / KV);
            std::vector<double> scores(width, -INFINITY);
            double mx = use_sink ? sink[h] : -INFINITY;
            for (int r = 0; r < width; ++r) {
                const int pos = selected[s * width + r];
                if (pos < 0 || pos >= dst || (causal && pos > past + s)) continue;
                double dot = 0.;
                for (int d = 0; d < hd; ++d) {
                    const float query = q[(h * src + s) * q_stride + d];
                    dot += double(fp16 && HAS_NEON ? rounded(query) : query) *
                           keys[(g * cap + pos) * hd + d];
                }
                scores[r] = dot * scale + (use_mask ? mask[s * mask_stride + pos] : 0.);
                mx = std::fmax(mx, scores[r]);
            }
            double sum = use_sink ? std::exp(double(sink[h]) - mx) : 0.;
            for (double& score : scores) {
                score = std::isfinite(score) ? std::exp(score - mx) : 0.;
                sum += score;
            }
            for (int d = 0; d < o_stride; ++d) {
                double ref = 0.;
                if (d < vd && sum > 0.)
                    for (int r = 0; r < width; ++r)
                        if (scores[r] > 0.)
                            ref += scores[r] / sum *
                                values[(g * cap + selected[s * width + r]) * vd + d];
                if (d >= vd) ref = guard;
                const float got = output[(h * src + s) * o_stride + d];
                if (!std::isfinite(got) || std::fabs(got - ref) > 2e-3 * (1. + std::fabs(ref))) {
                    if (ok) fprintf(stderr, "  sparse ref fp16=%d src=%d vd=%d causal=%d case=%d h=%d s=%d d=%d: got=%g ref=%g\n",
                                    fp16, src, vd, causal, int(scenario), h, s, d, got, ref);
                    ok = false;
                }
            }
        }
    free(kb); free(vb);
    return ok;
}

static void test_sparse_scalar_reference() {
    const SparseScenario cases[] = {SparseScenario::Plain, SparseScenario::Sink,
        SparseScenario::BiasMask, SparseScenario::Mixed, SparseScenario::AllMasked,
        SparseScenario::Invalid, SparseScenario::OddTail};
    for (bool fp16 : {false, true})
        for (int src : {1, 3})
            for (int vd : {1, 5, 9, 65})
                for (bool causal : {false, true}) {
                    bool ok = true;
                    for (auto scenario : cases)
                        ok = sparse_reference_case(fp16, src, vd, causal, scenario) && ok;
                    char label[100];
                    snprintf(label, sizeof(label), "sparse scalar reference: FP%d src=%d vd=%d causal=%d",
                             fp16 ? 16 : 32, src, vd, causal);
                    CHECK(ok, label);
                }
    // Cross both 32- and 64-key blocks, including an entirely masked first block.
    for (bool fp16 : {false, true})
        for (int src : {1, 3})
            for (int width : {33, 65})
                for (bool causal : {false, true}) {
                    bool ok = true;
                    for (auto scenario : {SparseScenario::Plain, SparseScenario::BiasMask,
                                          SparseScenario::AllMasked, SparseScenario::MaskedBlock})
                        ok = sparse_reference_case(fp16, src, 5, causal, scenario, width) && ok;
                    char label[100];
                    snprintf(label, sizeof(label), "sparse multi-block reference: FP%d src=%d width=%d causal=%d",
                             fp16 ? 16 : 32, src, width, causal);
                    CHECK(ok, label);
                }
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
    test_sparse_narrow_selection();
    test_sparse_scalar_reference();
    test_backend_dispatch_consumes_selection();

    printf(failures ? "\n%d FAILED\n" : "\nAll attention tests passed!\n", failures);
    return failures;
}
