#include "kernels/attention.h"
#include "engine/engine.h"  // for CacheMetadata
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

static void fill_rand(float* d, int n) { for(int i=0;i<n;i++) d[i]=(float)rand()/RAND_MAX-0.5f; }

// Build reference: concat K/V, compute SDPA naively
static void ref_sdpa(const float* Q, const float* K_cache, const float* V_cache,
                     const float* K_cur, const float* V_cur, float* out,
                     int H, int KV, int hd, int vd, int src, int cur, int past, int cap,
                     float scale, bool causal) {
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
            float mx = -INFINITY;
            for (int j = 0; j < dst; j++) if (qk[j] > mx) mx = qk[j];
            float sm = 0;
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
static bool test_case(int H, int KV, int hd, int vd, int src, int cur, int past, int cap,
                      bool causal) {
    float scale = 1.f / sqrtf(hd);

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
    size_t k_cache_total = CacheMetadata::SIZE + KV * cap * hd * sizeof(float);
    size_t v_cache_total = CacheMetadata::SIZE + KV * cap * vd * sizeof(float);
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
    memcpy(cache_data(kc_buf), kc, KV * cap * hd * sizeof(float));
    memcpy(cache_data(vc_buf), vc, KV * cap * vd * sizeof(float));

    Tensor K_cache = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                     (int64_t)k_cache_total / 4, 1, 1, 1, kc_buf);
    Tensor V_cache = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                     (int64_t)v_cache_total / 4, 1, 1, 1, vc_buf);
    Tensor out = Tensor::create(Precision::FP32, MemoryType::OWNED, vd, src, H, 1, od);
    Tensor K_out = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                   (int64_t)k_cache_total / 4, 1, 1, 1, kc_buf);
    Tensor V_out = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                   (int64_t)v_cache_total / 4, 1, 1, 1, vc_buf);

    OpParams p; p.i32={2, causal?1:0, H, KV, hd, vd}; p.f32={scale};
    std::vector<const Tensor*> ins = {&Q, &K_cur, &V_cur, nullptr, &K_cache, &V_cache};
    std::vector<Tensor*> outs = {&out, &K_out, &V_out};
    kernel_sdpa(p, ins, outs);

    ref_sdpa(qd, (float*)cache_data(kc_buf), (float*)cache_data(vc_buf),
             kd, vdata, ref, H, KV, hd, vd, src, cur, past, cap, scale, causal);

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
        float* kc_data = (float*)cache_data(kc_buf);
        for (int g = 0; g < KV; g++)
            for (int j = 0; j < cur; j++)
                for (int d = 0; d < hd; d++)
                    if (fabsf(kc_data[g*cap*hd + (past+j)*hd + d] - kd[g*cur*hd + j*hd + d]) > 1e-5f) ok = false;
    }

    free(kc_buf); free(vc_buf);
    delete[] od; delete[] ref;
    return ok;
}

int main() {
    srand(42);

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

    printf(failures ? "\n%d FAILED\n" : "\nAll attention tests passed!\n", failures);
    return failures;
}
