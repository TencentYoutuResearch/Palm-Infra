#include "kernels/attention.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

static void fill_rand(float* d, int n) { for(int i=0;i<n;i++) d[i]=(float)rand()/RAND_MAX-0.5f; }

static bool check_approx(const float* a, const float* b, int n, float tol) {
    for(int i=0;i<n;i++) if(fabsf(a[i]-b[i])>tol) {
        fprintf(stderr,"  mismatch[%d]: %f vs %f\n",i,a[i],b[i]); return false;
    }
    return true;
}

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
    Tensor K_cache; K_cache.prec=Precision::FP32; K_cache.mem_type=MemoryType::EXTERNAL;
    K_cache.shape[0]=hd; K_cache.shape[1]=cap; K_cache.shape[2]=KV; K_cache.shape[3]=1;
    K_cache.compute_strides(); K_cache.shape[1]=past; K_cache.data=kc;
    Tensor V_cache; V_cache.prec=Precision::FP32; V_cache.mem_type=MemoryType::EXTERNAL;
    V_cache.shape[0]=vd; V_cache.shape[1]=cap; V_cache.shape[2]=KV; V_cache.shape[3]=1;
    V_cache.compute_strides(); V_cache.shape[1]=past; V_cache.data=vc;
    Tensor out = Tensor::create(Precision::FP32, MemoryType::OWNED, vd, src, H, 1, od);
    Tensor K_out = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, hd, cap, KV, 1, kc);
    Tensor V_out = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, vd, cap, KV, 1, vc);

    OpParams p; p.i32={2, causal?1:0, H, KV, hd, vd}; p.f32={scale};
    std::vector<const Tensor*> ins = {&Q, &K_cur, &V_cur, nullptr, &K_cache, &V_cache};
    std::vector<Tensor*> outs = {&out, &K_out, &V_out};
    kernel_sdpa(p, ins, outs);

    ref_sdpa(qd, kc, vc, kd, vdata, ref, H, KV, hd, vd, src, cur, past, cap, scale, causal);

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
        // dump first few elements
        fprintf(stderr,"  od[0..4]: %f %f %f %f %f\n", od[0],od[1],od[2],od[3],od[4]);
        fprintf(stderr,"  ref[0..4]: %f %f %f %f %f\n", ref[0],ref[1],ref[2],ref[3],ref[4]);
    }

    // check cache append (only when past > 0 — kernel skips append when past==0)
    if (past > 0) {
        for (int g = 0; g < KV; g++)
            for (int j = 0; j < cur; j++)
                for (int d = 0; d < hd; d++)
                    if (fabsf(kc[g*cap*hd + (past+j)*hd + d] - kd[g*cur*hd + j*hd + d]) > 1e-5f) ok = false;
    }

    delete[] qd; delete[] kd; delete[] vdata; delete[] kc; delete[] vc;
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

    printf(failures ? "\n%d FAILED\n" : "\nAll attention tests passed!\n", failures);
    return failures;
}
