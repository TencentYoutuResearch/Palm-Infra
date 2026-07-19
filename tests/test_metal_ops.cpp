// Per-op parity tests for the Metal backend vs the CPU reference kernels.
// Only built when MOLLM_METAL is defined.

#include "kernels/tensor.h"
#include "kernels/matmul.h"
#include "engine/metal_backend.h"
#include "graph/graph.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

static void fill_rand(float* d, int n) {
    for (int i = 0; i < n; i++) d[i] = (float)rand() / RAND_MAX - 0.5f;
}

// max abs / mean rel error check tuned for FP16 weights + FP32 accumulate.
static bool close(const float* got, const float* ref, int n, float atol, float rtol) {
    double max_abs = 0, worst = 0;
    int worst_i = -1;
    for (int i = 0; i < n; i++) {
        double a = std::fabs(got[i] - ref[i]);
        double denom = std::fabs(ref[i]) + 1e-6;
        double rel = a / denom;
        if (a > max_abs) max_abs = a;
        if (rel > worst && a > atol) { worst = rel; worst_i = i; }
    }
    if (worst > rtol) {
        fprintf(stderr, "  worst rel %.4g at %d (got %f ref %f), max_abs %.4g\n",
                worst, worst_i, worst_i>=0?got[worst_i]:0, worst_i>=0?ref[worst_i]:0, max_abs);
        return false;
    }
    return true;
}

// Allocate a device-resident tensor via the backend and return its host ptr.
static Tensor make_dev(MetalBackend& mb, Precision prec, int d0, int d1) {
    Tensor t = Tensor::create(prec, MemoryType::EXTERNAL, d0, d1, 1, 1, nullptr);
    mb.alloc_persistent(t, t.nbytes());
    return t;
}

// Run a single MATMUL node on the Metal backend.
static void metal_matmul(MetalBackend& mb, Tensor& A, Tensor& B, Tensor& C) {
    GraphNode node;
    node.op_type = OpType::MATMUL;
    std::vector<const Tensor*> ins = { &A, &B };
    mb.begin_graph();
    mb.dispatch(node, ins, &C, nullptr);
    mb.end_graph();
}

// 3-D device tensor: shape[0]=d0 (inner), shape[1]=d1, shape[2]=d2.
static Tensor make_dev3(MetalBackend& mb, Precision prec, int d0, int d1, int d2) {
    Tensor t = Tensor::create(prec, MemoryType::EXTERNAL, d0, d1, d2, 1, nullptr);
    mb.alloc_persistent(t, t.nbytes());
    return t;
}

// Run a single op with the given inputs/params on the Metal backend.
static void metal_op(MetalBackend& mb, OpType op,
                     std::vector<const Tensor*> ins, Tensor& out,
                     const std::vector<int>& i32 = {},
                     const std::vector<float>& f32 = {}) {
    GraphNode node;
    node.op_type = op;
    node.params.i32 = i32;
    node.params.f32 = f32;
    mb.begin_graph();
    mb.dispatch(node, ins, &out, nullptr);
    mb.end_graph();
}

int main() {
    srand(7);
    MetalBackend mb;
    if (!mb.available()) {
        fprintf(stderr, "Metal not available; skipping\n");
        return 0;  // not a failure on non-Metal CI
    }

    // ---- GEMM: M=8, K=64, N=32 ----
    {
        int M = 8, K = 64, N = 32;
        // A is [K(inner),M]; B is the weight [N,K] with shape[0]=N, shape[1]=K
        // (K contiguous); C is [N(inner),M]. Mirrors kernel_matmul_fp32.
        Tensor A = make_dev(mb, Precision::FP32, K, M);
        Tensor B = make_dev(mb, Precision::FP16, N, K);
        Tensor C = make_dev(mb, Precision::FP32, N, M);

        std::vector<float> a(M*K), bf(N*K);
        fill_rand(a.data(), M*K);
        fill_rand(bf.data(), N*K);
        // upload A (fp32)
        memcpy(A.data, a.data(), M*K*sizeof(float));
        // upload B as fp16
        __fp16* bh = (__fp16*)B.data;
        for (int i = 0; i < N*K; i++) bh[i] = (__fp16)bf[i];

        metal_matmul(mb, A, B, C);

        // Ground-truth scalar reference: C[m,n] = sum_k A[k+m*K] * B[n*K+k].
        std::vector<float> ref(M*N);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                double s = 0;
                for (int k = 0; k < K; k++) s += (double)a[k+m*K] * (double)(float)bh[n*K+k];
                ref[m*N+n] = (float)s;
            }

        CHECK(close((const float*)C.data, ref.data(), M*N, 1e-3f, 2e-2f), "GEMM M=8 K=64 N=32");
    }

    // ---- GEMM larger, tile-aligned (exercises fast simdgroup_store path) ----
    // and an unaligned case (partial edge tiles).
    for (int ci = 0; ci < 2; ci++) {
        int M = ci==0 ? 64 : 40, K = ci==0 ? 256 : 200, N = ci==0 ? 128 : 96;
        Tensor A = make_dev(mb, Precision::FP32, K, M);
        Tensor B = make_dev(mb, Precision::FP16, N, K);
        Tensor C = make_dev(mb, Precision::FP32, N, M);
        std::vector<float> a(M*K), bf(N*K);
        fill_rand(a.data(), M*K); fill_rand(bf.data(), N*K);
        memcpy(A.data, a.data(), M*K*sizeof(float));
        __fp16* bh = (__fp16*)B.data;
        for (int i = 0; i < N*K; i++) bh[i] = (__fp16)bf[i];
        metal_matmul(mb, A, B, C);
        std::vector<float> ref(M*N);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                double s = 0;
                for (int k = 0; k < K; k++) s += (double)a[k+m*K] * (double)(float)bh[n*K+k];
                ref[m*N+n] = (float)s;
            }
        char label[64];
        snprintf(label, sizeof(label), "GEMM tiled M=%d K=%d N=%d", M, K, N);
        CHECK(close((const float*)C.data, ref.data(), M*N, 3e-2f, 3e-2f), label);
    }

    // ---- GEMV: M=1, K=128, N=64 ----
    {
        int M = 1, K = 128, N = 64;
        Tensor A = make_dev(mb, Precision::FP32, K, M);
        Tensor B = make_dev(mb, Precision::FP16, N, K);
        Tensor C = make_dev(mb, Precision::FP32, N, M);

        std::vector<float> a(M*K), bf(N*K);
        fill_rand(a.data(), M*K);
        fill_rand(bf.data(), N*K);
        memcpy(A.data, a.data(), M*K*sizeof(float));
        __fp16* bh = (__fp16*)B.data;
        for (int i = 0; i < N*K; i++) bh[i] = (__fp16)bf[i];

        metal_matmul(mb, A, B, C);

        std::vector<float> ref(M*N);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                double s = 0;
                for (int k = 0; k < K; k++) s += (double)a[k+m*K] * (double)(float)bh[n*K+k];
                ref[m*N+n] = (float)s;
            }

        CHECK(close((const float*)C.data, ref.data(), M*N, 1e-3f, 2e-2f), "GEMV M=1 K=128 N=64");
    }

    // ---- GEMV large K (K > threadgroup A-staging cap, like down_proj) ----
    {
        int M = 1, K = 9728, N = 48;   // K exceeds AS_CAP=4096 in the shader
        Tensor A = make_dev(mb, Precision::FP32, K, M);
        Tensor B = make_dev(mb, Precision::FP16, N, K);
        Tensor C = make_dev(mb, Precision::FP32, N, M);
        std::vector<float> a(M*K), bf(N*K);
        fill_rand(a.data(), M*K);
        fill_rand(bf.data(), N*K);
        memcpy(A.data, a.data(), M*K*sizeof(float));
        __fp16* bh = (__fp16*)B.data;
        for (int i = 0; i < N*K; i++) bh[i] = (__fp16)bf[i];
        metal_matmul(mb, A, B, C);
        std::vector<float> ref(N);
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)a[k] * (double)(float)bh[n*K+k];
            ref[n] = (float)s;
        }
        CHECK(close((const float*)C.data, ref.data(), N, 5e-2f, 3e-2f), "GEMV M=1 K=9728 N=48 (large K)");
    }

    // ---- GEMV odd N (exercises NR0=2 row-boundary r0+1==N) + K non-mult-of-256 ----
    {
        int M = 1, K = 2560, N = 47;   // N odd, K=2560 (=10*256, clean) — use 2600 for tail
        K = 2600;                      // 2600 % 256 = 40 -> exercises the K tail loop
        Tensor A = make_dev(mb, Precision::FP32, K, M);
        Tensor B = make_dev(mb, Precision::FP16, N, K);
        Tensor C = make_dev(mb, Precision::FP32, N, M);
        std::vector<float> a(M*K), bf(N*K);
        fill_rand(a.data(), M*K);
        fill_rand(bf.data(), N*K);
        memcpy(A.data, a.data(), M*K*sizeof(float));
        __fp16* bh = (__fp16*)B.data;
        for (int i = 0; i < N*K; i++) bh[i] = (__fp16)bf[i];
        metal_matmul(mb, A, B, C);
        std::vector<float> ref(N);
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)a[k] * (double)(float)bh[n*K+k];
            ref[n] = (float)s;
        }
        CHECK(close((const float*)C.data, ref.data(), N, 5e-2f, 3e-2f), "GEMV M=1 K=2600 N=47 (odd N, K tail)");
    }

    // ---- ADD (elementwise) ----
    {
        int n = 512;
        Tensor A = make_dev(mb, Precision::FP32, n, 1);
        Tensor B = make_dev(mb, Precision::FP32, n, 1);
        Tensor O = make_dev(mb, Precision::FP32, n, 1);
        std::vector<float> a(n), b(n);
        fill_rand(a.data(), n); fill_rand(b.data(), n);
        memcpy(A.data, a.data(), n*4); memcpy(B.data, b.data(), n*4);
        metal_op(mb, OpType::ADD, {&A,&B}, O);
        std::vector<float> ref(n);
        for (int i=0;i<n;i++) ref[i]=a[i]+b[i];
        CHECK(close((const float*)O.data, ref.data(), n, 1e-5f, 1e-5f), "ADD n=512");
    }

    // ---- MUL (elementwise) ----
    {
        int n = 512;
        Tensor A = make_dev(mb, Precision::FP32, n, 1);
        Tensor B = make_dev(mb, Precision::FP32, n, 1);
        Tensor O = make_dev(mb, Precision::FP32, n, 1);
        std::vector<float> a(n), b(n);
        fill_rand(a.data(), n); fill_rand(b.data(), n);
        memcpy(A.data, a.data(), n*4); memcpy(B.data, b.data(), n*4);
        metal_op(mb, OpType::MUL, {&A,&B}, O);
        std::vector<float> ref(n);
        for (int i=0;i<n;i++) ref[i]=a[i]*b[i];
        CHECK(close((const float*)O.data, ref.data(), n, 1e-5f, 1e-5f), "MUL n=512");
    }

    // ---- SILU ----
    {
        int n = 512;
        Tensor X = make_dev(mb, Precision::FP32, n, 1);
        Tensor O = make_dev(mb, Precision::FP32, n, 1);
        std::vector<float> x(n);
        fill_rand(x.data(), n);
        memcpy(X.data, x.data(), n*4);
        metal_op(mb, OpType::SILU, {&X}, O);
        std::vector<float> ref(n);
        for (int i=0;i<n;i++) ref[i]=x[i]/(1.0f+std::exp(-x[i]));
        CHECK(close((const float*)O.data, ref.data(), n, 1e-5f, 1e-4f), "SILU n=512");
    }

    // ---- SWIGLU: silu(gate)*up over merged [2I, S] (dim0=2I inner) ----
    {
        int I = 48, S = 5;                 // merged dim0 = 2I = 96, S rows
        Tensor M = make_dev(mb, Precision::FP32, 2*I, S);  // [2I, S]
        Tensor O = make_dev(mb, Precision::FP32, I, S);    // [I, S]
        std::vector<float> m((size_t)2*I*S);
        fill_rand(m.data(), m.size());
        memcpy(M.data, m.data(), m.size()*4);
        metal_op(mb, OpType::SWIGLU, {&M}, O);
        // merged row-major [2I, S]: element (d, s) at m[d + s*2I]? No — make_dev
        // sets shape[0]=2I inner, shape[1]=S; row-major means stride[0]=1,
        // stride[1]=2I, so element (d,s) at m[s*2I + d]. gate=d in [0,I), up=[I,2I).
        std::vector<float> ref((size_t)I*S);
        for (int s=0;s<S;s++)
            for (int i=0;i<I;i++){
                float g = m[s*2*I + i], u = m[s*2*I + I + i];
                ref[s*I + i] = (g/(1.0f+std::exp(-g))) * u;
            }
        CHECK(close((const float*)O.data, ref.data(), I*S, 1e-5f, 1e-4f), "SWIGLU I=48 S=5");
    }

    // ---- RMS_NORM over dim0 ----
    {
        int D = 128, rows = 8;
        Tensor X = make_dev(mb, Precision::FP32, D, rows);
        Tensor W = make_dev(mb, Precision::FP32, D, 1);
        Tensor O = make_dev(mb, Precision::FP32, D, rows);
        std::vector<float> x(D*rows), w(D);
        fill_rand(x.data(), D*rows); fill_rand(w.data(), D);
        memcpy(X.data, x.data(), D*rows*4); memcpy(W.data, w.data(), D*4);
        float eps = 1e-6f;
        metal_op(mb, OpType::RMS_NORM, {&X,&W}, O, {}, {eps});
        std::vector<float> ref(D*rows);
        for (int r=0;r<rows;r++){
            double ss=0; for(int i=0;i<D;i++){double v=x[r*D+i]; ss+=v*v;}
            double inv=1.0/std::sqrt(ss/D + eps);
            for(int i=0;i<D;i++) ref[r*D+i]=(float)(x[r*D+i]*inv*w[i]);
        }
        CHECK(close((const float*)O.data, ref.data(), D*rows, 1e-4f, 1e-3f), "RMS_NORM D=128 rows=8");
    }

    // ---- ROPE (interleave=false), layout [head_dim, seq, heads] ----
    {
        int hd = 64, S = 4, H = 2;
        int half = hd/2;
        Tensor X   = make_dev3(mb, Precision::FP32, hd, S, H);
        Tensor IN  = make_dev3(mb, Precision::FP32, hd, S, H);
        Tensor COS = make_dev(mb, Precision::FP32, half, S);   // [half, S]
        Tensor SIN = make_dev(mb, Precision::FP32, half, S);
        std::vector<float> in(hd*S*H), cs(half*S), sn(half*S);
        fill_rand(in.data(), hd*S*H);
        for (int i=0;i<half*S;i++){ cs[i]=std::cos(0.01f*i); sn[i]=std::sin(0.01f*i); }
        memcpy(IN.data, in.data(), in.size()*4);
        memcpy(COS.data, cs.data(), cs.size()*4);
        memcpy(SIN.data, sn.data(), sn.size()*4);
        // X starts as a copy target; dispatch copies IN->X then rotates.
        metal_op(mb, OpType::ROTARY_EMBED, {&IN,&COS,&SIN}, X, {hd});
        // reference: for each head, pos, pair i: rotate (in[i], in[i+half]) by (cos,sin)[pos*half+i]
        std::vector<float> ref(hd*S*H);
        for (int h=0;h<H;h++)
            for (int p=0;p<S;p++)
                for (int i=0;i<half;i++){
                    int base = h*(hd*S) + p*hd;   // [hd, S, H] row-major: stride pos=hd, head=hd*S
                    float x0=in[base+i], x1=in[base+i+half];
                    float c=cs[p*half+i], s=sn[p*half+i];
                    ref[base+i]      = x0*c - x1*s;
                    ref[base+i+half] = x0*s + x1*c;
                }
        CHECK(close((const float*)X.data, ref.data(), hd*S*H, 1e-4f, 1e-3f), "ROPE hd=64 S=4 H=2");
    }

    // ---- CONTIGUOUS: materialize a permuted (strided) tensor ----
    {
        // source logical [a=8, b=4] contiguous; permute to [b, a] (strided), then
        // CONTIGUOUS should produce a dense [b, a] equal to the transpose.
        int a=8, b=4;
        Tensor SRC = make_dev(mb, Precision::FP32, a, b);   // shape[0]=a, shape[1]=b
        std::vector<float> s(a*b); fill_rand(s.data(), a*b);
        memcpy(SRC.data, s.data(), a*b*4);
        // build a permuted view by hand: swap dims 0 and 1
        Tensor P = SRC;
        P.shape[0]=b; P.shape[1]=a;
        P.stride[0]=SRC.stride[1]; P.stride[1]=SRC.stride[0];
        P.device_data = SRC.device_data; P.device_offset = SRC.device_offset;
        Tensor O = make_dev(mb, Precision::FP32, b, a);     // dense [b, a]
        metal_op(mb, OpType::CONTIGUOUS, {&P}, O);
        // ref: O[j + i*b]?  O is [b(inner), a]: O[i2*... ] row-major over shape.
        // O flat index t: i0=t%b, i1=t/b ; source element = SRC[i1 (a-idx), i0 (b-idx)]
        //   = s[i1 + i0*a]  (SRC row-major [a,b]: s[a_idx + b_idx*a])
        std::vector<float> ref(a*b);
        for (int t=0;t<a*b;t++){ int i0=t%b, i1=t/b; ref[t]=s[i1 + i0*a]; }
        CHECK(close((const float*)O.data, ref.data(), a*b, 1e-6f, 1e-6f), "CONTIGUOUS transpose 8x4");
    }

    // ---- CONTIGUOUS 3D transpose (attention case: swap dims 1<->2) ----
    {
        // source logical [d=16, h=6, s=5] contiguous; permute dims 1<->2 to
        // [d, s, h] (strided), CONTIGUOUS -> dense [d, s, h].
        int d=16, h=6, s=5;
        Tensor SRC = make_dev3(mb, Precision::FP32, d, h, s);
        std::vector<float> src(d*h*s); fill_rand(src.data(), d*h*s);
        memcpy(SRC.data, src.data(), d*h*s*4);
        Tensor P = SRC;                       // [d, s, h] view of [d,h,s]
        P.shape[0]=d; P.shape[1]=s; P.shape[2]=h;
        P.stride[0]=SRC.stride[0]; P.stride[1]=SRC.stride[2]; P.stride[2]=SRC.stride[1];
        P.device_data = SRC.device_data; P.device_offset = SRC.device_offset;
        Tensor O = make_dev3(mb, Precision::FP32, d, s, h);
        metal_op(mb, OpType::CONTIGUOUS, {&P}, O);
        // dense O index t -> (i0=d, i1=s, i2=h); source = SRC[i0, i2(h), i1(s)]
        //   SRC row-major [d,h,s]: src[i0 + i2*d + i1*d*h]
        std::vector<float> ref(d*h*s);
        for (int t=0;t<d*h*s;t++){
            int i0=t%d, r=t/d; int i1=r%s, i2=r/s;
            ref[t]=src[i0 + i2*d + i1*d*h];
        }
        CHECK(close((const float*)O.data, ref.data(), d*h*s, 1e-6f, 1e-6f), "CONTIGUOUS 3D transpose d16 h6 s5");
    }

    // ---- SDPA: prefill (S>1) and decode (S=1), GQA, causal, FP16 cache ----
    // Layout: Q [head_dim, S, num_heads], K/V_cur [head_dim, S, num_kv], cache
    // is a Shared buffer with a 64-byte metadata header then FP16 data laid out
    // [kv_head, position, feature].
    auto run_sdpa_cfg = [&](int S, int past, int hd, int num_heads, int num_kv,
                            int max_seq, const char* label) {
        int hpg = num_heads / num_kv;
        float scale = 1.0f / std::sqrt((float)hd);

        Tensor Q  = make_dev3(mb, Precision::FP32, hd, S, num_heads);
        Tensor Kc = make_dev3(mb, Precision::FP32, hd, S, num_kv);
        Tensor Vc = make_dev3(mb, Precision::FP32, hd, S, num_kv);
        Tensor O  = make_dev3(mb, Precision::FP32, hd, S, num_heads);

        // KV caches: 64B header + FP16 data [num_kv, max_seq, hd].
        size_t cache_bytes = 64 + (size_t)num_kv * max_seq * hd * 2;
        Tensor Kcache = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, 1,1,1,1, nullptr);
        Tensor Vcache = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, 1,1,1,1, nullptr);
        mb.alloc_persistent(Kcache, cache_bytes);
        mb.alloc_persistent(Vcache, cache_bytes);
        // metadata: current_seq_len=past, max_seq_len=max_seq
        auto set_meta = [&](Tensor& c){
            uint64_t* m = (uint64_t*)c.data;
            for (int i=0;i<8;i++) m[i]=0;
            m[0]=(uint64_t)past; m[1]=(uint64_t)max_seq;
        };
        set_meta(Kcache); set_meta(Vcache);

        std::vector<float> q(hd*S*num_heads), kc(hd*S*num_kv), vc(hd*S*num_kv);
        fill_rand(q.data(), q.size()); fill_rand(kc.data(), kc.size()); fill_rand(vc.data(), vc.size());
        memcpy(Q.data, q.data(), q.size()*4);
        memcpy(Kc.data, kc.data(), kc.size()*4);
        memcpy(Vc.data, vc.data(), vc.size()*4);

        // Pre-fill the cache "past" region with FP16 data we control, so the
        // reference and GPU see identical past K/V.
        std::vector<float> pastK((size_t)num_kv*past*hd), pastV((size_t)num_kv*past*hd);
        fill_rand(pastK.data(), pastK.size()); fill_rand(pastV.data(), pastV.size());
        __fp16* Kd = (__fp16*)((char*)Kcache.data + 64);
        __fp16* Vd = (__fp16*)((char*)Vcache.data + 64);
        for (int g=0; g<num_kv; g++)
            for (int j=0;j<past;j++)
                for (int d=0; d<hd; d++){
                    Kd[g*max_seq*hd + j*hd + d] = (__fp16)pastK[g*past*hd + j*hd + d];
                    Vd[g*max_seq*hd + j*hd + d] = (__fp16)pastV[g*past*hd + j*hd + d];
                }

        std::vector<int> i32 = {2 /*kv_cache*/, 1 /*causal*/, num_heads, num_kv, hd, hd};
        std::vector<float> f32 = {scale};
        std::vector<const Tensor*> ins = {&Q,&Kc,&Vc,nullptr,&Kcache,&Vcache};
        metal_op(mb, OpType::SDPA, ins, O, i32, f32);

        // Reference: build the full K/V (past + current) per kv-head in fp16,
        // then causal attention with online-equivalent softmax.
        int dst = past + S;
        auto kv_at = [&](std::vector<float>& past_data, std::vector<float>& cur_data,
                         int g, int j, int d, int width)->float {
            if (j < past) return (float)(__fp16)past_data[g*past*width + j*width + d];
            int s = j - past;
            return (float)(__fp16)cur_data[d + s*width + g*S*width]; // cur layout [hd,S,num_kv]
        };
        std::vector<float> ref(hd*S*num_heads);
        for (int h=0; h<num_heads; h++){
            int g = h/hpg;
            for (int s=0;s<S;s++){
                const float* qq = &q[h*(hd*S) + s*hd];
                int limit = past + s + 1;
                std::vector<float> sc(limit);
                float mx=-1e30f;
                for (int j=0;j<limit;j++){
                    double dot=0; for(int d=0;d<hd;d++) dot += (double)qq[d]*kv_at(pastK,kc,g,j,d,hd);
                    sc[j]=(float)(dot*scale); if(sc[j]>mx)mx=sc[j];
                }
                double den=0; for(int j=0;j<limit;j++){sc[j]=std::exp(sc[j]-mx); den+=sc[j];}
                for (int d=0; d<hd; d++){
                    double o=0; for(int j=0;j<limit;j++) o+=sc[j]*kv_at(pastV,vc,g,j,d,hd);
                    ref[h*(hd*S)+s*hd+d]=(float)(o/den);
                }
            }
        }
        CHECK(close((const float*)O.data, ref.data(), hd*S*num_heads, 2e-2f, 3e-2f), label);
    };
    // hd=32 harness (PV=PAD(32,64)=64) — original coverage.
    auto run_sdpa = [&](int S, int past, const char* label) {
        run_sdpa_cfg(S, past, 32, 4, 2, 256, label);
    };
    run_sdpa(1, 5, "SDPA decode S=1 past=5");
    run_sdpa(1, 100, "SDPA decode S=1 past=100");
    run_sdpa(4, 0, "SDPA prefill S=4 past=0");
    run_sdpa(8, 50, "SDPA prefill S=8 past=50");
    run_sdpa(64, 0, "SDPA prefill S=64 past=0");     // multi query-tile + KV blocks
    run_sdpa(200, 0, "SDPA prefill S=200 past=0");   // ragged tail (200%8=0? no: 200/8=25)
    run_sdpa(37, 11, "SDPA prefill S=37 past=11");   // ragged query tile + past
    run_sdpa(63, 0, "SDPA prefill S=63 past=0");     // < one C=64 block, ragged
    run_sdpa(65, 3, "SDPA prefill S=65 past=3");     // C=64 boundary cross + past

    // Production config: dk=dv=128, GQA 8/2 (exercises PV=128, NO=PV8/NSG=4 PV MMA).
    auto run_sdpa128 = [&](int S, int past, const char* label) {
        run_sdpa_cfg(S, past, 128, 8, 2, 512, label);
    };
    run_sdpa128(1, 7, "SDPA decode hd=128 S=1 past=7");
    run_sdpa128(16, 0, "SDPA prefill hd=128 S=16 past=0");
    run_sdpa128(128, 0, "SDPA prefill hd=128 S=128 past=0");
    run_sdpa128(256, 0, "SDPA prefill hd=128 S=256 past=0");
    run_sdpa128(70, 13, "SDPA prefill hd=128 S=70 past=13"); // ragged Q tile + C cross + past

    if (failures == 0) printf("All Metal op parity tests passed.\n");
    return failures ? 1 : 0;
}
