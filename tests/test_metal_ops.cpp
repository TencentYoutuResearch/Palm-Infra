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
    for (int i = 0; i < n; i++)
        d[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
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
static void metal_matmul(MetalBackend& mb, Tensor& A, Tensor& B, Tensor& C,
                         const std::vector<int>& i32 = {},
                         OpType op = OpType::MATMUL) {
    GraphNode node;
    node.op_type = op;
    node.params.i32 = i32;
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

    // A duplicate release must not enqueue the same MTLBuffer twice. Otherwise
    // two live tensors can alias one pooled allocation.
    {
        Tensor released =
            Tensor::create(Precision::FP32, MemoryType::NONE, 256);
        mb.alloc_output(released, released.nbytes(), nullptr);
        mb.free_output(released, nullptr);
        mb.free_output(released, nullptr);

        Tensor first = Tensor::create(Precision::FP32, MemoryType::NONE, 256);
        Tensor second = Tensor::create(Precision::FP32, MemoryType::NONE, 256);
        mb.alloc_output(first, first.nbytes(), nullptr);
        mb.alloc_output(second, second.nbytes(), nullptr);
        CHECK(first.device_data != second.device_data,
              "Metal pool rejects duplicate release");
        mb.free_output(first, nullptr);
        mb.free_output(second, nullptr);
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

    // ---- GEMM fused SiLU on only the gate half (MLA merged gate/up) --------
    {
        int M=3, K=64, N=32, gate=16;
        Tensor A=make_dev(mb, Precision::FP32, K, M);
        Tensor B=make_dev(mb, Precision::FP16, N, K);
        Tensor C=make_dev(mb, Precision::FP32, N, M);
        std::vector<float> a(M*K), bf(N*K), ref(M*N);
        fill_rand(a.data(), a.size()); fill_rand(bf.data(), bf.size());
        memcpy(A.data, a.data(), a.size()*sizeof(float));
        __fp16* bh=(__fp16*)B.data;
        for (int i=0;i<N*K;i++) bh[i]=(__fp16)bf[i];
        metal_matmul(mb, A, B, C, {1,0,gate});
        for (int m=0;m<M;m++) for (int n=0;n<N;n++) {
            double sum=0;
            for (int k=0;k<K;k++) sum += (double)a[m*K+k]*(double)(float)bh[n*K+k];
            float v=(float)sum;
            ref[m*N+n] = n < gate ? v/(1.0f+std::exp(-v)) : v;
        }
        CHECK(close((const float*)C.data, ref.data(), M*N, 1e-3f, 2e-2f),
              "GEMM partial fused SiLU gate=[0,16), up unchanged");
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

    // ---- GEMV_SPARSE_A correctness fallback through the tuned dense GEMV ---
    {
        int K = 128, N = 64;
        Tensor A = make_dev(mb, Precision::FP32, K, 1);
        Tensor B = make_dev(mb, Precision::FP16, N, K);
        Tensor C = make_dev(mb, Precision::FP32, N, 1);
        std::vector<float> a(K, 0.0f), bf(N * K), ref(N);
        for (int k = 0; k < K; k += 7) a[k] = (k & 1) ? -0.25f : 0.5f;
        fill_rand(bf.data(), N * K);
        memcpy(A.data, a.data(), K * sizeof(float));
        __fp16* bh = (__fp16*)B.data;
        for (int i = 0; i < N * K; ++i) bh[i] = (__fp16)bf[i];
        metal_matmul(mb, A, B, C, {}, OpType::GEMV_SPARSE_A);
        for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) sum += a[k] * (float)bh[n * K + k];
            ref[n] = (float)sum;
        }
        CHECK(close((const float*)C.data, ref.data(), N, 1e-3f, 2e-2f),
              "GEMV_SPARSE_A dense Metal fallback");
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

    // ---- MUL of gate/up slice views with inherited wide row stride --------
    {
        int I=24, rows=5, full=2*I;
        Tensor M=make_dev(mb,Precision::FP32,full,rows);
        Tensor G=M, U=M;
        G.shape[0]=I;
        U.shape[0]=I; U.device_offset=(size_t)I*sizeof(float);
        Tensor O=make_dev(mb,Precision::FP32,I,rows);
        std::vector<float> merged(full*rows),ref(I*rows);
        fill_rand(merged.data(),merged.size());
        memcpy(M.data,merged.data(),merged.size()*sizeof(float));
        metal_op(mb,OpType::MUL,{&G,&U},O);
        for(int r=0;r<rows;r++) for(int i=0;i<I;i++)
            ref[r*I+i]=merged[r*full+i]*merged[r*full+I+i];
        CHECK(close((const float*)O.data,ref.data(),ref.size(),1e-6f,1e-6f),
              "MUL strided gate/up slice views");
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

    // ---- SIGMOID ----
    {
        int n = 512;
        Tensor X = make_dev(mb, Precision::FP32, n, 1);
        Tensor O = make_dev(mb, Precision::FP32, n, 1);
        std::vector<float> x(n);
        fill_rand(x.data(), n);
        memcpy(X.data, x.data(), n*4);
        metal_op(mb, OpType::SIGMOID, {&X}, O);
        std::vector<float> ref(n);
        for (int i=0;i<n;i++) ref[i]=1.0f/(1.0f+std::exp(-x[i]));
        CHECK(close((const float*)O.data, ref.data(), n, 1e-5f, 1e-4f), "SIGMOID n=512");
    }

    // ---- Remaining scalar activations (including recurrent exact variants) --
    for (OpType op : {OpType::TANH, OpType::EXP, OpType::EXP_EXACT,
                      OpType::SIGMOID_EXACT, OpType::SOFTPLUS}) {
        int D = 37, rows = 3;
        Tensor X = make_dev(mb, Precision::FP32, D, rows);
        Tensor O = make_dev(mb, Precision::FP32, D, rows);
        std::vector<float> x(D * rows), ref(D * rows);
        for (int i = 0; i < D * rows; ++i)
            x[i] = -6.0f + 12.0f * (float)i / (float)(D * rows - 1);
        memcpy(X.data, x.data(), x.size() * sizeof(float));
        metal_op(mb, op, {&X}, O);
        for (int i = 0; i < D * rows; ++i) {
            float v = x[i];
            if (op == OpType::TANH)
                ref[i] = std::tanh(v);
            else if (op == OpType::EXP || op == OpType::EXP_EXACT)
                ref[i] = std::exp(v);
            else if (op == OpType::SIGMOID_EXACT)
                ref[i] = 1.0f / (1.0f + std::exp(-v));
            else
                ref[i] = std::log1p(std::exp(v));
        }
        char label[64];
        snprintf(label, sizeof(label), "%s D=37 rows=3",
                 op_type_name(op));
        CHECK(close((const float*)O.data, ref.data(), D * rows, 2e-5f, 2e-5f),
              label);
    }

    // ---- SHORTCONV: depth-wise causal conv1d + silu (ksize=4) ----
    // Reference (mirrors execute.cpp): window = [state(3) | x(seq)]; per group g,
    // out[g,i] = silu(Σ_{k} win[i+k]*w[g,k]); state' = last 3 real x values.
    auto run_shortconv = [&](int groups, int seq, const char* label) {
        const int ks = 4, pre = ks - 1;
        Tensor X = make_dev(mb, Precision::FP32, groups, seq);   // data [seq, groups]
        Tensor W = make_dev(mb, Precision::FP32, groups, ks);
        Tensor ST = make_dev(mb, Precision::FP32, groups, pre);  // persistent state
        Tensor O = make_dev(mb, Precision::FP32, groups, seq);
        std::vector<float> x(groups*seq), w(groups*ks), st(groups*pre);
        fill_rand(x.data(), x.size()); fill_rand(w.data(), w.size()); fill_rand(st.data(), st.size());
        memcpy(X.data, x.data(), x.size()*4);
        memcpy(W.data, w.data(), w.size()*4);
        memcpy(ST.data, st.data(), st.size()*4);
        std::vector<int> i32 = {ks, seq};   // kernel_size, n_real
        metal_op(mb, OpType::SHORTCONV, {&X,&W,&ST}, O, i32);
        // Reference output + state.
        std::vector<float> ref(groups*seq), rst(groups*pre);
        for (int g=0; g<groups; g++) {
            std::vector<float> win(pre+seq);
            for (int p=0;p<pre;p++) win[p]=st[g*pre+p];
            for (int s=0;s<seq;s++) win[pre+s]=x[s*groups+g];   // x layout [seq,groups]
            for (int i=0;i<seq;i++){
                float sum=0; for(int k=0;k<ks;k++) sum+=win[i+k]*w[g*ks+k];
                ref[g*seq+i]=sum/(1.0f+std::exp(-sum));
            }
            for (int p=0;p<pre;p++) rst[g*pre+p]=win[seq+p]; // last 3 of [state|x]
        }
        bool ok_out = close((const float*)O.data, ref.data(), groups*seq, 2e-4f, 2e-4f);
        bool ok_st  = close((const float*)ST.data, rst.data(), groups*pre, 2e-4f, 2e-4f);
        CHECK(ok_out && ok_st, label);
    };
    run_shortconv(64, 1, "SHORTCONV decode groups=64 seq=1");
    run_shortconv(48, 8, "SHORTCONV prefill groups=48 seq=8");
    run_shortconv(6144, 4, "SHORTCONV groups=6144 seq=4 (prod width)");

    // ---- GATED_DELTANET_DECODE (seq=1): GDN recurrence + RMSNormGated ----
    // Scalar reference mirrors kernels/gdn_neon.h gdn_recurrence.
    {
        int H = 4, VH = 4, K = 128, Vd = 128;   // repeat=1
        float l2eps = 1e-6f, rmseps = 1e-6f, scale = 1.0f/std::sqrt((float)K);
        int qkv_dim = H*K, qkv_total = 2*H*K + VH*Vd, zdim = VH*Vd;
        // buffers (seq=1)
        Tensor QKV = make_dev(mb, Precision::FP32, qkv_total, 1);
        Tensor A = make_dev(mb, Precision::FP32, H, 1);
        Tensor B = make_dev(mb, Precision::FP32, H, 1);
        Tensor Z = make_dev(mb, Precision::FP32, zdim, 1);
        Tensor ALG = make_dev(mb, Precision::FP32, VH, 1);
        Tensor DTB = make_dev(mb, Precision::FP32, VH, 1);
        Tensor NRM = make_dev(mb, Precision::FP32, Vd, 1);
        Tensor ST = make_dev(mb, Precision::FP32, VH*K*Vd, 1);
        Tensor O = make_dev(mb, Precision::FP32, zdim, 1);
        std::vector<float> qkv(qkv_total), a(H), b(H), z(zdim), alg(VH), dtb(VH), nrm(Vd), st((size_t)VH*K*Vd);
        fill_rand(qkv.data(), qkv.size()); fill_rand(a.data(), H); fill_rand(b.data(), H);
        fill_rand(z.data(), zdim); fill_rand(alg.data(), VH); fill_rand(dtb.data(), VH);
        fill_rand(nrm.data(), Vd); fill_rand(st.data(), st.size());
        memcpy(QKV.data,qkv.data(),qkv.size()*4); memcpy(A.data,a.data(),H*4); memcpy(B.data,b.data(),H*4);
        memcpy(Z.data,z.data(),zdim*4); memcpy(ALG.data,alg.data(),VH*4); memcpy(DTB.data,dtb.data(),VH*4);
        memcpy(NRM.data,nrm.data(),Vd*4); memcpy(ST.data,st.data(),st.size()*4);
        std::vector<int> i32 = {H, K, Vd, 1, 1, 4, 0, VH};
        std::vector<float> f32 = {rmseps, l2eps, scale};
        std::vector<const Tensor*> ins = {&QKV,&A,&B,&Z,&ALG,&DTB,&NRM,&ST};
        metal_op(mb, OpType::GATED_DELTANET_DECODE, ins, O, i32, f32);

        // reference
        std::vector<float> ref(zdim), rst = st;
        auto silu=[](float x){return x/(1.f+std::exp(-x));};
        for (int vh=0; vh<VH; vh++){
            int kh = vh; // repeat=1
            std::vector<float> q(K), kk(K), v(Vd);
            for(int d=0;d<K;d++){ q[d]=qkv[kh*K+d]; kk[d]=qkv[qkv_dim+kh*K+d]; }
            for(int d=0;d<Vd;d++) v[d]=qkv[2*qkv_dim+vh*Vd+d];
            double qs=0,ks=0; for(int d=0;d<K;d++){qs+=(double)q[d]*q[d];ks+=(double)kk[d]*kk[d];}
            float qi=1.f/std::sqrt((float)qs+l2eps), ki=1.f/std::sqrt((float)ks+l2eps);
            for(int d=0;d<K;d++){q[d]*=qi;kk[d]*=ki;}
            float sp = (a[vh]+dtb[vh]>20.f)?(a[vh]+dtb[vh]):((a[vh]+dtb[vh]<-20.f)?std::exp(a[vh]+dtb[vh]):std::log1p(std::exp(a[vh]+dtb[vh])));
            float gexp=std::exp(-std::exp(alg[vh])*sp), beta=1.f/(1.f+std::exp(-b[vh]));
            float* S = rst.data()+(size_t)vh*K*Vd;
            std::vector<float> kv(Vd,0);
            for(int dk=0;dk<K;dk++)for(int dvv=0;dvv<Vd;dvv++){ S[dk*Vd+dvv]*=gexp; kv[dvv]+=S[dk*Vd+dvv]*kk[dk]; }
            std::vector<float> delta(Vd); for(int dvv=0;dvv<Vd;dvv++) delta[dvv]=(v[dvv]-kv[dvv])*beta;
            std::vector<float> attn(Vd,0);
            for(int dk=0;dk<K;dk++)for(int dvv=0;dvv<Vd;dvv++){ S[dk*Vd+dvv]+=kk[dk]*delta[dvv]; attn[dvv]+=S[dk*Vd+dvv]*q[dk]; }
            double ss=0; for(int dvv=0;dvv<Vd;dvv++){attn[dvv]*=scale; ss+=(double)attn[dvv]*attn[dvv];}
            float rms=1.f/std::sqrt((float)(ss/Vd)+rmseps);
            for(int dvv=0;dvv<Vd;dvv++) ref[vh*Vd+dvv]=attn[dvv]*rms*nrm[dvv]*silu(z[vh*Vd+dvv]);
        }
        bool ok_out = close((const float*)O.data, ref.data(), zdim, 3e-3f, 3e-3f);
        bool ok_st  = close((const float*)ST.data, rst.data(), (int)st.size(), 3e-3f, 3e-3f);
        CHECK(ok_out && ok_st, "GDN_DECODE H=4 K=128 V=128");
    }

    // ---- GATED_DELTANET_PREFILL (seq>1): recurrence over tokens ----
    // Layout: qkv [qkv_total, seq] (dim-major); a/b/z/out [seq, dim] (seq-major).
    {
        int H = 4, VH = 4, K = 128, Vd = 128, S = 6;
        float l2eps = 1e-6f, rmseps = 1e-6f, scale = 1.0f/std::sqrt((float)K);
        int qkv_dim = H*K, qkv_total = 2*H*K + VH*Vd, zdim = VH*Vd;
        Tensor QKV = make_dev(mb, Precision::FP32, qkv_total, S);   // [qkv_total, seq]
        Tensor A = make_dev(mb, Precision::FP32, VH, S);            // [seq, VH] data
        Tensor B = make_dev(mb, Precision::FP32, VH, S);
        Tensor Z = make_dev(mb, Precision::FP32, zdim, S);          // [seq, zdim]
        Tensor ALG = make_dev(mb, Precision::FP32, VH, 1);
        Tensor DTB = make_dev(mb, Precision::FP32, VH, 1);
        Tensor NRM = make_dev(mb, Precision::FP32, Vd, 1);
        Tensor ST = make_dev(mb, Precision::FP32, VH*K*Vd, 1);
        Tensor O = make_dev(mb, Precision::FP32, zdim, S);          // [seq, zdim]
        std::vector<float> qkv((size_t)qkv_total*S), a((size_t)VH*S), b((size_t)VH*S),
                           z((size_t)zdim*S), alg(VH), dtb(VH), nrm(Vd), st((size_t)VH*K*Vd);
        fill_rand(qkv.data(),qkv.size()); fill_rand(a.data(),a.size()); fill_rand(b.data(),b.size());
        fill_rand(z.data(),z.size()); fill_rand(alg.data(),VH); fill_rand(dtb.data(),VH);
        fill_rand(nrm.data(),Vd); fill_rand(st.data(),st.size());
        memcpy(QKV.data,qkv.data(),qkv.size()*4); memcpy(A.data,a.data(),a.size()*4);
        memcpy(B.data,b.data(),b.size()*4); memcpy(Z.data,z.data(),z.size()*4);
        memcpy(ALG.data,alg.data(),VH*4); memcpy(DTB.data,dtb.data(),VH*4);
        memcpy(NRM.data,nrm.data(),Vd*4); memcpy(ST.data,st.data(),st.size()*4);
        std::vector<int> i32 = {H, K, Vd, S, 1, 4, 0, VH};
        std::vector<float> f32 = {rmseps, l2eps, scale};
        std::vector<const Tensor*> ins = {&QKV,&A,&B,&Z,&ALG,&DTB,&NRM,&ST};
        metal_op(mb, OpType::GATED_DELTANET_PREFILL, ins, O, i32, f32);

        std::vector<float> ref((size_t)zdim*S), rst = st;
        auto silu=[](float x){return x/(1.f+std::exp(-x));};
        for (int vh=0; vh<VH; vh++){
            int kh=vh; float* S_=rst.data()+(size_t)vh*K*Vd;
            for (int t=0;t<S;t++){
                std::vector<float> q(K),kk(K),v(Vd);
                for(int d=0;d<K;d++){ q[d]=qkv[(size_t)(kh*K+d)*S+t]; kk[d]=qkv[(size_t)(qkv_dim+kh*K+d)*S+t]; }
                for(int d=0;d<Vd;d++) v[d]=qkv[(size_t)(2*qkv_dim+vh*Vd+d)*S+t];
                double qs=0,ks=0; for(int d=0;d<K;d++){qs+=(double)q[d]*q[d];ks+=(double)kk[d]*kk[d];}
                float qi=1.f/std::sqrt((float)qs+l2eps), ki=1.f/std::sqrt((float)ks+l2eps);
                for(int d=0;d<K;d++){q[d]*=qi;kk[d]*=ki;}
                float ab=a[(size_t)t*VH+vh]+dtb[vh];
                float sp=(ab>20.f)?ab:((ab<-20.f)?std::exp(ab):std::log1p(std::exp(ab)));
                float gexp=std::exp(-std::exp(alg[vh])*sp), beta=1.f/(1.f+std::exp(-b[(size_t)t*VH+vh]));
                std::vector<float> kv(Vd,0);
                for(int dk=0;dk<K;dk++)for(int dvv=0;dvv<Vd;dvv++){ S_[dk*Vd+dvv]*=gexp; kv[dvv]+=S_[dk*Vd+dvv]*kk[dk]; }
                std::vector<float> delta(Vd); for(int dvv=0;dvv<Vd;dvv++) delta[dvv]=(v[dvv]-kv[dvv])*beta;
                std::vector<float> attn(Vd,0);
                for(int dk=0;dk<K;dk++)for(int dvv=0;dvv<Vd;dvv++){ S_[dk*Vd+dvv]+=kk[dk]*delta[dvv]; attn[dvv]+=S_[dk*Vd+dvv]*q[dk]; }
                double ss=0; for(int dvv=0;dvv<Vd;dvv++){attn[dvv]*=scale; ss+=(double)attn[dvv]*attn[dvv];}
                float rms=1.f/std::sqrt((float)(ss/Vd)+rmseps);
                for(int dvv=0;dvv<Vd;dvv++) ref[(size_t)t*zdim+vh*Vd+dvv]=attn[dvv]*rms*nrm[dvv]*silu(z[(size_t)t*zdim+vh*Vd+dvv]);
            }
        }
        bool ok_out = close((const float*)O.data, ref.data(), (int)((size_t)zdim*S), 5e-3f, 5e-3f);
        bool ok_st  = close((const float*)ST.data, rst.data(), (int)st.size(), 5e-3f, 5e-3f);
        CHECK(ok_out && ok_st, "GDN_PREFILL H=4 K=128 V=128 S=6");
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

    // ---- LAYER_NORM over dim0 ---------------------------------------------
    {
        int D = 127, rows = 5;
        Tensor X = make_dev(mb, Precision::FP32, D, rows);
        Tensor W = make_dev(mb, Precision::FP32, D, 1);
        Tensor B = make_dev(mb, Precision::FP32, D, 1);
        Tensor O = make_dev(mb, Precision::FP32, D, rows);
        std::vector<float> x(D * rows), w(D), b(D), ref(D * rows);
        fill_rand(x.data(), (int)x.size());
        fill_rand(w.data(), D);
        fill_rand(b.data(), D);
        memcpy(X.data, x.data(), x.size() * sizeof(float));
        memcpy(W.data, w.data(), w.size() * sizeof(float));
        memcpy(B.data, b.data(), b.size() * sizeof(float));
        const float eps = 1e-5f;
        metal_op(mb, OpType::LAYER_NORM, {&X, &W, &B}, O, {}, {eps});
        for (int row = 0; row < rows; ++row) {
            double mean = 0.0;
            for (int i = 0; i < D; ++i) mean += x[row * D + i];
            mean /= D;
            double var = 0.0;
            for (int i = 0; i < D; ++i) {
                double z = x[row * D + i] - mean;
                var += z * z;
            }
            var /= D;
            float scale = 1.0f / std::sqrt((float)var + eps);
            for (int i = 0; i < D; ++i)
                ref[row * D + i] =
                    (x[row * D + i] - (float)mean) * scale * w[i] + b[i];
        }
        CHECK(close((const float*)O.data, ref.data(), D * rows, 2e-4f, 2e-3f),
              "LAYER_NORM D=127 rows=5");
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
        metal_op(mb, OpType::ROTARY_EMBED, {&IN,&COS,&SIN}, X, {hd,1});
        // interleaved reference: rotate adjacent pairs (2i, 2i+1).
        std::vector<float> ref(hd*S*H);
        for (int h=0;h<H;h++)
            for (int p=0;p<S;p++)
                for (int i=0;i<half;i++){
                    int base = h*(hd*S) + p*hd;   // [hd, S, H] row-major: stride pos=hd, head=hd*S
                    float x0=in[base+2*i], x1=in[base+2*i+1];
                    float c=cs[p*half+i], s=sn[p*half+i];
                    ref[base+2*i]   = x0*c - x1*s;
                    ref[base+2*i+1] = x0*s + x1*c;
                }
        CHECK(close((const float*)X.data, ref.data(), hd*S*H, 1e-4f, 1e-3f),
              "ROPE interleaved hd=64 S=4 H=2");
    }

    // ---- ROPE materializes a non-zero-offset, wide-stride slice (MLA) ------
    {
        int full=96, off=32, hd=64, S=7, H=2, half=hd/2;
        Tensor PARENT=make_dev3(mb, Precision::FP32, full, S, H);
        Tensor IN=PARENT;
        IN.shape[0]=hd; IN.device_offset=(size_t)off*sizeof(float);
        Tensor X=make_dev3(mb, Precision::FP32, hd, S, H);
        Tensor COS=make_dev(mb, Precision::FP32, half, S);
        Tensor SIN=make_dev(mb, Precision::FP32, half, S);
        std::vector<float> parent(full*S*H), cs(half*S), sn(half*S), ref(hd*S*H);
        fill_rand(parent.data(), parent.size());
        for(int i=0;i<half*S;i++){cs[i]=std::cos(0.013f*i);sn[i]=std::sin(0.013f*i);}
        memcpy(PARENT.data,parent.data(),parent.size()*sizeof(float));
        memcpy(COS.data,cs.data(),cs.size()*sizeof(float));
        memcpy(SIN.data,sn.data(),sn.size()*sizeof(float));
        metal_op(mb,OpType::ROTARY_EMBED,{&IN,&COS,&SIN},X,{hd,0});
        for(int h=0;h<H;h++) for(int p=0;p<S;p++) for(int i=0;i<half;i++){
            int src=h*full*S+p*full+off;
            int dst=h*hd*S+p*hd;
            float x0=parent[src+i],x1=parent[src+i+half],c=cs[p*half+i],s=sn[p*half+i];
            ref[dst+i]=x0*c-x1*s; ref[dst+i+half]=x0*s+x1*c;
        }
        CHECK(close((const float*)X.data,ref.data(),ref.size(),1e-4f,1e-3f),
              "ROPE strided non-zero-offset slice -> dense output");
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

    // ---- W8A8 prefill GEMM (int8 activations x int8 per-channel weights) ----
    // Guards the int8xint8->int32 tensor path, whose cooperative-tensor layout
    // is compiler-sensitive (must match between offline metallib and runtime).
    // Enabled via MOLLM_METAL_W8A8; weights + scales live in a registered region.
    {
        setenv("MOLLM_METAL_W8A8", "1", 1);
        for (int ci = 0; ci < 3 && mb.has_tensor_path(); ci++) {
            int M = ci==0 ? 40 : (ci==1 ? 256 : 33);
            int K = ci==0 ? 256 : (ci==1 ? 1024 : 96);
            int N = ci==0 ? 96  : (ci==1 ? 512  : 80);

            std::vector<float> a(M*K); fill_rand(a.data(), M*K);
            // int8 per-channel weights + fp32 scales laid out in one region.
            std::vector<int8_t> wi(N*K);
            std::vector<float>  sw(N);
            for (int n = 0; n < N; n++) {
                float mx = 0; std::vector<float> row(K);
                for (int k = 0; k < K; k++) { row[k] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
                    mx = std::max(mx, std::fabs(row[k])); }
                float s = mx / 127.0f, inv = mx > 0 ? 127.0f/mx : 0; sw[n] = s;
                for (int k = 0; k < K; k++) {
                    int q = (int)std::lround(row[k]*inv);
                    wi[n*K+k] = (int8_t)std::max(-127, std::min(127, q));
                }
            }
            // One region: [ int8 weights (N*K) | pad to 4B | fp32 scales (N) ].
            size_t woff = 0, soff = (N*K + 3) & ~size_t(3);
            size_t region = soff + N*sizeof(float);
            std::vector<uint8_t> buf(region, 0);
            memcpy(buf.data()+woff, wi.data(), N*K);
            memcpy(buf.data()+soff, sw.data(), N*sizeof(float));
            mb.register_weight_region(buf.data(), region);

            Tensor A = make_dev(mb, Precision::FP32, K, M);
            memcpy(A.data, a.data(), M*K*sizeof(float));
            Tensor C = make_dev(mb, Precision::FP32, N, M);

            Tensor B = Tensor::create(Precision::INT8, MemoryType::EXTERNAL, N, K, 1, 1, nullptr);
            B.data = buf.data() + woff;
            B.scales = (const float*)(buf.data() + soff);
            B.group_size = (uint32_t)K;          // per-channel
            B.groups_per_row = 1;
            B.num_groups = (uint32_t)N;
            mb.wrap_weight(B);

            metal_matmul(mb, A, B, C);

            // Reference: quantize A per token, int8 matmul, dequant.
            std::vector<float> ref(M*N);
            for (int m = 0; m < M; m++) {
                float mx = 0; for (int k = 0; k < K; k++) mx = std::max(mx, std::fabs(a[k+m*K]));
                float sa = mx/127.0f, inv = mx>0 ? 127.0f/mx : 0;
                for (int n = 0; n < N; n++) {
                    long acc = 0;
                    for (int k = 0; k < K; k++) {
                        int qa = std::max(-127, std::min(127, (int)std::lround(a[k+m*K]*inv)));
                        acc += (long)qa * (long)wi[n*K+k];
                    }
                    ref[m*N+n] = (float)acc * sa * sw[n];
                }
            }
            char label[64];
            snprintf(label, sizeof(label), "W8A8 GEMM M=%d K=%d N=%d", M, K, N);
            CHECK(close((const float*)C.data, ref.data(), M*N, 2e-3f, 2e-2f), label);
        }
        unsetenv("MOLLM_METAL_W8A8");
    }

    // ---- W4A8 prefill GEMM (int8 activations x per-group int4 weights) ------
    // Weights ship in the CPU Q4B8G128Block layout; the Metal backend decodes
    // them to raw nibbles at wrap_weight time. Build a matching packed blob in a
    // registered weight region so wrap_weight's decode path is exercised.
    {
        struct alignas(16) Q4B8G128Block { float scales[8]; uint8_t q[4][8][16]; };
        for (int ci = 0; ci < 2 && mb.has_tensor_path(); ci++) {
            int M = ci==0 ? 40 : 128;
            int K = ci==0 ? 256 : 512;      // multiple of 128
            int N = ci==0 ? 48  : 80;
            int GS = 128, GPR = K / GS;

            std::vector<float> a(M*K); fill_rand(a.data(), M*K);
            // reference int4 weights (signed [-8,7]) + per-group scales.
            std::vector<int8_t> wq(N*K);
            std::vector<float>  sw(N*GPR);
            for (int n = 0; n < N; n++) {
                for (int g = 0; g < GPR; g++) sw[n*GPR+g] = 0.0006f + 0.00002f*((n+g)%11);
                for (int k = 0; k < K; k++) { int v=(rand()%16)-8; if(v==8)v=7; wq[n*K+k]=(int8_t)v; }
            }
            // pack into Q4B8G128Block[ (N/8 padded) x GPR ]: block(nt,g).q[qgi][c]
            // = channel (nt*8+c)'s 16 bytes for K-sub qgi (raw k=g*128+qgi*32+2b lo/+1 hi).
            int Np = ((N + 7)/8)*8;
            std::vector<Q4B8G128Block> blocks((size_t)(Np/8)*GPR);
            std::memset(blocks.data(), 0, blocks.size()*sizeof(Q4B8G128Block));
            for (int nt = 0; nt < Np/8; nt++)
              for (int g = 0; g < GPR; g++) {
                Q4B8G128Block& blk = blocks[(size_t)nt*GPR + g];
                for (int c = 0; c < 8; c++) {
                    int n = nt*8 + c;
                    blk.scales[c] = (n < N) ? sw[n*GPR+g] : 0.f;
                    if (n >= N) continue;
                    for (int qgi = 0; qgi < 4; qgi++)
                      for (int b = 0; b < 16; b++) {
                        int k = g*128 + qgi*32 + 2*b;
                        int lo = wq[n*K+k]   & 0x0F;
                        int hi = wq[n*K+k+1] & 0x0F;
                        blk.q[qgi][c][b] = (uint8_t)((hi<<4)|lo);
                      }
                }
              }
            size_t region = blocks.size()*sizeof(Q4B8G128Block);
            mb.register_weight_region(blocks.data(), region);

            Tensor A = make_dev(mb, Precision::FP32, K, M);
            memcpy(A.data, a.data(), M*K*sizeof(float));
            Tensor C = make_dev(mb, Precision::FP32, N, M);

            Tensor B = Tensor::create(Precision::INT4, MemoryType::EXTERNAL, N, K, 1, 1, nullptr);
            B.data = blocks.data();
            B.group_size = (uint32_t)GS;
            B.groups_per_row = (uint32_t)GPR;
            B.num_groups = (uint32_t)(N*GPR);
            B.is_q4_g128_packed = true;
            B.q4_g128_data = blocks.data();
            B.scales = sw.data();          // placeholder; decode rebuilds from blocks
            mb.wrap_weight(B);
            mb.wrap_weight_int4_g128(B);   // decode packed blocks -> raw nibbles

            metal_matmul(mb, A, B, C);

            // reference: per-token int8 quant of A, per-group int4 dot, dequant.
            std::vector<float> ref(M*N);
            for (int m = 0; m < M; m++) {
                float mx=0; for (int k=0;k<K;k++) mx=std::max(mx,std::fabs(a[k+m*K]));
                float sa=mx/127.0f, inv=mx>0?127.0f/mx:0;
                for (int n = 0; n < N; n++) {
                    double acc=0;
                    for (int g=0; g<GPR; g++){ long d=0;
                        for (int k=g*GS;k<(g+1)*GS;k++){
                            int qa=std::max(-127,std::min(127,(int)std::lround(a[k+m*K]*inv)));
                            d += (long)qa * (long)wq[n*K+k]; }
                        acc += (double)d * sw[n*GPR+g]; }
                    ref[m*N+n]=(float)(acc*sa);
                }
            }
            char label[64];
            snprintf(label, sizeof(label), "W4A8 GEMM M=%d K=%d N=%d", M, K, N);
            CHECK(close((const float*)C.data, ref.data(), M*N, 2e-3f, 3e-2f), label);
        }
    }

    // ---- MATMUL from PERMUTE -> non-zero dim-0 SLICE view ------------------
    // Mirrors MLA's v path: [qk_nope+v, heads, seq] is permuted to
    // [qk_nope+v, seq, heads], then the second dim-0 slab is consumed as a
    // matrix.  K remains contiguous, but A has both a non-zero device offset
    // and a row stride inherited from the wider parent.
    {
        int fullK=24, offset=16, K=8, M=7, N=12;
        Tensor SRC = make_dev3(mb, Precision::FP32, fullK, 1, M);
        Tensor P   = make_dev3(mb, Precision::FP32, fullK, M, 1);
        Tensor A   = make_dev(mb, Precision::FP32, K, M);
        Tensor B   = make_dev(mb, Precision::FP16, N, K);
        Tensor C   = make_dev(mb, Precision::FP32, N, M);
        std::vector<float> src(fullK*M), bf(N*K);
        fill_rand(src.data(), src.size()); fill_rand(bf.data(), bf.size());
        memcpy(SRC.data, src.data(), src.size()*sizeof(float));
        __fp16* bh = (__fp16*)B.data;
        for (int i=0;i<N*K;i++) bh[i] = (__fp16)bf[i];

        metal_op(mb, OpType::PERMUTE, {&SRC}, P, {0,2,1,3});
        metal_op(mb, OpType::SLICE, {&P}, A, {0,offset,K});
        metal_matmul(mb, A, B, C);

        std::vector<float> ref(M*N);
        for (int m=0;m<M;m++) for (int n=0;n<N;n++) {
            double sum=0;
            for (int k=0;k<K;k++)
                sum += (double)src[offset+k + m*fullK] * (double)(float)bh[n*K+k];
            ref[m*N+n] = (float)sum;
        }
        CHECK(close((const float*)C.data, ref.data(), M*N, 1e-3f, 2e-2f),
              "MATMUL A=PERMUTE->SLICE(dim0, offset>0)");
    }

    // ---- TILE dim-2 broadcast (MLA k_rope -> heads) ------------------------
    {
        int s0 = 16, s1 = 5, reps = 4;   // [rope_dim, seq, 1] -> [.., .., heads]
        Tensor IN  = make_dev3(mb, Precision::FP32, s0, s1, 1);
        Tensor OUT = make_dev3(mb, Precision::FP32, s0, s1, reps);
        std::vector<float> in(s0*s1); fill_rand(in.data(), s0*s1);
        memcpy(IN.data, in.data(), s0*s1*sizeof(float));
        metal_op(mb, OpType::TILE, {&IN}, OUT, {1,1,reps,1});
        std::vector<float> ref(s0*s1*reps);
        for (int r=0;r<reps;r++) for (int i1=0;i1<s1;i1++) for (int i0=0;i0<s0;i0++)
            ref[i0 + i1*s0 + r*s0*s1] = in[i0 + i1*s0];
        CHECK(close((const float*)OUT.data, ref.data(), s0*s1*reps, 1e-6f, 1e-6f),
              "TILE dim2 s0=16 s1=5 reps=4");
    }

    // ---- CONCAT dim-0 of two dense 3D tensors ------------------------------
    {
        int a0=6, b0=10, s1=5, s2=3;     // [feat, seq, heads]
        Tensor A = make_dev3(mb, Precision::FP32, a0, s1, s2);
        Tensor B = make_dev3(mb, Precision::FP32, b0, s1, s2);
        Tensor OUT = make_dev3(mb, Precision::FP32, a0+b0, s1, s2);
        std::vector<float> a(a0*s1*s2), b(b0*s1*s2);
        fill_rand(a.data(), a0*s1*s2); fill_rand(b.data(), b0*s1*s2);
        memcpy(A.data, a.data(), a.size()*sizeof(float));
        memcpy(B.data, b.data(), b.size()*sizeof(float));
        metal_op(mb, OpType::CONCAT, {&A,&B}, OUT, {0});
        int O0=a0+b0;
        std::vector<float> ref(O0*s1*s2);
        for (int i2=0;i2<s2;i2++) for (int i1=0;i1<s1;i1++) {
            for (int i0=0;i0<a0;i0++) ref[i0 + i1*O0 + i2*O0*s1] = a[i0 + i1*a0 + i2*a0*s1];
            for (int i0=0;i0<b0;i0++) ref[(a0+i0) + i1*O0 + i2*O0*s1] = b[i0 + i1*b0 + i2*b0*s1];
        }
        CHECK(close((const float*)OUT.data, ref.data(), O0*s1*s2, 1e-6f, 1e-6f),
              "CONCAT dim0 [6|10] s1=5 s2=3 dense");
    }

    // ---- CONCAT dim-0 with a STRIDED (sliced) input ------------------------
    // Mirrors MLA: one operand is a dim-0 slice view of a larger tensor.
    {
        int full0=12, keep=6, b0=6, s1=4, s2=3;
        Tensor FULL = make_dev3(mb, Precision::FP32, full0, s1, s2);  // parent
        Tensor B    = make_dev3(mb, Precision::FP32, b0, s1, s2);
        Tensor OUT  = make_dev3(mb, Precision::FP32, keep+b0, s1, s2);
        std::vector<float> full(full0*s1*s2), b(b0*s1*s2);
        fill_rand(full.data(), full.size()); fill_rand(b.data(), b.size());
        memcpy(FULL.data, full.data(), full.size()*sizeof(float));
        memcpy(B.data, b.data(), b.size()*sizeof(float));
        // A = FULL[0:keep] along dim0 as a view (stride preserved = full0 rows).
        Tensor A = FULL;
        A.shape[0] = keep;   // device_offset 0, strides unchanged (stride[1]=full0)
        metal_op(mb, OpType::CONCAT, {&A,&B}, OUT, {0});
        int O0=keep+b0;
        std::vector<float> ref(O0*s1*s2);
        for (int i2=0;i2<s2;i2++) for (int i1=0;i1<s1;i1++) {
            for (int i0=0;i0<keep;i0++) ref[i0 + i1*O0 + i2*O0*s1] = full[i0 + i1*full0 + i2*full0*s1];
            for (int i0=0;i0<b0;i0++)   ref[(keep+i0) + i1*O0 + i2*O0*s1] = b[i0 + i1*b0 + i2*b0*s1];
        }
        CHECK(close((const float*)OUT.data, ref.data(), O0*s1*s2, 1e-6f, 1e-6f),
              "CONCAT dim0 strided(A=slice) [6|6] s1=4 s2=3");
    }

    if (failures == 0) printf("All Metal op parity tests passed.\n");
    return failures ? 1 : 0;
}
