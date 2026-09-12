// Per-op parity tests for the Metal backend vs the CPU reference kernels.
// Only built when MOLLM_METAL is defined.

#include "kernels/tensor.h"
#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/moe/moe.h"
#include "kernels/cpu/models/rwkv.h"
#include "backends/metal/backend.h"
#include "core/quant_layouts.h"
#include "graph/graph.h"
#include <algorithm>
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
        return 77;  // Report unavailable hardware as skipped, not passed.
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

    // A hybrid operator can close the current command buffer for a CPU read.
    // The next Metal dispatch in the same graph must resume encoding lazily.
    {
        constexpr int N = 64;
        Tensor a = make_dev(mb, Precision::FP32, N, 1);
        Tensor b = make_dev(mb, Precision::FP32, N, 1);
        Tensor first = make_dev(mb, Precision::FP32, N, 1);
        Tensor second = make_dev(mb, Precision::FP32, N, 1);
        auto* ap = static_cast<float*>(a.data);
        auto* bp = static_cast<float*>(b.data);
        for (int i = 0; i < N; ++i) {
            ap[i] = 0.25f * i;
            bp[i] = 1.0f - 0.125f * i;
        }
        GraphNode add;
        add.op_type = OpType::ADD;
        mb.begin_graph();
        mb.dispatch(add, {&a, &b}, &first, nullptr);
        mb.synchronize_for_host_read();
        mb.dispatch(add, {&first, &b}, &second, nullptr);
        mb.end_graph();
        std::vector<float> ref(N);
        for (int i = 0; i < N; ++i)
            ref[i] = ap[i] + 2.0f * bp[i];
        CHECK(close(static_cast<const float*>(second.data), ref.data(), N,
                    1e-6f, 1e-6f),
              "Metal dispatch resumes after host-read synchronization");
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

    // ---- GEMM fused ReLU-squared (RWKV channel-mix) ------------------------
    {
        int M = 3, K = 64, N = 31;
        Tensor A = make_dev(mb, Precision::FP32, K, M);
        Tensor B = make_dev(mb, Precision::FP16, N, K);
        Tensor C = make_dev(mb, Precision::FP32, N, M);
        std::vector<float> a(M * K), bf(N * K), ref(M * N);
        fill_rand(a.data(), (int)a.size());
        fill_rand(bf.data(), (int)bf.size());
        memcpy(A.data, a.data(), a.size() * sizeof(float));
        __fp16* bh = (__fp16*)B.data;
        for (int i = 0; i < N * K; ++i) bh[i] = (__fp16)bf[i];
        metal_matmul(mb, A, B, C, {4, 0, -1});
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double sum = 0.0;
                for (int k = 0; k < K; ++k)
                    sum += a[m * K + k] * (float)bh[n * K + k];
                float y = std::max(0.0f, (float)sum);
                ref[m * N + n] = y * y;
            }
        CHECK(close((const float*)C.data, ref.data(), M * N, 1e-3f, 2e-2f),
              "GEMM fused ReLU-squared");
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

    // ---- ADD/MUL full broadcasting: [D,S,H] with [D,1,H] ------------------
    for (OpType op : {OpType::ADD, OpType::MUL}) {
        int D = 17, S = 5, H = 3;
        Tensor A = make_dev3(mb, Precision::FP32, D, S, H);
        Tensor B = make_dev3(mb, Precision::FP32, D, 1, H);
        Tensor O = make_dev3(mb, Precision::FP32, D, S, H);
        std::vector<float> a(D * S * H), b(D * H), ref(D * S * H);
        fill_rand(a.data(), (int)a.size());
        fill_rand(b.data(), (int)b.size());
        memcpy(A.data, a.data(), a.size() * sizeof(float));
        memcpy(B.data, b.data(), b.size() * sizeof(float));
        for (int h = 0; h < H; ++h)
            for (int s = 0; s < S; ++s)
                for (int d = 0; d < D; ++d) {
                    int oi = (h * S + s) * D + d;
                    float bv = b[h * D + d];
                    ref[oi] = op == OpType::ADD ? a[oi] + bv : a[oi] * bv;
                }
        metal_op(mb, op, {&A, &B}, O);
        char label[80];
        snprintf(label, sizeof(label), "%s broadcast [17,5,3] with [17,1,3]",
                 op_type_name(op));
        CHECK(close((const float*)O.data, ref.data(), D * S * H,
                    1e-6f, 1e-6f), label);
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
    for (OpType op : {OpType::GELU, OpType::TANH, OpType::EXP, OpType::EXP_EXACT,
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
            if (op == OpType::GELU) {
                float inner =
                    0.7978845608f * (v + 0.044715f * v * v * v);
                ref[i] = 0.5f * v * (1.0f + std::tanh(inner));
            } else if (op == OpType::TANH)
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
    auto run_shortconv = [&](int groups, int seq, const char* label,
                             int input_row_padding = 0) {
        const int ks = 4, pre = ks - 1;
        const int input_row_stride = groups + input_row_padding;
        Tensor Xparent =
            make_dev(mb, Precision::FP32, input_row_stride, seq);
        Tensor X = Xparent.view_2d(groups, seq);
        Tensor W = make_dev(mb, Precision::FP32, groups, ks);
        Tensor ST = make_dev(mb, Precision::FP32, groups, pre);  // persistent state
        Tensor O = make_dev(mb, Precision::FP32, groups, seq);
        std::vector<float> x(groups*seq), xp(input_row_stride*seq, 123.f);
        std::vector<float> w(groups*ks), st(groups*pre);
        fill_rand(x.data(), x.size()); fill_rand(w.data(), w.size()); fill_rand(st.data(), st.size());
        for (int s=0;s<seq;s++)
            memcpy(xp.data() + s*input_row_stride, x.data() + s*groups,
                   groups*sizeof(float));
        memcpy(Xparent.data, xp.data(), xp.size()*sizeof(float));
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
    run_shortconv(48, 8, "SHORTCONV strided prefill input", 7);
    run_shortconv(6144, 4, "SHORTCONV groups=6144 seq=4 (prod width)");

    // ---- GATED_DELTANET_DECODE (seq=1): GDN recurrence + RMSNormGated ----
    // Scalar reference mirrors kernels/cpu/arm/gdn_neon.h gdn_recurrence.
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
        for (bool sigmoid_gate : {false, true}) {
            memcpy(QKV.data,qkv.data(),qkv.size()*4); memcpy(A.data,a.data(),H*4); memcpy(B.data,b.data(),H*4);
            memcpy(Z.data,z.data(),zdim*4); memcpy(ALG.data,alg.data(),VH*4); memcpy(DTB.data,dtb.data(),VH*4);
            memcpy(NRM.data,nrm.data(),Vd*4); memcpy(ST.data,st.data(),st.size()*4);
            std::vector<int> i32 = {H, K, Vd, 1,
                                    1 | (sigmoid_gate ? 2 : 0), 4, 0, VH};
            std::vector<float> f32 = {rmseps, l2eps, scale};
            std::vector<const Tensor*> ins = {&QKV,&A,&B,&Z,&ALG,&DTB,&NRM,&ST};
            metal_op(mb, OpType::GATED_DELTANET_DECODE, ins, O, i32, f32);

            // reference
            std::vector<float> ref(zdim), rst = st;
            auto sigmoid=[](float x){return 1.f/(1.f+std::exp(-x));};
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
                for(int dvv=0;dvv<Vd;dvv++) {
                    const float gate = sigmoid(z[vh*Vd+dvv]);
                    const float gated = sigmoid_gate
                        ? gate : z[vh*Vd+dvv] * gate;
                    ref[vh*Vd+dvv] =
                        attn[dvv] * rms * nrm[dvv] * gated;
                }
            }
            bool ok_out = close((const float*)O.data, ref.data(), zdim, 3e-3f, 3e-3f);
            bool ok_st  = close((const float*)ST.data, rst.data(), (int)st.size(), 3e-3f, 3e-3f);
            CHECK(ok_out && ok_st,
                  sigmoid_gate ? "GDN_DECODE sigmoid gate H=4 K=128 V=128"
                               : "GDN_DECODE SiLU gate H=4 K=128 V=128");
        }
    }

    // ---- Decode ShortConv + GDN fusion: compare against the two-op path ----
    // Cover both Qwen3.5 head layouts: one and two value heads per key head.
    for (int repeat : {1, 2}) {
        const int H = 2, VH = H*repeat, K = 128, Vd = 128, ks = 4;
        const int qkv_dim = H*K;
        const int qkv_total = 2*qkv_dim + VH*Vd;
        const int zdim = VH*Vd;
        const float scale = 1.0f/std::sqrt((float)K);

        Tensor QKV = make_dev(mb, Precision::FP32, qkv_total, 1);
        Tensor CW = make_dev(mb, Precision::FP32, qkv_total, ks);
        Tensor CSref = make_dev(mb, Precision::FP32, qkv_total, ks-1);
        Tensor CSfused = make_dev(mb, Precision::FP32, qkv_total, ks-1);
        Tensor QKVC = make_dev(mb, Precision::FP32, qkv_total, 1);
        Tensor A = make_dev(mb, Precision::FP32, VH, 1);
        Tensor B = make_dev(mb, Precision::FP32, VH, 1);
        Tensor Z = make_dev(mb, Precision::FP32, zdim, 1);
        Tensor ALG = make_dev(mb, Precision::FP32, VH, 1);
        Tensor DTB = make_dev(mb, Precision::FP32, VH, 1);
        Tensor NRM = make_dev(mb, Precision::FP32, Vd, 1);
        Tensor STref = make_dev(mb, Precision::FP32, VH*K*Vd, 1);
        Tensor STfused = make_dev(mb, Precision::FP32, VH*K*Vd, 1);
        Tensor Oref = make_dev(mb, Precision::FP32, zdim, 1);
        Tensor Ofused = make_dev(mb, Precision::FP32, zdim, 1);

        std::vector<float> qkv(qkv_total), cw((size_t)qkv_total*ks);
        std::vector<float> cs((size_t)qkv_total*(ks-1));
        std::vector<float> a(VH), b(VH), z(zdim), alg(VH), dtb(VH),
                           nrm(Vd), st((size_t)VH*K*Vd);
        fill_rand(qkv.data(), qkv.size()); fill_rand(cw.data(), cw.size());
        fill_rand(cs.data(), cs.size()); fill_rand(a.data(), a.size());
        fill_rand(b.data(), b.size()); fill_rand(z.data(), z.size());
        fill_rand(alg.data(), alg.size()); fill_rand(dtb.data(), dtb.size());
        fill_rand(nrm.data(), nrm.size()); fill_rand(st.data(), st.size());
        memcpy(QKV.data,qkv.data(),qkv.size()*4);
        memcpy(CW.data,cw.data(),cw.size()*4);
        memcpy(CSref.data,cs.data(),cs.size()*4);
        memcpy(CSfused.data,cs.data(),cs.size()*4);
        memcpy(A.data,a.data(),a.size()*4); memcpy(B.data,b.data(),b.size()*4);
        memcpy(Z.data,z.data(),z.size()*4); memcpy(ALG.data,alg.data(),alg.size()*4);
        memcpy(DTB.data,dtb.data(),dtb.size()*4); memcpy(NRM.data,nrm.data(),nrm.size()*4);
        memcpy(STref.data,st.data(),st.size()*4);
        memcpy(STfused.data,st.data(),st.size()*4);

        metal_op(mb, OpType::SHORTCONV, {&QKV,&CW,&CSref}, QKVC,
                 {ks, 1});
        std::vector<int> i32 = {H, K, Vd, 1, 1, ks, 0, VH};
        std::vector<float> f32 = {1e-6f, 1e-6f, scale};
        metal_op(mb, OpType::GATED_DELTANET_DECODE,
                 {&QKVC,&A,&B,&Z,&ALG,&DTB,&NRM,&STref},
                 Oref, i32, f32);
        metal_op(mb, OpType::GATED_DELTANET_CONV_DECODE,
                 {&QKV,&A,&B,&Z,&ALG,&DTB,&NRM,&STfused,&CW,&CSfused},
                 Ofused, i32, f32);

        char label[96];
        snprintf(label, sizeof(label),
                 "GDN_CONV_DECODE repeat=%d output/state parity", repeat);
        const bool ok_out =
            close((const float*)Ofused.data, (const float*)Oref.data,
                  zdim, 4e-3f, 4e-3f);
        const bool ok_gdn =
            close((const float*)STfused.data, (const float*)STref.data,
                  (int)st.size(), 4e-3f, 4e-3f);
        const bool ok_conv =
            close((const float*)CSfused.data, (const float*)CSref.data,
                  (int)cs.size(), 1e-6f, 1e-6f);
        CHECK(ok_out && ok_gdn && ok_conv, label);
    }

    // ---- GATED_DELTANET_PREFILL (seq>1): recurrence over tokens ----
    // Layout: qkv [qkv_total, seq] (dim-major); a/b/z/out [seq, dim] (seq-major).
    {
        int H = 4, VH = 4, K = 128, Vd = 128, S = 6;
        float l2eps = 1e-6f, rmseps = 1e-6f, scale = 1.0f/std::sqrt((float)K);
        int qkv_dim = H*K, qkv_total = 2*H*K + VH*Vd, zdim = VH*Vd;
        Tensor QKV = make_dev(mb, Precision::FP32, qkv_total, S);   // [qkv_total, seq]
        // A/B are views into one merged [2*VH, S] projection. Their logical
        // row width is VH while stride[1] remains 2*VH, matching converter
        // output after a dim-0 slice.
        Tensor AB = make_dev(mb, Precision::FP32, 2*VH, S);
        Tensor A = AB;
        Tensor B = AB;
        A.shape[0] = VH;
        B.shape[0] = VH;
        B.device_offset += (size_t)VH * sizeof(float);
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
        std::vector<float> ab((size_t)2*VH*S);
        for (int t=0;t<S;t++) {
            memcpy(ab.data() + (size_t)t*2*VH,
                   a.data() + (size_t)t*VH, (size_t)VH*4);
            memcpy(ab.data() + (size_t)t*2*VH + VH,
                   b.data() + (size_t)t*VH, (size_t)VH*4);
        }
        memcpy(QKV.data,qkv.data(),qkv.size()*4);
        memcpy(AB.data,ab.data(),ab.size()*4); memcpy(Z.data,z.data(),z.size()*4);
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
        CHECK(ok_out && ok_st,
              "GDN_PREFILL H=4 K=128 V=128 S=6 strided A/B");
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

    // ---- SIGMOID_MUL: value * sigmoid(gate) -------------------------------
    {
        int D = 96, rows = 5;
        Tensor V = make_dev(mb, Precision::FP32, D, rows);
        Tensor G = make_dev(mb, Precision::FP32, D, rows);
        Tensor O = make_dev(mb, Precision::FP32, D, rows);
        std::vector<float> value((size_t)D*rows), gate((size_t)D*rows);
        std::vector<float> ref((size_t)D*rows);
        fill_rand(value.data(), (int)value.size());
        fill_rand(gate.data(), (int)gate.size());
        memcpy(V.data, value.data(), value.size()*4);
        memcpy(G.data, gate.data(), gate.size()*4);
        for (size_t i=0;i<ref.size();++i)
            ref[i] = value[i] / (1.0f + std::exp(-gate[i]));
        metal_op(mb, OpType::SIGMOID_MUL, {&V,&G}, O);
        CHECK(close((const float*)O.data, ref.data(), (int)ref.size(),
                    1e-5f, 1e-4f),
              "SIGMOID_MUL D=96 rows=5");
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

    // ---- RMS_NORM_ROPE: strided Q/K view -> dense [D,S,H] ----
    {
        const int D=256, S=5, H=3, rows=S*H, rope=64, pad=7;
        Tensor XP = make_dev(mb, Precision::FP32, D+pad, rows);
        Tensor X = XP.view_2d(D, rows);
        Tensor W = make_dev(mb, Precision::FP32, D, 1);
        Tensor COS = make_dev(mb, Precision::FP32, rope/2, S);
        Tensor SIN = make_dev(mb, Precision::FP32, rope/2, S);
        Tensor O = make_dev3(mb, Precision::FP32, D, S, H);
        std::vector<float> xp((size_t)(D+pad)*rows, 123.f);
        std::vector<float> x((size_t)D*rows), w(D);
        std::vector<float> cs((size_t)(rope/2)*S);
        std::vector<float> sn((size_t)(rope/2)*S);
        fill_rand(x.data(),x.size()); fill_rand(w.data(),w.size());
        fill_rand(cs.data(),cs.size()); fill_rand(sn.data(),sn.size());
        for(int r=0;r<rows;r++)
            memcpy(xp.data()+(size_t)r*(D+pad),
                   x.data()+(size_t)r*D, (size_t)D*4);
        memcpy(XP.data,xp.data(),xp.size()*4);
        memcpy(W.data,w.data(),w.size()*4);
        memcpy(COS.data,cs.data(),cs.size()*4);
        memcpy(SIN.data,sn.data(),sn.size()*4);
        metal_op(mb, OpType::RMS_NORM_ROPE, {&X,&W,&COS,&SIN}, O,
                 {rope,0}, {1e-6f});

        std::vector<float> ref((size_t)D*rows);
        for(int r=0;r<rows;r++){
            double ss=0;
            for(int d=0;d<D;d++)
                ss+=(double)x[(size_t)r*D+d]*x[(size_t)r*D+d];
            float scale=1.f/std::sqrt((float)(ss/D)+1e-6f);
            int pos=r%S;
            for(int i=0;i<rope/2;i++){
                float x0=x[(size_t)r*D+i]*scale*w[i];
                float x1=x[(size_t)r*D+i+rope/2]*scale*w[i+rope/2];
                float c=cs[(size_t)pos*(rope/2)+i];
                float s=sn[(size_t)pos*(rope/2)+i];
                ref[(size_t)r*D+i]=x0*c-x1*s;
                ref[(size_t)r*D+i+rope/2]=x1*c+x0*s;
            }
            for(int d=rope;d<D;d++)
                ref[(size_t)r*D+d]=x[(size_t)r*D+d]*scale*w[d];
        }
        CHECK(close((const float*)O.data,ref.data(),D*rows,4e-4f,4e-4f),
              "RMS_NORM_ROPE strided D=256 S=5 H=3");
    }

    // ---- QK_RMS_NORM_ROPE: Q and K rows share one dispatch --------------
    {
        const int D=128, S=5, QH=3, KH=2, rope=64;
        const int qrows=S*QH, krows=S*KH;
        Tensor QP = make_dev(mb, Precision::FP32, D+5, qrows);
        Tensor KP = make_dev(mb, Precision::FP32, D+7, krows);
        Tensor Q = QP.view_2d(D, qrows);
        Tensor K = KP.view_2d(D, krows);
        Tensor QW = make_dev(mb, Precision::FP32, D, 1);
        Tensor KW = make_dev(mb, Precision::FP32, D, 1);
        Tensor COS = make_dev(mb, Precision::FP32, rope/2, S);
        Tensor SIN = make_dev(mb, Precision::FP32, rope/2, S);
        Tensor O = make_dev3(mb, Precision::FP32, D, S, QH+KH);
        std::vector<float> qp((size_t)(D+5)*qrows, 123.f);
        std::vector<float> kp((size_t)(D+7)*krows, 123.f);
        std::vector<float> q((size_t)D*qrows), k((size_t)D*krows);
        std::vector<float> qw(D), kw(D);
        std::vector<float> cs((size_t)(rope/2)*S);
        std::vector<float> sn((size_t)(rope/2)*S);
        fill_rand(q.data(),q.size()); fill_rand(k.data(),k.size());
        fill_rand(qw.data(),qw.size()); fill_rand(kw.data(),kw.size());
        fill_rand(cs.data(),cs.size()); fill_rand(sn.data(),sn.size());
        for(int r=0;r<qrows;r++)
            memcpy(qp.data()+(size_t)r*(D+5),
                   q.data()+(size_t)r*D, (size_t)D*4);
        for(int r=0;r<krows;r++)
            memcpy(kp.data()+(size_t)r*(D+7),
                   k.data()+(size_t)r*D, (size_t)D*4);
        memcpy(QP.data,qp.data(),qp.size()*4);
        memcpy(KP.data,kp.data(),kp.size()*4);
        memcpy(QW.data,qw.data(),qw.size()*4);
        memcpy(KW.data,kw.data(),kw.size()*4);
        memcpy(COS.data,cs.data(),cs.size()*4);
        memcpy(SIN.data,sn.data(),sn.size()*4);
        metal_op(mb, OpType::QK_RMS_NORM_ROPE,
                 {&Q,&K,&QW,&KW,&COS,&SIN}, O,
                 {rope,0,QH}, {1e-6f});

        std::vector<float> ref((size_t)D*(qrows+krows));
        auto reference = [&](const std::vector<float>& x,
                             const std::vector<float>& w,
                             int rows, int row_offset) {
            for(int r=0;r<rows;r++){
                double ss=0;
                for(int d=0;d<D;d++)
                    ss+=(double)x[(size_t)r*D+d]*x[(size_t)r*D+d];
                float scale=1.f/std::sqrt((float)(ss/D)+1e-6f);
                int pos=r%S;
                for(int i=0;i<rope/2;i++){
                    float x0=x[(size_t)r*D+i]*scale*w[i];
                    float x1=x[(size_t)r*D+i+rope/2]*scale*w[i+rope/2];
                    float c=cs[(size_t)pos*(rope/2)+i];
                    float s=sn[(size_t)pos*(rope/2)+i];
                    ref[(size_t)(row_offset+r)*D+i]=x0*c-x1*s;
                    ref[(size_t)(row_offset+r)*D+i+rope/2]=x1*c+x0*s;
                }
                for(int d=rope;d<D;d++)
                    ref[(size_t)(row_offset+r)*D+d]=
                        x[(size_t)r*D+d]*scale*w[d];
            }
        };
        reference(q, qw, qrows, 0);
        reference(k, kw, krows, qrows);
        CHECK(close((const float*)O.data,ref.data(),D*(qrows+krows),
                    4e-4f,4e-4f),
              "QK_RMS_NORM_ROPE strided Q/K");
    }

    // ---- ADD_RMS_NORM: update residual in place + normalized output -------
    {
        int D = 128, rows = 5;
        Tensor R = make_dev(mb, Precision::FP32, D, rows);
        Tensor U = make_dev(mb, Precision::FP32, D, rows);
        Tensor W = make_dev(mb, Precision::FP32, D, 1);
        Tensor O = make_dev(mb, Precision::FP32, D, rows);
        std::vector<float> residual((size_t)D*rows);
        std::vector<float> update((size_t)D*rows);
        std::vector<float> weight(D);
        std::vector<float> expected_residual((size_t)D*rows);
        std::vector<float> expected_out((size_t)D*rows);
        fill_rand(residual.data(), (int)residual.size());
        fill_rand(update.data(), (int)update.size());
        fill_rand(weight.data(), D);
        for (size_t i=0;i<residual.size();++i)
            expected_residual[i] = residual[i] + update[i];
        for (int row=0;row<rows;++row) {
            double ss = 0.0;
            for (int d=0;d<D;++d) {
                const float v = expected_residual[(size_t)row*D+d];
                ss += (double)v*v;
            }
            const float scale =
                1.0f/std::sqrt((float)(ss/D)+1e-6f);
            for (int d=0;d<D;++d)
                expected_out[(size_t)row*D+d] =
                    expected_residual[(size_t)row*D+d]*scale*weight[d];
        }
        memcpy(R.data, residual.data(), residual.size()*4);
        memcpy(U.data, update.data(), update.size()*4);
        memcpy(W.data, weight.data(), weight.size()*4);
        metal_op(mb, OpType::ADD_RMS_NORM, {&R,&U,&W}, O, {}, {1e-6f});
        CHECK(close((const float*)R.data, expected_residual.data(),
                    (int)residual.size(), 1e-6f, 1e-6f),
              "ADD_RMS_NORM residual update");
        CHECK(close((const float*)O.data, expected_out.data(),
                    (int)expected_out.size(), 2e-4f, 2e-4f),
              "ADD_RMS_NORM normalized output");
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

    // ---- RWKV_TOKEN_SHIFT: FP16 persistent state, real<padded sequence ----
    {
        int hidden = 67, seq = 5, real = 3;
        Tensor X = make_dev(mb, Precision::FP32, hidden, seq);
        Tensor STATE = make_dev(mb, Precision::FP16, hidden, 1);
        Tensor O = make_dev(mb, Precision::FP32, hidden, seq);
        std::vector<float> x(hidden * seq), state(hidden), ref(hidden * seq, 0);
        fill_rand(x.data(), (int)x.size());
        fill_rand(state.data(), hidden);
        memcpy(X.data, x.data(), x.size() * sizeof(float));
        __fp16* state_h = (__fp16*)STATE.data;
        for (int d = 0; d < hidden; ++d) state_h[d] = (__fp16)state[d];
        std::vector<float> rounded_state(hidden);
        for (int d = 0; d < hidden; ++d) rounded_state[d] = (float)state_h[d];
        for (int t = 0; t < real; ++t)
            for (int d = 0; d < hidden; ++d)
                ref[t * hidden + d] =
                    (t == 0 ? rounded_state[d]
                            : (float)(__fp16)x[(t - 1) * hidden + d]) -
                    x[t * hidden + d];
        metal_op(mb, OpType::RWKV_TOKEN_SHIFT, {&X, &STATE}, O,
                 {hidden, seq, real});
        bool state_ok = true;
        for (int d = 0; d < hidden; ++d)
            state_ok &= std::fabs((float)state_h[d] -
                                  (float)(__fp16)x[(real - 1) * hidden + d]) <
                        1e-6f;
        CHECK(close((const float*)O.data, ref.data(), hidden * seq,
                    1e-5f, 1e-5f) && state_ok,
              "RWKV_TOKEN_SHIFT FP16 state hidden=67 real=3 seq=5");
    }

    // ---- RWKV_MIX: per-channel mix broadcast across tokens ----------------
    {
        int hidden = 65, tokens = 4;
        Tensor X = make_dev(mb, Precision::FP32, hidden, tokens);
        Tensor S = make_dev(mb, Precision::FP32, hidden, tokens);
        Tensor M = make_dev(mb, Precision::FP32, hidden, 1);
        Tensor O = make_dev(mb, Precision::FP32, hidden, tokens);
        std::vector<float> x(hidden * tokens), s(hidden * tokens), m(hidden);
        std::vector<float> ref(hidden * tokens);
        fill_rand(x.data(), (int)x.size());
        fill_rand(s.data(), (int)s.size());
        fill_rand(m.data(), hidden);
        memcpy(X.data, x.data(), x.size() * sizeof(float));
        memcpy(S.data, s.data(), s.size() * sizeof(float));
        memcpy(M.data, m.data(), m.size() * sizeof(float));
        for (int t = 0; t < tokens; ++t)
            for (int d = 0; d < hidden; ++d)
                ref[t * hidden + d] =
                    x[t * hidden + d] + s[t * hidden + d] * m[d];
        metal_op(mb, OpType::RWKV_MIX, {&X, &S, &M}, O);
        CHECK(close((const float*)O.data, ref.data(), hidden * tokens,
                    1e-6f, 1e-6f),
              "RWKV_MIX hidden=65 tokens=4");
    }

    // ---- RWKV_L2_NORM: normalize each token/head independently ------------
    {
        int heads = 3, head_size = 64, tokens = 4;
        int hidden = heads * head_size;
        Tensor X = make_dev(mb, Precision::FP32, hidden, tokens);
        Tensor O = make_dev(mb, Precision::FP32, hidden, tokens);
        std::vector<float> x(hidden * tokens), ref(hidden * tokens);
        fill_rand(x.data(), (int)x.size());
        memcpy(X.data, x.data(), x.size() * sizeof(float));
        const float eps = 1e-6f;
        for (int t = 0; t < tokens; ++t)
            for (int h = 0; h < heads; ++h) {
                int base = t * hidden + h * head_size;
                double sum = 0.0;
                for (int d = 0; d < head_size; ++d)
                    sum += (double)x[base + d] * x[base + d];
                float scale = 1.0f / (std::sqrt((float)sum) + eps);
                for (int d = 0; d < head_size; ++d)
                    ref[base + d] = x[base + d] * scale;
            }
        metal_op(mb, OpType::RWKV_L2_NORM, {&X}, O,
                 {heads, head_size}, {eps});
        CHECK(close((const float*)O.data, ref.data(), hidden * tokens,
                    2e-6f, 2e-5f),
              "RWKV_L2_NORM heads=3 head_size=64 tokens=4");
    }

    // ---- RWKV_POST: group norm + receptance/key bonus + gate ---------------
    {
        int heads = 3, head_size = 64, tokens = 2;
        int hidden = heads * head_size, total = hidden * tokens;
        Tensor RAW = make_dev(mb, Precision::FP32, hidden, tokens);
        Tensor R = make_dev(mb, Precision::FP32, hidden, tokens);
        Tensor K = make_dev(mb, Precision::FP32, hidden, tokens);
        Tensor V = make_dev(mb, Precision::FP32, hidden, tokens);
        Tensor RK = make_dev(mb, Precision::FP32, hidden, 1);
        Tensor W = make_dev(mb, Precision::FP32, hidden, 1);
        Tensor B = make_dev(mb, Precision::FP32, hidden, 1);
        Tensor G = make_dev(mb, Precision::FP32, hidden, tokens);
        Tensor O = make_dev(mb, Precision::FP32, hidden, tokens);
        std::vector<float> raw(total), r(total), k(total), v(total), gate(total);
        std::vector<float> rk(hidden), w(hidden), b(hidden), ref(total);
        for (auto* vec : {&raw, &r, &k, &v, &gate, &rk, &w, &b})
            fill_rand(vec->data(), (int)vec->size());
        memcpy(RAW.data, raw.data(), raw.size() * sizeof(float));
        memcpy(R.data, r.data(), r.size() * sizeof(float));
        memcpy(K.data, k.data(), k.size() * sizeof(float));
        memcpy(V.data, v.data(), v.size() * sizeof(float));
        memcpy(RK.data, rk.data(), rk.size() * sizeof(float));
        memcpy(W.data, w.data(), w.size() * sizeof(float));
        memcpy(B.data, b.data(), b.size() * sizeof(float));
        memcpy(G.data, gate.data(), gate.size() * sizeof(float));
        const float eps = 64e-5f;
        for (int t = 0; t < tokens; ++t)
            for (int h = 0; h < heads; ++h) {
                int base = t * hidden + h * head_size;
                int wb = h * head_size;
                double mean = 0.0, bonus = 0.0;
                for (int d = 0; d < head_size; ++d) {
                    mean += raw[base + d];
                    bonus += r[base + d] * k[base + d] * rk[wb + d];
                }
                mean /= head_size;
                double variance = 0.0;
                for (int d = 0; d < head_size; ++d) {
                    double z = raw[base + d] - mean;
                    variance += z * z;
                }
                variance /= head_size;
                float inv = 1.0f / std::sqrt((float)variance + eps);
                for (int d = 0; d < head_size; ++d) {
                    float norm = (raw[base + d] - (float)mean) * inv *
                                     w[wb + d] +
                                 b[wb + d];
                    ref[base + d] =
                        (norm + (float)bonus * v[base + d]) * gate[base + d];
                }
            }
        metal_op(mb, OpType::RWKV_POST,
                 {&RAW, &R, &K, &V, &RK, &W, &B, &G}, O,
                 {heads, head_size}, {eps});
        CHECK(close((const float*)O.data, ref.data(), total, 2e-5f, 3e-4f),
              "RWKV_POST heads=3 head_size=64 tokens=2");
    }

    // ---- RWKV7 recurrence: output and persistent FP32 state parity --------
    {
        int heads = 2, head_size = 64, seq = 4, real = 3;
        int hidden = heads * head_size, total = hidden * seq;
        int state_n = heads * head_size * head_size;
        Tensor R = make_dev(mb, Precision::FP32, hidden, seq);
        Tensor D = make_dev(mb, Precision::FP32, hidden, seq);
        Tensor K = make_dev(mb, Precision::FP32, hidden, seq);
        Tensor V = make_dev(mb, Precision::FP32, hidden, seq);
        Tensor A = make_dev(mb, Precision::FP32, hidden, seq);
        Tensor B = make_dev(mb, Precision::FP32, hidden, seq);
        Tensor STATE = make_dev(mb, Precision::FP32, state_n, 1);
        Tensor O = make_dev(mb, Precision::FP32, hidden, seq);
        std::vector<float> r(total), decay(total), k(total), v(total), a(total),
            b(total), initial_state(state_n), ref_state(state_n), ref(total);
        for (auto* vec : {&r, &k, &v, &a, &b, &initial_state})
            fill_rand(vec->data(), (int)vec->size());
        for (int i = 0; i < total; ++i)
            decay[i] = 0.9f + 0.09f * (float)(i % 11) / 10.0f;
        memcpy(R.data, r.data(), r.size() * sizeof(float));
        memcpy(D.data, decay.data(), decay.size() * sizeof(float));
        memcpy(K.data, k.data(), k.size() * sizeof(float));
        memcpy(V.data, v.data(), v.size() * sizeof(float));
        memcpy(A.data, a.data(), a.size() * sizeof(float));
        memcpy(B.data, b.data(), b.size() * sizeof(float));
        memcpy(STATE.data, initial_state.data(),
               initial_state.size() * sizeof(float));

        ref_state = initial_state;
        Tensor r_h = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    hidden, seq, 1, 1, r.data());
        Tensor d_h = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    hidden, seq, 1, 1, decay.data());
        Tensor k_h = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    hidden, seq, 1, 1, k.data());
        Tensor v_h = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    hidden, seq, 1, 1, v.data());
        Tensor a_h = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    hidden, seq, 1, 1, a.data());
        Tensor b_h = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    hidden, seq, 1, 1, b.data());
        Tensor s_h = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    state_n, 1, 1, 1, ref_state.data());
        Tensor o_h = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    hidden, seq, 1, 1, ref.data());
        Rwkv7Params cpu_params{heads, head_size, seq, real};
        kernel_rwkv7(cpu_params, {&r_h, &d_h, &k_h, &v_h, &a_h, &b_h, &s_h},
                     o_h, nullptr);

        metal_op(mb, OpType::RWKV7, {&R, &D, &K, &V, &A, &B, &STATE}, O,
                 {heads, head_size, seq, real});
        CHECK(close((const float*)O.data, ref.data(), total, 2e-5f, 3e-4f) &&
                  close((const float*)STATE.data, ref_state.data(), state_n,
                        2e-5f, 3e-4f),
              "RWKV7 H64 threadgroup-state prefill parity");
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
    auto run_sdpa_cfg = [&](int S, int past, int hd, int vd, int num_heads, int num_kv,
                            int max_seq, const char* label, float atol = 2e-2f) {
        int hpg = num_heads / num_kv;
        float scale = 1.0f / std::sqrt((float)hd);

        Tensor Q  = make_dev3(mb, Precision::FP32, hd, S, num_heads);
        Tensor Kc = make_dev3(mb, Precision::FP32, hd, S, num_kv);
        Tensor Vc = make_dev3(mb, Precision::FP32, vd, S, num_kv);
        Tensor O  = make_dev3(mb, Precision::FP32, vd, S, num_heads);

        // KV caches: 64B header + FP16 data [num_kv, max_seq, hd].
        size_t k_cache_bytes = 64 + (size_t)num_kv * max_seq * hd * 2;
        size_t v_cache_bytes = 64 + (size_t)num_kv * max_seq * vd * 2;
        Tensor Kcache = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, 1,1,1,1, nullptr);
        Tensor Vcache = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, 1,1,1,1, nullptr);
        mb.alloc_persistent(Kcache, k_cache_bytes);
        mb.alloc_persistent(Vcache, v_cache_bytes);
        // metadata: current_seq_len=past, max_seq_len=max_seq
        auto set_meta = [&](Tensor& c){
            uint64_t* m = (uint64_t*)c.data;
            for (int i=0;i<8;i++) m[i]=0;
            m[0]=(uint64_t)past; m[1]=(uint64_t)max_seq;
        };
        set_meta(Kcache); set_meta(Vcache);

        std::vector<float> q(hd*S*num_heads), kc(hd*S*num_kv), vc(vd*S*num_kv);
        fill_rand(q.data(), q.size()); fill_rand(kc.data(), kc.size()); fill_rand(vc.data(), vc.size());
        memcpy(Q.data, q.data(), q.size()*4);
        memcpy(Kc.data, kc.data(), kc.size()*4);
        memcpy(Vc.data, vc.data(), vc.size()*4);

        // Pre-fill the cache "past" region with FP16 data we control, so the
        // reference and GPU see identical past K/V.
        std::vector<float> pastK((size_t)num_kv*past*hd), pastV((size_t)num_kv*past*vd);
        fill_rand(pastK.data(), pastK.size()); fill_rand(pastV.data(), pastV.size());
        __fp16* Kd = (__fp16*)((char*)Kcache.data + 64);
        __fp16* Vd = (__fp16*)((char*)Vcache.data + 64);
        for (int g=0; g<num_kv; g++)
            for (int j=0;j<past;j++)
                for (int d=0; d<hd; d++)
                    Kd[g*max_seq*hd + j*hd + d] = (__fp16)pastK[g*past*hd + j*hd + d];
        for (int g=0; g<num_kv; g++)
            for (int j=0;j<past;j++)
                for (int d=0; d<vd; d++)
                    Vd[g*max_seq*vd + j*vd + d] = (__fp16)pastV[g*past*vd + j*vd + d];

        std::vector<int> i32 = {2 /*kv_cache*/, 1 /*causal*/, num_heads, num_kv, hd, vd};
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
        std::vector<float> ref(vd*S*num_heads);
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
                for (int d=0; d<vd; d++){
                    double o=0; for(int j=0;j<limit;j++) o+=sc[j]*kv_at(pastV,vc,g,j,d,vd);
                    ref[h*(vd*S)+s*vd+d]=(float)(o/den);
                }
            }
        }
        CHECK(close((const float*)O.data, ref.data(), vd*S*num_heads, atol, 3e-2f), label);
    };
    // hd=32 harness (PV=PAD(32,64)=64) — original coverage.
    auto run_sdpa = [&](int S, int past, const char* label) {
        run_sdpa_cfg(S, past, 32, 32, 4, 2, 256, label);
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

    // Production config: dk=dv=128, GQA 8/2.
    auto run_sdpa128 = [&](int S, int past, const char* label) {
        run_sdpa_cfg(S, past, 128, 128, 8, 2, 512, label);
    };
    run_sdpa128(1, 7, "SDPA decode hd=128 S=1 past=7");
    run_sdpa128(1, 255, "SDPA decode hd=128 S=1 past=255");
    run_sdpa_cfg(1, 511, 128, 128, 8, 2, 1024,
                 "SDPA decode hd=128 S=1 past=511");
    run_sdpa_cfg(1, 1023, 128, 128, 8, 2, 2048,
                 "SDPA decode hd=128 S=1 past=1023", 3e-2f);
    run_sdpa_cfg(1, 3071, 128, 128, 8, 2, 4096,
                 "SDPA decode hd=128 S=1 past=3071", 3e-2f);
    run_sdpa_cfg(1, 7, 256, 256, 8, 2, 512,
                 "SDPA decode hd=256 S=1 past=7");
    run_sdpa_cfg(1, 255, 256, 256, 8, 2, 512,
                 "SDPA decode hd=256 S=1 past=255");
    run_sdpa_cfg(1, 511, 256, 256, 8, 2, 1024,
                 "SDPA decode hd=256 S=1 past=511", 3e-3f);
    run_sdpa_cfg(1, 1023, 256, 256, 8, 2, 1024,
                 "SDPA decode hd=256 S=1 past=1023", 3e-3f);
    run_sdpa128(16, 0, "SDPA prefill hd=128 S=16 past=0");
    run_sdpa128(128, 0, "SDPA prefill hd=128 S=128 past=0");
    run_sdpa128(256, 0, "SDPA prefill hd=128 S=256 past=0");
    run_sdpa128(70, 13, "SDPA prefill hd=128 S=70 past=13"); // ragged Q tile + C cross + past
    run_sdpa_cfg(64, 0, 256, 256, 8, 2, 512,
                 "SDPA prefill hd=256 S=64 past=0");
    run_sdpa_cfg(64, 0, 192, 128, 16, 16, 512,
                 "SDPA prefill dk=192 dv=128 S=64 past=0");
    run_sdpa_cfg(1, 0,   192, 128, 16, 16, 512, "SDPA fused decode dk=192 dv=128 past=0",   3e-3f);
    run_sdpa_cfg(1, 31,  192, 128, 16, 16, 512, "SDPA fused decode dk=192 dv=128 past=31",  3e-3f);
    run_sdpa_cfg(1, 32,  192, 128, 16, 16, 512, "SDPA fused decode dk=192 dv=128 past=32",  3e-3f);
    run_sdpa_cfg(1, 62,  192, 128, 16, 16, 512, "SDPA fused decode dk=192 dv=128 past=62",  3e-3f);
    run_sdpa_cfg(1, 63,  192, 128, 16, 16, 512, "SDPA fused decode dk=192 dv=128 past=63",  3e-3f);
    run_sdpa_cfg(1, 64,  192, 128, 16, 16, 512, "SDPA fused decode dk=192 dv=128 past=64",  3e-3f);
    run_sdpa_cfg(1, 127, 192, 128, 16, 16, 512, "SDPA fused decode dk=192 dv=128 past=127", 3e-3f);
    run_sdpa_cfg(1, 255, 192, 128, 16, 16, 512, "SDPA fused decode dk=192 dv=128 past=255", 3e-3f);
    run_sdpa_cfg(1, 511, 192, 128, 16, 16, 512, "SDPA fused decode dk=192 dv=128 past=511", 3e-3f);
    run_sdpa_cfg(1, 767, 192, 128, 16, 16, 1024, "SDPA multipart decode dk=192 dv=128 past=767", 3e-3f);
    run_sdpa_cfg(1, 1023, 192, 128, 16, 16, 1024, "SDPA multipart decode dk=192 dv=128 past=1023", 3e-3f);

    // ---- Default W8A16 prefill GEMM ----------------------------------------
    // Cover both cooperative output-tile specializations. The second shape
    // deliberately trips the large-projection M128 dispatch rule.
    if (mb.has_tensor_path()) {
        // Initialize the diagnostic W8A8 switch before its first dispatch.
        // These cases use two identical scale groups, which intentionally
        // selects the production W8A16 path even while W8A8 is enabled.
        setenv("MOLLM_METAL_W8A8", "1", 1);
        const int shapes[][3] = {
            {64, 256, 96},
            {129, 8192, 64},
        };
        for (const auto& shape : shapes) {
            const int M = shape[0], K = shape[1], N = shape[2];
            std::vector<float> a(M*K); fill_rand(a.data(), M*K);
            std::vector<int8_t> wi(N*K);
            std::vector<float> sw(2*N);
            for (int n = 0; n < N; ++n) {
                float mx = 0.0f;
                std::vector<float> row(K);
                for (int k = 0; k < K; ++k) {
                    row[k] = static_cast<float>(rand()) /
                             static_cast<float>(RAND_MAX) - 0.5f;
                    mx = std::max(mx, std::fabs(row[k]));
                }
                const float s = mx / 127.0f;
                const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
                sw[2*n] = s;
                sw[2*n+1] = s;
                for (int k = 0; k < K; ++k) {
                    const int q = (int)std::lround(row[k] * inv);
                    wi[n*K+k] =
                        (int8_t)std::max(-127, std::min(127, q));
                }
            }
            const size_t soff = (N*K + 3) & ~size_t(3);
            const size_t region = soff + 2*N*sizeof(float);
            std::vector<uint8_t> buf(region, 0);
            memcpy(buf.data(), wi.data(), N*K);
            memcpy(buf.data()+soff, sw.data(), 2*N*sizeof(float));
            mb.register_weight_region(buf.data(), region);

            Tensor A = make_dev(mb, Precision::FP32, K, M);
            memcpy(A.data, a.data(), M*K*sizeof(float));
            Tensor C = make_dev(mb, Precision::FP32, N, M);
            Tensor B = Tensor::create(
                Precision::INT8, MemoryType::EXTERNAL, N, K, 1, 1, nullptr);
            B.data = buf.data();
            B.scales = (const float*)(buf.data()+soff);
            B.group_size = (uint32_t)(K/2);
            B.groups_per_row = 2;
            B.num_groups = (uint32_t)(2*N);
            mb.wrap_weight(B);
            metal_matmul(mb, A, B, C);

            std::vector<float> ref(M*N);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    double acc = 0.0;
                    for (int k = 0; k < K; ++k)
                        acc += (double)a[m*K+k] * (double)wi[n*K+k];
                    ref[m*N+n] = (float)(acc * sw[2*n]);
                }
            }
            char label[80];
            snprintf(label, sizeof(label),
                     "W8A16 GEMM M=%d K=%d N=%d (%s tile)",
                     M, K, N, K >= 8192 ? "M128" : "M64");
            CHECK(close((const float*)C.data, ref.data(), M*N,
                        1e-2f, 3e-2f), label);
        }
    } else {
        std::printf("  SKIP: W8A16 prefill GEMM requires the Metal 4 tensor "
                    "path (M5/A19+ and compatible toolchain)\n");
    }

    // ---- W4A16 prefill GEMM (generic G32/G64 and specialized G128) -----------
    // Both variants use the decoded offset-binary [nibbles|scales] layout.
    // Keeping G64 here guards the non-specialized function-constant path.
    if (mb.has_tensor_path()) {
        setenv("MOLLM_METAL_W4_PREFILL_MODE", "fast", 1);
        constexpr int M = 64;
        constexpr int K = 256;
        constexpr int N = 64;
        for (int group_size : {32, 64, 128}) {
            const int groups_per_row = K / group_size;
            std::vector<float> a(M * K);
            for (int i = 0; i < M * K; ++i)
                a[i] =
                    0.001f *
                    static_cast<float>(
                        (i * 29 + 17) % 101 - 50);
            std::vector<int8_t> weights(N * K);
            std::vector<float> scales(N * groups_per_row);
            for (int n = 0; n < N; ++n) {
                for (int group = 0;
                     group < groups_per_row; ++group) {
                    scales[n * groups_per_row + group] =
                        0.002f *
                        static_cast<float>(
                            1 + (n + group) % 5);
                }
                for (int k = 0; k < K; ++k)
                    weights[n * K + k] =
                        static_cast<int8_t>(
                            (n * 7 + k * 3) % 16 - 8);
            }

            const size_t packed_bytes =
                static_cast<size_t>(N) * K / 2;
            const size_t scale_bytes =
                scales.size() * sizeof(float);
            std::vector<uint8_t> storage(
                packed_bytes + scale_bytes);
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; k += 2) {
                    const int lo =
                        static_cast<int>(
                            weights[n * K + k]) +
                        8;
                    const int hi =
                        static_cast<int>(
                            weights[n * K + k + 1]) +
                        8;
                    storage[
                        static_cast<size_t>(n) *
                            (K / 2) +
                        k / 2] =
                        static_cast<uint8_t>(
                            (hi << 4) | lo);
                }
            }
            std::memcpy(
                storage.data() + packed_bytes,
                scales.data(), scale_bytes);
            mb.register_weight_region(
                storage.data(), storage.size());

            Tensor A =
                make_dev(mb, Precision::FP32, K, M);
            std::memcpy(
                A.data, a.data(),
                a.size() * sizeof(float));
            Tensor C =
                make_dev(mb, Precision::FP32, N, M);
            Tensor B = Tensor::create(
                Precision::INT4, MemoryType::EXTERNAL,
                N, K, 1, 1, storage.data());
            B.scales =
                reinterpret_cast<const float*>(
                    storage.data() + packed_bytes);
            B.group_size =
                static_cast<uint32_t>(group_size);
            B.groups_per_row =
                static_cast<uint32_t>(groups_per_row);
            B.num_groups =
                static_cast<uint32_t>(
                    N * groups_per_row);
            mb.wrap_weight(B);
            metal_matmul(mb, A, B, C);

            std::vector<float> reference(M * N);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    double sum = 0.0;
                    for (int k = 0; k < K; ++k) {
                        const float weight =
                            static_cast<float>(
                                static_cast<__fp16>(
                                    static_cast<float>(
                                        weights[n * K + k]) *
                                    scales[
                                        n * groups_per_row +
                                        k / group_size]));
                        sum +=
                            static_cast<double>(
                                a[m * K + k]) *
                            static_cast<double>(weight);
                    }
                    reference[m * N + n] =
                        static_cast<float>(sum);
                }
            }
            char label[80];
            std::snprintf(
                label, sizeof(label),
                "W4A16 GEMM G%d function-constant path",
                group_size);
            CHECK(
                close(
                    static_cast<const float*>(C.data),
                    reference.data(), M * N,
                    1e-2f, 3e-2f),
                label);
        }
        unsetenv("MOLLM_METAL_W4_PREFILL_MODE");
    } else {
        std::printf("  SKIP: W4A16 prefill GEMM requires the Metal 4 tensor "
                    "path (M5/A19+ and compatible toolchain)\n");
    }

    // ---- Small-M W4 projection: one weight scan for M=2..4 ----------------
    {
        constexpr int K = 256;
        constexpr int N = 64;
        for (const auto [GS, M] : {
                 std::pair{32, 2}, std::pair{32, 3}, std::pair{32, 4},
                 std::pair{128, 2}, std::pair{128, 3},
                 std::pair{128, 4}}) {
            const int GPR = K / GS;
            std::vector<float> activations(M * K);
            fill_rand(activations.data(), M * K);
            std::vector<int8_t> weights(N * K);
            std::vector<float> scales(N * GPR);
            for (int n = 0; n < N; ++n) {
                for (int g = 0; g < GPR; ++g)
                    scales[n * GPR + g] =
                        0.0007f + 0.00003f * ((n + g) % 7);
                for (int k = 0; k < K; ++k)
                    weights[n * K + k] =
                        static_cast<int8_t>((n * 7 + k * 5) % 16 - 8);
            }

            const size_t packed_bytes =
                static_cast<size_t>(N) * K / 2;
            std::vector<uint8_t> storage(
                packed_bytes + scales.size() * sizeof(float));
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; k += 2) {
                    const int lo = weights[n * K + k] + 8;
                    const int hi = weights[n * K + k + 1] + 8;
                    storage[static_cast<size_t>(n) * (K / 2) + k / 2] =
                        static_cast<uint8_t>((hi << 4) | lo);
                }
            }
            std::memcpy(storage.data() + packed_bytes, scales.data(),
                        scales.size() * sizeof(float));
            mb.register_weight_region(storage.data(), storage.size());

            Tensor A = make_dev(mb, Precision::FP32, K, M);
            std::memcpy(A.data, activations.data(),
                        activations.size() * sizeof(float));
            Tensor C = make_dev(mb, Precision::FP32, N, M);
            Tensor B = Tensor::create(
                Precision::INT4, MemoryType::EXTERNAL,
                N, K, 1, 1, storage.data());
            B.scales = reinterpret_cast<const float*>(
                storage.data() + packed_bytes);
            B.group_size = GS;
            B.groups_per_row = GPR;
            B.num_groups = N * GPR;
            mb.wrap_weight(B);
            metal_matmul(mb, A, B, C);

            std::vector<float> reference(M * N, 0.0f);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    double sum = 0.0;
                    for (int g = 0; g < GPR; ++g) {
                        double group_sum = 0.0;
                        for (int k = g * GS; k < (g + 1) * GS; ++k)
                            group_sum +=
                                static_cast<double>(activations[m * K + k]) *
                                weights[n * K + k];
                        sum += group_sum * scales[n * GPR + g];
                    }
                    reference[m * N + n] = static_cast<float>(sum);
                }
            }
            char label[64];
            snprintf(label, sizeof(label),
                     "W4 G%d small-M GEMV M=%d", GS, M);
            CHECK(close(static_cast<const float*>(C.data), reference.data(),
                        M * N, 1e-4f, 3e-2f),
                  label);
            std::vector<float> batch_output(M * N);
            const bool batch_ok = mb.lm_head_small_batch(
                activations.data(), B, batch_output.data(), M, N, K);
            snprintf(label, sizeof(label),
                     "W4 G%d small-M lm_head batch M=%d", GS, M);
            CHECK(batch_ok &&
                      close(batch_output.data(), reference.data(), M * N,
                            1e-4f, 3e-2f),
                  label);
            std::vector<float> fused_output(M * N);
            mb.begin_graph();
            const bool fused_ok =
                mb.lm_head_small_batch_device_and_end_graph(
                    A, B, fused_output.data(), M, N, K);
            snprintf(label, sizeof(label),
                     "W4 G%d small-M fused lm_head M=%d", GS, M);
            CHECK(fused_ok &&
                      close(fused_output.data(), reference.data(), M * N,
                            1e-4f, 3e-2f),
                  label);
            std::vector<int> batch_top1(M, -1);
            mb.begin_graph();
            const bool batch_top1_ok =
                mb.lm_head_small_batch_argmax_device_and_end_graph(
                    A, B, batch_top1.data(), M, N, K);
            bool batch_top1_matches = batch_top1_ok;
            for (int m = 0; m < M; ++m) {
                const int expected = static_cast<int>(
                    std::max_element(reference.begin() + m * N,
                                     reference.begin() + (m + 1) * N) -
                    (reference.begin() + m * N));
                batch_top1_matches =
                    batch_top1_matches && batch_top1[m] == expected;
            }
            snprintf(label, sizeof(label),
                     "W4 G%d small-M batched GPU argmax M=%d", GS, M);
            CHECK(batch_top1_matches, label);
            if (M == 2) {
                mb.begin_graph();
                const int top1 =
                    mb.lm_head_argmax_device_and_end_graph(
                        A, 0, B, N, K);
                const int expected = static_cast<int>(
                    std::max_element(reference.begin(),
                                     reference.begin() + N) -
                    reference.begin());
                CHECK(top1 == expected, "W4 fused lm_head GPU argmax");
            }
        }
    }

    // ---- Small-M W8 projection: one weight scan for M=2..4 ----------------
    {
        constexpr int K = 256;
        constexpr int N = 64;
        constexpr int GS = 128;
        constexpr int GPR = K / GS;
        for (int M : {2, 3, 4}) {
            std::vector<float> activations(M * K);
            for (int i = 0; i < M * K; ++i)
                activations[i] =
                    0.001f * static_cast<float>((i * 29 + 7) % 997 - 498);
            std::vector<int8_t> weights(N * K);
            std::vector<float> scales(N * GPR);
            for (int n = 0; n < N; ++n) {
                for (int g = 0; g < GPR; ++g)
                    scales[n * GPR + g] =
                        0.0007f + 0.00003f * ((n + g) % 7);
                for (int k = 0; k < K; ++k)
                    weights[n * K + k] = static_cast<int8_t>(
                        (n * 17 + k * 11) % 255 - 127);
            }

            const size_t weight_bytes = weights.size();
            const size_t scales_offset =
                (weight_bytes + sizeof(float) - 1) &
                ~(sizeof(float) - 1);
            std::vector<uint8_t> storage(
                scales_offset + scales.size() * sizeof(float));
            std::memcpy(storage.data(), weights.data(), weight_bytes);
            std::memcpy(storage.data() + scales_offset, scales.data(),
                        scales.size() * sizeof(float));
            mb.register_weight_region(storage.data(), storage.size());

            Tensor A = make_dev(mb, Precision::FP32, K, M);
            std::memcpy(A.data, activations.data(),
                        activations.size() * sizeof(float));
            Tensor C = make_dev(mb, Precision::FP32, N, M);
            Tensor B = Tensor::create(
                Precision::INT8, MemoryType::EXTERNAL,
                N, K, 1, 1, storage.data());
            B.scales = reinterpret_cast<const float*>(
                storage.data() + scales_offset);
            B.group_size = GS;
            B.groups_per_row = GPR;
            B.num_groups = N * GPR;
            mb.wrap_weight(B);
            metal_matmul(mb, A, B, C);

            std::vector<float> reference(M * N, 0.0f);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    double sum = 0.0;
                    for (int k = 0; k < K; ++k) {
                        sum +=
                            static_cast<double>(activations[m * K + k]) *
                            weights[n * K + k] *
                            scales[n * GPR + k / GS];
                    }
                    reference[m * N + n] = static_cast<float>(sum);
                }
            }
            char label[64];
            snprintf(label, sizeof(label),
                     "W8 small-M GEMV M=%d", M);
            CHECK(close(static_cast<const float*>(C.data), reference.data(),
                        M * N, 1e-4f, 3e-2f),
                  label);
            std::vector<float> batch_output(M * N);
            const bool batch_ok = mb.lm_head_small_batch(
                activations.data(), B, batch_output.data(), M, N, K);
            snprintf(label, sizeof(label),
                     "W8 small-M lm_head batch M=%d", M);
            CHECK(batch_ok &&
                      close(batch_output.data(), reference.data(), M * N,
                            1e-4f, 3e-2f),
                  label);
            std::vector<float> fused_output(M * N);
            mb.begin_graph();
            const bool fused_ok =
                mb.lm_head_small_batch_device_and_end_graph(
                    A, B, fused_output.data(), M, N, K);
            snprintf(label, sizeof(label),
                     "W8 small-M fused lm_head M=%d", M);
            CHECK(fused_ok &&
                      close(fused_output.data(), reference.data(), M * N,
                            1e-4f, 3e-2f),
                  label);
            std::vector<int> batch_top1(M, -1);
            mb.begin_graph();
            const bool batch_top1_ok =
                mb.lm_head_small_batch_argmax_device_and_end_graph(
                    A, B, batch_top1.data(), M, N, K);
            bool batch_top1_matches = batch_top1_ok;
            for (int m = 0; m < M; ++m) {
                const int expected = static_cast<int>(
                    std::max_element(reference.begin() + m * N,
                                     reference.begin() + (m + 1) * N) -
                    (reference.begin() + m * N));
                batch_top1_matches =
                    batch_top1_matches && batch_top1[m] == expected;
            }
            snprintf(label, sizeof(label),
                     "W8 small-M batched GPU argmax M=%d", M);
            CHECK(batch_top1_matches, label);
            if (M == 2) {
                Tensor hidden_copy =
                    make_dev(mb, Precision::FP32, K, 1);
                mb.begin_graph();
                const int top1 =
                    mb.lm_head_argmax_device_and_end_graph(
                        A, 0, B, N, K, 0, &hidden_copy);
                const int expected = static_cast<int>(
                    std::max_element(reference.begin(),
                                     reference.begin() + N) -
                    reference.begin());
                CHECK(top1 == expected, "W8 fused lm_head GPU argmax");
                CHECK(close(static_cast<const float*>(hidden_copy.data),
                            activations.data(), K, 0.0f, 0.0f),
                      "W8 fused lm_head keeps hidden state on GPU");
            }
        }
    }

    // ---- W8A8 prefill GEMM (int8 activations x int8 per-channel weights) ----
    // Guards the int8xint8->int32 tensor path, whose cooperative-tensor layout
    // is compiler-sensitive (must match between offline metallib and runtime).
    // Enabled via MOLLM_METAL_W8A8; weights + scales live in a registered region.
    {
        setenv("MOLLM_METAL_W8A8", "1", 1);
        for (int ci = 0; ci < 5 && mb.has_tensor_path(); ci++) {
            int M = ci==0 ? 40 : (ci==1 ? 256 : (ci==2 ? 33 : 1));
            int K = ci==0 ? 256 : (ci==1 ? 1024 :
                    (ci==2 ? 96 : (ci==3 ? 2048 : 1024)));
            int N = ci==0 ? 96  : (ci==1 ? 512  : (ci==2 ? 80 : 264));

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
                if (M == 1) {
                    for (int n = 0; n < N; n++) {
                        double acc = 0.0;
                        for (int k = 0; k < K; k++)
                            acc += (double)a[k] * (double)wi[n*K+k];
                        ref[n] = (float)(acc * sw[n]);
                    }
                    continue;
                }
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
            snprintf(label, sizeof(label), "W8%s M=%d K=%d N=%d",
                     M == 1 ? " GEMV" : "A8 GEMM", M, K, N);
            CHECK(close((const float*)C.data, ref.data(), M*N,
                        M == 1 ? 1e-4f : 2e-3f,
                        M == 1 ? 2e-4f : 2e-2f), label);
        }
        unsetenv("MOLLM_METAL_W8A8");
    }

    // ---- W4A8 prefill GEMM (int8 activations x per-group int4 weights) ------
    // Weights ship in the CPU Q4B8G128Block layout; the Metal backend decodes
    // them to raw nibbles at wrap_weight time. Build a matching packed blob in a
    // registered weight region so wrap_weight's decode path is exercised.
    {
        setenv("MOLLM_METAL_W4_PREFILL_MODE", "accurate", 1);
        for (int ci = 0; ci < 6 && mb.has_tensor_path(); ci++) {
            const bool quant_probe = ci == 3;
            const bool balanced_probe = ci >= 4;
            const bool large_balanced_probe = ci == 5;
            int M = ci==0 ? 40 : (ci==1 ? 128 : (ci==2 ? 1 :
                    (balanced_probe ? 64 : 8)));
            int K = ci==0 ? 256 : (ci==1 ? 512 : (ci==2 ? 2048 :
                    (balanced_probe
                         ? (large_balanced_probe ? 2048 : 256)
                         : 128)));
            int N = ci==0 ? 48  : (ci==1 ? 80 : (ci==2 ? 264 : 128));
            int GS = 128, GPR = K / GS;

            std::vector<float> a(M*K); fill_rand(a.data(), M*K);
            // reference int4 weights (signed [-8,7]) + per-group scales.
            std::vector<int8_t> wq(N*K);
            std::vector<float>  sw(N*GPR);
            for (int n = 0; n < N; n++) {
                for (int g = 0; g < GPR; g++)
                    sw[n*GPR+g] =
                        quant_probe ? 1.0f
                                    : 0.0006f + 0.00002f*((n+g)%11);
                for (int k = 0; k < K; k++) {
                    int v = quant_probe ? (n == k ? 1 : 0)
                                        : (rand()%16)-8;
                    wq[n*K+k]=(int8_t)v;
                }
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
            mb.wrap_weight_int4(B);   // decode packed blocks -> raw nibbles

            if (balanced_probe)
                unsetenv("MOLLM_METAL_W4_PREFILL_MODE");
            metal_matmul(mb, A, B, C,
                         balanced_probe
                             ? std::vector<int>{1, 0, N/2}
                             : std::vector<int>{});
            if (balanced_probe) {
                std::vector<float> first((const float*)C.data,
                                         (const float*)C.data + M*N);
                metal_matmul(mb, A, B, C,
                             std::vector<int>{1, 0, N/2});
                CHECK(std::memcmp(first.data(), C.data,
                                  (size_t)M*N*sizeof(float)) == 0,
                      "W4 balanced K64 deterministic replay");
            }
            if (balanced_probe)
                setenv("MOLLM_METAL_W4_PREFILL_MODE", "accurate", 1);

            // Prefill follows the CPU production W4 path: independently
            // quantize every 32 activation values to Q8, take an integer dot,
            // combine activation/weight scales, and accumulate with FMA.
            // Decode consumes FP32 activations directly.
            std::vector<float> ref(M*N);
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float acc = 0.0f;
                    if (M == 1) {
                        for (int g=0; g<GPR; g++) {
                            double d=0.0;
                            for (int k=g*GS;k<(g+1)*GS;k++)
                                d += (double)a[m*K+k] *
                                     (double)wq[n*K+k];
                            acc += (float)(d * sw[n*GPR+g]);
                        }
                    } else if (balanced_probe) {
                        for (int kb=0; kb<K; kb+=64) {
                            float amax = 0.0f;
                            for (int k=kb; k<kb+64; k++)
                                amax = std::max(
                                    amax, std::fabs(a[m*K+k]));
                            float as =
                                amax > 0.0f ? amax/127.0f : 1.0f;
                            float inv =
                                amax > 0.0f ? 127.0f/amax : 0.0f;
                            int dot = 0;
                            for (int k=kb; k<kb+64; k++) {
                                int q = (int)std::nearbyint(
                                    a[m*K+k]*inv);
                                q = std::max(
                                    -127, std::min(127, q));
                                dot += q * (int)wq[n*K+k];
                            }
                            float combined =
                                sw[n*GPR + kb/GS] * as;
                            acc = std::fma(
                                (float)dot, combined, acc);
                        }
                    } else {
                        for (int kb=0; kb<K; kb+=32) {
                            float amax = 0.0f;
                            for (int k=kb; k<kb+32; k++)
                                amax = std::max(amax, std::fabs(a[m*K+k]));
                            float as = amax > 0.0f ? amax/127.0f : 1.0f;
                            float inv = amax > 0.0f ? 127.0f/amax : 0.0f;
                            int dot = 0;
                            for (int k=kb; k<kb+32; k++) {
                                int q = (int)std::nearbyint(a[m*K+k]*inv);
                                q = std::max(-127, std::min(127, q));
                                dot += q * (int)wq[n*K+k];
                            }
                            float combined = sw[n*GPR + kb/GS] * as;
                            acc = std::fma((float)dot, combined, acc);
                        }
                    }
                    if (balanced_probe && n < N/2)
                        acc = acc/(1.0f + std::exp(-acc));
                    ref[m*N+n]=acc;
                }
            }
            char label[64];
            snprintf(label, sizeof(label), "W4%s M=%d K=%d N=%d",
                     M == 1 ? " GEMV" :
                     (balanced_probe ? " balanced K64 GEMM" :
                                     " block32 GEMM"),
                     M, K, N);
            CHECK(close((const float*)C.data, ref.data(), M*N,
                        M == 1 ? 1e-4f :
                        (quant_probe ? 1e-7f :
                         (balanced_probe
                              ? (large_balanced_probe ? 3e-5f : 1e-5f)
                              : 3e-5f)),
                        M == 1 ? 3e-2f :
                        (quant_probe ? 1e-6f :
                         (balanced_probe ? 2e-4f : 1e-4f))),
                  label);
        }
        unsetenv("MOLLM_METAL_W4_PREFILL_MODE");
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

    // ---- Resident native-BG128 MoE decode ---------------------------------
    // Covers GPU routing with a no-shared-expert input layout and the direct
    // selected-expert kernel used by resident W4 packages.
    if (mb.has_tensor_path()) {
        constexpr int H = 128;
        constexpr int I = 128;
        constexpr int E = 256;
        constexpr int TOP = 8;
        constexpr int GU_ROWS = E * 2 * I;
        constexpr int DOWN_ROWS = E * H;
        constexpr int GU_BLOCKS = (GU_ROWS / 8) * (H / 128);
        constexpr int DOWN_BLOCKS = (DOWN_ROWS / 8) * (I / 128);
        std::vector<Q4B8G128Block> packed(GU_BLOCKS + DOWN_BLOCKS);
        for (size_t block = 0; block < packed.size(); ++block) {
            for (int channel = 0; channel < 8; ++channel)
                packed[block].scales[channel] =
                    0.001f * static_cast<float>(1 + (block + channel) % 5);
            for (int qgi = 0; qgi < 4; ++qgi)
                for (int channel = 0; channel < 8; ++channel)
                    for (int byte = 0; byte < 16; ++byte) {
                        int lo = static_cast<int>(
                                     (block * 13 + qgi * 7 +
                                      channel * 3 + byte) %
                                     16) -
                                 8;
                        int hi = static_cast<int>(
                                     (block * 5 + qgi * 11 +
                                      channel + byte * 3) %
                                     16) -
                                 8;
                        packed[block].q[qgi][channel][byte] =
                            static_cast<uint8_t>(
                                ((hi & 15) << 4) | (lo & 15));
                    }
        }
        CHECK(mb.register_weight_region(
                  packed.data(), packed.size() * sizeof(Q4B8G128Block)),
              "register native-BG128 MoE weight region");

        Tensor x = make_dev(mb, Precision::FP32, H, 1);
        Tensor router = make_dev(mb, Precision::FP16, E, H);
        Tensor bias = make_dev(mb, Precision::FP32, E, 1);
        Tensor out = make_dev(mb, Precision::FP32, H, 1);
        fill_rand(static_cast<float*>(x.data), H);
        std::memset(router.data, 0, router.nbytes());
        auto* bias_data = static_cast<float*>(bias.data);
        for (int e = 0; e < E; ++e)
            bias_data[e] =
                0.01f * static_cast<float>((e * 37) % E) - 0.64f;

        auto make_bg128 = [&](int rows, int k,
                              Q4B8G128Block* data) {
            Tensor weight = Tensor::create(
                Precision::INT4, MemoryType::EXTERNAL,
                rows, k, 1, 1, data);
            weight.group_size = 128;
            weight.groups_per_row = k / 128;
            weight.num_groups =
                static_cast<uint32_t>(rows * weight.groups_per_row);
            weight.is_q4_g128_packed = true;
            weight.q4_g128_data = data;
            mb.wrap_weight_int4(weight, true);
            return weight;
        };
        Tensor gate_up = make_bg128(GU_ROWS, H, packed.data());
        Tensor down = make_bg128(
            DOWN_ROWS, I, packed.data() + GU_BLOCKS);
        std::vector<const Tensor*> inputs = {
            &x, &router, &gate_up, &down, &bias};

        GraphNode moe;
        moe.op_type = OpType::MOE;
        moe.params.i32 = {
            H, E, TOP, I, I, 1, 1, 0, 1, 1, 0, 4, -1, -1};
        moe.params.f32 = {1.0f, 0.0f};
        mb.begin_graph();
        mb.dispatch(moe, inputs, &out, nullptr);
        mb.end_graph();

        std::vector<float> ref_data(H);
        Tensor ref = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            H, 1, 1, 1, ref_data.data());
        bool ref_ok = kernel_qwen3_moe(
            inputs, ref, nullptr, H, E, TOP, I, I,
            1, true, false, 1, 1, 1.0f, false, 4);
        CHECK(ref_ok, "CPU reference native-BG128 MoE decode");
        CHECK(close(static_cast<const float*>(out.data), ref_data.data(), H,
                    2e-3f, 3e-2f),
              "Metal resident native-BG128 MoE matches CPU");

        // The production 128-expert path uses a smaller selector
        // threadgroup than the 256-expert case above. Reuse the first half of
        // each resident tensor so this scheduling specialization has direct
        // CPU parity coverage.
        constexpr int E128 = 128;
        Tensor router128 = router;
        router128.shape[0] = E128;
        Tensor bias128 = bias;
        bias128.shape[0] = E128;
        Tensor gate_up128 = gate_up;
        gate_up128.shape[0] = E128 * 2 * I;
        Tensor down128 = down;
        down128.shape[0] = E128 * H;
        std::vector<const Tensor*> inputs128 = {
            &x, &router128, &gate_up128, &down128, &bias128};
        GraphNode moe128 = moe;
        moe128.params.i32[1] = E128;
        mb.begin_graph();
        mb.dispatch(moe128, inputs128, &out, nullptr);
        mb.end_graph();
        ref_ok = kernel_qwen3_moe(
            inputs128, ref, nullptr, H, E128, TOP, I, I,
            1, true, false, 1, 1, 1.0f, false, 4);
        CHECK(ref_ok,
              "CPU reference 128-expert native-BG128 MoE decode");
        CHECK(close(static_cast<const float*>(out.data), ref_data.data(), H,
                    2e-3f, 3e-2f),
              "Metal 128-expert native-BG128 MoE matches CPU");

        // MTP verification uses 2..4 token rows. This covers the fused
        // small-M router/BG128 activation quantizer and the selected-expert
        // scheduling specialization with repeated GPU-resident weights.
        for (int small_m : {2, 3, 4}) {
            Tensor x_small =
                make_dev(mb, Precision::FP32, H, small_m);
            Tensor out_small =
                make_dev(mb, Precision::FP32, H, small_m);
            fill_rand(
                static_cast<float*>(x_small.data),
                static_cast<size_t>(H) * small_m);
            std::vector<const Tensor*> inputs_small = {
                &x_small, &router128, &gate_up128, &down128, &bias128};
            mb.begin_graph();
            mb.dispatch(moe128, inputs_small, &out_small, nullptr);
            mb.end_graph();

            std::vector<float> ref_small_data(
                static_cast<size_t>(H) * small_m);
            Tensor ref_small = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL,
                H, small_m, 1, 1, ref_small_data.data());
            ref_ok = kernel_qwen3_moe(
                inputs_small, ref_small, nullptr,
                H, E128, TOP, I, I,
                1, true, false, 1, 1, 1.0f, false, 4);
            char label[96];
            std::snprintf(
                label, sizeof(label),
                "CPU reference native-BG128 MoE small-M=%d",
                small_m);
            CHECK(ref_ok, label);
            std::snprintf(
                label, sizeof(label),
                "Metal native-BG128 MoE small-M=%d matches CPU",
                small_m);
            CHECK(close(
                      static_cast<const float*>(out_small.data),
                      ref_small_data.data(),
                      static_cast<size_t>(H) * small_m,
                      2e-3f, 3e-2f),
                  label);
        }

        moe.params.i32[8] = 8;
        moe.params.i32[9] = 4;
        mb.begin_graph();
        mb.dispatch(moe, inputs, &out, nullptr);
        mb.end_graph();
        ref_ok = kernel_qwen3_moe(
            inputs, ref, nullptr, H, E, TOP, I, I,
            1, true, false, 8, 4, 1.0f, false, 4);
        CHECK(ref_ok,
              "CPU reference grouped-sigmoid native-BG128 MoE decode");
        CHECK(close(static_cast<const float*>(out.data), ref_data.data(), H,
                    2e-3f, 3e-2f),
              "Metal grouped-sigmoid native-BG128 MoE matches CPU");

        // Exercise the softmax specialization with non-uniform router logits.
        auto* router_data = static_cast<__fp16*>(router.data);
        for (int e = 0; e < E; ++e) {
            const float coefficient =
                0.0005f *
                static_cast<float>((e * 17) % E - E / 2);
            for (int h = 0; h < H; ++h)
                router_data[e * H + h] = (__fp16)coefficient;
        }
        moe.params.i32[5] = 0;
        moe.params.i32[8] = 1;
        moe.params.i32[9] = 1;
        moe.params.i32[11] = -1;
        mb.begin_graph();
        mb.dispatch(moe, inputs, &out, nullptr);
        mb.end_graph();

        ref_ok = kernel_qwen3_moe(
            inputs, ref, nullptr, H, E, TOP, I, I,
            0, true, false, 1, 1, 1.0f, false, -1);
        CHECK(ref_ok, "CPU reference softmax native-BG128 MoE decode");
        CHECK(close(static_cast<const float*>(out.data), ref_data.data(), H,
                    2e-3f, 3e-2f),
              "Metal softmax native-BG128 MoE matches CPU");

        // Multi-token prefill uses the expert-grouped native-BG128 path. Keep
        // this separate from the M=1 checks above so route compaction, grouped
        // activation indexing, and canonical output scatter are all covered.
        constexpr int PREFILL_S = 64;
        Tensor prefill_x =
            make_dev(mb, Precision::FP32, H, PREFILL_S);
        Tensor prefill_out =
            make_dev(mb, Precision::FP32, H, PREFILL_S);
        fill_rand(
            static_cast<float*>(prefill_x.data),
            H * PREFILL_S);
        std::vector<const Tensor*> prefill_inputs = {
            &prefill_x, &router, &gate_up, &down, &bias};
        mb.begin_graph();
        mb.dispatch(moe, prefill_inputs, &prefill_out, nullptr);
        mb.end_graph();

        std::vector<float> prefill_ref_data(H * PREFILL_S);
        Tensor prefill_ref = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            H, PREFILL_S, 1, 1, prefill_ref_data.data());
        ref_ok = kernel_qwen3_moe(
            prefill_inputs, prefill_ref, nullptr,
            H, E, TOP, I, I,
            0, true, false, 1, 1, 1.0f, false, -1);
        CHECK(ref_ok,
              "CPU reference grouped native-BG128 MoE prefill");
        CHECK(close(
                  static_cast<const float*>(prefill_out.data),
                  prefill_ref_data.data(), H * PREFILL_S,
                  2e-3f, 3e-2f),
              "Metal grouped native-BG128 MoE prefill matches CPU");
        std::vector<float> prefill_first(
            static_cast<const float*>(prefill_out.data),
            static_cast<const float*>(prefill_out.data) +
                H * PREFILL_S);
        mb.begin_graph();
        mb.dispatch(moe, prefill_inputs, &prefill_out, nullptr);
        mb.end_graph();
        CHECK(std::memcmp(
                  prefill_first.data(), prefill_out.data,
                  prefill_first.size() * sizeof(float)) == 0,
              "Metal grouped native-BG128 MoE replay is deterministic");

        // The parallel selector is also used for grouped sigmoid routing in
        // multi-token prefill. Cover its group filtering and correction-bias
        // path independently from the softmax case above.
        moe.params.i32[5] = 1;
        moe.params.i32[8] = 8;
        moe.params.i32[9] = 4;
        moe.params.i32[11] = 4;
        mb.begin_graph();
        mb.dispatch(moe, prefill_inputs, &prefill_out, nullptr);
        mb.end_graph();
        ref_ok = kernel_qwen3_moe(
            prefill_inputs, prefill_ref, nullptr,
            H, E, TOP, I, I,
            1, true, false, 8, 4, 1.0f, false, 4);
        CHECK(ref_ok,
              "CPU reference grouped-sigmoid BG128 MoE prefill");
        CHECK(close(
                  static_cast<const float*>(prefill_out.data),
                  prefill_ref_data.data(), H * PREFILL_S,
                  2e-3f, 3e-2f),
              "Metal grouped-sigmoid BG128 MoE prefill matches CPU");

        // Force routes to spread evenly across all experts. Each token is a
        // different one-hot hidden row and owns eight router rows, so every
        // expert receives exactly two routes. This exercises the adaptive
        // grouped-MoE Route16 branch; the concentrated cases above select
        // Route32.
        std::memset(router.data, 0, router.nbytes());
        std::memset(prefill_x.data, 0, prefill_x.nbytes());
        router_data = static_cast<__fp16*>(router.data);
        auto* spread_x =
            static_cast<float*>(prefill_x.data);
        for (int token = 0; token < PREFILL_S; ++token) {
            const int hidden = token % H;
            spread_x[token * H + hidden] = 1.0f;
            for (int rank = 0; rank < TOP; ++rank) {
                const int expert =
                    (hidden * TOP + rank) % E;
                router_data[expert * H + hidden] =
                    (__fp16)(1.0f + 0.01f * rank);
            }
        }
        moe.params.i32[5] = 0;
        moe.params.i32[8] = 1;
        moe.params.i32[9] = 1;
        moe.params.i32[11] = -1;
        mb.begin_graph();
        mb.dispatch(moe, prefill_inputs, &prefill_out, nullptr);
        mb.end_graph();
        ref_ok = kernel_qwen3_moe(
            prefill_inputs, prefill_ref, nullptr,
            H, E, TOP, I, I,
            0, true, false, 1, 1, 1.0f, false, -1);
        CHECK(ref_ok,
              "CPU reference spread-route BG128 MoE prefill");
        CHECK(close(
                  static_cast<const float*>(prefill_out.data),
                  prefill_ref_data.data(), H * PREFILL_S,
                  2e-3f, 3e-2f),
              "Metal spread-route Route16 MoE prefill matches CPU");
    }

    // ---- Resident native-BG32 MoE decode and grouped prefill --------------
    // Package W4G32 tensors use embedded per-eight-channel scales and signed
    // two's-complement nibbles. Cover both the selected decode kernel and the
    // expert-grouped prefill kernel against the CPU packed-BG32 path.
    if (mb.has_tensor_path()) {
        constexpr int H = 128;
        constexpr int I = 128;
        constexpr int E = 128;
        constexpr int TOP = 8;
        constexpr int GPR = H / 32;
        constexpr int GU_ROWS = E * 2 * I;
        constexpr int DOWN_ROWS = E * H;
        constexpr int GU_BLOCKS = (GU_ROWS / 8) * GPR;
        constexpr int DOWN_BLOCKS = (DOWN_ROWS / 8) * GPR;
        std::vector<Q4B8G32Block> packed(
            GU_BLOCKS + DOWN_BLOCKS);
        for (size_t block = 0; block < packed.size(); ++block) {
            for (int channel = 0; channel < 8; ++channel) {
                packed[block].scales[channel] =
                    0.001f *
                    static_cast<float>(1 + (block + channel) % 5);
                for (int byte = 0; byte < 16; ++byte) {
                    const int lo =
                        static_cast<int>(
                            (block * 13 + channel * 3 + byte) % 16) -
                        8;
                    const int hi =
                        static_cast<int>(
                            (block * 5 + channel + byte * 3) % 16) -
                        8;
                    packed[block].q[channel][byte] =
                        static_cast<uint8_t>(
                            ((hi & 15) << 4) | (lo & 15));
                }
            }
        }
        CHECK(mb.register_weight_region(
                  packed.data(),
                  packed.size() * sizeof(Q4B8G32Block)),
              "register native-BG32 MoE weight region");

        auto make_bg32 = [&](int rows, int k,
                             Q4B8G32Block* data) {
            Tensor weight = Tensor::create(
                Precision::INT4, MemoryType::EXTERNAL,
                rows, k, 1, 1, data);
            weight.group_size = 32;
            weight.groups_per_row = k / 32;
            weight.num_groups =
                static_cast<uint32_t>(
                    rows * weight.groups_per_row);
            weight.is_q4_g32_packed = true;
            weight.q4_g32_data = data;
            mb.wrap_weight_int4(weight, true);
            return weight;
        };
        Tensor gate_up =
            make_bg32(GU_ROWS, H, packed.data());
        Tensor down = make_bg32(
            DOWN_ROWS, I, packed.data() + GU_BLOCKS);
        Tensor router =
            make_dev(mb, Precision::FP16, E, H);
        Tensor bias =
            make_dev(mb, Precision::FP32, E, 1);
        std::memset(router.data, 0, router.nbytes());
        auto* bias_data = static_cast<float*>(bias.data);
        for (int expert = 0; expert < E; ++expert)
            bias_data[expert] =
                0.01f *
                    static_cast<float>((expert * 37) % E) -
                0.64f;

        GraphNode moe;
        moe.op_type = OpType::MOE;
        moe.params.i32 = {
            H, E, TOP, I, I, 1, 1, 0, 8, 4, 0, 4, -1, -1};
        moe.params.f32 = {1.0f, 0.0f};

        Tensor x =
            make_dev(mb, Precision::FP32, H, 1);
        Tensor out =
            make_dev(mb, Precision::FP32, H, 1);
        fill_rand(static_cast<float*>(x.data), H);
        std::vector<const Tensor*> inputs = {
            &x, &router, &gate_up, &down, &bias};
        mb.begin_graph();
        mb.dispatch(moe, inputs, &out, nullptr);
        mb.end_graph();

        std::vector<float> ref_data(H);
        Tensor ref = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            H, 1, 1, 1, ref_data.data());
        bool ref_ok = kernel_qwen3_moe(
            inputs, ref, nullptr, H, E, TOP, I, I,
            1, true, false, 8, 4, 1.0f, false, 4);
        CHECK(ref_ok, "CPU reference native-BG32 MoE decode");
        CHECK(close(
                  static_cast<const float*>(out.data),
                  ref_data.data(), H, 2e-4f, 3e-3f),
              "Metal resident native-BG32 MoE matches CPU");

        for (int small_m : {2, 3, 4}) {
            Tensor small_x =
                make_dev(mb, Precision::FP32, H, small_m);
            Tensor small_out =
                make_dev(mb, Precision::FP32, H, small_m);
            fill_rand(
                static_cast<float*>(small_x.data), H * small_m);
            std::vector<const Tensor*> small_inputs = {
                &small_x, &router, &gate_up, &down, &bias};
            mb.begin_graph();
            mb.dispatch(moe, small_inputs, &small_out, nullptr);
            mb.end_graph();

            std::vector<float> small_ref_data(H * small_m);
            Tensor small_ref = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL,
                H, small_m, 1, 1, small_ref_data.data());
            ref_ok = kernel_qwen3_moe(
                small_inputs, small_ref, nullptr,
                H, E, TOP, I, I,
                1, true, false, 8, 4, 1.0f, false, 4);
            char label[96];
            std::snprintf(
                label, sizeof(label),
                "CPU reference native-BG32 MoE small-M=%d", small_m);
            CHECK(ref_ok, label);
            std::snprintf(
                label, sizeof(label),
                "Metal native-BG32 MoE small-M=%d matches CPU", small_m);
            CHECK(close(
                      static_cast<const float*>(small_out.data),
                      small_ref_data.data(), H * small_m,
                      2e-4f, 3e-3f),
                  label);
        }

        constexpr int PREFILL_S = 64;
        Tensor prefill_x =
            make_dev(mb, Precision::FP32, H, PREFILL_S);
        Tensor prefill_out =
            make_dev(mb, Precision::FP32, H, PREFILL_S);
        fill_rand(
            static_cast<float*>(prefill_x.data),
            H * PREFILL_S);
        std::vector<const Tensor*> prefill_inputs = {
            &prefill_x, &router, &gate_up, &down, &bias};
        mb.begin_graph();
        mb.dispatch(
            moe, prefill_inputs, &prefill_out, nullptr);
        mb.end_graph();

        std::vector<float> prefill_ref_data(
            H * PREFILL_S);
        Tensor prefill_ref = Tensor::create(
            Precision::FP32, MemoryType::EXTERNAL,
            H, PREFILL_S, 1, 1, prefill_ref_data.data());
        ref_ok = kernel_qwen3_moe(
            prefill_inputs, prefill_ref, nullptr,
            H, E, TOP, I, I,
            1, true, false, 8, 4, 1.0f, false, 4);
        CHECK(ref_ok,
              "CPU reference grouped native-BG32 MoE prefill");
        CHECK(close(
                  static_cast<const float*>(prefill_out.data),
                  prefill_ref_data.data(), H * PREFILL_S,
                  2e-3f, 3e-2f),
              "Metal grouped native-BG32 MoE prefill matches CPU");
    }

    // ---- Resident W8 MoE decode and prefill ------------------------------
    // The production W8PC package stores expert rows and per-channel scales
    // separately. Register one combined region so this test also exercises
    // independent Metal buffer offsets for the two scale arrays.
    if (mb.has_tensor_path()) {
        // Keep both K dimensions above the K128 activation group so the
        // grouped prefill test covers multi-group accumulation, not just the
        // legacy one-scale-per-row special case.
        constexpr int H = 256;
        constexpr int I = 256;
        constexpr int E = 32;
        constexpr int TOP = 4;
        constexpr int GU_ROWS = E * 2 * I;
        constexpr int DOWN_ROWS = E * H;
        constexpr size_t GU_DATA = (size_t)GU_ROWS * H;
        constexpr size_t GU_SCALES = (size_t)GU_ROWS * sizeof(float);
        constexpr size_t DOWN_DATA = (size_t)DOWN_ROWS * I;
        constexpr size_t DOWN_SCALES = (size_t)DOWN_ROWS * sizeof(float);
        std::vector<uint8_t> storage(
            GU_DATA + GU_SCALES + DOWN_DATA + DOWN_SCALES);
        int8_t* gu_data = reinterpret_cast<int8_t*>(storage.data());
        float* gu_scales = reinterpret_cast<float*>(
            storage.data() + GU_DATA);
        int8_t* down_data = reinterpret_cast<int8_t*>(
            storage.data() + GU_DATA + GU_SCALES);
        float* down_scales = reinterpret_cast<float*>(
            storage.data() + GU_DATA + GU_SCALES + DOWN_DATA);
        for (size_t i = 0; i < GU_DATA; ++i)
            gu_data[i] = static_cast<int8_t>((i * 17 + 5) % 31 - 15);
        for (int row = 0; row < GU_ROWS; ++row)
            gu_scales[row] = 0.001f * (1 + row % 7);
        for (size_t i = 0; i < DOWN_DATA; ++i)
            down_data[i] = static_cast<int8_t>((i * 11 + 3) % 29 - 14);
        for (int row = 0; row < DOWN_ROWS; ++row)
            down_scales[row] = 0.0015f * (1 + row % 5);

        CHECK(mb.register_weight_region(storage.data(), storage.size()),
              "register W8 MoE weight and scale region");
        auto make_w8 = [&](int rows, int k, int8_t* data,
                           float* scales) {
            Tensor weight = Tensor::create(
                Precision::INT8, MemoryType::EXTERNAL,
                rows, k, 1, 1, data);
            weight.scales = scales;
            weight.group_size = k;
            weight.groups_per_row = 1;
            weight.num_groups = static_cast<uint32_t>(rows);
            mb.wrap_weight(weight);
            return weight;
        };
        Tensor gate_up = make_w8(GU_ROWS, H, gu_data, gu_scales);
        Tensor down = make_w8(DOWN_ROWS, I, down_data, down_scales);
        Tensor router = make_dev(mb, Precision::FP16, E, H);
        auto* router_data = static_cast<__fp16*>(router.data);
        for (int e = 0; e < E; ++e)
            for (int h = 0; h < H; ++h)
                router_data[e * H + h] = (__fp16)(
                    0.001f * static_cast<float>((e * 13 + h * 7) % 19 - 9));

        GraphNode moe;
        moe.op_type = OpType::MOE;
        moe.params.i32 = {
            H, E, TOP, I, I, 0, 1, 0, 1, 1, 0, -1, -1, -1};
        moe.params.f32 = {1.0f, 0.0f};

        auto run_w8_case = [&](int seq, const char* label) {
            Tensor x = make_dev(mb, Precision::FP32, H, seq);
            Tensor out = make_dev(mb, Precision::FP32, H, seq);
            fill_rand(static_cast<float*>(x.data), H * seq);
            if (seq >= 64) {
                // Make adjacent K128 groups differ strongly in magnitude.
                // A legacy row-wise scale loses most of the first group,
                // while grouped activation quantization preserves it.
                auto* x_data = static_cast<float*>(x.data);
                for (int row = 0; row < seq; ++row)
                    for (int k = 0; k < H / 2; ++k)
                        x_data[row * H + k] *= 0.02f;
            }
            std::vector<const Tensor*> inputs = {
                &x, &router, &gate_up, &down};
            mb.begin_graph();
            mb.dispatch(moe, inputs, &out, nullptr);
            mb.end_graph();

            std::vector<float> ref_data(H * seq);
            Tensor ref = Tensor::create(
                Precision::FP32, MemoryType::EXTERNAL,
                H, seq, 1, 1, ref_data.data());
            const bool ref_ok = kernel_qwen3_moe(
                inputs, ref, nullptr, H, E, TOP, I, I,
                0, true, false, 1, 1, 1.0f,
                false, -1);
            CHECK(ref_ok, label);
            CHECK(close(static_cast<const float*>(out.data),
                        ref_data.data(), H * seq,
                        seq >= 64 ? 6e-4f : 2e-4f,
                        seq >= 64 ? 5e-2f : 3e-3f),
                  seq == 1
                      ? "Metal resident W8 MoE decode matches CPU"
                      : seq >= 64
                          ? "Metal grouped W8 MoE prefill matches CPU"
                          : "Metal resident W8 MoE prefill matches CPU");
        };
        run_w8_case(1, "CPU reference W8 MoE decode");
        run_w8_case(2, "CPU reference W8 MoE M=2 prefill");
        run_w8_case(3, "CPU reference W8 MoE M=3 prefill");
        run_w8_case(4, "CPU reference W8 MoE prefill");
        run_w8_case(64, "CPU reference grouped W8 MoE prefill");
    }

    if (failures == 0) printf("All Metal op parity tests passed.\n");
    return failures ? 1 : 0;
}
