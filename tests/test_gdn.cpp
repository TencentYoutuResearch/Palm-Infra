// Unit tests for the fused GDN kernel (Qwen3.5 linear attention core).
//
// The kernel consumes 8 inputs in their native [seq, dim] row-major data layout:
//   qkv_conv [seq, qkv_total], a_out [seq, num_heads], b_out [seq, num_heads],
//   z_out [seq, num_heads*v_dim], A_log [num_heads], dt_bias [num_heads],
//   norm_weight [v_dim], gdn_state [num_heads, k_dim, v_dim] (in-place).
//
// Each test builds synthetic inputs, runs the kernel, and compares against a
// scalar reference implementation of the same math (g/beta → recurrence →
// RMSNormGated). Tolerance 1e-3: the NEON kernel uses sigmoid_f32_neon
// (polynomial approximation, ~7-bit precision) for RMSNormGated gating,
// which introduces ~1e-3 error vs the scalar std::exp path. This does not
// affect end-to-end PPL (test_e2e confirms PPL 8.49 vs HF 8.50).

#include "kernels/gdn.h"
#include "graph/graph.h"
#include "kernels/tensor.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

static void fill_rand(float* d, int n, unsigned int* seed) {
    for (int i = 0; i < n; i++) {
        // rand_r for thread-safe-ish determinism across runs
        d[i] = ((float)rand_r(seed) / (float)RAND_MAX) - 0.5f;
    }
}

static inline float ref_sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }
static inline float ref_softplusf(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return std::exp(x);
    return std::log1pf(std::exp(x));
}

// Scalar reference for the fused op. Mirrors gdn.cpp's gdn_core exactly.
// Produces `out` in [seq, num_heads*v_dim] row-major and updates `state` in-place.
static void ref_fused_gdn(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* A_log, const float* dt_bias, const float* norm_w,
    float* state, float* out,
    int num_heads, int k_dim, int v_dim, int seq_len,
    bool use_l2norm, float rms_eps, float l2norm_eps, float scale,
    int num_v_heads = -1)
{
    if (num_v_heads < 0) num_v_heads = num_heads;
    int qkv_dim   = num_heads * k_dim;
    int qkv_total = qkv_dim * 2 + num_v_heads * v_dim;
    int z_dim     = num_v_heads * v_dim;
    int state_size = k_dim * v_dim;
    int repeat = num_v_heads / num_heads;

    std::vector<float> neg_exp_A(num_v_heads), dt_bias_h(num_v_heads);
    for (int h = 0; h < num_v_heads; h++) {
        neg_exp_A[h] = -std::exp(A_log[h]);
        dt_bias_h[h] = dt_bias[h];
    }

    std::vector<float> q_pre(k_dim), k_pre(k_dim), v_t(v_dim);
    std::vector<float> q_n(k_dim), k_n(k_dim);
    std::vector<float> kv_mem(v_dim), delta(v_dim), attn_out(v_dim);

    for (int vh = 0; vh < num_v_heads; vh++) {
        int kh = vh / repeat;
        float* state_h = state + vh * state_size;
        float nea = neg_exp_A[vh];
        float dtb = dt_bias_h[vh];

        for (int t = 0; t < seq_len; t++) {
            // qkv layout: [qkv_total, seq] row-major → qkv[qkv_idx * seq_len + t]
            for (int d = 0; d < k_dim; d++) {
                q_pre[d] = qkv[(kh * k_dim + d) * seq_len + t];
                k_pre[d] = qkv[(qkv_dim + kh * k_dim + d) * seq_len + t];
            }
            for (int d = 0; d < v_dim; d++)
                v_t[d] = qkv[(2 * qkv_dim + vh * v_dim + d) * seq_len + t];

            float a_ht   = a[t * num_v_heads + vh];
            float b_ht   = b[t * num_v_heads + vh];
            float sp     = ref_softplusf(a_ht + dtb);
            float g_t    = nea * sp;
            float beta_t = ref_sigmoidf(b_ht);
            float g_t_exp = std::exp(g_t);

            const float* q_ptr = q_pre.data();
            const float* k_ptr = k_pre.data();
            if (use_l2norm) {
                float sq_q = 0, sq_k = 0;
                for (int d = 0; d < k_dim; d++) { sq_q += q_pre[d]*q_pre[d]; sq_k += k_pre[d]*k_pre[d]; }
                float inv_q = 1.f/std::sqrt(sq_q + l2norm_eps);
                float inv_k = 1.f/std::sqrt(sq_k + l2norm_eps);
                for (int d = 0; d < k_dim; d++) { q_n[d] = q_pre[d]*inv_q; k_n[d] = k_pre[d]*inv_k; }
                q_ptr = q_n.data();
                k_ptr = k_n.data();
            }

            // recurrence step
            for (int i = 0; i < state_size; i++) state_h[i] *= g_t_exp;
            for (int dv = 0; dv < v_dim; dv++) kv_mem[dv] = 0.f;
            for (int dk = 0; dk < k_dim; dk++) {
                float kv = k_ptr[dk];
                const float* row = state_h + dk * v_dim;
                for (int dv = 0; dv < v_dim; dv++) kv_mem[dv] += row[dv] * kv;
            }
            for (int dv = 0; dv < v_dim; dv++) delta[dv] = (v_t[dv] - kv_mem[dv]) * beta_t;
            for (int dk = 0; dk < k_dim; dk++) {
                float kv = k_ptr[dk];
                float* row = state_h + dk * v_dim;
                for (int dv = 0; dv < v_dim; dv++) row[dv] += kv * delta[dv];
            }
            for (int dv = 0; dv < v_dim; dv++) {
                float s = 0.f;
                for (int dk = 0; dk < k_dim; dk++) s += state_h[dk * v_dim + dv] * q_ptr[dk];
                attn_out[dv] = s * scale;
            }

            // RMSNormGated
            // Output layout: [z_dim, seq_len] row-major → out[(h*v_dim+d) + t * z_dim]
            const float* z_row = z + t * z_dim + vh * v_dim;
            float sum_sq = 0.f;
            for (int d = 0; d < v_dim; d++) sum_sq += attn_out[d] * attn_out[d];
            float rms = 1.f / std::sqrt(sum_sq / (float)v_dim + rms_eps);
            for (int d = 0; d < v_dim; d++) {
                float normed = attn_out[d] * rms * norm_w[d];
                float silu_z = z_row[d] * ref_sigmoidf(z_row[d]);
                int global_dim = vh * v_dim + d;
                out[global_dim + t * z_dim] = normed * silu_z;
            }
        }
    }
}

// Build the 8 input tensors + output tensor and call kernel_gdn_prefill/decode.
// `state` is passed as EXTERNAL so the caller can inspect it post-call.
static bool run_kernel(bool prefill,
                       int num_heads, int k_dim, int v_dim, int seq_len,
                       const float* qkv, const float* a, const float* b, const float* z,
                       const float* A_log, const float* dt_bias, const float* norm_w,
                       float* state, float* out_buf,
                       int num_v_heads = -1) {
    if (num_v_heads < 0) num_v_heads = num_heads;
    int qkv_total = num_heads * k_dim * 2 + num_v_heads * v_dim;
    int z_dim     = num_v_heads * v_dim;

    // Input shapes use the [dim, seq] declared convention (kernel ignores them
    // and indexes by data layout [seq, dim]); we set strides to match the data.
    auto make_2d = [](Precision prec, int d0, int d1, const void* data) {
        Tensor t = Tensor::create(prec, MemoryType::EXTERNAL, d0, d1, 1, 1,
                                   const_cast<void*>(data));
        return t;
    };

    Tensor qkv_t   = make_2d(Precision::FP32, qkv_total, seq_len,   qkv);
    Tensor a_t     = make_2d(Precision::FP32, num_v_heads, seq_len, a);
    Tensor b_t     = make_2d(Precision::FP32, num_v_heads, seq_len, b);
    Tensor z_t     = make_2d(Precision::FP32, z_dim,      seq_len,  z);
    Tensor A_log_t = make_2d(Precision::FP32, num_v_heads, 1,       A_log);
    Tensor dtb_t   = make_2d(Precision::FP32, num_v_heads, 1,       dt_bias);
    Tensor norm_t  = make_2d(Precision::FP32, v_dim,      1,        norm_w);
    Tensor state_t = make_2d(Precision::FP32, v_dim * k_dim * num_v_heads, 1, state);
    Tensor out_t   = make_2d(Precision::FP32, z_dim,      seq_len,  out_buf);

    OpParams params;
    params.i32 = {num_heads, k_dim, v_dim, seq_len, 1 /*use_l2norm*/,
                  4 /*conv_kernel*/, seq_len /*n_real*/, num_v_heads};
    params.f32 = {1e-6f /*rms_eps*/, 1e-6f /*l2norm_eps*/, 1.f / std::sqrt((float)k_dim)};

    std::vector<const Tensor*> inputs = {&qkv_t, &a_t, &b_t, &z_t,
                                          &A_log_t, &dtb_t, &norm_t, &state_t};
    std::vector<Tensor*> outputs = {&out_t};

    if (prefill) {
        kernel_gdn_prefill(params, inputs, outputs, nullptr);
    } else {
        kernel_gdn_decode(params, inputs, outputs, nullptr);
    }
    return true;
}

// Compare two buffers, print first mismatch.
static bool compare(const float* a, const float* b, int n, const char* name, float tol = 1e-3f) {
    float max_err = 0; int max_idx = 0;
    for (int i = 0; i < n; i++) {
        float e = std::fabs(a[i] - b[i]);
        if (e > max_err) { max_err = e; max_idx = i; }
    }
    if (max_err > tol) {
        fprintf(stderr, "  MISMATCH %s: max_err=%e at idx %d (ref=%f got=%f)\n",
                name, max_err, max_idx, a[max_idx], b[max_idx]);
        return false;
    }
    printf("  PASS: %s (max_err=%e at idx %d)\n", name, max_err, max_idx);
    return true;
}

// ---- Test 1: prefill, small dims, compare against scalar reference ----
static bool test_prefill_basic() {
    const int num_heads = 2, k_dim = 8, v_dim = 8, seq_len = 4;
    const int qkv_dim = num_heads * k_dim;
    const int qkv_total = qkv_dim * 3;
    const int z_dim = num_heads * v_dim;
    const int state_size = num_heads * k_dim * v_dim;
    unsigned int seed = 42;

    std::vector<float> qkv(seq_len * qkv_total), a(seq_len * num_heads),
                       b(seq_len * num_heads), z(seq_len * z_dim),
                       A_log(num_heads), dt_bias(num_heads), norm_w(v_dim);
    fill_rand(qkv.data(), qkv.size(), &seed);
    fill_rand(a.data(), a.size(), &seed);
    fill_rand(b.data(), b.size(), &seed);
    fill_rand(z.data(), z.size(), &seed);
    fill_rand(A_log.data(), A_log.size(), &seed);
    fill_rand(dt_bias.data(), dt_bias.size(), &seed);
    fill_rand(norm_w.data(), norm_w.size(), &seed);

    std::vector<float> state_k(state_size, 0), state_r(state_size, 0);
    std::vector<float> out_k(seq_len * z_dim, 0), out_r(seq_len * z_dim, 0);

    float scale = 1.f / std::sqrt((float)k_dim);
    ref_fused_gdn(qkv.data(), a.data(), b.data(), z.data(),
                  A_log.data(), dt_bias.data(), norm_w.data(),
                  state_r.data(), out_r.data(),
                  num_heads, k_dim, v_dim, seq_len,
                  true, 1e-6f, 1e-6f, scale);

    run_kernel(/*prefill=*/true, num_heads, k_dim, v_dim, seq_len,
               qkv.data(), a.data(), b.data(), z.data(),
               A_log.data(), dt_bias.data(), norm_w.data(),
               state_k.data(), out_k.data());

    bool ok = compare(out_r.data(), out_k.data(), seq_len * z_dim, "prefill out");
    ok &= compare(state_r.data(), state_k.data(), state_size, "prefill state");
    return ok;
}

// ---- Test 2: prefill with repeated value heads (Qwen3.5-4B style) ----
static bool test_prefill_repeated_value_heads() {
    const int num_heads = 2, num_v_heads = 4, k_dim = 8, v_dim = 8, seq_len = 5;
    const int qkv_dim = num_heads * k_dim;
    const int qkv_total = qkv_dim * 2 + num_v_heads * v_dim;
    const int z_dim = num_v_heads * v_dim;
    const int state_size = num_v_heads * k_dim * v_dim;
    unsigned int seed = 123;

    std::vector<float> qkv(seq_len * qkv_total), a(seq_len * num_v_heads),
                       b(seq_len * num_v_heads), z(seq_len * z_dim),
                       A_log(num_v_heads), dt_bias(num_v_heads), norm_w(v_dim);
    fill_rand(qkv.data(), qkv.size(), &seed);
    fill_rand(a.data(), a.size(), &seed);
    fill_rand(b.data(), b.size(), &seed);
    fill_rand(z.data(), z.size(), &seed);
    fill_rand(A_log.data(), A_log.size(), &seed);
    fill_rand(dt_bias.data(), dt_bias.size(), &seed);
    fill_rand(norm_w.data(), norm_w.size(), &seed);

    std::vector<float> state_k(state_size, 0), state_r(state_size, 0);
    std::vector<float> out_k(seq_len * z_dim, 0), out_r(seq_len * z_dim, 0);

    float scale = 1.f / std::sqrt((float)k_dim);
    ref_fused_gdn(qkv.data(), a.data(), b.data(), z.data(),
                  A_log.data(), dt_bias.data(), norm_w.data(),
                  state_r.data(), out_r.data(),
                  num_heads, k_dim, v_dim, seq_len,
                  true, 1e-6f, 1e-6f, scale, num_v_heads);

    run_kernel(/*prefill=*/true, num_heads, k_dim, v_dim, seq_len,
               qkv.data(), a.data(), b.data(), z.data(),
               A_log.data(), dt_bias.data(), norm_w.data(),
               state_k.data(), out_k.data(), num_v_heads);

    bool ok = compare(out_r.data(), out_k.data(), seq_len * z_dim, "repeat prefill out");
    ok &= compare(state_r.data(), state_k.data(), state_size, "repeat prefill state");
    return ok;
}

// ---- Test 3: decode (seq_len=1), compare against scalar reference ----
static bool test_decode_basic() {
    const int num_heads = 2, k_dim = 8, v_dim = 8;
    const int qkv_dim = num_heads * k_dim;
    const int qkv_total = qkv_dim * 3;
    const int z_dim = num_heads * v_dim;
    const int state_size = num_heads * k_dim * v_dim;
    unsigned int seed = 7;

    std::vector<float> qkv(qkv_total), a(num_heads), b(num_heads),
                       z(z_dim), A_log(num_heads), dt_bias(num_heads), norm_w(v_dim);
    fill_rand(qkv.data(), qkv.size(), &seed);
    fill_rand(a.data(), a.size(), &seed);
    fill_rand(b.data(), b.size(), &seed);
    fill_rand(z.data(), z.size(), &seed);
    fill_rand(A_log.data(), A_log.size(), &seed);
    fill_rand(dt_bias.data(), dt_bias.size(), &seed);
    fill_rand(norm_w.data(), norm_w.size(), &seed);

    // Non-zero initial state (simulates prior context).
    std::vector<float> state_k(state_size), state_r(state_size);
    fill_rand(state_k.data(), state_k.size(), &seed);
    std::copy(state_k.begin(), state_k.end(), state_r.begin());

    std::vector<float> out_k(z_dim, 0), out_r(z_dim, 0);

    float scale = 1.f / std::sqrt((float)k_dim);
    ref_fused_gdn(qkv.data(), a.data(), b.data(), z.data(),
                  A_log.data(), dt_bias.data(), norm_w.data(),
                  state_r.data(), out_r.data(),
                  num_heads, k_dim, v_dim, 1,
                  true, 1e-6f, 1e-6f, scale);

    run_kernel(/*prefill=*/false, num_heads, k_dim, v_dim, 1,
               qkv.data(), a.data(), b.data(), z.data(),
               A_log.data(), dt_bias.data(), norm_w.data(),
               state_k.data(), out_k.data());

    bool ok = compare(out_r.data(), out_k.data(), z_dim, "decode out");
    ok &= compare(state_r.data(), state_k.data(), state_size, "decode state");
    return ok;
}

// ---- Test 4: prefill then decode — state continuity ----
static bool test_prefill_then_decode() {
    const int num_heads = 4, k_dim = 16, v_dim = 16, seq_len = 8;
    const int qkv_dim = num_heads * k_dim;
    const int qkv_total = qkv_dim * 3;
    const int z_dim = num_heads * v_dim;
    const int state_size = num_heads * k_dim * v_dim;
    unsigned int seed = 99;

    std::vector<float> qkv(seq_len * qkv_total), a(seq_len * num_heads),
                       b(seq_len * num_heads), z(seq_len * z_dim),
                       A_log(num_heads), dt_bias(num_heads), norm_w(v_dim);
    fill_rand(qkv.data(), qkv.size(), &seed);
    fill_rand(a.data(), a.size(), &seed);
    fill_rand(b.data(), b.size(), &seed);
    fill_rand(z.data(), z.size(), &seed);
    fill_rand(A_log.data(), A_log.size(), &seed);
    fill_rand(dt_bias.data(), dt_bias.size(), &seed);
    fill_rand(norm_w.data(), norm_w.size(), &seed);

    // Prefill (both ref and kernel share the same zero-initial state).
    std::vector<float> state_k(state_size, 0), state_r(state_size, 0);
    std::vector<float> out_prefill_k(seq_len * z_dim, 0), out_prefill_r(seq_len * z_dim, 0);
    float scale = 1.f / std::sqrt((float)k_dim);

    ref_fused_gdn(qkv.data(), a.data(), b.data(), z.data(),
                  A_log.data(), dt_bias.data(), norm_w.data(),
                  state_r.data(), out_prefill_r.data(),
                  num_heads, k_dim, v_dim, seq_len,
                  true, 1e-6f, 1e-6f, scale);
    run_kernel(/*prefill=*/true, num_heads, k_dim, v_dim, seq_len,
               qkv.data(), a.data(), b.data(), z.data(),
               A_log.data(), dt_bias.data(), norm_w.data(),
               state_k.data(), out_prefill_k.data());

    bool ok = compare(out_prefill_r.data(), out_prefill_k.data(), seq_len * z_dim, "p→d prefill out");
    ok &= compare(state_r.data(), state_k.data(), state_size, "p→d prefill state");
    if (!ok) return false;

    // Now decode one more token using the updated state.
    std::vector<float> qkv_d(qkv_total), a_d(num_heads), b_d(num_heads),
                       z_d(z_dim);
    fill_rand(qkv_d.data(), qkv_d.size(), &seed);
    fill_rand(a_d.data(), a_d.size(), &seed);
    fill_rand(b_d.data(), b_d.size(), &seed);
    fill_rand(z_d.data(), z_d.size(), &seed);

    std::vector<float> out_d_k(z_dim, 0), out_d_r(z_dim, 0);
    ref_fused_gdn(qkv_d.data(), a_d.data(), b_d.data(), z_d.data(),
                  A_log.data(), dt_bias.data(), norm_w.data(),
                  state_r.data(), out_d_r.data(),
                  num_heads, k_dim, v_dim, 1,
                  true, 1e-6f, 1e-6f, scale);
    run_kernel(/*prefill=*/false, num_heads, k_dim, v_dim, 1,
               qkv_d.data(), a_d.data(), b_d.data(), z_d.data(),
               A_log.data(), dt_bias.data(), norm_w.data(),
               state_k.data(), out_d_k.data());

    ok &= compare(out_d_r.data(), out_d_k.data(), z_dim, "p→d decode out");
    ok &= compare(state_r.data(), state_k.data(), state_size, "p→d decode state");
    return ok;
}

int main() {
    CHECK(test_prefill_basic(),                "GDN prefill matches reference");
    CHECK(test_prefill_repeated_value_heads(), "GDN repeat-value-head prefill matches reference");
    CHECK(test_decode_basic(),                 "GDN decode matches reference");
    CHECK(test_prefill_then_decode(),          "GDN prefill→decode state continuity");
    printf(failures ? "\n%d FAILED\n" : "\nAll GDN tests passed!\n", failures);
    return failures;
}
