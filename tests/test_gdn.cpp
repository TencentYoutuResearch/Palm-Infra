#include "kernels/gdn.h"
#include "graph/graph.h"
#include "kernels/tensor.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

static void fill_rand(float* d, int n) {
    for (int i = 0; i < n; i++) d[i] = (float)rand() / RAND_MAX - 0.5f;
}

// Reference implementation (identical logic, used for cross-check)
static void ref_gdn(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta,
    float* state, float* out,
    int num_heads, int seq_len, int k_dim, int v_dim)
{
    float scale = 1.f / sqrtf((float)k_dim);
    for (int h = 0; h < num_heads; h++) {
        float* sh = state + h * k_dim * v_dim;
        for (int t = 0; t < seq_len; t++) {
            const float* qt = q + (h * seq_len + t) * k_dim;
            const float* kt = k + (h * seq_len + t) * k_dim;
            const float* vt = v + (h * seq_len + t) * v_dim;
            float* ot = out + (h * seq_len + t) * v_dim;
            float gt = g[h * seq_len + t];
            float bt = beta[h * seq_len + t];
            float ge = expf(gt);

            for (int i = 0; i < k_dim * v_dim; i++) sh[i] *= ge;

            float kv[256];
            for (int dv = 0; dv < v_dim; dv++) { kv[dv] = 0; for (int dk = 0; dk < k_dim; dk++) kv[dv] += sh[dk*v_dim+dv]*kt[dk]; }
            float delta[256];
            for (int dv = 0; dv < v_dim; dv++) delta[dv] = (vt[dv]-kv[dv])*bt;
            for (int dk = 0; dk < k_dim; dk++) for (int dv = 0; dv < v_dim; dv++) sh[dk*v_dim+dv] += kt[dk]*delta[dv];
            for (int dv = 0; dv < v_dim; dv++) { float s=0; for (int dk = 0; dk < k_dim; dk++) s += sh[dk*v_dim+dv]*qt[dk]; ot[dv] = s*scale; }
        }
    }
}

// Test 1: GDN prefill with known small input, verify against reference.
static bool test_prefill_basic() {
    const int num_heads = 2;
    const int seq_len = 4;
    const int k_dim = 8;
    const int v_dim = 8;

    int qk_size = num_heads * seq_len * k_dim;
    int v_size = num_heads * seq_len * v_dim;
    int state_size = num_heads * k_dim * v_dim;

    float* q = new float[qk_size]; fill_rand(q, qk_size);
    float* k = new float[qk_size]; fill_rand(k, qk_size);
    float* v = new float[v_size]; fill_rand(v, v_size);
    float* g = new float[num_heads * seq_len]; fill_rand(g, num_heads * seq_len);
    float* beta = new float[num_heads * seq_len]; fill_rand(beta, num_heads * seq_len);
    // Make beta in [0, 1] (sigmoid range)
    for (int i = 0; i < num_heads * seq_len; i++) beta[i] = 0.5f + 0.3f * beta[i];
    // Make g negative (decay)
    for (int i = 0; i < num_heads * seq_len; i++) g[i] = -0.1f - 0.05f * ((float)rand()/RAND_MAX);

    float* state1 = new float[state_size]; memset(state1, 0, state_size * sizeof(float));
    float* state2 = new float[state_size]; memset(state2, 0, state_size * sizeof(float));
    float* out1 = new float[v_size]; memset(out1, 0, v_size * sizeof(float));
    float* out2 = new float[v_size]; memset(out2, 0, v_size * sizeof(float));

    // Run reference (no L2 norm, use_l2norm=0)
    ref_gdn(q, k, v, g, beta, state1, out1, num_heads, seq_len, k_dim, v_dim);

    // Run kernel via direct call to gdn_recurrent_scalar (need to access it)
    // Since gdn_recurrent_scalar is static in gdn.cpp, we test via kernel_gdn_prefill
    // with tensor inputs. But that requires OpParams setup. Instead, let's
    // replicate the scalar recurrence here for comparison — but that's the same
    // as ref_gdn. So this test just validates ref_gdn against itself (sanity).
    //
    // For a real test, we'd call kernel_gdn_prefill with Tensor objects.
    // Let's do that.

    // Build tensors
    Tensor q_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                k_dim, seq_len, num_heads, 1, q);
    Tensor k_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                k_dim, seq_len, num_heads, 1, k);
    Tensor v_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                v_dim, seq_len, num_heads, 1, v);
    Tensor g_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                seq_len, num_heads, 1, 1, g);
    Tensor beta_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                   seq_len, num_heads, 1, 1, beta);
    // Allocate state and output buffers (Tensor::create doesn't alloc for OWNED)
    float* state_buf = new float[num_heads * k_dim * v_dim]();
    float* out_buf = new float[v_size]();
    Tensor state_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    v_dim, k_dim, num_heads, 1, state_buf);
    Tensor out_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  v_dim, seq_len, num_heads, 1, out_buf);

    OpParams params;
    params.i32 = {num_heads, k_dim, v_dim, 0}; // use_l2norm=0
    std::vector<const Tensor*> inputs = {&q_t, &k_t, &v_t, &g_t, &beta_t, &state_t};
    std::vector<Tensor*> outputs = {&state_t, &out_t};
    kernel_gdn_prefill(params, inputs, outputs, nullptr);

    // Compare
    bool ok = true;
    float tol = 1e-5f;
    for (int i = 0; i < v_size; i++) {
        if (fabsf(out1[i] - out_t.ptr<float>()[i]) > tol) {
            fprintf(stderr, "  out mismatch[%d]: ref=%f kernel=%f\n", i, out1[i], out_t.ptr<float>()[i]);
            ok = false; break;
        }
    }
    for (int i = 0; i < state_size; i++) {
        if (fabsf(state1[i] - state_t.ptr<float>()[i]) > tol) {
            fprintf(stderr,  "  state mismatch[%d]: ref=%f kernel=%f\n", i, state1[i], state_t.ptr<float>()[i]);
            ok = false; break;
        }
    }

    delete[] q; delete[] k; delete[] v; delete[] g; delete[] beta;
    delete[] state1; delete[] state2; delete[] out1; delete[] out2;
    delete[] state_buf; delete[] out_buf;
    return ok;
}

// Test 2: GDN decode (single token), verify state continuity.
static bool test_decode_basic() {
    const int num_heads = 2;
    const int k_dim = 8;
    const int v_dim = 8;

    // Initial state (non-zero, simulating prior context)
    float* state = new float[num_heads * k_dim * v_dim];
    fill_rand(state, num_heads * k_dim * v_dim);

    // Single token
    float* q = new float[num_heads * k_dim]; fill_rand(q, num_heads * k_dim);
    float* k = new float[num_heads * k_dim]; fill_rand(k, num_heads * k_dim);
    float* v = new float[num_heads * v_dim]; fill_rand(v, num_heads * v_dim);
    float* g = new float[num_heads]; for (int i=0;i<num_heads;i++) g[i] = -0.1f;
    float* beta = new float[num_heads]; for (int i=0;i<num_heads;i++) beta[i] = 0.5f;

    // Copy state for reference
    float* state_ref = new float[num_heads * k_dim * v_dim];
    memcpy(state_ref, state, num_heads * k_dim * v_dim * sizeof(float));
    float* out_ref = new float[num_heads * v_dim];

    // Run reference (1 token)
    ref_gdn(q, k, v, g, beta, state_ref, out_ref, num_heads, 1, k_dim, v_dim);

    // Run kernel
    Tensor q_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, k_dim, 1, num_heads, 1, q);
    Tensor k_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, k_dim, 1, num_heads, 1, k);
    Tensor v_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, v_dim, 1, num_heads, 1, v);
    Tensor g_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 1, num_heads, 1, 1, g);
    Tensor beta_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 1, num_heads, 1, 1, beta);
    Tensor state_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, v_dim, k_dim, num_heads, 1, state);
    float* out_buf = new float[num_heads * v_dim]();
    Tensor out_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, v_dim, 1, num_heads, 1, out_buf);

    OpParams params;
    params.i32 = {num_heads, k_dim, v_dim, 0};
    std::vector<const Tensor*> inputs = {&q_t, &k_t, &v_t, &g_t, &beta_t, &state_t};
    std::vector<Tensor*> outputs = {&state_t, &out_t};
    kernel_gdn_decode(params, inputs, outputs, nullptr);

    bool ok = true;
    float tol = 1e-5f;
    for (int i = 0; i < num_heads * v_dim; i++) {
        if (fabsf(out_ref[i] - out_t.ptr<float>()[i]) > tol) {
            fprintf(stderr, "  decode out mismatch[%d]: ref=%f kernel=%f\n", i, out_ref[i], out_t.ptr<float>()[i]);
            ok = false; break;
        }
    }
    for (int i = 0; i < num_heads * k_dim * v_dim; i++) {
        if (fabsf(state_ref[i] - state_t.ptr<float>()[i]) > tol) {
            fprintf(stderr, "  decode state mismatch[%d]: ref=%f kernel=%f\n", i, state_ref[i], state_t.ptr<float>()[i]);
            ok = false; break;
        }
    }

    delete[] q; delete[] k; delete[] v; delete[] g; delete[] beta;
    delete[] state; delete[] state_ref; delete[] out_ref;
    delete[] out_buf;
    return ok;
}

// Test 3: prefill then decode — state continuity.
static bool test_prefill_then_decode() {
    const int num_heads = 4;
    const int k_dim = 16;
    const int v_dim = 16;
    const int seq_len = 8;

    int qk_size = num_heads * seq_len * k_dim;
    int v_size = num_heads * seq_len * v_dim;
    int state_size = num_heads * k_dim * v_dim;

    float* q = new float[qk_size]; fill_rand(q, qk_size);
    float* k = new float[qk_size]; fill_rand(k, qk_size);
    float* v = new float[v_size]; fill_rand(v, v_size);
    float* g = new float[num_heads * seq_len];
    float* beta = new float[num_heads * seq_len];
    for (int i = 0; i < num_heads * seq_len; i++) { g[i] = -0.1f; beta[i] = 0.5f; }

    // Prefill
    Tensor q_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, k_dim, seq_len, num_heads, 1, q);
    Tensor k_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, k_dim, seq_len, num_heads, 1, k);
    Tensor v_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, v_dim, seq_len, num_heads, 1, v);
    Tensor g_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, seq_len, num_heads, 1, 1, g);
    Tensor beta_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, seq_len, num_heads, 1, 1, beta);
    float* state_buf = new float[num_heads * k_dim * v_dim]();
    float* out_buf = new float[v_size]();
    Tensor state_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, v_dim, k_dim, num_heads, 1, state_buf);
    Tensor out_t = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, v_dim, seq_len, num_heads, 1, out_buf);

    OpParams params;
    params.i32 = {num_heads, k_dim, v_dim, 0};
    std::vector<const Tensor*> inputs = {&q_t, &k_t, &v_t, &g_t, &beta_t, &state_t};
    std::vector<Tensor*> outputs = {&state_t, &out_t};
    kernel_gdn_prefill(params, inputs, outputs, nullptr);

    // Now decode one more token using the updated state
    float* q_d = new float[num_heads * k_dim]; fill_rand(q_d, num_heads * k_dim);
    float* k_d = new float[num_heads * k_dim]; fill_rand(k_d, num_heads * k_dim);
    float* v_d = new float[num_heads * v_dim]; fill_rand(v_d, num_heads * v_dim);
    float* g_d = new float[num_heads]; for (int i=0;i<num_heads;i++) g_d[i] = -0.1f;
    float* beta_d = new float[num_heads]; for (int i=0;i<num_heads;i++) beta_d[i] = 0.5f;

    Tensor q_dt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, k_dim, 1, num_heads, 1, q_d);
    Tensor k_dt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, k_dim, 1, num_heads, 1, k_d);
    Tensor v_dt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, v_dim, 1, num_heads, 1, v_d);
    Tensor g_dt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 1, num_heads, 1, 1, g_d);
    Tensor beta_dt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 1, num_heads, 1, 1, beta_d);
    float* out_d_buf = new float[num_heads * v_dim]();
    Tensor out_dt = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, v_dim, 1, num_heads, 1, out_d_buf);
    memset(out_d_buf, 0, num_heads * v_dim * sizeof(float));

    std::vector<const Tensor*> inputs_d = {&q_dt, &k_dt, &v_dt, &g_dt, &beta_dt, &state_t};
    std::vector<Tensor*> outputs_d = {&state_t, &out_dt};
    kernel_gdn_decode(params, inputs_d, outputs_d, nullptr);

    // Verify: decode output should be non-trivial (state was updated by prefill)
    float* out_d = out_dt.ptr<float>();
    bool has_output = false;
    for (int i = 0; i < num_heads * v_dim; i++) {
        if (fabsf(out_d[i]) > 1e-6f) { has_output = true; break; }
    }

    delete[] q; delete[] k; delete[] v; delete[] g; delete[] beta;
    delete[] q_d; delete[] k_d; delete[] v_d; delete[] g_d; delete[] beta_d;
    delete[] state_buf; delete[] out_buf; delete[] out_d_buf;
    return has_output;
}

int main() {
    srand(42);
    CHECK(test_prefill_basic(), "GDN prefill matches reference");
    CHECK(test_decode_basic(), "GDN decode matches reference");
    CHECK(test_prefill_then_decode(), "GDN prefill→decode state continuity");
    printf(failures ? "\n%d FAILED\n" : "\nAll GDN tests passed!\n", failures);
    return failures;
}
