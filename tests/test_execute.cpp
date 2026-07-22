#include "graph/execute.h"
#include "engine/backend.h"
#include <cstdio>
#include <cstring>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

int main() {
    CPUBackend cpu_backend;  // shared across all ExecContexts below
    // ---- build a simple chain: INPUT → RESHAPE → (end) ----
    Graph g;

    GraphNode in0;
    in0.id = 0;
    in0.op_type = OpType::INPUT;
    in0.out_shape[0] = 8;
    in0.out_shape[1] = 4;
    in0.out_prec = Precision::FP32;
    g.nodes.push_back(in0);

    GraphNode reshape;
    reshape.id = 1;
    reshape.op_type = OpType::RESHAPE;
    reshape.inputs = {0};
    reshape.out_shape[0] = 4;
    reshape.out_shape[1] = 8;
    reshape.out_prec = Precision::FP32;
    reshape.params.i32 = {4, 8, 1, 1};
    g.nodes.push_back(reshape);

    g.graph_inputs  = {0};
    g.graph_outputs = {1};

    // set up runtime
    g.runtime.tensors.resize(2);

    // allocate input data
    float* input_data = new float[32];
    for (int i = 0; i < 32; i++) input_data[i] = (float)i;

    auto& t0 = g.runtime.tensors[0];
    t0 = Tensor::create(Precision::FP32, MemoryType::OWNED, 8, 4, 1, 1, input_data);

    BufferPool pool;

    ExecContext ctx;
    ctx.graph = &g;
    ctx.pool  = &pool;
    ctx.backend = &cpu_backend;

    // ---- prepare ----
    prepare_execution(ctx);
    CHECK(ctx.release_queue.size() == 2, "release_queue sized");

    // ---- execute ----
    execute_graph(ctx);

    auto& t1 = g.runtime.tensors[1];
    CHECK(t1.data != nullptr, "output allocated");
    CHECK(t1.shape[0] == 4 && t1.shape[1] == 8, "reshaped to 4x8");
    // reshape is a zero-copy view — it copies mem_type from the input (OWNED)
    CHECK(t1.mem_type == t0.mem_type, "output shares input mem_type (view)");
    CHECK(t1.prec == Precision::FP32, "output precision FP32");

    // verify data is shared (zero-copy reshape)
    CHECK(t1.data == t0.data, "reshape shares data pointer");
    CHECK(pool.active_bytes() == 0, "zero-copy reshape does not allocate pool buffer");

    // verify pool active dropped (input was released after node 1)
    // input was OWNED, not POOLED, so it won't be released by the engine
    // reshape output is POOLED and still held — just check no crash

    // ---- test: permute ----
    Graph g2;
    GraphNode in1;
    in1.id = 0;
    in1.op_type = OpType::INPUT;
    in1.out_shape[0] = 3;
    in1.out_shape[1] = 4;
    in1.out_prec = Precision::FP32;
    g2.nodes.push_back(in1);

    GraphNode perm;
    perm.id = 1;
    perm.op_type = OpType::PERMUTE;
    perm.inputs = {0};
    perm.out_shape[0] = 4;
    perm.out_shape[1] = 3;
    perm.out_prec = Precision::FP32;
    perm.params.i32 = {1, 0, 2, 3};
    g2.nodes.push_back(perm);

    g2.graph_inputs  = {0};
    g2.graph_outputs = {1};
    g2.runtime.tensors.resize(2);

    float* d2 = new float[12]{1,2,3, 4,5,6, 7,8,9, 10,11,12};
    g2.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::OWNED, 3, 4, 1, 1, d2);

    BufferPool pool2;
    ExecContext ctx2;
    ctx2.graph = &g2;
    ctx2.pool  = &pool2;
    ctx2.backend = &cpu_backend;

    prepare_execution(ctx2);
    execute_graph(ctx2);

    auto& tp = g2.runtime.tensors[1];
    CHECK(tp.shape[0] == 4 && tp.shape[1] == 3, "permuted to 4x3");
    // permute(1,0,2,3) swaps dim0↔dim1: stride[0] was 4, stride[1] was 12
    // after swap: stride[0]=12, stride[1]=4
    CHECK(tp.stride[0] == 12 && tp.stride[1] == 4, "permute strides correct");
    CHECK(tp.at<float>(0, 1) == 2.0f, "permute data access via stride");
    CHECK(pool2.active_bytes() == 0, "zero-copy permute does not allocate pool buffer");

    // ---- test: per-channel [D, 1] broadcast across sequence [D, S] ----
    // RWKV time-mix parameters use this shape.  It must not advance the
    // parameter pointer for every token row.
    {
        Graph gb;
        for (uint32_t id = 0; id < 2; ++id) {
            GraphNode in;
            in.id = id;
            in.op_type = OpType::INPUT;
            in.out_shape[0] = 4;
            in.out_shape[1] = id == 0 ? 3 : 1;
            in.out_prec = Precision::FP32;
            gb.nodes.push_back(in);
        }
        GraphNode mul;
        mul.id = 2; mul.op_type = OpType::MUL; mul.inputs = {0, 1};
        mul.out_shape[0] = 4; mul.out_shape[1] = 3; mul.out_prec = Precision::FP32;
        gb.nodes.push_back(mul);
        GraphNode add;
        add.id = 3; add.op_type = OpType::ADD; add.inputs = {0, 1};
        add.out_shape[0] = 4; add.out_shape[1] = 3; add.out_prec = Precision::FP32;
        gb.nodes.push_back(add);
        gb.graph_inputs = {0, 1};
        gb.graph_outputs = {2, 3};
        gb.runtime.tensors.resize(4);
        float values[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        float channels[4] = {10, 20, 30, 40};
        gb.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                                4, 3, 1, 1, values);
        gb.runtime.tensors[1] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                                4, 1, 1, 1, channels);
        BufferPool pool_b;
        ExecContext ctx_b;
        ctx_b.graph = &gb; ctx_b.pool = &pool_b; ctx_b.backend = &cpu_backend;
        prepare_execution(ctx_b);
        execute_graph(ctx_b);
        CHECK(gb.runtime.tensors[2].at<float>(0, 2) == 90.0f &&
              gb.runtime.tensors[2].at<float>(3, 2) == 480.0f,
              "MUL broadcasts channel vector across sequence");
        CHECK(gb.runtime.tensors[3].at<float>(0, 2) == 19.0f &&
              gb.runtime.tensors[3].at<float>(3, 2) == 52.0f,
              "ADD broadcasts channel vector across sequence");
    }

    // ---- test: fused RWKV time mix x + shift * channel_mix ----
    {
        Graph gm;
        for (uint32_t id = 0; id < 3; ++id) {
            GraphNode in;
            in.id = id;
            in.op_type = OpType::INPUT;
            in.out_shape[0] = 4;
            in.out_shape[1] = id == 2 ? 1 : 2;
            in.out_prec = Precision::FP32;
            gm.nodes.push_back(in);
        }
        GraphNode mix;
        mix.id = 3; mix.op_type = OpType::RWKV_MIX; mix.inputs = {0, 1, 2};
        mix.out_shape[0] = 4; mix.out_shape[1] = 2;
        mix.out_prec = Precision::FP32;
        gm.nodes.push_back(mix);
        gm.graph_inputs = {0, 1, 2};
        gm.graph_outputs = {3};
        gm.runtime.tensors.resize(4);
        float x[8] = {1,2,3,4, 5,6,7,8};
        float shift[8] = {2,2,2,2, 4,4,4,4};
        float channels[4] = {0.5f,1.f,1.5f,2.f};
        gm.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                               4, 2, 1, 1, x);
        gm.runtime.tensors[1] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                               4, 2, 1, 1, shift);
        gm.runtime.tensors[2] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                               4, 1, 1, 1, channels);
        BufferPool pool_m;
        ExecContext ctx_m;
        ctx_m.graph = &gm; ctx_m.pool = &pool_m; ctx_m.backend = &cpu_backend;
        prepare_execution(ctx_m);
        execute_graph(ctx_m);
        CHECK(gm.runtime.tensors[3].at<float>(0, 0) == 2.f &&
              gm.runtime.tensors[3].at<float>(3, 0) == 8.f &&
              gm.runtime.tensors[3].at<float>(0, 1) == 7.f &&
              gm.runtime.tensors[3].at<float>(3, 1) == 16.f,
              "RWKV_MIX fuses channel-wise multiply-add");
    }

    // ---- test: slice(dim=0) + concat(dim=0) preserve row layout ----
    Graph g3;
    GraphNode in2;
    in2.id = 0;
    in2.op_type = OpType::INPUT;
    in2.out_shape[0] = 4;
    in2.out_shape[1] = 2;
    in2.out_prec = Precision::FP32;
    g3.nodes.push_back(in2);

    GraphNode slice0;
    slice0.id = 1;
    slice0.op_type = OpType::SLICE;
    slice0.inputs = {0};
    slice0.out_shape[0] = 2;
    slice0.out_shape[1] = 2;
    slice0.out_prec = Precision::FP32;
    slice0.params.i32 = {0, 0, 2};
    g3.nodes.push_back(slice0);

    GraphNode slice1;
    slice1.id = 2;
    slice1.op_type = OpType::SLICE;
    slice1.inputs = {0};
    slice1.out_shape[0] = 2;
    slice1.out_shape[1] = 2;
    slice1.out_prec = Precision::FP32;
    slice1.params.i32 = {0, 2, 2};
    g3.nodes.push_back(slice1);

    GraphNode concat;
    concat.id = 3;
    concat.op_type = OpType::CONCAT;
    concat.inputs = {1, 2};
    concat.out_shape[0] = 4;
    concat.out_shape[1] = 2;
    concat.out_prec = Precision::FP32;
    concat.params.i32 = {0};
    g3.nodes.push_back(concat);

    g3.graph_inputs = {0};
    // Keep slices as explicit outputs because this test inspects them after
    // execute_graph(); non-output internal views may be cleared by liveness.
    g3.graph_outputs = {1, 2, 3};
    g3.runtime.tensors.resize(4);

    float* d3 = new float[8]{0, 1, 2, 3, 4, 5, 6, 7};
    g3.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::OWNED, 4, 2, 1, 1, d3);

    BufferPool pool3;
    ExecContext ctx3;
    ctx3.graph = &g3;
    ctx3.pool = &pool3;
    ctx3.backend = &cpu_backend;
    ctx3.reuse_static_workspace = true;

    prepare_execution(ctx3);
    execute_graph(ctx3);

    auto& ts0 = g3.runtime.tensors[1];
    auto& ts1 = g3.runtime.tensors[2];
    auto& tc = g3.runtime.tensors[3];
    CHECK(ts0.stride[1] == g3.runtime.tensors[0].stride[1], "slice keeps parent row stride");
    CHECK(ts0.at<float>(0, 1) == 4.0f && ts0.at<float>(1, 1) == 5.0f, "slice row 1 reads original left half");
    CHECK(ts1.at<float>(0, 1) == 6.0f && ts1.at<float>(1, 1) == 7.0f, "slice row 1 reads original right half");
    CHECK(tc.at<float>(0, 0) == 0.0f && tc.at<float>(1, 0) == 1.0f &&
          tc.at<float>(2, 0) == 2.0f && tc.at<float>(3, 0) == 3.0f,
          "concat row 0 preserves dim0 ordering");
    CHECK(tc.at<float>(0, 1) == 4.0f && tc.at<float>(1, 1) == 5.0f &&
          tc.at<float>(2, 1) == 6.0f && tc.at<float>(3, 1) == 7.0f,
          "concat row 1 preserves dim0 ordering");
    size_t concat_active = pool3.active_bytes();
    size_t concat_acquires = pool3.acquire_count();
    CHECK(concat_active > 0, "concat materializes pooled output");
    execute_graph(ctx3);
    CHECK(pool3.active_bytes() == concat_active, "concat repeated execute does not grow active pool bytes");
    CHECK(pool3.acquire_count() == concat_acquires, "static workspace reuses concat buffer without reacquire");

    // ---- test: dynamic same-shape workspace reuse ----
    {
        Graph gd;
        GraphNode a;
        a.id = 0;
        a.op_type = OpType::INPUT;
        a.out_shape[0] = 4;
        a.out_shape[1] = 1;
        a.dim_expr[1].kind = DIM_SEQ;
        a.out_prec = Precision::FP32;
        gd.nodes.push_back(a);

        GraphNode b;
        b.id = 1;
        b.op_type = OpType::INPUT;
        b.out_shape[0] = 4;
        b.out_shape[1] = 1;
        b.dim_expr[1].kind = DIM_SEQ;
        b.out_prec = Precision::FP32;
        gd.nodes.push_back(b);

        GraphNode add;
        add.id = 2;
        add.op_type = OpType::ADD;
        add.inputs = {0, 1};
        add.out_shape[0] = 4;
        add.out_shape[1] = 1;
        add.dim_expr[1].kind = DIM_SEQ;
        add.out_prec = Precision::FP32;
        gd.nodes.push_back(add);

        gd.graph_inputs = {0, 1};
        gd.graph_outputs = {2};
        gd.runtime.tensors.resize(3);

        float da[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        float db[12] = {10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30};
        gd.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 3, 1, 1, da);
        gd.runtime.tensors[1] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 3, 1, 1, db);

        BufferPool pool_d;
        ExecContext ctx_d;
        ctx_d.graph = &gd;
        ctx_d.pool = &pool_d;
        ctx_d.backend = &cpu_backend;
        ctx_d.reuse_same_shape_workspace = true;
        ctx_d.runtime_seq_len = 3;

        prepare_execution(ctx_d);
        execute_graph(ctx_d);
        size_t dyn_active = pool_d.active_bytes();
        size_t dyn_acquires = pool_d.acquire_count();
        CHECK(gd.runtime.tensors[2].shape[1] == 3, "dynamic output shape follows runtime seq");
        CHECK(gd.runtime.tensors[2].at<float>(0, 2) == 38.0f,
              "dynamic same-shape output correct");

        execute_graph(ctx_d);
        CHECK(pool_d.active_bytes() == dyn_active,
              "dynamic same-shape workspace keeps active bytes stable");
        CHECK(pool_d.acquire_count() == dyn_acquires,
              "dynamic same-shape workspace reuses buffer without reacquire");

        gd.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 2, 1, 1, da);
        gd.runtime.tensors[1] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 2, 1, 1, db);
        ctx_d.runtime_seq_len = 2;
        execute_graph(ctx_d);
        CHECK(gd.runtime.tensors[2].shape[1] == 2, "dynamic changed-shape output updates shape");
        CHECK(pool_d.acquire_count() > dyn_acquires,
              "dynamic changed-shape workspace reacquires buffer");
    }

    // ---- test: view-aware liveness releases materialized producer after view consumer ----
    {
        Graph gl;
        GraphNode a;
        a.id = 0;
        a.op_type = OpType::INPUT;
        a.out_shape[0] = 4;
        a.out_prec = Precision::FP32;
        gl.nodes.push_back(a);

        GraphNode b;
        b.id = 1;
        b.op_type = OpType::INPUT;
        b.out_shape[0] = 4;
        b.out_prec = Precision::FP32;
        gl.nodes.push_back(b);

        GraphNode add1;
        add1.id = 2;
        add1.op_type = OpType::ADD;
        add1.inputs = {0, 1};
        add1.out_shape[0] = 4;
        add1.out_prec = Precision::FP32;
        gl.nodes.push_back(add1);

        GraphNode reshape_l;
        reshape_l.id = 3;
        reshape_l.op_type = OpType::RESHAPE;
        reshape_l.inputs = {2};
        reshape_l.out_shape[0] = 2;
        reshape_l.out_shape[1] = 2;
        reshape_l.out_prec = Precision::FP32;
        reshape_l.params.i32 = {2, 2, 1, 1};
        gl.nodes.push_back(reshape_l);

        GraphNode add2;
        add2.id = 4;
        add2.op_type = OpType::ADD;
        add2.inputs = {3, 1};
        add2.out_shape[0] = 4;
        add2.out_prec = Precision::FP32;
        gl.nodes.push_back(add2);

        gl.graph_inputs = {0, 1};
        gl.graph_outputs = {4};
        gl.runtime.tensors.resize(5);

        float da[4] = {1, 2, 3, 4};
        float db[4] = {10, 20, 30, 40};
        gl.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1, da);
        gl.runtime.tensors[1] = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1, db);

        BufferPool pool_l;
        ExecContext ctx_l;
        ctx_l.graph = &gl;
        ctx_l.pool = &pool_l;
        ctx_l.backend = &cpu_backend;

        prepare_execution(ctx_l);
        CHECK(!ctx_l.release_queue.empty(), "liveness release_queue built");
        execute_graph(ctx_l);

        Tensor& out_l = gl.runtime.tensors[4];
        CHECK(out_l.at<float>(0) == 21.0f && out_l.at<float>(3) == 84.0f,
              "liveness graph output correct");
        CHECK(pool_l.active_bytes() == BufferPool::MIN_BUCKET,
              "view-aware liveness releases intermediate producer");
    }

    // ---- test: profiler counts executed ops ----
    {
        Graph gp;
        GraphNode in;
        in.id = 0;
        in.op_type = OpType::INPUT;
        in.out_shape[0] = 4;
        in.out_prec = Precision::FP32;
        gp.nodes.push_back(in);

        GraphNode reshape_p;
        reshape_p.id = 1;
        reshape_p.op_type = OpType::RESHAPE;
        reshape_p.inputs = {0};
        reshape_p.out_shape[0] = 2;
        reshape_p.out_shape[1] = 2;
        reshape_p.out_prec = Precision::FP32;
        reshape_p.params.i32 = {2, 2, 1, 1};
        gp.nodes.push_back(reshape_p);

        gp.graph_inputs = {0};
        gp.graph_outputs = {1};
        gp.runtime.tensors.resize(2);

        float* dp = new float[4]{1, 2, 3, 4};
        gp.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::OWNED, 4, 1, 1, 1, dp);

        BufferPool poolp;
        ExecContext ctxp;
        ctxp.graph = &gp;
        ctxp.pool = &poolp;
        ctxp.profile_enabled = true;
        ctxp.backend = &cpu_backend;

        prepare_execution(ctxp);
        execute_graph(ctxp);

        CHECK(ctxp.profile_stats.size() == 2, "profile stats sized to node count");
        CHECK(ctxp.profile_stats[1].calls == 1, "profile counts reshape call");
        CHECK(ctxp.profile_stats[1].op_type == OpType::RESHAPE, "profile records op type");

        delete[] dp;
    }

    // ---- test: reshape after permute materializes logical order ----
    Graph g4;
    GraphNode in3;
    in3.id = 0;
    in3.op_type = OpType::INPUT;
    in3.out_shape[0] = 2;
    in3.out_shape[1] = 3;
    in3.out_prec = Precision::FP32;
    g4.nodes.push_back(in3);

    GraphNode perm2;
    perm2.id = 1;
    perm2.op_type = OpType::PERMUTE;
    perm2.inputs = {0};
    perm2.out_shape[0] = 3;
    perm2.out_shape[1] = 2;
    perm2.out_prec = Precision::FP32;
    perm2.params.i32 = {1, 0, 2, 3};
    g4.nodes.push_back(perm2);

    GraphNode reshape2;
    reshape2.id = 2;
    reshape2.op_type = OpType::RESHAPE;
    reshape2.inputs = {1};
    reshape2.out_shape[0] = 2;
    reshape2.out_shape[1] = 3;
    reshape2.out_prec = Precision::FP32;
    reshape2.params.i32 = {2, 3, 1, 1};
    g4.nodes.push_back(reshape2);

    g4.graph_inputs = {0};
    g4.graph_outputs = {2};
    g4.runtime.tensors.resize(3);

    float* d4 = new float[6]{0, 1, 2, 3, 4, 5};
    g4.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::OWNED, 2, 3, 1, 1, d4);

    BufferPool pool4;
    ExecContext ctx4;
    ctx4.graph = &g4;
    ctx4.pool = &pool4;
    ctx4.backend = &cpu_backend;
    ctx4.reuse_static_workspace = true;

    prepare_execution(ctx4);
    execute_graph(ctx4);

    auto& tr = g4.runtime.tensors[2];
    CHECK(tr.data != g4.runtime.tensors[1].data, "reshape after permute materializes new buffer");
    CHECK(tr.at<float>(0, 0) == 0.0f && tr.at<float>(1, 0) == 2.0f &&
          tr.at<float>(0, 1) == 4.0f && tr.at<float>(1, 1) == 1.0f &&
          tr.at<float>(0, 2) == 3.0f && tr.at<float>(1, 2) == 5.0f,
          "reshape after permute preserves logical element order");
    size_t reshape_active = pool4.active_bytes();
    size_t reshape_acquires = pool4.acquire_count();
    CHECK(reshape_active > 0, "non-contiguous reshape materializes pooled output");
    execute_graph(ctx4);
    CHECK(pool4.active_bytes() == reshape_active, "non-contiguous reshape repeated execute does not grow active pool bytes");
    CHECK(pool4.acquire_count() == reshape_acquires, "static workspace reuses materialized reshape buffer without reacquire");

    delete[] input_data;
    delete[] d2;
    delete[] d3;
    delete[] d4;

    if (failures == 0) {
        printf("\nAll execute tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
