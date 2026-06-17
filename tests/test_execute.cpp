#include "graph/execute.h"
#include <cstdio>
#include <cstring>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

int main() {
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

    // ---- prepare ----
    prepare_execution(ctx);
    // Phase 1: release_queue is empty (no-release mode)
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

    prepare_execution(ctx2);
    execute_graph(ctx2);

    auto& tp = g2.runtime.tensors[1];
    CHECK(tp.shape[0] == 4 && tp.shape[1] == 3, "permuted to 4x3");
    // permute(1,0,2,3) swaps dim0↔dim1: stride[0] was 4, stride[1] was 12
    // after swap: stride[0]=12, stride[1]=4
    CHECK(tp.stride[0] == 12 && tp.stride[1] == 4, "permute strides correct");
    CHECK(tp.at<float>(0, 1) == 2.0f, "permute data access via stride");

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
    g3.graph_outputs = {3};
    g3.runtime.tensors.resize(4);

    float* d3 = new float[8]{0, 1, 2, 3, 4, 5, 6, 7};
    g3.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::OWNED, 4, 2, 1, 1, d3);

    BufferPool pool3;
    ExecContext ctx3;
    ctx3.graph = &g3;
    ctx3.pool = &pool3;

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

    prepare_execution(ctx4);
    execute_graph(ctx4);

    auto& tr = g4.runtime.tensors[2];
    CHECK(tr.data != g4.runtime.tensors[1].data, "reshape after permute materializes new buffer");
    CHECK(tr.at<float>(0, 0) == 0.0f && tr.at<float>(1, 0) == 2.0f &&
          tr.at<float>(0, 1) == 4.0f && tr.at<float>(1, 1) == 1.0f &&
          tr.at<float>(0, 2) == 3.0f && tr.at<float>(1, 2) == 5.0f,
          "reshape after permute preserves logical element order");

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
