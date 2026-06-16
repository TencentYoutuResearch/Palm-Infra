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

    delete[] input_data;
    delete[] d2;

    if (failures == 0) {
        printf("\nAll execute tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
