#include "graph/graph.h"
#include "graph/execute.h"
#include <cstdio>

int main() {
    printf("PROJECT_NAME — placeholder example\n");

    // Create a trivial graph: INPUT → RESHAPE
    Graph g;
    GraphNode in;
    in.id = 0;
    in.op_type = OpType::INPUT;
    in.out_shape[0] = 8;
    in.out_prec = Precision::FP32;
    g.nodes.push_back(in);

    GraphNode reshape;
    reshape.id = 1;
    reshape.op_type = OpType::RESHAPE;
    reshape.inputs = {0};
    reshape.out_shape[0] = 4;
    reshape.out_shape[1] = 2;
    reshape.out_prec = Precision::FP32;
    reshape.params.i32 = {4, 2, 1, 1};
    g.nodes.push_back(reshape);

    g.graph_inputs = {0};
    g.graph_outputs = {1};

    g.runtime.tensors.resize(2);

    float* data = new float[8]{1,2,3,4,5,6,7,8};
    g.runtime.tensors[0] = Tensor::create(Precision::FP32, MemoryType::OWNED, 8, 1, 1, 1, data);

    BufferPool pool;
    ExecContext ctx;
    ctx.graph = &g;
    ctx.pool = &pool;

    prepare_execution(ctx);
    execute_graph(ctx);

    auto& out = g.runtime.tensors[1];
    printf("Output shape: [%lld, %lld]\n", out.shape[0], out.shape[1]);
    for (int i = 0; i < 8; i++) {
        printf("  out[%d] = %f\n", i, out.ptr<float>()[i]);
    }

    delete[] data;
    return 0;
}
