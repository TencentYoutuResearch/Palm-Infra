#include "graph/graph.h"
#include <cstdio>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

int main() {
    // ---- empty graph ----
    Graph g;
    CHECK(g.nodes.empty(), "empty graph has no nodes");
    CHECK(g.graph_inputs.empty(), "no inputs");
    CHECK(g.graph_outputs.empty(), "no outputs");

    // ---- add nodes manually ----
    GraphNode in0;
    in0.id      = 0;
    in0.op_type = OpType::INPUT;
    in0.out_shape[0] = 2048;
    in0.out_prec = Precision::FP16;
    g.nodes.push_back(in0);

    GraphNode matmul;
    matmul.id      = 1;
    matmul.op_type = OpType::MATMUL;
    matmul.inputs  = {0};
    matmul.out_shape[0] = 4096;
    matmul.out_prec = Precision::FP16;
    matmul.params.i32 = {2048, 4096};  // K, N
    g.nodes.push_back(matmul);

    CHECK(g.nodes.size() == 2, "two nodes");
    CHECK(g.nodes[0].op_type == OpType::INPUT, "first is INPUT");
    CHECK(g.nodes[1].op_type == OpType::MATMUL, "second is MATMUL");
    CHECK(g.nodes[1].inputs[0] == 0, "matmul input is node 0");

    // ---- OpParams ----
    OpParams p;
    CHECK(p.empty(), "empty params is empty");
    p.i32 = {1, 2, 3};
    CHECK(!p.empty(), "non-empty params");
    CHECK(p.i32[0] == 1 && p.i32[2] == 3, "i32 access");
    p.f32 = {0.5f, 1.0f};
    CHECK(p.f32[1] == 1.0f, "f32 access");
    p.str = {"hello"};
    CHECK(p.str[0] == "hello", "str access");

    // ---- graph_params helpers ----
    CHECK(graph_params::get_i32(p, 0) == 1, "get_i32 idx 0");
    CHECK(graph_params::get_i32(p, 99) == 0, "get_i32 out of range → default");
    CHECK(graph_params::get_f32(p, 0) == 0.5f, "get_f32 idx 0");
    CHECK(graph_params::get_f32(p, 99) == 0.f, "get_f32 out of range → default");
    CHECK(graph_params::get_str(p, 0) == "hello", "get_str idx 0");
    CHECK(graph_params::get_str(p, 99).empty(), "get_str out of range → empty");

    // ---- graph_inputs / graph_outputs ----
    g.graph_inputs = {0};
    g.graph_outputs = {1};
    CHECK(g.input_node(0) == &g.nodes[0], "input_node(0)");
    CHECK(g.input_node(1) == nullptr, "input_node(1) out of range");
    CHECK(g.output_node(0) == &g.nodes[1], "output_node(0)");

    // ---- Runtime ----
    CHECK(g.runtime.tensors.empty(), "runtime tensors initially empty");
    g.runtime.tensors.resize(2);
    CHECK(g.runtime.tensors.size() == 2, "resized tensors");
    CHECK(g.runtime.tensors[0].data == nullptr, "tensor 0 unallocated");
    CHECK(g.runtime.tensors[1].data == nullptr, "tensor 1 unallocated");

    // ---- SDPA params ----
    GraphNode sdpa;
    sdpa.op_type = OpType::SDPA;
    sdpa.params.i32 = {
        2,     // kv_cache
        1,     // causal
        128,   // num_heads
        16,    // num_kv_heads
        192,   // head_dim
        128,   // v_head_dim
    };
    sdpa.params.f32 = {0.0721688f};  // scale
    CHECK(graph_params::get_i32(sdpa.params, 0) == 2, "SDPA kv_cache=2");
    CHECK(graph_params::get_i32(sdpa.params, 2) == 128, "SDPA num_heads");
    CHECK(graph_params::get_f32(sdpa.params, 0) == 0.0721688f, "SDPA scale");

    // ---- all OpType values ----
    CHECK((int)OpType::INPUT == 0, "INPUT=0");
    CHECK((int)OpType::MATMUL == 10, "MATMUL=10");
    CHECK((int)OpType::SDPA == 50, "SDPA=50");
    CHECK((int)OpType::SDPA_MLA == 51, "SDPA_MLA=51");

    if (failures == 0) {
        printf("\nAll graph tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
