#include "graph/graph.h"
#include <cstdio>
#include <cstring>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

int main() {
    // ---- build a test graph ----
    Graph g;

    // input node
    GraphNode in0;
    in0.id      = 0;
    in0.op_type = OpType::INPUT;
    in0.out_shape[0] = 2048;
    in0.out_prec     = Precision::FP16;
    g.nodes.push_back(in0);

    // matmul node
    GraphNode matmul;
    matmul.id      = 1;
    matmul.op_type = OpType::MATMUL;
    matmul.inputs  = {0};
    matmul.out_shape[0] = 4096;
    matmul.out_prec     = Precision::FP16;
    matmul.params.i32   = {2048, 4096};  // K, N
    matmul.params.f32   = {1.0f};
    matmul.params.str   = {"layer_0_q_proj.weights"};
    g.nodes.push_back(matmul);

    // rms_norm node
    GraphNode rms;
    rms.id      = 2;
    rms.op_type = OpType::RMS_NORM;
    rms.inputs  = {1};
    rms.out_shape[0] = 4096;
    rms.out_prec     = Precision::FP16;
    rms.params.f32   = {1e-6f};  // eps
    g.nodes.push_back(rms);

    // SDPA node
    GraphNode sdpa;
    sdpa.id      = 3;
    sdpa.op_type = OpType::SDPA;
    sdpa.inputs  = {2};
    sdpa.out_shape[0] = 128;
    sdpa.out_prec     = Precision::FP16;
    sdpa.params.i32   = {2, 1, 128, 16, 192, 128};  // kv_cache, causal, heads, kv_heads, hd, vd
    sdpa.params.f32   = {0.0721688f};
    g.nodes.push_back(sdpa);

    g.graph_inputs  = {0};
    g.graph_outputs = {3};

    CHECK(g.nodes.size() == 4, "4 nodes in test graph");

    // ---- save ----
    CHECK(graph_save(g, "/tmp/test_graph.gllm"), "save graph");

    // ---- load ----
    Graph g2;
    CHECK(graph_load(g2, "/tmp/test_graph.gllm"), "load graph");
    CHECK(g2.nodes.size() == 4, "loaded 4 nodes");
    CHECK(g2.graph_inputs.size() == 1, "1 graph input");
    CHECK(g2.graph_outputs.size() == 1, "1 graph output");
    CHECK(g2.graph_inputs[0] == 0, "graph_input[0]==0");
    CHECK(g2.graph_outputs[0] == 3, "graph_output[0]==3");

    // ---- verify node 0 (INPUT) ----
    CHECK(g2.nodes[0].id == 0, "node0 id");
    CHECK(g2.nodes[0].op_type == OpType::INPUT, "node0 INPUT");
    CHECK(g2.nodes[0].out_shape[0] == 2048, "node0 shape");
    CHECK(g2.nodes[0].out_prec == Precision::FP16, "node0 FP16");

    // ---- verify node 1 (MATMUL) ----
    CHECK(g2.nodes[1].op_type == OpType::MATMUL, "node1 MATMUL");
    CHECK(g2.nodes[1].inputs.size() == 1, "node1 1 input");
    CHECK(g2.nodes[1].inputs[0] == 0, "node1 input is node0");
    CHECK(g2.nodes[1].params.i32.size() == 2, "node1 2 i32 params");
    CHECK(g2.nodes[1].params.i32[0] == 2048, "node1 K=2048");
    CHECK(g2.nodes[1].params.i32[1] == 4096, "node1 N=4096");
    CHECK(g2.nodes[1].params.f32[0] == 1.0f, "node1 f32 param");
    CHECK(g2.nodes[1].params.str[0] == "layer_0_q_proj.weights", "node1 str param");

    // ---- verify node 2 (RMS_NORM) ----
    CHECK(g2.nodes[2].op_type == OpType::RMS_NORM, "node2 RMS_NORM");
    CHECK(g2.nodes[2].params.f32[0] == 1e-6f, "node2 eps");

    // ---- verify node 3 (SDPA) ----
    CHECK(g2.nodes[3].op_type == OpType::SDPA, "node3 SDPA");
    CHECK(g2.nodes[3].params.i32.size() == 6, "node3 6 i32 params");
    CHECK(g2.nodes[3].params.i32[0] == 2, "node3 kv_cache=2");
    CHECK(g2.nodes[3].params.i32[1] == 1, "node3 causal=1");
    CHECK(g2.nodes[3].params.i32[4] == 192, "node3 head_dim=192");
    CHECK(g2.nodes[3].params.f32[0] == 0.0721688f, "node3 scale");

    // ---- runtime tensors ----
    CHECK(g2.runtime.tensors.size() == 4, "runtime tensors resized");

    // ---- empty graph round-trip ----
    Graph g3;
    CHECK(graph_save(g3, "/tmp/test_graph_empty.gllm"), "save empty graph");
    Graph g4;
    CHECK(graph_load(g4, "/tmp/test_graph_empty.gllm"), "load empty graph");
    CHECK(g4.nodes.empty(), "empty graph loaded");

    // ---- edge case: params with strings only ----
    Graph g5;
    GraphNode n;
    n.id = 0;
    n.op_type = OpType::INPUT;
    n.params.str = {"path/to/weights.bin", "another/path.bin"};
    g5.nodes.push_back(n);
    g5.graph_inputs = {0};
    g5.graph_outputs = {0};
    CHECK(graph_save(g5, "/tmp/test_graph_strs.gllm"), "save str-only graph");
    Graph g6;
    CHECK(graph_load(g6, "/tmp/test_graph_strs.gllm"), "load str-only graph");
    CHECK(g6.nodes[0].params.str.size() == 2, "2 str params loaded");
    CHECK(g6.nodes[0].params.str[0] == "path/to/weights.bin", "str[0] matches");
    CHECK(g6.nodes[0].params.str[1] == "another/path.bin", "str[1] matches");

    // cleanup
    remove("/tmp/test_graph.gllm");
    remove("/tmp/test_graph_empty.gllm");
    remove("/tmp/test_graph_strs.gllm");

    if (failures == 0) {
        printf("\nAll io tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
