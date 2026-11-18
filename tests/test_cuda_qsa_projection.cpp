#include "backends/cuda/backend.h"
#include "graph/graph.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
}

int main(int argc, char** argv) {
    const bool segments = argc == 2 && std::strcmp(argv[1], "--segments") == 0;
    if (segments) unsetenv("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS");
    else setenv("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS", "1", 1);
    CudaBackend backend;
    if (!backend.available()) return 77;
    backend.set_operator_fallback_policy(OperatorFallbackPolicy::REQUIRE_NATIVE);
    constexpr int k = 2560, n = 640;
    std::vector<uint16_t> weights(size_t(n) * k);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = uint16_t(0x2800 + (i * 137 + i / k * 59) % 2048) |
                     uint16_t((i * 13 + i / k) % 3 ? 0 : 0x8000);
    Tensor weight = Tensor::create(Precision::FP16, MemoryType::EXTERNAL, n, k, 1, 1, weights.data());
    backend.wrap_weight(weight);
    require(!backend.dispatch_failed(), "wrap FP16 projection");

    // Test the projection's graph role and plan lifecycle directly. Attention
    // history is unnecessary for checking this producer's arithmetic contract.
    Graph graph;
    graph.nodes.resize(11);
    graph.runtime.tensors.resize(11);
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        graph.nodes[i].id = uint32_t(i);
        graph.nodes[i].op_type = OpType::INPUT;
        graph.nodes[i].out_prec = Precision::FP32;
    }
    graph.nodes[1].op_type = OpType::CONSTANT;
    graph.nodes[1].out_prec = Precision::FP16;
    auto& projection = graph.nodes[2];
    projection.op_type = OpType::MATMUL;
    projection.inputs = {0, 1};
    projection.out_shape[0] = n;
    graph.nodes[3].op_type = OpType::QSA_INDEXER;
    graph.nodes[3].out_prec = Precision::INT32;
    graph.nodes[3].inputs = {2, 4, 5, 6, 7, 8, 9, 10};
    size_t differing_ordinary_elements = 0;
    for (int rows : {2, 3, 4, 8, 9}) {
        Tensor input = Tensor::create(Precision::FP32, MemoryType::NONE, k, rows);
        Tensor output = Tensor::create(Precision::FP32, MemoryType::NONE, n, rows);
        require(backend.alloc_output(input, input.nbytes(), nullptr), "allocate input");
        require(backend.alloc_output(output, output.nbytes(), nullptr), "allocate output");
        std::vector<float> x(size_t(rows) * k), ordinary(size_t(rows) * n), serial(ordinary.size()), actual(ordinary.size());
        auto dispatch = [&] {
            backend.begin_graph();
            backend.dispatch(projection, {&input, &weight}, &output, nullptr);
            backend.end_graph();
            require(!backend.dispatch_failed(), "projection dispatch");
        };
        for (int replay = 0; replay < 2; ++replay) {
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = std::sin(float(i * 7 + replay * 13) * .017f) * .31f;
            require(backend.copy_from_host(x.data(), input, x.size() * sizeof(float)), "upload input");
            dispatch();
            require(backend.copy_to_host(output, ordinary.data(), ordinary.size() * sizeof(float)), "ordinary batch");
            // Single-token calls, with the same prepared weight, are the oracle.
            Tensor whole_input = input, whole_output = output;
            for (int row = 0; row < rows; ++row) {
                input = whole_input; output = whole_output;
                input.device.offset += size_t(row) * k * sizeof(float);
                output.device.offset += size_t(row) * n * sizeof(float);
                input.shape[1] = output.shape[1] = 1;
                input.compute_strides(); output.compute_strides();
                dispatch();
            }
            input = whole_input; output = whole_output;
            require(backend.copy_to_host(output, serial.data(), serial.size() * sizeof(float)), "serial oracle");
            for (size_t i = 0; i < serial.size(); ++i)
                differing_ordinary_elements += serial[i] != ordinary[i];
            backend.prepare_graph(graph);
            backend.prepare_graph(graph); // Reprepare must not leave stale entries.
            dispatch();
            require(backend.copy_to_host(output, actual.data(), actual.size() * sizeof(float)), "tagged batch");
            const auto& expected = rows <= 8 ? serial : ordinary;
            require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                    "indexer short batch must match serial; larger prefill must retain ordinary GEMM");
            backend.release_graph_plan(graph);
            dispatch();
            require(backend.copy_to_host(output, actual.data(), actual.size() * sizeof(float)), "released plan");
            require(std::memcmp(actual.data(), ordinary.data(), actual.size() * sizeof(float)) == 0,
                    "unmarked projection must retain ordinary GEMM after release");
        }
        backend.free_output(input, nullptr); backend.free_output(output, nullptr);
    }
    std::printf("QSA FP16 projection: exact short batches, ordinary prefill and plan lifecycle pass; ordinary/serial differing elements=%zu\n", differing_ordinary_elements);
}
