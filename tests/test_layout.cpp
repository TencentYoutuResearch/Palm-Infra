#include "backends/cpu/backend.h"
#include "graph/graph.h"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

Tensor external_tensor(std::vector<float>& data, int64_t d0, int64_t d1 = 1,
                       int64_t d2 = 1) {
    return Tensor::create(Precision::FP32, MemoryType::EXTERNAL, d0, d1, d2, 1,
                          data.data());
}

void test_permute_contiguous() {
    std::vector<float> input = {1, 2, 3, 4, 5, 6};
    Tensor source = external_tensor(input, 3, 2);
    Tensor permuted;
    GraphNode permute;
    permute.op_type = OpType::PERMUTE;
    permute.params.i32 = {1, 0, 2, 3};
    CPUBackend{}.dispatch(permute, {&source}, &permuted, nullptr);

    std::vector<float> dense(6);
    Tensor output = external_tensor(dense, 2, 3);
    GraphNode contiguous;
    contiguous.op_type = OpType::CONTIGUOUS;
    CPUBackend{}.dispatch(contiguous, {&permuted}, &output, nullptr);
    check(dense == std::vector<float>({1, 4, 2, 5, 3, 6}),
          "PERMUTE followed by CONTIGUOUS");
}

void test_general_tile() {
    std::vector<float> input = {1, 2, 3, 4};
    Tensor source = external_tensor(input, 2, 2);
    std::vector<float> tiled(16);
    Tensor output = external_tensor(tiled, 4, 4);
    GraphNode tile;
    tile.op_type = OpType::TILE;
    tile.params.i32 = {2, 2, 1, 1};
    CPUBackend{}.dispatch(tile, {&source}, &output, nullptr);

    bool matches = true;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            matches &= tiled[row * 4 + col] == input[(row % 2) * 2 + (col % 2)];
    check(matches, "TILE general multidimensional path");
}

void test_concat_dim1() {
    std::vector<float> first = {1, 2, 3, 4};
    std::vector<float> second = {5, 6};
    Tensor a = external_tensor(first, 2, 2);
    Tensor b = external_tensor(second, 2, 1);
    std::vector<float> joined(6);
    Tensor output = external_tensor(joined, 2, 3);
    GraphNode concat;
    concat.op_type = OpType::CONCAT;
    concat.params.i32 = {1};
    CPUBackend{}.dispatch(concat, {&a, &b}, &output, nullptr);
    check(joined == std::vector<float>({1, 2, 3, 4, 5, 6}),
          "CONCAT along dim1");
}

void test_slice_view() {
    std::vector<float> input(12);
    Tensor source = external_tensor(input, 6, 2);
    source.owner_id = 3;
    source.storage_id = 9;
    Tensor slice;
    GraphNode node;
    node.op_type = OpType::SLICE;
    node.params.i32 = {0, 2, 3};
    CPUBackend{}.dispatch(node, {&source}, &slice, nullptr);
    check(slice.data == input.data() + 2 && slice.shape[0] == 3,
          "SLICE offsets data and updates shape");
    check(slice.shares_storage_with(source),
          "SLICE preserves storage identity");
}

void test_reshape_dynamic_view() {
    std::vector<float> input = {1, 2, 3, 4, 5, 6};
    Tensor source = external_tensor(input, 6);
    source.owner_id = 3;
    source.storage_id = 9;
    Tensor output;
    output.shape[0] = 1;  // Static dimension comes from the graph literal.
    output.shape[1] = 3;  // Dynamic dimension has been resolved by the executor.
    GraphNode node;
    node.op_type = OpType::RESHAPE;
    node.params.i32 = {2, -1, 1, 1};
    node.dim_expr[1].kind = DIM_SEQ;
    CPUBackend{}.dispatch(node, {&source}, &output, nullptr);
    check(output.shape[0] == 2 && output.shape[1] == 3,
          "RESHAPE combines static literals and resolved dynamic dimensions");
    check(output.data == source.data && output.shares_storage_with(source),
          "contiguous RESHAPE preserves borrowed storage");
    check(output.is_contiguous(), "RESHAPE recomputes contiguous strides");
}

void test_reshape_strided_materialization() {
    std::vector<float> input = {1, 2, 3, 4, 5, 6};
    Tensor source = external_tensor(input, 3, 2).permute(1, 0, 2, 3);
    std::vector<float> dense(6);
    Tensor output = external_tensor(dense, 6);
    GraphNode node;
    node.op_type = OpType::RESHAPE;
    // Missing shape literals use the output dimensions resolved by the caller.
    CPUBackend{}.dispatch(node, {&source}, &output, nullptr);
    check(output.data == dense.data() && output.shape[0] == 6,
          "strided RESHAPE retains its destination allocation and shape");
    check(dense == std::vector<float>({1, 4, 2, 5, 3, 6}),
          "strided RESHAPE materializes logical element order");
}

void test_slice_default_size() {
    std::vector<float> input(12);
    Tensor source = external_tensor(input, 6, 2);
    Tensor output;
    output.shape[0] = 4;
    GraphNode node;
    node.op_type = OpType::SLICE;
    node.params.i32 = {0, 1};
    CPUBackend{}.dispatch(node, {&source}, &output, nullptr);
    check(output.shape[0] == 4 && output.data == input.data() + 1,
          "SLICE default length comes from destination before aliasing source");
}

} // namespace

int main() {
    test_permute_contiguous();
    test_general_tile();
    test_concat_dim1();
    test_slice_view();
    test_reshape_dynamic_view();
    test_reshape_strided_materialization();
    test_slice_default_size();

    if (failures == 0)
        std::printf("All layout tests passed!\n");
    return failures;
}
