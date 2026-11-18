#include "kernels/cuda/draft_compact_head.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace {
using namespace mollm_cuda;
constexpr int rows = 248320, inner = 2560, candidates = 1024;
void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "draft energy screen: %s\n", message); std::exit(1); }
}
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "draft energy screen CUDA: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}
uint32_t ordered_key(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    if (!(bits & 0x7fffffffU)) bits = 0; // Stable +0/-0 equivalence in CUB.
    return bits & 0x80000000U ? ~bits : bits ^ 0x80000000U;
}
struct Buffer {
    unsigned char *allocation = nullptr, *data = nullptr;
    size_t bytes;
    explicit Buffer(size_t size) : bytes(size) {
        check(cudaMalloc(&allocation, size + 512)); data = allocation + 256;
        check(cudaMemset(allocation, 0x39, size + 512));
    }
    ~Buffer() { cudaFree(allocation); }
    template<class T> T* as() { return reinterpret_cast<T*>(data); }
    void put(const void* source) { check(cudaMemcpy(data, source, bytes, cudaMemcpyHostToDevice)); }
    template<class T> std::vector<T> get() {
        std::vector<T> result(bytes / sizeof(T));
        check(cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost)); return result;
    }
    void guards() {
        unsigned char first[256], last[256];
        check(cudaMemcpy(first, allocation, 256, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(last, data + bytes, 256, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 256; ++i) require(first[i] == 0x39 && last[i] == 0x39, "buffer guards");
    }
};
void test_screen(int group_count) {
    std::vector<Q4CompactG32> weights(size_t(rows / 8) * (inner / 32));
    for (size_t i = 0; i < weights.size(); ++i) for (int row = 0; row < 8; ++row) {
        auto& block = weights[i];
        block.scales[row] = uint16_t(0x3d00 + (i * 13 + row * 7) % 512);
        uint32_t value = uint32_t(i * 1664525 + row * 1013904223u);
        for (int j = 0; j < 16; ++j) { value = value * 1664525 + 1013904223; block.q[row][j] = uint8_t(value >> 24); }
    }
    Buffer weight(weights.size() * sizeof(weights[0])), input(inner * 4), output(rows * 4),
        full(rows * 4), coarse(rows * 4), sorted(rows * 4), ids(rows * 4), selected(rows * 4), groups(32 * 4);
    weight.put(weights.data());
    DraftHeadScreenScratch scratch{coarse.as<float>(), sorted.as<float>(), ids.as<int>(),
        selected.as<int>(), groups.as<int>(), nullptr, 0};
    size_t bytes = 0;
    check(draft_head_screen_temp_bytes(scratch, bytes)); require(bytes > 0, "temporary query");
    Buffer temporary(bytes); scratch.sort_temp = temporary.data; scratch.sort_bytes = bytes;
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(initialize_draft_head_screen(scratch, stream)); check(cudaStreamSynchronize(stream));
    auto original_ids = ids.get<int>();
    for (int i = 0; i < rows; ++i) require(original_ids[i] == i, "initialized stable IDs");

    std::vector<float> values(inner, 0.f); input.put(values.data());
    cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    check(launch_draft_head_screen(input.as<float>(), weight.as<Q4CompactG32>(), output.as<float>(),
        rows, inner, scratch, stream, group_count));
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int fixture = 0; fixture < 10; ++fixture) {
        for (int i = 0; i < inner; ++i) {
            values[i] = fixture == 0 ? 0.f : fixture == 1 ? (i == 73 * 32 + 7 ? -2.f : 0.f) :
                float((i * 29 + fixture * 7) % 251 - 125) * (fixture == 3 ? 1e-40f : .013f);
        }
        if (fixture == 9) std::fill(values.begin(), values.end(), 1.f);
        if (fixture == 5) {
            const uint32_t bits = 0xffc00001U; float negative_nan;
            std::memcpy(&negative_nan, &bits, sizeof(bits));
            std::fill(values.begin(), values.end(), negative_nan);
        }
        if (fixture == 6) values[73 * 32 + 7] = INFINITY;
        if (fixture == 7) values[17] = -INFINITY;
        if (fixture == 8) std::fill(values.begin(), values.end(), 1e30f); // Energy overflows, logits need not.
        if (fixture == 4) {
            // Replay must observe changed weight contents without repacking.
            weights[0].scales[0] = 0x3f00; weights[0].q[0][0] ^= 0x5a;
            check(cudaMemcpy(weight.data, weights.data(), sizeof(weights[0]), cudaMemcpyHostToDevice));
        }
        input.put(values.data());
        launch_q4_g32_compact_gemv(input.as<float>(), weight.as<Q4CompactG32>(), full.as<float>(), rows, inner, stream);
        check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
        const auto reference = full.get<float>();
        check(launch_draft_head_screen(input.as<float>(), weight.as<Q4CompactG32>(), output.as<float>(),
            rows, inner, scratch, stream, group_count));
        check(cudaStreamSynchronize(stream));
        const auto eager = output.get<float>();
        check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
        const auto actual = output.get<float>();
        require(!std::memcmp(eager.data(), actual.data(), rows * 4), "mutable graph equals eager");

        float energies[80]; std::vector<int> expected_groups(80); std::iota(expected_groups.begin(), expected_groups.end(), 0);
        for (int group = 0; group < 80; ++group) {
            float sum = 0;
            for (int j = 0; j < 32; ++j) { volatile float product = values[group * 32 + j] * values[group * 32 + j]; sum += product; }
            energies[group] = std::isnan(sum) ? INFINITY : sum;
        }
        std::stable_sort(expected_groups.begin(), expected_groups.end(), [&](int a, int b) { return energies[a] > energies[b]; });
        const auto actual_groups = groups.get<int>();
        require(std::equal(actual_groups.begin(), actual_groups.end(), expected_groups.begin()), "CPU group selection");
        const auto scores = coarse.get<float>(); const auto selected_ids = selected.get<int>();
        std::vector<int> expected(rows); std::iota(expected.begin(), expected.end(), 0);
        std::partial_sort(expected.begin(), expected.begin() + candidates, expected.end(), [&](int a, int b) {
            const auto left = ordered_key(scores[a]), right = ordered_key(scores[b]);
            return left > right || (left == right && a < b);
        });
        std::vector<bool> included(rows, false);
        for (int j = 0; j < candidates; ++j) {
            require(expected[j] == selected_ids[j], "CPU vocabulary selection and ties");
            const int row = selected_ids[j];
            require(row >= 0 && row < rows && !included[row], "candidate bounds and uniqueness");
            included[row] = true;
            require((std::isnan(actual[row]) && std::isnan(reference[row])) ||
                !std::memcmp(&actual[row], &reference[row], 4), "selected full-K native bits / NaN semantics");
        }
        for (int row = 0; row < rows; ++row) if (!included[row]) require(actual[row] == -INFINITY, "non-candidate sentinel");
        if (fixture == 9) {
            // Equal energies select ascending K groups. A full-K projection
            // with the unselected input groups zeroed independently checks
            // that the requested group budget is used by the coarse pass.
            auto masked = values;
            std::fill(masked.begin() + group_count * 32, masked.end(), 0.f);
            input.put(masked.data());
            launch_q4_g32_compact_gemv(input.as<float>(), weight.as<Q4CompactG32>(),
                full.as<float>(), rows, inner, stream);
            check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
            const auto coarse_reference = full.get<float>();
            require(!std::memcmp(scores.data(), coarse_reference.data(), rows * 4),
                "coarse group budget matches masked full projection");
            input.put(values.data());
        }
        require(ids.get<int>() == original_ids, "stable IDs immutable");
        const auto copied_input = input.get<float>();
        require(!std::memcmp(copied_input.data(), values.data(), input.bytes), "input immutable");
        for (auto* buffer : {&weight, &input, &output, &full, &coarse, &sorted, &ids, &selected, &groups, &temporary}) buffer->guards();
    }
    const auto copied = weight.get<Q4CompactG32>();
    require(!std::memcmp(copied.data(), weights.data(), weight.bytes), "weights immutable");
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));

    // Rejected geometry, aliasing and capacity must not enqueue partial work.
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    auto invalid = scratch; invalid.sort_bytes = bytes - 1;
    require(launch_draft_head_screen(input.as<float>(), weight.as<Q4CompactG32>(), output.as<float>(), rows, inner, invalid, stream) == cudaErrorInvalidValue, "short temporary");
    invalid = scratch; invalid.sorted = invalid.coarse;
    require(initialize_draft_head_screen(invalid, stream) == cudaErrorInvalidValue, "aliased scratch");
    invalid = scratch; invalid.sort_temp = coarse.data;
    require(launch_draft_head_screen(input.as<float>(), weight.as<Q4CompactG32>(), output.as<float>(), rows, inner, invalid, stream) == cudaErrorInvalidValue, "aliased temporary");
    require(launch_draft_head_screen(input.as<float>(), weight.as<Q4CompactG32>(), coarse.as<float>(), rows, inner, scratch, stream) == cudaErrorInvalidValue, "aliased output");
    require(launch_draft_head_screen(nullptr, weight.as<Q4CompactG32>(), output.as<float>(), rows, inner, scratch, stream) == cudaErrorInvalidValue, "null input");
    require(launch_draft_head_screen(input.as<float>() + 1, weight.as<Q4CompactG32>(), output.as<float>(), rows, inner, scratch, stream) == cudaErrorInvalidValue, "unaligned input");
    require(launch_draft_head_screen(input.as<float>(), weight.as<Q4CompactG32>(), output.as<float>(), rows - 1, inner, scratch, stream) == cudaErrorInvalidValue, "unsupported rows");
    require(launch_draft_head_screen(input.as<float>(), weight.as<Q4CompactG32>(), output.as<float>(), rows, inner - 32, scratch, stream) == cudaErrorInvalidValue, "unsupported inner");
    for (int unsupported : {0, 15, 17, 20, 25, 80})
        require(launch_draft_head_screen(input.as<float>(), weight.as<Q4CompactG32>(),
            output.as<float>(), rows, inner, scratch, stream, unsupported) == cudaErrorInvalidValue,
            "unsupported group budget");
    check(cudaStreamEndCapture(stream, &graph)); size_t nodes = 1;
    check(cudaGraphGetNodes(graph, nullptr, &nodes)); require(nodes == 0, "invalid calls submit no kernels");
    check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
    std::printf("draft energy screen groups=%d: native bits, CPU selections, mutable graphs, guards, immutable inputs and rejection PASS\n", group_count);
}
}  // namespace
int main() {
    int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    test_screen(32);
    test_screen(24);
    test_screen(16);
}
