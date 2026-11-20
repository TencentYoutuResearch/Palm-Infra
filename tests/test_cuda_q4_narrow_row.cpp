#include "kernels/cuda/matmul.h"

#include "cuda_test_support.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using mollm_test::check;
using mollm_test::from_bits;
using mollm_test::GuardedBuffer;
using mollm_test::require;
using mollm_test::same_bits;

constexpr int N = 2560;
constexpr int K = 6144;
constexpr int max_tokens = 6;

template<class T> T* misaligned(T* pointer) {
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(pointer) + 1);
}

// Every 19th block carries signed-zero scales; all nibble patterns are valid.
std::vector<Q4B8G32Block> make_weights() {
    std::vector<Q4B8G32Block> source(size_t(N / 8) * (K / 32));
    for (size_t b = 0; b < source.size(); ++b) {
        for (int r = 0; r < 8; ++r) {
            source[b].scales[r] = (b % 19 == 0)
                ? from_bits((r % 2) * 0x80000000u)
                : (r % 2 ? -1.f : 1.f) * 0.01317f * (1 + (b + r) % 7);
            for (int p = 0; p < 16; ++p)
                source[b].q[r][p] = ((b * 7 + r * 3 + p) % 16) | (((b * 3 + r + p * 5) % 16) << 4);
        }
    }
    return source;
}

// Independent CPU view oracle: preserve scale bits and biased nibbles.
std::vector<mollm_cuda::Q4RowG32> make_view_oracle(const std::vector<Q4B8G32Block>& source) {
    std::vector<mollm_cuda::Q4RowG32> oracle(size_t(N) * (K / 32));
    for (int row = 0; row < N; ++row) {
        for (int g = 0; g < K / 32; ++g) {
            const auto& block = source[size_t(row / 8) * (K / 32) + g];
            auto& group = oracle[size_t(row) * (K / 32) + g];
            std::memcpy(&group.scale, &block.scales[row % 8], 4);
            std::memcpy(group.q, block.q[row % 8], 16);
        }
    }
    return oracle;
}

// Phases 0, 1 and 8 are finite patterns; 2..7 cover zeros, signed zeros,
// subnormals, a NaN payload and infinities.
void fill_phase(std::vector<float>& values, int phase) {
    for (size_t i = 0; i < values.size(); ++i) {
        float value = float(int((i * 13 + phase * 17) % 101) - 50) / 137.f;
        if (phase == 2) value = 0.f;
        if (phase == 3) value = from_bits(0x80000000u);
        if (phase == 4) value = i % 4 ? 0.f : from_bits(2);
        values[i] = value;
    }
    if (phase == 5) values[13] = from_bits(0x7fc12345u);
    if (phase == 6) values[17] = from_bits(0x7f800000u);
    if (phase == 7) {
        values[17] = from_bits(0x7f800000u);
        values[23] = from_bits(0xff800000u);
    }
}

void run() {
    const std::vector<Q4B8G32Block> source = make_weights();
    const std::vector<mollm_cuda::Q4RowG32> oracle = make_view_oracle(source);
    GuardedBuffer weight_buffer(source.size() * sizeof(Q4B8G32Block));
    GuardedBuffer view_buffer(oracle.size() * sizeof(mollm_cuda::Q4RowG32));
    auto* weight = weight_buffer.as<Q4B8G32Block>();
    auto* view = view_buffer.as<mollm_cuda::Q4RowG32>();
    weight_buffer.upload(source.data());
    mollm_cuda::launch_repack_q4_row_g32(weight, view, N, K);
    check(cudaGetLastError(), "repack launch");
    check(cudaDeviceSynchronize(), "repack");
    require(same_bits(view_buffer.download<mollm_cuda::Q4RowG32>(), oracle), "view oracle mismatch");

    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    for (int padding : {0, 2}) {
        const int lda = K + padding;
        const int ldc = N + 3;
        GuardedBuffer input_buffer(size_t(max_tokens) * lda * sizeof(float));
        GuardedBuffer expected_buffer(size_t(max_tokens) * ldc * sizeof(float));
        GuardedBuffer result_buffer(size_t(max_tokens) * ldc * sizeof(float));
        float* input = input_buffer.as<float>();
        float* expected = expected_buffer.as<float>();
        float* result = result_buffer.as<float>();
        std::vector<float> values(size_t(max_tokens) * lda);
        const std::vector<int> valid_shape = {N, K, lda, ldc};
        const auto rejected = [&](const float* x, const mollm_cuda::Q4RowG32* w, float* y, int tokens,
                                  const std::vector<int>& shape) {
            return mollm_cuda::launch_q4_narrow_row(x, w, y, tokens, shape[0], shape[1], shape[2],
                                                    shape[3], stream) == cudaErrorInvalidValue;
        };

        // Rejected calls must not enqueue work into a capturing stream.
        cudaGraph_t empty;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        for (int tokens : {0, 2, 3, 7, 8})
            require(rejected(input, view, result, tokens, valid_shape), "unsupported batch accepted");
        require(rejected(nullptr, view, result, 1, valid_shape), "null input accepted");
        require(rejected(input, nullptr, result, 1, valid_shape), "null view accepted");
        require(rejected(input, view, nullptr, 1, valid_shape), "null output accepted");
        require(rejected(input + 1, view, result, 4, valid_shape), "unaligned input accepted");
        require(rejected(input, misaligned(view), result, 4, valid_shape), "unaligned view accepted");
        require(rejected(input, view, misaligned(result), 4, valid_shape), "unaligned output accepted");
        const std::vector<std::vector<int>> invalid_shapes = {
            {N - 1, K, lda, ldc},
            {N, K - 32, lda, ldc},
            {N, K, K + 1, ldc},
            {N, K, K - 2, ldc},
            {N, K, lda, N - 1},
        };
        for (const auto& shape : invalid_shapes)
            require(rejected(input, view, result, 4, shape), "invalid shape/stride accepted");
        check(cudaStreamEndCapture(stream, &empty), "end capture");
        size_t nodes = 0;
        check(cudaGraphGetNodes(empty, nullptr, &nodes), "graph nodes");
        require(nodes == 0, "invalid calls enqueued work");
        check(cudaGraphDestroy(empty), "graph destroy");

        for (int tokens : {1, 4, 5, 6}) {
            require(mollm_cuda::q4_narrow_row_eligible(input, tokens, N, K, lda, ldc),
                    "supported batch is not eligible");
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
            check(mollm_cuda::launch_q4_narrow_row(input, view, result, tokens, N, K, lda, ldc, stream),
                  "narrow row launch");
            check(cudaStreamEndCapture(stream, &graph), "end capture");
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "graph instantiate");
            for (int phase = 0; phase < 9; ++phase) {
                fill_phase(values, phase);
                check(cudaMemcpyAsync(input, values.data(), input_buffer.bytes(), cudaMemcpyHostToDevice,
                                      stream), "input upload");
                check(cudaMemsetAsync(expected, 0x5a, expected_buffer.bytes(), stream), "expected fill");
                check(cudaMemsetAsync(result, 0x5a, result_buffer.bytes(), stream), "result fill");
                if (tokens == 1) {
                    mollm_cuda::launch_q4_g32_dense_gemv(input, weight, expected, N, K, stream);
                } else {
                    check(mollm_cuda::launch_q4_g32_small_batch_exact(input, weight, expected, tokens, N, K,
                                                                      lda, ldc, stream),
                          "reference launch");
                }
                check(cudaGraphLaunch(executable, stream), "graph launch");
                check(cudaStreamSynchronize(stream), "phase");
                require(same_bits(expected_buffer.download<float>(), result_buffer.download<float>()),
                        "changed graph/native FP32 bits or stride guards differ");
            }
            check(cudaGraphExecDestroy(executable), "graph exec destroy");
            check(cudaGraphDestroy(graph), "graph destroy");
        }
        check(cudaStreamSynchronize(stream), "padding");
        input_buffer.check_guards();
        expected_buffer.check_guards();
        result_buffer.check_guards();
    }
    check(cudaStreamSynchronize(stream), "final");
    require(same_bits(weight_buffer.download<Q4B8G32Block>(), source), "original weights changed");
    require(same_bits(view_buffer.download<mollm_cuda::Q4RowG32>(), oracle), "view changed");
    weight_buffer.check_guards();
    view_buffer.check_guards();
    check(cudaStreamDestroy(stream), "stream destroy");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s\n", argv[0]);
        return 2;
    }
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    run();
    std::puts("Q4 narrow row: exact FP32/special values, CPU view oracle, changed graph, "
              "strides/guards and immutable weights PASS");
    return 0;
}
