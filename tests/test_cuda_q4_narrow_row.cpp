#include "kernels/cuda/matmul.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int N = 2560, K = 6144;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class T> struct Buffer {
    T* base = nullptr;
    T* data = nullptr;
    size_t count;
    explicit Buffer(size_t count) : count(count) {
        check(cudaMalloc(&base, (count + 32) * sizeof(T)));
        data = base + 16;
        check(cudaMemset(base, 0x5a, (count + 32) * sizeof(T)));
        check(cudaStreamSynchronize(cudaStreamPerThread));
    }
    ~Buffer() { cudaFree(base); }
    std::vector<T> read(cudaStream_t stream) const {
        std::vector<T> result(count);
        check(cudaMemcpyAsync(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return result;
    }
    void guards(cudaStream_t stream) const {
        std::vector<unsigned char> bytes((count + 32) * sizeof(T));
        check(cudaMemcpyAsync(bytes.data(), base, bytes.size(), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < 16 * sizeof(T); ++i)
            require(bytes[i] == 0x5a && bytes[(count + 16) * sizeof(T) + i] == 0x5a,
                "allocation guard changed");
    }
};
float from_bits(unsigned bits) { float value; std::memcpy(&value, &bits, 4); return value; }

void run(bool prefetch_off, bool vector_off, bool grouped, bool narrow_prefetch) {
    std::vector<Q4B8G32Block> source(size_t(N / 8) * (K / 32));
    std::vector<mollm_cuda::Q4RowG32> oracle(size_t(N) * (K / 32));
    for (size_t b = 0; b < source.size(); ++b) for (int r = 0; r < 8; ++r) {
        source[b].scales[r] = (b % 19 == 0) ? from_bits((r % 2) * 0x80000000u)
            : (r % 2 ? -1.f : 1.f) * 0.01317f * (1 + (b + r) % 7);
        for (int p = 0; p < 16; ++p)
            source[b].q[r][p] = ((b * 7 + r * 3 + p) % 16) | (((b * 3 + r + p * 5) % 16) << 4);
    }
    // Independent CPU view oracle: preserve scale bits and biased nibbles.
    for (int row = 0; row < N; ++row) for (int g = 0; g < K / 32; ++g) {
        const auto& block = source[size_t(row / 8) * (K / 32) + g];
        auto& group = oracle[size_t(row) * (K / 32) + g];
        std::memcpy(&group.scale, &block.scales[row % 8], 4);
        std::memcpy(group.q, block.q[row % 8], 16);
    }
    Buffer<Q4B8G32Block> weight(source.size());
    Buffer<mollm_cuda::Q4RowG32> view(oracle.size());
    check(cudaMemcpy(weight.data, source.data(), source.size() * sizeof(source[0]), cudaMemcpyHostToDevice));
    mollm_cuda::launch_repack_q4_row_g32(weight.data, view.data, N, K);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(cudaStreamPerThread));
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto actual_view = view.read(stream);
    require(!std::memcmp(actual_view.data(), oracle.data(), oracle.size() * sizeof(oracle[0])), "view oracle mismatch");

    for (int padding : {0, 2}) {
        const int lda = K + padding, ldc = N + 3;
        Buffer<float> input(size_t(6) * lda), expected(size_t(6) * ldc), result(size_t(6) * ldc);
        std::vector<float> values(input.count);
        auto launch = [&](int tokens) {
            return mollm_cuda::launch_q4_narrow_row(input.data, view.data, result.data,
                tokens, N, K, lda, ldc, stream);
        };
        cudaGraph_t empty;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int tokens : {0, 2, 3, 7, 8})
            require(launch(tokens) == cudaErrorInvalidValue, "unsupported batch accepted");
        require(mollm_cuda::launch_q4_narrow_row(nullptr, view.data, result.data, 1, N, K, lda, ldc, stream) == cudaErrorInvalidValue, "null input accepted");
        require(mollm_cuda::launch_q4_narrow_row(input.data, nullptr, result.data, 1, N, K, lda, ldc, stream) == cudaErrorInvalidValue, "null view accepted");
        require(mollm_cuda::launch_q4_narrow_row(input.data, view.data, nullptr, 1, N, K, lda, ldc, stream) == cudaErrorInvalidValue, "null output accepted");
        require(mollm_cuda::launch_q4_narrow_row(input.data + 1, view.data, result.data, 4, N, K, lda, ldc, stream) == cudaErrorInvalidValue, "unaligned input accepted");
        require(mollm_cuda::launch_q4_narrow_row(input.data, reinterpret_cast<mollm_cuda::Q4RowG32*>(reinterpret_cast<char*>(view.data) + 1), result.data, 4, N, K, lda, ldc, stream) == cudaErrorInvalidValue, "unaligned view accepted");
        require(mollm_cuda::launch_q4_narrow_row(input.data, view.data, reinterpret_cast<float*>(reinterpret_cast<char*>(result.data) + 1), 4, N, K, lda, ldc, stream) == cudaErrorInvalidValue, "unaligned output accepted");
        for (auto shape : {std::vector<int>{N-1,K,lda,ldc}, {N,K-32,lda,ldc}, {N,K,K+1,ldc}, {N,K,K-2,ldc}, {N,K,lda,N-1}})
            require(mollm_cuda::launch_q4_narrow_row(input.data, view.data, result.data, 4, shape[0], shape[1], shape[2], shape[3], stream) == cudaErrorInvalidValue, "invalid shape/stride accepted");
        check(cudaStreamEndCapture(stream, &empty));
        size_t nodes = 0; check(cudaGraphGetNodes(empty, nullptr, &nodes));
        require(nodes == 0, "invalid calls enqueued work"); check(cudaGraphDestroy(empty));

        for (int tokens : {1, 4, 5, 6}) {
            const bool eligible = tokens == 1 ? !prefetch_off && !narrow_prefetch : !grouped && (tokens == 4 || !vector_off);
            require(mollm_cuda::q4_narrow_row_eligible(input.data, tokens, N, K, lda, ldc) == eligible, "override precedence mismatch");
            if (!eligible) {
                require(launch(tokens) == cudaErrorInvalidValue, "override ignored");
                continue;
            }
            cudaGraph_t graph; cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            check(launch(tokens));
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int phase = 0; phase < 9; ++phase) {
                for (size_t i = 0; i < values.size(); ++i) {
                    float v = float(int((i * 13 + phase * 17) % 101) - 50) / 137.f;
                    if (phase == 2) v = 0.f;
                    if (phase == 3) v = from_bits(0x80000000u);
                    if (phase == 4) v = i % 4 ? 0.f : from_bits(2);
                    values[i] = v;
                }
                if (phase == 5) values[13] = from_bits(0x7fc12345u);
                if (phase == 6) values[17] = from_bits(0x7f800000u);
                if (phase == 7) { values[17] = from_bits(0x7f800000u); values[23] = from_bits(0xff800000u); }
                check(cudaMemcpyAsync(input.data, values.data(), values.size() * 4, cudaMemcpyHostToDevice, stream));
                check(cudaMemsetAsync(expected.data, 0x5a, expected.count * 4, stream));
                check(cudaMemsetAsync(result.data, 0x5a, result.count * 4, stream));
                if (tokens == 1)
                    mollm_cuda::launch_q4_g32_dense_gemv(input.data, weight.data, expected.data, N, K, stream);
                else
                    check(mollm_cuda::launch_q4_g32_small_batch_exact(input.data, weight.data,
                        expected.data, tokens, N, K, lda, ldc, stream));
                check(cudaGraphLaunch(executable, stream));
                auto reference = expected.read(stream), actual = result.read(stream);
                require(!std::memcmp(reference.data(), actual.data(), actual.size() * 4), "changed graph/native FP32 bits or stride guards differ");
            }
            check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        }
        input.guards(stream); expected.guards(stream); result.guards(stream);
    }
    auto final_weight = weight.read(stream); actual_view = view.read(stream);
    require(!std::memcmp(final_weight.data(), source.data(), source.size() * sizeof(source[0])), "original weights changed");
    require(!std::memcmp(actual_view.data(), oracle.data(), oracle.size() * sizeof(oracle[0])), "view changed");
    weight.guards(stream); view.guards(stream); check(cudaStreamDestroy(stream));
}
}
int main(int argc, char** argv) {
    if (argc != 1) return 1;
    int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try { run(false, false, false, false); std::puts("Q4 narrow row: exact FP32/special values, CPU view oracle, changed graph, strides/guards and immutable weights PASS"); return 0; }
    catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
