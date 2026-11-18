#include "kernels/cuda/q8_head_screen.h"
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/elementwise.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {
using namespace mollm_cuda;
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "scalar head screen CUDA: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "scalar head screen: %s\n", message); std::exit(1); }
}
struct Buffer {
    unsigned char *allocation = nullptr, *data = nullptr;
    size_t bytes;
    explicit Buffer(size_t size) : bytes(size) {
        check(cudaMalloc(&allocation, bytes + 512)); data = allocation + 256;
        check(cudaMemset(allocation, 0x39, bytes + 512));
    }
    ~Buffer() { cudaFree(allocation); }
    template<class T> T* as() { return reinterpret_cast<T*>(data); }
    void put(const void* source) { check(cudaMemcpy(data, source, bytes, cudaMemcpyHostToDevice)); }
    template<class T> std::vector<T> get() {
        std::vector<T> out(bytes / sizeof(T));
        check(cudaMemcpy(out.data(), data, bytes, cudaMemcpyDeviceToHost)); return out;
    }
    void guards() {
        unsigned char first[256], last[256];
        check(cudaMemcpy(first, allocation, 256, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(last, data + bytes, 256, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 256; ++i) require(first[i] == 0x39 && last[i] == 0x39, "allocation guards");
    }
};
void run(bool large, int bits) {
    const auto pack_head = pack_q8_head_screen_q4;
    const auto screen_head = launch_q8_head_screen_q4;
    constexpr int inner = 2560;
    const int rows = large ? 248320 : 521;
    std::vector<int8_t> weights(size_t(rows) * inner);
    std::vector<float> scales(rows), values(inner);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = int8_t((i * 29 + i / 257) % 256);
    for (int i = 0; i < 256; ++i) weights[i] = int8_t(i);
    std::copy_n(weights.begin(), inner, weights.begin() + inner);
    for (int i = 0; i < rows; ++i) scales[i] = i % 13 ? .001f : -.001f;
    scales[1] = scales[0];
    Buffer w(weights.size()), packed(size_t(rows) * 320 * bits), sf(rows * 4), x(inner * 4),
        output(rows * 4), full(rows * 4), approximate(rows * 4), lower(257 * 4),
        norm(8), count(4), ids(rows * 4), partial(256 * sizeof(ArgMaxPair)), result(sizeof(ArgMaxPair));
    w.put(weights.data()); sf.put(scales.data());
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto repack = [&] {
        check(pack_head(w.as<int8_t>(), packed.as<uint8_t>(), rows, stream));
        check(cudaStreamSynchronize(stream));
        const auto codes = packed.get<uint8_t>();
        const size_t groups = weights.size() / 4;
        for (size_t g = 0; g < groups; ++g) {
            if (large && g >= 512 && g + 512 < groups) continue;
            const unsigned packet = unsigned(codes[g * 2]) | unsigned(codes[g * 2 + 1]) << 8;
            for (int j = 0; j < 4; ++j) {
                const int weight = int(weights[g * 4 + j]);
                const int code = int(packet >> (4 * j)) & 15;
                require(code == (weight + 128) / 16, "packed Q4 codes");
                require(std::fabs(weight - (16 * code - 120.5)) <= 7.5,
                        "Q4 half-center quantization bound");
            }
        }
        return codes;
    };
    auto original_packed = repack();
    Q8HeadScreenScratch scratch{approximate.as<float>(), lower.as<float>(), norm.as<double>(),
        count.as<unsigned>(), ids.as<int>()};
    auto launch = [&] { check(screen_head(x.as<float>(), w.as<int8_t>(),
        packed.as<uint8_t>(), sf.as<float>(), output.as<float>(), rows, scratch, stream)); };
    auto argmax = [&](const float* data) {
        launch_argmax(data, rows, partial.as<ArgMaxPair>(), 256, result.as<ArgMaxPair>());
        check(cudaGetLastError()); check(cudaDeviceSynchronize()); return result.get<ArgMaxPair>()[0].index;
    };
    x.put(values.data());
    cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); launch();
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    int fixtures = 0;
    for (int fixture = 0; fixture < 16; ++fixture) {
        if (large && fixture != 0 && fixture != 2 && fixture != 15) continue;
        ++fixtures;
        for (int j = 0; j < inner; ++j) values[j] = std::sin(float(j) * .31f);
        auto ss = scales;
        switch (fixture) {
            case 0: std::fill(values.begin(), values.end(), 0.f); break;
            case 1: for (int j = 0; j < inner; ++j) values[j] = float(weights[j]) / 128.f; break;
            case 3: for (auto& v : values) v *= 1e16f; break;
            case 4: for (auto& v : values) v *= 1e20f; break;
            case 5: for (auto& v : values) v *= 1e-40f; break;
            case 6: std::fill(ss.begin(), ss.end(), 1e-40f); break;
            case 7: std::fill(ss.begin(), ss.end(), 1e10f); break;
            case 8: std::fill(ss.begin(), ss.end(), 1e11f); break;
            case 9: values[17] = NAN; break;
            case 10: values[17] = INFINITY; break;
            case 11: ss[0] = INFINITY; ss[1] = NAN; break;
            case 12: std::fill(values.begin(), values.end(), 0.f);
                std::fill(ss.begin(), ss.end(), std::numeric_limits<float>::max()); break;
            case 13: std::fill(ss.begin(), ss.end(), NAN); break;
            case 14: std::fill(ss.begin(), ss.end(), -0.f); break;
            case 15:
                // Scale disparity makes the max-lower-bound anchor score zero.
                // The coarse-center anchor scores20; every zero row's certified
                // upper bound is <20, so precisely one candidate must survive.
                std::fill(values.begin(), values.end(), 1.f);
                std::fill(ss.begin(), ss.end(), 1e-4f); ss[0] = 1e-8f; ss[rows - 1] = 1.f;
                std::fill(weights.begin(), weights.end(), 0); weights[size_t(rows - 1) * inner] = 20;
                w.put(weights.data()); original_packed = repack(); break;
        }
        x.put(values.data()); sf.put(ss.data());
        check(launch_q8_tuned_gemv(x.as<float>(), w.as<int8_t>(), sf.as<float>(), full.as<float>(),
            rows, inner, 8, false, stream)); check(cudaStreamSynchronize(stream));
        const auto reference = full.get<float>(); const int expected = argmax(full.as<float>());
        check(cudaMemsetAsync(output.data, 0xa5, output.bytes, stream)); launch();
        check(cudaStreamSynchronize(stream)); const auto eager = output.get<float>();
        check(cudaMemsetAsync(output.data, 0xa5, output.bytes, stream));
        check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
        const auto actual = output.get<float>(), coarse = approximate.get<float>();
        require(!std::memcmp(eager.data(), actual.data(), output.bytes), "changed-input graph equals eager");
        require(argmax(output.as<float>()) == expected, "original full-Q8 argmax");
        const auto selected = ids.get<int>(); const unsigned kept = count.get<unsigned>()[0];
        require(kept > 0 && kept <= unsigned(rows), "candidate count");
        std::vector<bool> included(rows, false);
        for (unsigned j = 0; j < kept; ++j) {
            const int row = selected[j];
            require(row >= 0 && row < rows && !included[row], "candidate uniqueness and bounds");
            included[row] = true;
            require((std::isnan(actual[row]) && std::isnan(reference[row])) ||
                !std::memcmp(&actual[row], &reference[row], 4), "selected original-Q8 score bits");
        }
        double l1 = 0; for (float v : values) l1 += std::fabs(double(v));
        for (int row = 0; row < rows; ++row) {
            if (!included[row]) require(actual[row] == -INFINITY, "discarded sentinel");
            if (std::isfinite(coarse[row]) && std::isfinite(l1) && l1 <= 1e20 &&
                std::isfinite(ss[row]) && std::fabs(ss[row]) <= 1e10f)
                require(std::fabs(double(coarse[row]) - reference[row]) <=
                    (bits == 4 ? 7.51 : 4.14) * std::fabs(double(ss[row])) * l1 + 1e-5, "conservative interval bound");
        }
        if (fixture == 15) require(expected == rows - 1 && kept == 1, "scale-disparity anchor regression");
        const auto copied_x = x.get<float>(), copied_sf = sf.get<float>();
        require(!std::memcmp(copied_x.data(), values.data(), x.bytes) &&
            !std::memcmp(copied_sf.data(), ss.data(), sf.bytes), "input/scales immutable");
        for (auto* buffer : {&w, &packed, &sf, &x, &output, &full, &approximate, &lower,
                            &norm, &count, &ids, &partial, &result}) buffer->guards();
    }
    require(w.get<int8_t>() == weights && packed.get<uint8_t>() == original_packed, "weights immutable");
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));

    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    auto reject = [&](const float* input, float* out, int n, Q8HeadScreenScratch s) {
        require(screen_head(input, w.as<int8_t>(), packed.as<uint8_t>(),
            sf.as<float>(), out, n, s, stream) == cudaErrorInvalidValue, "invalid launch rejected");
    };
    reject(x.as<float>(), output.as<float>(), 511, scratch);
    reject(x.as<float>(), output.as<float>(), INT_MAX, scratch);
    reject(nullptr, output.as<float>(), rows, scratch);
    reject(x.as<float>() + 1, output.as<float>(), rows, scratch);
    reject(x.as<float>(), approximate.as<float>(), rows, scratch);
    reject(reinterpret_cast<const float*>(UINTPTR_MAX - 15), output.as<float>(), rows, scratch);
    auto invalid = scratch; invalid.norm = reinterpret_cast<double*>(norm.data + 4);
    reject(x.as<float>(), output.as<float>(), rows, invalid);
    invalid = scratch; invalid.indices = reinterpret_cast<int*>(lower.data);
    reject(x.as<float>(), output.as<float>(), rows, invalid);
    require(pack_head(w.as<int8_t>(), w.as<uint8_t>(), rows, stream) ==
        cudaErrorInvalidValue, "packing aliases rejected");
    require(pack_head(w.as<int8_t>(), packed.as<uint8_t>(), 511, stream) ==
        cudaErrorInvalidValue, "packing geometry rejected");
    check(cudaStreamEndCapture(stream, &graph)); size_t nodes = 1;
    check(cudaGraphGetNodes(graph, nullptr, &nodes)); require(nodes == 0, "rejections enqueue no kernels");
    check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
    std::printf("Q%d screen: rows=%d fixtures=%d native bits, argmax, bounds, graph, packing, guards, anchor and rejection PASS\n", bits, rows, fixtures);
}
} // namespace
int main(int argc, char** argv) {
    int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    bool large = false; int bits = 4;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--large")) large = true;
        else if (!std::strcmp(argv[i], "--q4")) continue;
        else require(false, "arguments");
    }
    run(large, bits);
}
