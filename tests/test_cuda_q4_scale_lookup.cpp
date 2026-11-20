#include "kernels/cuda/matmul.h"

#include "cuda_test_support.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace {

using mollm_test::check;
using mollm_test::GuardedBuffer;
using mollm_test::require;
using Block = mollm_cuda::Q4CompactG32;

// Preserve every finite bit, including signed zero. NaN payloads are not an API contract.
bool same(float a, float b) {
    return (std::isnan(a) && std::isnan(b)) || mollm_test::to_bits(a) == mollm_test::to_bits(b);
}

void fill_weights(std::vector<Block>& weights, int fixture) {
    for (size_t j = 0; j < weights.size(); ++j) {
        for (int r = 0; r < 8; ++r) {
            uint16_t scale;
            if (fixture == 2 || fixture == 3)
                scale = uint16_t(j * 8 + r);
            else if (fixture == 6)
                scale = 0x8000u;
            else if (fixture == 7)
                scale = 0x7f7fu;
            else
                scale = uint16_t(0x3c00u + (j * 17 + r * 23) % 768);
            weights[j].scales[r] = scale;
            for (int q = 0; q < 16; ++q) weights[j].q[r][q] = uint8_t(j * 13 + r * 31 + q * 7);
        }
    }
}

void fill_input(std::vector<float>& input, int fixture) {
    for (size_t j = 0; j < input.size(); ++j)
        input[j] = fixture == 1 ? 0.f : std::sin(j * .31f) * .125f;
    if (fixture == 2)
        for (auto& v : input) v *= 1e16f;
    if (fixture == 3)
        for (auto& v : input) v *= 1e-40f;
    if (fixture == 4) input[17] = NAN;
    if (fixture == 5) input[31] = INFINITY;
}

}  // namespace

int main() {
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    GuardedBuffer scales(65536 * 2 * sizeof(float));
    require(mollm_cuda::launch_q4_compact_scale_check(nullptr, stream) == cudaErrorInvalidValue,
            "scale check rejects null output");
    check(mollm_cuda::launch_q4_compact_scale_check(scales.as<float>(), stream));
    check(cudaStreamSynchronize(stream));
    const auto table = scales.download<float>();
    for (int c = 0; c < 65536; ++c)
        require(same(table[c * 2], table[c * 2 + 1]), "scale lookup equals native division");
    scales.check_guards();
    std::puts("scale_lookup_all65536 native_division_exact=1 guards=1");

    int cases = 0;
    const std::pair<int, int> shapes[] = {
        {4129, 32}, {4129, 2560}, {4129, 4096}, {16448, 2560}, {16480, 2560}};
    for (const auto& shape : shapes) {
        for (int weight_offset : {0, 4}) {
            const int n = shape.first;
            const int k = shape.second;
            const int modes = k == 2560 ? 3 : 2;
            std::vector<Block> weights(size_t((n + 7) / 8) * (k / 32));
            std::vector<float> input(k);
            GuardedBuffer w(weights.size() * sizeof(Block) + weight_offset);
            GuardedBuffer x(k * sizeof(float));
            std::unique_ptr<GuardedBuffer> y[3];
            for (int m = 0; m < modes; ++m) y[m] = std::make_unique<GuardedBuffer>(n * sizeof(float));
            auto* weight = reinterpret_cast<Block*>(w.as<unsigned char>() + weight_offset);
            float* activation = x.as<float>();

            require(mollm_cuda::launch_q4_compact_scale_gemv_probe(
                        activation + 1, weight, y[0]->as<float>(), n, k, true, stream) ==
                        cudaErrorInvalidValue,
                    "probe rejects unaligned input");
            require(mollm_cuda::launch_q4_compact_scale_gemv_probe(
                        activation, weight, y[0]->as<float>(), 4096, k, true, stream) ==
                        cudaErrorInvalidValue,
                    "probe rejects unsupported rows");
            require(mollm_cuda::launch_q4_compact_scale_gemv_probe(
                        activation, weight, y[0]->as<float>(), n, 4097, true, stream) ==
                        cudaErrorInvalidValue,
                    "probe rejects unsupported inner");

            // Mode 0: probe with native division, 1: probe with the scale lookup,
            // 2: the production compact GEMV.
            const auto launch = [&](int mode) {
                float* output = y[mode]->as<float>();
                if (mode < 2) {
                    check(mollm_cuda::launch_q4_compact_scale_gemv_probe(
                        activation, weight, output, n, k, mode == 1, stream));
                } else {
                    mollm_cuda::launch_q4_g32_compact_gemv(activation, weight, output, n, k, stream);
                    check(cudaGetLastError());
                }
            };
            cudaGraph_t graphs[3];
            cudaGraphExec_t executable[3];
            for (int m = 0; m < modes; ++m) {
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                launch(m);
                check(cudaStreamEndCapture(stream, &graphs[m]));
                check(cudaGraphInstantiate(&executable[m], graphs[m], nullptr, nullptr, 0));
            }

            for (int fixture = 0; fixture < 8; ++fixture) {
                fill_weights(weights, fixture);
                fill_input(input, fixture);
                std::vector<unsigned char> storage(w.bytes(), GuardedBuffer::guard_pattern);
                std::memcpy(storage.data() + weight_offset, weights.data(),
                            weights.size() * sizeof(Block));
                w.upload(storage.data());
                x.upload(input.data());
                // Uploads run on the default stream; order them before work on `stream`.
                check(cudaStreamSynchronize(cudaStreamPerThread));
                for (int phase = 0; phase < 2; ++phase) {
                    for (int mode = 0; mode < modes; ++mode) {
                        check(cudaMemsetAsync(y[mode]->data(), 0xa5, y[mode]->bytes(), stream));
                        if (phase)
                            check(cudaGraphLaunch(executable[mode], stream));
                        else
                            launch(mode);
                    }
                    check(cudaStreamSynchronize(stream));
                    const auto expected = y[0]->download<float>();
                    for (int mode = 1; mode < modes; ++mode) {
                        const auto actual = y[mode]->download<float>();
                        for (int j = 0; j < n; ++j)
                            require(same(expected[j], actual[j]), "scale lookup output bits");
                    }
                    for (int mode = 0; mode < modes; ++mode) y[mode]->check_guards();
                    w.check_guards();
                    require(w.download<unsigned char>() == storage, "weights immutable");
                    x.check_guards();
                    require(mollm_test::same_bits(x.download<float>(), input), "input immutable");
                }
                ++cases;
                std::printf("scale_lookup_case n=%d k=%d fixture=%d weight_offset=%d native_bits=1 "
                            "mutable_graph=1 guards=1\n",
                            n, k, fixture, weight_offset);
            }
            for (int m = 0; m < modes; ++m) {
                check(cudaGraphExecDestroy(executable[m]));
                check(cudaGraphDestroy(graphs[m]));
            }
        }
    }
    check(cudaStreamDestroy(stream));
    std::printf("scale_lookup_all cases=%d exact_native=1 mutable_graphs=1 guards=1\n", cases);
}
