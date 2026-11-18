#include "kernels/cuda/moe.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Job = mollm_cuda::Nvfp4Projection;
constexpr int rows = 1280, inner = 2560, experts = 3;
constexpr size_t code_bytes = size_t(rows) * inner / 2;
constexpr size_t scale_bytes = size_t(rows) * inner / 16;
constexpr size_t code_stride = code_bytes + 16, scale_stride = scale_bytes + 16;

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> struct Buffer {
    T* data = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) { check(cudaMalloc(&data, count * sizeof(T))); }
    ~Buffer() { if (data) cudaFree(data); }
    std::vector<T> read(cudaStream_t stream) const {
        std::vector<T> result(count);
        check(cudaMemcpyAsync(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return result;
    }
};
template<class T> bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(T));
}
float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
void run(int routes, bool mask_only=false) {
    require(routes >= 1 && routes <= 80, "routes must be in 1..80");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Buffer<uint8_t> codes(experts * code_stride), scales(experts * scale_stride);
    Buffer<float> row_scales(experts * rows), input(size_t(routes) * inner + 4);
    Buffer<float> output(size_t(routes) * rows + 32), reference(size_t(routes) * rows + 32);
    Buffer<Job> jobs_device(routes), reference_jobs_device(routes);
    auto candidate = [&](const Job* jobs, int count, int n, int k) {
        return mollm_cuda::launch_nvfp4_gate_paired8_projections(jobs, count, n, k, stream);
    };
    require(candidate(nullptr, routes, rows, inner) == cudaErrorInvalidValue, "null jobs admitted");
    for (int count : {-1, 0, 81})
        require(candidate(jobs_device.data, count, rows, inner) == cudaErrorInvalidValue, "invalid count admitted");
    require(candidate(jobs_device.data, routes, rows - 1, inner) == cudaErrorInvalidValue, "invalid N admitted");
    require(candidate(jobs_device.data, routes, rows, inner - 1) == cudaErrorInvalidValue, "invalid K admitted");
    const auto* unaligned = reinterpret_cast<const Job*>(reinterpret_cast<const uint8_t*>(jobs_device.data) + 1);
    require(candidate(unaligned, routes, rows, inner) == cudaErrorInvalidValue, "unaligned descriptors admitted");

    int fixtures = 0;
    for (bool pair : {false, true}) for (bool scale_major : {false, true}) for (int alignment = 0; alignment < 4; ++alignment) {
        if(mask_only && (!pair || !scale_major || alignment))continue;
        unsigned rng = 181203;
        auto random = [&] { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
        std::vector<uint8_t> weight_host(codes.count, 0x5a), scale_host(scales.count, 0x5a);
        std::vector<float> row_host(row_scales.count), input_host(input.count);
        for (int expert = 0; expert < experts; ++expert) for (int row = 0; row < rows; ++row) {
            row_host[expert * rows + row] = float((row + expert) % 7 + 1) / 64.f;
            for (int group = 0; group < inner / 16; ++group) {
                const unsigned values[] = {0, 1, 7, 8, 48, 56, 64, 126, 128, 129, 184, 254};
                const size_t offset = scale_major ? size_t(group / 4) * rows * 4 + row * 4 + group % 4
                                                  : size_t(row) * (inner / 16) + group;
                scale_host[expert * scale_stride + offset] = values[(group + row + expert) % 12];
            }
            for (int byte = 0; byte < inner / 2; ++byte) {
                const int col = byte * 2;
                const size_t offset = size_t(col / 64) * rows * 32 + (pair
                    ? size_t(row / 4) * 128 + (col % 64 / 16) * 32 + (row % 4) * 8 + col % 16 / 2
                    : size_t(row) * 32 + col % 64 / 2);
                weight_host[expert * code_stride + ((alignment & 1) ? 1 : 0) + offset] = random();
            }
        }
        check(cudaMemcpyAsync(codes.data, weight_host.data(), codes.count, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(scales.data, scale_host.data(), scales.count, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(row_scales.data, row_host.data(), row_scales.count * sizeof(float), cudaMemcpyHostToDevice, stream));
        std::vector<Job> jobs(routes), reference_jobs(routes);
        for (int route = 0; route < routes; ++route) {
            const int expert = route % experts;
            jobs[route] = {codes.data + expert * code_stride + ((alignment & 1) ? 1 : 0),
                scales.data + expert * scale_stride, row_scales.data + expert * rows,
                input.data + size_t(route) * inner + ((alignment & 2) ? 1 : 0),
                output.data + 4 + size_t(route) * rows, pair};
            jobs[route].gate_scales_kmajor = scale_major;
            reference_jobs[route] = jobs[route];
            reference_jobs[route].output = reference.data + 4 + size_t(route) * rows;
        }
        auto upload_jobs = [&] {
            check(cudaMemcpyAsync(jobs_device.data, jobs.data(), routes * sizeof(Job), cudaMemcpyHostToDevice, stream));
            check(cudaMemcpyAsync(reference_jobs_device.data, reference_jobs.data(), routes * sizeof(Job), cudaMemcpyHostToDevice, stream));
        };
        auto original = [&] { check((routes<=50 ? mollm_cuda::launch_nvfp4_gate_paired_projections(reference_jobs_device.data, routes, rows, inner, stream) : mollm_cuda::launch_nvfp4_gate_code_projections(reference_jobs_device.data, routes, rows, inner, false, stream))); };
        auto updated = [&] { check(candidate(jobs_device.data, routes, rows, inner)); };
        auto poison = [&] {
            check(cudaMemsetAsync(output.data, 0x5a, output.count * sizeof(float), stream));
            check(cudaMemsetAsync(reference.data, 0x5a, reference.count * sizeof(float), stream));
        };
        upload_jobs();
        check(cudaStreamSynchronize(stream));
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        original(); updated();
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int change = 0; change < 11; ++change) {
            for (size_t i = 0; i < input_host.size(); ++i) {
                float value = float(int(random() % 2001) - 1000) / 8192.f;
                if (change == 1) value = -value;
                else if (change == 2) value = value * .5f + float(int(i % 13) - 6) * .00013f;
                else if (change == 3) value = 0.f;
                else if (change == 4) value = from_bits(0x80000000u);
                else if (change == 5) value = i % 4 ? 0.f : from_bits(2);
                input_host[i] = value;
            }
            if (change == 6) input_host[13] = from_bits(0x7fc12345u);
            if (change == 7) input_host[17] = from_bits(0x7f800000u);
            if (change == 8) { input_host[17] = from_bits(0x7f800000u); input_host[23] = from_bits(0xff800000u); }
            auto active_scales = scale_host;
            if (change >= 9) active_scales[3] = change == 9 ? 127 : 255;
            for (int route = 0; route < routes; ++route) {
                const int expert = (change & 1) ? (route * (route + 3) + change) % experts : route % experts;
                jobs[route].weight = codes.data + expert * code_stride + ((alignment & 1) ? 1 : 0);
                jobs[route].block_scales = scales.data + expert * scale_stride;
                jobs[route].row_scales = row_scales.data + expert * rows;
                reference_jobs[route] = jobs[route];
                reference_jobs[route].output = reference.data + 4 + size_t(route) * rows;
            }
            upload_jobs();
            check(cudaMemcpyAsync(input.data, input_host.data(), input.count * sizeof(float), cudaMemcpyHostToDevice, stream));
            check(cudaMemcpyAsync(scales.data, active_scales.data(), scales.count, cudaMemcpyHostToDevice, stream));
            poison(); original(); updated();
            const auto expected = reference.read(stream);
            require(same_bytes(output.read(stream), expected), "eager output/guards differ from native16");
            poison(); check(cudaGraphLaunch(executable, stream));
            require(same_bytes(output.read(stream), expected), "changed graph output/guards differ");
            require(same_bytes(reference.read(stream), expected), "changed graph native oracle differs");
            require(codes.read(stream) == weight_host && scales.read(stream) == active_scales, "weights/scales modified");
            require(same_bytes(input.read(stream), input_host) && same_bytes(row_scales.read(stream), row_host), "inputs/row scales modified");
            require(same_bytes(jobs_device.read(stream), jobs) && same_bytes(reference_jobs_device.read(stream), reference_jobs), "descriptors modified");
            ++fixtures;
        }
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    }
    check(cudaStreamDestroy(stream));
    std::printf("PASS paired8 routes=%d fixtures=%d native_bits eager mutable_graphs guards immutable admission\n", routes, fixtures);
}
}
int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    if(argc>3 || (argc==3 && std::strcmp(argv[2],"--mask-only")))return 2;
    try { run(argc > 1 ? std::stoi(argv[1]) : 3,argc==3); return 0; }
    catch (const std::exception& error) { std::fprintf(stderr, "paired8: %s\n", error.what()); return 1; }
}
