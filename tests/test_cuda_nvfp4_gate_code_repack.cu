#include "kernels/cuda/moe.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}
std::vector<uint8_t> read(const uint8_t* ptr, size_t bytes) {
    std::vector<uint8_t> result(bytes);
    check(cudaMemcpy(result.data(), ptr, bytes, cudaMemcpyDeviceToHost));
    return result;
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));
    constexpr size_t bytes = 1280 * 2560 / 2, guard = 16;
    uint8_t *source = nullptr, *target = nullptr;
    check(cudaMalloc(&source, bytes + 2 * guard));
    check(cudaMalloc(&target, bytes + 2 * guard));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    mollm_cuda::Nvfp4Projection job{};
    job.weight = source + guard;
    check(cudaMemset(source, 0x5a, bytes + 2 * guard));
    check(cudaMemset(target, 0x5a, bytes + 2 * guard));
    check(cudaDeviceSynchronize());
    // Every invalid argument is rejected before enqueue, including within an
    // active capture. Equal/partial overlap and uintptr overflow are explicit.
    int rejected = 0;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int id = 0; id < 12; ++id) {
        auto bad = job;
        uint8_t* dst = target + guard;
        int rows = 1280, inner = 2560;
        switch (id) {
            case 0: bad.weight = nullptr; break;
            case 1: dst = nullptr; break;
            case 2: rows = 640; break;
            case 3: inner = 1280; break;
            case 4: bad.mma_layout = true; break;
            case 5: bad.weight += 1; break;
            case 6: dst += 1; break;
            case 7: dst = source + guard; break;
            case 8: dst = source + 2 * guard; break;
            case 9: bad.weight = target + 2 * guard; break;
            case 10: bad.weight = reinterpret_cast<const uint8_t*>(UINTPTR_MAX - 15); break;
            case 11: dst = reinterpret_cast<uint8_t*>(UINTPTR_MAX - 15); break;
        }
        if (mollm_cuda::launch_repack_nvfp4_gate_codes(bad, dst, rows, inner, stream)
            != cudaErrorInvalidValue) return 2;
        ++rejected;
    }
    cudaGraph_t empty;
    check(cudaStreamEndCapture(stream, &empty));
    size_t nodes = 0;
    check(cudaGraphGetNodes(empty, nullptr, &nodes));
    if (nodes) return 3;
    check(cudaGraphDestroy(empty));
    const auto sentinel = read(target, bytes + 2 * guard);
    if (!std::all_of(sentinel.begin(), sentinel.end(), [](uint8_t x) { return x == 0x5a; })) return 4;
    int replays = 0, host_staging = 0;
    for (bool pair : {false, true}) {
        job.pair_layout = pair;
        cudaGraph_t graph;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        check(mollm_cuda::launch_repack_nvfp4_gate_codes(job, target + guard, 1280, 2560, stream));
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphGetNodes(graph, nullptr, &nodes));
        if (nodes != 1) return 5;
        cudaGraphExec_t exec;
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int variant = 0; variant < 8; ++variant) {
            std::vector<uint8_t> input(bytes + 2 * guard, 0x5a);
            std::vector<uint8_t> expected(bytes + 2 * guard, 0x5a);
            // Include high address bits and replay changes; not a short
            // periodic pattern that could conceal swapped K/row indices.
            for (size_t i = 0; i < bytes; ++i) {
                uint32_t value = uint32_t(i) ^ (uint32_t(variant + 1) * 0x9e3779b9u);
                value ^= value >> 16; value *= 0x7feb352du; value ^= value >> 15;
                input[i + guard] = value & 255;
            }
            const size_t blocks = pair ? 320 : 1280, tile_bytes = pair ? 128 : 32;
            for (size_t q = 0; q < 40; ++q)
                for (size_t row = 0; row < blocks; ++row)
                    std::memcpy(expected.data() + guard + (q * blocks + row) * tile_bytes,
                        input.data() + guard + (row * 40 + q) * tile_bytes, tile_bytes);
            check(cudaMemcpyAsync(source, input.data(), input.size(), cudaMemcpyHostToDevice, stream));
            check(cudaMemsetAsync(target, 0x5a, bytes + 2 * guard, stream));
            check(cudaGraphLaunch(exec, stream));
            check(cudaStreamSynchronize(stream));
            if (read(target, expected.size()) != expected || read(source, input.size()) != input) return 6;
            // Resident reader uses bounded ordinary host staging on platforms
            // whose CUDA devices can access pageable memory via host tables.
            if (prop.pageableMemoryAccess && prop.pageableMemoryAccessUsesHostPageTables) {
                if (reinterpret_cast<uintptr_t>(input.data()) & 15) return 7;
                const auto snapshot = input;
                auto staged = job;
                staged.weight = input.data() + guard;
                check(cudaMemsetAsync(target, 0x5a, bytes + 2 * guard, stream));
                check(mollm_cuda::launch_repack_nvfp4_gate_codes(staged,
                    target + guard, 1280, 2560, stream));
                check(cudaStreamSynchronize(stream));
                if (read(target, expected.size()) != expected || input != snapshot) return 8;
                ++host_staging;
            }
            ++replays;
        }
        check(cudaGraphExecDestroy(exec));
        check(cudaGraphDestroy(graph));
    }
    check(cudaStreamDestroy(stream));
    check(cudaFree(source));
    check(cudaFree(target));
    std::printf("gate_code_repack rejected=%d empty_capture=1 changed_replays=%d pageable_host_staging=%d layouts=ROW,PAIR byte_exact=1 source_immutable=1 guards=1\n", rejected, replays, host_staging);
}
