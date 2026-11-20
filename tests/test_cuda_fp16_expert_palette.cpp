#include "backends/cuda/fp16_expert_palette_cache.h"
#include "backends/cuda/backend.h"
#include "kernels/cuda/fp16_expert_palette.h"
#include "kernels/cuda/moe.h"

#include "cuda_test_support.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using Cache = mollm::cuda_backend_detail::Fp16ExpertPaletteCache;
using Job = mollm_cuda::Nvfp4Projection;
using mollm_test::check;
using mollm_test::GuardedBuffer;
using mollm_test::require;

Tensor tensor(std::vector<uint16_t>& values, int rows, int inner) {
    Tensor t;
    t.prec = Precision::FP16;
    t.shape[0] = rows;
    t.shape[1] = inner;
    t.shape[2] = t.shape[3] = 1;
    t.compute_strides();
    t.data = values.data();
    t.rowmajor_data = values.data();
    t.mem_type = MemoryType::EXTERNAL;
    return t;
}

void packing() {
    std::vector<uint16_t> w(256 * 256);
    std::vector<uint16_t> different;
    std::vector<uint16_t> bad(128 * 128);
    Cache cache;
    // Every tile has 256 distinct patterns, including signed zeros and NaNs.
    for (size_t i = 0; i < w.size(); i++) {
        const unsigned q = unsigned((i / 256 * 3 + i % 256) % 256);
        const uint16_t edge[] = {0, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0xfe02};
        w[i] = q < 6 ? edge[q] : uint16_t(q * 257);
    }
    require(cache.prepare(nullptr, 128, 128) == Cache::Result::Ineligible, "null");
    require(cache.prepare(w.data(), 127, 128) == Cache::Result::Ineligible, "rows alignment");
    require(cache.prepare(w.data(), 128, 129) == Cache::Result::Ineligible, "inner alignment");
    const auto* unaligned =
        reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(w.data()) + 1);
    require(cache.prepare(unaligned, 128, 128) == Cache::Result::Ineligible, "source alignment");
    require(cache.prepare(w.data(), 256, 256) == Cache::Result::Ready, "256-values packing");
    const auto* packed = cache.find(w.data(), 256, 256);
    require(packed && cache.bytes() == 67584, "compressed size");
    require(cache.prepare(w.data(), 256, 256) == Cache::Result::Ready && cache.size() == 1 &&
                cache.find(w.data(), 256, 256) == packed,
            "cache identity");
    require(!cache.find(w.data(), 128, 256), "shape identity");

    std::vector<uint8_t> codes(w.size());
    std::vector<uint16_t> palette(4 * 256);
    check(cudaMemcpy(codes.data(), packed->codes, codes.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(palette.data(), packed->palette, palette.size() * sizeof(uint16_t),
                     cudaMemcpyDeviceToHost));
    for (int r = 0; r < 256; r++) {
        for (int k = 0; k < 256; k++) {
            const int tile = r / 128 * 2 + k / 128;
            require(palette[tile * 256 + codes[r * 256 + k]] == w[r * 256 + k], "bit reconstruction");
        }
    }

    different = w;
    require(cache.prepare(different.data(), 256, 256) == Cache::Result::Ready && cache.size() == 2,
            "distinct source");
    for (int i = 0; i < 257; i++) bad[i] = i;
    require(cache.prepare(bad.data(), 128, 128) == Cache::Result::Ineligible && cache.size() == 2,
            "257-values fallback");

    cudaGraph_t graph;
    check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    require(cache.prepare(w.data(), 256, 256) == Cache::Result::Error, "load under capture");
    check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    check(cudaGraphDestroy(graph));

    // Device-memory admission failures must not publish an incomplete cache entry.
    setenv("MOLLM_CUDA_HOST_HEADROOM_MB", "invalid", 1);
    Cache denied;
    require(denied.prepare(w.data(), 256, 256) == Cache::Result::Error && denied.size() == 0 &&
                denied.bytes() == 0,
            "allocation failure atomicity");
    unsetenv("MOLLM_CUDA_HOST_HEADROOM_MB");
}

void projections(int n, int k) {
    constexpr int experts = 3;
    constexpr int routes = 5;
    std::vector<uint16_t> w(size_t(experts) * n * k);
    for (size_t i = 0; i < w.size(); i++)
        w[i] = uint16_t(0x2800 + (i % 113) * 4 + (i / (n * k)) * 512);
    // Exact exceptional-weight conversion too.
    w[0] = 0x8000;
    w[1] = 0x7e01;
    w[2] = 0xfc00;

    Cache cache;
    require(cache.prepare(w.data(), experts * n, k) == Cache::Result::Ready, "projection pack");
    const auto* packed = cache.find(w.data(), experts * n, k);

    GuardedBuffer original(w.size() * sizeof(uint16_t));
    original.upload(w.data());
    GuardedBuffer x(size_t(routes) * k * sizeof(float));
    GuardedBuffer base(size_t(routes) * n * sizeof(float));
    GuardedBuffer result(size_t(routes) * n * sizeof(float));
    GuardedBuffer reference_jobs(routes * sizeof(Job));
    GuardedBuffer palette_jobs(routes * sizeof(Job));
    std::vector<Job> a(routes);
    std::vector<Job> b(routes);
    std::vector<float> input(size_t(routes) * k);
    std::vector<float> sentinel(size_t(routes) * n, -123.f);
    for (int r = 0; r < routes; r++) {
        const size_t expert = size_t((r * 2) % experts);
        a[r] = {};
        a[r].weight = reinterpret_cast<const uint8_t*>(original.as<uint16_t>() + expert * n * k);
        a[r].input = x.as<float>() + size_t(r) * k;
        a[r].output = base.as<float>() + size_t(r) * n;
        b[r] = a[r];
        b[r].weight = packed->codes + expert * n * k;
        b[r].block_scales =
            reinterpret_cast<const uint8_t*>(packed->palette + expert * n * k / 16384 * 256);
        b[r].output = result.as<float>() + size_t(r) * n;
    }
    reference_jobs.upload(a.data());
    palette_jobs.upload(b.data());
    base.upload(sentinel.data());
    result.upload(sentinel.data());

    const Job* jobs = palette_jobs.as<Job>();
    require(mollm_cuda::launch_fp16_palette_projections(nullptr, 1, n, k) == cudaErrorInvalidValue,
            "null jobs");
    require(mollm_cuda::launch_fp16_palette_projections(jobs, -1, n, k) == cudaErrorInvalidValue,
            "negative count");
    require(mollm_cuda::launch_fp16_palette_projections(jobs, 65536, n, k) == cudaErrorInvalidValue,
            "count limit");
    require(mollm_cuda::launch_fp16_palette_projections(jobs, 1, n - 1, k) == cudaErrorInvalidValue,
            "invalid rows");
    require(mollm_cuda::launch_fp16_palette_projections(jobs, 1, n, k - 1) == cudaErrorInvalidValue,
            "invalid inner");
    const auto* unaligned_jobs =
        reinterpret_cast<const Job*>(reinterpret_cast<const char*>(palette_jobs.data()) + 1);
    require(mollm_cuda::launch_fp16_palette_projections(unaligned_jobs, 1, n, k) ==
                cudaErrorInvalidValue,
            "unaligned jobs");
    check(mollm_cuda::launch_fp16_palette_projections(nullptr, 0, n, k));

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    check(mollm_cuda::launch_fp16_palette_projections(jobs, routes, n, k, cudaStreamPerThread));
    check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int pass = 0; pass < 3; pass++) {
        for (size_t i = 0; i < input.size(); i++)
            input[i] = float(int((i * 37 + pass * 91) % 8191) - 4095) / 1024.f;
        x.upload(input.data());
        check(mollm_cuda::launch_fp16_expert_projections(reference_jobs.as<Job>(), routes, n, k));
        if (pass)
            check(cudaGraphLaunch(exec, cudaStreamPerThread));
        else
            check(mollm_cuda::launch_fp16_palette_projections(jobs, routes, n, k));
        const auto expected = base.download<float>();
        const auto actual = result.download<float>();
        require(mollm_test::same_bits(expected, actual), "projection bits");
        base.check_guards();
        result.check_guards();
    }
    // Retire readers before cache destruction.
    check(cudaGraphExecDestroy(exec));
    check(cudaGraphDestroy(graph));
}

void backend() {
    std::vector<uint16_t> w(256 * 128, 0x3c00);
    std::vector<uint16_t> bad(128 * 128);
    for (int i = 0; i < 257; i++) bad[i] = i;
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0));
    const bool eligible = properties.major == 12 && properties.minor == 1;
    for (int repeat = 0; repeat < 2; repeat++) {
        CudaBackend b;
        require(b.available(), "backend");
        auto t = tensor(w, 256, 128);
        b.wrap_routed_expert_weight(t, 128);
        require(!b.dispatch_failed(), "backend prepare");
        if (eligible) {
            require(!t.device.buffer && b.fp16_expert_palette_bytes() == 33792,
                    "no fake FP16 device storage");
            b.wrap_routed_expert_weight(t, 128);
            require(b.fp16_expert_palette_bytes() == 33792, "backend reuse");
        }
        std::vector<uint16_t> copy(w.size());
        require(b.copy_to_host(t, copy.data(), copy.size() * sizeof(uint16_t)) && copy == w,
                "host fallback read");
        auto mismatch = tensor(bad, 128, 128);
        b.wrap_routed_expert_weight(mismatch, 128);
        require(mismatch.device.buffer && !b.dispatch_failed(), "ineligible normal upload");
        auto ordinary = tensor(w, 256, 128);
        b.wrap_weight(ordinary);
        require(ordinary.device.buffer, "ordinary mixed consumer remains FP16");
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool large = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--large") == 0) {
            large = true;
        } else {
            std::fprintf(stderr, "usage: %s [--large]\n", argv[0]);
            return 2;
        }
    }
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    packing();
    projections(128, 128);
    projections(256, 256);
    if (large) {
        projections(1280, 2560);
        projections(2560, 640);
    }
    backend();
    std::puts("FP16 palette: passed");
}
