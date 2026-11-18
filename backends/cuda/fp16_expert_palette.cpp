#include "backends/cuda/fp16_expert_palette.h"
#include "backends/cuda/device_runtime.h"

#include <algorithm>
#include <limits>

namespace mollm::cuda_backend_detail {

Fp16ExpertPaletteCache::View::~View() {
    if (codes) cudaFree(codes);
    if (palette) cudaFree(palette);
}

const Fp16ExpertPaletteCache::View* Fp16ExpertPaletteCache::find(
    const uint16_t* source, int rows, int inner) const {
    for (const auto& view : views_)
        if (view->source == source && view->rows == rows && view->inner == inner)
            return view.get();
    return nullptr;
}

size_t Fp16ExpertPaletteCache::bytes() const {
    size_t total = 0;
    for (const auto& view : views_) total += view->bytes;
    return total;
}

Fp16ExpertPaletteCache::Result Fp16ExpertPaletteCache::prepare(
    const uint16_t* source, int rows, int inner) {
    if (!source || reinterpret_cast<uintptr_t>(source) % alignof(uint16_t) ||
        rows <= 0 || inner <= 0 || rows % 128 || inner % 128 ||
        size_t(rows) > std::numeric_limits<size_t>::max() / size_t(inner) / 2)
        return Result::Ineligible;
    const size_t count = size_t(rows) * inner;
    // Two passes use monotonically increasing tile stamps. Bound them before
    // reading host memory, as well as the original FP16 byte extent above.
    if (count / 16384 > (uint64_t(UINT32_MAX) - 1) / 2)
        return Result::Ineligible;
    cudaStreamCaptureStatus capture;
    if (!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread, &capture),
                                "FP16 palette capture check") ||
        capture != cudaStreamCaptureStatusNone) return Result::Error;
    if (find(source, rows, inner)) return Result::Ready;

    std::vector<uint32_t> seen(65536);
    uint32_t stamp = 0;
    // Validate every block before allocating GPU storage. Cardinality uses
    // raw uint16 bits, including signed zeros, infinities and NaN payloads.
    for (int r = 0; r < rows; r += 128) for (int c = 0; c < inner; c += 128) {
        ++stamp;
        unsigned used = 0;
        for (int rr = 0; rr < 128; ++rr) for (int cc = 0; cc < 128; ++cc) {
            const uint16_t bits = source[size_t(r + rr) * inner + c + cc];
            if (seen[bits] != stamp) {
                seen[bits] = stamp;
                if (++used > 256) return Result::Ineligible;
            }
        }
    }
    auto view = std::make_unique<View>();
    const size_t palette_count = count / 16384 * 256;
    if (!mollm_cuda::malloc_device(reinterpret_cast<void**>(&view->codes), count,
                                  "FP16 expert palette codes") ||
        !mollm_cuda::malloc_device(reinterpret_cast<void**>(&view->palette), palette_count * 2,
                                  "FP16 expert palettes")) return Result::Error;

    // Bounded host staging, one 128-row strip independent of expert count.
    std::vector<uint16_t> lookup(65536), palette(size_t(inner / 128) * 256);
    std::vector<uint8_t> codes(size_t(128) * inner);
    for (int r = 0; r < rows; r += 128) {
        std::fill(palette.begin(), palette.end(), 0);
        for (int c = 0; c < inner; c += 128) {
            ++stamp;
            unsigned used = 0;
            auto* dictionary = palette.data() + size_t(c / 128) * 256;
            for (int rr = 0; rr < 128; ++rr) for (int cc = 0; cc < 128; ++cc) {
                const uint16_t bits = source[size_t(r + rr) * inner + c + cc];
                if (seen[bits] != stamp) {
                    seen[bits] = stamp;
                    if (used == 256) return Result::Error; // Source changed during load.
                    lookup[bits] = static_cast<uint16_t>(used);
                    dictionary[used++] = bits;
                }
                codes[size_t(rr) * inner + c + cc] = static_cast<uint8_t>(lookup[bits]);
            }
        }
        if (!mollm_cuda::copy_memory(view->codes + size_t(r) * inner, codes.data(), codes.size(),
                                     cudaMemcpyHostToDevice, "FP16 palette code upload") ||
            !mollm_cuda::copy_memory(view->palette + size_t(r / 128) * (inner / 128) * 256,
                                     palette.data(), palette.size() * 2,
                                     cudaMemcpyHostToDevice, "FP16 palette dictionary upload"))
            return Result::Error;
    }
    view->source = source;
    view->rows = rows;
    view->inner = inner;
    view->bytes = count + palette_count * 2;
    views_.push_back(std::move(view));
    return Result::Ready;
}

} // namespace mollm::cuda_backend_detail
