#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace mollm::cuda_backend_detail {

// A private device representation of immutable host FP16 constants. The
// Tensor retains its host storage; these bytes are never exposed as FP16.
class Fp16ExpertPaletteCache {
public:
    struct View {
        const uint16_t* source = nullptr;
        int rows = 0, inner = 0;
        uint8_t* codes = nullptr;
        uint16_t* palette = nullptr;
        size_t bytes = 0;
        ~View();
        View() = default;
        View(const View&) = delete;
        View& operator=(const View&) = delete;
    };
    enum class Result { Ready, Ineligible, Error };
    // Source must remain immutable/alive until the cache is destroyed. No
    // eviction: callers retire captured readers before destroying the cache.
    Result prepare(const uint16_t* source, int rows, int inner);
    const View* find(const uint16_t* source, int rows, int inner) const;
    size_t size() const { return views_.size(); }
    size_t bytes() const;
private:
    std::vector<std::unique_ptr<View>> views_;
};

} // namespace mollm::cuda_backend_detail
