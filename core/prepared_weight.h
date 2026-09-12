#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Backend-specific, load-time representations of one logical weight. Keep
// these layouts out of Tensor: adding a backend should add a layout kind and
// its provider, not another backend-specific Tensor field.
enum class WeightLayout : uint8_t {
    X86_VNNI_Q4_G32 = 0,
    COUNT,
};

struct PreparedWeight {
    static constexpr size_t kLayoutCount =
        static_cast<size_t>(WeightLayout::COUNT);

    std::array<std::vector<uint8_t>, kLayoutCount> layouts;

    std::vector<uint8_t>& layout(WeightLayout kind) {
        return layouts[static_cast<size_t>(kind)];
    }

    const std::vector<uint8_t>& layout(WeightLayout kind) const {
        return layouts[static_cast<size_t>(kind)];
    }

    const void* data(WeightLayout kind) const {
        const auto& bytes = layout(kind);
        return bytes.empty() ? nullptr : bytes.data();
    }
};

// The engine owns this store for at least as long as graph weight Tensors.
// Tensor only carries a non-owning pointer to an entry.
using PreparedWeightMap = std::unordered_map<std::string, PreparedWeight>;
