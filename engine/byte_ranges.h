#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace mollm::detail {

struct ByteRange {
    uint64_t begin;
    uint64_t end;

    bool operator==(const ByteRange& other) const {
        return begin == other.begin && end == other.end;
    }
};

// Convert (offset, length) pairs into sorted, clipped, non-overlapping
// half-open ranges. Empty and fully out-of-bounds inputs are discarded.
std::vector<ByteRange> normalize_byte_ranges(
    const std::vector<std::pair<uint64_t, uint64_t>>& offset_lengths,
    uint64_t limit);

// `ranges` must be the normalized output above.
bool range_contains(const std::vector<ByteRange>& ranges, uint64_t offset);

}  // namespace mollm::detail
