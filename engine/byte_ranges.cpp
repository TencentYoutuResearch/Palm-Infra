#include "engine/byte_ranges.h"

#include <algorithm>
#include <iterator>

namespace mollm::detail {

std::vector<ByteRange> normalize_byte_ranges(
    const std::vector<std::pair<uint64_t, uint64_t>>& offset_lengths,
    uint64_t limit) {
    std::vector<std::pair<uint64_t, uint64_t>> sorted = offset_lengths;
    std::sort(sorted.begin(), sorted.end());

    std::vector<ByteRange> merged;
    for (const auto& [offset, length] : sorted) {
        if (length == 0 || offset >= limit)
            continue;

        // Clamp before adding so corrupt offset/length metadata cannot wrap.
        const uint64_t end = offset + std::min(length, limit - offset);
        if (merged.empty() || offset > merged.back().end) {
            merged.push_back({offset, end});
        } else {
            merged.back().end = std::max(merged.back().end, end);
        }
    }
    return merged;
}

bool range_contains(const std::vector<ByteRange>& ranges, uint64_t offset) {
    const auto after = std::upper_bound(
        ranges.begin(), ranges.end(), offset,
        [](uint64_t value, const ByteRange& range) {
            return value < range.begin;
        });
    if (after == ranges.begin())
        return false;
    return offset < std::prev(after)->end;
}

}  // namespace mollm::detail
