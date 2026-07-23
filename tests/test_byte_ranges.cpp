#include "engine/byte_ranges.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void test_normalize_and_merge() {
    using mollm::detail::ByteRange;
    const std::vector<std::pair<uint64_t, uint64_t>> input = {
        {30, 10}, {10, 5}, {14, 10}, {24, 6}, {90, 20}, {50, 0}, {120, 5}};
    const auto ranges = mollm::detail::normalize_byte_ranges(input, 100);
    const std::vector<ByteRange> expected = {{10, 40}, {90, 100}};
    expect(ranges == expected, "ranges should sort, merge, and clip");
}

void test_overflowing_length_is_clipped() {
    const auto ranges = mollm::detail::normalize_byte_ranges(
        {{8, std::numeric_limits<uint64_t>::max()}}, 32);
    expect(ranges.size() == 1 && ranges[0].begin == 8 && ranges[0].end == 32,
           "overflowing length should clip without wrapping");
}

void test_contains_half_open_ranges() {
    const auto ranges =
        mollm::detail::normalize_byte_ranges({{10, 5}, {20, 5}}, 100);
    expect(!mollm::detail::range_contains(ranges, 9), "before first range");
    expect(mollm::detail::range_contains(ranges, 10), "range begin");
    expect(mollm::detail::range_contains(ranges, 14), "range final byte");
    expect(!mollm::detail::range_contains(ranges, 15), "range end is excluded");
    expect(mollm::detail::range_contains(ranges, 24), "second range");
    expect(!mollm::detail::range_contains(ranges, 25), "after second range");
}

}  // namespace

int main() {
    test_normalize_and_merge();
    test_overflowing_length_is_clipped();
    test_contains_half_open_ranges();
    if (failures != 0)
        return 1;
    std::puts("All byte range tests passed.");
    return 0;
}
