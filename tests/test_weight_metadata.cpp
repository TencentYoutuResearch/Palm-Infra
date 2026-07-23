#include "engine/weight_metadata.h"

#include <cstdio>
#include <limits>

namespace {

int failures = 0;

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL: %s\n", message);                   \
            ++failures;                                                     \
        } else {                                                            \
            std::printf("  PASS: %s\n", message);                          \
        }                                                                   \
    } while (0)

Tensor make_weight(Precision precision, int64_t rows, int64_t cols) {
    return Tensor::create(precision, MemoryType::EXTERNAL, rows, cols);
}

MappedFile::Header make_header(uint64_t data_size, uint64_t scales_size,
                               uint32_t group_size, uint32_t num_groups,
                               uint32_t flags = 0) {
    MappedFile::Header header{};
    header.flags = flags;
    header.data_size = data_size;
    header.scales_size = scales_size;
    header.group_size = group_size;
    header.num_groups = num_groups;
    return header;
}

}  // namespace

int main() {
    float scales[8] = {};

    {
        Tensor weight = make_weight(Precision::FP16, 2, 4);
        weight.scales = scales;
        weight.group_size = 7;
        weight.num_groups = 9;
        weight.groups_per_row = 3;
        weight.is_q4_repacked = true;
        weight.is_q4_g128_packed = true;
        MappedFile::Header header{};

        CHECK(mollm::detail::configure_weight_metadata(
                  weight, header, nullptr, "fp16"),
              "accept non-quantized weight");
        CHECK(weight.scales == nullptr && weight.group_size == 0 &&
                  weight.num_groups == 0 && weight.groups_per_row == 0 &&
                  !weight.is_q4_repacked && !weight.is_q4_g128_packed,
              "non-quantized weight clears stale quantization state");
    }

    {
        Tensor weight = make_weight(Precision::INT8, 2, 4);
        const MappedFile::Header header =
            make_header(8, 4 * sizeof(float), 2, 4);
        CHECK(mollm::detail::configure_weight_metadata(
                  weight, header, scales, "int8"),
              "accept valid INT8 metadata");
        CHECK(weight.scales == scales && weight.group_size == 2 &&
                  weight.groups_per_row == 2 && weight.num_groups == 4,
              "attach INT8 metadata");

        Tensor missing_scales = make_weight(Precision::INT8, 2, 4);
        CHECK(!mollm::detail::configure_weight_metadata(
                  missing_scales, header, nullptr, "missing-scales"),
              "reject quantized weight without scales");

        Tensor wrong_size = make_weight(Precision::INT8, 2, 4);
        MappedFile::Header bad_header = header;
        bad_header.data_size = 7;
        CHECK(!mollm::detail::configure_weight_metadata(
                  wrong_size, bad_header, scales, "wrong-size"),
              "reject inconsistent quantized data size");
    }

    {
        Tensor weight = make_weight(Precision::INT4, 2, 4);
        const MappedFile::Header header =
            make_header(4, 4 * sizeof(float), 2, 4);
        CHECK(mollm::detail::configure_weight_metadata(
                  weight, header, scales, "int4"),
              "accept valid plain INT4 metadata");
        CHECK(!weight.is_q4_repacked && !weight.is_q4_g128_packed,
              "plain INT4 has no packed-layout flags");

        Tensor conflicting = make_weight(Precision::INT4, 8, 128);
        MappedFile::Header conflicting_header =
            make_header(544, 8 * sizeof(float), 128, 8,
                        MappedFile::FLAG_INT4_Q4DOT |
                            MappedFile::FLAG_INT4_BG128);
        CHECK(!mollm::detail::configure_weight_metadata(
                  conflicting, conflicting_header, scales, "conflicting"),
              "reject conflicting INT4 layout flags");
    }

    {
        Tensor huge = make_weight(
            Precision::INT8, std::numeric_limits<int64_t>::max(),
            std::numeric_limits<int64_t>::max());
        const MappedFile::Header header =
            make_header(0, 0, 1, 0);
        CHECK(!mollm::detail::configure_weight_metadata(
                  huge, header, scales, "overflow"),
              "reject overflowing quantized dimensions");
    }

    if (failures == 0)
        std::printf("\nAll weight metadata tests passed!\n");
    else
        std::printf("\n%d test(s) FAILED\n", failures);
    return failures;
}
