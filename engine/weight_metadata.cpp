#include "engine/weight_metadata.h"

#include "kernels/matmul.h"

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        return false;
    result = lhs * rhs;
    return true;
}

}  // namespace

namespace mollm::detail {

bool configure_weight_metadata(Tensor& tensor,
                               const MappedFile::Header& header,
                               const void* scales, const char* label) {
    tensor.scales = nullptr;
    tensor.group_size = 0;
    tensor.num_groups = 0;
    tensor.groups_per_row = 0;
    tensor.is_q4_repacked = false;
    tensor.is_q4_g128_packed = false;

    const bool is_quantized =
        tensor.prec == Precision::INT8 || tensor.prec == Precision::INT4;
    if (!is_quantized)
        return true;

    const int64_t rows = tensor.shape[0];
    const int64_t cols = tensor.shape[1];
    if (!scales || header.group_size == 0 || rows <= 0 || cols <= 0) {
        std::fprintf(stderr,
                     "Engine: quantized weight %s missing scales/group "
                     "metadata\n",
                     label);
        return false;
    }

    const uint64_t rows_u = static_cast<uint64_t>(rows);
    const uint64_t cols_u = static_cast<uint64_t>(cols);
    const uint64_t groups_per_row_u =
        1 + (cols_u - 1) / header.group_size;
    uint64_t expected_groups = 0;
    if (groups_per_row_u > std::numeric_limits<uint32_t>::max() ||
        !checked_multiply(rows_u, groups_per_row_u, expected_groups)) {
        std::fprintf(stderr,
                     "Engine: quantized weight %s dimensions overflow "
                     "metadata limits\n",
                     label);
        return false;
    }
    const uint32_t groups_per_row =
        static_cast<uint32_t>(groups_per_row_u);
    constexpr uint32_t supported_flags =
        MappedFile::FLAG_INT4_Q4DOT | MappedFile::FLAG_INT4_BG128;
    const bool int4_q4dot_layout =
        tensor.prec == Precision::INT4 &&
        (header.flags & MappedFile::FLAG_INT4_Q4DOT);
    const bool int4_bg128_layout =
        tensor.prec == Precision::INT4 &&
        (header.flags & MappedFile::FLAG_INT4_BG128);

    if (header.flags & ~supported_flags) {
        std::fprintf(
            stderr,
            "Engine: quantized weight %s has unsupported flags 0x%x\n",
            label, header.flags);
        return false;
    }
    if ((header.flags & supported_flags) &&
        tensor.prec != Precision::INT4) {
        std::fprintf(stderr,
                     "Engine: weight %s has INT4 layout flag but precision "
                     "is not INT4\n",
                     label);
        return false;
    }
    if (int4_q4dot_layout && int4_bg128_layout) {
        std::fprintf(stderr,
                     "Engine: weight %s has mutually exclusive INT4 layout "
                     "flags\n",
                     label);
        return false;
    }
    if (int4_q4dot_layout &&
        (cols % 32 != 0 || header.group_size % 32 != 0)) {
        std::fprintf(stderr,
                     "Engine: INT4 q4dot weight %s requires K/group "
                     "multiple of 32 (K=%lld group=%u)\n",
                     label, static_cast<long long>(cols), header.group_size);
        return false;
    }
    if (int4_bg128_layout &&
        (cols % 128 != 0 || header.group_size != 128)) {
        std::fprintf(stderr,
                     "Engine: INT4 BG128 weight %s requires K multiple of "
                     "128 and group=128 (K=%lld group=%u)\n",
                     label, static_cast<long long>(cols), header.group_size);
        return false;
    }
    if ((int4_q4dot_layout || int4_bg128_layout) &&
        !matmul_int4_q4dot_kernel_available()) {
        std::fprintf(stderr,
                     "Engine: INT4 packed weight %s requires an ARM DOTPROD "
                     "build\n",
                     label);
        return false;
    }

    uint64_t expected_data_size = 0;
    if (!checked_multiply(rows_u, cols_u, expected_data_size)) {
        std::fprintf(stderr,
                     "Engine: quantized weight %s dimensions overflow data "
                     "size\n",
                     label);
        return false;
    }
    if (tensor.prec == Precision::INT4) {
        if (int4_q4dot_layout) {
            if (rows_u > std::numeric_limits<uint64_t>::max() - 7) {
                std::fprintf(stderr,
                             "Engine: quantized weight %s dimensions overflow "
                             "packed data size\n",
                             label);
                return false;
            }
            const uint64_t padded_rows = ((rows_u + 7) / 8) * 8;
            const uint64_t col_blocks = 1 + (cols_u - 1) / 32;
            uint64_t packed_blocks = 0;
            if (!checked_multiply(padded_rows, col_blocks, packed_blocks) ||
                !checked_multiply(packed_blocks, 16, expected_data_size)) {
                std::fprintf(stderr,
                             "Engine: quantized weight %s dimensions overflow "
                             "packed data size\n",
                             label);
                return false;
            }
        } else if (int4_bg128_layout) {
            if (rows > std::numeric_limits<int>::max() ||
                cols > std::numeric_limits<int>::max()) {
                std::fprintf(stderr,
                             "Engine: quantized weight %s dimensions exceed "
                             "packed kernel limits\n",
                             label);
                return false;
            }
            expected_data_size = static_cast<uint64_t>(
                pack_b_q4dot_g128_bytes(static_cast<int>(rows),
                                        static_cast<int>(cols)));
        } else {
            const uint64_t packed_cols = 1 + (cols_u - 1) / 2;
            if (!checked_multiply(rows_u, packed_cols,
                                  expected_data_size)) {
                std::fprintf(stderr,
                             "Engine: quantized weight %s dimensions overflow "
                             "packed data size\n",
                             label);
                return false;
            }
        }
    }

    uint64_t expected_scales_size = 0;
    if (!checked_multiply(expected_groups, sizeof(float),
                          expected_scales_size)) {
        std::fprintf(stderr,
                     "Engine: quantized weight %s dimensions overflow scales "
                     "size\n",
                     label);
        return false;
    }
    if (header.num_groups != expected_groups ||
        header.scales_size != expected_scales_size ||
        header.data_size != expected_data_size) {
        std::fprintf(
            stderr,
            "Engine: quantized weight %s bad metadata (N=%lld K=%lld "
            "group=%u groups=%u expected=%llu scales=%llu data=%llu "
            "expected_data=%llu)\n",
            label, static_cast<long long>(rows),
            static_cast<long long>(cols), header.group_size,
            header.num_groups,
            static_cast<unsigned long long>(expected_groups),
            static_cast<unsigned long long>(header.scales_size),
            static_cast<unsigned long long>(header.data_size),
            static_cast<unsigned long long>(expected_data_size));
        return false;
    }

    tensor.scales = static_cast<const float*>(scales);
    tensor.group_size = header.group_size;
    tensor.num_groups = header.num_groups;
    tensor.groups_per_row = groups_per_row;
    tensor.is_q4_repacked = int4_q4dot_layout;
    tensor.is_q4_g128_packed = int4_bg128_layout;
    if (int4_bg128_layout)
        tensor.q4_g128_data = tensor.data;
    return true;
}

}  // namespace mollm::detail
