#include "engine/weight_metadata.h"
#include "backends/cpu/platform.h"

#include "kernels/cpu/matmul/matmul.h"

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
    tensor.e8m0_scales = nullptr;
    tensor.nvfp4_scales = nullptr;
    tensor.nvfp4_row_scales = nullptr;
    tensor.nvfp4_q8_pair_data = nullptr;
    tensor.fp8_q8_scales = nullptr;
    tensor.group_size = 0;
    tensor.num_groups = 0;
    tensor.groups_per_row = 0;
    tensor.is_q4_repacked = false;
    tensor.is_q4_g32_packed = false;
    tensor.is_q4_g128_packed = false;
    tensor.is_fp8_block128 = false;
    tensor.q4_g32_data = nullptr;
    tensor.q4_g128_data = nullptr;
    tensor.prepared_weight = nullptr;
    tensor.prepared_weight_row_offset = 0;

    const bool is_quantized =
        tensor.prec == Precision::INT8 || tensor.prec == Precision::INT4 ||
        tensor.prec == Precision::FP8_E4M3 ||
        tensor.prec == Precision::MXFP4 || tensor.prec == Precision::NVFP4;
    if (!is_quantized)
        return true;

    const int64_t rows = tensor.shape[0];
    const int64_t cols = tensor.shape[1];
    if (rows <= 0 || cols <= 0) {
        std::fprintf(stderr, "Engine: quantized weight %s has bad shape\n",
                     label);
        return false;
    }

    const uint64_t rows_u = static_cast<uint64_t>(rows);
    const uint64_t cols_u = static_cast<uint64_t>(cols);

    // OCP MXFP4 fixes every part of the microscaling format: E2M1 values,
    // 32 consecutive K elements per group, and one E8M0 byte per group.
    if (tensor.prec == Precision::MXFP4) {
        uint64_t logical_elements = 0;
        uint64_t expected_groups = 0;
        if (header.flags != 0 || !scales || header.group_size != 32 ||
            cols % 32 != 0 ||
            !checked_multiply(rows_u, cols_u, logical_elements) ||
            !checked_multiply(rows_u, cols_u / 32, expected_groups) ||
            expected_groups > std::numeric_limits<uint32_t>::max() ||
            header.data_size != logical_elements / 2 ||
            header.scales_size != expected_groups ||
            header.num_groups != expected_groups) {
            std::fprintf(
                stderr,
                "Engine: MXFP4 weight %s requires packed E2M1 data and "
                "E8M0 block-32 scales (N=%lld K=%lld)\n",
                label, static_cast<long long>(rows),
                static_cast<long long>(cols));
            return false;
        }
        tensor.e8m0_scales = static_cast<const uint8_t*>(scales);
        tensor.group_size = 32;
        tensor.groups_per_row = static_cast<uint32_t>(cols / 32);
        tensor.num_groups = header.num_groups;
        return true;
    }

    // Native NVFP4 routed experts have a per-expert global scale, while a
    // graph constant describes the aggregate of every expert in a layer.
    // They must therefore be presented through MoeSsdCache, which constructs
    // one Tensor per expert and can attach the correct scalar.
    if (tensor.prec == Precision::NVFP4) {
        std::fprintf(stderr,
                     "Engine: NVFP4 weight %s requires SSD expert storage\n",
                     label);
        return false;
    }

    // DeepSeek-V4 dense FP8 tensors use one E8M0 scale per 128x128 output/K
    // tile. This is deliberately explicit: it is not the OCP MXFP8 layout.
    if (tensor.prec == Precision::FP8_E4M3) {
        uint64_t logical_elements = 0;
        const uint64_t n_blocks = (rows_u + 127) / 128;
        const uint64_t k_blocks = (cols_u + 127) / 128;
        uint64_t expected_groups = 0;
        if (header.flags != MappedFile::FLAG_FP8_BLOCK128 || !scales ||
            header.group_size != 128 ||
            !checked_multiply(rows_u, cols_u, logical_elements) ||
            !checked_multiply(n_blocks, k_blocks, expected_groups) ||
            expected_groups > std::numeric_limits<uint32_t>::max() ||
            header.data_size != logical_elements ||
            header.scales_size != expected_groups ||
            header.num_groups != expected_groups) {
            std::fprintf(
                stderr,
                "Engine: FP8 weight %s requires E4M3 data and E8M0 "
                "128x128 block scales (N=%lld K=%lld)\n",
                label, static_cast<long long>(rows),
                static_cast<long long>(cols));
            return false;
        }
        tensor.e8m0_scales = static_cast<const uint8_t*>(scales);
        tensor.group_size = 128;
        tensor.groups_per_row =
            static_cast<uint32_t>(k_blocks);
        tensor.num_groups = header.num_groups;
        tensor.is_fp8_block128 = true;
        return true;
    }

    const bool header_embeds_int4_scales =
        tensor.prec == Precision::INT4 &&
        (header.flags & (MappedFile::FLAG_INT4_BG32 |
                         MappedFile::FLAG_INT4_BG128)) != 0;
    if (header_embeds_int4_scales && header.scales_size != 0) {
        std::fprintf(
            stderr,
            "Engine: INT4 weight %s duplicates external scales; reconvert "
            "the model to canonical BG32/BG128 storage\n",
            label);
        return false;
    }
    if ((!scales && !header_embeds_int4_scales) ||
        header.group_size == 0) {
        std::fprintf(stderr,
                     "Engine: quantized weight %s missing scales/group "
                     "metadata\n",
                     label);
        return false;
    }

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
        MappedFile::FLAG_INT4_BG128 | MappedFile::FLAG_INT4_BG32;
    if (header.flags & MappedFile::FLAG_INT4_Q4DOT_LEGACY) {
        std::fprintf(
            stderr,
            "Engine: INT4 weight %s uses the legacy Q4DOT package layout; "
            "reconvert the model to canonical BG32/BG128 storage\n",
            label);
        return false;
    }
    const bool int4_bg128_layout =
        tensor.prec == Precision::INT4 &&
        (header.flags & MappedFile::FLAG_INT4_BG128);
    const bool int4_bg32_layout =
        tensor.prec == Precision::INT4 &&
        (header.flags & MappedFile::FLAG_INT4_BG32);

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
    const int int4_layout_count =
        static_cast<int>(int4_bg32_layout) +
        static_cast<int>(int4_bg128_layout);
    if (tensor.prec == Precision::INT4 && int4_layout_count != 1) {
        std::fprintf(stderr,
                     "Engine: INT4 weight %s must use canonical BG32 or "
                     "BG128 storage\n",
                     label);
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
    if (int4_bg32_layout &&
        (cols % 32 != 0 || header.group_size != 32)) {
        std::fprintf(stderr,
                     "Engine: INT4 BG32 weight %s requires K multiple of "
                     "32 and group=32 (K=%lld group=%u)\n",
                     label, static_cast<long long>(cols), header.group_size);
        return false;
    }
    // Packed INT4 is a serialized layout, not an ARM-only model format.  The
    // scalar provider decodes it directly, while the ARM provider retains its
    // historical requirement for the DOTPROD kernel.
    if ((int4_bg32_layout || int4_bg128_layout) &&
        mollm::cpu::capabilities().arm_neon &&
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
        if (int4_bg32_layout) {
            if (rows > std::numeric_limits<int>::max() ||
                cols > std::numeric_limits<int>::max()) {
                std::fprintf(stderr,
                             "Engine: quantized weight %s dimensions exceed "
                             "packed kernel limits\n",
                             label);
                return false;
            }
            expected_data_size = static_cast<uint64_t>(
                pack_b_q4dot_g32_bytes(static_cast<int>(rows),
                                       static_cast<int>(cols)));
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
    if (!header_embeds_int4_scales &&
        !checked_multiply(expected_groups, sizeof(float),
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
    tensor.is_q4_repacked = false;
    tensor.is_q4_g32_packed = int4_bg32_layout;
    tensor.is_q4_g128_packed = int4_bg128_layout;
    if (int4_bg32_layout)
        tensor.q4_g32_data = tensor.data;
    if (int4_bg128_layout)
        tensor.q4_g128_data = tensor.data;
    return true;
}

}  // namespace mollm::detail
