#include "engine/accelerator_backend.h"
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
    unsigned char e8m0_scales[16] = {};

    CHECK(mollm::detail::is_routed_expert_aggregate_ref(
              "./model_layers_0_mlp_experts_gate_up_proj.weights"),
          "classify routed expert aggregate");
    CHECK(mollm::detail::is_routed_expert_aggregate_ref(
              "./layer_0_experts_down.weights"),
          "classify compact routed expert aggregate");
    CHECK(!mollm::detail::is_routed_expert_aggregate_ref(
              "./model_layers_0_mlp_shared_experts_gate_proj_weight.weights"),
          "exclude shared expert matrix from routed aggregates");
    CHECK(!mollm::detail::is_routed_expert_aggregate_ref(
              "model.layers.0.mlp.shared_experts.down_proj.weight"),
          "exclude raw shared expert reference");

    {
        Tensor weight = make_weight(Precision::FP16, 2, 4);
        weight.scales = scales;
        weight.group_size = 7;
        weight.num_groups = 9;
        weight.groups_per_row = 3;
        weight.is_q4_repacked = true;
        weight.is_q4_g32_packed = true;
        weight.is_q4_g128_packed = true;
        MappedFile::Header header{};

        CHECK(mollm::detail::configure_weight_metadata(
                  weight, header, nullptr, "fp16"),
              "accept non-quantized weight");
        CHECK(weight.scales == nullptr && weight.group_size == 0 &&
                  weight.num_groups == 0 && weight.groups_per_row == 0 &&
                  !weight.is_q4_repacked && !weight.is_q4_g32_packed &&
                  !weight.is_q4_g128_packed,
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
        CHECK(!weight.is_q4_repacked && !weight.is_q4_g32_packed &&
                  !weight.is_q4_g128_packed,
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
        Tensor weight = make_weight(Precision::MXFP4, 2, 64);
        const MappedFile::Header header =
            make_header(64, 4, 32, 4);
        CHECK(mollm::detail::configure_weight_metadata(
                  weight, header, e8m0_scales, "mxfp4"),
              "accept valid MXFP4 metadata");
        CHECK(weight.e8m0_scales == e8m0_scales &&
                  weight.group_size == 32 &&
                  weight.groups_per_row == 2 &&
                  weight.num_groups == 4,
              "attach MXFP4 E8M0 block-32 metadata");

        MappedFile::Header wrong_group = header;
        wrong_group.group_size = 16;
        CHECK(!mollm::detail::configure_weight_metadata(
                  weight, wrong_group, e8m0_scales, "mxfp4-wrong-group"),
              "reject nonstandard MXFP4 group size");
    }

    {
        Tensor weight = make_weight(Precision::FP8_E4M3, 129, 257);
        const uint32_t groups = 2 * 3;
        const MappedFile::Header header =
            make_header(129 * 257, groups, 128, groups,
                        MappedFile::FLAG_FP8_BLOCK128);
        CHECK(mollm::detail::configure_weight_metadata(
                  weight, header, e8m0_scales, "fp8"),
              "accept FP8 E4M3 128x128 block metadata");
        CHECK(weight.e8m0_scales == e8m0_scales &&
                  weight.is_fp8_block128 &&
                  weight.groups_per_row == 3,
              "attach FP8 E8M0 2D block metadata");

        MappedFile::Header missing_layout = header;
        missing_layout.flags = 0;
        CHECK(!mollm::detail::configure_weight_metadata(
                  weight, missing_layout, e8m0_scales,
                  "fp8-missing-layout"),
              "reject FP8 without scale-layout flag");
    }

    {
        unsigned char packed[160] = {};
        Tensor weight = make_weight(Precision::INT4, 8, 32);
        weight.data = packed;
        const MappedFile::Header header =
            make_header(sizeof(packed), 0, 32, 8,
                        MappedFile::FLAG_INT4_BG32);
        CHECK(mollm::detail::configure_weight_metadata(
                  weight, header, nullptr, "bg32"),
              "accept BG32 with embedded scales");
        CHECK(weight.is_q4_g32_packed &&
                  weight.q4_g32_data == packed &&
                  weight.scales == nullptr &&
                  weight.groups_per_row == 1,
              "attach BG32 packed metadata");
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
