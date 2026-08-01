#include "engine/engine.h"
#include "engine/input_prep.h"
#include "engine/vision.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

static bool write_package_header(const char* path, uint32_t version,
                                 bool overlapping_sections,
                                 bool overlapping_vision = false) {
    std::vector<uint8_t> bytes(256, 0);
    const uint32_t magic = 0x4D4C4F4D;
    std::memcpy(bytes.data(), &magic, sizeof(magic));
    std::memcpy(bytes.data() + 4, &version, sizeof(version));
    if (overlapping_sections) {
        const uint64_t meta_offset = 128;
        const uint64_t meta_length = 8;
        const uint64_t prefill_offset = 132;
        const uint64_t prefill_length = 8;
        const uint64_t decode_offset = 140;
        const uint64_t decode_length = 8;
        std::memcpy(bytes.data() + 8, &meta_offset, 8);
        std::memcpy(bytes.data() + 16, &meta_length, 8);
        std::memcpy(bytes.data() + 56, &prefill_offset, 8);
        std::memcpy(bytes.data() + 64, &prefill_length, 8);
        std::memcpy(bytes.data() + 72, &decode_offset, 8);
        std::memcpy(bytes.data() + 80, &decode_length, 8);
    } else if (overlapping_vision) {
        const uint64_t meta_offset = 128;
        const uint64_t meta_length = 8;
        const uint64_t prefill_offset = 136;
        const uint64_t prefill_length = 8;
        const uint64_t decode_offset = 144;
        const uint64_t decode_length = 8;
        const uint64_t vision_offset = 148;
        const uint64_t vision_length = 8;
        std::memcpy(bytes.data() + 8, &meta_offset, 8);
        std::memcpy(bytes.data() + 16, &meta_length, 8);
        std::memcpy(bytes.data() + 56, &prefill_offset, 8);
        std::memcpy(bytes.data() + 64, &prefill_length, 8);
        std::memcpy(bytes.data() + 72, &decode_offset, 8);
        std::memcpy(bytes.data() + 80, &decode_length, 8);
        std::memcpy(bytes.data() + 104, &vision_offset, 8);
        std::memcpy(bytes.data() + 112, &vision_length, 8);
    }

    FILE* file = std::fopen(path, "wb");
    if (!file)
        return false;
    bool ok = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
    ok = std::fclose(file) == 0 && ok;
    return ok;
}

int main() {
    // ---- causal mask ----
    {
        std::vector<float> mask(9);
        mollm::detail::fill_causal_mask(mask.data(), 3, 0);
        // row 0: [0, -inf, -inf]
        CHECK(mask[0] == 0.f, "mask[0,0]=0");
        CHECK(mask[1] < -1e30f, "mask[0,1]=-inf");
        CHECK(mask[2] < -1e30f, "mask[0,2]=-inf");
        // row 1: [0, 0, -inf]
        CHECK(mask[3] == 0.f, "mask[1,0]=0");
        CHECK(mask[4] == 0.f, "mask[1,1]=0");
        CHECK(mask[5] < -1e30f, "mask[1,2]=-inf");
    }

    {
        std::vector<float> mask(6);
        mollm::detail::fill_causal_mask(mask.data(), 1, 5);
        bool all_visible = true;
        for (float value : mask)
            all_visible &= value == 0.f;
        CHECK(all_visible, "single-token mask sees all past positions");
    }

    // ---- RoPE cache ----
    {
        std::vector<float> cos_cache(32 * 4);
        std::vector<float> sin_cache(32 * 4);
        mollm::detail::fill_rope_cache(cos_cache.data(), sin_cache.data(), 4,
                                       0, 64, 500000.f);
        // cos(0) = 1, sin(0) = 0
        CHECK(std::fabs(cos_cache[0] - 1.f) < 1e-5f, "cos(0)=1");
        CHECK(std::fabs(sin_cache[0]) < 1e-5f, "sin(0)=0");
    }

    // ---- Qwen3.5 interleaved multimodal RoPE ----
    {
        std::vector<float> cos_cache(32);
        std::vector<float> sin_cache(32);
        const std::vector<int> positions = {1, 2, 3}; // T, H, W
        mollm::detail::fill_multimodal_rope_cache(
            cos_cache.data(), sin_cache.data(), 1, 0, positions,
            64, 10000000.f, 11, 11, 10);
        const auto expected_sin = [](int position, int frequency) {
            const float exponent = static_cast<float>(2 * frequency) / 64.f;
            return std::sin(
                position / std::pow(10000000.f, exponent));
        };
        CHECK(std::fabs(sin_cache[0] - expected_sin(1, 0)) < 1e-6f,
              "multimodal RoPE frequency 0 uses temporal position");
        CHECK(std::fabs(sin_cache[1] - expected_sin(2, 1)) < 1e-6f,
              "multimodal RoPE frequency 1 uses height position");
        CHECK(std::fabs(sin_cache[2] - expected_sin(3, 2)) < 1e-6f,
              "multimodal RoPE frequency 2 uses width position");
        CHECK(std::fabs(sin_cache[30] - expected_sin(1, 30)) < 1e-6f &&
                  std::fabs(sin_cache[31] - expected_sin(2, 31)) < 1e-6f,
              "multimodal RoPE tail follows configured sections");
    }

    // ---- Qwen3.5 one-image position IDs ----
    {
        const std::vector<int> tokens = {10, 99, 99, 99, 99, 11};
        std::vector<int> positions;
        int delta = 0;
        std::string error;
        CHECK(mollm::detail::build_multimodal_position_ids(
                  tokens, 99, 1, 4, 4, 4, 2, 0,
                  positions, delta, &error),
              "build one-image multimodal positions");
        const std::vector<int> expected = {
            0, 1, 1, 1, 1, 3,
            0, 1, 1, 2, 2, 3,
            0, 1, 2, 1, 2, 3,
        };
        CHECK(positions == expected,
              "multimodal positions match text plus 2x2 image grid");
        CHECK(delta == -2,
              "multimodal rope delta accounts for spatial compression");

        const std::vector<int> split = {10, 99, 99, 11, 99, 99};
        CHECK(!mollm::detail::build_multimodal_position_ids(
                  split, 99, 1, 4, 4, 4, 2, 0,
                  positions, delta, &error),
              "reject non-contiguous image placeholders");
    }

    // ---- Qwen image resize budget ----
    {
        int height = 0;
        int width = 0;
        mollm::detail::smart_resize_dimensions(
            768, 1024, 32, 256 * 256, 512 * 512, height, width);
        CHECK(height == 416 && width == 576,
              "default image budget downsizes 1024x768 to 576x416");
        CHECK(height * width <= 512 * 512,
              "resized image stays within configured pixel budget");

        mollm::detail::smart_resize_dimensions(
            200, 300, 32, 256 * 256, 512 * 512, height, width);
        CHECK(height == 224 && width == 320,
              "small image is aligned and raised to minimum budget");
    }

    // ---- transparent PNG keeps RGB underneath alpha=0 ----
    {
        static const unsigned char png[] = {
            0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,
            0x49,0x48,0x44,0x52,0x00,0x00,0x00,0x40,0x00,0x00,0x00,0x40,
            0x08,0x06,0x00,0x00,0x00,0xaa,0x69,0x71,0xde,0x00,0x00,0x00,
            0x7e,0x49,0x44,0x41,0x54,0x78,0x9c,0xe5,0xce,0x41,0x01,0x00,
            0x30,0x0c,0x84,0x30,0x86,0x7f,0xcf,0x9d,0x8c,0x7b,0x10,0x05,
            0x79,0xc7,0xd6,0xe3,0xa6,0x05,0x89,0x93,0x38,0x89,0x93,0x38,
            0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,
            0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,
            0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,
            0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,
            0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,
            0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,
            0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,0x89,0x93,0x38,
            0x89,0x93,0x38,0xd7,0x81,0xb5,0x0f,0x94,0x68,0x03,0x7e,0xb6,
            0x71,0xfd,0x61,0x00,0x00,0x00,0x00,0x49,0x45,0x4e,0x44,0xae,
            0x42,0x60,0x82,
        };
        const char* path = "/tmp/mollm_transparent_rgb.png";
        FILE* file = std::fopen(path, "wb");
        bool wrote =
            file && std::fwrite(png, 1, sizeof(png), file) == sizeof(png);
        if (file) wrote = std::fclose(file) == 0 && wrote;
        CHECK(wrote, "write transparent PNG fixture");

        std::vector<unsigned char> rgba;
        int width = 0;
        int height = 0;
        CHECK(mollm::detail::decode_image_file_rgba(
                  path, rgba, width, height),
              "decode transparent PNG fixture");
        CHECK(width == 64 && height == 64 && rgba.size() == 64 * 64 * 4,
              "transparent PNG dimensions are preserved");
        CHECK(rgba.size() >= 4 &&
                  rgba[0] == 255 && rgba[1] == 0 &&
                  rgba[2] == 0 && rgba[3] == 0,
              "transparent pixel keeps its hidden red RGB value");
        std::remove(path);
    }
    // ---- engine lifecycle (without graph) ----
    {
        LLMEngine e;
        EngineConfig cfg;
        cfg.package_path = "/tmp/nonexistent.mollm";
        CHECK(!e.load(cfg), "load fails on missing package");
        CHECK(e.config().package_path.empty(),
              "failed load rolls engine config back to empty state");
        CHECK(e.package_metadata().empty(),
              "failed load leaves no package metadata");
        CHECK(e.prefill_pool_stats().active == 0 &&
                  e.decode_pool_stats().active == 0,
              "failed load leaves graph pools empty");
        CHECK(!e.load(cfg), "engine remains reusable after failed load");

        cfg.image_max_pixels =
            EngineConfig::kAbsoluteImageMaxPixels + 1;
        CHECK(!e.load(cfg), "reject image budget above absolute limit");

        // test that prefill/decode don't crash on empty graph
        e.reset();
        CHECK(e.past_len() == 0, "past_len=0 after reset");
    }

    // ---- package header validation ----
    {
        const char* wrong_version = "/tmp/mollm_wrong_version.mollm";
        const char* overlap = "/tmp/mollm_overlap.mollm";
        const char* vision_overlap = "/tmp/mollm_vision_overlap.mollm";
        CHECK(write_package_header(wrong_version, 99, false),
              "write wrong-version package");
        CHECK(write_package_header(overlap, 1, true),
              "write overlapping-section package");
        CHECK(write_package_header(vision_overlap, 1, false, true),
              "write overlapping vision-section package");

        EngineConfig cfg;
        LLMEngine wrong_version_engine;
        cfg.package_path = wrong_version;
        CHECK(!wrong_version_engine.load(cfg),
              "reject unsupported package version");

        LLMEngine overlap_engine;
        cfg.package_path = overlap;
        CHECK(!overlap_engine.load(cfg), "reject overlapping package sections");
        LLMEngine vision_overlap_engine;
        cfg.package_path = vision_overlap;
        CHECK(!vision_overlap_engine.load(cfg),
              "reject overlapping optional vision graph");
        std::remove(wrong_version);
        std::remove(overlap);
        std::remove(vision_overlap);
    }

    if (failures == 0) {
        printf("\nAll engine tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
