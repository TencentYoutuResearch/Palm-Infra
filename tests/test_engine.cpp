#include "engine/engine.h"
#include "engine/input_prep.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){fprintf(stderr,"FAIL: %s\n",msg);failures++;}else{printf("  PASS: %s\n",msg);} } while(0)

static bool write_package_header(const char* path, uint32_t version,
                                 bool overlapping_sections) {
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

    // ---- engine lifecycle (without graph) ----
    {
        LLMEngine e;
        EngineConfig cfg;
        cfg.package_path = "/tmp/nonexistent.mollm";
        CHECK(!e.load(cfg), "load fails on missing package");

        // test that prefill/decode don't crash on empty graph
        e.reset();
        CHECK(e.past_len() == 0, "past_len=0 after reset");
    }

    // ---- package header validation ----
    {
        const char* wrong_version = "/tmp/mollm_wrong_version.mollm";
        const char* overlap = "/tmp/mollm_overlap.mollm";
        CHECK(write_package_header(wrong_version, 99, false),
              "write wrong-version package");
        CHECK(write_package_header(overlap, 1, true),
              "write overlapping-section package");

        EngineConfig cfg;
        LLMEngine wrong_version_engine;
        cfg.package_path = wrong_version;
        CHECK(!wrong_version_engine.load(cfg),
              "reject unsupported package version");

        LLMEngine overlap_engine;
        cfg.package_path = overlap;
        CHECK(!overlap_engine.load(cfg), "reject overlapping package sections");
        std::remove(wrong_version);
        std::remove(overlap);
    }

    if (failures == 0) {
        printf("\nAll engine tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
