#include "graph/mmap_file.h"
#include "kernels/tensor.h"
#include <cstdio>
#include <cstring>
#include <utility>   // std::move

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

// Helper: write a test weight file
static bool write_test_file(const char* path, const MappedFile::Header& hdr,
                            const void* data, size_t data_len,
                            const void* scales = nullptr, size_t scales_len = 0) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    // compute offsets
    MappedFile::Header h = hdr;
    h.data_offset   = sizeof(MappedFile::Header);
    h.data_size     = data_len;
    h.scales_offset = scales ? h.data_offset + data_len : 0;
    h.scales_size   = scales ? scales_len : 0;

    fwrite(&h, sizeof(h), 1, f);
    fwrite(data, 1, data_len, f);
    if (scales && scales_len > 0) {
        fwrite(scales, 1, scales_len, f);
    }
    fclose(f);
    return true;
}

int main() {
    // ---- Header size ----
    CHECK(sizeof(MappedFile::Header) == 88, "Header is 88 bytes");

    // ---- open / close ----
    {
        float data[8] = {1,2,3,4,5,6,7,8};
        MappedFile::Header hdr = {};
        hdr.magic     = MappedFile::MAGIC;
        hdr.precision = (uint32_t)Precision::FP32;
        hdr.ndim      = 2;
        hdr.shape[0]  = 4;
        hdr.shape[1]  = 2;
        hdr.shape[2]  = 1;
        hdr.shape[3]  = 1;

        CHECK(write_test_file("/tmp/test_mmap.bin", hdr, data, sizeof(data)),
              "write test file");

        MappedFile mf;
        CHECK(mf.open("/tmp/test_mmap.bin"), "open test file");
        CHECK(mf.is_open(), "is_open after open");
        CHECK(mf.header().magic == MappedFile::MAGIC, "magic matches");
        CHECK(mf.header().shape[0] == 4, "shape[0]");
        CHECK(mf.header().shape[1] == 2, "shape[1]");
        CHECK(mf.header().precision == (uint32_t)Precision::FP32, "precision");
        CHECK(mf.data_size() == sizeof(data), "data_size");
        CHECK(mf.scales() == nullptr, "no scales");

        // verify data
        const float* d = static_cast<const float*>(mf.data());
        CHECK(d[0] == 1.0f && d[7] == 8.0f, "data contents");

        // prefetch / release_pages (no-ops on some platforms, but must not crash)
        mf.prefetch();
        mf.release_pages();

        mf.close();
        CHECK(!mf.is_open(), "is_open after close");
    }

    // ---- with scales ----
    {
        float data[4] = {10,20,30,40};
        float scales[2] = {0.5f, 2.0f};
        MappedFile::Header hdr = {};
        hdr.magic       = MappedFile::MAGIC;
        hdr.flags       = MappedFile::FLAG_INT4_Q4DOT;
        hdr.precision   = (uint32_t)Precision::FP16;
        hdr.ndim        = 1;
        hdr.shape[0]    = 4;
        hdr.shape[1]    = 1;
        hdr.shape[2]    = 1;
        hdr.shape[3]    = 1;
        hdr.group_size  = 32;
        hdr.num_groups  = 1;

        CHECK(write_test_file("/tmp/test_mmap2.bin", hdr, data, sizeof(data),
                              scales, sizeof(scales)),
              "write file with scales");

        MappedFile mf;
        CHECK(mf.open("/tmp/test_mmap2.bin"), "open file with scales");
        CHECK(mf.data_size() == sizeof(data), "data_size with scales");
        CHECK(mf.scales_size() == sizeof(scales), "scales_size");
        CHECK(mf.scales() != nullptr, "scales not null");
        CHECK(mf.header().flags == MappedFile::FLAG_INT4_Q4DOT, "flags");
        CHECK(mf.header().group_size == 32, "group_size");
        CHECK(mf.header().num_groups == 1, "num_groups");

        const float* s = static_cast<const float*>(mf.scales());
        CHECK(s[0] == 0.5f && s[1] == 2.0f, "scales contents");

        mf.close();
    }

    // ---- bad magic ----
    {
        MappedFile::Header hdr = {};
        hdr.magic = 0xDEADBEEF;
        CHECK(write_test_file("/tmp/test_mmap_bad.bin", hdr, nullptr, 0),
              "write bad magic file");

        MappedFile mf;
        CHECK(!mf.open("/tmp/test_mmap_bad.bin"), "reject bad magic");
    }

    // ---- move semantics ----
    {
        float data[2] = {99, 100};
        MappedFile::Header hdr = {};
        hdr.magic     = MappedFile::MAGIC;
        hdr.precision = (uint32_t)Precision::FP32;
        hdr.ndim      = 1;
        hdr.shape[0]  = 2;
        hdr.shape[1]  = 1;
        hdr.shape[2]  = 1;
        hdr.shape[3]  = 1;
        CHECK(write_test_file("/tmp/test_mmap_move.bin", hdr, data, sizeof(data)),
              "write move test file");

        MappedFile mf;
        CHECK(mf.open("/tmp/test_mmap_move.bin"), "open for move");
        CHECK(mf.is_open(), "src is_open");

        MappedFile mf2 = std::move(mf);
        CHECK(!mf.is_open(), "moved-from not open");
        CHECK(mf2.is_open(), "moved-to is open");
        CHECK(mf2.data_size() == sizeof(data), "moved-to data_size");
        CHECK(static_cast<const float*>(mf2.data())[0] == 99.0f, "moved-to data");

        mf2.close();
    }

    // cleanup
    remove("/tmp/test_mmap.bin");
    remove("/tmp/test_mmap2.bin");
    remove("/tmp/test_mmap_bad.bin");
    remove("/tmp/test_mmap_move.bin");

    if (failures == 0) {
        printf("\nAll mmap_file tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
