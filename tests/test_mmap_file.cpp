#include "graph/mmap_file.h"
#include "kernels/tensor.h"
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>   // std::move
#include <vector>

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

static bool write_header_only(const char* path,
                              const MappedFile::Header& header) {
    FILE* file = fopen(path, "wb");
    if (!file)
        return false;
    bool ok = fwrite(&header, sizeof(header), 1, file) == 1;
    ok = fclose(file) == 0 && ok;
    return ok;
}

int main() {
    // ---- Header size ----
    CHECK(sizeof(MappedFile::Header) == 88, "Header is 88 bytes");

    // ---- in-memory package blob parsing ----
    {
        std::vector<uint8_t> blob(sizeof(MappedFile::Header) + 4, 0);
        MappedFile::Header source = {};
        source.magic = MappedFile::MAGIC;
        source.ndim = 1;
        source.precision = static_cast<uint32_t>(Precision::FP16);
        source.shape[0] = 2;
        source.data_offset = sizeof(MappedFile::Header);
        source.data_size = 4;
        std::memcpy(blob.data(), &source, sizeof(source));

        MappedFile::Header parsed;
        CHECK(MappedFile::parse_header(blob.data(), blob.size(), parsed),
              "parse valid in-memory weight blob");
        CHECK(parsed.data_offset == sizeof(MappedFile::Header) &&
                  parsed.data_size == 4,
              "in-memory weight header fields");
        CHECK(!MappedFile::parse_header(blob.data(),
                                        sizeof(MappedFile::Header) - 1, parsed),
              "reject truncated in-memory weight header");

        source.data_size = 5;
        std::memcpy(blob.data(), &source, sizeof(source));
        CHECK(!MappedFile::parse_header(blob.data(), blob.size(), parsed),
              "reject out-of-range in-memory weight data");

        source.data_size = 2;
        source.scales_offset = sizeof(MappedFile::Header) + 1;
        source.scales_size = 2;
        std::memcpy(blob.data(), &source, sizeof(source));
        CHECK(!MappedFile::parse_header(blob.data(), blob.size(), parsed),
              "reject overlapping in-memory data and scales");

        source.data_size = 4;
        source.scales_offset = 0;
        source.scales_size = 0;
        source.shape[0] = 0;
        std::memcpy(blob.data(), &source, sizeof(source));
        CHECK(!MappedFile::parse_header(blob.data(), blob.size(), parsed),
              "reject zero-sized active dimension");
    }

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

    // ---- overflowed and structurally invalid ranges ----
    {
        MappedFile::Header hdr = {};
        hdr.magic = MappedFile::MAGIC;
        hdr.ndim = 1;
        hdr.precision = (uint32_t)Precision::FP32;
        hdr.shape[0] = 1;
        hdr.data_offset = std::numeric_limits<uint64_t>::max() - 3;
        hdr.data_size = 8;
        CHECK(write_header_only("/tmp/test_mmap_overflow.bin", hdr),
              "write overflowed range file");

        MappedFile mf;
        CHECK(!mf.open("/tmp/test_mmap_overflow.bin"),
              "reject overflowed data range");
        CHECK(!mf.is_open() && mf.data() == nullptr,
              "failed open leaves no mapped data");

        hdr.data_offset = 0;
        hdr.data_size = 0;
        hdr.ndim = 0;
        CHECK(write_header_only("/tmp/test_mmap_bad_ndim.bin", hdr),
              "write invalid ndim file");
        CHECK(!mf.open("/tmp/test_mmap_bad_ndim.bin"),
              "reject invalid dimension count");
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
    remove("/tmp/test_mmap_overflow.bin");
    remove("/tmp/test_mmap_bad_ndim.bin");

    if (failures == 0) {
        printf("\nAll mmap_file tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
