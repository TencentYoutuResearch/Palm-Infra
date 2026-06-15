#pragma once

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// PROJECT_NAME — MappedFile
//
// Memory-mapped read-only file for zero-copy weight loading.
//
// File format (little-endian):
//
//   Offset  Size   Field
//   ------  ----   -----
//   0       4      magic   — 0x50414D58 ("XMAP")
//   4       4      flags   — reserved (0)
//   8       4      ndim    — number of dimensions (1-4)
//   12      4      precision — Precision enum value
//   16      8      shape[0]
//   24      8      shape[1]
//   32      8      shape[2]
//   40      8      shape[3]
//   48      8      data_offset
//   56      8      data_size
//   64      8      scales_offset  (0 if no scales)
//   72      8      scales_size    (0 if no scales)
//   80      4      group_size
//   84      4      num_groups
//   88      -      [padding to alignment]
//   data_offset  [weight data]
//   scales_offset [quantization scales, if present]
//
// All integers are stored in native (little-endian) byte order.
// ---------------------------------------------------------------------------

class MappedFile {
public:
    static constexpr uint32_t MAGIC = 0x50414D58; // "XMAP"

    struct Header {
        uint32_t magic;
        uint32_t flags;
        uint32_t ndim;
        uint32_t precision;      // Precision enum value
        uint64_t shape[4];
        uint64_t data_offset;
        uint64_t data_size;
        uint64_t scales_offset;
        uint64_t scales_size;
        uint32_t group_size;
        uint32_t num_groups;
    };

    static_assert(sizeof(Header) == 88, "Header size must be 88 bytes");

    MappedFile() = default;

    ~MappedFile() { close(); }

    // not copyable
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    // movable
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    /// Open and memory-map the file at `path`.  Returns true on success.
    bool open(const char* path);

    /// Unmap and close the file.
    void close();

    /// True if the file is currently mapped.
    bool is_open() const { return mapped_ != nullptr; }

    // --- accessors ---

    const Header& header() const { return header_; }

    /// Pointer to the weight data (mmap'd).
    const void* data() const {
        return static_cast<const char*>(mapped_) + header_.data_offset;
    }

    /// Pointer to quantization scales (may be nullptr).
    const void* scales() const {
        if (header_.scales_size == 0) return nullptr;
        return static_cast<const char*>(mapped_) + header_.scales_offset;
    }

    size_t data_size()   const { return header_.data_size; }
    size_t scales_size() const { return header_.scales_size; }

    // --- madvise hints ---

    /// Prefetch pages into RAM (MADV_WILLNEED).
    void prefetch();

    /// Release pages back to OS (MADV_DONTNEED).
    void release_pages();

private:
    int    fd_      = -1;
    void*  mapped_  = nullptr;
    size_t file_size_ = 0;
    Header header_  = {};
};
