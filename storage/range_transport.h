#pragma once

// Ranged reads of an immutable byte object, independent of where it lives.
//
// A model package addresses every offloaded payload (routed experts, PLE
// rows) by absolute byte offset and length. A transport serves exactly that
// address space: a local file, a block of host memory, or memory on another
// machine. Policy that sits above it (which ranges to keep resident, what to
// evict, what to fetch ahead of need) does not depend on the transport, and a
// reader that needs no cache can call it directly.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mollm {

// What a scheduler needs to know to predict how long a batch of reads takes.
// A read of `bytes` on one plane costs about
//     latency_seconds + bytes / plane_bytes_per_second
// and up to `planes` such reads proceed at once. Zero means not measured.
struct TransportProfile {
    int planes = 1;                     // Independent paths (devices, links, queues).
    double latency_seconds = 0;         // Fixed cost per read.
    double plane_bytes_per_second = 0;  // Sustained rate of one plane.
};

struct ReadSegment {
    void* data = nullptr;
    size_t bytes = 0;
};

class RangeTransport {
public:
    virtual ~RangeTransport() = default;

    // Total bytes of the object.
    virtual uint64_t size() const = 0;

    // Fill dst with the bytes at [offset, offset + bytes). Blocks until all
    // of them arrived; false on any error, in which case dst is unspecified.
    // Safe to call from several threads at once.
    virtual bool read(uint64_t offset, void* dst, size_t bytes) = 0;

    // One contiguous range starting at offset, delivered into consecutive
    // segments. The default issues one read per segment.
    virtual bool read_segments(uint64_t offset, const std::vector<ReadSegment>& segments);

    virtual TransportProfile profile() const { return {}; }

    // Nonzero when the transport has a faster path for aligned reads: a read
    // whose offset, length and destination are all multiples of this value
    // takes it. Such a read may also run past size() by less than one unit,
    // filling only the bytes that exist, so a caller can always round a range
    // out to whole units. Every other read still works.
    virtual size_t direct_alignment() const { return 0; }

    // Short name for diagnostics, such as "file" or "memory".
    virtual const char* kind() const = 0;
};

// How a file transport treats the operating system's file cache.
enum class FileReadMode {
    // The platform default: the page cache on Linux; on macOS the descriptor
    // asks to bypass the unified buffer cache.
    Default,
    // Linux: aligned reads go to the device through a second O_DIRECT
    // descriptor and never touch the page cache; unaligned reads use the
    // ordinary descriptor. Elsewhere this is the same as Default.
    Direct,
};

// A local file read with pread.
class FileRangeTransport final : public RangeTransport {
public:
    // Null when the file cannot be opened in the requested mode; the reason
    // is printed. Some filesystems (tmpfs, for one) refuse Direct.
    static std::shared_ptr<FileRangeTransport> open(const std::string& path,
                                                    FileReadMode mode = FileReadMode::Default);
    ~FileRangeTransport() override;

    uint64_t size() const override { return size_; }
    bool read(uint64_t offset, void* dst, size_t bytes) override;
    bool read_segments(uint64_t offset, const std::vector<ReadSegment>& segments) override;
    size_t direct_alignment() const override { return direct_fd_ >= 0 ? kDirectAlignment : 0; }
    const char* kind() const override { return direct_fd_ >= 0 ? "file-direct" : "file"; }

    // How many reads took the device path and how many the page cache.
    uint64_t direct_reads() const { return direct_reads_.load(std::memory_order_relaxed); }
    uint64_t cached_reads() const { return cached_reads_.load(std::memory_order_relaxed); }

private:
    // Satisfies the logical block size of every device in use; also the page size.
    static constexpr size_t kDirectAlignment = 4096;
    FileRangeTransport(int fd, int direct_fd, uint64_t size)
        : fd_(fd), direct_fd_(direct_fd), size_(size) {}
    bool read_direct(uint64_t offset, void* dst, size_t bytes);
    int fd_ = -1;
    int direct_fd_ = -1;
    uint64_t size_ = 0;
    std::atomic<uint64_t> direct_reads_{0};
    std::atomic<uint64_t> cached_reads_{0};
};

// Bytes already in this process's address space, copied on read. The memory
// must outlive the transport; pass an owner to tie the two together.
class MemoryRangeTransport final : public RangeTransport {
public:
    MemoryRangeTransport(const void* data, uint64_t size, std::shared_ptr<void> owner = nullptr)
        : data_(static_cast<const uint8_t*>(data)), size_(size), owner_(std::move(owner)) {}

    uint64_t size() const override { return size_; }
    bool read(uint64_t offset, void* dst, size_t bytes) override;
    const char* kind() const override { return "memory"; }

private:
    const uint8_t* data_ = nullptr;
    uint64_t size_ = 0;
    std::shared_ptr<void> owner_;
};

}  // namespace mollm
