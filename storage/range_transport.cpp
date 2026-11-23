#include "storage/range_transport.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

namespace mollm {

bool RangeTransport::read_segments(uint64_t offset, const std::vector<ReadSegment>& segments) {
    for (const ReadSegment& segment : segments) {
        if (!read(offset, segment.data, segment.bytes)) return false;
        offset += segment.bytes;
    }
    return true;
}

std::shared_ptr<FileRangeTransport> FileRangeTransport::open(const std::string& path,
                                                             FileReadMode mode) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "range transport: failed to open %s: %s\n",
                     path.c_str(), std::strerror(errno));
        return nullptr;
    }
    struct stat file_stat {};
    if (fstat(fd, &file_stat) != 0 || file_stat.st_size < 0) {
        std::fprintf(stderr, "range transport: failed to stat %s: %s\n",
                     path.c_str(), std::strerror(errno));
        close(fd);
        return nullptr;
    }
#if defined(__APPLE__)
    // Request uncached reads on this dedicated descriptor; a separate mmap of
    // the same file keeps its own policy. F_NOCACHE does not guarantee that
    // every read reaches the device: existing UBC pages and alignment can
    // still affect the path.
    if (fcntl(fd, F_NOCACHE, 1) != 0) {
        std::fprintf(stderr, "range transport: warning: could not disable macOS file cache: %s\n",
                     std::strerror(errno));
    }
#endif
    int direct_fd = -1;
#if defined(O_DIRECT)
    if (mode == FileReadMode::Direct) {
        direct_fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (direct_fd < 0) {
            std::fprintf(stderr, "range transport: %s does not support direct reads: %s\n",
                         path.c_str(), std::strerror(errno));
            close(fd);
            return nullptr;
        }
    }
#else
    (void)mode;
#endif
    return std::shared_ptr<FileRangeTransport>(
        new FileRangeTransport(fd, direct_fd, static_cast<uint64_t>(file_stat.st_size)));
}

FileRangeTransport::~FileRangeTransport() {
    if (direct_fd_ >= 0) {
        std::fprintf(stderr, "range transport: direct reads=%llu cached reads=%llu\n",
                     static_cast<unsigned long long>(direct_reads()),
                     static_cast<unsigned long long>(cached_reads()));
        close(direct_fd_);
    }
    if (fd_ >= 0) close(fd_);
}

// The caller aligned everything, so the device writes straight into dst. A
// direct descriptor returns fewer bytes than asked only at the end of the file.
bool FileRangeTransport::read_direct(uint64_t offset, void* dst, size_t bytes) {
    if (offset > size_) return false;
    const size_t expected = static_cast<size_t>(std::min<uint64_t>(bytes, size_ - offset));
    uint8_t* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < expected) {
        const ssize_t n = pread(direct_fd_, out + done, bytes - done,
                                static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0 || (static_cast<size_t>(n) % kDirectAlignment && done + static_cast<size_t>(n) < expected)) {
            std::fprintf(stderr, "range transport: direct pread failed at offset %llu: %s\n",
                         static_cast<unsigned long long>(offset + done),
                         n < 0 ? std::strerror(errno) : "unexpected EOF");
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool FileRangeTransport::read(uint64_t offset, void* dst, size_t bytes) {
    if (direct_fd_ >= 0 && bytes && offset % kDirectAlignment == 0 && bytes % kDirectAlignment == 0 &&
        reinterpret_cast<uintptr_t>(dst) % kDirectAlignment == 0) {
        direct_reads_.fetch_add(1, std::memory_order_relaxed);
        return read_direct(offset, dst, bytes);
    }
    cached_reads_.fetch_add(1, std::memory_order_relaxed);
    uint8_t* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = pread(fd_, out + done, bytes - done,
                                static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            std::fprintf(stderr, "range transport: pread failed at offset %llu: %s\n",
                         static_cast<unsigned long long>(offset + done),
                         n == 0 ? "unexpected EOF" : std::strerror(errno));
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool FileRangeTransport::read_segments(uint64_t offset, const std::vector<ReadSegment>& segments) {
    cached_reads_.fetch_add(1, std::memory_order_relaxed);
    std::vector<iovec> vectors;
    vectors.reserve(segments.size());
    for (const ReadSegment& segment : segments)
        if (segment.bytes) vectors.push_back({segment.data, segment.bytes});
    size_t index = 0;
    while (index < vectors.size()) {
        const ssize_t bytes = preadv(
            fd_, vectors.data() + static_cast<ptrdiff_t>(index),
            static_cast<int>(vectors.size() - index), static_cast<off_t>(offset));
        if (bytes < 0 && errno == EINTR) continue;
        if (bytes <= 0) {
            std::fprintf(stderr, "range transport: preadv failed at offset %llu: %s\n",
                         static_cast<unsigned long long>(offset),
                         bytes == 0 ? "unexpected EOF" : std::strerror(errno));
            return false;
        }
        offset += static_cast<uint64_t>(bytes);
        size_t consumed = static_cast<size_t>(bytes);
        while (index < vectors.size() && consumed >= vectors[index].iov_len) {
            consumed -= vectors[index].iov_len;
            ++index;
        }
        if (index < vectors.size() && consumed != 0) {
            vectors[index].iov_base = static_cast<uint8_t*>(vectors[index].iov_base) + consumed;
            vectors[index].iov_len -= consumed;
        }
    }
    return true;
}

bool MemoryRangeTransport::read(uint64_t offset, void* dst, size_t bytes) {
    if (offset > size_ || bytes > size_ - offset) {
        std::fprintf(stderr, "range transport: memory read past the end at offset %llu\n",
                     static_cast<unsigned long long>(offset));
        return false;
    }
    if (bytes) std::memcpy(dst, data_ + offset, bytes);
    return true;
}

}  // namespace mollm
