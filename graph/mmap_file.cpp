#include "graph/mmap_file.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// platform abstraction
// ---------------------------------------------------------------------------

#if defined(_WIN32)

static void* map_file(const char* path, size_t* out_size) {
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) return nullptr;
    LARGE_INTEGER li;
    GetFileSizeEx(hFile, &li);
    *out_size = (size_t)li.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(hFile);
    if (!hMap) return nullptr;
    void* ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    return ptr;
}

static void unmap_file(void* ptr) {
    if (ptr) UnmapViewOfFile(ptr);
}

static void prefetch_pages(void* ptr, size_t len) {
    // Windows: VirtualAlloc with MEM_RESET_UNDO?  No direct MADV_WILLNEED.
    // We touch the first byte of each page as a crude prefetch.
    if (!ptr || len == 0) return;
    volatile char* p = (volatile char*)ptr;
    size_t page = 4096;
    for (size_t i = 0; i < len; i += page) {
        char v = p[i];
        (void)v;
    }
}

static void release_pages_m(void* ptr, size_t len) {
    // VirtualUnlock?  No direct MADV_DONTNEED on Windows.
    (void)ptr; (void)len;
}

#else  // POSIX

static void* map_file(const char* path, size_t* out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return nullptr; }
    *out_size = (size_t)st.st_size;
    void* ptr = mmap(nullptr, *out_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return nullptr;
    return ptr;
}

static void unmap_file(void* ptr, size_t size) {
    if (ptr) munmap(ptr, size);
}

static void prefetch_pages(void* ptr, size_t len) {
#if defined(MADV_WILLNEED)
    if (ptr && len > 0) madvise(ptr, len, MADV_WILLNEED);
#else
    (void)ptr; (void)len;
#endif
}

static void release_pages_m(void* ptr, size_t len) {
#if defined(MADV_DONTNEED)
    if (ptr && len > 0) madvise(ptr, len, MADV_DONTNEED);
#else
    (void)ptr; (void)len;
#endif
}

#endif // _WIN32

namespace {

bool range_within_file(uint64_t offset, uint64_t length, size_t file_size) {
    return offset <= file_size && length <= file_size - offset;
}

bool valid_header(const MappedFile::Header& header, size_t file_size) {
    constexpr uint32_t known_flags = MappedFile::FLAG_INT4_Q4DOT |
                                     MappedFile::FLAG_INT4_BG128;
    if ((header.flags & ~known_flags) != 0 ||
        header.ndim == 0 || header.ndim > 4 ||
        header.precision > 3) {
        return false;
    }
    for (uint32_t dim = 0; dim < header.ndim; ++dim) {
        if (header.shape[dim] == 0)
            return false;
    }
    if (!range_within_file(header.data_offset, header.data_size, file_size) ||
        !range_within_file(header.scales_offset, header.scales_size,
                           file_size)) {
        return false;
    }
    if ((header.data_size != 0 &&
         header.data_offset < sizeof(MappedFile::Header)) ||
        (header.scales_size != 0 &&
         header.scales_offset < sizeof(MappedFile::Header))) {
        return false;
    }
    if (header.data_size != 0 && header.scales_size != 0) {
        const uint64_t data_end = header.data_offset + header.data_size;
        const uint64_t scales_end =
            header.scales_offset + header.scales_size;
        if (header.data_offset < scales_end &&
            header.scales_offset < data_end) {
            return false;
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// MappedFile
// ---------------------------------------------------------------------------

bool MappedFile::parse_header(const void* bytes, size_t size, Header& header) {
    header = {};
    if (!bytes || size < sizeof(Header))
        return false;

    std::memcpy(&header, bytes, sizeof(Header));
    if (header.magic != MAGIC || !valid_header(header, size)) {
        header = {};
        return false;
    }
    return true;
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : mapped_(other.mapped_), file_size_(other.file_size_),
      header_(other.header_) {
    other.mapped_   = nullptr;
    other.file_size_ = 0;
    other.header_ = {};
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close();
        mapped_    = other.mapped_;
        file_size_ = other.file_size_;
        header_    = other.header_;
        other.mapped_   = nullptr;
        other.file_size_ = 0;
        other.header_ = {};
    }
    return *this;
}

bool MappedFile::open(const char* path) {
    close();

    size_t size = 0;
    void* ptr = map_file(path, &size);
    if (!ptr) {
        fprintf(stderr, "MappedFile: failed to mmap %s: %s\n", path, strerror(errno));
        return false;
    }

    if (!parse_header(ptr, size, header_)) {
        fprintf(stderr, "MappedFile: %s has an invalid header or range\n",
                path);
        unmap_file(ptr, size);
        return false;
    }

    mapped_    = ptr;
    file_size_ = size;
    return true;
}

void MappedFile::close() {
    if (mapped_) {
        unmap_file(mapped_, file_size_);
        mapped_    = nullptr;
        file_size_ = 0;
    }
    header_ = {};
}

void MappedFile::prefetch() {
    if (!mapped_) return;
    prefetch_pages(mapped_, file_size_);
}

void MappedFile::release_pages() {
    if (!mapped_) return;
    release_pages_m(mapped_, file_size_);
}
