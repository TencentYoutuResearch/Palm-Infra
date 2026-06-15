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

// ---------------------------------------------------------------------------
// MappedFile
// ---------------------------------------------------------------------------

MappedFile::MappedFile(MappedFile&& other) noexcept
    : fd_(other.fd_), mapped_(other.mapped_), file_size_(other.file_size_),
      header_(other.header_) {
    other.fd_       = -1;
    other.mapped_   = nullptr;
    other.file_size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close();
        fd_        = other.fd_;
        mapped_    = other.mapped_;
        file_size_ = other.file_size_;
        header_    = other.header_;
        other.fd_       = -1;
        other.mapped_   = nullptr;
        other.file_size_ = 0;
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

    if (size < sizeof(Header)) {
        fprintf(stderr, "MappedFile: %s too small (%zu < %zu)\n",
                path, size, sizeof(Header));
        unmap_file(ptr, size);
        return false;
    }

    // copy header from mmap'd region
    memcpy(&header_, ptr, sizeof(Header));

    if (header_.magic != MAGIC) {
        fprintf(stderr, "MappedFile: %s bad magic 0x%08x (expected 0x%08x)\n",
                path, header_.magic, MAGIC);
        unmap_file(ptr, size);
        return false;
    }

    // validate offsets
    if (header_.data_offset + header_.data_size > size ||
        header_.scales_offset + header_.scales_size > size) {
        fprintf(stderr, "MappedFile: %s data/scales extend beyond file\n", path);
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
    fd_ = -1;
}

void MappedFile::prefetch() {
    if (!mapped_) return;
    prefetch_pages(mapped_, file_size_);
}

void MappedFile::release_pages() {
    if (!mapped_) return;
    release_pages_m(mapped_, file_size_);
}
